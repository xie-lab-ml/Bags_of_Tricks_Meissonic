import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from . import merge
import time
from .utils import isinstance_str, init_generator


def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // (x.shape[1]))))
    args = tome_info["args"]
    if downsample <= (args["max_downsample"]):
        w = int(math.ceil(original_w / downsample)) # 32
        h = int(math.ceil(original_h / downsample)) # 32
        r = int(x.shape[1] * args["ratio"]) # 512
        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, 
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    return m, u


from diffusers.utils.torch_utils import maybe_allow_in_graph

def make_diffusers_tome_block(block_class: Type[torch.nn.Module], if_single = False, root_class = None) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    @maybe_allow_in_graph
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _root_class = root_class
        
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
        ):
            if not hasattr(self._root_class, "merge"):
                merge, unmerge = compute_merge(hidden_states, self._tome_info)
                self._root_class.merge = merge
                self._root_class.unmerge = unmerge
                self._root_class.count = 0
            else:
                merge = self._root_class.merge
                unmerge = self._root_class.unmerge
                self._root_class.count += 1
                if self._root_class.count > 3:
                    del self._root_class.merge, self._root_class.unmerge
            
            torch.cuda.synchronize()

            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            txt_emb_shape = image_rotary_emb[0].shape[0] - norm_hidden_states.shape[1]
            norm_hidden_states = merge(norm_hidden_states, rotaty = False)
            if not hasattr(self._root_class, "rotary_list"): 
                aa = image_rotary_emb[0][None, txt_emb_shape:, :].expand(norm_hidden_states.shape[0], -1, -1)
                bb = image_rotary_emb[1][None, txt_emb_shape:, :].expand(norm_hidden_states.shape[0], -1, -1)
                cc = image_rotary_emb[0][None, :txt_emb_shape, :].expand(norm_hidden_states.shape[0], -1, -1)
                dd = image_rotary_emb[1][None, :txt_emb_shape, :].expand(norm_hidden_states.shape[0], -1, -1)
                self._root_class.rotary_list = [aa,bb,cc,dd]
            else:
                aa, bb, cc, dd = self._root_class.rotary_list
            merged_rotary_emb = (merge(aa, rotaty = False), merge(bb, rotaty = False))
            merged_rotary_emb = (torch.cat([cc, merged_rotary_emb[0]], dim=1), 
                                torch.cat([dd, merged_rotary_emb[1]], dim=1))
            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=merged_rotary_emb,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            
            attn_output = unmerge(attn_output, rotaty = False)
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            norm_hidden_states = merge(norm_hidden_states)
            ff_output = self.ff(norm_hidden_states)

            ff_output = gate_mlp.unsqueeze(1) * ff_output

            ff_output = unmerge(ff_output)
            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
            return encoder_hidden_states, hidden_states


    class SingleToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class
        _root_class = root_class
        
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
        ) -> torch.Tensor:
            residual = hidden_states
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            # mlp_hidden_states = torch.cat([mlp_hidden_states[:, :77, :], unmerge(mlp_hidden_states[:, 77:, :])], dim=1)
            # norm_hidden_states = res_norm_hidden_states
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )
            # attn_output = torch.cat([attn_output[:, :unused_hidden_states.shape[1], :], unmerge(attn_output[:, unused_hidden_states.shape[1]:, :])], dim=1)
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            
            hidden_states = gate * self.proj_out(hidden_states)
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states

    if if_single:
        return SingleToMeBlock
    else:
        return ToMeBlock



def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args, kwargs):
        if "hidden_states" in kwargs:
            module._tome_info["size"] = (kwargs["hidden_states"].shape[1] // 2, kwargs["hidden_states"].shape[2] // 2)
        else:
            module._tome_info["size"] = (args[0].shape[1] // 2, args[0].shape[2] // 2)
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook, with_kwargs=True))


def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.transformer" and "unet"
        diffusion_model = model.transformer if hasattr(model, "transformer") else model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }
    hook_tome_model(diffusion_model)
    
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "TransformerBlock"):
            make_tome_block_fn = make_diffusers_tome_block
            module.__class__ = make_tome_block_fn(module.__class__, if_single=False, root_class=diffusion_model)
            module._tome_info = diffusion_model._tome_info
            
            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False
                
    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.transformer if hasattr(model, "transformer") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
        if module.__class__.__name__ == "SingleToMeBlock":
            module.__class__ = module._parent
    
    return model
