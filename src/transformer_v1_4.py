# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings,FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class   SingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

@maybe_allow_in_graph
class TransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        # self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="swiglu")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        # self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="swiglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        # print("norm_hidden_states",norm_hidden_states.shape)
        # print("norm_encoder_hidden_states",norm_encoder_hidden_states.shape)
        # print("image_rotary_emb",image_rotary_emb[0].shape)
        # norm_hidden_states torch.Size([64, 256, 1024])
        # norm_encoder_hidden_states torch.Size([64, 77, 1024])
        # image_rotary_emb torch.Size([333, 128])

        ###
        # norm_hidden_states torch.Size([64, 256, 1024])
        # norm_encoder_hidden_states torch.Size([64, 77, 1024])
        # image_rotary_emb torch.Size([64, 333, 64, 2, 2])
        # image_rotary_emb torch.Size([1, 333, 64, 2, 2])
        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

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

from diffusers.models.normalization import GlobalResponseNorm, RMSNorm
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
class UVit2DConvEmbed(nn.Module):
    def __init__(self, in_channels, block_out_channels, vocab_size, elementwise_affine, eps, bias):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, in_channels)
        self.layer_norm = RMSNorm(in_channels, eps, elementwise_affine)
        self.conv = nn.Conv2d(in_channels, block_out_channels, kernel_size=1, bias=bias)

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = self.conv(embeddings)
        return embeddings

class ConvMlmLayer(nn.Module):
    def __init__(
        self,
        block_out_channels: int,
        in_channels: int,
        use_bias: bool,
        ln_elementwise_affine: bool,
        layer_norm_eps: float,
        codebook_size: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(block_out_channels, in_channels, kernel_size=1, bias=use_bias)
        self.layer_norm = RMSNorm(in_channels, layer_norm_eps, ln_elementwise_affine)
        self.conv2 = nn.Conv2d(in_channels, codebook_size, kernel_size=1, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        logits = self.conv2(hidden_states)
        return logits

class SwiGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function. It's similar to `GEGLU`
    but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * self.activation(gate)

class ConvNextBlock(nn.Module):
    def __init__(
        self, channels, layer_norm_eps, ln_elementwise_affine, use_bias, hidden_dropout, hidden_size, res_ffn_factor=4
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=use_bias,
        )
        self.norm = RMSNorm(channels, layer_norm_eps, ln_elementwise_affine)
        self.channelwise_linear_1 = nn.Linear(channels, int(channels * res_ffn_factor), bias=use_bias)
        # self.channelwise_act = SwiGLU(channels, int(channels * res_ffn_factor), bias=use_bias)            #
        self.channelwise_act = nn.GELU()
        self.channelwise_norm = GlobalResponseNorm(int(channels * res_ffn_factor))
        self.channelwise_linear_2 = nn.Linear(int(channels * res_ffn_factor), channels, bias=use_bias)
        self.channelwise_dropout = nn.Dropout(hidden_dropout)
        self.cond_embeds_mapper = nn.Linear(hidden_size, channels * 2, use_bias)

    def forward(self, x, cond_embeds):
        x_res = x

        x = self.depthwise(x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.channelwise_linear_1(x)
        x = self.channelwise_act(x)
        x = self.channelwise_norm(x)
        x = self.channelwise_linear_2(x)
        x = self.channelwise_dropout(x)

        x = x.permute(0, 3, 1, 2)

        x = x + x_res

        scale, shift = self.cond_embeds_mapper(F.silu(cond_embeds)).chunk(2, dim=1)
        x = x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        return x

# BasicTransformerBlock(
#     dim=hidden_size,
#     num_attention_heads=num_attention_heads,
#     attention_head_dim=hidden_size // num_attention_heads,
#     dropout=hidden_dropout,
#     cross_attention_dim=hidden_size,
#     attention_bias=use_bias,
#     norm_type="ada_norm_continuous",
#     ada_norm_continous_conditioning_embedding_dim=hidden_size,
#     norm_elementwise_affine=ln_elementwise_affine,
#     norm_eps=layer_norm_eps,
#     ada_norm_bias=use_bias,
#     ff_inner_dim=intermediate_size,
#     ff_bias=use_bias,
#     attention_out_bias=use_bias,
#     positional_embeddings="sinusoidal", # I add
#     num_positional_embeddings=64, # I add
#     activation_fn="swiglu"
# )
# for _ in range(num_hidden_layers)
from diffusers.models.attention import BasicTransformerBlock, SkipFFTransformerBlock
from diffusers.models.resnet import Downsample2D, Upsample2D

class Simple_UVitBlock(nn.Module):
    def __init__(
        self,
        channels,
        ln_elementwise_affine,
        layer_norm_eps,
        use_bias,
        downsample: bool,
        upsample: bool,
    ):
        super().__init__()

        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
            )
        else:
            self.downsample = None

        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
            )
        else:
            self.upsample = None

    def forward(self, x):
        # print("before,", x.shape)
        if self.downsample is not None:
            # print('downsample')
            x = self.downsample(x)

        if self.upsample is not None:
            # print('upsample')
            x = self.upsample(x)
        # print("after,", x.shape)
        return x


class UVitBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_res_blocks: int,
        hidden_size,
        hidden_dropout,
        ln_elementwise_affine,
        layer_norm_eps,
        use_bias,
        block_num_heads,
        attention_dropout,
        downsample: bool,
        upsample: bool,
    ):
        super().__init__()

        if downsample:
            self.downsample = Downsample2D(
                channels,
                use_conv=True,
                padding=0,
                name="Conv2d_0",
                kernel_size=2,
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
            )
        else:
            self.downsample = None

        self.res_blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    channels,
                    layer_norm_eps,
                    ln_elementwise_affine,
                    use_bias,
                    hidden_dropout,
                    hidden_size,
                )
                for i in range(num_res_blocks)
            ]
        )

        self.attention_blocks = nn.ModuleList(
            [
                SkipFFTransformerBlock(
                    channels,
                    block_num_heads,
                    channels // block_num_heads,
                    hidden_size,
                    use_bias,
                    attention_dropout,
                    channels,
                    attention_bias=use_bias,
                    attention_out_bias=use_bias,
                )
                for _ in range(num_res_blocks)
            ]
        )

        if upsample:
            self.upsample = Upsample2D(
                channels,
                use_conv_transpose=True,
                kernel_size=2,
                padding=0,
                name="conv",
                norm_type="rms_norm",
                eps=layer_norm_eps,
                elementwise_affine=ln_elementwise_affine,
                bias=use_bias,
                interpolate=False,
            )
        else:
            self.upsample = None

    def forward(self, x, pooled_text_emb, encoder_hidden_states, cross_attention_kwargs):
        if self.downsample is not None:
            x = self.downsample(x)

        for res_block, attention_block in zip(self.res_blocks, self.attention_blocks):
            x = res_block(x, pooled_text_emb)

            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, height * width).permute(0, 2, 1)
            x = attention_block(
                x, encoder_hidden_states=encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs
            )
            x = x.permute(0, 2, 1).view(batch_size, channels, height, width)

        if self.upsample is not None:
            x = self.upsample(x)

        return x

class Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = False #True 
    # Due to NotImplementedError: DDPOptimizer backend: Found a higher order op in the graph. This is not supported. Please turn off DDP optimizer using torch._dynamo.config.optimize_ddp=False. Note that this can cause performance degradation because there will be one bucket for the entire Dynamo graph. 
    # Please refer to this issue - https://github.com/pytorch/pytorch/issues/104674.
    _no_split_modules = ["TransformerBlock", "SingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False, # unused in our implementation
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        vocab_size: int = 8256,
        codebook_size: int = 8192,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim # 64 * 16 = 1024 # 24 * 128 = 3072  # 6 * 128 = 768; 12 * 64 = 768

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        # self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        # self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        # self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        # add-on to flux from amused
        in_channels_embed = self.inner_dim #768
        # block_out_channels = self.inner_dim #768
        # vocab_size = 8256
        # codebook_size = 8192
        ln_elementwise_affine = True
        layer_norm_eps = 1e-06
        use_bias = False
        micro_cond_embed_dim = 1280
        # cond_embed_dim = 1024 #768 #256
        # hidden_size = 1024 #768 #1024
        self.embed = UVit2DConvEmbed(
            in_channels_embed, self.inner_dim, self.config.vocab_size, ln_elementwise_affine, layer_norm_eps, use_bias
        )
        self.mlm_layer = ConvMlmLayer(
            self.inner_dim, in_channels_embed, use_bias, ln_elementwise_affine, layer_norm_eps, self.config.codebook_size
        )
        self.cond_embed = TimestepEmbedding( #1024+1280=2304 -> 1024
            micro_cond_embed_dim + self.config.pooled_projection_dim, self.inner_dim, sample_proj_bias=use_bias
        )
        self.encoder_proj_layer_norm = RMSNorm(self.inner_dim, layer_norm_eps, ln_elementwise_affine)
        # self.encoder_proj_layer_norm = AdaLayerNormContinuous(hidden_size, hidden_size, ln_elementwise_affine,layer_norm_eps)
        self.project_to_hidden_norm = RMSNorm(in_channels_embed, layer_norm_eps, ln_elementwise_affine)
        # self.project_to_hidden_norm = AdaLayerNormContinuous(hidden_size, hidden_size, ln_elementwise_affine,layer_norm_eps)
       
        self.project_to_hidden = nn.Linear(in_channels_embed, self.inner_dim, bias=use_bias)
        self.project_from_hidden_norm = RMSNorm(self.inner_dim, layer_norm_eps, ln_elementwise_affine)
        # self.project_from_hidden_norm = AdaLayerNormContinuous(hidden_size, hidden_size, ln_elementwise_affine,layer_norm_eps)
       
        self.project_from_hidden = nn.Linear(self.inner_dim, in_channels_embed, bias=use_bias)
        # num_res_blocks = 3
        # hidden_dropout = 0.0
        # block_num_heads = 12
        # attention_dropout = 0.0
        
        self.down_block = Simple_UVitBlock(
            self.inner_dim, #block_out_channels,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            downsample,
            False,
        )
        # self.mid_block = UVitBlock(
        #     in_channels, #block_out_channels,
        #     num_res_blocks,
        #     hidden_size,
        #     hidden_dropout,
        #     ln_elementwise_affine,
        #     layer_norm_eps,
        #     use_bias,
        #     block_num_heads,
        #     attention_dropout,
        #     downsample,
        #     False,
        # )
        self.up_block = Simple_UVitBlock(
            self.inner_dim, #block_out_channels,
            ln_elementwise_affine,
            layer_norm_eps,
            use_bias,
            False,
            upsample=upsample,
        )
        # add-on to flux from amused

        # self.fuse_qkv_projections()

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples= None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        micro_conds: torch.Tensor = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # original input for 256x256 images
        # hidden_states: [2, 16, 16]
        # encoder_hidden_states: [2, 512, 768]
        # pooled_projections: [2, 768]
        # micro_conds: [2, 5]
        # self.fuse_qkv_projections()
        # my add-on to flux
        from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
        micro_cond_encode_dim = 256 # same as self.config.micro_cond_encode_dim = 256 from amused
        micro_cond_embeds = get_timestep_embedding(
            micro_conds.flatten(), micro_cond_encode_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        ) # shape of micro_cond_embeds [10, 256]
        micro_cond_embeds = micro_cond_embeds.reshape((hidden_states.shape[0], -1)) # shape of micro_cond_embeds [2, 1280]  

        pooled_projections = torch.cat([pooled_projections, micro_cond_embeds], dim=1) # [2, 768] [2, 1280] -> [2, 2048] # [2, 1024] [2, 1280] -> [2, 2304]
        pooled_projections = pooled_projections.to(dtype=self.dtype)
        pooled_projections = self.cond_embed(pooled_projections).to(encoder_hidden_states.dtype)    
        # temb = pooled_projections # [2, 1024] 

        hidden_states = self.embed(hidden_states) # output will be [2,768,16,16] # [2, 1024, 16, 16]

        encoder_hidden_states = self.context_embedder(encoder_hidden_states) # 1024 -> 1024
        # should we add norm here? next line is copied from amused
        encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states) # to figure out how many parameters here

        # hidden_states = self.down_block(
        #     hidden_states,
        #     pooled_text_emb=pooled_projections,
        #     encoder_hidden_states=encoder_hidden_states,
        #     cross_attention_kwargs=None,
        # )
        hidden_states = self.down_block(hidden_states)

        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        hidden_states = self.project_to_hidden_norm(hidden_states) # norm first or proj first?
        hidden_states = self.project_to_hidden(hidden_states)

       
        
        # my add-on to flux


        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        # hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (      # flux seems 768 -> 768
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        ) # self.inner_dim -> self.config.pooled_projection_dim
        # encoder_hidden_states = self.context_embedder(encoder_hidden_states) 
        # # should we add norm here? next line is copied from amused
        # encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]
            
        # import pdb
        # pdb.set_trace()
        try:
            ids = torch.cat((txt_ids, img_ids), dim=0)
        except:
            import pdb
            pdb.set_trace()

            # print("here ids",ids.shape) #here ids torch.Size([333, 3]) uploaded

        image_rotary_emb = self.pos_embed(ids) #a tuple
        # print("here image_rotary_emb",image_rotary_emb[0].shape) #here image_rotary_emb torch.Size([333, 128])
        # image_rotary_emb = None

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                # self.mid_block(
                #     hidden_states,
                #     pooled_text_emb=pooled_projections,
                #     encoder_hidden_states=encoder_hidden_states,
                #     cross_attention_kwargs=None,
                # )
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,  
                    image_rotary_emb=image_rotary_emb, #
                )
                

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...] 

        # hidden_states = self.norm_out(hidden_states, temb) # [2, 256, 768]
        # add-on from amused
        hidden_states = self.project_from_hidden_norm(hidden_states) # should we norm here?
        hidden_states = self.project_from_hidden(hidden_states)
        # add-on from amused

        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)

        hidden_states = self.up_block(hidden_states)
        
        # output = self.proj_out(hidden_states)
        # hidden_states = self.up_block(
        #     hidden_states,
        #     pooled_text_emb=pooled_projections,
        #     encoder_hidden_states=encoder_hidden_states,
        #     cross_attention_kwargs=None,
        # )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)
        
        output = self.mlm_layer(hidden_states)
        # self.unfuse_qkv_projections()
        if not return_dict:
            return (output,)

        # return self.mlm_layer(Transformer2DModelOutput(sample=output))
        return output
        

# testModel = Transformer2DModel(
#                 patch_size = 1,
#                 in_channels = 64,
#                 num_layers = 14, #9, #flux: 19
#                 num_single_layers = 28, #18, # flux:38
#                 attention_head_dim = 128, #64, #64,
#                 num_attention_heads = 8, #12, # flux: 16
#                 joint_attention_dim = 768, #4096,
#                 pooled_projection_dim = 768,
#                 guidance_embeds = False,
#                 axes_dims_rope = (16, 56, 56),
#             )
# testModel = Transformer2DModel( # Total parameters: 995143680
#                 patch_size = 1,
#                 in_channels = 64,
#                 num_layers = 14, #9, #flux: 19
#                 num_single_layers = 28, #18, # flux:38
#                 attention_head_dim = 128, #64, #64,
#                 num_attention_heads = 8, #16, #12, # flux: 16
#                 joint_attention_dim = 1024, #4096,
#                 pooled_projection_dim = 1024,
#                 guidance_embeds = False,
#                 axes_dims_rope = (16, 56, 56),
#             ).cuda().half()
# total_params = sum(p.numel() for p in testModel.parameters())
# trainable_params = sum(p.numel() for p in testModel.parameters() if p.requires_grad)
# non_trainable_params = total_params - trainable_params

# print(f"Total parameters: {total_params}")
# print(f"Trainable parameters: {trainable_params}")
# print(f"Non-trainable (frozen) parameters: {non_trainable_params}")

# model_input = torch.ones([2,16,16]).long().cuda()#.half()
# micro_conds = torch.ones([2,5]).cuda().half()
# pooled_text_emb = torch.ones([2,1024]).cuda().half() #torch.ones([2,768])
# encoder_hidden_states = torch.ones([2,77,1024]).cuda().half()#torch.ones([2,77,768])
# # input_ids torch.Size([2, 16, 16])oaded
# # encoder_hidden_states torch.Size([2, 512, 768])
# # pooled_text_emb torch.Size([2, 728])
# # micro_conds torch.Size([2, 5])
# txt_ids = torch.ones([77,3]).cuda().half()
# img_ids = torch.ones([256,3]).cuda().half()

# import time 

# for i in range(1000):
#     if i==100:
#         begin = time.time()
# # forward(self, input_ids, encoder_hidden_states, pooled_text_emb, micro_conds, cross_attention_kwargs=None):
#     out = testModel(#hidden_states= model_input, encoder_hidden_states= encoder_hidden_states,pooled_projections=pooled_text_emb ,micro_conds = micro_conds,
#         hidden_states=model_input, # should be (batch size, channel, height, width)
#         encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
#         micro_conds=micro_conds, # 
#         pooled_projections=pooled_text_emb, # should be (batch_size, projection_dim)
#         img_ids = img_ids,
#         txt_ids = txt_ids,        
#     )
# print(out.shape)
# print((time.time()-begin)/900)

