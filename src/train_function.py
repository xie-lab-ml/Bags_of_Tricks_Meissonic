import torch
from torch.utils.data import Dataset
import json
import csv
import os, PIL, math
from PIL import Image
import torchvision.transforms as transforms
from PIL.ImageOps import exif_transpose

def load_prompt(path):
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.strip().split(', ')
            new_prompt = ""
            for i in range(1, len(parts)-1):
                if i == len(parts)-2:
                    new_prompt += parts[i]
                else:
                    new_prompt += parts[i] + ", "
            prompts.append(new_prompt)
    tmp_prompt_list = []
    for prompt in prompts:
        if prompt != "":
            tmp_prompt_list.append(prompt)
    prompts = tmp_prompt_list
    return prompts


def load_single_image_list(file_path):
    image_list = os.listdir(file_path)
    res = []
    for idx in range(len(image_list)):
        res.append(Image.open(os.path.join(file_path, f'{idx}.png')))
    return res

def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    # latent_image_ids = latent_image_ids.unsqueeze(0).repeat(batch_size, 1, 1)

    return latent_image_ids.to(device=device, dtype=dtype)

def process_image(image, size, Norm=False, hps_score = 6.0): 
    image = exif_transpose(image)

    if not image.mode == "RGB":
        image = image.convert("RGB")

    orig_height = image.height
    orig_width = image.width

    image = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)(image)

    c_top, c_left, _, _ = transforms.RandomCrop.get_params(image, output_size=(size, size))
    image = transforms.functional.crop(image, c_top, c_left, size, size)
    image = transforms.ToTensor()(image)

    if Norm:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)

    micro_conds = torch.tensor(
        [orig_width, orig_height, c_top, c_left, hps_score],
    )

    return {"image": image, "micro_conds": micro_conds}


@torch.no_grad()
def tokenize_prompt(tokenizer, prompt, text_encoder_architecture='open_clip'): # only support open_clip and CLIP
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        return tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).input_ids
    elif text_encoder_architecture == 'CLIP_T5_base': # we have two tokenizers, 1st for CLIP, 2nd for T5
        input_ids = []
        input_ids.append(tokenizer[0](
            prompt,
            truncation=True,
            padding="max_length",
            max_length=77,
            return_tensors="pt",
        ).input_ids)
        input_ids.append(tokenizer[1](
            prompt,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).input_ids)
        return input_ids
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")
    
def encode_prompt(text_encoder, input_ids, text_encoder_architecture='open_clip'):  # only support open_clip and CLIP
    if text_encoder_architecture == 'CLIP' or text_encoder_architecture == 'open_clip':
        outputs = text_encoder(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs.hidden_states[-2]
        cond_embeds = outputs[0]
        return encoder_hidden_states, cond_embeds

    elif text_encoder_architecture == 'CLIP_T5_base':
        outputs_clip = text_encoder[0](input_ids=input_ids[0], return_dict=True, output_hidden_states=True)
        outputs_t5 = text_encoder[1](input_ids=input_ids[1], decoder_input_ids=torch.zeros_like(input_ids[1]),
                               return_dict=True, output_hidden_states=True)
        encoder_hidden_states = outputs_t5.encoder_hidden_states[-2]
        cond_embeds = outputs_clip[0]
        return encoder_hidden_states, cond_embeds
    else:
        raise ValueError(f"Unknown text_encoder_architecture: {text_encoder_architecture}")

class MeissonicDataset(Dataset):
    def __init__(self, tokenizer):
        
        self.tokenizer = tokenizer
        self.promptpath_list = [...] # TODO
        self.datapath_list = [...] # TODO
        
        self.prompts = []
        self.images = []
        for promptpath, datapath in zip(self.promptpath_list, self.datapath_list):
            self.prompts.extend(load_prompt(promptpath))
            self.images.extend(load_single_image_list(datapath))
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt, image = self.prompts[idx], self.images[idx]
        image = process_image(image, 1024, Norm=False)
        image["prompt_input_ids"] = tokenize_prompt(self.tokenizer, prompt, "open_clip")
        return image
    

@torch.no_grad()
def pre_process_batch(batch, model, vq_model, accelerator, weight_dtype=torch.float16):
    micro_conds = batch["micro_conds"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
    pixel_values = batch["image"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True)
    batch_size = pixel_values.shape[0]
    split_batch_size = 1
    num_splits = math.ceil(batch_size / split_batch_size)
    image_tokens = []
    for i in range(num_splits):
        start_idx = i * split_batch_size
        end_idx = min((i + 1) * split_batch_size, batch_size)
        image_tokens.append(
            vq_model.quantize(vq_model.encode(pixel_values[start_idx:end_idx]).latents)[2][2].reshape(
                split_batch_size, -1
            )
        )
    image_tokens = torch.cat(image_tokens, dim=0)
    batch_size, seq_len = image_tokens.shape

    timesteps = torch.rand(batch_size, device=image_tokens.device)
    mask_prob = torch.cos(timesteps * math.pi * 0.5)
    mask_prob = mask_prob.clip(0.0)
    num_token_masked = (seq_len * mask_prob).round().clamp(min=1)
    batch_randperm = torch.rand(batch_size, seq_len, device=image_tokens.device).argsort(dim=-1)
    mask = batch_randperm < num_token_masked.unsqueeze(-1)
    mask_id = accelerator.unwrap_model(model).config.vocab_size - 1
    input_ids = torch.where(mask, mask_id, image_tokens)
    labels = torch.where(mask, image_tokens, -100)
    return input_ids, labels, mask_prob, mask_id, micro_conds, timesteps

def scale_list():
    name_scale = [
        ('time_text_embed.timestep_embedder.linear_1', 0.00159454345703125),
        ('time_text_embed.timestep_embedder.linear_2', 0.0020904541015625),
        ('cond_embed.linear_2', 0.003875732421875),
        ('time_text_embed.text_embedder.linear_2', 0.005157470703125),
        ('transformer_blocks.1.norm1_context.linear', 0.012451171875),
        ('transformer_blocks.2.norm1.linear', 0.012939453125),
        ('transformer_blocks.3.norm1_context.linear', 0.01318359375),
        ('transformer_blocks.5.norm1_context.linear', 0.01318359375),
        ('transformer_blocks.0.norm1_context.linear', 0.013671875),
        ('transformer_blocks.4.norm1_context.linear', 0.01373291015625),
        ('single_transformer_blocks.10.norm.linear', 0.01409912109375),
        ('single_transformer_blocks.1.norm.linear', 0.01434326171875),
        ('single_transformer_blocks.4.norm.linear', 0.014404296875),
        ('single_transformer_blocks.5.norm.linear', 0.014404296875),
        ('transformer_blocks.6.norm1_context.linear', 0.01446533203125),
        ('single_transformer_blocks.8.norm.linear', 0.0146484375),
        ('single_transformer_blocks.0.norm.linear', 0.014892578125),
        ('single_transformer_blocks.2.norm.linear', 0.01495361328125),
        ('transformer_blocks.1.attn.to_v', 0.01556396484375),
        ('transformer_blocks.9.norm1.linear', 0.01556396484375),
        ('transformer_blocks.9.norm1_context.linear', 0.01556396484375),
        ('single_transformer_blocks.9.norm.linear', 0.015625),
        ('single_transformer_blocks.11.norm.linear', 0.015869140625),
        ('single_transformer_blocks.3.norm.linear', 0.0159912109375),
        ('transformer_blocks.10.norm1.linear', 0.017578125),
        ('transformer_blocks.10.norm1_context.linear', 0.017578125),
        ('single_transformer_blocks.7.norm.linear', 0.017578125),
        ('single_transformer_blocks.13.norm.linear', 0.017578125),
        ('single_transformer_blocks.14.norm.linear', 0.017578125),
        ('single_transformer_blocks.19.norm.linear', 0.017578125),
        ('transformer_blocks.8.norm1.linear', 0.01806640625),
        ('transformer_blocks.5.norm1.linear', 0.0185546875),
        ('transformer_blocks.7.norm1.linear', 0.018798828125),
        ('transformer_blocks.6.norm1.linear', 0.01904296875),
        ('project_to_hidden', 0.0194091796875),
        ('transformer_blocks.0.attn.to_v', 0.0194091796875),
        ('single_transformer_blocks.18.norm.linear', 0.0205078125),
        ('transformer_blocks.2.norm1_context.linear', 0.02099609375),
        ('single_transformer_blocks.16.norm.linear', 0.0213623046875),
        ('transformer_blocks.2.attn.to_v', 0.021728515625),
        ('transformer_blocks.0.attn.to_k', 0.02197265625),
        ('transformer_blocks.3.attn.to_v', 0.02197265625),
        ('transformer_blocks.1.norm1.linear', 0.0224609375),
        ('single_transformer_blocks.17.norm.linear', 0.0224609375),
        ('single_transformer_blocks.21.norm.linear', 0.0224609375),
        ('transformer_blocks.4.attn.to_v', 0.022705078125),
        ('transformer_blocks.3.attn.to_out.0', 0.0228271484375),
        ('single_transformer_blocks.20.norm.linear', 0.02294921875),
        ('transformer_blocks.5.attn.to_add_out', 0.023681640625),
        ('transformer_blocks.3.norm1.linear', 0.0240478515625),
        ('transformer_blocks.6.ff_context.net.2', 0.02490234375),
        ('single_transformer_blocks.23.norm.linear', 0.02587890625),
        ('single_transformer_blocks.27.attn.to_v', 0.026123046875),
        ('transformer_blocks.4.norm1.linear', 0.0262451171875),
        ('transformer_blocks.12.norm1_context.linear', 0.0263671875),
        ('single_transformer_blocks.22.norm.linear', 0.02685546875),
        ('transformer_blocks.3.attn.to_add_out', 0.0277099609375),
        ('transformer_blocks.0.attn.to_q', 0.02783203125),
        ('transformer_blocks.5.attn.to_v', 0.02783203125),
        ('transformer_blocks.6.attn.to_v', 0.0283203125),
        ('transformer_blocks.7.attn.to_add_out', 0.0283203125),
        ('transformer_blocks.0.norm1.linear', 0.0286865234375),
        ('transformer_blocks.1.attn.to_q', 0.0291748046875),
        ('transformer_blocks.1.attn.to_out.0', 0.03173828125),
        ('transformer_blocks.6.attn.to_out.0', 0.03173828125),
        ('transformer_blocks.1.attn.to_k', 0.031982421875),
        ('transformer_blocks.3.attn.to_k', 0.0322265625),
        ('transformer_blocks.4.ff_context.net.2', 0.0322265625),
        ('transformer_blocks.2.ff_context.net.2', 0.033203125),
        ('transformer_blocks.7.attn.to_out.0', 0.033935546875),
        ('single_transformer_blocks.24.norm.linear', 0.033935546875),
        ('transformer_blocks.2.attn.to_add_out', 0.0341796875),
        ('transformer_blocks.4.attn.to_out.0', 0.03466796875),
        ('transformer_blocks.7.attn.to_v', 0.03466796875),
        ('single_transformer_blocks.12.attn.to_v', 0.034912109375),
        ('single_transformer_blocks.25.norm.linear', 0.035400390625),
        ('transformer_blocks.2.attn.to_q', 0.03564453125),
        ('transformer_blocks.2.attn.to_k', 0.03564453125),
        ('single_transformer_blocks.17.attn.to_v', 0.03564453125),
        ('transformer_blocks.9.attn.to_v', 0.035888671875),
        ('transformer_blocks.10.attn.to_add_out', 0.035888671875),
        ('transformer_blocks.4.attn.to_q', 0.0361328125),
        ('transformer_blocks.0.attn.to_add_out', 0.036376953125),
        ('single_transformer_blocks.26.attn.to_v', 0.036376953125),
        ('transformer_blocks.3.ff_context.net.2', 0.03759765625),
        ('transformer_blocks.8.attn.to_add_out', 0.03759765625),
        ('single_transformer_blocks.22.attn.to_v', 0.03759765625),
        ('transformer_blocks.13.norm1_context.linear', 0.0380859375),
        ('single_transformer_blocks.25.attn.to_v', 0.038818359375),
        ('project_from_hidden', 0.039794921875),
        ('single_transformer_blocks.21.attn.to_v', 0.039794921875),
        ('single_transformer_blocks.10.attn.to_v', 0.04052734375),
        ('single_transformer_blocks.4.attn.to_v', 0.041259765625),
        ('single_transformer_blocks.11.attn.to_v', 0.04150390625),
        ('single_transformer_blocks.24.attn.to_v', 0.041748046875),
        ('transformer_blocks.6.attn.to_add_out', 0.0419921875),
        ('single_transformer_blocks.7.attn.to_v', 0.0419921875),
        ('transformer_blocks.5.attn.to_out.0', 0.04248046875),
        ('transformer_blocks.11.attn.to_v', 0.04248046875),
        ('single_transformer_blocks.26.norm.linear', 0.042724609375),
        ('single_transformer_blocks.9.attn.to_v', 0.04296875),
        ('single_transformer_blocks.23.attn.to_v', 0.04296875),
        ('transformer_blocks.3.attn.to_q', 0.043212890625),
        ('single_transformer_blocks.8.attn.to_v', 0.04345703125),
        ('transformer_blocks.5.attn.to_k', 0.0439453125),
        ('single_transformer_blocks.5.attn.to_v', 0.04443359375),
        ('single_transformer_blocks.3.attn.to_v', 0.045166015625),
        ('transformer_blocks.3.attn.add_v_proj', 0.04541015625),
        ('single_transformer_blocks.19.attn.to_v', 0.0458984375),
        ('single_transformer_blocks.6.attn.to_v', 0.046875),
        ('transformer_blocks.11.attn.to_add_out', 0.04736328125),
        ('single_transformer_blocks.14.attn.to_v', 0.04736328125),
        ('single_transformer_blocks.0.attn.to_v', 0.04833984375),
        ('single_transformer_blocks.18.attn.to_v', 0.04833984375),
        ('transformer_blocks.5.ff_context.net.2', 0.048828125),
        ('single_transformer_blocks.16.attn.to_v', 0.048828125),
        ('transformer_blocks.0.attn.to_out.0', 0.05029296875),
        ('transformer_blocks.10.attn.to_v', 0.05029296875),
        ('single_transformer_blocks.12.proj_mlp', 0.05029296875),
        ('time_text_embed.text_embedder.linear_1', 0.050537109375),
        ('transformer_blocks.5.attn.add_v_proj', 0.050537109375),
        ('single_transformer_blocks.24.proj_mlp', 0.050537109375),
        ('transformer_blocks.2.attn.add_v_proj', 0.05078125),
        ('transformer_blocks.7.attn.to_q', 0.05078125),
        ('transformer_blocks.2.attn.to_out.0', 0.051513671875),
        ('transformer_blocks.9.ff_context.net.2', 0.0517578125),
        ('single_transformer_blocks.20.attn.to_v', 0.0517578125),
        ('transformer_blocks.12.attn.to_add_out', 0.052001953125),
        ('transformer_blocks.4.attn.add_v_proj', 0.052734375),
        ('transformer_blocks.4.attn.to_add_out', 0.052978515625),
        ('single_transformer_blocks.13.attn.to_v', 0.0537109375),
        ('transformer_blocks.10.attn.to_out.0', 0.05419921875),
        ('transformer_blocks.1.ff.net.2', 0.0546875),
        ('transformer_blocks.11.ff_context.net.2', 0.0546875),
        ('single_transformer_blocks.27.norm.linear', 0.0546875),
        ('transformer_blocks.3.attn.add_q_proj', 0.054931640625),
        ('transformer_blocks.8.attn.to_out.0', 0.054931640625),
        ('single_transformer_blocks.1.attn.to_v', 0.0556640625),
        ('transformer_blocks.7.attn.add_v_proj', 0.056640625),
        ('transformer_blocks.13.ff.net.0.proj', 0.056640625),
        ('single_transformer_blocks.6.proj_mlp', 0.056884765625),
        ('transformer_blocks.9.attn.to_out.0', 0.05712890625),
        ('single_transformer_blocks.1.proj_out', 0.05712890625),
        ('single_transformer_blocks.4.proj_out', 0.057861328125),
        ('transformer_blocks.4.ff.net.2', 0.05810546875),
        ('single_transformer_blocks.8.proj_mlp', 0.05810546875),
        ('transformer_blocks.6.ff_context.net.0.proj', 0.058349609375),
        ('single_transformer_blocks.15.attn.to_v', 0.058349609375),
        ('single_transformer_blocks.27.proj_mlp', 0.058837890625),
        ('transformer_blocks.9.ff.net.0.proj', 0.05908203125),
        ('transformer_blocks.12.attn.to_out.0', 0.05908203125),
        ('transformer_blocks.9.ff_context.net.0.proj', 0.059326171875),
        ('transformer_blocks.3.ff_context.net.0.proj', 0.0595703125),
        ('transformer_blocks.7.ff.net.0.proj', 0.0595703125),
        ('transformer_blocks.11.ff_context.net.0.proj', 0.0595703125),
        ('transformer_blocks.12.ff.net.0.proj', 0.06005859375),
        ('single_transformer_blocks.10.proj_mlp', 0.06005859375),
        ('single_transformer_blocks.0.proj_mlp', 0.060302734375),
        ('transformer_blocks.4.ff_context.net.0.proj', 0.060546875),
        ('transformer_blocks.6.ff.net.0.proj', 0.060546875),
        ('transformer_blocks.8.ff.net.0.proj', 0.060546875),
        ('transformer_blocks.11.ff.net.0.proj', 0.060791015625),
        ('single_transformer_blocks.4.proj_mlp', 0.06103515625),
        ('transformer_blocks.6.attn.add_v_proj', 0.061279296875),
        ('transformer_blocks.10.ff.net.0.proj', 0.0615234375),
        ('single_transformer_blocks.2.attn.to_v', 0.0615234375),
        ('single_transformer_blocks.14.proj_mlp', 0.0615234375),
        ('transformer_blocks.11.attn.to_out.0', 0.061767578125),
        ('transformer_blocks.2.ff_context.net.0.proj', 0.06201171875),
        ('transformer_blocks.5.ff_context.net.0.proj', 0.06201171875),
        ('transformer_blocks.6.ff.net.2', 0.06201171875),
        ('transformer_blocks.12.attn.to_v', 0.06201171875),
        ('single_transformer_blocks.7.proj_out', 0.062255859375),
        ('transformer_blocks.5.attn.add_q_proj', 0.0625),
        ('transformer_blocks.6.attn.to_q', 0.0625),
        ('transformer_blocks.9.attn.add_v_proj', 0.0625),
        ('single_transformer_blocks.22.proj_mlp', 0.0625),
        ('transformer_blocks.1.attn.to_add_out', 0.06298828125),
        ('transformer_blocks.2.ff.net.0.proj', 0.0634765625),
        ('single_transformer_blocks.5.proj_mlp', 0.06396484375),
        ('single_transformer_blocks.6.proj_out', 0.064453125),
        ('single_transformer_blocks.27.attn.to_k', 0.06494140625),
        ('single_transformer_blocks.1.proj_mlp', 0.0654296875),
        ('single_transformer_blocks.2.proj_mlp', 0.0654296875),
        ('transformer_blocks.0.ff.net.2', 0.06640625),
        ('transformer_blocks.1.ff_context.net.0.proj', 0.06689453125),
        ('single_transformer_blocks.12.proj_out', 0.06689453125),
        ('single_transformer_blocks.27.attn.to_q', 0.06689453125),
        ('transformer_blocks.8.attn.to_k', 0.0673828125),
        ('transformer_blocks.10.ff_context.net.0.proj', 0.06787109375),
        ('single_transformer_blocks.8.proj_out', 0.06787109375),
        ('transformer_blocks.0.ff_context.net.0.proj', 0.068359375),
        ('transformer_blocks.5.ff.net.0.proj', 0.068359375),
        ('transformer_blocks.7.ff.net.2', 0.06884765625),
        ('transformer_blocks.8.attn.add_v_proj', 0.0693359375),
        ('transformer_blocks.9.attn.to_q', 0.0693359375),
        ('transformer_blocks.11.ff.net.2', 0.0703125),
        ('transformer_blocks.6.attn.to_k', 0.07080078125),
        ('single_transformer_blocks.25.proj_mlp', 0.072265625),
        ('single_transformer_blocks.12.attn.to_q', 0.07275390625),
        ('single_transformer_blocks.3.proj_mlp', 0.0732421875),
        ('single_transformer_blocks.11.proj_mlp', 0.0732421875),
        ('single_transformer_blocks.7.proj_mlp', 0.07421875),
        ('single_transformer_blocks.5.proj_out', 0.07470703125),
        ('single_transformer_blocks.11.attn.to_q', 0.07470703125),
        ('transformer_blocks.3.ff.net.0.proj', 0.0751953125),
        ('transformer_blocks.5.ff.net.2', 0.07568359375),
        ('transformer_blocks.7.attn.to_k', 0.076171875),
        ('transformer_blocks.6.attn.add_q_proj', 0.0771484375),
        ('transformer_blocks.13.ff.net.2', 0.0771484375),
        ('transformer_blocks.1.ff_context.net.2', 0.078125),
        ('transformer_blocks.11.attn.add_v_proj', 0.078125),
        ('transformer_blocks.1.attn.add_v_proj', 0.07861328125),
        ('transformer_blocks.12.ff.net.2', 0.0791015625),
        ('transformer_blocks.13.attn.add_v_proj', 0.0791015625),
        ('transformer_blocks.1.ff.net.0.proj', 0.080078125),
        ('transformer_blocks.4.attn.add_q_proj', 0.080078125),
        ('single_transformer_blocks.16.attn.to_k', 0.080078125),
        ('single_transformer_blocks.22.attn.to_k', 0.08056640625),
        ('single_transformer_blocks.2.proj_out', 0.0810546875),
        ('transformer_blocks.3.ff.net.2', 0.08154296875),
        ('single_transformer_blocks.10.proj_out', 0.08154296875),
        ('single_transformer_blocks.0.proj_out', 0.08203125),
        ('single_transformer_blocks.2.attn.to_q', 0.08203125),
        ('single_transformer_blocks.14.proj_out', 0.08203125),
        ('single_transformer_blocks.26.attn.to_k', 0.08251953125),
        ('transformer_blocks.13.attn.to_v', 0.0830078125),
        ('single_transformer_blocks.20.proj_mlp', 0.0830078125),
        ('single_transformer_blocks.3.proj_out', 0.08349609375),
        ('transformer_blocks.0.attn.add_v_proj', 0.083984375),
        ('transformer_blocks.0.attn.add_q_proj', 0.083984375),
        ('single_transformer_blocks.0.attn.to_q', 0.083984375),
        ('single_transformer_blocks.6.attn.to_q', 0.083984375),
        ('single_transformer_blocks.9.proj_mlp', 0.083984375),
        ('transformer_blocks.1.attn.add_q_proj', 0.0859375),
        ('transformer_blocks.12.ff_context.net.0.proj', 0.0859375),
        ('single_transformer_blocks.8.attn.to_q', 0.0859375),
        ('transformer_blocks.13.attn.to_out.0', 0.0869140625),
        ('single_transformer_blocks.4.attn.to_q', 0.0869140625),
        ('single_transformer_blocks.21.proj_mlp', 0.0869140625),
        ('single_transformer_blocks.3.attn.to_q', 0.08740234375),
        ('single_transformer_blocks.16.proj_mlp', 0.087890625),
        ('transformer_blocks.8.ff_context.net.0.proj', 0.0888671875),
        ('transformer_blocks.12.attn.add_v_proj', 0.0888671875),
        ('single_transformer_blocks.7.attn.to_q', 0.0888671875),
        ('transformer_blocks.9.ff.net.2', 0.08935546875),
        ('transformer_blocks.11.attn.to_q', 0.08984375),
        ('single_transformer_blocks.16.attn.to_q', 0.08984375),
        ('single_transformer_blocks.18.proj_out', 0.08984375),
        ('single_transformer_blocks.13.proj_mlp', 0.09033203125),
        ('transformer_blocks.10.attn.add_v_proj', 0.0908203125),
        ('single_transformer_blocks.25.attn.to_k', 0.0908203125),
        ('single_transformer_blocks.18.proj_mlp', 0.091796875),
        ('single_transformer_blocks.15.proj_out', 0.09228515625),
        ('transformer_blocks.8.ff.net.2', 0.09375),
        ('single_transformer_blocks.12.attn.to_k', 0.09375),
        ('single_transformer_blocks.26.attn.to_q', 0.0947265625),
        ('transformer_blocks.2.attn.add_k_proj', 0.09521484375),
        ('single_transformer_blocks.11.proj_out', 0.095703125),
        ('single_transformer_blocks.15.proj_mlp', 0.095703125),
        ('transformer_blocks.1.attn.add_k_proj', 0.09716796875),
        ('single_transformer_blocks.14.attn.to_k', 0.09765625),
        ('single_transformer_blocks.10.attn.to_q', 0.09814453125),
        ('single_transformer_blocks.5.attn.to_q', 0.09912109375),
        ('transformer_blocks.2.attn.add_q_proj', 0.099609375),
        ('transformer_blocks.7.attn.add_q_proj', 0.1005859375),
        ('single_transformer_blocks.21.attn.to_k', 0.1005859375),
        ('single_transformer_blocks.19.attn.to_q', 0.10107421875),
        ('transformer_blocks.8.attn.to_q', 0.1015625),
        ('single_transformer_blocks.22.attn.to_q', 0.1015625),
        ('single_transformer_blocks.23.proj_mlp', 0.10302734375),
        ('transformer_blocks.9.attn.to_add_out', 0.103515625),
        ('transformer_blocks.10.ff.net.2', 0.103515625),
        ('single_transformer_blocks.9.proj_out', 0.1044921875),
        ('single_transformer_blocks.20.attn.to_k', 0.1044921875),
        ('transformer_blocks.0.attn.add_k_proj', 0.10498046875),
        ('transformer_blocks.8.attn.add_q_proj', 0.107421875),
        ('transformer_blocks.9.attn.to_k', 0.107421875),
        ('single_transformer_blocks.26.proj_mlp', 0.107421875),
        ('transformer_blocks.13.attn.to_k', 0.10791015625),
        ('single_transformer_blocks.17.attn.to_k', 0.1083984375),
        ('single_transformer_blocks.18.attn.to_k', 0.1083984375),
        ('single_transformer_blocks.19.proj_mlp', 0.1103515625),
        ('single_transformer_blocks.4.attn.to_k', 0.111328125),
        ('single_transformer_blocks.8.attn.to_k', 0.11181640625),
        ('transformer_blocks.13.attn.to_q', 0.1123046875),
        ('single_transformer_blocks.14.attn.to_q', 0.1142578125),
        ('single_transformer_blocks.15.attn.to_q', 0.1142578125),
        ('single_transformer_blocks.16.proj_out', 0.1142578125),
        ('transformer_blocks.4.ff.net.0.proj', 0.115234375),
        ('single_transformer_blocks.17.proj_mlp', 0.1171875),
        ('single_transformer_blocks.19.attn.to_k', 0.1171875),
        ('single_transformer_blocks.9.attn.to_q', 0.11767578125),
        ('single_transformer_blocks.13.attn.to_k', 0.1181640625),
        ('single_transformer_blocks.2.attn.to_k', 0.11865234375),
        ('transformer_blocks.3.attn.add_k_proj', 0.11962890625),
        ('single_transformer_blocks.10.attn.to_k', 0.1220703125),
        ('single_transformer_blocks.11.attn.to_k', 0.1220703125),
        ('single_transformer_blocks.6.attn.to_k', 0.123046875),
        ('transformer_blocks.12.attn.to_q', 0.125),
        ('single_transformer_blocks.0.attn.to_k', 0.1259765625),
        ('transformer_blocks.0.ff.net.0.proj', 0.1279296875),
        ('transformer_blocks.10.attn.to_q', 0.1279296875),
        ('transformer_blocks.9.attn.add_k_proj', 0.130859375),
        ('single_transformer_blocks.13.proj_out', 0.130859375),
        ('single_transformer_blocks.23.attn.to_k', 0.1318359375),
        ('transformer_blocks.10.attn.to_k', 0.1328125),
        ('transformer_blocks.10.attn.add_q_proj', 0.134765625),
        ('transformer_blocks.12.attn.to_k', 0.134765625),
        ('single_transformer_blocks.5.attn.to_k', 0.1357421875),
        ('single_transformer_blocks.24.attn.to_k', 0.1357421875),
        ('single_transformer_blocks.15.attn.to_k', 0.13671875),
        ('single_transformer_blocks.22.proj_out', 0.13671875),
        ('transformer_blocks.11.attn.add_k_proj', 0.138671875),
        ('single_transformer_blocks.7.attn.to_k', 0.138671875),
        ('single_transformer_blocks.21.proj_out', 0.138671875),
        ('single_transformer_blocks.3.attn.to_k', 0.1396484375),
        ('transformer_blocks.11.attn.to_k', 0.140625),
        ('single_transformer_blocks.9.attn.to_k', 0.1416015625),
        ('transformer_blocks.13.attn.add_q_proj', 0.14453125),
        ('transformer_blocks.9.attn.add_q_proj', 0.146484375),
        ('single_transformer_blocks.19.proj_out', 0.146484375),
        ('single_transformer_blocks.21.attn.to_q', 0.146484375),
        ('single_transformer_blocks.25.attn.to_q', 0.1484375),
        ('single_transformer_blocks.23.attn.to_q', 0.150390625),
        ('single_transformer_blocks.24.attn.to_q', 0.15234375),
        ('transformer_blocks.4.attn.add_k_proj', 0.154296875),
        ('transformer_blocks.11.attn.add_q_proj', 0.1552734375),
        ('single_transformer_blocks.1.attn.to_k', 0.1552734375),
        ('transformer_blocks.2.ff.net.2', 0.15625),
        ('single_transformer_blocks.18.attn.to_q', 0.15625),
        ('single_transformer_blocks.20.attn.to_q', 0.158203125),
        ('single_transformer_blocks.25.proj_out', 0.1640625),
        ('single_transformer_blocks.17.proj_out', 0.166015625),
        ('transformer_blocks.5.attn.add_k_proj', 0.173828125),
        ('single_transformer_blocks.24.proj_out', 0.17578125),
        ('single_transformer_blocks.17.attn.to_q', 0.177734375),
        ('transformer_blocks.7.attn.add_k_proj', 0.1796875),
        ('transformer_blocks.13.attn.to_add_out', 0.193359375),
        ('single_transformer_blocks.23.proj_out', 0.19921875),
        ('transformer_blocks.10.ff_context.net.2', 0.22265625),
        ('transformer_blocks.8.ff_context.net.2', 0.2265625),
        ('transformer_blocks.12.attn.add_q_proj', 0.2265625),
        ('transformer_blocks.13.attn.add_k_proj', 0.2275390625),
        ('single_transformer_blocks.13.attn.to_q', 0.2421875),
        ('single_transformer_blocks.27.proj_out', 0.248046875),
        ('single_transformer_blocks.26.proj_out', 0.26171875),
        ('transformer_blocks.6.attn.add_k_proj', 0.29296875),
        ('single_transformer_blocks.20.proj_out', 0.3125),
        ('transformer_blocks.8.attn.add_k_proj', 0.32421875),
        ('transformer_blocks.7.ff_context.net.2', 0.33984375),
        ('transformer_blocks.12.attn.add_k_proj', 0.375),
        ('transformer_blocks.10.attn.add_k_proj', 0.41796875),
        ('transformer_blocks.13.ff_context.net.0.proj', 0.421875),
        ('cond_embed.linear_1', 0.4609375),
        ('context_embedder', 0.53125),
        ('transformer_blocks.0.ff_context.net.2', 0.5390625),
        ('transformer_blocks.13.ff_context.net.2', 11.5),
    ]
    return name_scale