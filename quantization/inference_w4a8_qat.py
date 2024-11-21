import os
import sys
import math
sys.path.append("./")
sys.path.append("../")

import numpy as np
import torch
import csv
import torch.nn.functional as F
from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from diffusers import VQModel
import time
import argparse
import re, json
import torch._dynamo 
torch._dynamo.config.suppress_errors = True
from optimum.quanto import freeze, qfloat8, quantize, QTensor, qint8, requantize, qint4, \
    Calibration, qint2, set_input_scale, QModuleMixin
from torch.utils.data import DataLoader
device = 'cuda'
from bitsandbytes.optim import AdamW8bit
from src.train_function import MeissonicDataset, _prepare_latent_image_ids, encode_prompt, \
    tokenize_prompt, pre_process_batch, scale_list
    
import torchvision.transforms as transforms
from accelerate import Accelerator

def load_models():
    model_path = "MeissonFlow/Meissonic"
    dtype = torch.bfloat16
    model = Transformer2DModel.from_pretrained(model_path, subfolder="transformer", torch_dtype=dtype)
    vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", torch_dtype=dtype)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=dtype
    )
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", torch_dtype=dtype)
    
    include_modules = []
    if not os.path.exists("./quantized/transformer_f8/model.safetensors"):
        include_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                include_modules.append(name)
        quantize(model, weights=qint4, activations=qint8, include=include_modules)
        scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler).to(device)
        pipe.transformer = calibration_for_f8(pipe)
        print("Finetuning for f8, begin...")
        model.to(dtype=torch.float32)
        pipe.vqvae.to(dtype=torch.float32)
        pipe.text_encoder.to(dtype=torch.float32)
        pipe.transformer = finetune_for_f8(pipe)
        print("Finetuning for f8, end...")
        pipe.transformer.eval()
        model.to(dtype=torch.bfloat16)
        pipe.vqvae.to(dtype=torch.bfloat16)
        pipe.text_encoder.to(dtype=torch.bfloat16)
        freeze(pipe.transformer)
        from safetensors.torch import save_file
        save_file(pipe.transformer.state_dict(), './quantized/transformer_f8/model.safetensors')
        return pipe
    else:
        scheduler = Scheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = Pipeline(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler).to(device)
        from safetensors.torch import load_file
        state_dict = load_file('./quantized/transformer_f8/model.safetensors')
        with open('./quantized/quantization_map.json', "r") as f:
            quantization_map = json.load(f)
        name_scale = scale_list()
        name_scale = {item[0]: item[1] for item in name_scale[:len(name_scale)//3]}
        count = 0
        for key, value in quantization_map.items():
            if key in name_scale.keys():
                quantization_map[key]["activations"] = "qint8"
        
        requantize(pipe.transformer, dict(), quantization_map, device=torch.device('cuda'))        
        pipe.transformer = calibration_for_f8(pipe)
        freeze(pipe.transformer)
        # from safetensors.torch import save_file
        # if not os.path.exists("./quantized/transformer_after"):
        #     os.makedirs("./quantized/transformer_after")
        # save_file(pipe.transformer.state_dict(), './quantized/transformer_after/model.safetensors')
        return pipe

def define_accelerator():
    accelerator = Accelerator(mixed_precision="no",
                              gradient_accumulation_steps=32)
    return accelerator


def finetune_for_f8(pipe):
    pipe.transformer.train()
    weight_dtype = torch.float16
    pipe.transformer.requires_grad_(True)
    pipe.vqvae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder = pipe.text_encoder.to(weight_dtype)
    pipe.vqvae = pipe.vqvae.to(weight_dtype)
    model = pipe.transformer
    model.train()
    accelerator = define_accelerator()
    optimizer = AdamW8bit(model.parameters(), lr=5e-6, weight_decay=0.01)
    train_dataset = MeissonicDataset(tokenizer=pipe.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    train_loader, optimizer, model = accelerator.prepare(train_loader, optimizer, model)
    from optimum.quanto import Calibration

    torch.cuda.empty_cache()
    
    for batch_idx, (batch) in enumerate(train_loader):
        # if batch_idx > 258:
        #     break
        input_ids, labels, mask_prob, mask_id, micro_conds, timesteps = pre_process_batch(batch, model, pipe.vqvae, accelerator, weight_dtype=weight_dtype)
        input_ids = input_ids
        mask_prob = mask_prob.float()
        micro_conds = micro_conds.float()
        
        if "prompt_input_ids" in batch:
            with torch.no_grad():
                encoder_hidden_states, cond_embeds = encode_prompt(
                        pipe.text_encoder, batch["prompt_input_ids"].to(accelerator.device, non_blocking=True), "open_clip"
                    )
                encoder_hidden_states = encoder_hidden_states.float()
                cond_embeds = cond_embeds.float()
                
        bs = input_ids.shape[0]
        vae_scale_factor = 2 ** (len(pipe.vqvae.config.block_out_channels) - 1)
        resolution = 1024 // vae_scale_factor
        input_ids = input_ids.reshape(bs, resolution, resolution)
        
        with accelerator.accumulate(model):
            codebook_size = accelerator.unwrap_model(model).config.codebook_size
            img_ids = _prepare_latent_image_ids(input_ids.shape[0], input_ids.shape[-2],input_ids.shape[-1],input_ids.device,input_ids.dtype)
            txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = input_ids.device, dtype = input_ids.dtype)
            torch.cuda.empty_cache()
            logits = (
                        model(
                            hidden_states=input_ids, # should be (batch size, channel, height, width)
                            encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                            micro_conds=micro_conds, # 
                            pooled_projections=cond_embeds, # should be (batch_size, projection_dim)
                            img_ids = img_ids,
                            txt_ids = txt_ids,
                            # timestep = timesteps * 20,
                            timestep = mask_prob * 1000,
                            # guidance = 9,
                        )
                        .reshape(bs, codebook_size, -1)
                        .permute(0, 2, 1)
                        .reshape(-1, codebook_size)
                    )
            if isinstance(logits, QTensor):
                logits = logits.dequantize()

            loss = F.cross_entropy(
                    logits,
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, loss: {loss.item()}")
            if accelerator.sync_gradients:
                from optimum.quanto import quantization_map
                if not os.path.exists("./quantized"):
                    os.makedirs("./quantized")
                if not os.path.exists("./quantized/transformer_f8"):
                    os.makedirs("./quantized/transformer_f8")
                _model =  accelerator.unwrap_model(model)
                with open('./quantized/quantization_map.json', "w") as f:
                    json.dump(quantization_map(_model), f)
    return accelerator.unwrap_model(model)

@torch.no_grad()
def calibration_for_f8(pipe, resolution=1024, steps=64, CFG=9):
    print("Start calibration...")
    pipe.transformer.eval()
    model = pipe.transformer
    weight_dtype = torch.bfloat16
    accelerator = define_accelerator()
    train_dataset = MeissonicDataset(tokenizer=pipe.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    torch.cuda.empty_cache()
    with Calibration(momentum=0.9):
        for _ in range(1):
            for batch_idx, (batch) in enumerate(train_loader):
                if batch_idx > 500:
                    break
                input_ids, labels, mask_prob, mask_id, micro_conds, timesteps = pre_process_batch(batch, model, pipe.vqvae, accelerator, weight_dtype=weight_dtype)
                input_ids = input_ids
                mask_prob = mask_prob.to(weight_dtype)
                micro_conds = micro_conds.to(weight_dtype)
                # now_idx = 7 - int(timesteps * 8)
                # set_input_scale(model, now_idx, type="get")
                if "prompt_input_ids" in batch:
                    with torch.no_grad():
                        encoder_hidden_states, cond_embeds = encode_prompt(
                                pipe.text_encoder, batch["prompt_input_ids"].to(accelerator.device, non_blocking=True), "open_clip"
                            )
                        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
                        cond_embeds = cond_embeds.to(weight_dtype)
                        
                        pool_empty_embeds, pool_empty_clip_embeds = encode_prompt(
                            pipe.text_encoder, tokenize_prompt(pipe.tokenizer, "","open_clip").to(accelerator.device, non_blocking=True), "open_clip"
                        )
                        encoder_hidden_states = torch.cat([pool_empty_embeds, encoder_hidden_states], dim=0)
                        cond_embeds = torch.cat([pool_empty_clip_embeds, cond_embeds], dim=0)
                bs = input_ids.shape[0]
                vae_scale_factor = 2 ** (len(pipe.vqvae.config.block_out_channels) - 1)
                resolution = 1024 // vae_scale_factor
                input_ids = input_ids.reshape(bs, resolution, resolution)
                codebook_size = model.config.codebook_size
                img_ids = _prepare_latent_image_ids(input_ids.shape[0], input_ids.shape[-2],input_ids.shape[-1],input_ids.device, weight_dtype)
                txt_ids = torch.zeros(encoder_hidden_states.shape[1],3).to(device = input_ids.device, dtype = weight_dtype)
                input_ids = torch.cat([input_ids, input_ids], dim=0)
                micro_conds = torch.cat([micro_conds, micro_conds], dim=0)
                _ = (
                                model(
                                    hidden_states=input_ids, # should be (batch size, channel, height, width)
                                    encoder_hidden_states=encoder_hidden_states, # should be (batch size, sequence_len, embed_dims)
                                    micro_conds=micro_conds, # 
                                    pooled_projections=cond_embeds, # should be (batch_size, projection_dim)
                                    img_ids = img_ids,
                                    txt_ids = txt_ids,
                                    # timestep = timesteps * 20,
                                    timestep = mask_prob * 1000,
                                    # guidance = 9,
                                )
                                .reshape(bs, codebook_size, -1)
                                .permute(0, 2, 1)
                                .reshape(-1, codebook_size)
                            )
                if batch_idx % 50 == 0:
                    result_list = []
                    for name1, module in model.named_modules():
                        for name, child in module.named_children():
                            if isinstance(child, QModuleMixin):
                                result_list.append((name1 + name, child.output_scale.mean().item()))
                    sorted_result_list = sorted(result_list, key=lambda x: x[1])
                    for item in sorted_result_list:
                        print(item)
                    
                print(f"\nPass calibration with {batch_idx} batches\n")
    torch.cuda.empty_cache()
    print("Calibration end...")
    return model
     
def run_inference(pipe, prompt, negative_prompt, resolution, cfg, steps):
    return pipe(prompt=prompt, negative_prompt=negative_prompt, height=resolution, width=resolution, guidance_scale=cfg, num_inference_steps=steps).images[0]


def load_prompt(path, seed_path, prompt_version):
    if prompt_version == 'pick':
        prompts = []
        with open(path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])

        prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list
    
    elif prompt_version == 'draw':
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "Prompts":
                    continue
                prompts.append(row[0])

        prompts = prompts[0:200]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)

        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)
        return prompts, seed_list
    elif prompt_version == 'challengebench':
        prompts = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                new_prompt = line.strip()
                prompts.append(new_prompt)
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
        print("prompts: ", len(prompts))
        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)    
        return prompts, seed_list
    else:
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
        print("prompts: ", len(prompts))
        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list
        

def main():
    steps = 64
    CFG = 9
    resolution = 1024 
    negative_prompts = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    prompt_list, seed_list = load_prompt("./hpsv2_prompt2seed.txt", "./HPD_prompt2seed.json", prompt_version="hpsv2")
    print(len(prompt_list), len(seed_list))
    seed_list = [int(seed) for _, seed in seed_list.items()]
    seed_list = seed_list + seed_list
    seed_list = seed_list[:len(prompt_list)]
    
    output_dir = "./challengebench_quant_a8w4_best"
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_models()
    
    idx = 0
    for prompt, seed in zip(prompt_list, seed_list):
        if os.path.exists(os.path.join(output_dir, f'{idx}.png')):
            print("already exists: ", idx)
            idx += 1
            continue
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed(int(seed))
        image = run_inference(pipe, prompt, negative_prompts, resolution, CFG, steps)

        image.save(os.path.join(output_dir, f'{idx}.png'))
        idx += 1

if __name__ == "__main__":
    main()
