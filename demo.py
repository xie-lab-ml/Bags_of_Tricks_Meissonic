import os
import random
import sys
import argparse
sys.path.append("./")

import numpy as np
import torch
from torchvision import transforms


from src.transformer import Transformer2DModel
from src.pipeline import Pipeline
from src.scheduler import Scheduler
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)
from src.tomemgt import patch
from diffusers import VQModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default='Groot depicted as a flower', type=str)
    
    parser.add_argument("--inference-step", default=64, type=int)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--cfg", default=9, type=float)
    parser.add_argument("--inversion-cfg", type=float, default=0.0)

    # noise factor
    parser.add_argument("--add-noise", type=str, default='cos',
                         help="enable noise regularization or not?", choices=['none', 'cos'])
    parser.add_argument("--method", type=str, default='origin', choices=['origin', 'zigzag'])

    # low entropy sampling
    parser.add_argument("--low-entropy", type=bool, default=False)
    # 1024 [0-3072]  512 [0-768]
    parser.add_argument("--entropy-interval", type=list, default=[0, 3072])
    parser.add_argument("--token-merging", action='store_true', default=False)
    
    
    args =  parser.parse_args()
    return args


if __name__ == "__main__":
    device = 'cuda'
    args = get_args()

    model_path = "MeissonFlow/Meissonic"
    clip_model_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model = Transformer2DModel.from_pretrained(model_path,
                                               subfolder="transformer")
    
    vq_model = VQModel.from_pretrained(model_path, 
                                       subfolder="vqvae")
    
    text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_path)
    
    tokenizer = CLIPTokenizer.from_pretrained(model_path,
                                              subfolder="tokenizer",)
    
    scheduler = Scheduler.from_pretrained(model_path,
                                          subfolder="scheduler",)
    scheduler.entropy_interval = args.entropy_interval
    
    pipe = Pipeline(vq_model, 
                  tokenizer=tokenizer,
                  text_encoder=text_encoder,
                  transformer=model,
                  scheduler=scheduler)

    pipe = pipe.to(device)

    if args.token_merging:
        patch.apply_patch(pipe, ratio=0.25)
    
    negative_prompts = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"

    if args.method == 'origin':
        image = pipe(prompt=args.prompt,
                    negative_prompt=negative_prompts,
                    height=args.size,
                    width=args.size,
                    guidance_scale=args.cfg,
                    num_inference_steps=args.inference_step,
                    add_noise=args.add_noise,
                    low_entropy=args.low_entropy).images[0]

        image.save(f"{args.prompt[:10]}_{args.size}_{args.inference_step}_{args.cfg}_{args.token_merging}.png")

    else:
        image = pipe.forward_ours(prompt=args.prompt,
                    negative_prompt=negative_prompts,
                    height=args.size,
                    width=args.size,
                    guidance_scale=args.cfg,
                    inversion_cfg=args.inversion_cfg,
                    num_inference_steps=args.inference_step,
                    add_noise=args.add_noise,
                    low_entropy=args.low_entropy).images[0]

        image.save(f"{args.prompt[:10]}_{args.size}_{args.inference_step}_{args.cfg}_zigzag.png")


