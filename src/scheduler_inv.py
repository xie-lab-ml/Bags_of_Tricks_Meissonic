# Copyright 2024 The HuggingFace Team and The MeissonFlow Team. All rights reserved.
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
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin


def gumbel_noise(t, generator=None):
    device = generator.device if generator is not None else t.device
    noise = torch.zeros_like(t, device=device).uniform_(0, 1, generator=generator).to(t.device)
    return -torch.log((-torch.log(noise.clamp(1e-20))).clamp(1e-20))


def mask_by_random_topk(mask_len, probs, temperature=1.0, generator=None):
    confidence = torch.log(probs.clamp(1e-20)) + temperature * gumbel_noise(probs, generator=generator)
    sorted_confidence = torch.sort(confidence, dim=-1).values
    cut_off = torch.gather(sorted_confidence, 1, mask_len.long())
    masking = confidence < cut_off
    return masking


@dataclass
class SchedulerOutput(BaseOutput):
      """
      Output class for the scheduler's `step` function output.

      Args:
            prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
                  Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
                  denoising loop.
            pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
                  The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
                  `pred_original_sample` can be used to preview progress or for guidance.
      """

      prev_sample: torch.Tensor
      pred_original_sample: torch.Tensor = None


class SchedulerInv(SchedulerMixin, ConfigMixin):
      order = 1

      temperatures: torch.Tensor

      @register_to_config
      def __init__(
            self,
            mask_token_id: int,
            masking_schedule: str = "cosine",
      ):
            self.temperatures = None
            self.timesteps = None

      def set_timesteps(
            self,
            num_inference_steps: int,
            temperature: Union[int, Tuple[int, int], List[int]] = (2, 0),
            device: Union[str, torch.device] = None,
      ):
            self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)

            if isinstance(temperature, (tuple, list)):
                  self.temperatures = torch.linspace(temperature[0], temperature[1], num_inference_steps, device=device)
            else:
                  self.temperatures = torch.linspace(temperature, 0.01, num_inference_steps, device=device)

      def step(
            self,
            model_output: torch.Tensor,
            timestep: torch.long,
            sample: torch.LongTensor,
            starting_mask_ratio: int = 1,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
            flag: str = 'none',
            sigma_end: float = 0.,
            rho: float = 1.,
            pre_mask_tokens_sum = 4095,
            pre_unknown_map = None,
      ) -> Union[SchedulerOutput, Tuple]:
            two_dim_input = sample.ndim == 3 and model_output.ndim == 4

            if two_dim_input:
                  batch_size, codebook_size, height, width = model_output.shape
                  sample = sample.reshape(batch_size, height * width)
                  model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)
            
            unknown_map = sample == self.config.mask_token_id

            probs = model_output.softmax(dim=-1)

            device = probs.device
            probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
            if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
                  probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
            probs_ = probs_.reshape(-1, probs.size(-1))


            pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=device)            
            pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])
            pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)

            if timestep == 0:
                  prev_sample = pred_original_sample
            else:
                  seq_len = sample.shape[1]
                  step_idx = (self.timesteps == timestep).nonzero()
                  ratio = (step_idx + 1) / len(self.timesteps)

                  if self.config.masking_schedule == "cosine":
                        mask_ratio = torch.cos(ratio * math.pi / 2)
                  elif self.config.masking_schedule == "linear":
                        mask_ratio = 1 - ratio
                  else:
                        raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

                  mask_ratio = starting_mask_ratio * mask_ratio

                  mask_len = (seq_len * mask_ratio).floor()

                  mask_len += pre_mask_tokens_sum - mask_len

                  selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
                  # Ignores the tokens given in the input by overwriting their confidence.
                  selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

                  masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)

                  # Masks tokens with lower confidence.
                  prev_sample = torch.where(masking, self.config.mask_token_id, pred_original_sample)

            if two_dim_input:
                  prev_sample = prev_sample.reshape(batch_size, height, width)
                  pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

            if not return_dict:
                  return (prev_sample, pred_original_sample)

            return SchedulerOutput(prev_sample, pred_original_sample)

      def step_zigzag(
            self,
            model_output: torch.Tensor,
            timestep: torch.long,
            sample: torch.LongTensor,
            starting_mask_ratio: int = 1,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
            pre_mask_tokens_sum = 4095,
            pre_unknown_map = None
      ) -> Union[SchedulerOutput, Tuple]:
            two_dim_input = sample.ndim == 3 and model_output.ndim == 4

            if two_dim_input:
                  batch_size, codebook_size, height, width = model_output.shape
                  sample = sample.reshape(batch_size, height * width)
                  model_output = model_output.reshape(batch_size, codebook_size, height * width).permute(0, 2, 1)
            
            unknown_map = sample == self.config.mask_token_id
            

            probs = model_output.softmax(dim=-1)

            device = probs.device
            probs_ = probs.to(generator.device) if generator is not None else probs  # handles when generator is on CPU
            if probs_.device.type == "cpu" and probs_.dtype != torch.float32:
                  probs_ = probs_.float()  # multinomial is not implemented for cpu half precision
            probs_ = probs_.reshape(-1, probs.size(-1))

            # TODO: random choose tokens to mask
            false_indices = (unknown_map == False).nonzero(as_tuple=True)[1]  # 获取 False 的索引 
            max_probs, max_indices = torch.max(probs_, dim=-1)
            false_indices = false_indices.squeeze(0)
            selected_max_probs = max_probs[false_indices]

            pred_original_sample = torch.multinomial(probs_, 1, generator=generator).to(device=device)            
            pred_original_sample = pred_original_sample[:, 0].view(*probs.shape[:-1])

            if timestep == 0:
                  # TODO: change 
                  pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)

                  prev_sample = pred_original_sample
            else:
                  seq_len = sample.shape[1]
                  step_idx = (self.timesteps == timestep).nonzero()
                  ratio = (step_idx + 1) / len(self.timesteps)

                  if self.config.masking_schedule == "cosine":
                        mask_ratio = torch.cos(ratio * math.pi / 2)
                  elif self.config.masking_schedule == "linear":
                        mask_ratio = 1 - ratio
                  else:
                        raise ValueError(f"unknown masking schedule {self.config.masking_schedule}")

                  mask_ratio = starting_mask_ratio * mask_ratio

                  mask_len = (seq_len * mask_ratio).floor()

                  # TODO: change
                  n = int((pre_mask_tokens_sum - mask_len).item())

                  if n > 0:
                        top_n_probs, top_n_indices = torch.topk(selected_max_probs, n)                        
                        top_n_false_indices = false_indices[top_n_indices]

                        unknown_map[0, top_n_false_indices] = True

                  pred_original_sample = torch.where(unknown_map, pred_original_sample, sample)
                  # ----------------------------------------
                  mask_len += pre_mask_tokens_sum - mask_len

                  selected_probs = torch.gather(probs, -1, pred_original_sample[:, :, None])[:, :, 0]
                  # Ignores the tokens given in the input by overwriting their confidence.
                  selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)

                  masking = mask_by_random_topk(mask_len, selected_probs, self.temperatures[step_idx], generator)

                  # Masks tokens with lower confidence.
                  prev_sample = torch.where(masking, self.config.mask_token_id, pred_original_sample)

            if two_dim_input:
                  prev_sample = prev_sample.reshape(batch_size, height, width)
                  pred_original_sample = pred_original_sample.reshape(batch_size, height, width)

            if not return_dict:
                  return (prev_sample, pred_original_sample)

            return SchedulerOutput(prev_sample, pred_original_sample)
