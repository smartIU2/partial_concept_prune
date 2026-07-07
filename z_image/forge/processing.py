from __future__ import annotations

import math
import os

import numpy as np
import torch
from PIL import Image

from . import devices, rng, memory_management
from .sd_samplers_kdiffusion import KDiffusionSampler

opt_f = 8


class DecodedSamples(list):
    already_decoded = True


class Txt2ImgProcessing:
    
    def __init__(self,
        model: object = None,
        prompts: object = "",
        seed: int = -1,
        sampler_name: str = "sample_res_multistep",
        batch_size: int = 1,
        steps: int = 9,
        shift: float = 7.0,
        width: int = 512,
        height: int = 512,
        ):
            
        self.model = model
        self.prompts = [prompts] if isinstance(prompts, str) else prompts
        self.seeds = [seed]
        self.sampler_name = sampler_name
        self.batch_size = batch_size
        self.steps = steps
        self.distilled_cfg_scale = shift
        self.width = width
        self.height = height
        
        self.c = None
        self.tiling = False
        self.sampler = None
        self.cfg_scale = 1.0
        self.step_multiplier = 1
        self.rng = None


    def setup_conds(self):

        with devices.autocast():
            self.c = self.model.get_learned_conditioning(self.prompts)

    def sample(self):
        
        if self.sampler is None:
            self.sampler = KDiffusionSampler(self.sampler_name, self.model)

        x = self.rng.next()
        samples = self.sampler.sample(self, x, self.c)
        del x

        return samples

    def close(self):
        self.sampler = None

    def decode_latent_batch(self, model, batch, target_device=None, check_for_nans=False):
        samples = DecodedSamples()
        samples_pytorch = model.decode_first_stage(batch).to(target_device)

        for x in samples_pytorch:
            samples.append(x)

        return samples

    def process_images(self, decode=True):

        assert self.prompts is not None

        image = None

        devices.torch_gc()

        with torch.inference_mode():
            
            self.setup_conds()

            latent_channels = self.model.forge_objects.vae.latent_channels
            _shape = (latent_channels, self.height // opt_f, self.width // opt_f)
            
            self.rng = rng.ImageRNG(_shape, self.seeds)

            samples_ddim = self.sample()

            if decode:
                
                devices.test_for_nans(samples_ddim, "unet")
                x_samples_ddim = self.decode_latent_batch(self.model, samples_ddim, target_device=devices.cpu, check_for_nans=True)

                x_samples_ddim = torch.stack(x_samples_ddim).float()
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                if len(x_samples_ddim.shape) == 5:
                    x_samples_ddim = x_samples_ddim.reshape(-1, *x_samples_ddim.shape[-3:])

                del samples_ddim

                devices.torch_gc()

                for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255.0 * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                   
                    x_sample = np.clip(x_sample, 0, 255).astype(np.uint8)
                    
                    image = Image.fromarray(x_sample)

                del x_samples_ddim

        devices.torch_gc()

        return image