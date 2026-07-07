import torch

from . import memory_management
from .k_prediction import PredictionDiscreteFlow
from .qwen3_engine import Qwen3TextProcessingEngine
from .diffusion_engine import ForgeDiffusionEngine, ForgeObjects
from .clip import CLIP
from .unet import UnetPatcher
from .vae import VAE


class ZImage(ForgeDiffusionEngine):

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(model_dict={"qwen3": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"])

        k_predictor = PredictionDiscreteFlow(estimated_config)

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_gemma = Qwen3TextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3,
            tokenizer=clip.tokenizer.qwen3,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        
        self.use_shift = True
        self.is_flux = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        shift = getattr(prompt, "distilled_cfg_scale", 7.0)
        self.forge_objects.unet.model.predictor.set_parameters(shift=shift)
        cond = self.text_processing_engine_gemma(prompt)
        return cond

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
