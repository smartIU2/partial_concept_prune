# reference: https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/supported_models.py

from enum import Enum

import torch


def state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=False):
    if filter_keys:
        out = {}
    else:
        out = state_dict
    for rp in replace_prefix:
        replace = list(map(lambda a: (a, "{}{}".format(replace_prefix[rp], a[len(rp) :])), filter(lambda a: a.startswith(rp), state_dict.keys())))
        for x in replace:
            w = state_dict.pop(x[0])
            out[x[1]] = w
    return out
    
class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    FLUX = 3
    FLOW = 4


class LatentFormat:
    scale_factor: float = 1.0
    latent_channels: int = 4
    latent_rgb_factors: list[list[float]] = None
    latent_rgb_factors_bias: list[list[float]] = None
    taesd_decoder_name: str = None

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent / self.scale_factor

class Flux(LatentFormat):
    def __init__(self):
        self.latent_channels = 16
        self.scale_factor = 0.3611
        self.shift_factor = 0.1159
        self.latent_rgb_factors = [
            [-0.0346,  0.0244,  0.0681],
            [ 0.0034,  0.0210,  0.0687],
            [ 0.0275, -0.0668, -0.0433],
            [-0.0174,  0.0160,  0.0617],
            [ 0.0859,  0.0721,  0.0329],
            [ 0.0004,  0.0383,  0.0115],
            [ 0.0405,  0.0861,  0.0915],
            [-0.0236, -0.0185, -0.0259],
            [-0.0245,  0.0250,  0.1180],
            [ 0.1008,  0.0755, -0.0421],
            [-0.0515,  0.0201,  0.0011],
            [ 0.0428, -0.0012, -0.0036],
            [ 0.0817,  0.0765,  0.0749],
            [-0.1264, -0.0522, -0.1103],
            [-0.0280, -0.0881, -0.0499],
            [-0.1262, -0.0982, -0.0778],
        ]
        self.latent_rgb_factors_bias = [-0.0329, -0.0718, -0.0851]
        self.taesd_decoder_name = "taef1_decoder"

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor


class BASE:
    
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    memory_usage_factor = 2.0

    manual_cast_dtype = None
    unet_target = "unet"
    vae_target = "vae"

    @classmethod
    def matches(cls, unet_config, state_dict=None):
        for k in cls.unet_config:
            if k not in unet_config or cls.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in cls.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict):
        return ModelType.EPS

    def clip_target(self, state_dict: dict):
        return {}

    def inpaint_model(self):
        return self.unet_config.get("in_channels", -1) > 4

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {k: "" for k in self.text_encoder_key_prefix}
        return state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.text_encoder_key_prefix[0]}
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def process_clip_vision_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        if self.clip_vision_prefix is not None:
            replace_prefix[""] = self.clip_vision_prefix
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.diffusion_model."}
        return state_dict_prefix_replace(state_dict, replace_prefix)

    def process_vae_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.vae_key_prefix[0]}
        return state_dict_prefix_replace(state_dict, replace_prefix)

class Lumina2(BASE):

    unet_config = {
        "image_model": "lumina2",
        "dim": 2304,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 6.0,
    }

    memory_usage_factor = 1.4

    unet_extra_config = {}
    latent_format = Flux

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def model_type(self, state_dict):
        return ModelType.FLOW

    def clip_target(self, state_dict: dict):
        pref = self.text_encoder_key_prefix[0]
        if "{}gemma2_2b.transformer.model.embed_tokens.weight".format(pref) in state_dict:
            state_dict.pop("{}gemma2_2b.logit_scale".format(pref), None)
            state_dict.pop("{}spiece_model".format(pref), None)
            return {"gemma2_2b.transformer": "text_encoder"}
        else:
            return {"gemma2_2b": "text_encoder"}

class ZImage(Lumina2):

    unet_config = {
        "image_model": "lumina2",
        "dim": 3840,
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.0,
    }

    memory_usage_factor = 2.8

    supported_inference_dtypes = [torch.bfloat16, torch.float32]

    def __init__(self, unet_config):
        super().__init__(unet_config)
        if self.unet_config.pop("allow_fp16", False):
            self.supported_inference_dtypes = ZImage.supported_inference_dtypes.copy()
            self.supported_inference_dtypes.insert(1, torch.float16)

    def clip_target(self, state_dict={}):
        return {"qwen3_4b.transformer": "text_encoder"}


models = [
    Lumina2,
    ZImage,
]
