from . import utils


class ForgeObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return ForgeObjects(self.unet, self.clip, self.vae, self.clipvision)


class ForgeDiffusionEngine:
    matched_guesses = []

    def __init__(self, estimated_config, huggingface_components):
        self.model_config = estimated_config
        self.is_inpaint = False

        self.forge_objects: "ForgeObjects" = None
        self.forge_objects_original: "ForgeObjects" = None
        self.forge_objects_after_applying_lora: "ForgeObjects" = None

        self.current_lora_hash = str([])

        self.ini_latent: "torch.Tensor" = None  # image from img2img input
        self.ref_latents: list["torch.Tensor"] = []  # images from ImageStitch

    def set_clip_skip(self, clip_skip):
        pass

    def get_first_stage_encoding(self, x):
        return x

    def get_learned_conditioning(self, prompt: list[str]):
        raise NotImplementedError

    def encode_first_stage(self, x):
        raise NotImplementedError

    def decode_first_stage(self, x):
        raise NotImplementedError

    @property
    def first_stage_model(self):
        try:
            return self.forge_objects.vae.first_stage_model
        except Exception:
            return None

    @property
    def cond_stage_model(self):
        try:
            return self.forge_objects.clip.cond_stage_model
        except Exception:
            return None

    def save_unet(self, filename):
        import safetensors.torch as sf

        sd = utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model)
        sf.save_file(sd, filename)
        return filename

    def save_checkpoint(self, filename):
        import safetensors.torch as sf

        sd = {}
        sd.update(utils.get_state_dict_after_quant(self.forge_objects.unet.model.diffusion_model, prefix="model.diffusion_model."))
        sd.update(utils.get_state_dict_after_quant(self.forge_objects.clip.cond_stage_model, prefix="text_encoders."))
        sd.update(utils.get_state_dict_after_quant(self.forge_objects.vae.first_stage_model, prefix="vae."))
        sf.save_file(sd, filename)
        return filename
