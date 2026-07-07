import inspect
import torch

from . import devices, memory_management, sampling
from .sd_samplers_common import Sampler
from .sd_schedulers import Scheduler, simple_scheduler


class KDiffusionSampler(Sampler):
    def __init__(self, funcname, sd_model, options=None):
        super().__init__(funcname)

        self.extra_params = []

        self.options = {}
        self.func = funcname if callable(funcname) else getattr(sampling, self.funcname)

        self.model_wrap = sd_model.forge_objects.unet
        self.predictor = sd_model.forge_objects.unet.model.predictor

  
    def sampling_prepare(self, x: torch.Tensor):
        shape = list(x.shape)
        mem_shape = [2 * shape[0]] + shape[1:]

        unet_inference_memory = self.model_wrap.memory_required(mem_shape)
        additional_inference_memory = self.model_wrap.extra_preserved_memory_during_sampling

        memory_management.load_models_gpu(models=[self.model_wrap], memory_required=unet_inference_memory + additional_inference_memory, minimum_memory_required=unet_inference_memory // 2 + additional_inference_memory)

        percent_to_timestep_function = lambda p: self.predictor.percent_to_sigma(p)

    def sampling_cleanup(self):

        memory_management.soft_empty_cache()


    def get_sigmas(self, p, steps):

        scheduler = Scheduler("simple", "Simple", simple_scheduler, need_inner_model=True)

        sigmas_kwargs = {"sigma_min": self.predictor.sigmas[0].item(), "sigma_max": self.predictor.sigmas[-1].item()}

        if scheduler.need_inner_model:
            sigmas_kwargs["inner_model"] = self.predictor

        sigmas = scheduler.function(n=steps, **sigmas_kwargs, device=devices.cpu)

        return sigmas.cpu()


    def sample(self, p, x, conditioning, steps=None):
        self.sampling_prepare(x)

        steps = steps or p.steps

        sigmas = self.get_sigmas(p, steps).to(x.device)

        x = self.predictor.noise_scaling(sigmas[0], x, torch.zeros_like(x), max_denoise=False)

        extra_params_kwargs = self.initialize(p)
        parameters = inspect.signature(self.func).parameters

        if "n" in parameters:
            extra_params_kwargs["n"] = steps

        if "sigma_min" in parameters:
            extra_params_kwargs["sigma_min"] = self.predictor.sigmas[0].item()
            extra_params_kwargs["sigma_max"] = self.predictor.sigmas[-1].item()

        if "sigmas" in parameters:
            extra_params_kwargs["sigmas"] = sigmas

        self.last_latent = x
        self.sampler_extra_args = {
            "c_crossattn": torch.unsqueeze(conditioning[0], 0),
        }
        
        samples = self.launch_sampling(
            lambda: self.func(self.model_wrap.model, x, extra_args=self.sampler_extra_args, disable=False, callback=None, **extra_params_kwargs),
        )
        
        self.sampling_cleanup()

        return samples