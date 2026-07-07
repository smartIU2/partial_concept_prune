import inspect

import torch

from . import sampling

class InterruptedException(BaseException):
    pass


class TorchHijack:
    """This is here to replace torch.randn_like of k-diffusion.

    k-diffusion has random_sampler argument for most samplers, but not for all, so
    this is needed to properly replace every use of torch.randn_like.

    We need to replace to make images generated in batches to be same as images generated individually."""

    def __init__(self, p):
        self.rng = p.rng

    def __getattr__(self, item):
        if item == "randn_like":
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return self.rng.next()


class Sampler:
    def __init__(self, funcname):
        self.funcname = funcname
        self.func = funcname
        self.extra_params = []
        self.sampler_noises = None
        self.stop_at = None
        self.eta = None
        self.config: SamplerData = None  # set by the function calling the constructor
        self.last_latent = None
        self.s_min_uncond = None
        self.s_churn = 0.0
        self.s_tmin = 0.0
        self.s_tmax = float("inf")
        self.s_noise = 1.0

        self.eta_option_field = "eta_ancestral"
        self.eta_infotext_field = "Eta"
        self.eta_default = 1.0

        self.conditioning_key = "crossattn"

        self.p = None
        self.model_wrap_cfg = None
        self.sampler_extra_args = None
        self.options = {}

    def callback_state(self, d):
        step = d["i"]

        if self.stop_at is not None and step > self.stop_at:
            raise InterruptedException

    def launch_sampling(self, func):

        try:
            return func()
        except RecursionError:
            print("Encountered RecursionError during sampling; try to use a smaller rho value instead")
            return self.last_latent
        except InterruptedException:
            return self.last_latent

    def number_of_needed_noises(self, p):
        return p.steps

    def initialize(self, p) -> dict:
        self.p = p
        self.eta = 0.0
        self.s_min_uncond = getattr(p, "s_min_uncond", 0.0)
        
        sampling.torch = TorchHijack(p)
        
        extra_params_kwargs = {}
        for param_name in self.extra_params:
            if hasattr(p, param_name) and param_name in inspect.signature(self.func).parameters:
                extra_params_kwargs[param_name] = getattr(p, param_name)

        if "eta" in inspect.signature(self.func).parameters:
            if self.eta != self.eta_default:
                p.extra_generation_params[self.eta_infotext_field] = self.eta

            extra_params_kwargs["eta"] = self.eta

        return extra_params_kwargs

    def sample(self, p, x, conditioning, unconditional_conditioning, steps=None, image_conditioning=None):
        raise NotImplementedError()
