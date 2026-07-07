import torch


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class AbstractPrediction(torch.nn.Module):
    def __init__(self, sigma_data=1.0, prediction_type="epsilon"):
        super().__init__()
        self.sigma_data = sigma_data
        self.prediction_type = prediction_type
        assert self.prediction_type in ["epsilon", "const", "v_prediction", "edm"]

    def calculate_input(self, sigma, noise):
        if self.prediction_type == "const":
            return noise
        else:
            sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
            return noise / (sigma**2 + self.sigma_data**2) ** 0.5

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        if self.prediction_type == "v_prediction":
            return model_input * self.sigma_data**2 / (sigma**2 + self.sigma_data**2) - model_output * sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        elif self.prediction_type == "edm":
            return model_input * self.sigma_data**2 / (sigma**2 + self.sigma_data**2) + model_output * sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        else:
            return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        if self.prediction_type == "const":
            return sigma * noise + (1.0 - sigma) * latent_image
        else:
            if max_denoise:
                noise = noise * torch.sqrt(1.0 + sigma**2.0)
            else:
                noise = noise * sigma

            noise += latent_image
            return noise

    def inverse_noise_scaling(self, sigma, latent):
        if self.prediction_type == "const":
            return latent / (1.0 - sigma)
        else:
            return latent


class PredictionDiscreteFlow(AbstractPrediction):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/model_sampling.py#L243"""

    def __init__(self, model_config):
        super().__init__(sigma_data=None, prediction_type="const")
        sampling_settings: dict = model_config.sampling_settings
        self.set_parameters(shift=sampling_settings.get("shift", 1.0), multiplier=sampling_settings.get("multiplier", 1000))

    def set_parameters(self, *, shift=None, multiplier=None, timesteps=1000):
        self.shift = shift or self.shift
        self.multiplier = multiplier or self.multiplier
        ts = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * self.multiplier)
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent):
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)
