import torch

from . import memory_management
from .patcher import ModelPatcher


class JointTextEncoder(torch.nn.Module):
    def __init__(self, module_dict):
        super(JointTextEncoder, self).__init__()
        for name, module in module_dict.items():
            self.add_module(name, module)

class ObjectDict:
    def __init__(self, module_dict):
        for name, module in module_dict.items():
            setattr(self, name, module)

class CLIP:
    def __init__(self, model_dict={}, tokenizer_dict={}, no_init=False):
        if no_init:
            return

        load_device = memory_management.text_encoder_device()
        offload_device = memory_management.text_encoder_offload_device()

        self.cond_stage_model = JointTextEncoder(model_dict)
        self.tokenizer = ObjectDict(tokenizer_dict)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, *arg, **kwargs):
        return self.patcher.add_patches(*arg, **kwargs)
