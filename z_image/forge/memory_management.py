# https://github.com/comfyanonymous/ComfyUI/blob/v0.7.0/comfy/model_management.py
# forge comment: Cherry-picked some good parts from ComfyUI with some bad parts fixed
# smartIU2 comment: removed dependence on commandline arguments by defining init function

"""
This file is part of ComfyUI.
Copyright (C) 2024 Comfy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import gc
import importlib
import logging
import os
import platform
import sys
import time
import weakref
from contextlib import nullcontext
from enum import Enum
from typing import TYPE_CHECKING
from dataclasses import dataclass

import psutil
import torch
import logging

if TYPE_CHECKING:
    from .patcher import ModelPatcher

from .cuda_malloc import get_torch_version, try_cuda_malloc
    
logger = logging.getLogger("memory_management")


class VRAMProfile(Enum):
    AUTO = 0 #No preference
    GPU_ONLY = 1 #Store and run everything on the GPU
    HIGH = 2 #Keeps models in VRAM after usage
    NORMAL = 3 #Force NORMAL_VRAM in case LOW_VRAM gets automatically enabled
    LOW = 4 #Split the diffusion model in parts to use less VRAM
    NO = 5 #When even LOW_VRAM is still not enough
    CPU_ONLY = 6 #Use the CPU for everything (slow)

class VAEPrecision(Enum):
    AUTO = 0 #No preference
    FP32 = 1 #Run the VAE in full precision fp32
    BF16 = 2 #Run the VAE in bf16
    FP16 = 3 #Run the VAE in fp16 (might cause black images)
    
class VRAMState(Enum):
    DISABLED = 0  # No vram present: no need to move models to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


@dataclass
class Args:
    # collection of settings that were collected from commandline arguments in ComfUI / forge
    
    benchmark: bool = False #torch.backends.cudnn.benchmark
    cpu_text_enc: bool = False
    cpu_vae: bool = False
    cuda_malloc: bool = False #improve memory allocation (if supported)
    cuda_streams: int = 0 #set to 2 or more to improve offloading (if supported)
    deterministic: bool = False #use slower deterministic algorithms when possible
    disable_gpu_warning: bool = False #disable the low VRAM warnings
    disable_smart_memory: bool = False #aggressively offload to RAM instead of keeping models in VRAM when possible
    disable_sage: bool = False
    disable_flash : bool = False
    disable_xformers : bool = False
    disable_ipex_optimize : bool = False
    fast_fp8: bool = False #use torch._scaled_mm
    fast_fp16: bool = False #torch.backends.cuda.matmul.allow_fp16_accumulation
    force_fp16: bool = False
    force_fp32: bool = False
    force_upcast_attention: bool = False #always upcast to fp32 during attention
    force_non_blocking: bool = False #use non-blocking operations for all applicable tensors
    gpu_device_id: int = None #set the id of device to use (all other devices will not be visible)
    pin_shared_memory: bool = False #improve RAM utilization
    reserve_vram: float = None #set the amount of VRAM (in MB) you want to reserve for other software, if "None" 400-700 MB are reserved
    use_pytorch_cross_attention: bool = False #the PyTorch cross attention (override sageattention/flash_attn/xformers)
    use_xformers_vae: bool = False #force VAE to use xformers attention (meant to use with PyTorch cross attention)
    vae_precision: VAEPrecision = VAEPrecision.AUTO
    vram_profile: VRAMProfile = VRAMProfile.AUTO
    
args = Args()


# internal
cpu = torch.device("cpu")
lowvram_available: bool = True
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
min_weight_memory_ratio = 0.4
signal_empty_cache = False
current_loaded_models: list["LoadedModel"] = []
torch_version = ""
torch_version_numeric: tuple[int, int] = None
bnb_is_available: bool = False
xpu_available: bool = False
sage_is_available: bool = False
flash_is_available: bool = False
xformers_is_available: bool = False
support_fp8_ops: bool = False
amd_old_arches = ("gfx1030", "gfx1031", "gfx1010", "gfx1011", "gfx1012", "gfx906", "gfx900", "gfx803")
prioritize_fp16: bool = False
enable_pytorch_attention: bool = False
torch_device_name: str = ""
total_vram = 0 #MB
total_ram = psutil.virtual_memory().total / (1024 * 1024) #MB
reserved_vram = 400 * 1024 * 1024 #MB


# public
OOM_EXCEPTION = getattr(torch, "OutOfMemoryError", Exception) # used by attention.py & vae.py 
VAE_ALWAYS_TILED: bool = False # used by vae.py
WINDOWS: bool = any(platform.win32_ver()) # used by operations.py
NVIDIA_CONV3D_WORKAROUND: bool = False # used by operations.py


float8_types: list[torch.dtype] = []
for dtype in ("e4m3fn", "e4m3fnuz", "e5m2", "e5m2fnuz", "e8m0fnu"):
    try:
        float8_types.append(getattr(torch, f"float8_{dtype}"))
    except Exception:
        pass

try:
    torch_version: str = torch.__version__
    _ver: list[str] = torch_version.split(".", 2)
    torch_version_numeric = (int(_ver[0]), int(_ver[1]))
except Exception:
    logger.warning("Could not determine PyTorch version...") 
    
try:
    import intel_extension_for_pytorch as ipex  # noqa: F401

    _ = torch.xpu.device_count()
    xpu_available = torch.xpu.is_available()
except Exception:
    pass
    
try:
    import bitsandbytes  # noqa: F401
    bnb_is_available = True
except Exception:
    pass
        
try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except Exception:
    pass
    
try:
    if torch_version_numeric >= (2, 5):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
except Exception:
    pass


def mac_version():
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except Exception:
        return None
        
def is_intel_xpu() -> bool:
    return cpu_state is CPUState.GPU and xpu_available

def is_nvidia() -> bool:
    return cpu_state is CPUState.GPU and torch.version.cuda

def is_amd() -> bool:
    return cpu_state is CPUState.GPU and torch.version.hip

def get_torch_device() -> torch.device:
    if cpu_state is CPUState.MPS:
        return torch.device("mps")
    if cpu_state is CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())

def get_total_memory(dev: torch.device = None, torch_total_too: bool = False):
    dev = dev or get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
            mem_total = mem_total_xpu
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats["reserved_bytes.all.current"]
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total
        
def amd_min_version(device: torch.device = None, min_rdna_version: int = 0) -> bool:
    if not is_amd():
        return False

    if is_device_cpu(device):
        return False

    arch = torch.cuda.get_device_properties(device).gcnArchName
    if arch.startswith("gfx") and len(arch) == 7:
        try:
            cmp_rdna_version = int(arch[4]) + 2
        except Exception:
            cmp_rdna_version = 0
        if cmp_rdna_version >= min_rdna_version:
            return True

    return False

def get_torch_device_name(device: torch.device) -> str:
    if hasattr(device, "type"):
        if device.type == "cuda":
            try:
                allocator_backend = f"- {torch.cuda.get_allocator_backend()}"
            except Exception:
                allocator_backend = ""
            return "{} ({}) {}".format(torch.cuda.get_device_name(device), device, allocator_backend)
        elif device.type == "xpu":
            return "{} ({})".format(torch.xpu.get_device_name(device), device)
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} ({})".format(torch.xpu.get_device_name(device), device)
    else:
        return "{} (CUDA {})".format(torch.cuda.get_device_name(device), device)

def bake_gguf_model(model):
    if getattr(model, "gguf_baked", False):
        return

    for p in model.parameters():
        gguf_cls = getattr(p, "gguf_cls", None)
        if gguf_cls is not None:
            gguf_cls.bake(p)

    global signal_empty_cache
    signal_empty_cache = True

    model.gguf_baked = True
    return model

def module_size(module: torch.nn.Module) -> int:
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nelement() * t.element_size()
    return module_mem


class LoadedModel:
    def __init__(self, model: "ModelPatcher"):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)

    @property
    def model(self) -> "ModelPatcher":
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)

        real_model = self.model.model

        if is_intel_xpu() and not args.disable_ipex_optimize and "ipex" in globals() and real_model is not None:
            with torch.no_grad():
                real_model = ipex.optimize(real_model.eval(), inplace=True, graph_mode=True, concat_linear=True)

            global signal_empty_cache
            signal_empty_cache = True

        bake_gguf_model(real_model)

        self.model.refresh_loras()

        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model() is not None and self.model is None


def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break


def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem


def minimum_inference_memory() -> float:
    return (1024 * 1024 * 1024) * 0.8 + reserved_vram


def free_memory(memory_required: float, device: torch.device, keep_loaded: list["LoadedModel"] = []):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.synchronize()

    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not args.disable_smart_memory:
            free_mem = get_free_memory(device)
            if free_mem > memory_required:
                break
            memory_to_free = memory_required - free_mem
        logger.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
        if current_loaded_models[i].model_unload(memory_to_free):
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state is not VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()
    return unloaded_models


def load_models_gpu(models: list["ModelPatcher"], memory_required: float = 0, force_patch_weights: bool = False, minimum_memory_required: float = None, force_full_load: bool = False):
    execution_start_time = time.perf_counter()
    cleanup_models_gc(target=models)

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + reserved_vram)
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + reserved_vram)

    models_temp = set()
    for m in models:
        models_temp.add(m)
        for mm in m.model_patches_models():
            models_temp.add(mm)

    models = models_temp

    models_to_load: list["LoadedModel"] = []

    for x in models:
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except Exception:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            if hasattr(x, "model"):
                logger.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            model_to_unload = current_loaded_models.pop(i)
            model_to_unload.model.detach(unpatch_all=False)
            model_to_unload.model_finalizer.detach()

    total_memory_required = {}
    for loaded_model in models_to_load:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_mem = get_free_memory(device)
            if free_mem < minimum_memory_required:
                models_l = free_memory(minimum_memory_required, device)
                logger.debug("{} models unloaded.".format(len(models_l)))

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and vram_set_state in (VRAMState.LOW_VRAM, VRAMState.NORMAL_VRAM) and not force_full_load:
            loaded_memory = loaded_model.model_loaded_memory()
            current_free_mem = get_free_memory(torch_dev) + loaded_memory

            lowvram_model_memory = max(0, (current_free_mem - minimum_memory_required), min(current_free_mem * min_weight_memory_ratio, current_free_mem - minimum_inference_memory()))
            lowvram_model_memory = lowvram_model_memory - loaded_memory

            if lowvram_model_memory == 0:
                lowvram_model_memory = 0.1

        if vram_set_state is VRAMState.NO_VRAM:
            lowvram_model_memory = 0.1

        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    logger.info(f"Moving model(s) has taken {moving_time:.2f} seconds")


def load_model_gpu(model: "ModelPatcher"):
    return load_models_gpu([model])


def loaded_models(only_currently_used: bool = False) -> list["LoadedModel"]:
    output = []
    for m in current_loaded_models:
        if only_currently_used and not m.currently_used:
            continue
        output.append(m.model)
    return output


def cleanup_models_gc(*, target: list["ModelPatcher"] = []):
    _gc: bool = False
    _del: list[int] = []

    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if not cur.is_dead():
            continue
        if any(mdl.model is cur.real_model() for mdl in target):
            _del.append(i)
            break

        logger.info("Potential memory leak detected with model {}...".format(cur.real_model().__class__.__name__))
        _gc = True

    if not _gc and len(_del) == 0:
        return

    for i in reversed(_del):
        m = current_loaded_models.pop(i)
        del m

    gc.collect()
    soft_empty_cache()

    for mdl in current_loaded_models:
        if mdl.is_dead():
            logger.warning("Memory Leak with model {} !".format(mdl.real_model().__class__.__name__))


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x


def dtype_size(dtype: torch.dtype) -> int:
    return getattr(dtype, "itemsize", 4)


def unet_offload_device():
    if vram_state is VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return cpu


def unet_initial_load_device(parameters: int, dtype: torch.dtype) -> torch.device:
    torch_dev = get_torch_device()
    if vram_state in (VRAMState.HIGH_VRAM, VRAMState.SHARED):
        return torch_dev

    cpu_dev = torch.device("cpu")
    if args.disable_smart_memory or vram_state is VRAMState.NO_VRAM:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)

    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def maximum_vram_for_weights(device: torch.device = None) -> float:
    return get_total_memory(device) * 0.88 - minimum_inference_memory()


def unet_dtype(device: torch.device = None, model_params: int = 0, supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32], weight_dtype: torch.dtype = None) -> torch.dtype:
    if model_params < 0:
        model_params = 1e32
    # if args.fp32_unet:
        # return torch.float32
    # if args.bf16_unet:
        # return torch.bfloat16
    # if args.fp16_unet:
        # return torch.float16
    # if args.fp8_e4m3fn_unet:
        # return torch.float8_e4m3fn
    # if args.fp8_e5m2_unet:
        # return torch.float8_e5m2
    # if args.fp8_e8m0fnu_unet:
        # return torch.float8_e8m0fnu

    if weight_dtype in float8_types:
        if supports_fp8_compute(device):
            return weight_dtype

        free_model_memory = maximum_vram_for_weights(device)
        if model_params * 2 > free_model_memory:
            return weight_dtype

    if prioritize_fp16 or weight_dtype == torch.float16:
        if torch.float16 in supported_dtypes and should_use_fp16(device=device, model_params=model_params):
            return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    return torch.float32


def inference_cast(weight_dtype: torch.device, inference_device: torch.device, supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]) -> torch.dtype:
    if weight_dtype == torch.float32:
        return weight_dtype

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return weight_dtype

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return weight_dtype

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
    if prioritize_fp16 and fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and fp16_supported:
            return torch.float16
        if dt == torch.bfloat16 and bf16_supported:
            return torch.bfloat16

    return torch.float32


def text_encoder_offload_device() -> torch.device:
    return get_torch_device() if args.vram_profile == VRAMProfile.GPU_ONLY else cpu


def text_encoder_device() -> torch.device:
    # if args.text_enc_device is not None:
        # return torch.device(args.text_enc_device)
    if args.vram_profile == VRAMProfile.GPU_ONLY:
        return get_torch_device()
    if args.cpu_text_enc:
        return cpu
    elif vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM):
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return cpu
    else:
        return cpu


def text_encoder_initial_device(load_device: torch.device, offload_device: torch.device, model_size: int = 0) -> torch.device:
    if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
        return offload_device

    if is_device_mps(load_device):
        return load_device

    mem_l = get_free_memory(load_device)
    mem_o = get_free_memory(offload_device)
    if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
        return load_device
    else:
        return offload_device


def text_encoder_dtype(device=None) -> torch.dtype:
    # if args.fp8_e4m3fn_text_enc:
        # return torch.float8_e4m3fn
    # if args.fp8_e5m2_text_enc:
        # return torch.float8_e5m2
    # if args.fp16_text_enc:
        # return torch.float16
    # if args.bf16_text_enc:
        # return torch.bfloat16
    # if args.fp32_text_enc:
        # return torch.float32

    return torch.float16


def intermediate_device() -> torch.device:
    return get_torch_device() if args.vram_profile == VRAMProfile.GPU_ONLY else cpu


def vae_device() -> torch.device:
    # if args.vae_device is not None:
        # return torch.device(args.vae_device)
    return cpu if args.cpu_vae else get_torch_device()


def vae_offload_device() -> torch.device:
    return get_torch_device() if args.vram_profile == VRAMProfile.GPU_ONLY else cpu


def vae_dtype(device=None, allowed_dtypes=None) -> torch.dtype:
    if args.vae_precision == VAEPrecision.FP16:
        return torch.float16
    if args.vae_precision == VAEPrecision.BF16:
        return torch.bfloat16
    if args.vae_precision == VAEPrecision.FP32:
        return torch.float32

    if should_use_bf16(vae_device()):
        return torch.bfloat16

    return torch.float32


def get_autocast_device(dev: torch.device) -> str:
    return getattr(dev, "type", "cuda")


def supports_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def supports_cast(device: torch.device, dtype: torch.dtype) -> bool:
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False


def pick_weight_dtype(dtype: torch.dtype, fallback_dtype: torch.dtype, device: torch.device = None) -> torch.dtype:
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype


def device_supports_non_blocking(device: torch.device) -> bool:
    if args.force_non_blocking:
        return True
    if is_device_mps(device):
        return False
    if is_intel_xpu():
        return False
    if args.deterministic:
        return False
    return True


def cast_to(weight: torch.Tensor, dtype: torch.dtype = None, device: torch.device = None, non_blocking: bool = False, copy: bool = False, context=nullcontext()):
    if device is None or weight.device == device:
        if not copy and (dtype is None or weight.dtype == dtype):
            return weight
        with context:
            return weight.to(dtype=dtype, copy=copy)

    with context:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
        return r


def cast_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, copy: bool = False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


def xformers_enabled() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    return xformers_is_available


def xformers_enabled_vae() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    return xformers_is_available or args.use_xformers_vae


def sage_enabled() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if not is_nvidia():
        return False
    return sage_is_available


def flash_enabled() -> bool:
    if cpu_state is not CPUState.GPU:
        return False
    if not is_nvidia():
        return False
    return flash_is_available


def bnb_enabled() -> bool:
    return bnb_is_available


def pytorch_attention_enabled() -> bool:
    return enable_pytorch_attention


def pytorch_attention_enabled_vae() -> bool:
    return enable_pytorch_attention and not is_amd()

def fast_fp8_enabled() -> bool:
    return args.fast_fp8 and supports_fp8_compute(get_torch_device())

def pytorch_attention_flash_attention() -> bool:
    if enable_pytorch_attention:
        if is_nvidia():
            return True
        if is_intel_xpu():
            return True
        if is_amd():
            return True
    return False


def force_upcast_attention_dtype() -> dict[torch.dtype, torch.dtype]:
    upcast: bool = args.force_upcast_attention

    macos_version = mac_version()
    if macos_version is not None and macos_version >= (14, 5):
        upcast = True

    return {torch.float16: torch.float32} if upcast else {}


def get_free_memory(dev: torch.device = None, torch_free_too: bool = False) -> int | tuple[int, int]:
    dev = dev or get_torch_device()

    if hasattr(dev, "type") and (dev.type == "cpu" or dev.type == "mps"):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats["active_bytes.all.current"]
            mem_reserved = stats["reserved_bytes.all.current"]
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode() -> bool:
    return cpu_state is CPUState.CPU


def mps_mode() -> bool:
    return cpu_state is CPUState.MPS


def is_device_type(device: torch.device, type: str) -> bool:
    return getattr(device, "type", False) == type


def is_device_cpu(device: torch.device) -> bool:
    return is_device_type(device, "cpu")


def is_device_mps(device: torch.device) -> bool:
    return is_device_type(device, "mps")


def is_device_xpu(device: torch.device) -> bool:
    return is_device_type(device, "xpu")


def is_device_cuda(device: torch.device) -> bool:
    return is_device_type(device, "cuda")


def should_use_fp16(device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False) -> bool:
    if device is not None and is_device_cpu(device):
        return False

    if args.force_fp16:
        return True

    if args.force_fp32:
        return False

    if (device is not None and is_device_mps(device)) or mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        if torch_version_numeric < (2, 3):
            return True
        else:
            return torch.xpu.get_device_properties(device).has_fp16

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    nvidia_10_series = ("1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4")
    for x in nvidia_10_series:
        if x in props.name.lower():
            if WINDOWS or manual_cast:
                return True
            else:
                return False

    if manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    nvidia_16_series = ("1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200")
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(device: torch.device = None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False) -> bool:
    if device is not None and is_device_cpu(device):
        return False

    if args.force_fp32:
        return False

    if (device is not None and is_device_mps(device)) or mps_mode():
        if mac_version() < (14,):
            return False
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        if torch_version_numeric < (2, 3):
            return True
        else:
            return torch.xpu.is_bf16_supported()

    if is_amd():
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if any((a in arch) for a in amd_old_arches):
            if manual_cast:
                return True
            return False

    props = torch.cuda.get_device_properties(device)

    if props.major >= 8:
        return True

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works and manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def supports_fp8_compute(device: torch.device = None) -> bool:
    if support_fp8_ops:
        return True

    if not is_nvidia():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major >= 9:
        return True
    if props.major < 8:
        return False
    if props.minor < 9:
        return False

    if WINDOWS:
        if torch_version_numeric < (2, 4):
            return False
    else:
        if torch_version_numeric < (2, 3):
            return False

    return True


def extended_fp16_support() -> bool:
    return torch_version_numeric >= (2, 7)



lora_compute_dtypes: dict[torch.device, torch.dtype] = {}
def lora_compute_dtype(device: torch.device) -> torch.dtype:
    if device in lora_compute_dtypes:
        return lora_compute_dtypes[device]

    if should_use_fp16(device):
        dtype = torch.float16
    else:
        dtype = torch.float32

    lora_compute_dtypes[device] = dtype
    return dtype


def soft_empty_cache(force=False):
    if cpu_state is CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    global signal_empty_cache
    signal_empty_cache = False


def unload_model(model: "ModelPatcher") -> bool:
    index = None
    for i, p in enumerate(current_loaded_models):
        if p.model == model:
            index = i
            break

    if index is not None:
        mdl = current_loaded_models.pop(index)
        del mdl
        return True

    return False


def unload_all_models():
    free_memory(1e30, get_torch_device())


def current_stream(device: torch.device):
    if device is None:
        return None
    if is_device_cuda(device):
        return torch.cuda.current_stream()
    elif is_device_xpu(device):
        return torch.xpu.current_stream()
    else:
        return None

streams = {}
stream_counters: dict[torch.device, int] = {}
def get_offload_stream(device: torch.device):
    if args.cuda_streams == 0:
        return None
    if torch.compiler.is_compiling():
        return None

    stream_counter = stream_counters.get(device, 0)

    if device in streams:
        ss = streams[device]
        ss[stream_counter].wait_stream(current_stream(device))
        stream_counter = (stream_counter + 1) % len(ss)
        stream_counters[device] = stream_counter
        return ss[stream_counter]
    elif is_device_cuda(device):
        ss = []
        for _ in range(args.cuda_streams):
            s1 = torch.cuda.Stream(device=device, priority=0)
            ss.append(s1)
        streams[device] = ss
        s = ss[stream_counter]
        stream_counters[device] = stream_counter
        return s
    elif is_device_xpu(device):
        ss = []
        for _ in range(args.cuda_streams):
            s1 = torch.xpu.Stream(device=device, priority=0)
            ss.append(s1)
        streams[device] = ss
        s = ss[stream_counter]
        stream_counters[device] = stream_counter
        return s

    return None

def sync_stream(device: torch.device, stream):
    if stream is None or current_stream(device) is None:
        return
    current_stream(device).wait_stream(stream)

def num_streams():
    return args.cuda_streams

pinning_allowed_types = "Parameter"
pinned_memory = {}
total_pinned_memory = 0
max_pinned_memory = -1

def discard_cuda_async_error():
    try:
        a = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        b = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        _ = a + b
        torch.cuda.synchronize()
    except torch.AcceleratorError:
        pass


def pin_memory(tensor):
    global total_pinned_memory
    if max_pinned_memory <= 0:
        return False

    if type(tensor).__name__ != pinning_allowed_types:
        return False

    if not is_device_cpu(tensor.device):
        return False

    if tensor.is_pinned():
        return False

    if not tensor.is_contiguous():
        return False

    size = tensor.numel() * tensor.element_size()
    if (total_pinned_memory + size) > max_pinned_memory:
        return False

    ptr = tensor.data_ptr()
    if ptr == 0:
        return False

    if torch.cuda.cudart().cudaHostRegister(ptr, size, 1) == 0:
        pinned_memory[ptr] = size
        total_pinned_memory += size
        return True
    else:
        discard_cuda_async_error()

    return False

def unpin_memory(tensor):
    global total_pinned_memory
    if max_pinned_memory <= 0:
        return False

    if not is_device_cpu(tensor.device):
        return False

    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()

    size_stored = pinned_memory.get(ptr, None)
    if size_stored is None:
        return False

    if size != size_stored:
        return False

    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:
        total_pinned_memory -= pinned_memory.pop(ptr)
        if len(pinned_memory) == 0:
            total_pinned_memory = 0
        return True
    else:
        discard_cuda_async_error()

    return False



def init():

    # set environment variables (from forge initialization.py)

    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    if args.gpu_device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device_id)

    if "rocm" in get_torch_version():
        # https://github.com/Comfy-Org/ComfyUI/blob/v0.10.0/main.py
        os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        os.environ["OCL_SET_SVM_SIZE"] = "262144"

    if args.cuda_malloc:
        try_cuda_malloc()


    # init parameters dependent on Args

    global cpu_state, total_vram, torch_device_name, min_weight_memory_ratio
    global sage_is_available, flash_is_available, xformers_is_available, enable_pytorch_attention
    global support_fp8_ops, prioritize_fp16
    global set_vram_to, lowvram_available, vram_state, reserved_vram, max_pinned_memory
    global NVIDIA_CONV3D_WORKAROUND


    if args.vram_profile == VRAMProfile.CPU_ONLY:
        cpu_state = CPUState.CPU

    total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
    logger.info("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

    try:
        torch_device_name = get_torch_device_name(get_torch_device())
        logger.info("Device: {}".format(torch_device_name))
    except Exception:
        logger.warning("Could not determine default device...")
                 
    if "rtx" in torch_device_name.lower() and not args.cuda_malloc:
        logger.warning("Hint: your device supports --cuda-malloc for potential speed improvements")
        
    if args.deterministic:
        logger.info("Using deterministic algorithms for PyTorch")
        torch.use_deterministic_algorithms(True, warn_only=True)

    if is_nvidia():
        min_weight_memory_ratio = 0.0

    if not args.disable_sage:
        try:
            from sageattention import sageattn  # noqa: F401
            sage_is_available = True
        except Exception:
            pass

    if not args.disable_flash:
        try:
            from flash_attn import flash_attn_func  # noqa: F401
            flash_is_available = True
        except Exception:
            pass
            
    if not args.disable_xformers:
        try:
            import xformers
            import xformers.ops  # noqa: F401

            xformers_is_available = xformers._has_cpp_library
        except Exception:
            pass
            
    if args.use_pytorch_cross_attention:
        enable_pytorch_attention = True
        xformers_is_available = False
        sage_is_available = False
        flash_is_available = False


    if is_nvidia() and torch_version_numeric[0] >= 2:
        enable_pytorch_attention = True
    elif is_intel_xpu():
        enable_pytorch_attention = True


    if is_amd():
        
        try:
            arch = torch.cuda.get_device_properties(get_torch_device()).gcnArchName
            if not (any((a in arch) for a in amd_old_arches)):
                if os.getenv("ENABLE_MIOPEN") != "1":
                    torch.backends.cudnn.enabled = False

            try:
                rocm_version = tuple(map(int, str(torch.version.hip).split(".")[:2]))
            except Exception:
                rocm_version = (6, -1)

            logger.info("AMD Arch: {}".format(arch))
            logger.info("ROCm Version: {}".format(rocm_version))
            if importlib.util.find_spec("triton") is not None:
                if torch_version_numeric >= (2, 7):
                    if any((a in arch) for a in ["gfx90a", "gfx942", "gfx1100", "gfx1101", "gfx1151"]):
                        enable_pytorch_attention = True
                if rocm_version >= (7, 0):
                    if any((a in arch) for a in ["gfx1201"]):
                        enable_pytorch_attention = True
            if torch_version_numeric >= (2, 7) and rocm_version >= (6, 4):
                if any((a in arch) for a in ["gfx1200", "gfx1201", "gfx950"]):
                    support_fp8_ops = True

        except Exception:
            pass


    if enable_pytorch_attention:
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    try:
        if args.fast_fp16 and (is_nvidia() or is_amd()):
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            logger.info("allow_fp16_accumulation: {}".format(torch.backends.cuda.matmul.allow_fp16_accumulation))
            prioritize_fp16 = True
    except Exception:
        pass

    if args.benchmark and torch.cuda.is_available() and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        logger.info("benchmark: {}".format(torch.backends.cudnn.benchmark))

    if args.vram_profile == VRAMProfile.LOW:
        set_vram_to = VRAMState.LOW_VRAM
        lowvram_available = True
    elif args.vram_profile == VRAMProfile.NO:
        set_vram_to = VRAMState.NO_VRAM
    elif args.vram_profile == VRAMProfile.HIGH or args.vram_profile == VRAMProfile.GPU_ONLY:
        vram_state = VRAMState.HIGH_VRAM

    if lowvram_available:
        if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
            vram_state = set_vram_to

    if cpu_state is not CPUState.GPU:
        vram_state = VRAMState.DISABLED

    if cpu_state is CPUState.MPS:
        vram_state = VRAMState.SHARED

    logger.info(f"VRAM State: {vram_state.name}")
    
    if args.reserve_vram is not None:
        reserved_vram = args.reserve_vram * 1024 * 1024
        logger.info("Reserving {:0.0f} MB VRAM".format(args.reserve_vram))
    elif WINDOWS:
        reserved_vram = 600 * 1024 * 1024
        if total_vram > (15 * 1024):
            reserved_vram += 100 * 1024 * 1024
  
    if args.pin_shared_memory:
        if is_nvidia() or is_amd():
            if WINDOWS:
                max_pinned_memory = get_total_memory(torch.device("cpu")) * 0.45  # Windows limit is apparently 50%
            else:
                max_pinned_memory = get_total_memory(torch.device("cpu")) * 0.95
            logger.info("Pinned Memory: {} MB".format(round(max_pinned_memory / (1024 * 1024))))
  
    try:
        if is_nvidia():
            cudnn_version = torch.backends.cudnn.version()
            if (91002 <= cudnn_version < 91500) and ((2, 9) <= torch_version_numeric <= (2, 10)):
                NVIDIA_CONV3D_WORKAROUND = True
    except Exception:
        pass