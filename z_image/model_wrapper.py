import gc
import json
import numpy as np
import random
import torch

class ZImageWrapper:
    '''
        This wrapper allows handling of both WanGP and Forge memory management & quantization implementations for the Z-Image Turbo model
    '''

    def __init__(self):
        self.num_layers = 30 # number of feed forward layers in main transformer block
        self.sampling_steps = 9 # number of time steps (between 8 and 10 for Z-Image Turbo)
        self.flow_shift = 7.0 # updated from the original 3.0, as it helps mitigate some Z-Image Turbo noise problems
        
        self.module_name = 'feed_forward.w2' # module that contains prunable weights / hidden features

    @property
    def transformer(self):
        # returns the actual Z-Image transformer underneath all the wrappers
        raise NotImplementedError

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def encode_text(
        self,
        prompts):
        raise NotImplementedError

    def generate_image(
        self,
        prompts,    
        width = 1024,
        height = 1024,
        batch_size = 1,
        seed = -1,
        sample_solver = "default",
        decode = True):
        raise NotImplementedError
        
    def prune(self
            , module #FeedForward module containing 3xForgeOperations.Linear or 3xQLinearQuantoRouter modules
            , mask #binary mask with shape [in_feature, hidden_feature], 1 = prunable
            , structured_pruning = True # False = set individual weights to zero (no change in dimensionality), True = reduce dimensionality
            , threshold = 0.5 #ratio of prunable weights per in_feature to maybe prune related hidden_feature
            , width = 16 #new dimensionality has to be divisible by this width, to optimize matrix multiplications on the GPU
            #, layer_no = 0
            ):
        '''
            Prunes a transformer module, given a binary mask for the prunable weights.
            Returns the hidden dimensionality after pruning.
        '''
        
        hidden_features = module.w2.in_features
        
        if not structured_pruning:
            
            # print(f"pruning {mask.sum()} weights")
            
            if module.w2.weight.dtype == torch.float8_e4m3fn:
                #cast to bf16 and back, because of limited support for fp8 ops in torch 2.10 / CUDA 
                weight = module.w2.weight.data.clone().detach().bfloat16()
                weight *= torch.from_numpy(1 - mask).to(module.w2.weight.device)
                weight = weight.to(torch.float8_e4m3fn)
                module.w2.weight.data = weight
            else: #quantized int8
                #using "_data" to directly update quantized weights, without dequantizing
                weight = module.w2.weight._data.clone().detach()
                weight *= (1- mask)
                weight = weight.to(torch.int8)
                module.w2.weight._data = weight
        
        else:
            
            threshold = mask.shape[0] * threshold
            feature_mask = []
            for i in range(mask.shape[1]):
                feature_mask.append(mask[:,i].sum())
            
            feature_mask = np.array(feature_mask)
            
            k = (feature_mask > threshold).sum() % width
            if k > 0:
                # select number of features divisible by width
                feature_mask[feature_mask <= threshold] = 9999
                remove_indices = np.argpartition(feature_mask, k)[:k]
                feature_mask[remove_indices] = 9999
                    
                feature_mask = (feature_mask != 9999)
            else:
                feature_mask = (feature_mask > threshold)
            
            prunable_features = feature_mask.sum()
            if prunable_features > 0:

                gated_feature_mask = np.invert(feature_mask)
                hidden_features = gated_feature_mask.sum()
                    
                module.w1 = self._prune_linear(module.w1, module.w1.in_features, hidden_features, module.w1.weight.data[gated_feature_mask, :])
                module.w2 = self._prune_linear(module.w2, hidden_features, module.w2.out_features, module.w2.weight.data[:, gated_feature_mask])
                module.w3 = self._prune_linear(module.w3, module.w3.in_features, hidden_features, module.w3.weight.data[gated_feature_mask, :])

        return int(hidden_features)
        
        
    def _prune_linear(self, module, in_features, out_features, weights):
        raise NotImplementedError
        
    def save(self, ckpt_path, hidden_dims):
        raise NotImplementedError
        
        
class ZImageWrapper_Forge(ZImageWrapper):
    
    def __init__(self,
        model="./models/zImage_turbo_FP8.safetensors",
        text_encoder="./models/Qwen3-4B.Q6_K.gguf",
        vae="./models/ae.safetensors",
        deterministic=False,
        reserve_vram=100,
        fast_fp8=False):
            
        super(ZImageWrapper_Forge, self).__init__()
        
        from z_image.forge import memory_management
        
        memory_management.args.reserve_vram = reserve_vram
        memory_management.args.fast_fp8 = fast_fp8
        memory_management.args.deterministic = deterministic
        memory_management.init()
        
        from z_image.forge.loader import forge_loader
        from z_image.forge.lumina import FeedForward
        from z_image.forge.operations import ForgeOperations
        
        self.model = forge_loader(model, [vae, text_encoder])
        self.model.set_clip_skip(2)
        
        self.replace_fn = FeedForward
        self.module_type = ForgeOperations.Linear
        self.chunks = 1 #forge does not split feed forward in chunks
         
    @property
    def transformer(self):
        return self.model.forge_objects.unet.model
        
    def encode_text(
        self,
        prompts):
        
        if isinstance(prompts, str):
            prompts = [prompts]
        
        from z_image.forge import devices
        
        with devices.autocast():
            return self.model.get_learned_conditioning(prompts)

    def generate_image(
        self,
        prompts,    
        width = 1024,
        height = 1024,
        batch_size = 1,
        seed = -1, #this needs to be set beforehand - a seed of -1 means a blank starting noise, which will lead to terrible outputs
        sample_solver = "sample_res_multistep", #other samplers are not implemented in this minimal forge reproduction
        decode = True #use to disable VAE decoding / actual image generation, for faster collection of activations
        ):
        
        self.set_seed(seed)
        
        from z_image.forge.processing import Txt2ImgProcessing

        p = Txt2ImgProcessing(model=self.model, prompts=prompts, width=width, height=height, seed=seed)

        try:
            images = p.process_images(decode=decode)
            
        finally:
            p.close()

        return images
        
    def _prune_linear(self, module, in_features, out_features, weights):
        # create a new ForgeOperations.Linear module with a reduced number of features, and populate with pruned weights
        
        from z_image.forge.operations import ForgeOperations
     
        linear = ForgeOperations.Linear(in_features, out_features, bias=False)
        linear.weight = torch.nn.parameter.Parameter(weights)
        linear.scale_weight = module.scale_weight

        return linear
        
    def save(self, ckpt_path, hidden_dims):
        
        # save transformer model
        self.model.save_unet(ckpt_path + ".safetensors")
        
        # forge loads the model directly from the state dict, so we need no separate config (in contrast to WanGP)

    
class ZImageWrapper_WanGP(ZImageWrapper):
    
    def __init__(self,
        model="./models/ZImageTurbo_quanto_bf16_int8.safetensors",
        text_encoder="./models/Qwen3/qwen3_quanto_bf16_int8.safetensors",
        vae="./models/vae/ZImageTurbo_VAE_bf16.safetensors",
        scheduler="./models/ZImageTurbo_scheduler_config.json",
        model_type="z_image",
        deterministic=False,
        vram_profile=5,
        vram_safety_coefficient=0.85,
        perc_reserved_mem_max=0):
            
        super(ZImageWrapper_WanGP, self).__init__()
        
        from mmgp import offload
        from z_image.wgp.z_image_main import model_factory
        from z_image.wgp.z_image_transformer2d import FeedForward
        
        torch.set_default_device('cpu')
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
        
        self.model_type = model_type
        self.model = model_factory(model_filename=model, model_type=model_type, text_encoder_filename=text_encoder, vae_filename=vae)
        self.replace_fn = FeedForward
        self.module_type = torch.nn.Linear
        self.chunks = 3 #Each feed forward step is repeated 3 times, for chunks of the input tensor
        
        pipe = {
                "transformer": self.model.transformer,
                "text_encoder": self.model.text_encoder,
                "vae": self.model.vae,
            }
            
        kwargs = {}
        mmgp_profile = self._init_pipe(pipe, kwargs, vram_profile)

        self.offloadobj = offload.profile(pipe, profile_no= mmgp_profile, compile = "", quantizeTransformer = False, loras = [], perc_reserved_mem_max = perc_reserved_mem_max , vram_safety_coefficient = vram_safety_coefficient , convertWeightsFloatTo = torch.bfloat16, **kwargs)  

        offload.shared_state["_attention"] = self.attn


    def _init_pipe(self, pipe, kwargs, profile):
        # sets WanGP memory optimization profile, between 1 and 5
        # 1 = no optimizations / highest speed
        # 5 = all available optimizations / lowest speed

        kwargs["extraModelsToQuantize"]=  None
        if profile in (2, 4, 5):
            default_transformer_budget = kwargs.get("budgets", 100) 
            if isinstance(default_transformer_budget, dict):
                default_transformer_budget = default_transformer_budget.get("transformer", 100) 

            budgets = { "transformer" : default_transformer_budget, "text_encoder" : 100, "*" : 1000 if profile==5 else 3000}
            
            kwargs["budgets"] = budgets
        elif profile == 3:
            kwargs["budgets"] = { "*" : "70%" }

        if profile == 4.5:
            mmgp_profile = 4
            kwargs["asyncTransfers"] = False
        elif profile == 3.5:
            mmgp_profile = 3
            kwargs["pinnedMemory"] = False
        else:
            mmgp_profile = profile

        return mmgp_profile

    @property
    def transformer(self):
        return self.model.transformer

    def generate_image(
        self,
        prompts,    
        width = 1024,
        height = 1024,
        batch_size = 1,
        seed = -1,
        sample_solver = "default", # WanGP always uses unified solver
        decode = True #TODO: implement for WanGP wrapper
        ):

        from mmgp import offload
        
        if not isinstance(prompts, list):
            prompts = [prompts]

        self.set_seed(seed)

        torch.set_grad_enabled(False) 
       
        gc.collect()
        torch.cuda.empty_cache()
        self.transformer.cache = None

        prompt = prompts[-1]

        try:
            
            samples = self.model.generate(
                input_prompt = prompt,
                sampling_steps = self.sampling_steps,
                batch_size = batch_size,
                height = height,
                width = width,
                shift=self.flow_shift,
                sample_solver=sample_solver,
                seed=seed,
                offloadobj = self.offloadobj,                    
            )
            
        finally:
            
            if "_cache" in offload.shared_state:
                del offload.shared_state["_cache"]
                
            self.offloadobj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()
            self.transformer.cache = None

        if samples == None:
            return None
            
        return samples.cpu()


    def prune_linear(self, module, in_features, out_features, weights):
        # create a new QLinearQuantoRouter module with a reduced number of features, and populate with pruned weights
        
        from mmgp.quant_router import QLinearQuantoRouter
     
        linear = torch.nn.Linear(in_features, out_features, bias=False)
        linear.weight.data = weights.to("cuda") #for immediate test image
        
        qlinear = QLinearQuantoRouter.from_module(module=linear, weights=module.weight_qtype, optimizer=module.optimizer)
        qlinear._router_default_dtype = module._router_default_dtype
        qlinear.freeze()
        
        return qlinear


    def save(self, ckpt_path, hidden_dims):
        
        from z_image.wgp.z_image_main import get_config
        from mmgp.offload import save_model
        
        # save transformer model
        save_model(self.transformer, ckpt_path + ".safetensors")
        
        
        # save config file - needed, as WanGP is first initiating the model from a hardcoded config, and then loads the weights
        config = get_config(self.model_type)
        
        config["hidden_dims"] = hidden_dims
        
        with open(ckpt_path + ".json", 'w') as f:
            json.dump(config, f)