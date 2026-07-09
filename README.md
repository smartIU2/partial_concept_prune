## Features

Partial Concept Pruning further reduces the size of a quantized Z-Image Turbo text-to-image model, to increase inference speed on low-end hardware with 8GB VRAM or less.

Build upon [ConceptPrune](https://github.com/ruchikachavhan/concept-prune) and the [Wanda score](https://doi.org/10.48550/arXiv.2306.11695) it structurally prunes a given range of FFN layers, by removing weights that are attributed to unwanted or unnecessary concepts for your intended application.

The implementation is designed to work locally on the same low-end hardware in less than half an hour. To this end, the memory management of [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) or alternatively [WanGP](https://github.com/deepbeepmeep/Wan2GP) is used for handling the model loading during pruning.


## Partial Concept Pruning

### Environment Setup

If you already have a working environment for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), you can reuse it without any further installations.

Otherwise: 

Python version 3.10 or newer needs to be installed on your machine.

PyTorch is required, so you need to install torch & torchvision. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) for your environment. The newest version 2.10 is highly recommended.

Then install the remaining requirements:

```commandline
pip install -r requirements.txt
```

### Model Download

Naturally, you need a Z-Image Turbo model to be pruned. The method is designed to work with either an FP8 quantized variant for use with [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) or the BF16 - INT8 variant that was created specifically for [WanGP](https://github.com/deepbeepmeep/Wan2GP). Other variants might or might not work.

If you haven't used either application before, [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) is recommended.


For [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo):

Download the LLM, VAE and Z-Image Turbo model of your choice and put them into the "models" folder in the repository. GGUF and safetensors are supported.

The implementation was tested with the following:

https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q6_K.gguf  
https://huggingface.co/Kijai/flux-fp8/blob/main/flux-vae-bf16.safetensors  
https://civitai.com/models/2170391/z-image-turbo-fp8-kijai  


For [WanGP](https://github.com/deepbeepmeep/Wan2GP):

If not done yet, use the Z-Image Turbo model once in the application to automatically download all model files.

Then copy the following files and folders from the "ckpts" folder in WanGP to the "models" folder in this repository:

```files
Qwen3
-config.json
-generation_config.json
-merges.txt
-special_tokens_map.json
-tokenizer.json
-tokenizer_config.json
-vocab.json
-added_tokens.json
-chat_template.jinja
-model.txt
-qwen3_quanto_bf16_int8.safetensors
ZImageTurbo_quanto_bf16_int8.safetensors
ZImageTurbo_scheduler_config.json
ZImageTurbo_VAE_bf16.safetensors
ZImageTurbo_VAE_bf16_config.json
```


### Prompt Preparation

The pruning method is meant to remove weights that are not needed for your specific application. Therefore, you need to provide a set of prompts to define what you will generate, and what not.

First, collect approximately 10 to 20 prompts that are *typical* of your intended application and put them into a .txt file inside the "prompts" folder. One prompt per line. These can and should be as long and detailed as optimal for Z-Image Turbo, i.e., at least 80 words are recommended.

Then, create a copy of the file and adapt each prompt in a way that is very *unlikely* for your intended application. If you are used to adding negative prompts (for other text-to-image models, that actually support them) these would be good examples. The method works best if you stick to one unwanted overall concept at a time that is properly described with different words and phrases for each prompt, and ideally put into varying positions of the prompt. That is, do not just add a comma separated list of unwanted adjectives at the end, such as "bad, ugly, deformed". If your unwanted concept contradicts an existing description in the prompt you should directly exchange it.

Repeat the second step, until you have at least 3 unwanted concepts.


### Configuration

The pruning method supports a number of settings that need to be defined in a .yaml file in the "configs" folder. See the included example config for the structure.

The individual settings have the following effects:

- hook_module: remnant of ConceptPrune, only 'transformer' allowed at the moment
- wrapper: 'forge' or 'wangp' depending on your chosen model / inference interface
- model: Z-Image Turbo model path; default for WanGP is "./models/ZImageTurbo_quanto_bf16_int8.safetensors"
- text_encoder: Qwen3 model path; default for WanGP is "./models/Qwen3/qwen3_quanto_bf16_int8.safetensors"
- vae: Flux VAE path; default for WanGP is "./models/ZImageTurbo_VAE_bf16.safetensors"
- scheduler: only relevant for wangp wrapper, sets scheduler config; default is "./models/ZImageTurbo_scheduler_config.json"
- profile: only relevant for wangp wrapper, sets [performance profile](https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/CLI.md#performance-profiles)
- target: name of the concept, has to be valid as part of a folder and filename in your environment
- prompts_neutral: name of the neutral prompts file under 'prompts', without .txt
- prompts_target: name of the concept prompts file under 'prompts', without .txt
- early_stop_layers: number of layers to consider for pruning; default is 10
- threshold_wanda_base: inverse fraction of total wanda score per layer to consider as prunable; set to 1 to include everything, similar to the original ConceptPrune implementation, whereas values between 8 and 24 will lead to better image quality 
- threshold_weight_ratio: minimal ratio of prunable weights to prune entire neuron (row of weights); default is 0.5
- threshold_timesteps: minimum number of timesteps a Wanda score has to be higher for the targeted concept than for the neutral prompt; values between 5 and 7 lead to good image quality
- model_type: only 'zImage' allowed at the moment
- model_id: name for created folder under 'results', and prefix for name of the pruned model
- seed: fixed seed for all image generations
- timesteps: number of timesteps for denoising; should be between 8 and 10 for Z-Image Turbo
- create_sample_images: set to 'false' to skip creating sample images during pruning, which considerably reduces memory consumption

Rule of thumb for the three thresholds: higher values lead to less pruning.


### Pruning

To start the actual pruning process run

```commandline
partial_concept_prune.py --config <config>
```

where <config> is the name of the config file(s), without '.yaml' to use for pruning. When supplying multiple configs, the union of all concepts will be pruned. The model and thresholds will be set according to the first config.


## Pruned Model Usage

### Forge

For [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), the pruned model will be generated in the 'models' folder.
This file can simply be copied to Forge's "\models\Stable-diffusion" subfolder and used normally.


### WanGP

For [WanGP](https://github.com/deepbeepmeep/Wan2GP), the pruned model along with a file 'z_image_pruned.json' will be generated in the 'models' folder.

Minor adjustments have to be made to "z_image_handler.py" and "z_image_transformer2d.py" inside WanGP's "\models\z_image" subfolder, to allow dynamically sized hidden dimensions:

Inside z_image_handler.py, add "z_image_pruned" to the list of query_supported_types and the query_family_maps, such as:

```code
def query_supported_types():
        return ["z_image", "z_image_pruned", "z_image_base", "z_image_control", "z_image_control2", "z_image_control2_1"]
[...]

def query_family_maps():
        models_eqv_map = {
            "z_image_control2_1" : "z_image_control2",
            "z_image_base": "z_image",
            "z_image_pruned": "z_image",
        }
[...]
```

Inside z_image_transformer2d.py, a "hidden_dim(s)" parameter has to be implemented in the init of "ZImageTransformerBlock" and "ZImageTransformer2DModel":

```code
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        [...],
        hidden_dim=None,
    ):
        [...]

        if hidden_dim is None:
            # SwiGLU FFN: hidden_dim = dim * 8/3 ≈ 2.67x expansion
            hidden_dim = int(dim / 3 * 8)
            
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)
		
		[...]
		
class ZImageTransformer2DModel(nn.Module):
    [...]
    def __init__(
        self,
        [...],
        hidden_dims=None,
    ) -> None:
	
		[...]
		
		# Main layers - use control version if enable_control
        if enable_control:
            [...]
        else:
		
            if hidden_dims is None or not len(hidden_dims) == n_layers:
                self.layers = nn.ModuleList(
                    [
                        ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                        for layer_id in range(n_layers)
                    ]
                )
            else: # pruned architecture
                self.layers = nn.ModuleList(
                    [
                        ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, hidden_dim=hidden_dims[layer_id])
                        for layer_id in range(n_layers)
                    ]
                )
		[...]
```

Put the generated 'z_image_pruned.json' file into WanGP's "\models\z_image\configs" subfolder. The changes will enable WanGP to read the individual hidden dimensions from this file.

Put the pruned model into the "ckpts" subfolder, and create a .json file under "finetunes", as usual practice in WanGP. See [Finetunes](https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/FINETUNES.md) for info. Essentially the .json should look like this:

```code
{
    "model": {
        "name": "Z-Image Turbo 6B Pruned",
        "architecture": "z_image_pruned",
        "description": "Z-Image Turbo Pruned",
        "URLs": [
            "<complete path to the pruned model file inside 'ckpts'>"
        ]
    },
    "resolution": "1024x1024",
    "batch_size": 1,
    "num_inference_steps": 9,
	"flow_shift":7.0,
    "guidance_scale": 0
}
```

## Disclaimer

No part of the source code in this repository or this documentation was created by or with the help of artificial intelligence.
