import os
import scipy
import pickle
import torch
import sys
import tqdm
import numpy as np
from argparse import ArgumentParser
from torchvision.utils import save_image

sys.path.append(os.getcwd())

from utils import Config
from hooks import NormCollector
from z_image import model_wrapper


def collect_activations(model, norms, prompts, args):
    norms_path = os.path.join(args.activations_path, f"{norms}.pt")
    
    if not os.path.exists(norms_path):
        print(f"Collecting normalized activations for {norms} prompts.")
        
        collector = NormCollector(args.timesteps, args.early_stop_layers, model.chunks, hook_module=args.hook_module)
        
        collector.observe_activation(model)
     
        # acquire norm values
        i = 0
        for prompt in tqdm.tqdm(prompts):
                
            print(f"{norms}: ", prompt)

            #observe activation
            collector.reset_time_layer()
            
            out = model.generate_image(prompt, seed=args.seed, decode=args.create_sample_images)
           
            if args.create_sample_images and (out is not None):
                # save images
                print(f"saving {norms} image")
                
                if isinstance(out, torch.Tensor):
                    save_image(out, os.path.join(args.images_path, f"{norms}_{i}.png"), nrow=8, normalize=True, value_range=(-1, 1))
                else:
                    out.save(os.path.join(args.images_path, f"{norms}_{i}.jpg"))
                    
                i += 1
        
        collector.remove_hooks()
        
        # save to pickle
        collector.activation_norm.save(norms_path)
        print(f"Saved normalized activations for {norms} prompts.")

    return norms_path

def get_weight_mask(weights_shape, layer_no, concepts):
    
    args = concepts[0]
    
    zeros = np.zeros(weights_shape)
    union_concepts = scipy.sparse.csr_matrix(zeros)
    
    # sum binary masks over all timesteps
    for t in range(0, args.timesteps):
        union_indices = scipy.sparse.csr_matrix(zeros)
        for c in concepts:
            with open(os.path.join(c.weights_path, f'timestep_{t}_layer_{layer_no}.pkl'), 'rb') as f:
                # load sparse matrix
                indices = pickle.load(f)
                # logical 'or' over concepts
                union_indices = union_indices.maximum(indices)
                
        # 'add' over timesteps
        union_concepts += union_indices

    # set layer mask if weight was affected in at least <threshold> number of timesteps
    union_concepts = union_concepts >= args.threshold_timesteps
    array = union_concepts.astype('bool').astype('int')
    
    return array.toarray()

def detect_prunable_weights(model, args):
    
    # get prompts
    base_prompts = args.get_neutral_prompts()
    target_prompts = args.get_target_prompts()

    # get activation norms
    base_norms = collect_activations(model, args.prompts_neutral, base_prompts, args)
    target_norms = collect_activations(model, args.prompts_target, target_prompts, args)
        
    act_norms_base = torch.load(base_norms)
    act_norms_target = torch.load(target_norms)

    # get the absolute value of FFN weights in layer w2
    abs_weights = {}
    layer_names = []

    if args.hook_module == 'transformer':
        
        # (feed_forward): FeedForward(
            # (w1): LinearModule*(in_features=3840, out_features=10240, bias=False)
            # (w2): LinearModule*(in_features=10240, out_features=3840, bias=False)
            # (w3): LinearModule*(in_features=3840, out_features=10240, bias=False)
        # )
        # *
        #  ForgeOperations.Linear for Forge
        #  QLinearQuantoRouter for WanGP
        
        layer = 0
        for name, module in model.transformer.named_modules():
            if isinstance(module, model.module_type) and model.module_name in name and not 'refiner' in name:
                layer_names.append(name)
  
                weight = module.weight.detach()
                if weight.dtype == torch.float8_e4m3fn:
                    # torch 2.10 / CUDA does not support some operations for fp8 yet
                    weight = weight.bfloat16()
                    # the values are only used for sorting, so no need to cast back later
                abs_weights[name] = weight.abs().cpu()
                
                layer += 1
                if layer == args.early_stop_layers:
                    break

    saved_masks = 0
    for t in range(args.timesteps):
        for l in range(args.early_stop_layers):

            mask_path = os.path.join(args.weights_path, f'timestep_{t}_layer_{l}.pkl')
            if not os.path.exists(mask_path):
                    
                print("Time step: ", t, "Layer: ", layer_names[l])
                
                # Wanda score is weights.abs() * activation
                metric_base = abs_weights[layer_names[l]] * act_norms_base[t][l]
                metric_target = abs_weights[layer_names[l]] * act_norms_target[t][l]
                
                data = metric_base.flatten().float().detach().numpy()
                
                max_098 = np.quantile(data, 0.98)
                threshold = max_098 / args.threshold_wanda_base
                
                binary_mask = torch.logical_and((metric_base <= threshold), (metric_target > metric_base))
                binary_mask = binary_mask.float()

                # convert binary mask to array
                binary_mask = binary_mask.cpu().numpy().astype(int)
                binary_mask = scipy.sparse.csr_matrix(binary_mask)
                print("Binary mask density: ", np.mean(binary_mask.toarray()))

                # save in pickle file
                with open(mask_path, 'wb') as f:
                    pickle.dump(binary_mask, f)
                
                saved_masks += 1

    if saved_masks > 0:
        print(f"Saved {saved_masks} binary masks for prunable weights.")


def prune_model(model, args, configs):
    
    # Prune and save the model
    print("Pruning model...")
   
    layer = 0
    hidden_dims = []
    if args.hook_module == 'transformer':
        
        for name, module in model.transformer.named_modules():
            if isinstance(module, model.replace_fn) and 'feed_forward' in name and not 'refiner' in name:
                
                mask = get_weight_mask(module.w2.weight.shape, layer, configs)

                dims_after_pruning = model.prune(module, mask, threshold=args.threshold_weight_ratio)
                hidden_dims.append(dims_after_pruning)
                
                layer += 1
                if layer == args.early_stop_layers:
                    #TODO: add full hidden_dims for rest of layers, for WanGP config
                    break

    # save the model (& config)
    print("Saving model...")
    target = '_'.join([c.target for c in configs])
    ckpt_name = os.path.join('models', f'{args.model_id}_{target}_w_{args.threshold_wanda_base}_t_{args.threshold_timesteps}_r_{args.threshold_weight_ratio}')
    
    model.save(ckpt_name, hidden_dims)


if __name__ == '__main__':
    
    # get config(s)
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, nargs='+', default=None, required=True)
    cmd_args = parser.parse_args()
    
    configs = []
    
    for c in cmd_args.config:
        configs.append(Config(c))
    
    args = configs[0]

    # load diffusion model
    if args.wrapper.lower() == "wangp":
        model = model_wrapper.ZImageWrapper_WanGP(profile=args.profile)
    else: #forge
        model = model_wrapper.ZImageWrapper_Forge()
        
        
    for config in configs:
        
        # assure working directories
        config.make_dirs()
        
        # collect activations and detect prunable weights
        detect_prunable_weights(model, config)
        
        
    # prune model
    prune_model(model, args, configs)