# adapted from https://github.com/ruchikachavhan/concept-prune/blob/main/neuron_receivers/wanda_receiver.py

import torch
from hooks.hook_manager import HookManager
from hooks.norms import TimeLayerColumnNorm


class NormCollector(HookManager):
    
    def __init__(self, T, n_layers, n_chunks=1, hook_module='transformer'):
        super(NormCollector, self).__init__(T, n_layers, n_chunks, hook_module)
        
        if hook_module == 'transformer':
            # create a dictionary to store activation norms for every time step and layer
            self.activation_norm = TimeLayerColumnNorm(T, n_layers)

    
    def hook_fn(self, module, input, output):
        ''' 
            Store the norm of the gate for each layer and timestep of the FFNs
        '''
        
        #SwiGLU gate
        
        hidden_states = module.w3(input[0])
        gate = module.w1(input[0])
        out = module._forward_silu_gating(gate, hidden_states)

        # get the input activation
        save_gate = out.clone().view(-1, out.shape[-1]).detach().cpu()
        
        # normalize across the sequence length to avoid inf values
        save_gate = torch.nn.functional.normalize(save_gate, p=2, dim=1)
        self.activation_norm.update(save_gate, self.timestep, self.layer)

        # update the time step, layer and chunk
        self.update_time_layer()

        return module.w2(out)
 
 
    def observe_activation(self, model):
        
        self.reset_time_layer()
        self.hooks = []
        
        layer = 0
        
        # hook the DiT
        if self.hook_module == 'transformer':
            for name, module in model.transformer.named_modules():
                if isinstance(module, model.replace_fn) and 'feed_forward' in name and not 'refiner' in name:
                    
                    hook = module.register_forward_hook(self.hook_fn)
                    self.hooks.append(hook)
                    
                    layer += 1
                    if layer == self.n_layers:
                        break
