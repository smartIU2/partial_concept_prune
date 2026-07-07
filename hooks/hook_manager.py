# adapted from https://github.com/ruchikachavhan/concept-prune/blob/main/neuron_receivers/base_receiver.py

class HookManager:
    '''
    This is the base class for applying hooks to a torch module
    '''

    def __init__(self, T, n_layers, n_chunks, hook_module):
        self.T = T
        self.n_layers = n_layers # number of feed forward layers in main transformer block
        self.n_chunks = n_chunks # WanGP implementation for DiT has chunked feed_forward
        self.hook_module = hook_module # FeedForward module of the respective implementation
        self.hooks = []
                        
        self.timestep = 0
        self.layer = 0
        self.chunk = 0
      
    def hook_fn(self, module, input, output):
        # custom hook function
        raise NotImplementedError

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        
        self.hooks = []
    
    def update_time_layer(self):
        if self.chunk == self.n_chunks - 1:
            self.chunk = 0
            if self.layer == self.n_layers - 1:
                self.layer = 0
                self.timestep += 1
            else:
                self.layer += 1
        else:
            self.chunk += 1


    def reset_time_layer(self):
        self.timestep = 0
        self.layer = 0
        self.chunk = 0
       
       
    def observe_activation(self, model):
        # register hooks
        raise NotImplementedError