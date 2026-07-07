# adapted from https://github.com/ruchikachavhan/concept-prune/blob/main/utils/base_utils.py

import os
import yaml

class Config:
    
    def __init__(self, path):
        # load config file
        with open(os.path.join('configs', path + '.yaml')) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)   

        # set directory paths        
        self.res_path = os.path.join('results', f'seed_{self.seed}', self.model_type, self.model_id)
        
        self.activations_path = os.path.join(self.res_path, 'activations')
        self.images_path = os.path.join(self.res_path, 'images', self.target)
        self.weights_path = os.path.join(self.res_path, 'prunable_weights', self.target, str(self.threshold_wanda_base))

    def make_dirs(self):
        # create all directories for this config
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        
        if not os.path.exists(self.activations_path):
            os.makedirs(self.activations_path)
        
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
      
        if self.create_sample_images and not os.path.exists(self.images_path):
            os.makedirs(self.images_path)

    def get_neutral_prompts(self):
        # return the neutral prompts
        
        prompts = []
        with open(os.path.join('prompts', self.prompts_neutral + '.txt'), 'r', encoding="utf-8") as f:
            prompts = f.readlines()
            
        return [prompt.strip() for prompt in prompts]  

    def get_target_prompts(self):
        # return the prompts containing prunable concept
        
        prompts = []
        with open(os.path.join('prompts', self.prompts_target + '.txt'), 'r', encoding="utf-8") as f:
            prompts = f.readlines()
            
        return [prompt.strip() for prompt in prompts]  
        
    def __repr__(self):
        for key, value in self.config.items():
            if value is not None:
                print(f"{key}: {value}")