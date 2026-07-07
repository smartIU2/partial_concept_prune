# extracted from https://github.com/ruchikachavhan/concept-prune/blob/main/utils/base_utils.py

import torch
import json
import numpy as np

class Average:
    '''
    Class to measure average of a set of values
    for all timesteps and layers
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    
class StandardDev:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.M2 = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        if self.n < 2:
            return float('nan')
        else:
            return self.M2 / (self.n - 1)

    def stddev(self):
        return self.variance() ** 0.5


class StatMeter:
    '''
    Class to measure average and standard deviation of a set of values
    for all timesteps and layers
    '''
    def __init__(self, T, n_layers):
        self.reset()
        self.results = {}
        self.results['time_steps'] = {}
        self.T = T
        self.n_layers = n_layers
        for t in range(T):
            self.results['time_steps'][t] = {}
            for i in range(n_layers):
                self.results['time_steps'][t][i] = {}
                self.results['time_steps'][t][i]['avg'] = Average()
                self.results['time_steps'][t][i]['std'] = StandardDev()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, t, n_layer):
        self.results['time_steps'][t][n_layer]['avg'].update(val)
        self.results['time_steps'][t][n_layer]['std'].update(val)
        
    
    def save(self, path):
        for t in range(self.T):
            for i in range(self.n_layers):
                self.results['time_steps'][t][i]['avg'] = self.results['time_steps'][t][i]['avg'].avg
                self.results['time_steps'][t][i]['std'] = self.results['time_steps'][t][i]['std'].stddev()
                # check if its and array
                if isinstance(self.results['time_steps'][t][i]['avg'], np.ndarray):
                    self.results['time_steps'][t][i]['avg'] = self.results['time_steps'][t][i]['avg'].tolist()
                if isinstance(self.results['time_steps'][t][i]['std'], np.ndarray):
                    self.results['time_steps'][t][i]['std'] = self.results['time_steps'][t][i]['std'].tolist()

        with open(path, 'w') as f:
            json.dump(self.results, f)


class ColumnNormCalculator:
    def __init__(self):
        '''
        Calculated Column Norm of a matrix incrementally as rows are added
        Assumes 2D matrix
        '''
        self.A = np.zeros((0, 0))
        self.column_norms = torch.tensor([])

    def add_rows(self, rows):
        if len(self.A) == 0:  # If it's the first row
            self.A = rows
            self.column_norms = torch.norm(self.A, dim=0)
        else:
            new_row_norms = torch.norm(rows, dim=0)
            self.column_norms = torch.sqrt(self.column_norms**2 + new_row_norms**2)

    def get_column_norms(self):
        return self.column_norms



class TimeLayerColumnNorm:
    '''
    Column Norm calculator for all timesteps and layers
    '''

    def __init__(self, T, n_layers):
        self.T = T
        self.n_layers = n_layers
        self.column_norms = {}
        for t in range(T):
            self.column_norms[t] = {}
            for i in range(n_layers):
                self.column_norms[t][i] = ColumnNormCalculator()

    def update(self, rows, t, n_layer):
        self.column_norms[t][n_layer].add_rows(rows)

    def get_column_norms(self):
        results = {}
        for t in range(self.T):
            results[t] = {}
            for i in range(self.n_layers):
                results[t][i] = self.column_norms[t][i].get_column_norms()
        return results
    
    def save(self, path):
        results = self.get_column_norms()
        torch.save(results, path)