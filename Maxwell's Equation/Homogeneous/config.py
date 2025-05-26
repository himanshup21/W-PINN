import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from scipy.stats import qmc

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

global device
device = torch.device('cuda:0')
torch.manual_seed(121)

class DataConfig:
    def __init__(self):
        self.n_collocation = 10000
        self.n_initial = 500
        self.n_boundary = 500
        self.n_validation = 1000
        self.n_test = 200
        
        self.x_lower = 0
        self.x_upper = 1
        self.t_lower = 0
        self.t_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""

        # sampler = qmc.Sobol(d = 2, scramble = True, seed = 501)
        # sobol_sequence_collocation = sampler.random(n = self.n_collocation)
        # sobol_sequence_icbc = sampler.random(n = self.n_boundary)

        # x_collocation = torch.tensor(sobol_sequence_collocation[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float().to(device)
        # t_collocation = torch.tensor(sobol_sequence_collocation[:,1].flatten()*(self.t_upper - self.t_lower) + self.t_lower).float().to(device)

        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        t_collocation = (torch.rand(self.n_collocation) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        
        # Boundary condition points
        x_ic = (torch.rand(self.n_initial) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        # x_ic = torch.tensor(sobol_sequence_icbc[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float().to(device)
        t_ic = self.t_lower * torch.ones(self.n_boundary).to(self.device)
        
        
        t_bc = (torch.rand(self.n_boundary) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        # t_bc = torch.tensor(sobol_sequence_icbc[:,1].flatten()*(self.t_upper - self.t_lower) + self.t_lower).float().to(device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower)
        t_validation = (torch.rand(self.n_validation) * (self.t_upper - self.t_lower) + self.t_lower)
        
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        ttest = torch.linspace(self.t_lower, self.t_upper, self.n_test)
        x_grid, t_grid = torch.meshgrid(xtest, ttest)
        x_test = x_grid.reshape(-1)
        t_test = t_grid.reshape(-1)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.t_lower, self.t_upper),
            'collocation': (self.n_collocation, x_collocation, t_collocation),
            'initial': (x_ic, t_ic),
            'boundary': (t_bc, x_bc_left, x_bc_right),
            'validation': (x_validation, t_validation),
            'test': (self.n_test, x_test, t_test)
        }
    
config = DataConfig()
points = config.generate_training_points()

x_lower, x_upper, t_lower, t_upper = points['domain']
n_collocation, x_collocation, t_collocation = points['collocation']
x_ic, t_ic = points['initial']
t_bc, x_bc_left, x_bc_right = points['boundary']
x_validation, t_validation = points['validation']
n_test, x_test, t_test = points['test']