import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init


from scipy.stats import qmc

import numpy as np
from numpy import random as npr
import matplotlib.pyplot as plt

from tqdm import tqdm


# Global device configuration
global device, Re
device = torch.device('cuda:1')
torch.cuda.empty_cache()

Re = 100

torch.cuda.manual_seed(121)

class DataConfig:
    def __init__(self):
        # Sample sizes
        self.n_collocation = 40000
        self.n_boundary = 1000
        self.n_test = 100
        
        # Domain
        self.x_lower = 0
        self.x_upper = 1
        self.y_lower = 0
        self.y_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower).to(device)



        # sampler = qmc.Sobol(d = 2, scramble = True, seed = 501)
        # sobol_sequence_collocation = sampler.random(n = self.n_collocation)
        # sobol_sequence_boundary = sampler.random(n = self.n_boundary)

        # x_collocation = torch.tensor(sobol_sequence_collocation[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float()
        # y_collocation = torch.tensor(sobol_sequence_collocation[:,1].flatten()*(self.y_upper - self.y_lower) + self.y_lower).float()


        # Boundary condition points
        x_bc = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        # x_bc = torch.tensor(sobol_sequence_boundary[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float().to(device)
        y_bc_lower = self.y_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_upper = (torch.rand(self.n_boundary) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc_upper = self.y_upper * torch.ones(self.n_boundary).to(self.device)
        
        y_bc = (torch.rand(self.n_boundary) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        # y_bc = torch.tensor(sobol_sequence_boundary[:,1].flatten()*(self.y_upper - self.y_lower) + self.y_lower).float().to(device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        
        vel_ref = torch.tensor(np.loadtxt("ref_vel.csv"))
        
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        ytest = torch.linspace(self.y_lower, self.y_upper, self.n_test)
        x_grid, y_grid = torch.meshgrid(xtest, ytest)
        x_test = x_grid.reshape(-1)
        y_test = y_grid.reshape(-1)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper),
            'collocation': (self.n_collocation, x_collocation, y_collocation),
            'boundary': (x_bc, x_bc_upper, y_bc_lower, y_bc_upper, y_bc, x_bc_left, x_bc_right),
            'test': (self.n_test,x_test, y_test),
            'ref': (vel_ref)
        }
    

# Initialize the configuration
config = DataConfig()

# Generate all training points
points = config.generate_training_points()

# Access the points as needed
x_lower, x_upper, y_lower, y_upper = points['domain']
len_collocation, x_collocation, y_collocation = points['collocation']
x_bc, x_bc_upper, y_bc_lower, y_bc_upper, y_bc, x_bc_left, x_bc_right = points['boundary']
n_test, x_test, y_test = points['test']
vel_ref = points['ref']