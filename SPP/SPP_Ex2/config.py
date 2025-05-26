import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from scipy.stats import qmc

import csv 
import os

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Global device configuration
global device
device = torch.device('cuda:0')
torch.manual_seed(121)

class DataConfig:
    def __init__(self):
        # Sample sizes
        self.n_collocation = 10**4
        self.n_validation = 1000
        self.n_intial = 100
        self.ntest = 10000
        
        # Domain bounds
        self.t_lower = 0
        self.t_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points
        # t_collocation = (torch.rand(self.n_collocation) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        sampler = qmc.Sobol(d = 1, scramble = True, seed = 501)
        sobol_sequence = sampler.random(n = self.n_collocation)* (self.t_upper - self.t_lower) + self.t_lower 

        t_collocation = torch.tensor(sobol_sequence.flatten()).float().to(device)

        # Validation points
        t_validation = (torch.rand(self.n_validation) * (self.t_upper - self.t_lower) + self.t_lower)

        # intial condition points
        t_ic = self.t_lower * torch.ones(self.n_intial).to(self.device)

        # Testing and Plotting points
        t_test = torch.linspace(self.t_lower, self.t_upper, self.ntest)
        
        return {
            'domain': (self.t_lower, self.t_upper),  
            'collocation': (self.n_collocation, t_collocation),
            'validation': (t_validation),
            'intial': (t_ic),
            'test': (t_test)
        }
    

config = DataConfig()
points = config.generate_training_points()

# Access the points as needed
t_lower, t_upper = points['domain']
n_collocation, t_collocation = points['collocation']
t_validation = points['validation']
t_ic = points['intial']
t_test = points['test']
