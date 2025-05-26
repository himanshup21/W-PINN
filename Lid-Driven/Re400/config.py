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
device = torch.device('cuda:0')
torch.cuda.empty_cache()

Re = 400

torch.cuda.manual_seed(121)

class DataConfig:
    def __init__(self):
        # Sample sizes
        self.n_collocation = 40000
        self.n_boundary = 1000
        self.n_test = 251
        self.n_additional = 5000
        
        # Domain
        self.x_lower = 0
        self.x_upper = 1
        self.y_lower = 0
        self.y_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower)#.to(device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower)#.to(device)
        add_x1 = torch.rand(self.n_additional) * 0.1
        add_x2 = torch.rand(self.n_additional) * 0.1 + 0.9
        add_y1 = torch.rand(self.n_additional) * 0.1 + 0.9
        add_y2 = torch.rand(self.n_additional) * 0.1 + 0.9

        x_collocation = torch.cat([x_collocation, add_x1, add_x2]).to(device)
        y_collocation = torch.cat([y_collocation, add_y1, add_y2]).to(device)


        # sampler = qmc.Sobol(d = 2, scramble = True, seed = 501)
        # sobol_sequence_collocation = sampler.random(n = self.n_collocation)
        # sobol_sequence_boundary = sampler.random(n = self.n_boundary)

        # x_collocation = torch.tensor(sobol_sequence_collocation[:,0].flatten()*(self.x_upper - self.x_lower) + self.x_lower).float()
        # y_collocation = torch.tensor(sobol_sequence_collocation[:,1].flatten()*(self.y_upper - self.y_lower) + self.y_lower).float()



        # add_x1 = torch.rand(self.n_additional) * 0.1
        # add_x2 = torch.rand(self.n_additional) * 0.1 + 0.9
        # add_y1 = torch.rand(self.n_additional) * 0.1 + 0.9
        # add_y2 = torch.rand(self.n_additional) * 0.1 + 0.9

        # x_collocation = torch.cat([x_collocation, add_x1, add_x2]).to(device)
        # y_collocation = torch.cat([y_collocation, add_y1, add_y2]).to(device)

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

        x_data = torch.tensor([0.1, 0.5, 0.9, 0.5, 0.5]).to(device)
        y_data = torch.tensor([0.5, 0.5, 0.5, 0.1, 0.9]).to(device)
        vel_data = torch.tensor([0.23844, 0.12619, 0.42537, 0.14352, 0.35486]).to(device)

        # u_y_validation = torch.tensor([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]).to(self.device)
        # u_x_validation = torch.tensor([0.5]*len(u_y_validation)).to(self.device)

        # v_x_validation = torch.tensor([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]).to(self.device)
        # v_y_validation = torch.tensor([0.5]*len(v_x_validation)).to(self.device)

        # u_validation_100 = torch.tensor([0.0, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.0]).to(device)
        # u_validation_400 = torch.tensor([0.0, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299, -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093, 0.55892, 0.61756, 0.68439, 0.75837, 1.0]).to(device)

        # v_validation_100 = torch.tensor([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.0]).to(device)
        # v_validation_400 = torch.tensor([0.0, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124, 0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827, -0.22847, -0.19254, -0.15663, -0.12146, 0.0]).to(device)


        # u_ref = torch.tensor(np.loadtxt("ref_Re400.csv", delimiter=','))
        # v_ref = torch.tensor(np.loadtxt("v_ref_3000", delimiter=','))
        # vel_ref = (u_ref**2 + v_ref**2)**0.5
        vel_ref = torch.tensor(np.loadtxt("ref_vel.csv", delimiter=','))
        
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        ytest = torch.linspace(self.y_lower, self.y_upper, self.n_test)
        x_grid, y_grid = torch.meshgrid(xtest, ytest)
        x_test = x_grid.reshape(-1)
        y_test = y_grid.reshape(-1)
        
        return {
            'domain': (self.x_lower, self.x_upper, self.y_lower, self.y_upper),
            'collocation': (self.n_collocation+2*self.n_additional, x_collocation, y_collocation),
            'boundary': (x_bc, x_bc_upper, y_bc_lower, y_bc_upper, y_bc, x_bc_left, x_bc_right),
            # 'validation': (u_x_validation, u_y_validation, v_x_validation, v_y_validation, u_validation_100, v_validation_100),
            'data': (x_data, y_data, vel_data),
            'test': (self.n_test,x_test, y_test),
            'ref': (vel_ref.T)
        }
    

# Initialize the configuration
config = DataConfig()

# Generate all training points
points = config.generate_training_points()

# Access the points as needed
x_lower, x_upper, y_lower, y_upper = points['domain']
len_collocation, x_collocation, y_collocation = points['collocation']
x_data, y_data, vel_data = points['data']
x_bc, x_bc_upper, y_bc_lower, y_bc_upper, y_bc, x_bc_left, x_bc_right = points['boundary']
# u_x_validation, u_y_validation, v_x_validation, v_y_validation, u_validation_100, v_validation_100 = points['validation']
n_test, x_test, y_test = points['test']
vel_ref = points['ref']