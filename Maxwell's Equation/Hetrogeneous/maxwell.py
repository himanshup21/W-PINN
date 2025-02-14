import torch
import numpy as np
from config import DataConfig, device

global device, mu2, epsilon2
mu2 = 4.5
epsilon2 = 0.5

def analytical1(x,t):
    
    arg1 = 2*t - 2*x + 1
    arg2 = 2*t + 2*x - 1
    arg3 = 2*t - 3*x + 1.5
    
    c1 = torch.cos(arg1)
    c2 = torch.cos(arg2)
    c3 = torch.cos(arg3)

    E = c1 + 0.5*c2
    H = c1 - 0.5*c2

    return (E,H)


def analytical2(x,t):
    
    arg1 = 2*t - 2*x + 1
    arg2 = 2*t + 2*x - 1
    arg3 = 2*t - 3*x + 1.5
    
    c1 = torch.cos(arg1)
    c2 = torch.cos(arg2)
    c3 = torch.cos(arg3)

    E = 1.5*c3
    H = 0.5*c3

    return (E,H)