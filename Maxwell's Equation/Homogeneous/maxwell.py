import torch
import numpy as np

global mode
mode = 4

def analytical(x,t):
    p = torch.pi
    sx = torch.sin(mode*p*x)
    st = torch.sin(mode*p*t)
    cx = torch.cos(mode*p*x)
    ct = torch.cos(mode*p*t)
    
    E = sx*ct
    H = -cx*st

    return (E,H)

