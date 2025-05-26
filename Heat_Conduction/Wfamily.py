from config import*

def wavelet_family():
    Jx = torch.arange(-6.0,6.0)
    Jt = torch.arange(-6.0,6.0)

    a = 0.2

    family = torch.tensor([(2**jx,2**jt,kx,kt) for jx in Jx for jt in Jt 
                                               for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1) 
                                               for kt in range(int(torch.floor((t_lower-a)*2**(jt))), int(torch.ceil((t_upper+a)*2**(jt))) + 1)])
    return len(family), family.to(device)


def gaussian(x, t, jx, jt, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    T = jt[:, None] * t[None, :] - kt[:, None]
    return X * T * torch.exp(-(X**2 + T**2)/2)

def D1tgaussian(x, t, jx, jt, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    T = jt[:, None] * t[None, :] - kt[:, None]
    return jt[:, None] * (1 - T**2) * X * torch.exp(-(X**2 + T**2)/2)

def D2xgaussian(x, t, jx, jt, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    T = jt[:, None] * t[None, :] - kt[:, None]
    return -(jx[:, None]**2) * X * T * (3 - X**2) * torch.exp(-(X**2 + T**2)/2)


len_family, family = wavelet_family()
print("family_len: ", len(family)) 

jx = family[:, 0]
jt = family[:, 1] 
kx = family[:, 2] 
kt = family[:, 3] 


Wfamily =  gaussian(x_collocation, t_collocation, jx, jt, kx, kt).T
DWt = D1tgaussian(x_collocation, t_collocation, jx, jt, kx, kt).T
DW2x = D2xgaussian(x_collocation, t_collocation, jx, jt, kx, kt).T

Wbc_left = gaussian(x_bc_left, t_bc, jx, jt, kx, kt).T
Wbc_right = gaussian(x_bc_right, t_bc, jx, jt, kx, kt).T
Wic = gaussian(x_ic, t_ic, jx, jt, kx, kt).T
Wval = gaussian(x_validation, t_validation, jx.cpu(), jt.cpu(), kx.cpu(), kt.cpu()).T
Wtest = gaussian(x_test, t_test, jx.cpu(), jt.cpu(), kx.cpu(), kt.cpu()).T
