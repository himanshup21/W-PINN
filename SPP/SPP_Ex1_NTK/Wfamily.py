from config import*

def wavelet_family():
    Jx = torch.arange(0.0,10.0)

    a = 1
    family = torch.tensor([(2**jx,kx) for jx in Jx for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1)])
    # family = torch.tensor([(2**j,k) for j in Jx for k in range(int(2**(j+1)))])

    return len(family), family.to(device)


def gaussian(x, jx, kx):
    X = jx[:, None] * x[None, :] - kx[:, None]
    return -X * torch.exp(-(X**2)/2)

def D1xgaussian(x, jx, kx):
    X = jx[:, None] * x[None, :] - kx[:, None]
    return jx[:, None] * (X**2-1) * torch.exp(-(X**2)/2)

def D2xgaussian(x, jx, kx):
    X = jx[:, None] * x[None, :] - kx[:, None]
    return (jx[:, None]**2) * X * (3 - X**2) * torch.exp(-(X**2)/2)


len_family, family = wavelet_family()
print("family_len: ", len(family)) 

jx = family[:, 0]
kx = family[:, 1] 


Wfamily =  gaussian(x_collocation, jx, kx).T
DWx = D1xgaussian(x_collocation, jx, kx).T
DW2x = D2xgaussian(x_collocation, jx, kx).T

Wbc_left = gaussian(x_bc_left, jx, kx).T
Wbc_right = gaussian(x_bc_right, jx, kx).T

Wval = gaussian(x_validation, jx.cpu(), kx.cpu()).T
Wtest = gaussian(x_test, jx.cpu(), kx.cpu()).T
