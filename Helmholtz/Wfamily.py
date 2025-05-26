from config import*

def waveley_family():
    Jx = torch.arange(-4.0,6.0)
    Jy = torch.arange(-4.0,6.0)

    a = 0.5

    family = torch.tensor([(2**jx,2**jy,kx,ky) for jx in Jx for jy in Jy 
                                               for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1) 
                                               for ky in range(int(torch.floor((y_lower-a)*2**(jy))), int(torch.ceil((y_upper+a)*2**(jy))) + 1)])
    
    

    return len(family), family.to(device)


def gaussian(x, y, jx, jy, kx, ky):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    return X * Y * torch.exp(-(X**2 + Y**2)/2)


def D2xgaussian(x, y, jx, jy, kx, ky):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    return -(jx[:, None]**2) * X * Y * (3 - X**2) * torch.exp(-(X**2 + Y**2)/2)

def D2ygaussian(x, y, jx, jy, kx, ky):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    return -(jy[:, None]**2) * X * Y * (3 - Y**2) * torch.exp(-(X**2 + Y**2)/2)


len_family, family = waveley_family()
print("family_len: ", len(family)) 

jx = family[:, 0]
jy = family[:, 1] 
kx = family[:, 2] 
ky = family[:, 3] 


Wfamily =  gaussian(x_collocation, y_collocation, jx, jy, kx, ky).T
DW2x = D2xgaussian(x_collocation, y_collocation, jx, jy, kx, ky).T
DW2y = D2ygaussian(x_collocation, y_collocation, jx, jy, kx, ky).T

Wbc_left = gaussian(x_bc_left, y_bc, jx, jy, kx, ky).T
Wbc_right = gaussian(x_bc_right, y_bc, jx, jy, kx, ky).T
Wbc_bottom = gaussian(x_bc, y_bc_bottom, jx, jy, kx, ky).T
Wbc_top = gaussian(x_bc, y_bc_top, jx, jy, kx, ky).T
Wval = gaussian(x_validation, y_validation, jx.cpu(), jy.cpu(), kx.cpu(), ky.cpu()).T
Wtest = gaussian(x_test, y_test, jx.cpu(), jy.cpu(), kx.cpu(), ky.cpu()).T
