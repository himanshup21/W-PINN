from config import*

def wavelet_family():
    Jx = torch.arange(-5.0,6.0)
    Jt = torch.arange(-5.0,6.0)

    # family = torch.tensor([(2**jx,2**jt,kx,kt) for jx in Jx for jt in Jt for kx in range(-2,max(int(2*2**jx),1)) for kt in range(-2, max(int(2*2**jt),1))])
    a = 0.5

    family = torch.tensor([(2**jx,2**jt,kx,kt) for jx in Jx for jt in Jt 
                                               for kx in range(int(torch.floor((x_lower-a)*2**(jx))), int(torch.ceil((x_upper+a)*2**(jx))) + 1) 
                                               for kt in range(int(torch.floor((t_lower-a)*2**(jt))), int(torch.ceil((t_upper+a)*2**(jt))) + 1)])
    return len(family), family.to(device)

def gaussian(x, t, jx, jt, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    T = jt[:, None] * t[None, :] - kt[:, None]
    ex = torch.exp(-(X**2 + T**2)/2)

    return X * T * ex

def D1xgaussian(x, t, jx, jt, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    T = jt[:, None] * t[None, :] - kt[:, None]
    ex = torch.exp(-(X**2 + T**2)/2)

    return jx[:, None] * (1 - X**2) * T * ex
def D1tgaussian(x, t, jx, jt, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    T = jt[:, None] * t[None, :] - kt[:, None]
    ex = torch.exp(-(X**2 + T**2)/2)

    return jt[:, None] * (1 - T**2) * X * ex


len_family, family = wavelet_family()
print("family_len: ", len(family)) 

jx = family[:, 0]
jt = family[:, 1] 
kx = family[:, 2] 
kt = family[:, 3] 

Wfamily1 = gaussian(x_collocation1, t_collocation, jx, jt, kx, kt).T
Wfamily2 = gaussian(x_collocation2, t_collocation, jx, jt, kx, kt).T

DWx1 = D1xgaussian(x_collocation1, t_collocation, jx, jt, kx, kt).T
DWt1 = D1tgaussian(x_collocation1, t_collocation, jx, jt, kx, kt).T
DWx2 = D1xgaussian(x_collocation2, t_collocation, jx, jt, kx, kt).T
DWt2 = D1tgaussian(x_collocation2, t_collocation, jx, jt, kx, kt).T

Wic1 = gaussian(x_ic1, t_ic, jx, jt, kx, kt).T
Wic2 = gaussian(x_ic2, t_ic, jx, jt, kx, kt).T
Wbc_left = gaussian(x_bc_left, t_bc, jx, jt, kx, kt).T
Wbc_right = gaussian(x_bc_right, t_bc, jx, jt, kx, kt).T

Wint = gaussian(x_interface, t_interface, jx, jt, kx, kt).T

WValidation1 = gaussian(x_validation1, t_validation, jx, jt, kx, kt).T
WValidation2 = gaussian(x_validation2, t_validation, jx, jt, kx, kt).T

WTest1 = gaussian(x_test1, t_test, jx.cpu(), jt.cpu(), kx.cpu(), kt.cpu()).T
WTest2 = gaussian(x_test2, t_test, jx.cpu(), jt.cpu(), kx.cpu(), kt.cpu()).T
