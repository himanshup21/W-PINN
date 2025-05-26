from config import*

def wavelet_family():
    Jx = torch.arange(-15.0,6.0)
    Jy = torch.arange(-15.0,6.0)

    # family = torch.tensor([(2**jx,2**jy,kx,ky) for jx in Jx for jy in Jy for kx in range(-2,max(int(2*2**jx),2)) for ky in range(-2, max(int(2*2**jy),2))])
    a_fact = 0.5
    family = torch.tensor([(2**jx,2**jy,kx,ky) for jx in Jx for jy in Jy
                                               for kx in range(int(torch.floor((x_lower-a_fact)*2**(jx))), int(torch.ceil((x_upper+a_fact)*2**(jx))))
                                               for ky in range(int(torch.floor((y_lower-a_fact)*2**(jy))), int(torch.ceil((y_upper+a_fact)*2**(jy))))])

    return family



def gaussian(x, y, jx, jy, kx, ky):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    ex = torch.exp(-(X**2 + Y**2)/2)

    return X * Y * ex

def D1xgaussian(x, y, jx, jy, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - kt[:, None]
    ex = torch.exp(-(X**2 + Y**2)/2)

    return jx[:, None] * (1 - X**2) * Y * ex

def D2xgaussian(x, y, jx, jy, kx, ky):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    ex = torch.exp(-(X**2 + Y**2)/2)

    return -(jx[:, None]**2) * X * Y * (3 - X**2) * ex

def D1ygaussian(x, y, jx, jy, kx, kt):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - kt[:, None]
    ex = torch.exp(-(X**2 + Y**2)/2)

    return jy[:, None] * (1 - Y**2) * X * ex

def D2ygaussian(x, y, jx, jy, kx, ky):
    X = jx[:, None] * x[None, :] - kx[:, None]
    Y = jy[:, None] * y[None, :] - ky[:, None]
    ex = torch.exp(-(X**2 + Y**2)/2)

    return -(jy[:, None]**2) * X * Y * (3 - Y**2) * ex


family = wavelet_family().to(device)
print("family_len: ", len(family))

# store_batch_size = int(len(family)*0.1)

jx = family[:, 0]
jy = family[:, 1] 
kx = family[:, 2] 
ky = family[:, 3] 

Wfamily = gaussian(x_collocation, y_collocation, jx, jy, kx, ky).T

DWx = D1xgaussian(x_collocation, y_collocation, jx, jy, kx, ky).T
DWy = D1ygaussian(x_collocation, y_collocation, jx, jy, kx, ky).T
DW2x = D2xgaussian(x_collocation, y_collocation, jx, jy, kx, ky).T
DW2y = D2ygaussian(x_collocation, y_collocation, jx, jy, kx, ky).T

Wbc_x_left = gaussian(x_bc_left, y_bc, jx, jy, kx, ky).T
Wbc_x_right = gaussian(x_bc_right, y_bc, jx, jy, kx, ky).T
Wbc_y_lower = gaussian(x_bc, y_bc_lower, jx, jy, kx, ky).T
Wbc_y_upper = gaussian(x_bc_upper, y_bc_upper, jx, jy, kx, ky).T

Wzero = gaussian(torch.zeros(1).to(device), torch.zeros(1).to(device), jx, jy, kx, ky).T

WTest = gaussian(x_test, y_test, jx.cpu(), jy.cpu(), kx.cpu(), ky.cpu()).T
