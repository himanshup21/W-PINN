from config import*

def wavelet_family():
    Jt = torch.arange(0.0,10.0)

    a = 1
    family = torch.tensor([(2**jt,kt) for jt in Jt for kt in range(int(torch.floor((t_lower-a)*2**(jt))), int(torch.ceil((t_upper+a)*2**(jt))) + 1)])
    # family = torch.tensor([(2**j,k) for j in Jt for k in range(int(2**(j+1)))])

    return len(family), family.to(device)


def gaussian(t, jt, kt):
    T = jt[:, None] * t[None, :] - kt[:, None]
    return -T * torch.exp(-(T**2)/2)

def D1tgaussian(t, jt, kt):
    T = jt[:, None] * t[None, :] - kt[:, None]
    return jt[:, None] * (T**2-1) * torch.exp(-(T**2)/2)

def D2tgaussian(t, jt, kt):
    T = jt[:, None] * t[None, :] - kt[:, None]
    return (jt[:, None]**2) * T * (3 - T**2) * torch.exp(-(T**2)/2)


len_family, family = wavelet_family()
print("family_len: ", len(family)) 

jt = family[:, 0]
kt = family[:, 1] 


Wfamily =  gaussian(t_collocation, jt, kt).T
DWt = D1tgaussian(t_collocation, jt, kt).T
DW2t = D2tgaussian(t_collocation, jt, kt).T

Wic = gaussian(t_ic, jt, kt).T
DWic = D1tgaussian(t_ic, jt, kt).T

Wval = gaussian(t_validation, jt.cpu(), kt.cpu()).T
Wtest = gaussian(t_test, jt.cpu(), kt.cpu()).T
