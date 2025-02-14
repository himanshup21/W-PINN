import torch

def wavelet_family():
    Jx = torch.arange(-10.0,6.0)
    Jt = torch.arange(-10.0,6.0)


    family = torch.tensor([(2**jx,2**jt,kx,kt) for jx in Jx for jt in Jt for kx in range(-2,max(int(2*2**jx),1)) for kt in range(-2, max(int(2*2**jt),1))])

    return family

def gaussian(x,t,jx,jt,kx,kt):
    return (jx*x - kx)*(jt*t - kt)*torch.exp(-((jx*x - kx)**2 + (jt*t - kt)**2)/2)

def D1xgaussian(x,t,jx,jt,kx,kt):
    return jx*(1-(jx*x - kx)**2)*(jt*t - kt)*torch.exp(-((jx*x - kx)**2 + (jt*t - kt)**2)/2)

def D1tgaussian(x,t,jx,jt,kx,kt):
    return jt*(1-(jt*t - kt)**2)*(jx*x - kx)*torch.exp(-((jx*x - kx)**2 + (jt*t - kt)**2)/2)

def D2xgaussian(x,t,jx,jt,kx,kt):
    return -(jx**2)*(jx*x - kx)*(jt*t-kt)*(3 - (jx*x - kx)**2)*torch.exp(-((jx*x - kx)**2 + (jt*t - kt)**2)/2)

def D2tgaussian(x,t,jx,jt,kx,kt):
    return -(jt**2)*(jx*x - kx)*(jt*t-ky)*(3 - (jt*t - kt)**2)*torch.exp(-((jx*x - kx)**2 + (jt*t - kt)**2)/2)


    