import torch

def wavelet_family():
    Jx = torch.arange(-10.0,6.0)
    Jy = torch.arange(-10.0,6.0)


    family = torch.tensor([(2**jx,2**jy,kx,ky) for jx in Jx for jy in Jy for kx in range(-2,max(int(2*2**jx),1)) for ky in range(-2, max(int(2*2**jy),1))])

    return family

def gaussian(x,y,jx,jy,kx,ky):
    return (jx*x - kx)*(jy*y - ky)*torch.exp(-((jx*x - kx)**2 + (jy*y - ky)**2)/2)

def D1xgaussian(x,y,jx,jy,kx,ky):
    return jx*(1-(jx*x - kx)**2)*(jy*y - ky)*torch.exp(-((jx*x - kx)**2 + (jy*y - ky)**2)/2)

def D1ygaussian(x,y,jx,jy,kx,ky):
    return jy*(1-(jy*y - ky)**2)*(jx*x - kx)*torch.exp(-((jx*x - kx)**2 + (jy*y - ky)**2)/2)

def D2xgaussian(x,y,jx,jy,kx,ky):
    return -(jx**2)*(jx*x - kx)*(jy*y-ky)*(3 - (jx*x - kx)**2)*torch.exp(-((jx*x - kx)**2 + (jy*y - ky)**2)/2)

def D2ygaussian(x,y,jx,jy,kx,ky):
    return -(jy**2)*(jx*x - kx)*(jy*y-ky)*(3 - (jy*y - ky)**2)*torch.exp(-((jx*x - kx)**2 + (jy*y - ky)**2)/2)


    