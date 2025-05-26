from config import*

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


E_validation, H_validation = analytical(x_validation, t_validation)
E_ic, H_ic = analytical(x_ic, t_ic)
E_exact, H_exact = analytical(x_test, t_test)
E_exact = E_exact.reshape(n_test, n_test).numpy()
H_exact = H_exact.reshape(n_test, n_test).numpy()