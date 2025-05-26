from config import*

global e
e = torch.tensor(2**-10)

def analytical(t):
    return t**2 + 2 - torch.exp(-t/e)

def right_side(t):
    expo = torch.exp(-t/e)
    res = (2 - expo + t**2)**2 + (3 + t)*(2*t + expo/e) + e*(2 - expo/e**2) - torch.sin(2 - expo + t**2)
    
    return res

u_ic = 1.0
Du_ic = 1/e

rhs = right_side(t_collocation)
exact = analytical(t_validation)
u_exact = analytical(t_test).numpy()