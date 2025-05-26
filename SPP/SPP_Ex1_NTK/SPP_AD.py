from config import*

global e
e = torch.tensor(2**-7)

def analytical(x):
    f1 = torch.exp(-x) - torch.exp(-x/e)
    f2 = torch.exp(torch.tensor(-1.0) - torch.exp(-1/e))

    return f1/f2



u_bc_left = analytical(x_bc_left)
u_bc_right = analytical(x_bc_right)

exact = analytical(x_validation)
u_exact = analytical(x_test).numpy()