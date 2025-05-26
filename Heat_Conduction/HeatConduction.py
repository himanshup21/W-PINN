from config import*

global e
e = 0.15

def analytical(x,t):
    et = 2*t-1
    ex = torch.exp(1/(et**2 + e))

    return (1-x**2)*ex


def right_side(x,t):
    et = 2*t-1
    ex = torch.exp(1/(et**2 + e))

    return 2*ex*(1 + 2*et*(x**2-1)/(et**2+e)**2)


u_ic_left = analytical(x_ic, t_ic)

u_bc_left = analytical(x_bc_left, t_bc)
u_bc_right = analytical(x_bc_right, t_bc)

rhs = right_side(x_collocation, t_collocation)
exact_validation = analytical(x_validation, t_validation)
exact_test = analytical(x_test, t_test).reshape(n_test, n_test).numpy()

