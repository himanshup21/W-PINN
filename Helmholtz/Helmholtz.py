from config import*

global a1, a2
a1 = 1
a2 = 8


def analytical(x,y):
    p = torch.pi
    s1 = torch.sin(a1*p*x)
    s2 = torch.sin(a2*p*y)

    return s1*s2


def righy_side(x,y):
    p = torch.pi
    s1 = torch.sin(a1*p*x)
    s2 = torch.sin(a2*p*y)

    return (1 - (a1**2 + a2**2)*p**2)*s1*s2



u_bc_left = analytical(x_bc_left, y_bc)
u_bc_right = analytical(x_bc_right, y_bc)
u_bc_bottom = analytical(x_bc, y_bc_bottom)
u_bc_top = analytical(x_bc, y_bc_top)

rhs = righy_side(x_collocation, y_collocation)
exact_validation = analytical(x_validation, y_validation)
exact_test = analytical(x_test, y_test).reshape(n_test, n_test).numpy()

