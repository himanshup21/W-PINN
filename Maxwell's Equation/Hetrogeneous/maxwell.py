from config import*

global mu2, epsilon2
mu2 = 4.5
epsilon2 = 0.5

def analytical1(x,t):
    
    arg1 = 2*t - 2*x + 1
    arg2 = 2*t + 2*x - 1
    arg3 = 2*t - 3*x + 1.5
    
    c1 = torch.cos(arg1)
    c2 = torch.cos(arg2)
    c3 = torch.cos(arg3)

    E = c1 + 0.5*c2
    H = c1 - 0.5*c2

    return (E,H)


def analytical2(x,t):
    
    arg1 = 2*t - 2*x + 1
    arg2 = 2*t + 2*x - 1
    arg3 = 2*t - 3*x + 1.5
    
    c1 = torch.cos(arg1)
    c2 = torch.cos(arg2)
    c3 = torch.cos(arg3)

    E = 1.5*c3
    H = 0.5*c3

    return (E,H)


E_validation1, H_validation1 = analytical1(x_validation1, t_validation)
E_validation2, H_validation2 = analytical2(x_validation2, t_validation)
E_validation = torch.cat((E_validation1, E_validation2))
H_validation = torch.cat((H_validation1, H_validation2))

E_ic1, H_ic1 = analytical1(x_ic1, t_ic)
E_ic2, H_ic2 = analytical2(x_ic2, t_ic)
E_bc_left, H_bc_left = analytical1(x_bc_left, t_bc)
E_bc_right, H_bc_right = analytical2(x_bc_right, t_bc)


E_exact1, H_exact1 = analytical1(x_test1, t_test)
E_exact2, H_exact2 = analytical2(x_test2, t_test)
E_exact = torch.cat((E_exact1, E_exact2)).reshape(n_test, n_test).numpy()
H_exact = torch.cat((H_exact1, H_exact2)).reshape(n_test, n_test).numpy()