from config import*
from SPP_AD import*
from Wfamily import*


x_r = torch.linspace(0, 1, 100)
fam = family.cpu()

W_r = gaussian(x_r, fam[:,0], fam[:,1]).T
DWx_r = D1xgaussian(x_r, fam[:,0], fam[:,1]).T
DW2x_r = D2xgaussian(x_r, fam[:,0], fam[:,1]).T



def PINN_NTK(model):
    x_r_pinn = x_r.to(device)
    model.zero_grad()
    
    params = list(model.parameters())


    # === Residual Points ===
    grads_N_r = []
    for i in range(x_r_pinn.shape[0]):
        x_i = x_r_pinn[i].unsqueeze(0).requires_grad_(True)
        u_i = model(x_i).reshape(-1)

        u_x = torch.autograd.grad(u_i, x_i, retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x_i, retain_graph=True, create_graph=True)[0]

        N_i = e * u_xx  # Adjust PDE residual as needed

        grad_i = torch.autograd.grad(N_i, params, retain_graph=True, allow_unused=True)
        grad_i = torch.cat([
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grad_i, params)
        ])
        grads_N_r.append(grad_i)
    grads_N_r = torch.stack(grads_N_r)  # [N_r, P]
    K_rr = grads_N_r @ grads_N_r.T      # [N_r, N_r]


    # Eigenvalue Analysis
    eigvals = torch.linalg.eigvals(K_rr).real
    eigvals_sorted = torch.sort(eigvals, descending=True).values

    return abs(eigvals_sorted.cpu().numpy())






def WPINN_NTK(model):
    model.zero_grad()
    c,b = model(x_collocation)


    # === Residual Points ===
    grads_N_r = []
    for i in range(x_r.shape[0]):
        u_i = torch.dot(W_r[i], c.cpu()) + b.cpu()
        u_x_i = torch.dot(DWx_r[i], c.cpu())
        u_xx_i = torch.dot(DW2x_r[i], c.cpu())
        N_i = e * u_xx_i + (1 + e) * u_x_i + u_i
        grad_i = torch.autograd.grad(N_i, model.parameters(), retain_graph=True, allow_unused=True)
        grad_i = torch.cat([g.flatten() for g in grad_i if g is not None])
        grads_N_r.append(grad_i)
    grads_N_r = torch.stack(grads_N_r)  # Shape: [N_r, P]
    K_rr = grads_N_r @ grads_N_r.T     # [N_r, N_r]


    # Eigen Values Analysis
    eigvals = torch.linalg.eigvals(K_rr).real
    eigvals_sorted = torch.sort(eigvals, descending=True).values

    return abs(eigvals_sorted.cpu().numpy())  
