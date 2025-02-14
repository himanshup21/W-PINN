import torch

# Global device configuration
global device
device = torch.device('cuda:1')
torch.cuda.empty_cache()

class DataConfig:
    def __init__(self):
        # Sample sizes
        self.n_collocation = 10000
        self.n_initial = 500
        self.n_boundary = 500
        self.n_interface = 10000
        self.n_validation = 1000
        self.n_test = 100
        
        # Domain
        self.x_lower = 0
        self.x_mid = 0.5
        self.x_upper = 1
        self.t_lower = 0
        self.t_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points

        x_collocation1 = (torch.rand(self.n_collocation) * (self.x_mid - self.x_lower) + self.x_lower).to(self.device)
        x_collocation2 = (torch.rand(self.n_collocation) * (self.x_upper - self.x_mid) + self.x_mid).to(self.device)
        t_collocation = (torch.rand(self.n_collocation) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        
        # Boundary condition points
        x_ic1 = (torch.rand(self.n_initial) * (self.x_mid - self.x_lower) + self.x_lower).to(self.device)
        x_ic2 = (torch.rand(self.n_initial) * (self.x_upper - self.x_mid) + self.x_mid).to(self.device)
        t_ic = self.t_lower * torch.ones(self.n_boundary).to(self.device)
        
        
        t_bc = (torch.rand(self.n_boundary) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        t_interface = (torch.rand(self.n_interface) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        x_interface = self.x_mid*torch.ones(self.n_interface).to(self.device)

        x_validation1 = (torch.rand(self.n_validation) * (self.x_mid - self.x_lower) + self.x_lower).to(self.device)
        x_validation2 = (torch.rand(self.n_validation) * (self.x_upper - self.x_mid) + self.x_mid).to(self.device)
        t_validation = (torch.rand(self.n_validation) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
    
        xtest1 = torch.linspace(self.x_lower, self.x_mid, int(self.n_test/2))
        xtest2 = torch.linspace(self.x_mid, self.x_upper, int(self.n_test/2))
        
        xtest = torch.cat((xtest1,xtest2))
        ttest = torch.linspace(self.t_lower, self.t_upper, self.n_test)
        
        x_grid1, t_grid = torch.meshgrid(xtest1, ttest)
        x_grid2, t_grid = torch.meshgrid(xtest2, ttest)
        
        x_test1 = x_grid1.reshape(-1)
        x_test2 = x_grid2.reshape(-1)
        t_test = t_grid.reshape(-1)

        x_grid, t_grid = torch.meshgrid(xtest, ttest)

        
        return {
            'collocation': (self.n_collocation, x_collocation1, x_collocation2, t_collocation),
            'initial': (x_ic1, x_ic2, t_ic),
            'boundary': (t_bc, x_bc_left, x_bc_right),
            'interface': (x_interface, t_interface),
            'validation': (x_validation1, x_validation2, t_validation),
            'test': (self.n_test, x_grid, t_grid, x_test1, x_test2, t_test)
        }