import torch

# Global device configuration
global device
device = torch.device('cuda:1')
torch.cuda.empty_cache()

class DataConfig:
    def __init__(self):
        # Sample sizes
        self.n_collocation = 20000
        self.n_initial = 1000
        self.n_boundary = 1000
        self.n_validation = 1000
        self.n_test = 100
        
        # Domain
        self.x_lower = 0
        self.x_upper = 1
        self.t_lower = 0
        self.t_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        t_collocation = (torch.rand(self.n_collocation) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        
        # Boundary condition points
        x_ic = (torch.rand(self.n_initial) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        t_ic = self.t_lower * torch.ones(self.n_boundary).to(self.device)
        
        
        t_bc = (torch.rand(self.n_boundary) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        x_validation = (torch.rand(self.n_validation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        t_validation = (torch.rand(self.n_validation) * (self.t_upper - self.t_lower) + self.t_lower).to(self.device)
        
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        ttest = torch.linspace(self.t_lower, self.t_upper, self.n_test)
        x_grid, t_grid = torch.meshgrid(xtest, ttest)
        x_test = x_grid.reshape(-1)
        t_test = t_grid.reshape(-1)
        
        return {
            'collocation': (self.n_collocation, x_collocation, t_collocation),
            'initial': (x_ic, t_ic),
            'boundary': (t_bc, x_bc_left, x_bc_right),
            'validation': (x_validation, t_validation),
            'test': (self.n_test, x_grid, t_grid, x_test, t_test)
        }