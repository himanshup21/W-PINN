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
        self.n_test = 500
        
        # Domain
        self.x_lower = 0
        self.x_upper = 1
        self.y_lower = 0
        self.y_upper = 1
        
        self.device = device
    
    def generate_training_points(self):
        """Generate all training points for the PINN"""
        # Collocation points
        x_collocation = (torch.rand(self.n_collocation) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_collocation = (torch.rand(self.n_collocation) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        
        # Boundary condition points
        x_bc = (torch.rand(self.n_initial) * (self.x_upper - self.x_lower) + self.x_lower).to(self.device)
        y_bc_lower = self.y_lower * torch.ones(self.n_boundary).to(self.device)
        y_bc_upper = self.y_upper * torch.ones(self.n_boundary).to(self.device)
        
        
        y_bc = (torch.rand(self.n_boundary) * (self.y_upper - self.y_lower) + self.y_lower).to(self.device)
        x_bc_left = self.x_lower * torch.ones(self.n_boundary).to(self.device)
        x_bc_right = self.x_upper * torch.ones(self.n_boundary).to(self.device)

        u_y_validation = torch.tensor([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0]).to(self.device)
        u_x_validation = torch.tensor([0.5]*len(u_y_validation)).to(self.device)

        v_x_validation = torch.tensor([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344, 0.5, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0]).to(self.device)
        v_y_validation = torch.tensor([0.5]*len(v_x_validation)).to(self.device)
        
        xtest = torch.linspace(self.x_lower, self.x_upper, self.n_test)
        ytest = torch.linspace(self.y_lower, self.y_upper, self.n_test)
        x_grid, y_grid = torch.meshgrid(xtest, ytest)
        x_test = x_grid.reshape(-1)
        y_test = y_grid.reshape(-1)
        
        return {
            'collocation': (self.n_collocation, x_collocation, y_collocation),
            'boundary': (x_bc, y_bc_lower, y_bc_upper, y_bc, x_bc_left, x_bc_right),
            'validation': (u_x_validation, u_y_validation, v_x_validation, v_y_validation),
            'test': (self.n_test, xtest, ytest, x_test, y_test)
        }