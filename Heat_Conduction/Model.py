from config import *


class WPINN(nn.Module):
    def __init__(self, input_size, num_hidden_layers1, num_hidden_layers2, hidden_neurons, family_size):
        
        super(WPINN, self).__init__()
        
        self.activation = nn.Tanh()
        
        # First network: processes each (x,t) point to create single feature
        first_stage_layers = []
        
        # Input layer
        first_stage_layers.append(nn.Linear(2, hidden_neurons))  # Takes (x,t) as input
        first_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers1-1):
            first_stage_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            first_stage_layers.append(self.activation)
        
        # Output of first stage: single feature per point
        first_stage_layers.append(nn.Linear(hidden_neurons, 1))
        self.first_stage = nn.Sequential(*first_stage_layers)
        
        # Second network: processes all point features to create global coefficients
        second_stage_layers = []
        
        # Input size is now just input_size (number of points) since each point has 1 feature
        second_stage_layers.append(nn.Linear(input_size, hidden_neurons))
        second_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers2-1):  # Fewer layers in second stage
            second_stage_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            second_stage_layers.append(self.activation)
        
        # Final layer outputs the wavelet coefficients
        second_stage_layers.append(nn.Linear(hidden_neurons, family_size))
        self.second_stage = nn.Sequential(*second_stage_layers)
        
        # Initialize weights
        for network in [self.first_stage, self.second_stage]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)
        
        self.bias = nn.Parameter(torch.tensor(0.5))


    def forward(self, x, t):
        # Combine x and t into single input
        inputs = torch.stack([x, t], dim=-1)  # Shape: [batch_size, 2]
        
        # First stage: process each point to get single feature
        point_features = self.first_stage(inputs)  
        point_features = point_features.squeeze(-1)  
        # Second stage: generate global coefficients from all point features
        coefficients = self.second_stage(point_features)  # Shape: [family_size]

        bias = self.bias
        
        return coefficients, bias





class CoefficientRefinementNetwork(nn.Module):
    def __init__(self, initial_coefficients, initial_bias):
        
        super(CoefficientRefinementNetwork, self).__init__()
        
        # Store initial coefficients from two-stage network
        self.coefficients = nn.Parameter(initial_coefficients.clone().detach())
        self.bias = nn.Parameter(initial_bias.clone().detach())
        
        # Output layers for the different derivatives

    def forward(self, x, t):

        bias = self.bias
        
        return self.coefficients, bias

