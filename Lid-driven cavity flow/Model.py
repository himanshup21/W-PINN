import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class WPINN(nn.Module):
    def __init__(self, input_size, family_size,
                 num_hidden_layers1 = 2, 
                 num_hidden_layers2 = 8, 
                 hidden_neurons = 80):
        
        super(WPINN, self).__init__()
        
        self.activation = nn.Tanh()
        
        # First network: processes each (x,t) point to create single feature
        first_stage_layers = []
        
        # Input layer
        first_stage_layers.append(nn.Linear(2, hidden_neurons))  # Takes (x,t) as input
        first_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers1):
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
        
        for _ in range(num_hidden_layers2):  # Fewer layers in second stage
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
        
        # Output layers for the different derivatives
        self.output_layers = nn.ModuleList()
        for i in range(5):
            output_layer = nn.Linear(family_size, 1)
            output_layer.weight.requires_grad = False
            output_layer.bias.data = torch.tensor(0.0 if i > 0 else 0.5)
            output_layer.bias.requires_grad = i == 0
            self.output_layers.append(output_layer)

    def forward(self, x, y, W):
        # Combine x and t into single input
        inputs = torch.stack([x, y], dim=-1)  # Shape: [batch_size, 2]
        
        # First stage: process each point to get single feature
        point_features = self.first_stage(inputs)  # Shape: [batch_size, 1]
        point_features = point_features.squeeze(-1)  # Shape: [batch_size]
        # Second stage: generate global coefficients from all point features
        coefficients = self.second_stage(point_features)  # Shape: [family_size]
        
        # Generate outputs using the wavelet family
        outputs = []
        for i, layer in enumerate(self.output_layers):
            layer.weight.data = W[i]
            outputs.append(layer(coefficients))

        bias = self.output_layers[0].bias
        
        return coefficients, bias, outputs



class CoefficientRefinementNetwork(nn.Module):
    def __init__(self, initial_coefficients, initial_bias, family_size):
        
        super(CoefficientRefinementNetwork, self).__init__()
        
        # Store initial coefficients from two-stage network
        self.coefficients = nn.Parameter(initial_coefficients.clone().detach())
        
        # Output layers for the different derivatives
        self.output_layers = nn.ModuleList()
        for i in range(5):
            output_layer = nn.Linear(family_size, 1)
            output_layer.weight.requires_grad = False
            output_layer.bias.data = torch.tensor(0.0 if i > 0 else initial_bias.item())
            output_layer.bias.requires_grad = i == 0
            self.output_layers.append(output_layer)

    def forward(self, x, y, W):
        # Just use the stored coefficients directly
        outputs = []
        for i, layer in enumerate(self.output_layers):
            layer.weight.data = W[i]
            outputs.append(layer(self.coefficients))

        bias = self.output_layers[0].bias
        
        return self.coefficients, bias, outputs