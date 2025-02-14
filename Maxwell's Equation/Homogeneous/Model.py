import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class WPINN(nn.Module):
    def __init__(self, input_size, family_size,
                 num_hidden_layers1 = 2, 
                 num_hidden_layers2 = 5, 
                 hidden_neurons = 50):
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
        
        self.second_stage_E = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons)
        self.second_stage_H = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons)
        
        # Initialize weights
        for network in [self.first_stage, self.second_stage_E, self.second_stage_H]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)


        self.output_layers_E = self.create_output_layers(family_size)
        self.output_layers_H = self.create_output_layers(family_size)
        

    def create_second_stage(self, input_size, family_size, num_layers, hidden_neurons):
        layers = []
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(self.activation)

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)
        
        layers.append(nn.Linear(hidden_neurons, family_size))
        return nn.Sequential(*layers)
        

    def create_output_layers(self, family_size):
        layers = nn.ModuleList()

        layer = nn.Linear(family_size, 1)
        layer.weight.requires_grad = False
        layer.bias.data = torch.tensor(0.5)
        layer.bias.requires_grad = True
        layers.append(layer)
        
        return layers

    def forward(self, x, t, W):
        # Combine x and t into single input
        inputs = torch.stack([x, t], dim=-1)  # Shape: [batch_size, 2]
        
        # First stage: process each point to get single feature
        point_features = self.first_stage(inputs)  # Shape: [batch_size, 1]
        point_features = point_features.squeeze(-1)  # Shape: [batch_size]

        coeff_E = self.second_stage_E(point_features)
        coeff_H = self.second_stage_H(point_features)

        # Generate outputs using the wavelet family
        self.output_layers_E[0].weight.data = W
        self.output_layers_H[0].weight.data = W
        
        outputs_E = self.output_layers_E[0](coeff_E)
        outputs_H = self.output_layers_H[0](coeff_H)


        bias_E = self.output_layers_E[0].bias
        bias_H = self.output_layers_H[0].bias
        
        return (coeff_E, coeff_H), (bias_E, bias_H), (outputs_E, outputs_H)



class CoefficientRefinementNetwork(nn.Module):
    def __init__(self, initial_coefficients, initial_bias, family_size):
        
        super(CoefficientRefinementNetwork, self).__init__()
        
        # Store initial coefficients from two-stage network
        self.E_coefficients = nn.Parameter(initial_coefficients[0].clone().detach())
        self.H_coefficients = nn.Parameter(initial_coefficients[1])

        
        self.output_layers_E = self.create_output_layers(family_size, initial_bias[0].item())
        self.output_layers_H = self.create_output_layers(family_size, initial_bias[1].item())


    def create_output_layers(self, family_size, bias):
        layers = nn.ModuleList()
    
        layer = nn.Linear(family_size, 1)
        layer.weight.requires_grad = False
        layer.bias.data = torch.tensor(bias)
        layer.bias.requires_grad = True
        layers.append(layer)
        
        return layers

    def forward(self, x, t, W):

        # Just use the stored coefficients directly
                
        self.output_layers_E[0].weight.data = W
        self.output_layers_H[0].weight.data = W
        
        outputs_E = self.output_layers_E[0](self.E_coefficients)
        outputs_H = self.output_layers_H[0](self.H_coefficients)

        bias_E = self.output_layers_E[0].bias
        bias_H = self.output_layers_H[0].bias
        
        return (self.E_coefficients, self.H_coefficients), (bias_E, bias_H), (outputs_E, outputs_H)