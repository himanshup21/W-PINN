import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

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
        
        self.second_stage_E = self.create_second_stage(input_size, family_size, num_hidden_layers2-1, hidden_neurons)
        self.second_stage_H = self.create_second_stage(input_size, family_size, num_hidden_layers2-1, hidden_neurons)
        
        # Initialize weights
        for network in [self.first_stage, self.second_stage_E, self.second_stage_H]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)


        self.bias_E = nn.Parameter(torch.tensor(0.5))
        self.bias_H = nn.Parameter(torch.tensor(0.5))
        

    def create_second_stage(self, input_size, family_size, num_layers, hidden_neurons):
        layers = []
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(self.activation)

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)
        
        layers.append(nn.Linear(hidden_neurons, family_size))
        return nn.Sequential(*layers)
        


    def forward(self, x, t):
        # Combine x and t into single input
        inputs = torch.stack([x, t], dim=-1)  # Shape: [batch_size, 2]
        
        # First stage: process each point to get single feature
        point_features = self.first_stage(inputs)  
        point_features = point_features.squeeze(-1)  

        coeff_E = self.second_stage_E(point_features)
        coeff_H = self.second_stage_H(point_features)

        bias_E = self.bias_E
        bias_H = self.bias_H
        
        return (coeff_E, coeff_H), (bias_E, bias_H)





class CoefficientRefinementNetwork(nn.Module):
    def __init__(self, initial_coefficients, initial_bias):
        
        super(CoefficientRefinementNetwork, self).__init__()
        
        self.coefficients_E = nn.Parameter(initial_coefficients[0].clone().detach())
        self.coefficients_H = nn.Parameter(initial_coefficients[1].clone().detach())

        self.bias_E = nn.Parameter(initial_bias[0].clone().detach())
        self.bias_H = nn.Parameter(initial_bias[1].clone().detach())

    
    def forward(self, x, t):

        coeff_E = self.coefficients_E
        coeff_H = self.coefficients_H 

        bias_E = self.bias_E
        bias_H = self.bias_H
        
        return (coeff_E, coeff_H), (bias_E, bias_H)