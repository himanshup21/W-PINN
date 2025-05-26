import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

class WPINN(nn.Module):
    def __init__(self, input_size, family_size, num_hidden_layers1, num_hidden_layers2, hidden_neurons1, hidden_neurons2):
        super(WPINN, self).__init__()
        
        self.activation = nn.Tanh()
        
        # First network: processes each (x,t) point to create single feature
        first_stage_layers = []
        
        # Input layer
        first_stage_layers.append(nn.Linear(2, hidden_neurons1))  # Takes (x,t) as input
        first_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers1):
            first_stage_layers.append(nn.Linear(hidden_neurons1, hidden_neurons1))
            first_stage_layers.append(self.activation)
        
        # Output of first stage: single feature per point
        first_stage_layers.append(nn.Linear(hidden_neurons1, 1))
        self.first_stage = nn.Sequential(*first_stage_layers)
        
        self.second_stage_u = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons2)
        self.second_stage_v = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons2)
        self.second_stage_p = self.create_second_stage(input_size, family_size, num_hidden_layers2, hidden_neurons2)
        
        # Initialize weights
        for network in [self.first_stage, self.second_stage_u, self.second_stage_v, self.second_stage_p]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)

        self.bias_u = nn.Parameter(torch.tensor(0.5))
        self.bias_v = nn.Parameter(torch.tensor(0.5))
        self.bias_p = nn.Parameter(torch.tensor(0.5))
        

    def create_second_stage(self, input_size, family_size, num_layers, hidden_neurons):
        layers = []
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(self.activation)

        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)
        
        layers.append(nn.Linear(hidden_neurons, family_size))
        return nn.Sequential(*layers)
        

    def forward(self, x, y):
        # Combine x and t into single input
        inputs = torch.stack([x, y], dim=-1)  # Shape: [batch_size, 2]
        
        # First stage: process each point to get single feature
        point_features = self.first_stage(inputs)  # Shape: [batch_size, 1]
        point_features = point_features.squeeze(-1)  # Shape: [batch_size]

        coeff_u = self.second_stage_u(point_features)
        coeff_v = self.second_stage_v(point_features)
        coeff_p = self.second_stage_p(point_features)

        bias_u = self.bias_u
        bias_v = self.bias_v
        bias_p = self.bias_p
        
        return (coeff_u, coeff_v, coeff_p), (bias_u, bias_v, bias_p)

