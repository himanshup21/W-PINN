from config import *

class WPINN(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_neurons, family_size):
        
        super(WPINN, self).__init__()
        
        self.activation = nn.Tanh()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(self.activation)
        
        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)
        
        # Output of first stage: single feature per point
        layers.append(nn.Linear(hidden_neurons, family_size))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for network in [self.network]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)
        
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):

        inputs = x.reshape(-1) 
        coefficients = self.network(inputs)  
        bias = self.bias.data
        
        return coefficients, bias
    




class PINN(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_neurons):
        
        super(PINN, self).__init__()
        
        self.activation = nn.Tanh()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(1, hidden_neurons))
        layers.append(self.activation)
        
        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)
        
        # Output of first stage: single feature per point
        layers.append(nn.Linear(hidden_neurons, 1))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for network in [self.network]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)


    def forward(self, x):

        inputs = x.reshape(-1,1) 
        output = self.network(inputs)  
        
        return output