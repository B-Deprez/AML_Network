import torch.nn as nn
import torch.nn.functional as F

class Decoder_linear(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            output_dim: int=2
            ):
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, output_dim)

    def forward(self, embedding):
        h = self.layer1(embedding)
        return h
    
class Decoder_deep(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            num_layers: int,
            hidden_dim: int,
            output_dim: int=2
            ):
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding):
        h = self.layer1(embedding)
        h = F.relu(h)
        for layer in self.layers:
            h = layer(h)
            h = F.relu(h)
        h = self.layer2(h)
        return h
    
class Decoder_linear_norm(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            output_dim: int=2
            ):
        super().__init__()
        self.normalise = nn.LayerNorm(embedding_dim)
        self.layer1 = nn.Linear(embedding_dim, output_dim)

    def forward(self, embedding):
        h = self.normalise(embedding)
        h = self.layer1(h)
        return h

class Decoder_deep_norm(nn.Module):
    def __init__(
            self, 
            embedding_dim: int, 
            num_layers: int,
            hidden_dim: int,
            output_dim: int=2
            ):
        super().__init__()
        self.normalise = nn.LayerNorm(embedding_dim)
        self.layer1 = nn.Linear(embedding_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding):
        h = self.normalise(embedding)
        h = self.layer1(h)
        h = F.relu(h)
        for layer in self.layers:
            h = layer(h)
            h = F.relu(h)
        h = self.layer2(h)
        return h
