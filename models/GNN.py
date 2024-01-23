import torch
import torch.nn as nn
import torch.Functional as F
from torch_geometric.nn import GraphConv, GCNConv, GATv2Conv, SAGEConv, GINConv



class GNN(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden, dropout, model):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden = hidden
        self.dropout = dropout
        self.model = model

        if self.model == 'GCN':
            self.conv1 = GCNConv(self.num_features, self.hidden)
            self.conv2 = GCNConv(self.hidden, self.num_classes)
        elif self.model == 'GAT':
            self.conv1 = GATv2Conv(self.num_features, self.hidden, heads=8, dropout=self.dropout)
            self.conv2 = GATv2Conv(8 * self.hidden, self.num_classes, heads=1, concat=False, dropout=self.dropout)
        elif self.model == 'SAGE':
            self.conv1 = SAGEConv(self.num_features, self.hidden)
            self.conv2 = SAGEConv(self.hidden, self.num_classes)
        elif self.model == 'GIN':
            self.conv1 = GINConv(nn.Sequential(nn.Linear(self.num_features, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.num_classes)))
        else:
            raise NotImplementedError
        
        # Functions below are just to have examples of how to add hidden layers
        # Need to adapt this to the models above
        # Add hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.hidden_layers.append(nn.Linear(self.hidden, self.hidden))

        def forward(self, x, edge_index):
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return x
            

