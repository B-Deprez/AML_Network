import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GINConv, GINEConv
from models.decoder import *

# Look at having hidden_dim and only embedding_dim in final layer

class GCN(nn.Module):
    def __init__(
            self, 
            edge_index: Tensor,
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int= 1, 
            n_layers: int = 3, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.edge_index = edge_index
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_hidden = nn.ModuleList()
        self.n_layers = n_layers

        if n_layers == 1:
            self.gcn1 = GCNConv(num_features, embedding_dim)
        else:
            self.gcn1 = GCNConv(num_features, hidden_dim)
            for _ in range(n_layers-2): #first and last layer seperately
                self.gcn_hidden.append(GCNConv(hidden_dim, hidden_dim))
            self.gcn2 = GCNConv(hidden_dim, embedding_dim)

        self.out = Decoder_linear(embedding_dim, output_dim)

    def forward(self, x, edge_index, edge_features=None):
        h = self.gcn1(x, edge_index, edge_weight=edge_features)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.gcn_hidden:
                h = layer(h, edge_index, edge_weight=edge_features)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.gcn2(h, edge_index, edge_weight=edge_features)
        out = self.out(h)

        return out, h

class GraphSAGE(nn.Module): #Neighbourhood sampling only in training step (via DataLoader)
    def __init__(
            self, 
            edge_index: Tensor,
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            n_layers: int, 
            dropout_rate: float = 0, 
            sage_aggr: str='mean'
            ):
        super().__init__()
        self.edge_index = edge_index
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers

        if n_layers == 1:
            self.sage1 = SAGEConv(num_features, embedding_dim, aggr=sage_aggr)
        else:
            self.sage1 = SAGEConv(num_features, hidden_dim, aggr=sage_aggr)
            self.sage_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.sage_hidden.append(SAGEConv(hidden_dim, hidden_dim, aggr=sage_aggr))
            
            self.sage2 = SAGEConv(hidden_dim, embedding_dim, aggr=sage_aggr)

        self.out = Decoder_linear(embedding_dim, output_dim)
        
    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.sage_hidden:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.sage2(h, edge_index)
        out = self.out(h)
        
        return out, h


class GAT(nn.Module):
    def __init__(
            self, 
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            heads: int, 
            n_layers: int, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers

        if n_layers == 1:
            self.gat1 = GATv2Conv(num_features, embedding_dim, heads=heads, concat=False)
        else:
            self.gat1 = GATv2Conv(num_features, hidden_dim, heads=heads)
            self.gat_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.gat_hidden.append(GATv2Conv(heads*hidden_dim, hidden_dim, heads=heads))
            self.gat2 = GATv2Conv(heads*hidden_dim, embedding_dim, heads=heads, concat=False)

        self.out = Decoder_linear(embedding_dim)

    def forward(self, x, edge_index, edge_features=None):
        h = self.gat1(x, edge_index, edge_weight=edge_features)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.gat_hidden:
                h = layer(h, edge_index, edge_weight=edge_features)
                h = F.relu(h)
                h = self.dropout(h)
            
            h = self.gat2(h, edge_index, edge_weight=edge_features)
        out = self.out(h)
        
        return out, h

class GIN(nn.Module):
    def __init__(
            self, 
            num_features: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            n_layers: int, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)

        if n_layers == 1:
            self.gin1 = GINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ))
            
        else:
            self.gin1 = GINConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                    ))
            
            self.gin_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.gin_hidden.append(GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                        )))
            
            self.gin2 = GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ))

        self.out = Decoder_linear(embedding_dim)
    
    def forward(self, x, edge_index):
        h = self.gin1(x, edge_index)

        if self.n_layers > 1:
            for layer in self.gin_hidden:
                h = layer(h, edge_index)

            h = self.gin2(h, edge_index)
        out = self.out(h)

        return out, h


class GINE(nn.Module):
    def __init__(
            self, 
            num_features: int, 
            edge_dim: int, 
            hidden_dim: int, 
            embedding_dim: int, 
            output_dim: int, 
            n_layers: int, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout_rate)

        if n_layers == 1:
            self.gine1 = GINEConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ),
                    edge_dim=edge_dim)
        else:
            self.gine1 = GINEConv(
                nn.Sequential(
                    nn.Linear(num_features, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                    ),
                    edge_dim=edge_dim)
            
            self.gine_hidden = nn.ModuleList()
            for _ in range(n_layers-2):
                self.gine_hidden.append(GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim), 
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU()
                        ),
                        edge_dim=edge_dim))
                
            self.gine2 = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), 
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embedding_dim)
                    ),
                    edge_dim=edge_dim)
        
        self.out = Decoder_linear(embedding_dim)
        
    def forward(self, x, edge_index, edge_features):
        h = self.gine1(x, edge_index, edge_features)

        for layer in self.gine_hidden:
            h = layer(h, edge_index, edge_features)

        h = self.gine2(h, edge_index, edge_features)
        out = self.out(h)

        return out, h

# class GNN(nn.Module):
#     def __init__(self, num_features, num_classes, num_hidden_layers, hidden, dropout, model):
#         super(GNN, self).__init__()
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.num_hidden_layers = num_hidden_layers
#         self.hidden = hidden
#         self.dropout = dropout
#         self.model = model

#         self.hidden_layers = nn.ModuleList()

#         if self.model == 'GCN':
#             self.hidden_layers.append(GCNConv(self.num_features, self.hidden))
#             for _ in range(self.num_hidden_layers):
#                 self.hidden_layers.append(GCNConv(self.hidden, self.hidden))
#             self.hidden_layers.append(GCNConv(self.hidden, self.num_classes))
            
#         elif self.model == 'GAT':
#             self.hidden_layers.append(GATv2Conv(self.num_features, self.hidden, heads=8, dropout=self.dropout))
#             for _ in range(self.num_hidden_layers):
#                 self.hidden_layers.append(GATv2Conv(8 * self.hidden, self.hidden, heads=8, dropout=self.dropout))
#             self.hidden_layers.append(GATv2Conv(8 * self.hidden, self.num_classes, heads=1, concat=False, dropout=self.dropout))

#         elif self.model == 'SAGE':
#             self.hidden_layers.append(SAGEConv(self.num_features, self.hidden))
#             for _ in range(self.num_hidden_layers):
#                 self.hidden_layers.append(SAGEConv(self.hidden, self.hidden))
#             self.hidden_layers.append(SAGEConv(self.hidden, self.num_classes))

#         elif self.model == 'GIN':
#             self.conv1 = GINConv(nn.Sequential(nn.Linear(self.num_features, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden)))
#             self.conv2 = GINConv(nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.num_classes)))
#         else:
#             raise NotImplementedError

#         def forward(self, x, edge_index):
#             for layer in self.hidden_layers:
#                 x = F.relu(layer(x))
#                 x = F.dropout(x, p=self.dropout, training=self.training)
#             return x
            

