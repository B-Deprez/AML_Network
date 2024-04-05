import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GINConv, GINEConv
from src.methods.utils.decoder import *

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
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
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

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.gcn_hidden:
                h = layer(h, edge_index)
                h = F.relu(h)
                h = self.dropout(h)
            h = self.gcn2(h, edge_index)
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
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers
        self.sage_aggr = sage_aggr

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
            n_layers: int, 
            heads: int = 1, 
            dropout_rate: float = 0
            ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers
        self.heads = heads

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
        h = self.gat1(x, edge_index, edge_attr=edge_features)
        h = F.relu(h)
        h = self.dropout(h)
        if self.n_layers > 1:
            for layer in self.gat_hidden:
                h = layer(h, edge_index, edge_attr=edge_features)
                h = F.relu(h)
                h = self.dropout(h)
            
            h = self.gat2(h, edge_index, edge_attr=edge_features)
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
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers

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
        self.num_features = num_features
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.n_layers = n_layers

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