import torch
import torch.nn as nn
import torch.Functional as F
from torch_geometric.nn import GraphConv, GCNConv, GATv2Conv, SAGEConv, GINConv

class GCN(nn.Module):
    def __init__(self, num_features, embedding_dim, output_dim, n_layers, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn1 = GCNConv(num_features, embedding_dim)
        self.gcn_hidden = nn.ModuleList()
        for _ in range(n_layers-2): #first and last layer seperately
            self.gcn_hidden.append(GCNConv(embedding_dim, embedding_dim))
        self.gcn2 = GCNConv(embedding_dim, output_dim)

    def forward(self, x, edge_index, edge_features=None):
        h = self.gcn1(x, edge_index, edge_weight=edge_features)
        h = F.relu(h)
        h = self.dropout(h)
        for layer in self.gcn_hidden:
            h = layer(h, edge_index, edge_weight=edge_features)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.gcn2(h, edge_index, edge_weight=edge_features)
        h = F.relu(h)

        return h

class GraphSAGE(nn.Module): #Neighbourhood sampling only in training step (via DataLoader)
    def __init__(self, num_features, embedding_dim, output_dim, n_layers, dropout_rate, sage_aggr='mean'):
        super.__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sage_aggr = sage_aggr
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.sage1 = SAGEConv(self.num_features, self.embedding_dim, aggr=self.sage_aggr)
        self.sage_hidden = nn.ModuleList()

        for _ in range(n_layers-2):
            self.sage_hidden.append(SAGEConv(self.embedding_dim, self.embedding_dim, aggr=self.sage_aggr))
        
        self.sage2 = SAGEConv(self.embedding_dim, self.output_dim, aggr=self.sage_aggr)
        
    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        for layer in self.sage_hidden:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
        h = self.sage2(h, edge_index)
        h = F.relu(h)
        return h


class GAT(nn.Module):
    def __init__(self, num_features, embedding_dim, output_dim, heads, n_layers, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.gat1 = GATv2Conv(num_features, embedding_dim, heads=heads)
        self.gat_hidden = nn.ModuleList()
        for _ in range(n_layers-2):
            self.gcn_hidden.append(GATv2Conv(heads*embedding_dim, embedding_dim, heads=heads))
        self.gat2 = GATv2Conv(heads*embedding_dim, output_dim, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_features=None):
        h = self.gat1(x, edge_index, edge_weight=edge_features)
        h = F.relu(h)
        h = self.dropout(h)
        for layer in self.gat_hidden:
            h = layer(h, edge_index, edge_weight=edge_features)
            h = F.relu(h)
            h = self.dropout(h)
        
        h = self.gat2(h, edge_index, edge_weight=edge_features)
        h = F.relu(h)
        return h

class GIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index, edge_features=None):
        pass

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
            

