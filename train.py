import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.functionsNetworkX import *
from models.functionsNetworKit import *
from models.functionsTorch import *
from models.GNN import *
from models.LINE import *
from utils.Network import *
from utils.DatasetConstruction import *

def train_node2vec():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def train_LINE():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

if __name__ == "__main__":
    ### Load Elliptic Dataset ###
    ntw = load_elliptic()

    ### Train networkx ###
    ntw_nx = ntw.get_network_nx()
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}
    features_nx_df = local_features_nx(ntw_nx, fraud_dict)

    ### Train NetworkKit ###
    #ntw_nk = ntw.get_network_nk()
    #features_nk_df = features_nk(ntw_nk)
    
    ### Train PyTorch ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch()
    ntw_torch.x = ntw_torch.x[:,1:]
    edge_index = ntw_torch.edge_index
    num_features = ntw_torch.num_features
    num_classes = 3
    hidden_dim = 64
    embedding_dim = 16
    output_dim = 2
    n_layers = 3
    dropout_rate = 0
    batch_size=128
    lr = 0.02

    n_epochs = 5

    # GCN
    print("GCN: ")
    model_gcn = GCN(
        edge_index=edge_index, 
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate
        ).to(device)
    
    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model_gcn, batch_size=batch_size, lr=lr)
        loss_test = test_GNN(ntw_torch, model_gcn)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')

    # GraphSAGE
    print("GraphSAGE: ")
    num_neighbors = [2]*n_layers
    sage_aggr = 'mean'

    model_sage = GraphSAGE(
        edge_index=edge_index, 
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        sage_aggr=sage_aggr
        ).to(device)
    
    loader = NeighborLoader(
        ntw_torch, 
        num_neighbors = num_neighbors,
        input_nodes = ntw_torch.train_mask,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
        )
    
    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model_sage, loader=loader, lr=lr)
        loss_test = test_GNN(ntw_torch, model_sage)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')

    # GAT
    print("GAT: ")
    heads = 6

    model_gat = GAT(
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        heads=heads,
        dropout_rate=dropout_rate
    ).to(device)

    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model_gat, batch_size=batch_size, lr=lr)
        loss_test = test_GNN(ntw_torch, model_gat)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')

    # GIN
    print("GIN: ")
    model_gin = GIN(
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    ).to(device)

    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model_gin, batch_size=batch_size, lr=lr)
        loss_test = test_GNN(ntw_torch, model_gin)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')