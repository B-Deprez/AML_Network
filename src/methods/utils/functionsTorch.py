from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from torch_geometric.nn import Node2Vec
from multiprocessing import cpu_count
import sys

def node2vec_representation_torch(G_torch: Data, train_mask: Tensor, test_mask: Tensor,
                            embedding_dim: int = 128,walk_length: int =20,context_size: int =10,walks_per_node: int =10,num_negative_samples: int =1,p: float =1.0,q: float =1.0, #node2vec hyper-parameters
                            batch_size: int =128, lr: float =0.01, max_iter: int =150, n_epochs: int =100): #learning hyper-parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    n_obs = G_torch.num_nodes

    model = Node2Vec(
        G_torch.edge_index,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        sparse=True,
        num_nodes=n_obs,
    ).to(device)

    num_workers = int(cpu_count()/2) if sys.platform == 'linux' else 0 
    batch_size = int(n_obs/10)
    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)
    
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    
    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(
            train_z=z[train_mask],
            train_y=G_torch.y[train_mask],
            test_z=z[test_mask],
            test_y=G_torch.y[test_mask],
            max_iter=max_iter,
        )
        return acc
    
    print("Epochs: ", n_epochs)
    for epoch in range(n_epochs):
        loss = train()
        #acc = test()
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: NA')

    return model

def train_GNN_old(
        data: Data,
        model: nn.Module,
        loader: DataLoader = None,
        lr: float = 0.02,
        train_mask: Tensor = None
        ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
    criterion = nn.CrossEntropyLoss()  # Define loss function.
    if loader is None:
        optimizer.zero_grad()
        y_hat, h = model(data.x, data.edge_index.to(device))
        y = data.y
        loss = criterion(y_hat[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    else:
        loader = loader #User-specified loader. Intetended mainly for GraphSAGE.
        for batch in loader:
            optimizer.zero_grad()
            out, h = model(batch.x, batch.edge_index.to(device))
            y_hat = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

    return(loss)

def test_GNN(
        data: Data, 
        model: nn.Module,
        test_mask: Tensor = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    criterion = nn.CrossEntropyLoss()
    out, h = model(data.x, data.edge_index.to(device))
    if test_mask is None: # If no test_mask is provided, use all data
        y_hat = out
        y = data.y
    else:
        y_hat = out[test_mask].squeeze()
        y = data.y[test_mask].squeeze()
    loss = criterion(y_hat, y)
    return(loss)