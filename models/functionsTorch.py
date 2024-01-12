import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

def node2vec_representation(G_torch, 
                            embedding_dim=128,walk_length=20,context_size=10,walks_per_node=10,num_negative_samples=1,p=1.0,q=1.0, #node2vec hyper-parameters
                            batch_size=128, lr=0.01, max_iter=150, epochs=100): #learning hyper-parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    ).to(device)
    
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
    
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
            train_z=z[G_torch.train_mask],
            train_y=G_torch.y[G_torch.train_mask],
            test_z=z[G_torch.test_mask],
            test_y=G_torch.y[G_torch.test_mask],
            max_iter=max_iter,
        )
        return acc
    
    
    for epoch in range(epochs):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
