from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

class LINE_w1(nn.Module):
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        num_negative_samples: int = 1,
        order: int = 2,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__()
        
        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
        
        self.row, self.col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        
        self.EPS = 1e-15
        
        self.embedding_dim = embedding_dim
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(self.num_nodes, embedding_dim,
                                   sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
    
    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]
    
    def loader(self, **kwargs) -> DataLoader:
            return DataLoader(range(self.num_nodes), collate_fn=self.sample,
                            **kwargs)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.num_negative_samples)

        ns = torch.randint(self.num_nodes, (batch.size(0), 1),
                           dtype=batch.dtype, device=batch.device)
        ns = torch.cat([batch.view(-1, 1), ns], dim=-1) #concatenates the tensors
        return ns
    
    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch_neigh = []
        for v in batch:
            batch_neigh.append(self.neighbourhood_samples(v))
        ps = torch.cat(batch_neigh)
        return ps
    
    @torch.jit.export
    def neighbourhood_samples(self, target_node) -> Tensor:
        row, col = self.row, self.col
        edges_connected_to_node = torch.where(row == target_node)[0]
        connected_nodes = col[edges_connected_to_node]
        target_node_dup = torch.tensor(target_node).repeat(len(connected_nodes))
        return(torch.cat([target_node_dup.view(-1,1), connected_nodes.view(-1,1)], dim = -1))
    
    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor]) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)
    
    @torch.jit.export
    def loss(self, ps: Tensor, ns: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        if self.order == 1:
            start, rest = ps[:, 0], ps[:, 1:].contiguous().view(-1)

            h_start = self.embedding(start).view(ps.size(0), 1, self.embedding_dim)
            h_rest = self.embedding(rest.view(-1)).view(ps.size(0), -1, self.embedding_dim)

            out = (h_start * h_rest).sum(dim=-1).view(-1)
            pos_loss = -torch.sigmoid(out).mean()
            neg_loss = 0

        else:
            # Positive loss.
            start, res = ps[:, 0], ps[:, 1:].contiguous().view(-1)
            
            h_start = self.embedding(start).view(ps.size(0), 1, self.embedding_dim)
            h_rest = self.embedding(res.view(-1)).view(ps.size(0), -1, self.embedding_dim)
            
            out = (h_start * h_rest).sum(dim=-1).view(-1)
            
            pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

            # Negative loss.
            start, res = ns[:, 0], ns[:, 1:].contiguous().view(-1)

            h_start = self.embedding(start).view(ns.size(0), 1, self.embedding_dim)
            h_rest = self.embedding(res.view(-1)).view(ns.size(0), -1, self.embedding_dim)
            
            out = (h_start * h_rest).sum(dim=-1).view(-1)
            neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
            

        return pos_loss + neg_loss