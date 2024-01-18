from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import index2ptr

class LINE(nn.Module): # Need to look at how node2vec does this. Delayed untill after deep learning methods
    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        num_negative_samples: int = 1,
        order: int = 2
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        self.num_nodes = maybe_num_nodes(edge_index, num_nodes)
        
        row, col = sort_edge_index(edge_index, num_nodes=self.num_nodes).cpu()
        self.rowptr, self.col = index2ptr(row, self.num_nodes), col 
        
        self.EPS = 1e-15
        
        self.embedding_dim = embedding_dim
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(self.num_nodes, embedding_dim,
                                   sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)
    
    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]
    
    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node)

        ns = torch.randint(self.num_nodes, (batch.size(0), 1),
                           dtype=batch.dtype, device=batch.device)
        ns = torch.cat([batch.view(-1, 1), ns], dim=-1) #concatenates the tensors
        return(ns)
    
    @torch.jit.export
    def pos_sample(self, batch: Tensor) -> Tensor:
        batch_neigh = []
        for v in batch:
            batch_neigh.append(neighbourhood(v, data))
        ps =torch.cat(batch_neigh)
        return ps
    
    def neighbourhood_samples(self, v):
        data = self.data
        edges_connected_to_node = torch.where(data.edge_index[0] == target_node)[0]
        connected_nodes = data.edge_index[1, edges_connected_to_node]
        target_node_dup = torch.tensor(target_node).repeat(len(connected_nodes))
        return(torch.cat([target_node_dup.view(-1,1), connected_nodes.view(-1,1)], dim = -1))