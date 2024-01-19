from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Embedding
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_CLUSTER
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes

class LINE(nn.Module):
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
        batch = batch.repeat(self.walks_per_node)

        ns = torch.randint(self.num_nodes, (batch.size(0), 1),
                           dtype=batch.dtype, device=batch.device)
        ns = torch.cat([batch.view(-1, 1), ns], dim=-1) #concatenates the tensors
        return(ns)
    
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
    
    