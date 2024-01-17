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