import torch
from torch_geometric.data import Data


class network_AML(df_features, df_edges, directed = False):
    def __init__(self):
        self.df_features = df_features
        self.df_edges = df_edges
        
        self.nodes, self.edges, self.map_id = set_up_network_info()
        
        self.network_nx = construct_network_nx()
        self.network_nk = construct_network_nk()
        self.network_torch = construct_network_torch()
        
    def set_up_network_info(self):
        nodes = self.df_features['txId']
        map_id = {j:i for i,j in enumerate(self.nodes)} #add map to have uniform mapping over the three networks
        if self.directed:
            edges_direct = self.df_edges[['txId1', 'txId2']]
            edges_rev = edges_direct[['txId2', 'txId1']]
            edges_rev.columns = ['txId1', 'txId2']
            edges = pd.concat([edges_direct, edges_rev])
        else:
            edges = self.df_edges[['txId1', 'txId2']]    
            
        edges.txId1 = edges.txId1.map(self.map_id)
        edges.txId2 = edges.txId2.map(self.map_id)
        
        edges = edges.astype(int)
        
        return(nodes, edges, map_id)
        
    def construct_network_nx(self):
        
    def construct_network_nk(self):
    
    def construct_network_torch(self):
        labels = self.df_features['class']
        features = self.df_features[self.df_features.columns.drop(['txId', 'class'])]
        
        x = torch.tensor(np.array(features.values, dtype=np.double), dtype=torch.double)
        
        # Reformat and convert to tensor
        edge_index = np.array(self.edges.values).T 
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        #create weights tensor with same shape of edge_index
        weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.double) 
        
        # Create pyG dataset
        data = Data(x=x, edge_index=edge_index)
        
        return(data)