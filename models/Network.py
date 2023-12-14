import torch
from torch_geometric.data import Data

def construct_network_data(df_features, df_edges, directed = False):
    nodes = df_features['txId']
    labels = df_features['class']
    features = df_features[df_features.columns.drop(['txId', 'class'])]
    
    x = torch.tensor(np.array(features.values, dtype=np.double), dtype=torch.double)

    map_id = {j:i for i,j in enumerate(nodes)}
    
    if directed:
        edges_direct = df_edges[['txId1', 'txId2']]
        edges_rev = edges_direct[['txId2', 'txId1']]
        edges_rev.columns = ['txId1', 'txId2']
        edges = pd.concat([edges_direct, edges_rev])
    else:
        edges = df_edges[['txId1', 'txId2']]
        
    edges.txId1 = edges.txId1.map(map_id)
    edges.txId2 = edges.txId2.map(map_id)
    
    edges = edges.astype(int)
    
    # Reformat and convert to tensor
    edge_index = np.array(edges.values).T 
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    #create weights tensor with same shape of edge_index
    weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.double) 
    
    # Create pyG dataset
    data = Data(x=x, edge_index=edge_index)
    
    return(data)