import pandas as pd
import torch
from torch_geometric.data import Data

#### Elliptic dataset ####
def load_elliptic():
    raw_paths = [
        './data/elliptic_bitcoin/raw/elliptic_txs_features.csv',
        './data/elliptic_bitcoin/raw/elliptic_txs_edgelist.csv',
        './data/elliptic_bitcoin/raw/elliptic_txs_classes.csv',
                    ]

    feat_df = pd.read_csv(raw_paths[0], header=None)
    edge_df = pd.read_csv(raw_paths[1])
    class_df = pd.read_csv(raw_paths[2])

    columns = {0: 'txId', 1: 'time_step'}
    feat_df = feat_df.rename(columns=columns)

    x = torch.from_numpy(feat_df.loc[:, 'time step':].values).to(torch.float)

    # There exists 3 different classes in the dataset:
    # 0=licit,  1=illicit, 2=unknown
    mapping = {'unknown': 2, '1': 1, '2': 0}
    class_df['class'] = class_df['class'].map(mapping)
    y = torch.from_numpy(class_df['class'].values)

    mapping = {idx: i for i, idx in enumerate(feat_df['txId'].values)}
    edge_df['txId1'] = edge_df['txId1'].map(mapping)
    edge_df['txId2'] = edge_df['txId2'].map(mapping)
    edge_index = torch.from_numpy(edge_df.values).t().contiguous()

    # Timestamp based split:
    # train_mask: 1 - 34 time_step, test_mask: 35-49 time_step
    time_step = torch.from_numpy(feat_df['time_step'].values)
    train_mask = (time_step < 30) & (y != 2)
    val_mask = (time_step >= 30) & (time_step < 40) & (y != 2) 
    test_mask = (time_step >= 40) & (y != 2)

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                test_mask=test_mask)
    
    return(data)