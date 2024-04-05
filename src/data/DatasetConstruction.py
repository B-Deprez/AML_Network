import pandas as pd
import torch
from torch_geometric.data import Data
from utils.Network import network_AML

#### Elliptic dataset ####
def load_elliptic():
    raw_paths = [
        'data/elliptic_bitcoin/raw/elliptic_txs_features.csv',
        'data/elliptic_bitcoin/raw/elliptic_txs_edgelist.csv',
        'data/elliptic_bitcoin/raw/elliptic_txs_classes.csv',
                    ]
    feat_df = pd.read_csv(raw_paths[0], header=None)
    edge_df = pd.read_csv(raw_paths[1])
    class_df = pd.read_csv(raw_paths[2])

    columns = {0: 'txId', 1: 'time_step'}
    feat_df = feat_df.rename(columns=columns)

    x = torch.from_numpy(feat_df.loc[:, 'time_step':].values).to(torch.float)

    # There exists 3 different classes in the dataset:
    # 0=licit,  1=illicit, 2=unknown
    mapping = {'unknown': 2, '1': 1, '2': 0}
    class_df['class'] = class_df['class'].map(mapping)
    y = torch.from_numpy(class_df['class'].values)
    feat_df["class"] = y

    # Timestamp based split:
    time_step = torch.from_numpy(feat_df['time_step'].values)
    train_mask = (time_step < 30) & (y != 2)
    val_mask = (time_step >= 30) & (time_step < 40) & (y != 2) 
    test_mask = (time_step >= 40) & (y != 2)
    
    ntw = network_AML(feat_df, edge_df, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return(ntw)

#### Cora dataset ####
from torch_geometric.datasets import Planetoid
def load_cora(y = 0, p_train = 0.6, p_val = 0.2):
    path = './data/Planetoid'
    dataset = Planetoid(path, name='Cora')
    data = dataset[0]

    mask = torch.tensor([False]*data.x.shape[0])
    train_size = int(p_train*data.x.shape[0])
    val_size = int(p_val*data.x.shape[0])

    train_mask = mask.clone()
    train_mask[:train_size] = True
    val_mask = mask.clone()
    val_mask[train_size:train_size+val_size] = True
    test_mask = mask.clone()
    test_mask[train_size+val_size:] = True

    feat_df = pd.DataFrame(data.x.detach().numpy())
    feat_df.reset_index(inplace=True)
    feat_df = feat_df.rename(columns={"index": "txId"})
    edge_df = pd.DataFrame(data.edge_index.detach().numpy().T)
    edge_df.columns = ['txId1', 'txId2']
    feat_df["class"] = (data.y.detach().numpy()==1)*1

    ntw = network_AML(feat_df, edge_df, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return(ntw)