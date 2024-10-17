import pandas as pd
import torch
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
    
    ntw = network_AML(feat_df, edge_df, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, name='elliptic')

    return(ntw)

#### IBM dataset ####
from datetime import timedelta
import os

def preprocess_ibm():
    date_format = '%Y/%m/%d %H:%M'

    data_df = pd.read_csv('data/IBM/LI-Small_Trans.csv')
    data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'], format=date_format)
    data_df.sort_values('Timestamp', inplace=True)
    data_df = data_df[data_df['Account']!= data_df['Account.1']]
    data_df.reset_index(drop=True, inplace=True)
    data_df.reset_index(inplace=True)

    data_df_accounts = data_df[['index', 'Account', 'Account.1', 'Timestamp']]
    delta = 12*60 # 12 hours

    num_obs = len(data_df_accounts)
    pieces = 100

    source = []
    target = []

    for i in range(pieces):
        start = i*num_obs//pieces
        end = (i+1)*num_obs//pieces
        data_df_right = data_df_accounts[start:end]
        min_timestamp = data_df_right['Timestamp'].iloc[0]
        max_timestamp = data_df_right['Timestamp'].iloc[-1]

        data_df_left = data_df_accounts[(data_df_accounts['Timestamp']>=min_timestamp-timedelta(minutes=delta)) & (data_df_accounts['Timestamp']<=max_timestamp)]

        data_df_join = data_df_left.merge(data_df_right, left_on='Account.1', right_on='Account', suffixes=('_1', '_2'))

        for j in range(len(data_df_join)):
            row = data_df_join.iloc[j]
            delta_trans = row['Timestamp_2']-row['Timestamp_1']
            if (delta_trans.days*24*60+delta_trans.seconds/60 <= delta) & (delta_trans.days*24*60+delta_trans.seconds/60 >= 0):
                source.append(row['index_1'])
                target.append(row['index_2'])

    pd.DataFrame({'txId1': source, 'txId2': target}).to_csv('data/IBM/edges.csv', index=False)

def load_ibm():
    path = 'data/IBM'
    if not os.path.exists(path+'/edges.csv'):
        preprocess_ibm()
    df_edges = pd.read_csv(path+'/edges.csv')

    df_features = pd.read_csv(path+'/LI-Small_Trans.csv')
    df_features['Timestamp'] = pd.to_datetime(df_features['Timestamp'], format='%Y/%m/%d %H:%M')
    df_features.sort_values('Timestamp', inplace=True)
    df_features = df_features[df_features['Account']!= df_features['Account.1']]
    df_features.reset_index(drop=True, inplace=True)
    df_features.reset_index(inplace=True)

    df_features.columns = ['txId', 'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']
    df_features = df_features[['txId', 'Timestamp', 'Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Currency', 'Payment Format', 'class']]

    list_day = []
    list_hour = []
    list_minute = []
    for date in list(df_features['Timestamp']):
        list_day.append(date.day)
        list_hour.append(date.hour)
        list_minute.append(date.minute)
    df_features['Day'] = list_day
    df_features['Hour'] = list_hour
    df_features['Minute'] = list_minute

    df_features = df_features.drop(columns=['Timestamp'])
    df_features = pd.get_dummies(df_features, columns=['Receiving Currency', 'Payment Currency', 'Payment Format'])

    # Timestamp based split:
    mask = torch.tensor([False]*df_features.shape[0])
    train_size = int(0.6*df_features.shape[0])
    val_size = int(0.2*df_features.shape[0])

    train_mask = mask.clone()
    train_mask[:train_size] = True
    val_mask = mask.clone()
    val_mask[train_size:train_size+val_size] = True
    test_mask = mask.clone()
    test_mask[train_size+val_size:] = True

    ntw = network_AML(df_features, df_edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, name='ibm')

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

    ntw = network_AML(feat_df, edge_df, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, name='cora')

    return(ntw)