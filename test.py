from train import *
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import networkx as nx
import networkit as nk
import numpy as np

from models.functionsNetworkX import *
from models.functionsNetworKit import *
from models.functionsTorch import *
from models.GNN import *
from models.LINE import *
from utils.Network import *
from utils.DatasetConstruction import *

from models.decoder import *

from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import random

def positional_features_calc(
        ntw,
        alpha_pr: float,
        alpha_ppr: float=None,
        fraud_dict_train: dict=None,
        fraud_dict_test: dict=None
):
    print("intrinsic and summary: ")
    X = ntw.get_features(full=True)

    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train)

    ## Train NetworkKit
    print("networkit: ")
    ntw_nk = ntw.get_network_nk()
    features_nk_df = features_nk(ntw_nk)

    ## Concatenate features
    features_df = pd.concat([X, features_nx_df, features_nk_df], axis=1)
    features_df["fraud"] = [fraud_dict_test[x] for x in features_df.index]
    return features_df

def train_model_shallow(x_train, y_train , n_epochs_decoder, lr,
                        n_layers_decoder=2, hidden_dim_decoder=5, device_decoder="cpu"):
    decoder = Decoder_deep_norm(x_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    return(decoder)

def train_model_deep(data, model, train_mask, n_epochs, lr, batch_size, loader = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for _ in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model, train_mask=train_mask, batch_size=batch_size, lr=lr, loader=loader)
    
def stratified_sampling(x_test, y_test):
    n_samples = x_test.shape[0]
    x_new, y_new = resample(x_test, y_test, n_samples=n_samples, stratify=y_test)
    return(x_new, y_new)

def evaluate_model_shallow(model, x_test, y_test, percentile_q = 99, n_samples=1000, device = "cpu"):
    AUC_list = []
    AP_list = []
    precision_list = []
    recall_list = []
    F1_list = []

    model.eval()

    for _ in range(n_samples):
        x_new, y_new = stratified_sampling(x_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())
        x_new = torch.from_numpy(x_new).to(device)
        y_new = torch.from_numpy(y_new).to(device)
        y_pred = model(x_new)

        AUC = roc_auc_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

        cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
        y_pred = (y_pred[:,1] >= cutoff)*1
        precision = precision_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        recall = recall_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
        F1 = f1_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

        AUC_list.append(AUC)
        AP_list.append(AP)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(F1)

    return(AUC_list, AP_list, precision_list, recall_list, F1_list)

def resample_testmask(test_mask, p=0.5):
    sample_size = int(np.floor(test_mask.sum()*p))
    # Get indices where value is True
    true_indices = [i for i, val in enumerate(test_mask) if val]

    # Randomly select a subset of these indices
    sampled_indices = random.sample(true_indices, min(sample_size, len(true_indices)))

    # Create new list with False at all indices except the sampled ones
    output_list = [i in sampled_indices for i in range(len(test_mask))]

    return output_list

import torch

def subsample_true_values_tensor(test_mask, p=0.5):
    # Get indices where value is True
    sample_size = int(np.floor(test_mask.sum()*p))
    true_indices = torch.where(test_mask)[0]

    # Randomly select a subset of these indices
    sampled_indices = true_indices[torch.randperm(true_indices.size(0))[:min(sample_size, true_indices.size(0))]]

    # Create new tensor with False at all indices except the sampled ones
    output_tensor = torch.zeros_like(test_mask, dtype=torch.bool)
    output_tensor[sampled_indices] = True

    return output_tensor

def evaluate_model_deep(model, test_mask, percentile_q = 99, n_samples=1000, device = "cpu", loader = None):
    AUC_list = []
    AP_list = []
    precision_list = []
    recall_list = []
    F1_list = []

    model.eval()

    for _ in range(n_samples):
        test_mask_new = resample_testmask(test_mask)
        if loader is None:
            model.eval()
            out, h = model(ntw_torch.x, ntw_torch.edge_index.to(device))
            y_hat = out[test_mask_new].to(device)
            y = ntw_torch.y[test_mask_new].to(device)
            
        else:
            batch = next(iter(loader))
            batch = batch.to(device, 'edge_index')
            out, h = model(batch.x, batch.edge_index)
            y_hat = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
        
        AUC = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])

        cutoff = np.percentile(y_hat.cpu().detach().numpy()[:,1], percentile_q)
        y_hat = (y_hat[:,1] >= cutoff)*1
        precision = precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        recall = recall_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
        F1 = f1_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())

        AUC_list.append(AUC)
        AP_list.append(AP)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(F1)

    return(AUC_list, AP_list, precision_list, recall_list, F1_list)

def save_results(AUC_list, AP_list, precision_list, recall_list, F1_list, model_name):
    dict = {'AUC': AUC_list, 'AP': AP_list, 'precision': precision_list, 'recall': recall_list, 'F1': F1_list}
    df = pd.DataFrame(dict)
    df.to_csv('misc/'+model_name+'.csv')

if __name__ == "__main__":
    ntw = load_elliptic()
    #ntw = load_cora()
    train_mask, val_mask, test_mask = ntw.get_masks()
    train_mask = torch.logical_or(train_mask, val_mask).detach()

    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    ### Intrinsic features ###
    print("Intrinsic features")
    with open("misc/intrinsic_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)

    model_trained = train_model_shallow(X_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"], n_layers_decoder=param_dict["n_layers_decoder"], hidden_dim_decoder=param_dict["hidden_dim_decoder"], device_decoder=device_decoder)

    AUC_list_intr, AP_list_intr, precision_list_intr, recall_list_intr, F1_list_intr = evaluate_model_shallow(model_trained, X_test, y_test, device=device_decoder)
    save_results(AUC_list_intr, AP_list_intr, precision_list_intr, recall_list_intr, F1_list_intr, "intrinsic")

    ### Positional features ###
    x_intrinsic = ntw.get_features_torch()

    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}

    print("Positional features")
    with open("misc/positional_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    features_df = positional_features_calc(
        ntw,
        alpha_pr = param_dict["alpha_pr"],
        alpha_ppr=None,
        fraud_dict_train=None,
        fraud_dict_test=fraud_dict
    )

    features_df_train = features_df[train_mask.numpy()]

    x_train = torch.tensor(features_df_train.drop(["PSP","fraud"], axis=1).values, dtype=torch.float32).to(device_decoder)
    y_train = torch.tensor(features_df_train["fraud"].values, dtype=torch.long).to(device_decoder)

    features_df_test = features_df[test_mask.numpy()]

    x_test = torch.tensor(features_df_test.drop(["PSP","fraud"], axis=1).values, dtype=torch.float32).to(device_decoder)
    y_test = torch.tensor(features_df_test["fraud"].values, dtype=torch.long).to(device_decoder)

    model_trained = train_model_shallow(
        x_train,
        y_train,
        param_dict["n_epochs_decoder"],
        param_dict["lr"],
        n_layers_decoder=param_dict["n_layers_decoder"],
        hidden_dim_decoder=param_dict["hidden_dim_decoder"],
        device_decoder=device_decoder
    )
    AUC_list_pos, AP_list_pos, precision_list_pos, recall_list_pos, F1_list_pos = evaluate_model_shallow(model_trained, x_test, y_test, device=device_decoder)
    save_results(AUC_list_pos, AP_list_pos, precision_list_pos, recall_list_pos, F1_list_pos, "positional")
    
    #### Troch models ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch()
    ntw_torch.x = ntw_torch.x[:,1:]
    edge_index = ntw_torch.edge_index
    num_features = ntw_torch.num_features
    num_classes = 3
    output_dim = 2
    batch_size=128
    
    ## Deepwalk
    print("Deepwalk")
    with open("misc/deepwalk_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    model_deepwalk = node2vec_representation(
        ntw_torch, 
        train_mask,
        test_mask, 
        embedding_dim=param_dict["embedding_dim"],
        walk_length= param_dict["walk_length"],
        context_size= param_dict["context_size"],
        walks_per_node=param_dict["walks_per_node"],
        num_negative_samples=param_dict["num_negative_samples"],
        p=1, 
        q=1,
        lr=param_dict["lr"], 
        n_epochs=param_dict["n_epochs"]
    )

    x = model_deepwalk()
    x = x.detach()
    x = torch.cat((x, x_intrinsic), 1)
    x_train = x[train_mask].to(device_decoder)
    x_test = x[test_mask].to(device_decoder).squeeze()
    y_train = ntw_torch.y[train_mask].to(device_decoder)
    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

    model_trained = train_model_shallow(x_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"], device_decoder=device_decoder)
    AUC_list_dw, AP_list_dw, precision_list_dw, recall_list_dw, F1_list_dw = evaluate_model_shallow(model_trained, x_test, y_test, device=device_decoder)
    save_results(AUC_list_dw, AP_list_dw, precision_list_dw, recall_list_dw, F1_list_dw, "deepwalk")

    ## node2vec
    print("Node2vec")
    with open("misc/node2vec_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    model_node2vec = node2vec_representation(
        ntw_torch, 
        train_mask,
        test_mask, 
        embedding_dim=param_dict["embedding_dim"],
        walk_length= param_dict["walk_length"],
        context_size= param_dict["context_size"],
        walks_per_node=param_dict["walks_per_node"],
        num_negative_samples=param_dict["num_negative_samples"],
        p=param_dict["p"], 
        q=param_dict["q"],
        lr=param_dict["lr"], 
        n_epochs=param_dict["n_epochs"] 
    )

    x = model_node2vec()
    x = x.detach()
    x = torch.cat((x, x_intrinsic), 1)
    x_train = x[train_mask].to(device_decoder)
    x_test = x[test_mask].to(device_decoder).squeeze()
    y_train = ntw_torch.y[train_mask].to(device_decoder)
    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

    model_trained = train_model_shallow(x_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"], device_decoder=device_decoder)
    AUC_list_n2v, AP_list_n2v, precision_list_n2v, recall_list_n2v, F1_list_n2v = evaluate_model_shallow(model_trained, x_test, y_test, device=device_decoder)
    save_results(AUC_list_n2v, AP_list_n2v, precision_list_n2v, recall_list_n2v, F1_list_n2v, "node2vec")

    ## line 

    ## GCN
    print("GCN")
    with open("misc/gcn_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    n_epochs = param_dict["n_epochs"]
    lr = param_dict["lr"]

    model_GCN = GCN(
        edge_index, 
        num_features,
        hidden_dim=param_dict["hidden_dim"],
        embedding_dim=param_dict["embedding_dim"],
        output_dim=output_dim,
        n_layers=param_dict["n_layers"],
        dropout_rate=param_dict["dropout_rate"]
    ).to(device)

    train_model_deep(ntw_torch, model_GCN, train_mask, n_epochs, lr, batch_size, loader = None)
    AUC_list_gcn, AP_list_gcn, precision_list_gcn, recall_list_gcn, F1_list_gcn = evaluate_model_deep(model_GCN, test_mask, n_samples=1000, device = device)
    save_results(AUC_list_gcn, AP_list_gcn, precision_list_gcn, recall_list_gcn, F1_list_gcn, "gcn")

    #GraphSAGE
    print("GraphSAGE")
    with open("misc/sage_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    n_epochs = param_dict["n_epochs"]
    lr = param_dict["lr"]
    num_neighbors = param_dict["num_neighbors"]
    n_layers = param_dict["n_layers"]

    model_sage = GraphSAGE(
        edge_index, 
        num_features,
        hidden_dim=param_dict["hidden_dim"],
        embedding_dim=param_dict["embedding_dim"],
        output_dim=output_dim,
        n_layers=n_layers,
        dropout_rate=param_dict["dropout_rate"],
        sage_aggr = param_dict["sage_aggr"]
    ).to(device)

    train_loader = NeighborLoader(
        ntw_torch, 
        num_neighbors=[num_neighbors]*n_layers, 
        input_nodes=train_mask,
        batch_size = batch_size, 
        shuffle=True,
        num_workers=0
    )

    test_loader = NeighborLoader(
        ntw_torch,
        num_neighbors=[num_neighbors]*n_layers,
        input_nodes=test_mask,
        batch_size = int(test_mask.sum()),
        shuffle=False,
        num_workers=0
    )

    train_model_deep(ntw_torch, model_sage, train_mask, n_epochs, lr, batch_size, loader = train_loader)
    AUC_list_sage, AP_list_sage, precision_list_sage, recall_list_sage, F1_list_sage = evaluate_model_deep(model_sage, test_mask, n_samples=1000, device = device, loader=test_loader)
    save_results(AUC_list_sage, AP_list_sage, precision_list_sage, recall_list_sage, F1_list_sage, "sage")

    # GAT
    print("GAT")
    with open("misc/gat_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    n_epochs = param_dict["n_epochs"]
    lr = param_dict["lr"]

    model_gat = GAT(
        num_features,
        hidden_dim=param_dict["hidden_dim"],
        embedding_dim=param_dict["embedding_dim"],
        output_dim=output_dim,
        n_layers=param_dict["n_layers"],
        heads=param_dict["heads"],
        dropout_rate=param_dict["dropout_rate"]
    ).to(device)

    train_model_deep(ntw_torch, model_gat, train_mask, n_epochs, lr, batch_size, loader = None)
    AUC_list_gat, AP_list_gat, precision_list_gat, recall_list_gat, F1_list_gat = evaluate_model_deep(model_gat, test_mask, n_samples=1000, device = device)
    save_results(AUC_list_gat, AP_list_gat, precision_list_gat, recall_list_gat, F1_list_gat, "gat")

    # GIN
    print("GIN")
    with open("misc/gin_params.txt", "r") as f:
        params = f.readlines()
    string_dict = params[0].strip()
    param_dict = eval(string_dict)

    n_epochs = param_dict["n_epochs"]
    lr = param_dict["lr"]

    model_gin = GIN(
        num_features,
        hidden_dim=param_dict["hidden_dim"],
        embedding_dim=param_dict["embedding_dim"],
        output_dim=output_dim,
        n_layers=param_dict["n_layers"],
        dropout_rate=param_dict["dropout_rate"]
    ).to(device)

    train_model_deep(ntw_torch, model_gin, train_mask, n_epochs, lr, batch_size, loader = None)
    AUC_list_gin, AP_list_gin, precision_list_gin, recall_list_gin, F1_list_gin = evaluate_model_deep(model_gin, test_mask, n_samples=1000, device = device)
    save_results(AUC_list_gin, AP_list_gin, precision_list_gin, recall_list_gin, F1_list_gin, "gin")