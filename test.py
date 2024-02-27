from train import *
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from models.functionsNetworkX import *
from models.functionsNetworKit import *
from models.functionsTorch import *
from models.GNN import *
from models.LINE import *
from utils.Network import *
from utils.DatasetConstruction import *

from models.decoder import *

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample
import random

def train_model_shallow(model, x_train, y_train , n_epochs_decoder, lr):
    model.eval()

    decoder = Decoder_deep_norm(x_train.shape[1], 2, 5).to(device_decoder)

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

def evaluate_model_shallow(model, x_test, y_test, n_samples=1000, device = "cpu"):
    AUC_list = []
    AP_list = []
    model.eval()

    for _ in range(n_samples):
        x_new, y_new = stratified_sampling(x_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())
        x_new = torch.from_numpy(x_new).to(device)
        y_new = torch.from_numpy(y_new).to(device)
        y_pred = model(x_new)
        AUC = roc_auc_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
        AUC_list.append(AUC)
        AP_list.append(AP)

    return(AUC_list, AP_list)

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

def evaluate_model_deep(model, test_mask, n_samples=1000, device = "cpu"):
    AUC_list = []
    AP_list = []
    model.eval()

    for _ in range(n_samples):
        test_mask_new = resample_testmask(test_mask)
        model.eval()
        out, h = model(ntw_torch.x, ntw_torch.edge_index.to(device))
        y_hat = out[test_mask_new].to(device)
        y = ntw_torch.y[test_mask_new].to(device)
        AUC = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
        AUC_list.append(AUC)
        AP_list.append(AP)

    return(AUC_list, AP_list)

def save_results(AUC_list, AP_list, model_name):
    dict = {'AUC': AUC_list, 'AP': AP_list}
    df = pd.DataFrame(dict)
    df.to_csv('misc/'+model_name+'.csv')

if __name__ == "__main__":
    ntw = load_elliptic()
    #ntw = load_cora()
    train_mask, val_mask, test_mask = ntw.get_masks()
    train_mask = torch.logical_or(train_mask, val_mask).detach()

    ### Positional features ###
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}
    
    #### Troch models ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

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
    x_train = x[train_mask].to(device_decoder)
    x_test = x[test_mask].to(device_decoder)
    y_train = ntw_torch.y[train_mask].to(device_decoder)
    y_test = ntw_torch.y[test_mask].to(device_decoder)

    model_trained = train_model_shallow(model_deepwalk, x_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"])
    AUC_list_dw, AP_list_dw = evaluate_model_shallow(model_trained, x_test, y_test, device=device_decoder)
    save_results(AUC_list_dw, AP_list_dw, "deepwalk")

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
    x_train = x[train_mask].to(device_decoder)
    x_test = x[test_mask].to(device_decoder)
    y_train = ntw_torch.y[train_mask].to(device_decoder)
    y_test = ntw_torch.y[test_mask].to(device_decoder)

    model_trained = train_model_shallow(model_node2vec, x_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"])
    AUC_list_n2v, AP_list_n2v = evaluate_model_shallow(model_trained, x_test, y_test, device=device_decoder)
    save_results(AUC_list_n2v, AP_list_n2v, "node2vec")

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
    AUC_list_gcn, AP_list_gcn = evaluate_model_deep(model_GCN, test_mask, n_samples=1000, device = device)
    save_results(AUC_list_gcn, AP_list_gcn, "gcn")

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

    loader = NeighborLoader(
        ntw_torch, 
        num_neighbors=[num_neighbors]*n_layers, 
        input_nodes=train_mask,
        batch_size = batch_size, 
        shuffle=True,
        num_workers=0
    )

    train_model_deep(ntw_torch, model_sage, train_mask, n_epochs, lr, batch_size, loader = None)
    AUC_list_sage, AP_list_sage = evaluate_model_deep(model_sage, test_mask, n_samples=1000, device = device)
    save_results(AUC_list_sage, AP_list_sage, "sage")

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
    AUC_list_gat, AP_list_gat = evaluate_model_deep(model_gat, test_mask, n_samples=1000, device = device)
    save_results(AUC_list_gat, AP_list_gat, "gat")

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
    AUC_list_gin, AP_list_gin = evaluate_model_deep(model_gin, test_mask, n_samples=1000, device = device)
    save_results(AUC_list_gin, AP_list_gin, "gin")