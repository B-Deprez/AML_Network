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

from sklearn.metrics import average_precision_score

import optuna

def positinal_features(
        ntw, train_mask, test_mask, fraud_dict,
        n_epochs_decoder_list: list, 
        lr: float,
        w_a,
        file: str = "misc/positional_results.txt"
        ):
    
    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, fraud_dict)

    ## Train NetworkKit
    print("networkit: ")
    ntw_nk = ntw.get_network_nk()
    features_nk_df = features_nk(ntw_nk)

    ## Concatenate features
    features_df = pd.concat([features_nx_df, features_nk_df], axis=1)
    features_based_on_labels = ["fraud_degree", "legit_degree", "fraud_triangle", "semifraud_triangle", "legit_triangle", "RNC_F_node", "RNC_NF_node"]
    features_df = features_df.drop(features_based_on_labels, axis=1)
    features_df["fraud"] = [fraud_dict[x] for x in features_df.index]

    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    features_df_train = features_df[train_mask.numpy()]

    x_train = torch.tensor(features_df_train.drop(["PSP","fraud"], axis=1).values, dtype=torch.float32).to(device_decoder)
    y_train = torch.tensor(features_df_train["fraud"].values, dtype=torch.long).to(device_decoder)

    features_df_test = features_df[test_mask.numpy()]

    x_test = torch.tensor(features_df_test.drop(["PSP","fraud"], axis=1).values, dtype=torch.float32).to(device_decoder)
    y_test = torch.tensor(features_df_test["fraud"].values, dtype=torch.long).to(device_decoder)

    decoder = Decoder_deep_norm(x_train.shape[1], 2, 5).to(device_decoder)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_epochs_decoder = max(n_epochs_decoder_list)
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        #print(f"Epoch {epoch+1}: Loss: {loss.item()}")
        if (epoch+1) in n_epochs_decoder_list:
            y_pred = decoder(x_test)
            ap_score = round(average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1]), 4)
            with open(file, w_a) as f:
                f.write(f"Epochs: {epoch+1},")
                f.write(f"Learning Rate: {lr},")
                f.write(f"Loss: {loss.item()},")
                f.write(f"AP Score: {ap_score} \n")
            w_a = "a"

def node2vec_features(
        ntw_torch, train_mask, test_mask,
        embedding_dim, 
        walk_length, 
        context_size,
        walks_per_node,
        num_negative_samples,
        p,
        q,
        lr,
        n_epochs,
        n_epochs_decoder
):
    model_n2v = node2vec_representation(
        ntw_torch,
        train_mask = train_mask,
        test_mask = test_mask,
        embedding_dim=embedding_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        num_negative_samples=num_negative_samples,
        p=p,
        q=q,
        lr=lr,
        n_epochs=n_epochs
        )
    
    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    model_n2v.eval()
    x = model_n2v()
    x = x.detach()

    x_train = x[train_mask].to(device_decoder)
    x_test = x[test_mask].to(device_decoder)

    y_train = ntw_torch.y[train_mask].to(device_decoder)
    y_test = ntw_torch.y[test_mask].to(device_decoder)

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
        
    y_pred = decoder(x_test)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return(ap_score)

def LINE_features(
        ntw_torch, train_mask, test_mask,
        embedding_dim,
        num_negative_samples,
        lr,
        n_epochs,
        n_epochs_decoder_list: list,
        w_a,
        file: str = "misc/LINE_results.txt"
):
    model_LINE = LINE_representation(
        ntw_torch,
        embedding_dim=embedding_dim,
        num_negative_samples=num_negative_samples,
        lr=lr,
        n_epochs=n_epochs
        )

    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    x = model_LINE()
    x = x.detach()

    x_train = x[train_mask].to(device_decoder)
    x_test = x[test_mask].to(device_decoder)

    y_train = ntw_torch.y[train_mask].to(device_decoder)
    y_test = ntw_torch.y[test_mask].to(device_decoder)

    decoder = Decoder_deep_norm(x_train.shape[1], 2, 5).to(device_decoder)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_epochs_decoder = max(n_epochs_decoder_list)
    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        #print(f"Epoch {epoch+1}: Loss: {loss.item()}")
        if (epoch+1) in n_epochs_decoder_list:
            y_pred = decoder(x_test)
            ap_score = round(average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1]), 4)
            with open(file, w_a) as f:
                f.write(f"Embedding Dim: {embedding_dim},")
                f.write(f"Negative Samples: {num_negative_samples},")
                f.write(f"Epochs: {n_epochs},")
                f.write(f"Epochs decoder: {epoch+1},")
                f.write(f"Learning Rate: {lr},")
                f.write(f"Loss: {loss.item()},")
                f.write(f"AP Score: {ap_score} \n")
            w_a = "a"

def GNN_features(
        ntw_torch: Data,
        model: nn.Module,
        batch_size: int, 
        lr: float,
        n_epochs: int, 
        loader: DataLoader =None,
        train_mask: torch.Tensor = None,
        test_mask: torch.Tensor = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model, train_mask=train_mask, batch_size=batch_size, lr=lr, loader=loader)
        #loss_test = test_GNN(ntw_torch, model, test_mask=test_mask)
    
    model.eval()
    out, h = model(ntw_torch.x, ntw_torch.edge_index.to(device))
    if test_mask is None: # If no test_mask is provided, use all data
        y_hat = out
        y = ntw_torch.y
    else:
        y_hat = out[test_mask]
        y = ntw_torch.y[test_mask]
    
    ap_score = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
    return(ap_score)

#### Optuna objective ####
def objective_node2vec(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 2, 64)
    walk_length = trial.suggest_int('walk_length', 10, 100)
    context_size = trial.suggest_int('context_size', 2, 10)
    walks_per_node = trial.suggest_int('walks_per_node', 1, 10)
    num_negative_samples = trial.suggest_int('num_negative_samples', 1, 10)
    p = trial.suggest_float('p', 0.5, 2)
    q = trial.suggest_float('q', 0.5, 2)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 100)
    n_epochs_decoder = trial.suggest_int('n_epochs_decoder', 5, 100)

    ap_loss = node2vec_features(
        ntw_torch, 
        train_mask, 
        val_mask,
        embedding_dim, 
        walk_length, 
        context_size,
        walks_per_node,
        num_negative_samples,
        p,
        q,
        lr,
        n_epochs,
        n_epochs_decoder
        )
    return(ap_loss)

def objective_line(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 2, 64)
    num_negative_samples = trial.suggest_int('num_negative_samples', 1, 10)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 100)
    n_epochs_decoder = trial.suggest_int('n_epochs_decoder', 5, 100)

    ap_loss = LINE_features(
        ntw_torch, 
        train_mask, 
        val_mask,
        embedding_dim,
        num_negative_samples,
        lr,
        n_epochs,
        n_epochs_decoder
        )
    return(ap_loss)

def objective_gcn(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 100)

    model_gcn = GCN(
        edge_index=edge_index, 
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate
        ).to(device)
    ap_loss = GNN_features(ntw_torch, model_gcn, batch_size, lr, n_epochs, train_mask=train_mask, test_mask=val_mask)
    return(ap_loss)

def objective_sage(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 100)

    sage_aggr = trial.suggest_categorical('sage_aggr', ["min","mean","max"])
    num_neighbors = trial.suggest_int("num_neighbors", 2, 16)

    model_sage = GraphSAGE(
        edge_index=edge_index, 
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate,
        sage_aggr=sage_aggr
    ).to(device)
    loader = NeighborLoader(
        ntw_torch, 
        num_neighbors = num_neighbors*n_layers,
        input_nodes = ntw_torch.train_mask,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )
    ap_loss = GNN_features(ntw_torch, model_sage, batch_size, lr, n_epochs, loader=loader, train_mask=train_mask, test_mask=val_mask)
    return(ap_loss)

def objective_gat(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 100)

    heads = trial.suggest_int("heads", 1, 8)

    model_gat = GAT(
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        heads=heads,
        dropout_rate=dropout_rate
    ).to(device)

    ap_loss = GNN_features(ntw_torch, model_gat, batch_size, lr, n_epochs, train_mask=train_mask, test_mask=val_mask)
    return(ap_loss)

def objective_gin(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 100)

    model_gin = GIN(
                    num_features=num_features,
                    hidden_dim=hidden_dim,
                    embedding_dim=embedding_dim,
                    output_dim=output_dim,
                    n_layers=n_layers,
                    dropout_rate=dropout_rate
                ).to(device)
    
    ap_loss = GNN_features(ntw_torch, model_gin, batch_size, lr, n_epochs, train_mask=train_mask, test_mask=val_mask)
    return(ap_loss)

if __name__ == "__main__":
    ### Load Elliptic Dataset ###
    #ntw = load_elliptic()
    ntw = load_cora()
    train_mask, val_mask, test_mask = ntw.get_masks()

    ### Train positional features ###
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}
    lr_list = [0.02]
    n_epochs_decoder = [10, 50, 100]
    
    w_a = "w" #string to indicate whether the file is being written or appended
    for lr in lr_list:
        positinal_features(ntw, train_mask, val_mask, fraud_dict, n_epochs_decoder, lr, w_a)
        w_a = "a"

    ### Train Torch ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch()
    ntw_torch.x = ntw_torch.x[:,1:]
    edge_index = ntw_torch.edge_index
    num_features = ntw_torch.num_features
    num_classes = 3
    output_dim = 2
    batch_size=128

    ## Train node2vec 
    print("node2vec: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_node2vec, n_trials=50)
    node2vec_params = study.best_params
    with open("misc/node2vec_params.txt", "w") as f:
        f.write(str(node2vec_params))

    ### Train LINE ###
    print("LINE: ")
    w_a = "w" #string to indicate whether the file is being written or appended
    for embedding_dim in embedding_dim_list:
        for num_negative_samples in num_negative_samples_list:
            for lr in lr_list:
                for n_epochs in n_epochs_list:
                    LINE_features(
                        ntw_torch, train_mask, val_mask,
                        embedding_dim, num_negative_samples, lr, n_epochs, n_epochs_decoder, w_a
                    )
                    w_a = "a"

    ### Train GNN ###
    ## GCN                
    print("GCN: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_gcn, n_trials=100)
    gcn_params = study.best_params
    with open("misc/gcn_params.txt", "w") as f:
        f.write(str(gcn_params))
                                
    # GraphSAGE
    print("GraphSAGE: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_sage, n_trials=100)
    sage_params = study.best_params
    with open("misc/sage_params.txt", "w") as f:
        f.write(str(sage_params))

    # GAT
    print("GAT: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_gat, n_trials=100)
    gat_params = study.best_params
    with open("misc/gat_params.txt", "w") as f:
        f.write(str(gat_params))

    # GIN
    print("GIN: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_gin, n_trials=100)
    gin_params = study.best_params
    with open("misc/gin_params.txt", "w") as f:
        f.write(str(gin_params))