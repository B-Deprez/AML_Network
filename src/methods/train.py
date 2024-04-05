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

def intrinsic_features(
        ntw, train_mask, test_mask,
        n_layers_decoder, hidden_dim_decoder, lr, n_epochs_decoder
):
    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)

    decoder = Decoder_deep_norm(X_train.shape[1], n_layers_decoder, hidden_dim_decoder).to(device_decoder)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    decoder.eval()
    y_pred = decoder(X_test)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return(ap_score)

def positional_features(
        ntw, train_mask, test_mask,
        alpha_pr: float,
        alpha_ppr: float,
        n_epochs_decoder: list, 
        lr: float,
        fraud_dict_train: dict = None, 
        fraud_dict_test: dict = None,
        n_layers_decoder: int = 2,
        hidden_dim_decoder: int = 5
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
        #print(f"Epoch {epoch+1}: Loss: {loss.item()}")
    
    decoder.eval()
    y_pred = decoder(x_test)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return(ap_score)

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
    x = torch.cat((x, ntw_torch.x), 1) # Concatenate node2vec and intrinsic features

    x_train = x[train_mask].to(device_decoder).squeeze()
    x_test = x[test_mask].to(device_decoder).squeeze()

    y_train = ntw_torch.y[train_mask].to(device_decoder).squeeze()
    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

    decoder = Decoder_deep_norm(x_train.shape[1], 2, 10).to(device_decoder)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs_decoder):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    decoder.eval()
    y_pred = decoder(x_test)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return(ap_score)

def LINE_features(
        ntw_torch, train_mask, test_mask,
        embedding_dim,
        num_negative_samples,
        lr,
        n_epochs,
        n_epochs_decoder, 
        order = 2
):
    model_LINE = LINE_representation(
        ntw_torch,
        embedding_dim=embedding_dim,
        num_negative_samples=num_negative_samples,
        lr=lr,
        n_epochs=n_epochs, 
        order=order
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
    x = torch.cat((x, ntw_torch.x), 1) # Concatenate node2vec and intrinsic features

    x_train = x[train_mask].to(device_decoder).squeeze()
    x_test = x[test_mask].to(device_decoder).squeeze()

    y_train = ntw_torch.y[train_mask].to(device_decoder).squeeze()
    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

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
    decoder.eval()
    y_pred = decoder(x_test)
    ap_score = average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
    return(ap_score)

def GNN_features(
        ntw_torch: Data,
        model: nn.Module,
        batch_size: int, 
        lr: float,
        n_epochs: int, 
        train_loader: DataLoader =None,
        test_loader: DataLoader =None,
        train_mask: torch.Tensor = None,
        test_mask: torch.Tensor = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model, train_mask=train_mask, batch_size=batch_size, lr=lr, loader=train_loader)
        #loss_test = test_GNN(ntw_torch, model, test_mask=test_mask)
    
    model.eval()
    if test_loader is None:
        out, h = model(ntw_torch.x, ntw_torch.edge_index.to(device))
        if test_mask is None: # If no test_mask is provided, use all data
            y_hat = out
            y = ntw_torch.y
        else:
            y_hat = out[test_mask].squeeze()
            y = ntw_torch.y[test_mask].squeeze()
    else:
        for batch in test_loader:
            batch = batch.to(device, 'edge_index')
            out, h = model(batch.x, batch.edge_index)
            y_hat = out[:batch.batch_size]
            y = batch.y[:batch.batch_size]
    
    ap_score = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
    return(ap_score)

#### Optuna objective ####
def objective_intrinsic(trial):
    n_layers_decoder = trial.suggest_int('n_layers_decoder', 1, 3)
    hidden_dim_decoder = trial.suggest_int('hidden_dim_decoder', 5, 20)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs_decoder = trial.suggest_int('n_epochs_decoder', 5, 500)

    ap_loss = intrinsic_features(
        ntw, 
        train_mask, 
        val_mask,
        n_layers_decoder, 
        hidden_dim_decoder, 
        lr, 
        n_epochs_decoder
        )
    return(ap_loss)

def objective_positional(trial):
    alpha_pr = trial.suggest_float('alpha_pr', 0.1, 0.9)
    alpha_ppr = 0#trial.suggest_float('alpha_ppr', 0.1, 0.9)
    n_epochs_decoder = trial.suggest_int('n_epochs_decoder', 5, 100)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_layers_decoder = trial.suggest_int('n_layers_decoder', 1, 3)
    hidden_dim_decoder = trial.suggest_int('hidden_dim_decoder', 5, 20)

    ap_loss = positional_features(
        ntw, 
        train_mask, 
        val_mask,
        alpha_pr, 
        alpha_ppr,
        n_epochs_decoder,
        lr,
        fraud_dict_test=fraud_dict,
        n_layers_decoder=n_layers_decoder,
        hidden_dim_decoder=hidden_dim_decoder
        )
    return(ap_loss)

def objective_deepwalk(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 2, 64)
    walk_length = trial.suggest_int('walk_length', 3, 10)
    context_size = trial.suggest_int('context_size', 2, walk_length)
    walks_per_node = trial.suggest_int('walks_per_node', 1, 3)
    num_negative_samples = trial.suggest_int('num_negative_samples', 1, 5)
    p = 1
    q = 1
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 500)
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


def objective_node2vec(trial):
    embedding_dim = trial.suggest_int('embedding_dim', 2, 64)
    walk_length = trial.suggest_int('walk_length', 3, 10)
    context_size = trial.suggest_int('context_size', 2, walk_length)
    walks_per_node = trial.suggest_int('walks_per_node', 1, 3)
    num_negative_samples = trial.suggest_int('num_negative_samples', 1, 5)
    p = trial.suggest_float('p', 0.5, 2)
    q = trial.suggest_float('q', 0.5, 2)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 500)
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
    n_epochs = trial.suggest_int('n_epochs', 5, 500)
    n_epochs_decoder = trial.suggest_int('n_epochs_decoder', 5, 100)
    order = trial.suggest_int('order', 1, 2)

    ap_loss = LINE_features(
        ntw_torch, 
        train_mask, 
        val_mask,
        embedding_dim,
        num_negative_samples,
        lr,
        n_epochs,
        n_epochs_decoder, 
        order = order
        )
    return(ap_loss)

def objective_gcn(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 500)

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
    n_epochs = trial.suggest_int('n_epochs', 5, 500)

    sage_aggr = trial.suggest_categorical('sage_aggr', ["min","mean","max"])
    num_neighbors = trial.suggest_int("num_neighbors", 2, 5) # Keep number of neighbours low to have scaling benefits

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
    train_loader = NeighborLoader(
        ntw_torch, 
        num_neighbors = [num_neighbors]*n_layers,
        input_nodes = train_mask,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )

    val_loader = NeighborLoader(
        ntw_torch, 
        num_neighbors = [num_neighbors]*n_layers,
        input_nodes = val_mask,
        batch_size = int(val_mask.sum()),
        shuffle = False,
        num_workers = 0
    )

    ap_loss = GNN_features(ntw_torch, model_sage, batch_size, lr, n_epochs, train_loader=train_loader,test_loader=val_loader, train_mask=train_mask, test_mask=val_mask)
    return(ap_loss)

def objective_gat(trial):
    print('-------------------')
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256); print("hidden_dim: ", hidden_dim)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128); print("embedding_dim: ", embedding_dim)
    n_layers = trial.suggest_int('n_layers', 1, 3); print("n_layers: ", n_layers)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5); print("dropout_rate: ", dropout_rate)
    lr = trial.suggest_float('lr', 0.01, 0.1); print("lr: ", lr)
    n_epochs = trial.suggest_int('n_epochs', 5, 500); print("n_epochs: ", n_epochs)

    heads = trial.suggest_int("heads", 1, 5); print("heads: ", heads)

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
    n_epochs = trial.suggest_int('n_epochs', 5, 500)

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
    ### Load Dataset ###
    ntw = load_elliptic()
    #ntw = load_cora()
    train_mask, val_mask, test_mask = ntw.get_masks()

    ### Train intrinsic features ###
    print("intrinsic: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_intrinsic, n_trials=100)
    intrinsic_params = study.best_params   
    intrinsic_values = study.best_value
    with open("misc/intrinsic_params.txt", "w") as f:
        f.write(str(intrinsic_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(intrinsic_values))

    ### Train positional features ###
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}
    
    ## Train positional features
    print("positional: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_positional, n_trials=100)
    positional_params = study.best_params
    positional_values = study.best_value
    with open("misc/positional_params.txt", "w") as f:
        f.write(str(positional_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(positional_values))
    
    ### Train Torch ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch()
    ntw_torch.x = ntw_torch.x[:,1:94] # Remove time_step and summary features
    edge_index = ntw_torch.edge_index
    num_features = ntw_torch.num_features
    num_classes = 3
    output_dim = 2
    batch_size=128

    ## Train deepwalk
    print("deepwalk: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_deepwalk, n_trials=50)
    deepwalk_params = study.best_params
    deepwalk_values = study.best_value
    with open("misc/deepwalk_params.txt", "w") as f:
        f.write(str(deepwalk_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(deepwalk_values))

    ## Train node2vec 
    print("node2vec: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_node2vec, n_trials=50)
    node2vec_params = study.best_params
    node2vec_values = study.best_value
    with open("misc/node2vec_params.txt", "w") as f:
        f.write(str(node2vec_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(node2vec_values))

    ### Train LINE ###
    print("LINE: ")
    #study = optuna.create_study(direction='maximize')
    #study.optimize(objective_line, n_trials=50)
    #line_params = study.best_params
    #line_values = study.best_value
    #with open("misc/line_params.txt", "w") as f:
    #    f.write(str(line_params))
    #    f.write("\n")
    #    f.write("AUC-PRC: "+str(line_values))

    ### Train GNN ###
    ## GCN                
    print("GCN: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_gcn, n_trials=100)
    gcn_params = study.best_params
    gcn_values = study.best_value
    with open("misc/gcn_params.txt", "w") as f:
        f.write(str(gcn_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(gcn_values))
                                
    # GraphSAGE
    print("GraphSAGE: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_sage, n_trials=100)
    sage_params = study.best_params
    sage_values = study.best_value
    with open("misc/sage_params.txt", "w") as f:
        f.write(str(sage_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(sage_values))

    # GAT
    print("GAT: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_gat, n_trials=100)
    gat_params = study.best_params
    gat_values = study.best_value
    with open("misc/gat_params.txt", "w") as f:
        f.write(str(gat_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(gat_values))

    # GIN
    print("GIN: ")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective_gin, n_trials=100)
    gin_params = study.best_params
    gin_values = study.best_value
    with open("misc/gin_params.txt", "w") as f:
        f.write(str(gin_params))
        f.write("\n")
        f.write("AUC-PRC: "+str(gin_values))