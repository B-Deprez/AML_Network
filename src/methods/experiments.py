import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.methods.utils.functionsNetworkX import *
from src.methods.utils.functionsNetworKit import *
from src.methods.utils.functionsTorch import *
from src.methods.utils.unsupervised  import *
from src.methods.utils.GNN import *
from utils.Network import *

from src.methods.utils.decoder import *

from sklearn.metrics import average_precision_score

import optuna

def intrinsic_features_unsupervised(
        ntw, train_mask, test_mask,
        n_estimators, max_features, bootstrap
):
    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)
    # Combine train and test to train isolation forest and do hyper-parameter tuning
    X_train = torch.cat((X_train, X_test), 0)
    y_test = torch.cat((y_train, y_test), 0)
    # The data is given as tensors. First convert back to numpy darray
    X_train = X_train.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    # Train the isolation forest
    y_pred = isolation_forest(X_train, n_estimators, max_features, bootstrap)
    ap_score = average_precision_score(y_test, y_pred)
    return(ap_score)

def intrinsic_features_supervised(
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
    y_pred = y_pred.softmax(dim=1)
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
        hidden_dim_decoder: int = 5,
        ntw_name: str = None,
        supervised: bool = True
        ):
    
    print("intrinsic and summary: ")
    X = ntw.get_features(full=True)
    
    print("networkx: ")
    ntw_nx = ntw.get_network_nx()
    features_nx_df = local_features_nx(ntw_nx, alpha_pr, alpha_ppr, fraud_dict_train=fraud_dict_train, ntw_name=ntw_name)

    ## Train NetworkKit
    print("networkit: ")
    ntw_nk = ntw.get_network_nk()
    features_nk_df = features_nk(ntw_nk, ntw_name=ntw_name)

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

    if supervised:
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
        y_pred = y_pred.softmax(dim=1)
    else:
        pass

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
        n_epochs_decoder,
        supervised: bool = True
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
    # For ease of use, move both tensors to the same device (cpu will always work)
    x = x.detach().to('cpu')
    x_intrinsic = ntw_torch.x.detach().to('cpu')
    x = torch.cat((x, x_intrinsic), 1) # Concatenate node2vec and intrinsic features

    x_train = x[train_mask].to(device_decoder).squeeze()
    x_test = x[test_mask].to(device_decoder).squeeze()

    y_train = ntw_torch.y[train_mask].to(device_decoder).squeeze()
    y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

    if supervised:
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
        y_pred = y_pred.softmax(dim=1)
    else:
        pass
    
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
    y = y.softmax(dim=1)
    ap_score = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
    return(ap_score)