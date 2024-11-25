import torch
import numpy as np

from src.methods.utils.functionsNetworkX import *
from src.methods.utils.functionsNetworKit import *
from src.methods.utils.functionsTorch import *
from src.methods.utils.isolation_forest  import *
from src.methods.utils.GNN import *
from utils.Network import *

from src.methods.utils.decoder import *

from sklearn.metrics import average_precision_score

def intrinsic_features(
        ntw, train_mask, test_mask,
        n_estimators, 
        max_samples,max_features, 
        bootstrap
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
    max_features_int = int(np.ceil(max_features*X_train.shape[1]/100))

    y_pred = isolation_forest(X_train, n_estimators, max_samples, max_features_int, bootstrap)
    ap_score = average_precision_score(y_test, y_pred)
    return(ap_score)

def positional_features(
        ntw, train_mask, test_mask,
        alpha_pr: float,
        alpha_ppr: float,
        n_estimators: int, 
        max_samples: float,
        max_features: int, # max_features% of features to consider 
        bootstrap: bool = False,
        fraud_dict_train: dict = None, 
        fraud_dict_test: dict = None,
        ntw_name: str = None
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

    # Combine train and test to train isolation forest and do hyper-parameter tuning
    x_train = torch.cat((x_train, x_test), 0)
    y_test = torch.cat((y_train, y_test), 0)
    # The data is given as tensors. First convert back to numpy darray
    x_train = x_train.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    # Train the isolation forest
    max_features_int = int(np.ceil(max_features*x_train.shape[1]/100))
    y_pred = isolation_forest(x_train, n_estimators, max_samples, max_features_int, bootstrap)
    ap_score = average_precision_score(y_test, y_pred)
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
        n_estimators, 
        max_samples,
        max_features, 
        bootstrap
):
    model_n2v = node2vec_representation_torch(
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

    # Combine train and test to train isolation forest and do hyper-parameter tuning
    X_train = torch.cat((x_train, x_test), 0)
    y_test = torch.cat((y_train, y_test), 0)
    # The data is given as tensors. First convert back to numpy darray
    X_train = X_train.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    # Train the isolation forest
    max_features_int = int(np.ceil(max_features*X_train.shape[1]/100))
    y_pred = isolation_forest(X_train, n_estimators, max_samples, max_features_int, bootstrap)
    ap_score = average_precision_score(y_test, y_pred)
    return(ap_score)
