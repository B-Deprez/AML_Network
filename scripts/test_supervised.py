import torch
import os
import sys

#os.chdir("/data/leuven/364/vsc36429/AML_Network")
#sys.path.append("/data/leuven/364/vsc36429/AML_Network")

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

from src.methods.experiments_supervised import *
from src.data.DatasetConstruction import *
from src.methods.evaluation import *

if __name__ == "__main__":
    to_test = [
        #"intrinsic",
        #"positional",
        #"deepwalk",
        #"node2vec",
        "gcn",
        #"sage",
        "gat",
        "gin"
    ]

    ### Load Dataset ###
    ntw_name = "elliptic"

    if ntw_name == "ibm":
        ntw = load_ibm()
    elif ntw_name == "elliptic":
        ntw = load_elliptic()
    elif ntw_name == "cora":
        ntw = load_cora()
    else:
        raise ValueError("Network not found")

    percentile_q_list = [90, 99, 99.9]

    train_mask, val_mask, test_mask = ntw.get_masks()
    train_mask = torch.logical_or(train_mask, val_mask).detach()

    device_decoder = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    if 'intrinsic' in to_test:
        ### Intrinsic features ###
        print("Intrinsic features")
        with open("res/intrinsic_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)

        percentage_labels = torch.mean(y_train.float()).item()
        percentile_q_list.append((1-percentage_labels)*100)

        model_trained = train_model_shallow(X_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"], n_layers_decoder=param_dict["n_layers_decoder"], hidden_dim_decoder=param_dict["hidden_dim_decoder"], device_decoder=device_decoder)

        AUC_list_intr, AP_list_intr, precision_dict_intr, recall_dict_intr, F1_dict_intr = evaluate_model_shallow(model_trained, X_test, y_test, percentile_q_list=percentile_q_list, device=device_decoder)
        save_results_TI(AUC_list_intr, AP_list_intr, ntw_name+"_intrinsic")
        save_results_TD(precision_dict_intr, recall_dict_intr, F1_dict_intr, ntw_name+"_intrinsic")

    ### Positional features ###
    x_intrinsic = ntw.get_features_torch()

    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}

    if 'positional' in to_test:
        print("Positional features")
        with open("res/positional_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        features_df = positional_features_calc(
            ntw,
            alpha_pr = param_dict["alpha_pr"],
            alpha_ppr=None,
            fraud_dict_train=None,
            fraud_dict_test=fraud_dict,
            ntw_name=ntw_name+"_test"
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
        AUC_list_pos, AP_list_pos, precision_dict_pos, recall_dict_pos, F1_dict_pos = evaluate_model_shallow(model_trained, x_test, y_test, percentile_q_list=percentile_q_list, device=device_decoder)
        save_results_TI(AUC_list_pos, AP_list_pos, ntw_name+"_positional")
        save_results_TD(precision_dict_pos, recall_dict_pos, F1_dict_pos, ntw_name+"_positional")

    #### Troch models ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch().to(device)
    ntw_torch.x = ntw_torch.x[:,1:]
    edge_index = ntw_torch.edge_index
    num_features = ntw_torch.num_features
    output_dim = 2
    batch_size=128

    ## Deepwalk
    if 'deepwalk' in to_test:
        print("Deepwalk")
        with open("res/deepwalk_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        model_deepwalk = node2vec_representation_torch(
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
        # Move x and x_intrinsic to cpu, so that they can be concatenated
        x = x.detach().to('cpu')
        x_intrinsic = x_intrinsic.to('cpu')
        x = torch.cat((x, x_intrinsic), 1)
        x_train = x[train_mask].to(device_decoder)
        x_test = x[test_mask].to(device_decoder).squeeze()
        y_train = ntw_torch.y[train_mask].to(device_decoder)
        y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

        model_trained = train_model_shallow(x_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"], device_decoder=device_decoder)
        AUC_list_dw, AP_list_dw, precision_dict_dw, recall_dict_dw, F1_dict_dw = evaluate_model_shallow(model_trained, x_test, y_test, percentile_q_list=percentile_q_list, device=device_decoder)
        save_results_TI(AUC_list_dw, AP_list_dw, ntw_name+"_deepwalk")
        save_results_TD(precision_dict_dw, recall_dict_dw, F1_dict_dw, ntw_name+"_deepwalk")

    if 'node2vec' in to_test:
        ## node2vec
        print("Node2vec")
        with open("res/node2vec_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        model_node2vec = node2vec_representation_torch(
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
        # Move x and x_intrinsic to cpu, so that they can be concatenated
        x = x.detach().to('cpu')
        x_intrinsic = x_intrinsic.to('cpu')
        x = torch.cat((x, x_intrinsic), 1)
        x_train = x[train_mask].to(device_decoder)
        x_test = x[test_mask].to(device_decoder).squeeze()
        y_train = ntw_torch.y[train_mask].to(device_decoder)
        y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

        model_trained = train_model_shallow(x_train, y_train, param_dict["n_epochs_decoder"], param_dict["lr"], device_decoder=device_decoder)
        AUC_list_n2v, AP_list_n2v, precision_dict_n2v, recall_dict_n2v, F1_dict_n2v = evaluate_model_shallow(model_trained, x_test, y_test, percentile_q_list=percentile_q_list, device=device_decoder)
        save_results_TI(AUC_list_n2v, AP_list_n2v, ntw_name+"_node2vec")
        save_results_TD(precision_dict_n2v, recall_dict_n2v, F1_dict_n2v, ntw_name+"_node2vec")

    if 'gcn' in to_test:
        ## GCN
        print("GCN")
        with open("res/gcn_params_"+ntw_name+".txt", "r") as f:
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
        AUC_list_gcn, AP_list_gcn, precision_dict_gcn, recall_dict_gcn, F1_dict_gcn = evaluate_model_deep(ntw_torch, model_GCN, test_mask, percentile_q_list=percentile_q_list, n_samples=1000, device = device)
        save_results_TI(AUC_list_gcn, AP_list_gcn, ntw_name+"_gcn")
        save_results_TD(precision_dict_gcn, recall_dict_gcn, F1_dict_gcn, ntw_name+"_gcn")

    if 'sage' in to_test:
        #GraphSAGE
        print("GraphSAGE")
        with open("res/sage_params_"+ntw_name+".txt", "r") as f:
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
        AUC_list_sage, AP_list_sage, precision_dict_sage, recall_dict_sage, F1_dict_sage = evaluate_model_deep(ntw_torch, model_sage, test_mask, percentile_q_list=percentile_q_list, n_samples=1000, device = device, loader=test_loader)
        save_results_TI(AUC_list_sage, AP_list_sage, ntw_name+"_sage")
        save_results_TD(precision_dict_sage, recall_dict_sage, F1_dict_sage, ntw_name+"_sage")

    if 'gat' in to_test:
        # GAT
        print("GAT")
        with open("res/gat_params_"+ntw_name+".txt", "r") as f:
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
        AUC_list_gat, AP_list_gat, precision_dict_gat, recall_dict_gat, F1_dict_gat = evaluate_model_deep(ntw_torch, model_gat, test_mask, percentile_q_list=percentile_q_list, n_samples=1000, device = device)
        save_results_TI(AUC_list_gat, AP_list_gat, ntw_name+"_gat")
        save_results_TD(precision_dict_gat, recall_dict_gat, F1_dict_gat, ntw_name+"_gat")

    if 'gin' in to_test:
        # GIN
        print("GIN")
        with open("res/gin_params_"+ntw_name+".txt", "r") as f:
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
        AUC_list_gin, AP_list_gin, precision_dict_gin, recall_dict_gin, F1_dict_gin = evaluate_model_deep(ntw_torch, model_gin, test_mask, percentile_q_list=percentile_q_list, n_samples=1000, device = device)
        save_results_TI(AUC_list_gin, AP_list_gin, ntw_name+"_gin")
        save_results_TD(precision_dict_gin, recall_dict_gin, F1_dict_gin, ntw_name+"_gin")