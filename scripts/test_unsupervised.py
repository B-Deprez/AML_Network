import torch
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

from sklearn.ensemble import IsolationForest

from src.methods.experiments_unsupervised import *
from src.data.DatasetConstruction import *
from src.methods.evaluation import *

if __name__ == "__main__":
    use_intrinsic = False
    intrinsic_str = "_intrinsic" if use_intrinsic else "_no_intrinsic"

    if use_intrinsic:
        to_test =[
            "intrinsic",
            "positional",
            "deepwalk",
            "node2vec"
        ]
    else:
        to_test = [
            "positional",
            "deepwalk",
            "node2vec"
        ]

    ### Load Dataset ###
    ntw_name = "ibm"

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

    # Calculate prevalence of fraud
    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)

    percentage_labels = torch.mean(y_train.float()).item()
    percentile_q_list.append((1-percentage_labels)*100)

    X_test = torch.cat((X_train, X_test), 0)
    y_test = torch.cat((y_train, y_test), 0)

    if 'intrinsic' in to_test:
        ### Intrinsic features ###
        print("Intrinsic features")
        with open("res/intrinsic_params_"+ntw_name+"_unsupervised.txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        max_features = int(np.ceil(param_dict["max_features_dec%"]*X_train.shape[1]/10))

        model_intrinsic = IsolationForest(
            n_estimators=param_dict["n_estimators"],
            max_samples=param_dict["max_samples"],
            max_features=max_features,
            bootstrap=param_dict["bootstrap"]
        )

        AUC_list_intr, AP_list_intr, precision_dict_intr, recall_dict_intr, F1_dict_intr = evaluate_if(model_intrinsic, X_test, y_test, percentile_q_list=percentile_q_list)
        save_results_TI(AUC_list_intr, AP_list_intr, ntw_name+"_intrinsic_unsupervised")
        save_results_TD(precision_dict_intr, recall_dict_intr, F1_dict_intr, ntw_name+"_intrinsic_unsupervised")

    ### Positional features ###
    x_intrinsic = ntw.get_features_torch()

    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}

    if 'positional' in to_test:
        print("Positional features")
        with open("res/positional_params_"+ntw_name+"_unsupervised.txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        features_df = positional_features_calc(
            ntw,
            alpha_pr = param_dict["alpha_pr"],
            alpha_ppr=None,
            fraud_dict_train=None,
            fraud_dict_test=fraud_dict,
            ntw_name=ntw_name+"_test", 
            use_intrinsic=use_intrinsic
        )

        x = torch.tensor(features_df.drop(["PSP","fraud"], axis=1).values, dtype=torch.float32).to(device_decoder)
        y = torch.tensor(features_df["fraud"].values, dtype=torch.long).to(device_decoder)

        max_features = int(np.ceil(param_dict["max_features_dec%"]*x.shape[1]/10))

        model_pos = IsolationForest(
            n_estimators=param_dict["n_estimators"],
            max_samples=param_dict["max_samples"],
            max_features=max_features,
            bootstrap=param_dict["bootstrap"]
        )

        AUC_list_pos, AP_list_pos, precision_dict_pos, recall_dict_pos, F1_dict_pos = evaluate_if(model_pos, x, y, percentile_q_list=percentile_q_list)
        save_results_TI(AUC_list_pos, AP_list_pos, ntw_name+"_positional_unsupervised"+intrinsic_str)
        save_results_TD(precision_dict_pos, recall_dict_pos, F1_dict_pos, ntw_name+"_positional_unsupervised"+intrinsic_str)

    #### Troch models ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch().to(device)
    ntw_torch.x = ntw_torch.x[:,1:]

    ## Deepwalk
    if 'deepwalk' in to_test:
        print("Deepwalk")
        with open("res/deepwalk_params_"+ntw_name+"_unsupervised.txt", "r") as f:
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
        mask = torch.logical_or(train_mask, test_mask)
        if use_intrinsic:
            x_intrinsic = x_intrinsic.to('cpu')
            x = torch.cat((x, x_intrinsic), 1)[mask]
        else:
            x = x[mask]
        y = ntw_torch.y.clone().detach().to('cpu')[mask]
        max_features = int(np.ceil(param_dict["max_features_dec%"]*x.shape[1]/10))

        model_deepwalk = IsolationForest(
            n_estimators=param_dict["n_estimators"],
            max_samples=param_dict["max_samples"],
            max_features=max_features,
            bootstrap=param_dict["bootstrap"]
        )

        AUC_list_dw, AP_list_dw, precision_dict_dw, recall_dict_dw, F1_dict_dw = evaluate_if(model_deepwalk, x, y, percentile_q_list=percentile_q_list)
        save_results_TI(AUC_list_dw, AP_list_dw, ntw_name+"_deepwalk_unsupervised"+intrinsic_str)
        save_results_TD(precision_dict_dw, recall_dict_dw, F1_dict_dw, ntw_name+"_deepwalk_unsupervised"+intrinsic_str)

    if 'node2vec' in to_test:
        ## node2vec
        print("Node2vec")
        with open("res/node2vec_params_"+ntw_name+"_unsupervised.txt", "r") as f:
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
        mask = torch.logical_or(train_mask, test_mask)
        if use_intrinsic:
            x_intrinsic = x_intrinsic.to('cpu')
            x = torch.cat((x, x_intrinsic), 1)[mask]
        else:
            x = x[mask]
        y = ntw_torch.y.clone().detach().to('cpu')[mask]
        max_features = int(np.ceil(param_dict["max_features_dec%"]*x.shape[1]/10))

        model_node2vec = IsolationForest(
            n_estimators=param_dict["n_estimators"],
            max_samples=param_dict["max_samples"],
            max_features=max_features,
            bootstrap=param_dict["bootstrap"]
        )

        AUC_list_n2v, AP_list_n2v, precision_dict_n2v, recall_dict_n2v, F1_dict_n2v = evaluate_if(model_node2vec, x, y, percentile_q_list=percentile_q_list)
        save_results_TI(AUC_list_n2v, AP_list_n2v, ntw_name+"_node2vec_unsupervised"+intrinsic_str)
        save_results_TD(precision_dict_n2v, recall_dict_n2v, F1_dict_n2v, ntw_name+"_node2vec_unsupervised"+intrinsic_str)
