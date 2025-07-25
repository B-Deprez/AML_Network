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
    use_intrinsic = True
    n_samples = 10
    intrinsic_str = "_intrinsic" if use_intrinsic else "_no_intrinsic"

    if use_intrinsic:
        to_test = [
            "intrinsic",
            "positional",
            "deepwalk",
            "node2vec",
            "gcn",
            "sage",
            "gat",
            "gin"
        ]
    else:
        to_test = [
            "positional",
            "deepwalk",
            "node2vec",
            "gcn",
            "sage",
            "gat",
            "gin"
        ]

    ### Load Dataset ###
    ntw_name = "cora"

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

    # Calculate prevalence of fraud in the training set
    X_train, y_train, X_test, y_test = ntw.get_train_test_split_intrinsic(train_mask, test_mask, device=device_decoder)

    percentage_labels = torch.mean(y_train.float()).item()
    percentile_q_list.append((1-percentage_labels)*100)

    if 'intrinsic' in to_test:
        ### Intrinsic features ###
        print("Intrinsic features")
        with open("res/intrinsic_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        AUC_list_intr = []
        AP_list_intr = []
        precision_dict_intr = dict()
        recall_dict_intr = dict()
        F1_dict_intr = dict()
        for percentile_q in percentile_q_list:
            precision_dict_intr[percentile_q] = []
            recall_dict_intr[percentile_q] = []
            F1_dict_intr[percentile_q] = []

        for _ in tqdm(range(n_samples)):
            X_new, y_new = stratified_sampling(X_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
            X_new = torch.from_numpy(X_new).to(device_decoder)
            y_new = torch.from_numpy(y_new).to(device_decoder)

            model_trained = train_model_shallow(X_new, y_new, param_dict["n_epochs_decoder"], param_dict["lr"], n_layers_decoder=param_dict["n_layers_decoder"], hidden_dim_decoder=param_dict["hidden_dim_decoder"], device_decoder=device_decoder)
            
            #Threshold independent metrics
            AUC, AP = evaluate_model_shallow_AUC(model_trained, X_test, y_test, device_decoder)
            AUC_list_intr.append(AUC)
            AP_list_intr.append(AP)

            #Threshold dependent metrics
            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_shallow_PRF(model_trained, X_test, y_test, percentile_q=percentile_q, device=device_decoder)
                precision_dict_intr[percentile_q].append(precision)
                recall_dict_intr[percentile_q].append(recall)
                F1_dict_intr[percentile_q].append(F1)

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
            ntw_name=ntw_name+"_test",
            use_intrinsic=use_intrinsic
        )

        features_df_train = features_df[train_mask.numpy()]

        x_train = torch.tensor(features_df_train.drop(["PSP","fraud"], axis=1).values, dtype=torch.float32).to(device_decoder)
        y_train = torch.tensor(features_df_train["fraud"].values, dtype=torch.long).to(device_decoder)

        features_df_test = features_df[test_mask.numpy()]

        x_test = torch.tensor(features_df_test.drop(["PSP","fraud"], axis=1).values, dtype=torch.float32).to(device_decoder)
        y_test = torch.tensor(features_df_test["fraud"].values, dtype=torch.long).to(device_decoder)

        AUC_list_pos = []
        AP_list_pos = []
        precision_dict_pos = dict()
        recall_dict_pos = dict()
        F1_dict_pos = dict()
        for percentile_q in percentile_q_list:
            precision_dict_pos[percentile_q] = []
            recall_dict_pos[percentile_q] = []
            F1_dict_pos[percentile_q] = []

        for _ in tqdm(range(n_samples)):
            X_new, y_new = stratified_sampling(x_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
            X_new = torch.from_numpy(X_new).to(device_decoder)
            y_new = torch.from_numpy(y_new).to(device_decoder)
            model_trained = train_model_shallow(
                X_new,
                y_new,
                param_dict["n_epochs_decoder"],
                param_dict["lr"],
                n_layers_decoder=param_dict["n_layers_decoder"],
                hidden_dim_decoder=param_dict["hidden_dim_decoder"],
                device_decoder=device_decoder
            )

            # Threshold independent metrics
            AUC, AP = evaluate_model_shallow_AUC(model_trained, x_test, y_test, device_decoder)
            AUC_list_pos.append(AUC)
            AP_list_pos.append(AP)

            # Threshold dependent metrics
            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_shallow_PRF(model_trained, x_test, y_test, percentile_q=percentile_q, device=device_decoder)
                precision_dict_pos[percentile_q].append(precision)
                recall_dict_pos[percentile_q].append(recall)
                F1_dict_pos[percentile_q].append(F1)

        save_results_TI(AUC_list_pos, AP_list_pos, ntw_name+"_positional"+intrinsic_str)
        save_results_TD(precision_dict_pos, recall_dict_pos, F1_dict_pos, ntw_name+"_positional"+intrinsic_str)

    #### Troch models ####
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch().to(device)
    ntw_torch.x = ntw_torch.x[:,1:]
    edge_index = ntw_torch.edge_index
    if use_intrinsic:
        num_features = ntw_torch.num_features
    else:
        num_features = 1
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
        if use_intrinsic:
            x_intrinsic = x_intrinsic.to('cpu')
            x = torch.cat((x, x_intrinsic), 1)
        x_train = x[train_mask].to(device_decoder)
        x_test = x[test_mask].to(device_decoder).squeeze()
        y_train = ntw_torch.y[train_mask].to(device_decoder)
        y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

        AUC_list_dw = []
        AP_list_dw = []
        precision_dict_dw = dict()
        recall_dict_dw = dict()
        F1_dict_dw = dict()
        for percentile_q in percentile_q_list:
            precision_dict_dw[percentile_q] = []
            recall_dict_dw[percentile_q] = []
            F1_dict_dw[percentile_q] = []

        for _ in tqdm(range(n_samples)):
            X_new, y_new = stratified_sampling(x_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
            X_new = torch.from_numpy(X_new).to(device_decoder)
            y_new = torch.from_numpy(y_new).to(device_decoder)
            model_trained = train_model_shallow(X_new, y_new, param_dict["n_epochs_decoder"], param_dict["lr"], device_decoder=device_decoder)

            AUC_dw, AP_dw = evaluate_model_shallow_AUC(model_trained, x_test, y_test, device=device_decoder)
            AUC_list_dw.append(AUC_dw)
            AP_list_dw.append(AP_dw)

            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_shallow_PRF(model_trained, x_test, y_test, percentile_q=percentile_q, device=device_decoder)
                precision_dict_dw[percentile_q].append(precision)
                recall_dict_dw[percentile_q].append(recall)
                F1_dict_dw[percentile_q].append(F1)


        save_results_TI(AUC_list_dw, AP_list_dw, ntw_name+"_deepwalk"+intrinsic_str)
        save_results_TD(precision_dict_dw, recall_dict_dw, F1_dict_dw, ntw_name+"_deepwalk"+intrinsic_str)

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
        if use_intrinsic:
            x_intrinsic = x_intrinsic.to('cpu')
            x = torch.cat((x, x_intrinsic), 1)
        x_train = x[train_mask].to(device_decoder)
        x_test = x[test_mask].to(device_decoder).squeeze()
        y_train = ntw_torch.y[train_mask].to(device_decoder)
        y_test = ntw_torch.y[test_mask].to(device_decoder).squeeze()

        AUC_list_n2v = []
        AP_list_n2v = []
        precision_dict_n2v = dict()
        recall_dict_n2v = dict()
        F1_dict_n2v = dict()

        for percentile_q in percentile_q_list:
            precision_dict_n2v[percentile_q] = []
            recall_dict_n2v[percentile_q] = []
            F1_dict_n2v[percentile_q] = []

        for _ in tqdm(range(n_samples)):
            X_new, y_new = stratified_sampling(x_train.cpu().detach().numpy(), y_train.cpu().detach().numpy())
            X_new = torch.from_numpy(X_new).to(device_decoder)
            y_new = torch.from_numpy(y_new).to(device_decoder)
            model_trained = train_model_shallow(X_new, y_new, param_dict["n_epochs_decoder"], param_dict["lr"], device_decoder=device_decoder)

            AUC_n2v, AP_n2v = evaluate_model_shallow_AUC(model_trained, x_test, y_test, device=device_decoder)
            AUC_list_n2v.append(AUC_n2v)
            AP_list_n2v.append(AP_n2v)

            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_shallow_PRF(model_trained, x_test, y_test, percentile_q=percentile_q, device=device_decoder)
                precision_dict_n2v[percentile_q].append(precision)
                recall_dict_n2v[percentile_q].append(recall)
                F1_dict_n2v[percentile_q].append(F1)

        save_results_TI(AUC_list_n2v, AP_list_n2v, ntw_name+"_node2vec"+intrinsic_str)
        save_results_TD(precision_dict_n2v, recall_dict_n2v, F1_dict_n2v, ntw_name+"_node2vec"+intrinsic_str)

    if 'gcn' in to_test:
        ## GCN
        print("GCN")
        with open("res/gcn_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        n_epochs = param_dict["n_epochs"]
        lr = param_dict["lr"]

        AUC_list_gcn = []
        AP_list_gcn = []

        precision_dict_gcn = dict()
        recall_dict_gcn = dict()
        F1_dict_gcn = dict()
        for percentile_q in percentile_q_list:
            precision_dict_gcn[percentile_q] = []
            recall_dict_gcn[percentile_q] = []
            F1_dict_gcn[percentile_q] = []

        for _ in tqdm(range(n_samples)):
            # Re-initialize the model for each sample
            model_GCN = GCN(
                edge_index, 
                num_features,
                hidden_dim=param_dict["hidden_dim"],
                embedding_dim=param_dict["embedding_dim"],
                output_dim=output_dim,
                n_layers=param_dict["n_layers"],
                dropout_rate=param_dict["dropout_rate"]
            ).to(device)

            train_mask_new = resample_mask(train_mask)
            train_model_deep(ntw_torch, model_GCN, train_mask_new, n_epochs, lr, batch_size, loader = None, use_intrinsic=use_intrinsic)

            AUC, PR = evaluate_model_deep_AUC(ntw_torch, model_GCN, test_mask, device = device, use_intrinsic=use_intrinsic)
            AUC_list_gcn.append(AUC)
            AP_list_gcn.append(PR)

            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_deep_PRF(ntw_torch, model_GCN, test_mask, percentile_q=percentile_q, device = device, use_intrinsic=use_intrinsic)
                precision_dict_gcn[percentile_q].append(precision)
                recall_dict_gcn[percentile_q].append(recall)
                F1_dict_gcn[percentile_q].append(F1)

        save_results_TI(AUC_list_gcn, AP_list_gcn, ntw_name+"_gcn"+intrinsic_str)
        save_results_TD(precision_dict_gcn, recall_dict_gcn, F1_dict_gcn, ntw_name+"_gcn"+intrinsic_str)

    if 'sage' in to_test:
        #GraphSAGE
        print("GraphSAGE")
        with open("res/sage_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        n_epochs = param_dict["n_epochs"]
        lr = param_dict["lr"]
        #num_neighbors = param_dict["num_neighbors"]
        n_layers = param_dict["n_layers"]

        AUC_list_sage = []
        AP_list_sage = []

        precision_dict_sage = dict()
        recall_dict_sage = dict()
        F1_dict_sage = dict()
        for percentile_q in percentile_q_list:
            precision_dict_sage[percentile_q] = []
            recall_dict_sage[percentile_q] = []
            F1_dict_sage[percentile_q] = []

        for _ in tqdm(range(n_samples)):
            # Re-initialize the model for each sample 
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

            train_mask_new = resample_mask(train_mask)

            #train_loader = NeighborLoader(
            #    ntw_torch, 
            #    num_neighbors=[num_neighbors]*n_layers, 
            #    input_nodes=train_mask,
            #    batch_size = batch_size, 
            #    shuffle=True,
            #    num_workers=0
            #)

            #test_loader = NeighborLoader(
            #    ntw_torch,
            #    num_neighbors=[num_neighbors]*n_layers,
            #    input_nodes=test_mask,
            #    batch_size = int(test_mask.sum()),
            #    shuffle=False,
            #    num_workers=0
            #)

            class_weights = torch.tensor([1.0, (1 / percentage_labels) - 1], device=device)  # Calculate class weights.
            #print(class_weights)
            optimizer = torch.optim.Adam(model_sage.parameters(), lr=lr, weight_decay=5e-4)  # Define optimizer.
            criterion = nn.CrossEntropyLoss(weight=class_weights)  # Define weighted loss function.

            if use_intrinsic:
                features =ntw_torch.x
            else:
                features = torch.ones((ntw_torch.x.shape[0], 1),dtype=torch.float32).to(device)

            def train_GNN():
                model_sage.train()
                optimizer.zero_grad()
                y_hat, h = model_sage(features, ntw_torch.edge_index.to(device))
                y = ntw_torch.y
                loss = criterion(y_hat[train_mask_new], y[train_mask_new])
                loss.backward()
                optimizer.step()

                return(loss)
            #print('n_epochs: ', n_epochs)
            for _ in range(n_epochs):
                loss_train = train_GNN()
                #print('Epoch: {:03d}, Loss: {:.4f}'.format(_, loss_train))

            #train_model_deep(ntw_torch, model_sage, train_mask, n_epochs, lr, batch_size)#, loader = train_loader)
            AUC, AP = evaluate_model_deep_AUC(ntw_torch, model_sage, test_mask, device = device, use_intrinsic=use_intrinsic)
            AUC_list_sage.append(AUC)
            AP_list_sage.append(AP)
            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_deep_PRF(ntw_torch, model_sage, test_mask, percentile_q=percentile_q, device = device, use_intrinsic=use_intrinsic)
                precision_dict_sage[percentile_q].append(precision)
                recall_dict_sage[percentile_q].append(recall)
                F1_dict_sage[percentile_q].append(F1)

        
        save_results_TI(AUC_list_sage, AP_list_sage, ntw_name+"_sage"+intrinsic_str)
        save_results_TD(precision_dict_sage, recall_dict_sage, F1_dict_sage, ntw_name+"_sage"+intrinsic_str)

    if 'gat' in to_test:
        # GAT
        print("GAT")
        with open("res/gat_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        n_epochs = param_dict["n_epochs"]
        lr = param_dict["lr"]

        AUC_list_gat = []
        AP_list_gat = []    

        precision_dict_gat = dict()
        recall_dict_gat = dict()
        F1_dict_gat = dict()
        for percentile_q in percentile_q_list:
            precision_dict_gat[percentile_q] = []
            recall_dict_gat[percentile_q] = []
            F1_dict_gat[percentile_q] = []
        
        for _ in tqdm(range(n_samples)):
            # Re-initialize the model for each sample
            model_gat = GAT(
                num_features,
                hidden_dim=param_dict["hidden_dim"],
                embedding_dim=param_dict["embedding_dim"],
                output_dim=output_dim,
                n_layers=param_dict["n_layers"],
                heads=param_dict["heads"],
                dropout_rate=param_dict["dropout_rate"]
                ).to(device)

            train_mask_new = resample_mask(train_mask)
            train_model_deep(ntw_torch, model_gat, train_mask_new, n_epochs, lr, batch_size, loader = None, use_intrinsic=use_intrinsic)
            AUC, AP = evaluate_model_deep_AUC(ntw_torch, model_gat, test_mask, device = device, use_intrinsic=use_intrinsic)
            AUC_list_gat.append(AUC)
            AP_list_gat.append(AP)
            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_deep_PRF(ntw_torch, model_gat, test_mask, percentile_q=percentile_q, device = device, use_intrinsic=use_intrinsic)
                precision_dict_gat[percentile_q].append(precision)
                recall_dict_gat[percentile_q].append(recall)
                F1_dict_gat[percentile_q].append(F1)
  
        save_results_TI(AUC_list_gat, AP_list_gat, ntw_name+"_gat"+intrinsic_str)
        save_results_TD(precision_dict_gat, recall_dict_gat, F1_dict_gat, ntw_name+"_gat"+intrinsic_str)

    if 'gin' in to_test:
        # GIN
        print("GIN")
        with open("res/gin_params_"+ntw_name+".txt", "r") as f:
            params = f.readlines()
        string_dict = params[0].strip()
        param_dict = eval(string_dict)

        n_epochs = param_dict["n_epochs"]
        lr = param_dict["lr"]

        AUC_list_gin = []
        AP_list_gin = []
        precision_dict_gin = dict()
        recall_dict_gin = dict()
        F1_dict_gin = dict()
        for percentile_q in percentile_q_list:
            precision_dict_gin[percentile_q] = []
            recall_dict_gin[percentile_q] = []
            F1_dict_gin[percentile_q] = []


        for _ in tqdm(range(n_samples)):
            model_gin = GIN(
                num_features,
                hidden_dim=param_dict["hidden_dim"],
                embedding_dim=param_dict["embedding_dim"],
                output_dim=output_dim,
                n_layers=param_dict["n_layers"],
                dropout_rate=param_dict["dropout_rate"]
            ).to(device)
            train_mask_new = resample_mask(train_mask)
            train_model_deep(ntw_torch, model_gin, train_mask_new, n_epochs, lr, batch_size, loader = None, use_intrinsic=use_intrinsic)
            AUC, AP = evaluate_model_deep_AUC(ntw_torch, model_gin, test_mask, device = device, use_intrinsic=use_intrinsic)
            AUC_list_gin.append(AUC)
            AP_list_gin.append(AP)
            for percentile_q in percentile_q_list:
                precision, recall, F1 = evaluate_model_deep_PRF(ntw_torch, model_gin, test_mask, percentile_q=percentile_q, device = device, use_intrinsic=use_intrinsic)
                precision_dict_gin[percentile_q].append(precision)
                recall_dict_gin[percentile_q].append(recall)
                F1_dict_gin[percentile_q].append(F1)    
        
        save_results_TI(AUC_list_gin, AP_list_gin, ntw_name+"_gin"+intrinsic_str)
        save_results_TD(precision_dict_gin, recall_dict_gin, F1_dict_gin, ntw_name+"_gin"+intrinsic_str)