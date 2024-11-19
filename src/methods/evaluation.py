from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import random


import torch.nn as nn
import numpy as np

from src.methods.utils.functionsNetworkX import *
from src.methods.utils.functionsNetworKit import *
from src.methods.utils.functionsTorch import *
from src.methods.utils.GNN import *
from utils.Network import *
from src.data.DatasetConstruction import *

from src.methods.utils.decoder import *

def positional_features_calc(
        ntw,
        alpha_pr: float,
        alpha_ppr: float=None,
        fraud_dict_train: dict=None,
        fraud_dict_test: dict=None,
        ntw_name: str=None
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
    return features_df

def train_model_shallow(x_train, y_train , n_epochs_decoder, lr,
                        n_layers_decoder=2, hidden_dim_decoder=10, device_decoder="cpu"):
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
        loss_train = train_GNN(data, model, train_mask=train_mask, batch_size=batch_size, lr=lr, loader=loader)
    
def stratified_sampling(x_test, y_test):
    n_samples = x_test.shape[0]
    x_new, y_new = resample(x_test, y_test, n_samples=n_samples, stratify=y_test)
    return(x_new, y_new)

def evaluate_model_shallow(model, x_test, y_test, percentile_q_list = [99], n_samples=1000, device = "cpu"):
    AUC_list = []
    AP_list = []
    
    precision_dict = dict()
    recall_dict = dict()
    F1_dict = dict()
    for percentile_q in percentile_q_list:
        precision_dict[percentile_q] = []
        recall_dict[percentile_q] = []
        F1_dict[percentile_q] = []

    model.eval()

    for _ in range(n_samples):
        x_new, y_new = stratified_sampling(x_test.cpu().detach().numpy(), y_test.cpu().detach().numpy())
        x_new = torch.from_numpy(x_new).to(device)
        y_new = torch.from_numpy(y_new).to(device)
        y_pred = model(x_new)
        y_pred = y_pred.softmax(dim=1)

        AUC = roc_auc_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y_new.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1])

        AUC_list.append(AUC)
        AP_list.append(AP)

        for percentile_q in percentile_q_list:
            cutoff = np.percentile(y_pred.cpu().detach().numpy()[:,1], percentile_q)
            y_pred_hard = (y_pred[:,1] >= cutoff)*1
            precision = precision_score(y_new.cpu().detach().numpy(), y_pred_hard.cpu().detach().numpy())
            recall = recall_score(y_new.cpu().detach().numpy(), y_pred_hard.cpu().detach().numpy())
            F1 = f1_score(y_new.cpu().detach().numpy(), y_pred_hard.cpu().detach().numpy())

            precision_dict[percentile_q].append(precision)
            recall_dict[percentile_q].append(recall)
            F1_dict[percentile_q].append(F1)

    return(AUC_list, AP_list, precision_dict, recall_dict, F1_dict)

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

def evaluate_model_deep(data, model, test_mask, percentile_q_list = [99], n_samples=1000, device = "cpu", loader = None):
    AUC_list = []
    AP_list = []

    precision_dict = dict()
    recall_dict = dict()
    F1_dict = dict()
    for percentile_q in percentile_q_list:
        precision_dict[percentile_q] = []
        recall_dict[percentile_q] = []
        F1_dict[percentile_q] = []

    model.eval()

    for _ in range(n_samples):
        test_mask_new = resample_testmask(test_mask)
        if loader is None:
            model.eval()
            out, h = model(data.x, data.edge_index.to(device))
            y_hat = out[test_mask_new].to(device) # Prediction
            y = data.y[test_mask_new].to(device) # True value
            
        else:
            batch = next(iter(loader))
            batch = batch.to(device, 'edge_index')
            out, h = model(batch.x, batch.edge_index)
            y_hat = out[:batch.batch_size] # Prediction
            y = batch.y[:batch.batch_size] # True value
        y_hat  = y_hat.softmax(dim=1) # Get probability of fraud
        
        AUC = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])

        AUC_list.append(AUC)
        AP_list.append(AP)

        for percentile_q in percentile_q_list:
            cutoff = np.percentile(y_hat.cpu().detach().numpy()[:,1], percentile_q)
            y_hat_hard = (y_hat[:,1] >= cutoff)*1
            precision = precision_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
            recall = recall_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
            F1 = f1_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())

            precision_dict[percentile_q].append(precision)
            recall_dict[percentile_q].append(recall)
            F1_dict[percentile_q].append(F1)

    return(AUC_list, AP_list, precision_dict, recall_dict, F1_dict)

def evaluate_if(model, x_test, y_test, percentile_q_list = [99], n_samples=1000):
    AUC_list = []
    AP_list = []

    precision_dict = dict()
    recall_dict = dict()
    F1_dict = dict()

    for percentile_q in percentile_q_list:
        precision_dict[percentile_q] = []
        recall_dict[percentile_q] = []
        F1_dict[percentile_q] = []
    
    for _ in range(n_samples):
        x_new, y_new = stratified_sampling(x_test, y_test)
        y_pred = model.score_samples(x_new)

        AUC = roc_auc_score(y_new, y_pred)
        AP = average_precision_score(y_new, y_pred)

        AUC_list.append(AUC)
        AP_list.append(AP)

        for percentile_q in percentile_q_list:
            cutoff = np.percentile(y_pred, percentile_q)
            y_pred_hard = (y_pred >= cutoff)*1
            precision = precision_score(y_new, y_pred_hard)
            recall = recall_score(y_new, y_pred_hard)
            F1 = f1_score(y_new, y_pred_hard)

            precision_dict[percentile_q].append(precision)
            recall_dict[percentile_q].append(recall)
            F1_dict[percentile_q].append(F1)
        
    return(AUC_list, AP_list, precision_dict, recall_dict, F1_dict)

def save_results_TI(AUC_list, AP_list, model_name):
    res_dict = {'AUC': AUC_list, 'AP': AP_list}
    df = pd.DataFrame(res_dict)
    df.to_csv('res/'+model_name+'_TI.csv')

def save_results_TD(precision_dict, recall_dict, F1_dict, model_name):
    res_dict = dict()
    for key in precision_dict.keys():
        res_dict['precision_'+str(key)] = precision_dict[key]
        res_dict['recall_'+str(key)] = recall_dict[key]
        res_dict['F1_'+str(key)] = F1_dict[key]
    
    df = pd.DataFrame(res_dict)
    df.to_csv('res/'+model_name+'_TD.csv')