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

def positinal_features(
        ntw, train_mask, test_mask, fraud_dict,
        n_epochs_decoder_list: list, 
        lr: float,
        w_a,
        file: str = "positional_results.txt"
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
        print(f"Epoch {epoch+1}: Loss: {loss.item()}")
        if epoch in n_epochs_decoder_list:
            y_pred = decoder(x_test)
            ap_score = round(average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1]), 2)
            with open(file, w_a) as f:
                f.write(f"Epochs: {epoch},")
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
        n_epochs_decoder_list: list,
        w_a, 
        file: str = "node2vec_results.txt"
):
    model_n2v = node2vec_representation(
        ntw_torch,
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

    n_epochs = 100
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
        print(f"Epoch {epoch+1}: Loss: {loss.item()}")
        if epoch in n_epochs_decoder_list:
            y_pred = decoder(x_test)
            ap_score = round(average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1]), 2)
            with open(file, w_a) as f:
                f.write(f"Embedding Dim: {embedding_dim},")
                f.write(f"Walk Length: {walk_length},")
                f.write(f"Context Size: {context_size},")
                f.write(f"Walks per Node: {walks_per_node},")
                f.write(f"Negative Samples: {num_negative_samples},")
                f.write(f"P: {p},")
                f.write(f"Q: {q},")
                f.write(f"Epochs: {epoch},")
                f.write(f"Learning Rate: {lr},")
                f.write(f"Loss: {loss.item()},")
                f.write(f"AP Score: {ap_score} \n")
            w_a = "a"

def LINE_features(
        ntw_torch, train_mask, test_mask,
        embedding_dim,
        num_negative_samples,
        lr,
        n_epochs,
        n_epochs_decoder_list: list,
        w_a,
        file: str = "LINE_results.txt"
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

    n_epochs = 100
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
        print(f"Epoch {epoch+1}: Loss: {loss.item()}")
        if epoch in n_epochs_decoder_list:
            y_pred = decoder(x_test)
            ap_score = round(average_precision_score(y_test.cpu().detach().numpy(), y_pred.cpu().detach().numpy()[:,1]), 2)
            with open(file, w_a) as f:
                f.write(f"Embedding Dim: {embedding_dim},")
                f.write(f"Negative Samples: {num_negative_samples},")
                f.write(f"Epochs: {epoch},")
                f.write(f"Learning Rate: {lr},")
                f.write(f"Loss: {loss.item()},")
                f.write(f"AP Score: {ap_score} \n")
            w_a = "a"

def GNN_features(
        ntw_torch: Data,
        model: nn.Module,
        batch_size: int, 
        lr: float,
        n_epochs_list: List, 
        loader: DataLoader =None
):
    hidden_dim = model.hidden_dim
    embedding_dim = model.embedding_dim
    output_dim = model.output_dim
    dropout_rate = model.dropout_rate
    n_layers = model.n_layers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    name = model._get_name()

    n_epochs = max(n_epochs_list)
    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model, batch_size=batch_size, lr=lr, loader=loader)
        loss_test = test_GNN(ntw_torch, model)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')
        if epoch in n_epochs_list:
            with open(name+"_results.txt", "a") as f:
                if name == "GraphSAGE":
                    f.write(f"Aggregation: {model.sage_aggr},")
                if name == "GAT":
                    f.write(f"Heads: {model.heads},")
                f.write(f"Hidden Dim: {hidden_dim},")
                f.write(f"Embedding Dim: {embedding_dim},")
                f.write(f"Output Dim: {output_dim},")
                f.write(f"Layers: {n_layers},")
                f.write(f"Dropout Rate: {dropout_rate},")
                f.write(f"Epochs: {epoch},")
                f.write(f"Learning Rate: {lr},")
                f.write(f"Loss Train: {loss_train:.4f},")
                f.write(f"Loss Test: {loss_test:.4f} \n")

if __name__ == "__main__":
    ### Load Elliptic Dataset ###
    ntw = load_elliptic()
    train_mask, val_mask, test_mask = ntw.get_masks()

    ### Train positional features ###
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}
    lr_list = [0.01, 0.02, 0.03]
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
    hidden_dim_list = 64
    embedding_dim_list = 16
    output_dim_list = 2
    n_layers_list = 3
    dropout_rate_list = 0
    batch_size=128
    n_epochs_list = 2 #make list

    ## Train node2vec 
    print("node2vec: ")
    walk_length_list = 20
    context_size_list = 10
    walks_per_node_list = 10
    num_negative_samples_list = 1
    p_list = 1.0
    q_list = 1.0

    w_a = "w" #string to indicate whether the file is being written or appended
    for embedding_dim in embedding_dim_list:
        for walk_length in walk_length_list:
            for context_size in [cs for cs in context_size_list if cs <= walk_length]:
                for walks_per_node in walks_per_node_list:
                    for num_negative_samples in num_negative_samples_list:
                        for p in p_list:
                            for q in q_list:
                                for lr in lr_list:
                                    for n_epochs in n_epochs_list:
                                        node2vec_features(
                                            ntw_torch, train_mask, val_mask,
                                            embedding_dim, walk_length, context_size, walks_per_node, num_negative_samples, p, q, lr, n_epochs, n_epochs_decoder, w_a
                                        )
                                        w_a = "a"


    #Include loops for different parameters node2vec

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
    for hidden_dim in hidden_dim_list:
        for embedding_dim in embedding_dim_list:
            for output_dim in output_dim_list:
                for n_layers in n_layers_list:
                    for dropout_rate in dropout_rate_list:
                        for lr in lr_list:
                            for n_epochs in n_epochs_list:
                                ## GCN
                                model_gcn = GCN(
                                    edge_index=edge_index, 
                                    num_features=num_features,
                                    hidden_dim=hidden_dim,
                                    embedding_dim=embedding_dim,
                                    output_dim=output_dim,
                                    n_layers=n_layers,
                                    dropout_rate=dropout_rate
                                    ).to(device)
                                GNN_features(ntw_torch, model_gcn, batch_size, lr, n_epochs_list)

                                ## GraphSAGE
                                for sage_aggr in sage_aggr_list:
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
                                        num_neighbors = num_neighbors,
                                        input_nodes = ntw_torch.train_mask,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        num_workers = 0
                                    )
                                    GNN_features(ntw_torch, model_sage, batch_size, lr, n_epochs_list, loader=loader)

                                ## GAT
                                for heads in heads_list:
                                    model_gat = GAT(
                                        num_features=num_features,
                                        hidden_dim=hidden_dim,
                                        embedding_dim=embedding_dim,
                                        output_dim=output_dim,
                                        n_layers=n_layers,
                                        heads=heads,
                                        dropout_rate=dropout_rate
                                    ).to(device)

                                    GNN_features(ntw_torch, model_gat, batch_size, lr, n_epochs_list)

                                ## GIN
                                model_gin = GIN(
                                    num_features=num_features,
                                    hidden_dim=hidden_dim,
                                    embedding_dim=embedding_dim,
                                    output_dim=output_dim,
                                    n_layers=n_layers,
                                    dropout_rate=dropout_rate
                                ).to(device)

                                GNN_features(ntw_torch, model_gin, batch_size, lr, n_epochs_list)
                                
    # GraphSAGE
    print("GraphSAGE: ")
    num_neighbors = [2]*n_layers
    sage_aggr = 'mean'

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
        num_neighbors = num_neighbors,
        input_nodes = ntw_torch.train_mask,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
        )
    
    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model_sage, loader=loader, lr=lr)
        loss_test = test_GNN(ntw_torch, model_sage)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')

    # GAT
    print("GAT: ")
    heads = 6

    model_gat = GAT(
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        heads=heads,
        dropout_rate=dropout_rate
    ).to(device)

    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model_gat, batch_size=batch_size, lr=lr)
        loss_test = test_GNN(ntw_torch, model_gat)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')

    # GIN
    print("GIN: ")
    model_gin = GIN(
        num_features=num_features,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout_rate=dropout_rate
    ).to(device)

    for epoch in range(n_epochs):
        loss_train = train_GNN(ntw_torch, model_gin, batch_size=batch_size, lr=lr)
        loss_test = test_GNN(ntw_torch, model_gin)
        print(f'Epoch: {epoch+1:03d}, Loss Train: {loss_train:.4f}, Loss Test: {loss_test:.4f}')

    