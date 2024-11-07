import os 
import sys

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

from src.methods.experiments import *
from src.data.DatasetConstruction import *

#### Optuna objective ####
def objective_intrinsic(trial):
    n_layers_decoder = trial.suggest_int('n_layers_decoder', 1, 3)
    hidden_dim_decoder = trial.suggest_int('hidden_dim_decoder', 5, 20)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs_decoder = trial.suggest_int('n_epochs_decoder', 5, 500)

    ap_loss = intrinsic_features_supervised(
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
        hidden_dim_decoder=hidden_dim_decoder, 
        ntw_name=ntw_name+"_train"
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
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0, 0.5)
    lr = trial.suggest_float('lr', 0.01, 0.1)
    n_epochs = trial.suggest_int('n_epochs', 5, 500)

    heads = trial.suggest_int("heads", 1, 5)

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
    ntw_name = "ibm"

    if ntw_name == "ibm":
        ntw = load_ibm()
    elif ntw_name == "elliptic":
        ntw = load_elliptic()
    elif ntw_name == "cora":
        ntw = load_cora()
    else:
        raise ValueError("Network not found")

    train_mask, val_mask, test_mask = ntw.get_masks()

    to_train = [
        "intrinsic",
        "positional",
        "deepwalk",
        "node2vec",
        "gcn",
        "sage",
        "gat",
        "gin"
    ]

    ### Train intrinsic features ###
    if "intrinsic" in to_train:
        print("="*10)
        print("intrinsic: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_intrinsic, n_trials=100)
        intrinsic_params = study.best_params   
        intrinsic_values = study.best_value
        with open("res/intrinsic_params_"+ntw_name+".txt", "w") as f:
            f.write(str(intrinsic_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(intrinsic_values))

    ### Train positional features ###
    fraud_dict = ntw.get_fraud_dict()
    fraud_dict = {k: 0 if v == 2 else v for k, v in fraud_dict.items()}

    ## Train positional features
    if "positional" in to_train:
        print("="*10)
        print("positional: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_positional, n_trials=100)
        positional_params = study.best_params
        positional_values = study.best_value
        with open("res/positional_params_"+ntw_name+".txt", "w") as f:
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
    if "deepwalk" in to_train:
        print("="*10)
        print("deepwalk: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_deepwalk, n_trials=100)
        deepwalk_params = study.best_params
        deepwalk_values = study.best_value
        with open("res/deepwalk_params_"+ntw_name+".txt", "w") as f:
            f.write(str(deepwalk_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(deepwalk_values))

    ## Train node2vec 
    if "node2vec" in to_train:
        print("="*10)
        print("node2vec: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_node2vec, n_trials=100)
        node2vec_params = study.best_params
        node2vec_values = study.best_value
        with open("res/node2vec_params_"+ntw_name+".txt", "w") as f:
            f.write(str(node2vec_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(node2vec_values))

    ### Train GNN ###
    ## GCN     
    if "gcn" in to_train:     
        print("="*10)      
        print("GCN: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_gcn, n_trials=100)
        gcn_params = study.best_params
        gcn_values = study.best_value
        with open("res/gcn_params_"+ntw_name+".txt", "w") as f:
            f.write(str(gcn_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(gcn_values))
                                
    # GraphSAGE
    if "sage" in to_train:
        print("="*10)
        print("GraphSAGE: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_sage, n_trials=100)
        sage_params = study.best_params
        sage_values = study.best_value
        with open("res/sage_params_"+ntw_name+".txt", "w") as f:
            f.write(str(sage_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(sage_values))

    # GAT
    if "gat" in to_train:
        print("="*10)
        print("GAT: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_gat, n_trials=100)
        gat_params = study.best_params
        gat_values = study.best_value
        with open("res/gat_params_"+ntw_name+".txt", "w") as f:
            f.write(str(gat_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(gat_values))

    # GIN
    if "gin" in to_train:
        print("="*10)
        print("GIN: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_gin, n_trials=100)
        gin_params = study.best_params
        gin_values = study.best_value
        with open("res/gin_params_"+ntw_name+".txt", "w") as f:
            f.write(str(gin_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(gin_values))