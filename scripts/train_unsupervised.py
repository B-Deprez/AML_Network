import os 
import sys

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

from src.methods.experiments_unsupervised import *
from src.data.DatasetConstruction import *
import optuna

#### Optuna objective ####
def objective_intrinsic(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_samples = trial.suggest_float('max_samples', 0.1, 1)
    max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
    max_features = max_features_dec*10
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    ap_loss = intrinsic_features(
        ntw, 
        train_mask, 
        val_mask,
        n_estimators, 
        max_samples,
        max_features, 
        bootstrap
        )
    return(ap_loss)

def objective_positional(trial):
    alpha_pr = trial.suggest_float('alpha_pr', 0.1, 0.9)
    alpha_ppr = 0#trial.suggest_float('alpha_ppr', 0.1, 0.9)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_samples = trial.suggest_float('max_samples', 0.1, 1)
    max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
    max_features = max_features_dec*10
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    ap_loss = positional_features(
        ntw, 
        train_mask, 
        val_mask,
        alpha_pr,
        alpha_ppr,
        n_estimators, 
        max_samples,
        max_features, 
        bootstrap,
        fraud_dict_test=fraud_dict,
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
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_samples = trial.suggest_float('max_samples', 0.1, 1)
    max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
    max_features = max_features_dec*10
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

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
        n_estimators, 
        max_samples,
        max_features,
        bootstrap
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
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_samples = trial.suggest_float('max_samples', 0.1, 1)
    max_features_dec = trial.suggest_int('max_features_dec%', 1, 10)
    max_features = max_features_dec*10
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

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
        n_estimators, 
        max_samples,
        max_features, 
        bootstrap
        )
    return(ap_loss)

if __name__ == "__main__":
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

    train_mask, val_mask, test_mask = ntw.get_masks()

    to_train = [
        "intrinsic",
        "positional",
        "deepwalk",
        "node2vec"
    ]

    ### Train intrinsic features ###
    if "intrinsic" in to_train:
        print("="*10)
        print("intrinsic: ")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_intrinsic, n_trials=100)
        intrinsic_params = study.best_params   
        intrinsic_values = study.best_value
        with open("res/intrinsic_params_"+ntw_name+"_unsupervised.txt", "w") as f:
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
        with open("res/positional_params_"+ntw_name+"_unsupervised.txt", "w") as f:
            f.write(str(positional_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(positional_values))

    ### Train Torch ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ntw_torch = ntw.get_network_torch().to(device)
    ntw_torch.x = ntw_torch.x[:,1:94] # Remove time_step and summary features
    edge_index = ntw_torch.edge_index
    num_features = ntw_torch.num_features
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
        with open("res/deepwalk_params_"+ntw_name+"_unsupervised.txt", "w") as f:
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
        with open("res/node2vec_params_"+ntw_name+"_unsupervised.txt", "w") as f:
            f.write(str(node2vec_params))
            f.write("\n")
            f.write("AUC-PRC: "+str(node2vec_values))
