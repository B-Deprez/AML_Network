import networkx as nx
import pandas as pd
from multiprocessing import Pool, cpu_count
from node2vec import Node2Vec

def calculate_node_features(arg):
    node, G_nx, fraud_dict_train, use_fraud_features = arg
    ego_net = nx.ego_graph(G_nx, node)
    
    ## Density ##
    N = len(ego_net.nodes())
    M = len(ego_net.edges())
    
    try:
        density = 2 * M / (N * (N - 1))
    except:
        density = 1
    
    ego_net.remove_node(node)
    nodes_ego = ego_net.nodes()
    edges_ego = ego_net.edges()
    
    if use_fraud_features:
        ## Degree & RNC ##
        fraud_degree = legit_degree = 0
        RNC_F_node = RNC_NF_node = 0
        num_ego_nodes = len(nodes_ego)
        
        for node_ego in nodes_ego:
            # Fraud degree
            fraud_degree += fraud_dict_train.get(node_ego, 0)
            # Legit degree
            legit_label = -1 * (fraud_dict_train.get(node_ego, 0) - 1)
            legit_degree += legit_label
        
        # RNC fraud
        RNC_F_node = fraud_degree / num_ego_nodes if num_ego_nodes else 0
        # RNC legit
        RNC_NF_node = legit_degree / num_ego_nodes if num_ego_nodes else 0
        
        ## Triangles ##
        legit_triangle = semifraud_triangle = fraud_triangle = 0
        
        for edge in edges_ego:
            fraud_0 = fraud_dict_train.get(edge[0], 0)
            fraud_1 = fraud_dict_train.get(edge[1], 0)
            num_fraud = fraud_0 + fraud_1
            
            if num_fraud == 0:
                legit_triangle += 1
            elif num_fraud == 1:
                semifraud_triangle += 1
            elif num_fraud == 2:
                fraud_triangle += 1
        
        return (node, [
            fraud_degree, 
            legit_degree, 
            fraud_triangle, 
            semifraud_triangle,
            legit_triangle,
            density, 
            RNC_F_node, 
            RNC_NF_node
        ])
    else:
        return (node, [density])


def local_features_nx_calculation(
        G_nx: nx.Graph, 
        fraud_dict_train: dict=None, 
        use_fraud_features: bool=False
):
    feature_dict = dict() #dictionary to save all values

    

    if use_fraud_features:
        features_df = pd.DataFrame(feature_dict, index=["fraud_degree", "legit_degree", "fraud_triangle", "semifraud_triangle", "legit_triangle", "density", "RNC_F_node", "RNC_NF_node"]).T
    else:
        features_df = pd.DataFrame(feature_dict, index=["density"]).T

    args = [(node, G_nx, fraud_dict_train, use_fraud_features) for node in G_nx.nodes()]
    print("Calculating node features...")
    with Pool(cpu_count()) as pool:
        results = list(pool.map(calculate_node_features, args))
    
    print("Saving node features...")
    for node, features in results:
        feature_dict[node] = features
    
    if use_fraud_features:
        features_df = pd.DataFrame(
            feature_dict, 
            index=["fraud_degree", "legit_degree", "fraud_triangle", "semifraud_triangle", "legit_triangle", "density", "RNC_F_node", "RNC_NF_node"]
        ).T
    else:
        features_df = pd.DataFrame(feature_dict, index=["density"]).T
    
    return features_df

def local_features_nx(
        G_nx: nx.Graph, 
        alpha_pr: float, 
        alpha_ppr: float, 
        fraud_dict_train: dict=None,
        ntw_name: str='elliptic'
        ):
    
    if fraud_dict_train is None:
        use_fraud_features = False
    else:
        use_fraud_features = True
    
    location = 'res/'+ntw_name+'_features_nx.csv'
    try:
        features_df = pd.read_csv(location, index_col=0)
    except:
        features_df = local_features_nx_calculation(G_nx, fraud_dict_train=fraud_dict_train, use_fraud_features=use_fraud_features)
        features_df.to_csv(location)

    pr_nx = nx.pagerank(G_nx, alpha=alpha_pr)
    features_df['PageRank'] = [pr_nx[x] for x in features_df.index]

    if use_fraud_features:
        ppr_nx = nx.pagerank(G_nx, alpha=alpha_ppr, personalization=fraud_dict_train)
        features_df['PersonalisedPageRank'] = [ppr_nx[x] for x in features_df.index]

    return(features_df)

def node2vec_embedding(
        graph, 
        dimensions, 
        walk_length,
        num_walks,
        workers,
        p, 
        q,
        window_size, 
        negative
                    ):
    node2vec = Node2Vec(graph, dimensions=dimensions, p=p, q=q, walk_length=walk_length, num_walks=num_walks, workers=workers)
    # Embed nodes
    model = node2vec.fit(window=window_size, negative=negative, min_count=1)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    def get_embedding(u):
        return model.wv[u]

    return get_embedding

def node2vec_nx(
        G_nx: nx.Graph, 
        embedding_dim: int = 128,
        walk_length: int =20,
        context_size: int =10,
        walks_per_node: int =10,
        num_negative_samples: int =1,
        p: float =1.0,
        q: float =1.0, #node2vec hyper-parameters
        batch_size: int =128, 
        lr: float =0.01, 
        max_iter: int =150, 
        n_epochs: int =100
):
    print("Calculating node2vec embedding...")
    embedding = node2vec_embedding(
        G_nx, 
        embedding_dim, 
        walk_length, 
        walks_per_node, 
        int(cpu_count()/1.5), # Not use all to save some memory
        p, 
        q, 
        context_size,
        num_negative_samples
    )
    print("Saving node2vec embedding...")
    features_df = pd.DataFrame([embedding(x) for x in G_nx.nodes()], index=G_nx.nodes())
    return features_df