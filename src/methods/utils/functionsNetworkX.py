import networkx as nx
import pandas as pd

def local_features_nx(
        G_nx: nx.Graph, 
        alpha_pr: float, 
        alpha_ppr: float, 
        fraud_dict_train: dict=None
        ):
    
    if fraud_dict_train is None:
        use_fraud_features = False
    else:
        use_fraud_features = True

    feature_dict = dict() #dictionary to save all values

    pr_nx = nx.pagerank(G_nx, alpha=alpha_pr)

    if use_fraud_features:
        ppr_nx = nx.pagerank(G_nx, alpha=alpha_ppr, personalization=fraud_dict_train)

    for node in G_nx.nodes():
        ### Calculate the network features for a single node in the network ###
        ego_net = nx.ego_graph(G_nx, node)
        
        ## Density ##
        N = len(ego_net.nodes())
        M = len(ego_net.edges())
        
        try:
            density = 2*M/(N*(N-1))
        except:
            density = 1
        
        ##
        
        ego_net.remove_node(node)
        nodes_ego = ego_net.nodes()
        edges_ego = ego_net.edges()
        
        if use_fraud_features:
            ## Degree & RNC ##
            fraud_degree = legit_degree = 0
            
            RNC_F_node = RNC_NF_node = 0
            
            num_ego_nodes = len(nodes_ego)
            
            for node_ego in nodes_ego:
                #Fraud degree
                try:
                    fraud_degree += fraud_dict_train[node_ego]
                except:
                    fraud_degree += 0 #not in dictionary: label 0
                #Legit degree
                try:
                    legit_label = -1*(fraud_dict_train[node_ego]-1)
                    legit_degree += legit_label
                except:
                    legit_degree += 1 #not in dictionary: label 0
                    
            #RNC fraud
            try:
                RNC_F_node = fraud_degree/num_ego_nodes
            except:
                RNC_F_node = 0
            #RNC legit
            try:
                RNC_NF_node = legit_degree/num_ego_nodes
            except:
                RNC_NF_node = 0
            ##
            
            ## Triangles ##
            legit_triangle = semifraud_triangle = fraud_triangle = 0 
            
            for edge in edges_ego:
                try:
                    fraud_0 = fraud_dict_train[edge[0]]
                except: 
                    fraud_0 = 0 #not in dictionary: label 0
                try:
                    fraud_1 = fraud_dict_train[edge[1]]
                except: 
                    fraud_1 = 0 #not in dictionary: label 0
                    
                num_fraud = fraud_0 + fraud_1
                
                if num_fraud == 0:
                    legit_triangle += 1
                    
                if num_fraud == 1:
                    semifraud_triangle += 1
                    
                if num_fraud == 2:
                    fraud_triangle += 1
                    
            feature_dict[node] = [
                fraud_degree, 
                legit_degree, 
                fraud_triangle, 
                semifraud_triangle,
                legit_triangle,
                density, 
                RNC_F_node, 
                RNC_NF_node, 
                pr_nx[node],
                ppr_nx[node]
            ]
        else:
            feature_dict[node] = [
                density, 
                pr_nx[node]
                ]

    if use_fraud_features:
        features_df = pd.DataFrame(feature_dict, index=["fraud_degree", "legit_degree", "fraud_triangle", "semifraud_triangle", "legit_triangle", "density", "RNC_F_node", "RNC_NF_node", "PageRank", "PersonalisedPageRank"]).T
    else:
        features_df = pd.DataFrame(feature_dict, index=["density", "PageRank"]).T


    return(features_df)