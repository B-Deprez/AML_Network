import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# To do with unsupervised methods
# > implementation of isolation forest as backbone for unsupervised methods
# > implementation of k-means clustering as backbone for unsupervised methods (https://medium.com/@tommaso.romani2000/harnessing-the-power-of-k-means-for-anomaly-detection-24dc71d260a8)
# > application of unsupervised learning on intrinsic features
# > application of unsupervised learning on local features
# > application of unsupervised learning on deepwalk
# > application of unsupervised learning on node2vec

def isolation_forest(
        X_train: np.ndarray, 
        n_estimators: int, 
        max_features: int, 
        bootstrap: bool
):
    clf = IsolationForest(
        n_estimators= n_estimators,
        max_features= max_features,
        bootstrap=bootstrap,
        random_state=1997)
    clf.fit(X_train)
    y_scores = clf.score_samples(X_train)
    return(y_scores)