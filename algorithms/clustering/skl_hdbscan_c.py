import numpy as np
from sklearn.cluster import HDBSCAN
from common import NELEMS, get_closest_elems, hamming_distance
from matplotlib import pyplot as plt

def plot(clf, X, labels):
    colors = ['c.', 'b.', 'r.', 'y.', 'g.']
    for Class, colour in zip(range(0, 5), colors):
        Xk = X[clf.labels_ == Class]
        plt.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha = 0.3)

    #plt.plot(X.iloc[clf.labels_ == -1, 0], X.iloc[clf.labels_ == -1, 1], 'k+', alpha = 0.1)
    plt.title("HDBSCAN")
    plt.tight_layout()
    plt.savefig("images/hdbscan-clusters.png")
    plt.close()


def get_classifier(n_clusters, X):
    return HDBSCAN(min_samples=10, n_jobs=-1, cluster_selection_epsilon=.25, leaf_size=40, store_centers="centroid").fit(X)

def classify(clf, df, X, query):
    query = np.nan_to_num(query)
    dist = hamming_distance(clf.centroids_, query)
    closest = np.argmin(dist)

    # identify the label of the new sample
    label = clf.labels_[closest]

    # find the other elements in the cluster
    elems = np.where(label == clf.labels_)[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)
    
    return elems

