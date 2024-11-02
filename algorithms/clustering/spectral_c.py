import numpy as np
from sklearn.cluster import SpectralClustering
from common import NELEMS, SEED, get_closest_elems, hamming_distance
from matplotlib import pyplot as plt

# compute the clusters for clustering
def get_classifier(n_clusters, X):
    return SpectralClustering(n_clusters=n_clusters, random_state=0, assign_labels='discretize').fit(X)

def classify(clf, df, X, query):
    # find the closest element
    query = np.nan_to_num(query)
    
    # calculate clusters' centroids excluding noise
    cluster_labels = np.unique(clf.labels_[clf.labels_ != -1])
    centroids = np.array([X[clf.labels_ == label].mean(axis=0) for label in cluster_labels])

    # calculate the distance between the query and all the centroids
    dist = np.zeros(centroids.shape[0])
    for i, cent in enumerate(centroids):
        dist[i] = hamming_distance(cent, query)
        
    #dist = hamming_distance(X, query)
    closest = np.argmin(dist)

    # identify the label of the new sample
    label = clf.labels_[closest]

    # find the other elements in the cluster
    elems = np.where(label == clf.labels_)[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems
