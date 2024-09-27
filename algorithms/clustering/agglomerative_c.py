import numpy as np
from sklearn.cluster import AgglomerativeClustering
from common import NELEMS, SEED, get_closest_elems, hamming_distance
from matplotlib import pyplot as plt

#def plot_dendogram(X):
#	#Using the dendrogram to find the optimal number of clusters
#	import scipy.cluster.hierarchy as sch
#	dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
#	plt.title('Agglomerativae Dendrogram')
#	plt.ylabel('Euclidean distances')
#	plt.savefig('images/agglomerative-dendogram.png')
#	plt.close()

def plot(clf, X, labels):
    plt.title('Agglomerative Clustering')
    plt.tight_layout()
    plt.scatter(x=X[:,0], y=X[:,1], c= clf.labels_, cmap='rainbow' )
    plt.savefig('images/agglomerative-clusters.png')
    plt.close()

# compute the clusters for clustering
def get_classifier(n_clusters, X):
    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    return AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity).fit(X)

    #return AgglomerativeClustering(n_clusters=n_clusters).fit(X)

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
