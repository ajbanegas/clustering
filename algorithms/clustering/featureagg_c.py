import numpy as np
from sklearn.cluster import FeatureAgglomeration, KMeans
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
    plt.title('Feature Agglomeration')
    plt.tight_layout()
    plt.scatter(x=X[:,0], y=X[:,1], c= clf.labels_, cmap='rainbow' )
    plt.savefig('images/featureagg-clusters.png')
    plt.close()

# compute the clusters for clustering
def get_classifier(n_clusters, X):
    #from sklearn.neighbors import kneighbors_graph
    #connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
    n_clusters = 100
    return FeatureAgglomeration(n_clusters=n_clusters, memory="./tmp", linkage="average", compute_full_tree=True).fit(X)

def classify(clf, df, X, query):
    # prepare the query compound
    query = np.nan_to_num(query)
    query_reduced = clf.transform(query.reshape(1, -1))
    
    # calculate the distance between the query and all the centroids
    X_reduced = clf.fit_transform(X)
    
    # apply KMeans to the reduced space
    kmeans = KMeans(n_clusters=5, random_state=42)  # Ajustar el número de clusters según sea necesario
    kmeans.fit(X_reduced)    
    
    # collect the centroids from the clusters
    centroids = kmeans.cluster_centers_
    
    # find the closest centroid
    dist = np.zeros(centroids.shape[0])
    for i, cent in enumerate(centroids):
        dist[i] = hamming_distance(cent, query_reduced[0])
    
    #dist = euclidean_distance(X, query)
    closest_centroid_idx  = np.argmin(dist)
    
    # obtain the index of the points belonging to the cluster with the closest centroid
    points_in_closest_cluster = np.where(kmeans.labels_ == closest_centroid_idx)[0]
    
    # identify the label of the new sample
    closest_point_idx = points_in_closest_cluster[0]
    label = clf.labels_[closest_point_idx]

    # find the other elements in the cluster
    elems = np.where(label == clf.labels_)[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems
