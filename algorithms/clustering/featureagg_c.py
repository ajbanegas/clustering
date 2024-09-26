import numpy as np
from sklearn.cluster import FeatureAgglomeration
from common import NELEMS, SEED, get_closest_elems, euclidean_distance
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
    # find the closest element
    query = np.nan_to_num(query)
    dist = euclidean_distance(X, query)
    closest = np.argmin(dist)

    # identify the label of the new sample
    label = clf.labels_[closest]

    # find the other elements in the cluster
    elems = np.where(label == clf.labels_)[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems
