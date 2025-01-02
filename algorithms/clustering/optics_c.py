import numpy as np
from sklearn.cluster import OPTICS
from common import NELEMS, get_closest_elems, hamming_distance
from matplotlib import pyplot as plt
import matplotlib.cm as cm

def plot(clf, X, labels):
	for Class in np.unique(clf.labels_):
		if Class == -1:
			continue
		Xk = X[clf.labels_ == Class]
		plt.scatter(x=Xk.iloc[:,0], y=Xk.iloc[:,1], cmap='rainbow')

	plt.title("OPTICS")
	plt.tight_layout()
	plt.savefig("images/optics-clusters.png")
	plt.close()

def get_classifier(n_clusters, X):
    #clf = OPTICS(min_samples=3, max_eps=.9, xi=0.01, n_jobs=-1, metric='cityblock', algorithm='kd_tree', leaf_size=5).fit(X)
    return OPTICS(min_samples=20, max_eps=100, n_jobs=-1).fit(X)

def classify(clf, df, X, query, dataset=""):
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

