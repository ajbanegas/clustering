import numpy as np
from sklearn.cluster import DBSCAN
from common import NELEMS, get_closest_elems, hamming_distance
from matplotlib import pyplot as plt


def plot(clf, X, labels):
	unique_labels = set(labels)
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	core_samples_mask = np.zeros_like(labels, dtype=bool)
	core_samples_mask[clf.core_sample_indices_] = True

	colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			#col = [0, 0, 0, 1]
			continue

		class_member_mask = labels == k

		xy = X[class_member_mask & core_samples_mask]
		plt.plot(
			xy[:, 0],
			xy[:, 1],
			"o",
			markerfacecolor=tuple(col),
			markeredgecolor="k",
			markersize=14,
		)

		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(
			xy[:, 0],
			xy[:, 1],
			"o",
			markerfacecolor=tuple(col),
			markeredgecolor="k",
			markersize=6,
		)

	plt.title("DBSCAN")
	plt.savefig('images/dbscan-clusters.png')

def get_classifier(n_clusters, X):
    #return DBSCAN(eps=.005, min_samples=3, n_jobs=-1, metric="precomputed", leaf_size=5).fit(X)
    return DBSCAN(eps=.005, min_samples=3, n_jobs=-1, leaf_size=5).fit(X)

def classify(clf, df, X, query):
    query = np.nan_to_num(query)
    
    # calculate clusters' centroids excluding noise
    cluster_labels = np.unique(clf.labels_[clf.labels_ != -1])
    centroids = np.array([X[clf.labels_ == label].mean(axis=0) for label in cluster_labels])

    # calculate the distance between the query and all the centroids
    dist = np.zeros(centroids.shape[0])
    for i, cent in enumerate(centroids):
        dist[i] = hamming_distance(cent, query)

    # find the closest element
    #dist = hamming_distance(X[clf.core_sample_indices_], query)
    closest = np.argmin(dist)

    # identify the label of the new sample
    label = clf.labels_[closest]

    # find the other elements in the cluster
    elems = np.where(label == clf.labels_)[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems

