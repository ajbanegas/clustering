import numpy as np
import time
from sklearn.cluster import DBSCAN
from common import load_dataset, get_query_list, save_items, NELEMS, compute_metrics, save_model
from matplotlib import pyplot as plt
import pandas as pd


def plot_dbscan(clf, X, labels):
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

def euclidean_distance(x, y):
	dist = (x - y)**2
	dist = np.sqrt(np.sum(dist, axis=1))
	return dist

def get_closest_elems(df, l_idx, query):
	dist = euclidean_distance(df.iloc[l_idx], query)
	dist = dist.reset_index()
	dist.columns = ['id', 'dist']

	# sor values by distance but keeping the index
	df2 = dist.sort_values(by="dist")
	return df2[:NELEMS].index.to_numpy()


# load dataset
#df, X = load_dataset(foo='dataset-745-1conf.csv', sample=True)
df, X = load_dataset("test.csv")

# load the list of queries from DUD-E
query_list = get_query_list()

# compute the clusters for clustering
#from sklearn.metrics import pairwise_distances
#distance_matrix = pairwise_distances(X, metric="cosine")
#print(distance_matrix)

clf = DBSCAN(eps=.005, min_samples=3, n_jobs=-1, metric="precomputed", leaf_size=5).fit(X)
save_model(clf, "models/dbscan.joblib")

# summarize the number of elements of each cluster
unique, counts = np.unique(clf.labels_, return_counts=True)
print(f"Number of samples per cluster: {counts}")

# plot the clusters
plot_dbscan(clf, X, clf.labels_)

# evaluate the model
compute_metrics(X, clf.labels_)

time1 = time.time()

# predict querys to chose N closest elements
for i, query in enumerate(query_list):
	print(f"Processing query {i+1}")

	# find the closest element
	dist = euclidean_distance(X[clf.core_sample_indices_], query)
	closest = np.argmin(dist)

	# identify the label of the new sample
	label = clf.labels_[closest]

	# find the other elements in the cluster
	elems_dbscan = np.where(label == clf.labels_)[0]
	if len(elems_dbscan) > NELEMS:
		# chose the NELEMS closest elements according to euclidean distance
		elems_dbscan = get_closest_elems(df, elems_dbscan, query)

	# save compund indexes for plotting
	save_items(f"elements/dbscan-q-{i+1}.txt", elems_dbscan)
	save_items(f"compounds/dbscan-q-{i+1}.txt", df.iloc[elems_dbscan].index.to_numpy(), fmt="%s")

	#print('Compounds:', df.iloc[elems_dbscan].index.to_numpy())
	#print()

time2 = time.time()
print(f"Time: {time2-time1} sec")

