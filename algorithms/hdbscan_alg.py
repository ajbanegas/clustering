import numpy as np
import time
from sklearn.cluster import HDBSCAN
from common import load_dataset, get_query_list, save_items, NELEMS, compute_metrics, save_model
from matplotlib import pyplot as plt
import pandas as pd

# https://github.com/scikit-learn-contrib/hdbscan

def euclidean_distance(x, y):
	dist = (x - y)**2
	dist = np.sqrt(np.sum(dist, axis=1))
	return dist

def get_closest_elems(df, l_idx, query):
	dist = euclidean_distance(df.iloc[l_idx], query)
	dist = dist.reset_index()
	dist.columns = ['id', 'dist']

	# sort values by distance but keeping the index
	df2 = dist.sort_values(by="dist")
	return df2[:NELEMS].index.to_numpy()

def plot_hdbscan(clf, X):
        colors = ['c.', 'b.', 'r.', 'y.', 'g.']
        for Class, colour in zip(range(0, 5), colors):
                Xk = X[clf.labels_ == Class]
                plt.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha = 0.3)

        plt.plot(X.iloc[clf.labels_ == -1, 0], X.iloc[clf.labels_ == -1, 1], 'k+', alpha = 0.1)
        plt.title("HDBSCAN")
        plt.tight_layout()
        plt.savefig("images/hdbscan-clusters.png")
        plt.close()


# load dataset
df, X = load_dataset(foo='dataset-745-1conf.csv', sample=False)

# load the list of queries from DUD-E
query_list = get_query_list()

# compute the clusters for clustering
clf = HDBSCAN(min_samples=3, n_jobs=-1, cluster_selection_epsilon=.25, metric='cityblock', algorithm='kdtree', leaf_size=3, store_centers="centroid").fit(X)
#clf = HDBSCAN(min_samples=3, min_cluster_size=15, n_jobs=-1, store_centers="centroid").fit(X)
save_model(clf, "models/hdbscan.joblib")

# summarize the number of elements of each cluster
unique, counts = np.unique(clf.labels_, return_counts=True)
non_noisy_labels = clf.labels_[clf.labels_ != -1]
print(f"Number of clusters found: {len(np.unique(non_noisy_labels))}")

# plot the clusters
plot_hdbscan(clf, df)

# evaluate the model
#compute_metrics(X, clf.labels_)

time1 = time.time()

# predict querys to chose N closest elements
for i, query in enumerate(query_list):
	print(f"Processing query {i+1}")

	# find the closest element
	query = np.nan_to_num(query)
	dist = euclidean_distance(clf.centroids_, query)
	closest = np.argmin(dist)

	# identify the label of the new sample
	label = clf.labels_[closest]

	# find the other elements in the cluster
	elems_hdb = np.where(label == clf.labels_)[0]
	if len(elems_hdb) > NELEMS:
		# chose the NELEMS closest elements according to euclidean distance
		elems_hdb = get_closest_elems(df, elems_hdb, query)

	# save compund indexes for plotting
	save_items(f"elements/HDBSCAN-q-{i+1}.txt", elems_hdb)
	save_items(f"compounds/HDBSCAN-q-{i+1}.txt", df.iloc[elems_hdb].index.to_numpy(), fmt="%s")

	print('Compounds:', df.iloc[elems_hdb].index.to_numpy())
	print()

time2 = time.time()
print(f"Time: {time2-time1} sec")
