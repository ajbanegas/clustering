import numpy as np
import time
from sklearn.cluster import Birch
from common import load_dataset, get_query_list, save_items, NELEMS, compute_metrics, save_model
from matplotlib import pyplot as plt
import pandas as pd
import sys


# https://scicoding.com/diving-deep-into-k-means-clustering-a-scikit-learn-guide/
NCLUSTERS = 4


def plot_birch(clf, df):
	y_birch = clf.predict(df)
	plt.title('Birch')
	plt.scatter(df[:, 0], df[:, 1], c = y_birch, cmap = 'rainbow', alpha = 0.7, edgecolors = 'b')
	plt.tight_layout()
	plt.savefig('images/birch-clusters.png')
	plt.close()

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
sys.setrecursionlimit(1000000)
df, X = load_dataset(foo='dataset-745-1conf.csv', sample=True)

# determine the ideal number of clusters
#select_clusters(X, 20)
#exit()

# load the list of queries from DUD-E
query_list = get_query_list()

# compute the clusters for clustering
clf = Birch(n_clusters=NCLUSTERS, branching_factor=5, threshold=0.9).fit(X)
save_model(clf, "models/birch.joblib")

# summarize the number of elements of each cluster
unique, counts = np.unique(clf.subcluster_labels_, return_counts=True)
print(f"Number of samples per cluster: {counts}")

# plot the clusters
plot_birch(clf, X)

# evaluate the model
#compute_metrics(X, clf.subcluster_labels_)

time1 = time.time()

# predict querys to chose N closest elements
for i, query in enumerate(query_list):
	print(f"Processing query {i+1}")

	# birch
	query = np.nan_to_num(query)
	y_birch = clf.predict(query.reshape(1, -1))

	elems_b = np.where(clf.subcluster_labels_ == y_birch[0])[0]
	if len(elems_b) > NELEMS:
		# chose the NELEMS closest elements according to euclidean distance
		elems_b = get_closest_elems(df, elems_b, query)

	# save compund indexes for plotting
	save_items(f"elements/birch-q-{i+1}.txt", elems_b)
	save_items(f"compounds/birch-q-{i+1}.txt", df.iloc[elems_b].index.to_numpy(), fmt="%s")

	print('Compounds:', df.iloc[elems_b].index.to_numpy())
	print()

time2 = time.time()
print(f"Time: {time2-time1} sec")
