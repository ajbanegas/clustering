import numpy as np
import time
from sklearn.cluster import AgglomerativeClustering
from common import load_dataset, get_query_list, save_items, NELEMS, SEED, compute_metrics, save_model
from matplotlib import pyplot as plt
import pandas as pd


NCLUSTERS = 3

def plot_dendogram(X):
	#Using the dendrogram to find the optimal number of clusters
	import scipy.cluster.hierarchy as sch
	dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
	plt.title('Agglomerativae Dendrogram')
	plt.ylabel('Euclidean distances')
	plt.savefig('images/agglomerative-dendogram.png')
	plt.close()

def plot_agglomerative(clf, X):
	plt.title('Agglomerative Clustering')
	plt.tight_layout()
	plt.scatter(x=X[:,0], y=X[:,1], c= clf.labels_, cmap='rainbow' )
	plt.savefig('images/agglomerative-clusters.png')
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
df, X = load_dataset(foo='dataset-745-1conf.csv', sample=True)

# determine the ideal number of clusters
#select_clusters(X, 20)
#exit()

# load the list of queries from DUD-E
query_list = get_query_list()

# compute the clusters for clustering
clf = AgglomerativeClustering(n_clusters=NCLUSTERS).fit(X)
save_model(clf, "models/agglomerative.joblib")

# summarize the number of elements of each cluster
unique, counts = np.unique(clf.labels_, return_counts=True)
print(f"Number of samples per cluster: {counts}")

# plot the clusters
plot_agglomerative(clf, X)

# evaluate the model
#compute_metrics(X, clf.labels_)

time1 = time.time()

# predict querys to chose N closest elements
for i, query in enumerate(query_list):
	print(f"Processing query {i+1}")

	# find the closest element
	query = np.nan_to_num(query)
	dist = euclidean_distance(X, query)
	closest = np.argmin(dist)

	# identify the label of the new sample
	label = clf.labels_[closest]

	# find the other elements in the cluster
	elems_agg = np.where(label == clf.labels_)[0]
	if len(elems_agg) > NELEMS:
		# chose the NELEMS closest elements according to euclidean distance
		elems_agg = get_closest_elems(df, elems_agg, query)

	# save compund indexes for plotting
	save_items(f"elements/agglomerative-q-{i+1}.txt", elems_agg)
	save_items(f"compounds/agglomerative-q-{i+1}.txt", df.iloc[elems_agg].index.to_numpy(), fmt="%s")

	print('Compounds:', df.iloc[elems_agg].index.to_numpy())
	#compute_metrics([], [], y_agglomerative)
	print()

time2 = time.time()
print(f"Time: {time2-time1} sec")
