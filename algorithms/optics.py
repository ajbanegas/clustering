import numpy as np
import time
from sklearn.cluster import OPTICS
from common import load_dataset, get_query_list, save_items, NELEMS, compute_metrics, save_model
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd


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

def plot_optics(clf, X):
	for Class in np.unique(clf.labels_):
		if Class == -1:
			continue
		Xk = X[clf.labels_ == Class]
		plt.scatter(x=Xk.iloc[:,0], y=Xk.iloc[:,1], cmap='rainbow')

	plt.title("OPTICS")
	plt.tight_layout()
	plt.savefig("images/optics-clusters.png")
	plt.close()

# load dataset
#df, X = load_dataset(foo='dataset-745-1conf.csv', sample=False)
df, X = load_dataset(foo='test.csv')

# load the list of queries from DUD-E
query_list = get_query_list()

# compute the clusters for clustering
#clf = OPTICS(min_samples=3, max_eps=.9, xi=0.01, n_jobs=-1, metric='cityblock', algorithm='kd_tree', leaf_size=5).fit(X)
clf = OPTICS(min_samples=3, max_eps=.1, n_jobs=-1, metric='cityblock', algorithm='kd_tree', leaf_size=5).fit(X)
save_model(clf, "models/optics.joblib")

# summarize the number of elements of each cluster
unique, counts = np.unique(clf.labels_, return_counts=True)
print(f"Number of samples per cluster: {counts}")
print(f"Clusters: {unique}")

# evaluate the model
#compute_metrics(X, clf.labels_)

# plot the clusters
plot_optics(clf, df)

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
	elems_opt = np.where(label == clf.labels_)[0]
	if len(elems_opt) > NELEMS:
		# chose the NELEMS closest elements according to euclidean distance
		elems_opt = get_closest_elems(df, elems_opt, query)

	# save compund indexes for plotting
	save_items(f"elements/OPTICS-q-{i+1}.txt", elems_opt)
	save_items(f"compounds/OPTICS-q-{i+1}.txt", df.iloc[elems_opt].index.to_numpy(), fmt="%s")

	print('Compounds:', df.iloc[elems_opt].index.to_numpy())
	print()
	
time2 = time.time()
print(f"Time: {time2-time1} sec")
