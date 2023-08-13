import numpy as np
import time
from sklearn.cluster import BisectingKMeans
from common import load_dataset, get_query_list, save_items, NELEMS, SEED, compute_metrics, save_model
from matplotlib import pyplot as plt
import pandas as pd

# https://scicoding.com/diving-deep-into-k-means-clustering-a-scikit-learn-guide/
NCLUSTERS = 3


#######################################
# selection of the number of clusters
#https://jarroba.com/seleccion-del-numero-optimo-clusters/
def plot_inertia(inertials):
	x, y = zip(*[inertia for inertia in inertials])
	plt.plot(x, y, 'ro-', markersize=8, lw=2)
	plt.grid(True)
	plt.xlabel('Num Clusters')
	plt.ylabel('Inertia')
	plt.savefig('images/biskmeans-inertia.png')
	plt.close()

def select_clusters(points, loops, max_iterations, init_cluster, tolerance, num_threads):
    inertia_clusters = list()

    for i in range(1, loops + 1, 1):
        # Object BisectingKMeans
        bkmeans = BisectingKMeans(n_clusters=i, max_iter=max_iterations, init=init_cluster, tol=tolerance)

        # Calculate Kmeans
        bkmeans.fit(points)

        # Obtain inertia
        inertia_clusters.append([i, bkmeans.inertia_])

    plot_inertia(inertia_clusters)

#######################################

def plot_bkmeans(clf, X):
	y_kmeans = clf.predict(X)
	centers = clf.cluster_centers_
	plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow')
	plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
	plt.title('Bisecting K-Means')
	plt.tight_layout()
	plt.savefig('images/biskmeans-clusters.png')
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
df, X = load_dataset(foo='dataset-745-1conf.csv')

# determine the ideal number of clusters
#select_clusters(X, 20, 10, 'k-means++', 0.001, 8)
#exit()

# load the list of queries from DUD-E
query_list = get_query_list()

# compute the clusters for clustering
clf = BisectingKMeans(n_clusters=NCLUSTERS, random_state=SEED).fit(X)
save_model(clf, "models/biskmeans.joblib")

# summarize the number of elements of each cluster
unique, counts = np.unique(clf.labels_, return_counts=True)
print(f"Number of samples per cluster: {counts}")

# plot the clusters
plot_bkmeans(clf, X)

# evaluate the model
#compute_metrics(X, clf.labels_)

time1 = time.time()

# predict querys to chose N closest elements
for i, query in enumerate(query_list):
	print(f"Processing query {i+1}")

	# k-means
	query = np.nan_to_num(query)
	y_kmeans = clf.predict(query.reshape(1, -1))

	elems_km = np.where(clf.labels_ == y_kmeans[0])[0]
	if len(elems_km) > NELEMS:
		# chose the NELEMS closest elements according to euclidean distance
		elems_km = get_closest_elems(df, elems_km, query)

	# save compund indexes for plotting
	save_items(f"elements/bkmeans-q-{i+1}.txt", elems_km)
	save_items(f"compounds/bkmeans-q-{i+1}.txt", df.iloc[elems_km].index.to_numpy(), fmt="%s")

	print('Compounds:', df.iloc[elems_km].index.to_numpy())
	print()

time2 = time.time()
print(f"Time: {time2-time1} sec")
