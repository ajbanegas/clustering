from common import load_dataset, get_query_list, save_items, save_model, compute_metrics, select_clusters, plot_clusters
from importlib import import_module
import numpy as np
from os.path import basename
import sys
import time


DISTANCE_KEYS = ['manhattan','euclidean','dice','hamming','canberra','chebyshev']
CLUSTERING_KEYS = ['kmeans','bisecting','agglomerative','dbscan','hdbscan','optics','birch']
N_CLUSTERS = 3

ds_file = sys.argv[1]
q_file = sys.argv[2]
alg_key = sys.argv[3]
dataset = ds_file.replace(".csv", "").split("_")[-1]
query_id = basename(q_file).split('/')[-1].replace(".csv", "").replace("rdkit_", "")

if alg_key not in DISTANCE_KEYS and alg_key not in CLUSTERING_KEYS:
	print(f"Algorithm {alg_key} does not exist.")
	print(f"Try with: {DISTANCE_KEYS} or {CLUSTERING_KEYS}")
	exit()

# load dataset
sys.setrecursionlimit(1000000)
df, X = load_dataset(foo=ds_file, fillnan=True)

# load the list of queries
query_list = get_query_list(q_file)

# calculate distance functions
if alg_key in DISTANCE_KEYS:

    module = import_module(f"distances.{alg_key}_d")

    time1 = time.time()

    # select the N closest ligands to the query
    for i, query in enumerate(query_list):
        print(f"Processing [{i+1}] {query_id} against {dataset} with {alg_key}")        

        # classical distances
        dist, elems = module.distance(X, query)

        # save compund indexes for plotting
        path = f"{alg_key}-{dataset.lower()}-{query_id}-{i+1}"
        save_items(f"elements/{path}.txt", elems)
        save_items(f"compounds/{path}.txt", df.iloc[elems].index.to_numpy(), fmt="%s")

        # print compound names
        print(f'Compounds {alg_key} - {dataset} - {query_id}:', df.iloc[elems].index.to_numpy())
        print()

    time2 = time.time()
    print(f"Time {alg_key} - {dataset}: {time2-time1} sec")

# calculate clustering
elif alg_key in CLUSTERING_KEYS:

    N_CLUSTERS = int(X.shape[0] / 300) + 1
    module = import_module(f"clustering.{alg_key}_c")

    #select_clusters(X, 20, 10, 'k-means++', 0.001, 8)

    # cluster the database
    time1 = time.time()
    clf = module.get_classifier(N_CLUSTERS, X)
    time2 = time.time()

    print(f"Clustering time {alg_key} - {dataset}: {time2-time1} sec")

    compress_level = 0 if alg_key == 'birch' else 3
    save_model(clf, f"models/{alg_key}-{dataset}.pkl", compress_level)
    exit()

    # summarize the number of elements of each cluster
    unique, counts = np.unique(clf.labels_, return_counts=True)
    print(f"Number of samples per cluster: {counts}")

    # obtain graphics
    labels = clf.labels_ if alg_key=='dbscan' else None
    data = df if alg_key=='hdbscan' or alg_key=='optics' else X
    #module.plot(clf, data, labels)
    plot_clusters(data, labels, alg_key, dataset)

    # evaluate the performance of the clustering
    compute_metrics(clf, X, clf.labels_)

    time1 = time.time()

    # select the N ligands closest to the query
    for i, query in enumerate(query_list):
        print(f"Processing query {i+1}")

        # cluster the dataset
        elems = module.classify(clf, df, X, query)

        # save compund indexes for later plotting
        path = f"{alg_key}-{dataset.lower()}-{query_id}-{i+1}"
        save_items(f"elements/{path}.txt", elems)
        save_items(f"compounds/{path}.txt", df.iloc[elems].index.to_numpy(), fmt="%s")

        print(f"Compounds {alg_key} - {dataset}:", df.iloc[elems].index.to_numpy())

    time2 = time.time()
    print(f"Inference time {alg_key} - {dataset}: {time2-time1} sec")

