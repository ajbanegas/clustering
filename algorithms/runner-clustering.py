from common import load_dataset, get_query_list, save_items, save_model, compute_metrics, plot_clusters, load_model
from common import DISTANCE_KEYS, CLUSTERING_KEYS
from importlib import import_module
import numpy as np
from os.path import basename
from sklearn.preprocessing import StandardScaler
import sys
import time

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
df, X = load_dataset(foo=ds_file, fillnan=True)
X = StandardScaler().fit_transform(X)

# load the list of queries
query_list = get_query_list(q_file, fillnan=True)

# calculate clustering
module = import_module(f"clustering.{alg_key}_c")

# load the model
clf = load_model(f"models/{alg_key}-{dataset}.pkl")

# summarize the number of elements of each cluster
unique, counts = np.unique(clf.labels_, return_counts=True)
print(f"Number of samples per cluster: {counts}")

# obtain graphics
labels = None if alg_key == 'featureagg' else clf.labels_
data = df.fillna(0.0).to_numpy() if alg_key=='hdbscan' or alg_key=='optics' else X
plot_clusters(data, labels, alg_key, dataset)

# evaluate the performance of the clustering
compute_metrics(clf, X, clf.labels_)

time1 = time.time()

# select the N ligands closest to the query
for i, query in enumerate(query_list):
    print(f"Processing query {i+1}")

    # cluster the dataset
    elems = module.classify(clf, df, X, query, dataset=dataset)

    # save compund indexes for later plotting
    path = f"{alg_key}-{dataset.lower()}-{query_id}-{i+1}"
    save_items(f"elements/{path}.txt", elems)
    save_items(f"compounds/{path}.txt", df.iloc[elems].index.to_numpy(), fmt="%s")

time2 = time.time()
print(f"Inference time {alg_key} - {dataset}: {time2-time1} sec")

