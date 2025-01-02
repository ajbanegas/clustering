import numpy as np
import pandas as pd
from metrics import *
from eda.EDA import *
from eda.PreProcessing import *
from itertools import cycle, islice
import joblib
from matplotlib import pyplot as plt
import matplotlib
from sklearn.cluster import KMeans
from scipy.spatial.distance import hamming
import pickle

NELEMS = 300
SEED = 42
FRAC = 0.05
#N_CLUSTERS = 3 # 7
DISTANCE_KEYS = ['manhattan','euclidean','dice','hamming','canberra','chebyshev']
CLUSTERING_KEYS = ['kmeans','bisecting','agglomerative','dbscan','hdbscan','optics','birch','minibatch','meanshift','autoencoder','vae']
N_CLUSTERS = {'Drugbank':7, 'ChEBI':5, 'ChemSpaceQED':6, 'BioSynth':7, 'ChEMBL':9, 'Enamine':5}

def load_dataset(foo='dataset-745-allconf.csv', outliers=True, fillnan=True, sample=False):
	df = pd.read_csv(foo, index_col=0)
	#df = (df-df.min())/(df.max()-df.min())
	if sample:
		df = df.sample(frac=FRAC)
	print(df.head())

	if outliers:
		df = handle_outliers(df)

	X = df.to_numpy()
	if fillnan:
		X = np.nan_to_num(X)	# to avoid errors when predicting

	return df, X

def handle_outliers(df):
	cat_cols, num_cols, cat_but_car = grab_col_names(df)
	for col in num_cols:
		if check_outlier(df, col):
			replace_with_thresholds(df, col)
	return df

def get_query_list(foo, fillnan=False):
    df = pd.read_csv(foo, index_col=0)
    if fillnan:
        for c in df.columns:
            df[c] = df[c].fillna(df[c].mean())

    return df.to_numpy()

def save_items(foo, items, fmt="%i"):
    np.savetxt(foo, items, fmt=fmt)

def compute_metrics(clf, X, y):
    print('###################')
    print('Computing metrics')
    print('###################')

    x_aux = X
    if X.shape[0] != y.shape[0]:
        x_aux = X[:min(X.shape[0], y.shape[0])]
  
    yhat = clf.labels_ if hasattr(clf, 'labels_') else clf.predict(X)

    print(f'Silhouette: {silhouette(x_aux, y, random_state=SEED):.3f}')
    print(f'Davies-Bouldin: {davies_bouldin(x_aux, y):.3f}')
    print(f'Calinski-Harabasz: {calinski_harabasz(x_aux, y):.3f}')
    print(f'Completeness: {completeness(y, yhat):.3f}')
    print(f'Homogeneity: {homogeneity(y, yhat):.3f}')
    print(f'Mutual info: {mutual_info(y, yhat):.3f}')
    print(f'Precision: {precision(y, yhat):.3f}')
    print(f'Recall: {recall(y, yhat):.3f}')
    print()
    print()

def save_model(clf, foo, compress_level=3):
    with open(foo, 'wb') as f:
        pickle.dump(clf, f)

def load_model(foo):
    return pickle.load(open(foo, 'rb'))

def plot_inertia(inertials, dataset):
    x, y = zip(*[inertia for inertia in inertials])
    plt.plot(x, y, 'ro-', markersize=8, lw=2)
    plt.grid(True)
    font = {'family': 'normal', 'size': 10}
    matplotlib.rc('font', **font)
    plt.title(dataset)
    plt.xlabel('Num Clusters')
    plt.ylabel('Inertia')
    plt.savefig(f'images/kmeans-{dataset}-inertia.png')

def select_clusters(X, init_cluster, dataset):
    inertia_clusters = list()
    loops = 30
    max_iterations = 10
    tolerance = .001
    num_threads = 8

    for i in range(1, loops + 1, 1):
        clf = KMeans(n_clusters=i, max_iter=max_iterations, init=init_cluster, tol=tolerance)
        clf.fit(X)
        inertia_clusters.append([i, clf.inertia_])

    plot_inertia(inertia_clusters, dataset)

def euclidean_distance(x, y):
    dist = (x - y)**2
    dist = np.sqrt(np.sum(dist, axis=0))
    return dist

def hamming_distance(x, y):
    return hamming(x, y)

def get_closest_elems(df, l_idx, query, X=None):
    #calculate distance to each element in the cluster
    measure = np.zeros(len(l_idx))
    for i, idx in enumerate(l_idx):
        measure[i] = hamming_distance(df.iloc[idx].to_numpy(), query)
        #measure[i] = euclidean_distance(df.iloc[idx].to_numpy(), query)

    # find the index of the NELEMS closest elements
    sorted_idx = np.argsort(measure)

    # return the elements
    return sorted_idx[:NELEMS]

def plot_clusters(X, labels, alg_key, dataset):
    if labels is None:
        return

    y_pred = labels.astype(int)
    plt.scatter(X[:, 35], X[:, 122], s=10, alpha=.4, c=[i for i in labels])
    #plt.scatter(X[labels != -1, 35], X[labels != -1, 122], s=10, alpha=.4, c=[i for i in labels if i != -1])
    plt.title(f'Clusters of {alg_key} with {dataset}')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(f'images/{alg_key}-{dataset}.png')
    plt.close()

"""
https://cdanielaam.medium.com/how-to-compare-and-evaluate-unsupervised-clustering-methods-84f3617e3769

only for supervised clustering

def completeness(y, yhat):
def homogeneity(y, yhat):
def mutual_info(y, yhat):
def recall(y, yhat):
def precision(y, yhat):
"""

