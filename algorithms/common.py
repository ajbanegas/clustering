import numpy as np
import pandas as pd
from metrics import *
from eda.EDA import *
from eda.PreProcessing import *
import joblib
from matplotlib import pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from scipy.spatial.distance import hamming
import pickle

NELEMS = 300
SEED = 42
FRAC = 0.05

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
    #joblib.dump(clf, foo, compress=compress_level)
    with open(foo, 'wb') as f:
        pickle.dump(clf, f)

def load_model(foo):
    #return joblib.load(foo)
    return pickle.load(open(foo, 'rb'))

def plot_inertia(inertials):
    x, y = zip(*[inertia for inertia in inertials])
    plt.plot(x, y, 'ro-', markersize=8, lw=2)
    plt.grid(True)
    font = {'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}
    matplotlib.rc('font', **font)
    plt.xlabel('Num Clusters')
    plt.ylabel('Inertia')
    plt.savefig('images/kmeans-inertia.png')

def select_clusters(points, loops, max_iterations, init_cluster, tolerance, num_threads):
    inertia_clusters = list()

    for i in range(1, loops + 1, 1):
        # Object KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=i, max_iter=max_iterations, init=init_cluster, tol=tolerance)

        # Calculate Kmeans
        kmeans.fit(points)

        # Obtain inertia
        inertia_clusters.append([i, kmeans.inertia_])

    plot_inertia(inertia_clusters)

def euclidean_distance(x, y):
    dist = (x - y)**2
    dist = np.sqrt(np.sum(dist, axis=1))
    return dist

def hamming_distance(x, y):
    return hamming(x, y)

def get_closest_elems(df, l_idx, query):
    #calculate distance to each element in the cluster
    measure = np.zeros(len(l_idx))
    for i, idx in enumerate(l_idx):
        measure[i] = hamming_distance(df.iloc[idx].to_numpy(), query)

    # find the index of the NELEMS closest elements
    pos = np.argpartition(measure, NELEMS)

    # return the elements
    return pos

def plot_clusters(X, labels, alg_key, dataset):
    # plot distribution after tSNE algorithm to reduce the dimensionality
    tsne = TSNE(n_components=3, perplexity=30, n_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=5)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f't-SNE of {alg_key} with {dataset}')
    plt.colorbar()
    plt.savefig(f'images/{alg_key}-{dataset}.png')
    plt.close()

    # plot cluster distribution (number of elements per cluster)
    plt.hist(labels)
    plt.ylabel('Number of items')
    plt.xlabel('Cluster')
    plt.title(f'Distribution of items per cluster: {alg_key} in {dataset}')
    plt.tight_layout()
    plt.savefig(f'images/hist-{alg_key}-{dataset}.png')
    plt.close()

    #yhat = clf.labels_ if hasattr(clf, 'labels_') else clf.predict(X)

"""
https://cdanielaam.medium.com/how-to-compare-and-evaluate-unsupervised-clustering-methods-84f3617e3769

only for supervised clustering

def completeness(y, yhat):
def homogeneity(y, yhat):
def mutual_info(y, yhat):
def recall(y, yhat):
def precision(y, yhat):
"""

