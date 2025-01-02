import numpy as np
from sklearn.cluster import Birch
from common import NELEMS, get_closest_elems
from matplotlib import pyplot as plt
import sys

def plot(clf, df, labels):
	y = clf.predict(df)
	plt.title('Birch')
	plt.scatter(df[:, 0], df[:, 1], c = y, cmap = 'rainbow', alpha = 0.7, edgecolors = 'b')
	plt.tight_layout()
	plt.savefig('images/birch-clusters.png')
	plt.close()

# load dataset
#sys.setrecursionlimit(1000000)

def get_classifier(n_clusters, X, dataset=""):
    return Birch(n_clusters=n_clusters, branching_factor=200, threshold=1.5).fit(X)

# predict querys to chose N closest elements
def classify(clf, df, X, query, dataset=""):
    query = np.nan_to_num(query)
    y = clf.predict(query.reshape(1, -1))

    elems = np.where(clf.subcluster_labels_ == y[0])[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems

