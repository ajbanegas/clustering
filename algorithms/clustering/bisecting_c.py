import numpy as np
from sklearn.cluster import BisectingKMeans
from common import NELEMS, SEED, get_closest_elems
from matplotlib import pyplot as plt

def plot(clf, X, labels):
    y = clf.predict(X)
    centers = clf.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title('Bisecting K-Means')
    plt.tight_layout()
    plt.savefig('images/biskmeans-clusters.png')
    plt.close()

def get_classifier(n_clusters, X, dataset=""):
    return BisectingKMeans(n_clusters=n_clusters, random_state=SEED).fit(X)

def classify(clf, df, X, query, dataset=""):
    query = np.nan_to_num(query)
    y = clf.predict(query.reshape(1, -1))

    elems = np.where(clf.labels_ == y[0])[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems
