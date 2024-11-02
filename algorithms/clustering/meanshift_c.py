import numpy as np
from sklearn.cluster import MeanShift
from common import NELEMS, SEED, get_closest_elems

def get_classifier(n_clusters, X):
    return MeanShift(bandwidth=2, n_jobs=-1).fit(X)

def classify(clf, df, X, query):
    query = np.nan_to_num(query)
    y = clf.predict(query.reshape(1, -1))

    elems = np.where(clf.labels_ == y[0])[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems

