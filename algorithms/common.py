import numpy as np
import pandas as pd
from metrics import *
from eda.EDA import *
from eda.PreProcessing import *
import joblib


NELEMS = 300
SEED = 42
#FRAC = 0.05	# agglomerative
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

def get_query_list():
	df = pd.read_csv('query_list.csv', index_col=0)
	return df.to_numpy()

def save_items(foo, items, fmt="%i"):
	np.savetxt(foo, items, fmt=fmt)

def compute_metrics(X, y):
	print('###################')
	print('Computing metrics')
	print('###################')

	x_aux = X
	if X.shape[0] != y.shape[0]:
		x_aux = X[:min(X.shape[0], y.shape[0])]

	print(f'Silhouette: {silhouette(x_aux, y, random_state=SEED):.3f}')
	print(f'Davies-Bouldin: {davies_bouldin(x_aux, y):.3f}')
	print(f'Calinski-Harabasz: {calinski_harabasz(x_aux, y):.3f}')
	print()
	print()

def save_model(clf, foo):
        joblib.dump(clf, foo)

def load_model(foo):
        return joblib.load(foo)

"""
https://cdanielaam.medium.com/how-to-compare-and-evaluate-unsupervised-clustering-methods-84f3617e3769

only for supervised clustering

def completeness(y, yhat):
def homogeneity(y, yhat):
def mutual_info(y, yhat):
def recall(y, yhat):
def precision(y, yhat):
"""
