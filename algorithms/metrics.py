from sklearn.metrics import silhouette_score, completeness_score, mutual_info_score, homogeneity_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import recall_score, precision_score


# internal metrics
def completeness(y, yhat):
	return completeness_score(y, yhat)

def homogeneity(y, yhat):
	return homogeneity_score(y, yhat)

def silhouette(X, y, random_state=42):
	return silhouette_score(X, y, random_state=random_state)

def davies_bouldin(X, y):
	return davies_bouldin_score(X, y)

def calinski_harabasz(X, y):
	return calinski_harabasz_score(X, y)
	
# external metrics
def mutual_info(y, yhat):
	return mutual_info_score(y, yhat)

def recall(y, yhat):
	return recall_score(y, yhat)

def precision(y, yhat):
	return precision_score(y, yhat)
