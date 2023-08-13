from common import load_model, load_dataset, compute_metrics
import sys


# load dataset
df, X = load_dataset(foo='dataset-745-1conf.csv')
print("The dataset is loaded")

clf = load_model(sys.argv[1])
print(clf)

# evaluate the model
print("Computing metrics")
compute_metrics(X, clf.labels_)
