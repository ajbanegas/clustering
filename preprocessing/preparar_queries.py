import pandas as pd
from glob import glob


def get_query_list(path):
	return glob(path)

def get_features(foo):
	df = pd.read_csv(foo)
	return df.columns.to_numpy()

def load_file(foo, features):
	return pd.read_csv(foo, header=0, sep='\t', usecols=features)[features]

def export(df):
	df.to_csv(f"query_list.csv")

if __name__ == "__main__":
	features = get_features('features.txt')
	files = query_list = get_query_list('./queries-txt/*.txt')

	df_all = None
	for i, foo in enumerate(files):
		print(f"Loading {foo}")
		df = load_file(foo, features)
		if df_all is None:
			df_all = df.copy()
		else:
			df_all = df_all.append(df)

	export(df_all)
