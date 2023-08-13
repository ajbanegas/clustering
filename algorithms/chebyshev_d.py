#import cudf as pd
#import cupy as np
import pandas as pd
import numpy as np
import time
from common import load_dataset, get_query_list, save_items, NELEMS
from scipy.spatial.distance import chebyshev


def distance(x, y):
	dist = []
	for i in range(x.shape[0]):
		dist.append(chebyshev(x[i], y))

	# locate the elements with the smallest distance
	result = np.argpartition(dist, NELEMS)
	elems = result[:NELEMS]

	return dist, elems


# load dataset
df, X = load_dataset(fillnan=False)

# load the list of queries from DUD-E
query_list = get_query_list()

time1 = time.time()

# calculate the distance for all the queries
for i, query in enumerate(query_list):
	print(f"Processing query {i+1}")

	# classical distances
	d_che, elems_che = distance(X, query)

	# save compund indexes for plotting
	save_items(f"elements/chebyshev-q-{i+1}.txt", elems_che)
	save_items(f"compounds/chebyshev-q-{i+1}.txt", df.iloc[elems_che].index.to_numpy(), fmt="%s")

	print('Compounds:', df.iloc[elems_che].index.to_numpy())
	print()

time2 = time.time()
print(f"Time: {time2-time1} sec")
