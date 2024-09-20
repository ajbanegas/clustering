import numpy as np
from common import NELEMS
from scipy.spatial.distance import hamming


def distance(x, y):
	dist = []
	for i in range(x.shape[0]):
		dist.append(hamming(x[i], y))

	# locate the elements with the smallest distance
	result = np.argpartition(dist, NELEMS)
	elems = result[:NELEMS]

	return dist, elems