import numpy as np
from common import NELEMS
from scipy.spatial.distance import chebyshev


def distance(x, y):
	dist = []
	for i in range(x.shape[0]):
		dist.append(chebyshev(x[i], y))

	# locate the elements with the smallest distance
	result = np.argpartition(dist, NELEMS)
	elems = result[:NELEMS]

	return dist, elems