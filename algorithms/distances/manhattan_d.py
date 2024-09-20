import numpy as np
from common import NELEMS


def distance(x, y):
	dist = np.sum(np.abs(x - y), axis=1)

	# locate the elements with the smallest distance
	result = np.argpartition(dist, NELEMS)
	elems = result[:NELEMS]
	
	return dist, elems
