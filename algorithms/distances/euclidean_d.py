import numpy as np
import time
from common import NELEMS


def distance(x, y):
	dist = (x - y)**2
	dist = np.sqrt(np.sum(dist, axis=1))

	# locate the elements with the smallest distance
	result = np.argpartition(dist, NELEMS)
	elems = result[:NELEMS]
	
	return dist, elems