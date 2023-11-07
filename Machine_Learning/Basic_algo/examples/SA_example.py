import numpy as np
from scipy.optimize import dual_annealing

def simple_function(point):
    x = point[0]
    return 2*x**2 + 4*x + 1

def simple_gradient(point):
    x = point[0]
    return np.array([4*x + 4,])

if __name__ == '__main__':
	# define range for input
	bounds = np.asarray([[-5.0, 5.0]])
	ret = dual_annealing(simple_function, bounds=bounds)

	print('Simulated annealing Done! Result:', 	ret.x)