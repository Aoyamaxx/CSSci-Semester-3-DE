import numpy as np

def simple_function(point):
    x = point[0]
    return 2*x**2 + 4*x + 1

def simple_gradient(point):
    x = point[0]
    return np.array([4*x + 4,])

########################################################################################
# Function to implement the adam algorithm
# It is designed for academic purpose
# Various details have been omitted and no program optimization has done.
# Refer to https://www.geeksforgeeks.org/how-to-implement-adam-gradient-descent-from-scratch-using-python/
def adam(objective, derivative, bounds, n_iter=500,
		alpha=0.02, beta1=0.8, beta2=0.999, eps=1e-8):
	# Generate an initial point
	x = bounds[:, 0] + np.random.rand(len(bounds))\
	* (bounds[:, 1] - bounds[:, 0])
	scores = []
	trajectory = []

	# Initialize first and second moments
	m = np.zeros(bounds.shape[0])
	v = np.zeros(bounds.shape[0])

	# Run the gradient descent updates
	for t in range(n_iter):
		# Calculate gradient g(t)
		g = derivative(x)

		# Build a solution one variable at a time
		for i in range(x.shape[0]):
			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			# v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
			# mhat(t) = m(t) / (1 - beta1(t))
			mhat = m[i] / (1.0 - beta1 ** (t + 1))
			# vhat(t) = v(t) / (1 - beta2(t))
			vhat = v[i] / (1.0 - beta2 ** (t + 1))
			# x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
			x[i] = x[i] - alpha * mhat / (np.sqrt(vhat) + eps)

		# Evaluate candidate point
		score = objective(x)
		scores.append(score)
		trajectory.append(x.copy())

	return x, scores, trajectory
########################################################################################

if __name__ == '__main__':
	# Define the range for input 
	bounds = np.array([[-3.0, 3.0]])

	# Perform the gradient descent search with Adam
	best, scores, trajectory = adam(simple_function, simple_gradient, bounds, n_iter=1000)

	print(best)
