import numpy as np

def simple_function(point):
    x = point[0]
    return 2*x**2 + 4*x + 1

def simple_gradient(point):
    x = point[0]
    return np.array([4*x + 4,])

def gradient_descent(gradient_function, starting_point, learning_rate=0.1, threshold=0.0001, max_iterations=1000):
    x = starting_point # initializes x
    for i in range(max_iterations):
        # Iterate over x by updating the value using gradient descent
        grad = gradient_function(x) # gradient
        x = x - learning_rate * grad

        if np.linalg.norm(grad) < threshold:
            break
    
    return x, i+1 # return the minimum (and optionally the number of iterations it took)

if __name__ == '__main__':
    starting_point = [10,]  # Feel free to change this

    minimum, iterations = gradient_descent(simple_gradient, starting_point)
    print("Found minimum at x =", minimum)
    print("Number of iterations:", iterations)