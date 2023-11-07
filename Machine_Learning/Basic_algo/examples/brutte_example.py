import numpy as np
import scipy
import matplotlib.pyplot as plt

def simple_function(point):
    x = point[0]
    return 2*x**2 + 4*x + 1

def simple_gradient(point):
    x = point[0]
    return np.array([4*x + 4,])

if __name__ == '__main__':
    rranges = (slice(-3, 3, 0.01),)
    resbrute = scipy.optimize.brute(simple_function, rranges, full_output=True)
    print("Brute force:",resbrute[0])
    plt.plot(resbrute[2],resbrute[3],".")
    plt.show()