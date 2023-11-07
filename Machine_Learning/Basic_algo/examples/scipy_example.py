import numpy as np
import scipy

def simple_function(point):
    x = point[0]
    return 2*x**2 + 4*x + 1

def simple_gradient(point):
    x = point[0]
    return np.array([4*x + 4,])

if __name__ == '__main__':
    x0 = [1.3]
    res = scipy.optimize.minimize(simple_function, x0, method='CG', tol=1e-6)
    print("Mininum for CG method: ", res.x)

    x0 = [1003]
    res = scipy.optimize.minimize(simple_function, x0, method='Newton-CG', jac=simple_gradient, tol=1e-6)
    print("Mininum for Newton-CG method: ", res.x)

    # The default method is chosen to be one of BFGS, L-BFGS-B, SLSQP, depending on whether or not the problem has constraints or bounds.
    x0 = [1003]
    res = scipy.optimize.minimize(simple_function, x0)
    print("Mininum for default method: ", res.x)