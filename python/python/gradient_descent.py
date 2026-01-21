import numpy as np

def gradient_descent(grad, x0, lr=0.1, max_iter=1000, tol=1e-6):
    """
    Basic gradient descent algorithm.

    Parameters
    ----------
    grad : callable
        Gradient of the objective function.
    x0 : ndarray
        Initial point.
    lr : float
        Learning rate.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for stopping criterion.

    Returns
    -------
    x : ndarray
        Approximate minimizer.
    """
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g
    return x


if __name__ == "__main__":
    grad = lambda x: 2 * x  # gradient of f(x) = ||x||^2
    x_star = gradient_descent(grad, x0=[1.0, -1.0])
    print("Approximate minimizer:", x_star)
          
