import numpy as np

def conjugate_gradient(A, b, x0=None, tol=1e-6, max_iter=None):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method.

    This implementation assumes that A is symmetric and positive definite.

    Parameters
    ----------
    A : ndarray or callable
        Symmetric positive definite matrix or linear operator.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess for the solution.
    tol : float, optional
        Convergence tolerance on the residual norm.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    x : ndarray
        Approximate solution to the system.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    if callable(A):
        r = b - A(x)
    else:
        r = b - A @ x

    p = r.copy()
    rs_old = np.dot(r, r)

    if max_iter is None:
        max_iter = n

    for _ in range(max_iter):
        if callable(A):
            Ap = A(p)
        else:
            Ap = A @ p

        alpha = rs_old / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.dot(r, r)
        if np.sqrt(rs_new) < tol:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


if __name__ == "__main__":
    A = np.array([[4.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    x_star = conjugate_gradient(A, b)
    print("Approximate solution:", x_star)
      
