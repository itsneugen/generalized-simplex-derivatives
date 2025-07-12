import numpy as np

def gcsg(fun, x0, T, h=1.0):
    """
    Generalized Centered Simplex Gradient (GCSG) approximation.

    Parameters:
        fun: callable, function from R^n to R, accepts a 1D numpy array
        x0: numpy array, point in R^n (1D)
        T: numpy array, n x m matrix of direction vectors (columns)
        h: float, step size (default=1.0)

    Returns:
        Approximate gradient at x0 as a numpy array of length n.
    """
    x0 = np.array(x0, dtype=float).ravel()
    T = np.array(T, dtype=float)
    n, m = T.shape

    if len(x0) != n:
        raise ValueError("T and x0 must have same number of rows")

    T_h = h * T
    delta_f = np.zeros(m)

    for i in range(m):
        f_forward  = fun(*(x0 + T_h[:, i]))
        f_backward = fun(*(x0 - T_h[:, i]))
        delta_f[i] = (f_forward - f_backward) / 2.0

    grad_approx = np.linalg.pinv(T_h).T @ delta_f
    return grad_approx
