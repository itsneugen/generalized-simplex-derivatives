import numpy as np
import sympy as sp

def gsg_from_func(fun, x0, S, h=0.01):
    """
    Compute the Generalized Simplex Gradient (GSG) using a function.
    Parameters:
        fun : callable
            Function from R^n to R.
        x0 : numpy array
            Point in R^n.
        S : numpy array, shape (n, m)
            Normalized direction matrix (columns have norm 1).
        h : float, optional
            Step size (default 0.01).
    Returns:
        grad : numpy array, shape (n,)
            Gradient estimate.
    """
    x0 = np.asarray(x0, dtype=float)
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    v = [fun(*x0)]
    for i in range(m):
        v.append(fun(*(x0 + h * S[:, i])))
    v = np.array(v)
    delta_s = (v[1:] - v[0]) / h
    grad = np.linalg.pinv(S.T) @ delta_s
    return grad

def gsg_from_values(v, S):
    """
    Compute the Generalized Simplex Gradient (GSG) using only function values.
    Parameters:
        v : array-like, shape (m+1,)
            v[0] = f(x0), v[1] = f(x0 + s1), ..., v[m] = f(x0 + sm).
        S : numpy array, shape (n, m)
            Columns are the direction vectors s1, ..., sm.
    Returns:
        grad : numpy array, shape (n,)
            Gradient estimate.
    """
    v = np.asarray(v, dtype=float)
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    if v.shape[0] != m + 1:
        raise ValueError(f"v must have length m+1 (got {v.shape[0]}, expected {m+1})")
    delta_s = v[1:] - v[0]
    grad = np.linalg.pinv(S.T) @ delta_s
    return grad

def gsg_error_bound(x0, S, hess_func):
    """
    Estimate the error bound for the GSG at x0 using directions S and a Hessian function.
    Parameters:
        x0 : numpy array
            Point in R^n.
        S : numpy array, shape (n, m)
            Direction matrix.
        hess_func : callable
            Function returning Hessian matrix at a point.
    Returns:
        bound : float
            Estimated error bound.
    """
    x0 = np.asarray(x0, dtype=float)
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    H = np.array(hess_func(*x0))  # shape (n, n)
    quad_forms = []
    for i in range(m):
        s = S[:, i]
        q = np.abs(s @ H @ s)
        quad_forms.append(q)
    max_quad = max(quad_forms)
    max_norm = max(np.linalg.norm(S[:, i]) for i in range(m))
    return 0.5 * max_quad / max_norm

def gsg_error_bound_lipschitz(m, h, L):
    """
    First-order (Lipschitz) error bound for GSG, per Theorem 4.10.
    Parameters:
        m : int
            Number of columns in S.
        h : float
            Step size (used as delta_u).
        L : float
            Lipschitz constant for the gradient.
    Returns:
        bound : float
            Error bound.
    """
    return (np.sqrt(m) / 2) * L * h

def estimate_lipschitz_from_gradients(x0, S, grad_func):
    """
    Estimate the Lipschitz constant of the gradient numerically using finite differences of gradients.
    Parameters:
        x0 : numpy array
            Base point.
        S : numpy array, shape (n, m)
            Direction matrix.
        grad_func : callable
            Function returning gradient at a point.
    Returns:
        L : float
            Estimated Lipschitz constant.
    """
    x0 = np.asarray(x0, dtype=float)
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    grad0 = np.array(grad_func(*x0))
    lipschitz_estimates = []
    for i in range(m):
        x1 = x0 + S[:, i]
        gradi = np.array(grad_func(*x1))
        diff = np.linalg.norm(gradi - grad0)
        step = np.linalg.norm(S[:, i])
        if step > 0:
            lipschitz_estimates.append(diff / step)
    return max(lipschitz_estimates) if lipschitz_estimates else 0.0