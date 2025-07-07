import numpy as np

def gsg_from_func(fun, x0, S):
    """
    Compute the Generalized Simplex Gradient (GSG) using a function.
    fun: callable, function from R^n to R
    x0: numpy array, point in R^n
    S: numpy array, n x m direction matrix
    Returns: gradient estimate (numpy array, shape n)
    """
    x0 = np.asarray(x0, dtype=float)
    S = np.asarray(S, dtype=float)
    n, m = S.shape
    v = [fun(*x0)]
    for i in range(m):
        v.append(fun(*(x0 + S[:, i])))
    v = np.array(v)
    delta_s = v[1:] - v[0]
    grad = np.linalg.pinv(S.T) @ delta_s
    return grad

def gsg_from_values(v, S):
    """
    Compute the Generalized Simplex Gradient (GSG) using only function values.
    v: array-like, shape (m+1,)
        v[0] = f(x0), v[1] = f(x0 + s1), ..., v[m] = f(x0 + sm)
    S: numpy array, shape (n, m)
        Columns are the direction vectors s1, ..., sm
    Returns: gradient estimate (numpy array, shape n)
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
    hess_func: function returning Hessian matrix at a point.
    Returns: estimated error bound (float)
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
    # Error bound (ignoring constants): (1/2) * max_quad / max_norm
    return 0.5 * max_quad / max_norm

def gsg_error_bound_lipschitz(S, L):
    """
    First-order (Lipschitz) error bound for GSG.
    S: direction matrix (n, m)
    L: Lipschitz constant for the gradient (user-supplied or estimated)
    Returns: error bound (float)
    """
    max_step = max(np.linalg.norm(S[:, i]) for i in range(S.shape[1]))
    return L * max_step

def estimate_lipschitz_from_gradients(x0, S, grad_func):
    """
    Estimate the Lipschitz constant of the gradient numerically using finite differences of gradients.
    x0: base point (numpy array)
    S: direction matrix (n, m)
    grad_func: function returning gradient at a point (callable)
    Returns: estimated Lipschitz constant (float)
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
