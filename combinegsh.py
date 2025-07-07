import numpy as np

def gsh_from_func(fun, x0, S, T):
    """
    Compute the Generalized Simplex Hessian (GSH) using a function.
    """
    x0 = np.asarray(x0, dtype=float)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    n, m = S.shape
    _, k = T.shape
    delta = np.zeros((m, k))
    for i in range(m):
        for j in range(k):
            f1 = fun(*(x0 + S[:, i] + T[:, j]))
            f2 = fun(*(x0 + S[:, i]))
            f3 = fun(*(x0 + T[:, j]))
            f4 = fun(*x0)
            delta[i, j] = f1 - f2 - f3 + f4
    S_pinv = np.linalg.pinv(S.T)
    T_pinv = np.linalg.pinv(T.T)
    H_approx = S_pinv @ delta @ T_pinv.T
    return H_approx

def gsh_from_values(v, S, T):
    """
    Compute the GSH using only function values.
    v: array-like, shape (m+1, k+1)
        v[0,0] = f(x0)
        v[i,0] = f(x0 + s_i)
        v[0,j] = f(x0 + t_j)
        v[i,j] = f(x0 + s_i + t_j)
    S: (n, m), T: (n, k)
    Returns: Hessian estimate (n x n)
    """
    v = np.asarray(v, dtype=float)
    m = S.shape[1]
    k = T.shape[1]
    delta = v[1:,1:] - v[1:,0:1] - v[0:1,1:] + v[0,0]
    S_pinv = np.linalg.pinv(S.T)
    T_pinv = np.linalg.pinv(T.T)
    H_approx = S_pinv @ delta @ T_pinv.T
    return H_approx

def gsh_error_bound(S, T, L_hess):
    """
    Error bound for GSH (special case T1=...=Tm=T).
    """
    m = S.shape[1]
    k = T.shape[1]
    delta_S = max(np.linalg.norm(S[:, i]) for i in range(m))
    delta_T = max(np.linalg.norm(T[:, j]) for j in range(k))
    delta_u = max(delta_S, delta_T)
    delta_l = min(delta_S, delta_T)
    S_pinv_norm = np.linalg.norm(np.linalg.pinv(S.T), 2)
    T_pinv_norm = np.linalg.norm(np.linalg.pinv(T.T), 2)
    bound = 4 * np.sqrt(m * k) * L_hess * (delta_u / delta_l) * S_pinv_norm * T_pinv_norm * delta_u
    return bound

def estimate_lipschitz_hessian_from_symbolic(x_syms, f_expr):
    """
    Estimate the Lipschitz constant of the Hessian by taking the max norm of the third derivatives at x0.
    """
    import sympy as sp
    n = len(x_syms)
    third_derivs = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                d3 = sp.diff(f_expr, x_syms[i], x_syms[j], x_syms[k])
                third_derivs.append(d3)
    # For a rough estimate, take the sum of absolute values at x0
    def lipschitz_func(*x0):
        vals = [abs(d3.evalf(subs=dict(zip(x_syms, x0)))) for d3 in third_derivs]
        return max(vals)
    return lipschitz_func