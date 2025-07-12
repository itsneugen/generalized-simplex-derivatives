import numpy as np
import sympy as sp

def gsh_from_func(fun, x0, S, T, h=0.01):
    """
    Compute the Generalized Simplex Hessian (GSH) using a function.
    Parameters:
        fun : callable
            Function of n variables.
        x0 : ndarray (n,)
            Base point.
        S : ndarray (n, m)
            Normalized direction matrix for S (columns have norm 1).
        T : ndarray (n, k)
            Normalized direction matrix for T (columns have norm 1).
        h : float, optional
            Step size (default 0.01).
    Returns:
        H_approx : ndarray (n, n)
            Approximated Hessian matrix.
    """
    x0 = np.asarray(x0, dtype=float)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    n, m = S.shape
    _, k = T.shape
    delta = np.zeros((m, k))

    for i in range(m):
        for j in range(k):
            f1 = fun(*(x0 + h * S[:, i] + h * T[:, j]))
            f2 = fun(*(x0 + h * S[:, i]))
            f3 = fun(*(x0 + h * T[:, j]))
            f4 = fun(*x0)
            delta[i, j] = (f1 - f2 - f3 + f4) / (h**2)

    S_pinv = np.linalg.pinv(S.T)
    T_pinv = np.linalg.pinv(T.T)
    H_approx = S_pinv @ delta @ T_pinv.T
    return H_approx

def gsh_from_values(v, S, T):
    """
    Compute the GSH using only function values.
    Parameters:
        v : ndarray (m+1, k+1)
            v[0,0] = f(x0)
            v[i,0] = f(x0 + s_i)
            v[0,j] = f(x0 + t_j)
            v[i,j] = f(x0 + s_i + t_j)
        S : ndarray (n, m)
            Direction matrix S.
        T : ndarray (n, k)
            Direction matrix T.
    Returns:
        H_approx : ndarray (n, n)
            Approximated Hessian matrix.
    """
    v = np.asarray(v, dtype=float)
    delta = v[1:,1:] - v[1:,0:1] - v[0:1,1:] + v[0,0]
    S_pinv = np.linalg.pinv(S.T)
    T_pinv = np.linalg.pinv(T.T)
    H_approx = S_pinv @ delta @ T_pinv.T
    return H_approx

def gsh_error_bound(m, k, L_hess, h):
    """
    Estimate error bound for the GSH method.
    Parameters:
        m : int
            Number of columns in S.
        k : int
            Number of columns in T.
        L_hess : float
            Estimated Lipschitz constant of Hessian.
        h : float
            Step size (used as delta_u).
    Returns:
        bound : float
            Error bound.
    """
    bound = 4 * np.sqrt(m * k) * L_hess * h
    return bound

def estimate_lipschitz_hessian_from_symbolic(x_syms, f_expr):
    """
    Estimate Lipschitz constant of the Hessian via third-order symbolic derivatives.
    Parameters:
        x_syms : list of sympy symbols
        f_expr : sympy expression
    Returns:
        lipschitz_func : callable
            Evaluates estimated Lipschitz constant at a given point.
    """
    n = len(x_syms)
    third_derivs = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                d3 = sp.diff(f_expr, x_syms[i], x_syms[j], x_syms[k])
                third_derivs.append(d3)

    def lipschitz_func(*x0):
        vals = [abs(d3.evalf(subs=dict(zip(x_syms, x0)))) for d3 in third_derivs]
        return max(vals)

    return lipschitz_func