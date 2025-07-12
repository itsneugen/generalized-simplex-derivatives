import numpy as np
import sympy as sp

def gst_from_func(fun, x0, S, T, U, h=0.01):
    """
    Compute the Generalized Simplex Tressian (GST) from a function.

    Parameters:
        fun : callable
            Scalar function f: R^n -> R
        x0 : ndarray (n,)
            Base point
        S, T, U : ndarray (n, m), (n, k), (n, l)
            Normalized direction matrices (columns have norm 1)
        h : float, optional
            Step size (default 0.01)
    Returns:
        Tressian approximation: ndarray (n, n, n)
    """
    x0 = np.asarray(x0, dtype=float)
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    U = np.asarray(U, dtype=float)
    
    m, k, l = S.shape[1], T.shape[1], U.shape[1]
    
    # Build delta tensor
    delta = np.zeros((m, k, l))
    for i in range(m):
        for j in range(k):
            for r in range(l):
                f111 = fun(*(x0 + h * S[:, i] + h * T[:, j] + h * U[:, r]))
                f100 = fun(*(x0 + h * S[:, i]))
                f010 = fun(*(x0 + h * T[:, j]))
                f001 = fun(*(x0 + h * U[:, r]))
                f110 = fun(*(x0 + h * S[:, i] + h * T[:, j]))
                f101 = fun(*(x0 + h * S[:, i] + h * U[:, r]))
                f011 = fun(*(x0 + h * T[:, j] + h * U[:, r]))
                f000 = fun(*x0)
                
                delta[i, j, r] = (
                    f111 - f110 - f101 - f011 + f100 + f010 + f001 - f000
                ) / (h**3)

    S_pinv = np.linalg.pinv(S.T)
    T_pinv = np.linalg.pinv(T.T)
    U_pinv = np.linalg.pinv(U.T)

    Tressian = np.einsum('ia,jb,kc,ijk->abc', S_pinv, T_pinv, U_pinv, delta)
    return Tressian

def gst_from_values(v, S, T, U):
    """
    Compute the GST using pre-evaluated function values.

    Parameters:
        v : ndarray of shape (m+1, k+1, l+1)
            v[i,j,r] = f(x0 + s_i + t_j + u_r), with appropriate special cases
        S, T, U : ndarray (n, m), (n, k), (n, l)
            Direction matrices
    Returns:
        Tressian estimate: ndarray (n, n, n)
    """
    v = np.asarray(v, dtype=float)
    m, k, l = S.shape[1], T.shape[1], U.shape[1]

    delta = (
        v[1:,1:,1:] - v[1:,1:,0:1] - v[1:,0:1,1:] - v[0:1,1:,1:]
        + v[1:,0:1,0:1] + v[0:1,1:,0:1] + v[0:1,0:1,1:] - v[0,0,0]
    )

    S_pinv = np.linalg.pinv(S.T)
    T_pinv = np.linalg.pinv(T.T)
    U_pinv = np.linalg.pinv(U.T)

    Tressian = np.einsum('ia,jb,kc,ijk->abc', S_pinv, T_pinv, U_pinv, delta)
    return Tressian

def gst_error_bound(m, k, l, L_tress, h):
    """
    Error bound for Generalized Simplex Tressian (GST).
    Parameters:
        m, k, l : int
            Number of columns in S, T, U
        L_tress : float
            Lipschitz constant of the Tressian
        h : float
            Step size
    Returns:
        bound : float
            Error bound
    """
    return (np.sqrt(m * k * l) / 2) * L_tress * h

def estimate_lipschitz_tressian_from_symbolic(x_syms, f_expr):
    """
    Estimate the Lipschitz constant of the third derivative (Tressian) using max of 4th-order derivatives.
    """
    n = len(x_syms)
    fourth_derivs = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    d4 = sp.diff(f_expr, x_syms[i], x_syms[j], x_syms[k], x_syms[l])
                    fourth_derivs.append(d4)

    def lipschitz_func(*x0):
        vals = [abs(d4.evalf(subs=dict(zip(x_syms, x0)))) for d4 in fourth_derivs]
        return max(vals)
    
    return lipschitz_func