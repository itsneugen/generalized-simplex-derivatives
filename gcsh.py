import numpy as np
import sympy as sp

def gsg(fun, x0, T, h=1.0):
    x0 = np.array(x0, dtype=float).ravel()
    T = np.array(T, dtype=float)
    n, m = T.shape
    f_val = fun(*x0)
    if np.size(f_val) != 1:
        raise ValueError("Function must return a scalar value at x0")
    T_h = h * T
    delta_f = np.zeros(m)
    for i in range(m):
        point = x0 + T_h[:, i]
        fi = fun(*point)
        if np.size(fi) != 1:
            raise ValueError("Function must return a scalar at each direction")
        delta_f[i] = float(fi) - float(f_val)
    return np.linalg.pinv(T_h).T @ delta_f

def gsh(fun, x0, S, Ti, h1=1.0, h2=1.0):
    x0 = np.array(x0, dtype=float).ravel()
    S = np.array(S, dtype=float)
    n, m = S.shape
    Ti_h2 = [h2 * np.array(T, dtype=float) for T in Ti] if isinstance(Ti, list) else h2 * np.array(Ti, dtype=float)
    S_h1 = h1 * S

    if isinstance(Ti_h2, list):
        sgMat = [ [gsg(fun, x0 + S_h1[:, i], Ti_h2[i]), gsg(fun, x0, Ti_h2[i])] for i in range(m) ]
        delta_s = np.array([ (a - b) for a, b in sgMat ])
    else:
        sgX0T = gsg(fun, x0, Ti_h2)
        sgMat = np.column_stack([ gsg(fun, x0 + S_h1[:, i], Ti_h2) for i in range(m) ])
        delta_s = sgMat.T - sgX0T

    SHessValue = np.linalg.pinv(S_h1).T @ delta_s
    return SHessValue

def gcsh(fun, x0, S, Ti, h1=1.0, h2=1.0):
    Ti_neg = [ -1 * np.array(T, dtype=float) for T in Ti ] if isinstance(Ti, list) else -1 * np.array(Ti, dtype=float)
    SHessPlus = gsh(fun, x0, S, Ti, h1, h2)
    SHessMinus = gsh(fun, x0, -1 * S, Ti_neg, h1, h2)
    return 0.5 * (SHessPlus + SHessMinus)
