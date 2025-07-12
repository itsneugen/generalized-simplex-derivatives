import numpy as np


def generate_simplex_derivative(f, x0, S_list, h_list):
    """
    Compute simplex derivatives up to order P.
    Returns a dict: order -> derivative tensor.
    """
    P = len(S_list)
    layers = {}
    for p in range(1, P + 1):
        layers[p] = _simplex_derivative_order_p(f, x0, S_list[:p], h_list[:p])
    return layers


def _simplex_derivative_order_p(f, x0, S_sub, h_sub):
    p = len(S_sub)
    if p == 1:
        return _gsg(f, x0, S_sub[0], h_sub[0])

    S1 = S_sub[0]
    h1 = h_sub[0]
    base = _simplex_derivative_order_p(f, x0, S_sub[1:], h_sub[1:])

    deltas = []
    for j in range(S1.shape[1]):
        xj = x0 + h1 * S1[:, j]
        lower = _simplex_derivative_order_p(f, xj, S_sub[1:], h_sub[1:])
        deltas.append((lower - base) / h1)  # <-- divide by h1 here!

    delta_arr = np.stack(deltas, axis=0)  # shape: (m1, ...)
    pinv = np.linalg.pinv(S1.T)  # shape: (n, m1)
    # Fold pinv into first axis of delta_arr
    return np.tensordot(pinv, delta_arr, axes=[1, 0])


def _gsg(f, x0, S1, h1):
    m1 = S1.shape[1]
    f0 = f(x0)
    vals = np.array([f(x0 + h1 * S1[:, j]) for j in range(m1)])
    diffs = (vals - f0) / h1
    pinv = np.linalg.pinv(S1.T)
    grad = pinv.dot(diffs)
    return grad


# Example usage:
# layers = generate_simplex_derivative(my_function, x0, [S1, S2, S3], [h1, h2, h3])
# grad = layers[1]; hess = layers[2]; tress = layers[3]
