import numpy as np
import sympy as sp
import itertools
import argparse
from gen import generate_simplex_derivative

def compute_symbolic_tensor(f_sym, variables, order):
    n = len(variables)
    tensor = np.empty((n,) * order, dtype=object)
    for index in itertools.product(range(n), repeat=order):
        expr = f_sym
        for axis in index:
            expr = sp.diff(expr, variables[axis])
        tensor[index] = expr
    return tensor

def evaluate_tensor(tensor, variables, point):
    flat_tensor = tensor.flatten().tolist()
    funcs = [sp.lambdify(variables, expr, 'numpy') for expr in flat_tensor]
    evaluated = np.array([f(*point) for f in funcs], dtype=float)
    return evaluated.reshape(tensor.shape)

def run_simplex_test(f_expr, x0, P, mode='auto', show_all=False, custom_layers=None, S_list=None, h=None, h_list=None):
    x0 = np.array(x0, dtype=float)
    n = len(x0)
    variables = sp.symbols([f'x{i}' for i in range(n)])

    local_dict = {str(v): v for v in variables}
    local_dict.update({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp})
    f_sym = sp.sympify(f_expr, locals=local_dict)
    f_lambda = sp.lambdify(variables, f_sym, 'numpy')
    f_numeric = lambda x: f_lambda(*x)

    if mode == 'auto':
        S_list = []
        h_list_normalized = []
        for _ in range(P):
            S = np.eye(n)
            delta = np.max(np.linalg.norm(S, axis=0))
            S_hat = S / delta
            S_list.append(S_hat)
            h_list_normalized.append(h / delta)  # Adjust step size accordingly
    elif mode == 'manual':
        if S_list is None:
            raise ValueError("In manual mode, S_list must be provided.")
        if h_list is not None:
            h_list = h_list
        elif h is not None:
            h_list = [h for _ in range(P)]
        else:
            raise ValueError("In manual mode, either h or h_list must be provided.")

    numeric_derivatives = generate_simplex_derivative(f_numeric, x0, S_list, h_list_normalized)

    if custom_layers:
        layers_to_show = custom_layers
    elif show_all:
        layers_to_show = list(range(1, P + 1))
    else:
        layers_to_show = [P]

    for order in layers_to_show:
        if order not in numeric_derivatives:
            print(f"\n[!] Order {order} not computed.")
            continue

        numeric_tensor = numeric_derivatives[order]
        symbolic_tensor = compute_symbolic_tensor(f_sym, variables, order)
        evaluated_tensor = evaluate_tensor(symbolic_tensor, variables, x0)
        abs_error = np.abs(numeric_tensor - evaluated_tensor)
        rel_error = abs_error / (np.abs(evaluated_tensor) + 1e-8)   #doublecheck, how we define norm for 3d and 4d

        print(f"\n--- Order {order} Derivative ---")
        print(f"Numeric Derivative (shape {numeric_tensor.shape}):\n{numeric_tensor}")
        print(f"Symbolic Derivative Evaluated:\n{evaluated_tensor}")
        print("Max Absolute Error:", np.max(abs_error))
        print("Max Relative Error:", np.max(rel_error))

def main():
    parser = argparse.ArgumentParser(description="Run generalized simplex derivative test.")
    parser.add_argument('--expr', type=str, required=True, help="Function expression in x0, x1, ...")
    parser.add_argument('--x0', type=str, required=True, help="Point as comma-separated values, e.g., 1,2")
    parser.add_argument('--order', type=int, required=True, help="P-th order of derivative to compute")
    parser.add_argument('--layers', type=str, default='', help="Comma-separated list of layers to show (e.g. 1,3,5)")
    parser.add_argument('--all', action='store_true', help="If set, show all layers from 1 to P")
    parser.add_argument('--mode', choices=['auto', 'manual'], default='auto', help="Direction mode")
    parser.add_argument('--h', type=float, default=1.0, help="Step size to use if h_list not provided")
    parser.add_argument('--hlist', type=str, help="Comma-separated list of step sizes for each order")
    args = parser.parse_args()

    f_expr = args.expr
    x0 = tuple(map(float, args.x0.split(',')))
    P = args.order
    custom_layers = list(map(int, args.layers.split(','))) if args.layers else None

    h_list = list(map(float, args.hlist.split(','))) if args.hlist else None

    run_simplex_test(
        f_expr=f_expr,
        x0=x0,
        P=P,
        mode=args.mode,
        show_all=args.all,
        custom_layers=custom_layers,
        h=args.h,
        h_list=h_list
    )

if __name__ == '__main__':
    main()