import numpy as np
import sympy as sp
import argparse
from tres import gst_from_func, gst_error_bound, estimate_lipschitz_tressian_from_symbolic

def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate third-order derivative tensor (Tressian) using Generalized Simplex method."
    )
    parser.add_argument("--x0", nargs="+", type=float, help="Base point x0 (space-separated list)")
    parser.add_argument("--function", type=str, help="Function expression in terms of x0, x1, ...")
    parser.add_argument("--h", type=float, help="Step size h")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--manual", action="store_true", help="Use manual value-only mode")
    return parser.parse_args()

def run_symbolic_mode(x0, f_expr_str, h):
    n = len(x0)
    x_syms = sp.symbols(f"x0:{n}")
    f_expr = sp.sympify(f_expr_str)
    f_func = sp.lambdify(x_syms, f_expr, "numpy")

    # Normalized directions
    S = np.eye(n)
    T = np.eye(n)
    U = np.eye(n)

    print("\nNormalized S, T, U direction matrices:\n", S)

    T_est = gst_from_func(f_func, x0, S, T, U, h)
    print("\nEstimated third-order tensor (Tressian):\n", T_est)

    # Compare with symbolic
    third_derivs = np.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                d3 = sp.diff(f_expr, x_syms[i], x_syms[j], x_syms[k])
                third_derivs[i, j, k] = float(d3.evalf(subs=dict(zip(x_syms, x0))))
    print("\nTrue Tressian at x0:\n", third_derivs)
    print("\nAbsolute error tensor:\n", np.abs(T_est - third_derivs))

    # Lipschitz bound
    lipschitz_func = estimate_lipschitz_tressian_from_symbolic(x_syms, f_expr)
    L_tress = lipschitz_func(*x0)
    print("\nEstimated Lipschitz constant for Tressian at x0:", L_tress)
    bound = gst_error_bound(n, n, n, L_tress, h)
    print("Tressian error bound (auto-estimated L):", bound)

def run_interactive():
    print("\nWelcome to interactive Tressian estimation mode.")
    n = int(input("Enter number of variables: "))
    x0 = np.array(list(map(float, input("Enter base point x0 (space-separated): ").split())))
    h = float(input("Enter step size h (e.g., 0.01): "))
    f_expr = input("Enter the function (in terms of x0, x1, ...): ")
    run_symbolic_mode(x0, f_expr, h)

def main():
    args = parse_args()

    if args.interactive:
        run_interactive()
        return

    if args.x0 and args.function and args.h:
        x0 = np.array(args.x0)
        run_symbolic_mode(x0, args.function, args.h)
    else:
        print("\n[!] Missing required arguments. Use --interactive for guided mode or see --help for CLI usage.")

if __name__ == "__main__":
    main()