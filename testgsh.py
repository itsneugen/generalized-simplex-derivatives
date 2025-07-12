import numpy as np
import sympy as sp
import argparse
import sys
from gsh import gsh_from_func, gsh_from_values, gsh_error_bound

def run_interactive_mode():
    print("Interactive GSH Tester Mode\n")
    n = int(input("Enter number of variables (n): "))
    x0 = np.array(list(map(float, input(f"Enter x0 ({n} values, space-separated): ").split())))
    h = float(input("Enter step size h (e.g., 0.01): "))
    fun_str = input("Enter function in terms of x0, x1, ... (e.g., 'x0**2 * x1 + 3*x1**2'): ")
    x_syms = sp.symbols(f'x0:{n}')
    f_expr = sp.sympify(fun_str)
    f_func = sp.lambdify(x_syms, f_expr, "numpy")

    S = np.eye(n)
    hess = gsh_from_func(f_func, x0, S, h)
    print("\nApproximate Hessian (GSH):", hess)

    hess_expr = sp.hessian(f_expr, x_syms)
    hess_func = sp.lambdify(x_syms, hess_expr, "numpy")
    hess_true = np.array(hess_func(*x0), dtype=float)
    print("True Hessian:", hess_true)

    abs_error = np.abs(hess - hess_true)
    print("Max Absolute Error:", np.max(abs_error))

    L = float(input("Enter estimated Lipschitz constant L (or guess): "))
    bound = gsh_error_bound(n, h, L)
    print("Lipschitz-based error bound:", bound)

def parse_matrix(matrix_str):
    rows = matrix_str.split(';')
    return np.array([list(map(float, row.strip().split())) for row in rows])

def main():
    if "--help" in sys.argv:
        print("""
Usage: python testgsh.py [OPTIONS]

Options:
  --x0          Base point x0, space-separated (e.g., --x0 1 2)
  --function    Function in terms of x0, x1, ... (e.g., "x0**2 * x1 + 3*x1**2")
  --h           Step size (e.g., 0.01)
  --S           Direction matrix S in row format (e.g., "0.01 0; 0 0.01")
  --values      Function values: f(x0), f(x0+s1), ..., f(x0+si+sj) (e.g., "14.0, 14.0201, 14.1303, 14.170601")
  --manual      Use function-value mode instead of symbolic
  --interactive Run with prompts (for beginners)
  --help        Show this help message and exit

Manual Mode Hint:
  - --S "r1c1 r1c2 ...; r2c1 r2c2 ...; ..." defines the direction matrix S (n x m).
  - --values "v0,v1,...,vm,v(m+1),..." provides values: v0 = f(x0), v1 = f(x0 + S[:,0]), ..., vm = f(x0 + S[:,m-1]), followed by f(x0 + S[:,i] + S[:,j]) for all i ≤ j.
  Example for f(x0, x1) = x0^2 * x1 + 3*x1^2 at x0 = [1, 2], h = 0.01:
    python testgsh.py --manual --x0 1 2 --S "0.01 0; 0 0.01" --values "14.0, 14.0201, 14.1303, 14.170601"
""")
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--x0", nargs="*", type=float)
    parser.add_argument("--function", type=str)
    parser.add_argument("--h", type=float)
    parser.add_argument("--S", type=str)
    parser.add_argument("--values", type=str)
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_mode()
        return

    if args.manual:
        print("\nUsing function values only:")
        x0 = np.array(args.x0)
        S = parse_matrix(args.S)
        v = np.array(list(map(float, args.values.split(","))))
        print("S matrix:\n", S)
        print("v values:", v)
        hess = gsh_from_values(v, S)
        print("Approximate Hessian (GSH):", hess)

    elif args.function and args.x0 and args.h:
        print("\nUsing symbolic function and standard basis:")
        x0 = np.array(args.x0)
        h = args.h
        n = len(x0)
        x_syms = sp.symbols(f'x0:{n}')
        f_expr = sp.sympify(args.function)
        f_func = sp.lambdify(x_syms, f_expr, "numpy")
        S = np.eye(n)  # Normalized

        hess = gsh_from_func(f_func, x0, S, h)
        print("S matrix:\n", S)
        print("Approximate Hessian (GSH):", hess)

        hess_expr = sp.hessian(f_expr, x_syms)
        hess_func = sp.lambdify(x_syms, hess_expr, "numpy")
        hess_true = np.array(hess_func(*x0), dtype=float)
        print("True Hessian:", hess_true)

        abs_error = np.abs(hess - hess_true)
        print("Max Absolute Error:", np.max(abs_error))

        L = np.max(np.abs(np.linalg.eigvals(hess_true)))
        print("Estimated Lipschitz constant L:", L)
        bound = gsh_error_bound(n, h, L)
        print("Lipschitz-based error bound:", bound)

    else:
        print("\nInvalid or incomplete arguments. Run with --help for usage.")

if __name__ == "__main__":
    main()