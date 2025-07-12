import numpy as np
import sympy as sp
import argparse
import sys
from gsg import gsg_from_func, gsg_from_values, gsg_error_bound, gsg_error_bound_lipschitz

def run_interactive_mode():
    print("Interactive GSG Tester Mode\n")
    n = int(input("Enter number of variables (n): "))
    x0 = np.array(list(map(float, input(f"Enter x0 ({n} values, space-separated): ").split())))
    h = float(input("Enter step size h (e.g., 0.01): "))
    fun_str = input("Enter function in terms of x0, x1, ... (e.g., 'x0**2 + 3*x1**2'): ")
    x_syms = sp.symbols(f'x0:{n}')
    f_expr = sp.sympify(fun_str)
    f_func = sp.lambdify(x_syms, f_expr, "numpy")

    S = np.eye(n)
    grad = gsg_from_func(f_func, x0, S, h)
    print("\nApproximate gradient (GSG):", grad)

    grad_expr = [sp.diff(f_expr, var) for var in x_syms]
    grad_func = sp.lambdify(x_syms, grad_expr, "numpy")
    grad_true = np.array(grad_func(*x0), dtype=float)
    print("True gradient:", grad_true)

    abs_error = np.abs(grad - grad_true)
    print("Max Absolute Error:", np.max(abs_error))

    L = float(input("Enter estimated Lipschitz constant L (or guess): "))
    bound = gsg_error_bound_lipschitz(n, h, L)
    print("Lipschitz-based error bound:", bound)

def parse_matrix(matrix_str):
    rows = matrix_str.split(';')
    return np.array([list(map(float, row.strip().split())) for row in rows])

def main():
    if "--help" in sys.argv:
        print("""
Usage: python testgsg.py [OPTIONS]

Options:
  --x0          Base point x0, space-separated (e.g., --x0 1 2)
  --function    Function in terms of x0, x1, ... (e.g., "x0**2 + 3*x1**2")
  --h           Step size (e.g., 0.01)
  --S           Direction matrix S in row format (e.g., "0.01 0; 0 0.01")
  --values      Function values: f(x0), f(x0+s1), ..., (e.g., "14.0, 14.0201, 14.1303")
  --manual      Use function-value mode instead of symbolic
  --interactive Run with prompts (for beginners)
  --help        Show this help message and exit

Manual Mode Hint:
  - --S "r1c1 r1c2 ...; r2c1 r2c2 ...; ..." defines the direction matrix S (n x m).
  - --values "v0,v1,...,vm" provides m+1 values: v0 = f(x0), v1 = f(x0 + S[:,0]), ..., vm = f(x0 + S[:,m-1]).
  Example for f(x0, x1) = x0^2 * x1 + 3*x1^2 at x0 = [1, 2], h = 0.01:
    python testgsg.py --manual --x0 1 2 --S "0.01 0; 0 0.01" --values "14.0, 14.0201, 14.1303"

Examples:
  python testgsg.py --x0 1 2 --function "x0**2 * x1 + 3*x1**2" --h 0.01
  python testgsg.py --manual --x0 1 2 --S "0.01 0; 0 0.01" --values "14.0, 14.0201, 14.1303"
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
        grad = gsg_from_values(v, S)
        print("Approximate gradient (GSG):", grad)

    elif args.function and args.x0 and args.h:
        print("\nUsing symbolic function and standard basis:")
        x0 = np.array(args.x0)
        h = args.h
        n = len(x0)
        x_syms = sp.symbols(f'x0:{n}')
        f_expr = sp.sympify(args.function)
        f_func = sp.lambdify(x_syms, f_expr, "numpy")
        S = np.eye(n)  # Normalized

        grad = gsg_from_func(f_func, x0, S, h)
        print("S matrix:\n", S)
        print("Approximate gradient (GSG):", grad)

        grad_expr = [sp.diff(f_expr, var) for var in x_syms]
        grad_func = sp.lambdify(x_syms, grad_expr, "numpy")
        grad_true = np.array(grad_func(*x0), dtype=float)
        print("True gradient:", grad_true)

        abs_error = np.abs(grad - grad_true)
        print("Max Absolute Error:", np.max(abs_error))

        H = sp.hessian(f_expr, x_syms)
        hess_func = sp.lambdify(x_syms, H, "numpy")
        H_at_x0 = np.array(hess_func(*x0), dtype=float)
        lipschitz_L = np.max(np.abs(np.linalg.eigvals(H_at_x0)))
        print("Estimated Lipschitz constant L:", lipschitz_L)

        bound = gsg_error_bound_lipschitz(n, h, lipschitz_L)
        print("Lipschitz-based error bound:", bound)

    else:
        print("\nInvalid or incomplete arguments. Run with --help for usage.")

if __name__ == "__main__":
    main()