import numpy as np
import sympy as sp
from gcsg import gcsg

def true_gradient(fun_str, x0):
    n = len(x0)
    x_sym = sp.symbols(f"x0:{n}")
    f_expr = sp.sympify(fun_str)
    grad_sym = [sp.diff(f_expr, var) for var in x_sym]
    grad_func = sp.lambdify(x_sym, grad_sym, "numpy")
    return np.array(grad_func(*x0), dtype=float)

def test_gcsg():
    print("Generalized Centered Simplex Gradient (GCSG) Tester")
    print("--------------------------------------------------")
    
    fun_str = input("Enter function f(x0, x1, ...) (e.g., 'x0**2 + 3*x1**2'): ").strip()
    n = int(input("Enter dimension n: "))
    x_sym = sp.symbols(f"x0:{n}")
    f_expr = sp.sympify(fun_str)
    fun = sp.lambdify(x_sym, f_expr, "numpy")

    x0 = np.array(list(map(float, input(f"Enter x0 ({n} values): ").split())), dtype=float)

    m = int(input("Enter number of directions m: "))
    print(f"Enter T as a {n}x{m} matrix (row by row):")
    T_rows = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        if len(row) != m:
            raise ValueError(f"Expected {m} values in row {i+1}")
        T_rows.append(row)
    T = np.array(T_rows, dtype=float)

    h_vals = list(map(float, input("Enter step sizes h (space-separated): ").split()))
    grad_true = true_gradient(fun_str, x0)

    print("\nResults:")
    print("-" * 90)
    print("h\t\tApprox Gradient\t\tTrue Gradient\t\tAbs Error\tRel Error")
    print("-" * 90)

    for h in h_vals:
        try:
            grad_approx = gcsg(fun, x0, T, h)
            abs_err = np.linalg.norm(grad_approx - grad_true)
            rel_err = abs_err / np.linalg.norm(grad_true) if np.linalg.norm(grad_true) > 1e-16 else float("inf")
            print(f"{h:.1e}\t{grad_approx.tolist()}\t{grad_true.tolist()}\t{abs_err:.5e}\t{rel_err:.5e}")
        except Exception as e:
            print(f"Error for h={h:.1e}: {e}")

if __name__ == "__main__":
    test_gcsg()
