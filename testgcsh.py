import numpy as np
import sympy as sp
from gcsh import gcsh
from gsh import gsh  # reuse if needed for symbolic comparison

def true_hessian(fun_str, x0):
    n = len(x0)
    x_sym = sp.symbols(f"x0:{n}")
    f_expr = sp.sympify(fun_str)
    hessian = sp.hessian(f_expr, x_sym)
    h_func = sp.lambdify(x_sym, hessian, 'numpy')
    return np.array(h_func(*x0), dtype=float)

def test_gcsh():
    print("Generalized Centered Simplex Hessian (GCSH) Tester")
    fun_str = input("Enter function f(x0, x1, ...) (e.g., 'x0**2 + 3*x1**2'): ").strip()
    n = int(input("Enter dimension n: "))
    x0 = np.array(list(map(float, input(f"Enter x0 ({n} values): ").split())))
    m = int(input("Enter number of directions in S (columns): "))
    print(f"Enter S ({n}x{m}):")
    S = np.array([list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)])
    k = int(input("Enter number of directions in Ti (columns): "))
    print(f"Enter Ti ({n}x{k}):")
    Ti = np.array([list(map(float, input(f"Row {i+1}: ").split())) for i in range(n)])
    h_values = list(map(float, input("Enter step sizes h (space-separated): ").split()))

    x_sym = sp.symbols(f"x0:{n}")
    f_expr = sp.sympify(fun_str)
    fun = sp.lambdify(x_sym, f_expr, 'numpy')

    true_H = true_hessian(fun_str, x0)

    print("\nResults:")
    print("-" * 90)
    print(f"{'h':<10}{'Approx Hessian':<25}{'True Hessian':<25}{'AbsErr':<12}{'RelErr':<12}")
    print("-" * 90)

    for h in h_values:
        H_approx = gcsh(fun, x0, S, Ti, h, h)
        abs_err = np.linalg.norm(H_approx - true_H)
        rel_err = abs_err / np.linalg.norm(true_H) if np.linalg.norm(true_H) > 1e-16 else float("inf")
        print(f"{h:<10.2e}{H_approx.round(3).tolist()!s:<25}{true_H.round(3).tolist()!s:<25}{abs_err:<12.4e}{rel_err:<12.4e}")

if __name__ == "__main__":
    test_gcsh()
