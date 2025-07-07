# generalized-simplex-derivatives
Python implementation of Generalized Simplex Gradient, Hessian, and generalized estimators with error bounds
#  Generalized Simplex Derivative Estimation and Error Bounds

## Project Summary

This project implements and tests **derivative-free approximations** for gradients, Hessians, and third-order tensors using **generalized simplex methods** (GSG, GSH, GST), including:

-  Accurate estimation of order‑p derivatives from function values
-  Symbolic validation via SymPy
-  Automatic or manual direction matrices
-  Analytical Lipschitz-based error bounds
-  Flexible interfaces (CLI, function-based, manual)

This work is part of a **research project under Prof. Gabriel Jarry-Bolduc** at Mount Royal University (MRU), focusing on *black-box optimization and error-bound analysis*.

---

##  Methods Implemented

| Method | Description |
|--------|-------------|
| **GSG** | Generalized Simplex Gradient (∇f) |
| **GSH** | Generalized Simplex Hessian (∇²f) |
| **GST** | Generalized Simplex Tressian (∇³f) |
| **Generalized** | Recursive simplex derivative up to any order-p |
| **Error Bounds** | Lipschitz-based theoretical bounds for each approximation |

---


```

---

##  How to Use

###  Option 1: Run in One Line (GSG/GSH/GST/generalized)
```bash
python testgen.py --expr "x0**2*x1 + 3*x1**2" --x0 1,2 --order 2 --h 0.01 --all
```
- Change the expression, point, or order as needed
- Displays numeric result, symbolic validation, and errors

###  Option 2: Modify Inputs in File
Use `run_example.py` to:
- Set your function, input point, directions, and step sizes
- Run once to see full output

###  Option 3: Interactive CLI for GSH
```bash
python testcombinedgsh.py
```
- Choose between symbolic function mode or raw values
- Optionally compare with true Hessian and error bound

---

##  Example Output

```
Approximate Hessian at x0 (GSH):
 [[4. 3.]
  [3. 6.]]
True Hessian at x0:
 [[4. 2.]
  [2. 6.]]
Error:
 [[0. 1.]
  [1. 0.]]
GSH error bound: 16.0
```

---

##  Notes

- All matrices **S**, **T** used for direction vectors are **normalized** before derivative computation to tighten error bounds.
- Symbolic derivatives and Lipschitz constants are automatically computed using **SymPy**.

---

##  Dependencies

- Python 3.8+
- `numpy`
- `sympy`

Install via:
```bash
pip install numpy sympy
```

---

##  License

MIT License – see [LICENSE](./LICENSE) for full details.

---

##  Authors

- **Gaurav [Lead Student Researcher]**
- **Prof. Gabriel Jarry-Bolduc [Supervisor]**
