# Generalized Simplex Derivative Approximations
*A research-driven toolkit for implementing and testing optimization methods across platforms.*




##  Overview

This repository contains a clean Python toolkit for estimating **gradients**, **Hessians**, and early prototypes of **third-order derivatives** using simplex-based methods.

It supports:
- Generalized Simplex Gradient (GSG)
- Generalized Simplex Hessian (GSH)
- Preliminary implementation of third-order (Tressian) estimation

The framework is designed for flexibility: it works with both symbolic functions and raw function evaluations, making it useful for a wide range of numerical analysis and optimization tasks.

Ideal for:
- Derivative-free optimization
- Numerical testing and benchmarking
- Symbolic vs numerical method comparisons
- Research and classroom use


This work is part of a **research project under Prof. Gabriel Jarry-Bolduc** at Mount Royal University (MRU), focusing on *black-box optimization and error-bound analysis*.


---
##  Project Structure

This repository contains Python and MATLAB implementations related to derivative-free approximation techniques, primarily focusing on estimating gradients, Hessians, and third-order tensors using Generalized Simplex methods.

```
├── gsg.py                # Core logic for Generalized Simplex Gradient (GSG)
├── gsh.py                # Core logic for Generalized Simplex Hessian (GSH)
├── tres.py               # Core logic for third-order Tressian estimation (GST)
│
├── testgsg.py            # CLI and interactive tester for GSG
├── testgsh.py            # CLI and interactive tester for GSH
├── testgst.py            # CLI and interactive tester for GST (Tressian)
│
├── run_example.py        # General-purpose script to test all variants (GSG, GSH, GST)
├── gen.py                # Early work on generalized higher-order simplex approximation(on progress)
├── testgen.py            # Testing harness for gen.py
├── gcsg.py               # Centered variant of GSG (optional/experimental)
├── gcsh.py               # Centered variant of GSH (optional/experimental)
│
├── matlab/               # MATLAB implementations (kept minimal but usable)

```

>  Note: While the spotlight is on GSG, GSH, and GST, other files are included for completeness and experimentation. MATLAB versions are also available for users who prefer that environment.

##  Features Overview

This repository provides Python tools for estimating derivatives of different orders (gradient, Hessian, third-order tensor) using finite difference schemes. Each tool is designed to be flexible, accurate, and beginner-friendly, with both command-line and interactive modes.

###  Available Estimators

| File         | Derivative Order | Description                        |
|--------------|------------------|------------------------------------|
| `testgsg.py` | First-order      | Estimates gradient (GSG method)    |
| `testgsh.py` | Second-order     | Estimates Hessian (GSH method)     |
| `testgst.py` | Third-order      | Estimates 3rd-order tensor (GST)   |

All three scripts share a **uniform interface**, so once you learn one, you can use all easily.

---

###  How to Run (Any of GSG / GSH / GST)

#### Command-Line Mode
Run script in one line:

```bash
python testgsg.py --x0 1 2 --function "x0**2 + 3*x1**2" --h 0.01
```

All will:
- Use `x0 = [1, 2]`
- Use the given function string
- Use standard basis (normalized)
- Compute and display the derivative (GSG, GSH, or GST)
- Compare with symbolic derivative (if possible)
- Show Lipschitz-based error bound (where supported)


####  Interactive Mode (Optional)
If a user prefers prompts:
```bash
python testgsh.py --interactive
```
It will guide you through input step-by-step.



####  Manual Function-Value Mode

Use `--manual` to compute GSG, GSH, or GST with pre-evaluated function values.

**Inputs**:
- `--x0`: Base point, e.g., `--x0 1 2` for \( x_0 = [1, 2] \).
- `--S`: Direction matrix \( S \) (and \( T, U = S \) for GST), e.g., `"0.01 0; 0 0.01"` for \( S = 0.01 \cdot I \).
- `--values`: Comma-separated values in order:

**Example** (for \( f(x_0, x_1) = x_0^2 x_1 + 3 x_1^2 \), \( x_0 = [1, 2] \), \( h = 0.01 \)):
- Compute: \( f(1, 2) = 14 \), \( f(1.01, 2) = 14.0201 \), \( f(1, 2.01) = 14.1303 \), \( f(1.01, 2.01) = 14.170601 \).


**Tip**: Compute values at \( x_0 + \) combinations of \( S[:,i] \). Check `--help` for details.


**Command**:
```bash
python testgsg.py --manual --x0 1 2 --S "0.01 0; 0 0.01" --values "14.0, 14.0201, 14.1303"
For black-box or experimental functions, provide values directly:
```bash
python testgsh.py --manual --x0 1 2 --S "0.01 0; 0 0.01" --values "7.0, 7.02, 7.12"
```
(Works for all three: `testgsg.py`, `testgsh.py`, `testgst.py`)


####  Help Mode
To quickly see help and usage instructions:
```bash
python testgsg.py --help
```
You’ll get a summary of all available flags, expected input format, and usage examples.

---

###  Sample Outputs

Each tool prints consistent outputs including approximations, true values (if available), absolute errors, and Lipschitz-based error bounds.

###  Gradient (GSG)

```bash
python testgsg.py --x0 1 2 --function "x0**2 + 3*x1**2" --h 0.01
```

**Output:**
```
Using symbolic function and standard basis:
S matrix:
 [[1. 0.]
 [0. 1.]]
Approximate gradient (GSG): [ 2.01 12.03]
True gradient: [ 2. 12.]
Max Absolute Error: 0.0299
Estimated Lipschitz constant L: 6.0
Lipschitz-based error bound: 0.0424
```

---

###  Hessian (GSH)

```bash
python testgsh.py --x0 1 2 --function "x0**2 * x1 + 3*x1**2" --h 0.01
```

**Output:**
```
Using symbolic function and standard basis:
S matrix:
 [[1. 0.]
  [0. 1.]]
T matrix:
 [[1. 0.]
  [0. 1.]]
Approximate Hessian (GSH): [[4.   2.01]
                           [2.01 6.  ]]
True Hessian: [[4. 2.]
              [2. 6.]]
Max Absolute Error: 0.01
Estimated Lipschitz constant: 2.00
Lipschitz-based error bound: 0.16
```

---

###  Third-Order Tensor (GST)

```bash
python testgst.py --x0 1 2 --function "x0**2 * x1 + 3*x1**2" --h 0.01
```

**Output:**
```
Normalized S, T, U direction matrices:
 [[1. 0.]
  [0. 1.]]

Estimated third-order tensor (Tressian):
 [[[24.01  0.   ]
   [ 0.    0.   ]]
  [[ 0.    0.   ]
   [ 0.   48.01 ]]]

True Tressian at x0:
 [[[24.  0.]
   [ 0.  0.]]
  [[ 0.  0.]
   [ 0. 48.]]]

Absolute error tensor:
 [[[0.01 0.  ]
   [0.   0.  ]]
  [[0.   0.  ]
   [0.   0.01]]]

Estimated Lipschitz constant for Tressian at x0: 24.0
Tressian error bound (auto-estimated L): 0.3394
```

---


###  MATLAB Support (Optional)

MATLAB versions of some methods (GSG, GSH) are included in the repository for academic benchmarking and comparison purposes.

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
## References
Jarry-Bolduc, G. (2023). A Matrix Algebra Approach to Approximate the Hessian. University of British Columbia.

Audet, C., & Hare, W. (2017). Derivative-Free and Blackbox Optimization. Springer.

Kelley, C. T. (1999). Iterative Methods for Optimization. SIAM.

Vicente, L. N. (2008). Optimization Without Derivatives.

---
##  License

MIT License – see [LICENSE](./LICENSE) for full details.

---

##  Authors & Acknowledgments

This project was developed as part of a supervised research collaboration at Mount Royal University (MRU), with the guidance of:

Prof. Gabriel Jarry-Bolduc
Department of Mathematics and Computing
https://github.com/DFOdude

Gaurav Neupane
Undergraduate Research Assistant
B.Sc. Computer Science, MRU
https://github.com/itsneugen

I thank Prof. Jarry-Bolduc for his mentorship and for providing key mathematical insights and theoretical materials during the development of these implementations.

