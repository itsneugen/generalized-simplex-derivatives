# Generalized Simplex Derivative Approximations
*A research-focused framework for applying and evaluating optimization techniques across various platforms, such as methods for approximating derivatives.*




##  Overview

This repository provides a Python-based framework for computing approximations of gradients, Hessians, and experimental third-order derivatives, with additional MATLAB implementations.

The main methods contained in this repository are:
- Generalized Simplex Gradient (GSG)
- Generalized Simplex Hessian (GSH)
- Generalized Simplex Tressian (GST)

The framework is designed for flexibility: The input can be either symbolic function or a list of functional values, making it useful in numerical analysis and optimization tasks.

The codes may be useful in:
- Derivative-free optimization
- Numerical testing and benchmarking
- Research and classroom use


This work is part of a **research project under Prof. Gabriel Jarry-Bolduc** at Mount Royal University (MRU), focusing on *derivative-free optimization and numerical analysis*.


---
##  Project Structure

This repository contains Python and MATLAB implementations related to derivative-free approximation techniques, primarily focusing on approximating gradients, Hessians, and third-order tensors using Generalized Simplex methods.

```
├── gsg.py                # Core logic for Generalized Simplex Gradient (GSG)
├── gsh.py                # Core logic for Generalized Simplex Hessian (GSH)
├── tres.py               # Core logic for Generalized Simplex Tressian (GST)
│
├── testgsg.py            # CLI and interactive tester for GSG
├── testgsh.py            # CLI and interactive tester for GSH
├── testgst.py            # CLI and interactive tester for GST 
│
├── run_example.py        # General-purpose script to test all variants (GSG, GSH, GST)
├── gen.py                # Early work on generalized higher-order simplex approximation(on progress)
├── testgen.py            # Testing harness for gen.py
├── gcsg.py               # Centered variant of GSG (optional/experimental)
├── gcsh.py               # Centered variant of GSH (optional/experimental)
│
├── matlab/               # MATLAB implementations (kept minimal but usable)

```

>  Note: While the spotlight is on GSG, GSH, and GST, other files are included for completeness and experimentation. MATLAB versions are also available for users.

##  Features Overview

This repository provides Python tools for approximating derivatives of different orders (gradient, Hessian, third-order tensor) using generalization of finite difference approximation methods. Each tool is designed to be flexible and beginner-friendly, with both command-line and interactive modes. The accuracy of each method can be controlled by adjusting the step size and the choice of direction matrices.

###  Available Estimators

| File         | Derivative Order | Description                        |
|--------------|------------------|------------------------------------|
| `testgsg.py` | First-order      | Approximate gradient (GSG method)    |
| `testgsh.py` | Second-order     | Approximate Hessian (GSH method)     |
| `testgst.py` | Third-order      | Approximate Tressian (GST method)   |

All three scripts follow a **consistent interface**, making it easy to switch between methods and apply the same workflow across different derivative orders.

---

###  How to Run (Any of GSG / GSH / GST)

#### Command-Line Mode
Run script in one line:

```bash
python testgsg.py --x0 1 2 --function "x0**2 + 3*x1**2" --h 0.01
```

**Parameter Descriptions**:

- `--x0`:  The initial point of interest for derivative estimation, provided as a row vector of space-separated values
    (e.g., `--x0 1 2` for the point $[1, 2]$).
- `--function`:  Function to differentiate, written using Python syntax with variables `x0`, `x1`, etc.  
    (e.g., The function `x0**2 + 3*x1**2` represents `f(x0, x1) = x0² + 3x1²`. The number of variables in your function          depends on how many values you provide to `--x0`. Each input value becomes one variable:
   
    `--x0 1 2 → 2 variables → x0, x1`

    `--x0 1 2 3 → 3 variables → x0, x1, x2`

  You must use these variable names (`x0, x1, x2`, etc.) inside the `--function` string.

- `--h`:  Step size for the finite difference approximation. Controls the precision (e.g., `--h 0.01`).


 **What the Script Does** :
- Evaluates the derivative (Gradient for GSG, Hessian for GSH, or Tressian for GST) at the specified point `x0`.
- Uses a normalized standard basis for computations.
- Outputs the computed derivative.
- Compares the result with the symbolic derivative, if available.
- Provides a Lipschitz-based error bound for the approximation, where applicable.

####  Interactive Mode (Optional)
If a user prefers prompts:
```bash
python testgsh.py --interactive
```
It will guide you through input step-by-step.




####  Manual Function-Value Mode
Use the `--manual` flag to compute GSG, GSH, or GST using pre-evaluated function values where the function formula is unavailable.


**Parameter Descriptions**:

`--manual`: Enable manual mode

`--x0`: Point of interest (e.g., `--x0 1 2` gives x0 = [1, 2])

`--S`: Direction matrix S (and also T or U for higher orders) entered as a string of semicolon-separated rows with space-separated values 


`--values`: Function values as comma-separated list


**Example** 
- Compute: `f(1, 2) = 14`, `f(1.01, 2) = 14.0201`, `f(1, 2.01) = 14.1303`, `f(1.01, 2.01) = 14.170601`.



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

Each method prints consistent outputs including approximations, true values (if available), absolute errors, and an approximation of the Lipschitz-based error bounds defined in **Jarry- Bolduc's PhD thesis, 2023**.

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
Absolute Error: 0.0299
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
Absolute Error: 0.01
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

Absolute error:
 [[[0.01 0.  ]
   [0.   0.  ]]
  [[0.   0.  ]
   [0.   0.01]]]

Estimated Lipschitz constant for Tressian at x0: 24.0
Tressian error bound (using Estimated Lipschitz contant): 0.3394
```

---


###  MATLAB Support (Optional)

MATLAB versions of some methods (GSG, GSH) are included in the repository.

---



##  Notes

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
Jarry-Bolduc, G. (2023). Numerical analysis for derivative-free optimization. University of British Columbia.

Audet, C., & Hare, W. (2017). Derivative-Free and Blackbox Optimization. Springer.

Kelley, C. T. (1999). Iterative Methods for Optimization. SIAM.

Vicente, L. N. (2008). Optimization Without Derivatives.

---
##  License

MIT License – see [LICENSE](./LICENSE) for full details.

---

##  Authors & Acknowledgments

This project was developed as part of a supervised research collaboration at Mount Royal University (MRU), with the guidance of:

**Prof. Gabriel Jarry-Bolduc**  

Department of Mathematics and Computing, Mount Royal University  

B.Sc. Hons., Université du Québec à Trois-Rivières, 2017  

 M.Sc., The University of British Columbia, 2019  

 https://github.com/DFOdude

**Gaurav Neupane**  

Undergraduate Research Assistant  

B.Sc. Computer Science, MRU  

https://github.com/itsneugen  



I thank Prof. Jarry-Bolduc for his mentorship and for providing key mathematical insights and theoretical materials during the development of these implementations.

