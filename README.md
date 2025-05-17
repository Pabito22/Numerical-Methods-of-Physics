# Numerical-Methods-of-Physics

## PoissonEquation
# 2D Poisson Equation Solver

This file provides a suite of solver classes for the two‑dimensional Poisson equation on a square grid. The solvers are organized under an abstract base class and implement a variety of iterative relaxation and minimization methods:

- **`PoissonSolver2D`** (abstract base class)  
  Defines the grid setup, source‐term initialization, and convergence functional. Subclasses must implement:
  - `_update_u_grid()`: one iteration’s worth of updates to the potential field.
  - `update()`: public method that calls `_update_u_grid()`, increments iteration count, and performs any additional bookkeeping.

- **`PoissonSolver1`** (Jacobi relaxation)  
  - Standard five‑point stencil Jacobi update:  
    \[
      u_{i,j} \leftarrow \frac{1}{4}\Bigl(u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1} + \rho_{i,j} \, dx^2\Bigr).
    \]  
  - Computes a residual grid (`ro_grid_prim`) to monitor convergence.

- **`PoissonSolver2`** (Successive Over‑Relaxation, SOR)  
  - Weighted Jacobi (SOR) update with relaxation factor ω:  
    \[
      u_{i,j} \leftarrow (1-\omega)\,u_{i,j} + \frac{\omega}{4}\Bigl(\dots\Bigr).
    \]  
  - Faster convergence than pure Jacobi for optimal ω ∈ (1,2).

- **`PoissonSolver3`** (Local functional minimization)  
  - At each interior point, probes a small set of discrete offsets \(\{d_0,d_1,d_2\}\) and extrapolates a fourth via finite differences.
  - Chooses the offset that minimizes the local action \(S\).

- **`PoissonSolver4`** (Steepest‑descent on \(S\))  
  - Finite‐difference approximation of \(\partial S/\partial u_{i,j}\) via \(S(u\pm d)\).
  - Updates \(u_{i,j}\leftarrow u_{i,j}-\beta\,\partial S/\partial u_{i,j}\), with β tuned for fastest convergence.

- **`PoissonSolver5`** (Stochastic probe minimization)  
  - Randomly samples offsets \(\delta\in[-r,r]\) at each grid point, accepts the first that lowers the local action.
  - Can escape shallow local minima and offers an alternative convergence path.

## Features

- **Modular design** via an abstract base class—new relaxation or minimization schemes can be added by subclassing `PoissonSolver2D`.
- **Convergence monitoring** through the action functional \(S\) and, for Jacobi/SOR, the discrete residual grid.
- **Performance comparison**: easily benchmark iterations‐to‐tolerance across all five methods.

## Usage

<pre markdown>```python from poisson import PoissonSolver1, PoissonSolver2, PoissonSolver3, PoissonSolver4, PoissonSolver5

# Create solver instance (choose your method)
solver = PoissonSolver2(N=50, dx=0.1, w=1.8)

# Iterate until convergence
tolerance = 1e-6
while solver.s_conv() > tolerance and solver.nr_iterations < 10000:
    solver.update()

print(f"Converged in {solver.nr_iterations} iterations.")
 ```</pre>


## Plotting & Analysis

    Use solver.s_conv() to obtain the current action SS.

    Compare S vs. iteration for different methods or parameter choices (ω, β, etc.).

    Visualize solver.u_grid with Matplotlib or your preferred plotting library.

## Dependencies

    Python 3.7+

    NumPy

(Optional: Matplotlib for plotting convergence curves and potential fields.)
