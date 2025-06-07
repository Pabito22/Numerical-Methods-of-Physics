# Numerical Methods of Physics

This repository contains solutions to tasks from numerical methods courses at AGH, focusing on computational techniques used in physics. The projects include implementations and simulations for two main topics:

## Poisson Equation

### 2D Poisson Equation Solver

This module provides a suite of solver classes for the two-dimensional Poisson equation on a square grid. The solvers are organized under an abstract base class and implement various iterative relaxation and minimization methods:

* **`PoissonSolver2D`** (abstract base class)
  Defines the grid setup, source-term initialization, and convergence functional. Subclasses must implement:

  * `_update_u_grid()`: one iteration’s worth of updates to the potential field.
  * `update()`: public method that calls `_update_u_grid()`, increments iteration count, and performs any additional bookkeeping.

* **`PoissonSolver1`** (Jacobi relaxation)

  * Uses the standard five-point stencil Jacobi update, averaging neighboring grid values and adding the source term.
  * Computes a residual grid (`ro_grid_prim`) to monitor convergence.

* **`PoissonSolver2`** (Successive Over-Relaxation, SOR)

  * Applies a weighted Jacobi update using a relaxation factor to accelerate convergence.
  * Typically converges faster than pure Jacobi when the relaxation factor is between 1 and 2.

* **`PoissonSolver3`** (Local functional minimization)

  * Examines a set of discrete offsets around each point and chooses the one that minimizes the local action.

* **`PoissonSolver4`** (Steepest-descent minimization)

  * Computes the gradient of the action with respect to the grid values and updates in the direction that reduces the action.

* **`PoissonSolver5`** (Stochastic probe minimization)

  * Randomly probes small offsets at each grid point and accepts changes that lower the action.
  * Useful for escaping shallow local minima and exploring the solution space differently.

## Features

* **Modular design** via an abstract base class—new relaxation or minimization schemes can be added by subclassing `PoissonSolver2D`.
* **Convergence monitoring** through the action functional and, for Jacobi/SOR, the discrete residual grid.
* **Performance comparison**: easily benchmark iterations-to-tolerance across all five methods.

## Usage

```python
from poisson import PoissonSolver1, PoissonSolver2, PoissonSolver3, PoissonSolver4, PoissonSolver5

# Create solver instance (choose your method)
solver = PoissonSolver2(N=50, dx=0.1, w=1.8)

# Iterate until convergence
tolerance = 1e-6
while solver.s_conv() > tolerance and solver.nr_iterations < 10000:
    solver.update()

print(f"Converged in {solver.nr_iterations} iterations.")
```

## Plotting & Analysis

* Use `solver.s_conv()` to obtain the current action.
* Compare the action versus iteration for different methods or parameter choices.
* Visualize `solver.u_grid` with Matplotlib or your preferred plotting library.

## Dependencies

* Python 3.7+
* NumPy
* (Optional: Matplotlib for plotting convergence curves and potential fields.)

## Ion Trap Project

This module simulates the electric potential in an ion trap configuration using numerical methods. It models the trapping behavior and potential distribution relevant for experimental setups in atomic and plasma physics.
