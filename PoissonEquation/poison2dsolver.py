import numpy as np
from abc import ABC, abstractmethod

class PoissonSolver2D(ABC):
    """
    Abstract base class for solving the 2D Poisson equation on a square grid.

    The domain is discretized as a (2N+1)×(2N+1) grid with grid spacing dx.
    Subclasses must implement their own relaxation/update strategy by
    overriding `_update_u_grid()` and `update()`.
    """

    def __init__(self, N=31, dx=1):
        """
        Initialize solver state.

        Parameters
        ----------
        N : int
            Half the grid extent in each coordinate. The full grid spans
            indices [-N..+N] in both x and y, for a total size of 2N+1.
        dx : float
            Grid spacing in both x and y directions.
        """
        self.size = 2*N+1
        self.dx = dx
        self.u_grid = np.zeros((self.size, self.size))
        self.ro_grid = np.zeros_like(self.u_grid)
        self._update_ro_grid()
        #how many iterations of the alghoritm were performed
        self.nr_iterations = 0

    @abstractmethod
    def _update_u_grid(self):
        """
        Perform one relaxation sweep over the interior grid points to
        update the potential u_grid according to a specific scheme.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Public method to advance the solver by one iteration. Should call
        `_update_u_grid()` and then increment `nr_iterations`, and perform
        any scheme-specific bookkeeping.
        """
        pass 

    def _update_ro_grid(self):
        """
        Recompute the charge density grid ro_grid by evaluating the
        analytic function ρ(x,y) at each grid point.

        The mapping from array indices (i,j) to physical coordinates (x,y)
        is x = i - N, y = j - N, where N = (size-1)/2.
        """
        N = (self.size - 1) // 2  # N = 31 dla rozmiaru 63
        for i in range(self.size):
            for j in range(self.size):
                x = i - N  # przeskalowanie indeksu na współrzędną x
                y = j - N  # przeskalowanie indeksu na współrzędną y
                self.ro_grid[i, j] = self.ro(x, y)
    
    def ro(self, x,y,d=4, x0=4, dx=1):
        """
        Analytical charge density function ρ(x,y):
        two Gaussian charges of opposite sign centered at (+x0,0) and (-x0,0).

        Parameters
        ----------
        x, y : int
            Grid coordinates in the range [-N..N].
        d : float
            Gaussian width parameter.
        x0 : float
            Offset of the charge centers along the x-axis.
        dx : float
            (Unused) placeholder for potential scaling; grid spacing is
            already stored in self.dx.

        Returns
        -------
        float
            ρ(x,y) = exp(-((x-x0)**2 + y**2)/d^2) - exp(-((x+x0)**2 + y**2)/d^2)
        """
        exp1 = np.exp(-((x-x0)**2+y**2)/(d**2))
        exp2 = np.exp(-((x+x0)**2+y**2)/(d**2))
        return exp1 - exp2
    
    def s_conv(self):
        """
        Compute the global action functional S for the current potential.

        S = -∑_{i,j} [ 0.5 u_{i,j} ∇²u_{i,j} + ρ_{i,j} u_{i,j} ] dx²

        Here ∇²u is approximated by the standard 5-point stencil.

        Returns
        -------
        float
            The value of the discrete action S.
        """
        s = 0
        dx2 = self.dx**2
        for x in range(1, self.size - 1):
            for y in range(1, self.size - 1):
                u_1 = self.u_grid[x, y]
                fst = 0.5 * (self.u_grid[x+1, y] + self.u_grid[x-1, y] - 2 * u_1) / dx2
                snd = 0.5 * (self.u_grid[x, y+1] + self.u_grid[x, y-1] - 2 * u_1) / dx2
                trd = self.ro_grid[x, y]
                s += (u_1 * dx2) * (fst + snd + trd)
        return -s
    



class PoissonSolver1(PoissonSolver2D):
    """
    Task 1: Standard Jacobi relaxation solver for the 2D Poisson equation.

    Uses a simple five‑point stencil update to relax the potential u_grid,
    followed by an explicit computation of the discrete residual (ro_grid_prim)
    to monitor convergence.
    """
    
    def __init__(self, N=31, dx=1):
        """
        Initialize the Jacobi solver.

        Parameters
        ----------
        N : int
            Half the grid size (total size = 2N+1).
        dx : float
            Grid spacing.
        """
        super().__init__(N, dx)
        self.ro_grid_prim = np.zeros_like(self.u_grid)


    def _update_u_grid(self):
        """
        Perform one Jacobi relaxation sweep over interior points.

        Updates each u[i,j] to the average of its four neighbors plus
        the source term contribution: (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}
        + ρ_{i,j}·dx²) / 4. Boundary values remain fixed at zero.
        """
        # Vectorized slicing for internal grid points (excluding boundaries)
        u = self.u_grid
        ro = self.ro_grid
        dx2 = self.dx ** 2

        # Update u_grid without looping
        u[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + 
                        u[1:-1, 2:] + u[1:-1, :-2] + 
                        ro[1:-1, 1:-1] * dx2) / 4
                
    def _update_ro_grid_prim(self):
        """
        Compute the discrete Laplacian of u_grid minus the source term,
        storing the result in ro_grid_prim for monitoring the residual.

        ro_grid_prim[i,j] = (∇²u)_{i,j} / dx² evaluated via the five‑point stencil.
        """
        # Vectorized slicing for internal grid points (excluding boundaries)
        u = self.u_grid
        dx2 = self.dx ** 2

        # Vectorized update
        self.ro_grid_prim[1:-1, 1:-1] = (4 * u[1:-1, 1:-1] -
                                        u[2:, 1:-1] - u[:-2, 1:-1] -
                                        u[1:-1, 2:] - u[1:-1, :-2]) / dx2
 
    def update(self):
        """
        Advance the solver by one full iteration:
        1. Relax u_grid via Jacobi.
        2. Recompute the residual grid ro_grid_prim.
        3. Increment the iteration counter.
        """
        self._update_u_grid()
        self._update_ro_grid_prim()
        self.nr_iterations +=1 

    

class PoissonSolver2(PoissonSolver2D):
    """
    Task 2: Successive Over‑Relaxation (SOR) solver for the 2D Poisson equation.

    Applies weighted Jacobi updates to accelerate convergence, using a relaxation
    factor ω. Also maintains a residual grid ro_grid_prim for diagnostic purposes.
    """

    def __init__(self, N=31, dx=1, w=1.9):
        """
        Initialize the SOR solver.

        Parameters
        ----------
        N : int
            Half the grid size (total size = 2N+1).
        dx : float
            Grid spacing.
        w : float
            Over‑relaxation factor (typically 1 < ω < 2 for optimal speed).
        """
        super().__init__(N, dx)
        self.w = w
        self.ro_grid_prim = np.zeros_like(self.u_grid)

    def _update_u_grid(self):
        """
        Perform one SOR sweep over the interior points.

        For each interior (i,j):
            u_new = (1 - ω) u_old + (ω/4) [u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} + ρ_{i,j}·dx²]
        """
        w = self.w
        #perform the iteration on all values of the gird except on boundaries
        for i in range(1,self.size-1):
            for j in range(1, self.size-1):
                self.u_grid[i,j] = (1-w) * self.u_grid[i,j] + w*(
                                (self.u_grid[i+1,j] + self.u_grid[i-1, j]+
                                self.u_grid[i,j+1] + self.u_grid[i, j-1]+
                                self.ro_grid[i,j]*self.dx**2))/4
    
    def update(self):
        """
        Advance the solver by one full iteration:
        1. Perform an SOR sweep to update u_grid.
        2. Increment the iteration counter.
        """
        self._update_u_grid()
        self.nr_iterations += 1



class PoissonSolver3(PoissonSolver2D):
    """
    Task 3: Local functional‐minimization solver for the 2D Poisson equation.

    At each grid point, we perform a local search over discrete perturbations
    of the potential u_{i,j} to minimize the action functional S.  This
    combines finite‐difference estimates of S with a small set of probe
    values to choose the optimal local update.
    """

    def __init__(self, N=31, dx=1, deltas = [0.0, 0.5, 1.0]):
        """
        Initialize the local minimization solver.

        Parameters
        ----------
        N : int
            Half‐extent of the grid (total size = 2N+1).
        dx : float
            Grid spacing.
        deltas : list of float, optional
            List of probe offsets [d0, d1, d2] to test for each u[i,j].
            Must have length ≤ 3.  Default probes are [0.0, 0.5, 1.0].
        """
        super().__init__(N, dx)
        if len(deltas) > 3:
            raise ValueError("len(deltas) > 3!!, should be smaller!")
        self.deltas = deltas

    def _Sloc(self, i, j, dlt):
        """
        Compute the local action S after perturbing u[i,j] by dlt.

        Temporarily add dlt to u_grid[i,j], sum the contribution of the
        3×3 neighborhood to the discrete action, then restore u[i,j].

        Returns
        -------
        float
            The local action S at (i,j) with u[i,j] → u[i,j] + dlt.
        """
        u0 = self.u_grid[i, j]
        self.u_grid[i, j] = u0 + dlt
        Sloc = 0.0
        dx2 = self.dx**2
        for ii in (i-1, i, i+1):
            for jj in (j-1, j, j+1):
                lapx = (self.u_grid[ii+1, jj] + self.u_grid[ii-1, jj] - 2*self.u_grid[ii, jj]) / dx2
                lapy = (self.u_grid[ii, jj+1] + self.u_grid[ii, jj-1] - 2*self.u_grid[ii, jj]) / dx2
                Sloc -= (0.5 * self.u_grid[ii, jj] * (lapx + lapy)
                         + self.ro_grid[ii, jj] * self.u_grid[ii, jj]) * dx2
        # restore the original value
        self.u_grid[i, j] = u0
        return Sloc

    def _u_point_update(self, i, j,S0):
        """
        Find the optimal local update for u[i,j] by testing probe offsets.

        Steps:
        1. Evaluate S at u + each probe d ∈ self.deltas.
        2. Use three probe values to extrapolate a 4th candidate via finite‐differences.
        3. Compute S at the extrapolated offset.
        4. Choose the offset that yields the minimum local S, and update u[i,j].

        Parameters
        ----------
        i, j : int
            Indices of the grid point to update.
        S0 : float
            (Unused) The global action S before any local changes; provided for API symmetry.

        Returns
        -------
        float
            The new value of u[i,j] after applying the best probe offset.
        """
        deltas = self.deltas
        # original value
        u0 = self.u_grid[i, j]
        # compute S at baseline (delta=0)
        Sloc0 = self._Sloc(i, j, 0.0)
        # compute S for each delta in list
        S_vals = [self._Sloc(i, j, d) for d in deltas]
        # extrapolate a 4th delta via finite-difference
        S1, S2, S3 = S_vals
        num = 3*S1 - 4*S2 + S3
        den = S1 - 2*S2 + S3
        delta4 = 0.25 * num/den if abs(den) > 1e-12 else 0.0
        # compute S at delta4
        S4 = self._Sloc(i, j, delta4)
        # choose best delta among all
        candidates = list(zip(deltas, S_vals)) + [(delta4, S4)]
        delta_best, _ = min(candidates, key=lambda x: x[1])
        # update the grid point with best delta
        self.u_grid[i, j] = u0 + delta_best
        return self.u_grid[i, j]
    
    def _update_u_grid(self):
        """
        Perform one global iteration by applying local minimization at each interior point.

        1. Compute the global action S_total (optional for diagnostics).
        2. For each interior i,j (excluding a one‐cell boundary), call _u_point_update.
        """
        S_total = self.s_conv()
        deltas = self.deltas
        Nbound = self.size - 1
        # update interior points excluding 2-layer boundary offset
        for i in range(2, Nbound-1):
            for j in range(2, Nbound-1):
                self._u_point_update(i, j, S0=S_total)

    def update(self):
        """
        Advance the solver by one full iteration:
        - Execute one sweep of local minimization across the grid.
        - Increment the iteration counter.
        """
        self._update_u_grid()
        # increment iteration count
        self.nr_iterations += 1



class PoissonSolver4(PoissonSolver2D):
    """
    Task 4: Gradient‐descent minimization of the action S for the 2D Poisson equation.

    At each grid point, we approximate the partial derivative ∂S/∂u_{i,j}
    by finite differences using S(u+d) and S(u−d), then step opposite to the gradient
    with step size β. This is equivalent to a local steepest‐descent update.
    """

    def __init__(self, N=31, dx=1, beta = 0.4, d=0.001):
        """
        Initialize the steepest‐descent solver.

        Parameters
        ----------
        N : int
            Half‐extent of the grid (total grid size = 2N+1).
        dx : float
            Grid spacing.
        beta : float
            Descent step parameter; must satisfy 0 < β < 0.5 for stability.
        d : float
            Finite‐difference probe size used to approximate the local gradient.
        """
        super().__init__(N, dx)
        if beta < 0:
            raise ValueError("beta must be bigger than 0!")
        self.beta = beta
        self.d = d
    
    def _Sloc(self, i, j):
        """
        Compute the local action S in the 3×3 neighborhood around (i,j)
        using the current u_grid values.

        This reuses the same finite‐difference Laplacian logic as other solvers.
        """
        dx2 = self.dx**2
        Sloc = 0.0
        for ii in (i-1, i, i+1):
            for jj in (j-1, j, j+1):
                lapx = (self.u_grid[ii+1, jj] + self.u_grid[ii-1, jj] - 2*self.u_grid[ii, jj]) / dx2
                lapy = (self.u_grid[ii, jj+1] + self.u_grid[ii, jj-1] - 2*self.u_grid[ii, jj]) / dx2
                Sloc -= (0.5 * self.u_grid[ii, jj] * (lapx + lapy)
                         + self.ro_grid[ii, jj] * self.u_grid[ii, jj]) * dx2
        return Sloc
    
    def _update_u_point(self, i, j):
        """
        Compute the new potential at (i,j) by a finite‐difference descent step.

        1. Evaluate S_plus = S(u+d) and S_minus = S(u−d).
        2. Approximate the partial derivative ∂S/∂u_{i,j} ≈ (S_plus − S_minus)/(2d).
        3. Return u0 − β * ∂S/∂u_{i,j}.
        """
        B = self.beta
        d = self.d
        #Save the value of u at (i,j) before doing anything
        u0_ij = self.u_grid[i,j]
        #Update the u(i,j) by d
        self.u_grid[i,j] += d
        #get S+ value
        S_plus = self._Sloc(i,j)
        #get the old val for u[i,j]
        self.u_grid[i,j] = u0_ij
        #Update the u(i,j) by -d
        self.u_grid[i,j] -= d
        #get the S- value 
        S_minus = self._Sloc(i,j)

        #get the divS
        divS = (S_plus - S_minus) / (2*d)

        #Returns the new value for u[i,j]
        return u0_ij - B*divS
        
    def _update_u_grid(self):
        """
        Perform one full descent sweep: update each interior grid point
        by computing its new value via _update_u_point.
        """
        # iterate interior points (exclude boundaries)
        for i in range(1, self.size-2):
            for j in range(1, self.size-2):
                self.u_grid[i,j] = self._update_u_point(i,j)
    
    def update(self):
        """
        Advance the solver by one iteration:
        - Execute a steepest‐descent sweep over the interior.
        - Increment the iteration counter.
        """
        self._update_u_grid()
        self.nr_iterations += 1



class PoissonSolver5(PoissonSolver2D):
    """
    Task 5: Stochastic local‐probe minimization solver for the 2D Poisson equation.

    For each grid point, this method attempts a sequence of random perturbations
    of the potential and accepts the first one that reduces the local action S.
    This randomized approach can escape shallow minima and may converge faster
    when properly tuned.
    """


    def __init__(self, r=0.1, N=31, dx=1.0, max_probes=100):
        """
        Initialize the stochastic probe solver.

        Parameters
        ----------
        r : float
            Maximum magnitude of random probe offsets (uniform in [-r, r]).
        N : int
            Half‐extent of the grid (total size = 2N+1).
        dx : float
            Grid spacing.
        max_probes : int
            Maximum number of random trials per grid point per iteration.
        """
        super().__init__(N, dx)
        self.r = r
        self.max_probes = max_probes

    def _Sloc(self, i, j):
        """
        Compute the local action S at (i,j) for the current u_grid.

        Uses the five‑point stencil to approximate the Laplacian in the
        3×3 neighborhood around (i,j), summing the local contribution to S.

        Returns
        -------
        float
            The local action S in the vicinity of (i,j).
        """
        dx2 = self.dx**2
        Sloc = 0.0
        for ii in (i-1, i, i+1):
            for jj in (j-1, j, j+1):
                lapx = (self.u_grid[ii+1, jj] + self.u_grid[ii-1, jj] - 2*self.u_grid[ii, jj]) / dx2
                lapy = (self.u_grid[ii, jj+1] + self.u_grid[ii, jj-1] - 2*self.u_grid[ii, jj]) / dx2
                Sloc -= (0.5 * self.u_grid[ii, jj] * (lapx + lapy)
                         + self.ro_grid[ii, jj] * self.u_grid[ii, jj]) * dx2
        return Sloc

    def _find_u_point_val(self, i, j):
        """
        Search for a random perturbation of u[i,j] that decreases the local S.

        Repeatedly draw δ ∈ Uniform[-r, r] up to max_probes times. If
        S(u + δ) < S(u), accept δ and leave u[i,j] updated. Otherwise revert.

        Returns
        -------
        float
            The accepted offset δ if found, or 0.0 if none improves S within max_probes.
        """
        u0 = self.u_grid[i, j]
        Sloc0 = self._Sloc(i, j)
        for _ in range(self.max_probes):
            delta = np.random.uniform(-self.r, self.r)
            self.u_grid[i, j] = u0 + delta
            if self._Sloc(i, j) < Sloc0:
                return delta
        # revert and give up
        self.u_grid[i, j] = u0
        return 0.0

    def _update_u_grid(self):
        """
        Perform one stochastic-probe sweep over interior grid points.

        At each point, attempt up to max_probes random local updates
        and apply the first one that reduces the local action S.
        """
        # iterate interior points (exclude boundaries)
        for i in range(1, self.size-2):
            for j in range(1, self.size-2):
                delta = self._find_u_point_val(i, j)
                # apply successful delta (already applied inside _find), nothing else needed

    def update(self):
        """
        Advance the solver by one full iteration:
        - Execute one stochastic-probe sweep.
        - Increment the iteration counter.
        """
        self._update_u_grid()
        self.nr_iterations += 1

