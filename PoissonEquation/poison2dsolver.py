import numpy as np


class PoissonSolver2D():
    """Solve 2D poisson equation"""
    def __init__(self, N=31, dx=1):
        """
        Initialize the grids.
        N (int): defines the [-N,..0...N]x[-N,..0...N] grid.
        """
        self.size = 2*N+1
        self.dx = dx
        self.u_grid = np.zeros((self.size, self.size))
        self.ro_grid = np.zeros_like(self.u_grid)
        self._update_ro_grid()
        self.ro_grid_prim = np.zeros_like(self.u_grid)
        #how many iterations of the alghoritm were performed
        self.nr_iterations = 0

    def ro(self, x,y,d=4, x0=4, dx=1):
        """
        Returns the charge density for a position (x,y).
        x,y (int): values that are in range [-N,N]
        REMEMBER TO UPDATE THE x,y values from [0,2N+1]
        to valus in range [-N, N]!!!!
        """
        exp1 = np.exp(-((x-x0)**2+y**2)/(d**2))
        exp2 = np.exp(-((x+x0)**2+y**2)/(d**2))
        return exp1 - exp2
    
    def _update_ro_grid(self):
        N = (self.size - 1) // 2  # N = 31 dla rozmiaru 63
        for i in range(self.size):
            for j in range(self.size):
                x = i - N  # przeskalowanie indeksu na współrzędną x
                y = j - N  # przeskalowanie indeksu na współrzędną y
                self.ro_grid[i, j] = self.ro(x, y)

    def _update_u_grid(self):
        """
         Updates the u_gird values, by performing one iteration
         of relaxation loop on the u_grid.
         Boundaries always 0!
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
        Update the ro_grid_prim on the basis of u_grid in order
        to check the quality of the solution.
        Boundaries 0!
        """
        # Vectorized slicing for internal grid points (excluding boundaries)
        u = self.u_grid
        dx2 = self.dx ** 2

        # Vectorized update
        self.ro_grid_prim[1:-1, 1:-1] = (4 * u[1:-1, 1:-1] -
                                        u[2:, 1:-1] - u[:-2, 1:-1] -
                                        u[1:-1, 2:] - u[1:-1, :-2]) / dx2
                
    def s_conv(self):
        """
        Value S, for investigating the convergence
        of the iterated function into the Poisson solution.
        Returns:
            s (float) : the Value S
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
    
    def update(self):
        self._update_u_grid()
        self._update_ro_grid_prim()
        self.nr_iterations +=1 

    

class PoissonSolver2DFast(PoissonSolver2D):
    """
    Solves 2D Poisson equation, with alternative way to 
    update the u grid, as described in the second task.
    """

    def __init__(self, N=31, dx=1):
        super().__init__(N, dx)

    def _update_u_grid(self, w = 1.9):
        """
         Updates the u_gird values, by performing one iteration
         of relaxation loop on the u_grid (over-relaxation).
         Boundaries always 0!
        """
        #perform the iteration on all values of the gird except on boundaries
        for i in range(1,self.size-1):
            for j in range(1, self.size-1):
                self.u_grid[i,j] = (1-w) * self.u_grid[i,j] + w*(
                                (self.u_grid[i+1,j] + self.u_grid[i-1, j]+
                                self.u_grid[i,j+1] + self.u_grid[i, j-1]+
                                self.ro_grid[i,j]*self.dx**2))/4



class PoissonSolver3(PoissonSolver2D):
    """
    Solve the 2D poisson equation using local functional minimization (task 3).
    """
    def __init__(self, N=31, dx=1):
        super().__init__(N, dx)

    def _Sloc(self, i, j, dlt):
        # compute local S value after perturbing u_grid[i,j] by dlt
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

    def _u_point_update(self, i, j, deltas, S0):
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

    def update(self, deltas=[0.0, 0.5, 1.0]):
        # global convergence functional S (unused in point update but can monitor)
        S_total = self.s_conv()
        Nbound = self.size - 1
        # update interior points excluding 2-layer boundary offset
        for i in range(2, Nbound-1):
            for j in range(2, Nbound-1):
                self._u_point_update(i, j, deltas, S0=S_total)
        # increment iteration count
        self.nr_iterations += 1



class PoissonSolver4(PoissonSolver2D):
    """Solve 2d poisson equation, by the method described in task4."""

    def __init__(self, N=31, dx=1, beta = 0.4, d=0.001):
        """beta (float): iteration parameter < 0.5
           d (float) : value, by which we will be changing the u(i,j), to calculate divS."""
        super().__init__(N, dx)
        if beta < 0:
            raise ValueError("beta must be bigger than 0!")
        self.beta = beta
        self.d = d
    
    def _Sloc(self, i, j):
        """Local contribution to S at (i,j) using current u_grid."""
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
        Returns the new value for u_grid at the point (i,j).
        return: u[i,j] - beta*divS
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
        """One sweep: update each interior grid point by the new value for u,
        Calculated by the method from task 4."""
        # iterate interior points (exclude boundaries)
        for i in range(1, self.size-2):
            for j in range(1, self.size-2):
                self.u_grid[i,j] = self._update_u_point(i,j)
    
    def update(self):
        self._update_u_grid()
        self.nr_iterations += 1



class PoissonSolver5(PoissonSolver2D):
    """Solves 2D Poisson eq. via random local probes reducing the S functional."""

    def __init__(self, r=0.1, N=31, dx=1.0, max_probes=100):
        super().__init__(N, dx)
        self.r = r
        self.max_probes = max_probes

    def _Sloc(self, i, j):
        """Local contribution to S at (i,j) using current u_grid."""
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
        """Try up to max_probes random deltas; return first that decreases local S, else 0."""
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
        """One sweep: update each interior grid point by random probe decreasing S locally."""
        # iterate interior points (exclude boundaries)
        for i in range(1, self.size-2):
            for j in range(1, self.size-2):
                delta = self._find_u_point_val(i, j)
                # apply successful delta (already applied inside _find), nothing else needed

    def update(self):
        """Perform a full iteration and increment counter."""
        self._update_u_grid()
        self.nr_iterations += 1

