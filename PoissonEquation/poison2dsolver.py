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
        """
        exp1 = np.exp(-((x-x0)**2+y**2)/(d**2))
        exp2 = np.exp(-((x+x0)**2+y**2)/(d**2))
        return exp1 - exp2
    
    def _update_ro_grid(self):
        """
        Updates ro grid with the charge density.
        grid (np.array): grid to be updated
        """
        for i in range(self.size):
            for j in range(self.size):
                self.ro_grid[i,j] = self.ro(i,j)

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
        for x in range(1, self.size - 2):
            for y in range(1, self.size - 2):
                u_1 = self.u_grid[x, y]
                fst = 0.5 * (self.u_grid[x+1, y] + self.u_grid[x-1, y] - 2 * u_1) / dx2
                snd = 0.5 * (self.u_grid[x, y+1] + self.u_grid[x, y-1] - 2 * u_1) / dx2
                trd = self.ro_grid[x, y]
                s += (u_1 * dx2) * (fst + snd + self.ro_grid[x, y])
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
                                self.ro(i,j)*self.dx**2))/4


class PS2DFunctionalMinimization(PoissonSolver2D):
    """ 
    Solve the 2D poisson equation, by the method described 
    in task 3.
    """
    def __init__(self, N=31, dx=1):
        super().__init__(N, dx)
    
    def _s_conv_loc(self, i,j, dlt=0):
        """
         Calcuates the local value of the functional S
         at the point (i,j), in order to then update
         u_grid values at the [i,j] point.
         Params:
         i (int): x position, all grid, but no at boundaries
         j (int): y position, all grid, but no at boundaries
         dlt (float): value, by which the u value will be shifted
         Return:
         The local S value around the point [i,j]
        """
        #check if the values are in the range defined by the grid and out of boundaries
        if abs(i) >= self.size-1 or abs(j) >= self.size-1 or i<=0 or j<=0:
            raise ValueError(f"i={i} or j={j} out of grid size index that is in (0, {self.size-1})!") 
        s = 0
        dx2 = self.dx**2
        # Use valid index ranges only
        for x in range(i-1, i+1):
            for y in range(j-1,  j+1):
                u_1 = self.u_grid[x, y] + dlt
                fst = 0.5 * (self.u_grid[x+1, y] + self.u_grid[x-1, y] - 2 * u_1) / dx2
                snd = 0.5 * (self.u_grid[x, y+1] + self.u_grid[x, y-1] - 2 * u_1) / dx2
                trd = self.ro_grid[x, y]
                s += (u_1 * dx2) * (fst + snd + self.ro_grid[x, y])
        return -1*s

        
    def _u_point_updater(self, i,j,S0, dlts=[0,0.5,1]):
        """
        Finds the right delta to update the grid at point 
        u_grid[i,j] and returns the updated balue for the position [i,j].
        dlts (list) : values of dlts to be used 
        S0 (float) : S value for one iteration trough the whole grid
        """
        #check if the values are in the range defined by the grid and out of boundaries
        if abs(i) >= self.size-1 or abs(j) >= self.size-1 or i<=0 or j<=0:
            raise ValueError(f"i={i} or j={j} out of grid size index that is in (0, {self.size-1})!") 
        
        S1 = S0 - self._s_conv_loc(i,j,0) + self._s_conv_loc(i,j,dlts[0])
        S2 = S0 - self._s_conv_loc(i,j,0) + self._s_conv_loc(i,j,dlts[1])
        S3 = S0 - self._s_conv_loc(i,j,0) + self._s_conv_loc(i,j,dlts[2])
        # Calculate the denominator for dlt4
        denominator = (S1 - 2 * S2 + S3)
        # Check if the denominator is zero (or close to zero)
        if abs(denominator) < 1e-10:
            # Handle division by zero case
            print("Warning: Division by zero encountered in dlt4 calculation.")
            dlt4 = 0  # or some other small default value
        else:
            # Safe calculation
            dlt4 = 0.25 * (3 * S1 - 4 * S2 + S3) / denominator

        S4 = S0 - self._s_conv_loc(i,j,0) + self._s_conv_loc(i,j,dlt4)
        #create a array of Si
        Sarray = np.array([S1,S2,S3,S4])
        dltas = dlts[:]
        dltas.append(dlt4)
        #array for deltas
        deltas = np.array(dltas)
        i_min = np.argmin(Sarray)
        delta_fin = deltas[i_min]

        return self.u_grid[i,j] + delta_fin
    
    def _update_u_grid(self, dlts=[0,0.5,1]):
        """
        Update the ro_grid_prim on the basis of u_grid in order
        to check the quality of the solution.
        Boundaries 0!
        """
        S0 = self.s_conv()
        # Create a copy to avoid overwriting during iteration
        new_u = self.u_grid.copy()
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                new_u[i, j] = self._u_point_updater(i,j,S0, dlts)
        self.u_grid = new_u

    def update(self):
        self._update_u_grid()
        self.nr_iterations +=1 