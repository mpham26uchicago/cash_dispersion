import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.special import lambertw, factorial
from scipy.optimize import linprog

def system_of_odes(t, y):
    """
    Defines a system of ordinary differential equations (ODEs) to be solved.

    This function represents a system of ODEs that depends on a parameter `mu` 
    and an input vector `y`. The function calculates the derivatives of the 
    elements in `y` with respect to time `t`.

    Parameters:
    -----------
    t : float
        The independent variable, typically representing time. This parameter 
        is included for compatibility with ODE solvers but is not used directly 
        in the calculations.
    y : list or numpy array of floats
        The dependent variables, a vector representing the state of the system.
        The length of `y` is N+1, where N is the maximum number of states, 
        assumed to be at least 2.

    Returns:
    --------
    dyn_dt : list of floats
        A list containing the time derivatives of the elements of `y`, with 
        each entry corresponding to the derivative of the corresponding element 
        in `y`.
    
    Notes:
    ------
    - The parameter `mu` is a global variable and must be defined before 
      calling this function.
    - The system of ODEs is defined for the first element (`y[0]`), the second 
      element (`y[1]`), and the last element (`y[N]`) separately as edge cases.
    - For elements between the second and the last (`y[2]` to `y[N-1]`), the 
      derivatives are computed using a general formula.
    """

    # Global mu, since mu cannot be added inside the call
    global mu
    
    # Let's say N = 5 is max, so y has dim N+1, N>=2
    N = len(y) - 1
    
    # Define the edge cases separately
    dy0_dt = -(mu - y[1]) * y[0]
    dy1_dt = 2 * y[2] + (mu - y[1]) * y[0] - (mu - y[1]) * y[1]
    dyN_dt = (mu - y[1]) * y[N-1] - (N + mu - y[1]) * y[N]

    # Define the general cases
    dyn_dt = [(n + 1) * y[n + 1] + (mu - y[1]) * y[n - 1] - (n + mu - y[1]) * y[n] for n in range(2, N)]

    # Insert/append the edge cases
    dyn_dt.insert(0, dy0_dt)
    dyn_dt.insert(1, dy1_dt)
    dyn_dt.append(dyN_dt)
    
    return dyn_dt

def random_starting_state(N, mu):
    """
    Generates a random starting state for a system, satisfying certain constraints.

    This function creates a random starting state vector `x` of length `N+1` by solving a 
    linear programming problem with specific equality constraints. The random values are 
    generated and then optimized to satisfy these constraints.

    Parameters:
    -----------
    N : int
        The number of states minus one, defining the length of the state vector `x` as `N+1`.
    mu : float
        A parameter that defines the sum of the product of indices and state variables, 
        used as a constraint in the linear programming problem.

    Returns:
    --------
    x : numpy array of floats
        The optimized starting state vector of length `N+1`, which satisfies the specified 
        constraints if the linear programming problem is successful.
        Returns `None` if no solution exists.

    Notes:
    ------
    - The linear programming problem is solved using the `linprog` function from the `scipy.optimize` module.
    - The constraints are defined as follows:
        - The sum of all elements in the state vector `x` must be equal to 1.
        - The weighted sum of the elements in `x` by their indices must be equal to `mu`.
    - If the linear programming problem does not find a feasible solution, the function 
      prints a message and returns `None`.
    """

    # Generate random weights for the linear programming objective function
    weights = np.random.uniform(0, 1, N+1) * np.arange(1, N+2)**2
    
    # Define the equality constraints matrix and right-hand side
    equality_lhs = np.vstack([np.ones(N+1).reshape(1, -1), np.arange(N+1).reshape(1, -1)])
    equality_rhs = np.array([[1], [mu]])
    
    # Solve the linear programming problem
    sol = linprog(c=weights, A_eq=equality_lhs, b_eq=equality_rhs)
    
    # Check if a feasible solution was found
    if sol.success:
        return sol.x
    else:
        print("No solution exists")
        return None