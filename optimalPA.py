import numpy as np
from scipy.optimize import minimize
import variablesPA

# This code is a benchmark that reproduces the results from the following paper:
"""
E.V.Pereira and F. R.M.Lima,‘‘Adaptive powerallocation among private
 and common streams for multicarrier RSMA system,’’ in Proc. 19th Int.
 Symp. Wireless Commun. Syst. (ISWCS), Jul. 2024, pp. 1–6.
 """
def optimizedPowerAllocation(h,combVect,uj,N,nUsers,Pmax):
   """
    Optimizes Power Allocation using SciPy (Sequential Least Squares Programming).
    
    This function maximizes the Weighted Sum-Rate (WSR) by minimizing the 
    negative WSR using a gradient-based non-linear solver.

    Parameters
    ----------
    h : np.ndarray
        Channel state information.
    combVect : np.ndarray
        User combination/pairing vector.
    uj : np.ndarray
        User weights (priorities).
    N : int
        Number of subcarriers.
    nUsers : int
        Total number of users.
    Pmax : float
        Maximum power budget.

    Returns
    -------
    P_opt : np.ndarray
        Optimized power vector of shape (2*N,).
        - Indices [0 to N-1]: Private Power (P_p)
        - Indices [N to 2N-1]: Common Power (P_c)
    """
   weights,g1,g2,alpha = variablesPA.varCalculate(h,combVect,uj,N,nUsers);
    #Objective Function
   def funObj(Pn):
       term1 = weights[0, :] * np.log2(1 + g1 * Pn[:N])
       term2 = ((weights[1, :] - weights[0, :]) / 2) * np.log2(1 + g2 * Pn[:N])
       term3 = ((weights[1, :] + weights[0, :]) / 2) * np.log2(1 + g2 * Pn[:N] + alpha * Pn[N:])
       return -np.sum(term1 + term2 + term3)
    
   lb = np.zeros((2 * N)) # lower bound
    # total power constraint
   def nonlcon(Pn):
       return Pmax - (np.sum(Pn[:N]) + np.sum(Pn[N:]));
    
   x0 = np.ones(2 * N) # Initial estimation
   constraints = {'type': 'ineq', 'fun': nonlcon}
   result = minimize(funObj, x0, bounds=[(0, None)] * (2 * N), constraints=constraints, options={'disp': False, 'maxiter': 1000})

   P_opt = result.x
   fval = result.fun
   return P_opt

