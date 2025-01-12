import numpy as np
from scipy.optimize import minimize
import variablesPA

def optimizedPowerAllocation(h,combVect,uj,N,nUsers,Pmax):
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

