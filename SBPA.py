import cvxpy as cp
import numpy as np
import variablesPA

def optimizedPowerAllocation(h, combVect, uj, N, nUsers, Pmax):
    """
    Solves the Convex Power Allocation problem for RSMA (Equation 13 of the reference paper).
    
    This function optimizes the Private and Common power allocated to each subcarrier 
    to maximize the Weighted Sum-Rate (WSR), subject to a total power constraint.

    Parameters
    ----------
    h : np.ndarray
        Channel state information matrix.
    combVect : np.ndarray
        Vector indicating the user pairing/combinations on subcarriers.
    uj : np.ndarray
        User weights (priority) vector.
    N : int
        Number of subcarriers.
    nUsers : int
        Total number of users.
    Pmax : float
        Maximum total transmit power budget (in linear units, e.g., Watts).

    Returns
    -------
    Pn : np.ndarray
        A stacked vector of shape (2*N, 1) containing the optimized power allocation.
        - First N elements: Private Power (P_p) per subcarrier.
        - Last N elements: Common Power (P_c) per subcarrier.
    """
    # --- 1. Decision Variables ---
    # P_p: Power allocated to private messages
    # P_c: Power allocated to the common message
    P_p = cp.Variable(N, nonneg=True)  
    P_c = cp.Variable(N, nonneg=True) 

    # --- 2. Objective Function Construction ---
    # We construct the Weighted Sum-Rate (WSR) expression term by term.
    # The log functions are base-e (natural log) in cvxpy, so we divide by np.log(2) 
    # to convert the result to bits/s/Hz.
    
    W = []
    for n in range(N):
        u_1,u_2,g1,g2,alpha = variablesPA.varCalculate_per_subcarrier(h,combVect,uj,n,nUsers);
        term1 = u_1 * cp.log(1 + g1 * P_p[n])/np.log(2);
        term2 = ((u_2 - u_1) / 2) * (cp.log(1 + g2 *  P_p[n]) / np.log(2));
        term3 = ((u_1+u_2) / 2) * (cp.log(1 + g2 *  P_p[n] + alpha *  P_c[n])/np.log(2));
        W.append(term1 + term2 + term3)

    objective = cp.Maximize(cp.sum(W))

# --- 3. Constraints ---
    constraints = [
        cp.sum(P_p) +cp.sum(P_c) <= Pmax,  # Total power constraint
        P_p >= 0,  # Non-negative private power
        P_c >= 0,  # Non-negative common power
    ]

    # Optimization problem (Equation 13 from paper)
    problem = cp.Problem(objective, constraints)

    # Solve the problem by cvx
    problem.solve()
    Pn = np.zeros((2*N,1));
    Pn[:N] = (P_p.value).reshape(-1, 1);
    Pn[N:] = P_c.value.reshape(-1, 1);
    return Pn