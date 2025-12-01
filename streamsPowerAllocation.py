import numpy as np
import waterFilling
import numpy as np
import math

# This code is a benchmark that reproduces the results from the following paper:
"""
E.V.Pereira and F. R.M.Lima,‘‘Adaptive powerallocation among private
 and common streams for multicarrier RSMA system,’’ in Proc. 19th Int.
 Symp. Wireless Commun. Syst. (ISWCS), Jul. 2024, pp. 1–6.
 """
 
def streams_power_allocation(h, N, nUsers, x, uj, P):
    """
    Determines the optimal power splitting fraction 't' for each subcarrier.
    
    This function iterates through subcarriers, identifies the active user pair,
    calculates the regime thresholds (Gamma), and solves for the optimal 
    fraction of power allocated to private streams.

    Parameters
    ----------
    h : np.ndarray
        Channel matrix (Nt, N, nUsers).
    N : int
        Number of subcarriers.
    nUsers : int
        Total number of users.
    matching_matrix : np.ndarray
        Matrix [Subcarrier, User1, User2, Flag] defining active pairs.
    uj : np.ndarray
        User priority weights.
    P : np.ndarray
        Power allocation vector per subcarrier.

    Returns
    -------
    best_t : np.ndarray
        Vector of optimal splitting fractions t for each subcarrier.
        P_private = t * P_total
        P_common = (1-t) * P_total
    """

    comb = int((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    
   # Initialize output vector
    rate_per_subcarrier = np.zeros(N)
    best_t = np.zeros(N)
    
    for n in range(N):
        # --- 1. Identify Active User Pair ---
        # Filter for the current subcarrier and active flag
        aux = np.where(x[:, 0].astype(int) == int(n+1))[0];
        idx = int(np.where(x[aux[0]:aux[comb-1]+1,3] != 0)[0]);
        ii = int(x[aux[idx],1]-1); # decrease in 1 because was increased 1 when the matrix was it built
        jj = int(x[aux[idx],2]-1);
        hni = h[:, n, ii]
        hnj = h[:, n, jj]
        
        # --- 2. Order Users (Strong vs Weak) ---
        # This order is crucial for the SIC decoding assumption in the formulas
        strongest = max(np.linalg.norm(hni), np.linalg.norm(hnj))
        if strongest == np.linalg.norm(hni):
            hn1, hn2 = hni, hnj
            u1, u2 = uj[ii], uj[jj]
        else:
            hn1, hn2 = hnj, hni
            u1, u2 = uj[jj], uj[ii]
        
        # --- 3. Calculate Geometry Parameters ---
        # Normalized Directions 
        h_n1 = hn1 / np.linalg.norm(hn1)
        h_n2 = hn2 / np.linalg.norm(hn2)
        
        # Rho (Orthogonality factor): 1 - |h1.h2|^2
        rho = (1-(np.abs(np.dot(np.conj(h_n1).T , h_n2))**2));
        # Gamma: A threshold parameter derived from the WSR derivative conditions
        # Determines which interference regime the system is in.
        Gamma = (1 / rho) * ((u1 / np.linalg.norm(hn2)**2) - (u2 / np.linalg.norm(hn1)**2))
        
        t0 = np.random.rand()
        # --- 4. Solve for Optimal t ---
        if P[n] > 0:
            t_optimal = optimal_fraction_t(P[n], hn1, hn2, rho, Gamma, np.random.rand(), u1, u2)
        else:
            t_optimal = -np.inf
        
        best_t[n] = t_optimal
    
    return best_t


def optimal_fraction_t(P, hn1, hn2, rho, Gamma, t0, u1, u2):
    """
    Solves the closed-form quadratic equation for the optimal power split 't'.
    
    The Weighted Sum-Rate (WSR) function in RSMA is non-convex, but its 
    stationary points can be found by solving roots of a polynomial.
    The solution space is divided into 3 regimes based on weights and channel gains.
    """
    t = np.linspace(0.001, 1, 1000)
    Pk = t0 * P
    Pc = (1 - t0) * P
    g1 = (np.linalg.norm(hn1) ** 2) * rho
    g2 = (np.linalg.norm(hn2) ** 2) * rho
    g = np.array([g1, g2])
    WF = waterFilling.WF(Pk,g,u1,u2);
    P1 = WF[0]
    P2 = WF[1]
    mu = (1 / (u1 + u2)) * (t0 * P + (1 / g1) + (1 / g2))
    best_rate = np.zeros(len(t))
    possibles_t = np.zeros(2)
    h_n1 = hn1 / np.linalg.norm(hn1)
    h_n2 = hn2 / np.linalg.norm(hn2)
    fc = (h_n1 + (h_n2 * np.exp(-1j * np.angle(np.conjugate(h_n1).dot(h_n2))))) * (1 / (np.sqrt(2 * (1 + abs(np.conjugate(h_n1).dot(h_n2))))))

    # OMA/NOMA/Multicasting Regime
    if (t0 * P * u2) <= Gamma:
        num = ((np.linalg.norm(hn1) ** 2) - (np.linalg.norm(hn2) ** 2))
        den = (np.linalg.norm(hn1) ** 2) * (np.linalg.norm(hn2) ** 2) * rho * P
        a = (np.linalg.norm(hn1) ** 2) * rho * P
        if t0 <= (num / den):
            b = 1 + (abs(np.conjugate(hn2).dot(fc)) ** 2) * P
            c = b - 1
            tx = ((-c * (u1 + u2) + (2 * u1 * a * b)) / (3 * u1 * a * c + u2 * a * c))
            
            if (((u1 * a) / (np.log(2) * (1 + a * tx)) + ((u2 + u1) / 2) * (-c / (np.log(2) * (b - c * tx))) <= 1e-6) and 
                ((1 + a * tx) >= 0) and ((b - c * tx) >= 0)):
                t_optimal = max(min(tx, 1), 0)
            else:
                rate1 = u1 * np.log2(1 + a * 0) + ((u2 + u1) / 2) * np.log2(b - c * 0)
                rate2 = u1 * np.log2(1 + a * 1) + ((u2 + u1) / 2) * np.log2(b - c * 1)
                t_optimal = 0 if rate1 >= rate2 else 1
            rate = np.log2(1 + a * t_optimal) + np.log2(b - c * t_optimal)
        else:
            b = 1 + (abs(np.conjugate(hn1).dot(fc)) ** 2) * P
            c = 1 + a - b
            tx = ((u2 * (((a * b) - c) - u1 * ((a * b) + c))) / (2 * u1 * a * c))
            if (((((u1 - u2) / 2) * (a / (np.log(2) * (1 + a * tx))) + 
                 ((u1 + u2) / 2) * (c / (np.log(2) * (b + c * tx)))) <= 1e-6) and 
                ((1 + a * tx) >= 0) and ((b + c * tx) >= 0)):
                t_optimal = max(min(tx, 1), 0)
            else:
                rate1 = ((u1 - u2) / 2) * np.log2(1 + a * 0) + ((u1 + u2) / 2) * np.log2(b + c * 0)
                rate2 = ((u1 - u2) / 2) * np.log2(1 + a * 1) + ((u1 + u2) / 2) * np.log2(b + c * 1)
                t_optimal = 0 if rate1 >= rate2 else 1
            rate = np.log2(1 + a * t_optimal) + np.log2(b + c * t_optimal)
    
    elif (t0 * P * u1) < -Gamma:
        if (np.linalg.norm(hn2) ** 2) == (np.linalg.norm(hn1) ** 2):
            raise Exception("Equal norms detected")
        
        a = (np.linalg.norm(hn2) ** 2) * rho * P
        b = 1 + (abs(np.conjugate(hn2).dot(fc)) ** 2) * P
        c = 1 + a - b
        tx = (((u1 * (a * b - c)) - (u2 * (a * b + c))) / (2 * a * c * u2))

        if (((((u2 - u1) / 2) * (a / (np.log(2) * (1 + a * tx))) + 
             ((u2 + u1) / 2) * (c / (np.log(2) * (b + c * tx)))) <= 1e-6) and 
            ((1 + a * tx) >= 0) and ((b + c * tx) >= 0)):
            t_optimal = max(min((((u1 * (a * b - c)) - (u2 * (a * b + c))) / (2 * a * c * u2)), 1), 0)
        else:
            rate1 = ((u2 - u1) / 2) * np.log2(1 + a * 0) + ((u2 + u1) / 2) * np.log2(b + c * 0)
            rate2 = ((u2 - u1) / 2) * np.log2(1 + a) + ((u2 + u1) / 2) * np.log2(b + c)
            t_optimal = 0 if rate1 >= rate2 else 1
        rate = np.log2(1 + a * t_optimal) + np.log2(b + c * t_optimal)

    if ((t0 * P * u2) > Gamma) and ((t0 * P * u1) > -Gamma):
        b = ((np.linalg.norm(hn1) ** 2) * rho * P * u1) / (u1 + u2)
        a = 1 + ((b * Gamma) / (P * u1))
        d = ((np.linalg.norm(hn2) ** 2) * rho * P * u2) / (u1 + u2)
        c = 1 - ((d * Gamma) / (P * u2))
        
        if u2 >= u1:
            e = (abs(np.conjugate(hn2).dot(fc)) ** 2) * P
            f = c + e
            
            alpha = 4 * (u1 + u2) * ((b * d ** 2) - (b * d * e))
            beta = (u1 * ((2 * b * d * f) + (6 * b * c * d) - (6 * b * c * e)) + 
                   u2 * ((2 * b * d * f) + (4 * a * d ** 2) - (4 * a * d * e) - (2 * b * c * e) + (2 * b * c * d)))
            eta = (u1 * ((4 * b * c * f) - (2 * a * d * f) - (2 * a * c * e) + (2 * a * c * d)) + 
                  u2 * ((2 * a * d * f) - (2 * a * c * e) + (2 * a * c * d)))
            
            possibles_t[0] = (-beta + np.sqrt((beta ** 2) - (4 * alpha * eta))) / (2 * alpha)
            possibles_t[1] = (-beta - np.sqrt((beta ** 2) - (4 * alpha * eta))) / (2 * alpha)
            positive_possibles_t = possibles_t[possibles_t > 0]
            
            if len(positive_possibles_t) == 0:
                rate1 = (u1 * np.log(a + b * 0.01) / np.log(2) + 
                        (u2 - u1) / 2 * np.log(c + d * 0.01) / np.log(2) + 
                        (u1 + u2) / 2 * np.log(f + (d - e) * 0.01) / np.log(2))
                rate2 = (u1 * np.log(a + b) / np.log(2) + 
                        (u2 - u1) / 2 * np.log(c + d) / np.log(2) + 
                        (u1 + u2) / 2 * np.log(f + (d - e)) / np.log(2))
                t_optimal = 0.001 if rate1 >= rate2 else 1
            else:
                t_optimal = min(min(positive_possibles_t), 1)
        else:
            e = (abs(np.conjugate(hn1).dot(fc)) ** 2) * P
            f = a + e
            
            alpha = 4 * (u1 + u2) * (d * b ** 2 - b * d * e)
            beta = (u1 * (4 * c * b ** 2 - 4 * b * c * e + 2 * b * d * f - 2 * a * d * e + 2 * a * b * d) + 
                   u2 * (6 * a * b * d - 6 * a * d * e + 2 * b * d * f))
            eta = (u1 * (2 * b * c * f - 2 * a * c * e + 2 * a * b * c) + 
                  u2 * (4 * a * d * f + 2 * a * b * c - 2 * b * c * f - 2 * a * c * e))
            
            possibles_t[0] = (-beta + np.sqrt((beta ** 2) - (4 * alpha * eta))) / (2 * alpha)
            possibles_t[1] = (-beta - np.sqrt((beta ** 2) - (4 * alpha * eta))) / (2 * alpha)
            positive_possibles_t = possibles_t[possibles_t > 0]
            
            if len(positive_possibles_t) == 0:
                rate1 = ((u1 - u2) / 2 * np.log(a + b * 0.01) / np.log(2) + 
                        u2 * np.log(c + d * 0.01) / np.log(2) + 
                        (u1 + u2) / 2 * np.log(f + (b - e) * 0.01) / np.log(2))
                rate2 = ((u1 - u2) / 2 * np.log(a + b) / np.log(2) + 
                        u2 * np.log(c + d) / np.log(2) + 
                        (u1 + u2) / 2 * np.log(f + (b - e)) / np.log(2))
                t_optimal = 0.01 if rate1 >= rate2 else 1
            else:
                t_optimal = min(min(positive_possibles_t), 1)

        if t_optimal < 0:
            raise Exception("Negative t_optimal detected")
            
    return t_optimal
