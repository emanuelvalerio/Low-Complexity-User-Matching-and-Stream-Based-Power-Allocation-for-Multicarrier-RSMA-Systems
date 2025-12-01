import numpy as np
import math
def commomRate(h,Pnc,n,ii,jj,gamma1,gamma2,rho):
    """
    Calculates the Achievable Rate for the Common Message (RSMA).
    
    This function computes the Signal-to-Interference-plus-Noise Ratio (SINR) components
    (Numerator and Denominator) and the resulting Shannon capacity.

    Parameters
    ----------
    h : np.ndarray
        Channel state information matrix of shape (N_rx, N_subcarriers, N_users).
    Pnc : float
        Power allocated to the common message on subcarrier n.
    n : int
        Index of the subcarrier.
    ii : int
        Index of the first user (User 1).
    jj : int
        Index of the second user (User 2).
    gamma1 : float
        SINR/Channel parameter (weight) for User 1.
    gamma2 : float
        SINR/Channel parameter (weight) for User 2.
    rho : float
        Correlation/Orthogonality factor.

    Returns
    -------
    Nnij : float
        The Numerator term (Signal Power component).
    Dnij : float
        The Denominator term (Interference + Noise component).
    R_common : float
        The calculated common rate in bits/s/Hz (log2(1 + N/D)).
    """
    
    hni = h[:,n,ii];
    hnj = h[:,n,jj];
    Nnij = rho*(np.linalg.norm(hni)**2)*(np.linalg.norm(hnj)**2)*Pnc;
    Dnij = (gamma1*(np.linalg.norm(hnj)**2))+(gamma2*(np.linalg.norm(hni)**2))-(2*(math.sqrt(gamma1)*math.sqrt(gamma2)*np.abs(np.dot(np.conj(hni).T,hnj))));
    return Nnij,Dnij,math.log2(1+(Nnij/Dnij));