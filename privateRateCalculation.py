import numpy as np
import math
import normalizeVector as norm
def privateRate(h,n,ii,jj,Pn1,Pn2):
    """
    Calculates the Private Rates for a pair of users using Zero-Forcing (ZF) 
    or Orthogonal Projection logic.

    The private rate is calculated based on the effective channel gain after 
    projecting onto the orthogonal subspace of the paired user (to mitigate interference).

    Parameters
    ----------
    h : np.ndarray
        Channel state information matrix (Nt, N_subcarriers, N_users).
    n : int
        Index of the current subcarrier.
    ii : int
        Index of the first user (User 1).
    jj : int
        Index of the second user (User 2).
    Pn1 : float
        Power allocated to the private stream of User 1.
    Pn2 : float
        Power allocated to the private stream of User 2.

    Returns
    -------
    pRate1 : float
        Achievable private rate for User 1 (bits/s/Hz).
    pRate2 : float
        Achievable private rate for User 2 (bits/s/Hz).
    gamma1 : float
        (1 + SINR) term for User 1.
    gamma2 : float
        (1 + SINR) term for User 2.
    rho : float
        The orthogonality factor (1 - correlation^2). 
        Ranges from 0 (collinear) to 1 (perfectly orthogonal).
    """
    hni = h[:,n,ii];
    hnj = h[:,n,jj]; 
    h_hni  =norm.normalization(h,n,ii);
    h_hnj  = norm.normalization(h,n,jj);
    rho = (1-(np.abs(np.dot(np.conj(h_hni).T , h_hnj))**2));
    gamma1 = 1+((np.linalg.norm(hni)**2)*rho*Pn1);
    gamma2 = 1+((np.linalg.norm(hnj)**2)*rho*Pn2);
    pRate1 = math.log2(gamma1);
    pRate2 = math.log2(gamma2);
    return pRate1,pRate2,gamma1,gamma2,rho