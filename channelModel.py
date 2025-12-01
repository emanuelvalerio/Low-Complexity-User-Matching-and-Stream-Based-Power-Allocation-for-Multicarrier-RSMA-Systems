import numpy as np
import pandas as pd
import distances as dt
def channel(Nt,N,nUsers,gamma,dOuterRadius,dInnerRadius):
   """
    Generates the Channel State Information (CSI) matrix incorporating 
    Small-Scale Fading (Rayleigh) and Large-Scale Fading (Path Loss).

    Model: h = sqrt(PathLoss) * h_small_scale

    Parameters
    ----------
    Nt : int
        Number of Transmit Antennas (Base Station).
    N : int
        Number of Subcarriers.
    nUsers : int
        Number of Users.
    gamma : float
        Path Loss Exponent (typically between 2.0 and 4.0).
    dOuterRadius : float
        Cell outer radius.
    dInnerRadius : float
        Cell inner radius.

    Returns
    -------
    h : np.ndarray
        Complex channel matrix of shape (Nt, N, nUsers).
    """
   d = dt.distance(nUsers,dOuterRadius,dInnerRadius);
   h = np.zeros((Nt,N,nUsers), dtype=complex);
   for jj in range(0,nUsers):
      variance = 1/(Nt * (d[jj]**gamma));
      for n in range(0,N):
          h[:,n,jj] = (np.sqrt(variance/2))*(np.random.randn(1,Nt) + 1j*np.random.randn(1,Nt));
   return h