import numpy as np
import pandas as pd
import random

def distance(nUsers,dOuterRadius,dInnerRadius):
    """
    Generates random user distances uniformly distributed within a ring (annulus).

    This simulates users in a cell where they cannot be closer to the Base Station 
    than 'dInnerRadius' and no further than 'dOuterRadius'.

    Parameters
    ----------
    nUsers : int
        Number of users to generate.
    dOuterRadius : float
        The maximum radius of the cell (cell edge).
    dInnerRadius : float
        The minimum radius (minimum distance from BS).

    Returns
    -------
    dist : np.ndarray
        Column vector (nUsers, 1) containing the Euclidean distance 
        of each user from the origin.
    """
    theta = 2*np.pi * (np.random.rand(nUsers,1));
    r = (dOuterRadius-dInnerRadius) * np.sqrt(np.random.rand(nUsers, 1)) + dInnerRadius;
    x = r * np.cos(theta);
    y = r * np.sin(theta);
    return np.sqrt(x**2 + y**2);