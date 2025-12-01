import numpy as np

def WF(Pp,g,u1,u2):
    """
    Performs Weighted Water Filling power allocation for exactly 2 streams.

    Solves the optimization problem:
        Maximize: u1*log2(1 + g1*p1) + u2*log2(1 + g2*p2)
        Subject to: p1 + p2 = P_total
                    p1, p2 >= 0

    Parameters
    ----------
    P_total : float
        Total private power budget available for the pair.
    g : np.ndarray or list
        Channel gains (normalized by noise) for the two streams [g1, g2].
    u1 : float
        Weight (priority) for User 1.
    u2 : float
        Weight (priority) for User 2.

    Returns
    -------
        Array of shape (2,) containing [Power1, Power2].
    """
    initPowerAllocation = np.zeros((2,1));
    initPowerAllocation[0] = (1/(u1+u2))*((u1*Pp) + (u1/g[1]) - (u2/g[0]) );
    initPowerAllocation[1] = (1/(u1+u2))*((u2*Pp) + (u2/g[0]) - (u1/g[1]) );
    while np.any(initPowerAllocation < 0): # while there's negative power alocated
        indxNeg = np.where(initPowerAllocation < 0)[0]; # store the indice of negative power allocated
        indPos = np.where(initPowerAllocation >= 0)[0]; # store the indice of positive or null power allocated
        nPositiveSubChannel = len(indPos); # stores the number of positive power allocated
        initPowerAllocation[indxNeg] = 0; # change the value of negative power to zero.
        newSNR = g[indPos]; # takes only the SNRs in which the positive power was allocated
        auxPowerAllocation = ((Pp + np.sum(1/newSNR))/nPositiveSubChannel) - 1/(newSNR);
        initPowerAllocation[indPos] = auxPowerAllocation; # adds to the vector of powers allocated from the current positive index
    return initPowerAllocation;

