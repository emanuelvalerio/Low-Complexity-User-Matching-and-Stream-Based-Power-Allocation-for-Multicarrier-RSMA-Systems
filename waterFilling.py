import numpy as np

def WF(Pp,g,u1,u2):
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

