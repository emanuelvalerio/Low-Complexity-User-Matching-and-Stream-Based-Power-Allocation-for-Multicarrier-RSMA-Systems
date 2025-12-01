import numpy as np
import math 
def combVector(nUsers,N):
    """
    Generates a matrix containing all possible user pairs for all subcarriers.
    
    This function creates a lookup table where each row represents a specific 
    combination of 2 users assigned to a specific subcarrier.

    Parameters
    ----------
    nUsers : int
        Total number of users.
    N : int
        Total number of subcarriers.

    Returns
    -------
    combMatrix : np.ndarray
        Matrix of shape (N * n_combinations, 3) with integer type.
        Columns structure:
        - Col 0: Subcarrier Index (1-based)
        - Col 1: User 1 Index (1-based)
        - Col 2: User 2 Index (1-based)
    """
    
    comb = int((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    combV= np.zeros((N*comb,3)); 
    aux = 0;
    for n in range(0,N):
        prev = 0;
        for ii in range(0,comb):
            for jj in range( prev+1,nUsers):
                combV[aux,0] = int(n+1);
                combV[aux,1] = int(prev+1);
                combV[aux,2] = int(jj+1);
                aux += 1;
            prev += 1;
         
            if aux > comb*N:
                break;
    return combV;