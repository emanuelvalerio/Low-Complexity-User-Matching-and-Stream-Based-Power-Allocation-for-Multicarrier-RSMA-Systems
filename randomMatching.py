import numpy as np
import math
import pandas as pd
import random

def randomUserMatching(combVect,N,nUsers):
    """
    Benchmark Algorithm: Random User Pairing.
    
    This function randomly selects one user pair for each subcarrier from the 
    pool of possible combinations. It serves as a baseline (lower bound) to 
    evaluate the performance of intelligent matching algorithms.

    Parameters
    ----------
    combVect : np.ndarray
        Matrix containing all possible user pairs for all subcarriers.
        Shape: (N * n_combinations, 3).
        Columns: [Subcarrier_Index, User1, User2].
    N : int
        Number of subcarriers.
    nUsers : int
        Number of users.

    Returns
    -------
    allocation_matrix : np.ndarray
        The input matrix with an appended 4th column (binary flag).
        Shape: (N * n_combinations, 4).
        Columns: [Subcarrier, User1, User2, Active_Flag].
        
        The 'Active_Flag' is 1 for the randomly selected pair on that subcarrier,
        and 0 for all other pairs.
    """
    comb = int((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    x = [];
    for n in range(N):
        auxMatrix = combVect[n*comb:(n+1)*comb,:].copy();
        np.random.shuffle(auxMatrix);
        aux = np.zeros((comb,1));
        idx_sorted = random.randint(0, comb-1);
        aux[idx_sorted] = 1;
        x.extend( np.hstack((auxMatrix, aux)));
    df = pd.DataFrame(x) # Convert the numpy array to a pandas DataFrame
    return df.sort_values(by=[0, 1, 2]).values # Sort by the first, second, and third columns
    