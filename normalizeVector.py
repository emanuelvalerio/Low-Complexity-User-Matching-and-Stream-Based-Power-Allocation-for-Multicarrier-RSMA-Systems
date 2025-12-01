import numpy as np

def normalization(h,n,idx):
    """
    Extracts and normalizes the channel vector for a specific user and subcarrier.
    
    This converts the Channel State Information (CSI) into 
    Channel Direction Information (CDI) by scaling the vector to unit norm.

    Parameters
    ----------
    h : np.ndarray
        Channel matrix of shape (Nt, N_subcarriers, N_users).
    n : int
        Index of the subcarrier.
    user_idx : int
        Index of the user.

    Returns
    -------
    h_normalized : np.ndarray
        The normalized channel vector (Unit Norm).
        Returns a zero vector if the input vector has zero norm.
    """
    h_normalized = h[:,n,idx]/np.linalg.norm(h[:,n,idx])
    return h_normalized
    