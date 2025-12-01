import numpy as np
def calculationFc(h_1,h_2):
    """
    Calculates the normalized composite precoder (beamforming vector) for a pair of users.
    (Equation 5 from the reference paper).

    This function aligns the phase of h2 to match h1 and combines them into 
    a single unit-norm vector fc. This is typically used for the Common Message in RSMA.

    Parameters
    ----------
    h1 : np.ndarray
        Channel vector for User 1 (expected unit norm direction).
    h2 : np.ndarray
        Channel vector for User 2 (expected unit norm direction).

    Returns
    -------
    f_c : np.ndarray
        The normalized composite beamforming vector.
    """
    inner_product = np.dot(h_1.conj().T, h_2) 
    angle_inner_product = np.angle(inner_product) # Calculate fc (Equation 5 from paper)
    f_c = ((h_1 + (h_2 * np.exp(-1j * angle_inner_product))) * (1 / (np.sqrt(2 * (1 + np.abs(inner_product))))))
    return f_c
