import numpy as np
def calculationFc(h_1,h_2):
    inner_product = np.dot(h_1.conj().T, h_2) 
    angle_inner_product = np.angle(inner_product) # Calculando fc 
    return ((h_1 + (h_2 * np.exp(-1j * angle_inner_product))) * (1 / (np.sqrt(2 * (1 + np.abs(inner_product))))));
