import numpy as np
from scipy.linalg import inv
from scipy.optimize import minimize

# Função objetivo para a otimização
def objective(P, h1, h2, f_c, u1, u2, rho_n):
    N = len(P)  # Número de subportadoras
    Pn_p = P[:(N//2)]  # Potência privada
    Pn_c = P[(N//2):]  # Potência comum
    
    Wn = 0
    for n in range(N//2):
        term1 = 1 * np.log2(1 + np.abs(h1[n])**2 * rho_n * (Pn_p[n] / 2))
        term2 = (1 - 1) / 2 * np.log2(1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2))
        term3 = (1 + 1) / 2 * np.log2(1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2) + np.abs(np.dot(h2[n].conj(), f_c[n]))**2 * Pn_c[n])
        
        Wn += term1 + term2 + term3
    
    return -Wn  # Negativo para maximizar o somatório

# Função gradiente
def gradient(P, h1, h2, f_c, u1, u2, rho_n):
    N = len(P)
    Pn_p = P[:(N//2)]
    Pn_c = P[(N//2):]
    
    grad = np.zeros_like(P)
    
    # Gradiente para Pn_p e Pn_c
    for n in range(N//2):
        # Gradiente em relação a Pn_p
        grad[n] = -1 * np.abs(h1[n])**2 * rho_n / (2 * (1 + np.abs(h1[n])**2 * rho_n * (Pn_p[n] / 2))) / np.log(2)
        grad[n] += -(1 - 1) * np.abs(h2[n])**2 * rho_n / (2 * (1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2))) / np.log(2)
        grad[n] += -(1 + 1) * np.abs(h2[n])**2 * rho_n / (2 * (1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2) + np.abs(np.dot(h2[n].conj(), f_c[n]))**2 * Pn_c[n])) / np.log(2)
        
        # Gradiente em relação a Pn_c
        grad[N + n] = -(1 + 1) * np.abs(np.dot(h2[n].conj(), f_c[n]))**2 * rho_n / (2 * (1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2) + np.abs(np.dot(h2[n].conj(), f_c[n]))**2 * Pn_c[n])) / np.log(2)
    
    return grad

# Função Hessiana
def hessian(P, h1, h2, f_c, u1, u2, rho_n):
    N = len(P)
    Pn_p = P[:(N//2)]
    Pn_c = P[(N//2):]
    
    H = np.zeros((2 * N, 2 * N))
    
    # Hessiana para Pn_p e Pn_c
    for n in range(N//2):
        # Hessiana em relação a Pn_p
        H[n, n] = 1 * np.abs(h1[n])**2 * rho_n / (2 * (1 + np.abs(h1[n])**2 * rho_n * (Pn_p[n] / 2))**2) / np.log(2)
        H[n, n] += (1 - 1) * np.abs(h2[n])**2 * rho_n / (2 * (1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2))**2) / np.log(2)
        H[n, n] += (1 + 1) * np.abs(h2[n])**2 * rho_n / (2 * (1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2) + np.abs(np.dot(h2[n].conj(), f_c[n]))**2 * Pn_c[n])**2) / np.log(2)
        
        # Hessiana em relação a Pn_c
        H[N + n, N + n] = (1 + 1) * np.abs(np.dot(h2[n].conj(), f_c[n]))**2 * rho_n / (2 * (1 + np.abs(h2[n])**2 * rho_n * (Pn_p[n] / 2) + np.abs(np.dot(h2[n].conj(), f_c[n]))**2 * Pn_c[n])**2) / np.log(2)
    
    return H

# Função para otimização usando Newton-Raphson
def newton_raphson(h1, h2, f_c, u1, u2, rho_n, Pmax, max_iter=100, tol=1e-6):
    N = len(h1)
    P0 = np.ones(2 * N)  # Inicialização com potências positivas

    for iter in range(max_iter):
        # Cálculo da função objetivo, gradiente e Hessiana
        obj = objective(P0, h1, h2, f_c, u1, u2, rho_n)
        grad = gradient(P0, h1, h2, f_c, u1, u2, rho_n)
        H = hessian(P0, h1, h2, f_c, u1, u2, rho_n)
        
        # Atualização de Newton-Raphson
        P_new = P0 - np.linalg.inv(H).dot(grad)
        
        # Imposição das restrições
        P_new = np.clip(P_new, 0, Pmax)  # Pn_p e Pn_c devem estar dentro de 0 e Pmax
        
        # Verificação da convergência
        if np.linalg.norm(grad) < tol:
            break
        
        P0 = P_new  # Atualizar P0 para a próxima iteração
    
    return P_new

# Exemplo de uso:
h1 = np.random.rand(10)  # Vetor de canais para o primeiro usuário
h2 = np.random.rand(10)  # Vetor de canais para o segundo usuário
f_c = np.random.rand(10)  # Vetor de canais comuns
u1 = np.random.rand(10)  # Vetor de pesos u1
u2 = np.random.rand(10)  # Vetor de pesos u2
rho_n = 1  # Fator de escalonamento
Pmax = 10  # Potência máxima

# Chamar a função de otimização
P_opt = newton_raphson(h1, h2, f_c, u1, u2, rho_n, Pmax)

print("Potências ótimas: ", P_opt)
