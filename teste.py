import cvxpy as cp
import numpy as np
import variablesPA

# Parâmetros do problema
N = 5  # Número de subportadoras
P_total = 10  # Potência total

# Parâmetros específicos para o cálculo da função objetivo
u_1 = np.random.rand(N)  # Pesos para os usuários
u_2 = np.random.rand(N)
h_1 = np.random.rand(N) + 1j * np.random.rand(N)  # Canais para o usuário 1
h_2 = np.random.rand(N) + 1j * np.random.rand(N)  # Canais para o usuário 2
rho = np.random.rand(N)  # Fator de escalonamento do canal
f_c = np.random.rand(N) + 1j * np.random.rand(N)  # Vetores beamforming

# Variáveis de decisão
P_p = cp.Variable(N, nonneg=True)  # Potência privada para cada subportadora
P_c = cp.Variable(N, nonneg=True)  # Potência comum para cada subportadora

# Cálculo da função objetivo
W = []
for n in range(N):
    u1,u2,g1,g2,alpha = variablesPA.varCalculate_per_subcarrier(h,)
    term1 = u_1[n] * cp.log(1 + cp.norm(h_1[n])**2 * rho[n] * P_p[n] / 2) / cp.log(2)
    term2 = (u_2[n] - u_1[n]) / 2 * cp.log(1 + cp.norm(h_2[n])**2 * rho[n] * P_p[n] / 2) / cp.log(2)
    term3 = (u_1[n] + u_2[n]) / 2 * cp.log(
        1 + cp.norm(h_2[n])**2 * rho[n] * P_p[n] / 2 + cp.norm(cp.conj(h_2[n]).T @ f_c[n])**2 * P_c[n]
    ) / cp.log(2)
    W.append(term1 + term2 + term3)

objective = cp.Maximize(cp.sum(W))

# Restrições
constraints = [
    cp.sum(P_p + P_c) <= P_total,  # Restrição de potência total
    P_p >= 0,  # Potência privada não negativa
    P_c >= 0,  # Potência comum não negativa
]

# Problema de otimização
problem = cp.Problem(objective, constraints)

# Resolvendo o problema
problem.solve()

# Resultados
print("Status:", problem.status)
print("Valor ótimo da função objetivo:", problem.value)
print("Potências privadas:", P_p.value)
print("Potências comuns:", P_c.value)
