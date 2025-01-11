import cvxpy as cp
import numpy as np
import variablesPA

def optimizedPowerAllocation(h, combVect, uj, N, nUsers, Pmax):
    # Calcular os parâmetros necessários
    # Variáveis de decisão
    P_p = cp.Variable(N, nonneg=True)  # Potência privada para cada subportadora
    P_c = cp.Variable(N, nonneg=True)  # Potência comum para cada subportadora

    # Cálculo da função objetivo
    W = []
    for n in range(N):
        u_1,u_2,g1,g2,alpha = variablesPA.varCalculate_per_subcarrier(h,combVect,uj,n,nUsers);
        term1 = u_1 * cp.log(1 + g1 * P_p[n])/np.log(2);
        term2 = ((u_2 - u_1) / 2) * (cp.log(1 + g2 *  P_p[n]) / np.log(2));
        term3 = ((u_1+u_2) / 2) * (cp.log(1 + g2 *  P_p[n] + alpha *  P_c[n])/np.log(2));
        W.append(term1 + term2 + term3)

    objective = cp.Maximize(cp.sum(W))

    # Restrições
    constraints = [
        cp.sum(P_p) +cp.sum(P_c) <= Pmax,  # Restrição de potência total
        P_p >= 0,  # Potência privada não negativa
        P_c >= 0,  # Potência comum não negativa
    ]

    # Problema de otimização
    problem = cp.Problem(objective, constraints)

    # Resolvendo o problema
    problem.solve()
    Pn = np.zeros((2*N,1));
    Pn[:N] = (P_p.value).reshape(-1, 1);
    Pn[N:] = P_c.value.reshape(-1, 1);
    return Pn