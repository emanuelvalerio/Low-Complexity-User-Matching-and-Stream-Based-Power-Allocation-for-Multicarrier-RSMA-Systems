import pandas as pd
import numpy as np
import distances as dt
import channelModel as ch
import lowComplexityUserMatching as uMatch
import combinationsVectors as combV
import optimalPA as optPA
import rateCalculation
import unimodularMatrixBenchmark as benchmark
import cvxPA

# Configurações do sistema
Nt = 4  # Número de antenas no transmissor
Pmax = 100  # Potência total disponível
nUsers = 8 # Número de usuários
dInnerRadius = 1
dOuterRadius = 10
gamma = 3  # Expoente de perda de caminho
num_iterations = 1000  # Número de repetições
epsilon = 1e-4
uj = np.ones((nUsers, 1))  # Vetor de pesos

# Intervalo para o número de subportadoras
N_values = [16]  # Lista com os valores de N a serem testados

# Lista para armazenar os resultados
results = []

# Loop sobre os valores de N
for N in N_values:
    print(f"\nIniciando experimentos para N = {N} subportadoras...")
    Pn = [(Pmax / N) * x for x in np.ones((N, 1))]  # Potência inicial por subportadora

    # Repetição dos experimentos
    for iteration in range(num_iterations):
        # Geração do canal
        h = ch.channel(Nt, N, nUsers, gamma, dOuterRadius, dInnerRadius)

        # Algoritmos de alocação de usuários
        x_lc = uMatch.userMatchingAlgorithm(h, N, nUsers, uj)
        x_tum = benchmark.unimodularMatrixUserMatching(h, Pmax, N, nUsers, uj)
        x_tum = x_tum.reshape((-1,1));
        # Combinação de vetores
        combVect = combV.combVector(nUsers, N)
        userMatch = np.hstack((combVect, x_lc))

        # Alocação de potência
        P_opt1 = optPA.optimizedPowerAllocation(h, userMatch, uj, N, nUsers, Pmax)
        P_opt2 = optPA.optimizedPowerAllocation(h, np.hstack((combVect, x_tum)), uj, N, nUsers, Pmax)
        P_opt3 = cvxPA.optimizedPowerAllocation(h,userMatch,uj,N,nUsers,Pmax);

        # Cálculo da taxa
        userRate_lc = rateCalculation.ASPA(h, userMatch, P_opt1, uj, N, nUsers)
        userRate_tum = rateCalculation.ASPA(h, np.hstack((combVect, x_tum)), P_opt2, uj, N, nUsers)
        userRate_lc_cvx = rateCalculation.ASPA(h, userMatch, P_opt3, uj, N, nUsers)

        # Armazenar os resultados
        results.append({
            "N": N,
            "Iteration": iteration + 1,
            "Sum Rate LC": np.sum(userRate_lc),
            "Sum Rate TUM": np.sum(userRate_tum)
        })

        # Feedback no console
        if (iteration + 1) % 100 == 0 or iteration == 0:
            print(f"Iteração {iteration + 1}/{num_iterations} - N = {N}, LC: {np.sum(userRate_lc):.4f}, TUM: {np.sum(userRate_tum):.4f}")

# Conversão para DataFrame
df_results = pd.DataFrame(results)

# Salvar os resultados em um arquivo CSV
df_results.to_csv("Results/sum_rate_results_varying_16.csv", index=False)

# Exibir resumo final
summary = df_results.groupby("N")[["Sum Rate LC", "Sum Rate TUM"]].mean()
print("\nResumo dos Resultados (Médias):")
print(summary)
print("\nResultados salvos em 'sum_rate_results_varying_N.csv'")
