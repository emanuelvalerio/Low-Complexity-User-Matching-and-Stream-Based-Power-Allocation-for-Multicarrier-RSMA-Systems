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
import randomMatching as randMatch
import gradientDescent as gradDes
import streamsPowerAllocation as streamPA

# Configurações do sistema
Nt = 4  # Número de antenas no transmissor
Pmax = 10  # Potência total disponível
nUsers = 8 # Número de usuários
dInnerRadius = 1
dOuterRadius = 10
gamma = 3  # Expoente de perda de caminho
num_iterations = 1000  # Número de repetições
epsilon = 1e-4
uj = np.ones((nUsers, 1))  # Vetor de pesos

# Intervalo para o número de subportadoras
N_values = [2]  # Lista com os valores de N a serem testados

# Lista para armazenar os resultados
results = []

# Repetição dos experimentos
for N in N_values:
    print(f"\nIniciando experimentos para N = {N} subportadoras...")
    Pn = [(Pmax / N) * x for x in np.ones((N, 1))]  # Potência inicial por subportadora

    # Inicializa listas para acumular as taxas para cada algoritmo
    sum_rates_streamPA_rand = []
    sum_rates_streamPA_tum = []
    sum_rates_streamPA_lc = []
    sum_rates_EPA_rand = []
    sum_rates_EPA_tum = []
    sum_rates_EPA_lc = []
    sum_rates_ASPA_rand = []
    sum_rates_ASPA_tum = []
    sum_rates_ASPA_lc = []

    # Repetição dos experimentos
    for iteration in range(num_iterations):
        print(iteration)
        # Geração do canal
        h = ch.channel(Nt, N, nUsers, gamma, dOuterRadius, dInnerRadius)

        # Algoritmos de alocação de usuários
        x_lc = uMatch.userMatchingAlgorithm(h, N, nUsers, uj)
        x_tum = benchmark.unimodularMatrixUserMatching(h, Pmax, N, nUsers, uj)
        x_tum = x_tum.reshape((-1, 1))
        
        # Combinação de vetores
        combVect = combV.combVector(nUsers, N)
        userMatch_lc = np.hstack((combVect, x_lc))
        userMatch_rand = randMatch.randomUserMatching(combVect, N, nUsers)
        userMatch_tum = np.hstack((combVect, x_tum))
        
        # Alocação de potência
        P_opt_rand = cvxPA.optimizedPowerAllocation(h, userMatch_rand, uj, N, nUsers, Pmax)
        P_opt_tum = cvxPA.optimizedPowerAllocation(h, userMatch_tum, uj, N, nUsers, Pmax)
        P_opt_lc = cvxPA.optimizedPowerAllocation(h, userMatch_lc, uj, N, nUsers, Pmax)
        
        P_EPA_rand = [(Pmax / (2 * N)) * x for x in np.ones((2 * N))]
        P_EPA_tum = [(Pmax / (2 * N)) * x for x in np.ones((2 * N, 1))]
        P_EPA_lc = [(Pmax / (2 * N)) * x for x in np.ones((2 * N, 1))]

        P_ASPA_rand = gradDes.gradDes(h, userMatch_rand, nUsers, Pmax, N, uj, epsilon)
        P_ASPA_tum = gradDes.gradDes(h, userMatch_tum, nUsers, Pmax, N, uj, epsilon)
        P_ASPA_lc = gradDes.gradDes(h, userMatch_lc, nUsers, Pmax, N, uj, epsilon)
        
        optimal_t_rand = streamPA.streams_power_allocation(h, N, nUsers, userMatch_rand, uj, P_ASPA_rand)
        optimal_t_tum = streamPA.streams_power_allocation(h, N, nUsers, userMatch_tum, uj, P_ASPA_tum)
        optimal_t_lc = streamPA.streams_power_allocation(h, N, nUsers, userMatch_lc, uj, P_ASPA_lc)
        
        # Cálculo da taxa
        userRate_opt_rand = rateCalculation.ASPA(h, userMatch_rand, P_opt_rand, uj, N, nUsers)
        userRate_opt_tum = rateCalculation.ASPA(h, userMatch_tum, P_opt_tum, uj, N, nUsers)
        userRate_opt_lc = rateCalculation.ASPA(h, userMatch_lc, P_opt_lc, uj, N, nUsers)
        
        userRate_EPA_rand = rateCalculation.ASPA(h, userMatch_rand, P_EPA_rand, uj, N, nUsers)
        userRate_EPA_tum = rateCalculation.ASPA(h, userMatch_tum, P_EPA_tum, uj, N, nUsers)
        userRate_EPA_lc = rateCalculation.ASPA(h, userMatch_lc, P_EPA_lc, uj, N, nUsers)
        
        userRate_ASPA_rand = rateCalculation.ASPA_optimal_fraction_t(h, userMatch_rand, P_ASPA_rand, optimal_t_rand, uj, N, nUsers)
        userRate_ASPA_tum = rateCalculation.ASPA_optimal_fraction_t(h, userMatch_tum, P_ASPA_tum, optimal_t_tum, uj, N, nUsers)
        userRate_ASPA_lc = rateCalculation.ASPA_optimal_fraction_t(h, userMatch_lc, P_ASPA_lc, optimal_t_lc, uj, N, nUsers)
        
        # Armazenar os resultados para cada iteração
        sum_rates_streamPA_rand.append(np.sum(userRate_opt_rand))
        sum_rates_streamPA_tum.append(np.sum(userRate_opt_tum))
        sum_rates_streamPA_lc.append(np.sum(userRate_opt_lc))
        sum_rates_EPA_rand.append(np.sum(userRate_EPA_rand))
        sum_rates_EPA_tum.append(np.sum(userRate_EPA_tum))
        sum_rates_EPA_lc.append(np.sum(userRate_EPA_lc))
        sum_rates_ASPA_rand.append(np.sum(userRate_ASPA_rand))
        sum_rates_ASPA_tum.append(np.sum(userRate_ASPA_tum))
        sum_rates_ASPA_lc.append(np.sum(userRate_ASPA_lc))

    # Calcular a média das taxas após 1000 iterações
    media_rates = {
        "Sum Rate Stream-Based-PA_rand": np.mean(sum_rates_streamPA_rand),
        "Sum Rate Stream-Based-PA_tum": np.mean(sum_rates_streamPA_tum),
        "Sum Rate Stream-Based-PA_lc": np.mean(sum_rates_streamPA_lc),
        "Sum Rate EPA_rand": np.mean(sum_rates_EPA_rand),
        "Sum Rate EPA_tum": np.mean(sum_rates_EPA_tum),
        "Sum Rate EPA_lc": np.mean(sum_rates_EPA_lc),
        "Sum Rate ASPA_rand": np.mean(sum_rates_ASPA_rand),
        "Sum Rate ASPA_tum": np.mean(sum_rates_ASPA_tum),
        "Sum Rate ASPA_lc": np.mean(sum_rates_ASPA_lc)
    }
    
    # Exibir
        # Exibir as médias na tela
    print(f"\nMédias após {num_iterations} iterações para N = {N} subportadoras:")
    for key, value in media_rates.items():
        print(f"{key}: {value:.4f}")
