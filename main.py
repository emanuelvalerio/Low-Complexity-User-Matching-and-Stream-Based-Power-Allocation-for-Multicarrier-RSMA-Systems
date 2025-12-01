import pandas as pd
import numpy as np
import distances as dt
import channelModel as ch
import lowComplexityUserMatching as uMatch
import combinationsVectors as combV
import optimalPA as optPA
import rateCalculation
import unimodularMatrixBenchmark as benchmark
import SBPA
import randomMatching as randMatch
import gradientDescent as gradDes
import streamsPowerAllocation as streamPA

# ==========================================
# System Configuration
# ==========================================

Nt = 4                 # Number of Transmit Antennas (Tx)
Pmax = 100             # Total Available Power (Watts)
nUsers = 8             # Number of Users
dInnerRadius = 1       # Inner cell radius
dOuterRadius = 10      # Outer cell radius
gamma = 3              # Path loss exponent
num_iterations = 1000  # Number of Monte Carlo iterations
epsilon = 1e-4         # Convergence threshold
uj = np.ones((nUsers, 1)) # Weight vector for users


N_values = [4,8,12,16]; # Range of subcarriers to be tested
results = [] # List to store simulation results

df_results = pd.DataFrame(results) # Initial conversion to DataFrame

# ==========================================
# Main Simulation Loop
# ==========================================

for N in N_values:
    print(f"\nStarting experiments for N = {N} subcarriers...")
    # Initial power per subcarrier (Uniform)
    Pn = [(Pmax / N) * x for x in np.ones((N, 1))]  # Potência inicial por subportadora

    # Monte Carlo Iterations
    for iteration in range(num_iterations):
        
        # 1. Channel Generation
        h = ch.channel(Nt, N, nUsers, gamma, dOuterRadius, dInnerRadius)

        # 2. User Matching Algorithms (Clustering)
        x_lc = uMatch.userMatchingAlgorithm(h, N, nUsers, uj) # Low Complexity Algorithm
        x_tum = benchmark.unimodularMatrixUserMatching(h, Pmax, N, nUsers, uj) # Total Unimodular Matrix (TUM) Benchmark
        x_tum = x_tum.reshape((-1, 1))
        
        # 3. Vector Combination
        combVect = combV.combVector(nUsers, N)
        
        # Formulate matching matrices
        userMatch_lc = np.hstack((combVect, x_lc))
        userMatch_rand = randMatch.randomUserMatching(combVect, N, nUsers)
        userMatch_tum = np.hstack((combVect, x_tum))
        
        # 4. Power Allocation (PA) Algorithms
        
        # A) Optimized Power Allocation (General)
        P_opt_rand = optPA.optimizedPowerAllocation(h, userMatch_rand, uj, N, nUsers, Pmax)
        P_opt_tum = optPA.optimizedPowerAllocation(h, userMatch_tum, uj, N, nUsers, Pmax)
        P_opt_lc = optPA.optimizedPowerAllocation(h, userMatch_lc, uj, N, nUsers, Pmax)
        
        # B) Equal Power Allocation (EPA)
        P_EPA_rand = [(Pmax / (2 * N)) * x for x in np.ones((2 * N))]
        P_EPA_tum = [(Pmax / (2 * N)) * x for x in np.ones((2 * N, 1))]
        P_EPA_lc = [(Pmax / (2 * N)) * x for x in np.ones((2 * N, 1))]
        
        # C) Alternative Stream Power Allocation (ASPA - Gradient Descent)
        P_ASPA_rand = gradDes.gradDes(h, userMatch_rand, nUsers, Pmax, N, uj, epsilon)
        P_ASPA_tum = gradDes.gradDes(h, userMatch_tum, nUsers, Pmax, N, uj, epsilon)
        P_ASPA_lc = gradDes.gradDes(h, userMatch_lc, nUsers, Pmax, N, uj, epsilon)
        
        # D) Stream-Based Power Allocation (SBPA)
        P_SBPA_lc = SBPA.optimizedPowerAllocation(h,userMatch_lc,uj,N,nUsers,Pmax)
        P_SBPA_tum = SBPA.optimizedPowerAllocation(h,userMatch_tum,uj,N,nUsers,Pmax)
        P_SBPA_rand = SBPA.optimizedPowerAllocation(h,userMatch_rand,uj,N,nUsers,Pmax)
        
        # E) Stream Optimal Fraction Allocation (ASPA)
        optimal_t_rand = streamPA.streams_power_allocation(h, N, nUsers, userMatch_rand, uj, P_ASPA_rand)
        optimal_t_tum = streamPA.streams_power_allocation(h, N, nUsers, userMatch_tum, uj, P_ASPA_tum)
        optimal_t_lc = streamPA.streams_power_allocation(h, N, nUsers, userMatch_lc, uj, P_ASPA_lc)
        
        # 5. Rate Calculation (Throughput)
        
        # Calculate rates for Optimized PA 
        userRate_opt_rand = rateCalculation.ASPA(h, userMatch_rand, P_opt_rand, uj, N, nUsers)
        userRate_opt_tum = rateCalculation.ASPA(h, userMatch_tum, P_opt_tum, uj, N, nUsers)
        userRate_opt_lc = rateCalculation.ASPA(h, userMatch_lc, P_opt_lc, uj, N, nUsers)
        
        # Calculate rates for EPA
        userRate_EPA_rand = rateCalculation.ASPA(h, userMatch_rand, P_EPA_rand, uj, N, nUsers)
        userRate_EPA_tum = rateCalculation.ASPA(h, userMatch_tum, P_EPA_tum, uj, N, nUsers)
        userRate_EPA_lc = rateCalculation.ASPA(h, userMatch_lc, P_EPA_lc, uj, N, nUsers)
        
        # Calculate rates for ASPA
        userRate_ASPA_rand = rateCalculation.ASPA_optimal_fraction_t(h, userMatch_rand, P_ASPA_rand, optimal_t_rand, uj, N, nUsers)
        userRate_ASPA_tum = rateCalculation.ASPA_optimal_fraction_t(h, userMatch_tum, P_ASPA_tum, optimal_t_tum, uj, N, nUsers)
        userRate_ASPA_lc = rateCalculation.ASPA_optimal_fraction_t(h, userMatch_lc, P_ASPA_lc, optimal_t_lc, uj, N, nUsers)
        
        # Calculate rates for SBPA
        userRate_SBPA_lc = rateCalculation.ASPA(h, userMatch_lc, P_SBPA_lc, uj, N, nUsers)
        userRate_SBPA_tum = rateCalculation.ASPA(h, userMatch_tum, P_SBPA_tum, uj, N, nUsers)
        userRate_SBPA_rand = rateCalculation.ASPA(h, userMatch_rand, P_SBPA_rand, uj, N, nUsers)
        
        # 6. Store Results
        result = {
            "N": N,
            "Iteration": iteration + 1,
            "Sum Rate Stream-Based-PA_rand": np.sum(userRate_SBPA_rand),
            "Sum Rate Stream-Based-PA_tum": np.sum(userRate_SBPA_tum),
            "Sum Rate Stream-Based-PA_lc": np.sum(userRate_SBPA_lc)
            # Uncomment below to log other PA methods
            #"Sum Rate EPA_rand": np.sum(userRate_EPA_rand),
            #"Sum Rate EPA_tum": np.sum(userRate_EPA_tum),
            #"Sum Rate EPA_lc": np.sum(userRate_EPA_lc),
            #"Sum Rate ASPA_rand": np.sum(userRate_ASPA_rand),
            #"Sum Rate ASPA_tum": np.sum(userRate_ASPA_tum),
            #"Sum Rate ASPA_lc": np.sum(userRate_ASPA_lc)
        }

        # Append to DataFrame
        df_results = pd.concat([df_results, pd.DataFrame([result])], ignore_index=True)
 
        
        # Save to CSV continuously (checkpointing)
        filename = "sum_rate_x_num_subcarriers_P_100W_8_users.csv"
        df_results.to_csv(filename, index=False)
        print(f"Saving results for iteration {iteration + 1} at N = {N}")
