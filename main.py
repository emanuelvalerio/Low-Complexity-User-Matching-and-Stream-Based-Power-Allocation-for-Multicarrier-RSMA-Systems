# adaptive power allocation based RSMA sytem

import pandas as pd
import numpy  as np
import distances as dt
import channelModel as ch
import lowComplexityUserMatching as uMatch
import combinationsVectors as combV
import optimalPA as optPA
import rateCalculation
import unimodularMatrixBenchmark as benchmark

Nt = 4; # Number of antenas at transmitter
Pmax = 100; # Total power disponible
N = 2; # Number of subcarriers 
nUsers = 4; # Number of users 
dInnerRadius = 1;
dOuterRadius = 10;
gamma = 3; # expoent of path loss Python: Select Interpreter
num_iterations = 1000;
epsilon = 10^-4;
posERBx = 0;
posERBy = 0;
uj = np.ones((nUsers, 1)); # Vector of weights
h = ch.channel(Nt,N,nUsers,gamma,dOuterRadius,dInnerRadius);
x_lc = uMatch.userMatchingAlgorithm(h,N,nUsers,uj);
x_tum = benchmark.unimodularMatrixUserMatching(h,Pmax,N,nUsers,uj);
x_tum = x_tum.reshape(-1, 1)
combVect = combV.combVector(nUsers,N);
userMatch = np.hstack((combVect,x_lc));
P_opt1 = optPA.optimizedPowerAllocation(h,userMatch,uj,N,nUsers,Pmax);
P_opt2 = optPA.optimizedPowerAllocation(h,np.hstack((combVect,x_tum)),uj,N,nUsers,Pmax);
userRate_lc = rateCalculation.ASPA(h,userMatch,P_opt1,uj,N,nUsers);
userRate_tum = rateCalculation.ASPA(h,np.hstack((combVect,x_tum)),P_opt2,uj,N,nUsers);
print(np.sum(userRate_tum))
print(np.sum(userRate_lc))