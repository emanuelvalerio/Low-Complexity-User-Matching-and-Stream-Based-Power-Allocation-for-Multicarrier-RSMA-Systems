# adaptive power allocation based RSMA sytem

import pandas as pd
import numpy  as np
import distances as dt
import channelModel as ch
import userMatching as uMatch
import combinationsVectors as combV
import optimalPA as optPA
import rateCalculation

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
x = uMatch.userMatchingAlgorithm(h,N,nUsers,uj);
combVect = combV.combVector(nUsers,N);
userMatch = np.hstack((combVect,x));
P_opt = optPA.optimizedPowerAllocation(h,userMatch,uj,N,nUsers,Pmax);
userRate = rateCalculation.ASPA(h,userMatch,P_opt,uj,N,nUsers);
print(np.sum(userRate))