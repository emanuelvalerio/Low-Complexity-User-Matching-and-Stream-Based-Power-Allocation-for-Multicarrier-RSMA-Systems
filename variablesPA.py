import numpy as np
import math
import normalizeVector as norm
import calculateFc

def varCalculate(h,restX,uj,N,nUsers):
    comb = int((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    vetX = np.zeros((1,N));
    vetY = np.zeros((1,N));
    g1 = np.zeros((1,N));
    g2 = np.zeros((1,N));
    alpha = np.zeros((1,N));
    weights= np.zeros((2,N));
    for n in range(0,N):
        aux = np.where(restX[:, 0].astype(int) == int(n+1))[0];
        idx = int(np.where(restX[aux[0]:aux[comb-1]+1,3] != 0)[0]);
        ii = int(restX[aux[idx],1]-1); # decrease in 1 because was increased 1 when the matrix was it built
        jj = int(restX[aux[idx],2]-1);
        hni = h[:,n,ii];
        hnj = h[:,n,jj];
        strongest = max(np.linalg.norm(hni),np.linalg.norm(hnj));
        if strongest == np.linalg.norm(hni):
            h1 = hni;
            h2 = hnj;
            u1 = uj[ii];
            u2 = uj[jj];
            h_1 = norm.normalization(h,n,ii);
            h_2 = norm.normalization(h,n,jj);
            rho = (1-(np.abs(np.dot(np.conj(h_1).T , h_2))**2));
            fc = calculateFc.calculationFc(h_1,h_2);
        else:
            h1 = hnj;
            h2 = hni;
            u1 = uj[jj];
            u2 = uj[ii];
            h_1 = norm.normalization(h,n,jj);
            h_2 = norm.normalization(h,n,ii);
            rho = (1-(np.abs(np.dot(np.conj(h_1).T , h_2))**2));
            fc = calculateFc.calculationFc(h_1,h_2);
        h1_norm = np.linalg.norm(h_1);
        h2_norm = np.linalg.norm(h_2);
       # vetX[n] = ((h1_norm**2)*rho/3)+((h2_norm**2)*rho/3)+ ((np.abs(np.dot(np.conj(h_1).T,fc))**2)*(h2_norm**2)/3);
       # vetY[n] = (((h1_norm**2)*(h2_norm**2)*rho**2)/9)+(((h1_norm**2)*(np.abs(np.dot(np.conj(h_1).T,fc))**2)*(h2_norm**2)*rho**2)/9);
        g1[:,n] = ((h1_norm**2)*rho/2);
        g2[:,n] = ((h2_norm**2)*rho/2);
        alpha[:,n] = (np.abs(np.dot(np.conj(h_2).T,fc))**2)*(h2_norm**2);
        weights[0,n] = u1;
        weights[1,n] = u2;
    
    return weights,g1,g2,alpha
        

        
    