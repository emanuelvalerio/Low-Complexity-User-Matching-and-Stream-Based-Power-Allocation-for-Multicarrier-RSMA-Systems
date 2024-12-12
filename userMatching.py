import numpy as np
import math
import normalizeVector as nv

def userMatchingAlgorithm(h,N,nUsers,uj):

    gainMatrix = np.zeros((N,nUsers));
    minMaxGains = np.zeros((N,2));
    rho_matrix = np.zeros((nUsers-1,1)); 
    aux_rho_matrix = np.zeros((nUsers-1,2),dtype=int); 
    comb = ((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    x = np.zeros((int(N*comb),1));
    for n in range(0,N):
        for jj in range(0,nUsers):
            gainMatrix[n,jj] = (np.linalg.norm(h[:,n,jj]**2))*(uj[jj]/np.sum(uj));
    for n in range(0,N):
        t = 0;
        idxMax = np.argmax(gainMatrix[n,:]);
        aux_rho_matrix[:,0] = idxMax+1;
        h_n2 = nv.normalization(h,n,idxMax);
        users_set = {i for i in range(0, nUsers) if i != idxMax}

        for jj in users_set:
            beta = .99;
            h_n1 = nv.normalization(h,n,jj);
            aux_rho_matrix[t,1] = jj+1;
            rho_matrix[t] = beta * (1-(np.abs(np.dot(np.conj(h_n1).T , h_n2))**2)) + (1-beta)*(uj[jj]/np.sum(uj));
            t = t+1;
        idxMax2 = np.argmax(rho_matrix);
        minMaxGains[n,0] = idxMax+1;
        minMaxGains[n,1] = aux_rho_matrix[idxMax2,1];

    # create a vector with the lenght of N*comb with all possibles combinations.
    t = 0;
    for n in range(0,N):
        for ii in range(0,nUsers-1):
            for jj in range(ii+1,nUsers):
                if((ii+1 == minMaxGains[n,0] or ii+1 == minMaxGains[n,1])and(jj+1==minMaxGains[n,0] or jj+1 == minMaxGains[n,1])):
                    x[t] = 1;
                else:
                    x[t] = 0;
                t +=1;
    return x;



