import numpy as np
import math
def commomRate(h,Pnc,n,ii,jj,gamma1,gamma2,rho):
    hni = h[:,n,ii];
    hnj = h[:,n,jj];
    Nnij = rho*(np.linalg.norm(hni)**2)*(np.linalg.norm(hnj)**2)*Pnc;
    Dnij = (gamma1*(np.linalg.norm(hnj)**2))+(gamma2*(np.linalg.norm(hni)**2))-(2*(math.sqrt(gamma1)*math.sqrt(gamma2)*np.abs(np.dot(np.conj(hni).T,hnj))));
    return Nnij,Dnij,math.log2(1+(Nnij/Dnij));