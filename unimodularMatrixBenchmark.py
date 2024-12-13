import numpy as np
from scipy.optimize import linprog
import pandas as pd
import math
import normalizeVector as norm

def privateRate(h,n,ii,jj,Pn1,Pn2):
    hni = h[:,n,ii];
    hnj = h[:,n,jj]; 
    h_hni  =norm.normalization(h,n,ii);
    h_hnj  = norm.normalization(h,n,jj);
    rho = (1-(np.abs(np.dot(np.conj(h_hni).T , h_hnj))**2));
    gamma1 = 1+((np.linalg.norm(hni)**2)*rho*Pn1);
    gamma2 = 1+((np.linalg.norm(hnj)**2)*rho*Pn2);
    pRate1 = np.log2(gamma1);
    pRate2 = np.log2(gamma2);
    return pRate1,pRate2,gamma1,gamma2,rho

def commomRate(h,Pnc,n,ii,jj,gamma1,gamma2,rho):
    hni = h[:,n,ii];
    hnj = h[:,n,jj];
    Nnij = rho*(np.linalg.norm(hni)**2)*(np.linalg.norm(hnj)**2)*Pnc;
    Dnij = (gamma1*(np.linalg.norm(hnj)**2))+(gamma2*(np.linalg.norm(hni)**2))-(2*(np.sqrt(gamma1)*np.sqrt(gamma2)*np.abs(np.dot(np.conj(hni).T,hnj))));
    return np.log2(1+(Nnij/Dnij));
     
def totalRateCalculate(h,nUsers,N,Pn,uj):
    cont = 0;
    faux = np.zeros((N*nUsers*(nUsers-1),1));
    f = np.zeros((faux.shape[0] // 2,1));
    test = np.zeros((N*nUsers*(nUsers-1),3));
    comb = ((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    for jj in range(0,nUsers):
        contInit = cont+1;
        for n in range(0,N):
            Pn1 = Pn[n]/3;
            Pn2 = Pn[n]/3;
            Pnc = Pn[n]/3; 
            for ii in range(0,jj-1):
                cont +=1;
                pRate1,pRate2,gamma1,gamma2,rho = privateRate(h,n,ii,jj,Pn1,Pn2);
                cRate = commomRate(h,Pnc,n,ii,jj,gamma1,gamma2,rho);
                Cnij2 = cRate/2;
                faux[cont] = (pRate2+Cnij2);
                values = [n,ii,jj];
                test[cont,:] = values;
            for ii in range(jj+1,nUsers):
                cont +=1;
                pRate1,pRate2,gamma1,gamma2,rho = privateRate(h,n,jj,ii,Pn1,Pn2);
                cRate = commomRate(h,Pnc,n,jj,ii,gamma1,gamma2,rho);
                Cnij1 = cRate/2;
                faux[cont] = (pRate1+Cnij1);
                values = [n,jj,ii];
                test[cont,:] = values;
        faux[contInit:cont+1] = faux[contInit:cont+1]*uj[jj];
    auxMatrix =  np.hstack((test,faux));   
    df = pd.DataFrame(auxMatrix) # Convert the numpy array to a pandas DataFrame
    orderedMatrix = df.sort_values(by=[0, 1, 2]).values # Sort by the first, second, and third columns
    idxEqualRows = np.zeros((orderedMatrix.shape[0] // 2, 2));
    cont = 0;
    for ii in range(0,orderedMatrix.shape[0]):
        for jj in range(ii+1,orderedMatrix.shape[0]):
                # Check if the first three columns of the rows are the same
                if np.array_equal(orderedMatrix[ii, :2], orderedMatrix[jj, :2]):
                    if cont>=N*comb:
                        break;
                    # Add indexes to the list
                    idxEqualRows[cont, 0] = ii;
                    idxEqualRows[cont, 1] = jj;
                    cont += 1

    
    for ii in range(0,idxEqualRows.shape[0]):
        f[ii] = orderedMatrix[int(idxEqualRows[ii,0]),3] + orderedMatrix[int(idxEqualRows[ii,1]),3];

    return f


def unimodularMatrixUserMatching(h,Pmax,N,nUsers,uj):
    comb = int((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    # Assuming Pmax, N, comb, and nUsers are defined elsewhere in the code
    Pn = (Pmax / N) * np.ones(N)
    # Call to totalRateCalculate2 (this should be defined elsewhere in the Python code)
    f = totalRateCalculate(h,nUsers,N,Pn,uj);
    # Create the Aeq matrix (equivalent to the MATLAB loop)
    Aeq = np.zeros((N, N * comb))
    for n in range(N):
        Aeq[n, (((n+1) * comb - (comb - 1))-1):((n+1) * comb)] = np.ones(comb)

    # Define the beq vector
    beq = np.ones(N)

    # Set the bounds for the linear programming
    ub = np.ones_like(f)  # Upper bounds
    lb = np.zeros_like(f)  # Lower bounds
    bounds = [(0, 1) for _ in range(len(f))]
    result = linprog(
    -f,               # Minimizar -f (maximizar f)
    A_eq=Aeq,         # Restrições de igualdade
    b_eq=beq,         # Vetor da restrição de igualdade
    bounds=bounds,  # Limites inferior e superior
    method='highs'    # Método de solução
)

    # Verificar resultados
    if result.success:
        x2 = result.x  # Solução do problema
        fval1 = -result.fun  # Valor da função objetivo
        print("Solução encontrada:")
        print("x2:", x2)
        print("Valor da função objetivo:", fval1)
    else:
        print("Falha ao resolver o problema:", result.message)

    # Extract the solution
    x2 = result.x
    fval1 = result.fun
    exitflag1 = result.status
    output1 = result.message

    return x2
