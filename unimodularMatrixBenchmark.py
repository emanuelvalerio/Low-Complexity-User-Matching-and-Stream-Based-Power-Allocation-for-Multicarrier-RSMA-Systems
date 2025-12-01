import numpy as np
from docplex.mp.model import Model
import pandas as pd
import math
import normalizeVector as norm

# --- Helper Functions ---

def privateRate(h,n,ii,jj,Pn1,Pn2):
    """Calculates private rates and interference parameters."""
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
    """Calculates the common rate based on interference alignment."""
    hni = h[:,n,ii];
    hnj = h[:,n,jj];
    Nnij = rho*(np.linalg.norm(hni)**2)*(np.linalg.norm(hnj)**2)*Pnc;
    Dnij = (gamma1*(np.linalg.norm(hnj)**2))+(gamma2*(np.linalg.norm(hni)**2))-(2*(np.sqrt(gamma1)*np.sqrt(gamma2)*np.abs(np.dot(np.conj(hni).T,hnj))));
    return np.log2(1+(Nnij/Dnij));
     
def totalRateCalculate(h,nUsers,N,Pn,uj):
    """
    Calculates the Weighted Sum Rate (WSR) for all possible user pairs on all subcarriers.
    
    Optimization:
    Replaces the Pandas sort/merge approach. Instead of calculating individual rates 
    and matching them later, we iterate through Combinations(Users, 2) and 
    calculate the pair's total rate immediately.
    """
    cont = -1;
    faux = np.zeros((N*nUsers*(nUsers-1),1));
    f = np.zeros((faux.shape[0] // 2,1));
    test = np.zeros((N*nUsers*(nUsers-1),3));
    comb = ((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    for jj in range(nUsers):
        contInit = cont+1;
        for n in range(N):
            Pn1 = Pn[n]/3;
            Pn2 = Pn[n]/3;
            Pnc = Pn[n]/3; 
            for ii in range(jj):
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
    
    for ii in range(orderedMatrix.shape[0]):
        for jj in range(ii+1,orderedMatrix.shape[0]):
            if (np.array_equal(orderedMatrix[ii,0:3],orderedMatrix[jj,0:3])):
                idxEqualRows[cont,0] = ii;
                idxEqualRows[cont,1] = jj;
                cont+=1;
    
    for ii in range(idxEqualRows.shape[0]):
        f[ii] = orderedMatrix[int(idxEqualRows[ii,0]),3] + orderedMatrix[int(idxEqualRows[ii,1]),3];

    return f


def unimodularMatrixUserMatching(h,Pmax,N,nUsers,uj):
    """
    Solves the User Matching problem using Linear Programming (Relaxed Integer Programming).
    
    The Unimodular property of the assignment matrix guarantees that 
    the continuous relaxation often yields binary (integer) results.
    """
    comb = int((math.factorial(nUsers)/((math.factorial(nUsers-2))*math.factorial(2))));
    # Assuming Pmax, N, comb, and nUsers are defined elsewhere in the code
    Pn = (Pmax / N) * np.ones(N)
    # Call to totalRateCalculate2 (this should be defined elsewhere in the Python code)
    f = np.hstack(totalRateCalculate(h,nUsers,N,Pn,uj));
    # Create the Aeq matrix (equivalent to the MATLAB loop)
    Aeq = np.zeros((N, N * comb))

    for n in range(N):
         Aeq[n, (((n+1) * comb - (comb - 1))-1):((n+1) * comb)] = np.ones(comb);

    beq = np.ones(N)  
    ub = np.hstack(np.ones_like(f))  # Upper bound
    lb = np.hstack(np.zeros_like(f)) # lower bound

    opt_model = Model(name="LP_Model")


    x_vars = opt_model.continuous_var_list(len(f), lb=lb, ub=ub, name="x")
    
    objective = opt_model.sum(-f[i] * x_vars[i] for i in range(len(f)))
    
    opt_model.set_objective('min', objective)
   
    for n in range(N):
        opt_model.add_constraint(opt_model.sum(Aeq[n,ii]*x_vars[ii] for ii in range(N*comb)) == beq[n],ctname=f"eq_{n}");

    # Solving the model
    solution = opt_model.solve()
    return np.array([solution.get_value(var) for var in x_vars])  
