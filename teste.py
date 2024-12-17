from docplex.mp.model import Model
import numpy as np

# Supondo que os dados h, uj, Pmax, N, e nUsers já estejam definidos
# Esses valores precisam ser inicializados para que o código funcione
# Exemplos fictícios para demonstração:
N = 5  # Número de subportadoras (por exemplo)
comb = 3  # Número de combinações (exemplo)
Pmax = 10  # Potência máxima
nUsers = 2  # Número de usuários

# Suponha que f seja a saída de uma função 'totalRateCalculate2'
# Aqui, usamos um exemplo fictício
f = np.random.rand(N * comb)  # Vetor de taxa fictícia
Aeq = np.zeros((N, N * comb))

# Preencher Aeq conforme o código MATLAB
for n in range(N):
    Aeq[n, n * comb:(n + 1) * comb] = np.ones(comb)

beq = np.ones(N)  # Vetor de igualdade
ub = np.ones_like(f)  # Limites superiores
lb = np.zeros_like(f)  # Limites inferiores

# Criar o modelo de otimização
opt_model = Model(name="LP_Model")

# Variáveis de decisão (x2 no MATLAB)
x_vars = opt_model.continuous_var_list(len(f), lb=lb.tolist(), ub=ub.tolist(), name="x")

# Função objetivo (maximização de f -> minimização de -f)
objective = opt_model.sum(-f[i] * x_vars[i] for i in range(len(f)))
opt_model.set_objective('min', objective)

# Adicionar restrições de igualdade Aeq * x = beq
for i in range(Aeq.shape[0]):
    opt_model.add_constraint(
        opt_model.sum(Aeq[i, j] * x_vars[j] for j in range(Aeq.shape[1])) == beq[i],
        ctname=f"eq_{i}"
    )

# Resolver o modelo
solution = opt_model.solve()

# Resultados
if solution:
    x2 = np.array([solution.get_value(var) for var in x_vars])  # Variáveis de decisão
    fval1 = solution.objective_value  # Valor da função objetivo
    print("Solução encontrada:")
    print("x2:", x2)
    print("Valor da função objetivo (fval1):", fval1)
else:
    print("Não foi possível encontrar uma solução viável.")


