
import pandas as pd
import matplotlib.pyplot as plt

# Carregando o arquivo CSV
df = pd.read_csv('sum_rate_x_num_subcarriers_P_10W_8_users.csv')


# Extrair as colunas de interesse (ignorando a coluna 'Iteration')
subportadoras = df['N']  # Número de subportadoras
taxas = df.iloc[:, 2:]  # As taxas obtidas (do 3º ao último)

# Agrupar pelos números de subportadoras e calcular a média das taxas para cada valor de 'N'
media_taxas = df.groupby('N').mean() # Média das taxas (sem a coluna de iteração)

# Plotar o gráfico para cada algoritmo (colunas de taxas)
plt.figure(figsize=(10, 6))

for col in media_taxas.columns[1:]:
    plt.plot(media_taxas.index, media_taxas[col], marker='o', label=col)

# Configurar o gráfico
plt.xlabel('Número de Subportadoras', fontsize=12)
plt.ylabel('Taxa de Dados Média', fontsize=12)
plt.title('Média da Taxa de Dados por Número de Subportadoras para Cada Algoritmo', fontsize=14)
plt.legend(title='Algoritmos', loc='best')
plt.grid(True)
plt.show()
