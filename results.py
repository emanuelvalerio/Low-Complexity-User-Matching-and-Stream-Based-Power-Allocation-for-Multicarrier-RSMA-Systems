import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados salvos no arquivo CSV
df_results = pd.read_csv("Results/sum_rate_results_varying_N.csv")

# Calcular a média das taxas para cada valor de N
summary = df_results.groupby("N")[["Sum Rate LC", "Sum Rate TUM"]].mean().reset_index()

# Configurar o estilo do seaborn
sns.set(style="whitegrid")

# Criar o gráfico
plt.figure(figsize=(10, 6))
sns.lineplot(data=summary, x="N", y="Sum Rate LC", label="Sum Rate LC", marker="o", linewidth=2)
sns.lineplot(data=summary, x="N", y="Sum Rate TUM", label="Sum Rate TUM", marker="s", linewidth=2)

# Configurar o título e os rótulos dos eixos
plt.title("Variação da Taxa em Função do Número de Subportadoras", fontsize=14)
plt.xlabel("Número de Subportadoras (N)", fontsize=12)
plt.ylabel("Taxa Total (Sum Rate)", fontsize=12)

# Legenda
plt.legend(title="Algoritmo", fontsize=10)

# Mostrar o gráfico
plt.tight_layout()
plt.show()
