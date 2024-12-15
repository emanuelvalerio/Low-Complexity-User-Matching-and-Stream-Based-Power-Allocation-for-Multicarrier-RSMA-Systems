import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configurações iniciais
num_users = 3
num_subcarriers = 2
num_antennas = 4
gamma = 3.5
outer_radius = 500
inner_radius = 35
num_repetitions = 1000  # Número de repetições para expandir a base de dados

# Função para gerar as distâncias entre a estação e os usuários
def generate_distances(n_users, outer_radius, inner_radius):
    return np.random.uniform(inner_radius, outer_radius, n_users)

# Função para gerar os canais
def generate_channels(n_antennas, n_subcarriers, n_users, gamma, distances):
    h = np.zeros((n_antennas, n_subcarriers, n_users), dtype=complex)
    for user in range(n_users):
        variance = 1 / (n_antennas * distances[user]**gamma)
        for subcarrier in range(n_subcarriers):
            h[:, subcarrier, user] = (np.sqrt(variance / 2) *
                                      (np.random.randn(n_antennas) + 1j * np.random.randn(n_antennas)))
    return h

# Função para calcular os ganhos e ortogonalidade
def calculate_features(h, distances):
    num_users = h.shape[2]
    num_subcarriers = h.shape[1]
    gains = np.zeros((num_users, num_subcarriers))
    ortho_features = []

    for user in range(num_users):
        for subcarrier in range(num_subcarriers):
            gains[user, subcarrier] = np.linalg.norm(h[:, subcarrier, user])**2

    for i in range(num_users):
        for j in range(i + 1, num_users):
            rho = 1 - (np.abs(np.dot(np.conj(h[:, :, i].T), h[:, :, j]))**2).mean()
            ortho_features.append(rho)

    # Variância das ortogonalidades entre pares
    ortho_variance = np.var(ortho_features)

    return gains, ortho_features, ortho_variance

# Função para calcular a taxa ponderada
def calculate_weighted_rate(gains, weights, ortho_features):
    rates = []
    for i in range(len(ortho_features)):
        for subcarrier in range(gains.shape[1]):
            rate = (weights[i, 0] * np.log2(1 + gains[i, subcarrier]) +
                    weights[i, 1] * np.log2(1 + ortho_features[i]))
            rates.append(rate)
    return rates

# Função para gerar dados de entrada e saída
def generate_data(num_users, num_subcarriers, num_antennas, gamma, outer_radius, inner_radius, num_repetitions):
    X = []
    y = []

    for _ in range(num_repetitions):
        distances = generate_distances(num_users, outer_radius, inner_radius)
        h = generate_channels(num_antennas, num_subcarriers, num_users, gamma, distances)
        gains, ortho_features, ortho_variance = calculate_features(h, distances)

        for subcarrier in range(num_subcarriers):
            for i in range(num_users):
                for j in range(i + 1, num_users):
                    # Calcula a distância Euclidiana entre pares de usuários
                    user_pair_distance = np.abs(distances[i] - distances[j])

                    # Razão sinal-ruído (SNR) aproximada
                    snr_i = gains[i, subcarrier] / (1 + ortho_variance)
                    snr_j = gains[j, subcarrier] / (1 + ortho_variance)

                    input_features = [
                        gains[i, subcarrier],
                        gains[j, subcarrier],
                        ortho_features[(i * num_users + j) - ((i + 1) * (i + 2)) // 2],
                        distances[i],
                        distances[j],
                        user_pair_distance,
                        ortho_variance,
                        snr_i,
                        snr_j
                    ]
                    X.append(input_features)

                    # Define a etiqueta binária (0 ou 1) com base no par que maximiza a taxa
                    rate_i = np.log2(1 + gains[i, subcarrier])
                    rate_j = np.log2(1 + gains[j, subcarrier])
                    y.append(1 if rate_i + rate_j > 0 else 0)

    return np.array(X), np.array(y)

# Gerar base de dados
X, y = generate_data(num_users, num_subcarriers, num_antennas, gamma, outer_radius, inner_radius, num_repetitions)

# Normalização e divisão dos dados
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construção do modelo DNN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Avaliação do modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Gráficos de avaliação
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.show()
