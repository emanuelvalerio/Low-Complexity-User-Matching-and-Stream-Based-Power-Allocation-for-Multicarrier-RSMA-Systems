import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools
import calculateFc
import normalizeVector as norm
# Configurações iniciais
num_users = 3
num_subcarriers = 2
num_antennas = 4
Pmax = 100;
gamma = 3
outer_radius = 10
inner_radius = 1
num_repetitions = 7000  # Número de repetições para expandir a base de dados
Pn = [(Pmax/num_subcarriers) * x for x in np.ones((num_subcarriers,1))];
uj = np.ones((1,num_users));

def distance(nUsers, dOuterRadius, dInnerRadius):
    # Gerar as posições dos usuários
    theta = 2 * np.pi * (np.random.rand(nUsers, 1))
    r = (dOuterRadius - dInnerRadius) * np.sqrt(np.random.rand(nUsers, 1)) + dInnerRadius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Distância dos usuários até a estação rádio base (0,0)
    dist_to_base = np.sqrt(x**2 + y**2)
    
    # Calcular as distâncias entre todos os pares de usuários
        # Gerar as combinações únicas de pares de usuários
    pairs = list(itertools.combinations(range(nUsers), 2))

    # Calcular distâncias para os pares
    dist_pairs = np.zeros(len(pairs))  # Vetor para armazenar as distâncias entre os pares
    for i, (user1, user2) in enumerate(pairs):
        dist_pairs[i] = np.sqrt((x[user1] - x[user2])**2 + (y[user1] - y[user2])**2)
    
    return dist_to_base, dist_pairs

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
    correlation_features = []
    entropy_features = []

    for user in range(num_users):
        for subcarrier in range(num_subcarriers):
            gains[user, subcarrier] = (np.linalg.norm(h[:, subcarrier, user])**2)

    for subcarrier in range(num_subcarriers):  # Adiciona variação por subportadora
        for i in range(num_users):
            for j in range(i + 1, num_users):
                # Ortogonalidade dos canais por subportadora
                rho = (1 - np.abs(np.dot(np.conj(h[:, subcarrier, i]), h[:, subcarrier, j]))**2)
                ortho_features.append(rho)

                # Correlação dos canais por subportadora
                correlation = np.abs(np.correlate(h[:, subcarrier, i], h[:, subcarrier, j])).mean()
                correlation_features.append(correlation)

                # Entropia dos canais por subportadora
                combined_channel = np.concatenate((h[:, subcarrier, i].real, h[:, subcarrier, j].real,
                                                    h[:, subcarrier, i].imag, h[:, subcarrier, j].imag))
                prob, _ = np.histogram(combined_channel, bins=10, density=True)
                prob = prob[prob > 0]  # Remover valores zero para evitar log(0)
                entropy = -np.sum(prob * np.log2(prob))
                entropy_features.append(entropy)

    # Variação das distâncias
    distance_variance = np.var(distances)

    return gains, ortho_features, entropy_features, distance_variance, correlation_features

def minimumGain(h,subcarrier,i,j):
    gain_i = np.linalg.norm(h[:, subcarrier, i]);
    gain_j = np.linalg.norm(h[:, subcarrier, j]);
    if gain_i >= gain_j:
        return j;
    else:
        return i;

# Função para gerar dados de entrada e saída
def generate_data(num_users, num_subcarriers, num_antennas, gamma, outer_radius, inner_radius, num_repetitions,uj,Pn):
    X = []
    y = []

    for _ in range(num_repetitions):
       # distances = distance.distance(num_users,dOuterRadius,dInnerRadius);
        distances, dist_pairs = distance(num_users, outer_radius, inner_radius);
        distances = distances.flatten();
        dist_pairs = dist_pairs.flatten();
        h = generate_channels(num_antennas, num_subcarriers, num_users, gamma, distances)
        gains, ortho_features, entropy_features, distance_variance, correlation_features = calculate_features(h, distances);

        pair_rates = []
        max_rate = np.zeros((num_subcarriers,1));
        max_pair = np.zeros((num_subcarriers,2));
        for subcarrier in range(num_subcarriers):
            max_rate[subcarrier] = -np.inf ;
            max_pair[subcarrier] = None ; 
            

            for i in range(num_users):
                for j in range(i + 1, num_users):
                    h_i = norm.normalization(h,subcarrier,i);
                    h_j = norm.normalization(h,subcarrier,j);
                    fc = calculateFc.calculationFc(h_i,h_j);
                    idx = minimumGain(h,subcarrier,i,j);
                    rate_i = np.log2(1 + gains[i, subcarrier]*np.linalg.norm(h[:, subcarrier, i])*Pn[subcarrier]/3)*uj[:,i] + ((uj[:,i]+uj[:,j])/2)*np.log2(1+((np.dot(np.conj(h[:,subcarrier,idx]).T,fc)**2)*Pn[subcarrier]/3)/(1+gains[idx, subcarrier]));
                    rate_j = np.log2(1 + gains[j, subcarrier]*np.linalg.norm(h[:, subcarrier, j])*Pn[subcarrier]/3)*uj[:,j] + ((uj[:,i]+uj[:,j])/2)*np.log2(1+((np.dot(np.conj(h[:,subcarrier,idx]).T,fc)**2)*Pn[subcarrier]/3)/(1+gains[idx, subcarrier]));
                    total_rate = rate_i + rate_j
                    pair_rates.append((subcarrier,i, j, total_rate))

                    if total_rate > max_rate[subcarrier]:
                        max_rate[subcarrier] = total_rate ; 
                        max_pair[subcarrier,0] = i;
                        max_pair[subcarrier,1] = j;

        cont = 0;
        ant = 0;
        for subcarrier,i, j, total_rate in pair_rates:
            if subcarrier!=ant:
                cont = 0;
                ant = subcarrier;
            
            input_features = [
                gains[i, subcarrier],
                gains[j, subcarrier],
                ortho_features[subcarrier],
                entropy_features[subcarrier],
                correlation_features[subcarrier],
                distances[i],
                distances[j],
                dist_pairs[cont],
                distance_variance,
                np.abs(distances[i] - distances[j])
                ]
            X.append(input_features)
            y.append(1 if (i == max_pair[subcarrier,0] and j == max_pair[subcarrier,1]) else 0)
            cont +=1;
    return np.array(X), np.array(y)

# Gerar base de dados
X, y = generate_data(num_users, num_subcarriers, num_antennas, gamma, outer_radius, inner_radius, num_repetitions,uj,Pn)

# Normalização e divisão dos dados
scaler = MinMaxScaler();
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construção do modelo DNN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=1)

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
