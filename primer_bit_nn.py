import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
import itertools

def primer_bit(entrada):
    return 1 if entrada[0] == 1 else 0

def red_neuronal(model):
    # Generar todas las combinaciones de 20 bits
    N = 2**21
    bits = np.arange(21, dtype=np.uint32)
    X_all = ((np.arange(N, dtype=np.uint32)[:, None] >> bits) & 1).astype(np.float32)
    
    # Muestreo aleatorio de 500_000 ejemplos
    rng = np.random.default_rng(42)
    idx = rng.choice(N, size=500_000, replace=False)
    X = X_all[idx]
    
    y = np.array([primer_bit(row) for row in X], dtype=np.float32)
    
    # Dividir 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Entrenamiento rápido
    model.fit(X_train, y_train, epochs=10, batch_size=2048, verbose=2)
    
    # Evaluación
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print("Test accuracy:", accuracy_score(y_test, y_pred))


model_1 = Sequential([
    Input(shape=(21,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_2 = Sequential([
    Input(shape=(21,)),
    Dense(9, activation='relu'),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_3 = Sequential([
    Input(shape=(21,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_4 = Sequential([
    Input(shape=(21,)),
    Dense(40, activation='relu'),
    Dense(20, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_5 = Sequential([
    Input(shape=(21,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(3, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_6 = Sequential([
    Input(shape=(21,)),
    Dense(30, activation='relu'),
    Dense(20, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid')
])



def sensitivity_stats(f, n):
    total_changes = 0
    sensitivities = []

    for x in itertools.product([0, 1], repeat=n):
        fx = f(x)
        sensitivity = 0
        for i in range(n):
            x_flip = list(x)
            x_flip[i] ^= 1
            f_flip = f(tuple(x_flip))
            change = int(fx != f_flip)
            sensitivity += change
            total_changes += change

        sensitivities.append(sensitivity)

    avg_sens = total_changes / (2**n * n)
    mean_sens = sum(sensitivities) / len(sensitivities)
    var_sens = sum((s - mean_sens) ** 2 for s in sensitivities) / len(sensitivities)
    norm_var = var_sens / ((n**2)/4) if n > 0 else 0


    print(f"Sensibilidad media: {avg_sens}")
    print(f"Varianza sensibilidad (normalizada): {norm_var}")

red_neuronal(model_1)
red_neuronal(model_2)
red_neuronal(model_3)
red_neuronal(model_4)
red_neuronal(model_5)
red_neuronal(model_6)
sensitivity_stats(primer_bit, 21)
