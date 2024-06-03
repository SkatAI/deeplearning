'''
- start with x 1 to 9
- puis 1 to 21 => exploding
=> normaliser avec max(x)

faire varier les epochs
lister les layers
regarder les coeffs de la layer

'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


xs = np.array([i for i in range(1, 11)], dtype=float)
coef = np.max(xs)
# coef = 1.0
xs = xs / coef
ys = 2.0 * xs - 1.0

# standardize


# ys = ys / coef

# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Define your model (should be a model with 1 dense layer and 1 unit)

# Define the model
model = Sequential([
    Dense(units=1, input_shape=[1])
])

model.summary()

# Compile your model
model.compile(optimizer='sgd', loss='mse')

# Train your model for 1000 epochs by feeding the i/o tensors
model.fit(xs, ys, epochs=1000, verbose = 0)

new_y = 10.0
print(f" predict({new_y}) ",model.predict(np.array([new_y/coef]))[0] * coef)

print("layer coefs ", model.layers[0].get_weights())

