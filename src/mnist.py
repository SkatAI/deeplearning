# mnist

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
'''
- quelle dimensions de x_train, y_train etc ?
- quelle type ?
'''


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Define the model architecture

'''
- tester avec diffrente activation pour la couche du milieu
- tester avec diffrente dimension pour la couche du milieu
-
'''

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
'''
- tester different optimzer
- peut on utiliser une autre fonction de cout ?
- essayer avec d'autres metriques
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
'''
- faire varier le nombre d'epochs
- le batch size
'''
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

'''
- ajouter un call back pour memoriser la valeur de l;a fonction de cout a chauqe epoch
'''

# Evaluate the model
train_loss, train_accuracy = model.evaluate(x_train, y_train)
print(f'Train Loss: {train_loss:.4f}')
print(f'Train Accuracy: {train_accuracy:.4f}')
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

'''
si on augmente les epochs et reduit le batch size
est ce que ca overfit ?
'''

# Make predictions on new data
new_data = x_test[:5]  # Select the first 5 images from the test set
predictions = model.predict(new_data)
predicted_labels = np.argmax(predictions, axis=1)

# Print the predicted labels
print("Predicted Labels:")
print(predicted_labels)