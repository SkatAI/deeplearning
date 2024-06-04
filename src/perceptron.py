import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, x):
        x = np.insert(x, 0, 1)  # Adding bias term to the input
        return 1 if np.dot(self.weights, x) >= 0 else 0

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # print(f" -- Epoch {epoch+1}/{self.epochs}")
            # print(self.weights)
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error
            accuracy = self.evaluate(X, y)
            print(f" E [{epoch+1}/{self.epochs}] Accuracy: {accuracy * 100:.4f}%")


    def evaluate(self, X, y):
        predictions = [self.predict(xi) for xi in X]
        accuracy = np.mean(predictions == y)
        return accuracy

# Generate more complex training data
X, y = make_blobs(n_samples=400, centers=2, random_state=42, cluster_std=3)

# Visualize the training data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.show()

# Train and evaluate the perceptron
perceptron = Perceptron(input_size=2, learning_rate=0.0001, epochs=20)
perceptron.fit(X, y)
accuracy = perceptron.evaluate(X, y)
print(f"Accuracy: {accuracy * 100}%")

# Test predictions
# for x in X[:10]:  # Display first 10 predictions for brevity
#     print(f"Input: {x}, Prediction: {perceptron.predict(x)}")
