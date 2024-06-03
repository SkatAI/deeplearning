import numpy as np


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])
w = np.ones(X.shape[1])
# w = np.array([0, 1])
b = 0
alpha = 0.1

n_iters = 50

def predict(x, w, b ):
    if (np.dot(x, w) + b) >= 0 :
        return 1
    else:
        return 0


for _ in range(n_iters):
    idx = np.random.randint(0, len(y))
    xi = X[idx]
    target = y[idx]
    # print(xi, target, predict(xi, w,b))
    # Compute the prediction and update weights and bias
    if target != predict(xi, w,b):
        w =  w + alpha * target * xi
        b = b + alpha * target

        print(idx, xi, target, predict(xi, w,b), w)
    else:
        print("-",idx, xi, target, predict(xi, w,b), w)

    # else:
    #     print('-')



for
    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iters):
            # Randomly pick an input sample
            idx = np.random.randint(0, len(y))
            xi = X[idx]
            target = y[idx]
            # Compute the prediction and update weights and bias
            if target != self.predict(xi):
                self.weights += target * xi
                self.bias += target

    def predict(self, X):
        # Linear combination + bias
        linear_output = np.dot(X, self.weights) + self.bias
        # Apply step function
        return np.where(linear_output >= 0, 1, 0)

# Example usage
if __name__ == "__main__":
    # Training data: OR logic gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    perceptron = Perceptron(n_iters=100)
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)

    print("Predictions:", predictions)
