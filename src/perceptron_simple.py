import numpy as np

# Input data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Labels OR
y = np.array([0, 1, 1, 1])
# XOR
# y = np.array([0, 1, 1, 0])

# Initialize weights and bias
w = np.zeros(X.shape[1])
w = [0.1, -0.2]
b = 0
# Number of iterations
n_iters = 1000
# Learning rate
lr = 0.001

for n in range(n_iters):
    # Random sample
    idx = np.random.randint(0, len(y))
    xi = X[idx]
    target = y[idx]

    # Compute the linear combination of inputs and weights plus bias
    linear_output = np.dot(xi, w) + b
    # Prediction using step function
    y_pred = 1 if linear_output >= 0 else 0

    # Update weights and bias
    update = lr * (target - y_pred)
    # update = target - y_pred
    w += update * xi
    b += update

    # test de convergence
    output = np.dot(X, np.transpose(w))  + b
    output = np.where(output < 0, 0, 1)
    loss = np.sum(np.abs(output-y))
    if loss == 0:
        print(f"-- convergence ! {n} iterations")
        print(f"Iteration {n+1}: Target={target}, Input={xi}, Weights={w}, Bias={b}, Loss={loss}")
        break;
    else:
        print(f"Iteration {n+1}: Target={target}, Input={xi}, Weights={w}, Bias={b}, Loss={loss}")

print("Final Weights:", w)
print("Final Bias:", b)
