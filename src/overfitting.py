import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train overfitting SGD model: small learning rate, no L2 regularization
sgd_no_l2 = SGDClassifier(alpha=0.0, max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.001, random_state=42)
sgd_no_l2.fit(X_train, y_train)

# Calculate score on train and test subsets
train_score_no_l2 = accuracy_score(y_train, sgd_no_l2.predict(X_train))
test_score_no_l2 = accuracy_score(y_test, sgd_no_l2.predict(X_test))

# Add L2 regularization
sgd_with_l2 = SGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3, learning_rate='constant', eta0=0.001, random_state=42)
sgd_with_l2.fit(X_train, y_train)

# Calculate score on train and test subsets
train_score_with_l2 = accuracy_score(y_train, sgd_with_l2.predict(X_train))
test_score_with_l2 = accuracy_score(y_test, sgd_with_l2.predict(X_test))

# Print the results
print(f"Train accuracy without L2: {train_score_no_l2}")
print(f"Test accuracy without L2: {test_score_no_l2}")
print(f"Train accuracy with L2: {train_score_with_l2}")
print(f"Test accuracy with L2: {test_score_with_l2}")

# Plot decision boundaries (only feasible if dimensionality reduction is applied)
# from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)

# sgd_no_l2.fit(X_train_pca, y_train)
# sgd_with_l2.fit(X_train_pca, y_train)

# def plot_decision_boundary(clf, X, y, ax, title):
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     ax.contourf(xx, yy, Z, alpha=0.8)
#     ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=20)
#     ax.set_title(title)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# plot_decision_boundary(sgd_no_l2, X_test_pca, y_test, ax1, "SGD without L2 regularization")
# plot_decision_boundary(sgd_with_l2, X_test_pca, y_test, ax2, "SGD with L2 regularization")

# plt.show()
