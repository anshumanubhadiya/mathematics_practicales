"""Practical 20: Linear regression with Gradient Descent on the Iris dataset."""
import numpy as np
from sklearn.datasets import load_iris

# Load Iris dataset and select features
data = load_iris()
X = data.data[:, 0]  # Sepal length
y = data.data[:, 2]  # Petal length

# Normalization
X_norm = (X - np.mean(X)) / np.std(X)

# Gradient Descent Initialization
theta0, theta1 = 0.0, 0.0
alpha = 0.1
epochs = 200
m = len(X_norm)

# Optimization Loop
for _ in range(epochs):
    y_pred = theta0 + theta1 * X_norm
    error = y_pred - y

    d_theta0 = (1 / m) * np.sum(error)
    d_theta1 = (1 / m) * np.sum(error * X_norm)

    theta0 -= alpha * d_theta0
    theta1 -= alpha * d_theta1

print(f"Trained theta0={theta0:.3f}, theta1={theta1:.3f}")