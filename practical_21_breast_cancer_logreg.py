"""Practical 21: Logistic regression with Gradient Descent on the Breast Cancer dataset."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Breast Cancer dataset load karna aur pehle 2 features select karna
data = load_breast_cancer()
X = data.data[:, :2]  # mean radius, mean texture
y = data.target

# Bias term add karna
X = np.hstack([np.ones((X.shape[0], 1)), X])
w = np.zeros(X.shape[1])

alpha = 0.01
epochs = 500
loss_history = []

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent Loop
for _ in range(epochs):
    z = X @ w
    p = sigmoid(z)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    loss_history.append(loss)

    grad = X.T @ (p - y) / len(y)
    w -= alpha * grad

print(f"Weights: {w}")