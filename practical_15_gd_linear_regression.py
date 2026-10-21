"""Practical 15: Gradient Descent on a linear regression cost function."""
import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1, 2, 3, 4])
y = np.array([3, 5, 7, 9])  # y = 2x + 1

# Parameters Initialize karna
theta0 = 0
theta1 = 0
alpha = 0.1
epochs = 100
m = len(X)

# Loss History
loss_history = []

# Gradient Descent Loop
for epoch in range(epochs):
    y_pred = theta0 + theta1 * X
    error = y_pred - y
    cost = np.mean(error ** 2)
    loss_history.append(cost)

    # Gradients calculate karna
    d_theta0 = (2 / m) * np.sum(error)
    d_theta1 = (2 / m) * np.sum(error * X)

    # Parameters update karna
    theta0 = theta0 - alpha * d_theta0
    theta1 = theta1 - alpha * d_theta1

print(f"Final (theta0, theta1): ({theta0:.4f}, {theta1:.4f})")

# Plot MSE Loss
plt.plot(loss_history)
plt.title("MSE Loss over Iterations")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()