"""Practical 14: Gradient Descent on a univariate quadratic cost function."""
import matplotlib.pyplot as plt

# Initializer
theta = 0
alpha = 0.1
iterations = 20
history = []

# Gradient Descent Loop
for _ in range(iterations):
    grad = 2 * (theta - 3)
    theta = theta - alpha * grad
    history.append(theta)

print("Final theta:", theta)