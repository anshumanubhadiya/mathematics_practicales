"""Practical 16: Gradient Descent on a multivariable quadratic function."""

# Initialization
theta1, theta2 = 4.0, 4.0
alpha = 0.1
epochs = 50

theta1_list, theta2_list = [], []

# Gradient Descent Loop
for _ in range(epochs):
    grad1 = 2 * theta1
    grad2 = 4 * theta2

    theta1 = theta1 - alpha * grad1
    theta2 = theta2 - alpha * grad2

    theta1_list.append(theta1)
    theta2_list.append(theta2)

print(f"Final (theta1, theta2): ({theta1:.4f}, {theta2:.4f})")