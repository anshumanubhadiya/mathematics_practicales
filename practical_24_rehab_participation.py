"""Practical 24: Model patient participation in a rehabilitation program using logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression

np.random.seed(42)  # reproducible output ke liye

# Synthetic dataset simulation
n = 1000
distance = np.random.normal(10, 5, n)
age = np.random.normal(65, 12, n)
no_car = np.random.binomial(1, 0.3, n)

# Logits generation
logits = 1.7760 - 0.0675 * distance - 0.0199 * age - 1.8868 * no_car
p = 1 / (1 + np.exp(-logits))
participate = (np.random.rand(n) < p).astype(int)

X = np.column_stack([distance, age, no_car])
model = LogisticRegression().fit(X, participate)

print("Estimated coefficients:", model.coef_, "Intercept:", model.intercept_)
print("Training accuracy:", model.score(X, participate))