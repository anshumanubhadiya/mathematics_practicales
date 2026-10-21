"""Practical 23: Predict hospital readmission risk using logistic regression."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Synthetic dataset: Age, Hospital Stay, Comorbidity Count
np.random.seed(42)
n_samples = 500
age = np.random.randint(30, 90, n_samples)
stay = np.random.randint(1, 15, n_samples)
comorb = np.random.randint(0, 5, n_samples)

# Simulate readmission with probability
X = np.column_stack([age, stay, comorb])
logits = -2 + 0.03 * age + 0.2 * stay + 0.6 * comorb
p = 1 / (1 + np.exp(-logits))
y = (np.random.rand(n_samples) < p).astype(int)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Output Results
print("Coefficients:", model.coef_, "Intercept:", model.intercept_)
print("Test Accuracy:", model.score(X_test, y_test))