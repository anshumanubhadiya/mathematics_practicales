"""Practical 25: Predict customer churn using logistic regression."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Simulate customer dataset (1000 samples)
np.random.seed(42)
n_samples = 1000
tenure = np.random.randint(1, 60, n_samples)
monthly_charges = np.random.uniform(20, 120, n_samples)
num_support_calls = np.random.poisson(1.5, n_samples)

# Generate target churn using sigmoid probability
churn_prob = 1 / (1 + np.exp(-(-0.02 * tenure + 0.01 * monthly_charges + 0.4 * num_support_calls - 0.5)))
churn = np.random.binomial(1, churn_prob)

df = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'num_support_calls': num_support_calls,
    'churn': churn
})

# Train/Test Split
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", model.score(X_test, y_test))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))