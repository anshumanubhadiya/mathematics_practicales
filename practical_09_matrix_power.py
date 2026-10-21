"""Practical 09: Compute the power of a matrix (A^3) using eigen-decomposition."""
import numpy as np

# Matrix A define karna
A = np.array([[2, 0], [0, 3]])

# Eigenvalues aur Eigenvectors calculate karna
eigenvals, eigenvecs = np.linalg.eig(A)

P = eigenvecs
D = np.diag(eigenvals)

# Eigen-decomposition formula: A^k = P @ (D^k) @ P_inv
A_cubed = P @ (D**3) @ np.linalg.inv(P)

# Result print karna
print("A^3:\n", A_cubed)