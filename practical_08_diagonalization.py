"""Practical 08: Diagonalize a matrix using eigen-decomposition."""
import numpy as np

# Matrix A define karna
A = np.array([[2, 3], [4, 5]])

# Eigenvalues aur Eigenvectors calculate karna
eigenvals, eigenvecs = np.linalg.eig(A)

# Matrices P (eigenvectors), D (diagonal matrix of eigenvalues), aur P_inv
P = eigenvecs
D = np.diag(eigenvals)
P_inv = np.linalg.inv(P)

# Diagonalization / Reconstruction formula (P @ D @ P_inv)
A_diag = P @ D @ P_inv

# Result print karna
print("Diagonalized A:\n", A_diag)