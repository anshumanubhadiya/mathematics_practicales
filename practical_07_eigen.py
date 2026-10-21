"""Practical 07: Calculate the eigenvalues and eigenvectors of a matrix."""
import numpy as np

# Matrix define karna
A = np.array([[2, 1], [0, 3]])

# Eigenvalues aur Eigenvectors calculate karna
eigenvals, eigenvecs = np.linalg.eig(A)

# Results print karna
print("Eigenvalues:", eigenvals)
print("Eigenvectors:\n", eigenvecs)