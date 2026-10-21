"""Practical 04: Compute the Singular Value Decomposition (SVD) of a matrix."""
import numpy as np

# Matrix define karna
A = np.array([[2, 3]])

# Singular Value Decomposition (SVD)
U, S, VT = np.linalg.svd(A)

# Results print karna
print("U:\n", U)
print("Singular Values:", S)
print("V^T (VT):\n", VT)