"""Practical 03: Calculate the determinant and rank of a matrix."""
import numpy as np

# Matrix define karna
A = np.array([[2, 3], [4, 5]])

# Determinant aur Rank calculate karna
det = np.linalg.det(A)
rank = np.linalg.matrix_rank(A)

# Results print karna
print("Determinant:", det)
print("Rank:", rank)