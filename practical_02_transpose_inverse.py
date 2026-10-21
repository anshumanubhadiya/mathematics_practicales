"""Practical 02: Find the transpose and inverse of a matrix."""
import numpy as np

# Matrix define karna
A = np.array([[2, 3], [4, 5]])

# Transpose aur Inverse calculate karna
transpose = A.T
inverse = np.linalg.inv(A)

# Results print karna
print("Transpose:\n", transpose)
print("Inverse:\n", inverse)