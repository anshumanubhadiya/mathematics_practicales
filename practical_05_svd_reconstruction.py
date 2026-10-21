"""Practical 05: Reconstruct a matrix from its SVD components."""
import numpy as np

# Matrix define karna aur SVD calculate karna
A = np.array([[3, 4]])
U, S, VT = np.linalg.svd(A)

# Singular values ko full matrix (diagonal) mein convert karna
S_full = np.zeros(A.shape, dtype=float)
np.fill_diagonal(S_full, S)

# Matrix reconstruction (U @ S_full @ VT)
A_reconstructed = U @ S_full @ VT

# Result print karna
print("Reconstructed A:\n", A_reconstructed)