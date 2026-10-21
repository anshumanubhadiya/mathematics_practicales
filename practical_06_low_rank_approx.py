"""Practical 06: Find the low-rank approximation of a matrix using SVD."""
import numpy as np

# Matrix define karna aur SVD calculate karna
A = np.array([[2, 3]])
U, S, VT = np.linalg.svd(A)

# Keep only top-1 singular value
S_reduced = np.diag(S[:1])
U_reduced = U[:, :1]
VT_reduced = VT[:1, :]

# Low-rank matrix approximation
A_approx = U_reduced @ S_reduced @ VT_reduced

# Result print karna
print("Low-rank approximation:\n", A_approx)