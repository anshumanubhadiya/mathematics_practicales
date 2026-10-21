"""Practical 01: Addition, subtraction and multiplication of two matrices using NumPy."""
import numpy as np

# Matrices define karna
A = np.array([[2, 3], [4, 5]])
B = np.array([[6, 7], [8, 9]])

# Operations
add = A + B
sub = A - B
mul = A @ B  # Matrix multiplication

# Results print karna
print("Addition:\n", add)
print("Subtraction:\n", sub)
print("Multiplication:\n", mul)