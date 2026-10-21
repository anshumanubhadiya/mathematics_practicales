"""Practical 13: Compute the sample variance and standard deviation of a dataset."""
import numpy as np

# Sample dataset
data = [2, 3, 4, 5, 6, 7]

# Sample Variance aur Standard Deviation calculate karna (ddof=1)
variance = np.var(data, ddof=1)
std_dev = np.std(data, ddof=1)

# Results print karna
print(f"Variance = {variance}")
print(f"Standard Deviation = {std_dev}")