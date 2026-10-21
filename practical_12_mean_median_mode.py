"""Practical 12: Compute the mean, median and mode of a dataset."""
import numpy as np
from scipy import stats

# Dataset
data = [2, 4, 7, 7, 7, 9, 10]

# Mean, Median, Mode calculation
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True)

# Results print karna
print(f"Mean = {mean}")
print(f"Median = {median}")
print(f"Mode = {mode.mode}")