"""Practical 11: Poisson probability of exactly 1 call in a minute (avg rate 2 calls/min)."""
from scipy.stats import poisson

l = 2  # average rate (lambda)
k = 1  # number of occurrences

# PMF calculate karna
prob = poisson.pmf(k, l)
print(f"P(X={k}) = {prob:.4f}")