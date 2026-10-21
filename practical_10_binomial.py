"""Practical 10: Binomial probability of getting exactly 3 heads in 5 tosses of a fair coin."""
from scipy.stats import binom

n = 5     # number of trials
p = 0.5   # probability of success
k = 3     # number of successes

# PMF calculate karna
prob = binom.pmf(k, n, p)
print(f"P(X={k}) = {prob:.4f}")