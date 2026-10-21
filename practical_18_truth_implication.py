"""Practical 18: Print the truth table of logical implication."""
import itertools

print("A B A -> B")
for A, B in itertools.product([True, False], repeat=2):
    res = (not A) or B
    print(f"{str(A):<5} {str(B):<5} {res}")