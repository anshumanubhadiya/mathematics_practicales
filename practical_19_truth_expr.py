"""Practical 19: Print the truth table of the expression (A or B) and (not A)."""
import itertools

print("A B | A v B | ~A | (A v B) ^ ~A")
for A, B in itertools.product([True, False], repeat=2):
    res = (A or B) and (not A)
    print(f"{str(A):<5} {str(B):<5} | {str(A or B):<5} | {str(not A):<5} | {res}")