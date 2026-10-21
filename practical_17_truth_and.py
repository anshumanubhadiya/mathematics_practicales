"""Practical 17: Print the truth table of logical AND."""
import itertools

print("A B A and B")
for A, B in itertools.product([True, False], repeat=2):
    print(f"{str(A):<5} {str(B):<5} {A and B}")