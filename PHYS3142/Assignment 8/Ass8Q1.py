import numpy as np


T = np.array([[0, 1, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 1, 0],
              [1, 0, 0, 0]])

M = np.array([[ 1,  2,  3,  4],   # row 0
              [ 5,  6,  7,  8],   # row 1
              [ 9, 10, 11, 12],   # row 2
              [13, 14, 15, 16]])  # row 3


M_prime = T @ M

print("T =");       print(T);       print()
print("M =");       print(M);       print()
print("M' = TM ="); print(M_prime)
