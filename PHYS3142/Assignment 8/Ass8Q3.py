import numpy as np
import time
from scipy.linalg import solve_banded

Vplus = 5.0


# (a)
A3 = np.array([[ 3, -1, -1],
               [-1,  4, -1],
               [-1, -1,  3]], dtype=float)
w3 = np.array([Vplus, Vplus, 0.0])

v3 = np.linalg.solve(A3, w3)
print("Part (a)  N=3")
print(f"  A =\n{A3}")
print(f"  w = {w3}")
print(f"  Solution V1,V2,V3 = {v3}")

# (b)
def build_system(N, Vplus=5.0):
    A = np.zeros((N, N))
    w = np.zeros(N)

    for i in range(N):
        # Diagonal
        if i == 0 or i == N - 1:
            A[i, i] = 3
        else:
            A[i, i] = 4

        # Left neighbours
        if i - 1 >= 0: A[i, i-1] = -1
        if i - 2 >= 0: A[i, i-2] = -1

        # Right neighbours
        if i + 1 < N:  A[i, i+1] = -1
        if i + 2 < N:  A[i, i+2] = -1

        # RHS vector
        if i == 0 or i == 1:
            w[i] = Vplus
        else:
            w[i] = 0.0

    return A, w

print("Part (b)  N=5")
A5, w5 = build_system(5)
print(f"  A =\n{A5}")
print(f"  w = {w5}")

print("\nPart (b)  N=6")
A6, w6 = build_system(6)
print(f"  A =\n{A6}")
print(f"  w = {w6}")

# (c)
print("Part (c)  Solving N=5 and N=6")
for N in [5, 6]:
    A, w = build_system(N)
    v = np.linalg.solve(A, w)
    print(f"  N={N}: V = {np.round(v, 4)}")

# (d) 
N = 10000
A_large, w_large = build_system(N)

# Method 1
t0 = time.perf_counter()
v_full = np.linalg.solve(A_large, w_large)
t_full = time.perf_counter() - t0

# Method 2
def build_banded(N, Vplus=5.0):
    ab = np.zeros((5, N))
    _, w = build_system(N, Vplus)

    for i in range(N):
        ab[2, i] = 3 if (i == 0 or i == N-1) else 4   # main diag
        if i + 1 < N: ab[1, i]   = -1                  # 1st superdiag
        if i + 2 < N: ab[0, i]   = -1                  # 2nd superdiag
        if i - 1 >= 0: ab[3, i]  = -1                  # 1st subdiag
        if i - 2 >= 0: ab[4, i]  = -1                  # 2nd subdiag
    return ab, w

ab, w_b = build_banded(N)

t1 = time.perf_counter()
v_band = solve_banded((2, 2), ab, w_b)
t_band = time.perf_counter() - t1

# Results
print(f"Part (d)  N=10000")
print(f"  numpy.linalg.solve  : {t_full:.4f} s")
print(f"  scipy solve_banded  : {t_band:.6f} s")
print(f"  Speedup             : {t_full/t_band:.1f}x")
print(f"  Max difference      : {np.max(np.abs(v_full - v_band)):.2e}")
