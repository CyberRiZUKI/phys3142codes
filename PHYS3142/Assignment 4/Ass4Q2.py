import numpy as np
from scipy.integrate import quad

f = lambda x: np.exp(-x**2)
a, b = 0, 1
n = 5

R = np.zeros((n, n))

# trapezoid rule
for i in range(n):
    N  = 2**i
    h  = (b - a) / N
    x  = np.linspace(a, b, N+1)
    R[i, 0] = h * (0.5*f(x[0]) + np.sum(f(x[1:-1])) + 0.5*f(x[-1]))

# Richardson extrapolation
for m in range(1, n):
    for i in range(m, n):
        R[i, m] = (4**m * R[i, m-1] - R[i-1, m-1]) / (4**m - 1)

# Replicate romberg(show=True) output
print("── Romberg progress (replicating show=True) ──")
print(f"{'Iter':>5} {'N':>6} {'Trapezoid R[i,0]':>20} {'Best Romberg R[i,i]':>22}")
print("-" * 58)
for i in range(n):
    N_i = 2**i
    trap_val  = R[i, 0]
    romb_val  = R[i, i]
    print(f"{i+1:>5} {N_i:>6} {trap_val:>20.10f} {romb_val:>22.10f}")

# Full Romberg table
print("\nRomberg Table R[i,m]  (i=row, m=col, 1-indexed)\n")
print(f"{'':>4}", end="")
for m in range(n):
    print(f"  m={m+1}      ", end="")
print()
for i in range(n):
    print(f"i={i+1} ", end="")
    for m in range(n):
        val = f"{R[i,m]:.6f}" if m <= i else "   ---   "
        print(f"  {val:>9}", end="")
    print()

# quad verification
print("\n── scipy.integrate.quad verification ──")
result, error = quad(f, a, b)
print(f"quad result:     {result:.10f}")
print(f"estimated error: {error:.2e}")
print(f"Romberg result:  {R[n-1, n-1]:.10f}")
print(f"Difference:      {abs(R[n-1, n-1] - result):.2e}")
