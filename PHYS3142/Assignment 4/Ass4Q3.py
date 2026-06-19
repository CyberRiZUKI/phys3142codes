import numpy as np
import matplotlib.pyplot as plt
from gaussxw import gaussxwab

f      = lambda x: 5 * x**4 * np.exp(x**5)
a, b   = 0, 1
exact  = np.e - 1                          # analytical result
Ns     = [4, 8, 16, 32, 64, 128]

err_trap  = []
err_romb  = []
err_gauss = []

# Adaptive Trapezoidal Method
for N in Ns:
    x = np.linspace(a, b, N+1)
    h = (b - a) / N
    result = h * (0.5*f(x[0]) + np.sum(f(x[1:-1])) + 0.5*f(x[-1]))
    err_trap.append(abs(result - exact))

# Romberg's Method
def romberg_integrate(f, a, b, N):
    """Build Romberg table up to row where trapezoid uses N intervals."""
    max_i = int(np.log2(N)) + 1
    R = np.zeros((max_i, max_i))
    for i in range(max_i):
        ni = 2**i
        x  = np.linspace(a, b, ni+1)
        h  = (b - a) / ni
        R[i, 0] = h * (0.5*f(x[0]) + np.sum(f(x[1:-1])) + 0.5*f(x[-1]))
    for m in range(1, max_i):
        for i in range(m, max_i):
            R[i, m] = (4**m * R[i, m-1] - R[i-1, m-1]) / (4**m - 1)
    return R[max_i-1, max_i-1]

for N in Ns:
    result = romberg_integrate(f, a, b, N)
    err_romb.append(abs(result - exact))

# Gaussian Quadrature
def gauss_composite(f, a, b, N):
    """Split [a,b] into N panels, apply Gauss-2 on each."""
    panels = np.linspace(a, b, N+1)
    total  = 0.0
    for k in range(N):
        xg, wg = gaussxwab(2, panels[k], panels[k+1])
        total  += np.sum(wg * f(xg))
    return total

for N in Ns:
    result = gauss_composite(f, a, b, N)
    err_gauss.append(abs(result - exact))

# Plot
plt.figure(figsize=(7, 5))
plt.loglog(Ns, err_trap,  'b-o',  label='Trapezoidal')
plt.loglog(Ns, err_romb,  'r-s',  label="Romberg")
plt.loglog(Ns, err_gauss, 'g-^',  label='Gauss Quadrature')


plt.xlabel('Number of intervals N')
plt.ylabel('Absolute error')
plt.title(r'Error comparison: $\int_0^1 5x^4 e^{x^5}\,dx$')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig("q3_error_comparison.png", dpi=150)
plt.show()

# Summary table
print(f"\n{'N':>5} | {'Trap error':>12} | {'Romberg error':>14} | {'Gauss error':>12}")
print("-" * 52)
for i, N in enumerate(Ns):
    print(f"{N:>5} | {err_trap[i]:>12.2e} | {err_romb[i]:>14.2e} | {err_gauss[i]:>12.2e}")
