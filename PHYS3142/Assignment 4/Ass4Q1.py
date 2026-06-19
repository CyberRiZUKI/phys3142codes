import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

f      = lambda x: np.exp(x**2)
a, b   = 0, 1
exact, _ = quad(f, a, b)
Ns     = [2, 4, 8, 16, 32, 64, 128, 256]

err_left, err_mid, err_right = [], [], []

for N in Ns:
    h = (b - a) / N
    x = np.linspace(a, b, N+1)          # N+1 endpoints

    left  = h * np.sum(f(x[:-1]))        # left endpoints
    right = h * np.sum(f(x[1:]))         # right endpoints
    mid   = h * np.sum(f((x[:-1] + x[1:]) / 2))  # midpoints

    err_left.append(abs(left  - exact))
    err_right.append(abs(right - exact))
    err_mid.append(abs(mid   - exact))

# ── Log-log plot ──────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.loglog(Ns, err_left,  'b-o', label='Left-point')
plt.loglog(Ns, err_right, 'r-s', label='Right-point')
plt.loglog(Ns, err_mid,   'g-^', label='Midpoint')

# Reference lines
plt.loglog(Ns, [2/n     for n in Ns], 'k:',  alpha=0.5, label=r'$O(h)$')
plt.loglog(Ns, [1/n**2  for n in Ns], 'k--', alpha=0.5, label=r'$O(h^2)$')

plt.xlabel('Number of intervals N')
plt.ylabel('Absolute error')
plt.title(r'Rectangle rule errors: $\int_0^1 e^{x^2}\,dx$')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig("q1_rectangle_errors.png", dpi=150)
plt.show()

# ── Print table ───────────────────────────────────────
print(f"\n{'N':>6} | {'Left error':>12} | {'Right error':>12} | {'Mid error':>12}")
print("-" * 50)
for i, N in enumerate(Ns):
    print(f"{N:>6} | {err_left[i]:>12.2e} | {err_right[i]:>12.2e} | {err_mid[i]:>12.2e}")
print(f"\nExact (quad): {exact:.10f}")
