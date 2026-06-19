import numpy as np
import time
import matplotlib.pyplot as plt

# ── Gaussian Elimination (no pivoting) ───────────────────────
def gausselim(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    # Forward elimination
    for k in range(n):
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i]     -= factor * b[k]

    # Back substitution
    x = np.zeros(n)
    for k in range(n-1, -1, -1):
        x[k] = (b[k] - A[k, k+1:] @ x[k+1:]) / A[k, k]
    return x

# ── Timing over a range of n ──────────────────────────────────
n_vals = np.logspace(1, 2.8, 20, dtype=int)   # n from ~10 to ~630
times  = []

for n in n_vals:
    A = np.random.rand(n, n) + n * np.eye(n)   # diag-dominant → stable
    b = np.random.rand(n)
    repeats = max(1, int(500 / n))              # more repeats for small n

    start = time.perf_counter()
    for _ in range(repeats):
        gausselim(A.copy(), b.copy())
    elapsed = (time.perf_counter() - start) / repeats
    times.append(elapsed)

n_vals = np.array(n_vals, dtype=float)
times  = np.array(times)

# ── Log-log fit to extract empirical slope ────────────────────
log_n = np.log10(n_vals)
log_t = np.log10(times)
slope, intercept = np.polyfit(log_n, log_t, 1)
fit_line = 10**(intercept + slope * log_n)

print(f"Empirical slope (log-log fit): {slope:.3f}  →  O(n^{slope:.2f})")

# ── Reference O(n^3) line ─────────────────────────────────────
ref = (n_vals / n_vals[0])**3 * times[0]

# ── Plot ──────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))
plt.loglog(n_vals, times,    'bo-', ms=5, label='Measured time')
plt.loglog(n_vals, fit_line, 'r--',       label=f'Fit slope = {slope:.2f}')
plt.loglog(n_vals, ref,      'k:',        label='Reference $O(n^3)$')
plt.xlabel('Matrix size $n$')
plt.ylabel('Time (seconds)')
plt.title('Gaussian Elimination — Time Complexity (log-log)')
plt.legend(); plt.grid(True, which='both', ls=':')
plt.tight_layout(); plt.show()
