import numpy as np

rng = np.random.default_rng(42)
N = 100_000

while True:
    pts = rng.uniform(-1, 1, (N, 3))
    inside = (pts**2).sum(axis=1) < 1
    p_hat  = inside.mean()
    V = 8 * p_hat
    ci = 8 * 1.96 * np.sqrt(p_hat * (1 - p_hat) / N)

    if ci < 0.05:
        break
    N *= 2

print(f"N = {N:,}")
print(f"Volume = {V:.5f}")
print(f"Exact = {4/3 * np.pi:.5f}")
print(f"Error = {abs(V - 4/3*np.pi):.5f}")
print(f"95% CI = ± {ci:.5f}")
