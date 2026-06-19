import numpy as np
import matplotlib.pyplot as plt

# MC function
def mc_pi(n=10_000, rng=None):
    rng = rng or np.random.default_rng()
    x, y = rng.uniform(0, 1, (2, n))
    return 4 * np.mean(x**2 + y**2 < 1)

# Repeat 4000 times
rng = np.random.default_rng(42)
results = np.array([mc_pi(rng=rng) for _ in range(4000)])

mu, sigma = results.mean(), results.std()
print(f"Mean μ = {mu:.6f}")
print(f"Std σ = {sigma:.6f}")
print(f"Error |μ-π| = {abs(mu - np.pi):.6f}")

# P
in_band = np.mean(np.abs(results - mu) < sigma)
print(f"\nP(within ±σ of mean) = {in_band:.4f}")

# Histogram 
plt.figure(figsize=(8, 4))
plt.hist(results, bins=60, color='steelblue', edgecolor='white', density=True)
plt.axvline(mu, color='red',    lw=2, label=f'Mean = {mu:.5f}')
plt.axvline(mu - sigma, color='orange', lw=1.5, linestyle='--', label=f'μ ± σ  (σ={sigma:.5f})')
plt.axvline(mu + sigma, color='orange', lw=1.5, linestyle='--')
plt.axvline(np.pi, color='green', lw=1.5, linestyle=':', label=f'True π = {np.pi:.5f}')
plt.xlabel('Estimated π')
plt.ylabel('Density')
plt.title('Monte Carlo π — 4000 runs × 10⁴ points')
plt.legend()
plt.tight_layout()
plt.show()
