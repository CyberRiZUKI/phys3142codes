import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def integrand(x):
    """sqrt(tan x) / x, with limit = 1 at x = 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x == 0, 1.0, np.sqrt(np.tan(x)) / x)

I_exact, _ = integrate.quad(
    lambda x: np.sqrt(np.tan(x)) / x, 0, 1,
    limit=200, points=[1e-10]
)
print(f"Exact value: I = {I_exact:.6f}")


N_samples = 10**6
N_MC      = 100
N_repeat  = 50000
rng       = np.random.default_rng(42)


# Uniform random numbers distribution
uniform_samples = rng.uniform(0, 1, N_samples)

fig, ax = plt.subplots(figsize=(7, 4))
data = uniform_samples
bins = np.linspace(np.min(data), np.max(data), 501)
ax.hist(data, bins=bins, color='steelblue', edgecolor='none', density=True)
mu_u  = np.mean(data)
sig_u = np.std(data, ddof=1)
sig_x_u = sig_u / np.sqrt(N_samples)
ymax = ax.get_ylim()[1]
ax.vlines([mu_u - sig_x_u, mu_u + sig_x_u], 0, ymax,
          colors='grey', linestyles='--', lw=1.5, label=r'$\mu \pm \sigma_{\bar{x}}$')
ax.set_xlim(mu_u - 2*sig_u, mu_u + 2*sig_u)
ax.set_title("Distribution of Uniform Random Numbers ($10^6$ samples)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("fig1_uniform_rng.png", dpi=150)
plt.show()


# Non-uniform random numbers

nonuniform_samples = rng.uniform(0, 1, N_samples) ** 2

fig, ax = plt.subplots(figsize=(7, 4))
data = nonuniform_samples
bins = np.linspace(np.min(data), np.max(data), 501)
ax.hist(data, bins=bins, color='darkorange', edgecolor='none', density=True)
x_plot = np.linspace(0.001, 1, 500)
ax.plot(x_plot, 0.5 / np.sqrt(x_plot), 'k--', lw=1.5,
        label=r'$p(x)=\frac{1}{2\sqrt{x}}$')
mu_nu  = np.mean(data)
sig_nu = np.std(data, ddof=1)
sig_x_nu = sig_nu / np.sqrt(N_samples)
ymax = ax.get_ylim()[1]
ax.vlines([mu_nu - sig_x_nu, mu_nu + sig_x_nu], 0, ymax,
          colors='grey', linestyles='--', lw=1.5, label=r'$\mu \pm \sigma_{\bar{x}}$')
ax.set_xlim(mu_nu - 2*sig_nu, mu_nu + 2*sig_nu)
ax.set_title(r"Non-Uniform Random Numbers  $\omega(x)=1/\sqrt{x}$  ($10^6$ samples)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig("fig2_nonuniform_rng.png", dpi=150)
plt.show()

# Helper: plot histogram of I estimates
def plot_I_histogram(estimates, I_exact, title, filename, color='steelblue'):
    mu      = np.mean(estimates)
    sigma   = np.std(estimates, ddof=1)          
    sigma_x = sigma / np.sqrt(N_MC)              

    data = estimates
    bins = np.linspace(np.min(data), np.max(data), 501)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=bins, color=color, edgecolor='none', density=True)

    ymax = ax.get_ylim()[1]

    # Grey lines
    ax.vlines([mu - sigma_x, mu + sigma_x], 0, ymax,
              colors='grey', linestyles='--', lw=1.5,
              label=r'$\mu \pm \sigma_{\bar{x}}$')

    # Red line
    ax.vlines(I_exact, 0, ymax,
              colors='red', linestyles='-', lw=2,
              label=f'Exact = {I_exact:.5f}')

    ax.set_xlim(mu - 2*sigma, mu + 2*sigma)

    ax.set_title(title)
    ax.set_xlabel("Estimate of $I$")
    ax.set_ylabel("Count / Density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

    return mu, sigma, sigma_x

# Uniform MC

x_all_a    = rng.uniform(0, 1, (N_repeat, N_MC))
estimates_a = np.mean(integrand(x_all_a), axis=1)   

mu_a, sigma_a, sigma_x_a = plot_I_histogram(
    estimates_a, I_exact,
    title="(a) Uniform MC — Estimates of $I$",
    filename="fig3_uniform_MC.png",
    color='steelblue'
)
print(f"\n(a) μ = {mu_a:.5f},  σ = {sigma_a:.5f},  σ_x̄ = {sigma_x_a:.5f}")
in_range_a = (mu_a - sigma_x_a) < I_exact < (mu_a + sigma_x_a)
print(f"    True value in [μ-σ_x̄, μ+σ_x̄]? {in_range_a}")

# Importance Sampling

u_all_b    = rng.uniform(0, 1, (N_repeat, N_MC))
x_all_b    = u_all_b ** 2                              
g_all_b    = integrand(x_all_b) * 2.0 * np.sqrt(x_all_b)
estimates_b = np.mean(g_all_b, axis=1)

mu_b, sigma_b, sigma_x_b = plot_I_histogram(
    estimates_b, I_exact,
    title=r"(b) Importance Sampling  $\omega(x)=1/\sqrt{x}$ — Estimates of $I$",
    filename="fig4_nonuniform_MC.png",
    color='darkorange'
)
print(f"\n(b) μ = {mu_b:.5f},  σ = {sigma_b:.5f},  σ_x̄ = {sigma_x_b:.5f}")
in_range_b = (mu_b - sigma_x_b) < I_exact < (mu_b + sigma_x_b)
print(f"    True value in [μ-σ_x̄, μ+σ_x̄]? {in_range_b}")
print(f"    σ^a_x̄ / σ^b_x̄ = {sigma_x_a / sigma_x_b:.4f}")


# Metropolis Algorithm

def metropolis_chain(n_total, delta=0.5, x0=0.5):
    """
    Metropolis chain targeting p(x) ∝ 1/√x on (0,1).
    Acceptance ratio: p(x')/p(x) = √(x/x').
    All random draws pre-generated for speed.
    """
    steps    = rng.uniform(-delta, delta, n_total)
    u_accept = rng.uniform(0, 1,          n_total)
    samples  = np.empty(n_total)
    x = x0
    for i in range(n_total):
        x_new = x + steps[i]
        if 0.0 < x_new < 1.0 and u_accept[i] < np.sqrt(x / x_new):
            x = x_new
        samples[i] = x
    return samples

BURN     = 2000
THIN     = 3
N_needed = BURN + N_repeat * N_MC * THIN   

print(f"\nRunning Metropolis chain ({N_needed:,} steps)...")
chain   = metropolis_chain(N_needed, delta=0.5)
thinned = chain[BURN::THIN]                            # discard burn-in, thin
x_blocks = thinned[:N_repeat * N_MC].reshape(N_repeat, N_MC)


g_blocks    = integrand(x_blocks) * 2.0 * np.sqrt(x_blocks)
estimates_c = np.mean(g_blocks, axis=1)
print("Done.")

mu_c, sigma_c, sigma_x_c = plot_I_histogram(
    estimates_c, I_exact,
    title=r"(c) Metropolis  $p(x)\propto 1/\sqrt{x}$ — Estimates of $I$",
    filename="fig5_metropolis_MC.png",
    color='mediumseagreen'
)
print(f"\n(c) μ = {mu_c:.5f},  σ = {sigma_c:.5f},  σ_x̄ = {sigma_x_c:.6f}")
in_range_c = (mu_c - sigma_x_c) < I_exact < (mu_c + sigma_x_c)
print(f"    True value in [μ-σ_x̄, μ+σ_x̄]? {in_range_c}")
print(f"    σ^a_x̄ / σ^c_x̄ = {sigma_x_a / sigma_x_c:.4f}")
