import numpy as np
from scipy.integrate import quad

f_original = lambda x: np.cos(x) / x**0.99

# after substitution t = x^0.01
f_transformed = lambda t: 100 * np.cos(t**100)

# Method 1: Direct integration (skip singularity, start from epsilon)
eps = 1e-6
result_direct, err_direct = quad(f_original, eps, 1)

# ── Method 2: Direct integration with singularity hint ──
result_hint, err_hint = quad(f_original, 0, 1,points=[1e-10], limit=200)

# ── Method 3: Substitution — smooth, no singularity ──
result_sub, err_sub = quad(f_transformed, 0, 1)

# ── Method 4: quad with weight='alg' (built-in algebraic singularity) ──
# integrand = cos(x) * x^(-0.99) → weight x^alpha, alpha = -0.99
result_weight, err_weight = quad(np.cos, 0, 1,weight='alg', wvar=(-0.99, 0))

# Layout
print(f"{'Method':<35} {'Result':>12}  {'Est. Error':>10}")
print(f"{'Direct (skip x<eps=1e-6)':<35} {result_direct:>12.8f}  {err_direct:>10.2e}")
print(f"{'Direct (singularity hint)':<35} {result_hint:>12.8f}  {err_hint:>10.2e}")
print(f"{'Substitution t=x^0.01':<35} {result_sub:>12.8f}  {err_sub:>10.2e}")
print(f"{'quad weight=alg (reference)':<35} {result_weight:>12.8f}  {err_weight:>10.2e}")

# Show error relative to substitution result
ref = result_sub
print(f"\nRelative error of direct method vs substitution:")
print(f"|direct - sub|/|sub| = {abs(result_direct - ref)/abs(ref):.4e}")
print(f"|hint   - sub|/|sub| = {abs(result_hint - ref)/abs(ref):.4e}")
