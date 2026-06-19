import numpy as np
import matplotlib.pyplot as plt

alpha, beta, gamma = -1.5, 1, 1
N = 500

v = np.linspace(-4, 4, N)
px, py = np.meshgrid(v, v)

# Free En
p2 = px**2 + py**2
F = (alpha / 2) * p2 \
  + (beta  / 4) * p2**2 \
  + (gamma / 3) * (px * (px**2 - 3*py**2) + py * (3*px**2 - py**2))

# Truncate outside
F[(F < -5) | (F > 15)] = np.nan

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

im = ax.contourf(px, py, F,levels=100,cmap='RdBu_r',vmin=-5,vmax=15)

plt.colorbar(im, ax=ax)
ax.set_aspect('equal')                          
ax.set_xlabel(r'$p_x$')
ax.set_ylabel(r'$p_y$')
ax.set_title('Free Energy')

plt.tight_layout()
plt.savefig("free_energy.png", dpi=150, transparent=True)  
plt.show()
