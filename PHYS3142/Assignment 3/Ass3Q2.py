import numpy as np
import matplotlib.pyplot as plt

def make_grid(lim, N=500):
    v = np.linspace(-lim, lim, N)
    x, z = np.meshgrid(v, v)
    r = np.sqrt(x**2 + z**2)
    return x, z, r

# 2pz 
def psi_2pz(x, z, r):
    cosT = np.where(r > 0, z / r, 0)
    return r * np.exp(-r / 2) * cosT

# 3dz² 
def psi_3dz2(x, z, r):
    cos2T = np.where(r > 0, z**2 / r**2, 0)
    return r**2 * np.exp(-r / 3) * (3 * cos2T - 1)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 2pz
x, z, r = make_grid(lim=10)
psi = psi_2pz(x, z, r)
vmax = np.max(np.abs(psi))
im0 = axes[0].pcolormesh(x, z, psi, cmap='RdBu', vmin=-vmax, vmax=vmax, shading='auto')
axes[0].set_title(r'$2p_z$')
axes[0].set_xlabel(r'$x\ (a_0)$')
axes[0].set_ylabel(r'$z\ (a_0)$')
plt.colorbar(im0, ax=axes[0])

# 3dz²
x, z, r = make_grid(lim=20)
psi = psi_3dz2(x, z, r)
vmax = np.max(np.abs(psi))
im1 = axes[1].pcolormesh(x, z, psi, cmap='RdBu', vmin=-vmax, vmax=vmax, shading='auto')
axes[1].set_title(r'$3d_{z^2}$')
axes[1].set_xlabel(r'$x\ (a_0)$')
axes[1].set_ylabel(r'$z\ (a_0)$')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("orbitals.png", dpi=150)
plt.show()
