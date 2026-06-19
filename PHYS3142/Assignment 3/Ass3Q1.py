import numpy as np
import matplotlib.pyplot as plt

r = 1
t = np.linspace(0, 2 * np.pi, 1000) 

# Point at MIDPOINT of radius -> factor of r/2
x = t - (r / 2) * np.sin(t)
# Fix: change + to -
y = 1 - (r / 2) * np.cos(t) 


# Scatter plot
fig, ax = plt.subplots(figsize=(9, 4))

# Draw the rolling circle at t = pi for reference
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.pi + r * np.cos(theta), 1 + r * np.sin(theta), 'k-', lw=1.5)
ax.plot(np.pi, 1, 'ko', ms=5)

ax.scatter(x[::5], y[::5], s=30, facecolors='none', edgecolors='red', linewidths=0.5)

ax.set_title("Cycloid")
ax.set_xlabel("x (r)")
ax.set_ylabel("y (r)")
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2.1)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("cycloid.png", dpi=150)
plt.show()

# (b) Arc length
dx = np.diff(x)
dy = np.diff(y)
arc_length = np.sum(np.sqrt(dx**2 + dy**2))
print(f"Arc length (numerical) : {arc_length:.6f} r")
print(f"Arc length (exact 6r)  : {6 * r:.6f} r")
