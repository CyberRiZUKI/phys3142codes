import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────
g, v0, h = 10, 200, 0.1

# ── Derivative function ───────────────────────────────────────
def deriv(s, b):
    x, y, vx, vy = s
    speed = np.sqrt(vx**2 + vy**2)
    return np.array([vx, vy, -b*vx*speed, -g - b*vy*speed])

# ── RK4 trajectory: returns (x_arr, y_arr) ───────────────────
def trajectory(theta, b):
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    s   = np.array([0.0, 0.0, vx0, vy0])
    xs, ys = [0.0], [0.0]
    while True:
        k1 = deriv(s,           b)
        k2 = deriv(s + h/2*k1, b)
        k3 = deriv(s + h/2*k2, b)
        k4 = deriv(s + h*k3,   b)
        s  = s + h/6*(k1 + 2*k2 + 2*k3 + k4)
        xs.append(s[0]); ys.append(s[1])
        if s[1] < 0:                       # hit the ground
            # linear interpolation for exact landing x
            x_land = xs[-2] - ys[-2]*(xs[-1]-xs[-2])/(ys[-1]-ys[-2])
            return np.array(xs), np.array(ys), x_land

# ── Range function (scalar) ───────────────────────────────────
def range_fn(theta, b=0.003):
    _, _, x_land = trajectory(theta, b)
    return x_land

# ═══════════════════════════════════════════════════════════════
# Part (a): θ₀ = π/4, b=0 vs b=0.0003
# ═══════════════════════════════════════════════════════════════
theta_a = np.pi / 4
x0, y0, _ = trajectory(theta_a, b=0)
xb, yb, _ = trajectory(theta_a, b=0.0003)

plt.figure(figsize=(8, 4))
plt.plot(x0, np.clip(y0, 0, None), 'b-',  label='No air resistance (b=0)')
plt.plot(xb, np.clip(yb, 0, None), 'r--', label='Air resistance (b=0.0003)')
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.title('(a) Trajectory  θ₀ = π/4')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# ═══════════════════════════════════════════════════════════════
# Part (b): Golden Section Search for optimal θ, b=0.003
# ═══════════════════════════════════════════════════════════════
phi = (np.sqrt(5) - 1) / 2          
a, b_gs = 0.1, np.pi/2 - 0.01      

for _ in range(50):                  
    c = b_gs - phi * (b_gs - a)
    d = a    + phi * (b_gs - a)
    if range_fn(c) < range_fn(d):
        a = c
    else:
        b_gs = d

theta_opt = (a + b_gs) / 2
range_opt = range_fn(theta_opt)

print(f"Optimal angle : {np.degrees(theta_opt):.2f}°  ({theta_opt:.4f} rad)")
print(f"Maximum range : {range_opt:.1f} m")

# ── Plot range vs angle ───────────────────────────────────────
angles = np.linspace(0.1, np.pi/2 - 0.01, 60)
ranges = [range_fn(th) for th in angles]

plt.figure(figsize=(7, 4))
plt.plot(np.degrees(angles), ranges, 'b-')
plt.axvline(np.degrees(theta_opt), color='r', linestyle='--',
            label=f'Optimal θ = {np.degrees(theta_opt):.2f}°')
plt.xlabel('Launch angle θ₀ (degrees)'); plt.ylabel('Range (m)')
plt.title('(b) Range vs Launch Angle  (b=0.003)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
