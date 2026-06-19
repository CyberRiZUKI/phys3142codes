import numpy as np
import matplotlib.pyplot as plt

g, L, m = 4, 1, 1
t = np.linspace(0, 5*np.pi, 101)
dt = t[1] - t[0]
T1 = 2 * np.pi * np.sqrt(L / g)          

# Derivative function 
def f(state):
    th, w = state
    return np.array([w, -(g/L) * np.sin(th)])

# RK2 midpoint 
def rk2(th0, w0):
    s = np.array([th0, w0])
    hist = [s.copy()]
    for _ in t[:-1]:
        k1 = f(s)
        k2 = f(s + dt/2 * k1)
        s  = s + dt * k2
        hist.append(s.copy())
    return np.array(hist)

# RK4 
def rk4(th0, w0):
    s = np.array([th0, w0])
    hist = [s.copy()]
    for _ in t[:-1]:
        k1 = f(s)
        k2 = f(s + dt/2 * k1)
        k3 = f(s + dt/2 * k2)
        k4 = f(s + dt    * k3)
        s  = s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        hist.append(s.copy())
    return np.array(hist)

# Energy
def energy(hist, th0):
    th, w = hist[:,0], hist[:,1]
    return m*g*L*(np.cos(th0) - np.cos(th)) + 0.5*m*L**2*w**2

# Part (a): θ₀ = 0.7 — nonlinear RK2 vs linear approximation
th0_a   = 0.7
sol_a   = rk2(th0_a, 0)
linear  = th0_a * np.cos(2*np.pi*t / T1)

plt.figure(figsize=(7, 4))
plt.plot(t, sol_a[:,0], 'b-',  label='Nonlinear (RK2)')
plt.plot(t, linear,     'r--', label='Linear approx')
plt.xlabel('t'); plt.ylabel('θ (rad)')
plt.title('(a) Nonlinear vs Linear Pendulum  θ₀=0.7')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# Part (b): θ₀ = 1 — total energy, RK2 vs RK4
th0_b = 1.0
E_rk2 = energy(rk2(th0_b, 0), th0_b)
E_rk4 = energy(rk4(th0_b, 0), th0_b)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, E, label, color in zip(axes,
                                [E_rk2,       E_rk4],
                                ['RK2',        'RK4'],
                                ['b',          'r']):
    ax.plot(t, np.abs(E), color)                          # ← abs(E)
    ax.set_title(f'(b) |Total Energy Error| — {label}  θ₀=1')
    ax.set_xlabel('t')
    ax.set_ylabel('|Energy|')
    ax.grid(True)

plt.tight_layout()
plt.show()

