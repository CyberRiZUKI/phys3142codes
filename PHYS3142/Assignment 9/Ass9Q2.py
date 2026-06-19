import numpy as np

# ── Core functions ─────────────────────────────────────────
f  = lambda x: x**5 / (np.exp(x) - 1)
g  = lambda x: (5 - x) * np.exp(x) - 5        # f'(x) numerator = 0
dg = lambda x: (4 - x) * np.exp(x)             # g'(x) for Newton

tol = 1e-6

# (a) 
x1, x2 = 0.1, 20.0
for i in range(1, 1000):
    xm = (x1 + x2) / 2
    if g(x1) * g(xm) < 0: x2 = xm
    else:                  x1 = xm
    if abs(x2 - x1) < tol:
        print(f"(a) Binary:      x_peak = {(x1+x2)/2:.8f},  iterations = {i}")
        break

# (b)
phi = (np.sqrt(5) - 1) / 2
x1, x4 = 0.1, 20.0
for i in range(1, 1000):
    x2 = x4 - phi * (x4 - x1)
    x3 = x1 + phi * (x4 - x1)
    if f(x2) > f(x3): x4 = x3
    else:              x1 = x2
    if abs(x4 - x1) < tol:
        print(f"(b) Golden:      x_peak = {(x1+x4)/2:.8f},  iterations = {i}")
        break

# (c)
for x1_init in [1, 5, 10]:
    x = float(x1_init)
    for i in range(1, 1000):
        x_new = x - g(x) / dg(x)
        if abs(x_new - x) < tol:
            x = x_new
            if x > 1.0:  # physical root must be > 0 and away from trivial root
                print(f"(c) Newton x1={x1_init}: x_peak = {x:.8f},  iterations = {i}")
            else:
                print(f"(c) Newton x1={x1_init}: converged to WRONG root x={x:.4f}, iterations = {i}")
            break
        x = x_new

