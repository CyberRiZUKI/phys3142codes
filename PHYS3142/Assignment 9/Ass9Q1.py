import numpy as np
import matplotlib.pyplot as plt

C, T, x0 = 1.0, 0.9, 1.0

# runner
def run_relaxation(omega=1.0, n=50):
    x = x0
    history = [x]
    for _ in range(n):
        x = (1 - omega) * x + omega * np.tanh(C * x / T)
        history.append(x)
    return np.array(history)

# (a)
T_vals, x_final_std, x_final_over = np.linspace(0.1, 2, 500), [], []
for T_i in T_vals:
    x = 1.0
    for _ in range(50):
        x = np.tanh(C * x / T_i)                        # standard
    x_final_std.append(x)

    x = 1.0
    for _ in range(50):
        x = (1 - 1.5) * x + 1.5 * np.tanh(C * x / T_i) # over-relaxation
    x_final_over.append(x)

# (b) & (c)
xs          = run_relaxation(omega=1.0, n=50)
xs_over     = run_relaxation(omega=1.5, n=50)
x_star      = xs[-1]
x_star_over = xs_over[-1]

i_vals      = np.arange(5, 31)
actual      = np.abs(xs[i_vals] - x_star)
estimated   = np.abs((xs[i_vals-1] - xs[i_vals])**2 /
                     (2*xs[i_vals-1] - xs[i_vals-2] - xs[i_vals]))
actual_over = np.abs(xs_over[i_vals] - x_star_over)
estimated_over = np.abs((xs_over[i_vals-1] - xs_over[i_vals])**2 /
                        (2*xs_over[i_vals-1] - xs_over[i_vals-2] - xs_over[i_vals]))

# (a)
plt.figure(figsize=(6, 4))
plt.plot(T_vals, x_final_std,  'b',    label=r'Standard ($\omega=1.0$)')
plt.plot(T_vals, x_final_over, 'r--',  label=r'Over-relax ($\omega=1.5$)')
plt.axvline(C, color='k', linestyle=':', label='$T_c = 1$')
plt.xlabel('Temperature T'); plt.ylabel('Magnetization x')
plt.title('(a)+(c) Phase Transition — Standard vs Over-Relaxation')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# (b)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, log in zip(axes, [False, True]):
    ax.plot(i_vals, actual,    'b-o',  ms=4, label='Actual')
    ax.plot(i_vals, estimated, 'r--s', ms=4, label='Estimated')
    if log: ax.set_yscale('log')
    ax.set_xlabel('Iteration i'); ax.set_ylabel('Error')
    ax.set_title('(b) Standard Error — ' + ('Log scale' if log else 'Linear'))
    ax.legend(); ax.grid(True)
plt.tight_layout(); plt.show()

# (c)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, log in zip(axes, [False, True]):
    ax.plot(i_vals, actual,      'b-o',  ms=4, label=r'Standard ($\omega=1.0$)')
    ax.plot(i_vals, actual_over, 'r--s', ms=4, label=r'Over-relax ($\omega=1.5$)')
    if log: ax.set_yscale('log')
    ax.set_xlabel('Iteration i'); ax.set_ylabel('Actual Error')
    ax.set_title('(c) Standard vs Over-Relaxation — ' + ('Log scale' if log else 'Linear'))
    ax.legend(); ax.grid(True)
plt.tight_layout(); plt.show()
