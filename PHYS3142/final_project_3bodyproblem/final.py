import numpy as np
import matplotlib.pyplot as plt
import time                                          

G     = 6.6738e-11
M_sun = 1.9891e30
M     = M_sun
R     = 1e13
vR    = np.sqrt(5*G*M/(4*R))
T_char= R/vR

M_jup = 1.898e27
a_J   = 7.783e11
v_J   = np.sqrt(G*M_sun/a_J)
omega = v_J/a_J
T_J   = 2*np.pi/omega

def rk4(s, dt, deriv, *args):
    k1 = deriv(s, *args)
    k2 = deriv(s + dt/2*k1, *args)
    k3 = deriv(s + dt/2*k2, *args)
    k4 = deriv(s + dt  *k3, *args)
    return s + dt/6*(k1 + 2*k2 + 2*k3 + k4)

# (a)
planets = {
    'Mercury': {'r':4.60e10,  'v':57200., 'T_known':88.0,  'color':'gray'},
    'Earth'  : {'r':1.471e11, 'v':30300., 'T_known':365.2, 'color':'royalblue'},
    'Mars'   : {'r':2.067e11, 'v':26400., 'T_known':687.0, 'color':'tomato'},
}

def deriv_1body(s, *_):
    x,y,vx,vy = s;  r3=(x**2+y**2)**1.5
    return np.array([vx, vy, -G*M_sun*x/r3, -G*M_sun*y/r3])

def simulate_planet(r0, v0, T_days, dt=3600):
    s = np.array([r0,0.,0.,v0])
    n = int(2*T_days*86400/dt)+1
    xs,ys = np.empty(n),np.empty(n)
    xs[0],ys[0] = s[0],s[1]
    angle,period,t = 0.,None,0.
    for i in range(1,n):
        prev = np.arctan2(s[1],s[0])
        s = rk4(s,dt,deriv_1body);  t+=dt
        xs[i],ys[i] = s[0],s[1]
        da = np.arctan2(s[1],s[0])-prev
        da -= 2*np.pi*(da>np.pi);  da += 2*np.pi*(da<-np.pi)
        angle += da
        if angle>=2*np.pi and period is None:
            period = t-(angle-2*np.pi)/abs(da)*dt
    return xs[:i+1],ys[:i+1],period

print("PART (a)...")
res_a = {n: simulate_planet(p['r'],p['v'],p['T_known']) for n,p in planets.items()}

print(f"\n{'Planet':<10}{'Simulated':>14}{'Known':>10}{'Err%':>8}")
for n,p in planets.items():
    Td = res_a[n][2]/86400
    print(f"  {n:<10}{Td:>12.2f}{p['T_known']:>10.1f}{abs(Td-p['T_known'])/p['T_known']*100:>7.3f}%")

fig,ax = plt.subplots(figsize=(7,7))
ax.plot(0,0,'o',color='gold',ms=20,markeredgecolor='orange',label='Sun',zorder=5)
for n,p in planets.items():
    xs,ys,_ = res_a[n]
    ax.plot(xs,ys,color=p['color'],lw=1.5,label=n)
    ax.plot(xs[0],ys[0],'o',color=p['color'],ms=7,markeredgecolor='k',zorder=4)
ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=.3)
ax.set_title('Part (a) — Mercury, Earth, Mars Orbits')
ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
fig.tight_layout(); fig.savefig('part_a_orbits.png',dpi=150)
print("Saved: part_a_orbits.png")


# (b)
def deriv_3body(s, *_):
    p   = s.reshape(3,4); pos,vel = p[:,:2],p[:,2:]
    d   = pos[:,None,:]-pos[None,:,:]         
    r2  = np.sum(d**2,axis=2); np.fill_diagonal(r2,1.)
    acc = -G*M*np.sum(d/(r2**1.5)[:,:,None],axis=1)
    ds  = np.zeros(12)
    ds[0::4],ds[1::4],ds[2::4],ds[3::4] = vel[:,0],vel[:,1],acc[:,0],acc[:,1]
    return ds

def simulate_3body(name, s0, dt, n):                
    traj=np.empty((n,12)); traj[0]=s0; s=s0.copy()
    t0=time.time()                                   
    for i in range(1,n):
        s=rk4(s,dt,deriv_3body); traj[i]=s
        if i % (n//20) == 0:                         
            pct=i/n*100; elapsed=time.time()-t0
            eta=elapsed/pct*(100-pct) if pct>0 else 0
            print(f"\r  {name}: {pct:5.1f}%  ETA {eta:5.1f}s", end='', flush=True)
    print(f"\r  {name}: done in {time.time()-t0:.1f}s          ")  
    return traj

V0 = np.sqrt(G*M/R)

def make_ic_3b(pos_coeffs, vel_coeffs, L, V):
    s = []
    for (px,py),(vx,vy) in zip(pos_coeffs, vel_coeffs):
        s += [px*L, py*L, vx*V, vy*V]
    return np.array(s, dtype=float)

L4 = 2*R;  V4 = np.sqrt(G*M/L4)
L8 = R/.97; V8 = np.sqrt(G*M/L8)

configs = {
    "1. Euler Collinear": {
        's0': np.array([R,0,0,vR, -R,0,0,-vR, 0,0,0,0], dtype=float),
        'dt':5e5, 'steps':400000, 'lim':2.5*R},

    "2. Lagrange Triangle": {
        's0': np.array([R,0,0,V0,
                        -R/2, R*np.sqrt(3)/2, -V0*np.sqrt(3)/2, -V0/2,
                        -R/2,-R*np.sqrt(3)/2,  V0*np.sqrt(3)/2, -V0/2], dtype=float),
        'dt':5e5, 'steps':400000, 'lim':2.5*R},

    "3. Figure-8": {
        's0': make_ic_3b(
            [( 0.97, -0.24308753), (-0.97,  0.24308753), (0.0,  0.0)],
            [( 0.46620369,  0.43236573), ( 0.46620369,  0.43236573), (-0.93240737, -0.86473146)],
            L8, V8),
        'dt':1e6, 'steps':500000, 'lim':1.8*L8},

    "4. Butterfly I": {
        's0': make_ic_3b(
            [( 0.3059,  0.0), (-0.3059,  0.0), (0.0,  0.0)],
            [( 0.3393,  0.5030), ( 0.3393,  0.5030), (-0.6786, -1.0060)],
            L4, V4),
        'dt':5e5, 'steps':500000, 'lim':1.5*L4},

    "5. Yin-Yang I": {
        's0': make_ic_3b(
            [( 0.5130,  0.4814), (-0.5130, -0.4814), (0.0,  0.0)],
            [( 0.3063,  0.1257), ( 0.3063,  0.1257), (-0.6126, -0.2514)],
            L4, V4),
        'dt':5e5, 'steps':500000, 'lim':2.0*L4},
}

colors = ['crimson','steelblue','gold']

print("\nPART (b)...")
trajs = {n: simulate_3body(n,c['s0'],c['dt'],c['steps']) for n,c in configs.items()}  

fig,axes = plt.subplots(2,3,figsize=(16,10))
for ax,(n,c) in zip(axes.flat,configs.items()):
    t=trajs[n]
    for b,col in enumerate(colors):
        ax.plot(t[:,4*b],t[:,4*b+1],color=col,lw=.6,alpha=.85,label=f'B{b+1}')
        ax.plot(t[0,4*b],t[0,4*b+1],'o',color=col,ms=5,markeredgecolor='k')
    lim=c['lim']; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_aspect('equal'); ax.set_title(n,fontsize=9,fontweight='bold')
    ax.legend(fontsize=6); ax.grid(alpha=.25)
axes.flat[-1].set_visible(False)
fig.suptitle('Part (b) — Three-Body Periodic Orbits',fontsize=13,fontweight='bold')
fig.tight_layout(); fig.savefig('part_b_orbits.png',dpi=150)
print("Saved: part_b_orbits.png")

# PART (c)
# Expand s(τ) = Σ c[n] τ^n,  τ = t/T_char
# Recurrence via Leibniz rule on r^{-3}:
#   u[n] = (1/2n·r²[0]) Σ_{k=1}^{n} (2k-3n)·r²[k]·u[n-k]
# Position/velocity advance:
#   px[n+1] = T_char·vx[n]/(n+1),  vx[n+1] = T_char·ax[n]/(n+1)

def conv(A,B,n):
    """Convolution (A*B)[n] = Σ_{k=0}^{n} A[k]*B[n-k]"""
    k=np.arange(n+1); return np.dot(A[:n+1],B[n::-1])

def inv_r3(r2,N):
    """Taylor coeffs of r^{-3} given r^2 coeffs, up to order N."""
    u=np.zeros(N+1); u[0]=r2[0]**-1.5
    for n in range(1,N+1):
        k=np.arange(1,n+1)
        u[n]=np.dot((2*k-3*n)*r2[1:n+1],u[n-1::-1])/(2*n*r2[0])
    return u

def taylor_coeffs(s0,N):
    """Return coeffs shape (N+1,12); τ=t/T_char."""
    px=np.zeros((3,N+1)); py=np.zeros((3,N+1))
    vx=np.zeros((3,N+1)); vy=np.zeros((3,N+1))
    for b in range(3):
        px[b,0],py[b,0],vx[b,0],vy[b,0] = s0[4*b],s0[4*b+1],s0[4*b+2],s0[4*b+3]
    for n in range(N):
        ax_n=np.zeros(3); ay_n=np.zeros(3)
        for b in range(3):
            for j in range(3):
                if j==b: continue
                dx=px[b,:n+1]-px[j,:n+1]; dy=py[b,:n+1]-py[j,:n+1]
                r2=np.array([conv(dx,dx,k)+conv(dy,dy,k) for k in range(n+1)])
                u =inv_r3(r2,n)
                ax_n[b]-=G*M*conv(dx,u,n)
                ay_n[b]-=G*M*conv(dy,u,n)
        for b in range(3):
            px[b,n+1]=T_char*vx[b,n]/(n+1)
            py[b,n+1]=T_char*vy[b,n]/(n+1)
            vx[b,n+1]=T_char*ax_n[b]/(n+1)
            vy[b,n+1]=T_char*ay_n[b]/(n+1)
    C=np.zeros((N+1,12))
    for b in range(3):
        C[:,4*b],C[:,4*b+1],C[:,4*b+2],C[:,4*b+3]=px[b],py[b],vx[b],vy[b]
    return C

def horner(C,tau):
    """Evaluate polynomial via Horner. tau:(T,) → (T,12)"""
    tau=np.atleast_1d(tau); r=np.outer(np.ones(len(tau)),C[-1])
    for c in C[-2::-1]: r=r*tau[:,None]+c
    return r

# Convergence test on Euler Collinear 
print("\nPART (c) — Taylor series...")
N_MAX  = 20
s0_t   = configs["1. Euler Collinear"]['s0']
dt_t   = configs["1. Euler Collinear"]['dt']
tau_ev = 0.5;  t_ev = tau_ev*T_char

# RK4 reference
s_ref=s0_t.copy()
for _ in range(int(t_ev/dt_t)): s_ref=rk4(s_ref,dt_t,deriv_3body)

C_full = taylor_coeffs(s0_t,N_MAX)
orders = np.arange(2,N_MAX+1)
errors = [np.linalg.norm(horner(C_full[:N+1],[tau_ev])[0,:6]-s_ref[:6])
          /np.linalg.norm(s_ref[:6]) for N in orders]

# rk4 reference trajectory
tau_arr = np.linspace(0,.8,300); t_arr=tau_arr*T_char
n_ref   = int(t_arr[-1]/dt_t)+1
traj_ref= np.empty((n_ref,12)); traj_ref[0]=s0_t; s=s0_t.copy()
for i in range(1,n_ref): s=rk4(s,dt_t,deriv_3body); traj_ref[i]=s
t_ref   = np.arange(n_ref)*dt_t

s_tay   = horner(C_full,tau_arr)
err_t   = [np.linalg.norm(s_tay[i,:6]-traj_ref[np.argmin(np.abs(t_ref-tau_arr[i]*T_char)),:6])
           /(np.linalg.norm(traj_ref[np.argmin(np.abs(t_ref-tau_arr[i]*T_char)),:6])+1e-30)
           for i in range(len(tau_arr))]

# (c) plot
fig,axes=plt.subplots(1,3,figsize=(16,5))

axes[0].semilogy(orders,errors,'o-',color='steelblue',lw=2,ms=5)
axes[0].set_xlabel('Order N'); axes[0].set_ylabel('Relative Error')
axes[0].set_title(f'Convergence at τ={tau_ev}'); axes[0].grid(alpha=.3)

axes[1].plot(t_ref/T_char,traj_ref[:,0]/R,color='crimson',lw=2,label='RK4',alpha=.8)
axes[1].plot(tau_arr,s_tay[:,0]/R,'--',color='navy',lw=2,label=f'Taylor N={N_MAX}')
axes[1].set_xlabel('τ = t/T_char'); axes[1].set_ylabel('x₁/R')
axes[1].set_title('Taylor vs RK4 — Body 1 x(t)'); axes[1].legend(); axes[1].grid(alpha=.3)

axes[2].semilogy(tau_arr,err_t,color='darkorange',lw=1.5)
axes[2].set_xlabel('τ = t/T_char'); axes[2].set_ylabel('Relative Error')
axes[2].set_title(f'Error Growth (N={N_MAX})'); axes[2].grid(alpha=.3)

fig.suptitle('Part (c) — Taylor Series: Convergence & Accuracy',fontsize=13,fontweight='bold')
fig.tight_layout(); fig.savefig('part_c_analysis.png',dpi=150)
print("Saved: part_c_analysis.png")

# Taylor vs RK4 for all Part (b) configs
fig,axes=plt.subplots(2,3,figsize=(16,10))
for ax,(n,c) in zip(axes.flat,configs.items()):
    s0c=c['s0']; dtc=c['dt']; lim=c['lim']
    Cc  = taylor_coeffs(s0c,N_MAX)
    tau = np.linspace(0,.6,200)
    st  = horner(Cc,tau)
    # RK4 reference
    nr=int(tau[-1]*T_char/dtc)+1; tr=np.empty((nr,12)); tr[0]=s0c; sc=s0c.copy()
    for i in range(1,nr): sc=rk4(sc,dtc,deriv_3body); tr[i]=sc
    for b,col in enumerate(colors):
        ax.plot(tr[:,4*b]/R,tr[:,4*b+1]/R,'-',color=col,lw=1.5,alpha=.7,label=f'B{b+1} RK4')
        ax.plot(st[:,4*b]/R,st[:,4*b+1]/R,'--',color=col,lw=1.5,label=f'B{b+1} Tay')
    ax.set_xlim(-lim/R,lim/R); ax.set_ylim(-lim/R,lim/R); ax.set_aspect('equal')
    ax.set_title(n,fontsize=9,fontweight='bold'); ax.legend(fontsize=5,ncol=2); ax.grid(alpha=.25)
axes.flat[-1].set_visible(False)
fig.suptitle(f'Part (c) — Taylor N={N_MAX} (dashed) vs RK4 (solid)',fontsize=13,fontweight='bold')
fig.tight_layout(); fig.savefig('part_c_vs_rk4.png',dpi=150)
print("Saved: part_c_vs_rk4.png")

# (d)

print("\nPART (d) — Sun-Jupiter Asteroid Groups")

M_sun = 1.989e30       # kg
M_jup = 1.898e27       # kg
a_J   = 7.783e11       # m  (Jupiter semi-major axis)
T_J   = 3.743e8        # s  (Jupiter orbital period ~11.86 yr)
omega = 2*np.pi / T_J  # rad/s
v_J   = omega * a_J    # Jupiter orbital speed

def deriv_ast(S, t):
    x,  y  = S[:,0], S[:,1]
    vx, vy = S[:,2], S[:,3]

    # Sun gravity
    rs = (x**2 + y**2)**1.5
    ax = -G*M_sun*x / rs
    ay = -G*M_sun*y / rs

    # Jupiter position (circular orbit)
    jx = a_J * np.cos(omega * t)
    jy = a_J * np.sin(omega * t)
    rj = ((x - jx)**2 + (y - jy)**2)**1.5
    ax -= G*M_jup*(x - jx) / rj
    ay -= G*M_jup*(y - jy) / rj

    return np.column_stack([vx, vy, ax, ay])

# ── Helper: circular velocity at position p ───────────────────
def circ_vel(p):
    r = np.linalg.norm(p)
    return np.sqrt(G*M_sun/r) * np.array([-p[1]/r, p[0]/r])

# ── Helper: make ICs near a centre point ──────────────────────
def make_ic(centre, n, pos_spread, vel_spread):
    np.random.seed(42)
    return np.hstack([
        centre + np.random.uniform(-pos_spread, pos_spread, (n, 2)),
        circ_vel(centre) + np.random.uniform(-vel_spread, vel_spread, (n, 2))
    ])

# ── Lagrange points (t=0, Jupiter on +x axis) ─────────────────
L4 = np.array([ a_J*np.cos( np.pi/3),  a_J*np.sin( np.pi/3)])
L5 = np.array([ a_J*np.cos(-np.pi/3),  a_J*np.sin(-np.pi/3)])

# ── Trojan & Greek ICs ────────────────────────────────────────
N       = 50
ps_tj   = 0.02 * a_J
vs_tj   = 0.005 * v_J
S_troy  = make_ic(L4, N, ps_tj, vs_tj)
S_greek = make_ic(L5, N, ps_tj, vs_tj)

# ── Hilda ICs — eccentric orbits at 3:2 resonance ─────────────
a_hilda = a_J * (2/3)**(2/3)           # semi-major axis
e_hilda = 0.15                          # realistic eccentricity
r_peri  = a_hilda * (1 - e_hilda)      # perihelion distance
v_peri  = np.sqrt(G*M_sun * (2/r_peri - 1/a_hilda))  # vis-viva

def hilda_ic(theta0, n_each, seed=42):
    """Place n_each Hildas near perihelion at resonant angle theta0."""
    np.random.seed(seed)
    spread_r = 0.01 * a_hilda
    spread_v = 0.002 * v_peri
    ics = []
    for _ in range(n_each):
        dr  = np.random.uniform(-spread_r, spread_r, 2)
        dv  = np.random.uniform(-spread_v, spread_v, 2)
        pos = np.array([ r_peri*np.cos(theta0),
                         r_peri*np.sin(theta0)]) + dr
        vel = np.array([-v_peri*np.sin(theta0),
                         v_peri*np.cos(theta0)]) + dv
        ics.append([pos[0], pos[1], vel[0], vel[1]])
    return np.array(ics)

N_h     = 20   # asteroids per vertex
S_hilda = np.vstack([
    hilda_ic(0.0,         N_h, seed=42),   # vertex 1 — 0°
    hilda_ic(2*np.pi/3,   N_h, seed=43),   # vertex 2 — 120°
    hilda_ic(4*np.pi/3,   N_h, seed=44),   # vertex 3 — 240°
])

S_all = np.vstack([S_troy, S_greek, S_hilda])
sl_t  = slice(0,   N)
sl_g  = slice(N,   2*N)
sl_h  = slice(2*N, len(S_all))

dt_d = T_J / 3000          # timestep
n_d  = 30000               # 10 Jupiter years
times_d = np.arange(n_d) * dt_d

print(f"  Simulating {len(S_all)} asteroids × {n_d} steps...")
traj_d = np.empty((n_d, len(S_all), 2))
S = S_all.copy()
traj_d[0] = S[:, :2]
for i in range(1, n_d):
    S = rk4(S, dt_d, deriv_ast, times_d[i-1])
    traj_d[i] = S[:, :2]
    if i % (n_d // 20) == 0:
        print(f"\r  {i/n_d*100:.0f}%", end='', flush=True)
print("Done.")

def to_rotating(xy_inertial, t_arr):
    theta = omega * t_arr
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)
    x = xy_inertial[:, :, 0]
    y = xy_inertial[:, :, 1]
    xr =  cos_t[:,None]*x - sin_t[:,None]*y
    yr =  sin_t[:,None]*x + cos_t[:,None]*y
    return np.stack([xr, yr], axis=2)

traj_rot = to_rotating(traj_d, times_d)

# Lagrange points in rotating frame (fixed)
L4_rot = np.array([ a_J*np.cos( np.pi/3),  a_J*np.sin( np.pi/3)])
L5_rot = np.array([ a_J*np.cos(-np.pi/3),  a_J*np.sin(-np.pi/3)])
L3_rot = np.array([-a_J, 0.])

th     = np.linspace(0, 2*np.pi, 500)
groups = [('Trojans', sl_t, 'limegreen'),
          ('Greeks',  sl_g, 'dodgerblue'),
          ('Hildas',  sl_h, 'red')]

fig, ax = plt.subplots(figsize=(9, 9))

# Reference circles
ax.plot(a_J*np.cos(th),      a_J*np.sin(th),      '--', color='gray',   lw=0.8, alpha=0.5, label='Jupiter orbit')
ax.plot(a_hilda*np.cos(th),  a_hilda*np.sin(th),  ':',  color='salmon',  lw=0.8, alpha=0.6, label='Hilda orbit')

ax.plot(0,    0,   'o', color='gold',   ms=18, markeredgecolor='orange', label='Sun',       zorder=6)
ax.plot(a_J,  0,   's', color='orange', ms=12, markeredgecolor='k',      label='Jupiter',   zorder=6)

for ln, lp, lc in [('L4', L4_rot, 'green'),
                   ('L5', L5_rot, 'blue'),
                   ('L3', L3_rot, 'red')]:
    ax.plot(*lp, '*', color=lc, ms=14, zorder=5)
    ax.annotate(ln, lp, xytext=(6,6), textcoords='offset points',
                fontsize=11, color=lc, fontweight='bold')

for gn, sl, col in groups:
    xy = traj_rot[:, sl, :]
    for k in range(xy.shape[1]):
        ax.plot(xy[-3000:, k, 0], xy[-3000:, k, 1],
                '-', color=col, lw=0.3, alpha=0.12)
    ax.scatter(xy[-1, :, 0], xy[-1, :, 1],
               c=col, s=12, zorder=4, label=gn)

ax.set_xlim(-1.4*a_J, 1.4*a_J)
ax.set_ylim(-1.4*a_J, 1.4*a_J)
ax.set_aspect('equal')
ax.set_xlabel("x' (rotating frame, m)", fontsize=12)
ax.set_ylabel("y' (rotating frame, m)", fontsize=12)
ax.set_title("Part (d) — Sun-Jupiter Asteroid Groups\n(Jupiter co-rotating frame)", fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig('part_d_asteroids.png', dpi=150)
print("Saved: part_d_asteroids.png")


plt.show()
print("\nDone.")
