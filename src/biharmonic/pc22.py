"""
PC22 Scheme for the Generalized Kirchhoff-Love Plate Equation
=============================================================
Implements Algorithm 3.1 from:
  "Stable and Accurate Numerical Methods for Generalized Kirchhoff-Love Plates"
  Nguyen, Li, Ji (2020)

Solves:
  rho_h * w_tt = -K0*w + T*lap(w) - D*bilap(w) - K1*w_t + T1*lap(w_t) + F

on [0,1]x[0,1] with simply supported boundary conditions,
verified against the analytical standing wave solution (Section 5.2.1).

Key design: ghost-point approach for the bilaplacian, with boundary conditions
applied explicitly before each operator application.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

save_file_name = "pc22_kirchhoff_love.png"


# 1. LAPLACIAN WITH GHOST POINTS
def laplacian(w, dx, dy, bc="supported"):
    """
    Compute centered finite difference Laplacian on interior + boundary.
    For simply supported BC:
      - w = 0 on boundary  (enforced before calling)
      - d²w/dn² = 0 on boundary (since w=0 and nu*d²w/dt²=0 with w=0 on edge)
        => ghost point: w_ghost = -w_interior (reflection that gives d²w/dn²=0)

    Returns lap(w) on the full grid (boundary values set to 0).
    """
    Nx, Ny = w.shape
    lap = np.zeros_like(w)

    # Pad with ghost points using simply-supported reflection
    # For simply supported: w=0 on boundary, d²w/dn²=0
    # Ghost point outside boundary i=0: w_{-1} = -w_{1} (so d²w/dn² at i=0 -> 0)
    # This gives: lap[i,j] at i=0: (w[-1] - 2*w[0] + w[1])/dx² = (-w[1] - 0 + w[1])/dx² = 0
    # which is correct since w=0 on boundary.
    # For interior we just use standard stencil.

    # x-direction second derivative
    d2x = np.zeros_like(w)
    d2x[1:-1, :] = (w[2:, :] - 2 * w[1:-1, :] + w[:-2, :]) / dx**2
    # boundary rows: use ghost w_{-1} = -w_{1} and w_{N} = -w_{N-2}
    d2x[0, :] = (-w[1, :] - 2 * w[0, :] + w[1, :]) / dx**2  # = -2*w[0]/dx² = 0
    d2x[-1, :] = (-w[-2, :] - 2 * w[-1, :] + w[-2, :]) / dx**2  # = -2*w[-1]/dx² = 0

    # y-direction second derivative
    d2y = np.zeros_like(w)
    d2y[:, 1:-1] = (w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]) / dy**2
    d2y[:, 0] = (-w[:, 1] - 2 * w[:, 0] + w[:, 1]) / dy**2
    d2y[:, -1] = (-w[:, -2] - 2 * w[:, -1] + w[:, -2]) / dy**2

    lap = d2x + d2y
    return lap


def bilaplacian(w, dx, dy):
    """∇⁴w = ∇²(∇²w) by composing two Laplacians."""
    lap_w = laplacian(w, dx, dy)
    return laplacian(lap_w, dx, dy)


# 2. SPATIAL OPERATORS Kh AND Bh  (eq 3.8)
def apply_Kh(w, dx, dy, K0, T, D):
    """Kh*w = K0*w - T*∇²w + D*∇⁴w"""
    result = K0 * w
    if T != 0:
        result -= T * laplacian(w, dx, dy)
    if D != 0:
        result += D * bilaplacian(w, dx, dy)
    return result


def apply_Bh(v, dx, dy, K1, T1):
    """Bh*v = K1*v - T1*∇²v"""
    result = K1 * v
    if T1 != 0:
        result -= T1 * laplacian(v, dx, dy)
    return result


def compute_acceleration(w, v, F, dx, dy, rho_h, K0, T, D, K1, T1):
    """rho_h * a = -Kh(w) - Bh(v) + F  (eq 3.7)"""
    return (-apply_Kh(w, dx, dy, K0, T, D) - apply_Bh(v, dx, dy, K1, T1) + F) / rho_h


# 3. BOUNDARY CONDITIONS
def apply_bc(w):
    """Simply supported: w = 0 on all edges."""
    w[0, :] = 0.0
    w[-1, :] = 0.0
    w[:, 0] = 0.0
    w[:, -1] = 0.0
    return w


# 4. STABLE TIME STEP  (Section 4.2, eq 4.4, 4.5, 4.9)
def compute_stable_dt(dx, dy, rho_h, K0, T, D, K1, T1, Csf=0.9):
    """
    Stable dt for PC22 via half super-ellipse approximation.
    Parameters a=1.75, b=1.2, n=1.5 from paper.
    """
    a_se, b_se, n_se = 1.75, 1.2, 1.5

    KhatM = (1.0 / rho_h) * (K0 + 4.0 * T * (1 / dx**2 + 1 / dy**2) + 16.0 * D * (1 / dx**2 + 1 / dy**2) ** 2)
    BhatM = (1.0 / rho_h) * (K1 + 4.0 * T1 * (1 / dx**2 + 1 / dy**2))

    disc = (BhatM / 2) ** 2 - KhatM
    if disc < 0:
        lam_r = -BhatM / 2
        lam_i = np.sqrt(KhatM - (BhatM / 2) ** 2)
    else:
        lam_r = -BhatM
        lam_i = 0.0

    denom = (abs(lam_r) / a_se) ** n_se + (abs(lam_i) / b_se) ** n_se
    if denom < 1e-30:
        return 1e-3
    return Csf * denom ** (-1.0 / n_se)


# 5. PC22 ALGORITHM 3.1
def pc22_step(w, v, a, F_next, dx, dy, dt, rho_h, K0, T, D, K1, T1, w_prev=None, v_prev=None, a_prev=None):
    """
    One PC22 predictor-corrector step.

    First call: pass w_prev=v_prev=a_prev=None to use 1st-order bootstrap.
    Subsequent calls: pass previous-step values for full AB2 predictor.
    """

    # ---- STAGE I: AB2 Predictor ----
    if w_prev is None:
        # First step bootstrap: forward Euler
        w_p = w + dt * v
        v_p = v + dt * a
    else:
        w_p = w + dt * (1.5 * v - 0.5 * v_prev)
        v_p = v + dt * (1.5 * a - 0.5 * a_prev)

    w_p = apply_bc(w_p)
    v_p = apply_bc(v_p)
    a_p = compute_acceleration(w_p, v_p, F_next, dx, dy, rho_h, K0, T, D, K1, T1)
    a_p = apply_bc(a_p)

    # ---- STAGE II: AM2 Corrector ----
    w_new = w + 0.5 * dt * (v + v_p)
    v_new = v + 0.5 * dt * (a + a_p)

    w_new = apply_bc(w_new)
    v_new = apply_bc(v_new)
    a_new = compute_acceleration(w_new, v_new, F_next, dx, dy, rho_h, K0, T, D, K1, T1)
    a_new = apply_bc(a_new)

    return w_new, v_new, a_new


# 6. SIMULATION RUNNER
def run_simulation(N=40, T_end=1.0, m=1, n_mode=2):
    """
    Standing wave test problem (Section 5.2.1).
    Exact solution: w_e = sin(m*pi*x)*sin(n*pi*y)*cos(omega_mn*t)
    """
    rho_h = 2.7
    K0 = 0.0
    T_c = 0.0
    D = 6.4527
    K1 = 0.0
    T1 = 0.0
    nu = 0.33
    L, H = 1.0, 1.0

    x = np.linspace(0, L, N)
    y = np.linspace(0, H, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    omega_mn = np.pi**2 * (m**2 / L**2 + n_mode**2 / H**2) * np.sqrt(D / rho_h)
    f_mn = omega_mn / (2 * np.pi)
    print(f"  Mode ({m},{n_mode}): f_mn = {f_mn:.4f} Hz, omega = {omega_mn:.4f}")

    def w_ex(t):
        return np.sin(m * np.pi * X / L) * np.sin(n_mode * np.pi * Y / H) * np.cos(omega_mn * t)

    def v_ex(t):
        return -omega_mn * np.sin(m * np.pi * X / L) * np.sin(n_mode * np.pi * Y / H) * np.sin(omega_mn * t)

    # Initial conditions
    w = apply_bc(w_ex(0.0).copy())
    v = apply_bc(v_ex(0.0).copy())
    F = np.zeros_like(w)
    a = compute_acceleration(w, v, F, dx, dy, rho_h, K0, T_c, D, K1, T1)

    # Time step
    dt = compute_stable_dt(dx, dy, rho_h, K0, T_c, D, K1, T1, Csf=0.9)
    n_steps = max(1, int(np.ceil(T_end / dt)))
    dt = T_end / n_steps
    print(f"  N={N}, dx={dx:.4f}, dt={dt:.6f}, steps={n_steps}")

    # Probe tracking
    xp_i = max(1, min(N - 2, int(round(0.2 / dx))))
    yp_j = max(1, min(N - 2, int(round(0.1 / dy))))
    probe_num = [w[xp_i, yp_j]]
    probe_ex = [w_ex(0.0)[xp_i, yp_j]]
    times = [0.0]

    w_prev = v_prev = a_prev = None

    for step in range(n_steps):
        t_next = (step + 1) * dt
        w_new, v_new, a_new = pc22_step(w, v, a, F, dx, dy, dt, rho_h, K0, T_c, D, K1, T1, w_prev, v_prev, a_prev)
        w_prev, v_prev, a_prev = w, v, a
        w, v, a = w_new, v_new, a_new

        probe_num.append(w[xp_i, yp_j])
        probe_ex.append(w_ex(t_next)[xp_i, yp_j])
        times.append(t_next)

    error = np.abs(w - w_ex(T_end))
    l_inf = np.max(error)
    print(f"  L-inf error at t={T_end}: {l_inf:.4e}")

    return dict(
        w=w,
        w_exact=w_ex(T_end),
        error=error,
        X=X,
        Y=Y,
        times=np.array(times),
        probe_num=np.array(probe_num),
        probe_ex=np.array(probe_ex),
        l_inf=l_inf,
        f_mn=f_mn,
        m=m,
        n=n_mode,
        N=N,
        dt=dt,
    )


# 7. CONVERGENCE STUDY
def convergence_study(N_list=(10, 20, 40, 80), m=1, n_mode=2, T_end=1.0):
    errors, h_list = [], []
    for N in N_list:
        r = run_simulation(N=N, T_end=T_end, m=m, n_mode=n_mode)
        errors.append(r["l_inf"])
        h_list.append(1.0 / N)
    h = np.array(h_list)
    e = np.array(errors)
    rates = np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])
    print(f"\nConvergence rates: {np.round(rates,3)}  (mean={np.mean(rates):.3f})")
    return h, e


# 8. PLOT
def make_plot(result, conv=None):
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    def ax_style(ax, title):
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#e6edf3", fontsize=9, fontweight="bold")
        ax.tick_params(colors="#8b949e", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    X, Y, m, n = result["X"], result["Y"], result["m"], result["n"]

    ax1 = fig.add_subplot(gs[0, 0])
    ax_style(ax1, f"Numerical  mode ({m},{n})")
    cf = ax1.contourf(X, Y, result["w"], 30, cmap="RdYlBu_r")
    ax1.contour(X, Y, result["w"], [0], colors="white", linewidths=0.7)
    plt.colorbar(cf, ax=ax1).ax.tick_params(colors="#8b949e", labelsize=6)
    ax1.set_xlabel("x", color="#c9d1d9", fontsize=8)
    ax1.set_ylabel("y", color="#c9d1d9", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax_style(ax2, f'Exact  f={result["f_mn"]:.3f} Hz')
    cf2 = ax2.contourf(X, Y, result["w_exact"], 30, cmap="RdYlBu_r")
    ax2.contour(X, Y, result["w_exact"], [0], colors="white", linewidths=0.7)
    plt.colorbar(cf2, ax=ax2).ax.tick_params(colors="#8b949e", labelsize=6)
    ax2.set_xlabel("x", color="#c9d1d9", fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax_style(ax3, f'Error  ||E||_inf={result["l_inf"]:.2e}')
    cf3 = ax3.contourf(X, Y, result["error"], 30, cmap="hot")
    plt.colorbar(cf3, ax=ax3).ax.tick_params(colors="#8b949e", labelsize=6)
    ax3.set_xlabel("x", color="#c9d1d9", fontsize=8)

    ax4 = fig.add_subplot(gs[1, 0:2])
    ax_style(ax4, "Displacement at probe xp=(0.2,0.1)")
    ax4.plot(result["times"], result["probe_ex"], color="#58a6ff", lw=1.5, label="Exact")
    ax4.plot(result["times"], result["probe_num"], color="#ff7b72", lw=1.0, ls="--", label="PC22")
    ax4.set_xlabel("Time", color="#c9d1d9", fontsize=8)
    ax4.set_ylabel("w", color="#c9d1d9", fontsize=8)
    ax4.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    ax4.grid(True, color="#21262d", lw=0.5)

    ax5 = fig.add_subplot(gs[1, 2])
    ax_style(ax5, "2nd-Order Convergence")
    if conv is not None:
        h, e = conv
        ax5.loglog(h, e, "o-", color="#3fb950", lw=1.5, ms=5, label="PC22")
        ref = e[0] * (h / h[0]) ** 2
        ax5.loglog(h, ref, "--", color="#8b949e", lw=1.0, label="slope 2")
        ax5.set_xlabel("h", color="#c9d1d9", fontsize=8)
        ax5.set_ylabel("||E||_inf", color="#c9d1d9", fontsize=8)
        ax5.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
        ax5.grid(True, which="both", color="#21262d", lw=0.5)

    fig.suptitle(
        "PC22 Scheme — Kirchhoff-Love Plate (Simply Supported BC)",
        color="#e6edf3",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(
        os.path.join(os.path.dirname(__file__), save_file_name), dpi=150, bbox_inches="tight", facecolor="#0d1117"
    )
    plt.close()
    print(f"Saved: {save_file_name}")


# MAIN
if __name__ == "__main__":
    print("=" * 55)
    print("PC22 — Kirchhoff-Love Plate")
    print("=" * 55)

    print("\n[1] Single run N=60, mode (1,2)")
    result = run_simulation(N=60, T_end=1.0, m=1, n_mode=2)

    print("\n[2] Convergence study N=10,20,40,80")
    h, e = convergence_study(N_list=[10, 20, 40, 80], m=1, n_mode=2, T_end=1.0)

    make_plot(result, conv=(h, e))
    print("\nAll done.")
