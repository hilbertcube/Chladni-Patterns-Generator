"""
NB2 Scheme for the Generalized Kirchhoff-Love Plate Equation
=============================================================
Implements Algorithm 3.2 from:
  "Stable and Accurate Numerical Methods for Generalized Kirchhoff-Love Plates"
  Nguyen, Li, Ji (2020)

Solves:
  rho_h * w_tt = -K0*w + T*lap(w) - D*bilap(w) - K1*w_t + T1*lap(w_t) + F

on [0,1]x[0,1] with simply supported boundary conditions.

Key difference from PC22: NB2 is fully implicit. At each step it solves
the linear system:
  (rho_h*I + beta*dt²*Kh + gamma*dt*Bh) * a^{n+1} = RHS
using scipy's sparse solver. This makes it unconditionally stable so
the time step can be chosen purely for accuracy, not stability.

Parameters: beta=1/4, gamma=1/2 (unconditionally stable, 2nd-order accurate).
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve

save_file_name = "nb2_kirchhoff_love.png"

# 1. LAPLACIAN WITH GHOST POINTS  (shared with PC22)
def laplacian(w, dx, dy):
    """
    Second-order centered finite difference Laplacian.
    Simply supported ghost points: w_{-1} = -w_1 (from d²w/dn² = 0, w_0 = 0).
    """
    d2x = np.zeros_like(w)
    d2x[1:-1, :] = (w[2:, :] - 2 * w[1:-1, :] + w[:-2, :]) / dx**2
    d2x[0, :] = (-w[1, :] - 2 * w[0, :] + w[1, :]) / dx**2
    d2x[-1, :] = (-w[-2, :] - 2 * w[-1, :] + w[-2, :]) / dx**2

    d2y = np.zeros_like(w)
    d2y[:, 1:-1] = (w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]) / dy**2
    d2y[:, 0] = (-w[:, 1] - 2 * w[:, 0] + w[:, 1]) / dy**2
    d2y[:, -1] = (-w[:, -2] - 2 * w[:, -1] + w[:, -2]) / dy**2

    return d2x + d2y


def bilaplacian(w, dx, dy):
    """∇⁴w = ∇²(∇²w) by composing two Laplacians."""
    return laplacian(laplacian(w, dx, dy), dx, dy)


# 2. SPATIAL OPERATORS  (shared with PC22)
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


# 3. BOUNDARY CONDITIONS  (shared with PC22)
def apply_bc(w):
    """Simply supported: w = 0 on all edges."""
    w[0, :] = 0.0
    w[-1, :] = 0.0
    w[:, 0] = 0.0
    w[:, -1] = 0.0
    return w


# 4. SPARSE SYSTEM MATRIX FOR NB2
def build_system_matrix(N, dx, dy, dt, rho_h, K0, T, D, K1, T1, beta=0.25, gamma=0.5):
    """
    Build the sparse matrix for Stage II of NB2 (Algorithm 3.2):

      (rho_h*I + beta*dt²*Kh + gamma*dt*Bh) * a^{n+1} = RHS

    We assemble Kh and Bh as sparse matrices acting on the
    flattened grid vector of size N*N.

    The 1D second-difference matrix L1 with simply-supported ghost points:
      Interior rows i=1..N-2: standard [-1, 2, -1]/h²  (with sign flip for -∇²)
      Boundary rows i=0, N-1: zero row (since lap=0 there from ghost formula)

    ∇²_h in 2D = L1x ⊗ I + I ⊗ L1y  (Kronecker sum)
    ∇⁴_h       = (∇²_h)²
    """

    def make_L1(n, h):
        """
        1D centered difference matrix for -d²/dx² with simply-supported BCs.
        Size n x n. Boundary rows are zero (Laplacian is zero there).
        Interior rows: standard tridiagonal [1, -2, 1]/h²
        Note: we build the Laplacian (not negative), so entries are [1,-2,1]/h²
        """
        # main diagonal: -2/h² for interior, 0 for boundary
        main = np.zeros(n)
        main[1:-1] = -2.0 / h**2
        # off-diagonals: 1/h² for interior connections only
        off = np.zeros(n - 1)
        # off[i] connects row i to row i+1
        # for interior rows i=1..n-2, connection to i+1 and i-1
        off[1:-1] = 1.0 / h**2  # connects row i (interior) to row i+1
        # Note: off[0] connects row 0 (boundary) to row 1 — keep 0
        #       off[n-2] connects row n-2 to row n-1 (boundary) — keep as is
        # Actually for interior row i, both neighbours i-1 and i+1 contribute
        # Sub-diagonal: off_sub[i] = entry at (i+1, i), i.e. row i+1 gets w_i
        # We need: row i (interior): [1/h², -2/h², 1/h²] at cols [i-1, i, i+1]
        off_sub = np.zeros(n - 1)
        off_sub[1:] = 1.0 / h**2  # row i+1 (if interior) gets 1/h² from col i
        # off_super: row i gets 1/h² from col i+1
        off_super = np.zeros(n - 1)
        off_super[:-1] = 1.0 / h**2  # row i (if interior) gets 1/h² from col i+1
        # Zero out boundary contributions
        off_sub[0] = 0.0  # row 1's connection to row 0: keep (interior row 1 uses w_0=0 naturally)
        off_super[-1] = 0.0  # row n-2's connection to row n-1: keep

        L = diags([off_sub, main, off_super], [-1, 0, 1], shape=(n, n), format="csr")
        return L

    Ix = eye(N, format="csr")
    Iy = eye(N, format="csr")
    Lx = make_L1(N, dx)
    Ly = make_L1(N, dy)

    # 2D Laplacian: ∇²_h = Lx⊗Iy + Ix⊗Ly
    LAP = kron(Lx, Iy) + kron(Ix, Ly)

    # 2D Bilaplacian: ∇⁴_h = (∇²_h)²
    BILAP = LAP @ LAP

    # Kh = K0*I - T*LAP + D*BILAP
    I_full = eye(N * N, format="csr")
    Kh_mat = K0 * I_full - T * LAP + D * BILAP

    # Bh = K1*I - T1*LAP
    Bh_mat = K1 * I_full - T1 * LAP

    # System matrix: M = rho_h*I + beta*dt²*Kh + gamma*dt*Bh
    M = rho_h * I_full + beta * dt**2 * Kh_mat + gamma * dt * Bh_mat

    # Zero out boundary rows and set diagonal to 1
    # (boundary acceleration will be set to 0 via RHS)
    M = M.tolil()
    boundary_indices = get_boundary_indices(N)
    for idx in boundary_indices:
        M[idx, :] = 0.0
        M[idx, idx] = 1.0
    M = M.tocsr()

    return M, Kh_mat, Bh_mat


def get_boundary_indices(N):
    """Return flat indices of all boundary grid points for an N×N grid."""
    idx = []
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                idx.append(i * N + j)
    return idx


# 5. NB2 ALGORITHM 3.2
def nb2_step(w, v, a, F_next, dx, dy, dt, rho_h, K0, T, D, K1, T1, M, Kh_mat, Bh_mat, beta=0.25, gamma=0.5):
    """
    One step of the NB2 (Newmark-Beta) scheme (Algorithm 3.2).

    Parameters
    ----------
    w, v, a   : displacement, velocity, acceleration at t_n  (2D arrays, N×N)
    F_next    : forcing at t_{n+1}
    M         : prebuilt sparse system matrix (rho_h*I + beta*dt²*Kh + gamma*dt*Bh)
    Kh_mat, Bh_mat : sparse operator matrices (N²×N²)
    beta=0.25, gamma=0.5 : Newmark parameters (unconditionally stable, 2nd order)

    Returns
    -------
    w_new, v_new, a_new : solutions at t_{n+1}
    """
    N = w.shape[0]

    # ---- STAGE I: First-order prediction of w and v ----
    # w^p = w^n + dt*v^n + (dt²/2)*(1-2β)*a^n
    # v^p = v^n + dt*(1-γ)*a^n
    w_p = w + dt * v + (dt**2 / 2) * (1 - 2 * beta) * a
    v_p = v + dt * (1 - gamma) * a

    # Apply BCs to predicted values
    w_p = apply_bc(w_p)
    v_p = apply_bc(v_p)

    # ---- STAGE II: Solve for a^{n+1} ----
    # Substituting w^{n+1} = w^p + β*dt²*a^{n+1}
    #              v^{n+1} = v^p + γ*dt*a^{n+1}
    # into rho_h*a^{n+1} = -Kh*w^{n+1} - Bh*v^{n+1} + F^{n+1}
    # gives the linear system (eq in Algorithm 3.2):
    #   (rho_h*I + β*dt²*Kh + γ*dt*Bh) * a^{n+1} = -Kh*w^p - Bh*v^p + F^{n+1}

    # Flatten to 1D for sparse solve
    wp_flat = w_p.flatten()
    vp_flat = v_p.flatten()
    F_flat = F_next.flatten()

    # Build RHS: -Kh*w^p - Bh*v^p + F^{n+1}
    rhs = -Kh_mat @ wp_flat - Bh_mat @ vp_flat + F_flat

    # Enforce boundary condition on RHS: a=0 on boundary
    boundary_idx = get_boundary_indices(N)
    for idx in boundary_idx:
        rhs[idx] = 0.0

    # Solve the sparse linear system
    a_new_flat = spsolve(M, rhs)
    a_new = a_new_flat.reshape(N, N)
    a_new = apply_bc(a_new)

    # ---- STAGE III: Correct w and v using a^{n+1} ----
    # w^{n+1} = w^p + β*dt²*a^{n+1}
    # v^{n+1} = v^p + γ*dt*a^{n+1}
    w_new = w_p + beta * dt**2 * a_new
    v_new = v_p + gamma * dt * a_new

    w_new = apply_bc(w_new)
    v_new = apply_bc(v_new)

    return w_new, v_new, a_new


# 6. TIME STEP FOR NB2
def compute_nb2_dt(dx, dy, rho_h, K0, T, D, K1, T1, Csf=90.0):
    """
    NB2 is unconditionally stable so any dt works for stability.
    The paper uses the same PC22 formula but with Csf=90 (100x larger)
    to choose an accuracy-driven dt rather than a stability-driven one.
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
        return 1e-1
    return Csf * denom ** (-1.0 / n_se)


# 7. SIMULATION RUNNER
def run_simulation(N=40, T_end=1.0, m=1, n_mode=2, beta=0.25, gamma=0.5):
    """
    Standing wave test problem (Section 5.2.1), solved with NB2.
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

    # Initial acceleration from equation of motion
    a = (-apply_Kh(w, dx, dy, K0, T_c, D) - apply_Bh(v, dx, dy, K1, T1) + F) / rho_h
    a = apply_bc(a)

    # Time step for NB2.
    # NB2 is unconditionally stable so dt is chosen for accuracy only.
    # We need enough steps to resolve the oscillation period T = 2*pi/omega_mn.
    # Rule: at least 100 steps per oscillation period.
    # This is independent of the grid size, unlike PC22 where dt ~ dx^4.
    dt_accuracy = (2 * np.pi / omega_mn) / 100.0
    n_steps = max(1, int(np.ceil(T_end / dt_accuracy)))
    dt = T_end / n_steps
    print(f"  N={N}, dx={dx:.4f}, dt={dt:.6f}, steps={n_steps}")

    # Build sparse system matrix ONCE — this is the key efficiency of NB2.
    # The matrix never changes between steps (constant coefficients, fixed dt),
    # so we factorize it once and reuse for every RHS.
    print(f"  Building {N*N}x{N*N} sparse system matrix...")
    M, Kh_mat, Bh_mat = build_system_matrix(N, dx, dy, dt, rho_h, K0, T_c, D, K1, T1, beta, gamma)

    # Probe tracking
    xp_i = max(1, min(N - 2, int(round(0.2 / dx))))
    yp_j = max(1, min(N - 2, int(round(0.1 / dy))))
    probe_num = [w[xp_i, yp_j]]
    probe_ex = [w_ex(0.0)[xp_i, yp_j]]
    times = [0.0]

    for step in range(n_steps):
        t_next = (step + 1) * dt
        F_next = np.zeros_like(w)  # no forcing for free vibration test

        w, v, a = nb2_step(w, v, a, F_next, dx, dy, dt, rho_h, K0, T_c, D, K1, T1, M, Kh_mat, Bh_mat, beta, gamma)

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


# 8. CONVERGENCE STUDY


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


# 9. PLOT


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
    ax4.plot(result["times"], result["probe_num"], color="#f0883e", lw=1.0, ls="--", label="NB2")
    ax4.set_xlabel("Time", color="#c9d1d9", fontsize=8)
    ax4.set_ylabel("w", color="#c9d1d9", fontsize=8)
    ax4.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    ax4.grid(True, color="#21262d", lw=0.5)

    ax5 = fig.add_subplot(gs[1, 2])
    ax_style(ax5, "2nd-Order Convergence")
    if conv is not None:
        h, e = conv
        ax5.loglog(h, e, "s-", color="#f0883e", lw=1.5, ms=5, label="NB2")
        ref = e[0] * (h / h[0]) ** 2
        ax5.loglog(h, ref, "--", color="#8b949e", lw=1.0, label="slope 2")
        ax5.set_xlabel("h", color="#c9d1d9", fontsize=8)
        ax5.set_ylabel("||E||_inf", color="#c9d1d9", fontsize=8)
        ax5.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
        ax5.grid(True, which="both", color="#21262d", lw=0.5)

    fig.suptitle(
        "NB2 Scheme — Kirchhoff-Love Plate (Simply Supported BC)",
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
    print("NB2 — Kirchhoff-Love Plate")
    print("=" * 55)

    print("\n[1] Single run N=30, mode (1,2)")
    result = run_simulation(N=30, T_end=1.0, m=1, n_mode=2)

    print("\n[2] Convergence study N=10,20,40")
    h, e = convergence_study(N_list=[10, 20, 40], m=1, n_mode=2, T_end=1.0)

    make_plot(result, conv=(h, e))
    print("\nAll done.")
