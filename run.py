"""Finite differences and operator splitting: stability, convergence, Fisher-KPP, Gray-Scott."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from fdsplit import (
    FisherSolver,
    advection_diffusion_split,
    amplification,
    fisher_exact,
    gray_scott,
    laplacian_1d,
    theta_step_factory,
)


def stability_figures():
    kh = np.linspace(0, np.pi, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for r, c in ((0.25, "C0"), (0.5, "C1"), (0.6, "C3")):
        a1.plot(kh, amplification(0.0, r, kh), color=c, label=f"FTCS, r = {r}")
    a1.plot(kh, amplification(1.0, 2.0, kh), "k--", label="BTCS, r = 2")
    a1.plot(kh, amplification(0.5, 2.0, kh), "m-.", label="Crank–Nicolson, r = 2")
    a1.plot(kh, np.exp(-2.0 * kh**2), "g:", lw=2, label="exact exp(−r (kh)²), r = 2")
    a1.axhline(-1, color="grey", lw=0.8)
    a1.axhline(1, color="grey", lw=0.8)
    a1.set(xlabel="k h", ylabel="G(k h)", title="Von Neumann amplification factor, u_t = u_xx  (r = Δt/h²)",
           ylim=(-1.5, 1.1))
    a1.grid(alpha=0.3)
    a1.legend(fontsize=7)

    # FTCS just below / above the stability limit r = 1/2
    n = 101
    x = np.linspace(0, 1, n)[1:-1]
    h = 1 / (n - 1)
    A = laplacian_1d(n - 2, h, periodic=False)
    u0 = np.where(np.abs(x - 0.5) < 0.2, 1.0, 0.0)
    for r, c in ((0.48, "C0"), (0.52, "C3")):
        step = theta_step_factory(A, r * h * h, 0.0)
        u = u0.copy()
        for _ in range(400):
            u = step(u)
        a2.plot(x, u, color=c, lw=1.5 if r < 0.5 else 0.5, alpha=1 if r < 0.5 else 0.6,
                label=f"FTCS r = {r}: max |u| = {np.abs(u).max():.2g}")
    step = theta_step_factory(A, 4.8 * h * h, 0.5)
    u = u0.copy()
    for _ in range(40):
        u = step(u)
    a2.plot(x, u, "m-.", label="Crank–Nicolson r = 4.8, 40 steps (same t as FTCS r = 0.48)")
    a2.set(xlabel="x", ylabel="u", title="Heat equation after 400 FTCS steps", ylim=(-0.6, 0.8))
    a2.grid(alpha=0.3)
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("docs/stability.png", dpi=120)
    plt.close(fig)


def convergence_study():
    print("Spatial convergence (Strang, dt = 1e-3, t = 2): max error")
    space = {}
    ns = [101, 201, 401, 801]
    for order in (2, 4):
        e = []
        for n in ns:
            s = FisherSolver(n=n, order=order)
            u, t = s.run(2.0, 1e-3)
            e.append(np.abs(u - fisher_exact(s.xi, t)).max())
        space[order] = e
        print(f"  order {order}: " + "  ".join(f"{v:.2e}" for v in e) +
              f"   observed {np.log2(e[-2] / e[-1]):.2f}")

    print("\nTemporal convergence (4th-order FD, n = 1601, t = 4): max error")
    dts = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    time = {}
    s = FisherSolver(n=1601, order=4)
    print("| dt | Lie | Strang | unsplit IMEX |\n|---|---|---|---|")
    for m in ("lie", "strang", "imex"):
        time[m] = [np.abs(s.run(4.0, dt, m)[0] - fisher_exact(s.xi, 4.0)).max() for dt in dts]
    for k, dt in enumerate(dts):
        print(f"| {dt} | {time['lie'][k]:.2e} | {time['strang'][k]:.2e} | {time['imex'][k]:.2e} |")
    print("| **order** | " + " | ".join(f"**{np.log2(time[m][-2] / time[m][-1]):.2f}**" for m in ("lie", "strang", "imex")) + " |")

    x = np.linspace(0, 1, 128, endpoint=False)
    for m in ("lie", "strang"):
        a, b = advection_diffusion_split(np.exp(-100 * (x - 0.5) ** 2), 1 / 128, 0.01, 100, method=m)
        print(f"commuting advection-diffusion, {m}: splitting error {np.abs(a - b).max():.1e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    h = 100 / (np.array(ns) - 1)
    for order, mk in ((2, "o-"), (4, "s-")):
        a1.loglog(h, space[order], mk, label=f"{order}th-order central" if order == 4 else "2nd-order central")
        a1.loglog(h, space[order][0] * (h / h[0]) ** order, "k:", lw=0.8)
    a1.set(xlabel="h", ylabel="max error", title="Spatial convergence (slopes 2 and 4)")
    a1.grid(alpha=0.3, which="both")
    a1.legend()
    names = {"lie": "Lie  R(Δt)·D(Δt)", "strang": "Strang  R(Δt/2)·D(Δt)·R(Δt/2)", "imex": "unsplit IMEX (CN + explicit)"}
    for m, mk in (("lie", "o-"), ("strang", "s-"), ("imex", "^--")):
        a2.loglog(dts, time[m], mk, label=names[m])
    a2.loglog(dts, time["lie"][-1] * (np.array(dts) / dts[-1]), "k:", lw=0.8)
    a2.loglog(dts, time["strang"][-1] * (np.array(dts) / dts[-1]) ** 2, "k--", lw=0.8)
    a2.set(xlabel="Δt", ylabel="max error", title="Temporal convergence (slopes 1 and 2)")
    a2.grid(alpha=0.3, which="both")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("docs/convergence.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    s = FisherSolver(n=801, order=4)
    for k, t in enumerate((0.0, 4.0, 8.0, 12.0)):
        u = fisher_exact(s.xi, 0.0) if t == 0 else s.run(t, 0.2, "strang")[0]
        ax.plot(s.xi, fisher_exact(s.xi, t), "k-", lw=2.5, alpha=0.25, label="exact" if k == 0 else None)
        ax.plot(s.xi[::8], u[::8], "o", ms=3, color=f"C{k}", label=f"Strang, Δt = 0.2, t = {t:.0f}")
    lu, _ = s.run(12.0, 0.2, "lie")
    ax.plot(s.xi, lu, "--", color="C3", lw=1, label="Lie, Δt = 0.2, t = 12 (lags behind)")
    ax.set(xlabel="x", ylabel="u", xlim=(-20, 45), title="Fisher–KPP travelling wave, u_t = u_xx + u(1 − u)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig("docs/fisher.png", dpi=120)
    plt.close(fig)


def gray_scott_figures():
    frames = gray_scott(n=200, steps=10_000, every=100)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
    for ax, k in zip(axes, (5, 20, 50, len(frames) - 1)):
        ax.imshow(frames[k], cmap="inferno", vmin=0, vmax=0.45)
        ax.set(xticks=[], yticks=[], title=f"t = {k * 100}")
    fig.suptitle("Gray–Scott reaction–diffusion (F = 0.035, k = 0.065), Strang splitting")
    fig.tight_layout()
    fig.savefig("docs/gray_scott.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.subplots_adjust(0, 0, 1, 1)
    im = ax.imshow(frames[0], cmap="inferno", vmin=0, vmax=0.45)
    ax.axis("off")

    def update(k):
        im.set_data(frames[k])
        return [im]

    FuncAnimation(fig, update, frames=len(frames)).save("docs/gray_scott.gif", writer=PillowWriter(fps=15), dpi=60)
    plt.close(fig)


def main():
    import sys

    stability_figures()
    if "--stability-only" in sys.argv:
        return
    convergence_study()
    gray_scott_figures()


if __name__ == "__main__":
    main()
