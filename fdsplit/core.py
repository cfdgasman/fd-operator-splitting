"""Finite-difference building blocks and operator splitting.

1D periodic or Dirichlet grids, second- and fourth-order central differences,
theta-method time stepping (FTCS theta=0, Crank-Nicolson theta=1/2, BTCS theta=1),
and Lie / Strang splitting of  u_t = D u_xx + R(u).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------- operators


def laplacian_1d(n, h, order=2, periodic=True):
    """Second-derivative matrix on n points (periodic, or interior points of a Dirichlet grid)."""
    if order == 2:
        coef = {-1: 1.0, 0: -2.0, 1: 1.0}
    elif order == 4:
        coef = {-2: -1 / 12, -1: 4 / 3, 0: -5 / 2, 1: 4 / 3, 2: -1 / 12}
    else:
        raise ValueError("order must be 2 or 4")
    A = sp.lil_matrix((n, n))
    for i in range(n):
        for k, c in coef.items():
            j = i + k
            if periodic:
                A[i, j % n] += c
            elif 0 <= j < n:
                A[i, j] += c
    return A.tocsr() / h**2


def theta_step_factory(A, dt, theta):
    """Return a function u -> u^{n+1} for u_t = A u with the theta method."""
    n = A.shape[0]
    I = sp.identity(n, format="csr")
    lhs = (I - theta * dt * A).tocsc()
    rhs = (I + (1 - theta) * dt * A).tocsr()
    if theta == 0:
        return lambda u: rhs @ u
    lu = splu(lhs)
    return lambda u: lu.solve(rhs @ u)


def amplification(theta, r, k_h):
    """Von Neumann amplification factor of the theta method for u_t = u_xx, r = dt / h^2."""
    lam = -4 * r * np.sin(k_h / 2) ** 2  # dt * eigenvalue of the 2nd-order Laplacian
    return (1 + (1 - theta) * lam) / (1 - theta * lam)


# ---------------------------------------------------------------- Fisher-KPP


def fisher_exact(x, t):
    """Ablowitz-Zeppetella travelling wave for u_t = u_xx + u(1 - u), speed 5/sqrt(6)."""
    return (1 + np.exp((x - 5 * t / np.sqrt(6)) / np.sqrt(6))) ** -2


def logistic_flow(u, dt):
    """Exact solution operator of the reaction step u' = u(1 - u)."""
    e = np.exp(dt)
    return u * e / (1 - u + u * e)


class FisherSolver:
    """u_t = u_xx + u(1-u) on [a, b] with exact Dirichlet data, second- or fourth-order FD in space."""

    def __init__(self, a=-40.0, b=60.0, n=801, order=4):
        self.x = np.linspace(a, b, n)
        self.h = self.x[1] - self.x[0]
        self.xi = self.x[1:-1]
        self.A = laplacian_1d(n - 2, self.h, order, periodic=False)
        self.order = order
        self._key = None

    def boundary_vector(self, t):
        """Contribution of the known boundary values to (A u)_interior at time t."""
        n = len(self.xi)
        g = np.zeros(n)
        left = fisher_exact(self.x[0], t)
        right = fisher_exact(self.x[-1], t)
        if self.order == 2:
            g[0] += left
            g[-1] += right
        else:
            # point x_1 uses x_0 (coef 4/3) and x_{-1}; x_2 uses x_0 (coef -1/12).
            # x_{-1} lies outside the grid: use the exact solution there.
            g[0] += 4 / 3 * left - 1 / 12 * fisher_exact(self.x[0] - self.h, t)
            g[1] += -1 / 12 * left
            g[-1] += 4 / 3 * right - 1 / 12 * fisher_exact(self.x[-1] + self.h, t)
            g[-2] += -1 / 12 * right
        return g / self.h**2

    def _factor(self, dt):
        if self._key != dt:
            I = sp.identity(self.A.shape[0], format="csc")
            self._lu = splu((I - 0.5 * dt * self.A).tocsc())
            self._rhs = (I + 0.5 * dt * self.A).tocsr()
            self._key = dt

    def _cn_rhs(self, u, t, dt):
        return self._rhs @ u + 0.5 * dt * (self.boundary_vector(t) + self.boundary_vector(t + dt))

    def diffusion_step(self, u, t, dt):
        """Crank-Nicolson for u_t = u_xx with time-dependent Dirichlet data."""
        self._factor(dt)
        return self._lu.solve(self._cn_rhs(u, t, dt))

    def run(self, t_end, dt, method="strang"):
        u = fisher_exact(self.xi, 0.0)
        t = 0.0
        steps = int(round(t_end / dt))
        for _ in range(steps):
            if method == "lie":  # R(dt) then D(dt)
                u = logistic_flow(u, dt)
                u = self.diffusion_step(u, t, dt)
            elif method == "strang":  # R(dt/2) D(dt) R(dt/2)
                u = logistic_flow(u, dt / 2)
                u = self.diffusion_step(u, t, dt)
                u = logistic_flow(u, dt / 2)
            elif method == "imex":  # unsplit: CN diffusion + explicit (forward Euler) reaction
                self._factor(dt)
                u = self._lu.solve(self._cn_rhs(u, t, dt) + dt * u * (1 - u))
            else:
                raise ValueError(method)
            t += dt
        return u, t


# ---------------------------------------------------------------- linear advection-diffusion (commuting case)


def advection_diffusion_split(u0, h, dt, steps, a=1.0, nu=0.01, method="strang"):
    """Periodic u_t + a u_x = nu u_xx, each sub-problem solved *exactly* in Fourier space.

    The two operators commute, so the splitting error must vanish (up to round-off)."""
    n = len(u0)
    k = 2 * np.pi * np.fft.fftfreq(n, d=h)
    adv = lambda tau: np.exp(-1j * a * k * tau)  # noqa: E731
    dif = lambda tau: np.exp(-nu * k**2 * tau)  # noqa: E731
    U = np.fft.fft(u0)
    for _ in range(steps):
        if method == "lie":
            U = dif(dt) * (adv(dt) * U)
        else:
            U = adv(dt / 2) * (dif(dt) * (adv(dt / 2) * U))
    exact = np.fft.fft(u0) * adv(dt * steps) * dif(dt * steps)
    return np.fft.ifft(U).real, np.fft.ifft(exact).real


# ---------------------------------------------------------------- Gray-Scott


def gray_scott(n=256, steps=12_000, dt=1.0, Du=0.16, Dv=0.08, F=0.035, k=0.065, every=100, seed=1):
    """2D Gray-Scott on a periodic n x n grid (h = 1) with Strang splitting:
    exact spectral diffusion for half steps, RK4 reaction for full steps."""
    rng = np.random.default_rng(seed)
    u = np.ones((n, n))
    v = np.zeros((n, n))
    c = slice(n // 2 - 10, n // 2 + 10)
    u[c, c], v[c, c] = 0.5, 0.25
    u += 0.02 * rng.random((n, n))
    v += 0.02 * rng.random((n, n))
    kx = 2 * np.pi * np.fft.fftfreq(n)
    # symbol of the 5-point Laplacian (so this is the exact solution of the FD semi-discretisation)
    lap = -(4 * np.sin(kx[:, None] / 2) ** 2 + 4 * np.sin(kx[None, :] / 2) ** 2)
    Eu, Ev = np.exp(Du * lap * dt / 2), np.exp(Dv * lap * dt / 2)

    def react(u, v):
        uvv = u * v * v
        return -uvv + F * (1 - u), uvv - (F + k) * v

    frames = []
    for s in range(steps):
        u = np.fft.ifft2(Eu * np.fft.fft2(u)).real
        v = np.fft.ifft2(Ev * np.fft.fft2(v)).real
        k1 = react(u, v)
        k2 = react(u + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
        k3 = react(u + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
        k4 = react(u + dt * k3[0], v + dt * k3[1])
        u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        u = np.fft.ifft2(Eu * np.fft.fft2(u)).real
        v = np.fft.ifft2(Ev * np.fft.fft2(v)).real
        if s % every == 0:
            frames.append(v.copy())
    return frames
