import numpy as np
import pytest

from fdsplit import FisherSolver, advection_diffusion_split, amplification, fisher_exact, laplacian_1d, logistic_flow


def test_fisher_exact_solution_satisfies_pde():
    x, t, e = np.linspace(-10, 10, 41), 1.3, 1e-4
    ut = (fisher_exact(x, t + e) - fisher_exact(x, t - e)) / (2 * e)
    uxx = (fisher_exact(x + e, t) - 2 * fisher_exact(x, t) + fisher_exact(x - e, t)) / e**2
    u = fisher_exact(x, t)
    assert np.allclose(ut, uxx + u * (1 - u), atol=1e-5)


def test_logistic_flow_is_a_semigroup():
    u = np.linspace(0.01, 0.99, 9)
    assert np.allclose(logistic_flow(logistic_flow(u, 0.3), 0.4), logistic_flow(u, 0.7))


@pytest.mark.parametrize("order", [2, 4])
def test_laplacian_order(order):
    errs = []
    for n in (64, 128):
        h = 2 * np.pi / n
        x = np.arange(n) * h
        errs.append(np.abs(laplacian_1d(n, h, order) @ np.sin(x) + np.sin(x)).max())
    assert np.log2(errs[0] / errs[1]) == pytest.approx(order, abs=0.1)


def test_ftcs_stability_limit():
    kh = np.linspace(0, np.pi, 200)
    assert np.abs(amplification(0.0, 0.5, kh)).max() <= 1 + 1e-12
    assert np.abs(amplification(0.0, 0.51, kh)).max() > 1
    assert np.abs(amplification(0.5, 100.0, kh)).max() <= 1  # CN unconditionally stable


@pytest.mark.parametrize("method,order", [("lie", 1), ("strang", 2)])
def test_splitting_order(method, order):
    s = FisherSolver(n=801, order=4)
    e = [np.abs(s.run(2.0, dt, method)[0] - fisher_exact(s.xi, 2.0)).max() for dt in (0.1, 0.05)]
    assert np.log2(e[0] / e[1]) == pytest.approx(order, abs=0.15)


def test_commuting_operators_have_no_splitting_error():
    x = np.linspace(0, 1, 64, endpoint=False)
    a, b = advection_diffusion_split(np.exp(-100 * (x - 0.5) ** 2), 1 / 64, 0.02, 50, method="lie")
    assert np.abs(a - b).max() < 1e-12
