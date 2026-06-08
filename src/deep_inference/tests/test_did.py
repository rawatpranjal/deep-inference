"""
Machine-precision firewall for the closed-form 2x2 DiD estimator (did_2x2).

The estimator is deterministic, so the right firewall is an exact-equivalence test
against the saturated OLS regression with HC0 robust standard errors, not a Monte
Carlo coverage test. We verify:

    mu_hat == saturated-OLS beta_3            (the G x T interaction coefficient)
    se^2   == HC0 vcov[3, 3]
    mean(psi) == mu_hat,  mean((psi-mu)^2)/n == se^2
"""

import numpy as np
import pytest

from deep_inference import did_2x2


def _saturated_ols_hc0(Y, G, P):
    """Return (beta_interaction, HC0 variance of the interaction) for Y ~ 1 + G + P + G*P."""
    n = len(Y)
    X = np.column_stack([np.ones(n), G, P, G * P])

    beta = np.linalg.solve(X.T @ X, X.T @ Y)
    residual = Y - X @ beta

    XtX_inv = np.linalg.inv(X.T @ X)
    meat = X.T @ (residual[:, None] ** 2 * X)
    vcov = XtX_inv @ meat @ XtX_inv

    return beta[3], vcov[3, 3]


def _make_unbalanced_2x2(seed=123):
    rng = np.random.default_rng(seed)
    n00, n01, n10, n11 = 83, 91, 77, 103

    G = np.r_[np.zeros(n00), np.zeros(n01), np.ones(n10), np.ones(n11)]
    P = np.r_[np.zeros(n00), np.ones(n01), np.zeros(n10), np.ones(n11)]

    Y = np.r_[
        rng.normal(1.0, 1.2, n00),
        rng.normal(1.5, 1.0, n01),
        rng.normal(2.0, 1.4, n10),
        rng.normal(3.1, 1.8, n11),
    ]
    return Y, G, P


def test_did_2x2_matches_saturated_ols_machine_precision():
    Y, G, P = _make_unbalanced_2x2()

    result = did_2x2(Y, G, P, use_bessel=False)
    beta_ols, var_hc0 = _saturated_ols_hc0(Y, G, P)

    assert abs(result.mu_hat - beta_ols) < 1e-12
    assert abs(result.se**2 - var_hc0) < 1e-14

    psi = result.psi_values.numpy()
    assert abs(psi.mean() - result.mu_hat) < 1e-14
    assert abs(((psi - result.mu_hat) ** 2).mean() / len(Y) - result.se**2) < 1e-16


def test_did_2x2_equals_four_cell_contrast():
    Y, G, P = _make_unbalanced_2x2(seed=7)

    def cell_mean(g, t):
        return Y[(G == g) & (P == t)].mean()

    four_cell = cell_mean(1, 1) - cell_mean(1, 0) - cell_mean(0, 1) + cell_mean(0, 0)
    result = did_2x2(Y, G, P)
    assert abs(result.mu_hat - four_cell) < 1e-12


def test_did_2x2_ci_is_normal_approx():
    Y, G, P = _make_unbalanced_2x2(seed=11)
    result = did_2x2(Y, G, P, alpha=0.05)
    z = 1.959963984540054  # norm.ppf(0.975)
    assert abs(result.ci_lower - (result.mu_hat - z * result.se)) < 1e-12
    assert abs(result.ci_upper - (result.mu_hat + z * result.se)) < 1e-12


def test_did_2x2_bessel_differs_from_hc0_by_known_factor():
    Y, G, P = _make_unbalanced_2x2(seed=5)
    n = len(Y)
    hc0 = did_2x2(Y, G, P, use_bessel=False).se ** 2
    bessel = did_2x2(Y, G, P, use_bessel=True).se ** 2
    # var_psi scales by n/(n-1) under Bessel; se^2 = var_psi/n carries the same factor.
    assert abs(bessel - hc0 * n / (n - 1)) < 1e-14


def test_did_2x2_empty_cell_raises():
    # No control-post observations -> empty (G=0, T=1) cell.
    G = np.array([0.0, 0.0, 1.0, 1.0])
    P = np.array([0.0, 0.0, 0.0, 1.0])
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="Empty DiD cell"):
        did_2x2(Y, G, P)


def test_did_2x2_non_binary_raises():
    Y, G, P = _make_unbalanced_2x2(seed=3)
    G_bad = G.copy()
    G_bad[0] = 2.0
    with pytest.raises(ValueError, match="binary"):
        did_2x2(Y, G_bad, P)
