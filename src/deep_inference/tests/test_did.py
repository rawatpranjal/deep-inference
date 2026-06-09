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


# ── Heterogeneous neural DiD: DiDModel + AnalyticLambda (NN-free checks) ──


def test_did_model_score_hessian_match_autodiff():
    """DiDModel closed-form score/Hessian must match torch autodiff of the loss."""
    import torch
    from deep_inference.models import DiDModel

    model = DiDModel()
    assert model.theta_dim == 4
    assert model.hessian_depends_on_theta is False
    assert model.hessian_depends_on_y is False

    rng = np.random.default_rng(0)
    max_g_err = 0.0
    max_h_err = 0.0
    for _ in range(100):
        y = torch.tensor(float(rng.standard_normal()), dtype=torch.float64)
        G, P = float(rng.integers(0, 2)), float(rng.integers(0, 2))
        t = torch.tensor([G, P, G * P], dtype=torch.float64)
        theta = torch.tensor(rng.standard_normal(4), dtype=torch.float64, requires_grad=True)

        loss = model.loss(y, t, theta)
        (grad_ad,) = torch.autograd.grad(loss, theta, create_graph=True)
        hess_ad = torch.zeros(4, 4, dtype=torch.float64)
        for j in range(4):
            (row,) = torch.autograd.grad(grad_ad[j], theta, retain_graph=True)
            hess_ad[j] = row

        score_cf = model.score(y, t, theta.detach())
        hess_cf = model.hessian(y, t, theta.detach())
        max_g_err = max(max_g_err, float((score_cf - grad_ad.detach()).abs().max()))
        max_h_err = max(max_h_err, float((hess_cf - hess_ad).abs().max()))

    assert max_g_err < 1e-6, f"score autodiff mismatch: {max_g_err}"
    assert max_h_err < 1e-6, f"hessian autodiff mismatch: {max_h_err}"


def test_analytic_lambda_supports_multicolumn_design():
    """AnalyticLambda must build (n,4,4) Lambda = E[WW'] for the DiD design [1,G,P,GP]."""
    import torch
    from deep_inference.lambda_.analytic import AnalyticLambda
    from deep_inference.models import DiDModel

    rng = np.random.default_rng(1)
    n = 3000
    X = torch.tensor(rng.standard_normal((n, 3)), dtype=torch.float32)
    G = (torch.rand(n) < 0.5).float()
    P = (torch.rand(n) < 0.5).float()
    T = torch.stack([G, P, G * P], dim=1)  # (n, 3)

    lam = AnalyticLambda(method="aggregate")
    lam.fit(X, T, torch.zeros(n), None, DiDModel())
    L = lam.predict(X)
    assert L.shape == (n, 4, 4)

    W = torch.cat([torch.ones(n, 1), T], dim=1)  # [1, G, P, GP]
    expected = torch.einsum("bi,bj->bij", W, W).mean(0)
    assert float((L[0] - expected).abs().max()) < 1e-5
    assert float(torch.linalg.eigvalsh(L[0]).min()) > 1e-4  # PSD / stable


def test_analytic_lambda_scalar_path_unchanged():
    """Regression: scalar T still yields the original (n,2,2) Lambda."""
    import torch
    from deep_inference.lambda_.analytic import AnalyticLambda
    from deep_inference.models import Linear

    rng = np.random.default_rng(2)
    n = 2000
    X = torch.tensor(rng.standard_normal((n, 2)), dtype=torch.float32)
    T = torch.tensor(rng.standard_normal(n), dtype=torch.float32)  # (n,)

    lam = AnalyticLambda(method="aggregate")
    lam.fit(X, T, torch.zeros(n), None, Linear())
    L = lam.predict(X)
    assert L.shape == (n, 2, 2)
    T_aug = torch.stack([torch.ones(n), T], dim=1)
    expected = torch.einsum("bi,bj->bij", T_aug, T_aug).mean(0)
    assert float((L[0] - expected).abs().max()) < 1e-5


# ── Two-way fixed-effects panel DiD ──


def test_residualize_fixed_effects_kills_additive_fe():
    """Two-way demean must remove additive unit+time effects (balanced: exact)."""
    from deep_inference.utils import residualize_fixed_effects

    rng = np.random.default_rng(0)
    N, T = 40, 5
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    additive = 3.0 + rng.standard_normal(N)[unit] + rng.standard_normal(T)[time]
    r = residualize_fixed_effects(additive, unit, time)
    assert np.abs(r).max() < 1e-9  # pure additive FE -> demeaned to ~0

    # idempotent: residualizing again changes nothing
    base = additive + rng.standard_normal(N * T)
    r1 = residualize_fixed_effects(base, unit, time)
    r2 = residualize_fixed_effects(r1, unit, time)
    assert np.abs(r1 - r2).max() < 1e-9


def test_fe_panel_model_score_hessian_match_autodiff():
    """FEPanelDiDModel closed-form score/Hessian vs torch autodiff."""
    import torch
    from deep_inference.models import FEPanelDiDModel

    model = FEPanelDiDModel()
    assert model.theta_dim == 1
    assert model.hessian_depends_on_theta is False
    assert model.analytic_intercept is False

    rng = np.random.default_rng(0)
    max_g, max_h = 0.0, 0.0
    for _ in range(100):
        y = torch.tensor(float(rng.standard_normal()), dtype=torch.float64)
        t = torch.tensor(float(rng.standard_normal()), dtype=torch.float64)  # Dtilde scalar
        theta = torch.tensor(rng.standard_normal(1), dtype=torch.float64, requires_grad=True)
        loss = model.loss(y, t, theta)
        (g_ad,) = torch.autograd.grad(loss, theta, create_graph=True)
        (h_ad,) = torch.autograd.grad(g_ad[0], theta)
        max_g = max(max_g, float((model.score(y, t, theta.detach()) - g_ad.detach()).abs().max()))
        max_h = max(max_h, float((model.hessian(y, t, theta.detach())[0, 0] - h_ad[0]).abs().max()))
    assert max_g < 1e-6
    assert max_h < 1e-6


def test_analytic_lambda_intercept_free_scalar():
    """AnalyticLambda(intercept=False) on scalar T -> (n,1,1) = E[Dtilde^2]."""
    import torch
    from deep_inference.lambda_.analytic import AnalyticLambda
    from deep_inference.models import FEPanelDiDModel

    rng = np.random.default_rng(1)
    n = 3000
    X = torch.tensor(rng.standard_normal((n, 2)), dtype=torch.float32)
    T = torch.tensor(rng.standard_normal(n), dtype=torch.float32)  # Dtilde

    lam = AnalyticLambda(method="aggregate", intercept=False)
    lam.fit(X, T, torch.zeros(n), None, FEPanelDiDModel())
    L = lam.predict(X)
    assert L.shape == (n, 1, 1)
    assert abs(float(L[0, 0, 0]) - float((T ** 2).mean())) < 1e-4


def test_did_panel_fe_matches_within_ols_homogeneous():
    """Homogeneous tau: neural FE DiD point estimate ~ within-OLS slope."""
    import deep_inference as di
    from deep_inference.utils import residualize_fixed_effects

    rng = np.random.default_rng(3)
    N, T = 200, 4
    unit = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)
    n = N * T
    G = (rng.random(N) < 0.5).astype(float)[unit]
    Post = (time >= 2).astype(float)
    D = G * Post
    X = rng.standard_normal((n, 3))
    tau = 0.5
    Y = rng.standard_normal(N)[unit] + np.array([0, .3, .6, .9])[time] + tau * D + rng.standard_normal(n)

    res = di.did_panel_fe(Y, D, X, unit, time, n_folds=3, epochs=40, patience=15, hidden_dims=[16])
    Yt = residualize_fixed_effects(Y, unit, time)
    Dt = residualize_fixed_effects(D, unit, time)
    beta_ols = float((Dt @ Yt) / (Dt @ Dt))
    assert res.diagnostics.get("regime") == "B"
    assert res.theta_hat.shape == (n, 1)
    # neural homogeneous estimate close to the within-OLS slope
    assert abs(res.mu_hat - beta_ols) < 0.1
