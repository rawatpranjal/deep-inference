"""Tests for the RieszNet automatic-debiasing procedure (deep_inference.riesz).

The fast invariants here fail if the port is broken. A heavier known-truth coverage
study (the 93-97% claim) lives in exploration/riesz_coverage.py and is run out of band,
since each replication trains several nets and is too slow for a per-CI gate.
"""

import numpy as np
import torch

from deep_inference import InferenceResult, riesz_inference
from deep_inference.riesz import SUPPORTED_OUTCOMES


def _linear_dgp(seed, n=1500):
    """Confounded binary treatment, continuous outcome. Truth: ATE = E[beta(X)] = 0.5."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    e = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0])))  # propensity depends on X0
    T = (rng.random(n) < e).astype(float)
    beta = 0.5 + 0.3 * X[:, 1]
    alpha = 1.0 + 0.5 * X[:, 0]  # X0 confounds e and the baseline
    Y = alpha + beta * T + rng.standard_normal(n)
    truth = float(beta.mean())
    return Y, T, X, truth


def _logit_dgp(seed, n=2000):
    """Confounded binary treatment, binary outcome. Truth: probability-scale ATE."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    e = 1.0 / (1.0 + np.exp(-(0.7 * X[:, 0])))
    T = (rng.random(n) < e).astype(float)
    beta = 0.8 + 0.4 * X[:, 1]
    alpha = 0.3 * X[:, 0]
    p1 = 1.0 / (1.0 + np.exp(-(alpha + beta * 1.0)))
    p0 = 1.0 / (1.0 + np.exp(-(alpha + beta * 0.0)))
    p_obs = 1.0 / (1.0 + np.exp(-(alpha + beta * T)))
    Y = (rng.random(n) < p_obs).astype(float)
    truth = float((p1 - p0).mean())  # E[g(1,X) - g(0,X)]
    return Y, T, X, truth


def test_returns_valid_inference_result():
    """The result has the right type, shapes, and finite, ordered CI bounds."""
    Y, T, X, _ = _linear_dgp(seed=0, n=800)
    r = riesz_inference(
        Y,
        T,
        X,
        outcome="linear",
        n_folds=3,
        n_repeats=1,
        max_epochs=80,
        patience=15,
        seed=0,
    )
    assert isinstance(r, InferenceResult)
    assert np.isfinite(r.mu_hat) and np.isfinite(r.se) and r.se > 0
    assert r.ci_lower < r.mu_hat < r.ci_upper
    assert r.psi_values.shape == (800,)
    assert r.theta_hat.shape == (800, 1)
    assert r.diagnostics["procedure"] == "riesznet"


def test_unsupported_outcome_raises():
    Y, T, X, _ = _linear_dgp(seed=0, n=300)
    try:
        riesz_inference(
            Y,
            T,
            X,
            outcome="weibull",
            n_folds=2,
            n_repeats=1,
            max_epochs=20,
            patience=5,
            seed=0,
        )
    except ValueError as exc:
        assert "weibull" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unsupported outcome")
    assert "weibull" not in SUPPORTED_OUTCOMES


def test_recovers_linear_truth_within_ci():
    """On a known-truth linear DGP the 95% CI contains the truth and the point estimate
    is within three standard errors of it."""
    Y, T, X, truth = _linear_dgp(seed=7)
    r = riesz_inference(
        Y,
        T,
        X,
        outcome="linear",
        n_folds=5,
        n_repeats=2,
        max_epochs=150,
        patience=20,
        seed=7,
    )
    assert abs(r.mu_hat - truth) < 3 * r.se, (r.mu_hat, truth, r.se)
    assert r.ci_lower <= truth <= r.ci_upper, (r.ci_lower, truth, r.ci_upper)


def test_recovers_logit_truth_within_ci():
    """Same check on the probability-scale ATE of a logit DGP."""
    Y, T, X, truth = _logit_dgp(seed=11)
    r = riesz_inference(
        Y,
        T,
        X,
        outcome="logit",
        n_folds=5,
        n_repeats=2,
        max_epochs=150,
        patience=20,
        seed=11,
    )
    assert abs(r.mu_hat - truth) < 3 * r.se, (r.mu_hat, truth, r.se)
    assert r.ci_lower <= truth <= r.ci_upper, (r.ci_lower, truth, r.ci_upper)


if __name__ == "__main__":
    torch.manual_seed(0)
    for fn in [
        test_returns_valid_inference_result,
        test_unsupported_outcome_raises,
        test_recovers_linear_truth_within_ci,
        test_recovers_logit_truth_within_ci,
    ]:
        fn()
        print(f"ok: {fn.__name__}")
