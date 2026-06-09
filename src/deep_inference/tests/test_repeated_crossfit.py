"""
Tests for repeated cross-fitting with Chernozhukov median DML aggregation
(Item 1) and the undersmoothing knobs three_way_theta_frac / dropout /
weight_decay (Item 2).

Invariants asserted:
  1. n_repeats > 1 runs end-to-end on a tiny logit DGP and returns finite
     mu_hat / se / CI.
  2. n_repeats = 1 (the default) is the byte-faithful single-split path:
       - default (no kwarg) == explicit n_repeats=1, to machine precision;
       - mu_hat == psi.mean() and se == the shared compute_se_ci estimator on
         the SAME psi (i.e. the median wrapper is a no-op for one repeat).
     This is the no-regression firewall: the default path is unchanged.
  3. The median DML rule ONLY WIDENS the SE: se >= mean per-repeat se_r (the
     across-split term (mu_r - mu_hat)^2 adds, never averages away).
  4. three_way_theta_frac / dropout / weight_decay are accepted and thread
     through both entry points (signatures + tiny runs returning finite output).
"""

import inspect

import numpy as np
import torch
from scipy.stats import norm

from deep_inference import inference, structural_dml
from deep_inference.engine.variance import compute_se_ci, median_dml_aggregate
from deep_inference.engine.crossfit import run_crossfit
from deep_inference.models.structural_net import train_structural_net, StructuralNet


def _logit_dgp(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    T = rng.standard_normal(n)
    alpha = 0.3 * X[:, 0]
    beta = 0.5
    p = 1.0 / (1.0 + np.exp(-(alpha + beta * T)))
    Y = (rng.random(n) < p).astype(np.float64)
    return Y, T, X


def _linear_dgp(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    T = rng.standard_normal(n)
    Y = X[:, 0] + 0.5 * T + rng.standard_normal(n) * 0.5
    return Y, T, X


# --- Item 1: median_dml_aggregate formula invariants -------------------------

def test_median_aggregate_single_repeat_reduces_exactly():
    """One repeat -> (mu_1, se_1) and CI = mu_1 +/- z*se_1 (median is a no-op)."""
    mu, se, lo, hi = median_dml_aggregate([0.42], [0.137])
    z = norm.ppf(0.975)
    assert np.isclose(mu, 0.42, atol=1e-12)
    assert np.isclose(se, 0.137, atol=1e-12)
    assert np.isclose(lo, 0.42 - z * 0.137, atol=1e-12)
    assert np.isclose(hi, 0.42 + z * 0.137, atol=1e-12)
    # z is the exact normal quantile, not the hardcoded 1.96.
    assert not np.isclose(z, 1.96, atol=1e-6)


def test_median_aggregate_se_only_widens():
    """Distinct mu_r with equal se_r: the across-split term strictly widens SE.

    mu_list=[.3,.5,.7], se=.1 -> augmented terms [.05,.01,.05], median .05,
    se=sqrt(.05)~0.2236 > mean(se_r)=0.10 and > median(se_r)=0.10.
    """
    mu_list = [0.30, 0.50, 0.70]
    se_list = [0.10, 0.10, 0.10]
    mu, se, _lo, _hi = median_dml_aggregate(mu_list, se_list)
    assert np.isclose(mu, 0.50, atol=1e-12)
    assert se >= float(np.mean(se_list))           # >= mean per-repeat se_r
    assert se >= float(np.median(se_list))          # >= median per-repeat se_r (theorem)
    assert se > float(np.mean(se_list)) + 1e-6      # strict widening here
    assert np.isclose(se, np.sqrt(0.05), atol=1e-12)


# --- Item 1: end-to-end repeated cross-fitting -------------------------------

def test_inference_n_repeats_runs_end_to_end():
    """n_repeats=3 on a tiny logit DGP returns finite mu_hat / se / CI."""
    Y, T, X = _logit_dgp(n=400, seed=1)
    np.random.seed(0)
    torch.manual_seed(0)
    r = inference(
        Y, T, X, model="logit", target="beta",
        n_folds=4, epochs=20, hidden_dims=[16], n_repeats=3,
    )
    assert np.isfinite(r.mu_hat)
    assert np.isfinite(r.se) and r.se > 0
    assert np.isfinite(r.ci_lower) and np.isfinite(r.ci_upper)
    assert r.ci_lower < r.ci_upper


# --- Item 1: no-regression at the n_repeats=1 default ------------------------

def test_n_repeats_1_is_single_split_inference():
    """Default path == explicit n_repeats=1, and equals the verbatim single
    split: mu==psi.mean and se==shared compute_se_ci on the SAME psi."""
    Y, T, X = _logit_dgp(n=400, seed=2)

    np.random.seed(11)
    torch.manual_seed(11)
    r_default = inference(
        Y, T, X, model="logit", target="beta",
        n_folds=4, epochs=20, hidden_dims=[16],
    )
    np.random.seed(11)
    torch.manual_seed(11)
    r_explicit = inference(
        Y, T, X, model="logit", target="beta",
        n_folds=4, epochs=20, hidden_dims=[16], n_repeats=1,
    )

    # default (no kwarg) and explicit n_repeats=1 are identical.
    assert np.allclose(
        r_default.psi_values.numpy(), r_explicit.psi_values.numpy(), atol=1e-10
    )
    assert np.isclose(r_default.mu_hat, r_explicit.mu_hat, atol=1e-12)
    assert np.isclose(r_default.se, r_explicit.se, atol=1e-12)

    # The median wrapper is a no-op for one repeat: verbatim single-split output.
    assert np.isclose(r_default.mu_hat, float(r_default.psi_values.mean()), atol=1e-6)
    se_shared, lo, hi, _ = compute_se_ci(r_default.psi_values, method="pooled")
    assert np.isclose(r_default.se, se_shared, rtol=1e-9, atol=1e-12)
    assert np.isclose(r_default.ci_lower, lo, atol=1e-9)
    assert np.isclose(r_default.ci_upper, hi, atol=1e-9)


def test_n_repeats_1_is_single_split_structural_dml():
    """Same no-regression guarantee for the legacy structural_dml core path."""
    Y, T, X = _linear_dgp(n=400, seed=3)

    np.random.seed(21)
    torch.manual_seed(21)
    r1 = structural_dml(
        Y, T, X, family="linear", n_folds=4, epochs=20, hidden_dims=[16],
    )
    np.random.seed(21)
    torch.manual_seed(21)
    r2 = structural_dml(
        Y, T, X, family="linear", n_folds=4, epochs=20, hidden_dims=[16],
        n_repeats=1,
    )

    assert np.allclose(r1.psi_values, r2.psi_values, atol=1e-10)
    assert np.isclose(r1.mu_hat, r2.mu_hat, atol=1e-12)
    assert np.isclose(r1.se, r2.se, atol=1e-12)

    # se is the verbatim shared estimator on its own psi; mu == psi.mean().
    se_shared, lo, hi, _ = compute_se_ci(r1.psi_values, method="pooled")
    assert np.isclose(r1.se, se_shared, rtol=1e-10, atol=1e-12)
    assert np.isclose(r1.mu_hat, r1.psi_values.mean(), atol=1e-12)
    assert np.isclose(r1.ci_lower, lo, atol=1e-9)
    assert np.isclose(r1.ci_upper, hi, atol=1e-9)


# --- Item 2: undersmoothing knobs accepted and threaded ----------------------

def test_signatures_expose_knobs():
    """The three knobs (plus n_repeats) appear on the relevant signatures."""
    cf = inspect.signature(run_crossfit).parameters
    for name in ("n_repeats", "three_way_theta_frac", "dropout", "weight_decay"):
        assert name in cf, f"run_crossfit missing {name}"

    inf = inspect.signature(inference).parameters
    sdml = inspect.signature(structural_dml).parameters
    for name in ("n_repeats", "three_way_theta_frac", "dropout", "weight_decay"):
        assert name in inf, f"inference missing {name}"
        assert name in sdml, f"structural_dml missing {name}"

    assert "weight_decay" in inspect.signature(train_structural_net).parameters
    assert "dropout" in inspect.signature(StructuralNet.__init__).parameters


def test_knobs_thread_through_inference():
    """Non-default knobs run end-to-end and return finite output (logit, 3-way)."""
    Y, T, X = _logit_dgp(n=400, seed=4)
    np.random.seed(0)
    torch.manual_seed(0)
    r = inference(
        Y, T, X, model="logit", target="beta",
        n_folds=4, epochs=20, hidden_dims=[16],
        three_way_theta_frac=0.5, dropout=0.2, weight_decay=1e-4,
    )
    assert np.isfinite(r.mu_hat)
    assert np.isfinite(r.se) and r.se > 0


def test_knobs_thread_through_structural_dml():
    """Non-default knobs run end-to-end via the legacy path (linear, 2-way)."""
    Y, T, X = _linear_dgp(n=400, seed=5)
    np.random.seed(0)
    torch.manual_seed(0)
    r = structural_dml(
        Y, T, X, family="linear", n_folds=4, epochs=20, hidden_dims=[16],
        three_way_theta_frac=0.5, dropout=0.2, weight_decay=1e-4,
    )
    assert np.isfinite(r.mu_hat)
    assert np.isfinite(r.se) and r.se > 0


def test_weight_decay_actually_reaches_optimizer():
    """A large weight_decay changes the fitted psi vs wd=0, proving it is not
    silently dropped on the way to torch.optim.Adam."""
    Y, T, X = _linear_dgp(n=400, seed=6)

    np.random.seed(31)
    torch.manual_seed(31)
    r0 = structural_dml(
        Y, T, X, family="linear", n_folds=4, epochs=20, hidden_dims=[16],
        weight_decay=0.0,
    )
    np.random.seed(31)
    torch.manual_seed(31)
    r_wd = structural_dml(
        Y, T, X, family="linear", n_folds=4, epochs=20, hidden_dims=[16],
        weight_decay=5.0,
    )
    # Same seeds, only weight_decay differs -> the estimate must move.
    assert not np.allclose(r0.psi_values, r_wd.psi_values, atol=1e-6)
