"""
Tests for the unified influence-function variance (Batch 3).

Both entry points -- structural_dml() (core path) and inference() (engine path)
-- now route their SE/CI computation through the single shared estimator
engine.variance.compute_se_ci. The default is the FLM POOLED variance (Bessel
sample variance of psi centered at the global mean); 'within_fold' is the legacy
per-fold-centered variant.

These tests assert:
  1. compute_se_ci implements both formulas exactly (z = norm.ppf, no 1.96).
  2. The shared function returns the SAME SE for the same psi regardless of the
     calling entry point (numpy core-style vs torch engine-style).
  3. within_fold <= pooled (it drops the between-fold component) and they differ.
  4. structural_dml's pooled SE is exactly reproduced by compute_se_ci on its
     own psi, and switching to variance='within_fold' on the IDENTICAL psi
     yields a strictly smaller (or equal) SE.
"""

import numpy as np
import torch
from scipy.stats import norm

from deep_inference.engine.variance import compute_se_ci, compute_inference_results
from deep_inference import structural_dml, inference


def _synthetic_psi(n_per_fold=40, n_folds=5, seed=7):
    """psi with genuine between-fold variation so within_fold < pooled strictly."""
    rng = np.random.default_rng(seed)
    psi = np.empty(n_per_fold * n_folds, dtype=np.float64)
    folds = np.empty(n_per_fold * n_folds, dtype=np.int64)
    for k in range(n_folds):
        offset = (k - n_folds / 2) * 0.5  # distinct fold mean -> between-fold var > 0
        sl = slice(k * n_per_fold, (k + 1) * n_per_fold)
        psi[sl] = rng.normal(loc=offset, scale=1.0, size=n_per_fold)
        folds[sl] = k
    return psi, folds


def test_pooled_formula_matches_bessel_and_z():
    psi, folds = _synthetic_psi()
    n = psi.shape[0]
    se, lo, hi, var = compute_se_ci(psi, fold_indices=folds, n=n, method="pooled")

    mu = psi.mean()
    var_manual = ((psi - mu) ** 2).sum() / (n - 1)
    se_manual = np.sqrt(var_manual / n)
    z = norm.ppf(0.975)

    assert np.isclose(var, var_manual, rtol=0, atol=1e-12)
    assert np.isclose(se, se_manual, rtol=0, atol=1e-12)
    assert np.isclose(lo, mu - z * se_manual, atol=1e-12)
    assert np.isclose(hi, mu + z * se_manual, atol=1e-12)
    # Explicitly the exact normal quantile, NOT the hardcoded 1.96.
    assert not np.isclose(z, 1.96, atol=1e-6)


def test_within_fold_formula_matches_manual():
    psi, folds = _synthetic_psi()
    n = psi.shape[0]
    se, lo, hi, var = compute_se_ci(psi, fold_indices=folds, n=n, method="within_fold")

    K = len(np.unique(folds))
    var_sum = 0.0
    for k in np.unique(folds):
        psi_k = psi[folds == k]
        var_sum += ((psi_k - psi_k.mean()) ** 2).mean()
    var_manual = var_sum / K
    se_manual = np.sqrt(var_manual / n)

    assert np.isclose(var, var_manual, rtol=0, atol=1e-12)
    assert np.isclose(se, se_manual, rtol=0, atol=1e-12)


def test_within_fold_le_pooled_and_differs():
    psi, folds = _synthetic_psi()
    n = psi.shape[0]
    se_pooled, *_ = compute_se_ci(psi, fold_indices=folds, n=n, method="pooled")
    se_within, *_ = compute_se_ci(psi, fold_indices=folds, n=n, method="within_fold")

    assert se_within <= se_pooled + 1e-12
    # With genuine between-fold variation the two estimators are distinct.
    assert not np.isclose(se_within, se_pooled, rtol=1e-3)


def test_within_fold_requires_fold_indices():
    psi, _ = _synthetic_psi()
    try:
        compute_se_ci(psi, fold_indices=None, method="within_fold")
        raised = False
    except ValueError:
        raised = True
    assert raised, "within_fold without fold_indices must raise ValueError"


def test_shared_function_unifies_entry_points():
    """Same psi -> same pooled SE whether called numpy-style (core) or via the
    engine helper compute_inference_results on a torch tensor."""
    psi_np, _ = _synthetic_psi()
    se_core, _, _, _ = compute_se_ci(psi_np, method="pooled")

    psi_t = torch.tensor(psi_np, dtype=torch.float64)
    engine = compute_inference_results(psi_t, method="pooled")

    assert np.isclose(se_core, engine["se"], rtol=1e-10, atol=1e-12)


# --- Integration: structural_dml toggle on the IDENTICAL psi -----------------

def _linear_dgp(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    T = rng.standard_normal(n)
    Y = X[:, 0] + 0.5 * T + rng.standard_normal(n) * 0.5
    return Y, T, X


def _run_sdml(variance):
    # Reset BOTH global RNGs so the full pipeline (folds + NN init/training) is
    # byte-identical across calls; only the final variance method differs, and it
    # consumes no RNG. Hence psi is identical and only the SE can move.
    np.random.seed(123)
    torch.manual_seed(123)
    Y, T, X = _linear_dgp()
    return structural_dml(
        Y, T, X, family="linear", n_folds=4, epochs=25, hidden_dims=[16],
        variance=variance, verbose=False,
    )


def test_structural_dml_pooled_vs_within_fold_same_psi():
    r_pooled = _run_sdml("pooled")
    r_within = _run_sdml("within_fold")

    # Identical psi and point estimate: only the variance estimator changed.
    assert np.allclose(r_pooled.psi_values, r_within.psi_values, rtol=0, atol=1e-10)
    assert np.isclose(r_pooled.mu_hat, r_within.mu_hat, rtol=0, atol=1e-10)

    # within_fold drops between-fold variation -> SE <= pooled, and they differ.
    assert r_within.se <= r_pooled.se + 1e-9
    assert not np.isclose(r_within.se, r_pooled.se, rtol=1e-4)

    # The reported pooled SE is exactly the shared estimator on its own psi.
    se_shared, _, _, _ = compute_se_ci(r_pooled.psi_values, method="pooled")
    assert np.isclose(r_pooled.se, se_shared, rtol=1e-10, atol=1e-12)


def test_structural_dml_and_inference_pooled_sane_on_shared_dgp():
    """Both top-level APIs run pooled on the same DGP and return finite, sane
    SEs. NOTE: the two pipelines use independent fold/NN RNG, so they do NOT
    assemble the same psi; the rigorous same-psi -> same-SE invariant is proven
    in test_shared_function_unifies_entry_points. Here we only sanity-check that
    both default to pooled and recover the truth within a few SE."""
    Y, T, X = _linear_dgp(seed=1)

    np.random.seed(5); torch.manual_seed(5)
    r_sdml = structural_dml(
        Y, T, X, family="linear", n_folds=4, epochs=25, hidden_dims=[16],
    )
    np.random.seed(5); torch.manual_seed(5)
    r_inf = inference(
        Y, T, X, model="linear", target="beta",
        n_folds=4, epochs=25, hidden_dims=[16],
    )

    for r in (r_sdml, r_inf):
        assert np.isfinite(r.se) and r.se > 0
        assert abs(r.mu_hat - 0.5) < 5 * r.se
