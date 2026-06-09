"""
Variance estimation for influence function-based inference.

Standard estimator (from paper):
    Ψ̂ = (1/n) Σ (ψᵢ - μ̂)²
    SE = √(Ψ̂/n)
"""

from typing import Optional, Tuple
import torch
from torch import Tensor


def estimate_variance(
    psi: Tensor,
    mu_hat: Optional[float] = None,
) -> float:
    """
    Estimate variance of influence function.

    Ψ̂ = (1/n) Σ (ψᵢ - μ̂)²

    Args:
        psi: (n,) influence function values
        mu_hat: Point estimate (if None, computed from psi)

    Returns:
        Variance estimate Ψ̂
    """
    n = psi.shape[0]

    if mu_hat is None:
        mu_hat = psi.mean().item()

    # Sample variance
    variance = ((psi - mu_hat) ** 2).sum().item() / n

    return variance


def estimate_variance_bessel(
    psi: Tensor,
    mu_hat: Optional[float] = None,
) -> float:
    """
    Estimate variance with Bessel correction.

    Ψ̂ = (1/(n-1)) Σ (ψᵢ - μ̂)²

    Args:
        psi: (n,) influence function values
        mu_hat: Point estimate (if None, computed from psi)

    Returns:
        Variance estimate Ψ̂
    """
    n = psi.shape[0]

    if mu_hat is None:
        mu_hat = psi.mean().item()

    variance = ((psi - mu_hat) ** 2).sum().item() / (n - 1)

    return variance


def compute_se(
    psi: Tensor,
    mu_hat: Optional[float] = None,
    use_bessel: bool = True,
) -> float:
    """
    Compute standard error of the mean.

    SE = √(Ψ̂/n)

    Args:
        psi: (n,) influence function values
        mu_hat: Point estimate
        use_bessel: Whether to use Bessel correction

    Returns:
        Standard error
    """
    n = psi.shape[0]

    if use_bessel:
        variance = estimate_variance_bessel(psi, mu_hat)
    else:
        variance = estimate_variance(psi, mu_hat)

    se = (variance / n) ** 0.5

    return se


def compute_confidence_interval(
    mu_hat: float,
    se: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute confidence interval.

    CI = [μ̂ - z_{α/2} × SE, μ̂ + z_{α/2} × SE]

    Args:
        mu_hat: Point estimate
        se: Standard error
        alpha: Significance level (default: 0.05 for 95% CI)

    Returns:
        (lower, upper) confidence interval bounds
    """
    import scipy.stats as stats

    z = stats.norm.ppf(1 - alpha / 2)

    lower = mu_hat - z * se
    upper = mu_hat + z * se

    return lower, upper


def compute_se_ci(
    psi,
    fold_indices=None,
    n: Optional[int] = None,
    method: str = "pooled",
    alpha: float = 0.05,
) -> Tuple[float, float, float, float]:
    """
    Shared standard-error / confidence-interval computation for IF inference.

    This is the single source of truth used by BOTH entry points (the engine
    `inference()` path and the legacy `structural_dml()` path). The point
    estimate is the global mean of psi in BOTH variants; only the variance
    estimator differs.

    method='pooled' (default, FLM):
        Ψ̂ = (1/(n-1)) Σ_i (ψ_i - μ̂)²
        Bessel-corrected sample variance centered at the GLOBAL mean. This is
        the influence-function variance prescribed by the paper (FLM2025 §3.2:
        "Ψ = V[ψ(...)]" estimated by the sample variance of ψ; FLM2021 §4:
        replace first moments with second moments).

    method='within_fold' (legacy per-fold-centered variant):
        Ψ̂ = (1/K) Σ_k mean_{i∈fold k} (ψ_i - μ̂_k)²
        Mean of the per-fold variances, each centered at its OWN fold mean.
        Requires `fold_indices`. Drops the between-fold component, so it is
        always <= pooled and yields (weakly) narrower confidence intervals.

    SE = √(Ψ̂ / n);  CI = μ̂ ± z·SE  with z = norm.ppf(1 - alpha/2).

    Args:
        psi: (n,) influence function values (torch.Tensor or np.ndarray).
        fold_indices: (n,) fold assignment; required for method='within_fold'.
        n: Sample size (defaults to len(psi)).
        method: 'pooled' (default, FLM) or 'within_fold' (legacy).
        alpha: Significance level (default 0.05 -> 95% CI).

    Returns:
        (se, ci_lower, ci_upper, variance)
    """
    import numpy as np
    from scipy.stats import norm

    # Accept torch tensors or numpy arrays; compute in float64 for stability.
    if hasattr(psi, "detach"):
        psi_arr = psi.detach().cpu().numpy()
    else:
        psi_arr = np.asarray(psi)
    psi_arr = psi_arr.reshape(-1).astype(np.float64)

    if n is None:
        n = psi_arr.shape[0]

    mu_hat = float(psi_arr.mean())

    if method == "pooled":
        # Bessel-corrected sample variance centered at the global mean.
        variance = float(((psi_arr - mu_hat) ** 2).sum() / (n - 1))
    elif method == "within_fold":
        # Mean of per-fold variances, each centered at its own fold mean.
        if fold_indices is None:
            raise ValueError("method='within_fold' requires fold_indices.")
        if hasattr(fold_indices, "detach"):
            fold_arr = fold_indices.detach().cpu().numpy()
        else:
            fold_arr = np.asarray(fold_indices)
        fold_arr = fold_arr.reshape(-1)
        fold_labels = np.unique(fold_arr)
        var_sum = 0.0
        for k in fold_labels:
            psi_k = psi_arr[fold_arr == k]
            mu_k = psi_k.mean()
            var_sum += float(((psi_k - mu_k) ** 2).mean())
        variance = var_sum / len(fold_labels)
    else:
        raise ValueError(
            f"Unknown variance method: {method!r}. Use 'pooled' or 'within_fold'."
        )

    se = (variance / n) ** 0.5
    z = norm.ppf(1 - alpha / 2)
    ci_lower = mu_hat - z * se
    ci_upper = mu_hat + z * se

    return se, ci_lower, ci_upper, variance


def compute_inference_results(
    psi: Tensor,
    alpha: float = 0.05,
    method: str = "pooled",
    fold_indices=None,
) -> dict:
    """
    Compute complete inference results.

    Args:
        psi: (n,) influence function values
        alpha: Significance level
        method: Variance estimator, 'pooled' (default, FLM) or 'within_fold'
            (legacy per-fold-centered). See compute_se_ci.
        fold_indices: (n,) fold assignment; required for method='within_fold'.

    Returns:
        Dictionary with mu_hat, se, ci_lower, ci_upper
    """
    n = psi.shape[0]
    mu_hat = psi.mean().item()
    se, ci_lower, ci_upper, _variance = compute_se_ci(
        psi, fold_indices=fold_indices, n=n, method=method, alpha=alpha
    )

    return {
        "mu_hat": mu_hat,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
    }
