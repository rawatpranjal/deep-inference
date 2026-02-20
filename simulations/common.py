"""
Shared utilities for semi-synthetic simulations.

Provides:
- PCA-based DGP construction helpers
- Coverage metrics computation
- Reporting utilities
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from typing import Tuple, Dict, List, Optional


def extract_pca_features(X: np.ndarray, n_components: int = 2,
                          seed: int = 42) -> Tuple[np.ndarray, PCA]:
    """Extract centered PCA components from feature matrix.

    The components are standardized (zero mean, unit variance) so that
    E[z_k] = 0 for all k. This means linear functions of z have
    analytically known expectations: E[a + b*z_k] = a.

    Args:
        X: (n, d) feature matrix
        n_components: number of PCA components
        seed: random state for PCA

    Returns:
        z: (n, n_components) standardized PCA components
        pca: fitted PCA object
    """
    pca = PCA(n_components=n_components, random_state=seed)
    z_raw = pca.fit_transform(X)
    # Standardize: zero mean, unit variance
    z = (z_raw - z_raw.mean(axis=0)) / (z_raw.std(axis=0) + 1e-8)
    return z, pca


def compute_coverage_metrics(mu_hats: List[float], ses: List[float],
                              mu_true: float) -> Dict[str, float]:
    """Compute standard coverage metrics from MC replications.

    Args:
        mu_hats: list of point estimates
        ses: list of standard errors
        mu_true: true parameter value

    Returns:
        dict with coverage, se_ratio, bias, z_mean, z_std
    """
    mu_hats = np.array(mu_hats)
    ses = np.array(ses)

    # Coverage
    ci_lower = mu_hats - 1.96 * ses
    ci_upper = mu_hats + 1.96 * ses
    covers = (ci_lower <= mu_true) & (mu_true <= ci_upper)
    coverage = covers.mean()

    # SE ratio
    empirical_sd = mu_hats.std()
    mean_se = ses.mean()
    se_ratio = mean_se / empirical_sd if empirical_sd > 0 else float('inf')

    # Bias
    bias = mu_hats.mean() - mu_true

    # z-scores
    z_scores = (mu_hats - mu_true) / ses
    z_mean = z_scores.mean()
    z_std = z_scores.std()

    return {
        'coverage': coverage,
        'se_ratio': se_ratio,
        'bias': bias,
        'z_mean': z_mean,
        'z_std': z_std,
        'empirical_sd': empirical_sd,
        'mean_se': mean_se,
        'n_reps': len(mu_hats),
    }


def print_coverage_report(metrics: Dict[str, float], sim_name: str,
                           mu_true: float) -> str:
    """Print and return formatted coverage report."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {sim_name}")
    lines.append(f"{'='*60}")
    lines.append(f"  True mu*:        {mu_true:.4f}")
    lines.append(f"  Replications:    {metrics['n_reps']}")
    lines.append(f"  Coverage:        {metrics['coverage']*100:.1f}%  {'PASS' if 0.90 <= metrics['coverage'] <= 0.99 else 'FAIL'}")
    lines.append(f"  SE ratio:        {metrics['se_ratio']:.3f}  {'PASS' if 0.7 <= metrics['se_ratio'] <= 1.5 else 'FAIL'}")
    lines.append(f"  |Bias|:          {abs(metrics['bias']):.4f}  {'PASS' if abs(metrics['bias']) < 0.05 else 'FAIL'}")
    lines.append(f"  z-mean:          {metrics['z_mean']:.3f}  {'PASS' if -0.3 < metrics['z_mean'] < 0.3 else 'FAIL'}")
    lines.append(f"  z-std:           {metrics['z_std']:.3f}  {'PASS' if 0.7 <= metrics['z_std'] <= 1.5 else 'FAIL'}")
    lines.append(f"  Empirical SD:    {metrics['empirical_sd']:.4f}")
    lines.append(f"  Mean SE:         {metrics['mean_se']:.4f}")
    lines.append(f"{'='*60}")

    report = '\n'.join(lines)
    print(report)
    return report


def confounded_treatment(z: np.ndarray, seed: int = None) -> np.ndarray:
    """Generate confounded treatment T = 0.3*z[:,0] + noise.

    Args:
        z: (n, d) PCA components (standardized)
        seed: random state

    Returns:
        T: (n,) treatment values
    """
    rng = np.random.RandomState(seed)
    T = 0.3 * z[:, 0] + rng.randn(len(z))
    return T
