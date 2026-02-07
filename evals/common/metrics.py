"""
Standardized Evaluation Metrics and Thresholds

Central source of truth for all pass/fail criteria across the eval suite.
Strict thresholds ensure meaningful validation — a passing eval that misses
bugs is worse than useless.
"""

from typing import Dict, Tuple, Optional
import numpy as np


# ── Coverage thresholds (eval_06, eval_09, etc.) ──

COVERAGE_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "coverage":  (0.90, 0.99),   # 90-99% coverage for valid CIs
    "se_ratio":  (0.7, 1.5),     # SE calibration: emp_SE / est_SE
    "abs_bias":  (0, 0.05),      # |bias| < 0.05 for unbiasedness
    "z_mean":    (-0.3, 0.3),    # z-score mean near 0
    "z_std":     (0.7, 1.5),     # z-score std near 1
}

# ── Parameter recovery thresholds (eval_01, etc.) ──

RECOVERY_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "rmse":        (0, 0.3),     # θ̂ accuracy
    "correlation": (0.7, 1.0),   # θ̂ manifold shape
}

# ── Autodiff validation thresholds (eval_02) ──

AUTODIFF_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "gradient_error": (0, 1e-6),   # |autodiff - analytic| for gradients
    "hessian_error":  (0, 1e-6),   # |autodiff - analytic| for Hessians
}

# ── Lambda estimation thresholds (eval_03) ──

LAMBDA_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "frobenius_error": (0, 0.15),          # ||Λ̂ - Λ*||_F / ||Λ*||_F
    "non_psd_count":   (0, 0),             # Must be 0 non-PSD estimates
    "min_eigenvalue":  (1e-4, float('inf')),  # Stability check
}

# ── Influence function thresholds (eval_05) ──

PSI_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "psi_correlation": (0.9, 1.0),   # corr(ψ_pkg, ψ_oracle)
    "psi_rmse":        (0, 0.1),     # RMSE between package and oracle ψ
}


def check_metric(
    name: str,
    value: float,
    thresholds: Dict[str, Tuple[float, float]],
) -> Tuple[bool, str]:
    """
    Check a single metric against its threshold.

    Args:
        name: Metric name (must be a key in thresholds)
        value: Observed metric value
        thresholds: Threshold dictionary

    Returns:
        (passed, message) tuple
    """
    if name not in thresholds:
        return False, f"Unknown metric: {name}"

    lo, hi = thresholds[name]

    if np.isnan(value):
        return False, f"{name} = NaN (expected [{lo}, {hi}])"

    passed = lo <= value <= hi
    status = "PASS" if passed else "FAIL"
    return passed, f"{name} = {value:.6f} [{lo}, {hi}]: {status}"


def validate_coverage_run(metrics: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate a coverage MC run against strict thresholds.

    Args:
        metrics: Dict with keys: coverage, se_ratio, bias, z_mean, z_std

    Returns:
        (all_pass, criteria_dict) where criteria_dict maps criterion name to pass/fail
    """
    criteria = {}

    # Coverage
    lo, hi = COVERAGE_THRESHOLDS["coverage"]
    cov = metrics.get("coverage", float('nan'))
    criteria[f"Coverage in [{lo*100:.0f}%, {hi*100:.0f}%]"] = lo <= cov <= hi

    # SE ratio
    lo, hi = COVERAGE_THRESHOLDS["se_ratio"]
    se_r = metrics.get("se_ratio", float('nan'))
    criteria[f"SE Ratio in [{lo}, {hi}]"] = lo <= se_r <= hi

    # Bias
    lo, hi = COVERAGE_THRESHOLDS["abs_bias"]
    bias = metrics.get("bias", float('nan'))
    criteria[f"|Bias| < {hi}"] = abs(bias) < hi if not np.isnan(bias) else False

    # z-score mean
    lo, hi = COVERAGE_THRESHOLDS["z_mean"]
    z_m = metrics.get("z_mean", float('nan'))
    criteria[f"|z_mean| < {abs(hi)}"] = lo <= z_m <= hi if not np.isnan(z_m) else False

    # z-score std
    lo, hi = COVERAGE_THRESHOLDS["z_std"]
    z_s = metrics.get("z_std", float('nan'))
    criteria[f"z_std in [{lo}, {hi}]"] = lo <= z_s <= hi if not np.isnan(z_s) else False

    all_pass = all(criteria.values())
    return all_pass, criteria


def validate_recovery_run(metrics: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate a parameter recovery run against strict thresholds.

    Args:
        metrics: Dict with keys like rmse_alpha, rmse_beta, corr_alpha, corr_beta

    Returns:
        (all_pass, criteria_dict)
    """
    criteria = {}
    lo_rmse, hi_rmse = RECOVERY_THRESHOLDS["rmse"]
    lo_corr, hi_corr = RECOVERY_THRESHOLDS["correlation"]

    for key, val in metrics.items():
        if "rmse" in key:
            criteria[f"{key} < {hi_rmse}"] = val < hi_rmse if not np.isnan(val) else False
        elif "corr" in key:
            criteria[f"{key} > {lo_corr}"] = val > lo_corr if not np.isnan(val) else False

    all_pass = all(criteria.values()) if criteria else False
    return all_pass, criteria


def validate_autodiff_run(metrics: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate autodiff vs analytic comparison.

    Args:
        metrics: Dict with keys like gradient_error, hessian_error

    Returns:
        (all_pass, criteria_dict)
    """
    criteria = {}
    for name, (lo, hi) in AUTODIFF_THRESHOLDS.items():
        val = metrics.get(name, float('nan'))
        criteria[f"{name} < {hi}"] = val < hi if not np.isnan(val) else False

    all_pass = all(criteria.values()) if criteria else False
    return all_pass, criteria


def validate_lambda_run(metrics: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate Lambda estimation accuracy.

    Args:
        metrics: Dict with keys like frobenius_error, non_psd_count, min_eigenvalue

    Returns:
        (all_pass, criteria_dict)
    """
    criteria = {}
    for name, (lo, hi) in LAMBDA_THRESHOLDS.items():
        val = metrics.get(name, float('nan'))
        if np.isnan(val):
            criteria[f"{name} in [{lo}, {hi}]"] = False
        else:
            criteria[f"{name} in [{lo}, {hi}]"] = lo <= val <= hi

    all_pass = all(criteria.values()) if criteria else False
    return all_pass, criteria


def validate_psi_run(metrics: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
    """
    Validate influence function assembly.

    Args:
        metrics: Dict with keys like psi_correlation, psi_rmse

    Returns:
        (all_pass, criteria_dict)
    """
    criteria = {}
    for name, (lo, hi) in PSI_THRESHOLDS.items():
        val = metrics.get(name, float('nan'))
        if np.isnan(val):
            criteria[f"{name} in [{lo}, {hi}]"] = False
        else:
            criteria[f"{name} in [{lo}, {hi}]"] = lo <= val <= hi

    all_pass = all(criteria.values()) if criteria else False
    return all_pass, criteria


def format_validation_table(criteria: Dict[str, bool]) -> str:
    """
    Format validation criteria as a printable table.

    Args:
        criteria: Dict mapping criterion name to pass/fail

    Returns:
        Formatted string with PASS/FAIL status for each criterion
    """
    lines = []
    for name, passed in criteria.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"  {name}: {status}")
    return "\n".join(lines)
