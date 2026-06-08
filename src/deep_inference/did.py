"""
Closed-form 2x2 difference-in-differences with influence-function inference.

Scope: the canonical homogeneous 2x2 *repeated cross-section* DiD. The estimand is
the group x post interaction

    beta = mu_11 - mu_10 - mu_01 + mu_00,    mu_gt = E[Y | G=g, T=t].

This is a design-based, closed-form estimator -- it does NOT use the neural
inference() path. It still follows the package's methodology: estimate a target by
averaging influence-function pseudo-outcomes, with SE = std(psi) / sqrt(n).

For the saturated cell-mean loss l_i = 0.5 (Y_i - W_i'theta)^2 with one-hot cell
indicator W_i and theta = (mu_00, mu_01, mu_10, mu_11), the expected Hessian is
Lambda = E[W W'] = diag(p_00, p_01, p_10, p_11). Plugging into the package IF
formula psi = H - H_theta Lambda^{-1} l_theta with H(theta) = a'theta,
a = (+1, -1, -1, +1) gives, for the observation in cell C_i,

    psi_i = beta_hat + a_{C_i} (Y_i - mu_hat_{C_i}) / p_hat_{C_i}.

Then beta_hat = mean(psi_i) and Var(psi)/n (with the 1/n / ddof=0 denominator)
equals the four-cell variance sum_gt sigma^2_gt / n_gt, which is exactly the HC0
robust OLS variance of the saturated regression Y = a + g G + l T + b (G T) + u.
Set use_bessel=False (the default) to match HC0 to machine precision.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from scipy.stats import norm


# Cell order is (group, post). Signs implement the DiD contrast
# beta = mu_11 - mu_10 - mu_01 + mu_00.
CELL_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
SIGN = {
    (0, 0): +1.0,
    (0, 1): -1.0,
    (1, 0): -1.0,
    (1, 1): +1.0,
}


def _as_binary_1d(x: Any, name: str) -> np.ndarray:
    """Validate that x is a 1-D array of binary {0, 1} values and return it as int."""
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}.")

    vals = np.unique(arr)
    if not np.all(np.isin(vals, [0, 1, False, True])):
        raise ValueError(f"{name} must be binary with values in {{0, 1}}.")

    return arr.astype(np.int64)


def did_2x2_arrays(
    Y: Any,
    group: Any,
    post: Any,
    *,
    alpha: float = 0.05,
    use_bessel: bool = False,
) -> dict[str, Any]:
    """
    Simple repeated-cross-section 2x2 DiD via the influence-function pseudo-outcome
    convention psi_i = beta_hat + IF_i.

    Then:
        beta_hat = mean(psi_i)
        se       = sqrt(var(psi_i) / n)

    use_bessel=False (default) gives the exact HC0 / plug-in IF variance; use_bessel=True
    applies a finite-sample (n-1) correction to the cell variances and will NOT equal HC0.

    Args:
        Y: Outcome, 1-D array of length n.
        group: Binary treatment-group indicator G in {0, 1}, length n.
        post: Binary post-period indicator T in {0, 1}, length n.
        alpha: CI level (default 0.05 -> 95% CI).
        use_bessel: If False (default), denom = n (HC0). If True, denom = n - 1.

    Returns:
        Dict with InferenceResult fields (mu_hat, se, ci_lower, ci_upper, psi_values,
        theta_hat, diagnostics) plus if_values and n.
    """
    Y = np.asarray(Y, dtype=np.float64)
    G = _as_binary_1d(group, "group")
    P = _as_binary_1d(post, "post")

    if Y.ndim != 1:
        raise ValueError(f"Y must be 1D, got shape {Y.shape}.")

    n = len(Y)
    if len(G) != n or len(P) != n:
        raise ValueError("Y, group, and post must have the same length.")

    if np.any(~np.isfinite(Y)):
        raise ValueError("Y contains NaN or Inf values.")

    cell_means: dict[tuple[int, int], float] = {}
    cell_counts: dict[tuple[int, int], int] = {}
    cell_props: dict[tuple[int, int], float] = {}

    for g, t in CELL_ORDER:
        idx = (G == g) & (P == t)
        count = int(idx.sum())

        if count == 0:
            raise ValueError(f"Empty DiD cell: group={g}, post={t}.")

        cell_counts[(g, t)] = count
        cell_props[(g, t)] = count / n
        cell_means[(g, t)] = float(Y[idx].mean())

    beta_hat = float(sum(SIGN[c] * cell_means[c] for c in CELL_ORDER))

    # Mean-zero influence-function values.
    if_values = np.zeros(n, dtype=np.float64)
    for g, t in CELL_ORDER:
        idx = (G == g) & (P == t)
        p_hat = cell_props[(g, t)]
        mu_hat = cell_means[(g, t)]
        if_values[idx] = SIGN[(g, t)] * (Y[idx] - mu_hat) / p_hat

    # Package convention: psi_values average to the target estimate.
    psi_values = beta_hat + if_values

    denom = n - 1 if use_bessel else n
    var_psi = float(np.sum((psi_values - beta_hat) ** 2) / denom)
    se = float(np.sqrt(var_psi / n))

    z = norm.ppf(1.0 - alpha / 2.0)
    ci_lower = float(beta_hat - z * se)
    ci_upper = float(beta_hat + z * se)

    theta_vec = np.array([cell_means[c] for c in CELL_ORDER], dtype=np.float64)

    # Stringify (g, t) tuple keys so summary()/JSON logging render cleanly.
    def _k(c: tuple[int, int]) -> str:
        return f"g{c[0]}_t{c[1]}"

    diagnostics = {
        "estimator": "did_2x2",
        "design": "repeated_cross_section",
        "se_type": "if_hc0" if not use_bessel else "if_bessel",
        "cell_order": [list(c) for c in CELL_ORDER],
        "cell_counts": {_k(c): cell_counts[c] for c in CELL_ORDER},
        "cell_props": {_k(c): cell_props[c] for c in CELL_ORDER},
        "cell_means": {_k(c): cell_means[c] for c in CELL_ORDER},
        "if_mean": float(if_values.mean()),
        "var_psi": var_psi,
    }

    return {
        "mu_hat": beta_hat,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "psi_values": torch.tensor(psi_values, dtype=torch.float64),
        "if_values": torch.tensor(if_values, dtype=torch.float64),
        "theta_hat": torch.tensor(
            np.tile(theta_vec, (n, 1)),
            dtype=torch.float64,
        ),
        "diagnostics": diagnostics,
        "n": n,
    }
