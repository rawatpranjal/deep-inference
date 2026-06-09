"""
Fixed-effects utilities for panel difference-in-differences.

Two-way (unit + time) within transformation by iterative alternating projection:

    Ytilde_it = Y_it - mean_i - mean_t + ...   (additive unit and time effects removed)

For a balanced panel a single sweep of (subtract unit means, subtract time means) is
exact up to the grand mean; the iteration below converges for unbalanced panels too.
After residualization, additive unit/time fixed effects are absorbed, so the within
regression of Ytilde on Dtilde identifies the treatment slope (the DiD effect).
"""

from __future__ import annotations

import numpy as np


def _group_demean(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Subtract the within-group mean of `values` for each level of `group`."""
    # group means via bincount (group must be non-negative integer codes)
    sums = np.bincount(group, weights=values)
    counts = np.bincount(group)
    means = sums / np.maximum(counts, 1)
    return values - means[group]


def residualize_fixed_effects(
    values,
    unit,
    time,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Two-way (unit + time) within transformation by alternating projections.

    Args:
        values: (n,) array to residualize (outcome or treatment).
        unit: (n,) unit identifiers (any hashable; mapped to integer codes).
        time: (n,) time/period identifiers (any hashable; mapped to integer codes).
        tol: convergence tolerance on the max absolute change between sweeps.
        max_iter: maximum number of alternating sweeps.

    Returns:
        (n,) residualized array with additive unit and time effects removed.
    """
    v = np.asarray(values, dtype=np.float64).copy()
    u_codes = np.unique(np.asarray(unit), return_inverse=True)[1].astype(np.int64)
    t_codes = np.unique(np.asarray(time), return_inverse=True)[1].astype(np.int64)

    for _ in range(max_iter):
        prev = v
        v = _group_demean(v, u_codes)
        v = _group_demean(v, t_codes)
        if np.max(np.abs(v - prev)) < tol:
            break
    return v
