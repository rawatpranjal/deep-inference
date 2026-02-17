"""
mle_core.py - Scalable MLE Core for Deep Logit Simulations

Copied from deep-aesthetics project for reference.
Original: /Users/pranjal/Dropbox/deep-aesthetics/final-analysis/choice/mle_core.py

Provides grouped count-based MLE, Newton optimization, and variance-based SE
computation that scales to millions of observations via O(M*G*J) complexity.

Key functions:
- aggregate_to_counts(): Convert individual choices to sufficient statistics
- mnl_grouped_nll(): Grouped multinomial logit negative log-likelihood
- alpha_newton_step(): One-dimensional Newton step for price coefficient
- compute_alpha_se(): Variance-based SE (no Hessian autodiff)
- center_function(): Location-invariant centering for g(x)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Union
import numpy as np


def aggregate_to_counts(
    y: torch.Tensor,
    groups: Optional[torch.Tensor] = None,
    markets: Optional[torch.Tensor] = None,
    J: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate individual choices to group-by-product (or market-group-product) counts.

    Sufficient statistic for multinomial logit: within (market, group), if all
    individuals face same (p, x), then counts[m, g, j] capture all information.

    Args:
        y: Individual choices [n] with values in {0, ..., J-1}
        groups: Group indicators [n] (optional, defaults to single group)
        markets: Market indicators [n] (optional, defaults to single market)
        J: Number of products (if None, inferred from y.max()+1)

    Returns:
        counts: [M, G, J] or [G, J] or [J] tensor of choice counts
        n_mg: [M, G, 1] or [G, 1] or scalar - total choices per (market, group)
    """
    n = len(y)
    J = J or (y.max().item() + 1)

    # Default to single group/market if not provided
    if groups is None:
        groups = torch.zeros(n, dtype=torch.long)
    if markets is None:
        markets = torch.zeros(n, dtype=torch.long)

    G = groups.max().item() + 1
    M = markets.max().item() + 1

    # Aggregate to counts[m, g, j]
    counts = torch.zeros(M, G, J, dtype=torch.float32)
    for m in range(M):
        for g in range(G):
            mask = (markets == m) & (groups == g)
            y_mg = y[mask]
            if len(y_mg) > 0:
                counts[m, g] = torch.bincount(y_mg, minlength=J).float()

    # Total choices per (m, g)
    n_mg = counts.sum(dim=-1, keepdim=True)  # [M, G, 1]

    # Squeeze singleton dimensions
    if M == 1 and G == 1:
        return counts.squeeze(0).squeeze(0), n_mg.squeeze()  # [J], scalar
    elif M == 1:
        return counts.squeeze(0), n_mg.squeeze(0)  # [G, J], [G, 1]
    else:
        return counts, n_mg  # [M, G, J], [M, G, 1]


def mnl_grouped_nll(
    alpha: Union[torch.Tensor, float],
    g_vals: torch.Tensor,
    p: torch.Tensor,
    counts: torch.Tensor,
    alpha_per_group: bool = False
) -> torch.Tensor:
    """
    Grouped multinomial logit negative log-likelihood using counts.

    Exact MLE via sufficient statistics. No O(n*J) expansion needed.

    Args:
        alpha: Price coefficient(s)
               - If scalar: shared across all groups
               - If [G]: different per group (requires alpha_per_group=True)
        g_vals: Product utility from network
                - Shape [J] (single group) or [G, J] or [M, G, J]
        p: Product prices
           - Shape [J] or [M, J]
        counts: Choice counts
                - Shape [J] or [G, J] or [M, G, J]
        alpha_per_group: If True, alpha is [G] vector

    Returns:
        Negative log-likelihood (scalar)

    Formula:
        NLL = -Sum_{m,g,j} c_{mgj} * log(softmax(u_{mgj}))
        where u_{mgj} = alpha_g * p_{mj} + g_g(x_{mj})
    """
    # Ensure proper shapes
    if counts.dim() == 1:  # [J] -> [1, 1, J]
        counts = counts.view(1, 1, -1)
        g_vals = g_vals.view(1, 1, -1) if g_vals.dim() == 1 else g_vals
        p = p.view(1, 1, -1) if p.dim() == 1 else p
        alpha = torch.tensor([alpha]) if isinstance(alpha, (int, float)) else alpha
        squeeze_output = True
    elif counts.dim() == 2:  # [G, J] -> [1, G, J]
        counts = counts.unsqueeze(0)
        g_vals = g_vals.unsqueeze(0) if g_vals.dim() == 2 else g_vals
        p = p.view(1, 1, -1)
        squeeze_output = True
    else:
        squeeze_output = False

    # Broadcast alpha
    if alpha_per_group:
        alpha = alpha.view(1, -1, 1)  # [1, G, 1]
    else:
        alpha = torch.tensor([[alpha]]) if isinstance(alpha, (int, float)) else alpha
        alpha = alpha.view(1, 1, 1)  # Scalar broadcast

    # Broadcast prices if needed
    if p.dim() == 2:  # [M, J] already
        p = p.unsqueeze(1)  # [M, 1, J]
    elif p.dim() == 1:
        p = p.view(1, 1, -1)  # [1, 1, J]

    # Compute utilities: u[M, G, J] = alpha * p + g(x)
    u = alpha * p + g_vals

    # Grouped cross-entropy via log_softmax
    log_probs = torch.log_softmax(u, dim=-1)  # softmax over J
    nll = -(counts * log_probs).sum()

    return nll


def alpha_newton_step(
    alpha_current: float,
    probs: torch.Tensor,
    p: torch.Tensor,
    counts: torch.Tensor,
    n_mg: torch.Tensor,
    lr: float = 1.0
) -> Tuple[float, float]:
    """
    One Newton step for alpha optimization (1D convex problem).

    Score: d_ell/d_alpha = -Sum c_{mgj} p_j + Sum n_{mg} E_{s_{mg}}[p]
    Hessian: d2_ell/d_alpha2 = Sum n_{mg} Var_{s_{mg}}[p]

    Args:
        alpha_current: Current alpha value
        probs: Model probabilities [M, G, J] or [G, J] or [J]
        p: Prices [M, J] or [J]
        counts: Counts [M, G, J] or [G, J] or [J]
        n_mg: Total per (m,g): [M, G, 1] or [G, 1] or scalar
        lr: Learning rate (default 1.0 for pure Newton)

    Returns:
        alpha_new: Updated alpha
        H: Observed Hessian (for SE computation)
    """
    # Ensure consistent shapes
    if probs.dim() == 1:
        probs = probs.view(1, 1, -1)
        p = p.view(1, 1, -1) if p.dim() == 1 else p.view(1, -1).unsqueeze(1)
        counts = counts.view(1, 1, -1)
        n_mg = n_mg.view(1, 1, 1) if n_mg.dim() == 0 else n_mg
    elif probs.dim() == 2:
        probs = probs.unsqueeze(0)
        p = p.view(1, 1, -1)
        counts = counts.unsqueeze(0)
        n_mg = n_mg.unsqueeze(0)

    if p.dim() == 2:  # [M, J]
        p = p.unsqueeze(1)  # [M, 1, J]

    # Score: derivative of NLL
    emp = (counts * p).sum()  # empirical weighted sum
    mod = (n_mg * (probs * p)).sum()  # model expectation
    score = -(emp - mod)  # negative because we minimize NLL

    # Hessian: sum of variances
    p_mean = (probs * p).sum(dim=-1, keepdim=True)  # [M, G, 1]
    p_var = (probs * (p - p_mean) ** 2).sum(dim=-1, keepdim=True)  # [M, G, 1]
    H = (n_mg * p_var).sum()  # scalar

    # Newton update: alpha_{t+1} = alpha_t - lr * score / H
    if H > 1e-8:  # Positive definite check
        alpha_new = alpha_current - lr * (score / H).item()
    else:
        alpha_new = alpha_current  # Fallback if Hessian is singular

    return alpha_new, H.item()


def compute_alpha_se(
    probs: torch.Tensor,
    p: torch.Tensor,
    n_mg: torch.Tensor
) -> float:
    """
    Compute standard error for alpha via variance trick (observed Fisher information).

    SE(alpha) = 1 / sqrt(Sum_{mg} n_{mg} * Var_{s_{mg}}[price])

    This is the conditional SE (treating g as fixed). Full sandwich estimator
    would account for g-estimation uncertainty, but this is sufficient for
    simulation validation and much faster than Hessian autodiff.

    Args:
        probs: Model probabilities [M, G, J] or [G, J] or [J]
        p: Prices [M, J] or [J]
        n_mg: Total per (m,g): [M, G, 1] or [G, 1] or scalar

    Returns:
        Standard error (float)
    """
    # Ensure consistent shapes (same logic as alpha_newton_step)
    if probs.dim() == 1:
        probs = probs.view(1, 1, -1)
        p = p.view(1, 1, -1) if p.dim() == 1 else p.view(1, -1).unsqueeze(1)
        n_mg = n_mg.view(1, 1, 1) if n_mg.dim() == 0 else n_mg
    elif probs.dim() == 2:
        probs = probs.unsqueeze(0)
        p = p.view(1, 1, -1)
        n_mg = n_mg.unsqueeze(0)

    if p.dim() == 2:  # [M, J]
        p = p.unsqueeze(1)  # [M, 1, J]

    # Variance of price under model probs
    p_mean = (probs * p).sum(dim=-1, keepdim=True)  # [M, G, 1]
    p_var = (probs * (p - p_mean) ** 2).sum(dim=-1, keepdim=True)  # [M, G, 1]

    # Observed Fisher information
    H_total = (n_mg * p_var).sum()

    # SE = 1 / sqrt(H)
    se = (1.0 / torch.sqrt(H_total)).item()

    return se


def center_function(
    g_vals: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    Center g(x) to remove location invariance (arbitrary additive constant).

    Multinomial logit identifies utilities only up to a constant per group/market.
    Always center before comparing learned vs true functions.

    Args:
        g_vals: Function values [J] or [G, J] or [M, G, J]
        dim: Dimension to center over (default -1 for products)

    Returns:
        Centered g_vals with mean 0 along specified dimension
    """
    return g_vals - g_vals.mean(dim=dim, keepdim=True)


def get_device(prefer_gpu: bool = False) -> torch.device:
    """
    Get compute device (GPU if available and requested, else CPU).

    Args:
        prefer_gpu: If True and CUDA available, use GPU

    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def move_to_device(*tensors, device: torch.device):
    """
    Move multiple tensors to device in one call.

    Args:
        *tensors: Variable number of tensors or None values
        device: Target device

    Returns:
        Tuple of tensors moved to device (preserving None values)
    """
    return tuple(t.to(device) if t is not None else None for t in tensors)


# ============================================================================
# Nested Logit Utilities
# ============================================================================

def compute_inclusive_values(
    utilities: torch.Tensor,
    nest_ids: torch.Tensor,
    mu: float = 1.0
) -> torch.Tensor:
    """
    Compute inclusive values for nested logit model.

    IV[m,g,n] = (1/mu) log Sum_{j in n} exp(mu*u[m,g,j])

    Args:
        utilities: Product utilities [M, G, J] or [G, J] or [J]
        nest_ids: Nest assignment for each product [J]
        mu: Scale parameter (default 1.0)

    Returns:
        Inclusive values per nest [M, G, N_nests] or [G, N_nests] or [N_nests]
    """
    # Determine dimensions
    original_dims = utilities.dim()

    # Standardize to [M, G, J]
    if original_dims == 1:  # [J]
        utilities = utilities.view(1, 1, -1)
    elif original_dims == 2:  # [G, J]
        utilities = utilities.unsqueeze(0)

    M, G, J = utilities.shape
    N_nests = int(nest_ids.max().item()) + 1

    # Compute IVs per nest
    IVs = torch.zeros(M, G, N_nests, device=utilities.device)

    for n in range(N_nests):
        mask = nest_ids == n
        if mask.sum() > 0:
            u_nest = utilities[:, :, mask]  # [M, G, J_n]
            # LogSumExp for numerical stability
            max_u = u_nest.max(dim=2, keepdim=True).values
            IVs[:, :, n] = (1.0 / mu) * (
                max_u.squeeze(2) + torch.log(torch.exp(mu * (u_nest - max_u)).sum(dim=2))
            )

    # Restore original dimensions
    if original_dims == 1:
        return IVs.squeeze(0).squeeze(0)
    elif original_dims == 2:
        return IVs.squeeze(0)
    else:
        return IVs


def compute_diversion_ratio(
    utilities: torch.Tensor,
    nest_ids: torch.Tensor,
    removed_product: int,
    lambda_nest: float,
    mu: float = 1.0
) -> torch.Tensor:
    """
    Compute diversion ratios when a product is removed.

    Measures where demand goes when product j is removed from the choice set.

    Args:
        utilities: Product utilities [J] (single market/segment)
        nest_ids: Nest assignment [J]
        removed_product: Index of product to remove
        lambda_nest: Nest dissimilarity parameter
        mu: Scale parameter

    Returns:
        Diversion shares to remaining products [J-1]
    """
    J = len(utilities)

    # Compute baseline choice probabilities
    IVs_base = compute_inclusive_values(utilities, nest_ids, mu)
    nest_probs_base = torch.softmax(lambda_nest * IVs_base, dim=0)

    # Within-nest probabilities
    N_nests = len(IVs_base)
    probs_base = torch.zeros(J, device=utilities.device)

    for n in range(N_nests):
        mask = nest_ids == n
        if mask.sum() > 0:
            u_nest = utilities[mask]
            within_probs = torch.softmax(mu * u_nest, dim=0)
            probs_base[mask] = nest_probs_base[n] * within_probs

    # Remove product and recompute
    keep_mask = torch.ones(J, dtype=torch.bool, device=utilities.device)
    keep_mask[removed_product] = False

    u_reduced = utilities[keep_mask]
    nest_ids_reduced = nest_ids[keep_mask]

    IVs_reduced = compute_inclusive_values(u_reduced, nest_ids_reduced, mu)
    nest_probs_reduced = torch.softmax(lambda_nest * IVs_reduced, dim=0)

    probs_reduced = torch.zeros(J-1, device=utilities.device)
    for n in range(N_nests):
        mask = nest_ids_reduced == n
        if mask.sum() > 0:
            u_nest = u_reduced[mask]
            within_probs = torch.softmax(mu * u_nest, dim=0)
            probs_reduced[mask] = nest_probs_reduced[n] * within_probs

    # Diversion ratio: (new_prob - old_prob) / old_prob_removed
    prob_removed = probs_base[removed_product]
    diversion = (probs_reduced - probs_base[keep_mask]) / prob_removed

    return diversion


def compute_own_elasticity(
    alpha: float,
    price: float,
    share: float,
    lambda_nest: float,
    share_in_nest: float
) -> float:
    """
    Compute own-price elasticity in nested logit model.

    eps_{jj} = alpha*p_j*(1/lambda - (1-lambda)/lambda*s_{j|n} - s_j)

    Args:
        alpha: Price coefficient (should be negative)
        price: Product price
        share: Market share s_j
        lambda_nest: Nest dissimilarity parameter
        share_in_nest: Within-nest share s_{j|n}

    Returns:
        Own-price elasticity (negative if alpha < 0)
    """
    elasticity = alpha * price * (
        (1.0 / lambda_nest) -
        ((1.0 - lambda_nest) / lambda_nest) * share_in_nest -
        share
    )
    return elasticity
