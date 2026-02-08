"""
Conditional variance target: Var(Y | θ(X), t̃).

For logit (Bernoulli): Var = p(1-p) where p = σ(α + β·t̃)
For Poisson: Var = λ = exp(α + β·t̃)
For linear/Gaussian: Var = σ² (constant if not modeled)

The model-implied variance of outcomes captures heterogeneity in risk/uncertainty
across the covariate distribution.

References:
    Melnychuk & Feuerriegel (2026, ICLR) — GDR-Learners for distributional effects
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class ConditionalVariance(BaseTarget):
    """
    Target: Conditional Variance of Outcome.

    μ* = E[Var(Y | θ(X), t̃)] — average model-implied variance.

    For logit: Var(Y|θ,t̃) = p(1-p) where p = σ(α + β·t̃)
    For Poisson: Var(Y|θ,t̃) = λ = exp(α + β·t̃)
    For linear: Var(Y|θ,t̃) = σ² (if modeled as theta[2])

    References:
        Melnychuk & Feuerriegel (2026, ICLR) — GDR-Learners
    """

    output_dim: int = 1

    def __init__(self, model_type: str = "logit"):
        """
        Initialize ConditionalVariance target.

        Args:
            model_type: Type of model ("logit", "poisson")
        """
        self.model_type = model_type

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute conditional variance Var(Y | θ, t̃).

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,) = [α, β, ...]
            t_tilde: Evaluation point

        Returns:
            Scalar: Var(Y | θ, t̃)
        """
        eta = theta[0] + theta[1] * t_tilde

        if self.model_type == "logit":
            p = torch.sigmoid(eta)
            return p * (1 - p)

        elif self.model_type == "poisson":
            return torch.exp(eta)

        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Supported: logit, poisson"
            )

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of conditional variance w.r.t. theta (closed-form).

        For logit: ∂[p(1-p)]/∂θ = p(1-p)(1-2p) · [1, t̃]
        For poisson: ∂exp(η)/∂θ = exp(η) · [1, t̃]

        Args:
            x: Covariates
            theta: Parameters
            t_tilde: Evaluation point

        Returns:
            (d_theta,) Jacobian or None for autodiff
        """
        eta = theta[0] + theta[1] * t_tilde
        d_theta = theta.shape[0]
        jac = torch.zeros(d_theta, dtype=theta.dtype, device=theta.device)

        if self.model_type == "logit":
            p = torch.sigmoid(eta)
            # d[p(1-p)]/dη = p(1-p)(1-2p)
            d_var = p * (1 - p) * (1 - 2 * p)
            jac[0] = d_var
            jac[1] = d_var * t_tilde
            return jac

        elif self.model_type == "poisson":
            lam = torch.exp(eta)
            jac[0] = lam
            jac[1] = lam * t_tilde
            return jac

        return None
