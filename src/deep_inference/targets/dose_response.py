"""
Dose-response target: average structural prediction at treatment level t̃.

For logit: E[σ(α(X) + β(X)·t̃)]  = avg purchase probability at price t̃
For Poisson: E[exp(α(X) + β(X)·t̃)] = avg count at exposure t̃
For linear: E[α(X) + β(X)·t̃]      = avg outcome at treatment t̃

This is the most fundamental counterfactual: what would happen on average
if everyone received treatment level t̃?

References:
    Colangelo & Lee (2026, JBES) — Inference for continuous treatments
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class DoseResponse(BaseTarget):
    """
    Target: Average Dose-Response.

    μ* = E[G(θ(X), t̃)] — average predicted outcome at treatment level t̃.

    For logit: G(θ, t) = σ(α + β·t)
    For linear: G(θ, t) = α + β·t
    For poisson: G(θ, t) = exp(α + β·t)

    References:
        Colangelo & Lee (2026, JBES) — Inference for continuous treatments
    """

    output_dim: int = 1

    def __init__(self, model_type: str = "logit"):
        """
        Initialize DoseResponse target.

        Args:
            model_type: Type of model ("logit", "linear", "poisson")
        """
        self.model_type = model_type

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute predicted outcome at treatment level t̃.

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,) = [α, β, ...]
            t_tilde: Evaluation point (treatment level)

        Returns:
            Scalar: predicted outcome G(θ, t̃)
        """
        eta = theta[0] + theta[1] * t_tilde

        if self.model_type == "logit":
            return torch.sigmoid(eta)
        elif self.model_type == "linear":
            return eta
        elif self.model_type == "poisson":
            return torch.exp(eta)
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Supported: logit, linear, poisson"
            )

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of dose-response w.r.t. theta (closed-form).

        For logit: ∂G/∂θ = p(1-p) · [1, t̃]
        For linear: ∂G/∂θ = [1, t̃]
        For poisson: ∂G/∂θ = exp(η) · [1, t̃]

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
            dp = p * (1 - p)
            jac[0] = dp
            jac[1] = dp * t_tilde
            return jac

        elif self.model_type == "linear":
            jac[0] = 1.0
            jac[1] = t_tilde
            return jac

        elif self.model_type == "poisson":
            lam = torch.exp(eta)
            jac[0] = lam
            jac[1] = lam * t_tilde
            return jac

        return None
