"""
Profit (expected revenue) target: t̃ · G(θ(X), t̃).

For logit demand: E[t̃ · σ(α(X) + β(X)·t̃)] = avg revenue at price t̃
For linear:       E[t̃ · (α(X) + β(X)·t̃)]   = avg revenue at price t̃
For poisson:      E[t̃ · exp(α(X) + β(X)·t̃)] = avg revenue at exposure t̃

Profit = price × purchase probability. This is the central object in personalized
pricing: find t̃* = argmax E[t̃ · P(buy | θ(X), t̃)].

References:
    Dube & Misra (2023, JPE) — Personalized Pricing and Consumer Welfare
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class Profit(BaseTarget):
    """
    Target: Expected Revenue / Profit.

    μ* = E[t̃ · G(θ(X), t̃)] — average revenue per consumer at price t̃.

    For logit demand: g(θ) = t̃ · σ(α + β·t̃)
    For linear:       g(θ) = t̃ · (α + β·t̃)
    For poisson:      g(θ) = t̃ · exp(α + β·t̃)

    References:
        Dube & Misra (2023, JPE) — Personalized Pricing and Consumer Welfare
    """

    output_dim: int = 1

    def __init__(self, model_type: str = "logit"):
        """
        Initialize Profit target.

        Args:
            model_type: Type of model ("logit", "linear", "poisson")
        """
        self.model_type = model_type

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute revenue at treatment/price level t̃.

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,) = [α, β, ...]
            t_tilde: Evaluation point (price level)

        Returns:
            Scalar: revenue = t̃ · G(θ, t̃)
        """
        eta = theta[0] + theta[1] * t_tilde

        if self.model_type == "logit":
            return t_tilde * torch.sigmoid(eta)
        elif self.model_type == "linear":
            return t_tilde * eta
        elif self.model_type == "poisson":
            return t_tilde * torch.exp(eta)
        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Supported: logit, linear, poisson"
            )

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of profit w.r.t. theta (closed-form).

        For logit: ∂[t̃·σ]/∂θ = t̃·p(1-p)·[1, t̃]
        For linear: ∂[t̃·(α+β·t̃)]/∂θ = t̃·[1, t̃]
        For poisson: ∂[t̃·exp(η)]/∂θ = t̃·exp(η)·[1, t̃]

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
            jac[0] = t_tilde * dp
            jac[1] = t_tilde * dp * t_tilde
            return jac

        elif self.model_type == "linear":
            jac[0] = t_tilde
            jac[1] = t_tilde * t_tilde
            return jac

        elif self.model_type == "poisson":
            lam = torch.exp(eta)
            jac[0] = t_tilde * lam
            jac[1] = t_tilde * lam * t_tilde
            return jac

        return None
