"""
Price elasticity of demand target.

For logit model: η = (1-p)·β·t where p = σ(α + β·t)
For log-link models (Poisson, Gamma, NegBin): η = β·t

This captures the percentage change in demand for a percentage change in price,
which is the key object in personalized pricing (Dube & Misra, 2022).
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class Elasticity(BaseTarget):
    """
    Target: Price Elasticity of Demand.

    For logit: η = (1-p)·β·t where p = σ(α + β·t)
    For log-link (poisson/gamma/negbin): η = β·t

    This is the elasticity of the outcome w.r.t. treatment,
    evaluated at t_tilde.

    References:
        Dube & Misra (2022, JPE) — Personalized Pricing and Consumer Welfare
    """

    output_dim: int = 1

    def __init__(
        self,
        model_type: str = "logit",
    ):
        """
        Initialize Elasticity target.

        Args:
            model_type: Type of model ("logit", "poisson", "gamma", "negbin")
        """
        self.model_type = model_type

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute elasticity at evaluation point.

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,) = [α, β, ...]
            t_tilde: Evaluation point (price level)

        Returns:
            Scalar: elasticity at t_tilde
        """
        alpha, beta = theta[0], theta[1]

        if self.model_type == "logit":
            # η = (1 - σ(α + β·t)) · β · t
            eta = alpha + beta * t_tilde
            p = torch.sigmoid(eta)
            return (1 - p) * beta * t_tilde

        elif self.model_type in ("poisson", "gamma", "negbin"):
            # Log-link: η = β · t
            return beta * t_tilde

        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Supported: logit, poisson, gamma, negbin"
            )

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of elasticity w.r.t. theta.

        For logit with θ = (α, β):
            ∂η/∂α = -p(1-p)·β·t
            ∂η/∂β = (1-p)·t - p(1-p)·β·t²

        For log-link with θ = (α, β):
            ∂η/∂α = 0
            ∂η/∂β = t

        Args:
            x: Covariates
            theta: Parameters
            t_tilde: Evaluation point

        Returns:
            (d_theta,) Jacobian or None for autodiff
        """
        alpha, beta = theta[0], theta[1]
        d_theta = theta.shape[0]
        jac = torch.zeros(d_theta, dtype=theta.dtype, device=theta.device)

        if self.model_type == "logit":
            eta = alpha + beta * t_tilde
            p = torch.sigmoid(eta)
            pp = p * (1 - p)  # p(1-p)

            # ∂η/∂α = -p(1-p)·β·t
            jac[0] = -pp * beta * t_tilde

            # ∂η/∂β = (1-p)·t + (-p(1-p))·t·β·t = (1-p)·t - p(1-p)·β·t²
            jac[1] = (1 - p) * t_tilde - pp * beta * t_tilde ** 2

            return jac

        elif self.model_type in ("poisson", "gamma", "negbin"):
            # ∂η/∂α = 0
            jac[0] = 0.0
            # ∂η/∂β = t
            jac[1] = t_tilde

            return jac

        else:
            return None
