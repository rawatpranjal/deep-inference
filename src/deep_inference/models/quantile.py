"""
Quantile regression model for structural estimation.

Q_τ(Y|T,X) = α(X) + β(X) * T

Loss: Smoothed check function (tilted Huber)
  ρ̃_τ(u) = τ·u + ε·log(1 + exp(-u/ε))
  where u = y - α - β·t

Score: -(τ - σ(u/ε)) · [1, t]
Hessian: (1/ε)·σ(u/ε)·(1-σ(u/ε))·[[1,t],[t,t²]]

ALWAYS Regime C (observational):
- hessian_depends_on_theta = True (via u = y - α - β·t)
- hessian_depends_on_y = True (via u = y - α - β·t)
→ Must ESTIMATE Λ with 3-way split
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseModel


class QuantileModel(BaseModel):
    """
    Quantile regression model via smoothed check function.

    Model: Q_τ(Y|T,X) = α(X) + β(X) * T

    Key property: Hessian depends on BOTH theta AND y (through residual u).
    - Always Regime C (3-way split, estimate Λ)
    - At τ=0.5 with Gaussian errors, matches linear model (median = mean)
    """

    theta_dim: int = 2
    hessian_depends_on_theta: bool = True   # Depends on u = y - α - β·t
    hessian_depends_on_y: bool = True       # Depends on u = y - α - β·t

    def __init__(self, tau: float = 0.5, smooth_eps: float = 0.01):
        """
        Initialize quantile model.

        Args:
            tau: Quantile level in (0, 1). Default 0.5 = median regression.
            smooth_eps: Smoothing parameter for check function.
                       Smaller = closer to true quantile loss but harder to optimize.
        """
        self.tau = tau
        self.smooth_eps = smooth_eps

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Smoothed check function loss (single observation).

        ρ̃_τ(u) = τ·u + ε·log(1 + exp(-u/ε))

        where u = y - α - β·t

        Args:
            y: Outcome (continuous)
            t: Treatment (scalar)
            theta: Parameters (2,) = [α, β]

        Returns:
            Scalar loss
        """
        u = y - theta[0] - theta[1] * t
        # Use logaddexp for numerical stability: log(1+exp(x)) = logaddexp(0, x)
        return self.tau * u + self.smooth_eps * torch.logaddexp(
            torch.tensor(0.0, dtype=theta.dtype, device=theta.device),
            -u / self.smooth_eps,
        )

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form score (gradient of loss).

        ∂ρ̃/∂u = τ - 1 + σ(u/ε)
        ∂u/∂θ = [-1, -t]
        ∂ρ̃/∂θ = (1 - τ - σ(u/ε)) · [1, t]

        Args:
            y: Outcome (continuous)
            t: Treatment (scalar)
            theta: Parameters (2,)

        Returns:
            (2,) gradient vector
        """
        u = y - theta[0] - theta[1] * t
        s = torch.sigmoid(u / self.smooth_eps)
        residual = 1 - self.tau - s
        return torch.stack([residual, residual * t])

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian (depends on BOTH theta AND y).

        ∂²ρ̃/∂θ² = (1/ε)·σ(u/ε)·(1-σ(u/ε))·[[1,t],[t,t²]]

        Args:
            y: Outcome (continuous)
            t: Treatment (scalar)
            theta: Parameters (2,)

        Returns:
            (2, 2) Hessian matrix
        """
        u = y - theta[0] - theta[1] * t
        s = torch.sigmoid(u / self.smooth_eps)
        w = s * (1 - s) / self.smooth_eps

        H = torch.zeros(2, 2, dtype=theta.dtype, device=theta.device)
        H[0, 0] = w
        H[0, 1] = w * t
        H[1, 0] = w * t
        H[1, 1] = w * t * t
        return H


# Convenience alias
Quantile = QuantileModel
