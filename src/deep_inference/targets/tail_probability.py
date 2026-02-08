"""
Tail probability target: P(Y > c | θ(X), t̃).

For logit (binary, c=0): σ(α + β·t̃) — same as DoseResponse for logit
For Poisson: 1 - PoissonCDF(c; λ) where λ = exp(α + β·t̃)
For linear/Gaussian: 1 - Φ((c - μ) / σ) where μ = α + β·t̃

Useful for risk analysis: what fraction of individuals would exceed threshold c
at treatment level t̃?

References:
    Melnychuk & Feuerriegel (2026, ICLR) — GDR-Learners for distributional effects
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class TailProbability(BaseTarget):
    """
    Target: Tail (Exceedance) Probability.

    μ* = E[P(Y > c | θ(X), t̃)] — average probability outcome exceeds c.

    For logit (binary): P(Y=1|θ,t̃) = σ(α + β·t̃) (threshold c=0 implied)
    For Poisson: P(Y > c | λ) = 1 - Σ_{k=0}^{c} e^{-λ}λ^k/k!
    For linear/Gaussian: P(Y > c | μ, σ) = 1 - Φ((c - μ) / σ)

    References:
        Melnychuk & Feuerriegel (2026, ICLR) — GDR-Learners
    """

    output_dim: int = 1

    def __init__(
        self,
        threshold: float = 0.0,
        model_type: str = "poisson",
        noise_std: float = 1.0,
    ):
        """
        Initialize TailProbability target.

        Args:
            threshold: Threshold c (P(Y > c))
            model_type: Type of model ("logit", "poisson", "linear")
            noise_std: Standard deviation for linear/Gaussian model
        """
        self.threshold = threshold
        self.model_type = model_type
        self.noise_std = noise_std

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute tail probability P(Y > c | θ, t̃).

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,) = [α, β, ...]
            t_tilde: Evaluation point

        Returns:
            Scalar: P(Y > c)
        """
        eta = theta[0] + theta[1] * t_tilde

        if self.model_type == "logit":
            # For binary outcome, P(Y > 0) = P(Y = 1) = σ(η)
            return torch.sigmoid(eta)

        elif self.model_type == "poisson":
            # P(Y > c) = 1 - Σ_{k=0}^{floor(c)} e^{-λ}λ^k/k!
            lam = torch.exp(eta)
            c = int(self.threshold)
            # Compute CDF by summing Poisson PMF terms
            log_lam = torch.log(lam + 1e-30)
            cdf = torch.zeros_like(lam)
            for k in range(c + 1):
                log_pmf = -lam + k * log_lam - torch.lgamma(
                    torch.tensor(k + 1, dtype=theta.dtype, device=theta.device)
                )
                cdf = cdf + torch.exp(log_pmf)
            return 1.0 - cdf

        elif self.model_type == "linear":
            # P(Y > c) = 1 - Φ((c - μ) / σ)
            mu = eta
            z = (self.threshold - mu) / self.noise_std
            normal = torch.distributions.Normal(0, 1)
            return 1.0 - normal.cdf(z)

        else:
            raise ValueError(
                f"Unknown model_type: {self.model_type}. "
                f"Supported: logit, poisson, linear"
            )

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of tail probability w.r.t. theta.

        For logit: same as DoseResponse (∂σ/∂θ = p(1-p)·[1,t̃])
        For linear: ∂/∂θ = φ(z)/σ · [1, t̃]  where φ = normal PDF
        For Poisson: autodiff (CDF derivative is complex)

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
            # ∂P/∂μ = φ(z)/σ, ∂μ/∂α = 1, ∂μ/∂β = t̃
            z = (self.threshold - eta) / self.noise_std
            normal = torch.distributions.Normal(0, 1)
            phi_z = torch.exp(normal.log_prob(z))
            dP_dmu = phi_z / self.noise_std
            jac[0] = dP_dmu
            jac[1] = dP_dmu * t_tilde
            return jac

        # Poisson: autodiff
        return None
