"""
Combinatorial multi-treatment structural model from DeDL2025.

Y = G(θ(X), T) + ε where T ∈ {0,1}^m (m binary treatments)

Four link functions:
- multiplicative: θ₀ · ∏(1 + θ_k · t_k)
- sigmoid: a/(1+exp(-(θ₀ + Σ θ_k·t_k))) + b
- gen_sigmoid_i: θ_{m+1} · σ(Σ θ_k·t_k)      [no intercept in exponent]
- gen_sigmoid_ii: θ_{m+1} · σ(θ₀ + Σ θ_k·t_k) [most flexible, recommended]

Loss: MSE = (y - G(θ, t))²

References:
    Ye et al. (2025, Management Science) — Debiased Deep Learning for
    Combinatorial Experiments
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseModel


class CombinatorialModel(BaseModel):
    """
    Multi-treatment structural model from DeDL2025.

    T is a binary vector in {0,1}^m encoding treatment combination.
    θ(X) maps covariates to structural parameters via DNN.

    Hessian: Fisher information 2·G_θ·G_θ' (does NOT depend on y).
    Under correct specification E[y - G|X,T] = 0, the G_{θθ}·(y-G)
    term has zero expectation, so the expected Hessian is 2·E[G_θ·G_θ'].

    References:
        Ye et al. (2025, Management Science)
    """

    hessian_depends_on_theta: bool = True   # G_θ depends on θ
    hessian_depends_on_y: bool = False      # Fisher information

    def __init__(
        self,
        n_treatments: int,
        link: str = "gen_sigmoid_ii",
        a: float = 1.0,
        b: float = 0.0,
    ):
        """
        Initialize CombinatorialModel.

        Args:
            n_treatments: Number of binary treatments m
            link: Link function type
                "multiplicative" — θ₀·∏(1+θ_k·t_k), d_θ = m+1
                "sigmoid"        — a/(1+exp(-(θ₀+Σθ_k·t_k)))+b, d_θ = m+1
                "gen_sigmoid_i"  — θ_{m+1}·σ(Σθ_k·t_k), d_θ = m+1
                "gen_sigmoid_ii" — θ_{m+1}·σ(θ₀+Σθ_k·t_k), d_θ = m+2
            a: Scale parameter for sigmoid link
            b: Shift parameter for sigmoid link
        """
        self.n_treatments = n_treatments
        self.link = link
        self.a = a
        self.b = b
        self.theta_dim = self._compute_theta_dim()

    def _compute_theta_dim(self) -> int:
        """Compute dimension of θ based on link function."""
        m = self.n_treatments
        if self.link in ("multiplicative", "sigmoid"):
            return m + 1  # θ₀, θ₁, ..., θ_m
        elif self.link == "gen_sigmoid_i":
            return m + 1  # θ₁, ..., θ_m, θ_{m+1} (no intercept)
        elif self.link == "gen_sigmoid_ii":
            return m + 2  # θ₀, θ₁, ..., θ_m, θ_{m+1}
        else:
            raise ValueError(
                f"Unknown link: {self.link}. "
                f"Supported: multiplicative, sigmoid, gen_sigmoid_i, gen_sigmoid_ii"
            )

    def G(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Structured link function G(θ, t).

        Args:
            theta: Parameters (d_theta,)
            t: Treatment vector (m,) with values in {0, 1}

        Returns:
            Scalar: predicted outcome
        """
        m = self.n_treatments

        if self.link == "multiplicative":
            # G = θ₀ · ∏_{k=1}^{m} (1 + θ_k · t_k)
            product = torch.ones(1, dtype=theta.dtype, device=theta.device)
            for k in range(m):
                product = product * (1 + theta[k + 1] * t[k])
            return theta[0] * product.squeeze()

        elif self.link == "sigmoid":
            # G = a / (1 + exp(-(θ₀ + Σ θ_k·t_k))) + b
            eta = theta[0]
            for k in range(m):
                eta = eta + theta[k + 1] * t[k]
            return self.a * torch.sigmoid(eta) + self.b

        elif self.link == "gen_sigmoid_i":
            # G = θ_{m+1} · σ(Σ_{k=1}^{m} θ_k · t_k)  [no intercept]
            # theta = [θ₁, ..., θ_m, θ_{m+1}]  (0-indexed: theta[0..m-1] = θ_k, theta[m] = θ_{m+1})
            eta = torch.zeros(1, dtype=theta.dtype, device=theta.device)
            for k in range(m):
                eta = eta + theta[k] * t[k]
            return theta[m] * torch.sigmoid(eta.squeeze())

        elif self.link == "gen_sigmoid_ii":
            # G = θ_{m+1} · σ(θ₀ + Σ_{k=1}^{m} θ_k · t_k)
            # theta = [θ₀, θ₁, ..., θ_m, θ_{m+1}]
            eta = theta[0]
            for k in range(m):
                eta = eta + theta[k + 1] * t[k]
            return theta[m + 1] * torch.sigmoid(eta)

        else:
            raise ValueError(f"Unknown link: {self.link}")

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        MSE loss: ℓ = (y - G(θ, t))².

        Args:
            y: Outcome (scalar)
            t: Treatment vector (m,)
            theta: Parameters (d_theta,)

        Returns:
            Scalar loss
        """
        return (y - self.G(theta, t)) ** 2

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """Score via autodiff for generality across link functions."""
        return None

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Optional[Tensor]:
        """
        Fisher information: 2·G_θ·G_θ' (does NOT depend on y).

        Under correct specification, E[(y-G)·G_{θθ}|X,T] = 0, so the
        expected Hessian is 2·E[G_θ·G_θ'|X,T].

        Args:
            y: Outcome — NOT USED (Fisher information)
            t: Treatment vector (m,)
            theta: Parameters (d_theta,)

        Returns:
            (d_theta, d_theta) Hessian matrix
        """
        theta_req = theta.detach().requires_grad_(True)
        g = self.G(theta_req, t)
        g_theta = torch.autograd.grad(g, theta_req, create_graph=False)[0]
        return 2 * torch.outer(g_theta, g_theta)

    def compute_lambda_integral(
        self,
        theta: Tensor,
        t_samples: Tensor,
    ) -> Tensor:
        """
        Compute Λ(x) via Monte Carlo for randomized experiments (Regime A).

        Λ(x) = 2·E[G_θ·G_θ' | X=x] ≈ (2/M) Σ_m G_θ(θ,t_m)·G_θ(θ,t_m)'

        Args:
            theta: (d_theta,) parameter vector for this x
            t_samples: (M, m) treatment sample matrix

        Returns:
            (d_theta, d_theta) Lambda matrix
        """
        M = t_samples.shape[0]
        d = self.theta_dim
        Lambda = torch.zeros(d, d, dtype=theta.dtype, device=theta.device)

        for i in range(M):
            theta_req = theta.detach().requires_grad_(True)
            g = self.G(theta_req, t_samples[i])
            g_theta = torch.autograd.grad(g, theta_req, create_graph=False)[0]
            Lambda += 2 * torch.outer(g_theta, g_theta)

        return Lambda / M


# Convenience alias
Combinatorial = CombinatorialModel
