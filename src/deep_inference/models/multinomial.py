"""
Conditional logit (McFadden) multinomial choice model.

Model (Hetzenecker & Osterhaus 2024, Section 2):
    V_ij = α_j(W) + x'_ij · β(W)     for j = 1, ..., J
    P(Y=j | W, X) = exp(V_ij) / Σ_m exp(V_im)

    Normalization: α_0 = 0 (alternative 0 is reference)
    θ = [α_1, ..., α_{J-1}, β_1, ..., β_K]

Data encoding:
    W = individual characteristics (NN input → θ(W))
    X = alternative-specific attributes, packed as T: (J*K,)
    Y = chosen alternative index (0, 1, ..., J-1)

Loss: Categorical cross-entropy: -log P(Y=y)
Score: (p_j - 1{Y=j}) stacked for α and β components
Hessian: Fisher information (does NOT depend on Y)
    - Enables Regime A under randomization
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseModel


class MultinomialLogitModel(BaseModel):
    """
    Conditional logit (McFadden) with J alternatives.

    P(Y=j | W, X) = exp(V_ij) / Σ_m exp(V_im)
    V_ij = α_j(W) + x'_ij · β(W)

    α_0 = 0 (normalization), θ = [α_1,...,α_{J-1}, β_1,...,β_K]
    """

    hessian_depends_on_theta: bool = True   # Depends on p_j via softmax
    hessian_depends_on_y: bool = False      # Fisher information form

    def __init__(self, n_alternatives: int = 3, n_attributes: int = 2,
                 eps: float = 1e-7):
        """
        Initialize multinomial logit model.

        Args:
            n_alternatives: J, number of choice alternatives
            n_attributes: K, number of alternative-specific attributes
            eps: Numerical stability constant
        """
        self.J = n_alternatives
        self.K = n_attributes
        self.theta_dim = (self.J - 1) + self.K
        self.eps = eps

    def _unpack(self, t_packed: Tensor, theta: Tensor):
        """
        Unpack treatment and parameters.

        Args:
            t_packed: (J*K,) packed alternative attributes
            theta: (theta_dim,) = [α_1,...,α_{J-1}, β_1,...,β_K]

        Returns:
            x: (J, K) alternative attribute matrix
            alphas: (J-1,) alternative-specific intercepts
            betas: (K,) attribute coefficients
        """
        x = t_packed.reshape(self.J, self.K)
        alphas = theta[:self.J - 1]
        betas = theta[self.J - 1:]
        return x, alphas, betas

    def _utilities(self, x: Tensor, alphas: Tensor, betas: Tensor) -> Tensor:
        """
        Compute utilities V_j = α_j + x'_j · β (with V_0 = 0).

        Args:
            x: (J, K) alternative attributes
            alphas: (J-1,) intercepts (α_1,...,α_{J-1})
            betas: (K,) attribute coefficients

        Returns:
            V: (J,) utility vector
        """
        V = torch.zeros(self.J, dtype=alphas.dtype, device=alphas.device)
        # V_0 = 0 (reference), V_j = α_j + x_j' β for j >= 1
        V[1:] = alphas + x[1:] @ betas
        # V_0 still gets attribute contribution: x_0' β
        V[0] = x[0] @ betas
        return V

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Categorical cross-entropy: -log P(Y=y).

        Args:
            y: Scalar (chosen alternative index, 0 to J-1)
            t: (J*K,) packed alternative attributes
            theta: (theta_dim,) parameters

        Returns:
            Scalar loss
        """
        x, alphas, betas = self._unpack(t, theta)
        V = self._utilities(x, alphas, betas)
        log_probs = V - torch.logsumexp(V, dim=0)
        return -log_probs[int(y)]

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form score: ∂ℓ/∂θ.

        For α_j (j >= 1): ∂ℓ/∂α_j = p_j - 1{Y=j}
        For β_k: ∂ℓ/∂β_k = Σ_j (p_j - 1{Y=j}) · x_{jk}

        Args:
            y: Chosen alternative (scalar)
            t: (J*K,) packed attributes
            theta: (theta_dim,) parameters

        Returns:
            (theta_dim,) gradient vector
        """
        x, alphas, betas = self._unpack(t, theta)
        V = self._utilities(x, alphas, betas)
        probs = torch.softmax(V, dim=0)

        # Residual: p_j - 1{Y=j}
        residual = probs.clone()
        y_idx = int(y)
        residual[y_idx] = residual[y_idx] - 1.0

        # Gradient w.r.t. α_j (j=1,...,J-1)
        grad_alpha = residual[1:]  # (J-1,)

        # Gradient w.r.t. β: Σ_j residual_j · x_j
        grad_beta = torch.zeros(self.K, dtype=theta.dtype, device=theta.device)
        for j in range(self.J):
            grad_beta = grad_beta + residual[j] * x[j]

        return torch.cat([grad_alpha, grad_beta])

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Fisher information (does NOT depend on y).

        H_αα[j,m] = p_{j+1}(δ_{jm} - p_{m+1})   for j,m in 0..J-2
        H_αβ[j,:] = p_{j+1}(x_{j+1} - x̄_p)
        H_ββ = Σ_j p_j (x_j - x̄_p)(x_j - x̄_p)'

        where x̄_p = Σ_j p_j x_j

        Args:
            y: Outcome - NOT USED (Hessian is Fisher information)
            t: (J*K,) packed attributes
            theta: (theta_dim,) parameters

        Returns:
            (theta_dim, theta_dim) Hessian matrix
        """
        x, alphas, betas = self._unpack(t, theta)
        V = self._utilities(x, alphas, betas)
        probs = torch.softmax(V, dim=0)

        d = self.theta_dim
        H = torch.zeros(d, d, dtype=theta.dtype, device=theta.device)

        # Probability-weighted mean attributes
        # x_bar = Σ_j p_j x_j
        x_bar = torch.zeros(self.K, dtype=theta.dtype, device=theta.device)
        for j in range(self.J):
            x_bar = x_bar + probs[j] * x[j]

        # H_αα block (J-1 × J-1): indexed by j,m in {1,...,J-1}
        for j in range(self.J - 1):
            for m in range(self.J - 1):
                delta = 1.0 if j == m else 0.0
                H[j, m] = probs[j + 1] * (delta - probs[m + 1])

        # H_αβ block (J-1 × K)
        for j in range(self.J - 1):
            H[j, self.J - 1:] = probs[j + 1] * (x[j + 1] - x_bar)

        # H_βα = H_αβ' (K × J-1)
        H[self.J - 1:, :self.J - 1] = H[:self.J - 1, self.J - 1:].T

        # H_ββ block (K × K): Σ_j p_j (x_j - x̄_p)(x_j - x̄_p)'
        for j in range(self.J):
            diff = x[j] - x_bar
            H[self.J - 1:, self.J - 1:] = (
                H[self.J - 1:, self.J - 1:] + probs[j] * torch.outer(diff, diff)
            )

        return H

    def compute_lambda_integral(
        self,
        theta: Tensor,
        t_samples: Tensor,
    ) -> Tensor:
        """
        Compute Λ(w) via Monte Carlo for randomized experiments (Regime A).

        Λ(w) = E[H(y, T, θ) | W=w] ≈ (1/M) Σ_m H(y, t_m, θ)

        Since Hessian doesn't depend on y, we just average Hessians over T samples.

        Args:
            theta: (theta_dim,) parameter vector for this individual
            t_samples: (M, J*K) samples of alternative attributes

        Returns:
            (theta_dim, theta_dim) Lambda matrix
        """
        M = t_samples.shape[0]
        d = self.theta_dim
        Lambda = torch.zeros(d, d, dtype=theta.dtype, device=theta.device)

        # Dummy y (Hessian doesn't depend on it)
        dummy_y = torch.tensor(0.0, dtype=theta.dtype, device=theta.device)

        for m in range(M):
            Lambda = Lambda + self.hessian(dummy_y, t_samples[m], theta)

        return Lambda / M


# Convenience alias
MultinomialLogit = MultinomialLogitModel
