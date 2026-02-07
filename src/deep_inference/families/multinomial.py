"""
Multinomial logit (conditional logit) family for the legacy structural_dml() API.

Wraps the MultinomialLogitModel for batched operations.

Model: P(Y=j | W, X) = exp(V_ij) / Σ_m exp(V_im)
       V_ij = α_j(W) + x'_ij · β(W)

Note: T must be packed as (n, J*K) with alternative attributes.
      Y is the chosen alternative index (0, 1, ..., J-1).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .base import BaseFamily


class MultinomialLogitFamily(BaseFamily):
    """
    Multinomial logit (conditional logit / McFadden) structural model.

    Model: P[Y=j | W, X] = softmax(V)[j]
           V_ij = α_j(W) + x'_ij · β(W)

    Parameters:
        theta = (α_1, ..., α_{J-1}, β_1, ..., β_K) where
        - α_j(w): alternative-specific intercepts (α_0 = 0)
        - β_k(w): attribute coefficients

    Target options:
        - 'beta': E[β_target_idx(W)] (average attribute coefficient)
        - 'choice_prob': E[P(Y=j | W, X)] (average choice probability)

    Note: The Hessian depends on theta through softmax probabilities,
    so three-way splitting is required (Regime C for observational data).
    """

    def __init__(self, n_alternatives: int = 3, n_attributes: int = 2,
                 target: str = "beta", target_idx: int = 0):
        """
        Initialize MultinomialLogitFamily.

        Args:
            n_alternatives: J, number of choice alternatives
            n_attributes: K, number of alternative-specific attributes
            target: Target functional - 'beta' for attribute coefficient
            target_idx: Which β to target (0-indexed)
        """
        self.J = n_alternatives
        self.K = n_attributes
        self.theta_dim = (self.J - 1) + self.K
        self._target = target
        self._target_idx = target_idx

    def _unpack_batch(self, t: Tensor, theta: Tensor):
        """
        Unpack batched inputs.

        Args:
            t: (n, J*K) packed alternative attributes
            theta: (n, theta_dim)

        Returns:
            x: (n, J, K) alternative attributes
            alphas: (n, J-1) intercepts
            betas: (n, K) coefficients
        """
        n = t.shape[0]
        x = t.reshape(n, self.J, self.K)
        alphas = theta[:, :self.J - 1]
        betas = theta[:, self.J - 1:]
        return x, alphas, betas

    def _utilities_batch(self, x: Tensor, alphas: Tensor, betas: Tensor) -> Tensor:
        """
        Compute utilities for batch.

        Args:
            x: (n, J, K) attributes
            alphas: (n, J-1) intercepts
            betas: (n, K) coefficients

        Returns:
            V: (n, J) utilities
        """
        n = x.shape[0]
        V = torch.zeros(n, self.J, dtype=alphas.dtype, device=alphas.device)
        # V_0 = x_0' β
        V[:, 0] = (x[:, 0, :] * betas).sum(dim=1)
        # V_j = α_j + x_j' β for j >= 1
        for j in range(1, self.J):
            V[:, j] = alphas[:, j - 1] + (x[:, j, :] * betas).sum(dim=1)
        return V

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Categorical cross-entropy loss.

        Args:
            y: (n,) chosen alternative indices (float, will be cast to long)
            t: (n, J*K) packed alternative attributes
            theta: (n, theta_dim) parameters

        Returns:
            (n,) per-observation losses
        """
        x, alphas, betas = self._unpack_batch(t, theta)
        V = self._utilities_batch(x, alphas, betas)
        return F.cross_entropy(V, y.long(), reduction='none')

    def gradient(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form gradient.

        ∂ℓ/∂α_j = p_j - 1{Y=j}  (for j=1,...,J-1)
        ∂ℓ/∂β_k = Σ_j (p_j - 1{Y=j}) x_{jk}

        Args:
            y: (n,) chosen alternatives
            t: (n, J*K) packed attributes
            theta: (n, theta_dim) parameters

        Returns:
            (n, theta_dim) gradient tensor
        """
        x, alphas, betas = self._unpack_batch(t, theta)
        V = self._utilities_batch(x, alphas, betas)
        probs = torch.softmax(V, dim=1)  # (n, J)

        n = y.shape[0]
        y_long = y.long()

        # Residual: p_j - 1{Y=j}
        residual = probs.clone()
        residual[torch.arange(n), y_long] -= 1.0

        # Gradient w.r.t. α (J-1 components, excluding alternative 0)
        grad_alpha = residual[:, 1:]  # (n, J-1)

        # Gradient w.r.t. β: Σ_j residual_j · x_j
        # residual: (n, J), x: (n, J, K) → (n, K)
        grad_beta = torch.einsum('nj,njk->nk', residual, x)

        return torch.cat([grad_alpha, grad_beta], dim=1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Fisher information Hessian (does NOT depend on y).

        Args:
            y: (n,) outcomes (unused)
            t: (n, J*K) packed attributes
            theta: (n, theta_dim) parameters

        Returns:
            (n, theta_dim, theta_dim) Hessian tensor
        """
        x, alphas, betas = self._unpack_batch(t, theta)
        V = self._utilities_batch(x, alphas, betas)
        probs = torch.softmax(V, dim=1)  # (n, J)

        n = y.shape[0]
        d = self.theta_dim
        H = torch.zeros(n, d, d, dtype=theta.dtype, device=theta.device)

        # Probability-weighted mean attributes: x̄_p = Σ_j p_j x_j
        # probs: (n, J), x: (n, J, K) → (n, K)
        x_bar = torch.einsum('nj,njk->nk', probs, x)

        # H_αα block (J-1 × J-1)
        for j in range(self.J - 1):
            for m in range(self.J - 1):
                if j == m:
                    H[:, j, m] = probs[:, j + 1] * (1.0 - probs[:, m + 1])
                else:
                    H[:, j, m] = -probs[:, j + 1] * probs[:, m + 1]

        # H_αβ block (J-1 × K)
        for j in range(self.J - 1):
            # probs[:, j+1] * (x[:, j+1, :] - x_bar) → (n, K)
            diff = x[:, j + 1, :] - x_bar
            H[:, j, self.J - 1:] = probs[:, j + 1].unsqueeze(1) * diff

        # H_βα = H_αβ' (K × J-1)
        H[:, self.J - 1:, :self.J - 1] = H[:, :self.J - 1, self.J - 1:].transpose(1, 2)

        # H_ββ block (K × K): Σ_j p_j (x_j - x̄_p)(x_j - x̄_p)'
        for j in range(self.J):
            diff = x[:, j, :] - x_bar  # (n, K)
            # outer product for each obs: (n, K, 1) @ (n, 1, K) → (n, K, K)
            outer = diff.unsqueeze(2) * diff.unsqueeze(1)
            H[:, self.J - 1:, self.J - 1:] += probs[:, j].unsqueeze(1).unsqueeze(2) * outer

        return H

    def hessian_depends_on_theta(self) -> bool:
        """
        Multinomial logit Hessian depends on theta through softmax probabilities.

        This means three-way splitting is required.
        """
        return True

    def per_obs_target(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Per-observation target h(θ, t).

        For target='beta': h(θ) = β_{target_idx}

        Args:
            theta: (n, theta_dim) parameters
            t: (n, J*K) packed attributes

        Returns:
            (n,) per-observation target values
        """
        if self._target == "beta":
            return theta[:, self.J - 1 + self._target_idx]
        raise ValueError(f"Unknown target: {self._target}")

    def per_obs_target_gradient(self, theta: Tensor, t: Tensor) -> Tensor:
        """
        Gradient of per-observation target: ∂h/∂θ.

        For target='beta': unit vector at position J-1+target_idx.

        Args:
            theta: (n, theta_dim) parameters
            t: (n, J*K) packed attributes

        Returns:
            (n, theta_dim) gradient
        """
        n = theta.shape[0]
        grad = torch.zeros(n, self.theta_dim, dtype=theta.dtype, device=theta.device)
        if self._target == "beta":
            grad[:, self.J - 1 + self._target_idx] = 1.0
        return grad

    def default_target(self, x: Tensor, theta: Tensor) -> Tensor:
        """
        Default target: E[β_{target_idx}(W)].

        Args:
            x: (n, d_w) individual characteristics (unused)
            theta: (n, theta_dim) parameters

        Returns:
            Scalar target value
        """
        return theta[:, self.J - 1 + self._target_idx].mean()
