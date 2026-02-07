"""
Choice probability and multinomial AME targets.

For the conditional logit model:
    P(Y=j | W, X) = exp(V_ij) / Σ_m exp(V_im)
    V_ij = α_j(W) + x'_ij · β(W)

Targets:
    - ChoiceProbabilityTarget: H = P(Y=j | W, X) for a specific alternative j
    - MultinomialAME: H = ∂P(Y=j)/∂x_{jk} = β_k · P_j · (1 - P_j)
      (at the evaluation point t_tilde)

Reference: Hetzenecker & Osterhaus (2024), Section 2.
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class ChoiceProbabilityTarget(BaseTarget):
    """
    Target: H = P(Y=j | W, X) — choice probability for alternative j.

    At evaluation point t_tilde (packed alternative attributes):
        H(w, θ; t̃) = softmax(V)[j]
        where V_j = α_j + x̃'_j · β

    Jacobian (closed-form via softmax derivative):
        ∂P_j/∂α_m = P_{m+1}(δ_{j,m+1} - P_j)  for m in 0..J-2
        ∂P_j/∂β_k = P_j · (x̃_{jk} - Σ_l P_l x̃_{lk})
    """

    output_dim: int = 1

    def __init__(self, alternative: int, n_alternatives: int, n_attributes: int):
        """
        Args:
            alternative: j, which alternative's probability to target (0-indexed)
            n_alternatives: J, total number of alternatives
            n_attributes: K, number of alternative-specific attributes
        """
        self.j = alternative
        self.J = n_alternatives
        self.K = n_attributes
        self.theta_dim = (self.J - 1) + self.K

    def _compute_probs(self, theta: Tensor, t_tilde: Tensor):
        """Compute softmax probabilities and unpack."""
        x = t_tilde.reshape(self.J, self.K)
        alphas = theta[:self.J - 1]
        betas = theta[self.J - 1:]

        V = torch.zeros(self.J, dtype=theta.dtype, device=theta.device)
        V[0] = x[0] @ betas
        V[1:] = alphas + x[1:] @ betas

        probs = torch.softmax(V, dim=0)
        return probs, x, alphas, betas

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute choice probability P(Y=j).

        Args:
            x: Covariates W (not used directly)
            theta: (theta_dim,) parameters
            t_tilde: (J*K,) evaluation point (packed attributes)

        Returns:
            Scalar: P(Y=j)
        """
        probs, _, _, _ = self._compute_probs(theta, t_tilde)
        return probs[self.j]

    def jacobian(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Closed-form Jacobian of P_j w.r.t. θ.

        ∂P_j/∂α_m = P_{m+1}(δ_{j,m+1} - P_j)
        ∂P_j/∂β_k = P_j(x̃_{jk} - x̄_pk)   where x̄_p = Σ_l P_l x̃_l

        Args:
            x: Covariates W (not used)
            theta: (theta_dim,) parameters
            t_tilde: (J*K,) evaluation point

        Returns:
            (theta_dim,) gradient vector
        """
        probs, x_mat, _, _ = self._compute_probs(theta, t_tilde)
        p_j = probs[self.j]

        jac = torch.zeros(self.theta_dim, dtype=theta.dtype, device=theta.device)

        # ∂P_j/∂α_m for m=0..J-2 (α_{m+1})
        for m in range(self.J - 1):
            delta = 1.0 if (m + 1) == self.j else 0.0
            jac[m] = probs[m + 1] * (delta - p_j)

        # x̄_p = Σ_l P_l x_l
        x_bar = torch.zeros(self.K, dtype=theta.dtype, device=theta.device)
        for l in range(self.J):
            x_bar = x_bar + probs[l] * x_mat[l]

        # ∂P_j/∂β_k = P_j(x_{jk} - x̄_pk)
        jac[self.J - 1:] = p_j * (x_mat[self.j] - x_bar)

        return jac


class MultinomialAME(BaseTarget):
    """
    Target: Average Marginal Effect for multinomial logit.

    H = ∂P(Y=j)/∂x_{jk} evaluated at t_tilde

    For the conditional logit:
        ∂P_j/∂x_{jk} = β_k · P_j · (1 - P_j)

    This is the marginal effect of attribute k on the probability of choosing
    alternative j, which generalizes the binary logit AME formula.

    Jacobian is closed-form via chain rule through softmax.
    """

    output_dim: int = 1

    def __init__(self, alternative: int, attribute: int,
                 n_alternatives: int, n_attributes: int):
        """
        Args:
            alternative: j, which alternative's probability
            attribute: k, which attribute's marginal effect
            n_alternatives: J, total alternatives
            n_attributes: K, total attributes
        """
        self.j = alternative
        self.k = attribute
        self.J = n_alternatives
        self.K = n_attributes
        self.theta_dim = (self.J - 1) + self.K

    def _compute_probs(self, theta: Tensor, t_tilde: Tensor):
        """Compute probabilities and unpack."""
        x = t_tilde.reshape(self.J, self.K)
        alphas = theta[:self.J - 1]
        betas = theta[self.J - 1:]

        V = torch.zeros(self.J, dtype=theta.dtype, device=theta.device)
        V[0] = x[0] @ betas
        V[1:] = alphas + x[1:] @ betas

        probs = torch.softmax(V, dim=0)
        return probs, x, alphas, betas

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute AME: β_k · P_j · (1 - P_j).

        Args:
            x: Covariates W (not used)
            theta: (theta_dim,) parameters
            t_tilde: (J*K,) evaluation point

        Returns:
            Scalar: ∂P_j/∂x_{jk}
        """
        probs, _, _, betas = self._compute_probs(theta, t_tilde)
        p_j = probs[self.j]
        beta_k = betas[self.k]
        return beta_k * p_j * (1.0 - p_j)

    def jacobian(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Optional[Tensor]:
        """
        Closed-form Jacobian of the AME target.

        H = β_k · P_j · (1 - P_j)

        ∂H/∂θ_i = β_k · (1 - 2P_j) · ∂P_j/∂θ_i     (for non-β_k components)
        ∂H/∂β_k = P_j(1-P_j) + β_k · (1-2P_j) · ∂P_j/∂β_k

        Falls back to autodiff if computation is complex.
        """
        # Return None to use autodiff — the chain rule is complex for the
        # cross-alternative effects. Autodiff handles this correctly.
        return None
