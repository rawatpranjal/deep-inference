"""
Multinomial Logit DGP: Heterogeneous Conditional Logit (Regime C)

Synthetic multinomial choice data modeled after Hetzenecker & Osterhaus (2024).

DGP Specification:
    J = 3 alternatives, K = 2 attributes, d_w = 3 individual characteristics

    W ~ N(0, I_{d_w})  (individual characteristics → NN input)
    X_ij ~ N(0, 1)     (alternative-specific attributes, iid)

    True parameter functions (heterogeneous in W[0]):
        α_0*(W) = 0  (reference alternative, normalized)
        α_1*(W) = 0.5 + 0.2 * W[0]
        α_2*(W) = -0.3 - 0.1 * W[0]

        β_1*(W) = -0.8 - 0.2 * W[0]
        β_2*(W) = 0.5 + 0.1 * W[0]

    θ*(W) = [α_1, α_2, β_1, β_2]  (dim = 4)

    V_ij = α_j*(W) + X'_ij · β*(W)
    P(Y=j | W, X) = exp(V_ij) / Σ_m exp(V_im)
    Y ~ Categorical(P)

Target: μ* = E[β_1*(W)] = -0.8  (since E[W] = 0)

Oracle functions:
    - oracle_score(): closed-form gradient
    - oracle_hessian(): Fisher information matrix
    - oracle_lambda_mc(): E[H | W=w] via Monte Carlo over X
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch import Tensor
from scipy.special import softmax


@dataclass
class MultinomialLogitDGP:
    """Configuration for the multinomial logit DGP."""

    J: int = 3   # Number of alternatives
    K: int = 2   # Number of attributes
    d_w: int = 3  # Dimension of individual characteristics

    # Alpha intercepts: α_j(W) = a0_j + a1_j * W[0]
    # α_0 = 0 (normalized)
    a0: List[float] = field(default_factory=lambda: [0.0, 0.5, -0.3])
    a1: List[float] = field(default_factory=lambda: [0.0, 0.2, -0.1])

    # Beta coefficients: β_k(W) = b0_k + b1_k * W[0]
    b0: List[float] = field(default_factory=lambda: [-0.8, 0.5])
    b1: List[float] = field(default_factory=lambda: [-0.2, 0.1])

    # Target: which beta to target (0-indexed)
    target_beta_idx: int = 0

    def alpha_star(self, W: np.ndarray) -> np.ndarray:
        """
        True α_j*(W) for j=0,...,J-1.

        Args:
            W: (n, d_w) individual characteristics

        Returns:
            (n, J) alpha values (α_0 = 0 always)
        """
        n = W.shape[0]
        alphas = np.zeros((n, self.J))
        for j in range(self.J):
            alphas[:, j] = self.a0[j] + self.a1[j] * W[:, 0]
        return alphas

    def beta_star(self, W: np.ndarray) -> np.ndarray:
        """
        True β_k*(W) for k=0,...,K-1.

        Args:
            W: (n, d_w)

        Returns:
            (n, K) beta values
        """
        n = W.shape[0]
        betas = np.zeros((n, self.K))
        for k in range(self.K):
            betas[:, k] = self.b0[k] + self.b1[k] * W[:, 0]
        return betas

    def theta_star(self, W: np.ndarray) -> np.ndarray:
        """
        True θ*(W) = [α_1,...,α_{J-1}, β_1,...,β_K].

        Note: α_0 is excluded (normalized to 0).

        Args:
            W: (n, d_w)

        Returns:
            (n, theta_dim) where theta_dim = (J-1) + K
        """
        alphas = self.alpha_star(W)  # (n, J)
        betas = self.beta_star(W)    # (n, K)
        # Exclude α_0 (index 0)
        return np.column_stack([alphas[:, 1:], betas])

    def utilities(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute utilities V_ij = α_j*(W) + X'_ij · β*(W).

        Args:
            W: (n, d_w)
            X: (n, J, K) alternative attributes

        Returns:
            V: (n, J) utilities
        """
        alphas = self.alpha_star(W)  # (n, J)
        betas = self.beta_star(W)    # (n, K)

        # V_ij = α_j + Σ_k X_{ijk} β_k
        V = alphas.copy()
        for j in range(self.J):
            V[:, j] += np.sum(X[:, j, :] * betas, axis=1)
        return V

    def choice_probs(self, W: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute P(Y=j | W, X) = softmax(V).

        Args:
            W: (n, d_w)
            X: (n, J, K)

        Returns:
            (n, J) choice probabilities
        """
        V = self.utilities(W, X)
        return softmax(V, axis=1)

    def mu_true(self) -> float:
        """
        True target: E[β_{target_idx}*(W)] = b0[target_idx].

        Since W ~ N(0, I) and β_k(W) = b0_k + b1_k * W[0],
        E[β_k(W)] = b0_k because E[W[0]] = 0.
        """
        return self.b0[self.target_beta_idx]

    def theta_dim(self) -> int:
        """Dimension of parameter vector."""
        return (self.J - 1) + self.K


def generate_multinomial_dgp(
    n: int,
    seed: Optional[int] = None,
    dgp: Optional[MultinomialLogitDGP] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    Generate data from the multinomial logit DGP.

    Args:
        n: Number of observations
        seed: Random seed
        dgp: DGP configuration

    Returns:
        Y: (n,) chosen alternative (float, values 0 to J-1)
        T: (n, J*K) packed alternative attributes
        W: (n, d_w) individual characteristics
        theta_true: (n, theta_dim) true parameters
        mu_true: True target value
    """
    if dgp is None:
        dgp = MultinomialLogitDGP()

    if seed is not None:
        np.random.seed(seed)

    # Generate individual characteristics W ~ N(0, I)
    W = np.random.normal(0, 1, (n, dgp.d_w))

    # Generate alternative attributes X ~ N(0, 1)
    X = np.random.normal(0, 1, (n, dgp.J, dgp.K))

    # Compute choice probabilities
    probs = dgp.choice_probs(W, X)

    # Sample choices Y ~ Categorical(probs)
    Y = np.zeros(n)
    for i in range(n):
        Y[i] = np.random.choice(dgp.J, p=probs[i])

    # True parameters
    theta_true = dgp.theta_star(W)

    # Pack X: (n, J, K) → (n, J*K)
    T_packed = X.reshape(n, -1)

    # True target
    mu_true = dgp.mu_true()

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T_packed, dtype=torch.float32)
    W_t = torch.tensor(W, dtype=torch.float32)
    theta_true_t = torch.tensor(theta_true, dtype=torch.float32)

    return Y_t, T_t, W_t, theta_true_t, mu_true


# === ORACLE FORMULAS (for validation) ===

def oracle_score(y: int, x: np.ndarray, theta: np.ndarray,
                 J: int = 3, K: int = 2) -> np.ndarray:
    """
    Oracle score: ∂ℓ/∂θ for multinomial logit loss.

    ∂ℓ/∂α_j = p_j - 1{Y=j}  (for j=1,...,J-1)
    ∂ℓ/∂β_k = Σ_j (p_j - 1{Y=j}) x_{jk}

    Args:
        y: Chosen alternative (0 to J-1)
        x: (J, K) alternative attributes
        theta: [α_1,...,α_{J-1}, β_1,...,β_K]
        J: Number of alternatives
        K: Number of attributes

    Returns:
        (theta_dim,) gradient
    """
    alphas = theta[:J - 1]
    betas = theta[J - 1:]

    # Utilities
    V = np.zeros(J)
    V[0] = x[0] @ betas
    for j in range(1, J):
        V[j] = alphas[j - 1] + x[j] @ betas

    # Softmax probabilities
    probs = softmax(V)

    # Residuals
    residual = probs.copy()
    residual[y] -= 1.0

    # Score components
    grad_alpha = residual[1:]
    grad_beta = np.zeros(K)
    for j in range(J):
        grad_beta += residual[j] * x[j]

    return np.concatenate([grad_alpha, grad_beta])


def oracle_hessian(x: np.ndarray, theta: np.ndarray,
                   J: int = 3, K: int = 2) -> np.ndarray:
    """
    Oracle Fisher information Hessian (does NOT depend on y).

    H_αα[j,m] = p_{j+1}(δ_{jm} - p_{m+1})
    H_αβ[j,k] = p_{j+1}(x_{j+1,k} - x̄_pk)
    H_ββ[k,l] = Σ_j p_j(x_{jk} - x̄_pk)(x_{jl} - x̄_pl)

    Args:
        x: (J, K) alternative attributes
        theta: [α_1,...,α_{J-1}, β_1,...,β_K]
        J: Number of alternatives
        K: Number of attributes

    Returns:
        (theta_dim, theta_dim) Hessian matrix
    """
    alphas = theta[:J - 1]
    betas = theta[J - 1:]

    # Utilities
    V = np.zeros(J)
    V[0] = x[0] @ betas
    for j in range(1, J):
        V[j] = alphas[j - 1] + x[j] @ betas

    probs = softmax(V)
    d = (J - 1) + K
    H = np.zeros((d, d))

    # x̄_p = Σ_j p_j x_j
    x_bar = np.zeros(K)
    for j in range(J):
        x_bar += probs[j] * x[j]

    # H_αα block
    for j in range(J - 1):
        for m in range(J - 1):
            delta = 1.0 if j == m else 0.0
            H[j, m] = probs[j + 1] * (delta - probs[m + 1])

    # H_αβ block
    for j in range(J - 1):
        H[j, J - 1:] = probs[j + 1] * (x[j + 1] - x_bar)

    # H_βα = H_αβ'
    H[J - 1:, :J - 1] = H[:J - 1, J - 1:].T

    # H_ββ block
    for j in range(J):
        diff = x[j] - x_bar
        H[J - 1:, J - 1:] += probs[j] * np.outer(diff, diff)

    return H


def oracle_lambda_mc(w: np.ndarray, theta: np.ndarray,
                     J: int = 3, K: int = 2,
                     n_mc: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Oracle Lambda via Monte Carlo: E[H(X, θ) | W=w].

    Since X is independent of W, this is just E_X[H(X, θ)].

    Args:
        w: (d_w,) individual characteristics (not used since X⊥W)
        theta: (theta_dim,) true parameters for this individual
        J, K: Dimensions
        n_mc: MC samples
        seed: Random seed

    Returns:
        (theta_dim, theta_dim) Lambda matrix
    """
    rng = np.random.RandomState(seed)
    d = (J - 1) + K
    Lambda = np.zeros((d, d))

    for _ in range(n_mc):
        x = rng.normal(0, 1, (J, K))
        Lambda += oracle_hessian(x, theta, J, K)

    return Lambda / n_mc
