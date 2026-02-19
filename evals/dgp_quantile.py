"""
Quantile DGP: Location-Shift Model (Regime C)

DGP Specification:
    X ~ Uniform(-2, 2), d_x = 1
    α*(x) = 1.0 + 0.5·x         (heterogeneous intercept)
    β*(x) = 0.3 + 0.2·x         (heterogeneous treatment effect)
    T = 0.5·x + N(0, 0.5²)      (confounded!)
    Y = α*(x) + β*(x)·T + σ·ε   (σ = 0.5, ε ~ N(0,1))

For any quantile τ:
    Q_τ(Y|T,X) = α*(x) + β*(x)·T + σ·Φ⁻¹(τ)
    α_τ(x) = α*(x) + σ·Φ⁻¹(τ)
    β_τ(x) = β*(x)  (slope is same at all quantiles for location-shift)

Target: E[β*(X)] = 0.3 (independent of τ)

Oracle formulas for smoothed check function (τ, ε):
    score(y, t, θ) = -(τ - σ(u/ε)) · [1, t]   where u = y - α - β·t
    hessian(y, t, θ) = (1/ε)·σ(u/ε)·(1-σ(u/ε))·[[1,t],[t,t²]]
    Λ(x) = E[H(Y,T,θ*) | X=x] via MC integration
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor
from scipy import integrate
from scipy.special import expit
from scipy.stats import norm


@dataclass
class QuantileDGP:
    """Configuration for the quantile DGP."""

    # True parameter functions: α*(x) = A0 + A1·x
    A0: float = 1.0
    A1: float = 0.5

    # β*(x) = B0 + B1·x
    B0: float = 0.3
    B1: float = 0.2

    # Noise scale
    SIGMA: float = 0.5

    # Confounding strength: T = CONFOUND·x + noise
    CONFOUND: float = 0.5
    T_noise_std: float = 0.5

    # Covariate distribution
    X_low: float = -2.0
    X_high: float = 2.0

    def alpha_star(self, x: np.ndarray) -> np.ndarray:
        """True α*(x) = A0 + A1·x."""
        return self.A0 + self.A1 * x

    def beta_star(self, x: np.ndarray) -> np.ndarray:
        """True β*(x) = B0 + B1·x."""
        return self.B0 + self.B1 * x

    def theta_star(self, x: np.ndarray) -> np.ndarray:
        """True θ*(x) = [α*(x), β*(x)]."""
        return np.column_stack([self.alpha_star(x), self.beta_star(x)])

    def theta_star_quantile(self, x: np.ndarray, tau: float = 0.5) -> np.ndarray:
        """
        True quantile-specific θ_τ*(x) = [α_τ*(x), β*(x)].

        α_τ(x) = α*(x) + σ·Φ⁻¹(τ)
        β_τ(x) = β*(x)  (same at all quantiles for location-shift)
        """
        q_shift = self.SIGMA * norm.ppf(tau)
        return np.column_stack([
            self.alpha_star(x) + q_shift,
            self.beta_star(x),
        ])

    def mu_true(self) -> float:
        """
        True target: E[β*(X)] = B0.

        For location-shift model, the QTE equals the ATE at all quantiles.
        Computed via numerical integration over X ~ Uniform(-2, 2).
        """
        def integrand(x):
            beta = self.B0 + self.B1 * x
            return beta * 0.25  # PDF of Uniform(-2, 2) is 1/4

        result, _ = integrate.quad(integrand, self.X_low, self.X_high)
        return result


def generate_quantile_dgp(
    n: int,
    seed: Optional[int] = None,
    dgp: Optional[QuantileDGP] = None,
    tau: float = 0.5,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, float]:
    """
    Generate data from the quantile DGP.

    Args:
        n: Number of observations
        seed: Random seed
        dgp: DGP configuration (default: QuantileDGP())
        tau: Quantile level (only affects theta_true returned, not Y generation)

    Returns:
        Y: (n,) continuous outcomes
        T: (n,) treatments (confounded!)
        X: (n, 1) covariates
        theta_true: (n, 2) true quantile-specific parameters [α_τ*(x), β*(x)]
        mu_true: True target μ* = E[β*(X)]
    """
    if dgp is None:
        dgp = QuantileDGP()

    if seed is not None:
        np.random.seed(seed)

    # Generate X ~ Uniform(-2, 2)
    X = np.random.uniform(dgp.X_low, dgp.X_high, n)

    # True structural parameters
    alpha_true = dgp.alpha_star(X)
    beta_true = dgp.beta_star(X)

    # Generate T = CONFOUND·x + noise [CONFOUNDED!]
    xi = np.random.normal(0, dgp.T_noise_std, n)
    T = dgp.CONFOUND * X + xi

    # Generate Y = α*(x) + β*(x)·T + σ·ε
    eps = np.random.normal(0, 1, n)
    Y = alpha_true + beta_true * T + dgp.SIGMA * eps

    # True quantile-specific theta
    theta_true = dgp.theta_star_quantile(X, tau=tau)

    # Compute true μ*
    mu_true = dgp.mu_true()

    # Convert to tensors
    Y_t = torch.tensor(Y, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    X_t = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    theta_true_t = torch.tensor(theta_true, dtype=torch.float32)

    return Y_t, T_t, X_t, theta_true_t, mu_true


# === ORACLE FORMULAS (for validation) ===

def oracle_score(y: float, t: float, theta: np.ndarray,
                 tau: float = 0.5, eps: float = 0.01) -> np.ndarray:
    """
    Oracle score for smoothed check function.

    ∂ρ̃/∂u = τ - 1 + σ(u/ε)
    ∂u/∂θ = [-1, -t]
    ∂ρ̃/∂θ = (1 - τ - σ(u/ε)) · [1, t]

    where u = y - α - β·t

    Args:
        y: Outcome
        t: Treatment
        theta: [alpha, beta]
        tau: Quantile level
        eps: Smoothing parameter

    Returns:
        [∂ρ̃/∂α, ∂ρ̃/∂β]
    """
    alpha, beta = theta[0], theta[1]
    u = y - alpha - beta * t
    s = expit(u / eps)
    residual = 1 - tau - s
    return np.array([residual, residual * t])


def oracle_hessian(y: float, t: float, theta: np.ndarray,
                   tau: float = 0.5, eps: float = 0.01) -> np.ndarray:
    """
    Oracle Hessian for smoothed check function.

    ∂²ρ̃/∂θ² = (1/ε)·σ(u/ε)·(1-σ(u/ε))·[[1,t],[t,t²]]
    where u = y - α - β·t

    Args:
        y: Outcome
        t: Treatment
        theta: [alpha, beta]
        tau: Quantile level
        eps: Smoothing parameter

    Returns:
        [[∂²ρ̃/∂α², ∂²ρ̃/∂α∂β],
         [∂²ρ̃/∂β∂α, ∂²ρ̃/∂β²]]
    """
    alpha, beta = theta[0], theta[1]
    u = y - alpha - beta * t
    s = expit(u / eps)
    w = s * (1 - s) / eps
    return np.array([
        [w, w * t],
        [w * t, w * t * t],
    ])


def oracle_lambda_conditional(x: float, dgp: QuantileDGP,
                              tau: float = 0.5, eps: float = 0.01,
                              n_mc: int = 50000) -> np.ndarray:
    """
    Oracle Λ(x) = E[H(Y,T,θ*) | X=x] via Monte Carlo integration.

    For QTE, the Hessian depends on BOTH Y and T, so we integrate over
    the joint distribution (T, Y) | X=x.

    T | X ~ N(CONFOUND·x, T_noise_std²)
    Y | T, X ~ N(α*(x) + β*(x)·T, σ²)

    Args:
        x: Covariate value
        dgp: DGP configuration
        tau: Quantile level
        eps: Smoothing parameter
        n_mc: MC samples

    Returns:
        (2, 2) conditional Hessian matrix
    """
    # True parameters at x
    alpha_x = dgp.alpha_star(np.array([x]))[0]
    beta_x = dgp.beta_star(np.array([x]))[0]

    # True quantile-adjusted alpha
    q_shift = dgp.SIGMA * norm.ppf(tau)
    alpha_tau = alpha_x + q_shift

    theta_true = np.array([alpha_tau, beta_x])

    # Sample (T, Y) | X=x
    T_samples = np.random.normal(dgp.CONFOUND * x, dgp.T_noise_std, n_mc)
    eps_samples = np.random.normal(0, 1, n_mc)
    Y_samples = alpha_x + beta_x * T_samples + dgp.SIGMA * eps_samples

    # Compute Hessians and average
    Lambda = np.zeros((2, 2))
    for i in range(n_mc):
        H = oracle_hessian(Y_samples[i], T_samples[i], theta_true,
                           tau=tau, eps=eps)
        Lambda += H
    Lambda /= n_mc

    return Lambda


if __name__ == "__main__":
    # Test DGP generation
    dgp = QuantileDGP()
    print("=== Quantile DGP Configuration ===")
    print(f"α*(x) = {dgp.A0} + {dgp.A1}·x")
    print(f"β*(x) = {dgp.B0} + {dgp.B1}·x")
    print(f"X ~ Uniform({dgp.X_low}, {dgp.X_high})")
    print(f"T = {dgp.CONFOUND}·x + N(0, {dgp.T_noise_std}²)")
    print(f"Y = α*(x) + β*(x)·T + {dgp.SIGMA}·ε")
    print(f"\nTrue μ* = E[β*(X)] = {dgp.mu_true():.6f}")

    # Generate sample
    Y, T, X, theta_true, mu_true = generate_quantile_dgp(n=1000, seed=42)
    print(f"\n=== Sample Statistics (n=1000) ===")
    print(f"Y: mean={Y.mean():.4f}, std={Y.std():.4f}")
    print(f"T: mean={T.mean():.4f}, std={T.std():.4f}")
    print(f"X: mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"α_τ*: mean={theta_true[:, 0].mean():.4f}, std={theta_true[:, 0].std():.4f}")
    print(f"β*: mean={theta_true[:, 1].mean():.4f}, std={theta_true[:, 1].std():.4f}")

    # Test oracle functions
    print(f"\n=== Oracle Functions (single point) ===")
    y_test, t_test = 1.5, 0.5
    theta_test = np.array([1.0, 0.3])

    print(f"Test: y={y_test}, t={t_test}, θ={theta_test}")
    print(f"Score: {oracle_score(y_test, t_test, theta_test)}")
    print(f"Hessian:\n{oracle_hessian(y_test, t_test, theta_test)}")

    # Test conditional Lambda
    print(f"\n=== Oracle Lambda (x=0) ===")
    Lambda_0 = oracle_lambda_conditional(0.0, dgp, n_mc=50000)
    print(f"Λ(0) =\n{Lambda_0}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(Lambda_0)}")
