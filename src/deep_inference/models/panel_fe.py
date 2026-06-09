"""
Within-transformed two-way fixed-effects panel difference-in-differences model.

After residualizing the outcome Y and treatment D = G*Post by unit and time fixed
effects (see deep_inference.utils.residualize_fixed_effects), additive unit/time
effects are absorbed and the model reduces to a single-parameter regression through
the origin:

    Ytilde_it = Dtilde_it * tau(X_it) + eps_it

A neural network maps covariates X to the heterogeneous treatment effect
tau(X) = theta_0(X). The "treatment" t passed to this model is the residualized
Dtilde (a scalar per observation). theta_dim = 1 and there is NO intercept (the FE
within transformation removed it), so the Regime-B analytic Lambda must be built
without a constant: Lambda(x) = E[Dtilde^2 | X] (1x1). This is signalled by
analytic_intercept = False.

Squared-error loss => Hessian = Dtilde^2 is constant in theta and Y (Regime B,
two-way cross-fitting). The influence-function standard error is heteroskedasticity
-robust, which also covers the binary (linear probability model) case. Used by
inference(model='did_fe', target='fe_effect') and the entry point
did(..., unit=..., time=...).
"""

import torch
from torch import Tensor

from .base import BaseModel


class FEPanelDiDModel(BaseModel):
    """
    Intercept-free FE panel DiD: E[Ytilde | Dtilde, X] = Dtilde * tau(X).

    theta = [tau], theta_dim = 1. Hessian = Dtilde^2 (constant) -> Regime B.
    """

    theta_dim: int = 1
    hessian_depends_on_theta: bool = False  # KEY: enables Regime B
    hessian_depends_on_y: bool = False
    analytic_intercept: bool = False  # within transform absorbed the intercept

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Squared error: 0.5 (ytilde - dtilde * tau)^2. t = Dtilde (scalar)."""
        pred = t * theta[0]
        return 0.5 * (y - pred) ** 2

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Closed-form score: d l / d tau = (pred - y) * dtilde -> (1,)."""
        pred = t * theta[0]
        return ((pred - y) * t).reshape(1)

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """Closed-form Hessian: d^2 l / d tau^2 = dtilde^2 -> (1, 1)."""
        return (t * t).reshape(1, 1)


# Convenience alias
FEPanelDiD = FEPanelDiDModel
