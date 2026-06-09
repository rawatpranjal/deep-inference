"""
Saturated 2x2 difference-in-differences structural model (heterogeneous / neural).

A neural network maps covariates X to the four coefficients of the saturated DiD
regression, allowing every coefficient -- in particular the DiD interaction tau --
to vary with X:

    Y = alpha(X) + gamma(X) G + lambda(X) P + tau(X) (G P) + eps

The per-observation "treatment" t packs the design columns t = [G, P, G*P], and the
design vector with intercept is W = [1, G, P, G*P]. The structural parameters are

    theta = [alpha, gamma, lambda, tau]   (theta_dim = 4)

so the DiD effect is tau = theta[3], targeted via AverageParameter(param_index=3).

Squared-error loss => Hessian = W W', which is constant in theta and Y. This is the
linear regime (Regime B): Lambda(x) = E[W W' | X = x] is estimated analytically and
two-way cross-fitting suffices.

This is the StructuralModel used by inference(model='did', target='tau') and by the
top-level convenience wrapper deep_inference.did_2x2_nn(). It is distinct from the
closed-form estimator in deep_inference/did.py (did_2x2), which handles the exact
homogeneous case.
"""

import torch
from torch import Tensor

from .base import BaseModel


class DiDModel(BaseModel):
    """
    Saturated 2x2 DiD model: E[Y | t, X] = W'theta(X), W = [1, G, P, G*P].

    Key property: Hessian = W W' does not depend on theta (Regime B), so
    Lambda(x) = E[W W' | X] can be estimated independently of theta_hat.
    """

    theta_dim: int = 4
    hessian_depends_on_theta: bool = False  # KEY: enables Regime B
    hessian_depends_on_y: bool = False

    @staticmethod
    def _design(t: Tensor) -> Tensor:
        """Build W = [1, G, P, G*P] from per-observation t = [G, P, G*P]."""
        one = torch.ones_like(t[0])
        return torch.stack([one, t[0], t[1], t[2]])

    def loss(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Squared error loss (single observation).

        Args:
            y: Outcome (scalar)
            t: Design columns (3,) = [G, P, G*P]
            theta: Parameters (4,) = [alpha, gamma, lambda, tau]

        Returns:
            Scalar loss: (y - W'theta)^2 / 2
        """
        W = self._design(t)
        pred = torch.dot(W, theta)
        return 0.5 * (y - pred) ** 2

    def score(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form score: d l / d theta = (pred - y) * W.

        Args:
            y: Outcome (scalar)
            t: Design columns (3,)
            theta: Parameters (4,)

        Returns:
            (4,) gradient vector
        """
        W = self._design(t)
        pred = torch.dot(W, theta)
        return (pred - y) * W

    def hessian(self, y: Tensor, t: Tensor, theta: Tensor) -> Tensor:
        """
        Closed-form Hessian: d^2 l / d theta^2 = W W' (constant in theta and y).

        Args:
            y: Outcome (scalar) - NOT USED
            t: Design columns (3,)
            theta: Parameters (4,) - NOT USED

        Returns:
            (4, 4) Hessian matrix
        """
        W = self._design(t)
        return torch.outer(W, W)


# Convenience alias
DiD = DiDModel
