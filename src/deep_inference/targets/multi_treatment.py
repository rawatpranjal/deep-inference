"""
Multi-treatment ATE target for combinatorial experiments.

μ(t) = E[G(θ(X), t) - G(θ(X), t₀)] for treatment combination t vs control t₀.

This is the average treatment effect of applying treatment combination t
relative to the control t₀ (default: all zeros).

References:
    Ye et al. (2025, Management Science) — Debiased Deep Learning for
    Combinatorial Experiments (DeDL2025, Proposition 2)
"""

import torch
from torch import Tensor
from typing import Optional, List

from .base import BaseTarget


class MultiTreatmentATE(BaseTarget):
    """
    Target: Average Treatment Effect for combinatorial experiments.

    μ(t) = E[G(θ(X), t) - G(θ(X), t₀)]

    where G is the structural link function from CombinatorialModel,
    t is a treatment combination vector, and t₀ is the control.

    The influence function Jacobian H_θ = ∂G/∂θ(θ,t) - ∂G/∂θ(θ,t₀)
    follows directly from DeDL2025 Proposition 2.

    References:
        Ye et al. (2025, Management Science)
    """

    output_dim: int = 1

    def __init__(
        self,
        model,
        treatment: List[int],
        control: Optional[List[int]] = None,
    ):
        """
        Initialize MultiTreatmentATE target.

        Args:
            model: CombinatorialModel instance (provides G function)
            treatment: Treatment combination vector, e.g. [1, 0, 1]
            control: Control combination vector (default: all zeros)
        """
        self.model = model
        self.treatment = torch.tensor(treatment, dtype=torch.float32)
        if control is not None:
            self.control = torch.tensor(control, dtype=torch.float32)
        else:
            self.control = torch.zeros(model.n_treatments, dtype=torch.float32)

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute ATE: G(θ, t) - G(θ, t₀).

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,)
            t_tilde: Evaluation point (not used — treatment/control set at init)

        Returns:
            Scalar: treatment effect
        """
        t_dev = self.treatment.to(dtype=theta.dtype, device=theta.device)
        c_dev = self.control.to(dtype=theta.dtype, device=theta.device)
        return self.model.G(theta, t_dev) - self.model.G(theta, c_dev)

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """Autodiff — G_θ differences across link functions."""
        return None
