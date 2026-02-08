"""
Willingness To Pay (WTP) target.

WTP = -β_attribute / β_price

Measures how much price can increase while maintaining the same probability
of purchase when an attribute improves by 1 unit. Standard object in discrete
choice demand analysis.

References:
    Dube & Misra (2022, JPE) — Personalized Pricing and Consumer Welfare
"""

import torch
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class WTP(BaseTarget):
    """
    Target: Willingness To Pay.

    WTP = -θ_attribute / θ_price

    The ratio of coefficients gives the marginal rate of substitution
    between an attribute and price, i.e., how much a consumer would pay
    for a one-unit improvement in the attribute.

    Delta method standard errors are computed automatically via autodiff.

    References:
        Dube & Misra (2022, JPE) — Personalized Pricing and Consumer Welfare
    """

    output_dim: int = 1

    def __init__(
        self,
        attribute_index: int = 1,
        price_index: int = 2,
    ):
        """
        Initialize WTP target.

        Args:
            attribute_index: Index of attribute coefficient in theta
            price_index: Index of price coefficient in theta
        """
        self.attribute_index = attribute_index
        self.price_index = price_index

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute WTP = -θ_attribute / θ_price.

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,)
            t_tilde: Evaluation point (not used for WTP)

        Returns:
            Scalar: willingness to pay
        """
        beta_attr = theta[self.attribute_index]
        beta_price = theta[self.price_index]

        # Guard against division by zero
        if torch.abs(beta_price) < 1e-8:
            return torch.tensor(float('nan'), dtype=theta.dtype, device=theta.device)

        return -beta_attr / beta_price

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Compute Jacobian of WTP w.r.t. theta (closed-form delta method).

        ∂WTP/∂θ_attr = -1/θ_price
        ∂WTP/∂θ_price = θ_attr / θ_price²

        Returns None if price coefficient is near zero.
        """
        beta_attr = theta[self.attribute_index]
        beta_price = theta[self.price_index]

        if torch.abs(beta_price) < 1e-8:
            return None

        d_theta = theta.shape[0]
        jac = torch.zeros(d_theta, dtype=theta.dtype, device=theta.device)

        # ∂WTP/∂θ_attr = -1/θ_price
        jac[self.attribute_index] = -1.0 / beta_price

        # ∂WTP/∂θ_price = β_attr / β_price²
        jac[self.price_index] = beta_attr / (beta_price ** 2)

        return jac
