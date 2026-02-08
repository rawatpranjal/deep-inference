"""
Consumer Welfare (Expected Consumer Surplus) target.

For binary logit:
    CS = log(1 + exp(α + β·t)) / |β_price|

Based on Small & Rosen (1981) logsum formula: the expected maximum utility
over alternatives, scaled by the marginal utility of income (price coefficient).

References:
    Small & Rosen (1981) — Applied Welfare Economics with Discrete Choice Models
    Dube & Misra (2022, JPE) — Personalized Pricing and Consumer Welfare
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .base import BaseTarget


class ConsumerWelfare(BaseTarget):
    """
    Target: Expected Consumer Surplus from logit demand.

    For binary logit:
        CS = log(1 + exp(V)) / |β_price|

    where V = α + β_price · t is the utility of the good (vs outside option
    normalized to 0).

    Uses softplus for numerical stability: log(1 + exp(x)) = softplus(x).

    References:
        Small & Rosen (1981)
        Dube & Misra (2022, JPE)
    """

    output_dim: int = 1

    def __init__(
        self,
        price_coef_index: int = 1,
    ):
        """
        Initialize ConsumerWelfare target.

        Args:
            price_coef_index: Index of price coefficient in theta
        """
        self.price_coef_index = price_coef_index

    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """
        Compute expected consumer surplus.

        CS = log(1 + exp(V)) / |β_price|

        where V = α + β_price · t_tilde.

        Args:
            x: Covariates (not used directly)
            theta: Parameters (d_theta,) = [α, β_price, ...]
            t_tilde: Evaluation point (price level)

        Returns:
            Scalar: consumer surplus in monetary units
        """
        alpha = theta[0]
        beta_price = theta[self.price_coef_index]

        # Utility at evaluation point
        V = alpha + beta_price * t_tilde

        # Inclusive value (logsum) — softplus for numerical stability
        inclusive_value = F.softplus(V)

        # Scale by |price coefficient| to convert to monetary units
        return inclusive_value / torch.abs(beta_price)

    def jacobian(
        self, x: Tensor, theta: Tensor, t_tilde: Tensor
    ) -> Optional[Tensor]:
        """
        Return None to use autodiff.

        The Jacobian involves derivatives of softplus and |β_price|,
        which autodiff handles correctly.
        """
        return None
