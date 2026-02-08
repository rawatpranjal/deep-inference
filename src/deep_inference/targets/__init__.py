"""
Target functionals for structural inference.

A Target defines what quantity we want to estimate and its derivatives.

H(x, θ; t̃) → scalar or vector target value
H_θ(x, θ; t̃) → Jacobian (gradient if scalar target)
"""

from .base import Target, BaseTarget, CustomTarget, TargetMetadata, target_from_fn
from .average_param import AverageParameter, AverageBeta
from .marginal_effect import AverageMarginalEffect, AME
from .choice_probability import ChoiceProbabilityTarget, MultinomialAME
from .elasticity import Elasticity
from .wtp import WTP
from .welfare import ConsumerWelfare

__all__ = [
    # Protocol and base classes
    "Target",
    "BaseTarget",
    "CustomTarget",
    "TargetMetadata",
    "target_from_fn",
    # Built-in targets
    "AverageParameter",
    "AverageBeta",
    "AverageMarginalEffect",
    "AME",
    # Economic targets
    "Elasticity",
    "WTP",
    "ConsumerWelfare",
    # Multinomial targets
    "ChoiceProbabilityTarget",
    "MultinomialAME",
]
