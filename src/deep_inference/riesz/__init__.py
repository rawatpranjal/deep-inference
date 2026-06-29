"""RieszNet automatic-debiasing inference.

Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022), "RieszNet and ForestRiesz"
(ICML). An alternative to the influence-function procedure: the bias correction is
learned by a neural-network Riesz head rather than derived from the model's Hessian.

Public API:
    riesz_inference(Y, T, X, outcome=...) -> InferenceResult
    RieszNet  (the underlying nn.Module)
"""

from .inference import riesz_inference
from .model import SUPPORTED_OUTCOMES, RieszNet

__all__ = ["riesz_inference", "RieszNet", "SUPPORTED_OUTCOMES"]
