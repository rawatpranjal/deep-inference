"""RieszNet model: a multitasking neural net for automatic debiasing.

Implements Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022), "RieszNet and
ForestRiesz" (arXiv 2110.03031, ICML). One shared trunk feeds two heads:

  - a regression head g(Z) for the outcome model, and
  - a linear Riesz head a(Z) for the Riesz representer (Section 3, Lemma 3.1).

Training minimizes the Equation (5) loss

    loss = RegLoss(g) + lambda1 * RieszLoss(a) + lambda2 * TargetedReg(eps)

where the Riesz loss E[a^2 - 2 m(a)] makes a(Z) the representer without an analytic
formula (Section 2.1 / Equation 3), and the targeted-regularization term fits an
unpenalized scalar eps in g_tilde(Z) = g(Z) + eps * a(Z) so that the first-order
condition forces E_n[(Y - g_tilde) a] = 0 (Section 3, around Equation 5). That zeroed
debiasing term is what buys approximately nominal coverage.

This module is a faithful port of the validated prototype in exploration/spike.py; the
architecture, loss, and moment match the paper. The fit schedule and the doubly-robust
moment used for inference live in `inference.py`.
"""

from __future__ import annotations

import torch
from torch import nn

# Equation (5) weights and the L2 penalty on net weights (paper Appendix A.1).
LAMBDA1, LAMBDA2, L2 = 0.1, 1.0, 1e-3

# Outcomes whose regression head and loss are implemented here.
SUPPORTED_OUTCOMES = ("linear", "logit", "poisson", "gamma", "negbin", "probit")


def _torch_Phi(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, for the probit regression head."""
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


class RieszNet(nn.Module):
    """Shared trunk plus a linear Riesz head and a regression head.

    Architecture follows Section 3 and Appendix A.1 of the paper:
      - shared trunk f1(Z): 3 hidden layers, width 200, ELU activations,
      - Riesz head: a linear map on the shared representation, a(Z) = <f1(Z), beta>,
      - regression head: 2 hidden layers, width 100, ELU, producing the linear index,
      - eps: a single unpenalized parameter for targeted regularization.

    The input is the concatenation of the (scalar) treatment t and the covariates x.
    """

    def __init__(self, d_x: int, k_width: int = 200, g_width: int = 100):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_x + 1, k_width),
            nn.ELU(),
            nn.Linear(k_width, k_width),
            nn.ELU(),
            nn.Linear(k_width, k_width),
            nn.ELU(),
        )
        self.riesz_head = nn.Linear(k_width, 1)  # linear on the shared representation
        self.reg_head = nn.Sequential(
            nn.Linear(k_width, g_width),
            nn.ELU(),
            nn.Linear(g_width, g_width),
            nn.ELU(),
            nn.Linear(g_width, 1),
        )
        self.eps = nn.Parameter(torch.zeros(1))  # unpenalized targeted-reg parameter

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Return (g_raw, a) where g_raw is the regression linear index and a the representer."""
        rep = self.trunk(torch.cat([t, x], dim=1))
        return self.reg_head(rep).squeeze(-1), self.riesz_head(rep).squeeze(-1)


def outcome_g_reg(g_raw, y, outcome: str, nb_dispersion: float = 3.0):
    """Map the regression linear index to the mean g(Z) and return (g, regression_loss).

    Each branch is the negative log-likelihood for that outcome, matching the GLM
    families used elsewhere in the package.
    """
    if outcome == "logit":
        return torch.sigmoid(g_raw), nn.functional.binary_cross_entropy_with_logits(
            g_raw, y
        )
    if outcome == "poisson":
        g = torch.exp(g_raw)
        return g, (g - y * g_raw).mean()  # Poisson NLL exp(eta) - y*eta
    if outcome == "gamma":
        g = torch.exp(g_raw)
        return g, (y * torch.exp(-g_raw) + g_raw).mean()  # Gamma NLL y/mu + log mu
    if outcome == "negbin":
        r = nb_dispersion
        g = torch.exp(g_raw)
        return g, ((r + y) * torch.log(r + g) - y * g_raw).mean()  # NB2 NLL
    if outcome == "probit":
        p = _torch_Phi(g_raw).clamp(1e-6, 1 - 1e-6)
        return p, -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    if outcome == "linear":
        return g_raw, ((g_raw - y) ** 2).mean()  # linear / squared error
    raise ValueError(
        f"Unsupported outcome {outcome!r}. Supported: {', '.join(SUPPORTED_OUTCOMES)}."
    )


def riesz_loss_term(net, t, x, ones, zeros):
    """The Riesz loss E[a^2 - 2 m(a)] for the binary-treatment ATE moment.

    For the ATE the moment of the representer is m(a) = a(1, X) - a(0, X), so minimizing
    a^2 - 2 (a1 - a0) drives a(Z) to the representer (Equation 3, Section 2.1).
    """
    _, a_obs = net(t, x)
    _, a1 = net(ones, x)
    _, a0 = net(zeros, x)
    return (a_obs**2 - 2.0 * (a1 - a0)).mean()


def combined_loss(net, t, x, y, ones, zeros, outcome: str, nb_dispersion: float = 3.0):
    """Equation (5) objective: RegLoss + LAMBDA1 * RieszLoss + LAMBDA2 * TargetedReg.

    The targeted-regularization term fluctuates the fitted mean by eps * a and pays a
    square loss (the canonical Poisson deviance on the log scale for counts), whose
    first-order condition zeroes the debiasing correction.
    """
    g_raw, a_obs = net(t, x)
    g_obs, reg = outcome_g_reg(g_raw, y, outcome, nb_dispersion)
    _, a1 = net(ones, x)
    _, a0 = net(zeros, x)
    riesz = (a_obs**2 - 2.0 * (a1 - a0)).mean()
    eps = torch.clamp(net.eps, -2.0, 2.0)  # clamp the targeted-reg parameter
    if outcome == "poisson":
        # Canonical Poisson targeting: fluctuate the log-mean eta -> eta + eps*a and
        # target via the Poisson deviance, whose FOC is E_n[a*(Y - lambda_tilde)] = 0.
        lt = g_raw + eps * a_obs
        tmle = (torch.exp(lt) - y * lt).mean()
    else:
        # Paper's generic squared-loss targeting on the mean scale.
        g_tilde = g_obs + eps * a_obs
        tmle = ((y - g_tilde) ** 2).mean()
    return reg + LAMBDA1 * riesz + LAMBDA2 * tmle
