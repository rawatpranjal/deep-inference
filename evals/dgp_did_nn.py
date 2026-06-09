"""
Oracle DGP for the heterogeneous neural 2x2 DiD model (eval_14).

Canonical repeated cross-section: group G and period P are assigned independently of
the covariates X (only the OUTCOME means depend on X). All four saturated coefficients
are heterogeneous in X[0], so the DiD effect tau(X) varies across individuals:

    alpha(X)  = A0 + A1 * X[0]
    gamma(X)  = G0 + G1 * X[0]
    lambda(X) = L0 + L1 * X[0]
    tau(X)    = T0 + T1 * X[0]            <- the DiD interaction coefficient

    mu(X,G,P) = alpha + gamma*G + lambda*P + tau*(G*P)
    Y         = mu + SIGMA * eps,   eps ~ N(0,1)

Target:  mu_true = E[tau(X)] = T0   (since E[X[0]] = 0).

Because (G,P) are independent of X, the conditional design second moment is constant:
    Lambda(x) = E[W W' | X] = E[W W'],   W = [1, G, P, G*P],
so Regime B with aggregate analytic Lambda is exactly correct.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DiDNNDGP:
    # alpha(X) = A0 + A1 X0
    A0: float = 1.0
    A1: float = 0.5
    # gamma(X) = G0 + G1 X0
    G0: float = 0.3
    G1: float = 0.2
    # lambda(X) = L0 + L1 X0
    L0: float = 0.4
    L1: float = 0.3
    # tau(X) = T0 + T1 X0   (DiD effect)
    T0: float = 0.5
    T1: float = 0.4
    SIGMA: float = 1.0
    d_x: int = 3
    p_group: float = 0.5
    p_post: float = 0.5

    def theta_star(self, X: np.ndarray) -> np.ndarray:
        """Return (n, 4) true [alpha, gamma, lambda, tau] at covariates X."""
        x0 = X[:, 0]
        return np.column_stack([
            self.A0 + self.A1 * x0,
            self.G0 + self.G1 * x0,
            self.L0 + self.L1 * x0,
            self.T0 + self.T1 * x0,
        ])

    def mu_true(self) -> float:
        """Target E[tau(X)] = T0 (E[X0] = 0)."""
        return self.T0

    def generate(self, n: int, seed: int):
        """Draw (Y, G, P, X). G, P independent of X (canonical repeated cross section)."""
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, self.d_x))
        G = (rng.random(n) < self.p_group).astype(np.float64)
        P = (rng.random(n) < self.p_post).astype(np.float64)
        theta = self.theta_star(X)
        mu = theta[:, 0] + theta[:, 1] * G + theta[:, 2] * P + theta[:, 3] * (G * P)
        Y = mu + self.SIGMA * rng.standard_normal(n)
        return Y, G, P, X

    def oracle_lambda(self, n_mc: int = 2_000_000, seed: int = 0) -> np.ndarray:
        """Oracle Lambda = E[W W'] (constant in X since (G,P) ⟂ X), via large-sample MC."""
        rng = np.random.default_rng(seed)
        G = (rng.random(n_mc) < self.p_group).astype(np.float64)
        P = (rng.random(n_mc) < self.p_post).astype(np.float64)
        W = np.column_stack([np.ones(n_mc), G, P, G * P])
        return (W[:, :, None] * W[:, None, :]).mean(0)
