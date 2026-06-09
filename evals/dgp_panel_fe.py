"""
Oracle DGP for the two-way fixed-effects panel DiD model (eval_15).

Balanced panel: N units x T periods. Unit i belongs to the treated group with prob
0.5 (G_i); the last half of periods are post (Post_t). Treatment D_it = G_i * Post_t.
The treatment effect is heterogeneous in a covariate: tau(X_it) = T0 + T1 * X0.

Continuous outcome:
    Y_it = alpha_i + lambda_t + tau(X_it) * D_it + SIGMA * eps,   eps ~ N(0,1)

Binary outcome (linear probability model) uses a SMALLER, interior effect so the
implied probabilities never clip (clipping would attenuate the linear effect and bias
the LPM). It therefore has its own target:
    P_it = P_BASE + a_i + l_t + tauB(X_it) * D_it,   tauB(X) = TB0 + TB1 * X0
    Y_it ~ Bernoulli(P_it)

Targets:  continuous E[tau(X)] = T0;  binary E[tauB(X)] = TB0  (since E[X0]=0).

(G, Post) are assigned independently of X, so after the within transformation the
design second moment E[Dtilde^2 | X] = E[Dtilde^2] is constant in X (oracle Lambda).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class PanelFEDGP:
    N: int = 400          # units
    T: int = 6            # periods (post = last T//2)
    d_x: int = 3
    # continuous effect tau(X) = T0 + T1 X0
    T0: float = 0.5
    T1: float = 0.6
    SIGMA: float = 0.5
    ALPHA_SD: float = 1.0
    LAMBDA_SD: float = 0.5
    # binary (LPM) effect tauB(X) = TB0 + TB1 X0 — small + interior, no clipping
    TB0: float = 0.15
    TB1: float = 0.10
    P_BASE: float = 0.45
    A_SD: float = 0.05
    L_SD: float = 0.05

    def mu_true_continuous(self) -> float:
        return self.T0

    def mu_true_binary(self) -> float:
        return self.TB0

    def _panel_skeleton(self, rng):
        unit = np.repeat(np.arange(self.N), self.T)
        time = np.tile(np.arange(self.T), self.N)
        G = (rng.random(self.N) < 0.5).astype(np.float64)[unit]
        Post = (time >= self.T // 2).astype(np.float64)
        D = G * Post
        X = rng.standard_normal((self.N * self.T, self.d_x))
        return unit, time, D, X

    def generate_continuous(self, seed: int):
        rng = np.random.default_rng(seed)
        unit, time, D, X = self._panel_skeleton(rng)
        tau = self.T0 + self.T1 * X[:, 0]
        alpha = (self.ALPHA_SD * rng.standard_normal(self.N))[unit]
        lam = (self.LAMBDA_SD * rng.standard_normal(self.T))[time]
        Y = alpha + lam + tau * D + self.SIGMA * rng.standard_normal(len(D))
        return Y, D, X, unit, time, tau

    def generate_binary(self, seed: int):
        rng = np.random.default_rng(seed)
        unit, time, D, X = self._panel_skeleton(rng)
        tau = self.TB0 + self.TB1 * X[:, 0]
        a = (self.A_SD * rng.standard_normal(self.N))[unit]
        l = (self.L_SD * rng.standard_normal(self.T))[time]
        P = self.P_BASE + a + l + tau * D
        # interior by construction; clip only as a numerical safety net
        P = np.clip(P, 1e-4, 1 - 1e-4)
        Y = (rng.random(len(D)) < P).astype(np.float64)
        return Y, D, X, unit, time, tau

    def oracle_lambda(self, seed: int = 0, n_rep: int = 200) -> float:
        """Oracle E[Dtilde^2] from many panels (Dtilde ⟂ X => Lambda constant)."""
        from deep_inference.utils import residualize_fixed_effects
        vals = []
        for s in range(n_rep):
            rng = np.random.default_rng(seed + s)
            unit, time, D, _ = self._panel_skeleton(rng)
            Dt = residualize_fixed_effects(D, unit, time)
            vals.append(float((Dt ** 2).mean()))
        return float(np.mean(vals))
