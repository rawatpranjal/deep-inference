"""
eval_13_did: Monte Carlo coverage for the closed-form 2x2 DiD estimator (did_2x2).

The estimator is deterministic and matches saturated-OLS HC0 to machine precision
(see src/deep_inference/tests/test_did.py). This eval is the frequentist firewall:
it confirms the IF-based 95% CI actually covers the true DiD effect at ~95% in
repeated sampling, with calibrated SE and negligible bias.

DGP: repeated cross-section. Each observation draws G ~ Bern(0.5), T ~ Bern(0.5),
then Y ~ N(mu_{G,T}, sigma_{G,T}^2). Heteroskedastic across cells so HC0 matters.

    mu_00=1.0  mu_01=1.5  mu_10=2.0  mu_11=3.1   -> beta* = 3.1 - 2.0 - 1.5 + 1.0 = 0.6

Run:
    python3 -m evals.eval_13_did 2>&1 | tee evals/reports/eval_13_did_$(date +%Y%m%d_%H%M%S).txt
"""

import sys
import time

import numpy as np

sys.path.insert(0, "/Users/pranjal/deepest/src")

from deep_inference import did  # noqa: E402
from evals.common.metrics import validate_coverage_run, format_validation_table  # noqa: E402


# ── DGP config ──
CELL_MEANS = {(0, 0): 1.0, (0, 1): 1.5, (1, 0): 2.0, (1, 1): 3.1}
CELL_SIGMAS = {(0, 0): 1.2, (0, 1): 1.0, (1, 0): 1.4, (1, 1): 1.8}
BETA_STAR = (
    CELL_MEANS[(1, 1)] - CELL_MEANS[(1, 0)] - CELL_MEANS[(0, 1)] + CELL_MEANS[(0, 0)]
)

# ── MC config ──
M = 500
N = 2000
P_GROUP = 0.5
P_POST = 0.5


def simulate(n: int, rng: np.random.Generator):
    """Draw one repeated-cross-section sample."""
    G = (rng.random(n) < P_GROUP).astype(np.float64)
    P = (rng.random(n) < P_POST).astype(np.float64)
    mu = np.empty(n)
    sd = np.empty(n)
    for (g, t), m in CELL_MEANS.items():
        idx = (G == g) & (P == t)
        mu[idx] = m
        sd[idx] = CELL_SIGMAS[(g, t)]
    Y = mu + sd * rng.standard_normal(n)
    return Y, G, P


def main():
    print("=" * 78)
    print("eval_13_did: 2x2 DiD closed-form estimator — Monte Carlo coverage")
    print("=" * 78)
    print(f"True cell means : {CELL_MEANS}")
    print(f"Cell sigmas     : {CELL_SIGMAS}")
    print(f"beta*           : {BETA_STAR:.6f}")
    print(f"M replications  : {M}")
    print(f"n per rep       : {N}")
    print(f"P(group=1)      : {P_GROUP},  P(post=1): {P_POST}")
    print("-" * 78)

    t0 = time.time()
    betas = np.empty(M)
    ses = np.empty(M)
    covered = np.zeros(M, dtype=bool)
    zscores = np.empty(M)

    for m in range(M):
        rng = np.random.default_rng(m + 1)
        Y, G, P = simulate(N, rng)
        r = did(Y, G, P)
        betas[m] = r.mu_hat
        ses[m] = r.se
        covered[m] = (r.ci_lower <= BETA_STAR) and (BETA_STAR <= r.ci_upper)
        zscores[m] = (r.mu_hat - BETA_STAR) / r.se

    elapsed = time.time() - t0

    coverage = float(covered.mean())
    emp_se = float(betas.std(ddof=1))
    mean_se = float(ses.mean())
    se_ratio = emp_se / mean_se
    bias = float(betas.mean() - BETA_STAR)
    z_mean = float(zscores.mean())
    z_std = float(zscores.std(ddof=1))

    print("RESULTS (raw)")
    print(f"  mean(beta_hat)   : {betas.mean():.6f}")
    print(f"  beta*            : {BETA_STAR:.6f}")
    print(f"  bias             : {bias:+.6f}")
    print(f"  empirical SE     : {emp_se:.6f}   (sd of beta_hat across reps)")
    print(f"  mean est. SE     : {mean_se:.6f}   (mean of IF/HC0 SE)")
    print(f"  SE ratio         : {se_ratio:.4f}   (emp_SE / est_SE)")
    print(f"  coverage (95% CI): {coverage*100:.1f}%   ({covered.sum()}/{M})")
    print(f"  z mean           : {z_mean:+.4f}")
    print(f"  z std            : {z_std:.4f}")
    print(f"  elapsed          : {elapsed:.2f}s")
    print("-" * 78)

    metrics = {
        "coverage": coverage,
        "se_ratio": se_ratio,
        "bias": bias,
        "z_mean": z_mean,
        "z_std": z_std,
    }
    all_pass, criteria = validate_coverage_run(metrics)
    print(format_validation_table(criteria))
    print("-" * 78)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 78)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
