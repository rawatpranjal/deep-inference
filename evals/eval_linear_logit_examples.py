"""
Eval: Linear and Logit Examples for Paper Section 4

Runs M=50 MC replications for:
  1. Linear model (Regime B): n=3000, E[beta]=0.5
  2. Binary logit (Regime C): n=5000, E[beta]=0.5 (naive vs IF comparison)

Outputs results tables for the paper.
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, "/Users/pranjal/deepest/src")


@dataclass
class SimResult:
    sim_id: int
    mu_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    covered: bool
    z_score: float
    # For logit: also track naive
    mu_naive: float = np.nan
    se_naive: float = np.nan
    covered_naive: bool = False


def run_linear_sim(sim_id: int, n: int = 3000) -> SimResult:
    """Single MC rep for linear model."""
    from deep_inference import inference

    np.random.seed(sim_id)
    X = np.random.randn(n, 5)
    T = np.random.randn(n)
    alpha_true = 1.0 + 0.3 * X[:, 0]
    beta_true = 0.5 + 0.2 * X[:, 1]
    Y = alpha_true + beta_true * T + np.random.randn(n)
    mu_true = 0.5

    result = inference(Y, T, X, model='linear', target='beta',
                       n_folds=20, epochs=100, patience=50, verbose=False)

    covered = result.ci_lower <= mu_true <= result.ci_upper
    z = (result.mu_hat - mu_true) / result.se if result.se > 0 else np.nan

    return SimResult(sim_id=sim_id, mu_hat=result.mu_hat, se=result.se,
                     ci_lower=result.ci_lower, ci_upper=result.ci_upper,
                     covered=covered, z_score=z)


def run_logit_sim(sim_id: int, n: int = 5000) -> SimResult:
    """Single MC rep for logit model (tracks both naive and IF)."""
    from deep_inference import inference

    np.random.seed(sim_id)
    X = np.random.randn(n, 5)
    T = np.random.randn(n)
    beta_true = 0.5 + 0.3 * X[:, 0]
    p = 1 / (1 + np.exp(-(1.0 + beta_true * T)))
    Y = np.random.binomial(1, p).astype(float)
    mu_true = 0.5

    result = inference(Y, T, X, model='logit', target='beta',
                       n_folds=20, epochs=200, patience=50, verbose=False)

    # IF results
    covered_if = result.ci_lower <= mu_true <= result.ci_upper
    z_if = (result.mu_hat - mu_true) / result.se if result.se > 0 else np.nan

    # Naive: just mean of theta_hat[:,1] with std/sqrt(n)
    beta_hats = result.theta_hat[:, 1]
    mu_naive = beta_hats.mean()
    se_naive = beta_hats.std() / np.sqrt(n)
    ci_naive_lo = mu_naive - 1.96 * se_naive
    ci_naive_hi = mu_naive + 1.96 * se_naive
    covered_naive = ci_naive_lo <= mu_true <= ci_naive_hi

    return SimResult(sim_id=sim_id, mu_hat=result.mu_hat, se=result.se,
                     ci_lower=result.ci_lower, ci_upper=result.ci_upper,
                     covered=covered_if, z_score=z_if,
                     mu_naive=mu_naive, se_naive=se_naive,
                     covered_naive=covered_naive)


def compute_metrics(results: List[SimResult], mu_true: float) -> Dict:
    valid = [r for r in results if not np.isnan(r.mu_hat)]
    mu_hats = np.array([r.mu_hat for r in valid])
    ses = np.array([r.se for r in valid])
    covered = np.array([r.covered for r in valid])
    zs = np.array([r.z_score for r in valid])
    zs = zs[~np.isnan(zs)]

    emp_se = mu_hats.std()
    mean_se = ses.mean()

    return {
        "coverage": covered.mean(),
        "se_ratio": mean_se / emp_se if emp_se > 0 else np.nan,
        "bias": mu_hats.mean() - mu_true,
        "z_mean": zs.mean() if len(zs) > 0 else np.nan,
        "z_std": zs.std() if len(zs) > 0 else np.nan,
        "mean_se": mean_se,
        "emp_se": emp_se,
    }


def main():
    M = 50

    # ===== LINEAR MODEL =====
    print("=" * 60)
    print(f"LINEAR MODEL: M={M}, n=3000, E[beta]=0.5")
    print("=" * 60)

    linear_results = []
    for m in range(1, M + 1):
        if m % 10 == 0:
            print(f"  Linear sim {m}/{M}...")
        r = run_linear_sim(m)
        linear_results.append(r)

    lin_metrics = compute_metrics(linear_results, 0.5)
    print(f"\n--- Linear Results ---")
    print(f"Coverage:  {lin_metrics['coverage']:.0%}")
    print(f"SE ratio:  {lin_metrics['se_ratio']:.3f}")
    print(f"|Bias|:    {abs(lin_metrics['bias']):.4f}")
    print(f"z-mean:    {lin_metrics['z_mean']:.3f}")
    print(f"z-std:     {lin_metrics['z_std']:.3f}")

    # ===== LOGIT MODEL =====
    print("\n" + "=" * 60)
    print(f"LOGIT MODEL: M={M}, n=5000, E[beta]=0.5")
    print("=" * 60)

    logit_results = []
    for m in range(1, M + 1):
        if m % 5 == 0:
            print(f"  Logit sim {m}/{M}...")
        r = run_logit_sim(m)
        logit_results.append(r)

    logit_metrics = compute_metrics(logit_results, 0.5)

    # Naive metrics
    naive_covered = np.array([r.covered_naive for r in logit_results if not np.isnan(r.mu_naive)])
    naive_ses = np.array([r.se_naive for r in logit_results if not np.isnan(r.se_naive)])

    print(f"\n--- Logit IF Results ---")
    print(f"Coverage:  {logit_metrics['coverage']:.0%}")
    print(f"SE ratio:  {logit_metrics['se_ratio']:.3f}")
    print(f"|Bias|:    {abs(logit_metrics['bias']):.4f}")
    print(f"z-mean:    {logit_metrics['z_mean']:.3f}")
    print(f"z-std:     {logit_metrics['z_std']:.3f}")

    print(f"\n--- Logit Naive Results ---")
    print(f"Coverage:  {naive_covered.mean():.0%}")
    print(f"Mean SE:   {naive_ses.mean():.4f}")
    print(f"IF/Naive SE ratio: {logit_metrics['mean_se']/naive_ses.mean():.1f}x")

    # ===== PAPER TABLE FORMAT =====
    print("\n" + "=" * 60)
    print("PAPER TABLE (LaTeX)")
    print("=" * 60)

    cov_l = lin_metrics['coverage'] * 100
    ser_l = lin_metrics['se_ratio']
    bias_l = abs(lin_metrics['bias'])
    zm_l = lin_metrics['z_mean']

    cov_i = logit_metrics['coverage'] * 100
    ser_i = logit_metrics['se_ratio']
    bias_i = abs(logit_metrics['bias'])
    zm_i = logit_metrics['z_mean']

    cov_n = naive_covered.mean() * 100

    print(f"Linear IF:  Coverage={cov_l:.0f}%, SE ratio={ser_l:.2f}, |Bias|={bias_l:.3f}, z-mean={zm_l:.2f}")
    print(f"Logit IF:   Coverage={cov_i:.0f}%, SE ratio={ser_i:.2f}, |Bias|={bias_i:.3f}, z-mean={zm_i:.2f}")
    print(f"Logit Naive: Coverage={cov_n:.0f}%, Mean SE={naive_ses.mean():.4f}")
    print(f"IF/Naive SE: {logit_metrics['mean_se']/naive_ses.mean():.1f}x")


if __name__ == "__main__":
    main()
