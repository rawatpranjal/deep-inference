"""
Method Comparison: Naive SE vs IF SE vs Bootstrap

Compares three approaches to uncertainty quantification:
    1. Naive SE: Hessian-based SE from the structural net (too small)
    2. IF SE: Influence function corrected SE (correct)
    3. Bootstrap: Nonparametric bootstrap (slow but assumption-free)

Usage:
    python application/04_compare_methods.py
    python application/04_compare_methods.py --n-bootstrap 100
"""

import sys
import os
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    J, K, THETA_DIM, ATTRIBUTE_NAMES,
    HIDDEN_DIMS, LEARNING_RATE, EPOCHS, PATIENCE, N_FOLDS,
    DATA_DIR, RESULTS_DIR,
)


def mnl_loss(y, t, theta):
    """MNL loss (same as 03_inference.py)."""
    x = t.reshape(J, K)
    V = x @ theta
    return -V[0] + torch.logsumexp(V, dim=0)


def price_target(x, theta, t_tilde):
    """Target: E[theta_0] = average price sensitivity."""
    return theta[0]


def run_if_inference(Y, T, X, verbose=False):
    """Run IF-corrected inference."""
    from deep_inference import inference

    result = inference(
        Y=Y, T=T, X=X,
        loss=mnl_loss,
        theta_dim=THETA_DIM,
        hessian_depends_on_theta=True,
        hessian_depends_on_y=False,
        target_fn=price_target,
        n_folds=N_FOLDS,
        epochs=EPOCHS,
        patience=PATIENCE,
        hidden_dims=HIDDEN_DIMS,
        lr=LEARNING_RATE,
        verbose=verbose,
    )
    return result


def run_bootstrap(Y, T, X, n_bootstrap=50, verbose=False):
    """Run nonparametric bootstrap for comparison.

    Resamples (Y, T, X) rows and re-runs inference each time.
    Reports bootstrap SE and CI.
    """
    from deep_inference import inference

    n = len(Y)
    boot_estimates = []

    # Use smaller settings for bootstrap (otherwise too slow)
    boot_n_folds = 10
    boot_epochs = 100
    boot_patience = 30

    for b in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        Y_b, T_b, X_b = Y[idx], T[idx], X[idx]

        try:
            result = inference(
                Y=Y_b, T=T_b, X=X_b,
                loss=mnl_loss,
                theta_dim=THETA_DIM,
                hessian_depends_on_theta=True,
                hessian_depends_on_y=False,
                target_fn=price_target,
                n_folds=boot_n_folds,
                epochs=boot_epochs,
                patience=boot_patience,
                hidden_dims=HIDDEN_DIMS,
                lr=LEARNING_RATE,
                verbose=False,
            )
            boot_estimates.append(result.mu_hat)
            if verbose:
                print(f"  Bootstrap [{b+1}/{n_bootstrap}]: mu={result.mu_hat:.4f}")
        except Exception as e:
            print(f"  Bootstrap [{b+1}/{n_bootstrap}]: ERROR {e}")

    boot_estimates = np.array(boot_estimates)
    boot_se = boot_estimates.std()
    boot_ci = (np.percentile(boot_estimates, 2.5), np.percentile(boot_estimates, 97.5))

    return boot_estimates, boot_se, boot_ci


def main():
    parser = argparse.ArgumentParser(description="Compare Naive vs IF vs Bootstrap")
    parser.add_argument("--n-bootstrap", type=int, default=50, help="Bootstrap replications")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap (slow)")
    args = parser.parse_args()

    # Load data
    Y = np.load(DATA_DIR / "Y.npy")
    T = np.load(DATA_DIR / "T.npy")
    X = np.load(DATA_DIR / "X.npy")

    print("=" * 70)
    print("  METHOD COMPARISON: Naive SE vs IF SE vs Bootstrap")
    print("=" * 70)
    print(f"  n={len(Y)}, J={J}, K={K}")

    # 1. IF inference
    print("\n[1/3] Running IF-corrected inference...")
    result = run_if_inference(Y, T, X, verbose=True)

    # Extract naive estimates
    theta_hat = result.theta_hat
    if isinstance(theta_hat, torch.Tensor):
        theta_hat = theta_hat.numpy()
    naive_mean = theta_hat[:, 0].mean()
    naive_se = theta_hat[:, 0].std() / np.sqrt(len(theta_hat))

    if_mean = result.mu_hat
    if_se = result.se

    print(f"\n  Naive: mu={naive_mean:.6f}, se={naive_se:.6f}")
    print(f"  IF:    mu={if_mean:.6f}, se={if_se:.6f}")

    # 2. Bootstrap
    if not args.skip_bootstrap:
        print(f"\n[2/3] Running bootstrap ({args.n_bootstrap} replications)...")
        boot_estimates, boot_se, boot_ci = run_bootstrap(
            Y, T, X, n_bootstrap=args.n_bootstrap, verbose=True
        )
        print(f"\n  Bootstrap: mu={boot_estimates.mean():.6f}, se={boot_se:.6f}")
        print(f"  Bootstrap CI: [{boot_ci[0]:.6f}, {boot_ci[1]:.6f}]")
    else:
        print("\n[2/3] Bootstrap skipped.")
        boot_se = float("nan")
        boot_ci = (float("nan"), float("nan"))

    # 3. Comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Method':<15} {'Estimate':>10} {'SE':>10} {'CI_lo':>10} {'CI_hi':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Naive':<15} {naive_mean:>10.4f} {naive_se:>10.4f} "
          f"{naive_mean - 1.96*naive_se:>10.4f} {naive_mean + 1.96*naive_se:>10.4f}")
    print(f"  {'IF':<15} {if_mean:>10.4f} {if_se:>10.4f} "
          f"{result.ci_lower:>10.4f} {result.ci_upper:>10.4f}")
    if not args.skip_bootstrap:
        print(f"  {'Bootstrap':<15} {boot_estimates.mean():>10.4f} {boot_se:>10.4f} "
              f"{boot_ci[0]:>10.4f} {boot_ci[1]:>10.4f}")

    print(f"\n  SE Ratios:")
    print(f"    IF / Naive:      {if_se / naive_se:.2f}x")
    if not args.skip_bootstrap:
        print(f"    Bootstrap / Naive: {boot_se / naive_se:.2f}x")
        print(f"    IF / Bootstrap:    {if_se / boot_se:.2f}x")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import json
    comparison = {
        "naive": {"mean": float(naive_mean), "se": float(naive_se)},
        "if": {"mean": float(if_mean), "se": float(if_se),
               "ci_lower": float(result.ci_lower), "ci_upper": float(result.ci_upper)},
    }
    if not args.skip_bootstrap:
        comparison["bootstrap"] = {
            "mean": float(boot_estimates.mean()),
            "se": float(boot_se),
            "ci_lower": float(boot_ci[0]),
            "ci_upper": float(boot_ci[1]),
        }
    with open(RESULTS_DIR / "comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n  Results saved to: {(RESULTS_DIR / 'comparison_results.json').resolve()}")


if __name__ == "__main__":
    main()
