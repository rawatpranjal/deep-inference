"""
Eval 09: Multinomial Choice Model Validation

Goal: Validate the multinomial logit (conditional logit / McFadden) implementation
against oracle formulas from the DGP.

Tests:
    1. Parameter recovery: θ̂(W) vs θ*(W) for J=3, K=2
    2. Autodiff validation: score/Hessian vs oracle closed-form
    3. Lambda estimation: Λ̂(W) vs MC oracle
    4. Coverage: M=50 MC simulations with strict thresholds

DGP: Heterogeneous conditional logit
    α_j(W) = a0_j + a1_j * W[0],  β_k(W) = b0_k + b1_k * W[0]
    Target: μ* = E[β_1(W)] = -0.8

Uses strict thresholds from evals/common/metrics.py.
"""

import sys
import os
import argparse
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, "/Users/pranjal/deepest/src")

from evals.dgp_multinomial import (
    MultinomialLogitDGP,
    generate_multinomial_dgp,
    oracle_score,
    oracle_hessian,
    oracle_lambda_mc,
)
from evals.common.metrics import (
    RECOVERY_THRESHOLDS,
    AUTODIFF_THRESHOLDS,
    LAMBDA_THRESHOLDS,
    COVERAGE_THRESHOLDS,
    validate_coverage_run,
    validate_recovery_run,
    validate_autodiff_run,
    validate_lambda_run,
    format_validation_table,
)


def eval_parameter_recovery(n: int = 10000, seed: int = 42, verbose: bool = True,
                            quick: bool = False):
    """
    Test 1: Parameter recovery — can the model learn θ*(W)?

    Generates data, trains structural network, and compares
    estimated θ̂(W) to true θ*(W).
    """
    print("\n" + "=" * 60)
    print("TEST 1: PARAMETER RECOVERY")
    print("=" * 60)

    dgp = MultinomialLogitDGP()
    Y, T, W, theta_true, mu_true = generate_multinomial_dgp(n, seed, dgp)

    n_folds = 20 if quick else 50
    epochs = 100 if quick else 300

    print(f"  n = {n}")
    print(f"  J = {dgp.J}, K = {dgp.K}")
    print(f"  theta_dim = {dgp.theta_dim()}")
    print(f"  mu_true = {mu_true:.4f}")
    print(f"  n_folds = {n_folds}, epochs = {epochs}")
    print(f"  Y distribution: {[float((Y == j).sum()) / n for j in range(dgp.J)]}")

    # Train structural network
    from deep_inference import structural_dml

    result = structural_dml(
        Y=Y.numpy(),
        T=T.numpy(),
        X=W.numpy(),
        family='multinomial_logit',
        n_folds=n_folds,
        epochs=epochs,
        lr=0.01,
        patience=50,
        hidden_dims=[64, 32],
        verbose=verbose,
        n_alternatives=dgp.J,
        n_attributes=dgp.K,
    )

    theta_hat = result.theta_hat  # (n, theta_dim)
    if isinstance(theta_hat, torch.Tensor):
        theta_hat = theta_hat.numpy()
    theta_true_np = theta_true.numpy()

    # Compute recovery metrics per parameter
    metrics = {}
    param_names = [f"alpha_{j}" for j in range(1, dgp.J)] + [f"beta_{k}" for k in range(dgp.K)]

    print(f"\n  {'Param':<10} {'RMSE':<10} {'Corr':<10} {'Bias':<10}")
    print(f"  {'-'*40}")
    for i, name in enumerate(param_names):
        rmse = np.sqrt(np.mean((theta_hat[:, i] - theta_true_np[:, i]) ** 2))
        corr = np.corrcoef(theta_hat[:, i], theta_true_np[:, i])[0, 1]
        bias = np.mean(theta_hat[:, i] - theta_true_np[:, i])
        metrics[f"rmse_{name}"] = rmse
        metrics[f"corr_{name}"] = corr
        print(f"  {name:<10} {rmse:<10.4f} {corr:<10.4f} {bias:<10.4f}")

    # Validate — use relaxed thresholds in quick mode (insufficient training)
    if quick:
        QUICK_RECOVERY = {"rmse": (0, 0.5), "correlation": (0.2, 1.0)}
        all_pass, criteria = validate_recovery_run(metrics)
        # Override with relaxed quick thresholds
        criteria_quick = {}
        for key, val in metrics.items():
            if "rmse" in key:
                criteria_quick[f"{key} < 0.5 (quick)"] = val < 0.5
            elif "corr" in key:
                criteria_quick[f"{key} > 0.2 (quick)"] = val > 0.2
        all_pass = all(criteria_quick.values())
        criteria = criteria_quick
        print(f"\n  VALIDATION (quick mode — relaxed thresholds):")
    else:
        all_pass, criteria = validate_recovery_run(metrics)
        print(f"\n  VALIDATION:")
    print(format_validation_table(criteria))
    status = "PASS" if all_pass else "FAIL"
    print(f"\n  TEST 1: {status}")
    return all_pass, metrics


def eval_autodiff_validation(n: int = 100, seed: int = 42):
    """
    Test 2: Autodiff validation — score/Hessian match oracle formulas.
    """
    print("\n" + "=" * 60)
    print("TEST 2: AUTODIFF VALIDATION")
    print("=" * 60)

    dgp = MultinomialLogitDGP()
    from deep_inference.models.multinomial import MultinomialLogitModel

    model = MultinomialLogitModel(n_alternatives=dgp.J, n_attributes=dgp.K)

    np.random.seed(seed)
    max_grad_err = 0.0
    max_hess_err = 0.0

    for i in range(n):
        # Random data
        theta_np = np.random.randn(dgp.theta_dim())
        x_np = np.random.randn(dgp.J, dgp.K)
        y_int = np.random.randint(0, dgp.J)

        theta_t = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
        t_packed = torch.tensor(x_np.flatten(), dtype=torch.float64)
        y_t = torch.tensor(float(y_int), dtype=torch.float64)

        # Package score
        pkg_score = model.score(y_t, t_packed, theta_t).detach().numpy()
        oracle_s = oracle_score(y_int, x_np, theta_np, dgp.J, dgp.K)
        grad_err = np.max(np.abs(pkg_score - oracle_s))
        max_grad_err = max(max_grad_err, grad_err)

        # Package Hessian
        pkg_hess = model.hessian(y_t, t_packed, theta_t).detach().numpy()
        oracle_h = oracle_hessian(x_np, theta_np, dgp.J, dgp.K)
        hess_err = np.max(np.abs(pkg_hess - oracle_h))
        max_hess_err = max(max_hess_err, hess_err)

    print(f"  Tested {n} random observations")
    print(f"  Max |score_pkg - score_oracle|  = {max_grad_err:.2e}")
    print(f"  Max |hessian_pkg - hessian_oracle| = {max_hess_err:.2e}")

    metrics = {
        "gradient_error": max_grad_err,
        "hessian_error": max_hess_err,
    }

    all_pass, criteria = validate_autodiff_run(metrics)
    print(f"\n  VALIDATION:")
    print(format_validation_table(criteria))
    status = "PASS" if all_pass else "FAIL"
    print(f"\n  TEST 2: {status}")
    return all_pass, metrics


def eval_lambda_estimation(n: int = 200, seed: int = 42):
    """
    Test 3: Lambda estimation — Λ̂(W) vs oracle MC Lambda.
    """
    print("\n" + "=" * 60)
    print("TEST 3: LAMBDA ESTIMATION")
    print("=" * 60)

    dgp = MultinomialLogitDGP()
    from deep_inference.models.multinomial import MultinomialLogitModel

    model = MultinomialLogitModel(n_alternatives=dgp.J, n_attributes=dgp.K)

    np.random.seed(seed)
    frob_errors = []
    min_eigs = []

    for i in range(n):
        # Random individual
        w = np.random.randn(dgp.d_w)
        theta_np = dgp.theta_star(w.reshape(1, -1))[0]

        # Oracle Lambda via MC
        Lambda_oracle = oracle_lambda_mc(w, theta_np, dgp.J, dgp.K, n_mc=500)

        # Package Lambda via MC integration
        theta_t = torch.tensor(theta_np, dtype=torch.float64)
        t_samples = torch.randn(500, dgp.J * dgp.K, dtype=torch.float64)
        Lambda_pkg = model.compute_lambda_integral(theta_t, t_samples).numpy()

        # Frobenius error
        frob_norm_oracle = np.linalg.norm(Lambda_oracle, 'fro')
        if frob_norm_oracle > 1e-10:
            frob_err = np.linalg.norm(Lambda_pkg - Lambda_oracle, 'fro') / frob_norm_oracle
        else:
            frob_err = 0.0
        frob_errors.append(frob_err)

        # Min eigenvalue
        eigvals = np.linalg.eigvalsh(Lambda_pkg)
        min_eigs.append(eigvals.min())

    mean_frob = np.mean(frob_errors)
    non_psd = sum(1 for e in min_eigs if e < -1e-10)
    min_eig = min(min_eigs)

    print(f"  Tested {n} random individuals")
    print(f"  Mean relative Frobenius error: {mean_frob:.6f}")
    print(f"  Non-PSD count: {non_psd} / {n}")
    print(f"  Min eigenvalue: {min_eig:.6f}")

    metrics = {
        "frobenius_error": mean_frob,
        "non_psd_count": non_psd,
        "min_eigenvalue": min_eig,
    }

    all_pass, criteria = validate_lambda_run(metrics)
    print(f"\n  VALIDATION:")
    print(format_validation_table(criteria))
    status = "PASS" if all_pass else "FAIL"
    print(f"\n  TEST 3: {status}")
    return all_pass, metrics


def eval_coverage(M: int = 50, n: int = 8000, seed: int = 42,
                  epochs: int = 300, n_folds: int = 50, quick: bool = False):
    """
    Test 4: Coverage — Monte Carlo validation of confidence intervals.
    """
    print("\n" + "=" * 60)
    print(f"TEST 4: COVERAGE (M={M}, n={n})")
    print("=" * 60)

    dgp = MultinomialLogitDGP()
    mu_true = dgp.mu_true()
    print(f"  mu_true = {mu_true:.4f}")
    print(f"  epochs = {epochs}, n_folds = {n_folds}")

    if quick:
        M = min(M, 10)
        n = min(n, 2000)
        epochs = 50
        n_folds = 20
        print(f"  QUICK MODE: M={M}, n={n}, epochs={epochs}, n_folds={n_folds}")

    from deep_inference import structural_dml

    results = []
    for m in range(1, M + 1):
        try:
            Y, T, W, theta_true, _ = generate_multinomial_dgp(n, seed=seed + m, dgp=dgp)

            result = structural_dml(
                Y=Y.numpy(),
                T=T.numpy(),
                X=W.numpy(),
                family='multinomial_logit',
                n_folds=n_folds,
                epochs=epochs,
                lr=0.01,
                patience=50,
                hidden_dims=[64, 32],
                verbose=False,
                n_alternatives=dgp.J,
                n_attributes=dgp.K,
            )

            mu_hat = result.mu_hat
            se = result.se
            ci_lo = result.ci_lower
            ci_hi = result.ci_upper
            covered = ci_lo <= mu_true <= ci_hi
            z_score = (mu_hat - mu_true) / se if se > 0 else float('nan')

            results.append({
                "sim": m, "mu_hat": mu_hat, "se": se,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "covered": covered, "z_score": z_score,
            })
            print(f"  [{m:3d}/{M}] mu_hat={mu_hat:.4f} se={se:.4f} "
                  f"CI=[{ci_lo:.4f}, {ci_hi:.4f}] cov={'T' if covered else 'F'}")
        except Exception as e:
            print(f"  [{m:3d}/{M}] FAILED: {e}")
            continue

    if len(results) == 0:
        print("  No successful simulations!")
        return False, {}

    # Compute metrics
    mu_hats = np.array([r["mu_hat"] for r in results])
    ses = np.array([r["se"] for r in results])
    coverages = np.array([r["covered"] for r in results])
    z_scores = np.array([r["z_score"] for r in results])
    z_valid = z_scores[~np.isnan(z_scores)]

    metrics = {
        "coverage": coverages.mean(),
        "se_ratio": mu_hats.std() / ses.mean() if ses.mean() > 0 else float('nan'),
        "bias": mu_hats.mean() - mu_true,
        "z_mean": z_valid.mean() if len(z_valid) > 0 else float('nan'),
        "z_std": z_valid.std() if len(z_valid) > 0 else float('nan'),
    }

    print(f"\n  SUMMARY (M={len(results)} successful):")
    print(f"  Coverage: {metrics['coverage']:.1%}")
    print(f"  SE ratio: {metrics['se_ratio']:.4f}")
    print(f"  Bias:     {metrics['bias']:.6f}")
    print(f"  z_mean:   {metrics['z_mean']:.4f}")
    print(f"  z_std:    {metrics['z_std']:.4f}")

    all_pass, criteria = validate_coverage_run(metrics)
    print(f"\n  VALIDATION:")
    print(format_validation_table(criteria))
    status = "PASS" if all_pass else "FAIL"
    print(f"\n  TEST 4: {status}")
    return all_pass, metrics


def main():
    parser = argparse.ArgumentParser(description="Eval 09: Multinomial Choice Model")
    parser.add_argument("--quick", action="store_true", help="Quick mode with reduced settings")
    parser.add_argument("--test", type=int, default=None, help="Run specific test (1-4)")
    args = parser.parse_args()

    print("=" * 60)
    print("EVAL 09: MULTINOMIAL CHOICE MODEL VALIDATION")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    all_pass = True

    if args.test is None or args.test == 2:
        # Test 2: Autodiff (fast, always run first)
        p, m = eval_autodiff_validation()
        results["autodiff"] = (p, m)
        all_pass = all_pass and p

    if args.test is None or args.test == 3:
        # Test 3: Lambda
        n_lambda = 50 if args.quick else 200
        p, m = eval_lambda_estimation(n=n_lambda)
        results["lambda"] = (p, m)
        all_pass = all_pass and p

    if args.test is None or args.test == 1:
        # Test 1: Recovery
        n_recovery = 2000 if args.quick else 10000
        p, m = eval_parameter_recovery(n=n_recovery, quick=args.quick)
        results["recovery"] = (p, m)
        all_pass = all_pass and p

    if args.test is None or args.test == 4:
        # Test 4: Coverage (slowest, run last)
        p, m = eval_coverage(quick=args.quick)
        results["coverage"] = (p, m)
        all_pass = all_pass and p

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for name, (passed, _) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<15}: {status}")

    print("\n" + "=" * 60)
    if all_pass:
        print("EVAL 09: PASS")
    else:
        print("EVAL 09: FAIL")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
