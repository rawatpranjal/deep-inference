"""
IF-Corrected Inference: H&M Fashion Demand

Runs the deep_inference package on H&M data prepared by 00_prep_data.py.
Uses a custom MNL loss (no alternative-specific constants) and targets
E[beta_price(X)] — average price sensitivity across consumers.

The package handles everything automatically:
    - Cross-fitting with 3-way split (Regime C)
    - Lambda estimation via ridge (alpha=1000)
    - Influence function assembly
    - SE and CI computation

Usage:
    python application/03_inference.py
    python application/03_inference.py --target-idx 0  # beta_price (default)
    python application/03_inference.py --target-idx 1  # beta_pca1
    python application/03_inference.py --all-params     # all K parameters

Report saved to: evals/reports/inference_YYYYMMDD_HHMMSS.txt
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


# ──────────────────────────────────────────────────────────────────────────────
# Custom MNL loss and target
# ──────────────────────────────────────────────────────────────────────────────

def mnl_loss(y, t, theta):
    """MNL loss without alternative-specific constants.

    V_ij = x_ij @ theta, P(Y=j) = softmax(V)
    Loss = -log P(Y=chosen) = -V[0] + log(sum(exp(V)))

    Note: Chosen alternative is always at position 0 (reordered in prep).
    Uses V[0] instead of V[int(y)] for vmap compatibility.
    """
    x = t.reshape(J, K)
    V = x @ theta
    return -V[0] + torch.logsumexp(V, dim=0)


def make_target_fn(param_idx):
    """Create target function for the k-th parameter.

    Target: mu* = E[theta_k(X)]
    """
    def target_fn(x, theta, t_tilde):
        return theta[param_idx]
    return target_fn


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(Y, T, X, target_idx=0, verbose=True):
    """Run IF-corrected inference for a single target parameter.

    Args:
        Y: (n,) chosen alternative (always 0)
        T: (n, J*K) packed alternative attributes
        X: (n, d_x) consumer embeddings
        target_idx: Index of theta parameter to target
        verbose: Print progress

    Returns:
        InferenceResult
    """
    from deep_inference import inference

    target_fn = make_target_fn(target_idx)
    param_name = ATTRIBUTE_NAMES[target_idx] if target_idx < len(ATTRIBUTE_NAMES) else f"theta_{target_idx}"

    print(f"\n  Running inference for E[{param_name}(X)]...")
    print(f"  n={len(Y)}, J={J}, K={K}, theta_dim={THETA_DIM}")
    print(f"  n_folds={N_FOLDS}, epochs={EPOCHS}, patience={PATIENCE}")

    result = inference(
        Y=Y, T=T, X=X,
        loss=mnl_loss,
        theta_dim=THETA_DIM,
        hessian_depends_on_theta=True,
        hessian_depends_on_y=False,
        target_fn=target_fn,
        n_folds=N_FOLDS,
        epochs=EPOCHS,
        patience=PATIENCE,
        hidden_dims=HIDDEN_DIMS,
        lr=LEARNING_RATE,
        verbose=verbose,
    )

    print(f"\n  {'='*50}")
    print(f"  RESULTS: E[{param_name}(X)]")
    print(f"  {'='*50}")
    print(f"  mu_hat:   {result.mu_hat:.6f}")
    print(f"  se:       {result.se:.6f}")
    print(f"  CI (95%): [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")

    if hasattr(result, 'diagnostics') and result.diagnostics:
        print(f"\n  Diagnostics:")
        for k, v in result.diagnostics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    # Naive SE for comparison
    if hasattr(result, 'theta_hat') and result.theta_hat is not None:
        theta_hat = result.theta_hat
        if isinstance(theta_hat, torch.Tensor):
            theta_hat = theta_hat.numpy()
        naive_mean = theta_hat[:, target_idx].mean()
        naive_se = theta_hat[:, target_idx].std() / np.sqrt(len(theta_hat))
        se_ratio = result.se / naive_se if naive_se > 0 else float("nan")

        print(f"\n  Comparison:")
        print(f"    Naive mean:  {naive_mean:.6f}")
        print(f"    Naive SE:    {naive_se:.6f}")
        print(f"    IF SE:       {result.se:.6f}")
        print(f"    SE ratio:    {se_ratio:.2f}x (IF / Naive)")
        print(f"    IF widens CI by {(se_ratio - 1) * 100:.0f}%")

    return result


def main():
    parser = argparse.ArgumentParser(description="IF-corrected inference on H&M data")
    parser.add_argument("--target-idx", type=int, default=0,
                        help="Parameter index to target (0=price, 1-5=PCA)")
    parser.add_argument("--all-params", action="store_true",
                        help="Run inference for all K parameters")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR

    # Load data
    print("=" * 60)
    print("  H&M FASHION DEMAND: IF-CORRECTED INFERENCE")
    print("=" * 60)

    Y = np.load(data_dir / "Y.npy")
    T = np.load(data_dir / "T.npy")
    X = np.load(data_dir / "X.npy")

    print(f"\n  Data loaded from: {data_dir.resolve()}")
    print(f"  Y: {Y.shape}  T: {T.shape}  X: {X.shape}")
    print(f"  n_occasions = {len(Y)}")

    # Create report directory
    report_dir = Path(os.path.dirname(__file__)) / ".." / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"inference_{timestamp}.txt"

    # Tee output
    import io

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    report_fh = open(report_file, "w")
    sys.stdout = Tee(sys.__stdout__, report_fh)

    print(f"Inference Report: {datetime.now().isoformat()}")
    print(f"Report: {report_file.resolve()}")

    # Run
    results = {}
    if args.all_params:
        for idx in range(K):
            results[idx] = run_inference(Y, T, X, target_idx=idx)
    else:
        results[args.target_idx] = run_inference(Y, T, X, target_idx=args.target_idx)

    # Summary table
    if len(results) > 1:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: All Parameter Estimates")
        print(f"{'='*70}")
        print(f"  {'Param':<12} {'mu_hat':>10} {'SE':>10} {'CI_lo':>10} {'CI_hi':>10}")
        print(f"  {'-'*52}")
        for idx, r in results.items():
            name = ATTRIBUTE_NAMES[idx] if idx < len(ATTRIBUTE_NAMES) else f"theta_{idx}"
            print(f"  {name:<12} {r.mu_hat:>10.4f} {r.se:>10.4f} "
                  f"{r.ci_lower:>10.4f} {r.ci_upper:>10.4f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_dict = {}
    for idx, r in results.items():
        name = ATTRIBUTE_NAMES[idx] if idx < len(ATTRIBUTE_NAMES) else f"theta_{idx}"
        results_dict[name] = {
            "mu_hat": float(r.mu_hat),
            "se": float(r.se),
            "ci_lower": float(r.ci_lower),
            "ci_upper": float(r.ci_upper),
        }

    import json
    with open(RESULTS_DIR / "inference_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    report_fh.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to: {report_file.resolve()}")
    print(f"Results saved to: {(RESULTS_DIR / 'inference_results.json').resolve()}")


if __name__ == "__main__":
    main()
