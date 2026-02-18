"""
Simulation Study: MC Validation of IF Coverage for Custom MNL Loss

This script validates that the deep_inference package delivers correct
coverage when used with a custom multinomial logit loss function —
the exact specification used in the H&M application.

DGP:
    - Consumer embeddings: d_i ~ N(0, I_{d_x}), d_x=10
    - Item attributes: x_j ~ N(0, I_K) for J=20 alternatives, K=6
      (first attribute = "log_price", rest = "embedding PCA dims")
    - True parameters: theta*(d) = W @ d + b, W is (K, d_x), b is (K,)
      Designed so theta_0 (price) is always negative
    - Utilities: V_ij = x_ij @ theta*(d_i)
    - Choices: Y_i = argmax_j(V_ij + gumbel_noise)
    - Target: mu* = E[theta_0*(d)] = b[0] (since E[d]=0)

Validation:
    - M=50 MC replications
    - Check 95% IF coverage, SE ratio ~1.0, |bias| < 0.05

Usage:
    python application/06_simulation_study.py
    python application/06_simulation_study.py --quick    # M=10, n=2000 for fast check
    python application/06_simulation_study.py --full     # M=50, n=5000

Report saved to: evals/reports/sim_study_YYYYMMDD_HHMMSS.txt
"""

import sys
import os
import argparse
import numpy as np
import torch
from datetime import datetime

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ──────────────────────────────────────────────────────────────────────────────
# DGP
# ──────────────────────────────────────────────────────────────────────────────

class MNLSimDGP:
    """DGP matching the H&M application specification."""

    def __init__(self, J=20, K=6, d_x=10, seed=42):
        self.J = J      # Number of alternatives
        self.K = K      # Number of attributes per alternative
        self.d_x = d_x  # Consumer embedding dimension

        # Fix random W, b for the DGP (consistent across replications)
        rng = np.random.RandomState(seed)

        # W: (K, d_x) — maps consumer embeddings to structural parameters
        # Scale down so theta values are moderate
        self.W = rng.randn(K, d_x) * 0.1

        # b: (K,) — intercept (determines E[theta*])
        # Make b[0] (price) negative, others moderate
        self.b = rng.randn(K) * 0.3
        self.b[0] = -1.5  # Strong negative price sensitivity

        # True target: E[theta_0*(d)] = b[0] since E[d] = 0
        self.mu_true = self.b[0]

    def theta_star(self, d):
        """True structural parameters: theta*(d) = W @ d + b.

        Args:
            d: (n, d_x) consumer embeddings

        Returns:
            (n, K) true theta values
        """
        return d @ self.W.T + self.b  # (n, K)

    def generate(self, n, seed):
        """Generate one dataset.

        Args:
            n: Number of choice occasions
            seed: Random seed

        Returns:
            Y: (n,) chosen alternative index (float, always 0 after reorder)
            T: (n, J*K) packed alternative attributes
            X: (n, d_x) consumer embeddings
            theta_true: (n, K) true theta values
        """
        rng = np.random.RandomState(seed)

        # Consumer embeddings
        d = rng.randn(n, self.d_x)

        # True theta
        theta = self.theta_star(d)  # (n, K)

        # Item attributes for each choice occasion
        # x_ij ~ N(0, 1) for all J alternatives
        X_items = rng.randn(n, self.J, self.K)

        # Make first attribute more "price-like" (positive, with variance)
        X_items[:, :, 0] = np.abs(X_items[:, :, 0]) + 0.5  # log_price > 0

        # Utilities: V_ij = x_ij @ theta_i
        V = np.einsum("njk,nk->nj", X_items, theta)  # (n, J)

        # Gumbel noise for multinomial choice
        gumbel = -np.log(-np.log(rng.uniform(size=(n, self.J)) + 1e-20) + 1e-20)
        choices = np.argmax(V + gumbel, axis=1)  # (n,)

        # Reorder: put chosen alternative first (so Y=0 always)
        T_packed = np.zeros((n, self.J * self.K))
        for i in range(n):
            chosen = choices[i]
            # Swap chosen to position 0
            reordered = np.copy(X_items[i])
            reordered[0], reordered[chosen] = X_items[i, chosen].copy(), X_items[i, 0].copy()
            T_packed[i] = reordered.reshape(-1)  # (J*K,)

        Y = np.zeros(n, dtype=np.float32)  # Always 0 (chosen is first)
        X = d.astype(np.float32)
        T_packed = T_packed.astype(np.float32)

        return Y, T_packed, X, theta.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Custom MNL loss (matches application specification)
# ──────────────────────────────────────────────────────────────────────────────

def make_mnl_loss(J, K):
    """Create MNL loss function for given J, K.

    Returns a loss function compatible with deep_inference.inference().
    """
    def mnl_loss(y, t, theta):
        """MNL loss without alternative-specific constants.

        Args:
            y: scalar, chosen alternative index (always 0 after reordering)
            t: (J*K,) packed alternative attributes
            theta: (K,) heterogeneous coefficients on attributes

        Note: Uses V[0] instead of V[int(y)] because chosen alternative
        is always reordered to position 0. This avoids int() which breaks
        torch.vmap used by autodiff.
        """
        x = t.reshape(J, K)          # (J, K) attribute matrix
        V = x @ theta                 # (J,) utilities
        return -V[0] + torch.logsumexp(V, dim=0)

    return mnl_loss


# ──────────────────────────────────────────────────────────────────────────────
# Custom target: E[theta_0] (average price sensitivity)
# ──────────────────────────────────────────────────────────────────────────────

def price_target(x, theta, t_tilde):
    """Target: theta_0 (price sensitivity parameter).

    Args:
        x: consumer covariates (not used)
        theta: (K,) structural parameters
        t_tilde: evaluation point (not used)

    Returns:
        scalar: theta[0]
    """
    return theta[0]


# ──────────────────────────────────────────────────────────────────────────────
# MC Simulation
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(M=50, n=5000, quick=False):
    """Run M replications and compute IF coverage.

    Args:
        M: Number of MC replications
        n: Sample size per replication
        quick: If True, use reduced settings

    Returns:
        dict with coverage, SE ratio, bias, etc.
    """
    from deep_inference import inference

    dgp = MNLSimDGP()
    mnl_loss = make_mnl_loss(dgp.J, dgp.K)

    print(f"\n{'='*70}")
    print(f"  SIMULATION STUDY: Custom MNL Loss + IF Coverage")
    print(f"{'='*70}")
    print(f"  DGP: J={dgp.J} alternatives, K={dgp.K} attributes, d_x={dgp.d_x}")
    print(f"  True target: E[theta_0*(d)] = b[0] = {dgp.mu_true:.4f}")
    print(f"  M={M} replications, n={n}")
    print(f"  Mode: {'quick' if quick else 'full'}")
    print(f"{'='*70}\n")

    # Cross-fitting settings
    n_folds = 20 if quick else 50
    epochs = 150 if quick else 300
    patience = 50

    results = []
    for m in range(1, M + 1):
        seed = 1000 + m  # Avoid overlap with DGP seed=42
        Y, T, X, theta_true = dgp.generate(n, seed)

        try:
            result = inference(
                Y=Y,
                T=T,
                X=X,
                loss=mnl_loss,
                theta_dim=dgp.K,
                hessian_depends_on_theta=True,
                hessian_depends_on_y=False,
                target_fn=price_target,
                n_folds=n_folds,
                epochs=epochs,
                patience=patience,
                hidden_dims=[64, 32],
                lr=0.01,
                verbose=False,
            )

            mu_hat = result.mu_hat
            se = result.se
            ci_lower = result.ci_lower
            ci_upper = result.ci_upper
            covers = ci_lower <= dgp.mu_true <= ci_upper
            bias = mu_hat - dgp.mu_true
            z_score = bias / se if se > 0 else float("nan")

            # Diagnostics
            diag = result.diagnostics if hasattr(result, "diagnostics") else {}
            correction_ratio = diag.get("correction_ratio", float("nan"))

            results.append({
                "m": m,
                "mu_hat": mu_hat,
                "se": se,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "covers": covers,
                "bias": bias,
                "z_score": z_score,
                "correction_ratio": correction_ratio,
            })

            status = "COVER" if covers else "MISS "
            print(f"  [{m:3d}/{M}] mu={mu_hat:+.4f}  se={se:.4f}  "
                  f"CI=[{ci_lower:.3f},{ci_upper:.3f}]  z={z_score:+.2f}  "
                  f"corr_ratio={correction_ratio:.1f}  {status}")

        except Exception as e:
            print(f"  [{m:3d}/{M}] ERROR: {e}")
            results.append({
                "m": m,
                "mu_hat": float("nan"),
                "se": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "covers": False,
                "bias": float("nan"),
                "z_score": float("nan"),
                "correction_ratio": float("nan"),
            })

    return results, dgp


def print_summary(results, dgp):
    """Print summary statistics."""
    valid = [r for r in results if not np.isnan(r["mu_hat"])]
    M_valid = len(valid)

    if M_valid == 0:
        print("\n  NO VALID RESULTS")
        return {}

    mu_hats = np.array([r["mu_hat"] for r in valid])
    ses = np.array([r["se"] for r in valid])
    biases = np.array([r["bias"] for r in valid])
    z_scores = np.array([r["z_score"] for r in valid])
    covers = np.array([r["covers"] for r in valid])
    correction_ratios = np.array([r["correction_ratio"] for r in valid])

    coverage = covers.mean()
    mean_bias = biases.mean()
    mean_se = ses.mean()
    empirical_se = mu_hats.std()
    se_ratio = mean_se / empirical_se if empirical_se > 0 else float("nan")
    z_mean = z_scores.mean()
    z_std = z_scores.std()
    mean_cr = np.nanmean(correction_ratios)

    print(f"\n{'='*70}")
    print(f"  SIMULATION SUMMARY")
    print(f"{'='*70}")
    print(f"  True target:      mu* = {dgp.mu_true:.4f}")
    print(f"  Valid runs:       {M_valid}/{len(results)}")
    print(f"")
    print(f"  Mean estimate:    {mu_hats.mean():.4f}")
    print(f"  Mean bias:        {mean_bias:.4f}")
    print(f"  Abs bias:         {abs(mean_bias):.4f}")
    print(f"  Empirical SE:     {empirical_se:.4f}")
    print(f"  Mean SE:          {mean_se:.4f}")
    print(f"  SE ratio:         {se_ratio:.4f}")
    print(f"  Coverage:         {coverage:.1%} ({int(covers.sum())}/{M_valid})")
    print(f"  z_mean:           {z_mean:.4f}")
    print(f"  z_std:            {z_std:.4f}")
    print(f"  Mean corr_ratio:  {mean_cr:.1f}")

    # Validation
    print(f"\n  {'Metric':<20} {'Value':<12} {'Threshold':<15} {'Status'}")
    print(f"  {'-'*60}")

    checks = {
        "coverage": (coverage, (0.90, 0.99)),
        "se_ratio": (se_ratio, (0.7, 1.5)),
        "abs_bias": (abs(mean_bias), (0.0, 0.05)),
        "z_mean": (z_mean, (-0.3, 0.3)),
        "z_std": (z_std, (0.7, 1.5)),
    }

    all_pass = True
    for name, (val, (lo, hi)) in checks.items():
        if name == "abs_bias":
            ok = val < hi
            thresh_str = f"< {hi}"
        else:
            ok = lo <= val <= hi
            thresh_str = f"[{lo}, {hi}]"
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:<20} {val:<12.4f} {thresh_str:<15} {status}")

    grade = "PASS" if all_pass else "FAIL"
    print(f"\n  OVERALL: {grade}")
    print(f"{'='*70}")

    return {
        "coverage": coverage,
        "se_ratio": se_ratio,
        "mean_bias": mean_bias,
        "z_mean": z_mean,
        "z_std": z_std,
        "all_pass": all_pass,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNL Simulation Study")
    parser.add_argument("--quick", action="store_true", help="Quick mode: M=10, n=2000")
    parser.add_argument("--full", action="store_true", help="Full mode: M=50, n=5000")
    parser.add_argument("-M", type=int, default=None, help="Number of replications")
    parser.add_argument("-n", type=int, default=None, help="Sample size")
    args = parser.parse_args()

    if args.quick:
        M = args.M or 10
        n = args.n or 2000
        quick = True
    elif args.full:
        M = args.M or 50
        n = args.n or 8000
        quick = False
    else:
        M = args.M or 50
        n = args.n or 8000
        quick = False

    # Create report directory
    report_dir = os.path.join(os.path.dirname(__file__), "..", "evals", "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(report_dir, f"sim_study_{timestamp}.txt")

    # Tee output to report file
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

    print(f"Simulation Study: {datetime.now().isoformat()}")
    print(f"Report: {os.path.abspath(report_file)}")

    results, dgp = run_simulation(M=M, n=n, quick=quick)
    summary = print_summary(results, dgp)

    report_fh.close()
    sys.stdout = sys.__stdout__

    print(f"\nReport saved to: {os.path.abspath(report_file)}")
