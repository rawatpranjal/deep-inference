"""
Simulation 2: Image Data with CNN Encoder

Dataset: FashionMNIST (1x28x28 grayscale images)
Encoder: Conv2d(1->16)->Pool->Conv2d(16->32)->Pool->Linear->theta
Model: Logit (Y ~ Bernoulli(sigmoid(alpha + beta*T)))
DGP: theta*(z) where z = (normalized_label, normalized_intensity), T confounded
Target: mu* = E[beta*] = 0.5

Usage:
    python -m simulations.sim_02_image
    python -m simulations.sim_02_image --M 5 --quick
"""

import sys
import os
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulations.common import (
    compute_coverage_metrics, print_coverage_report,
)
from simulations.network_factories import cnn_factory


# DGP parameters
B0 = 0.5   # E[beta*] = B0
B1 = 0.4   # beta depends on z1 (label)
B2 = -0.3  # beta depends on z2 (intensity)
A1 = 0.3   # alpha depends on z1
A2 = 0.2   # alpha depends on z2
MU_TRUE = B0  # E[beta*] = B0 since E[z1] = E[z2] = 0


def load_fashionmnist_with_labels(n: int = 10000, seed: int = 42):
    """Load FashionMNIST images with labels and intensity features.

    Returns:
        X: (n, 784) flattened images
        z: (n, 2) standardized features [label, mean_intensity]
    """
    from torchvision import datasets, transforms

    dataset = datasets.FashionMNIST(
        root=os.path.join(os.path.dirname(__file__), "data"),
        train=True, download=True,
        transform=transforms.ToTensor(),
    )

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), n, replace=False)
    images = []
    labels = []
    intensities = []

    for idx in indices:
        img, label = dataset[idx]
        arr = img.numpy().flatten()
        images.append(arr)
        labels.append(label)
        intensities.append(arr.mean())

    X = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32)

    # Standardize z features
    z1 = (labels - labels.mean()) / (labels.std() + 1e-8)
    z2 = (intensities - intensities.mean()) / (intensities.std() + 1e-8)
    z = np.column_stack([z1, z2])

    return X, z


def generate_data(X_pool, z_pool, n, seed):
    """Generate one replication of logit data."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_pool), n, replace=True)
    X = X_pool[idx]
    z = z_pool[idx]

    # True parameters
    alpha_star = A1 * z[:, 0] + A2 * z[:, 1]
    beta_star = B0 + B1 * z[:, 0] + B2 * z[:, 1]

    # Confounded treatment
    T = 0.3 * z[:, 0] + rng.randn(n)

    # Logit outcomes
    eta = alpha_star + beta_star * T
    p = 1.0 / (1.0 + np.exp(-eta))
    Y = rng.binomial(1, p).astype(np.float32)

    return Y, T.astype(np.float32), X, MU_TRUE


def run_simulation(M=20, n=10000, n_folds=20, epochs=150, patience=30,
                    n_repeats=1, lambda_method=None, verbose=True):
    """Run M replications of CNN + Logit simulation."""
    from deep_inference import inference

    print(f"\n{'='*60}")
    print(f"  SIMULATION 2: IMAGE DATA (CNN + LOGIT)")
    print(f"{'='*60}")
    print(f"  M={M}, n={n}, n_folds={n_folds}, epochs={epochs}")
    print(f"  n_repeats={n_repeats}, lambda_method={lambda_method}")
    print(f"  True mu* = E[beta*] = {MU_TRUE}")
    print(f"  DGP: beta* = {B0} + {B1}*z1 + {B2}*z2")
    print(f"  Encoder: CNN (Conv->Pool->Conv->Pool->Linear)")

    pool_size = min(60000, max(n * 2, 20000))
    print(f"\n  Loading FashionMNIST pool (n={pool_size})...")
    X_pool, z_pool = load_fashionmnist_with_labels(pool_size, seed=0)
    print(f"  Pool loaded: X={X_pool.shape}, z={z_pool.shape}")

    # Logit loss
    def logit_loss(y, t, theta):
        eta = theta[0] + theta[1] * t
        return torch.log(1 + torch.exp(eta)) - y * eta

    def beta_target(x, theta, t_tilde):
        return theta[1]

    mu_hats = []
    ses = []

    for m in range(M):
        seed = 2000 + m
        Y, T, X, _ = generate_data(X_pool, z_pool, n, seed)

        if verbose:
            print(f"\n  Rep {m+1}/{M} (seed={seed})...")

        result = inference(
            Y=Y, T=T, X=X,
            loss=logit_loss,
            theta_dim=2,
            hessian_depends_on_theta=True,
            target_fn=beta_target,
            n_folds=n_folds,
            n_repeats=n_repeats,
            lambda_method=lambda_method,
            epochs=epochs,
            patience=patience,
            hidden_dims=[64, 32],
            lr=0.01,
            network_factory=cnn_factory,
            verbose=False,
        )

        mu_hats.append(result.mu_hat)
        ses.append(result.se)

        if verbose:
            ci_lo, ci_hi = result.ci_lower, result.ci_upper
            covers = ci_lo <= MU_TRUE <= ci_hi
            print(f"    mu_hat={result.mu_hat:.4f}, se={result.se:.4f}, "
                  f"CI=[{ci_lo:.4f}, {ci_hi:.4f}], covers={covers}")

    metrics = compute_coverage_metrics(mu_hats, ses, MU_TRUE)
    report = print_coverage_report(metrics, "SIM 2: CNN + LOGIT", MU_TRUE)

    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="Sim 2: CNN + Logit")
    parser.add_argument("--M", type=int, default=20, help="MC replications")
    parser.add_argument("--n", type=int, default=10000, help="Sample size")
    parser.add_argument("--n-folds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="Repeated cross-fitting splits (median DML)")
    parser.add_argument("--lambda-method", type=str, default=None,
                        help="Lambda strategy override (e.g. ridge, lgbm, aggregate)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: M=5, n=5000")
    args = parser.parse_args()

    if args.quick:
        args.M = 5
        args.n = 5000
        args.epochs = 50

    report_dir = Path(__file__).parent.parent / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"sim_02_image_{timestamp}.txt"

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

    fh = open(report_file, "w")
    sys.stdout = Tee(sys.__stdout__, fh)

    metrics, report = run_simulation(
        M=args.M, n=args.n, n_folds=args.n_folds,
        epochs=args.epochs, patience=args.patience,
        n_repeats=args.n_repeats, lambda_method=args.lambda_method,
    )

    fh.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to: {report_file.resolve()}")

    return metrics


if __name__ == "__main__":
    main()
