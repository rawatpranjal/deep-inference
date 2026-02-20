"""
Simulation 1: Sequential Data with GRU Encoder

Dataset: FashionMNIST (28x28 images treated as 28-step sequences)
Encoder: GRU(input=28, hidden=32) -> Linear head -> theta
Model: Poisson (Y ~ Poisson(exp(alpha + beta*T)))
DGP: theta*(z) where z = PCA_2(flattened image), T confounded
Target: mu* = E[beta*] = -0.8

Usage:
    python -m simulations.sim_01_sequential
    python -m simulations.sim_01_sequential --M 5 --quick
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
    extract_pca_features, compute_coverage_metrics,
    print_coverage_report, confounded_treatment,
)
from simulations.network_factories import gru_factory


# DGP parameters
A0, A1 = 0.5, 0.3   # alpha*(z) = A0 + A1*z1
B0, B1 = -0.8, 0.2  # beta*(z) = B0 + B1*z2
MU_TRUE = B0         # E[beta*] = B0 (since E[z2] = 0)


def load_fashionmnist(n: int = 10000, seed: int = 42) -> np.ndarray:
    """Load FashionMNIST images as flattened arrays.

    Returns:
        X: (n, 784) flattened grayscale images
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
    for idx in indices:
        img, _ = dataset[idx]
        images.append(img.numpy().flatten())

    return np.array(images, dtype=np.float32)


def generate_data(X_pool: np.ndarray, z_pool: np.ndarray,
                  n: int, seed: int):
    """Generate one replication of Poisson data.

    Args:
        X_pool: (N, 784) pool of images
        z_pool: (N, 2) standardized PCA components of X_pool
        n: sample size for this replication
        seed: random seed

    Returns:
        Y, T, X as numpy arrays, plus true mu*
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_pool), n, replace=True)
    X = X_pool[idx]
    z = z_pool[idx]

    # True structural parameters
    alpha_star = A0 + A1 * z[:, 0]
    beta_star = B0 + B1 * z[:, 1]

    # Confounded treatment
    T = confounded_treatment(z, seed=seed)

    # Poisson outcomes: Y ~ Poisson(exp(alpha + beta*T))
    eta = alpha_star + beta_star * T
    lam = np.exp(np.clip(eta, -5, 5))  # Clip for numerical stability
    Y = rng.poisson(lam).astype(np.float32)

    return Y, T.astype(np.float32), X, MU_TRUE


def run_simulation(M: int = 20, n: int = 10000, n_folds: int = 20,
                    epochs: int = 100, patience: int = 30,
                    verbose: bool = True):
    """Run M replications of GRU + Poisson simulation."""
    from deep_inference import inference

    print(f"\n{'='*60}")
    print(f"  SIMULATION 1: SEQUENTIAL DATA (GRU + POISSON)")
    print(f"{'='*60}")
    print(f"  M={M}, n={n}, n_folds={n_folds}, epochs={epochs}")
    print(f"  True mu* = E[beta*] = {MU_TRUE}")
    print(f"  DGP: alpha* = {A0} + {A1}*z1, beta* = {B0} + {B1}*z2")
    print(f"  Encoder: GRU(28, hidden=32)")

    # Load image pool (larger than n for resampling)
    pool_size = min(60000, max(n * 2, 20000))
    print(f"\n  Loading FashionMNIST pool (n={pool_size})...")
    X_pool = load_fashionmnist(pool_size, seed=0)
    z_pool, _ = extract_pca_features(X_pool, n_components=2)
    print(f"  Pool loaded: X={X_pool.shape}, z={z_pool.shape}")

    # Poisson loss
    def poisson_loss(y, t, theta):
        eta = theta[0] + theta[1] * t
        lam = torch.exp(torch.clamp(eta, -5, 5))
        return lam - y * torch.log(lam + 1e-8)

    def beta_target(x, theta, t_tilde):
        return theta[1]

    mu_hats = []
    ses = []

    for m in range(M):
        seed = 1000 + m
        Y, T, X, _ = generate_data(X_pool, z_pool, n, seed)

        if verbose:
            print(f"\n  Rep {m+1}/{M} (seed={seed})...")

        result = inference(
            Y=Y, T=T, X=X,
            loss=poisson_loss,
            theta_dim=2,
            hessian_depends_on_theta=True,
            target_fn=beta_target,
            n_folds=n_folds,
            epochs=epochs,
            patience=patience,
            hidden_dims=[64, 32],
            lr=0.01,
            network_factory=gru_factory,
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
    report = print_coverage_report(metrics, "SIM 1: GRU + POISSON", MU_TRUE)

    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="Sim 1: GRU + Poisson")
    parser.add_argument("--M", type=int, default=20, help="MC replications")
    parser.add_argument("--n", type=int, default=10000, help="Sample size")
    parser.add_argument("--n-folds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: M=5, n=5000")
    args = parser.parse_args()

    if args.quick:
        args.M = 5
        args.n = 5000
        args.epochs = 50

    # Save report
    report_dir = Path(__file__).parent.parent / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"sim_01_sequential_{timestamp}.txt"

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
    )

    fh.close()
    sys.stdout = sys.__stdout__
    print(f"\nReport saved to: {report_file.resolve()}")

    return metrics


if __name__ == "__main__":
    main()
