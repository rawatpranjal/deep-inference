"""
Simulation 3: Text Data with Transformer Embeddings

Dataset: IMDB reviews (HuggingFace datasets)
Encoder: Pre-extracted sentence-transformer embeddings (all-MiniLM-L6-v2, 384D)
         Then standard MLP on embeddings (no custom network_factory needed).
Model: Logit (Y ~ Bernoulli(sigmoid(alpha + beta*T)))
DGP: theta*(z) where z = PCA_5(embeddings), T confounded
Target: mu* = E[beta*] = 1.0

Usage:
    python -m simulations.sim_03_text
    python -m simulations.sim_03_text --M 5 --quick
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
    print_coverage_report,
)


# DGP parameters
A_COEFFS = [0.2, -0.3, 0.1]  # alpha depends on z1, z2, z3
B0 = 1.0                      # E[beta*] = B0
B_COEFFS = [0.5, 0.3]         # beta depends on z1, z4
MU_TRUE = B0                  # E[beta*] = B0 since E[z_k] = 0


def extract_embeddings(n: int = 10000, cache_path: str = None,
                        seed: int = 42) -> np.ndarray:
    """Extract sentence-transformer embeddings from IMDB reviews.

    Uses all-MiniLM-L6-v2 (384-dim). Results are cached to disk.

    Args:
        n: number of reviews to embed
        cache_path: path to cache file
        seed: for sampling reviews

    Returns:
        embeddings: (n, 384) float32 array
    """
    if cache_path is None:
        cache_path = os.path.join(os.path.dirname(__file__), "data",
                                    "imdb_embeddings.npy")

    if os.path.exists(cache_path):
        emb = np.load(cache_path)
        if len(emb) >= n:
            print(f"  Loaded cached embeddings: {cache_path} ({emb.shape})")
            return emb[:n]

    print(f"  Extracting embeddings for {n} IMDB reviews...")
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    dataset = load_dataset("imdb", split="train")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), min(n, len(dataset)), replace=False)
    texts = [dataset[int(i)]["text"] for i in indices]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True,
                               batch_size=256)
    embeddings = np.array(embeddings, dtype=np.float32)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  Saved embeddings to: {cache_path} ({embeddings.shape})")

    return embeddings[:n]


def generate_data(emb_pool, z_pool, n, seed):
    """Generate one replication of logit data from text embeddings."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(emb_pool), n, replace=True)
    X = emb_pool[idx]  # (n, 384) — embeddings ARE the covariates
    z = z_pool[idx]

    # True parameters
    alpha_star = (A_COEFFS[0] * z[:, 0] + A_COEFFS[1] * z[:, 1] +
                  A_COEFFS[2] * z[:, 2])
    beta_star = B0 + B_COEFFS[0] * z[:, 0] + B_COEFFS[1] * z[:, 3]

    # Confounded treatment
    T = 0.3 * z[:, 0] + rng.randn(n)

    # Logit outcomes
    eta = alpha_star + beta_star * T
    p = 1.0 / (1.0 + np.exp(-eta))
    Y = rng.binomial(1, p).astype(np.float32)

    return Y, T.astype(np.float32), X, MU_TRUE


def run_simulation(M=20, n=10000, n_folds=50, epochs=200, patience=50,
                    verbose=True):
    """Run M replications of Transformer embeddings + Logit simulation."""
    from deep_inference import inference

    print(f"\n{'='*60}")
    print(f"  SIMULATION 3: TEXT DATA (SENTENCE-BERT + LOGIT)")
    print(f"{'='*60}")
    print(f"  M={M}, n={n}, n_folds={n_folds}, epochs={epochs}")
    print(f"  True mu* = E[beta*] = {MU_TRUE}")
    print(f"  DGP: beta* = {B0} + {B_COEFFS[0]}*z1 + {B_COEFFS[1]}*z4")
    print(f"  Encoder: Pre-extracted all-MiniLM-L6-v2 (384D) + MLP")

    # Extract/load embeddings
    pool_size = min(25000, max(n * 2, 15000))
    print(f"\n  Preparing embedding pool (n={pool_size})...")
    emb_pool = extract_embeddings(pool_size)
    z_pool, _ = extract_pca_features(emb_pool, n_components=5)
    print(f"  Pool: embeddings={emb_pool.shape}, z={z_pool.shape}")

    # Logit loss
    def logit_loss(y, t, theta):
        eta = theta[0] + theta[1] * t
        return torch.log(1 + torch.exp(eta)) - y * eta

    def beta_target(x, theta, t_tilde):
        return theta[1]

    mu_hats = []
    ses = []

    for m in range(M):
        seed = 3000 + m
        Y, T, X, _ = generate_data(emb_pool, z_pool, n, seed)

        if verbose:
            print(f"\n  Rep {m+1}/{M} (seed={seed})...")

        # No network_factory needed — standard MLP on 384-dim embeddings
        result = inference(
            Y=Y, T=T, X=X,
            loss=logit_loss,
            theta_dim=2,
            hessian_depends_on_theta=True,
            target_fn=beta_target,
            n_folds=n_folds,
            epochs=epochs,
            patience=patience,
            hidden_dims=[64, 32],
            lr=0.01,
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
    report = print_coverage_report(metrics, "SIM 3: SENTENCE-BERT + LOGIT", MU_TRUE)

    return metrics, report


def main():
    parser = argparse.ArgumentParser(description="Sim 3: Text + Logit")
    parser.add_argument("--M", type=int, default=20, help="MC replications")
    parser.add_argument("--n", type=int, default=10000, help="Sample size")
    parser.add_argument("--n-folds", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: M=5, n=5000")
    args = parser.parse_args()

    if args.quick:
        args.M = 5
        args.n = 5000
        args.epochs = 100

    report_dir = Path(__file__).parent.parent / "evals" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"sim_03_text_{timestamp}.txt"

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
