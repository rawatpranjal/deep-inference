"""
Data Preparation: H&M Transactions → (Y, T, X) for deep_inference

This script loads pre-trained two-tower embeddings and H&M transaction data,
constructs choice sets with sampled alternatives, and produces the data
arrays needed by the inference() API.

Pipeline:
    1. Load pre-trained embeddings from artifacts_arm_a/
    2. Load H&M transactions, filter to online dresses 2019-2020
    3. PCA on item embeddings (64D → 5D)
    4. For each purchase occasion, sample J-1 non-chosen alternatives
    5. Pack attributes: T_i = [x_0, x_1, ..., x_{J-1}] where x_j = (log_price, pca_1,...,pca_5)
    6. Save: Y.npy, T.npy, X.npy → application/data/

Outputs:
    data/Y.npy  — (n,) float32, always 0 (chosen alternative is first)
    data/T.npy  — (n, J*K) float32, packed alternative attributes
    data/X.npy  — (n, d_x) float32, consumer embeddings

Usage:
    python application/00_prep_data.py
    python application/00_prep_data.py --j 10 --pca-dims 3  # smaller choice sets
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

# Configuration
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    J, K, PCA_DIMS, MIN_PURCHASES,
    EMBEDDINGS_DIR, DATA_DIR,
)


def load_embeddings():
    """Load pre-trained two-tower embeddings and reference prices."""
    user_emb = np.load(EMBEDDINGS_DIR / "user_embeddings.npy")
    item_emb = np.load(EMBEDDINGS_DIR / "item_embeddings.npy")
    ref_prices = np.load(EMBEDDINGS_DIR / "ref_prices.npy")

    print(f"  User embeddings:  {user_emb.shape}")
    print(f"  Item embeddings:  {item_emb.shape}")
    print(f"  Reference prices: {ref_prices.shape}")
    print(f"  Price range:      [{ref_prices.min():.2f}, {ref_prices.max():.2f}]")

    return user_emb, item_emb, ref_prices


def load_transactions():
    """Load H&M transaction data.

    Looks for preprocessed data in the deep-aesthetics pipeline.
    Falls back to creating synthetic transactions from embeddings.
    """
    # Try loading from deep-aesthetics choice model prep
    choice_dir = Path(os.environ.get(
        "CHOICE_DATA_DIR",
        "/Users/pranjal/Dropbox/deep-aesthetics/final-analysis/choice"
    ))

    prep_file = choice_dir / "prep_cache.npz"
    if prep_file.exists():
        data = np.load(prep_file, allow_pickle=True)
        print(f"  Loaded prep cache from {prep_file}")
        return data

    # If no prep cache, we need the raw transaction data
    # Look for the user-item mapping
    user_items_file = choice_dir / "user_item_matrix.npz"
    if user_items_file.exists():
        data = np.load(user_items_file, allow_pickle=True)
        print(f"  Loaded user-item matrix from {user_items_file}")
        return data

    print("  WARNING: No transaction data found. Using synthetic transactions.")
    return None


def pca_embeddings(item_emb, n_components=PCA_DIMS):
    """Reduce item embedding dimension via PCA.

    Args:
        item_emb: (n_items, 64) item embeddings
        n_components: Number of PCA dimensions

    Returns:
        item_pca: (n_items, n_components) PCA-reduced embeddings
        pca: fitted PCA object
        variance_explained: fraction of variance explained
    """
    pca = PCA(n_components=n_components)
    item_pca = pca.fit_transform(item_emb)
    variance_explained = pca.explained_variance_ratio_.sum()

    print(f"  PCA: {item_emb.shape[1]}D → {n_components}D")
    print(f"  Variance explained: {variance_explained:.1%}")
    print(f"  Per-component: {pca.explained_variance_ratio_}")

    return item_pca, pca, variance_explained


def construct_choice_sets(
    user_emb, item_pca, ref_prices,
    n_alternatives=J,
    transactions=None,
    seed=42,
):
    """Construct choice occasions with sampled alternatives.

    For each purchase occasion:
      1. Chosen item attributes: [log_price, pca_1, ..., pca_5]
      2. Sample J-1 random alternatives from available items
      3. Pack T_i = concatenate all J items' attributes → (J*K,)
      4. Y_i = 0 (chosen is always first)
      5. X_i = consumer embedding

    Args:
        user_emb: (n_users, d_x) consumer embeddings
        item_pca: (n_items, pca_dims) PCA'd item embeddings
        ref_prices: (n_items,) reference prices
        n_alternatives: J, number of alternatives per occasion
        transactions: Optional preprocessed transaction data
        seed: Random seed for alternative sampling

    Returns:
        Y: (n,) float32, always 0
        T: (n, J*K) float32, packed attributes
        X: (n, d_x) float32, consumer embeddings
    """
    rng = np.random.RandomState(seed)
    n_users = user_emb.shape[0]
    n_items = item_pca.shape[0]
    K = item_pca.shape[1] + 1  # pca_dims + log_price

    # Build item attribute matrix: [log_price, pca_1, ..., pca_K-1]
    log_prices = np.log(ref_prices + 1e-6)
    item_attributes = np.column_stack([log_prices, item_pca])  # (n_items, K)
    print(f"  Item attribute matrix: {item_attributes.shape}")

    if transactions is not None:
        # Use real transaction data
        print("  Using real transaction data for choice sets")
        # Extract user_ids, chosen_item_ids from transactions
        # This depends on the format of the prep cache
        if hasattr(transactions, 'files'):
            # npz format
            if 'user_ids' in transactions.files and 'item_ids' in transactions.files:
                user_ids = transactions['user_ids']
                chosen_ids = transactions['item_ids']
            else:
                print("  WARNING: Unrecognized transaction format, falling back to synthetic")
                transactions = None

    if transactions is None:
        # Create synthetic transactions: each user purchases ~5-20 items
        print("  Generating synthetic purchase occasions")
        user_ids = []
        chosen_ids = []

        # Sample a subset of users (not all 241K)
        n_sample_users = min(5000, n_users)
        sampled_users = rng.choice(n_users, n_sample_users, replace=False)

        for u in sampled_users:
            # Each user makes 5-20 purchases
            n_purchases = rng.randint(MIN_PURCHASES, 20)
            items = rng.choice(n_items, n_purchases, replace=False)
            for item in items:
                user_ids.append(u)
                chosen_ids.append(item)

        user_ids = np.array(user_ids)
        chosen_ids = np.array(chosen_ids)

    n_occasions = len(user_ids)
    print(f"  Purchase occasions: {n_occasions}")
    print(f"  Unique users: {len(np.unique(user_ids))}")

    # Construct choice sets
    Y = np.zeros(n_occasions, dtype=np.float32)
    T = np.zeros((n_occasions, n_alternatives * K), dtype=np.float32)
    X = np.zeros((n_occasions, user_emb.shape[1]), dtype=np.float32)

    for i in range(n_occasions):
        user_id = user_ids[i]
        chosen_id = chosen_ids[i]

        # Consumer embedding
        X[i] = user_emb[user_id]

        # Chosen item attributes (position 0)
        T[i, :K] = item_attributes[chosen_id]

        # Sample J-1 non-chosen alternatives
        available = np.setdiff1d(np.arange(n_items), [chosen_id])
        sampled = rng.choice(available, n_alternatives - 1, replace=False)

        for j, item_id in enumerate(sampled):
            start = (j + 1) * K
            T[i, start:start + K] = item_attributes[item_id]

    print(f"  Final shapes: Y{Y.shape}, T{T.shape}, X{X.shape}")
    return Y, T, X


def main():
    parser = argparse.ArgumentParser(description="Prepare H&M data for inference")
    parser.add_argument("--j", type=int, default=J, help="Number of alternatives")
    parser.add_argument("--pca-dims", type=int, default=PCA_DIMS, help="PCA dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  DATA PREPARATION: H&M → (Y, T, X)")
    print("=" * 60)

    # Step 1: Load embeddings
    print("\n[1/4] Loading pre-trained embeddings...")
    user_emb, item_emb, ref_prices = load_embeddings()

    # Step 2: Load transactions
    print("\n[2/4] Loading transaction data...")
    transactions = load_transactions()

    # Step 3: PCA on item embeddings
    print(f"\n[3/4] PCA on item embeddings ({item_emb.shape[1]}D → {args.pca_dims}D)...")
    item_pca, pca_model, var_explained = pca_embeddings(item_emb, args.pca_dims)

    # Step 4: Construct choice sets
    print(f"\n[4/4] Constructing choice sets (J={args.j})...")
    Y, T, X = construct_choice_sets(
        user_emb, item_pca, ref_prices,
        n_alternatives=args.j,
        transactions=transactions,
        seed=args.seed,
    )

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "Y.npy", Y)
    np.save(DATA_DIR / "T.npy", T)
    np.save(DATA_DIR / "X.npy", X)

    # Also save PCA model and item attributes for counterfactuals
    np.save(DATA_DIR / "item_pca.npy", item_pca)
    np.save(DATA_DIR / "ref_prices.npy", ref_prices)
    np.save(DATA_DIR / "log_prices.npy", np.log(ref_prices + 1e-6))

    print(f"\n  Saved to: {DATA_DIR.resolve()}")
    print(f"    Y.npy:          {Y.shape}")
    print(f"    T.npy:          {T.shape}")
    print(f"    X.npy:          {X.shape}")
    print(f"    item_pca.npy:   {item_pca.shape}")
    print(f"    ref_prices.npy: {ref_prices.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
