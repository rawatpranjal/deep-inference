"""
Data Preparation: H&M Transactions -> (Y, T, X) for deep_inference

Loads real H&M purchase data from the deep-aesthetics pipeline
(choice/artifacts_arm_a/prep/) and constructs the arrays needed
by inference() with a custom MNL loss.

Data sources:
    purchases_train.parquet  — 10,681 real purchase occasions
    maps.pkl                 — cust_to_i / sku_to_j index mappings
    d_matrix.npy             — (4833, 64) user embeddings
    x_matrix.npy             — (83, 64) item embeddings
    price_jt.npz             — (83, 89) time-varying price panel

Pipeline:
    1. Load real purchases + index mappings from prep/
    2. Load user/item embeddings from prep/
    3. PCA on item embeddings (64D -> 5D)
    4. For each purchase: pack chosen + J-1 sampled alternatives
    5. Save: Y.npy, T.npy, X.npy -> application/data/

Outputs:
    data/Y.npy  — (n,) float32, always 0 (chosen alternative is first)
    data/T.npy  — (n, J*K) float32, packed alternative attributes
    data/X.npy  — (n, d_x) float32, consumer embeddings

Usage:
    python application/00_prep_data.py
    python application/00_prep_data.py --j 10 --pca-dims 3
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    J, K, PCA_DIMS, CHOICE_PREP_DIR, DATA_DIR,
)


def load_real_data():
    """Load real H&M purchases, embeddings, and mappings from prep directory."""

    print(f"  Source: {CHOICE_PREP_DIR}")

    # Load purchases
    purchases = pd.read_parquet(CHOICE_PREP_DIR / "purchases_train.parquet")
    print(f"  Purchases:     {purchases.shape[0]:,} occasions")
    print(f"  Columns:       {list(purchases.columns)}")
    print(f"  Price range:   [{purchases['price_euros'].min():.2f}, {purchases['price_euros'].max():.2f}] EUR")

    # Load index mappings
    with open(CHOICE_PREP_DIR / "maps.pkl", "rb") as f:
        maps = pickle.load(f)
    cust_to_i = maps['cust_to_i']
    sku_to_j = maps['sku_to_j']
    print(f"  Customers:     {len(cust_to_i):,} (mapped to 0..{len(cust_to_i)-1})")
    print(f"  Articles:      {len(sku_to_j):,} (mapped to 0..{len(sku_to_j)-1})")

    # Load embeddings (from prep dir — already subsampled to choice set)
    d_matrix = np.load(CHOICE_PREP_DIR / "d_matrix.npy")
    x_matrix = np.load(CHOICE_PREP_DIR / "x_matrix.npy")
    print(f"  User emb:      {d_matrix.shape}")
    print(f"  Item emb:      {x_matrix.shape}")

    # Load time-varying prices for reference prices of non-chosen alternatives
    price_data = np.load(CHOICE_PREP_DIR / "price_jt.npz", allow_pickle=True)
    price_panel = price_data['p']  # (83, 89) with NaNs
    ref_prices = np.nanmedian(price_panel, axis=1)  # median price per item
    print(f"  Price panel:   {price_panel.shape} (items x weeks)")
    print(f"  Ref prices:    [{ref_prices.min():.2f}, {ref_prices.max():.2f}] EUR (median)")

    return purchases, cust_to_i, sku_to_j, d_matrix, x_matrix, ref_prices


def pca_embeddings(item_emb, n_components=PCA_DIMS):
    """Reduce item embedding dimension via PCA."""
    pca = PCA(n_components=n_components)
    item_pca = pca.fit_transform(item_emb)
    variance_explained = pca.explained_variance_ratio_.sum()

    print(f"  PCA: {item_emb.shape[1]}D -> {n_components}D")
    print(f"  Variance explained: {variance_explained:.1%}")
    print(f"  Per-component: {pca.explained_variance_ratio_}")

    return item_pca, pca, variance_explained


def construct_choice_sets(
    purchases, cust_to_i, sku_to_j, d_matrix, x_matrix_pca, ref_prices,
    n_alternatives=J, seed=42,
):
    """Construct choice occasions with sampled alternatives from real purchases.

    For each purchase occasion:
      1. Look up consumer embedding via cust_to_i mapping
      2. Chosen item: [log(actual_price), pca_1, ..., pca_5]
      3. Sample J-1 non-chosen alternatives from the 83 items
      4. Non-chosen items: [log(ref_price), pca_1, ..., pca_5]
      5. Pack T_i = [chosen_attrs, alt1_attrs, ..., alt_{J-1}_attrs]
      6. Y_i = 0 (chosen is always first)
    """
    rng = np.random.RandomState(seed)
    n_items = x_matrix_pca.shape[0]
    K = x_matrix_pca.shape[1] + 1  # pca_dims + log_price

    # Pre-compute item attributes using reference prices (for non-chosen items)
    log_ref_prices = np.log(ref_prices + 1e-6)
    item_attributes = np.column_stack([log_ref_prices, x_matrix_pca])  # (83, K)
    print(f"  Item attribute matrix: {item_attributes.shape}")

    n_occasions = len(purchases)
    all_items = np.arange(n_items)

    Y = np.zeros(n_occasions, dtype=np.float32)
    T = np.zeros((n_occasions, n_alternatives * K), dtype=np.float32)
    X = np.zeros((n_occasions, d_matrix.shape[1]), dtype=np.float32)

    n_skipped = 0
    valid_indices = []

    for i in range(n_occasions):
        row = purchases.iloc[i]
        cust_id = int(row['customer_id_idx'])
        art_id = int(row['article_id_idx'])
        price = float(row['price_euros'])

        # Map to contiguous indices
        if cust_id not in cust_to_i or art_id not in sku_to_j:
            n_skipped += 1
            continue

        cust_i = cust_to_i[cust_id]
        item_j = sku_to_j[art_id]

        # Consumer embedding
        X[len(valid_indices)] = d_matrix[cust_i]

        # Chosen item: use ACTUAL transaction price
        chosen_attrs = np.concatenate([
            [np.log(price + 1e-6)],
            x_matrix_pca[item_j],
        ])
        T[len(valid_indices), :K] = chosen_attrs

        # Sample J-1 non-chosen alternatives
        available = np.setdiff1d(all_items, [item_j])
        sampled = rng.choice(available, n_alternatives - 1, replace=False)

        for slot, alt_j in enumerate(sampled):
            start = (slot + 1) * K
            T[len(valid_indices), start:start + K] = item_attributes[alt_j]

        valid_indices.append(i)

    n_valid = len(valid_indices)
    if n_skipped > 0:
        print(f"  WARNING: Skipped {n_skipped} occasions with unmapped IDs")

    # Trim to valid occasions
    Y = Y[:n_valid]
    T = T[:n_valid]
    X = X[:n_valid]

    print(f"  Valid occasions: {n_valid:,}")
    print(f"  Final shapes: Y{Y.shape}, T{T.shape}, X{X.shape}")

    return Y, T, X


def main():
    parser = argparse.ArgumentParser(description="Prepare H&M data for inference")
    parser.add_argument("--j", type=int, default=J, help="Number of alternatives")
    parser.add_argument("--pca-dims", type=int, default=PCA_DIMS, help="PCA dimensions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  DATA PREPARATION: H&M -> (Y, T, X)")
    print("=" * 60)

    # Step 1: Load real data
    print("\n[1/3] Loading real H&M data...")
    purchases, cust_to_i, sku_to_j, d_matrix, x_matrix, ref_prices = load_real_data()

    # Step 2: PCA on item embeddings
    print(f"\n[2/3] PCA on item embeddings ({x_matrix.shape[1]}D -> {args.pca_dims}D)...")
    item_pca, pca_model, var_explained = pca_embeddings(x_matrix, args.pca_dims)

    # Step 3: Construct choice sets from real purchases
    print(f"\n[3/3] Constructing choice sets (J={args.j})...")
    Y, T, X = construct_choice_sets(
        purchases, cust_to_i, sku_to_j, d_matrix, item_pca, ref_prices,
        n_alternatives=args.j,
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

    print(f"\n  Saved to: {DATA_DIR.resolve()}")
    print(f"    Y.npy:          {Y.shape}  (all zeros — chosen is always first)")
    print(f"    T.npy:          {T.shape}  (J={args.j} x K={item_pca.shape[1]+1})")
    print(f"    X.npy:          {X.shape}  (64D user embeddings)")
    print(f"    item_pca.npy:   {item_pca.shape}")
    print(f"    ref_prices.npy: {ref_prices.shape}")

    # Summary statistics
    print(f"\n  === Data Summary ===")
    print(f"  Unique customers: {len(set(int(r['customer_id_idx']) for _, r in purchases.iterrows())):,}")
    print(f"  Unique items:     {x_matrix.shape[0]}")
    print(f"  Purchase events:  {Y.shape[0]:,}")
    print(f"  Chosen price range: [{np.exp(T[:, 0]).min():.2f}, {np.exp(T[:, 0]).max():.2f}] EUR")
    print(f"  Mean chosen price:  {np.exp(T[:, 0]).mean():.2f} EUR")
    print("=" * 60)


if __name__ == "__main__":
    main()
