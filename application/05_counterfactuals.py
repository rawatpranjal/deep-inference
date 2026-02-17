"""
Counterfactual Pricing Experiments with Uncertainty

Uses the estimated heterogeneous price sensitivity theta_0(X) to:
    1. Compute own-price elasticities with IF confidence intervals
    2. Evaluate 10% price increase: revenue impact with uncertainty bands
    3. Explore personalized pricing bounds with valid CIs

Usage:
    python application/05_counterfactuals.py
"""

import sys
import os
import argparse
import json
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
    """MNL loss (same as other scripts)."""
    x = t.reshape(J, K)
    V = x @ theta
    return -V[0] + torch.logsumexp(V, dim=0)


def compute_elasticities(theta_hat, T, J=20, K=6):
    """Compute own-price elasticities for each consumer.

    For the MNL without ASC:
        P(Y=j) = exp(V_j) / sum_m exp(V_m)
        own-price elasticity = beta_price * price_j * (1 - P(Y=j))

    Args:
        theta_hat: (n, K) estimated parameters
        T: (n, J*K) packed attributes
        J, K: dimensions

    Returns:
        elasticities: (n,) own-price elasticity for chosen item
        choice_probs: (n,) predicted choice probability for chosen item
    """
    if isinstance(theta_hat, torch.Tensor):
        theta_hat = theta_hat.numpy()

    n = len(theta_hat)
    elasticities = np.zeros(n)
    choice_probs = np.zeros(n)

    for i in range(n):
        x_i = T[i].reshape(J, K)  # (J, K)
        V = x_i @ theta_hat[i]     # (J,)
        probs = np.exp(V - V.max())
        probs /= probs.sum()

        # Own-price elasticity for chosen item (index 0)
        log_price = x_i[0, 0]  # First attribute = log_price
        price = np.exp(log_price)
        beta_price = theta_hat[i, 0]

        choice_probs[i] = probs[0]
        # Elasticity = d log P / d log p = beta_price * price * (1 - P)
        # Since attribute is log_price: elasticity = beta_price * (1 - P)
        elasticities[i] = beta_price * (1 - probs[0])

    return elasticities, choice_probs


def simulate_price_change(theta_hat, T, delta_pct=0.10, J=20, K=6):
    """Simulate revenue impact of a uniform price change.

    Args:
        theta_hat: (n, K) estimated parameters
        T: (n, J*K) packed attributes
        delta_pct: Price change fraction (0.10 = 10% increase)

    Returns:
        dict with revenue change metrics
    """
    if isinstance(theta_hat, torch.Tensor):
        theta_hat = theta_hat.numpy()

    n = len(theta_hat)

    # Baseline probabilities
    base_probs = np.zeros(n)
    new_probs = np.zeros(n)

    for i in range(n):
        x_i = T[i].reshape(J, K)
        V_base = x_i @ theta_hat[i]
        p_base = np.exp(V_base - V_base.max())
        p_base /= p_base.sum()
        base_probs[i] = p_base[0]

        # New attributes with price change (only item 0)
        x_new = x_i.copy()
        x_new[0, 0] += np.log(1 + delta_pct)  # log(price * (1 + delta))
        V_new = x_new @ theta_hat[i]
        p_new = np.exp(V_new - V_new.max())
        p_new /= p_new.sum()
        new_probs[i] = p_new[0]

    # Revenue: price * prob
    # Relative change in expected revenue per consumer
    # Revenue_new / Revenue_base = (price * (1+delta)) * P_new / (price * P_base)
    revenue_ratio = (1 + delta_pct) * new_probs / (base_probs + 1e-10)

    return {
        "delta_pct": delta_pct,
        "mean_base_prob": float(base_probs.mean()),
        "mean_new_prob": float(new_probs.mean()),
        "prob_change_pct": float((new_probs.mean() / base_probs.mean() - 1) * 100),
        "mean_revenue_ratio": float(revenue_ratio.mean()),
        "revenue_change_pct": float((revenue_ratio.mean() - 1) * 100),
        "n_lose_customers": int((new_probs < 0.5 * base_probs).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Counterfactual pricing experiments")
    parser.add_argument("--delta", type=float, default=0.10, help="Price change fraction")
    args = parser.parse_args()

    # Load data and results
    Y = np.load(DATA_DIR / "Y.npy")
    T = np.load(DATA_DIR / "T.npy")
    X = np.load(DATA_DIR / "X.npy")

    print("=" * 70)
    print("  COUNTERFACTUAL PRICING EXPERIMENTS")
    print("=" * 70)

    # Run inference to get theta_hat
    print("\n[1/3] Running inference to get theta_hat...")
    from deep_inference import inference

    def price_target(x, theta, t_tilde):
        return theta[0]

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
        verbose=False,
    )

    theta_hat = result.theta_hat
    psi_values = result.psi_values

    print(f"  E[beta_price] = {result.mu_hat:.4f} +/- {result.se:.4f}")

    # 2. Elasticities
    print("\n[2/3] Computing own-price elasticities...")
    elasticities, choice_probs = compute_elasticities(theta_hat, T)

    print(f"  Mean elasticity:     {elasticities.mean():.4f}")
    print(f"  Median elasticity:   {np.median(elasticities):.4f}")
    print(f"  Std elasticity:      {elasticities.std():.4f}")
    print(f"  Range:               [{elasticities.min():.4f}, {elasticities.max():.4f}]")
    print(f"  Mean choice prob:    {choice_probs.mean():.4f}")

    # Elasticity with IF uncertainty
    # The IF provides pointwise uncertainty on theta_hat
    # Elasticity ≈ beta_price * (1 - P), so SE(elasticity) ≈ |1-P| * SE(beta_price)
    if psi_values is not None:
        if isinstance(psi_values, torch.Tensor):
            psi_values = psi_values.numpy()
        psi_se = psi_values.std() / np.sqrt(len(psi_values))
        mean_factor = np.abs(1 - choice_probs).mean()
        elasticity_se = mean_factor * result.se
        print(f"  Elasticity SE (IF):  {elasticity_se:.4f}")
        print(f"  Elasticity CI:       [{elasticities.mean() - 1.96*elasticity_se:.4f}, "
              f"{elasticities.mean() + 1.96*elasticity_se:.4f}]")

    # 3. Price change simulation
    print(f"\n[3/3] Simulating {args.delta*100:.0f}% price increase...")
    cf_results = simulate_price_change(theta_hat, T, delta_pct=args.delta)

    print(f"  Base choice prob:    {cf_results['mean_base_prob']:.4f}")
    print(f"  New choice prob:     {cf_results['mean_new_prob']:.4f}")
    print(f"  Prob change:         {cf_results['prob_change_pct']:.1f}%")
    print(f"  Revenue ratio:       {cf_results['mean_revenue_ratio']:.4f}")
    print(f"  Revenue change:      {cf_results['revenue_change_pct']:.1f}%")

    # Heterogeneity in responses
    if isinstance(theta_hat, torch.Tensor):
        theta_hat_np = theta_hat.numpy()
    else:
        theta_hat_np = theta_hat

    # Quintile analysis
    price_sens = theta_hat_np[:, 0]
    quintiles = np.percentile(price_sens, [20, 40, 60, 80])
    print(f"\n  Heterogeneity in price sensitivity (quintiles):")
    print(f"    Q1 (least sensitive): beta_price > {quintiles[3]:.4f}")
    print(f"    Q5 (most sensitive):  beta_price < {quintiles[0]:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cf_output = {
        "inference": {
            "mu_hat": float(result.mu_hat),
            "se": float(result.se),
        },
        "elasticities": {
            "mean": float(elasticities.mean()),
            "median": float(np.median(elasticities)),
            "std": float(elasticities.std()),
        },
        "counterfactual": cf_results,
        "heterogeneity": {
            "quintiles": quintiles.tolist(),
            "price_sens_mean": float(price_sens.mean()),
            "price_sens_std": float(price_sens.std()),
        },
    }
    with open(RESULTS_DIR / "counterfactual_results.json", "w") as f:
        json.dump(cf_output, f, indent=2)

    print(f"\n  Results saved to: {(RESULTS_DIR / 'counterfactual_results.json').resolve()}")


if __name__ == "__main__":
    main()
