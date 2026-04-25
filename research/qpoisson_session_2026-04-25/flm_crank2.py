"""
Second crank: explore DGP variations AFTER we know which IF fix is best.

This script reuses the best fix from flm_robustness.py and stress-tests it
against harder DGPs:
  G1: imbalanced T                P(T=1) ∈ {0.1, 0.3, 0.7, 0.9}
  G2: stronger nuisance           α*/β* scale × {0.5, 1.0, 1.5, 2.0}
  G3: noisier outcome             φ (overdispersion) ∈ {1.0, 2.0, 5.0, 10.0}
  G4: DNN capacity                hidden ∈ {(32,16), (64,32), (128,64), (256,128)}
  G5: Tanh vs ReLU matched to truth
  G6: continuous T                instead of binary

Goal: find which DGP characteristics degrade FML even with the best fix applied.
Output: diagnostic curves vs each nuisance parameter.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist

# reuse everything from flm_robustness
import sys
sys.path.insert(0, "/tmp/qpoisson_if")
from flm_robustness import (
    TrueNet, _build_truth, ThetaNet, poisson_nll, train_theta, pred,
    poisson_if, winsorize_psi, aggregate, tmle_one_step, cross_fit_psi,
    K_FOLDS, DNN_HIDDEN, DNN_LR, DNN_MAX_EPOCHS, DNN_PATIENCE, DNN_BATCH,
    cp_ci, summarize, M_MC, N_WORKERS,
)

# crank 1 winner: F1_trim_e010 (Λ-trim ε=0.1, no winsorize)
BEST_FIX = dict(trim_eps=0.10)


def simulate_with_phi(rng, n, d, p_t=0.5, phi=1.0, scale=1.0, truth=None):
    """Poisson (phi=1) or NB2 (phi>1).  scale multiplies α* and β* scales."""
    T = truth
    x = rng.standard_normal((n, d)).astype(np.float32)
    t = (rng.uniform(size=n) < p_t).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = T["alpha"](xt).numpy() * scale
        b = T["beta"](xt).numpy() * scale
    mu = np.exp(a + b * t).astype(np.float32)
    if phi <= 1.0:
        y = rng.poisson(mu).astype(np.float32)
    else:
        theta = mu / (phi - 1.0)
        p = theta / (theta + mu)
        y = rng.negative_binomial(theta, p).astype(np.float32)
    mu_star = float(T["beta"](torch.randn(500_000, d,
                    generator=torch.Generator().manual_seed(1234))).mean() * scale)
    return x, t, y, a, b, mu, mu_star


def run_worker(payload):
    group, label, dgp_params, mc_seed = payload
    rng = np.random.default_rng(mc_seed)
    truth = _build_truth(dgp_params["d"], seed=1001)
    x, t, y, a_obs, b_obs, mu_obs, mu_star = simulate_with_phi(
        rng, dgp_params["n"], dgp_params["d"],
        p_t=dgp_params.get("p_t", 0.5),
        phi=dgp_params.get("phi", 1.0),
        scale=dgp_params.get("scale", 1.0),
        truth=truth)
    # use BEST_FIX
    psi, beta_hat, alpha_hat, _, _ = cross_fit_psi(
        x, t, y, seed=mc_seed,
        trim_eps=BEST_FIX.get("trim_eps", 0.1),
        ensemble_r=BEST_FIX.get("ensemble_r", 1))
    if BEST_FIX.get("tmle_iters", 0) > 0:
        beta_hat = tmle_one_step(beta_hat, alpha_hat, t, y,
                                 trim_eps=BEST_FIX.get("trim_eps", 0.1),
                                 max_iter=BEST_FIX["tmle_iters"])
        psi, _, _ = poisson_if(alpha_hat, beta_hat, t, y,
                               trim_eps=BEST_FIX.get("trim_eps", 0.1))
    psi_w = winsorize_psi(psi, *BEST_FIX.get("winsor", (0.0, 1.0)))
    mu_hat = float(psi_w.mean())
    se = float(psi_w.std(ddof=1) / np.sqrt(len(psi_w)))
    return dict(group=group, label=label, mc_seed=mc_seed,
                mu_hat=mu_hat, se=se, mu_star=mu_star,
                covered=bool(abs(mu_hat - mu_star) <= 1.96 * se),
                mu_naive=float(beta_hat.mean()))


def main():
    N_BASE = 20_000
    D_BASE = 4
    M = 15
    payloads = []

    # G1: imbalanced T
    for p_t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for m in range(M):
            payloads.append(("G1_pT", f"p={p_t}",
                             dict(d=D_BASE, n=N_BASE, p_t=p_t), 7000 + m))

    # G2: scale (bigger α, β gives bigger mu range)
    for scale in [0.5, 1.0, 1.5, 2.0]:
        for m in range(M):
            payloads.append(("G2_scale", f"s={scale}",
                             dict(d=D_BASE, n=N_BASE, scale=scale), 8000 + m))

    # G3: overdispersion (NB2 with various phi)
    for phi in [1.0, 2.0, 5.0, 10.0]:
        for m in range(M):
            payloads.append(("G3_phi", f"phi={phi}",
                             dict(d=D_BASE, n=N_BASE, phi=phi), 9000 + m))

    print(f"[crank2] Total runs: {len(payloads)}")
    t0 = time.time()
    results_by_group = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_worker, p) for p in payloads]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            key = (r["group"], r["label"])
            results_by_group.setdefault(key, []).append(r)
            done += 1
            if done % 30 == 0:
                dt = time.time() - t0
                eta = dt * (len(futs) - done) / max(done, 1)
                print(f"  {done:4d}/{len(futs)}  t={dt:5.0f}s  eta={eta:.0f}s")

    # Report
    print("\n" + "=" * 90)
    print("CRANK 2: DGP stress curves".center(90))
    print("=" * 90)
    for grp_name in ["G1_pT", "G2_scale", "G3_phi"]:
        grp_rows = [(k[1], v) for k, v in results_by_group.items()
                    if k[0] == grp_name]
        grp_rows.sort(key=lambda r: r[0])
        print(f"\n--- {grp_name} ---")
        print(f"  {'level':<12}{'bias':>9}{'|b|/SE':>8}{'SEratio':>9}"
              f"{'cov':>7}{'CP-CI':>14}")
        for label, res in grp_rows:
            s = summarize(label, grp_name, res)
            print(f"  {label:<12}{s['bias']:>+9.4f}"
                  f"{s['abs_bias_over_se']:>8.3f}"
                  f"{s['se_ratio']:>9.3f}"
                  f"{s['coverage']:>7.3f}"
                  f" [{s['cp_lo']:.2f},{s['cp_hi']:.2f}]  {s['verdict']}")

    print(f"\n[crank2 total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
