"""
Crank 5: cluster-based subgroup averages.

Two estimators of per-cluster µ_c = E[β*(X) | X ∈ cluster_c]:

  (A) FML-subgroup
      1. Fit α̂(x), β̂(x) as a DNN over continuous X (K-fold cross-fit).
      2. Compute ψ_i (Poisson FML influence function with Λ-trim ε=0.10).
      3. For each cluster c:
            µ̂_c = mean(ψ_i | x_i ∈ cluster_c)
            SE_c = std(ψ_i | x_i ∈ cluster_c) / sqrt(|cluster_c|)

  (B) Saturated cluster-GLM
      For each cluster c separately:
        Fit Poisson GLM on {(x_i, t_i, y_i) | x_i ∈ cluster_c}
          log E[y|t] = a_c + b_c · t
        µ̂_c = b̂_c, SE_c from HC1 robust sandwich.

Clusters defined by K-means on a fixed reference pool (100k obs, seed 42),
K ∈ {5, 10, 20}.  Same centroids across MC reps so µ_c* is deterministic.

DGP: Poisson, d=4 continuous X, α*/β* from random Tanh nets (same truth as
prior cranks).  Binary T, p_T=0.5.  n=20,000, M_MC=30.

Headline output: per-cluster coverage for (A) and (B), averaged across MC
reps.  Also: which method has lower RMSE per cluster?
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist
from sklearn.cluster import KMeans
import statsmodels.api as sm

import sys
sys.path.insert(0, "/tmp/qpoisson_if")
from flm_robustness import (
    _build_truth, cross_fit_psi, poisson_if, cp_ci, N_WORKERS,
)


# ===========================================================
# config
# ===========================================================
D = 4
N = 20_000
M_MC = 30
K_VALUES = [5, 10, 20]
TRIM_EPS = 0.10
REF_POOL_SIZE = 100_000


# ===========================================================
# truth + K-means reference
# ===========================================================
TRUTH = _build_truth(D, seed=1001)

# reference pool to define stable cluster centroids across MC reps
_ref_rng = np.random.default_rng(42)
REF_X = _ref_rng.standard_normal((REF_POOL_SIZE, D)).astype(np.float32)
with torch.no_grad():
    REF_BETA = TRUTH["beta"](torch.from_numpy(REF_X)).numpy()

# fit K-means once per K and store the fitted object
KMEANS = {}
MU_C_STAR = {}    # truth per cluster
P_C = {}          # cluster prior probability
for K in K_VALUES:
    km = KMeans(n_clusters=K, random_state=42, n_init=5).fit(REF_X)
    KMEANS[K] = km
    labels = km.predict(REF_X)
    mu_c = np.array([REF_BETA[labels == c].mean() for c in range(K)])
    p_c = np.array([(labels == c).mean() for c in range(K)])
    MU_C_STAR[K] = mu_c
    P_C[K] = p_c
    print(f"[K={K:>2}]  cluster sizes (ref): "
          f"min={int(p_c.min()*REF_POOL_SIZE):>4}  "
          f"max={int(p_c.max()*REF_POOL_SIZE):>5}  "
          f"µ_c range=[{mu_c.min():+.3f}, {mu_c.max():+.3f}]  "
          f"std={mu_c.std():.3f}")


# ===========================================================
# DGP
# ===========================================================
def simulate(rng, n):
    x = rng.standard_normal((n, D)).astype(np.float32)
    t = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = TRUTH["alpha"](xt).numpy()
        b = TRUTH["beta"](xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return x, t, y, a, b, mu


# ===========================================================
# method A: FML-subgroup
# ===========================================================
def fml_subgroup(x, t, y, seed, K):
    """One DNN fit on continuous X, slice ψ by cluster for per-cluster ATEs."""
    psi, beta_hat, alpha_hat, mu_hat, _ = cross_fit_psi(
        x, t, y, seed=seed, trim_eps=TRIM_EPS, ensemble_r=1)
    labels = KMEANS[K].predict(x)
    results = {}
    for c in range(K):
        m = labels == c
        nc = int(m.sum())
        if nc < 5:
            results[c] = dict(mu_hat=np.nan, se=np.nan, n=nc)
            continue
        mu_c_hat = float(psi[m].mean())
        se_c = float(psi[m].std(ddof=1) / np.sqrt(nc))
        results[c] = dict(mu_hat=mu_c_hat, se=se_c, n=nc)
    return results


# ===========================================================
# method B: saturated cluster-Poisson
# ===========================================================
def saturated_cluster_glm(x, t, y, K):
    """Fit a separate Poisson GLM per cluster.  Returns b̂_c and HC1 SE."""
    labels = KMEANS[K].predict(x)
    results = {}
    for c in range(K):
        m = labels == c
        nc = int(m.sum())
        if nc < 10:
            results[c] = dict(mu_hat=np.nan, se=np.nan, n=nc)
            continue
        y_c = y[m].astype(float)
        t_c = t[m].astype(float)
        X_c = sm.add_constant(t_c)
        try:
            model = sm.GLM(y_c, X_c, family=sm.families.Poisson())
            res = model.fit(cov_type="HC1", maxiter=100)
            b_hat = float(res.params[1])
            se_b = float(res.bse[1])
        except Exception:
            b_hat = np.nan; se_b = np.nan
        results[c] = dict(mu_hat=b_hat, se=se_b, n=nc)
    return results


# ===========================================================
# MC worker
# ===========================================================
def run_one(payload):
    mc_seed, K = payload
    rng = np.random.default_rng(mc_seed)
    x, t, y, _, _, _ = simulate(rng, N)
    fml_res = fml_subgroup(x, t, y, seed=mc_seed, K=K)
    sat_res = saturated_cluster_glm(x, t, y, K=K)
    out = dict(mc_seed=mc_seed, K=K)
    for c in range(K):
        out[f"fml_c{c}_mu"] = fml_res[c]["mu_hat"]
        out[f"fml_c{c}_se"] = fml_res[c]["se"]
        out[f"fml_c{c}_n"] = fml_res[c]["n"]
        out[f"sat_c{c}_mu"] = sat_res[c]["mu_hat"]
        out[f"sat_c{c}_se"] = sat_res[c]["se"]
        out[f"sat_c{c}_n"] = sat_res[c]["n"]
    return out


# ===========================================================
# main
# ===========================================================
def main():
    t0 = time.time()
    payloads = []
    for K in K_VALUES:
        for m in range(M_MC):
            payloads.append((15000 + m * 31, K))
    print(f"\n[crank5] {len(payloads)} runs")

    results_by_K = {K: [] for K in K_VALUES}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_one, p) for p in payloads]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            results_by_K[r["K"]].append(r)
            done += 1
            if done % 10 == 0:
                dt = time.time() - t0
                print(f"  {done:3d}/{len(payloads)}  t={dt:5.0f}s  "
                      f"eta={dt*(len(payloads)-done)/max(done,1):.0f}s")

    print("\n" + "=" * 110)
    print("CRANK 5: PER-CLUSTER SUBGROUP ATEs — FML-subgroup vs Saturated GLM"
          .center(110))
    print("=" * 110)

    all_summary = {}
    for K in K_VALUES:
        reps = results_by_K[K]
        print(f"\n--- K={K} clusters, M={len(reps)} reps ---")
        mu_c_star = MU_C_STAR[K]
        p_c = P_C[K]

        # For each cluster: compute per-method coverage, bias, SE ratio
        print(f"  {'c':>3}{'n/rep':>8}{'µ_c*':>10}  "
              f"{'FML bias':>10}{'FML SEr':>9}{'FML cov':>9}  "
              f"{'SAT bias':>10}{'SAT SEr':>9}{'SAT cov':>9}  "
              f"{'FML<SAT (rmse)':>17}")

        cluster_stats = []
        for c in range(K):
            fml_mu = np.array([r[f"fml_c{c}_mu"] for r in reps])
            fml_se = np.array([r[f"fml_c{c}_se"] for r in reps])
            fml_n = np.array([r[f"fml_c{c}_n"] for r in reps])
            sat_mu = np.array([r[f"sat_c{c}_mu"] for r in reps])
            sat_se = np.array([r[f"sat_c{c}_se"] for r in reps])

            valid_f = ~np.isnan(fml_mu) & ~np.isnan(fml_se)
            valid_s = ~np.isnan(sat_mu) & ~np.isnan(sat_se)

            if valid_f.sum() < 3 or valid_s.sum() < 3:
                print(f"  {c:>3}  too few valid (skipping)")
                continue

            # FML
            fml_bias = float(fml_mu[valid_f].mean() - mu_c_star[c])
            fml_empSD = float(fml_mu[valid_f].std(ddof=1))
            fml_seR = float(fml_se[valid_f].mean() / fml_empSD) if fml_empSD > 0 else np.nan
            fml_cov = float(np.mean(
                np.abs(fml_mu[valid_f] - mu_c_star[c]) <= 1.96 * fml_se[valid_f]))
            fml_rmse = float(np.sqrt(((fml_mu[valid_f] - mu_c_star[c]) ** 2).mean()))

            # saturated
            sat_bias = float(sat_mu[valid_s].mean() - mu_c_star[c])
            sat_empSD = float(sat_mu[valid_s].std(ddof=1))
            sat_seR = float(sat_se[valid_s].mean() / sat_empSD) if sat_empSD > 0 else np.nan
            sat_cov = float(np.mean(
                np.abs(sat_mu[valid_s] - mu_c_star[c]) <= 1.96 * sat_se[valid_s]))
            sat_rmse = float(np.sqrt(((sat_mu[valid_s] - mu_c_star[c]) ** 2).mean()))

            fml_wins = "YES" if fml_rmse < sat_rmse else "no"
            print(f"  {c:>3}{int(fml_n.mean()):>8}{mu_c_star[c]:>+10.4f}  "
                  f"{fml_bias:>+10.4f}{fml_seR:>9.2f}{fml_cov:>9.3f}  "
                  f"{sat_bias:>+10.4f}{sat_seR:>9.2f}{sat_cov:>9.3f}  "
                  f"{fml_rmse:.4f} vs {sat_rmse:.4f} {fml_wins:>6}")

            cluster_stats.append(dict(
                K=K, c=c, mu_star=mu_c_star[c],
                p_c=float(p_c[c]),
                fml_bias=fml_bias, fml_emp_sd=fml_empSD,
                fml_se_ratio=fml_seR, fml_cov=fml_cov, fml_rmse=fml_rmse,
                sat_bias=sat_bias, sat_emp_sd=sat_empSD,
                sat_se_ratio=sat_seR, sat_cov=sat_cov, sat_rmse=sat_rmse,
                fml_wins_rmse=fml_rmse < sat_rmse,
                mean_cluster_n=float(fml_n.mean()),
            ))

        # summary across clusters
        if not cluster_stats:
            continue
        fml_cov_avg = np.mean([s["fml_cov"] for s in cluster_stats])
        sat_cov_avg = np.mean([s["sat_cov"] for s in cluster_stats])
        fml_rmse_avg = np.mean([s["fml_rmse"] for s in cluster_stats])
        sat_rmse_avg = np.mean([s["sat_rmse"] for s in cluster_stats])
        fml_wins_frac = np.mean([s["fml_wins_rmse"] for s in cluster_stats])
        print(f"\n  [K={K} summary across {len(cluster_stats)} clusters]")
        print(f"    avg FML coverage = {fml_cov_avg:.3f}   "
              f"avg SAT coverage = {sat_cov_avg:.3f}")
        print(f"    avg FML RMSE    = {fml_rmse_avg:.4f}  "
              f"avg SAT RMSE    = {sat_rmse_avg:.4f}  "
              f"RMSE ratio FML/SAT = {fml_rmse_avg/sat_rmse_avg:.3f}")
        print(f"    FML wins on RMSE in {int(fml_wins_frac*len(cluster_stats))}/"
              f"{len(cluster_stats)} clusters ({fml_wins_frac:.0%})")
        all_summary[K] = cluster_stats

    # save
    with open("/tmp/qpoisson_if/flm_crank5_results.json", "w") as f:
        json.dump({str(K): v for K, v in all_summary.items()}, f, indent=1,
                  default=str)
    print(f"\n[saved] /tmp/qpoisson_if/flm_crank5_results.json")
    print(f"[crank5 total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
