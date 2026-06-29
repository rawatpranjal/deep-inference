"""
Crank 6: FML-subgroup on user × item cross cells (40-cell grid).

Mirrors the paper's cross_cell_lockdown analysis: K_user × K_item cells,
per-cell treatment effect, per-cell standard error.

Setup:
  z_u ∈ R^{d_u},  z_j ∈ R^{d_j}   iid standard normal
  X = concat(z_u, z_j) ∈ R^{d_u + d_j}
  α*(X), β*(X) = random fixed Tanh nets on X
  K-means separately on z_u and z_j using a fixed 100k reference pool
  cross-cell c = user_cluster · K_item + item_cluster   (40 cells)

Target per cell:
  µ_c* = E[β*(X) | user_cluster(z_u) = u  AND  item_cluster(z_j) = j]

Estimators:
  (A) FML-subgroup: one DNN on continuous X + ψ sliced by cross-cell.
  (B) Saturated per-cell Poisson QMLE with HC1 robust SE (paper's baseline).

DGP: n = 40,000, K_u = 10, K_j = 4, d_u = d_j = 4, M_MC = 20.
Poisson counts, binary T, p_T = 0.5.

Report:
  - Per-cell coverage and RMSE for both methods
  - Average across cells + per-cell details
  - Cell-size histogram
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
    _build_truth, cross_fit_psi, cp_ci, N_WORKERS,
)


# ===========================================================
# config
# ===========================================================
D_U = 4
D_J = 4
D_TOTAL = D_U + D_J
K_U = 10
K_J = 4
N_CELLS = K_U * K_J      # 40
N_DATA = 40_000
M_MC = 20
TRIM_EPS = 0.10
REF_POOL_SIZE = 100_000


# ===========================================================
# truth
# ===========================================================
TRUTH = _build_truth(D_TOTAL, seed=1001)


# ===========================================================
# K-means reference pool (stable cluster centroids across reps)
# ===========================================================
_ref_rng = np.random.default_rng(42)
REF_ZU = _ref_rng.standard_normal((REF_POOL_SIZE, D_U)).astype(np.float32)
REF_ZJ = _ref_rng.standard_normal((REF_POOL_SIZE, D_J)).astype(np.float32)
REF_X = np.concatenate([REF_ZU, REF_ZJ], axis=1)

with torch.no_grad():
    REF_BETA = TRUTH["beta"](torch.from_numpy(REF_X)).numpy()

KM_U = KMeans(n_clusters=K_U, random_state=42, n_init=5).fit(REF_ZU)
KM_J = KMeans(n_clusters=K_J, random_state=42, n_init=5).fit(REF_ZJ)

ref_lu = KM_U.predict(REF_ZU)
ref_lj = KM_J.predict(REF_ZJ)
ref_cell = ref_lu * K_J + ref_lj

MU_CELL_STAR = np.zeros(N_CELLS)
P_CELL = np.zeros(N_CELLS)
for c in range(N_CELLS):
    m = ref_cell == c
    if m.sum() > 0:
        MU_CELL_STAR[c] = REF_BETA[m].mean()
        P_CELL[c] = m.mean()

print(f"[setup] {N_CELLS} cells (K_U={K_U} × K_J={K_J})")
print(f"  cell size range (ref): "
      f"min={int(P_CELL[P_CELL > 0].min()*REF_POOL_SIZE):>4}  "
      f"max={int(P_CELL.max()*REF_POOL_SIZE):>5}")
print(f"  µ_c* range: [{MU_CELL_STAR.min():+.3f}, {MU_CELL_STAR.max():+.3f}]  "
      f"std={MU_CELL_STAR.std():.3f}")
print(f"  n_data = {N_DATA}, expected obs/cell = {N_DATA/N_CELLS:.0f}")


# ===========================================================
# DGP
# ===========================================================
def simulate(rng):
    z_u = rng.standard_normal((N_DATA, D_U)).astype(np.float32)
    z_j = rng.standard_normal((N_DATA, D_J)).astype(np.float32)
    x = np.concatenate([z_u, z_j], axis=1)
    t = rng.choice([0.0, 1.0], size=N_DATA).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = TRUTH["alpha"](xt).numpy()
        b = TRUTH["beta"](xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return z_u, z_j, x, t, y, a, b, mu


# ===========================================================
# FML-subgroup: one DNN fit, slice ψ per cross-cell
# ===========================================================
def fml_per_cell(x, t, y, z_u, z_j, seed):
    psi, beta_hat, *_ = cross_fit_psi(
        x, t, y, seed=seed, trim_eps=TRIM_EPS, ensemble_r=1)
    lu = KM_U.predict(z_u)
    lj = KM_J.predict(z_j)
    cell = lu * K_J + lj
    out = {}
    for c in range(N_CELLS):
        m = cell == c
        nc = int(m.sum())
        if nc < 5:
            out[c] = dict(mu_hat=np.nan, se=np.nan, n=nc)
            continue
        mu_hat = float(psi[m].mean())
        se = float(psi[m].std(ddof=1) / np.sqrt(nc))
        out[c] = dict(mu_hat=mu_hat, se=se, n=nc)
    return out


# ===========================================================
# Saturated per-cell Poisson QMLE with HC1
# ===========================================================
def saturated_per_cell(t, y, z_u, z_j):
    lu = KM_U.predict(z_u)
    lj = KM_J.predict(z_j)
    cell = lu * K_J + lj
    out = {}
    for c in range(N_CELLS):
        m = cell == c
        nc = int(m.sum())
        if nc < 10:
            out[c] = dict(mu_hat=np.nan, se=np.nan, n=nc)
            continue
        y_c = y[m].astype(float)
        t_c = t[m].astype(float)
        X_c = sm.add_constant(t_c)
        try:
            res = sm.GLM(y_c, X_c, family=sm.families.Poisson()).fit(
                cov_type="HC1", maxiter=100)
            b_hat = float(res.params[1])
            se_b = float(res.bse[1])
        except Exception:
            b_hat = np.nan; se_b = np.nan
        out[c] = dict(mu_hat=b_hat, se=se_b, n=nc)
    return out


# ===========================================================
# MC worker
# ===========================================================
def run_rep(mc_seed):
    rng = np.random.default_rng(mc_seed)
    z_u, z_j, x, t, y, _, _, _ = simulate(rng)
    fml_res = fml_per_cell(x, t, y, z_u, z_j, seed=mc_seed)
    sat_res = saturated_per_cell(t, y, z_u, z_j)
    rec = dict(mc_seed=mc_seed)
    for c in range(N_CELLS):
        rec[f"fml_c{c}_mu"] = fml_res[c]["mu_hat"]
        rec[f"fml_c{c}_se"] = fml_res[c]["se"]
        rec[f"fml_c{c}_n"] = fml_res[c]["n"]
        rec[f"sat_c{c}_mu"] = sat_res[c]["mu_hat"]
        rec[f"sat_c{c}_se"] = sat_res[c]["se"]
        rec[f"sat_c{c}_n"] = sat_res[c]["n"]
    return rec


# ===========================================================
# main
# ===========================================================
def main():
    t0 = time.time()
    print(f"\n[crank6] launching {M_MC} MC reps...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_rep, 16000 + m * 31) for m in range(M_MC)]
        done = 0
        results = []
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            if done % 2 == 0:
                print(f"  {done}/{M_MC}  t={time.time()-t0:.0f}s")

    # aggregate per cell
    per_cell = []
    for c in range(N_CELLS):
        fml_mu = np.array([r[f"fml_c{c}_mu"] for r in results])
        fml_se = np.array([r[f"fml_c{c}_se"] for r in results])
        fml_n = np.array([r[f"fml_c{c}_n"] for r in results])
        sat_mu = np.array([r[f"sat_c{c}_mu"] for r in results])
        sat_se = np.array([r[f"sat_c{c}_se"] for r in results])

        vf = ~np.isnan(fml_mu) & ~np.isnan(fml_se)
        vs = ~np.isnan(sat_mu) & ~np.isnan(sat_se)
        if vf.sum() < 3:
            continue

        fml_bias = float(fml_mu[vf].mean() - MU_CELL_STAR[c])
        fml_empSD = float(fml_mu[vf].std(ddof=1))
        fml_seR = float(fml_se[vf].mean() / fml_empSD) if fml_empSD > 0 else np.nan
        fml_cov = float(np.mean(
            np.abs(fml_mu[vf] - MU_CELL_STAR[c]) <= 1.96 * fml_se[vf]))
        fml_rmse = float(np.sqrt(((fml_mu[vf] - MU_CELL_STAR[c]) ** 2).mean()))

        if vs.sum() >= 3:
            sat_bias = float(sat_mu[vs].mean() - MU_CELL_STAR[c])
            sat_empSD = float(sat_mu[vs].std(ddof=1))
            sat_seR = float(sat_se[vs].mean() / sat_empSD) if sat_empSD > 0 else np.nan
            sat_cov = float(np.mean(
                np.abs(sat_mu[vs] - MU_CELL_STAR[c]) <= 1.96 * sat_se[vs]))
            sat_rmse = float(np.sqrt(((sat_mu[vs] - MU_CELL_STAR[c]) ** 2).mean()))
        else:
            sat_bias = sat_seR = sat_cov = sat_rmse = np.nan

        uc, jc = divmod(c, K_J)
        per_cell.append(dict(
            c=c, uc=uc, jc=jc, mu_star=MU_CELL_STAR[c],
            p_c=float(P_CELL[c]), n_mean=float(fml_n.mean()),
            fml_bias=fml_bias, fml_cov=fml_cov, fml_rmse=fml_rmse,
            fml_seR=fml_seR, fml_empSD=fml_empSD,
            sat_bias=sat_bias, sat_cov=sat_cov, sat_rmse=sat_rmse,
            sat_seR=sat_seR, sat_empSD=sat_empSD,
        ))

    # -------- headline --------
    print(f"\n{'=' * 100}")
    print(f"CRANK 6: {K_U}×{K_J} = {N_CELLS} cross cells, M={M_MC} reps"
          .center(100))
    print(f"{'=' * 100}\n")

    fml_cov_avg = np.mean([p["fml_cov"] for p in per_cell])
    sat_cov_avg = np.nanmean([p["sat_cov"] for p in per_cell])
    fml_rmse_avg = np.mean([p["fml_rmse"] for p in per_cell])
    sat_rmse_avg = np.nanmean([p["sat_rmse"] for p in per_cell])
    fml_bias_avg = np.mean([abs(p["fml_bias"]) for p in per_cell])
    sat_bias_avg = np.nanmean([abs(p["sat_bias"]) for p in per_cell])
    fml_seR_med = np.median([p["fml_seR"] for p in per_cell])

    cell_n_min = int(min(p["n_mean"] for p in per_cell))
    cell_n_median = int(np.median([p["n_mean"] for p in per_cell]))
    cell_n_max = int(max(p["n_mean"] for p in per_cell))

    print(f"  cells observed                            {len(per_cell)} / {N_CELLS}")
    print(f"  obs/cell (min/median/max)                 "
          f"{cell_n_min} / {cell_n_median} / {cell_n_max}")
    print()
    print(f"  {'metric':<42}{'FML-subgroup':>16}{'saturated GLM':>18}")
    print(f"  {'-'*76}")
    print(f"  {'avg per-cell coverage':<42}"
          f"{fml_cov_avg:>16.4f}{sat_cov_avg:>18.4f}")
    print(f"  {'avg per-cell |bias|':<42}"
          f"{fml_bias_avg:>16.4f}{sat_bias_avg:>18.4f}")
    print(f"  {'avg per-cell RMSE':<42}"
          f"{fml_rmse_avg:>16.4f}{sat_rmse_avg:>18.4f}")
    print(f"  {'RMSE ratio FML/SAT':<42}{fml_rmse_avg/sat_rmse_avg:>16.3f}")
    print(f"  {'median SE ratio (FML)':<42}{fml_seR_med:>16.3f}")

    fml_beats_sat = sum(1 for p in per_cell if p["fml_rmse"] < p.get("sat_rmse", np.inf))
    print(f"\n  FML RMSE beats saturated in   {fml_beats_sat}/{len(per_cell)} cells "
          f"({fml_beats_sat/len(per_cell):.0%})")

    # -------- coverage histogram --------
    bins = [0, 0.7, 0.85, 0.90, 0.95, 0.99, 1.01]
    labels = ["<0.70", "0.70–0.85", "0.85–0.90", "0.90–0.95", "0.95–0.99", "≥0.99"]
    fml_hist = np.histogram([p["fml_cov"] for p in per_cell], bins=bins)[0]
    sat_covs = [p["sat_cov"] for p in per_cell if not np.isnan(p["sat_cov"])]
    sat_hist = np.histogram(sat_covs, bins=bins)[0]
    print(f"\n  COVERAGE HISTOGRAM (cells per bin)")
    print(f"  {'bin':<15}{'FML':>6}{'Sat':>6}")
    for lbl, f_, s_ in zip(labels, fml_hist, sat_hist):
        print(f"  {lbl:<15}{f_:>6}{s_:>6}")

    # -------- per-cell table (all 40 cells) --------
    print(f"\n  PER-CELL DETAIL (sorted by FML coverage)")
    print(f"  {'cell':<8}{'uc,jc':>8}{'µ_c*':>9}{'n/rep':>8}  "
          f"{'FML bias':>10}{'FML cov':>9}{'FML rmse':>10}  "
          f"{'Sat bias':>10}{'Sat cov':>9}{'Sat rmse':>10}")
    for p in sorted(per_cell, key=lambda p: p["fml_cov"]):
        print(f"  c={p['c']:<6}({p['uc']},{p['jc']}) "
              f"{p['mu_star']:>+9.3f}{int(p['n_mean']):>8}  "
              f"{p['fml_bias']:>+10.4f}{p['fml_cov']:>9.3f}{p['fml_rmse']:>10.4f}  "
              f"{p['sat_bias']:>+10.4f}{p['sat_cov']:>9.3f}{p['sat_rmse']:>10.4f}")

    # save
    with open("/tmp/qpoisson_if/flm_crank6_results.json", "w") as f:
        json.dump(per_cell, f, indent=1, default=str)
    print(f"\n[saved] /tmp/qpoisson_if/flm_crank6_results.json")
    print(f"[crank6 total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
