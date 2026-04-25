"""
Auto-research v3: Round 2.  Build on C12 (Tanh ensemble-5, 0.900 coverage).

Hypotheses:
  - Ensemble averaging reduces DNN pointwise variance, which is the binding
    bias term in subgroup averaging.  Bigger ensemble → lower bias.
  - Additive + Tanh + ensemble combines best of each round-1 ingredient.
  - Bigger N needs paired slower LR and more patience (crank 13 broke
    because training hyperparams didn't rescale).

New cranks (14+):
  C14_tanh_ensemble_10      : doubles ensemble size
  C15_additive_ensemble_5   : Tanh additive + 5-seed ensemble
  C16_tanh_ens5_n80k_tuned  : n=80k with LR=5e-4, patience=50, batch=2048
  C17_tanh_ens5_patience50  : same as C12 but train longer per member
  C18_subset_focus          : run C12 but on a SUBSET of cells (drop the 4
                              cells that keep failing) to isolate local-region
                              failures from global methodology

All documented to /tmp/qpoisson_if/FINDINGS.md.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans

import sys
sys.path.insert(0, "/tmp/qpoisson_if")
from flm_autoresearch_v2 import (
    D_U, D_J, D_TOTAL, K_U, K_J, N_CELLS, TRIM_EPS, K_FOLDS,
    REF_POOL_SIZE, N_WORKERS, FINDINGS_PATH,
    KM_U, KM_J, MU_CELL_STAR,
    alpha_t, beta_t, simulate,
    TanhThetaNet, AdditiveThetaNet, ReluThetaNet,
    pnll, pred, poisson_if, summarize, append_finding,
)


# ==========================================================
# custom train with tunable hyperparams
# ==========================================================
def train_custom(net, x_tr, t_tr, y_tr, x_va, t_va, y_va, seed=0,
                 lr=2e-3, patience=25, max_epochs=200, batch=1024):
    torch.manual_seed(seed)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    xt, tt, yt = map(torch.from_numpy, (x_tr, t_tr, y_tr))
    xv, tv, yv = map(torch.from_numpy, (x_va, t_va, y_va))
    n = len(xt); best = float("inf"); st = None; pat = 0
    for ep in range(max_epochs):
        net.train()
        idx = np.random.permutation(n)
        for s in range(0, n, batch):
            sel = idx[s:s + batch]
            a, b = net(xt[sel])
            loss = pnll(torch.exp(a + b * tt[sel]), yt[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            av, bv = net(xv)
            v = pnll(torch.exp(av + bv * tv), yv).item()
        if v < best - 1e-5:
            best = v
            st = {k: vv.detach().clone() for k, vv in net.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break
    if st is not None: net.load_state_dict(st)
    return net


def cross_fit_custom(x, t, y, net_factory, seed, ensemble_r=1,
                     lr=2e-3, patience=25, max_epochs=200, batch=1024):
    n = len(x)
    rng = np.random.default_rng(seed)
    folds = rng.integers(K_FOLDS, size=n)
    a_all = np.zeros(n); b_all = np.zeros(n)
    for k in range(K_FOLDS):
        me = folds == k
        tr_all = np.where(~me)[0]
        perm = np.random.default_rng(seed * 17 + k).permutation(len(tr_all))
        n_val = max(int(0.15 * len(tr_all)), 200)
        va = tr_all[perm[:n_val]]
        tr = tr_all[perm[n_val:]]
        a_sum = np.zeros(me.sum()); b_sum = np.zeros(me.sum())
        for r in range(ensemble_r):
            net = net_factory()
            net = train_custom(net,
                               x[tr], t[tr], y[tr],
                               x[va], t[va], y[va],
                               seed=seed * 100 + k * 10 + r,
                               lr=lr, patience=patience,
                               max_epochs=max_epochs, batch=batch)
            a_e, b_e = pred(net, x[me])
            a_sum += a_e; b_sum += b_e
        a_all[me] = a_sum / ensemble_r
        b_all[me] = b_sum / ensemble_r
    psi = poisson_if(a_all, b_all, t, y)
    return psi, b_all


# ==========================================================
# MC worker (generic)
# ==========================================================
def run_one_generic(payload):
    cfg_name, mc_seed, n_data, config = payload
    rng = np.random.default_rng(mc_seed)
    z_u, z_j, x, t, y = simulate(rng, n_data)

    # construct net factory from config
    arch = config["arch"]
    hidden = config.get("hidden", (128, 64))
    if arch == "tanh":
        factory = lambda: TanhThetaNet(D_TOTAL, hidden=hidden)
    elif arch == "additive":
        factory = lambda: AdditiveThetaNet(D_U, D_J,
                                           hidden=config.get("add_hidden", (64, 32)))
    elif arch == "relu":
        factory = lambda: ReluThetaNet(D_TOTAL, hidden=hidden)
    else:
        raise ValueError(arch)

    psi, beta_hat = cross_fit_custom(
        x, t, y, factory, seed=mc_seed,
        ensemble_r=config.get("ensemble_r", 1),
        lr=config.get("lr", 2e-3),
        patience=config.get("patience", 25),
        max_epochs=config.get("max_epochs", 200),
        batch=config.get("batch", 1024),
    )

    lu = KM_U.predict(z_u); lj = KM_J.predict(z_j)
    cell = lu * K_J + lj
    rec = dict(mc_seed=mc_seed, config=cfg_name)
    for c in range(N_CELLS):
        m = cell == c; nc = int(m.sum())
        if nc < 5:
            rec[f"fml_c{c}_mu"] = np.nan; rec[f"fml_c{c}_se"] = np.nan
        else:
            rec[f"fml_c{c}_mu"] = float(psi[m].mean())
            rec[f"fml_c{c}_se"] = float(psi[m].std(ddof=1) / np.sqrt(nc))
        rec[f"fml_c{c}_n"] = nc
    return rec


def run_config_generic(name, config, M_MC=15, n_data=40_000):
    t0 = time.time()
    print(f"\n[auto v3] === {name} ===  n={n_data}  M={M_MC}  cfg={config}")
    payloads = [(name, 30_000 + m * 31, n_data, config) for m in range(M_MC)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_one_generic, p) for p in payloads]
        results = [f.result() for f in as_completed(futs)]
    wall = time.time() - t0
    summary = summarize(name, results, wall, n_data,
                        config.get("ensemble_r", 1))
    append_finding(summary)
    print(f"  -> avg cov = {summary['avg_cov']:.4f}  "
          f"cells≥0.90 = {summary['cells_cov_ge_90']}/40  "
          f"cells<0.70 = {summary['cells_cov_lt_70']}/40   wall={wall:.0f}s")
    return summary


# ==========================================================
# main
# ==========================================================
if __name__ == "__main__":
    t_all = time.time()

    with open(FINDINGS_PATH, "a") as f:
        f.write(f"\n\n## Round 2 cranks  "
                f"(t={datetime.datetime.now().isoformat(timespec='seconds')})\n\n")
        f.write("Building on C12 Tanh-ensemble-5 (0.900 avg coverage).  "
                "Round 2 combines best ingredients: ensemble, additive, "
                "paired hyperparams for bigger N.\n")

    CRANKS = [
        ("C14_tanh_ensemble_10",
         dict(arch="tanh", hidden=(128, 64), ensemble_r=10)),
        ("C15_additive_ensemble_5",
         dict(arch="additive", add_hidden=(64, 32), ensemble_r=5)),
        ("C16_tanh_ens5_n80k_tuned",
         dict(arch="tanh", hidden=(128, 64), ensemble_r=5,
              lr=5e-4, patience=50, max_epochs=300, batch=2048)),
        ("C17_tanh_ens5_patience50",
         dict(arch="tanh", hidden=(128, 64), ensemble_r=5,
              patience=50, max_epochs=300)),
        ("C18_tanh_ens15",
         dict(arch="tanh", hidden=(128, 64), ensemble_r=15)),
    ]
    # C16 at n=80k, others at n=40k
    n_sizes = {"C16_tanh_ens5_n80k_tuned": 80_000}

    summaries = []
    for name, cfg in CRANKS:
        n = n_sizes.get(name, 40_000)
        try:
            s = run_config_generic(name, cfg, M_MC=15, n_data=n)
            summaries.append(s)
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {e}")
            with open(FINDINGS_PATH, "a") as f:
                f.write(f"\n### {name} — FAILED\n\n"
                        f"{type(e).__name__}: {e}\n\n---\n")

    # round 2 ranking
    summaries.sort(key=lambda s: -s["avg_cov"])
    print("\n" + "=" * 80)
    print("ROUND 2 RANKING".center(80))
    print("=" * 80)
    print(f"  {'config':<30}{'cov':>8}{'≥0.90':>8}{'<0.70':>8}"
          f"{'rmse':>8}{'SEr':>8}{'wall':>8}")
    for s in summaries:
        print(f"  {s['config']:<30}{s['avg_cov']:>8.3f}"
              f"{s['cells_cov_ge_90']:>8}{s['cells_cov_lt_70']:>8}"
              f"{s['avg_rmse']:>8.4f}{s['med_se_ratio']:>8.3f}"
              f"{s['wall_seconds']:>8.0f}")

    with open(FINDINGS_PATH, "a") as f:
        f.write("\n## Round 2 ranking\n\n")
        f.write("| config | avg cov | cells ≥ 0.90 | cells < 0.70 | RMSE | SE ratio |\n")
        f.write("|---|---|---|---|---|---|\n")
        for s in summaries:
            f.write(f"| {s['config']} | {s['avg_cov']:.3f} | "
                    f"{s['cells_cov_ge_90']}/40 | {s['cells_cov_lt_70']}/40 | "
                    f"{s['avg_rmse']:.4f} | {s['med_se_ratio']:.3f} |\n")

    print(f"\n[v3 total] {time.time()-t_all:.0f}s")
