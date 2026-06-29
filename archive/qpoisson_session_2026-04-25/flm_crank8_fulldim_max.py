"""
Crank 8: Same d=128 no-PCA setup as crank 7, but scaled up.
  n = 100_000  (2.5x more data → 2500 obs/cell avg)
  DNN hidden = (256, 128, 64) deeper + wider
  patience = 50, more epochs
  K-fold = 10 (smaller folds → more stable Λ)

Goal: close the 0.87 → 0.95 coverage gap seen in crank 7 at d=128 no-PCA.
If this works, FML-subgroup is production-ready for the paper's full spec.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import KMeans
import statsmodels.api as sm


# ---------- config ----------
D_U = 64
D_J = 64
D_TOTAL = D_U + D_J
K_U = 10
K_J = 4
N_CELLS = K_U * K_J
N_DATA = 100_000                  # 2.5x crank 7
M_MC = 12
TRIM_EPS = 0.10
REF_POOL_SIZE = 100_000
K_FOLDS = 10                      # more folds, smaller eval per fold
DNN_HIDDEN = (256, 128, 64)       # deeper + wider
DNN_LR = 1e-3                     # slower (bigger net)
DNN_MAX_EPOCHS = 300
DNN_PATIENCE = 50
DNN_BATCH = 2048
N_WORKERS = 8


# ---------- truth (d=128) ----------
class TrueNet(nn.Module):
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


torch.manual_seed(1001)
_alpha = TrueNet(D_TOTAL)
_beta = TrueNet(D_TOTAL)
for p in _alpha.parameters(): p.requires_grad_(False)
for p in _beta.parameters(): p.requires_grad_(False)

with torch.no_grad():
    xc = torch.randn(200_000, D_TOTAL, generator=torch.Generator().manual_seed(99))
    am, as_ = _alpha(xc).mean().item(), _alpha(xc).std().item()
    bm, bs = _beta(xc).mean().item(), _beta(xc).std().item()


def alpha_t(x): return (_alpha(x) - am) / as_ * 0.40 + 1.3
def beta_t(x):  return (_beta(x)  - bm) / bs * 0.55 - 0.35


# ---------- K-means reference ----------
_ref_rng = np.random.default_rng(42)
REF_ZU = _ref_rng.standard_normal((REF_POOL_SIZE, D_U)).astype(np.float32)
REF_ZJ = _ref_rng.standard_normal((REF_POOL_SIZE, D_J)).astype(np.float32)
REF_X = np.concatenate([REF_ZU, REF_ZJ], axis=1)
with torch.no_grad():
    REF_BETA = beta_t(torch.from_numpy(REF_X)).numpy()
KM_U = KMeans(n_clusters=K_U, random_state=42, n_init=5).fit(REF_ZU)
KM_J = KMeans(n_clusters=K_J, random_state=42, n_init=5).fit(REF_ZJ)
ref_cell = KM_U.predict(REF_ZU) * K_J + KM_J.predict(REF_ZJ)
MU_CELL_STAR = np.array([REF_BETA[ref_cell == c].mean() if (ref_cell == c).sum() > 0 else 0
                         for c in range(N_CELLS)])
P_CELL = np.array([(ref_cell == c).mean() for c in range(N_CELLS)])
print(f"[setup] d={D_TOTAL}, cells={N_CELLS}, n={N_DATA}")
print(f"  expected obs/cell ≈ {N_DATA/N_CELLS:.0f}")
print(f"  µ_c* range: [{MU_CELL_STAR.min():+.3f}, {MU_CELL_STAR.max():+.3f}]  "
      f"std={MU_CELL_STAR.std():.3f}")


# ---------- DGP ----------
def simulate(rng):
    z_u = rng.standard_normal((N_DATA, D_U)).astype(np.float32)
    z_j = rng.standard_normal((N_DATA, D_J)).astype(np.float32)
    x = np.concatenate([z_u, z_j], axis=1)
    t = rng.choice([0.0, 1.0], size=N_DATA).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = alpha_t(xt).numpy()
        b = beta_t(xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return z_u, z_j, x, t, y


# ---------- DNN ----------
class ThetaNet(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        L, prev = [], d_in
        for h in DNN_HIDDEN:
            L += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*L)
        self.head_a = nn.Linear(prev, 1)
        self.head_b = nn.Linear(prev, 1)
    def forward(self, x):
        h = self.trunk(x)
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


def pnll(mu, y): return (mu - y * torch.log(mu + 1e-10)).mean()


def train(net, x_tr, t_tr, y_tr, x_va, t_va, y_va, seed=0):
    torch.manual_seed(seed)
    opt = optim.Adam(net.parameters(), lr=DNN_LR, weight_decay=1e-5)
    xt, tt, yt = map(torch.from_numpy, (x_tr, t_tr, y_tr))
    xv, tv, yv = map(torch.from_numpy, (x_va, t_va, y_va))
    n = len(xt)
    best = float("inf"); st = None; pat = 0
    for ep in range(DNN_MAX_EPOCHS):
        net.train()
        idx = np.random.permutation(n)
        for s in range(0, n, DNN_BATCH):
            sel = idx[s:s + DNN_BATCH]
            a, b = net(xt[sel])
            loss = pnll(torch.exp(a + b * tt[sel]), yt[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            av, bv = net(xv)
            v = pnll(torch.exp(av + bv * tv), yv).item()
        if v < best - 1e-5:
            best = v; st = {k: vv.detach().clone() for k, vv in net.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= DNN_PATIENCE: break
    if st is not None: net.load_state_dict(st)
    return net


def pred(net, x):
    net.eval()
    with torch.no_grad():
        a, b = net(torch.from_numpy(x))
    return a.numpy(), b.numpy()


def poisson_if(a, b, t, y):
    mu_0 = np.maximum(np.exp(a), TRIM_EPS)
    mu_1 = np.maximum(np.exp(a + b), TRIM_EPS)
    mu_r = np.where(t > 0.5, mu_1, mu_0)
    coef = np.where(t > 0.5, 2.0 / mu_1, -2.0 / mu_0)
    return b - coef * (mu_r - y)


def cross_fit(x, t, y, seed):
    n = len(x)
    rng = np.random.default_rng(seed)
    folds = rng.integers(K_FOLDS, size=n)
    a_all = np.zeros(n); b_all = np.zeros(n)
    for k in range(K_FOLDS):
        me = folds == k
        tr_all = np.where(~me)[0]
        perm = np.random.default_rng(seed * 17 + k).permutation(len(tr_all))
        n_val = max(int(0.10 * len(tr_all)), 300)
        va = tr_all[perm[:n_val]]
        tr = tr_all[perm[n_val:]]
        net = ThetaNet(x.shape[1])
        net = train(net, x[tr], t[tr], y[tr], x[va], t[va], y[va],
                    seed=seed * 100 + k)
        a_e, b_e = pred(net, x[me])
        a_all[me] = a_e; b_all[me] = b_e
    psi = poisson_if(a_all, b_all, t, y)
    return psi, b_all


def run_rep(mc_seed):
    rng = np.random.default_rng(mc_seed)
    z_u, z_j, x, t, y = simulate(rng)
    psi, beta_hat = cross_fit(x, t, y, seed=mc_seed)
    lu = KM_U.predict(z_u); lj = KM_J.predict(z_j)
    cell = lu * K_J + lj
    rec = dict(mc_seed=mc_seed)
    for c in range(N_CELLS):
        m = cell == c; nc = int(m.sum())
        if nc < 5:
            rec[f"fml_c{c}_mu"] = np.nan; rec[f"fml_c{c}_se"] = np.nan
            rec[f"fml_c{c}_n"] = nc
        else:
            rec[f"fml_c{c}_mu"] = float(psi[m].mean())
            rec[f"fml_c{c}_se"] = float(psi[m].std(ddof=1) / np.sqrt(nc))
            rec[f"fml_c{c}_n"] = nc
        # saturated
        if nc < 10:
            rec[f"sat_c{c}_mu"] = np.nan; rec[f"sat_c{c}_se"] = np.nan
        else:
            y_c, t_c = y[m].astype(float), t[m].astype(float)
            try:
                r = sm.GLM(y_c, sm.add_constant(t_c),
                           family=sm.families.Poisson()).fit(cov_type="HC1", maxiter=100)
                rec[f"sat_c{c}_mu"] = float(r.params[1])
                rec[f"sat_c{c}_se"] = float(r.bse[1])
            except Exception:
                rec[f"sat_c{c}_mu"] = np.nan; rec[f"sat_c{c}_se"] = np.nan
    return rec


def main():
    t0 = time.time()
    print(f"\n[crank8] {M_MC} reps, n={N_DATA}, d={D_TOTAL}, "
          f"hidden={DNN_HIDDEN}, patience={DNN_PATIENCE}")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_rep, 18_000 + m * 31) for m in range(M_MC)]
        done = 0; results = []
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            print(f"  {done}/{M_MC}  t={time.time()-t0:.0f}s")

    per_cell = []
    for c in range(N_CELLS):
        fml_mu = np.array([r[f"fml_c{c}_mu"] for r in results])
        fml_se = np.array([r[f"fml_c{c}_se"] for r in results])
        fml_n = np.array([r[f"fml_c{c}_n"] for r in results])
        sat_mu = np.array([r[f"sat_c{c}_mu"] for r in results])
        sat_se = np.array([r[f"sat_c{c}_se"] for r in results])
        vf = ~np.isnan(fml_mu) & ~np.isnan(fml_se)
        vs = ~np.isnan(sat_mu) & ~np.isnan(sat_se)
        if vf.sum() < 3: continue
        mu_star = MU_CELL_STAR[c]
        fml_bias = float(fml_mu[vf].mean() - mu_star)
        fml_empSD = float(fml_mu[vf].std(ddof=1))
        fml_seR = float(fml_se[vf].mean() / fml_empSD) if fml_empSD > 0 else np.nan
        fml_cov = float(np.mean(
            np.abs(fml_mu[vf] - mu_star) <= 1.96 * fml_se[vf]))
        fml_rmse = float(np.sqrt(((fml_mu[vf] - mu_star) ** 2).mean()))
        if vs.sum() >= 3:
            sat_bias = float(sat_mu[vs].mean() - mu_star)
            sat_empSD = float(sat_mu[vs].std(ddof=1))
            sat_seR = float(sat_se[vs].mean() / sat_empSD) if sat_empSD > 0 else np.nan
            sat_cov = float(np.mean(
                np.abs(sat_mu[vs] - mu_star) <= 1.96 * sat_se[vs]))
            sat_rmse = float(np.sqrt(((sat_mu[vs] - mu_star) ** 2).mean()))
        else:
            sat_bias = sat_seR = sat_cov = sat_rmse = np.nan
        per_cell.append(dict(c=c, mu_star=mu_star, n_mean=float(fml_n.mean()),
            fml_bias=fml_bias, fml_cov=fml_cov, fml_rmse=fml_rmse,
            fml_seR=fml_seR,
            sat_bias=sat_bias, sat_cov=sat_cov, sat_rmse=sat_rmse,
            sat_seR=sat_seR))

    print(f"\n{'=' * 100}")
    print(f"CRANK 8: d={D_TOTAL} no-PCA, scaled up (n={N_DATA}, hidden={DNN_HIDDEN})"
          .center(100))
    print(f"{'=' * 100}\n")
    fml_cov = np.mean([p["fml_cov"] for p in per_cell])
    sat_cov = np.nanmean([p["sat_cov"] for p in per_cell])
    fml_rmse = np.mean([p["fml_rmse"] for p in per_cell])
    sat_rmse = np.nanmean([p["sat_rmse"] for p in per_cell])
    fml_seR = np.median([p["fml_seR"] for p in per_cell])

    print(f"  {'metric':<42}{'FML':>15}{'saturated':>15}")
    print(f"  avg per-cell coverage        {fml_cov:>14.4f}  {sat_cov:>14.4f}")
    print(f"  avg per-cell |bias|          "
          f"{np.mean([abs(p['fml_bias']) for p in per_cell]):>14.4f}  "
          f"{np.nanmean([abs(p['sat_bias']) for p in per_cell]):>14.4f}")
    print(f"  avg per-cell RMSE            {fml_rmse:>14.4f}  {sat_rmse:>14.4f}")
    print(f"  RMSE ratio FML/SAT           {fml_rmse/sat_rmse:>14.3f}")
    print(f"  median SE ratio (FML)        {fml_seR:>14.3f}")

    # histogram
    bins = [0, 0.70, 0.85, 0.90, 0.95, 0.99, 1.01]
    labels = ["<0.70", "0.70–0.85", "0.85–0.90", "0.90–0.95", "0.95–0.99", "≥0.99"]
    fml_hist = np.histogram([p["fml_cov"] for p in per_cell], bins=bins)[0]
    sat_covs = [p["sat_cov"] for p in per_cell if not np.isnan(p["sat_cov"])]
    sat_hist = np.histogram(sat_covs, bins=bins)[0]
    print(f"\n  COVERAGE HISTOGRAM")
    print(f"  {'bin':<15}{'FML':>6}{'Sat':>6}")
    for lbl, f_, s_ in zip(labels, fml_hist, sat_hist):
        print(f"  {lbl:<15}{f_:>6}{s_:>6}")

    # sorted
    print(f"\n  WORST 10 CELLS (by FML coverage)")
    print(f"  {'cell':<6}{'µ_c*':>9}{'n/rep':>8}  "
          f"{'FML bias':>10}{'FML cov':>9}  {'Sat cov':>9}")
    for p in sorted(per_cell, key=lambda p: p["fml_cov"])[:10]:
        print(f"  c={p['c']:<4}{p['mu_star']:>+9.3f}{int(p['n_mean']):>8}  "
              f"{p['fml_bias']:>+10.4f}{p['fml_cov']:>9.3f}  {p['sat_cov']:>9.3f}")

    with open("/tmp/qpoisson_if/flm_crank8_results.json", "w") as f:
        json.dump(per_cell, f, indent=1, default=str)
    print(f"\n[saved] /tmp/qpoisson_if/flm_crank8_results.json")
    print(f"[crank8 total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
