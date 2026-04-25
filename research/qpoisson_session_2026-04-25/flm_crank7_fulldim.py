"""
Crank 7: FML-subgroup on FULL 128-dim three-tower embeddings.

No PCA.  z_u ∈ R^64, z_j ∈ R^64, total X ∈ R^128.
K-means on 64-dim z_u (K_u=10) and z_j (K_j=4) separately → 40 cross cells.

This is the realistic paper spec: we take the three-tower embeddings as is,
cluster them, and deploy FML-subgroup on the raw 128-dim covariate vector.
Question: does subgroup averaging rescue inference even at d=128?

Setup:
  n = 40,000 (paper scale is larger; this is validation)
  K_u = 10, K_j = 4, 40 cells, ~1000 obs/cell average
  DNN estimator: hidden = (128, 64), ReLU, K=5-fold cross-fit
  Λ-trim ε = 0.10, no winsorize
  M_MC = 15 reps (each rep heavier due to 128-dim input)

Comparison: FML-subgroup vs saturated per-cell Poisson QMLE with HC1.

If this passes, the pipeline is ready to deploy on actual H&M embeddings.
If it fails, PCA to d=8-16 is the fallback.
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


# ===========================================================
# config
# ===========================================================
D_U = 64                 # paper spec: three-tower user embedding
D_J = 64                 # paper spec: three-tower item embedding
D_TOTAL = D_U + D_J      # 128
K_U = 10
K_J = 4
N_CELLS = K_U * K_J      # 40
N_DATA = 40_000
M_MC = 15
TRIM_EPS = 0.10
REF_POOL_SIZE = 100_000
K_FOLDS = 5
DNN_HIDDEN = (128, 64)   # bigger to match 128-d input
DNN_LR = 2e-3
DNN_MAX_EPOCHS = 200
DNN_PATIENCE = 25
DNN_BATCH = 1024
N_WORKERS = 10


# ===========================================================
# truth (d=128)
# ===========================================================
class TrueNet(nn.Module):
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        # slightly bigger truth for 128-d input
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

_g = torch.Generator().manual_seed(99)
with torch.no_grad():
    xc = torch.randn(200_000, D_TOTAL, generator=_g)
    am, as_ = _alpha(xc).mean().item(), _alpha(xc).std().item()
    bm, bs = _beta(xc).mean().item(), _beta(xc).std().item()

def alpha_t(x): return (_alpha(x) - am) / as_ * 0.40 + 1.3
def beta_t(x):  return (_beta(x)  - bm) / bs * 0.55 - 0.35


# ===========================================================
# K-means reference
# ===========================================================
_ref_rng = np.random.default_rng(42)
REF_ZU = _ref_rng.standard_normal((REF_POOL_SIZE, D_U)).astype(np.float32)
REF_ZJ = _ref_rng.standard_normal((REF_POOL_SIZE, D_J)).astype(np.float32)
REF_X = np.concatenate([REF_ZU, REF_ZJ], axis=1)

with torch.no_grad():
    REF_BETA = beta_t(torch.from_numpy(REF_X)).numpy()

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

print(f"[setup] d_u={D_U}, d_j={D_J}, total d={D_TOTAL} (NO PCA)")
print(f"  {N_CELLS} cross cells, n_data={N_DATA}")
print(f"  cell size range (ref): "
      f"min={int(P_CELL[P_CELL > 0].min()*REF_POOL_SIZE):>4}  "
      f"max={int(P_CELL.max()*REF_POOL_SIZE):>5}")
print(f"  µ_c* range: [{MU_CELL_STAR.min():+.3f}, {MU_CELL_STAR.max():+.3f}]  "
      f"std={MU_CELL_STAR.std():.3f}")


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
        a = alpha_t(xt).numpy()
        b = beta_t(xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return z_u, z_j, x, t, y


# ===========================================================
# ThetaNet for FML (bigger, to handle 128-d input)
# ===========================================================
class ThetaNet(nn.Module):
    def __init__(self, d_in, hidden=DNN_HIDDEN):
        super().__init__()
        L, prev = [], d_in
        for h in hidden:
            L += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*L)
        self.head_a = nn.Linear(prev, 1)
        self.head_b = nn.Linear(prev, 1)
    def forward(self, x):
        h = self.trunk(x)
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


def poisson_nll(mu, y):
    return (mu - y * torch.log(mu + 1e-10)).mean()


def train_theta(x_tr, t_tr, y_tr, x_va, t_va, y_va, seed=0):
    torch.manual_seed(seed)
    net = ThetaNet(x_tr.shape[1])
    opt = optim.Adam(net.parameters(), lr=DNN_LR, weight_decay=1e-5)
    xtr, ttr, ytr = map(torch.from_numpy, (x_tr, t_tr, y_tr))
    xva, tva, yva = map(torch.from_numpy, (x_va, t_va, y_va))
    n = len(xtr)
    best_val = float("inf"); best = None; pat = 0
    for ep in range(DNN_MAX_EPOCHS):
        net.train()
        idx = np.random.permutation(n)
        for s in range(0, n, DNN_BATCH):
            sel = idx[s:s + DNN_BATCH]
            a, b = net(xtr[sel])
            mu = torch.exp(a + b * ttr[sel])
            loss = poisson_nll(mu, ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            av, bv = net(xva)
            muv = torch.exp(av + bv * tva)
            val = poisson_nll(muv, yva).item()
        if val < best_val - 1e-5:
            best_val = val
            best = {k: v.detach().clone() for k, v in net.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= DNN_PATIENCE: break
    if best is not None:
        net.load_state_dict(best)
    return net


def pred(net, x):
    net.eval()
    with torch.no_grad():
        a, b = net(torch.from_numpy(x))
    return a.numpy(), b.numpy()


def poisson_if(alpha_hat, beta_hat, t, y, trim_eps=TRIM_EPS, p_t=0.5):
    mu_0 = np.exp(alpha_hat)
    mu_1 = np.exp(alpha_hat + beta_hat)
    if trim_eps > 0:
        mu_0 = np.maximum(mu_0, trim_eps)
        mu_1 = np.maximum(mu_1, trim_eps)
    mu_real = np.where(t > 0.5, mu_1, mu_0)
    coef = np.where(t > 0.5, 1.0 / (p_t * mu_1),
                    -1.0 / ((1 - p_t) * mu_0))
    return beta_hat - coef * (mu_real - y), mu_real


def cross_fit(x, t, y, seed):
    n = len(x)
    rng = np.random.default_rng(seed)
    folds = rng.integers(K_FOLDS, size=n)
    alpha_hat = np.zeros(n); beta_hat = np.zeros(n)
    for k in range(K_FOLDS):
        me = folds == k
        tr_all = np.where(~me)[0]
        perm = np.random.default_rng(seed * 17 + k).permutation(len(tr_all))
        n_val = max(int(0.15 * len(tr_all)), 100)
        va = tr_all[perm[:n_val]]
        tr = tr_all[perm[n_val:]]
        net = train_theta(x[tr], t[tr], y[tr], x[va], t[va], y[va],
                          seed=seed * 100 + k)
        a_e, b_e = pred(net, x[me])
        alpha_hat[me] = a_e
        beta_hat[me] = b_e
    psi, _ = poisson_if(alpha_hat, beta_hat, t, y)
    return psi, beta_hat, alpha_hat


# ===========================================================
# subgroup + saturated per cell
# ===========================================================
def fml_per_cell(x, t, y, z_u, z_j, seed):
    psi, beta_hat, _ = cross_fit(x, t, y, seed)
    lu = KM_U.predict(z_u); lj = KM_J.predict(z_j)
    cell = lu * K_J + lj
    out = {}
    for c in range(N_CELLS):
        m = cell == c; nc = int(m.sum())
        if nc < 5:
            out[c] = dict(mu_hat=np.nan, se=np.nan, n=nc); continue
        out[c] = dict(mu_hat=float(psi[m].mean()),
                      se=float(psi[m].std(ddof=1) / np.sqrt(nc)),
                      n=nc)
    return out


def saturated_per_cell(t, y, z_u, z_j):
    lu = KM_U.predict(z_u); lj = KM_J.predict(z_j)
    cell = lu * K_J + lj
    out = {}
    for c in range(N_CELLS):
        m = cell == c; nc = int(m.sum())
        if nc < 10:
            out[c] = dict(mu_hat=np.nan, se=np.nan, n=nc); continue
        y_c, t_c = y[m].astype(float), t[m].astype(float)
        Xc = sm.add_constant(t_c)
        try:
            r = sm.GLM(y_c, Xc, family=sm.families.Poisson()).fit(
                cov_type="HC1", maxiter=100)
            out[c] = dict(mu_hat=float(r.params[1]), se=float(r.bse[1]), n=nc)
        except Exception:
            out[c] = dict(mu_hat=np.nan, se=np.nan, n=nc)
    return out


def run_rep(mc_seed):
    rng = np.random.default_rng(mc_seed)
    z_u, z_j, x, t, y = simulate(rng)
    f = fml_per_cell(x, t, y, z_u, z_j, seed=mc_seed)
    s = saturated_per_cell(t, y, z_u, z_j)
    rec = dict(mc_seed=mc_seed)
    for c in range(N_CELLS):
        rec[f"fml_c{c}_mu"] = f[c]["mu_hat"]
        rec[f"fml_c{c}_se"] = f[c]["se"]
        rec[f"fml_c{c}_n"] = f[c]["n"]
        rec[f"sat_c{c}_mu"] = s[c]["mu_hat"]
        rec[f"sat_c{c}_se"] = s[c]["se"]
        rec[f"sat_c{c}_n"] = s[c]["n"]
    return rec


# ===========================================================
# main
# ===========================================================
def main():
    t0 = time.time()
    print(f"\n[crank7] launching {M_MC} reps (d=128, bigger DNN)...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_rep, 17000 + m * 31) for m in range(M_MC)]
        done = 0; results = []
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
            print(f"  {done}/{M_MC}  t={time.time()-t0:.0f}s")

    # aggregate
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
            sat_bias = sat_cov = sat_rmse = sat_seR = np.nan
        uc, jc = divmod(c, K_J)
        per_cell.append(dict(
            c=c, uc=uc, jc=jc, mu_star=mu_star, n_mean=float(fml_n.mean()),
            fml_bias=fml_bias, fml_cov=fml_cov, fml_rmse=fml_rmse,
            fml_seR=fml_seR, fml_empSD=fml_empSD,
            sat_bias=sat_bias, sat_cov=sat_cov, sat_rmse=sat_rmse,
            sat_seR=sat_seR,
        ))

    print(f"\n{'=' * 100}")
    print(f"CRANK 7: d={D_TOTAL} (NO PCA) — {N_CELLS} cross cells, M={M_MC} reps"
          .center(100))
    print(f"{'=' * 100}\n")

    fml_cov = np.mean([p["fml_cov"] for p in per_cell])
    sat_cov = np.nanmean([p["sat_cov"] for p in per_cell])
    fml_rmse = np.mean([p["fml_rmse"] for p in per_cell])
    sat_rmse = np.nanmean([p["sat_rmse"] for p in per_cell])
    fml_bias = np.mean([abs(p["fml_bias"]) for p in per_cell])
    sat_bias = np.nanmean([abs(p["sat_bias"]) for p in per_cell])
    fml_seR = np.median([p["fml_seR"] for p in per_cell])

    print(f"  {'metric':<42}{'FML (d=128)':>15}{'saturated':>15}")
    print(f"  {'-'*72}")
    print(f"  {'avg per-cell coverage':<42}{fml_cov:>15.4f}{sat_cov:>15.4f}")
    print(f"  {'avg per-cell |bias|':<42}{fml_bias:>15.4f}{sat_bias:>15.4f}")
    print(f"  {'avg per-cell RMSE':<42}{fml_rmse:>15.4f}{sat_rmse:>15.4f}")
    print(f"  {'RMSE ratio FML/SAT':<42}{fml_rmse/sat_rmse:>15.3f}")
    print(f"  {'median SE ratio (FML)':<42}{fml_seR:>15.3f}")

    fml_beats = sum(1 for p in per_cell if p["fml_rmse"] < p.get("sat_rmse", np.inf))
    print(f"\n  FML RMSE beats saturated in {fml_beats}/{len(per_cell)} cells "
          f"({fml_beats/len(per_cell):.0%})")

    # coverage histogram
    bins = [0, 0.70, 0.85, 0.90, 0.95, 0.99, 1.01]
    labels = ["<0.70", "0.70–0.85", "0.85–0.90", "0.90–0.95", "0.95–0.99", "≥0.99"]
    fml_hist = np.histogram([p["fml_cov"] for p in per_cell], bins=bins)[0]
    sat_covs = [p["sat_cov"] for p in per_cell if not np.isnan(p["sat_cov"])]
    sat_hist = np.histogram(sat_covs, bins=bins)[0]
    print(f"\n  COVERAGE HISTOGRAM")
    print(f"  {'bin':<15}{'FML':>6}{'Sat':>6}")
    for lbl, f_, s_ in zip(labels, fml_hist, sat_hist):
        print(f"  {lbl:<15}{f_:>6}{s_:>6}")

    print(f"\n  PER-CELL DETAIL (sorted by FML coverage ascending — worst first)")
    print(f"  {'cell':<8}{'µ_c*':>9}{'n/rep':>8}  "
          f"{'FML bias':>10}{'FML cov':>9}{'FML rmse':>10}  "
          f"{'Sat bias':>10}{'Sat cov':>9}{'Sat rmse':>10}")
    for p in sorted(per_cell, key=lambda p: p["fml_cov"])[:15]:
        print(f"  c={p['c']:<6}{p['mu_star']:>+9.3f}{int(p['n_mean']):>8}  "
              f"{p['fml_bias']:>+10.4f}{p['fml_cov']:>9.3f}{p['fml_rmse']:>10.4f}  "
              f"{p['sat_bias']:>+10.4f}{p['sat_cov']:>9.3f}{p['sat_rmse']:>10.4f}")
    print(f"  ... (top 15 worst of {len(per_cell)})")

    with open("/tmp/qpoisson_if/flm_crank7_results.json", "w") as f:
        json.dump(per_cell, f, indent=1, default=str)
    print(f"\n[saved] /tmp/qpoisson_if/flm_crank7_results.json")
    print(f"[crank7 total] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
