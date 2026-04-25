"""
Auto-research v2: iterate candidate fixes at d=128 no-PCA, 40 cross cells.

For each configuration:
  1. Run M_MC MC reps
  2. Compute per-cell coverage, bias, RMSE
  3. Append findings to /tmp/qpoisson_if/FINDINGS.md
  4. Save per-cell JSON to /tmp/qpoisson_if/flm_C{N}_results.json

Candidate fixes (cranks 9+):
  C9_tanh_matched      : same architecture as crank 7, Tanh activations
  C10_additive         : β(z_u, z_j) = f(z_u) + g(z_j) separable
  C11_bilinear_r8      : β = Σ_{k=1}^8 φ_k(z_u)·ψ_k(z_j), low-rank interaction
  C12_tanh_ensemble_5  : Tanh, 5-seed ensemble average of β̂
  C13_tanh_bigger_n    : Tanh arch, n=80k (more data at matched arch)

Keeps the crank-7 baseline config on everything else:
  n=40_000, K_folds=5, LR=2e-3, patience=25, batch=1024, Λ-trim ε=0.10, M=15.
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
import statsmodels.api as sm


# ========================================================
# shared config (matches crank 7)
# ========================================================
D_U = 64
D_J = 64
D_TOTAL = D_U + D_J
K_U = 10
K_J = 4
N_CELLS = K_U * K_J
N_DATA_DEFAULT = 40_000
M_MC_DEFAULT = 15
TRIM_EPS = 0.10
REF_POOL_SIZE = 100_000
K_FOLDS = 5
LR = 2e-3
MAX_EPOCHS = 200
PATIENCE = 25
BATCH = 1024
N_WORKERS = 10

FINDINGS_PATH = "/tmp/qpoisson_if/FINDINGS.md"


# ========================================================
# truth (unchanged)
# ========================================================
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


# ========================================================
# K-means reference
# ========================================================
_ref_rng = np.random.default_rng(42)
REF_ZU = _ref_rng.standard_normal((REF_POOL_SIZE, D_U)).astype(np.float32)
REF_ZJ = _ref_rng.standard_normal((REF_POOL_SIZE, D_J)).astype(np.float32)
REF_X = np.concatenate([REF_ZU, REF_ZJ], axis=1)
with torch.no_grad():
    REF_BETA = beta_t(torch.from_numpy(REF_X)).numpy()
KM_U = KMeans(n_clusters=K_U, random_state=42, n_init=5).fit(REF_ZU)
KM_J = KMeans(n_clusters=K_J, random_state=42, n_init=5).fit(REF_ZJ)
ref_cell = KM_U.predict(REF_ZU) * K_J + KM_J.predict(REF_ZJ)
MU_CELL_STAR = np.array(
    [REF_BETA[ref_cell == c].mean() if (ref_cell == c).sum() > 0 else 0
     for c in range(N_CELLS)])


# ========================================================
# DGP
# ========================================================
def simulate(rng, n):
    z_u = rng.standard_normal((n, D_U)).astype(np.float32)
    z_j = rng.standard_normal((n, D_J)).astype(np.float32)
    x = np.concatenate([z_u, z_j], axis=1)
    t = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = alpha_t(xt).numpy()
        b = beta_t(xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return z_u, z_j, x, t, y


# ========================================================
# Candidate DNN architectures
# ========================================================
class ReluThetaNet(nn.Module):
    def __init__(self, d_in, hidden=(128, 64)):
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


class TanhThetaNet(nn.Module):
    """C9: Tanh activations matching truth."""
    def __init__(self, d_in, hidden=(128, 64)):
        super().__init__()
        L, prev = [], d_in
        for h in hidden:
            L += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*L)
        self.head_a = nn.Linear(prev, 1)
        self.head_b = nn.Linear(prev, 1)
    def forward(self, x):
        h = self.trunk(x)
        return self.head_a(h).squeeze(-1), self.head_b(h).squeeze(-1)


class AdditiveThetaNet(nn.Module):
    """C10: β = f(z_u) + g(z_j), separable in user and item."""
    def __init__(self, d_u=D_U, d_j=D_J, hidden=(64, 32)):
        super().__init__()
        def mk(d_in):
            L, prev = [], d_in
            for h in hidden:
                L += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            L.append(nn.Linear(prev, 1))
            return nn.Sequential(*L)
        self.alpha_u = mk(d_u); self.alpha_j = mk(d_j)
        self.beta_u = mk(d_u);  self.beta_j = mk(d_j)
    def forward(self, x):
        z_u, z_j = x[:, :D_U], x[:, D_U:]
        a = self.alpha_u(z_u).squeeze(-1) + self.alpha_j(z_j).squeeze(-1)
        b = self.beta_u(z_u).squeeze(-1)  + self.beta_j(z_j).squeeze(-1)
        return a, b


class BilinearThetaNet(nn.Module):
    """C11: β = φ(z_u) · ψ(z_j), low-rank r=8 interaction."""
    def __init__(self, d_u=D_U, d_j=D_J, r=8, hidden=(64, 32)):
        super().__init__()
        def mk(d_in, d_out):
            L, prev = [], d_in
            for h in hidden:
                L += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            L.append(nn.Linear(prev, d_out))
            return nn.Sequential(*L)
        self.alpha_u = mk(d_u, 1); self.alpha_j = mk(d_j, 1)
        self.phi_u = mk(d_u, r);   self.psi_j = mk(d_j, r)
        self.r = r
    def forward(self, x):
        z_u, z_j = x[:, :D_U], x[:, D_U:]
        a = self.alpha_u(z_u).squeeze(-1) + self.alpha_j(z_j).squeeze(-1)
        phi = self.phi_u(z_u)
        psi = self.psi_j(z_j)
        b = (phi * psi).sum(dim=-1)
        return a, b


NET_FACTORIES = {
    "C9_tanh_matched":      lambda: TanhThetaNet(D_TOTAL, hidden=(128, 64)),
    "C10_additive":         lambda: AdditiveThetaNet(D_U, D_J, hidden=(64, 32)),
    "C11_bilinear_r8":      lambda: BilinearThetaNet(D_U, D_J, r=8,
                                                      hidden=(64, 32)),
    "C12_tanh_ensemble_5":  lambda: TanhThetaNet(D_TOTAL, hidden=(128, 64)),  # ensemble at runtime
    "C13_tanh_bigger_n":    lambda: TanhThetaNet(D_TOTAL, hidden=(128, 64)),
}


# ========================================================
# training / IF / cross-fit
# ========================================================
def pnll(mu, y): return (mu - y * torch.log(mu + 1e-10)).mean()


def train(net, x_tr, t_tr, y_tr, x_va, t_va, y_va, seed=0,
          patience=PATIENCE, max_epochs=MAX_EPOCHS, lr=LR, batch=BATCH):
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


def pred(net, x):
    net.eval()
    with torch.no_grad():
        a, b = net(torch.from_numpy(x))
    return a.numpy(), b.numpy()


def poisson_if(a, b, t, y, eps=TRIM_EPS):
    mu_0 = np.maximum(np.exp(a), eps)
    mu_1 = np.maximum(np.exp(a + b), eps)
    mu_r = np.where(t > 0.5, mu_1, mu_0)
    coef = np.where(t > 0.5, 2.0 / mu_1, -2.0 / mu_0)
    return b - coef * (mu_r - y)


def cross_fit(x, t, y, net_factory, seed, ensemble_r=1):
    """K-fold cross-fit.  If ensemble_r>1, train R DNNs per fold and
    average predictions."""
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
            net = train(net, x[tr], t[tr], y[tr], x[va], t[va], y[va],
                        seed=seed * 100 + k * 10 + r)
            a_e, b_e = pred(net, x[me])
            a_sum += a_e; b_sum += b_e
        a_all[me] = a_sum / ensemble_r
        b_all[me] = b_sum / ensemble_r
    psi = poisson_if(a_all, b_all, t, y)
    return psi, b_all


# ========================================================
# MC worker
# ========================================================
def run_one(payload):
    config_name, mc_seed, n_data, ensemble_r = payload
    rng = np.random.default_rng(mc_seed)
    z_u, z_j, x, t, y = simulate(rng, n_data)
    net_factory = NET_FACTORIES[config_name]
    psi, beta_hat = cross_fit(x, t, y, net_factory, seed=mc_seed,
                              ensemble_r=ensemble_r)
    lu = KM_U.predict(z_u); lj = KM_J.predict(z_j)
    cell = lu * K_J + lj
    rec = dict(mc_seed=mc_seed, config=config_name)
    for c in range(N_CELLS):
        m = cell == c; nc = int(m.sum())
        if nc < 5:
            rec[f"fml_c{c}_mu"] = np.nan; rec[f"fml_c{c}_se"] = np.nan
        else:
            rec[f"fml_c{c}_mu"] = float(psi[m].mean())
            rec[f"fml_c{c}_se"] = float(psi[m].std(ddof=1) / np.sqrt(nc))
        rec[f"fml_c{c}_n"] = nc
    return rec


# ========================================================
# per-config driver
# ========================================================
def run_config(config_name, M_MC=M_MC_DEFAULT, n_data=N_DATA_DEFAULT,
               ensemble_r=1):
    t0 = time.time()
    print(f"\n[auto] === {config_name} ===")
    print(f"  n_data={n_data}  M_MC={M_MC}  ensemble_r={ensemble_r}")
    payloads = [(config_name, 20_000 + m * 31, n_data, ensemble_r)
                for m in range(M_MC)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_one, p) for p in payloads]
        results = []
        done = 0
        for f in as_completed(futs):
            results.append(f.result())
            done += 1
    wall = time.time() - t0
    print(f"  {done}/{M_MC} done, wall {wall:.0f}s")
    return results, wall


def summarize(config_name, results, wall, n_data, ensemble_r):
    per_cell = []
    for c in range(N_CELLS):
        mu = np.array([r[f"fml_c{c}_mu"] for r in results])
        se = np.array([r[f"fml_c{c}_se"] for r in results])
        v = ~np.isnan(mu) & ~np.isnan(se)
        if v.sum() < 3:
            continue
        mu_star = MU_CELL_STAR[c]
        bias = float(mu[v].mean() - mu_star)
        emp_sd = float(mu[v].std(ddof=1))
        se_r = float(se[v].mean() / emp_sd) if emp_sd > 0 else np.nan
        cov = float(np.mean(np.abs(mu[v] - mu_star) <= 1.96 * se[v]))
        rmse = float(np.sqrt(((mu[v] - mu_star) ** 2).mean()))
        per_cell.append(dict(c=c, mu_star=float(mu_star), bias=bias, cov=cov,
                             rmse=rmse, se_ratio=se_r))
    avg_cov = float(np.mean([p["cov"] for p in per_cell]))
    avg_bias = float(np.mean([abs(p["bias"]) for p in per_cell]))
    avg_rmse = float(np.mean([p["rmse"] for p in per_cell]))
    med_se_r = float(np.median([p["se_ratio"] for p in per_cell]))
    cov_ge_95 = int(sum(p["cov"] >= 0.95 for p in per_cell))
    cov_ge_90 = int(sum(p["cov"] >= 0.90 for p in per_cell))
    cov_lt_70 = int(sum(p["cov"] < 0.70 for p in per_cell))
    summary = dict(
        config=config_name, n_data=n_data, ensemble_r=ensemble_r, M=len(results),
        wall_seconds=wall, avg_cov=avg_cov, avg_abs_bias=avg_bias,
        avg_rmse=avg_rmse, med_se_ratio=med_se_r,
        cells_cov_ge_95=cov_ge_95, cells_cov_ge_90=cov_ge_90,
        cells_cov_lt_70=cov_lt_70, per_cell=per_cell,
    )
    return summary


def append_finding(summary):
    """Append a markdown section for this config to FINDINGS.md."""
    s = summary
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    # coverage histogram
    covs = [p["cov"] for p in s["per_cell"]]
    bins = [0, 0.70, 0.85, 0.90, 0.95, 0.99, 1.01]
    labels = ["<0.70", "0.70–0.85", "0.85–0.90", "0.90–0.95", "0.95–0.99", "≥0.99"]
    hist = np.histogram(covs, bins=bins)[0]

    # top 5 worst cells
    worst = sorted(s["per_cell"], key=lambda p: p["cov"])[:5]

    md = f"""
### {s['config']}  (t={ts}, wall={s['wall_seconds']:.0f}s)

**Setup**: d=128 no-PCA, 40 cross cells, n={s['n_data']:,}, ensemble_r={s['ensemble_r']}, M={s['M']} MC reps

**Headline**:
- avg per-cell coverage: **{s['avg_cov']:.4f}**
- avg per-cell \\|bias\\|: {s['avg_abs_bias']:.4f}
- avg per-cell RMSE: {s['avg_rmse']:.4f}
- median SE ratio: {s['med_se_ratio']:.3f}
- cells with coverage ≥0.95: {s['cells_cov_ge_95']}/40
- cells with coverage ≥0.90: {s['cells_cov_ge_90']}/40
- cells with coverage <0.70: {s['cells_cov_lt_70']}/40

**Coverage histogram**:

| bin | n cells |
|---|---|
"""
    for lbl, h in zip(labels, hist):
        md += f"| {lbl} | {int(h)} |\n"
    md += "\n**Worst 5 cells**:\n\n| cell | µ_c* | bias | cov | rmse |\n|---|---|---|---|---|\n"
    for p in worst:
        md += (f"| c={p['c']} | {p['mu_star']:+.3f} | {p['bias']:+.4f} | "
               f"{p['cov']:.3f} | {p['rmse']:.4f} |\n")

    md += f"\n---\n"
    with open(FINDINGS_PATH, "a") as f:
        f.write(md)

    # save JSON
    with open(f"/tmp/qpoisson_if/flm_{s['config']}_results.json", "w") as f:
        json.dump(s, f, indent=1, default=str)


# ========================================================
# main
# ========================================================
if __name__ == "__main__":
    t_all = time.time()

    CRANKS = [
        ("C9_tanh_matched",     M_MC_DEFAULT, N_DATA_DEFAULT, 1),
        ("C10_additive",        M_MC_DEFAULT, N_DATA_DEFAULT, 1),
        ("C11_bilinear_r8",     M_MC_DEFAULT, N_DATA_DEFAULT, 1),
        ("C12_tanh_ensemble_5", M_MC_DEFAULT, N_DATA_DEFAULT, 5),
        ("C13_tanh_bigger_n",   M_MC_DEFAULT, 80_000,          1),
    ]

    all_summaries = []
    for name, M, N_, R in CRANKS:
        print(f"\n{'='*70}\nLaunching {name}\n{'='*70}")
        try:
            results, wall = run_config(name, M_MC=M, n_data=N_, ensemble_r=R)
            summary = summarize(name, results, wall, N_, R)
            append_finding(summary)
            print(f"  -> avg cov = {summary['avg_cov']:.4f}  "
                  f"cells≥0.90 = {summary['cells_cov_ge_90']}/40  "
                  f"cells<0.70 = {summary['cells_cov_lt_70']}/40")
            all_summaries.append(summary)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            with open(FINDINGS_PATH, "a") as f:
                f.write(f"\n### {name} — FAILED\n\n{type(e).__name__}: {e}\n\n---\n")

    # final consolidated ranking
    print("\n" + "=" * 80)
    print("AUTO-RESEARCH SESSION SUMMARY".center(80))
    print("=" * 80)
    all_summaries.sort(key=lambda s: -s["avg_cov"])
    print(f"  {'config':<26}{'cov':>8}{'≥0.90':>8}{'<0.70':>8}"
          f"{'rmse':>8}{'SEr':>8}{'wall':>8}")
    for s in all_summaries:
        print(f"  {s['config']:<26}{s['avg_cov']:>8.3f}"
              f"{s['cells_cov_ge_90']:>8}{s['cells_cov_lt_70']:>8}"
              f"{s['avg_rmse']:>8.4f}{s['med_se_ratio']:>8.3f}"
              f"{s['wall_seconds']:>8.0f}")

    # add ranking to FINDINGS.md
    with open(FINDINGS_PATH, "a") as f:
        f.write("\n## Session ranking\n\n")
        f.write("| config | avg cov | cells ≥ 0.90 | cells < 0.70 | avg RMSE | SE ratio | wall (s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for s in all_summaries:
            f.write(f"| {s['config']} | {s['avg_cov']:.3f} | "
                    f"{s['cells_cov_ge_90']}/40 | {s['cells_cov_lt_70']}/40 | "
                    f"{s['avg_rmse']:.4f} | {s['med_se_ratio']:.3f} | "
                    f"{s['wall_seconds']:.0f} |\n")
        f.write(f"\nTotal wall time: {time.time()-t_all:.0f}s\n")

    print(f"\n[auto_research] total wall {time.time()-t_all:.0f}s")
    print(f"[auto_research] findings in {FINDINGS_PATH}")
