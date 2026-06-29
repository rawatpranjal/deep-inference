"""
FML robustness auto-research: test 10 influence-function fixes against the
Poisson instability identified in the earlier sweep (1/μ blowups, outlier ψ
reps inflating SE and bias).

Fixes (column dimension of matrix):
  F0_vanilla              : standard FML IF, no fix
  F1_trim_e010            : Λ-trim clip μ̂₀, μ̂₁ ≥ 0.10
  F2_trim_e050            : Λ-trim ε = 0.50 (aggressive)
  F3_winsor_1             : ψ_i winsorized to (1%, 99%) per fold
  F4_winsor_5             : ψ_i winsorized to (5%, 95%) per fold
  F5_trim_winsor          : F1 + F3 combined
  F6_mom5                 : median-of-means with 5 blocks + F1
  F7_ensemble_3           : 3-DNN ensemble + F1
  F8_tmle_1step           : 1-step TMLE targeting + F1
  F9_all                  : F1 + F3 + 3-DNN ensemble + TMLE

DGPs (row dimension):
  D1_Poisson_d4_n20k      : baseline failing case from sweep
  D2_Poisson_d8_n20k      : harder, d=8
  D3_Poisson_d4_n50k      : larger N

For each config: M=20 MC reps, 5-fold cross-fit, parallel over MC reps.
Report: bias, |bias|/SE, SE ratio, coverage, Clopper-Pearson CI, PASS/FAIL.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist

# ===========================================================
# config
# ===========================================================
K_FOLDS = 5
DNN_HIDDEN = (64, 32)
DNN_LR = 2e-3
DNN_MAX_EPOCHS = 200
DNN_PATIENCE = 20
DNN_BATCH = 1024
M_MC = 20
N_WORKERS = 10

# ===========================================================
# truth nets (fixed, module-level so workers share)
# ===========================================================
torch.manual_seed(1001)
class TrueNet(nn.Module):
    def __init__(self, d_in, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def _build_truth(d, seed=1001):
    torch.manual_seed(seed)
    a_net = TrueNet(d); b_net = TrueNet(d)
    for p in a_net.parameters(): p.requires_grad_(False)
    for p in b_net.parameters(): p.requires_grad_(False)
    g = torch.Generator().manual_seed(99)
    with torch.no_grad():
        x = torch.randn(200_000, d, generator=g)
        am, as_ = a_net(x).mean().item(), a_net(x).std().item()
        bm, bs = b_net(x).mean().item(), b_net(x).std().item()
    def α(x): return (a_net(x) - am) / as_ * 0.40 + 1.3
    def β(x): return (b_net(x) - bm) / bs * 0.55 - 0.35
    with torch.no_grad():
        xb = torch.randn(1_500_000, d, generator=torch.Generator().manual_seed(101))
        mu_star = β(xb).mean().item()
    return dict(d=d, alpha=α, beta=β, mu_star=mu_star)


TRUTHS = {4: _build_truth(4), 8: _build_truth(8)}


def simulate(rng, n, d):
    T = TRUTHS[d]
    x = rng.standard_normal((n, d)).astype(np.float32)
    t = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = T["alpha"](xt).numpy()
        b = T["beta"](xt).numpy()
    mu = np.exp(a + b * t).astype(np.float32)
    y = rng.poisson(mu).astype(np.float32)
    return x, t, y, a, b, mu, T["mu_star"]


# ===========================================================
# first-stage DNN
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
    xtr = torch.from_numpy(x_tr); ttr = torch.from_numpy(t_tr)
    ytr = torch.from_numpy(y_tr); xva = torch.from_numpy(x_va)
    tva = torch.from_numpy(t_va); yva = torch.from_numpy(y_va)
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


# ===========================================================
# IF computation with optional Λ-trimming
# ===========================================================
def poisson_if(alpha_hat, beta_hat, t, y, trim_eps=0.0, p_t=0.5):
    """
    Binary T, Λ⁻¹ closed-form:  at T=0 correction_coef = -1/((1-p) μ₀),
    at T=1 correction_coef = 1/(p μ₁).  For p=0.5: -2/μ₀ or 2/μ₁.
    """
    mu_0 = np.exp(alpha_hat)
    mu_1 = np.exp(alpha_hat + beta_hat)
    if trim_eps > 0:
        mu_0 = np.maximum(mu_0, trim_eps)
        mu_1 = np.maximum(mu_1, trim_eps)
    mu_realized = np.where(t > 0.5, mu_1, mu_0)
    coef = np.where(t > 0.5, 1.0 / (p_t * mu_1),
                    -1.0 / ((1 - p_t) * mu_0))
    resid = mu_realized - y
    correction = coef * resid
    psi = beta_hat - correction
    return psi, mu_realized, correction


# ===========================================================
# FML with fixes (ensemble and TMLE logic)
# ===========================================================
def cross_fit_psi(x, t, y, seed, trim_eps=0.0, ensemble_r=1):
    """
    Returns: psi (n,), beta_hat (n,), alpha_hat (n,), mu_hat (n,).
    If ensemble_r>1, train R DNNs per fold and average predictions.
    """
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
        a_sum = np.zeros(me.sum()); b_sum = np.zeros(me.sum())
        for r in range(ensemble_r):
            net = train_theta(
                x[tr], t[tr], y[tr], x[va], t[va], y[va],
                seed=seed * 100 + k * 10 + r)
            a_e, b_e = pred(net, x[me])
            a_sum += a_e; b_sum += b_e
        alpha_hat[me] = a_sum / ensemble_r
        beta_hat[me] = b_sum / ensemble_r
    psi, mu_hat, correction = poisson_if(
        alpha_hat, beta_hat, t, y, trim_eps=trim_eps)
    return psi, beta_hat, alpha_hat, mu_hat, correction


def winsorize_psi(psi, lo_q=0.0, hi_q=1.0):
    if lo_q <= 0.0 and hi_q >= 1.0:
        return psi
    lo = np.percentile(psi, lo_q * 100)
    hi = np.percentile(psi, hi_q * 100)
    return np.clip(psi, lo, hi)


def aggregate(psi, method="mean", mom_blocks=None, rng=None):
    """Return point estimate and standard error."""
    n = len(psi)
    if method == "mean":
        return float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(n))
    if method == "mom":
        if rng is None: rng = np.random.default_rng(0)
        idx = rng.permutation(n)
        blocks = np.array_split(idx, mom_blocks)
        bm = np.array([psi[b].mean() for b in blocks])
        return float(np.median(bm)), float(bm.std(ddof=1) / np.sqrt(mom_blocks) * 1.2533)
    raise ValueError(method)


def tmle_one_step(beta_hat, alpha_hat, t, y, trim_eps=0.1, p_t=0.5, max_iter=3):
    """
    One-step TMLE: shift β̂ uniformly by ε chosen to make sum of IF ≈ 0.
    Parametric-submodel fluctuation along the Λ⁻¹ T_vec direction.
    For our H = β target, the fluctuation along the β direction reduces
    finite-sample bias.  ε is chosen by 1-step NLL minimization on the
    updated model.
    """
    for _ in range(max_iter):
        psi, _, correction = poisson_if(alpha_hat, beta_hat, t, y,
                                        trim_eps=trim_eps, p_t=p_t)
        mean_psi = psi.mean()
        mean_beta = beta_hat.mean()
        shift = mean_psi - mean_beta
        if abs(shift) < 1e-6:
            break
        beta_hat = beta_hat + shift
    return beta_hat


def apply_fix(config, x, t, y, seed):
    """Run FML with the configured fix and return (mu, se, diagnostics)."""
    trim = config.get("trim_eps", 0.0)
    ens = config.get("ensemble_r", 1)
    psi, beta_hat, alpha_hat, _, _ = cross_fit_psi(
        x, t, y, seed=seed, trim_eps=trim, ensemble_r=ens)

    if config.get("tmle_iters", 0) > 0:
        beta_hat = tmle_one_step(beta_hat, alpha_hat, t, y,
                                 trim_eps=trim, max_iter=config["tmle_iters"])
        psi, _, _ = poisson_if(alpha_hat, beta_hat, t, y, trim_eps=trim)

    lo, hi = config.get("winsor", (0.0, 1.0))
    psi_w = winsorize_psi(psi, lo, hi)

    agg_method = config.get("aggregate", "mean")
    mom_k = config.get("mom_blocks", 5)
    rng = np.random.default_rng(seed * 1234 + 7)
    mu_hat, se = aggregate(psi_w, method=agg_method, mom_blocks=mom_k, rng=rng)
    mu_naive = float(beta_hat.mean())
    return dict(
        mu_hat=mu_hat, se=se, mu_naive=mu_naive,
        psi_mean=float(psi.mean()), psi_sd=float(psi.std(ddof=1)),
        psi_p01=float(np.percentile(psi, 1)),
        psi_p99=float(np.percentile(psi, 99)),
    )


# ===========================================================
# Fix configs
# ===========================================================
FIXES = {
    "F0_vanilla":        dict(trim_eps=0.0),
    "F1_trim_e010":      dict(trim_eps=0.10),
    "F2_trim_e050":      dict(trim_eps=0.50),
    "F3_winsor_1":       dict(trim_eps=0.0, winsor=(0.01, 0.99)),
    "F4_winsor_5":       dict(trim_eps=0.0, winsor=(0.05, 0.95)),
    "F5_trim_winsor":    dict(trim_eps=0.10, winsor=(0.01, 0.99)),
    "F6_mom5":           dict(trim_eps=0.10, aggregate="mom", mom_blocks=5),
    "F7_ensemble_3":     dict(trim_eps=0.10, ensemble_r=3),
    "F8_tmle_1step":     dict(trim_eps=0.10, tmle_iters=1),
    "F9_all":            dict(trim_eps=0.10, winsor=(0.01, 0.99), ensemble_r=3,
                              tmle_iters=1),
}

# DGPs
DGPS = [
    ("D1_P_d4_n20k", 4, 20_000),
    ("D2_P_d8_n20k", 8, 20_000),
    ("D3_P_d4_n50k", 4, 50_000),
]


# ===========================================================
# MC worker
# ===========================================================
def run_one(payload):
    fix_name, fix_cfg, dgp_name, d, n, mc_seed = payload
    rng = np.random.default_rng(mc_seed)
    x, t, y, a_obs, b_obs, mu_obs, mu_star = simulate(rng, n, d)
    result = apply_fix(fix_cfg, x, t, y, seed=mc_seed)
    result["mu_star"] = mu_star
    result["covered"] = bool(
        result["mu_hat"] - 1.96 * result["se"] <= mu_star <=
        result["mu_hat"] + 1.96 * result["se"])
    result["fix"] = fix_name
    result["dgp"] = dgp_name
    result["mc_seed"] = mc_seed
    return result


def cp_ci(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def summarize(fix_name, dgp_name, results):
    mu_f = np.array([r["mu_hat"] for r in results])
    mu_n = np.array([r["mu_naive"] for r in results])
    se = np.array([r["se"] for r in results])
    cov = np.array([r["covered"] for r in results])
    mu_star = results[0]["mu_star"]
    bias = float(mu_f.mean() - mu_star)
    bias_n = float(mu_n.mean() - mu_star)
    sd = float(mu_f.std(ddof=1))
    se_r = float(se.mean() / sd) if sd > 0 else float("nan")
    cov_frac = float(cov.mean())
    k_cov = int(cov.sum())
    lo, hi = cp_ci(k_cov, len(results))
    mean_se = float(se.mean())
    pass_bias = abs(bias) < 0.1 * mean_se
    pass_se = 0.7 <= se_r <= 1.5
    pass_cov = 0.90 <= cov_frac <= 0.99
    verdict = "PASS" if all([pass_bias, pass_se, pass_cov]) else (
        "PARTIAL" if pass_cov else "FAIL")
    return dict(
        fix=fix_name, dgp=dgp_name,
        M=len(results), mu_star=mu_star,
        bias=bias, bias_naive=bias_n, mean_se=mean_se, emp_sd=sd,
        se_ratio=se_r, coverage=cov_frac, cp_lo=float(lo), cp_hi=float(hi),
        abs_bias_over_se=abs(bias) / mean_se if mean_se > 0 else float("nan"),
        pass_bias=pass_bias, pass_se=pass_se, pass_cov=pass_cov,
        verdict=verdict,
    )


def run_matrix():
    t0_all = time.time()
    payloads = []
    for dgp_name, d, n in DGPS:
        for fix_name, fix_cfg in FIXES.items():
            for m in range(M_MC):
                payloads.append((fix_name, fix_cfg, dgp_name, d, n,
                                 9000 + m * 31))
    print(f"Total runs: {len(payloads)} ({len(FIXES)} fixes x "
          f"{len(DGPS)} DGPs x {M_MC} reps)")
    print(f"Launching on {N_WORKERS} workers...\n")

    results_by_group = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_one, p) for p in payloads]
        done = 0
        for f in as_completed(futs):
            r = f.result()
            key = (r["fix"], r["dgp"])
            results_by_group.setdefault(key, []).append(r)
            done += 1
            if done % 20 == 0 or done == len(futs):
                dt = time.time() - t0
                eta = dt * (len(futs) - done) / done if done > 0 else 0
                print(f"  {done:4d}/{len(futs)}  t={dt:5.0f}s  eta={eta:.0f}s")

    summaries = []
    for (fix, dgp), res in sorted(results_by_group.items()):
        s = summarize(fix, dgp, res)
        summaries.append(s)

    # --------- matrix print ---------
    print("\n" + "=" * 110)
    print("ROBUSTNESS MATRIX (rows=fix, cols=DGP)".center(110))
    print("=" * 110)
    dgp_names = [n for n, _, _ in DGPS]
    fix_names = list(FIXES.keys())
    # Header
    print(f"\n  {'fix':<22}", end="")
    for d in dgp_names:
        print(f"  {d[:18]:<18}", end="")
    print()
    print("  " + "-" * (22 + len(dgp_names) * 20))
    # Rows
    lookup = {(s["fix"], s["dgp"]): s for s in summaries}
    for fix in fix_names:
        print(f"  {fix:<22}", end="")
        for dgp in dgp_names:
            s = lookup[(fix, dgp)]
            tag = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}[s["verdict"]]
            print(f"  {tag}{s['coverage']:>5.2f}/"
                  f"{s['abs_bias_over_se']:>4.2f}/"
                  f"{s['se_ratio']:<5.2f}", end="")
        print()
    print("  (cells: cov / |b|/SE / SE-ratio.  ✓ PASS  ~ PARTIAL  ✗ FAIL)")

    # -------- detail tables per DGP ----------
    for dgp in dgp_names:
        print(f"\n{'-' * 110}")
        print(f"DETAIL: {dgp}  (µ*={lookup[(fix_names[0], dgp)]['mu_star']:+.5f})"
              .center(110))
        print(f"{'-' * 110}")
        print(f"  {'fix':<22}{'bias':>9}{'b_naive':>10}"
              f"{'|b|/SE':>8}{'SEratio':>9}{'cov':>7}{'CP-CI':>14}"
              f"{'verdict':>10}")
        for fix in fix_names:
            s = lookup[(fix, dgp)]
            print(f"  {s['fix']:<22}"
                  f"{s['bias']:>+9.4f}"
                  f"{s['bias_naive']:>+10.4f}"
                  f"{s['abs_bias_over_se']:>8.3f}"
                  f"{s['se_ratio']:>9.3f}"
                  f"{s['coverage']:>7.3f}"
                  f" [{s['cp_lo']:.2f},{s['cp_hi']:.2f}]"
                  f"{s['verdict']:>10}")

    # --------- winner per DGP ---------
    print("\n" + "=" * 110)
    print("BEST FIX PER DGP (by coverage gate + lowest |bias|/SE)".center(110))
    print("=" * 110)
    for dgp in dgp_names:
        dgp_summaries = [s for s in summaries if s["dgp"] == dgp]
        # rank: first by pass_cov, then by lowest |b|/SE
        dgp_summaries.sort(
            key=lambda s: (-int(s["pass_cov"]), s["abs_bias_over_se"]))
        best = dgp_summaries[0]
        print(f"  {dgp}:  best fix = {best['fix']}  "
              f"(cov {best['coverage']:.3f}, |b|/SE {best['abs_bias_over_se']:.3f}, "
              f"SE ratio {best['se_ratio']:.3f}, {best['verdict']})")

    # --------- save ---------
    with open("/tmp/qpoisson_if/flm_robustness_results.json", "w") as f:
        json.dump(summaries, f, indent=1, default=str)
    print(f"\n[saved] /tmp/qpoisson_if/flm_robustness_results.json")
    print(f"[total wall time] {time.time() - t0_all:.0f}s")


if __name__ == "__main__":
    run_matrix()
