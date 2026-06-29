"""
Four rescue tests for FML coverage failure in the d=16 continuous-X setup.

Baseline (A)  : d=16, (64,32) ReLU DNN   [the failing case from flm_scaling.py]

Rescue (B)    : d=4 continuous covariates, same DNN.  Lower dim should give
                faster nonparametric rate (n^{-2p/(2p+d)}) and honest coverage.

Rescue (C)    : d=2, same DNN.  Extreme case; should basically match
                parametric √n.

Rescue (D)    : d=16, architecture-matched DNN: three independent Tanh MLPs
                with 32→32 hidden, exactly matching the true networks.
                Tests whether ReLU mismatch was the bottleneck.

Rescue (E)    : d=16, 3-way sample split.  Train θ̂_main on first half of the
                training folds, θ̂_Λ on second half.  Main uses θ̂_main for
                β̂(x_i); correction uses Λ̂ constructed from θ̂_Λ.  Decouples
                the two nuisances per FLM §3.3 prescription.

All at N=50_000, M_MC=60 reps.  Report bias, |bias|/emp_SD, coverage,
CP-CI, L2(α̂), L2(β1̂), DNN epochs used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist

# =====================================================
# Shared constants
# =====================================================
PHI = 2.0
K_FOLDS = 5
DNN_MAX_EPOCHS = 300
DNN_BATCH = 1024
DNN_PATIENCE = 25
DNN_LR = 2e-3
N_WORKERS = 10
N_TEST = 50_000
M_MC = 60

P_PERIODS = (0.4, 0.3, 0.3)
P0, P1, P2 = P_PERIODS


# =====================================================
# Fresh truth for a given input dim
# =====================================================
class _TrueNet(nn.Module):
    def __init__(self, d_in, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_truth(d_emb, seed=1001):
    """Create truth networks for given per-side embedding dim.  Returns a
    dict of callables + reference µ*."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    alpha_raw = _TrueNet(2 * d_emb)
    beta1_raw = _TrueNet(2 * d_emb)
    beta2_raw = _TrueNet(2 * d_emb)
    for p in alpha_raw.parameters(): p.requires_grad_(False)
    for p in beta1_raw.parameters(): p.requires_grad_(False)
    for p in beta2_raw.parameters(): p.requires_grad_(False)

    BASE = 1.3
    ALPHA_STD = 0.40
    B1_M, B1_S = -0.35, 0.55
    B2_M, B2_S = +0.10, 0.40

    g2 = torch.Generator().manual_seed(99)
    with torch.no_grad():
        xc = torch.randn(300_000, 2 * d_emb, generator=g2)
        a_c = alpha_raw(xc)
        b1_c = beta1_raw(xc)
        b2_c = beta2_raw(xc)
    A_MEAN, A_STD = a_c.mean().item(), a_c.std().item()
    B1_MEAN, B1_STD = b1_c.mean().item(), b1_c.std().item()
    B2_MEAN, B2_STD = b2_c.mean().item(), b2_c.std().item()

    def alpha_t(x):
        return (alpha_raw(x) - A_MEAN) / A_STD * ALPHA_STD + BASE
    def beta1_t(x):
        return (beta1_raw(x) - B1_MEAN) / B1_STD * B1_S + B1_M
    def beta2_t(x):
        return (beta2_raw(x) - B2_MEAN) / B2_STD * B2_S + B2_M

    with torch.no_grad():
        x_big = torch.randn(2_000_000, 2 * d_emb,
                            generator=torch.Generator().manual_seed(101))
        MU_STAR = beta1_t(x_big).mean().item()
    with torch.no_grad():
        x_eval = torch.randn(8000, 2 * d_emb,
                             generator=torch.Generator().manual_seed(202))
        A_EV = alpha_t(x_eval).numpy()
        B1_EV = beta1_t(x_eval).numpy()
        B2_EV = beta2_t(x_eval).numpy()
    return dict(d_emb=d_emb, alpha_t=alpha_t, beta1_t=beta1_t, beta2_t=beta2_t,
                MU_STAR=MU_STAR, X_EV=x_eval.numpy(),
                A_EV=A_EV, B1_EV=B1_EV, B2_EV=B2_EV)


# precompute truths for d ∈ {2, 4, 16}.  Must be top-level to be picklable.
TRUTH_D16 = make_truth(d_emb=8, seed=1001)
TRUTH_D4 = make_truth(d_emb=2, seed=1001)
TRUTH_D2 = make_truth(d_emb=1, seed=1001)
TRUTHS = {16: TRUTH_D16, 4: TRUTH_D4, 2: TRUTH_D2}
for d, T in TRUTHS.items():
    print(f"[truth] d={d}  µ* = {T['MU_STAR']:+.5f}  "
          f"β1* std={T['B1_EV'].std():.3f}  "
          f"β1* range=[{T['B1_EV'].min():+.3f}, {T['B1_EV'].max():+.3f}]")


def simulate_d(rng, n, truth):
    d_emb = truth["d_emb"]
    x = rng.standard_normal((n, 2 * d_emb)).astype(np.float32)
    t = rng.choice(3, size=n, p=P_PERIODS).astype(np.int64)
    D1 = (t == 1).astype(np.float32); D2 = (t == 2).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = truth["alpha_t"](xt).numpy()
        b1 = truth["beta1_t"](xt).numpy()
        b2 = truth["beta2_t"](xt).numpy()
    mu = np.exp(a + b1 * D1 + b2 * D2)
    theta = mu / (PHI - 1.0)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p).astype(np.float32)
    return x, t, D1, D2, y, a, b1, b2, mu


# =====================================================
# DNN architectures
# =====================================================
class ReluThetaNet(nn.Module):
    """Baseline: shared trunk + 3 heads, ReLU activations."""
    def __init__(self, d_in, hidden=(64, 32)):
        super().__init__()
        L, prev = [], d_in
        for h in hidden:
            L += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.trunk = nn.Sequential(*L)
        self.head_a = nn.Linear(prev, 1)
        self.head_b1 = nn.Linear(prev, 1)
        self.head_b2 = nn.Linear(prev, 1)
    def forward(self, x):
        h = self.trunk(x)
        return (self.head_a(h).squeeze(-1),
                self.head_b1(h).squeeze(-1),
                self.head_b2(h).squeeze(-1))


class TanhMatchedThetaNet(nn.Module):
    """Structured: three INDEPENDENT Tanh nets matching true architecture
    exactly (2*d_in → 32 → 32 → 1 with Tanh)."""
    def __init__(self, d_in, d_hidden=32):
        super().__init__()
        def mk():
            return nn.Sequential(
                nn.Linear(d_in, d_hidden), nn.Tanh(),
                nn.Linear(d_hidden, d_hidden), nn.Tanh(),
                nn.Linear(d_hidden, 1),
            )
        self.net_a = mk()
        self.net_b1 = mk()
        self.net_b2 = mk()
    def forward(self, x):
        return (self.net_a(x).squeeze(-1),
                self.net_b1(x).squeeze(-1),
                self.net_b2(x).squeeze(-1))


def poisson_nll(mu, y):
    return (mu - y * torch.log(mu + 1e-10)).mean()


def train_theta(net, x_tr, t_tr, D1_tr, D2_tr, y_tr,
                x_va, t_va, D1_va, D2_va, y_va,
                max_epochs=DNN_MAX_EPOCHS, patience=DNN_PATIENCE, lr=DNN_LR):
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    xtr = torch.from_numpy(x_tr); ytr = torch.from_numpy(y_tr)
    D1tr = torch.from_numpy(D1_tr); D2tr = torch.from_numpy(D2_tr)
    xva = torch.from_numpy(x_va); yva = torch.from_numpy(y_va)
    D1va = torch.from_numpy(D1_va); D2va = torch.from_numpy(D2_va)
    n = len(xtr)
    best_val = float("inf"); best_state = None; pat = 0; ep_used = 0
    for ep in range(max_epochs):
        net.train()
        idx = np.random.permutation(n)
        for s in range(0, n, DNN_BATCH):
            sel = idx[s:s + DNN_BATCH]
            a, b1, b2 = net(xtr[sel])
            mu = torch.exp(a + b1 * D1tr[sel] + b2 * D2tr[sel])
            loss = poisson_nll(mu, ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            av, b1v, b2v = net(xva)
            muv = torch.exp(av + b1v * D1va + b2v * D2va)
            val = poisson_nll(muv, yva).item()
        ep_used = ep + 1
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience: break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val, ep_used


def predict(net, x):
    net.eval()
    with torch.no_grad():
        a, b1, b2 = net(torch.from_numpy(x))
    return a.numpy(), b1.numpy(), b2.numpy()


# =====================================================
# IF for H=β1 with the 3-period (0,D1,D2) structure
# =====================================================
def fml_if(a_main, b1_main, b2_main, a_lam, b1_lam, b2_lam, t, D1, D2, y):
    """Main term uses (a_main, b1_main, b2_main) via β̂_main(x).
       Correction uses Λ̂ built from (a_lam, b1_lam, b2_lam) AND
       μ̂ built from (a_main, b1_main, b2_main) on the realized t.

    For the standard 2-way split, a_main==a_lam, etc.
    For 3-way split, they come from different halves of the training data.
    """
    mu_0_main = np.exp(a_main)
    mu_1_main = np.exp(a_main + b1_main)
    mu_2_main = np.exp(a_main + b2_main)
    mu_hat = np.where(t == 0, mu_0_main, np.where(t == 1, mu_1_main, mu_2_main))

    mu_0_lam = np.exp(a_lam)
    mu_1_lam = np.exp(a_lam + b1_lam)
    m_0 = P0 * mu_0_lam
    m_1 = P1 * mu_1_lam
    correction_coef = -1.0 / m_0 + D1 * (1.0 / m_1 + 1.0 / m_0) + D2 * (1.0 / m_0)
    resid = mu_hat - y
    psi = b1_main - correction_coef * resid
    return mu_hat, psi, correction_coef * resid, resid


# =====================================================
# DML2 driver (2-way split)
# =====================================================
def run_2way(payload):
    config_id, d, arch, N, seed = payload
    T = TRUTHS[d]
    np.random.seed(seed); torch.manual_seed(seed + 100)
    rng = np.random.default_rng(seed)
    x, t, D1, D2, y, a_obs, b1_obs, b2_obs, mu_obs = simulate_d(rng, N, T)
    folds = rng.integers(K_FOLDS, size=N)
    alpha_hat = np.zeros(N); b1_hat = np.zeros(N); b2_hat = np.zeros(N)
    psi = np.zeros(N)
    l2a = np.zeros(K_FOLDS); l2b1 = np.zeros(K_FOLDS)
    eps_used = np.zeros(K_FOLDS, dtype=int)
    for k in range(K_FOLDS):
        me = folds == k
        tr = np.where(~me)[0]
        perm = np.random.default_rng(seed * 17 + k).permutation(len(tr))
        n_val = max(int(0.15 * len(tr)), 100)
        va_idx = tr[perm[:n_val]]
        tr_idx = tr[perm[n_val:]]
        if arch == "relu":
            net = ReluThetaNet(x.shape[1])
        elif arch == "tanh_matched":
            net = TanhMatchedThetaNet(x.shape[1])
        net, val, ep = train_theta(
            net,
            x[tr_idx], t[tr_idx], D1[tr_idx], D2[tr_idx], y[tr_idx],
            x[va_idx], t[va_idx], D1[va_idx], D2[va_idx], y[va_idx],
        )
        eps_used[k] = ep
        with torch.no_grad():
            ae, b1e, b2e = net(torch.from_numpy(T["X_EV"]))
        l2a[k] = float(((ae.numpy() - T["A_EV"]) ** 2).mean())
        l2b1[k] = float(((b1e.numpy() - T["B1_EV"]) ** 2).mean())
        a_e, b1_e, b2_e = predict(net, x[me])
        alpha_hat[me] = a_e; b1_hat[me] = b1_e; b2_hat[me] = b2_e
        _, p_e, _, _ = fml_if(a_e, b1_e, b2_e, a_e, b1_e, b2_e,
                               t[me], D1[me], D2[me], y[me])
        psi[me] = p_e
    return _package(config_id, T, N, psi, b1_hat, alpha_hat, b1_obs, a_obs,
                    l2a, l2b1, eps_used)


def run_3way(payload):
    """3-way split: for each held-out fold k, split the K-1 training folds
    into two halves.  Train θ̂_main on first half (used for β̂(x_i) and μ̂),
    θ̂_Λ on second half (used to construct Λ̂(x_i) in the correction)."""
    config_id, d, N, seed = payload
    T = TRUTHS[d]
    np.random.seed(seed); torch.manual_seed(seed + 100)
    rng = np.random.default_rng(seed)
    x, t, D1, D2, y, a_obs, b1_obs, b2_obs, mu_obs = simulate_d(rng, N, T)
    folds = rng.integers(K_FOLDS, size=N)
    alpha_hat = np.zeros(N); b1_hat = np.zeros(N)
    psi = np.zeros(N)
    l2a = np.zeros(K_FOLDS); l2b1 = np.zeros(K_FOLDS)
    eps_used = np.zeros(K_FOLDS, dtype=int)
    for k in range(K_FOLDS):
        me = folds == k
        tr = np.where(~me)[0]
        perm = np.random.default_rng(seed * 17 + k).permutation(len(tr))
        n_val = max(int(0.15 * len(tr)), 100)
        va_idx = tr[perm[:n_val]]
        tr_rest = tr[perm[n_val:]]
        half = len(tr_rest) // 2
        tr_main = tr_rest[:half]
        tr_lam = tr_rest[half:]
        # train θ̂_main on tr_main
        net_main = ReluThetaNet(x.shape[1])
        net_main, _, ep_m = train_theta(
            net_main,
            x[tr_main], t[tr_main], D1[tr_main], D2[tr_main], y[tr_main],
            x[va_idx], t[va_idx], D1[va_idx], D2[va_idx], y[va_idx],
        )
        # train θ̂_Λ on tr_lam
        net_lam = ReluThetaNet(x.shape[1])
        net_lam, _, ep_l = train_theta(
            net_lam,
            x[tr_lam], t[tr_lam], D1[tr_lam], D2[tr_lam], y[tr_lam],
            x[va_idx], t[va_idx], D1[va_idx], D2[va_idx], y[va_idx],
        )
        eps_used[k] = (ep_m + ep_l) // 2
        with torch.no_grad():
            ae, b1e, b2e = net_main(torch.from_numpy(T["X_EV"]))
        l2a[k] = float(((ae.numpy() - T["A_EV"]) ** 2).mean())
        l2b1[k] = float(((b1e.numpy() - T["B1_EV"]) ** 2).mean())
        a_m, b1_m, b2_m = predict(net_main, x[me])
        a_l, b1_l, b2_l = predict(net_lam, x[me])
        alpha_hat[me] = a_m; b1_hat[me] = b1_m
        _, p_e, _, _ = fml_if(a_m, b1_m, b2_m, a_l, b1_l, b2_l,
                               t[me], D1[me], D2[me], y[me])
        psi[me] = p_e
    return _package(config_id, T, N, psi, b1_hat, alpha_hat, b1_obs, a_obs,
                    l2a, l2b1, eps_used)


def _package(config_id, T, N, psi, b1_hat, alpha_hat, b1_obs, a_obs,
             l2a, l2b1, eps_used):
    mu_naive = float(b1_hat.mean())
    mu_fml = float(psi.mean())
    se_fml = float(psi.std(ddof=1) / np.sqrt(N))
    u = alpha_hat - a_obs
    v = b1_hat - b1_obs
    return dict(config_id=config_id, MU_STAR=T["MU_STAR"],
                N=N, mu_naive=mu_naive, mu_fml=mu_fml, se_fml=se_fml,
                l2_a=float(l2a.mean()), l2_b1=float(l2b1.mean()),
                epochs_mean=float(eps_used.mean()),
                pred_bias=float(0.5 * (v ** 2).mean() + (u * v).mean()))


def cp_ci(k, n, a=0.05):
    lo = beta_dist.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def summarize(name, results):
    MU_STAR = results[0]["MU_STAR"]
    mu_n = np.array([r["mu_naive"] for r in results])
    mu_f = np.array([r["mu_fml"] for r in results])
    se_f = np.array([r["se_fml"] for r in results])
    pb = np.array([r["pred_bias"] for r in results])
    l2a = np.array([r["l2_a"] for r in results])
    l2b1 = np.array([r["l2_b1"] for r in results])
    eps = np.array([r["epochs_mean"] for r in results])
    bias = mu_f.mean() - MU_STAR
    emp_sd = mu_f.std(ddof=1)
    cov = np.abs(mu_f - MU_STAR) <= 1.96 * se_f
    k_cov = int(cov.sum())
    lo, hi = cp_ci(k_cov, len(results))
    print(f"\n  --- {name} ---  µ*={MU_STAR:+.5f}")
    print(f"    M={len(results)}  mean_epochs={eps.mean():.1f}  "
          f"L2(α̂)={l2a.mean():.5f}  L2(β1̂)={l2b1.mean():.5f}")
    print(f"    MC bias (naive/FML):    {mu_n.mean()-MU_STAR:+.5f}  /  {bias:+.5f}")
    print(f"    predicted FML bias      {pb.mean():+.5f}   "
          f"ratio pred/obs {pb.mean()/bias if abs(bias)>1e-6 else float('nan'):.2f}")
    print(f"    emp SD of µ̂_FML         {emp_sd:.5f}")
    print(f"    mean IF SE               {se_f.mean():.5f}  "
          f"(SE/SD {se_f.mean()/emp_sd:.3f})")
    print(f"    |bias|/emp_SD            {abs(bias)/emp_sd:.3f}  "
          f"(PASS if <0.1 or <0.3 tolerable)")
    print(f"    95% coverage             {cov.mean():.3f}  "
          f"CP-CI [{lo:.3f}, {hi:.3f}]  (PASS if CI includes 0.95)")
    return dict(name=name, bias=bias, emp_sd=emp_sd, cov=cov.mean(), cp_lo=lo,
                cp_hi=hi, l2_a=l2a.mean(), l2_b1=l2b1.mean(),
                mean_se=se_f.mean(), ratio_bias_sd=abs(bias)/emp_sd,
                predicted_bias=pb.mean(), epochs=eps.mean())


# =====================================================
# Runner
# =====================================================
def run_config(label, run_fn, payloads):
    print(f"\n{'='*78}")
    print(f"{label}".center(78))
    print(f"{'='*78}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = [pool.submit(run_fn, p) for p in payloads]
        results = [f.result() for f in as_completed(futs)]
    print(f"  wall time {time.time()-t0:.0f}s")
    return summarize(label, results)


if __name__ == "__main__":
    t_all = time.time()
    all_summaries = []

    print("\n" + "#" * 78)
    print(f"#  Rescue sweep: N={N_TEST}, M_MC={M_MC} per config, workers={N_WORKERS}")
    print("#" * 78)

    # (A) baseline (d=16, ReLU)
    payloads = [("A", 16, "relu", N_TEST, 7000 + m * 31) for m in range(M_MC)]
    all_summaries.append(run_config(
        "(A) BASELINE: d=16, (64,32) ReLU DNN", run_2way, payloads))

    # (B) d=4
    payloads = [("B", 4, "relu", N_TEST, 7000 + m * 31) for m in range(M_MC)]
    all_summaries.append(run_config(
        "(B) RESCUE: d=4, (64,32) ReLU DNN", run_2way, payloads))

    # (C) d=2
    payloads = [("C", 2, "relu", N_TEST, 7000 + m * 31) for m in range(M_MC)]
    all_summaries.append(run_config(
        "(C) RESCUE: d=2, (64,32) ReLU DNN", run_2way, payloads))

    # (D) d=16, Tanh-matched
    payloads = [("D", 16, "tanh_matched", N_TEST, 7000 + m * 31) for m in range(M_MC)]
    all_summaries.append(run_config(
        "(D) RESCUE: d=16, Tanh-matched 3-net architecture", run_2way, payloads))

    # (E) d=16, 3-way split
    payloads = [("E", 16, N_TEST, 7000 + m * 31) for m in range(M_MC)]
    all_summaries.append(run_config(
        "(E) RESCUE: d=16, 3-way split (decoupled θ̂_main, θ̂_Λ)", run_3way, payloads))

    # Final comparison
    print("\n" + "=" * 78)
    print("FINAL RESCUE COMPARISON".center(78))
    print("=" * 78)
    hdr = (f"  {'config':<42}{'bias':>9}{'|b|/SD':>8}"
           f"{'cov':>7}{'CP-CI':>14}{'L2(β1̂)':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in all_summaries:
        print(f"  {s['name']:<42}{s['bias']:>+9.5f}"
              f"{s['ratio_bias_sd']:>8.3f}"
              f"{s['cov']:>7.3f}"
              f" [{s['cp_lo']:.2f},{s['cp_hi']:.2f}] "
              f"{s['l2_b1']:>10.5f}")

    # winner
    print()
    verdicts = []
    for s in all_summaries:
        pass_bias = s['ratio_bias_sd'] < 0.3
        pass_cov = s['cp_lo'] <= 0.95 <= s['cp_hi']
        status = "PASS" if (pass_bias and pass_cov) else \
                 ("PARTIAL" if pass_cov else "FAIL")
        verdicts.append((s['name'], status))
        print(f"  {s['name']:<42}  {status}")

    # save
    out = {"MU_STAR_by_d": {d: T["MU_STAR"] for d, T in TRUTHS.items()},
           "summaries": [{k: (float(v) if isinstance(v, (np.floating,)) else v)
                          for k, v in s.items()} for s in all_summaries]}
    with open("/tmp/qpoisson_if/flm_rescue_results.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\n[saved] /tmp/qpoisson_if/flm_rescue_results.json")
    print(f"[total time] {time.time()-t_all:.0f}s")
