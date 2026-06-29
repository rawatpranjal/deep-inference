"""
FML deployment for a Poisson event study with event-time T ∈ {0=pre, 1=during, 2=post}.

Match to the paper's structure:
    log E[y_i | x_i, t_i] = α*(x_i) + β1*(x_i)·D1_i + β2*(x_i)·D2_i
    D1 = I{t=1} (during), D2 = I{t=2} (post)
    Var(y|x,t) = φ · E[y|x,t]            (quasi-Poisson, NB2 draws)
    X = (z_u, z_j) ∈ R^{16}  iid Gaussian
    α*, β1*, β2* are each outputs of a fixed random neural net ("truth")

Target (low-dim):
    µ* = E[β1*(X)]   (average lockdown effect)

FML influence function for H=β1:
    T_vec = (1, D1, D2)'
    Λ(x) = E[ μ(x,T) T_vec T_vec' | X=x ]
         = [[ Σ p_k μ_k,  p1 μ1,  p2 μ2 ],
            [  p1 μ1,     p1 μ1,  0    ],
            [  p2 μ2,     0,      p2 μ2 ]]
    det(Λ) = p0 p1 p2 μ0 μ1 μ2         (where μ_k = μ at T=k)
    [Λ⁻¹]_{β1-row} = (-1/m0,  1/m1 + 1/m0,  1/m0)     (m_k = p_k μ_k)
    correction = (-1/m0 + D1·(1/m1+1/m0) + D2·(1/m0)) · (μ̂ − y)
    ψ = β̂1(x) − correction

Neyman-orthogonality (verified analytically):
    E[∂ψ/∂θ | X] = 0 at θ = θ*.

Honest limitations of this sim vs the actual paper:
  - X is iid Gaussian, not three-tower embeddings with real cluster structure.
  - Each obs is iid; no panel repetition, no cluster-robust SE.
  - Target is scalar E[β1*]; paper's target is per-cell τ_{u,j} with FDR.

Diagnostic plan:
  SWEEP N over {5k, 10k, 25k, 50k, 100k, 200k}, M_MC reps per N.
  At each N track:
    - µ̂_naive (plug-in),  µ̂_FML (orthogonal)
    - IF SE, multiplier-bootstrap SE, empirical MC SD
    - 2nd-order predicted bias: 0.5·E[v²] + E[u·v] where v = β1̂ − β1*, u = α̂ − α*
    - DNN health: L2(α̂−α*), L2(β1̂−β1*), L2(β2̂−β2*), epochs used
    - coverage of 95% CI (Clopper-Pearson CI on coverage)
    - cross-N: show bias decays as N grows AND coverage converges to 0.95

HONEST PASS/FAIL CRITERIA:
    PASS iff at N=200k (or smaller): |FML bias|/empirical_SD < 0.1 AND
          coverage CP-CI contains 0.95 AND
          (predicted bias from L2 rates) matches observed bias within 30%.
    Anything less is a partial pass or fail; I will report it plainly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist

# ======================================================================
# config
# ======================================================================
D_EMB = 8
PHI = 2.0
K_FOLDS = 5
DNN_HIDDEN = (64, 32)
DNN_LR = 2e-3
DNN_MAX_EPOCHS = 300
DNN_BATCH = 1024
DNN_PATIENCE = 25
N_WORKERS = 10
P_PERIODS = (0.4, 0.3, 0.3)             # pre / during / post probabilities
P0, P1, P2 = P_PERIODS

# ======================================================================
# fixed "truth" nets for α*, β1*, β2*
# ======================================================================
torch.manual_seed(1001)
class TrueNet(nn.Module):
    def __init__(self, d_in, d_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, d_hidden), nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

_alpha_raw = TrueNet(2 * D_EMB)
_beta1_raw = TrueNet(2 * D_EMB)
_beta2_raw = TrueNet(2 * D_EMB)
for p in _alpha_raw.parameters(): p.requires_grad_(False)
for p in _beta1_raw.parameters(): p.requires_grad_(False)
for p in _beta2_raw.parameters(): p.requires_grad_(False)

# scales: baseline μ around exp(1.3)=3.7; lockdown effect avg -0.3, post effect avg +0.1
BASE = 1.3
ALPHA_STD = 0.40
BETA1_MEAN, BETA1_STD = -0.35, 0.55     # lockdown: negative average, heterogeneous
BETA2_MEAN, BETA2_STD = +0.10, 0.40     # post: mild positive average

_gen = torch.Generator().manual_seed(99)
with torch.no_grad():
    x_cal = torch.randn(300_000, 2 * D_EMB, generator=_gen)
    a_cal = _alpha_raw(x_cal); b1_cal = _beta1_raw(x_cal); b2_cal = _beta2_raw(x_cal)
A_MEAN, A_STD = a_cal.mean().item(), a_cal.std().item()
B1_MEAN, B1_STD = b1_cal.mean().item(), b1_cal.std().item()
B2_MEAN, B2_STD = b2_cal.mean().item(), b2_cal.std().item()

def alpha_true_torch(x):
    return (_alpha_raw(x) - A_MEAN) / A_STD * ALPHA_STD + BASE

def beta1_true_torch(x):
    return (_beta1_raw(x) - B1_MEAN) / B1_STD * BETA1_STD + BETA1_MEAN

def beta2_true_torch(x):
    return (_beta2_raw(x) - B2_MEAN) / B2_STD * BETA2_STD + BETA2_MEAN

# high-precision µ* = E[β1*(X)]
with torch.no_grad():
    _x_big = torch.randn(2_000_000, 2 * D_EMB,
                         generator=torch.Generator().manual_seed(101))
    MU_STAR = beta1_true_torch(_x_big).mean().item()

# shared eval grid for L2 tracking
with torch.no_grad():
    _x_eval = torch.randn(8000, 2 * D_EMB,
                          generator=torch.Generator().manual_seed(202))
    ALPHA_STAR_EVAL = alpha_true_torch(_x_eval).numpy()
    BETA1_STAR_EVAL = beta1_true_torch(_x_eval).numpy()
    BETA2_STAR_EVAL = beta2_true_torch(_x_eval).numpy()
X_EVAL_NP = _x_eval.numpy()

# ======================================================================
# DGP
# ======================================================================
def simulate(rng, n):
    x = rng.standard_normal((n, 2 * D_EMB)).astype(np.float32)
    t = rng.choice(3, size=n, p=P_PERIODS).astype(np.int64)
    D1 = (t == 1).astype(np.float32)
    D2 = (t == 2).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)
        a = alpha_true_torch(xt).numpy()
        b1 = beta1_true_torch(xt).numpy()
        b2 = beta2_true_torch(xt).numpy()
    log_mu = a + b1 * D1 + b2 * D2
    mu = np.exp(log_mu)
    theta = mu / (PHI - 1.0)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p).astype(np.float32)
    return x, t, D1, D2, y, a, b1, b2, mu

# ======================================================================
# θ(x) = (α(x), β1(x), β2(x)) DNN
# ======================================================================
class ThetaNet(nn.Module):
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


def poisson_nll(mu, y):
    return (mu - y * torch.log(mu + 1e-10)).mean()


def train_theta(x_tr, t_tr, D1_tr, D2_tr, y_tr,
                x_va, t_va, D1_va, D2_va, y_va):
    net = ThetaNet(x_tr.shape[1], DNN_HIDDEN)
    opt = optim.Adam(net.parameters(), lr=DNN_LR, weight_decay=1e-5)
    xtr = torch.from_numpy(x_tr); ytr = torch.from_numpy(y_tr)
    D1tr = torch.from_numpy(D1_tr); D2tr = torch.from_numpy(D2_tr)
    xva = torch.from_numpy(x_va); yva = torch.from_numpy(y_va)
    D1va = torch.from_numpy(D1_va); D2va = torch.from_numpy(D2_va)
    n = len(xtr)
    best_val = float("inf"); best_state = None; patience = 0
    ep_used = 0
    for ep in range(DNN_MAX_EPOCHS):
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
            patience = 0
        else:
            patience += 1
            if patience >= DNN_PATIENCE: break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, best_val, ep_used


def predict(net, x):
    net.eval()
    with torch.no_grad():
        a, b1, b2 = net(torch.from_numpy(x))
    return a.numpy(), b1.numpy(), b2.numpy()

# ======================================================================
# FML IF for H = β1, p=(P0, P1, P2)
# ======================================================================
def fml_if(alpha_hat, beta1_hat, beta2_hat, t, D1, D2, y):
    mu_0 = np.exp(alpha_hat)
    mu_1 = np.exp(alpha_hat + beta1_hat)
    mu_2 = np.exp(alpha_hat + beta2_hat)
    # realized μ̂_i
    mu_hat = np.where(t == 0, mu_0, np.where(t == 1, mu_1, mu_2))
    m_0 = P0 * mu_0; m_1 = P1 * mu_1; m_2 = P2 * mu_2
    # [Λ⁻¹]_{β1-row} = (-1/m_0,  1/m_1 + 1/m_0,  1/m_0)
    # correction_coef = -1/m_0 + D1·(1/m_1 + 1/m_0) + D2·(1/m_0)
    correction_coef = -1.0 / m_0 + D1 * (1.0 / m_1 + 1.0 / m_0) + D2 * (1.0 / m_0)
    resid = mu_hat - y
    correction = correction_coef * resid
    psi = beta1_hat - correction
    return mu_hat, psi, correction, resid

# ======================================================================
# FML cross-fitting (K folds)
# ======================================================================
def run_one_rep(payload):
    N, mc_seed = payload
    np.random.seed(mc_seed)
    torch.manual_seed(mc_seed + 100)
    rng = np.random.default_rng(mc_seed)

    x, t, D1, D2, y, a_obs, b1_obs, b2_obs, mu_obs = simulate(rng, N)
    folds = rng.integers(K_FOLDS, size=N)

    alpha_hat = np.zeros(N); b1_hat = np.zeros(N); b2_hat = np.zeros(N)
    psi = np.zeros(N); corr = np.zeros(N); mu_hat = np.zeros(N)
    fold_l2_a = np.zeros(K_FOLDS); fold_l2_b1 = np.zeros(K_FOLDS)
    fold_l2_b2 = np.zeros(K_FOLDS); fold_val = np.zeros(K_FOLDS)
    fold_eps = np.zeros(K_FOLDS, dtype=int); fold_time = np.zeros(K_FOLDS)

    for k in range(K_FOLDS):
        me = folds == k
        tr = np.where(~me)[0]
        perm = np.random.default_rng(mc_seed * 17 + k).permutation(len(tr))
        n_val = max(int(0.15 * len(tr)), 100)
        va_idx = tr[perm[:n_val]]
        tr_idx = tr[perm[n_val:]]
        t0 = time.time()
        net, val, eps = train_theta(
            x[tr_idx], t[tr_idx], D1[tr_idx], D2[tr_idx], y[tr_idx],
            x[va_idx], t[va_idx], D1[va_idx], D2[va_idx], y[va_idx],
        )
        fold_time[k] = time.time() - t0
        fold_val[k] = val; fold_eps[k] = eps
        with torch.no_grad():
            ae, b1e, b2e = net(torch.from_numpy(X_EVAL_NP))
        fold_l2_a[k] = float(((ae.numpy() - ALPHA_STAR_EVAL) ** 2).mean())
        fold_l2_b1[k] = float(((b1e.numpy() - BETA1_STAR_EVAL) ** 2).mean())
        fold_l2_b2[k] = float(((b2e.numpy() - BETA2_STAR_EVAL) ** 2).mean())
        a_e, b1_e, b2_e = predict(net, x[me])
        alpha_hat[me] = a_e; b1_hat[me] = b1_e; b2_hat[me] = b2_e
        m_e, p_e, c_e, _ = fml_if(a_e, b1_e, b2_e, t[me], D1[me], D2[me], y[me])
        mu_hat[me] = m_e; psi[me] = p_e; corr[me] = c_e

    mu_naive = float(b1_hat.mean())
    mu_fml = float(psi.mean())
    se_fml = float(psi.std(ddof=1) / np.sqrt(N))
    se_naive_pl = float(b1_hat.std(ddof=1) / np.sqrt(N))

    rng_b = np.random.default_rng(mc_seed * 7 + 3)
    e = rng_b.choice([-1.0, 1.0], size=(500, N))
    psi_c = psi - psi.mean()
    mu_b = psi.mean() + (e * psi_c).mean(axis=1)
    se_mult = float(mu_b.std(ddof=1))

    # 2nd-order predicted bias: 0.5 E[v²] + E[u·v]  where v=β1̂-β1*, u=α̂-α*
    u = alpha_hat - a_obs
    v = b1_hat - b1_obs
    pred_bias = float(0.5 * (v ** 2).mean() + (u * v).mean())

    return dict(
        N=N, mc_seed=mc_seed,
        mu_naive=mu_naive, mu_fml=mu_fml,
        se_fml=se_fml, se_naive_pl=se_naive_pl, se_mult=se_mult,
        l2_a=float(fold_l2_a.mean()),
        l2_b1=float(fold_l2_b1.mean()),
        l2_b2=float(fold_l2_b2.mean()),
        epochs_mean=float(fold_eps.mean()),
        val_nll_mean=float(fold_val.mean()),
        fold_time_mean=float(fold_time.mean()),
        pred_bias=pred_bias,
        v2_mean=float((v ** 2).mean()),
        uv_mean=float((u * v).mean()),
        corr_mean=float(corr.mean()),
        corr_sd=float(corr.std(ddof=1)),
    )

# ======================================================================
# main sweep
# ======================================================================
def cp_ci(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def run_sweep():
    N_GRID = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000]
    M_GRID = [120, 100, 80, 60, 50, 40]

    print(f"µ* = E[β1*(X)] = {MU_STAR:+.6f}  "
          f"(target mean ATE ≈ {BETA1_MEAN})")
    print(f"P(T=pre,during,post) = {P_PERIODS}")
    print(f"DNN hidden {DNN_HIDDEN}, max {DNN_MAX_EPOCHS} epochs, "
          f"patience {DNN_PATIENCE}, workers {N_WORKERS}")
    print(f"β1* eval grid: mean={BETA1_STAR_EVAL.mean():+.3f}  "
          f"std={BETA1_STAR_EVAL.std():.3f}  "
          f"range=[{BETA1_STAR_EVAL.min():+.3f}, {BETA1_STAR_EVAL.max():+.3f}]\n")

    all_results = {}
    overall_t0 = time.time()

    for N, M in zip(N_GRID, M_GRID):
        print(f"\n{'='*78}")
        print(f"N={N:,}  M_MC={M}  (elapsed {time.time()-overall_t0:.0f}s)".center(78))
        print(f"{'='*78}")
        payloads = [(N, 4000 + m * 31 + N) for m in range(M)]
        results = []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futs = [pool.submit(run_one_rep, p) for p in payloads]
            for i, f in enumerate(as_completed(futs)):
                results.append(f.result())
                if (i + 1) % max(1, M // 4) == 0:
                    done = len(results)
                    bf = np.mean([r["mu_fml"] for r in results]) - MU_STAR
                    bn = np.mean([r["mu_naive"] for r in results]) - MU_STAR
                    l2b = np.mean([r["l2_b1"] for r in results])
                    print(f"  {done:3d}/{M}  t={time.time()-t0:5.0f}s  "
                          f"bias_naive={bn:+.4f}  bias_fml={bf:+.4f}  "
                          f"L2(β1)={l2b:.5f}")
        all_results[N] = results

        mu_n = np.array([r["mu_naive"] for r in results])
        mu_f = np.array([r["mu_fml"] for r in results])
        se_f = np.array([r["se_fml"] for r in results])
        se_n = np.array([r["se_naive_pl"] for r in results])
        se_m = np.array([r["se_mult"] for r in results])
        pb = np.array([r["pred_bias"] for r in results])
        v2 = np.array([r["v2_mean"] for r in results])
        uv = np.array([r["uv_mean"] for r in results])
        l2a = np.array([r["l2_a"] for r in results])
        l2b1 = np.array([r["l2_b1"] for r in results])
        l2b2 = np.array([r["l2_b2"] for r in results])
        eps = np.array([r["epochs_mean"] for r in results])

        bias_n = mu_n.mean() - MU_STAR
        bias_f = mu_f.mean() - MU_STAR
        sd_n = mu_n.std(ddof=1)
        sd_f = mu_f.std(ddof=1)

        cov_f = np.abs(mu_f - MU_STAR) <= 1.96 * se_f
        cov_n = np.abs(mu_n - MU_STAR) <= 1.96 * se_n

        clo, chi = cp_ci(cov_f.sum(), M)
        nlo, nhi = cp_ci(cov_n.sum(), M)

        print(f"\n  [SUMMARY]  µ* = {MU_STAR:+.5f}")
        print(f"    {'':<35}{'naive':>14}{'FML':>14}")
        print(f"    {'-'*63}")
        print(f"    {'MC mean µ̂':<35}{mu_n.mean():>+14.5f}{mu_f.mean():>+14.5f}")
        print(f"    {'MC bias (µ̂ − µ*)':<35}{bias_n:>+14.5f}{bias_f:>+14.5f}")
        print(f"    {'|bias| / empirical SD':<35}"
              f"{abs(bias_n)/sd_n:>14.4f}{abs(bias_f)/sd_f:>14.4f}")
        print(f"    {'MC empirical SD':<35}{sd_n:>14.5f}{sd_f:>14.5f}")
        print(f"    {'mean reported SE':<35}{se_n.mean():>14.5f}{se_f.mean():>14.5f}")
        print(f"    {'reported SE / empirical SD':<35}"
              f"{se_n.mean()/sd_n:>14.4f}{se_f.mean()/sd_f:>14.4f}")
        print(f"    {'RMSE':<35}"
              f"{np.sqrt(((mu_n-MU_STAR)**2).mean()):>14.5f}"
              f"{np.sqrt(((mu_f-MU_STAR)**2).mean()):>14.5f}")
        print(f"    {'95% coverage':<35}{cov_n.mean():>14.4f}{cov_f.mean():>14.4f}")
        print(f"    {'Clopper-Pearson 95% CI cov':<35}"
              f"[{nlo:.3f},{nhi:.3f}]      [{clo:.3f},{chi:.3f}]")

        print(f"\n  [2nd-order bias prediction]")
        print(f"    observed FML bias              = {bias_f:+.5f}")
        print(f"    mean predicted (0.5 v² + u·v)  = {pb.mean():+.5f}")
        print(f"      component 0.5 E[v²]          = {0.5*v2.mean():+.5f}")
        print(f"      component E[u·v]             = {uv.mean():+.5f}")
        if abs(bias_f) > 1e-6:
            print(f"    ratio predicted / observed     = {pb.mean()/bias_f:.3f}")

        print(f"\n  [DNN health]")
        print(f"    L2(α̂ − α*)    mean={l2a.mean():.5f}")
        print(f"    L2(β1̂ − β1*)  mean={l2b1.mean():.5f}  "
              f"(RMS={np.sqrt(l2b1.mean()):.4f}, "
              f"rel to std(β1*)={np.sqrt(l2b1.mean())/BETA1_STD:.3f})")
        print(f"    L2(β2̂ − β2*)  mean={l2b2.mean():.5f}")
        print(f"    epochs used    mean={eps.mean():.1f}")

        print(f"\n  [multiplier bootstrap]")
        print(f"    mean SE_mult={se_m.mean():.5f}  SE_FML={se_f.mean():.5f}  "
              f"ratio={se_m.mean()/se_f.mean():.4f}")

    # ----------------- cross-N scaling ------------------
    print(f"\n{'='*78}")
    print(f"CROSS-N SCALING (HONEST)".center(78))
    print(f"{'='*78}")
    hdr = (f"  {'N':>8}{'bias_naive':>11}{'bias_FML':>10}{'pred_bias':>11}"
           f"{'emp SD':>10}{'IF SE':>10}{'coverage':>10}{'CP-CI':>18}"
           f"{'L2(β1)':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for N, res in all_results.items():
        mu_n = np.array([r["mu_naive"] for r in res])
        mu_f = np.array([r["mu_fml"] for r in res])
        se_f = np.array([r["se_fml"] for r in res])
        pb = np.array([r["pred_bias"] for r in res])
        l2b1 = np.array([r["l2_b1"] for r in res])
        cov_f = np.abs(mu_f - MU_STAR) <= 1.96 * se_f
        clo, chi = cp_ci(cov_f.sum(), len(res))
        print(f"  {N:>8,}"
              f"{mu_n.mean()-MU_STAR:>+11.5f}"
              f"{mu_f.mean()-MU_STAR:>+10.5f}"
              f"{pb.mean():>+11.5f}"
              f"{mu_f.std(ddof=1):>10.5f}"
              f"{se_f.mean():>10.5f}"
              f"{cov_f.mean():>10.3f}"
              f" [{clo:.2f},{chi:.2f}]  "
              f"{l2b1.mean():>10.5f}")

    # pass/fail verdict
    print(f"\n{'='*78}")
    print(f"VERDICT".center(78))
    print(f"{'='*78}")
    biggest_N = max(all_results.keys())
    res = all_results[biggest_N]
    mu_f = np.array([r["mu_fml"] for r in res])
    se_f = np.array([r["se_fml"] for r in res])
    pb = np.array([r["pred_bias"] for r in res])
    bias_f = mu_f.mean() - MU_STAR
    sd_f = mu_f.std(ddof=1)
    cov = (np.abs(mu_f - MU_STAR) <= 1.96 * se_f).mean()
    clo, chi = cp_ci(int(cov * len(res)), len(res))
    print(f"  At N={biggest_N:,} (M={len(res)} reps):")
    print(f"    |bias|/emp_SD = {abs(bias_f)/sd_f:.3f}   (PASS if <0.1)")
    print(f"    coverage      = {cov:.3f}  CP-CI [{clo:.3f}, {chi:.3f}]  "
          f"(PASS if CI contains 0.95)")
    pred_ratio = pb.mean() / bias_f if abs(bias_f) > 1e-6 else float("inf")
    print(f"    predicted/observed bias = {pred_ratio:.2f}   "
          f"(PASS if between 0.7 and 1.3)")
    pass_bias = abs(bias_f) / sd_f < 0.1
    pass_cov = clo <= 0.95 <= chi
    pass_pred = 0.7 <= pred_ratio <= 1.3 if abs(bias_f) > 1e-6 else True
    print(f"\n  OVERALL: "
          f"{'PASS' if (pass_bias and pass_cov and pass_pred) else 'FAIL / PARTIAL'}")
    print(f"    bias gate: {'PASS' if pass_bias else 'FAIL'}  "
          f"coverage gate: {'PASS' if pass_cov else 'FAIL'}  "
          f"theory gate: {'PASS' if pass_pred else 'FAIL'}")

    out = {"mu_star": MU_STAR,
           "n_grid": [int(n) for n in all_results],
           "results": {str(k): v for k, v in all_results.items()}}
    with open("/tmp/qpoisson_if/flm_scaling_results.json", "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\n[saved] /tmp/qpoisson_if/flm_scaling_results.json")
    print(f"[total time] {time.time()-overall_t0:.0f}s")


if __name__ == "__main__":
    run_sweep()
