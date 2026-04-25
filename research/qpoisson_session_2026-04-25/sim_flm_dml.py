"""
Full FLM deployment: enriched Poisson GLM with DNN-estimated theta(x),
closed-form Λ(x) correction, K-fold cross-fitting (DML), multiplier and
pairs bootstrap, Monte Carlo benchmark.

Structural model (enriched GLM, FLM Appendix B.2):
    Y | X=x, T=t  has  E[Y|x,t] = exp( α*(x) + β*(x) · t )
    Var(Y|x,t) = φ · E[Y|x,t]          (quasi-Poisson; simulated as NB2)
    X = (z_u, z_j) ∈ R^{2d},  T ~ Bern(0.5) independent of X
    α*(x), β*(x) from fixed "true" random neural nets

Target of inference:
    µ* = E[β*(X)]  (average treatment effect, on the log-mean scale)

FLM influence function (B.4 applied with p(T=1)=0.5, Poisson link):
    Λ(x) = E[ μ(x,T) T₁T₁' | X=x ]
         = [[p μ_1 + (1-p) μ_0,  p μ_1],
            [p μ_1,              p μ_1]]
    det(Λ(x)) = p(1-p) μ_0 μ_1
    H(x, θ) = β(x),  H_θ = (0, 1)
    ψ_i = β̂(x_i) − [Λ̂⁻¹]_{β-row} · (1, t_i)' · (μ̂_i − y_i)

Cross-fitting (DML2):
    K folds. For held-out fold k, train θ̂ on the other K-1 folds, compute
    Λ̂ from that θ̂, evaluate ψ on fold k. Stack all ψ and average.

Diagnostics:
  A. DGP pre-checks: α*, β*, μ*, Var(y)/E[y], E[β*(X)].
  B. DNN training: per-epoch train/val loss, L2 error to truth on a fresh
     grid, prediction-distribution health.
  C. Per-fold: convergence, epochs used, held-out NLL.
  D. IF internals: ψ_i distribution, E[ψ] across MC ≈ µ*, Var(ψ) vs empirical SD.
  E. Correction term: mean ≈ 0 under truth, variance component vs naive.
  F. Two estimators: µ̂_naive (plug-in), µ̂_DML (orthogonal).
  G. Three SEs: analytic IF SE, multiplier bootstrap, pairs bootstrap (costly).
  H. Monte Carlo benchmark: repeat the whole pipeline M times and check bias,
     empirical SD, coverage, IF-vs-bootstrap agreement.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from scipy import stats

# ======================================================================
# 0. Config
# ======================================================================
D_EMB = 8
N_FOCAL = 30_000
N_MC = 10_000
PHI = 2.0
K_FOLDS = 5
DNN_HIDDEN = (64, 32)
DNN_LR = 2e-3
DNN_MAX_EPOCHS = 200
DNN_BATCH = 1024
DNN_PATIENCE = 15
SEED = 17
M_MC = 40
B_MULT = 2000
B_PAIR = 30

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float32)

# ======================================================================
# 1. True networks (fixed) for α*(x) and β*(x)
# ======================================================================
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
_beta_raw = TrueNet(2 * D_EMB)
for p in _alpha_raw.parameters(): p.requires_grad_(False)
for p in _beta_raw.parameters(): p.requires_grad_(False)

# calibrate so α has mean BASE, std ALPHA_STD; β has mean ATE_TRUTH, std BETA_STD
BASE = 1.3
ALPHA_STD = 0.5
ATE_TRUTH_TARGET = 0.35
BETA_STD = 0.55
_rng = torch.Generator().manual_seed(99)
with torch.no_grad():
    x_cal = torch.randn(200_000, 2 * D_EMB, generator=_rng)
    a_raw = _alpha_raw(x_cal)
    b_raw = _beta_raw(x_cal)
    a_mean, a_std = a_raw.mean().item(), a_raw.std().item()
    b_mean, b_std = b_raw.mean().item(), b_raw.std().item()

def alpha_true_torch(x):
    return (_alpha_raw(x) - a_mean) / a_std * ALPHA_STD + BASE

def beta_true_torch(x):
    return (_beta_raw(x) - b_mean) / b_std * BETA_STD + ATE_TRUTH_TARGET

# ground-truth µ* computed by high-precision MC over the known X distribution
with torch.no_grad():
    x_big = torch.randn(1_000_000, 2 * D_EMB, generator=torch.Generator().manual_seed(101))
    MU_STAR = beta_true_torch(x_big).mean().item()
print(f"[truth] µ* = E[β*(X)] = {MU_STAR:+.6f}  (target {ATE_TRUTH_TARGET})")

# fresh evaluation grid for L2 error tracking
with torch.no_grad():
    x_eval = torch.randn(5000, 2 * D_EMB, generator=torch.Generator().manual_seed(202))
    alpha_star_eval = alpha_true_torch(x_eval).numpy()
    beta_star_eval = beta_true_torch(x_eval).numpy()
print(f"[truth] α* on eval grid: mean={alpha_star_eval.mean():+.3f}  "
      f"std={alpha_star_eval.std():.3f}")
print(f"[truth] β* on eval grid: mean={beta_star_eval.mean():+.3f}  "
      f"std={beta_star_eval.std():.3f}  "
      f"range=[{beta_star_eval.min():+.3f}, {beta_star_eval.max():+.3f}]")

# ======================================================================
# 2. Data simulation
# ======================================================================
def simulate(rng, n):
    x = rng.standard_normal((n, 2 * D_EMB)).astype(np.float32)
    t = rng.integers(2, size=n).astype(np.float32)
    with torch.no_grad():
        a = alpha_true_torch(torch.from_numpy(x)).numpy()
        b = beta_true_torch(torch.from_numpy(x)).numpy()
    mu = np.exp(a + b * t)
    theta = mu / (PHI - 1.0)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p).astype(np.float32)
    return x, t, y, a, b, mu

# ======================================================================
# 3. First-stage DNN: θ(x) = (α̂(x), β̂(x)) under Poisson NLL
# ======================================================================
class ThetaNet(nn.Module):
    def __init__(self, d_in, hidden=(64, 32)):
        super().__init__()
        L = []
        prev = d_in
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
    # up to const: mean of μ − y·log(μ)
    return (mu - y * torch.log(mu + 1e-10)).mean()


def train_theta(x_tr, t_tr, y_tr, x_va, t_va, y_va, track_truth=False, verbose=False):
    """Train with early stopping. If track_truth=True, record L2 to truth each epoch."""
    net = ThetaNet(x_tr.shape[1], DNN_HIDDEN)
    opt = optim.Adam(net.parameters(), lr=DNN_LR, weight_decay=1e-5)
    xtr = torch.from_numpy(x_tr)
    ttr = torch.from_numpy(t_tr)
    ytr = torch.from_numpy(y_tr)
    xva = torch.from_numpy(x_va)
    tva = torch.from_numpy(t_va)
    yva = torch.from_numpy(y_va)
    if track_truth:
        x_ev = x_eval
        a_true = alpha_star_eval
        b_true = beta_star_eval
    n = len(xtr)
    best_val = float("inf")
    best_state = None
    patience = 0
    history = []
    for ep in range(DNN_MAX_EPOCHS):
        net.train()
        idx = np.random.permutation(n)
        train_losses = []
        for s in range(0, n, DNN_BATCH):
            sel = idx[s:s + DNN_BATCH]
            a, b = net(xtr[sel])
            mu = torch.exp(a + b * ttr[sel])
            loss = poisson_nll(mu, ytr[sel])
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            av, bv = net(xva)
            muv = torch.exp(av + bv * tva)
            val = poisson_nll(muv, yva).item()
            if track_truth:
                ae, be = net(x_ev)
                l2_a = float(((ae.numpy() - a_true) ** 2).mean())
                l2_b = float(((be.numpy() - b_true) ** 2).mean())
                mean_b = float(be.mean().item())
                sd_b = float(be.std().item())
        rec = dict(ep=ep, train=float(np.mean(train_losses)), val=val)
        if track_truth:
            rec.update(l2_a=l2_a, l2_b=l2_b, mean_b=mean_b, sd_b=sd_b)
        history.append(rec)
        if verbose and (ep % 10 == 0 or ep < 5):
            msg = f"    ep {ep:3d}  train {rec['train']:.4f}  val {val:.4f}"
            if track_truth:
                msg += f"  l2_a {l2_a:.4f}  l2_b {l2_b:.4f}  β̂ mean={mean_b:+.3f} sd={sd_b:.3f}"
            print(msg)
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= DNN_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history


def predict(net, x):
    net.eval()
    with torch.no_grad():
        a, b = net(torch.from_numpy(x))
    return a.numpy(), b.numpy()

# ======================================================================
# 4. Λ(x) closed form and influence function for H=β
# ======================================================================
def if_components(alpha_hat, beta_hat, t, y, p_t=0.5):
    """Return:
        mu_hat_i       μ̂ at the realized (x_i, t_i)
        psi_i          β̂(x_i) − correction_i   (DML IF, centered on µ̂_DML)
        correction_i   [Λ̂⁻¹]_{β-row} · T₁_i · (μ̂_i − y_i)
        resid_i        μ̂_i − y_i
    Closed-form:
        μ_0 = exp(α̂), μ_1 = exp(α̂ + β̂),
        det = p(1-p) μ_0 μ_1,
        [Λ⁻¹]_β-row: (inv_10, inv_11) = (-p μ_1, p μ_1 + (1-p) μ_0) / det
        correction = (inv_10 + t·inv_11) · (μ̂ − y).
    """
    mu_0 = np.exp(alpha_hat)
    mu_1 = np.exp(alpha_hat + beta_hat)
    mu_hat = np.where(t > 0.5, mu_1, mu_0)
    det = p_t * (1 - p_t) * mu_0 * mu_1
    inv_10 = -(p_t * mu_1) / det            # = −1/((1-p) μ_0)
    inv_11 = (p_t * mu_1 + (1 - p_t) * mu_0) / det
    resid = mu_hat - y
    correction = (inv_10 + t * inv_11) * resid
    psi = beta_hat - correction
    return mu_hat, psi, correction, resid

# ======================================================================
# 5. DML2 driver (K-fold cross-fitting)
# ======================================================================
def dml_fit(x, t, y, K=K_FOLDS, verbose=False, track_truth=False, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    folds = rng.integers(K, size=n)
    out = dict(
        psi=np.zeros(n), alpha_hat=np.zeros(n), beta_hat=np.zeros(n),
        mu_hat=np.zeros(n), correction=np.zeros(n), resid=np.zeros(n),
        fold_val=np.zeros(K), fold_epochs=np.zeros(K, dtype=int),
        fold_time=np.zeros(K), fold_l2_a=np.zeros(K), fold_l2_b=np.zeros(K),
    )
    for k in range(K):
        mask_eval = folds == k
        tr_all = np.where(~mask_eval)[0]
        perm = np.random.default_rng(100 + k).permutation(len(tr_all))
        n_val = max(int(0.15 * len(tr_all)), 100)
        va_idx = tr_all[perm[:n_val]]
        tr_idx = tr_all[perm[n_val:]]
        t0 = time.time()
        net, hist = train_theta(
            x[tr_idx], t[tr_idx], y[tr_idx],
            x[va_idx], t[va_idx], y[va_idx],
            track_truth=track_truth, verbose=verbose and (k == 0),
        )
        out["fold_time"][k] = time.time() - t0
        out["fold_epochs"][k] = hist[-1]["ep"] + 1
        out["fold_val"][k] = hist[-1]["val"]
        # final truth-gap diagnostics (regardless of track_truth, we have x_eval)
        with torch.no_grad():
            ae, be = net(x_eval)
        out["fold_l2_a"][k] = float(((ae.numpy() - alpha_star_eval) ** 2).mean())
        out["fold_l2_b"][k] = float(((be.numpy() - beta_star_eval) ** 2).mean())
        a_e, b_e = predict(net, x[mask_eval])
        out["alpha_hat"][mask_eval] = a_e
        out["beta_hat"][mask_eval] = b_e
        mu_i, psi_i, corr_i, res_i = if_components(a_e, b_e, t[mask_eval], y[mask_eval])
        out["mu_hat"][mask_eval] = mu_i
        out["psi"][mask_eval] = psi_i
        out["correction"][mask_eval] = corr_i
        out["resid"][mask_eval] = res_i
        if verbose:
            print(f"    fold {k}: {out['fold_time'][k]:5.1f}s  "
                  f"{out['fold_epochs'][k]:3d} ep  val={out['fold_val'][k]:.4f}  "
                  f"L2(α)={out['fold_l2_a'][k]:.4f}  L2(β)={out['fold_l2_b'][k]:.4f}")
    out["mu_hat_naive"] = out["beta_hat"].mean()
    out["mu_hat_dml"] = out["psi"].mean()
    return out

# ======================================================================
# 6. Standard errors
# ======================================================================
def if_se(psi):
    n = len(psi)
    return float(np.std(psi, ddof=1) / np.sqrt(n))

def multiplier_bootstrap(psi, B=B_MULT, rng=None):
    """Multiplier (wild) bootstrap on the IF: valid for DML estimators."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(psi)
    psi_c = psi - psi.mean()
    # Rademacher multipliers
    e = rng.choice([-1.0, 1.0], size=(B, n))
    mu_b = psi.mean() + (e * psi_c).mean(axis=1)
    return mu_b, float(mu_b.std(ddof=1))

# ======================================================================
# 7. FOCAL replicate: deep diagnostics
# ======================================================================
print("\n" + "=" * 78)
print("A. DGP PRE-CHECKS".center(78))
print("=" * 78)
rng_focal = np.random.default_rng(SEED)
x, t, y, a_obs, b_obs, mu_obs = simulate(rng_focal, N_FOCAL)
print(f"\n[DGP] N={N_FOCAL}, PHI={PHI}")
print(f"[DGP] α*(X) at obs:   mean={a_obs.mean():+.3f}  std={a_obs.std():.3f}")
print(f"[DGP] β*(X) at obs:   mean={b_obs.mean():+.3f}  std={b_obs.std():.3f}  "
      f"range=[{b_obs.min():+.3f}, {b_obs.max():+.3f}]")
print(f"[DGP] μ(X,T) at obs:  mean={mu_obs.mean():.3f}  p5={np.percentile(mu_obs,5):.3f}  "
      f"p95={np.percentile(mu_obs,95):.3f}")
print(f"[DGP] y:              mean={y.mean():.3f}  var={y.var():.3f}  "
      f"Var/E={y.var()/y.mean():.3f}  target φ={PHI}")
print(f"[DGP] t:              fraction T=1 = {t.mean():.4f}  (target 0.5)")
print(f"[truth] µ* = E[β*(X)] (analytic MC): {MU_STAR:+.6f}")
# in-sample mean of β*(X_i) (what an oracle estimator would return at this replicate)
print(f"[truth] (1/n)Σβ*(X_i) in focal:      {b_obs.mean():+.6f}")

# score at truth sanity
print(f"\n[score@truth] mean of (y − μ(X,T,α*,β*)): "
      f"{(y - mu_obs).mean():+.4f} (target ~0)")
# Var check: Var(y|x,t) should be phi*mu(x,t)
local_var = ((y - mu_obs) ** 2).mean() / mu_obs.mean()
print(f"[score@truth] avg((y−μ)²)/avg(μ) = {local_var:.3f}  (target φ={PHI})")

# ---- B. DNN training with per-epoch diagnostics on fold 0 only ----
print("\n" + "=" * 78)
print("B. DNN TRAINING ON HELD-OUT FOLD (full per-epoch track for fold 0)".center(78))
print("=" * 78)
rng_pilot = np.random.default_rng(SEED + 1)
folds = rng_pilot.integers(K_FOLDS, size=N_FOCAL)
mk = folds == 0
tr_all = np.where(~mk)[0]
pp = np.random.default_rng(99).permutation(len(tr_all))
n_val = int(0.15 * len(tr_all))
va_idx = tr_all[pp[:n_val]]
tr_idx = tr_all[pp[n_val:]]
print(f"\n[pilot fit] train/val split: {len(tr_idx)} train, {len(va_idx)} val, "
      f"{mk.sum()} held-out")
t0 = time.time()
net_pilot, hist_pilot = train_theta(
    x[tr_idx], t[tr_idx], y[tr_idx], x[va_idx], t[va_idx], y[va_idx],
    track_truth=True, verbose=False,
)
print(f"[pilot fit] {time.time()-t0:.1f}s, {hist_pilot[-1]['ep']+1} epochs")
print(f"\nepoch trace (every 10th + last 3):")
print(f"  {'ep':>3}  {'train':>9}  {'val':>9}  {'L2(α)':>9}  {'L2(β)':>9}  "
      f"{'mean β̂':>9}  {'sd β̂':>9}")
sel_eps = list(range(0, len(hist_pilot), 10)) + list(range(max(0, len(hist_pilot)-3),
                                                            len(hist_pilot)))
for i in sorted(set(sel_eps)):
    h = hist_pilot[i]
    print(f"  {h['ep']:>3}  {h['train']:>9.4f}  {h['val']:>9.4f}  "
          f"{h['l2_a']:>9.5f}  {h['l2_b']:>9.5f}  "
          f"{h['mean_b']:>+9.4f}  {h['sd_b']:>9.4f}")

# ---- C-G. Full DML2 with all 5 folds ----
print("\n" + "=" * 78)
print("C. DML2 CROSS-FITTING (K={} folds)".format(K_FOLDS).center(78))
print("=" * 78)
t0 = time.time()
res_focal = dml_fit(x, t, y, K=K_FOLDS, verbose=True, seed=SEED + 10)
print(f"\n[DML2] total wall time: {time.time()-t0:.1f}s")
print(f"[DML2] val NLL by fold:     {res_focal['fold_val']}")
print(f"[DML2] epochs used by fold: {res_focal['fold_epochs']}")
print(f"[DML2] L2(α̂-α*) by fold:   {res_focal['fold_l2_a'].round(5)}  "
      f"mean={res_focal['fold_l2_a'].mean():.5f}")
print(f"[DML2] L2(β̂-β*) by fold:   {res_focal['fold_l2_b'].round(5)}  "
      f"mean={res_focal['fold_l2_b'].mean():.5f}")

psi = res_focal["psi"]
corr = res_focal["correction"]
bhat = res_focal["beta_hat"]
mu_hat = res_focal["mu_hat"]
resid = res_focal["resid"]

print("\n" + "=" * 78)
print("D. IF INTERNALS".center(78))
print("=" * 78)
print(f"\n[β̂(X)]         mean={bhat.mean():+.4f}  std={bhat.std():.4f}  "
      f"range=[{bhat.min():+.3f}, {bhat.max():+.3f}]")
print(f"[correction]   mean={corr.mean():+.4f}  std={corr.std():.4f}  "
      f"(target mean ~0 under truth)")
print(f"[residual μ̂-y] mean={resid.mean():+.4f}  std={resid.std():.4f}  "
      f"(target mean ~0)")
# correlation between correction term and β̂ (variance decomposition)
rho_corr_beta = np.corrcoef(corr, bhat)[0, 1]
print(f"[corr(correction, β̂)]  {rho_corr_beta:+.4f}  "
      f"(near 0 => correction is orthogonal noise)")
print(f"[ψ]            mean={psi.mean():+.4f}  std={psi.std():.4f}  "
      f"range=[{psi.min():+.3f}, {psi.max():+.3f}]")

print("\n" + "=" * 78)
print("E. ESTIMATORS AND SEs".center(78))
print("=" * 78)
mu_naive = res_focal["mu_hat_naive"]
mu_dml = res_focal["mu_hat_dml"]
se_dml = if_se(psi)
# Naive SE (treating β̂(x_i) as iid) — NOT valid inference, just for comparison
se_naive = float(np.std(bhat, ddof=1) / np.sqrt(N_FOCAL))
# multiplier bootstrap
mb_draws, se_mult = multiplier_bootstrap(psi, B=B_MULT, rng=np.random.default_rng(777))
# pairs bootstrap (B_PAIR refits). Each refit is a fresh DML2 on resampled indices.
t0 = time.time()
pairs_mu = np.zeros(B_PAIR)
for b in range(B_PAIR):
    rng_b = np.random.default_rng(2000 + b)
    idx = rng_b.integers(N_FOCAL, size=N_FOCAL)
    res_b = dml_fit(x[idx], t[idx], y[idx], K=K_FOLDS, verbose=False,
                    seed=SEED + 10 + b)
    pairs_mu[b] = res_b["mu_hat_dml"]
    if (b + 1) % 10 == 0:
        print(f"  pairs bootstrap {b+1}/{B_PAIR}  "
              f"t={time.time()-t0:.0f}s  mu_b_sd so far={pairs_mu[:b+1].std(ddof=1):.4f}")
se_pair = float(pairs_mu.std(ddof=1))
print(f"[pairs bootstrap] {B_PAIR} refits in {time.time()-t0:.1f}s")

print(f"\n  {'estimator':<14}{'mu_hat':>12}{'bias':>10}{'IF SE':>10}"
      f"{'mult SE':>10}{'pair SE':>10}")
print(f"  {'naive':<14}{mu_naive:>+12.5f}{mu_naive-MU_STAR:>+10.5f}"
      f"{se_naive:>10.5f}{'—':>10}{'—':>10}")
print(f"  {'DML2':<14}{mu_dml:>+12.5f}{mu_dml-MU_STAR:>+10.5f}"
      f"{se_dml:>10.5f}{se_mult:>10.5f}{se_pair:>10.5f}")
print(f"  (µ* = {MU_STAR:+.5f})")
print(f"\n  95% IF CI for DML: [{mu_dml - 1.96*se_dml:+.4f}, "
      f"{mu_dml + 1.96*se_dml:+.4f}]  covers µ*: "
      f"{abs(mu_dml - MU_STAR) <= 1.96 * se_dml}")

# ======================================================================
# 8. Monte Carlo benchmark
# ======================================================================
print("\n" + "=" * 78)
print("H. MONTE CARLO BENCHMARK  (M={} reps, N={})".format(M_MC, N_MC).center(78))
print("=" * 78)
print("  collecting: naive vs DML point estimates, IF SE, bootstrap SE, coverage.\n")
mc_naive = np.zeros(M_MC)
mc_dml = np.zeros(M_MC)
mc_se_dml = np.zeros(M_MC)
mc_se_mult = np.zeros(M_MC)
mc_se_naive_pl = np.zeros(M_MC)
mc_l2_a = np.zeros(M_MC)
mc_l2_b = np.zeros(M_MC)
mc_epochs = np.zeros(M_MC)
cover_dml = np.zeros(M_MC, dtype=bool)
cover_naive = np.zeros(M_MC, dtype=bool)

t_mc = time.time()
for m in range(M_MC):
    rng_m = np.random.default_rng(3000 + m)
    xm, tm, ym, *_ = simulate(rng_m, N_MC)
    res_m = dml_fit(xm, tm, ym, K=K_FOLDS, verbose=False, seed=SEED + m)
    mc_naive[m] = res_m["mu_hat_naive"]
    mc_dml[m] = res_m["mu_hat_dml"]
    mc_se_dml[m] = if_se(res_m["psi"])
    mc_se_naive_pl[m] = float(np.std(res_m["beta_hat"], ddof=1) / np.sqrt(N_MC))
    _, mc_se_mult[m] = multiplier_bootstrap(res_m["psi"], B=500,
                                             rng=np.random.default_rng(5000 + m))
    mc_l2_a[m] = res_m["fold_l2_a"].mean()
    mc_l2_b[m] = res_m["fold_l2_b"].mean()
    mc_epochs[m] = res_m["fold_epochs"].mean()
    cover_dml[m] = abs(mc_dml[m] - MU_STAR) <= 1.96 * mc_se_dml[m]
    cover_naive[m] = abs(mc_naive[m] - MU_STAR) <= 1.96 * mc_se_naive_pl[m]
    if (m + 1) % 5 == 0:
        dt = time.time() - t_mc
        print(f"  MC {m+1:>3}/{M_MC}  elapsed {dt:5.0f}s  "
              f"bias_naive={mc_naive[:m+1].mean()-MU_STAR:+.4f}  "
              f"bias_dml={mc_dml[:m+1].mean()-MU_STAR:+.4f}  "
              f"cover_dml={cover_dml[:m+1].mean():.3f}")

print(f"\n[MC complete]  {time.time()-t_mc:.1f}s  "
      f"({(time.time()-t_mc)/M_MC:.1f}s per rep)")

# ---- Final report ----
print("\n" + "=" * 78)
print("FINAL MC SUMMARY".center(78))
print("=" * 78)
def pct(p): return np.percentile(mc_dml - MU_STAR, p)
print(f"\n  µ* (truth, analytic) = {MU_STAR:+.6f}\n")
print(f"  {'metric':<40}{'naive':>14}{'DML':>14}")
print(f"  {'-'*68}")
print(f"  {'MC mean of µ̂':<40}{mc_naive.mean():>+14.5f}{mc_dml.mean():>+14.5f}")
print(f"  {'MC bias (µ̂ − µ*)':<40}"
      f"{mc_naive.mean()-MU_STAR:>+14.5f}{mc_dml.mean()-MU_STAR:>+14.5f}")
print(f"  {'MC empirical SD of µ̂':<40}"
      f"{mc_naive.std(ddof=1):>14.5f}{mc_dml.std(ddof=1):>14.5f}")
print(f"  {'mean reported SE':<40}"
      f"{mc_se_naive_pl.mean():>14.5f}{mc_se_dml.mean():>14.5f}")
print(f"  {'ratio  reported SE / empirical SD':<40}"
      f"{mc_se_naive_pl.mean()/mc_naive.std(ddof=1):>14.4f}"
      f"{mc_se_dml.mean()/mc_dml.std(ddof=1):>14.4f}")
print(f"  {'95% CI coverage of µ*':<40}"
      f"{cover_naive.mean():>14.4f}{cover_dml.mean():>14.4f}")
print(f"  {'rmse of µ̂':<40}"
      f"{np.sqrt(((mc_naive-MU_STAR)**2).mean()):>14.5f}"
      f"{np.sqrt(((mc_dml-MU_STAR)**2).mean()):>14.5f}")

print(f"\n  mean multiplier bootstrap SE: {mc_se_mult.mean():.5f}  "
      f"corr w/ IF SE: {np.corrcoef(mc_se_dml, mc_se_mult)[0,1]:.4f}")

print(f"\n  DNN convergence across MC reps:")
print(f"    L2(α̂ − α*) fold-avg: mean={mc_l2_a.mean():.5f}  "
      f"p5-p95=[{np.percentile(mc_l2_a,5):.5f}, {np.percentile(mc_l2_a,95):.5f}]")
print(f"    L2(β̂ − β*) fold-avg: mean={mc_l2_b.mean():.5f}  "
      f"p5-p95=[{np.percentile(mc_l2_b,5):.5f}, {np.percentile(mc_l2_b,95):.5f}]")
print(f"    epochs per fold (avg across folds): mean={mc_epochs.mean():.1f}")

# Coverage Clopper-Pearson CI
from scipy.stats import beta as beta_dist
k = cover_dml.sum()
lo = beta_dist.ppf(0.025, k, M_MC - k + 1) if k > 0 else 0.0
hi = beta_dist.ppf(0.975, k + 1, M_MC - k) if k < M_MC else 1.0
print(f"\n  DML coverage {k}/{M_MC} = {cover_dml.mean():.3f}  "
      f"95% Clopper-Pearson CI [{lo:.3f}, {hi:.3f}]  target 0.950")
k = cover_naive.sum()
lo = beta_dist.ppf(0.025, k, M_MC - k + 1) if k > 0 else 0.0
hi = beta_dist.ppf(0.975, k + 1, M_MC - k) if k < M_MC else 1.0
print(f"  naive coverage {k}/{M_MC} = {cover_naive.mean():.3f}  "
      f"95% Clopper-Pearson CI [{lo:.3f}, {hi:.3f}]")

print("\n" + "=" * 78)
print("DONE.".center(78))
print("=" * 78)
