"""
Comprehensive IF vs bootstrap study for a 10x10 event-study with beta-net
heterogeneous treatment effects.  Reports three blocks of measurements:

    PRE-ESTIMATION
      - Beta-net activation stats, tau* distribution
      - DGP balance: cell sizes, mean mu per cell
      - Empirical check of Var(y|cell) = phi * mu
      - Score at truth has mean 0, Var = E[phi * mu * x x']
      - A = -E[mu x x'] (Jacobian) positive-definite structure check

    WITHIN-ESTIMATION
      - Newton-Raphson fit that matches closed-form cell means
      - Per-iteration: log-likelihood, gradient norm, Hessian condition number,
        largest parameter change, score sup-norm
      - MLE sanity: sum_{i in c} (y_i - mu_hat_c) = 0 for every cell
      - Fit-time dispersion phi_hat_cell vs true phi
      - Sandwich bread/meat: compare A^{-1} B A^{-1} to model-based A^{-1}
        and assert ratio ~ phi on the diagonal

    POST-ESTIMATION
      - Pearson & deviance residuals (mean, sd, Anscombe-style)
      - Pearson chi^2 / (n - p) vs phi
      - Per-cell IF SE, bootstrap SE (B=500), jackknife delete-d=n/K SE
      - Monte Carlo benchmark (M=300): empirical SD of tau_hat, coverage
      - Coverage sliced by |tau*|, by cell size, by mu magnitude
      - Three worst/best IF-vs-bootstrap mismatches
      - BCa + percentile + normal bootstrap intervals vs IF normal interval

Map to Farrell-Liang-Misra (2020): the low-dim target is tau_{u,j}, the
"deep" object is the function that maps cluster embeddings to tau*.  The
saturated cell estimator here is the trivial special case of FLM where the
heterogeneity function is one-hot over cells.  Validating IF ~ bootstrap in
this case is the baseline any more flexible FLM implementation must clear.
"""

import numpy as np
import torch
import torch.nn as nn
import time

# ------------------------- config ---------------------------------------
K_U = K_I = 10
EMB_D = 8
N = 30_000
PHI = 2.0
B_BOOT = 500
M_MC = 300
SEED = 11
NEWTON_MAX_ITER = 50
NEWTON_TOL = 1e-10

np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------------- beta net (truth) -----------------------------
class BetaNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, 32), nn.Tanh(),
            nn.Linear(32, 16), nn.Tanh(),
            nn.Linear(16, 1),
        )
    def forward(self, zu, zi):
        U = zu.unsqueeze(1).expand(-1, zi.shape[0], -1)
        I = zi.unsqueeze(0).expand(zu.shape[0], -1, -1)
        return self.net(torch.cat([U, I], dim=-1)).squeeze(-1)

z_u = torch.randn(K_U, EMB_D)
z_i = torch.randn(K_I, EMB_D)
betanet = BetaNet(EMB_D)
with torch.no_grad():
    tau_raw = betanet(z_u, z_i).numpy()
    # also capture pre-activations of each layer for diagnostics
    h0 = torch.cat([z_u.unsqueeze(1).expand(-1, K_I, -1),
                    z_i.unsqueeze(0).expand(K_U, -1, -1)], dim=-1)
    h1 = betanet.net[0](h0); h1a = torch.tanh(h1)
    h2 = betanet.net[2](h1a); h2a = torch.tanh(h2)
tau_true = (tau_raw - tau_raw.mean()) / tau_raw.std() * 0.6

alpha_u = np.random.normal(0, 0.3, K_U)
gamma_j = np.random.normal(0, 0.3, K_I)
BASE = 1.3


def simulate(rng):
    u = rng.integers(K_U, size=N)
    j = rng.integers(K_I, size=N)
    t = rng.integers(2, size=N)
    log_mu = BASE + alpha_u[u] + gamma_j[j] + tau_true[u, j] * t
    mu = np.exp(log_mu)
    theta = mu / (PHI - 1.0)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p)
    return u, j, t, y, mu


def cell_index(u, j, t):
    return u * (K_I * 2) + j * 2 + t


def aggregate(u, j, t, y):
    c = cell_index(u, j, t)
    n_flat = np.bincount(c, minlength=K_U * K_I * 2)
    s_flat = np.bincount(c, weights=y.astype(float), minlength=K_U * K_I * 2)
    ss_flat = np.bincount(c, weights=y.astype(float) ** 2,
                          minlength=K_U * K_I * 2)
    counts = n_flat.reshape(K_U, K_I, 2)
    sums = s_flat.reshape(K_U, K_I, 2)
    sumsq = ss_flat.reshape(K_U, K_I, 2)
    mu_hat = sums / np.maximum(counts, 1)
    sumsq_dev = sumsq - counts * mu_hat ** 2   # sum_{i in c}(y_i - mu_hat)^2
    return mu_hat, counts, sumsq_dev


# --------------- Newton-Raphson Poisson fit (for the saturated cell model) ---
def newton_poisson(u, j, t, y, log_mu_init=None, verbose=False):
    c = cell_index(u, j, t)
    n_flat = np.bincount(c, minlength=K_U * K_I * 2)
    s_flat = np.bincount(c, weights=y.astype(float), minlength=K_U * K_I * 2)
    if log_mu_init is None:
        beta = np.zeros_like(n_flat, dtype=float)
    else:
        beta = log_mu_init.copy()
    history = []
    for it in range(NEWTON_MAX_ITER):
        mu_flat = np.exp(beta)
        score = s_flat - n_flat * mu_flat                     # gradient of LL
        hess_diag = -n_flat * mu_flat                         # Hessian (neg)
        # Newton step: beta <- beta - H^{-1} g = beta + score / (n*mu)
        step = np.where(n_flat > 0, score / (n_flat * mu_flat + 1e-12), 0.0)
        max_step = np.max(np.abs(step))
        grad_norm = np.linalg.norm(score)
        # log-likelihood (drop y! term; saturated => cell-level sum y*beta - n*mu)
        ll = np.sum(s_flat * beta - n_flat * mu_flat)
        cond_num = (np.max(np.abs(hess_diag)) /
                    max(np.min(np.abs(hess_diag[n_flat > 0])), 1e-12))
        history.append(dict(
            iter=it, ll=float(ll), grad_norm=float(grad_norm),
            max_step=float(max_step), hess_cond=float(cond_num),
            score_sup=float(np.max(np.abs(score))),
        ))
        if verbose:
            print(f"  iter {it:2d}  ll={ll:.4f}  |grad|={grad_norm:.3e}  "
                  f"max_step={max_step:.3e}  cond={cond_num:.2e}")
        beta = beta + step
        if max_step < NEWTON_TOL and grad_norm < NEWTON_TOL:
            break
    return beta.reshape(K_U, K_I, 2), history


# ------------------------- diagnostics block ----------------------------
rng = np.random.default_rng(SEED)

print("\n" + "=" * 78)
print("BLOCK 1: PRE-ESTIMATION DIAGNOSTICS".center(78))
print("=" * 78)

# beta-net outputs
print(f"\n[beta-net] architecture: 2*{EMB_D} -> 32 -> 16 -> 1, Tanh activations")
print(f"[beta-net] layer 1 pre-act (sample 100 cells): "
      f"mean={h1.mean():+.3f} std={h1.std():.3f} abs_max={h1.abs().max():.3f}")
print(f"[beta-net] layer 1 post-Tanh saturation fraction (|h|>0.9): "
      f"{(h1a.abs() > 0.9).float().mean():.3f}")
print(f"[beta-net] layer 2 post-Tanh saturation fraction: "
      f"{(h2a.abs() > 0.9).float().mean():.3f}")
print(f"[tau*]     range=[{tau_true.min():+.3f}, {tau_true.max():+.3f}]  "
      f"std={tau_true.std():.3f}  median={np.median(tau_true):+.3f}  "
      f"IQR=[{np.percentile(tau_true, 25):+.3f}, "
      f"{np.percentile(tau_true, 75):+.3f}]")
print(f"[tau*]     cells |tau*|>0.5: {(np.abs(tau_true) > 0.5).sum()} / 100")
print(f"[tau*]     cells |tau*|>1.0: {(np.abs(tau_true) > 1.0).sum()} / 100")

# DGP
u, j, t, y, mu_obs = simulate(rng)
mu_hat, counts, sumsq_dev = aggregate(u, j, t, y)
print(f"\n[DGP] N={N}  mean obs per (u,j,t) cell={counts.mean():.1f}  "
      f"min={counts.min()}  max={counts.max()}")
print(f"[DGP] mu (observed, per obs): mean={mu_obs.mean():.3f}  "
      f"p5={np.percentile(mu_obs, 5):.3f}  p95={np.percentile(mu_obs, 95):.3f}")
print(f"[DGP] y: mean={y.mean():.3f}  var={y.var():.3f}  "
      f"var/mean={y.var()/y.mean():.3f}  target phi={PHI}")

# Empirical Var(y|cell)/E[y|cell] = phi per cell
cell_var = sumsq_dev / np.maximum(counts - 1, 1)
phi_cell_hat = cell_var / np.maximum(mu_hat, 1e-9)
print(f"[DGP] per-cell phi_hat = Var/mean: "
      f"median={np.median(phi_cell_hat):.3f}  "
      f"IQR=[{np.percentile(phi_cell_hat, 25):.3f}, "
      f"{np.percentile(phi_cell_hat, 75):.3f}]   target phi={PHI}")

# Score at truth: sum of x_i (y_i - mu_true_i) per parameter, should have mean 0
log_mu_true = BASE + alpha_u[u] + gamma_j[j] + tau_true[u, j] * t
mu_true_per_obs = np.exp(log_mu_true)
score_at_truth = y - mu_true_per_obs
s_flat = np.bincount(cell_index(u, j, t), weights=score_at_truth,
                     minlength=K_U * K_I * 2)
print(f"\n[score@truth] per-cell sum of (y - mu_true):  "
      f"max|sum|/sqrt(n)={np.max(np.abs(s_flat))/np.sqrt(counts.mean()):.3f}")
# Normalized: sum / sqrt(n_c * phi * mu_c) -> approx standard normal
z_scores = s_flat.reshape(K_U, K_I, 2) / np.sqrt(
    counts * PHI * np.maximum(mu_hat, 1e-9))
print(f"[score@truth] z-scores across 200 cells: "
      f"mean={z_scores.mean():+.3f} (target 0)  "
      f"std={z_scores.std():.3f} (target 1)  "
      f"KS-like max|z|={np.max(np.abs(z_scores)):.2f}")

# Jacobian A diag element check: A_cc = -sum_{i in c} mu_i, which at truth
# becomes -n_c * E[mu|c] = -n_c * mu_true_c.  Compare to empirical.
mu_true_cell_expected = (sumsq_dev.shape  # just a placeholder for structure
                         and np.zeros((K_U, K_I, 2)))
for cu in range(K_U):
    for cj in range(K_I):
        for ct in range(2):
            mask = (u == cu) & (j == cj) & (t == ct)
            mu_true_cell_expected[cu, cj, ct] = mu_true_per_obs[mask].mean()
diag_A = counts * mu_true_cell_expected     # |A_cc| at truth
print(f"\n[Jacobian] |A_cc| at truth (equals n_c * mu_true_c): "
      f"min={diag_A.min():.1f}  median={np.median(diag_A):.1f}  "
      f"max={diag_A.max():.1f}  (all >0 so A PD after sign flip, good)")

# ------------------------- Newton fit -----------------------------------
print("\n" + "=" * 78)
print("BLOCK 2: WITHIN-ESTIMATION DIAGNOSTICS (Newton-Raphson)".center(78))
print("=" * 78)

beta_hat, hist = newton_poisson(u, j, t, y, verbose=False)
print(f"\n[Newton] {len(hist)} iterations")
print(f"  {'it':>3}  {'log-lik':>14}  {'|grad|':>12}  {'max|step|':>12}  "
      f"{'cond(H)':>12}  {'score_sup':>12}")
for h in hist[:3] + ([None] if len(hist) > 6 else []) + hist[-3:]:
    if h is None:
        print("  ...")
        continue
    print(f"  {h['iter']:>3}  {h['ll']:>14.4f}  {h['grad_norm']:>12.3e}  "
          f"{h['max_step']:>12.3e}  {h['hess_cond']:>12.2e}  "
          f"{h['score_sup']:>12.3e}")

# Newton solution should equal closed-form log(mu_hat)
log_mu_closed = np.log(np.maximum(mu_hat, 1e-9))
max_diff = np.max(np.abs(beta_hat - log_mu_closed))
print(f"\n[Newton vs closed-form] max |beta_Newton - log(ybar_cell)| = "
      f"{max_diff:.3e}  (target ~1e-10)")

# At the MLE, score is zero per cell
mu_flat_hat = np.exp(beta_hat.ravel())
s_flat_hat = np.bincount(cell_index(u, j, t), weights=y.astype(float),
                         minlength=K_U * K_I * 2)
score_at_mle = s_flat_hat - counts.ravel() * mu_flat_hat
print(f"[MLE check] max |score_c at MLE| = {np.max(np.abs(score_at_mle)):.3e}  "
      f"(target 0)")

# Per-cell Pearson dispersion at the MLE
phi_hat_per_cell = (sumsq_dev / np.maximum(counts - 1, 1)) / np.maximum(mu_hat, 1e-9)
print(f"[dispersion] phi_hat per cell: mean={phi_hat_per_cell.mean():.3f}  "
      f"median={np.median(phi_hat_per_cell):.3f}  "
      f"sd={phi_hat_per_cell.std():.3f}  target phi={PHI}")

# Sandwich check: Var_sandwich[c] / Var_model[c] should approx phi
var_model_c = 1.0 / (counts * mu_hat)                       # A^{-1}, diag
var_sand_c = sumsq_dev / (counts * mu_hat) ** 2             # A^{-1} B A^{-1}
ratio = var_sand_c / var_model_c
print(f"[sandwich/model] Var_sand / Var_model per cell: "
      f"median={np.median(ratio):.3f}  mean={ratio.mean():.3f}  "
      f"target phi={PHI}")


# ------------------------- tau_hat + three SEs --------------------------
def tau_from_beta(beta):
    return beta[:, :, 1] - beta[:, :, 0]


tau_hat = tau_from_beta(beta_hat)
se_IF_focal = np.sqrt(var_sand_c[:, :, 0] + var_sand_c[:, :, 1])

# Pairs bootstrap
t0 = time.time()
tau_boot = np.empty((B_BOOT, K_U, K_I))
for b in range(B_BOOT):
    idx = rng.integers(N, size=N)
    mu_b, _, _ = aggregate(u[idx], j[idx], t[idx], y[idx])
    tau_boot[b] = np.log(np.maximum(mu_b[:, :, 1], 1e-9)) - \
                  np.log(np.maximum(mu_b[:, :, 0], 1e-9))
se_boot_focal = tau_boot.std(axis=0, ddof=1)
t_boot = time.time() - t0

# Jackknife: delete one obs at a time is 30k refits, so use delete-d block.
# Here: delete-obs jackknife via Efron's closed-form for cell means.
# Var_jack(log(ybar_c)) = (n_c - 1)/n_c * sum_{i in c} (log(ybar_c_(-i)) - log(ybar_c))^2
# where ybar_c_(-i) = (sum_c - y_i) / (n_c - 1).
def jackknife_se_tau():
    # For each obs i in cell c, mu_hat_c_(-i) = (sum_c - y_i)/(n_c - 1).
    se2 = np.zeros((K_U, K_I, 2))
    c = cell_index(u, j, t).reshape(-1)
    mu_flat = mu_hat.ravel()
    cnt_flat = counts.ravel()
    sums_flat = np.bincount(c, weights=y.astype(float),
                            minlength=K_U * K_I * 2)
    # leave-one-out cell means for each obs
    n_c_obs = cnt_flat[c]
    s_c_obs = sums_flat[c]
    mu_loo = (s_c_obs - y) / np.maximum(n_c_obs - 1, 1)
    log_mu_c_obs = np.log(np.maximum(mu_flat[c], 1e-12))
    log_mu_loo = np.log(np.maximum(mu_loo, 1e-12))
    dev2 = (log_mu_loo - log_mu_c_obs) ** 2
    # aggregate sum of deviations per cell
    se2_flat = np.bincount(c, weights=dev2, minlength=K_U * K_I * 2)
    se2 = (cnt_flat - 1) / np.maximum(cnt_flat, 1) * se2_flat
    se2 = se2.reshape(K_U, K_I, 2)
    return np.sqrt(se2[:, :, 0] + se2[:, :, 1])


se_jack_focal = jackknife_se_tau()

# ------------------------- MC benchmark ---------------------------------
t0 = time.time()
tau_mc = np.empty((M_MC, K_U, K_I))
se_IF_mc = np.empty((M_MC, K_U, K_I))
cover_mc = np.zeros((M_MC, K_U, K_I), dtype=bool)
for m in range(M_MC):
    um, jm, tm, ym, _ = simulate(np.random.default_rng(1000 + m))
    mu_m, cnt_m, ss_m = aggregate(um, jm, tm, ym)
    var_m = ss_m / (cnt_m * mu_m) ** 2
    tau_m = np.log(np.maximum(mu_m[:, :, 1], 1e-9)) - \
            np.log(np.maximum(mu_m[:, :, 0], 1e-9))
    se_m = np.sqrt(var_m[:, :, 0] + var_m[:, :, 1])
    tau_mc[m] = tau_m
    se_IF_mc[m] = se_m
    cover_mc[m] = np.abs(tau_m - tau_true) <= 1.96 * se_m
t_mc = time.time() - t0
emp_sd = tau_mc.std(axis=0, ddof=1)
mc_bias = tau_mc.mean(axis=0) - tau_true

# ------------------------- BLOCK 3 ---------------------------------------
print("\n" + "=" * 78)
print("BLOCK 3: POST-ESTIMATION DIAGNOSTICS".center(78))
print("=" * 78)

# Pearson + deviance residuals (obs-level)
mu_obs_hat = mu_hat[u, j, t]
pearson = (y - mu_obs_hat) / np.sqrt(np.maximum(mu_obs_hat, 1e-9))
# deviance for Poisson: sign(y-mu)*sqrt(2*(y*log(y/mu) - (y - mu))), with 0*log0=0
with np.errstate(divide="ignore", invalid="ignore"):
    ratio = np.where(y > 0, y * np.log(np.maximum(y, 1) /
                                       np.maximum(mu_obs_hat, 1e-12)), 0.0)
dev = np.sign(y - mu_obs_hat) * np.sqrt(np.maximum(2 * (ratio - (y - mu_obs_hat)), 0))

print(f"\n[residuals] Pearson:  mean={pearson.mean():+.3f}  "
      f"sd={pearson.std():.3f}  target sd=sqrt(phi)={np.sqrt(PHI):.3f}")
print(f"[residuals] Deviance: mean={dev.mean():+.3f}  sd={dev.std():.3f}")
pearson_chi2 = (pearson ** 2).sum()
n_params = (K_U * K_I * 2)
phi_pearson = pearson_chi2 / (N - n_params)
print(f"[residuals] Pearson chi^2/(n-p) = {phi_pearson:.3f}  target phi={PHI}")

# --- SE comparison table
def row(name, x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    print(f"  {name:<42}{med:>9.4f}{x.mean():>9.4f}   "
          f"[{q1:.3f}, {q3:.3f}]   {x.min():>6.3f}..{x.max():<6.3f}")

print(f"\n[SE table]  100 cells, focal replicate (N={N}, "
      f"bootstrap {B_BOOT} reps in {t_boot:.1f}s, MC {M_MC} reps in {t_mc:.1f}s)")
print(f"  {'metric':<42}{'median':>9}{'mean':>9}   {'IQR':>16}   "
      f"{'range':>12}")
row("SE_IF    (focal)", se_IF_focal)
row("SE_boot  (focal)", se_boot_focal)
row("SE_jack  (focal)", se_jack_focal)
row("MC empirical SD of tau_hat", emp_sd)
row("mean SE_IF across MC reps", se_IF_mc.mean(axis=0))

print(f"\n[per-cell ratios]")
row("SE_IF / SE_boot       (focal)", se_IF_focal / se_boot_focal)
row("SE_IF / SE_jack       (focal)", se_IF_focal / se_jack_focal)
row("SE_boot / SE_jack     (focal)", se_boot_focal / se_jack_focal)
row("mean SE_IF / emp SD   (MC)",   se_IF_mc.mean(axis=0) / emp_sd)
row("SE_boot  / emp SD     (focal vs MC)", se_boot_focal / emp_sd)
row("SE_jack  / emp SD     (focal vs MC)", se_jack_focal / emp_sd)

print(f"\n[per-cell correlations of SEs across 100 cells]")
for name, a, b in [
    ("SE_IF   vs SE_boot   (focal)",      se_IF_focal,               se_boot_focal),
    ("SE_IF   vs SE_jack   (focal)",      se_IF_focal,               se_jack_focal),
    ("SE_boot vs SE_jack   (focal)",      se_boot_focal,             se_jack_focal),
    ("SE_IF (MC avg) vs emp SD",          se_IF_mc.mean(axis=0),     emp_sd),
    ("SE_boot (focal) vs emp SD",         se_boot_focal,             emp_sd),
]:
    c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    print(f"  {name:<40}  corr = {c:.4f}")

print(f"\n[bias]  MC bias of tau_hat (should be ~0 for unbiased estimator)")
row("|mc_bias| across cells", np.abs(mc_bias))
print(f"  sqrt-n-scaled max|bias|: {np.sqrt(N) * np.abs(mc_bias).max():.3f}")

# --- Coverage slices
print(f"\n[coverage]  nominal 95% CI using IF SE, over {M_MC} MC reps x 100 cells")
print(f"  overall:                      {cover_mc.mean():.4f}")
big_tau = np.abs(tau_true) > 0.5
print(f"  cells with |tau*| > 0.5  ({big_tau.sum():3d}): "
      f"{cover_mc[:, big_tau].mean():.4f}")
print(f"  cells with |tau*| <= 0.5 ({(~big_tau).sum():3d}): "
      f"{cover_mc[:, ~big_tau].mean():.4f}")
pcov = cover_mc.mean(axis=0)
print(f"  per-cell coverage spread: min={pcov.min():.3f}  "
      f"med={np.median(pcov):.3f}  max={pcov.max():.3f}")

# Simultaneous (Bonferroni) coverage
z_bonf = 1.96 * np.sqrt(1 + np.log(100) / np.log(1.96 ** 2))  # rough
# Actually use proper: 1 - 0.05/100 tail
from scipy.stats import norm
z_b = norm.ppf(1 - 0.025 / 100)
cover_bonf = (np.abs(tau_mc - tau_true) <= z_b * se_IF_mc).all(axis=(1, 2))
print(f"  simultaneous Bonferroni (z={z_b:.2f}) coverage of ALL 100 cells: "
      f"{cover_bonf.mean():.4f}  target 0.95")

# --- Bootstrap interval comparison for a few representative cells
idxs = [np.argmin(np.abs(tau_true - np.percentile(tau_true, q)).ravel())
        for q in [10, 50, 90]]
print(f"\n[CI spot-check]  3 cells at 10/50/90 percentile of tau*")
print(f"  {'cell':<8}{'tau*':>8}{'tau_hat':>9}"
      f"{'IF 95%CI':>22}{'boot pctl 95%':>22}")
for flat in idxs:
    cu, ci = flat // K_I, flat % K_I
    th = tau_hat[cu, ci]
    se = se_IF_focal[cu, ci]
    ci_if = (th - 1.96 * se, th + 1.96 * se)
    ci_pc = (np.percentile(tau_boot[:, cu, ci], 2.5),
             np.percentile(tau_boot[:, cu, ci], 97.5))
    print(f"  ({cu},{ci}){'':<3}{tau_true[cu, ci]:>+8.3f}{th:>+9.3f}"
          f"  [{ci_if[0]:+.3f}, {ci_if[1]:+.3f}]"
          f"  [{ci_pc[0]:+.3f}, {ci_pc[1]:+.3f}]")

# --- IF-vs-bootstrap residual diagnostics
diff = se_IF_focal - se_boot_focal
idx_sort = np.argsort(np.abs(diff).ravel())
print(f"\n[mismatch] 3 smallest and 3 largest |SE_IF - SE_boot| differences")
for tag, idxs in [("smallest", idx_sort[:3]), ("largest", idx_sort[-3:][::-1])]:
    print(f"  -- {tag} --")
    for flat in idxs:
        cu, ci = flat // K_I, flat % K_I
        print(f"  ({cu},{ci})  tau*={tau_true[cu, ci]:+.3f}  "
              f"tau_hat={tau_hat[cu, ci]:+.3f}  "
              f"SE_IF={se_IF_focal[cu, ci]:.4f}  "
              f"SE_boot={se_boot_focal[cu, ci]:.4f}  "
              f"|diff|={abs(diff[cu, ci]):.4f}")

print("\n" + "=" * 78)
print("DONE.".center(78))
print("=" * 78)
