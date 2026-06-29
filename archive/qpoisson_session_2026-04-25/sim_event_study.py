"""
10x10 event-study simulation with beta-net-generated heterogeneous effects.

DGP (obs i):
    user cluster u_i ~ Uniform{0..9},  item cluster j_i ~ Uniform{0..9},
    period t_i ~ Uniform{0,1}  (pre/post, i.e. D_i = t_i).
    log E[y_i] = alpha_{u_i} + gamma_{j_i} + tau*_{u_i, j_i} * t_i
    Var(y_i) = phi * E[y_i]          (quasi-Poisson overdispersion)
    tau*_{u,j} = BetaNet(z_u, z_j)   -- 10x10 "radically heterogeneous" truth

Estimator (saturated cell FE Poisson QMLE):
    tau_hat_{u,j} = log( mean(y | u,j,1) ) - log( mean(y | u,j,0) )

Three standard errors for each of the 100 cells:
    (A) IF sandwich:
        SE_IF(tau_{u,j})^2 = sum_{i in (u,j,1)} (y_i - mu_hat_1)^2 / (n_1 * mu_hat_1)^2
                           + sum_{i in (u,j,0)} (y_i - mu_hat_0)^2 / (n_0 * mu_hat_0)^2
    (B) Pairs bootstrap, B=500 resamples
    (C) Monte Carlo benchmark: empirical SD of tau_hat_{u,j} across M DGP reps

A correct IF means (A) and (B) agree per-cell in one draw, and both track (C)
on average. CI coverage using IF SE should hit 95%.
"""

import numpy as np
import torch
import torch.nn as nn
import time

# ---------- config ----------
K_U = K_I = 10
EMB_D = 8
N = 30_000          # one replicate
PHI = 2.0           # quasi-Poisson overdispersion
B_BOOT = 500        # bootstrap draws in the focal replicate
M_MC = 300          # MC reps for the SE-truth benchmark
SEED = 11
np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- beta net: true heterogeneous tau over 10x10 cells ----------
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
with torch.no_grad():
    tau_raw = BetaNet(EMB_D)(z_u, z_i).numpy()
# Rescale so tau* has std ~0.6 and range ~[-1.5, 1.5]: radically heterogeneous
tau_true = (tau_raw - tau_raw.mean()) / tau_raw.std() * 0.6
print(f"tau* range: [{tau_true.min():+.3f}, {tau_true.max():+.3f}]  "
      f"std={tau_true.std():.3f}  mean={tau_true.mean():+.3f}")

# baseline FEs
alpha_u = np.random.normal(0, 0.3, K_U)
gamma_j = np.random.normal(0, 0.3, K_I)
BASE = 1.3   # intercept so mu is ~exp(1.3) = 3.7


def simulate_dataset(rng):
    u = rng.integers(K_U, size=N)
    j = rng.integers(K_I, size=N)
    t = rng.integers(2, size=N)
    log_mu = BASE + alpha_u[u] + gamma_j[j] + tau_true[u, j] * t
    mu = np.exp(log_mu)
    # NB2 with var = phi * mu
    theta = mu / (PHI - 1.0)
    p = theta / (theta + mu)
    y = rng.negative_binomial(theta, p)
    return u, j, t, y


def fit_saturated(u, j, t, y):
    """Return mu_hat[K_U, K_I, 2], counts[K_U, K_I, 2], sumsq[K_U, K_I, 2]."""
    cell = u * (K_I * 2) + j * 2 + t   # flat index in [0, 200)
    counts = np.bincount(cell, minlength=K_U * K_I * 2).reshape(K_U, K_I, 2)
    sums = np.bincount(cell, weights=y, minlength=K_U * K_I * 2).reshape(K_U, K_I, 2)
    mu_hat = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    # sum of (y - mu)^2 per cell, computed as sum y^2 - n * mu^2
    sumsq_y = np.bincount(cell, weights=y.astype(float) ** 2,
                          minlength=K_U * K_I * 2).reshape(K_U, K_I, 2)
    sumsq_dev = sumsq_y - counts * mu_hat ** 2
    return mu_hat, counts, sumsq_dev


def tau_and_IF_se(mu_hat, counts, sumsq_dev):
    """tau_hat[K_U,K_I], se_IF[K_U,K_I]."""
    tau_hat = np.log(mu_hat[:, :, 1]) - np.log(mu_hat[:, :, 0])
    # Var(beta_hat_c) = sumsq_dev_c / (n_c * mu_c)^2
    var_cell = sumsq_dev / (counts * mu_hat) ** 2
    se_IF = np.sqrt(var_cell[:, :, 0] + var_cell[:, :, 1])
    return tau_hat, se_IF


# ---------- focal replicate: IF vs bootstrap ----------
rng = np.random.default_rng(SEED)
u, j, t, y = simulate_dataset(rng)
mu_hat, counts, sumsq_dev = fit_saturated(u, j, t, y)
tau_hat, se_IF = tau_and_IF_se(mu_hat, counts, sumsq_dev)

print(f"\nfocal dataset: N={N}, mean obs per (u,j,t) cell = {counts.mean():.1f}, "
      f"min = {counts.min()}")
print(f"tau_hat range: [{tau_hat.min():+.3f}, {tau_hat.max():+.3f}]")

t0 = time.time()
tau_boots = np.empty((B_BOOT, K_U, K_I))
for b in range(B_BOOT):
    idx = rng.integers(N, size=N)
    mu_b, counts_b, _ = fit_saturated(u[idx], j[idx], t[idx], y[idx])
    tau_boots[b] = np.log(mu_b[:, :, 1]) - np.log(mu_b[:, :, 0])
se_boot = tau_boots.std(axis=0, ddof=1)
print(f"bootstrap: {B_BOOT} reps in {time.time() - t0:.1f}s")

# ---------- MC benchmark: empirical SD across M DGP reps ----------
print(f"\nMC benchmark: simulating M={M_MC} replicate datasets for SE ground truth...")
t0 = time.time()
tau_mc = np.empty((M_MC, K_U, K_I))
se_IF_mc = np.empty((M_MC, K_U, K_I))
cover_mc = np.zeros((M_MC, K_U, K_I), dtype=bool)
for m in range(M_MC):
    um, jm, tm, ym = simulate_dataset(np.random.default_rng(1000 + m))
    mu_m, counts_m, sumsq_m = fit_saturated(um, jm, tm, ym)
    tau_m, se_m = tau_and_IF_se(mu_m, counts_m, sumsq_m)
    tau_mc[m] = tau_m
    se_IF_mc[m] = se_m
    cover_mc[m] = np.abs(tau_m - tau_true) <= 1.96 * se_m
emp_sd = tau_mc.std(axis=0, ddof=1)                # "truth" SE
print(f"MC replicates in {time.time() - t0:.1f}s")

# ---------- report ----------
print("\n" + "=" * 76)
print("Event-study cell-level SE comparison  (100 cells, 10 user x 10 item)")
print("=" * 76)
print(f"{'metric':<44}{'median':>10}{'mean':>10}{'IQR':>12}")
def row(name, x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    print(f"{name:<44}{med:>10.4f}{x.mean():>10.4f}   [{q1:.3f}, {q3:.3f}]")

row("SE_IF (focal rep)", se_IF)
row("SE_boot (focal rep)", se_boot)
row("empirical SD of tau_hat across MC reps", emp_sd)
row("SE_IF averaged across MC reps", se_IF_mc.mean(axis=0))

print(f"\nper-cell ratios (100 cells, one focal dataset):")
r_ifb  = se_IF  / se_boot
r_ifmc = se_IF_mc.mean(axis=0) / emp_sd
r_bmc  = se_boot / emp_sd
row("SE_IF / SE_boot  (focal)",     r_ifb)
row("mean SE_IF / empirical SD (MC)", r_ifmc)
row("SE_boot (focal) / empirical SD", r_bmc)

print(f"\nper-cell correlations:")
for name, a, b in [("IF vs boot (focal)", se_IF, se_boot),
                   ("IF (MC avg) vs empirical SD", se_IF_mc.mean(axis=0), emp_sd),
                   ("boot (focal) vs empirical SD", se_boot, emp_sd)]:
    c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    print(f"  {name:<40}  corr = {c:.4f}")

print(f"\n95% CI coverage of tau*_{{u,j}} using IF SE:")
print(f"  overall (across 100 cells x {M_MC} reps): "
      f"{cover_mc.mean():.4f}   (target 0.95)")
per_cell_cov = cover_mc.mean(axis=0)
print(f"  per-cell coverage spread: min={per_cell_cov.min():.3f}, "
      f"med={np.median(per_cell_cov):.3f}, max={per_cell_cov.max():.3f}")

# worst-case cells
flat_diff = np.abs(se_IF - se_boot).ravel()
worst = np.argsort(flat_diff)[-3:][::-1]
print(f"\n3 worst IF-vs-boot mismatches (focal rep):")
print(f"  {'cell':<10}{'tau*':>9}{'tau_hat':>10}{'SE_IF':>9}{'SE_boot':>10}{'|diff|':>9}")
for idx in worst:
    uu, ii = idx // K_I, idx % K_I
    print(f"  ({uu},{ii}){'':<5}{tau_true[uu,ii]:>+9.3f}"
          f"{tau_hat[uu,ii]:>+10.3f}{se_IF[uu,ii]:>9.4f}"
          f"{se_boot[uu,ii]:>10.4f}{flat_diff[idx]:>9.4f}")

print("\n" + "=" * 76)
