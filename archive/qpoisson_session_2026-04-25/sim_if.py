"""
Monte Carlo check of the quasi-Poisson QML influence function.

Model:    y_i | x_i has E[y|x] = mu_i = exp(x_i' beta),  Var(y|x) = phi * mu_i
Score:    psi_i(beta) = x_i (y_i - mu_i)
Jacobian: A = E[d psi / d beta'] = -E[mu x x']
IF:       IF_i = -A^{-1} psi_i = (E[mu x x'])^{-1} x_i (y_i - mu_i)

Asymptotic linearization:
    beta_hat - beta_0  ~=  (1/n) sum_i IF_i(beta_0)

Sandwich variance (under quasi-Poisson Var(y|x) = phi * mu):
    Var(beta_hat)  ~=  (1/n) * phi * (E[mu x x'])^{-1}

Simulation strategy: draw NB2 with var = phi * mu so overdispersion matches
quasi-Poisson. Fit by solving the Poisson score. Compare:
  (a) empirical bias of beta_hat
  (b) empirical SD of beta_hat  vs  sandwich SE  vs  naive (phi=1) SE
  (c) correlation between (beta_hat - beta_0) and (1/n) sum IF_i(beta_0), per replicate
"""

import numpy as np
from scipy.optimize import root

rng = np.random.default_rng(42)

# --- design ---------------------------------------------------------------
n = 2000
beta_true = np.array([0.5, 0.3, -0.2])  # intercept, x1, x2
k = len(beta_true)
X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
phi = 2.5  # overdispersion factor


def simulate_y(X, beta, phi, rng):
    mu = np.exp(X @ beta)
    if phi == 1.0:
        return rng.poisson(mu)
    # NB2 reparam: var = mu + mu^2/theta = phi*mu  =>  theta = mu / (phi - 1)
    theta = mu / (phi - 1.0)
    p = theta / (theta + mu)  # success prob in numpy's NB: E[y] = theta*(1-p)/p = mu
    return rng.negative_binomial(theta, p)


def poisson_score(beta, X, y):
    mu = np.exp(X @ beta)
    return X.T @ (y - mu)


def fit_qpoisson(X, y):
    sol = root(lambda b: poisson_score(b, X, y), np.zeros(X.shape[1]), method="hybr")
    assert sol.success, sol.message
    return sol.x


# --- Monte Carlo ----------------------------------------------------------
M = 2000
beta_hats = np.zeros((M, k))
IF_avg = np.zeros((M, k))  # (1/n) sum_i IF_i(beta_true), per replicate

mu0 = np.exp(X @ beta_true)
# E[mu x x'] estimated by sample mean (1/n) sum mu_i x_i x_i'
# => bread = (E[mu x x'])^{-1}  (NOT inv of the sum)
bread_true = np.linalg.inv((X.T * mu0) @ X / n)

for m in range(M):
    y = simulate_y(X, beta_true, phi=phi, rng=rng)
    bhat = fit_qpoisson(X, y)
    beta_hats[m] = bhat
    psi = X * (y - mu0)[:, None]        # n x k, score contributions at truth
    IF = psi @ bread_true.T             # n x k, IF_i = bread * x_i * (y_i - mu_i)
    IF_avg[m] = IF.mean(axis=0)

empirical_bias = beta_hats.mean(axis=0) - beta_true
empirical_sd = beta_hats.std(axis=0, ddof=1)
if_avg_sd = IF_avg.std(axis=0, ddof=1)

# Theoretical SEs at the truth (use sample design, so "theoretical" ≈ plug-in)
#   sandwich:  Var = A^{-1} B A^{-1}, with B = sum psi psi' ≈ phi * X' diag(mu) X
#              => diag(Var) = phi * diag(bread)
sandwich_se_theory = np.sqrt(phi * np.diag(bread_true) / n)
#   naive Poisson (ignoring overdispersion): SE = sqrt(diag(bread)/n)
naive_se_theory = np.sqrt(np.diag(bread_true) / n)

# Per-replicate sandwich SE (averaged over MC) to mimic what a practitioner reports
def sandwich_se_sample(bhat, X, y):
    mu = np.exp(X @ bhat)
    A = -(X.T * mu) @ X
    psi = X * (y - mu)[:, None]
    B = psi.T @ psi
    A_inv = np.linalg.inv(A)
    V = A_inv @ B @ A_inv.T
    return np.sqrt(np.diag(V))

# Re-run a lighter pass to collect sandwich SEs per replicate (reuses rng stream order -> new draw)
rng2 = np.random.default_rng(123)
sand_se_mc = np.zeros((M, k))
for m in range(M):
    y = simulate_y(X, beta_true, phi=phi, rng=rng2)
    bhat = fit_qpoisson(X, y)
    sand_se_mc[m] = sandwich_se_sample(bhat, X, y)

print("=" * 72)
print(f"Quasi-Poisson QML influence function check")
print(f"n={n}, M={M} replicates, phi={phi}, k={k} (intercept + 2 regressors)")
print("=" * 72)
print(f"\ntrue beta           : {beta_true}")
print(f"mean beta_hat       : {beta_hats.mean(axis=0).round(5)}")
print(f"empirical bias      : {empirical_bias.round(5)}")
print(f"\nempirical SD of beta_hat across {M} reps:")
print(f"                    : {empirical_sd.round(5)}")
print(f"sandwich SE (truth-plug):")
print(f"                    : {sandwich_se_theory.round(5)}")
print(f"sandwich SE (MC avg of per-rep):")
print(f"                    : {sand_se_mc.mean(axis=0).round(5)}")
print(f"naive Poisson SE (ignores phi):")
print(f"                    : {naive_se_theory.round(5)}")
print(f"  ratio naive/empirical : {(naive_se_theory/empirical_sd).round(3)}")
print(f"  expected if IF right  : 1/sqrt(phi) = {1/np.sqrt(phi):.3f}")

print(f"\nSD of (1/n) sum IF_i(beta_0), across reps:")
print(f"                    : {if_avg_sd.round(5)}")
print(f"  (should match empirical SD of beta_hat)")

print(f"\nPer-replicate correlation:  (beta_hat_j - beta_j)  vs  (1/n) sum IF_{{i,j}}")
for j in range(k):
    c = np.corrcoef(beta_hats[:, j] - beta_true[j], IF_avg[:, j])[0, 1]
    print(f"  j={j}: corr = {c:.4f}")
print("  (linearization says this should be -> 1 as n grows)")

# Regression of (beta_hat - beta_0) on IF_avg: slope should be ~1
print(f"\nRegression slope of (beta_hat_j - beta_j) on (1/n) sum IF_{{i,j}}:")
for j in range(k):
    x = IF_avg[:, j]
    yv = beta_hats[:, j] - beta_true[j]
    slope = np.cov(yv, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    print(f"  j={j}: slope = {slope:.4f}  (target 1.0)")

print("\n" + "=" * 72)
