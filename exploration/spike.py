"""
Spike: FLM vs RieszNet on a known-truth ATE, linear and logit DGPs.

Question this settles: are FLM (analytic influence function, via the package)
and a minimal RieszNet (automatic / learned Riesz representer) the SAME
estimator built two ways? If both land on the Monte-Carlo-known truth with
~95% CI coverage and SE ratio ~1, the "one debiased-inference spine, two
backends" thesis holds and the production backend is worth building.

Estimand (same for both methods): ATE = E[ g(1,X) - g(0,X) ].
  - Linear  Y = a(X) + b(X) T + noise           -> ATE = E[b(X)]  (analytic)
  - Logit   Y ~ Bernoulli(sigmoid(a(X)+b(X)T))   -> ATE on prob scale (MC truth)

Methods compared per replication:
  - Oracle    : correctly-specified parametric MLE + delta-method SE (gold std)
  - FLM       : the deep_inference package (structural_dml / inference)
  - RieszNet  : ~120-line standalone auto-DML, SE via the package's variance fn
  - Naive     : plug-in mean(g1-g0), no correction (shows debiasing is needed)

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/spike.py --smoke
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/spike.py
"""

import os
# Parallelism is ACROSS reps (one process per rep), so each worker must be
# single-threaded or the BLAS/OMP backends oversubscribe (14 workers x 16 BLAS
# threads = 260 threads on 16 cores). Must be set before numpy/torch import.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import warnings
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
import statsmodels.api as sm
from scipy.stats import norm

from deep_inference import structural_dml, inference
from deep_inference.engine.variance import compute_se_ci

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# ---- DGP ------------------------------------------------------------------
# a(X) = A0 + A1*X0 + A2*X1   (confounders enter the outcome)
# b(X) = B0 + B1*X0           (heterogeneous effect; E[b(X)] = B0 since E[X]=0)
# e(X) = sigmoid(G*(X0+X1))   (propensity: T is confounded through X0,X1)
D_X = 5
A0, A1, A2 = 0.5, 0.8, -0.6
B0, B1 = 1.0, 0.5
GAMMA = 1.0
SIGMA = 1.0  # linear outcome noise sd


def draw_X(n, rng):
    return rng.standard_normal((n, D_X))


def a_of(X):
    return A0 + A1 * X[:, 0] + A2 * X[:, 1]


def b_of(X):
    return B0 + B1 * X[:, 0]


def propensity(X):
    return 1.0 / (1.0 + np.exp(-GAMMA * (X[:, 0] + X[:, 1])))


def gen_linear(n, rng):
    X = draw_X(n, rng)
    T = (rng.uniform(size=n) < propensity(X)).astype(float)
    Y = a_of(X) + b_of(X) * T + SIGMA * rng.standard_normal(n)
    return Y, T, X


def gen_logit(n, rng):
    X = draw_X(n, rng)
    T = (rng.uniform(size=n) < propensity(X)).astype(float)
    p = 1.0 / (1.0 + np.exp(-(a_of(X) + b_of(X) * T)))
    Y = (rng.uniform(size=n) < p).astype(float)
    return Y, T, X


def truth_linear():
    # ATE = E[b(X)] = B0 (E[X0]=0), exact.
    return B0


def truth_logit(rng):
    # MC truth: E[ sigmoid(a+b) - sigmoid(a) ] over the X distribution.
    Xb = draw_X(2_000_000, rng)
    a, b = a_of(Xb), b_of(Xb)
    s1 = 1.0 / (1.0 + np.exp(-(a + b)))
    s0 = 1.0 / (1.0 + np.exp(-a))
    return float(np.mean(s1 - s0))


# ---- Poisson DGP: Y ~ Poisson(lambda), lambda = exp(a(X) + b(X) T) --------
# Smaller coefficients than linear/logit so lambda=exp(eta) stays moderate (counts ~0-8).
# theta-dependent Hessian (weight lambda) like logit, but exp link -> UNBOUNDED weight and
# NO saturation channel: Lambda*(x) near-singular only at overlap (det = e(1-e) lam0 lam1).
PA0, PA1, PA2 = 0.2, 0.4, -0.3
PB0, PB1 = 0.4, 0.2


def a_of_pois(X):
    return PA0 + PA1 * X[:, 0] + PA2 * X[:, 1]


def b_of_pois(X):
    return PB0 + PB1 * X[:, 0]


def gen_poisson(n, rng):
    X = draw_X(n, rng)
    T = (rng.uniform(size=n) < propensity(X)).astype(float)
    lam = np.exp(a_of_pois(X) + b_of_pois(X) * T)
    Y = rng.poisson(lam).astype(float)
    return Y, T, X


def truth_poisson(rng):
    # ATE on the count-mean scale: E[ exp(a+b) - exp(a) ].
    Xb = draw_X(2_000_000, rng)
    a, b = a_of_pois(Xb), b_of_pois(Xb)
    return float(np.mean(np.exp(a + b) - np.exp(a)))


# ---- Oracle (gold standard, correctly specified) --------------------------
def oracle_linear(Y, T, X):
    # True mean: A0 + A1 X0 + A2 X1 + (B0 + B1 X0) T. Design includes T, T*X0.
    D = np.column_stack([np.ones_like(T), X[:, 0], X[:, 1], T, T * X[:, 0]])
    m = sm.OLS(Y, D).fit()
    cov = m.cov_params()
    xbar0 = X[:, 0].mean()
    # mu = coef_T + coef_{T X0} * mean(X0)
    mu = m.params[3] + m.params[4] * xbar0
    # delta-method SE incl. variance of the sample mean of X0
    var = (cov[3, 3] + xbar0**2 * cov[4, 4] + 2 * xbar0 * cov[3, 4]
           + m.params[4] ** 2 * (X[:, 0].var() / len(Y)))
    return mu, np.sqrt(var)


def oracle_logit(Y, T, X):
    D = np.column_stack([np.ones_like(T), X[:, 0], X[:, 1], T, T * X[:, 0]])
    m = sm.Logit(Y, D).fit(disp=0)
    th = m.params
    cov = np.asarray(m.cov_params())
    D1 = np.column_stack([np.ones_like(T), X[:, 0], X[:, 1], np.ones_like(T), X[:, 0]])
    D0 = np.column_stack([np.ones_like(T), X[:, 0], X[:, 1], np.zeros_like(T), np.zeros_like(T)])
    s1 = 1.0 / (1.0 + np.exp(-D1 @ th))
    s0 = 1.0 / (1.0 + np.exp(-D0 @ th))
    mu = float(np.mean(s1 - s0))
    # delta method: grad of mean(s1-s0) wrt theta
    g = (((s1 * (1 - s1))[:, None] * D1) - ((s0 * (1 - s0))[:, None] * D0)).mean(0)
    var = float(g @ cov @ g)
    return mu, np.sqrt(var)


def oracle_poisson(Y, T, X):
    # correctly-specified Poisson GLM + delta-method SE for E[exp(D1 θ) - exp(D0 θ)].
    D = np.column_stack([np.ones_like(T), X[:, 0], X[:, 1], T, T * X[:, 0]])
    m = sm.GLM(Y, D, family=sm.families.Poisson()).fit()
    th = np.asarray(m.params)
    cov = np.asarray(m.cov_params())
    D1 = np.column_stack([np.ones_like(T), X[:, 0], X[:, 1], np.ones_like(T), X[:, 0]])
    D0 = np.column_stack([np.ones_like(T), X[:, 0], X[:, 1], np.zeros_like(T), np.zeros_like(T)])
    l1 = np.exp(D1 @ th)
    l0 = np.exp(D0 @ th)
    mu = float(np.mean(l1 - l0))
    g = ((l1[:, None] * D1) - (l0[:, None] * D0)).mean(0)  # d/dθ mean(exp(D1θ)-exp(D0θ))
    var = float(g @ cov @ g)
    return mu, np.sqrt(var)


# ---- RieszNet: faithful automatic debiased ML -----------------------------
# Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022), RieszNet, arXiv
# 2110.03031. Implements the Section 3 multitasking architecture and the
# Equation (5) loss, with the appendix A.1 hyperparameters.
#   - shared trunk f1(Z): k=3 hidden layers, width 200, ELU       (paper A.1)
#   - Riesz head: LINEAR on the shared rep, a(Z)=<f1(Z),beta>     (Sec 3, l.139)
#   - regression head: d-k=2 hidden layers, width 100, ELU, on f1 (Sec 3, l.145)
#   - targeted regularization: g~(Z)=g(Z)+eps*a(Z), eps unpenalized; the
#     lambda2 square-loss term E[(Y-g~)^2] forces E_n[a(Y-g~)]=0 by its FOC
#     (Sec 3, l.155-177), which is what buys coverage (ablation row 4, l.390).
#   - loss = RegLoss + lambda1*RieszLoss + lambda2*TargetedReg, L2=1e-3 on net
#     weights only (not eps); lambda1=0.1, lambda2=1, Adam            (A.1, l.471)
#   - DR moment for inference: psi = (g~1-g~0) + a(Z)(Y-g~)        (Sec 5, l.231)
# Deliberate adaptation for a fast spike: full-batch Adam at lr 1e-3 with a
# train/val early-stopping split, instead of the paper's 1e-4 then 1e-5 two-step
# minibatch schedule. The architecture, loss, and moment are faithful.
LAMBDA1, LAMBDA2, L2 = 0.1, 1.0, 1e-3


class RieszNet(nn.Module):
    def __init__(self, d_x, k_width=200, g_width=100):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_x + 1, k_width), nn.ELU(),
            nn.Linear(k_width, k_width), nn.ELU(),
            nn.Linear(k_width, k_width), nn.ELU(),
        )
        self.riesz_head = nn.Linear(k_width, 1)            # linear on shared rep
        self.reg_head = nn.Sequential(
            nn.Linear(k_width, g_width), nn.ELU(),
            nn.Linear(g_width, g_width), nn.ELU(),
            nn.Linear(g_width, 1),
        )
        self.eps = nn.Parameter(torch.zeros(1))            # unpenalized TMLE knob

    def forward(self, t, x):
        rep = self.trunk(torch.cat([t, x], dim=1))
        return self.reg_head(rep).squeeze(-1), self.riesz_head(rep).squeeze(-1)


def _outcome_g_reg(g_raw, y, outcome):
    """Mean g(Z) and the regression loss for each outcome type (g_raw is the linear index)."""
    if outcome == "logit":
        return torch.sigmoid(g_raw), nn.functional.binary_cross_entropy_with_logits(g_raw, y)
    if outcome == "poisson":
        g = torch.exp(g_raw)
        return g, (g - y * g_raw).mean()                   # Poisson NLL exp(η)-y·η
    return g_raw, ((g_raw - y) ** 2).mean()                # linear / MSE


def _loss(net, t, x, y, ones, zeros, outcome):
    g_raw, a_obs = net(t, x)
    g_obs, reg = _outcome_g_reg(g_raw, y, outcome)
    _, a1 = net(ones, x)
    _, a0 = net(zeros, x)
    riesz = (a_obs ** 2 - 2.0 * (a1 - a0)).mean()          # E[a^2 - 2 m(a)]
    g_tilde = g_obs + torch.clamp(net.eps, -2.0, 2.0) * a_obs  # clamp the TMLE knob
    tmle = ((y - g_tilde) ** 2).mean()                     # targeted reg (lambda2)
    return reg + LAMBDA1 * riesz + LAMBDA2 * tmle


def _fit_riesz(Xtr, Ttr, Ytr, outcome, max_epochs, patience, seed, restarts=3,
               batch_size=256, weight_decay=L2):
    """Faithful RieszNet fit: minibatch + two-stage LR (Adam 1e-3 then 2e-4). Restart
    selection uses a divergence-robust val metric -- the combined val loss PLUS a penalty
    on the validation representer magnitude. The DR moment ψ = (g̃1-g̃0) + a(Y-g̃) blows up
    when the learned representer a(Z) runs away; the true ATE representer (T-e)/(e(1-e)) is
    bounded by overlap, so an implausibly large val |a| flags a junk head that the plain
    val loss (dominated by the regression term) does not reject. Truth-free guard."""
    d_x = Xtr.shape[1]

    # train/val split within the fold for early stopping (fixed across restarts)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Ytr))
    n_val = max(1, int(0.2 * len(Ytr)))
    val, tr = idx[:n_val], idx[n_val:]
    n_tr = len(tr)
    stage2 = max_epochs // 2  # LR drop point

    def pack(ii):
        x = torch.tensor(Xtr[ii], dtype=torch.float32)
        t = torch.tensor(Ttr[ii], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(Ytr[ii], dtype=torch.float32)
        return t, x, y, torch.ones_like(t), torch.zeros_like(t)

    val_b = pack(val)

    def val_score(net):
        # combined val loss + soft penalty on a runaway representer (rms of a on val)
        with torch.no_grad():
            loss = _loss(net, *val_b, outcome).item()
            _, a_val = net(val_b[0], val_b[1])
            a_rms = float((a_val ** 2).mean().sqrt())
        return loss + 0.01 * max(0.0, a_rms - 25.0) ** 2

    global_best, global_state = float("inf"), None
    for r in range(restarts):
        torch.manual_seed(seed + 1000 * r)
        brng = np.random.default_rng(seed + 7 * r + 1)
        net = RieszNet(d_x)
        # L2 on net weights only, NOT on eps (paper: R does not take eps as input)
        opt = torch.optim.Adam([
            {"params": [p for n, p in net.named_parameters() if n != "eps"], "weight_decay": weight_decay},
            {"params": [net.eps], "weight_decay": 0.0},
        ], lr=1e-3)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(max_epochs):
            for g in opt.param_groups:
                g["lr"] = 1e-3 if ep < stage2 else 2e-4
            net.train()
            perm = tr[brng.permutation(n_tr)]
            for s in range(0, n_tr, batch_size):
                bb = pack(perm[s:s + batch_size])
                opt.zero_grad()
                _loss(net, *bb, outcome).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
            net.eval()
            v = val_score(net)
            if v < best - 1e-5:
                best, best_state, wait = v, {k: p.clone() for k, p in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None and best < global_best:
            global_best, global_state = best, best_state
    net = RieszNet(d_x)
    if global_state is not None:
        net.load_state_dict(global_state)
    net.eval()
    return net


def _riesz_single(Y, T, X, outcome, K, max_epochs, patience, seed):
    """One K-fold cross-fit RieszNet pass -> (mu, se, mu_naive, se_naive)."""
    n = len(Y)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), K)
    psi = np.zeros(n)
    g_diff = np.zeros(n)
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(K) if j != k])
        net = _fit_riesz(X[tr], T[tr], Y[tr], outcome, max_epochs, patience, seed + k)
        with torch.no_grad():
            xe = torch.tensor(X[te], dtype=torch.float32)
            te_t = torch.tensor(T[te], dtype=torch.float32).unsqueeze(1)
            ones, zeros = torch.ones_like(te_t), torch.zeros_like(te_t)
            eps = float(np.clip(net.eps.item(), -2.0, 2.0))  # clamp TMLE knob (match training)

            def gmean(traw):
                g, a = net(traw, xe)
                if outcome == "logit":
                    g = torch.sigmoid(g)
                elif outcome == "poisson":
                    g = torch.exp(g)
                return g.numpy(), a.numpy()

            g1, a1 = gmean(ones)
            g0, a0 = gmean(zeros)
            g_obs, a_obs = gmean(te_t)
        # TMLE-corrected regression g~ = g + eps*a, then the DR moment
        gt1 = g1 + eps * a1
        gt0 = g0 + eps * a0
        gt_obs = g_obs + eps * a_obs
        g_diff[te] = g1 - g0                                # uncorrected plug-in
        psi[te] = (gt1 - gt0) + a_obs * (Y[te] - gt_obs)    # DR moment
    se, lo, hi, _ = compute_se_ci(torch.tensor(psi))
    mu = float(psi.mean())
    mu_naive = float(g_diff.mean())
    se_naive = float(g_diff.std(ddof=1) / np.sqrt(n))
    return mu, float(se), mu_naive, se_naive


def riesz_ate(Y, T, X, outcome, K=5, max_epochs=400, patience=30, seed=0, repeats=3):
    """Median-DML over `repeats` independent cross-fit splits (Chernozhukov et al.). A junk
    Riesz head in ONE split makes that split's mu_r an outlier; the median across splits
    rejects it, where best-of-restarts and the representer guard could not. SE folds the
    across-split spread in: se² = median_r(se_r² + (mu_r - mu)²)."""
    res = [_riesz_single(Y, T, X, outcome, K, max_epochs, patience, seed + 100 * r)
           for r in range(repeats)]
    mus = np.array([r[0] for r in res])
    ses = np.array([r[1] for r in res])
    mu = float(np.median(mus))
    se = float(np.sqrt(np.median(ses ** 2 + (mus - mu) ** 2)))
    return mu, se, res[0][2], res[0][3]


# ---- FLM via the package --------------------------------------------------
def flm_linear(Y, T, X, n_folds, epochs, n_repeats=1,
               lambda_method="ridge", ridge_alpha=1000.0, three_way=False):
    # three_way=False -> package default (aggregate flat Lambda, ignores
    # lambda_method). three_way=True -> Lambda(x) estimated via lambda_method.
    r = structural_dml(Y, T, X.astype(float), family="linear",
                       n_folds=n_folds, epochs=epochs, n_repeats=n_repeats,
                       lambda_method=lambda_method, ridge_alpha=ridge_alpha,
                       three_way=three_way, hidden_dims=[32], verbose=False)
    # Parameter recovery: theta_hat[:,0]=alpha(x), theta_hat[:,1]=beta(x).
    a_hat, b_hat = r.theta_hat[:, 0], r.theta_hat[:, 1]
    a_star = A0 + A1 * X[:, 0] + A2 * X[:, 1]
    b_star = B0 + B1 * X[:, 0]

    def _r2(hat, star):
        ss_res = float(((hat - star) ** 2).sum())
        ss_tot = float(((star - star.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rec = {"alpha_r2": _r2(a_hat, a_star), "beta_r2": _r2(b_hat, b_star),
           "alpha_rmse": float(np.sqrt(((a_hat - a_star) ** 2).mean())),
           "beta_rmse": float(np.sqrt(((b_hat - b_star) ** 2).mean())),
           "alpha_bias": float((a_hat - a_star).mean()),
           "beta_bias": float((b_hat - b_star).mean())}
    return r.mu_hat, r.se, rec


# cholesky uses the SAME truth-free default (0.01) as logit -- one general setting
# across DGPs, not per-DGP tuned. Linear Λ=2[[1,e],[e,e]] is near-singular only at the
# overlap tails (one channel), so 0.01 is comfortably enough.
LINEAR_TIKHONOV = {"oracle": 1e-8, "cholesky": 0.01, "ridge": 0.01}


class OracleLinearLambda:
    """Oracle Λ(x)=[[1,e],[e,e]] for the linear DGP. The package's Linear loss is
    0.5*(y-pred)², so its per-obs Hessian is [[1,t],[t,t²]] with NO factor 2; the
    conditional mean over T~Bernoulli(e*(x)), e*=σ(γ(x0+x1)), is [[1,e],[e,e]].
    (The legacy structural_dml LinearFamily used (y-μ)² and so carried a 2 -- do NOT
    copy that here; this strategy is injected into the new inference() path.) Ceiling
    reference, NOT a general method. LambdaStrategy-shaped; fit is a no-op."""
    requires_theta = True
    requires_separate_fold = True

    def fit(self, X, T, Y, theta_hat, model):
        return None

    def predict(self, X, theta_hat=None):
        e = torch.sigmoid(GAMMA * (X[:, 0] + X[:, 1]))
        L = X.new_zeros(len(X), 2, 2)
        L[:, 0, 0] = 1.0
        L[:, 0, 1] = e
        L[:, 1, 0] = e
        L[:, 1, 1] = e
        return L


def flm_linear_general(Y, T, X, n_folds, epochs, n_repeats=1, lambda_spec="cholesky",
                       tikhonov=None, max_condition=None):
    """Linear ATE = E[β(X)] via the GENERAL new path (inference + cholesky), so linear
    and logit use the same general estimator. 'flat' = the legacy 2-way aggregate-Λ
    contrast (the under-covering bug); 'oracle' = ceiling reference (not general)."""
    if lambda_spec == "flat":
        r = structural_dml(Y, T, X.astype(float), family="linear", n_folds=n_folds,
                           epochs=epochs, n_repeats=n_repeats, three_way=False,
                           hidden_dims=[32], verbose=False)
        return r.mu_hat, r.se
    kw = dict(model="linear", target="average_slope", n_folds=n_folds, epochs=epochs,
              n_repeats=n_repeats, hidden_dims=[32], verbose=False)
    if lambda_spec == "cholesky":
        if max_condition is not None:
            kw["lambda_strategy"] = _cholesky_strategy(max_condition)
        else:
            kw["lambda_method"] = "cholesky"
    elif lambda_spec == "ridge":
        kw["lambda_method"] = "ridge"
    elif lambda_spec == "oracle":
        kw["lambda_strategy"] = OracleLinearLambda()
    else:
        raise ValueError(f"unknown linear lambda_spec: {lambda_spec}")
    kw["tikhonov_scale"] = tikhonov if tikhonov is not None else LINEAR_TIKHONOV[lambda_spec]
    r = inference(Y, T, X.astype(float), **kw)
    return r.mu_hat, r.se


class OracleLogitLambda:
    """Oracle Λ(x)=E[ℓ_θθ|X=x] for the logit DGP -- the ceiling diagnostic.

    Integrates T~Bernoulli(e*(x)) out of the per-obs logit Hessian w·[[1,t],[t,t²]]
    (w=p(1-p); NO factor 2 -- the package logit loss Hessian carries no 2):
      Λ*(x) = (1-e*)·w0·[[1,0],[0,0]] + e*·w1·[[1,1],[1,1]]
    with e*=σ(γ(x0+x1)), w0=p0(1-p0), w1=p1(1-p1), p0=σ(α*), p1=σ(α*+β*),
    α*=A0+A1 x0+A2 x1, β*=B0+B1 x0. Uses the TRUE DGP constants, so this is the
    achievable ceiling Λ̂(x) should approach. LambdaStrategy-shaped (3-way path,
    same cross-fitting as cholesky/ridge); fit is a no-op since the oracle needs none."""
    requires_theta = True
    requires_separate_fold = True

    def fit(self, X, T, Y, theta_hat, model):
        return None

    def predict(self, X, theta_hat=None):
        x0, x1 = X[:, 0], X[:, 1]
        a = A0 + A1 * x0 + A2 * x1
        b = B0 + B1 * x0
        e = torch.sigmoid(GAMMA * (x0 + x1))
        p0, p1 = torch.sigmoid(a), torch.sigmoid(a + b)
        w0, w1 = p0 * (1 - p0), p1 * (1 - p1)
        L = X.new_zeros(len(X), 2, 2)
        L[:, 0, 0] = (1 - e) * w0 + e * w1
        L[:, 0, 1] = e * w1
        L[:, 1, 0] = e * w1
        L[:, 1, 1] = e * w1
        return L


# tikhonov defaults per spec. cholesky needs REAL inverse regularization on logit:
# the true Λ(x) is near-singular at low overlap (w=p(1-p)->0), so a noisy Λ̂ inverts
# explosively under near-zero tikhonov. oracle-Λ is exactly right so 1e-8 is fine.
# cholesky uses the package DEFAULT tikhonov (0.01), the SAME for both DGPs -- a
# truth-free choice (NOT tuned per-DGP to hit a target SE-ratio, which would require
# knowing the truth). oracle/analytic are labeled reference ceilings, not the solution.
LOGIT_TIKHONOV = {"oracle": 1e-8, "oracleprop": 1e-2, "analytic": 1e-2,
                  "cholesky": 0.01, "ridge": 0.01}


class LogitAnalyticLambda:
    """Λ(x) from the KNOWN logit form. Instead of regressing the Hessian entries
    (cholesky/ridge) or collapsing them (aggregate), estimate the scalar propensity
    ê(x) -- a clean, well-conditioned 1-D problem -- and assemble Λ analytically from
    θ̂(x). This is the logit analogue of the linear 'estimate e(x) then build Λ' fix
    that CLAUDE.md flags as the recommended direction. The near-singularity at the tails
    is still present (it is real), but the inverse is large-and-correct (like oracle),
    not large-and-noisy (like cholesky's amplified fit error).
      Λ(x) = (1-ê)·w0·[[1,0],[0,0]] + ê·w1·[[1,1],[1,1]]
      w0 = σ(α̂)(1-σ(α̂)),  w1 = σ(α̂+β̂)(1-σ(α̂+β̂)),  ê(x) from a propensity model.
    Here the DGP propensity is logistic so LogisticRegression is correctly specified;
    swap in a flexible learner for robustness."""
    requires_theta = True
    requires_separate_fold = True

    def __init__(self, C=1.0, true_prop=False):
        self.C = C
        self.true_prop = true_prop  # oracle ladder rung: inject TRUE e(x) to isolate
        self._clf = None            # the propensity-estimation error (still uses θ̂ for w)

    def fit(self, X, T, Y, theta_hat, model):
        if self.true_prop:
            return None
        from sklearn.linear_model import LogisticRegression
        Xn = X.detach().cpu().numpy()
        Tn = T.detach().cpu().numpy().astype(int)
        self._clf = LogisticRegression(C=self.C, max_iter=500).fit(Xn, Tn)

    def predict(self, X, theta_hat=None):
        if self.true_prop:
            e = torch.sigmoid(GAMMA * (X[:, 0] + X[:, 1]))  # true DGP propensity
        else:
            e = self._clf.predict_proba(X.detach().cpu().numpy())[:, 1]
            e = torch.tensor(np.clip(e, 1e-3, 1 - 1e-3), dtype=X.dtype)
        a, b = theta_hat[:, 0], theta_hat[:, 1]
        p0, p1 = torch.sigmoid(a), torch.sigmoid(a + b)
        w0, w1 = p0 * (1 - p0), p1 * (1 - p1)
        L = X.new_zeros(len(X), 2, 2)
        L[:, 0, 0] = (1 - e) * w0 + e * w1
        L[:, 0, 1] = e * w1
        L[:, 1, 0] = e * w1
        L[:, 1, 1] = e * w1
        return L


def _cholesky_strategy(max_condition):
    """cholesky as a pre-built strategy so max_condition (the spectrum-adaptive inverse
    regularizer) can be swept; None -> the package default (lambda_method path, C=100)."""
    from deep_inference.lambda_.estimate import EstimateLambda
    return EstimateLambda(method="cholesky", max_condition=max_condition)


def flm_logit(Y, T, X, n_folds, epochs, n_repeats=1, lambda_spec="cholesky",
              tikhonov=None, max_condition=None):
    # discrete ATE on probability scale: g(1,X)-g(0,X) = sigmoid(a+b)-sigmoid(a)
    def ate_target(x, theta, t_tilde):
        return torch.sigmoid(theta[0] + theta[1]) - torch.sigmoid(theta[0])
    kw = dict(model="logit", target_fn=ate_target, t_tilde=0.0,
              n_folds=n_folds, epochs=epochs, n_repeats=n_repeats,
              hidden_dims=[32], verbose=False)
    if lambda_spec == "oracle":
        kw["lambda_strategy"] = OracleLogitLambda()
    elif lambda_spec == "oracleprop":
        kw["lambda_strategy"] = LogitAnalyticLambda(true_prop=True)
    elif lambda_spec == "analytic":
        kw["lambda_strategy"] = LogitAnalyticLambda()
    elif lambda_spec == "cholesky":
        if max_condition is not None:
            kw["lambda_strategy"] = _cholesky_strategy(max_condition)
        else:
            kw["lambda_method"] = "cholesky"
    elif lambda_spec == "ridge":
        kw["lambda_method"] = "ridge"            # package default; explicit contrast row
    else:
        raise ValueError(f"unknown logit lambda_spec: {lambda_spec}")
    kw["tikhonov_scale"] = tikhonov if tikhonov is not None else LOGIT_TIKHONOV[lambda_spec]
    r = inference(Y, T, X.astype(float), **kw)
    return r.mu_hat, r.se


POISSON_TIKHONOV = {"oracle": 1e-8, "cholesky": 0.01, "ridge": 0.01}


class OraclePoissonLambda:
    """Oracle Λ(x)=E[ℓ_θθ|X=x] for the Poisson DGP. Per-obs Hessian is λ·[[1,t],[t,t²]],
    λ=exp(α+β t) (loss exp(η)-y·η, NO factor). Integrate over T~Bernoulli(e*):
      Λ*(x) = (1-e)·λ0·[[1,0],[0,0]] + e·λ1·[[1,1],[1,1]], λ0=exp(α*), λ1=exp(α*+β*).
    Ceiling reference, NOT general."""
    requires_theta = True
    requires_separate_fold = True

    def fit(self, X, T, Y, theta_hat, model):
        return None

    def predict(self, X, theta_hat=None):
        x0, x1 = X[:, 0], X[:, 1]
        a = PA0 + PA1 * x0 + PA2 * x1
        b = PB0 + PB1 * x0
        e = torch.sigmoid(GAMMA * (x0 + x1))
        l0, l1 = torch.exp(a), torch.exp(a + b)
        L = X.new_zeros(len(X), 2, 2)
        L[:, 0, 0] = (1 - e) * l0 + e * l1
        L[:, 0, 1] = e * l1
        L[:, 1, 0] = e * l1
        L[:, 1, 1] = e * l1
        return L


def flm_poisson(Y, T, X, n_folds, epochs, n_repeats=1, lambda_spec="cholesky",
                tikhonov=None, max_condition=None):
    # ATE on the count-mean scale exp(a+b)-exp(a), via a CUSTOM LOSS -> fully autodiff (no
    # built-in poisson model), the most general test of the cholesky Λ path.
    def ploss(y, t, theta):
        eta = theta[0] + theta[1] * t
        return torch.exp(eta) - y * eta
    def ate_target(x, theta, t_tilde):
        return torch.exp(theta[0] + theta[1]) - torch.exp(theta[0])
    kw = dict(loss=ploss, target_fn=ate_target, theta_dim=2, t_tilde=0.0,
              hessian_depends_on_theta=True, hessian_depends_on_y=False,
              n_folds=n_folds, epochs=epochs, n_repeats=n_repeats,
              hidden_dims=[32], verbose=False)
    if lambda_spec == "oracle":
        kw["lambda_strategy"] = OraclePoissonLambda()
    elif lambda_spec == "cholesky":
        if max_condition is not None:
            kw["lambda_strategy"] = _cholesky_strategy(max_condition)
        else:
            kw["lambda_method"] = "cholesky"
    elif lambda_spec == "ridge":
        kw["lambda_method"] = "ridge"
    else:
        raise ValueError(f"unknown poisson lambda_spec: {lambda_spec}")
    kw["tikhonov_scale"] = tikhonov if tikhonov is not None else POISSON_TIKHONOV[lambda_spec]
    r = inference(Y, T, X.astype(float), **kw)
    return r.mu_hat, r.se


# ---- Monte Carlo driver ---------------------------------------------------
def covered(mu, se, truth):
    z = norm.ppf(0.975)
    return float(mu - z * se <= truth <= mu + z * se)


# DGP registry: name -> data generator, oracle-MLE anchor, general FLM, RieszNet outcome.
DGPS = {
    "linear":  dict(gen=gen_linear,  oracle=oracle_linear,  flm=flm_linear_general, outcome="linear"),
    "logit":   dict(gen=gen_logit,   oracle=oracle_logit,   flm=flm_logit,          outcome="logit"),
    "poisson": dict(gen=gen_poisson, oracle=oracle_poisson, flm=flm_poisson,        outcome="poisson"),
}
DEFAULT_SPECS = {  # 'flat'/'ridge' = naive-Λ contrast, 'oracle' = ceiling ref, cholesky = general
    "linear":  ("cholesky", "flat", "oracle"),
    "logit":   ("cholesky", "ridge", "oracle"),
    "poisson": ("cholesky", "ridge", "oracle"),
}
DGP_BASE_SEED = {"linear": 1000, "logit": 5000, "poisson": 9000}
TRUTH_FN = {"linear": lambda rng: truth_linear(), "logit": truth_logit, "poisson": truth_poisson}


def _one_rep(task):
    """One MC replication, all methods. Top-level so it pickles for Pool."""
    (truth, dgp_name, n, flm_folds, flm_epochs, flm_repeats, flm_tikhonov,
     flm_max_condition, specs, riesz_epochs, riesz_patience, seed) = task
    # Seed the GLOBAL torch/numpy RNG per rep so FLM (which uses global state for net
    # init) is reproducible and identical serial-vs-parallel; distinct per rep.
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    d = DGPS[dgp_name]
    Y, T, X = d["gen"](n, rng)
    out = {}
    mu, se = d["oracle"](Y, T, X)
    out["Oracle"] = (mu, se, covered(mu, se, truth))
    for spec in specs:  # general cholesky + contrast(ridge/flat) + ceiling(oracle) rows
        mu, se = d["flm"](Y, T, X, flm_folds, flm_epochs, n_repeats=flm_repeats,
                          lambda_spec=spec, tikhonov=flm_tikhonov, max_condition=flm_max_condition)
        out[f"FLM[{spec}]"] = (mu, se, covered(mu, se, truth))
    mu, se, mun, sen = riesz_ate(Y, T, X, d["outcome"], max_epochs=riesz_epochs,
                                 patience=riesz_patience, seed=seed)
    out["RieszNet"] = (mu, se, covered(mu, se, truth))
    out["Naive"] = (mun, sen, covered(mun, sen, truth))
    return out, None


def run(truth, dgp_name, M, n, flm_folds, flm_epochs,
        riesz_epochs, riesz_patience, base_seed, workers=1, flm_repeats=1,
        flm_tikhonov=None, flm_max_condition=None, specs=("cholesky", "oracle")):
    acc = {}  # method -> {est,se,cov}; built lazily so the FLM rows appear in order
    rec_acc = {}  # recovery panel retired (goal is coverage/SE-ratio)
    tasks = [(truth, dgp_name, n, flm_folds, flm_epochs, flm_repeats, flm_tikhonov,
              flm_max_condition, list(specs),
              riesz_epochs, riesz_patience, base_seed + i) for i in range(M)]

    def absorb(i, result):
        out, rec = result
        for m, (est, se, cov) in out.items():
            d = acc.setdefault(m, {"est": [], "se": [], "cov": []})
            d["est"].append(est); d["se"].append(se); d["cov"].append(cov)
        if rec is not None:
            for k in rec_acc:
                rec_acc[k].append(rec[k])
        flm_str = " ".join(f"{k}={out[k][0]:.3f}" for k in out if k.startswith("FLM"))
        print(f"  rep {i+1}/{M}: oracle={out['Oracle'][0]:.3f} {flm_str} "
              f"riesz={out['RieszNet'][0]:.3f} naive={out['Naive'][0]:.3f}", flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks):
            absorb(i, _one_rep(t))
    else:
        # imap keeps results in submission order so seeds map to reps deterministically
        with Pool(processes=workers) as pool:
            for i, out in enumerate(pool.imap(_one_rep, tasks)):
                absorb(i, out)
    return acc, rec_acc


def summarize(name, truth, acc, rec_acc, M):
    lines = [f"\n### {name}  (truth = {truth:.4f}, M = {M})\n",
             "| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |",
             "|---|---|---|---|---|---|---|"]
    for m, d in acc.items():
        est = np.array(d["est"]); se = np.array(d["se"]); cov = np.array(d["cov"])
        emp = est.std(ddof=1)
        lines.append(f"| {m} | {est.mean():.4f} | {est.mean()-truth:+.4f} | "
                     f"{emp:.4f} | {se.mean():.4f} | {se.mean()/emp:.2f} | "
                     f"{100*cov.mean():.0f}% |")
    if rec_acc and len(rec_acc["beta_r2"]) > 0:
        ar2 = np.array(rec_acc["alpha_r2"]); br2 = np.array(rec_acc["beta_r2"])
        arm = np.array(rec_acc["alpha_rmse"]); brm = np.array(rec_acc["beta_rmse"])
        ab = np.array(rec_acc["alpha_bias"]); bb = np.array(rec_acc["beta_bias"])
        lines += ["\n**FLM parameter recovery** (theta_hat(x) vs truth, mean over reps)\n",
                  "| param | R2 | RMSE | bias |",
                  "|---|---|---|---|",
                  f"| alpha(x) | {ar2.mean():.4f} | {arm.mean():.4f} | {ab.mean():+.4f} |",
                  f"| beta(x)  | {br2.mean():.4f} | {brm.mean():.4f} | {bb.mean():+.4f} |"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to check wiring")
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--flm-folds", type=int, default=5, help="FLM cross-fit folds")
    ap.add_argument("--flm-epochs", type=int, default=100, help="FLM nuisance epochs")
    ap.add_argument("--flm-repeats", type=int, default=1,
                    help="FLM repeated cross-fitting splits (n_repeats); >1 = median DML")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes over MC reps (torch pinned to 2 threads each)")
    ap.add_argument("--dgp", choices=["both", "all", "linear", "logit", "poisson"], default="both",
                    help="which DGP(s): both=linear+logit, all=+poisson, or a single name")
    ap.add_argument("--lambda-method", default="ridge",
                    choices=["ridge", "lgbm", "rf", "mlp", "aggregate"],
                    help="FLM Lambda estimator (linear DGP only)")
    ap.add_argument("--ridge-alpha", type=float, default=1000.0,
                    help="ridge_alpha for lambda_method=ridge")
    ap.add_argument("--three-way", action="store_true",
                    help="force 3-way split so Lambda(x) is estimated via "
                         "lambda_method (default 2-way uses flat aggregate Lambda)")
    ap.add_argument("--logit-lambdas", default="cholesky,analytic,ridge,oracle",
                    help="comma-sep FLM Lambda specs for the LOGIT DGP, one table row each: "
                         "cholesky (general PSD net), analytic (est. propensity + known "
                         "logit form), ridge (default contrast), oracle (true Lambda ceiling)")
    ap.add_argument("--linear-lambdas", default="cholesky,flat,oracle",
                    help="comma-sep FLM Lambda specs for the LINEAR DGP: cholesky (general "
                         "PSD net), flat (legacy 2-way aggregate bug contrast), oracle "
                         "(true Lambda ceiling). All but 'flat' run the general inference() path.")
    ap.add_argument("--poisson-lambdas", default="cholesky,ridge,oracle",
                    help="comma-sep FLM Lambda specs for the POISSON DGP: cholesky (general),"
                         " ridge (contrast), oracle (true Lambda ceiling).")
    ap.add_argument("--tikhonov", type=float, default=None,
                    help="global tikhonov_scale override applied to ALL FLM specs (matched-eps "
                         "oracle-ladder diagnostic). Default None = each spec's own default.")
    ap.add_argument("--max-condition", type=float, default=None,
                    help="cholesky inverse condition-number clamp (spectrum-adaptive regularizer). "
                         "Default None = package default 100. Lower = more inverse regularization.")
    args = ap.parse_args()
    specs_map = {
        "linear": [s for s in args.linear_lambdas.split(",") if s],
        "logit": [s for s in args.logit_lambdas.split(",") if s],
        "poisson": [s for s in args.poisson_lambdas.split(",") if s],
    }

    if args.smoke:
        M, n, flm_folds, flm_epochs, riesz_epochs, riesz_patience = 3, 1000, 5, 60, 150, 20
    else:
        M, n, flm_folds, flm_epochs, riesz_epochs, riesz_patience = (
            args.M, args.n, args.flm_folds, args.flm_epochs, 200, 25)

    sel = {"both": ["linear", "logit"],
           "all": ["linear", "logit", "poisson"]}.get(args.dgp, [args.dgp])
    truths = {name: TRUTH_FN[name](np.random.default_rng(99)) for name in sel}
    print("truths: " + "  ".join(f"{k} ATE={v:.4f}" for k, v in truths.items()))

    def do(name):
        print(f"\n[{name.upper()}]  (workers={args.workers})", flush=True)
        specs = specs_map[name]
        acc, rec_acc = run(truths[name], name, M, n, flm_folds, flm_epochs,
                  riesz_epochs, riesz_patience, base_seed=DGP_BASE_SEED[name],
                  workers=args.workers, flm_repeats=args.flm_repeats,
                  flm_tikhonov=args.tikhonov, flm_max_condition=args.max_condition, specs=specs)
        tik_desc = f", tikhonov={args.tikhonov:g}" if args.tikhonov is not None else ""
        cond_desc = f", maxcond={args.max_condition:g}" if args.max_condition is not None else ""
        rep_desc = f", repeats={args.flm_repeats}" if args.flm_repeats > 1 else ""
        s = summarize(f"{name.capitalize()} DGP (lambdas={','.join(specs)}, n={n}, "
                      f"folds={flm_folds}{rep_desc}{tik_desc}{cond_desc})", truths[name], acc, rec_acc, M)
        print(s, flush=True)  # print each summary AS its DGP finishes (partial-safe)
        return s

    summaries = [do(name) for name in sel]

    report = ("# Spike: FLM vs RieszNet (known-truth ATE)\n"
              + "\n".join(summaries)
              + "\n\nPass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; "
                "Naive should under-cover (shows the correction is necessary).\n")
    with open("exploration/results.md", "w") as f:
        f.write(report)
    print("wrote exploration/results.md")


if __name__ == "__main__":
    main()
