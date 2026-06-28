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


def _loss(net, t, x, y, ones, zeros, logit):
    g_raw, a_obs = net(t, x)
    g_obs = torch.sigmoid(g_raw) if logit else g_raw
    reg = nn.functional.binary_cross_entropy_with_logits(g_raw, y) if logit \
        else ((g_raw - y) ** 2).mean()
    _, a1 = net(ones, x)
    _, a0 = net(zeros, x)
    riesz = (a_obs ** 2 - 2.0 * (a1 - a0)).mean()          # E[a^2 - 2 m(a)]
    g_tilde = g_obs + net.eps * a_obs
    tmle = ((y - g_tilde) ** 2).mean()                     # targeted reg (lambda2)
    return reg + LAMBDA1 * riesz + LAMBDA2 * tmle


def _fit_riesz(Xtr, Ttr, Ytr, logit, max_epochs, patience, seed, restarts=2):
    d_x = Xtr.shape[1]

    # train/val split within the fold for early stopping (fixed across restarts)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Ytr))
    n_val = max(1, int(0.2 * len(Ytr)))
    val, tr = idx[:n_val], idx[n_val:]

    def pack(ii):
        x = torch.tensor(Xtr[ii], dtype=torch.float32)
        t = torch.tensor(Ttr[ii], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(Ytr[ii], dtype=torch.float32)
        return t, x, y, torch.ones_like(t), torch.zeros_like(t)

    tr_b, val_b = pack(tr), pack(val)

    # ponytail: grad-clip + best-of-restarts guard a known full-batch-Adam
    # failure mode (a rep occasionally diverges to a junk Riesz head, e.g.
    # est=-0.289). Keep the restart with the lowest val loss. The real upgrade
    # is the paper's two-stage minibatch LR schedule; restarts are the cheap one.
    net = RieszNet(d_x)
    global_best, global_state = float("inf"), None
    for r in range(restarts):
        torch.manual_seed(seed + 1000 * r)
        net = RieszNet(d_x)
        # L2 on net weights only, NOT on eps (paper: R does not take eps as input)
        opt = torch.optim.Adam([
            {"params": [p for n, p in net.named_parameters() if n != "eps"], "weight_decay": L2},
            {"params": [net.eps], "weight_decay": 0.0},
        ], lr=1e-3)
        best, best_state, wait = float("inf"), None, 0
        for _ in range(max_epochs):
            net.train(); opt.zero_grad()
            _loss(net, *tr_b, logit).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            net.eval()
            with torch.no_grad():
                v = _loss(net, *val_b, logit).item()
            if v < best - 1e-5:
                best, best_state, wait = v, {k: p.clone() for k, p in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= patience:
                    break
        if best_state is not None and best < global_best:
            global_best, global_state = best, best_state
    if global_state is not None:
        net.load_state_dict(global_state)
    net.eval()
    return net


def riesz_ate(Y, T, X, logit, K=5, max_epochs=400, patience=30, seed=0):
    """Cross-fit faithful RieszNet; returns (mu, se, mu_naive, se_naive)."""
    n = len(Y)
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), K)
    psi = np.zeros(n)
    g_diff = np.zeros(n)
    for k in range(K):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(K) if j != k])
        net = _fit_riesz(X[tr], T[tr], Y[tr], logit, max_epochs, patience, seed + k)
        with torch.no_grad():
            xe = torch.tensor(X[te], dtype=torch.float32)
            te_t = torch.tensor(T[te], dtype=torch.float32).unsqueeze(1)
            ones, zeros = torch.ones_like(te_t), torch.zeros_like(te_t)
            eps = net.eps.item()

            def gmean(traw):
                g, a = net(traw, xe)
                g = torch.sigmoid(g) if logit else g
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
    return mu, se, mu_naive, se_naive


# ---- FLM via the package --------------------------------------------------
def flm_linear(Y, T, X, n_folds, epochs, n_repeats=1):
    r = structural_dml(Y, T, X.astype(float), family="linear",
                       n_folds=n_folds, epochs=epochs, n_repeats=n_repeats,
                       hidden_dims=[32], verbose=False)
    return r.mu_hat, r.se


def flm_logit(Y, T, X, n_folds, epochs, n_repeats=1):
    # discrete ATE on probability scale: g(1,X)-g(0,X) = sigmoid(a+b)-sigmoid(a)
    def ate_target(x, theta, t_tilde):
        return torch.sigmoid(theta[0] + theta[1]) - torch.sigmoid(theta[0])
    r = inference(Y, T, X.astype(float), model="logit", target_fn=ate_target,
                  t_tilde=0.0, n_folds=n_folds, epochs=epochs, n_repeats=n_repeats,
                  hidden_dims=[32], verbose=False)
    return r.mu_hat, r.se


# ---- Monte Carlo driver ---------------------------------------------------
def covered(mu, se, truth):
    z = norm.ppf(0.975)
    return float(mu - z * se <= truth <= mu + z * se)


def _one_rep(task):
    """One MC replication, all four methods. Top-level so it pickles for Pool."""
    (truth, logit, n, flm_folds, flm_epochs, flm_repeats,
     riesz_epochs, riesz_patience, seed) = task
    # Seed the GLOBAL torch/numpy RNG per rep so FLM (which uses global state for
    # net init) is reproducible and identical serial-vs-parallel. Distinct per
    # rep, so cross-rep MC variability is preserved.
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    dgp = gen_logit if logit else gen_linear
    flm_fn = flm_logit if logit else flm_linear
    Y, T, X = dgp(n, rng)
    out = {}
    mu, se = (oracle_logit if logit else oracle_linear)(Y, T, X)
    out["Oracle"] = (mu, se, covered(mu, se, truth))
    mu, se = flm_fn(Y, T, X, flm_folds, flm_epochs, n_repeats=flm_repeats)
    out["FLM"] = (mu, se, covered(mu, se, truth))
    mu, se, mun, sen = riesz_ate(Y, T, X, logit, max_epochs=riesz_epochs,
                                 patience=riesz_patience, seed=seed)
    out["RieszNet"] = (mu, se, covered(mu, se, truth))
    out["Naive"] = (mun, sen, covered(mun, sen, truth))
    return out


def run(dgp, truth, flm_fn, logit, M, n, flm_folds, flm_epochs,
        riesz_epochs, riesz_patience, base_seed, workers=1, flm_repeats=1):
    acc = {m: {"est": [], "se": [], "cov": []} for m in
           ["Oracle", "FLM", "RieszNet", "Naive"]}
    tasks = [(truth, logit, n, flm_folds, flm_epochs, flm_repeats, riesz_epochs,
              riesz_patience, base_seed + i) for i in range(M)]

    def absorb(i, out):
        for m, (est, se, cov) in out.items():
            acc[m]["est"].append(est); acc[m]["se"].append(se); acc[m]["cov"].append(cov)
        print(f"  rep {i+1}/{M}: oracle={out['Oracle'][0]:.3f} flm={out['FLM'][0]:.3f} "
              f"riesz={out['RieszNet'][0]:.3f} naive={out['Naive'][0]:.3f}", flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks):
            absorb(i, _one_rep(t))
    else:
        # imap keeps results in submission order so seeds map to reps deterministically
        with Pool(processes=workers) as pool:
            for i, out in enumerate(pool.imap(_one_rep, tasks)):
                absorb(i, out)
    return acc


def summarize(name, truth, acc, M):
    lines = [f"\n### {name}  (truth = {truth:.4f}, M = {M})\n",
             "| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |",
             "|---|---|---|---|---|---|---|"]
    for m, d in acc.items():
        est = np.array(d["est"]); se = np.array(d["se"]); cov = np.array(d["cov"])
        emp = est.std(ddof=1)
        lines.append(f"| {m} | {est.mean():.4f} | {est.mean()-truth:+.4f} | "
                     f"{emp:.4f} | {se.mean():.4f} | {se.mean()/emp:.2f} | "
                     f"{100*cov.mean():.0f}% |")
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
    ap.add_argument("--dgp", choices=["both", "linear", "logit"], default="both",
                    help="which DGP(s) to run (logit/Regime-C is far heavier at high folds)")
    args = ap.parse_args()

    if args.smoke:
        M, n, flm_folds, flm_epochs, riesz_epochs, riesz_patience = 3, 1000, 5, 60, 150, 20
    else:
        M, n, flm_folds, flm_epochs, riesz_epochs, riesz_patience = (
            args.M, args.n, args.flm_folds, args.flm_epochs, 400, 30)

    truth_rng = np.random.default_rng(99)
    t_lin = truth_linear()
    t_log = truth_logit(truth_rng)
    print(f"truths: linear ATE={t_lin:.4f}  logit ATE={t_log:.4f}")

    def do(name, gen, truth, flm_fn, logit, base_seed):
        print(f"\n[{name.upper()}]  (workers={args.workers})", flush=True)
        acc = run(gen, truth, flm_fn, logit, M, n, flm_folds, flm_epochs,
                  riesz_epochs, riesz_patience, base_seed=base_seed,
                  workers=args.workers, flm_repeats=args.flm_repeats)
        s = summarize(f"{name.capitalize()} DGP", truth, acc, M)
        print(s, flush=True)  # print each summary AS its DGP finishes (partial-safe)
        return s

    summaries = []
    if args.dgp in ("both", "linear"):
        summaries.append(do("linear", gen_linear, t_lin, flm_linear, False, 1000))
    if args.dgp in ("both", "logit"):
        summaries.append(do("logit", gen_logit, t_log, flm_logit, True, 5000))

    report = ("# Spike: FLM vs RieszNet (known-truth ATE)\n"
              + "\n".join(summaries)
              + "\n\nPass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; "
                "Naive should under-cover (shows the correction is necessary).\n")
    with open("exploration/results.md", "w") as f:
        f.write(report)
    print("wrote exploration/results.md")


if __name__ == "__main__":
    main()
