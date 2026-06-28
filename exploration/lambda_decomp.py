"""
Full-observability decomposition of the FLM linear-ATE SE-undercount.

Scores EVERY object in the 7-step FLM influence-function chain against its DGP
oracle (out-of-sample R2 / RMSE / variance ratio), across a grid of Lambda-source
x theta-source. Answers: which entity's imperfect fit tracks the SE-ratio defect,
and does feeding the true Lambda(x) (oracle e(x)) restore SE-ratio -> 1?

Every object reconstructs faithfully from three arrays the real pipeline exposes
(theta_hat, psi_values, lambda_hat) plus X,T,Y -- correction = h - psi exactly --
so we instrument the real estimator, not a reimplementation.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/lambda_decomp.py --sanity
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/lambda_decomp.py --M 30 --workers 8
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import argparse
import warnings
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import spike` works
from spike import A0, A1, A2, B0, B1, GAMMA, SIGMA, gen_linear, truth_linear  # noqa: E402

from deep_inference import structural_dml                       # noqa: E402
from deep_inference.utils.linalg import batch_inverse           # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

ORACLE_TIKHONOV = 1e-8   # near-zero so the oracle anchor is not distorted by Tikhonov


# ---- Oracle Lambda(x) = 2[[1,e],[e,e]], e(x)=sigmoid(X0+X1), binary T --------
def oracle_lambda_fn(X_eval, *_):
    """X_eval: (n,d_x) torch tensor -> (n,2,2) torch tensor. Factor 2 matches
    LinearFamily's loss (y-mu)^2 whose per-obs Hessian is 2[[1,T],[T,T^2]].
    Extra args (X_lambda, T_lambda) ignored -- the oracle needs no fitting."""
    e = torch.sigmoid(GAMMA * (X_eval[:, 0] + X_eval[:, 1]))
    L = torch.zeros(len(e), 2, 2, dtype=X_eval.dtype)
    L[:, 0, 0] = 2.0
    L[:, 0, 1] = 2.0 * e
    L[:, 1, 0] = 2.0 * e
    L[:, 1, 1] = 2.0 * e
    return L


# ---- Oracle theta(x): a network_factory that returns exact (alpha*,beta*) -----
class OracleThetaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))  # inert: keeps Adam's param list non-empty

    def forward(self, x):
        a = A0 + A1 * x[:, 0] + A2 * x[:, 1]
        b = B0 + B1 * x[:, 0]
        out = torch.stack([a, b], dim=1)
        return out + 0.0 * self._dummy   # grad path to _dummy, value unchanged (stays oracle)


def oracle_theta_factory(d_x, theta_dim):
    return OracleThetaNet()


# ---- fit helpers ------------------------------------------------------------
def _r2(hat, star):
    hat = np.asarray(hat, float).ravel(); star = np.asarray(star, float).ravel()
    ss_res = ((hat - star) ** 2).sum()
    ss_tot = ((star - star.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")


def _rmse(hat, star):
    return float(np.sqrt(((np.asarray(hat, float) - np.asarray(star, float)) ** 2).mean()))


def _pooled_r2(H, S):
    """Pooled-Frobenius R2 over a batch of matrices. =1 exactly when H==S; the
    constant [0,0] element contributes 0 to both num and den (no artifact)."""
    H = np.asarray(H, float); S = np.asarray(S, float)
    num = ((H - S) ** 2).sum()
    den = ((S - S.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1 - num / den) if den > 1e-12 else float("nan")


def _frob(H, S):
    return float(np.sqrt(((np.asarray(H, float) - np.asarray(S, float)) ** 2).sum(axis=(1, 2))).mean())


def fit_panel(r, X, T, Y, tikhonov_scale):
    """All 7-step objects scored vs the DGP oracle. r is a DMLResult."""
    n = len(Y)
    # --- estimated objects (reconstructed from the exposed arrays) ---
    a_hat, b_hat = r.theta_hat[:, 0], r.theta_hat[:, 1]
    eps_hat = Y - a_hat - b_hat * T
    score_hat = np.stack([-2 * eps_hat, -2 * eps_hat * T], axis=1)
    lam_hat = r.lambda_hat                                  # (n,2,2)
    laminv_hat = batch_inverse(torch.tensor(lam_hat, dtype=torch.float64),
                               tikhonov_scale=tikhonov_scale).numpy()
    psi_hat = r.psi_values
    corr_hat = b_hat - psi_hat                              # h - psi = correction (exact)

    # --- oracle objects ---
    a_star = A0 + A1 * X[:, 0] + A2 * X[:, 1]
    b_star = B0 + B1 * X[:, 0]
    e_star = 1.0 / (1.0 + np.exp(-GAMMA * (X[:, 0] + X[:, 1])))
    eps_star = Y - a_star - b_star * T
    score_star = np.stack([-2 * eps_star, -2 * eps_star * T], axis=1)
    V = e_star * (1.0 - e_star)
    lam_star = np.zeros((n, 2, 2))
    lam_star[:, 0, 0] = 2.0; lam_star[:, 0, 1] = 2 * e_star
    lam_star[:, 1, 0] = 2 * e_star; lam_star[:, 1, 1] = 2 * e_star
    laminv_star = np.linalg.inv(lam_star)
    corr_star = -eps_star * (T - e_star) / V               # = h - psi* (sign matches corr_hat)
    psi_star = b_star + eps_star * (T - e_star) / V        # efficient IF

    return {
        "alpha_r2": _r2(a_hat, a_star), "beta_r2": _r2(b_hat, b_star),
        "resid_r2": _r2(eps_hat, eps_star), "score_r2": _r2(score_hat, score_star),
        "lam_r2p": _pooled_r2(lam_hat, lam_star), "lam_frob": _frob(lam_hat, lam_star),
        "laminv_r2p": _pooled_r2(laminv_hat, laminv_star), "laminv_frob": _frob(laminv_hat, laminv_star),
        "corr_r2": _r2(corr_hat, corr_star), "psi_r2": _r2(psi_hat, psi_star),
        "psi_corr": float(np.corrcoef(psi_hat, psi_star)[0, 1]),
        "var_ratio": float(np.var(psi_hat) / np.var(psi_star)),   # (SE_reported/SE_efficient)^2
        "beta_rmse": _rmse(b_hat, b_star),
    }


# ---- one FLM cell on one dataset --------------------------------------------
def run_flm_cell(Y, T, X, lambda_source, theta_source, folds, epochs, reps):
    kw = dict(family="linear", n_folds=folds, epochs=epochs, n_repeats=reps,
              hidden_dims=[32], verbose=False)
    tik = 0.01
    if lambda_source == "flat":
        kw["three_way"] = False
    elif lambda_source == "rf":
        kw.update(three_way=True, lambda_method="rf")
    elif lambda_source.startswith("ridge@"):
        kw.update(three_way=True, lambda_method="ridge", ridge_alpha=float(lambda_source.split("@")[1]))
    elif lambda_source == "oracle":
        kw["lambda_eval_fn"] = oracle_lambda_fn
        kw["tikhonov_scale"] = tik = ORACLE_TIKHONOV
    else:
        raise ValueError(lambda_source)
    if theta_source == "oracle":
        kw["network_factory"] = oracle_theta_factory
    r = structural_dml(Y, T, X.astype(float), **kw)
    return r.mu_hat, r.se, fit_panel(r, X, T, Y, tik)


# ---- MC driver ---------------------------------------------------------------
PANEL_KEYS = ["var_ratio", "psi_r2", "psi_corr", "corr_r2", "laminv_r2p", "lam_r2p",
              "lam_frob", "score_r2", "resid_r2", "beta_r2", "alpha_r2"]


def _one(task):
    cell, seed, n, folds, epochs, reps = task
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    Y, T, X = gen_linear(n, rng)
    try:
        mu, se, panel = run_flm_cell(Y, T, X, cell["lam"], cell["theta"], folds, epochs, reps)
    except Exception as ex:                       # detonating cells are data points, not crashes
        return {"fail": str(ex)[:80]}
    z = norm.ppf(0.975)
    cov = float(mu - z * se <= 1.0 <= mu + z * se)
    return {"mu": mu, "se": se, "cov": cov, **panel}


def run_cell(cell, M, n, folds, epochs, reps, base_seed, workers):
    tasks = [(cell, base_seed + i, n, folds, epochs, reps) for i in range(M)]
    outs = []
    if workers <= 1:
        outs = [_one(t) for t in tasks]
    else:
        with Pool(processes=workers) as pool:
            outs = list(pool.imap(_one, tasks))
    ok = [o for o in outs if "mu" in o]
    nfail = len(outs) - len(ok)
    if not ok:
        return {"n_ok": 0, "nfail": nfail}
    est = np.array([o["mu"] for o in ok]); se = np.array([o["se"] for o in ok])
    cov = np.array([o["cov"] for o in ok])
    emp = est.std(ddof=1)
    row = {"n_ok": len(ok), "nfail": nfail, "bias": est.mean() - 1.0,
           "emp_sd": emp, "mean_se": se.mean(), "se_ratio": se.mean() / emp,
           "coverage": cov.mean()}
    for k in PANEL_KEYS:
        row[k] = float(np.nanmean([o[k] for o in ok]))
    return row


CELLS = (
    [{"lam": s, "theta": "net"} for s in
     ["flat", "ridge@1", "ridge@10", "ridge@100", "ridge@1000", "rf", "oracle"]]
    + [{"lam": s, "theta": "oracle"} for s in ["flat", "ridge@1000", "oracle"]]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--M", type=int, default=30)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-sep cell-label substrings to keep")
    ap.add_argument("--out", default="exploration/results_lambda_decomp.md")
    args = ap.parse_args()

    if args.sanity:
        sanity()
        return

    print(f"truth=1.0  M={args.M} n={args.n} folds={args.folds} epochs={args.epochs} "
          f"reps={args.reps}  cells={len(CELLS)}", flush=True)
    rows = []
    for c in CELLS:
        label = f"{c['lam']}/th={c['theta']}"
        r = run_cell(c, args.M, args.n, args.folds, args.epochs, args.reps,
                     base_seed=1000, workers=args.workers)
        r["cell"] = label
        rows.append(r)
        if r.get("n_ok"):
            print(f"  {label:22s} SEratio={r['se_ratio']:.3f} cov={100*r['coverage']:.0f}% "
                  f"var_ratio={r['var_ratio']:.3f} psi_R2={r['psi_r2']:.3f} "
                  f"Linv_R2={r['laminv_r2p']:.3f} L_R2={r['lam_r2p']:.3f} "
                  f"beta_R2={r['beta_r2']:.3f} bias={r['bias']:+.4f} fail={r['nfail']}", flush=True)
        else:
            print(f"  {label:22s} ALL {r['nfail']} reps failed", flush=True)
    write_report(args, rows)


def write_report(args, rows):
    hdr = ("| cell | SE-ratio | cov | bias | Var(psi_h)/Var(psi*) | psi-R2 | psi-corr | "
           "corr-R2 | Linv-R2p | L-R2p | L-frob | score-R2 | resid-R2 | beta-R2 | alpha-R2 | fails |")
    sep = "|" + "---|" * 16
    lines = [f"# FLM linear SE decomposition (full observability)\n",
             f"M={args.M} n={args.n} folds={args.folds} epochs={args.epochs} reps={args.reps}. "
             f"truth mu=1.0. SE-ratio = mean(reported SE)/empirical SD. "
             f"Var ratio = (SE_reported/SE_efficient)^2.\n", hdr, sep]
    for r in rows:
        if not r.get("n_ok"):
            lines.append(f"| {r['cell']} | (all {r['nfail']} reps failed) |" + " |" * 15)
            continue
        lines.append(
            f"| {r['cell']} | {r['se_ratio']:.3f} | {100*r['coverage']:.0f}% | {r['bias']:+.4f} | "
            f"{r['var_ratio']:.3f} | {r['psi_r2']:.3f} | {r['psi_corr']:.3f} | {r['corr_r2']:.3f} | "
            f"{r['laminv_r2p']:.3f} | {r['lam_r2p']:.3f} | {r['lam_frob']:.3f} | {r['score_r2']:.3f} | "
            f"{r['resid_r2']:.3f} | {r['beta_r2']:.3f} | {r['alpha_r2']:.3f} | {r['nfail']} |")
    lines += ["",
              "Reading: oracle/th=net is the decisive cell (does true Lambda(x) restore SE-ratio?). ",
              "Objects whose R2 moves together with SE-ratio/Var-ratio locate the defect; flat-R2 "
              "objects are exonerated. th=oracle rows remove the net's beta-recovery as a confound."]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


def sanity():
    """One draw: assert oracle-theta+oracle-Lambda psi equals the closed-form
    efficient IF, and lambda_hat equals 2[[1,e],[e,e]]."""
    rng = np.random.default_rng(7)
    Y, T, X = gen_linear(3000, rng)
    r = structural_dml(Y, T, X.astype(float), family="linear", n_folds=5, epochs=80,
                       hidden_dims=[32], verbose=False, network_factory=oracle_theta_factory,
                       lambda_eval_fn=oracle_lambda_fn, tikhonov_scale=ORACLE_TIKHONOV)
    e = 1.0 / (1.0 + np.exp(-GAMMA * (X[:, 0] + X[:, 1])))
    a_star = A0 + A1 * X[:, 0] + A2 * X[:, 1]
    b_star = B0 + B1 * X[:, 0]
    eps = Y - a_star - b_star * T
    psi_closed = b_star + eps * (T - e) / (e * (1 - e))
    # lambda_hat == 2[[1,e],[e,e]]
    lam_err = np.abs(r.lambda_hat[:, 0, 1] - 2 * e).max()
    psi_err = np.abs(r.psi_values - psi_closed).max()
    print(f"SANITY  max|lambda_hat[0,1] - 2e| = {lam_err:.2e}  (expect ~0)")
    print(f"SANITY  max|psi_pipeline - psi_closedform| = {psi_err:.2e}")
    print(f"SANITY  theta recovery: beta R2 = {_r2(r.theta_hat[:,1], b_star):.4f} (expect ~1.0, oracle net)")
    se_eff = np.sqrt(np.var(psi_closed) / len(Y))
    print(f"SANITY  reported SE = {r.se:.4f}  vs  sqrt(Var(psi*)/n) = {se_eff:.4f}")
    assert lam_err < 1e-4, "oracle Lambda not injected correctly"
    assert psi_err < 1e-3, "pipeline psi != closed-form efficient IF"
    print("SANITY PASS")


if __name__ == "__main__":
    main()
