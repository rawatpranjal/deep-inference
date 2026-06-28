"""
Fix study for the FLM linear SE-undercount: a STABLE Lambda(x) that doesn't
blow up under inversion.

Diagnosis (see lambda_decomp.py / CLAUDE.md): regressing the per-obs Hessian
entries independently and inverting is unstable because Lambda(x) is near-singular
at low overlap. Fix idea: estimate the scalar propensity e(x)=E[T|X] DIRECTLY,
then form Lambda(x)=2[[1,e],[e,e]] analytically -- guaranteed PSD, det=4e(1-e)>0,
exactly invertible. We compare propensity estimators (logit-ridge / LPM-OLS /
HistGB / MLP / oracle) against the regress-and-invert baselines (ridge@100, rf),
and ask whether more data closes the gap.

Each propensity-Lambda cross-fits e(x) per fold via the widened lambda_eval_fn
hook (X_eval, X_lambda, T_lambda) -> Lambda(X_eval), so it runs in the REAL
pipeline with correct cross-fitting.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/lambda_inv_fix.py --M 50 --workers 8
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
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import gen_linear                                   # noqa: E402
from lambda_decomp import (fit_panel, oracle_lambda_fn, oracle_theta_factory,  # noqa: E402
                           PANEL_KEYS, ORACLE_TIKHONOV)
from deep_inference import structural_dml                      # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(1)
E_CLIP = 1e-3   # clip e(x) away from 0/1 for a finite (honest) overlap weight


def make_prop_lambda(kind):
    """Closure (built INSIDE the worker, never pickled): cross-fit e(x) on the
    fold's lambda-train (X,T), return the structured Lambda(x)=2[[1,e],[e,e]]."""
    def fn(X_eval, X_lambda, T_lambda):
        Xl = X_lambda.detach().cpu().numpy(); Tl = T_lambda.detach().cpu().numpy().astype(int)
        Xe = X_eval.detach().cpu().numpy()
        if kind == "logit_ridge":
            e = LogisticRegression(C=1.0, max_iter=500).fit(Xl, Tl).predict_proba(Xe)[:, 1]
        elif kind == "lpm_ols":
            e = LinearRegression().fit(Xl, Tl.astype(float)).predict(Xe)
        elif kind == "histgb":
            e = HistGradientBoostingClassifier(max_depth=3, max_iter=120,
                                               min_samples_leaf=40).fit(Xl, Tl).predict_proba(Xe)[:, 1]
        elif kind == "mlp":
            e = MLPClassifier(hidden_layer_sizes=(32,), max_iter=400,
                              alpha=1e-3).fit(Xl, Tl).predict_proba(Xe)[:, 1]
        else:
            raise ValueError(kind)
        e = np.clip(e, E_CLIP, 1 - E_CLIP)
        L = np.zeros((len(e), 2, 2))
        L[:, 0, 0] = 2.0; L[:, 0, 1] = 2 * e; L[:, 1, 0] = 2 * e; L[:, 1, 1] = 2 * e
        return torch.tensor(L, dtype=X_eval.dtype)
    return fn


def run_flm_cell(Y, T, X, lam, theta, folds, epochs):
    kw = dict(family="linear", n_folds=folds, epochs=epochs, n_repeats=1,
              hidden_dims=[32], verbose=False)
    tik = 0.01
    if lam == "flat":
        kw["three_way"] = False
    elif lam == "oracle":
        kw["lambda_eval_fn"] = oracle_lambda_fn; kw["tikhonov_scale"] = tik = ORACLE_TIKHONOV
    elif lam.startswith("prop:"):
        # structured Lambda is well-conditioned -> minimal Tikhonov, rely on the e-clip
        kw["lambda_eval_fn"] = make_prop_lambda(lam.split(":")[1]); kw["tikhonov_scale"] = tik = ORACLE_TIKHONOV
    elif lam in ("rf", "lgbm", "mlp"):
        kw.update(three_way=True, lambda_method=lam)   # general regress-Hessian-entries path
    elif lam.startswith("ridge@"):
        kw.update(three_way=True, lambda_method="ridge", ridge_alpha=float(lam.split("@")[1]))
    else:
        raise ValueError(lam)
    if theta == "oracle":
        kw["network_factory"] = oracle_theta_factory
    r = structural_dml(Y, T, X.astype(float), **kw)
    return r.mu_hat, r.se, fit_panel(r, X, T, Y, tik)


def _one(task):
    cell, seed, folds, epochs = task
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    Y, T, X = gen_linear(cell["n"], rng)
    try:
        mu, se, panel = run_flm_cell(Y, T, X, cell["lam"], cell["theta"], folds, epochs)
    except Exception as ex:
        return {"fail": str(ex)[:90]}
    z = norm.ppf(0.975)
    return {"mu": mu, "se": se, "cov": float(mu - z * se <= 1.0 <= mu + z * se), **panel}


def run_cell(cell, M, folds, epochs, base_seed, workers):
    tasks = [(cell, base_seed + i, folds, epochs) for i in range(M)]
    if workers <= 1:
        outs = [_one(t) for t in tasks]
    else:
        with Pool(processes=workers) as pool:
            outs = list(pool.imap(_one, tasks))
    ok = [o for o in outs if "mu" in o]
    nfail = len(outs) - len(ok)
    if not ok:
        return {"n_ok": 0, "nfail": nfail, "fail_eg": outs[0].get("fail", "")}
    est = np.array([o["mu"] for o in ok]); se = np.array([o["se"] for o in ok])
    cov = np.array([o["cov"] for o in ok]); emp = est.std(ddof=1)
    row = {"n_ok": len(ok), "nfail": nfail, "bias": est.mean() - 1.0, "emp_sd": emp,
           "mean_se": se.mean(), "se_ratio": se.mean() / emp, "coverage": cov.mean()}
    for k in PANEL_KEYS:
        row[k] = float(np.nanmean([o[k] for o in ok]))
    return row


# estimator comparison @ n=2000, theta=net ; then data-scaling on logit_ridge
CELLS = (
    [{"lam": l, "theta": "net", "n": 2000} for l in
     ["flat", "oracle", "prop:logit_ridge", "prop:lpm_ols", "prop:histgb", "prop:mlp", "ridge@100", "rf"]]
    + [{"lam": "prop:logit_ridge", "theta": "net", "n": nn} for nn in [1000, 4000, 8000]]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-sep label substrings to keep")
    ap.add_argument("--out", default="exploration/results_lambda_inv_fix.md")
    args = ap.parse_args()
    cells = CELLS
    if args.only:
        subs = args.only.split(",")
        cells = [c for c in CELLS if any(s in f"{c['lam']}/th={c['theta']}/n={c['n']}" for s in subs)]
    print(f"M={args.M} folds={args.folds} epochs={args.epochs} cells={len(cells)}", flush=True)
    rows = []
    for c in cells:
        label = f"{c['lam']}/th={c['theta']}/n={c['n']}"
        r = run_cell(c, args.M, args.folds, args.epochs, base_seed=1000, workers=args.workers)
        r["cell"] = label
        rows.append(r)
        if r.get("n_ok"):
            print(f"  {label:34s} SEratio={r['se_ratio']:.3f} cov={100*r['coverage']:.0f}% "
                  f"var_ratio={r['var_ratio']:.3f} Linv_R2={r['laminv_r2p']:.3f} "
                  f"psi_R2={r['psi_r2']:.3f} bias={r['bias']:+.4f} fail={r['nfail']}", flush=True)
        else:
            print(f"  {label:34s} ALL {r['nfail']} failed: {r.get('fail_eg','')}", flush=True)
    write_report(args, rows)


def write_report(args, rows):
    hdr = ("| cell | SE-ratio | cov | bias | Var(psi_h)/Var(psi*) | Linv-R2 | psi-R2 | "
           "L-R2 | beta-R2 | fails |")
    lines = ["# FLM linear: stable-Lambda fix study\n",
             f"M={args.M} folds={args.folds} epochs={args.epochs}. truth=1.0. "
             "prop:* = estimate e(x) directly then form Lambda=2[[1,e],[e,e]] analytically "
             "(structured, exactly invertible). ridge@/rf = regress Hessian entries + invert "
             "(the unstable baseline). oracle = true e(x).\n", hdr, "|" + "---|" * 10]
    for r in rows:
        if not r.get("n_ok"):
            lines.append(f"| {r['cell']} | all {r['nfail']} failed |" + " |" * 9); continue
        lines.append(
            f"| {r['cell']} | {r['se_ratio']:.3f} | {100*r['coverage']:.0f}% | {r['bias']:+.4f} | "
            f"{r['var_ratio']:.3f} | {r['laminv_r2p']:.3f} | {r['psi_r2']:.3f} | {r['lam_r2p']:.3f} | "
            f"{r['beta_r2']:.3f} | {r['nfail']} |")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
