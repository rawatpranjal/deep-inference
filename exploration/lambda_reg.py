"""
Regularization frontier for the general Lambda(x) regress-and-invert path.

Two questions:
  (1) lgbm with vs without regularization -- does reg strength trade Lambda FIT
      against Lambda CONDITIONING (and does any setting hit the target)?
  (2) can we beat the flattened-ridge baseline? Two attempts: condition-aware
      inversion (clamp the eigenvalue ratio of the estimated Lambda), and
      structure-imposition (regress the single scalar e(x), the linear ceiling).

The Lambda regressors run through a lambda_eval_fn closure that reconstructs the
linear per-obs Hessian from (X,T) on the fold's lambda-train -- h = 2[[1,T],[T,T^2]]
-- regresses the entries on X, and returns Lambda(X_eval). This IS the general
entry-wise mechanism (the package would autodiff the same Hessian), computed
directly. The package's own lambda_method='ridge'/'lgbm' cells are run too, as a
faithfulness check that the closure matches.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/lambda_reg.py --M 50 --workers 8
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import argparse
from multiprocessing import Pool
import numpy as np
import torch
from scipy.stats import norm
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import gen_linear                                            # noqa: E402
from lambda_decomp import (fit_panel, oracle_lambda_fn, PANEL_KEYS, ORACLE_TIKHONOV)  # noqa: E402
from deep_inference import structural_dml                              # noqa: E402

E_CLIP = 1e-3
LGBM_HEAVY = dict(n_estimators=20, max_depth=2, learning_rate=0.05, min_child_samples=150,
                  reg_alpha=5.0, reg_lambda=5.0, random_state=42, verbose=-1)
LGBM_LIGHT = dict(n_estimators=300, num_leaves=63, learning_rate=0.1, min_child_samples=5,
                  reg_alpha=0.0, reg_lambda=0.0, random_state=42, verbose=-1)
LGBM_NONE = dict(n_estimators=500, num_leaves=255, learning_rate=0.2, min_child_samples=1,
                 reg_alpha=0.0, reg_lambda=0.0, random_state=42, verbose=-1)


def _clamp_cond(L, max_cond):
    """Condition-aware regularization: clamp each matrix's min eigenvalue to
    max_eig/max_cond, bounding the condition number before inversion."""
    w, V = np.linalg.eigh(L)                  # (n,2),(n,2,2)
    floor = w[:, -1:] / max_cond
    w = np.maximum(w, floor)
    return (V * w[:, None, :]) @ V.transpose(0, 2, 1)


def make_hess_lambda(kind, clamp=None, structure=False):
    """Closure built INSIDE the worker. Regress the linear Hessian entries (or the
    scalar e) on X_lambda, return Lambda(X_eval)."""
    def fn(X_eval, X_lambda, T_lambda, *_):
        Xl = X_lambda.detach().cpu().numpy(); Tl = T_lambda.detach().cpu().numpy()
        Xe = X_eval.detach().cpu().numpy()
        if structure:                          # impose Lambda=2[[1,e],[e,e]] from one regression
            e = np.clip(LogisticRegression(C=1.0, max_iter=500).fit(Xl, Tl.astype(int))
                        .predict_proba(Xe)[:, 1], E_CLIP, 1 - E_CLIP)
            h01 = h11 = 2 * e
        else:                                  # general: regress h01=2T and h11=2T^2 independently
            params = {"heavy": LGBM_HEAVY, "light": LGBM_LIGHT, "none": LGBM_NONE}[kind]
            reg = MultiOutputRegressor(LGBMRegressor(**params))
            reg.fit(Xl, np.column_stack([2 * Tl, 2 * Tl ** 2]))
            pred = reg.predict(Xe); h01, h11 = pred[:, 0], pred[:, 1]
        n = len(Xe)
        L = np.zeros((n, 2, 2)); L[:, 0, 0] = 2.0
        L[:, 0, 1] = h01; L[:, 1, 0] = h01; L[:, 1, 1] = h11
        if clamp is not None:
            L = _clamp_cond(L, clamp)
        return torch.tensor(L, dtype=X_eval.dtype)
    return fn


def run_flm_cell(Y, T, X, lam, folds, epochs):
    kw = dict(family="linear", n_folds=folds, epochs=epochs, n_repeats=1,
              hidden_dims=[32], verbose=False)
    tik = 0.01
    if lam == "flat":
        kw["three_way"] = False
    elif lam == "oracle":
        kw["lambda_eval_fn"] = oracle_lambda_fn; kw["tikhonov_scale"] = tik = ORACLE_TIKHONOV
    elif lam.startswith("pkg_ridge@"):                 # package path (faithfulness ref)
        kw.update(three_way=True, lambda_method="ridge", ridge_alpha=float(lam.split("@")[1]))
    elif lam == "pkg_lgbm":
        kw.update(three_way=True, lambda_method="lgbm")
    else:                                               # closure variants
        kw["tikhonov_scale"] = tik = ORACLE_TIKHONOV
        if lam.startswith("lgbm_"):
            kind = lam.split("_")[1]
            clamp = 1e3 if lam.endswith("_clamp") else None
            kw["lambda_eval_fn"] = make_hess_lambda(kind, clamp=clamp)
        elif lam == "struct_logit":
            kw["lambda_eval_fn"] = make_hess_lambda(None, structure=True)
        else:
            raise ValueError(lam)
    r = structural_dml(Y, T, X.astype(float), **kw)
    return r.mu_hat, r.se, fit_panel(r, X, T, Y, tik)


def _one(task):
    lam, seed, n, folds, epochs = task
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    Y, T, X = gen_linear(n, rng)
    try:
        mu, se, panel = run_flm_cell(Y, T, X, lam, folds, epochs)
    except Exception as ex:
        return {"fail": str(ex)[:90]}
    z = norm.ppf(0.975)
    return {"mu": mu, "se": se, "cov": float(mu - z * se <= 1.0 <= mu + z * se), **panel}


def run_cell(lam, M, n, folds, epochs, workers):
    tasks = [(lam, 1000 + i, n, folds, epochs) for i in range(M)]
    with Pool(processes=workers) as pool:
        outs = list(pool.imap(_one, tasks))
    ok = [o for o in outs if "mu" in o]
    if not ok:
        return {"n_ok": 0, "nfail": len(outs), "fail_eg": outs[0].get("fail", "")}
    est = np.array([o["mu"] for o in ok]); se = np.array([o["se"] for o in ok])
    cov = np.array([o["cov"] for o in ok]); emp = est.std(ddof=1)
    row = {"n_ok": len(ok), "nfail": len(outs) - len(ok), "bias": est.mean() - 1.0,
           "se_ratio": se.mean() / emp, "coverage": cov.mean()}
    for k in PANEL_KEYS:
        row[k] = float(np.nanmean([o[k] for o in ok]))
    return row


LAMS = ["flat", "oracle",
        "pkg_ridge@100", "pkg_ridge@1000",            # ridge frontier (package)
        "pkg_lgbm", "lgbm_heavy", "lgbm_light", "lgbm_none",   # lgbm reg sweep
        "lgbm_light_clamp", "struct_logit"]           # the two "beat ridge" attempts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="exploration/results_lambda_reg.md")
    args = ap.parse_args()
    print(f"reg frontier. M={args.M} n={args.n} folds={args.folds} cells={len(LAMS)}", flush=True)
    rows = []
    for lam in LAMS:
        r = run_cell(lam, args.M, args.n, args.folds, args.epochs, args.workers)
        r["cell"] = lam; rows.append(r)
        if r.get("n_ok"):
            print(f"  {lam:18s} SEratio={r['se_ratio']:.3f} cov={100*r['coverage']:.0f}% "
                  f"var_ratio={r['var_ratio']:.3f} Linv_R2={r['laminv_r2p']:.3f} "
                  f"L_R2={r['lam_r2p']:.3f} bias={r['bias']:+.4f} fail={r['nfail']}", flush=True)
        else:
            print(f"  {lam:18s} ALL {r['nfail']} failed: {r.get('fail_eg','')}", flush=True)
    hdr = "| cell | SE-ratio | cov | bias | var_ratio | Linv-R2 | L-R2 | fails |"
    lines = ["# Lambda(x) regularization frontier (general path, n=%d)\n" % args.n,
             f"M={args.M} folds={args.folds}. truth=1.0. pkg_* = real package path; lgbm_* and "
             "struct_logit = closure variants. _clamp = condition-aware inversion. "
             "struct_logit imposes the linear structure (the ceiling).\n", hdr, "|" + "---|" * 7]
    for r in rows:
        if not r.get("n_ok"):
            lines.append(f"| {r['cell']} | all {r['nfail']} failed |" + " |" * 6); continue
        lines.append(f"| {r['cell']} | {r['se_ratio']:.3f} | {100*r['coverage']:.0f}% | "
                     f"{r['bias']:+.4f} | {r['var_ratio']:.3f} | {r['laminv_r2p']:.3f} | "
                     f"{r['lam_r2p']:.3f} | {r['nfail']} |")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
