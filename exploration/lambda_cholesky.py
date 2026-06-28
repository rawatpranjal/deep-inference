"""
PSD-by-construction general Lambda(x) regression.

The surface viz showed the general obstacle: regressing the Hessian entries
independently breaks PSD (det<0) in the tails, so the inverse detonates, and
post-hoc eigenvalue clamping does not rescue it. This enforces PSD DURING the
regression. A small net maps X -> the Cholesky factor L(x) (lower-triangular,
positive diagonal); the prediction is Lambda_hat(x) = L(x) L(x)^T, which is PSD
for ANY net output. It is trained to fit the per-observation Hessians H_i in
Frobenius norm. Fully general: any family (autodiff the Hessian), any theta_dim.

Injected through the real pipeline via the lambda_eval_fn hook (cross-fit per
fold). Compared against the flat bug, the structure ceiling (linear-only), and
the oracle.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/lambda_cholesky.py --M 50 --workers 8
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
import torch.nn as nn
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import gen_linear                                            # noqa: E402
from lambda_decomp import (fit_panel, oracle_lambda_fn, PANEL_KEYS, ORACLE_TIKHONOV)  # noqa: E402
from deep_inference import structural_dml                              # noqa: E402


class CholNet(nn.Module):
    """X -> Cholesky factor L(x) of a (d,d) PSD matrix; predict L L^T."""
    def __init__(self, d_x, d=2, hidden=32):
        super().__init__()
        self.d = d
        self.n_low = d * (d + 1) // 2
        self.net = nn.Sequential(nn.Linear(d_x, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.n_low))
        # indices of lower-triangle; diagonal positions get softplus
        self.tri = torch.tril_indices(d, d)
        self.diag_mask = (self.tri[0] == self.tri[1])

    def forward(self, X):
        p = self.net(X)                                  # (n, n_low)
        vals = p.clone()
        vals[:, self.diag_mask] = nn.functional.softplus(p[:, self.diag_mask]) + 1e-3
        L = X.new_zeros(len(X), self.d, self.d)
        L[:, self.tri[0], self.tri[1]] = vals
        return L @ L.transpose(-1, -2)                   # (n,d,d), PSD by construction


def make_cholesky_lambda(hidden=32, epochs=250, lr=5e-3):
    def fn(X_eval, X_lambda, T_lambda):
        Xl = X_lambda.detach().float(); Tl = T_lambda.detach().float()
        # general per-obs Hessian (linear loss): H = 2[[1,T],[T,T^2]]
        H = torch.zeros(len(Tl), 2, 2)
        H[:, 0, 0] = 2.0; H[:, 0, 1] = 2 * Tl; H[:, 1, 0] = 2 * Tl; H[:, 1, 1] = 2 * Tl ** 2
        net = CholNet(Xl.shape[1], d=2, hidden=hidden)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((net(Xl) - H) ** 2).mean()           # Frobenius fit to per-obs Hessians
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return net(X_eval.float())
    return fn


def run_flm_cell(Y, T, X, lam, folds, epochs):
    kw = dict(family="linear", n_folds=folds, epochs=epochs, n_repeats=1,
              hidden_dims=[32], verbose=False)
    tik = 0.01
    if lam == "flat":
        kw["three_way"] = False
    elif lam == "oracle":
        kw["lambda_eval_fn"] = oracle_lambda_fn; kw["tikhonov_scale"] = tik = ORACLE_TIKHONOV
    elif lam == "cholesky":
        kw["lambda_eval_fn"] = make_cholesky_lambda(); kw["tikhonov_scale"] = tik = ORACLE_TIKHONOV
    elif lam == "lgbm":
        kw.update(three_way=True, lambda_method="lgbm")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lams", default="flat,lgbm,cholesky,oracle")
    ap.add_argument("--out", default="exploration/results_lambda_cholesky.md")
    args = ap.parse_args()
    lams = args.lams.split(",")
    print(f"PSD-by-construction Lambda. M={args.M} n={args.n} folds={args.folds} cells={lams}", flush=True)
    rows = []
    for lam in lams:
        r = run_cell(lam, args.M, args.n, args.folds, args.epochs, args.workers)
        r["cell"] = lam; rows.append(r)
        if r.get("n_ok"):
            print(f"  {lam:10s} SEratio={r['se_ratio']:.3f} cov={100*r['coverage']:.0f}% "
                  f"var_ratio={r['var_ratio']:.3f} Linv_R2={r['laminv_r2p']:.3f} "
                  f"L_R2={r['lam_r2p']:.3f} bias={r['bias']:+.4f} fail={r['nfail']}", flush=True)
        else:
            print(f"  {lam:10s} ALL {r['nfail']} failed: {r.get('fail_eg','')}", flush=True)
    lines = ["# PSD-by-construction (Cholesky) general Lambda(x)\n",
             f"M={args.M} n={args.n} folds={args.folds}. cholesky = net outputs L(x), Lambda=LL^T, "
             "trained Frobenius to per-obs Hessians (general). vs flat bug, lgbm (general regress+invert), "
             "oracle.\n",
             "| cell | SE-ratio | cov | bias | var_ratio | Linv-R2 | L-R2 |", "|" + "---|" * 7]
    for r in rows:
        if not r.get("n_ok"):
            lines.append(f"| {r['cell']} | all {r['nfail']} failed |" + " |" * 6); continue
        lines.append(f"| {r['cell']} | {r['se_ratio']:.3f} | {100*r['coverage']:.0f}% | {r['bias']:+.4f} | "
                     f"{r['var_ratio']:.3f} | {r['laminv_r2p']:.3f} | {r['lam_r2p']:.3f} |")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
