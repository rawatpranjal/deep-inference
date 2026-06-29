"""Sweep the Tikhonov inverse-regularization for cholesky Λ̂ on the LOGIT DGP.

The probe showed the true logit Λ(x) is near-singular (max||Λ⁻¹||~6e3), so a noisy Λ̂
inverts explosively under near-zero tikhonov -> the cholesky estimator is unbiased but
high-variance. tikhonov_scale ε in (Λ+εI)⁻¹ regularizes that inverse. Find the ε that
stabilizes cholesky without over-shrinking the correction (which would re-introduce
under-coverage like ridge). No RieszNet here -- just the cholesky row, many seeds, fast.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/sweep_chol_tikhonov.py --M 24 --workers 8
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spike import gen_logit, flm_logit, truth_logit, covered  # noqa: E402

EPS_GRID = [1e-8, 1e-3, 1e-2, 5e-2, 1e-1, 3e-1]


def _one(task):
    eps, seed, n, folds, epochs = task
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    Y, T, X = gen_logit(n, rng)
    try:
        mu, se = flm_logit(Y, T, X, folds, epochs, lambda_spec="cholesky", tikhonov=eps)
    except Exception as ex:
        return {"fail": str(ex)[:80]}
    return {"eps": eps, "mu": mu, "se": se}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    truth = truth_logit(np.random.default_rng(99))
    print(f"logit truth ATE = {truth:.4f}   M={args.M} n={args.n} folds={args.folds}", flush=True)
    tasks = [(eps, 5000 + i, args.n, args.folds, args.epochs)
             for eps in EPS_GRID for i in range(args.M)]
    with Pool(args.workers) as pool:
        outs = list(pool.imap(_one, tasks))

    print(f"\n{'tikhonov':>10s} {'mean est':>9s} {'bias':>9s} {'emp SE':>8s} "
          f"{'mean SE':>8s} {'SE ratio':>9s} {'coverage':>9s} {'fails':>6s}")
    for eps in EPS_GRID:
        rows = [o for o in outs if o.get("eps") == eps and "mu" in o]
        nfail = sum(1 for o in outs if o.get("eps") == eps) - len(rows) \
            if any("eps" in o for o in outs) else 0
        nfail = args.M - len(rows)
        if not rows:
            print(f"{eps:10.0e}  all {args.M} failed"); continue
        est = np.array([o["mu"] for o in rows]); se = np.array([o["se"] for o in rows])
        cov = np.array([covered(o["mu"], o["se"], truth) for o in rows])
        emp = est.std(ddof=1)
        print(f"{eps:10.0e} {est.mean():9.4f} {est.mean()-truth:+9.4f} {emp:8.4f} "
              f"{se.mean():8.4f} {se.mean()/emp:9.2f} {100*cov.mean():8.0f}% {nfail:6d}")


if __name__ == "__main__":
    main()
