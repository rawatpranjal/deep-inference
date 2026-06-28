"""
Focused FLM-linear oracle for the autoresearch loop. Runs ONLY FLM (no RieszNet/
Oracle/Naive) so each Monte-Carlo eval is ~4x faster than the full spike. Exposes
every structural_dml knob so the loop can sweep folds / n_repeats / network /
weight_decay / variance method against the known-truth linear ATE (= 1.0).

Run (on the pod):
  PYTHONPATH=src python3 exploration/flm_debug.py --n 2000 --folds 50 --repeats 5 \
      --M 50 --workers 14 --label foldssweep
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from multiprocessing import Pool
torch.set_num_threads(1)

import spike  # reuse the SAME linear DGP, truth, covered() as the main spike
from deep_inference import structural_dml

TRUTH = 1.0  # E[b(X)] = B0 = 1.0 for the linear DGP


def _recovery(hat, tru):
    """Out-of-fold recovery of a nuisance function: RMSE, R^2, correlation."""
    hat = np.asarray(hat, dtype=np.float64)
    tru = np.asarray(tru, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((hat - tru) ** 2)))
    ss_tot = float(np.sum((tru - tru.mean()) ** 2))
    r2 = float(1.0 - np.sum((hat - tru) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(hat, tru)[0, 1])
    return rmse, r2, corr


def one(task):
    (n, folds, epochs, repeats, hidden, wd, variance, lr, patience, seed) = task
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    Y, T, X = spike.gen_linear(n, rng)
    r = structural_dml(Y, T, X.astype(float), family="linear",
                       n_folds=folds, epochs=epochs, n_repeats=repeats,
                       hidden_dims=hidden, weight_decay=wd, variance=variance,
                       lr=lr, patience=patience, verbose=False)
    # Out-of-fold nuisance recovery: theta_hat[:,0]=alpha(X), [:,1]=beta(X)
    th = np.asarray(r.theta_hat)
    a_rmse, a_r2, a_corr = _recovery(th[:, 0], spike.a_of(X))
    b_rmse, b_r2, b_corr = _recovery(th[:, 1], spike.b_of(X))
    return (r.mu_hat, r.se, spike.covered(r.mu_hat, r.se, TRUTH),
            a_rmse, a_r2, a_corr, b_rmse, b_r2, b_corr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--hidden", default="32", help="comma-separated, e.g. 64,32")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--variance", default="pooled", choices=["pooled", "within_fold"])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=0, help="0 -> package default")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    hidden = [int(x) for x in a.hidden.split(",")]
    patience = a.patience if a.patience and a.patience > 0 else None
    tasks = [(a.n, a.folds, a.epochs, a.repeats, hidden, a.weight_decay,
              a.variance, a.lr, patience, 1000 + i) for i in range(a.M)]

    if a.workers <= 1:
        res = [one(t) for t in tasks]
    else:
        with Pool(a.workers) as pool:
            res = list(pool.imap(one, tasks))

    arr = np.asarray(res, dtype=np.float64)  # (M, 9)
    est, se, cov = arr[:, 0], arr[:, 1], arr[:, 2]
    a_rmse, a_r2, a_corr = arr[:, 3].mean(), arr[:, 4].mean(), arr[:, 5].mean()
    b_rmse, b_r2, b_corr = arr[:, 6].mean(), arr[:, 7].mean(), arr[:, 8].mean()
    emp = est.std(ddof=1)
    print(f"RESULT [{a.label}] n={a.n} folds={a.folds} ep={a.epochs} rep={a.repeats} "
          f"hidden={a.hidden} wd={a.weight_decay} var={a.variance} pat={a.patience} M={a.M}: "
          f"bias={est.mean() - TRUTH:+.4f} empSD={emp:.4f} SE={se.mean():.4f} "
          f"ratio={se.mean() / emp:.3f} cov={100 * cov.mean():.0f}% "
          f"|| alpha[RMSE={a_rmse:.4f} R2={a_r2:.3f} corr={a_corr:.3f}] "
          f"beta[RMSE={b_rmse:.4f} R2={b_r2:.3f} corr={b_corr:.3f}]", flush=True)


if __name__ == "__main__":
    main()
