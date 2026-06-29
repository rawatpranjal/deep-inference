"""Does TRIMMING low-overlap obs stabilize the cholesky Λ̂ on logit?

The true logit Λ*(x) is near-singular when det Λ* = e(1-e)·w0·w1 -> 0, i.e. at
(a) propensity boundaries e->0/1 (treatment overlap) OR (b) outcome saturation
w_t=p_t(1-p_t)->0 (a Fisher-info channel with no linear analogue). Trimming on the
propensity targets (a) only. This probe trims on the TRUE propensity e*(x)∈[δ,1-δ]
(oracle trim, clean diagnostic), recomputes the trimmed-population truth, and runs
FLM[cholesky] (near-zero tikhonov, so trimming alone does the stabilizing) plus the
oracle-Λ ceiling on the trimmed sample. SE-ratio / coverage are scale-free so they
compare across δ even though emp SE grows as n_kept shrinks.

  --trim propensity  -> keep e*(x)∈[δ,1-δ]            (what 'low overlap' usually means)
  --trim lambda_eig  -> keep min-eig(Λ*(x)) ≥ δ·max  (the actual singularity, both channels)

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/probe_logit_trim.py --M 24 --workers 8
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
from spike import (gen_logit, flm_logit, covered, draw_X, a_of, b_of, propensity,
                   OracleLogitLambda, GAMMA)  # noqa: E402

DELTAS = [0.0, 0.05, 0.10, 0.15]


def keep_mask(X, delta, mode):
    """Boolean keep-mask on raw covariates X (numpy)."""
    if delta <= 0:
        return np.ones(len(X), dtype=bool)
    if mode == "propensity":
        e = propensity(X)
        return (e >= delta) & (e <= 1 - delta)
    elif mode == "lambda_eig":
        # min eigenvalue of the true Λ*(x); keep where it is not tiny
        Lam = OracleLogitLambda().predict(torch.tensor(X, dtype=torch.float32))
        mineig = torch.linalg.eigvalsh(Lam)[:, 0].numpy()
        return mineig >= delta * mineig.max()
    raise ValueError(mode)


def trimmed_truth(delta, mode, n=2_000_000, seed=99):
    rng = np.random.default_rng(seed)
    X = draw_X(n, rng)
    a, b = a_of(X), b_of(X)
    s1 = 1.0 / (1.0 + np.exp(-(a + b)))
    s0 = 1.0 / (1.0 + np.exp(-a))
    m = keep_mask(X, delta, mode)
    return float(np.mean((s1 - s0)[m])), float(m.mean())


def _one(task):
    delta, mode, spec, seed, n, folds, epochs = task
    torch.manual_seed(seed); np.random.seed(seed % (2 ** 32))
    rng = np.random.default_rng(seed)
    Y, T, X = gen_logit(n, rng)
    m = keep_mask(X, delta, mode)
    Yt, Tt, Xt = Y[m], T[m], X[m]
    try:
        tik = 1e-8 if spec in ("cholesky", "oracle") else None
        mu, se = flm_logit(Yt, Tt, Xt, folds, epochs, lambda_spec=spec, tikhonov=tik)
    except Exception as ex:
        return {"fail": str(ex)[:80]}
    return {"delta": delta, "spec": spec, "mu": mu, "se": se, "nkept": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=24)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--trim", default="propensity", choices=["propensity", "lambda_eig"])
    ap.add_argument("--specs", default="cholesky,oracle")
    args = ap.parse_args()
    specs = args.specs.split(",")

    truths = {d: trimmed_truth(d, args.trim) for d in DELTAS}
    print(f"trim={args.trim}  M={args.M} n={args.n} folds={args.folds}", flush=True)
    for d in DELTAS:
        t, frac = truths[d]
        print(f"  delta={d:.2f}: trimmed truth={t:.4f}  kept~{100*frac:.0f}%", flush=True)

    tasks = [(d, args.trim, spec, 5000 + i, args.n, args.folds, args.epochs)
             for d in DELTAS for spec in specs for i in range(args.M)]
    with Pool(args.workers) as pool:
        outs = list(pool.imap(_one, tasks))

    print(f"\n{'delta':>6s} {'spec':>9s} {'nkept':>6s} {'mean est':>9s} {'bias':>9s} "
          f"{'emp SE':>8s} {'mean SE':>8s} {'SEratio':>8s} {'cover':>7s} {'fail':>5s}")
    for d in DELTAS:
        truth = truths[d][0]
        for spec in specs:
            rows = [o for o in outs if o.get("delta") == d and o.get("spec") == spec and "mu" in o]
            nfail = args.M - len(rows)
            if not rows:
                print(f"{d:6.2f} {spec:>9s}  all {args.M} failed"); continue
            est = np.array([o["mu"] for o in rows]); se = np.array([o["se"] for o in rows])
            cov = np.array([covered(o["mu"], o["se"], truth) for o in rows])
            nk = int(np.mean([o["nkept"] for o in rows]))
            emp = est.std(ddof=1)
            print(f"{d:6.2f} {spec:>9s} {nk:6d} {est.mean():9.4f} {est.mean()-truth:+9.4f} "
                  f"{emp:8.4f} {se.mean():8.4f} {se.mean()/emp:8.2f} {100*cov.mean():6.0f}% {nfail:5d}")


if __name__ == "__main__":
    main()
