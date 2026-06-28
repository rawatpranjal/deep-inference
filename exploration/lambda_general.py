"""
GENERAL-case Lambda(x) estimation on the linear spike.

The structured-propensity fix (lambda_inv_fix.py) only works because the linear
Hessian is a closed form in e(x). The package's GENERAL path makes no such
assumption: it autodiffs the per-obs Hessian and regresses its entries on X
(multi-output ridge/rf/lgbm/mlp), then inverts. This script asks whether any
general Lambda(x) REGRESSOR -- lgbm (heavy reg) or mlp (early-stopped) -- stays
accurate AND well-conditioned enough to restore valid SEs, and whether more data
closes the gap (learning curves). No closed form is used.

Run:
  PYTHONPATH=src /opt/homebrew/bin/python3.11 exploration/lambda_general.py --M 50 --workers 8
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lambda_inv_fix import run_cell   # uses the same _one/run_flm_cell/fit_panel harness

# comparison @ n=2000 (theta=net) + learning curves for lgbm and mlp
CELLS = (
    [{"lam": l, "theta": "net", "n": 2000} for l in
     ["flat", "oracle", "lgbm", "mlp", "rf", "ridge@1000"]]
    + [{"lam": "lgbm", "theta": "net", "n": nn} for nn in [1000, 4000, 8000]]
    + [{"lam": "mlp", "theta": "net", "n": nn} for nn in [1000, 4000, 8000]]
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="exploration/results_lambda_general.md")
    args = ap.parse_args()
    print(f"GENERAL Lambda(x) regress-and-invert. M={args.M} folds={args.folds} cells={len(CELLS)}", flush=True)
    rows = []
    for c in CELLS:
        label = f"{c['lam']}/n={c['n']}"
        r = run_cell(c, args.M, args.folds, args.epochs, base_seed=1000, workers=args.workers)
        r["cell"] = label
        rows.append(r)
        if r.get("n_ok"):
            print(f"  {label:16s} SEratio={r['se_ratio']:.3f} cov={100*r['coverage']:.0f}% "
                  f"var_ratio={r['var_ratio']:.3f} Linv_R2={r['laminv_r2p']:.3f} "
                  f"L_R2={r['lam_r2p']:.3f} psi_R2={r['psi_r2']:.3f} bias={r['bias']:+.4f} "
                  f"fail={r['nfail']}", flush=True)
        else:
            print(f"  {label:16s} ALL {r['nfail']} failed: {r.get('fail_eg','')}", flush=True)
    hdr = ("| cell | SE-ratio | cov | bias | var_ratio | Linv-R2 | L-R2 | psi-R2 | fails |")
    lines = ["# General Lambda(x) regress-and-invert on the linear spike\n",
             f"M={args.M} folds={args.folds}. truth=1.0. lgbm/mlp/rf/ridge = the GENERAL path "
             "(regress autodiff Hessian entries on X, then invert). No closed form. "
             "L-R2 = out-of-sample Lambda fit (overfit shows as high in-sample but this low); "
             "Linv-R2 = the inverse that actually enters psi.\n", hdr, "|" + "---|" * 8]
    for r in rows:
        if not r.get("n_ok"):
            lines.append(f"| {r['cell']} | all {r['nfail']} failed |" + " |" * 7); continue
        lines.append(f"| {r['cell']} | {r['se_ratio']:.3f} | {100*r['coverage']:.0f}% | "
                     f"{r['bias']:+.4f} | {r['var_ratio']:.3f} | {r['laminv_r2p']:.3f} | "
                     f"{r['lam_r2p']:.3f} | {r['psi_r2']:.3f} | {r['nfail']} |")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
