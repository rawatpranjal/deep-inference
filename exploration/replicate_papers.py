"""
Paper replication harness for deep-inference.

Modes
-----
flm   -- FLM2021 Section 6 Monte Carlo (structural_dml, linear, d=20).
          Paper: Farrell, Liang, Misra (2021). Econometrica 89(1):181-213.
          Tables 6 and 7, architecture 2 = {60,30,20}.

riesz -- RieszNet IHDP ATE replication (riesz_inference, binary T, 25 confounders).
          Paper: Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022, ICML).
          §5.1 / Table 1: MAE 0.110, coverage ~95%.

both  -- run both (default).

Run:
    PYTHONPATH=src uv run python exploration/replicate_papers.py --smoke
    PYTHONPATH=src uv run python exploration/replicate_papers.py --mode riesz --smoke
    PYTHONPATH=src uv run python exploration/replicate_papers.py --mode flm --smoke
    PYTHONPATH=src uv run python exploration/replicate_papers.py --full
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress pytorch/package noise
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from deep_inference import structural_dml  # noqa: E402

# ---- DGP -------------------------------------------------------------------
# Verbatim from FLM2021 Section 6 (Econometrica 89(1), lines 524-530 of
# references/FLM2021_docling.md):
#
#   "For either d=20 or 100, X includes a constant term and d independent
#    uniform random variables, U(0,1). Treatment assignment is Bernoulli with
#    probability p(x), where p(x) is the propensity score. We consider both
#    (i) randomized treatments with p(x) = 0.5 and (ii) observational data
#    with p(x) = (1 + exp(-alpha'_p x))^{-1}, where alpha_{p,1} = 0.09 and
#    the remainder are drawn once as U(-0.55, 0.55), and then fixed for the
#    replications."
#
#   "Y_i = mu_0(X_i) + tau(X_i) T_i + epsilon_i where epsilon_i ~ N(0,1)
#    and phi(x) are second-degree polynomials including pairwise interactions.
#    For mu_0(x) and tau(x) we consider two cases, linear and nonlinear models.
#    In both cases the intercepts are alpha_{mu,1} = 0.09 and alpha_{tau,1} =
#    -0.05 and slopes are drawn (once) as alpha_{mu,k} ~ N(0.3, 0.7) and
#    alpha_{tau,k} ~ U(0.1, 0.22), k = 2, ..., d+1. The linear models set
#    beta_mu = beta_tau = 0 while the nonlinear models take ..."
#
# This file implements ONLY the linear case (beta_mu = beta_tau = 0) for d=20.
# Coefficients are drawn ONCE per design with a fixed seed, then fixed across
# all M replications.

D = 20  # number of uniform covariates
N_COEF_SEED = 0  # seed for drawing fixed coefficients (not specified in paper)

# Intercepts (exact from paper)
ALPHA_MU_1 = 0.09
ALPHA_TAU_1 = -0.05
ALPHA_P_1 = 0.09  # propensity intercept (observational design)


def draw_fixed_coefs(rng: np.random.Generator) -> dict:
    """Draw slope coefficients ONCE; returned dict is fixed across all reps.

    Paper: alpha_{mu,k} ~ N(0.3, 0.7)  [mean=0.3, std=0.7]
           alpha_{tau,k} ~ U(0.1, 0.22)
           alpha_{p,k}   ~ U(-0.55, 0.55)   (observational design)
    k = 2, ..., d+1  -> d slopes each.
    """
    alpha_mu = rng.normal(0.3, 0.7, size=D)  # d slopes for mu_0
    alpha_tau = rng.uniform(0.1, 0.22, size=D)  # d slopes for tau
    alpha_p = rng.uniform(-0.55, 0.55, size=D)  # d slopes for propensity
    return {"alpha_mu": alpha_mu, "alpha_tau": alpha_tau, "alpha_p": alpha_p}


def true_ate(coefs: dict) -> float:
    """Closed form: E[tau(X)] = alpha_{tau,1} + 0.5 * sum_k alpha_{tau,k}.

    X_k ~ U(0,1) so E[X_k] = 0.5 for k = 2, ..., d+1.
    tau(x) = alpha_{tau,1} + sum_k alpha_{tau,k} * x_k.
    """
    return ALPHA_TAU_1 + 0.5 * float(coefs["alpha_tau"].sum())


def gen_data(n: int, coefs: dict, design: str, rng: np.random.Generator):
    """Sample one replication.

    Returns Y (n,), T (n,), X (n, D).
    X passed to the estimator is the d=20 uniforms WITHOUT the constant column;
    the package's network learns its own bias (as documented in CLAUDE.md).
    """
    X = rng.uniform(0.0, 1.0, size=(n, D))  # d=20 uniforms, no explicit constant

    # propensity
    if design == "randomized":
        p = np.full(n, 0.5)
    else:
        # logit = alpha_p_1 + alpha_p_k' * x  (constant = 1 acts as x_1)
        logit = ALPHA_P_1 + X @ coefs["alpha_p"]
        p = 1.0 / (1.0 + np.exp(-logit))

    T = (rng.uniform(size=n) < p).astype(float)

    # linear outcome (beta_mu = beta_tau = 0 -> no polynomial terms)
    mu0 = ALPHA_MU_1 + X @ coefs["alpha_mu"]
    tau = ALPHA_TAU_1 + X @ coefs["alpha_tau"]
    Y = mu0 + tau * T + rng.standard_normal(n)

    return Y, T, X


# ---- Estimator wrapper -----------------------------------------------------
def run_one(
    rep: int,
    coefs: dict,
    design: str,
    n: int,
    epochs: int,
    n_folds: int,
) -> dict:
    rng = np.random.default_rng(rep)  # each rep gets its own rng
    Y, T, X = gen_data(n, coefs, design, rng)

    result = structural_dml(
        Y,
        T,
        X,
        family="linear",
        epochs=epochs,
        n_folds=n_folds,
        hidden_dims=[60, 30, 20],  # architecture 2 from FLM2021
        verbose=False,
    )
    return {
        "mu_hat": result.mu_hat,
        "se": result.se,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
    }


# ---- Metrics ----------------------------------------------------------------
def compute_metrics(records: list, tau_bar: float) -> dict:
    mu_hats = np.array([r["mu_hat"] for r in records])
    ses = np.array([r["se"] for r in records])
    ci_lowers = np.array([r["ci_lower"] for r in records])
    ci_uppers = np.array([r["ci_upper"] for r in records])

    bias = float(np.mean(mu_hats - tau_bar))
    il = float(np.mean(2 * 1.96 * ses))
    coverage = float(np.mean((ci_lowers <= tau_bar) & (tau_bar <= ci_uppers)))
    return {"bias": bias, "il": il, "coverage": coverage}


# ---- Report formatting ------------------------------------------------------
# Paper values verbatim from references/FLM2021_docling.md.
# Table 6 (constant propensity), architecture 2 (={60,30,20}), d=20, linear -- line 566:
#   Bias=-0.00032, IL=0.079, Coverage=0.951
# Table 7 (non-constant propensity), architecture 2 (={60,30,20}), d=20, linear -- line 594:
#   Bias=0.00011, IL=0.079, Coverage=0.946
# Note: paper across-architecture coverage range for d=20 is approximately 0.93-0.96.
PAPER = {
    "randomized": {"bias": -0.00032, "il": 0.079, "coverage": 0.951},
    "observational": {"bias": 0.00011, "il": 0.079, "coverage": 0.946},
}
PAPER_NOTE = "paper across-architecture coverage range for d=20 approx 0.93-0.96"

DESIGN_LABEL = {
    "randomized": "Randomized (constant propensity p=0.5)",
    "observational": "Observational (logistic propensity)",
}
TABLE_HEADER = (
    "### FLM2021 Monte Carlo -- {label} (d=20, linear outcome, n={n}, M={M})\n"
    "| Quantity | Package | Paper |\n"
    "|---|---|---|\n"
    "| Bias | {pkg_bias:.5f} | {paper_bias} |\n"
    "| Interval length | {pkg_il:.3f} | {paper_il} |\n"
    "| Coverage | {pkg_cov:.3f} | {paper_cov} |"
)


def format_table(design: str, metrics: dict, n: int, M: int) -> str:
    p = PAPER[design]
    return TABLE_HEADER.format(
        label=DESIGN_LABEL[design],
        n=n,
        M=M,
        pkg_bias=metrics["bias"],
        paper_bias=p["bias"],
        pkg_il=metrics["il"],
        paper_il=p["il"],
        pkg_cov=metrics["coverage"],
        paper_cov=p["coverage"],
    )


# ---- RieszNet IHDP ---------------------------------------------------------
# Paper benchmarks:
#   MAE 0.110 -- Table 1, DR RieszNet, RieszNet2022_docling.md line 219
#   Coverage ~0.95 -- §5.1 / Figure 2, line 227
RIESZ_PAPER_MAE = 0.110
RIESZ_PAPER_COV = 0.95
RIESZ_SMOKE_CFG = dict(N=3, n_folds=5, n_repeats=1, max_epochs=80, patience=15)
RIESZ_FULL_CFG = dict(N=100, n_folds=5, n_repeats=2, max_epochs=200, patience=20)

IHDP_URL = (
    "https://raw.githubusercontent.com/claudiashi57/dragonnet/master/dat/ihdp/csv/"
    "ihdp_npci_{i}.csv"
)


def fetch_ihdp(n: int, dest: str = "data/external/ihdp") -> tuple[Path, int]:
    """Download ihdp_npci_{i}.csv for i=1..n only if missing (idempotent).

    The public Dragonnet/CEVAE mirrors only ship the first ~50 individual CSVs,
    so on a 404 we stop and use however many contiguous realizations exist rather
    than hard-failing. Returns (dest_path, n_available).
    """
    import urllib.error as _err  # lazy: avoids import at module level
    import urllib.request as _req

    dest_path = Path(__file__).resolve().parent.parent / dest
    dest_path.mkdir(parents=True, exist_ok=True)
    avail = 0
    for i in range(1, n + 1):
        fpath = dest_path / f"ihdp_npci_{i}.csv"
        if not fpath.exists():
            try:
                _req.urlretrieve(IHDP_URL.format(i=i), str(fpath))
            except _err.HTTPError:
                fpath.unlink(missing_ok=True)  # remove the empty 404 stub
                break
        avail += 1
    return dest_path, avail


def load_ihdp(i: int, dest_path: Path):
    """Load realization i. Returns Y, T, X (estimator inputs only) and ate_i.

    CSV layout (no header, 747 rows):
      col 0 = T (binary), col 1 = y_factual, col 2 = y_cfactual (unused),
      col 3 = mu0, col 4 = mu1, cols 5..29 = 25 confounders.
    ate_i = mean(mu1 - mu0) used ONLY to score results, never passed to estimator.
    """
    data = np.loadtxt(str(dest_path / f"ihdp_npci_{i}.csv"), delimiter=",")
    T = data[:, 0]
    Y = data[:, 1]
    # col 2 = y_cfactual -- do not use
    mu0, mu1 = data[:, 3], data[:, 4]
    X = data[:, 5:30]
    ate_i = float(np.mean(mu1 - mu0))
    return Y, T, X, ate_i


def run_riesz_mode(args) -> str:
    """Run RieszNet IHDP replication. Returns the markdown table string."""
    from deep_inference.riesz import (
        riesz_inference,  # noqa: PLC0415  lazy: not needed for flm
    )

    cfg = (
        RIESZ_SMOKE_CFG.copy()
        if (args.smoke or not args.full)
        else RIESZ_FULL_CFG.copy()
    )
    if args.riesz_N is not None:
        cfg["N"] = args.riesz_N
    N, tag = cfg["N"], args.tag

    print(
        f"RieszNet IHDP replication: N={N} realizations, "
        f"n_folds={cfg['n_folds']}, n_repeats={cfg['n_repeats']}, "
        f"max_epochs={cfg['max_epochs']}, patience={cfg['patience']}"
    )
    dest_path, avail = fetch_ihdp(N)
    if avail < N:
        print(
            f"NOTE: only {avail} IHDP realizations available from the public "
            f"mirror (requested {N}); running on {avail}."
        )
        N = avail

    abs_errs: list[float] = []
    covers: list[float] = []
    t0 = time.time()
    for i in range(1, N + 1):
        Y, T, X, ate_i = load_ihdp(i, dest_path)
        res = riesz_inference(
            Y,
            T,
            X,
            outcome="linear",
            n_folds=cfg["n_folds"],
            n_repeats=cfg["n_repeats"],
            max_epochs=cfg["max_epochs"],
            patience=cfg["patience"],
            seed=i,
        )
        err = abs(res.mu_hat - ate_i)
        cov = float(res.ci_lower <= ate_i <= res.ci_upper)
        abs_errs.append(err)
        covers.append(cov)
        print(
            f"  i={i:3d} ate_true={ate_i:.4f} mu_hat={res.mu_hat:.4f} "
            f"se={res.se:.4f} err={err:.4f} cover={bool(cov)}"
        )

    elapsed = time.time() - t0
    mae = float(np.mean(abs_errs))
    coverage = float(np.mean(covers))

    tbl = (
        f"### RieszNet IHDP ATE replication "
        f"(N={N} of the paper's 1000 semi-synthetic datasets)\n"
        f"| Quantity | Package | Paper |\n"
        f"|---|---|---|\n"
        f"| MAE | {mae:.3f} | {RIESZ_PAPER_MAE} |\n"
        f"| Coverage | {coverage:.2f} | {RIESZ_PAPER_COV} |"
    )
    # ponytail: using original-T from CSVs for coverage, not redrawn T (paper §5.1, line 225-227)
    note = (
        "NOTE: Data = Dragonnet IHDP-1000 CSVs (Shi et al. 2019). "
        f"N={N} < the paper's 1000. "
        "CI coverage uses the CSVs' original confounded T; the paper's coverage figure "
        "(Figure 2, line 227) redraws T per propensity 'True' from NPCI -- "
        "these coverage numbers are not directly comparable. "
        "The MAE comparison (Table 1) is directly comparable."
    )

    print()
    print(tbl)
    print()
    print(note)
    print(f"\nWall time: {elapsed:.1f}s")

    out_path = Path(__file__).parent / f"replicate_papers_riesz_{tag}.md"
    out_path.write_text(
        "\n".join(
            [
                "# RieszNet IHDP Replication",
                "",
                f"Config: N={N}, n_folds={cfg['n_folds']}, n_repeats={cfg['n_repeats']}, "
                f"max_epochs={cfg['max_epochs']}, patience={cfg['patience']}",
                f"Wall time: {elapsed:.1f}s",
                "",
                tbl,
                "",
                note,
            ]
        )
    )
    print(f"\nReport saved to: {out_path.resolve()}")
    return tbl


# ---- CLI -------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Paper replication harness (flm | riesz | both)"
    )
    run_mode = ap.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--smoke", action="store_true", help="quick check (small M/N)"
    )
    run_mode.add_argument("--full", action="store_true", help="full run (large M/N)")
    ap.add_argument(
        "--mode",
        choices=["flm", "riesz", "both"],
        default="both",
        help="which replication to run (default: both)",
    )
    ap.add_argument("--M", type=int, default=None, help="FLM: override number of reps")
    ap.add_argument("--n", type=int, default=10_000, help="FLM: sample size per rep")
    ap.add_argument("--epochs", type=int, default=80, help="FLM: epochs per rep")
    ap.add_argument("--folds", type=int, default=5, help="FLM: cross-fitting folds")
    ap.add_argument(
        "--riesz-N",
        type=int,
        default=None,
        dest="riesz_N",
        help="RieszNet: override number of IHDP realizations",
    )
    ap.add_argument("--tag", type=str, default="smoke", help="report filename tag")
    return ap.parse_args()


def run_flm_mode(args):
    """Run the FLM2021 Monte Carlo replication."""
    if args.M is not None:
        M = args.M
    elif args.full:
        M = 200
    else:
        M = 5  # default = smoke

    n, epochs, n_folds, tag = args.n, args.epochs, args.folds, args.tag

    coef_rng = np.random.default_rng(N_COEF_SEED)
    coefs = draw_fixed_coefs(coef_rng)
    tau_bar = true_ate(coefs)
    print(f"True ATE tau_bar = {tau_bar:.6f}")
    print(f"Config: M={M}, n={n}, epochs={epochs}, n_folds={n_folds}\n")

    tables = []
    t0 = time.time()

    for design in ("randomized", "observational"):
        print(f"Running {design} design ({M} reps)...")
        records = []
        for rep in range(M):
            rec = run_one(rep, coefs, design, n, epochs, n_folds)
            records.append(rec)
            if (rep + 1) % max(1, M // 5) == 0:
                print(
                    f"  rep {rep + 1}/{M} mu_hat={rec['mu_hat']:.4f} se={rec['se']:.4f}"
                )

        metrics = compute_metrics(records, tau_bar)
        tbl = format_table(design, metrics, n, M)
        print()
        print(tbl)
        print()
        tables.append(tbl)

    elapsed = time.time() - t0
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Note: {PAPER_NOTE}")

    out_dir = Path(__file__).parent
    out_path = out_dir / f"replicate_papers_flm_{tag}.md"
    report_lines = [
        "# FLM2021 Monte Carlo Replication",
        "",
        f"Config: M={M}, n={n}, epochs={epochs}, n_folds={n_folds}, "
        f"hidden_dims=[60,30,20] (architecture 2)",
        "",
        f"True ATE tau_bar = {tau_bar:.6f}",
        f"Wall time: {elapsed:.1f}s",
        f"Note: {PAPER_NOTE}",
        "",
    ] + [t + "\n" for t in tables]
    out_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to: {out_path.resolve()}")


def main():
    args = parse_args()
    if args.mode in ("flm", "both"):
        run_flm_mode(args)
    if args.mode in ("riesz", "both"):
        run_riesz_mode(args)


if __name__ == "__main__":
    main()
