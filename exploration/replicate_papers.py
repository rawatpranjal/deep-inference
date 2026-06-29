"""
FLM2021 Section 6 Monte Carlo replication using THIS package (structural_dml).

Paper: Farrell, Liang, Misra (2021). "Deep Neural Networks for Estimation and
Inference." Econometrica 89(1):181-213. Section 6, Tables 6 and 7.

Compares package output vs paper-reported numbers for architecture 2 = {60,30,20},
d=20 covariates, linear outcome, n=10,000 per replication.

Run:
    PYTHONPATH=src uv run python exploration/replicate_papers.py --smoke
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


# ---- CLI -------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="FLM2021 Section 6 replication")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="M=5 quick check")
    mode.add_argument("--full", action="store_true", help="M=200 full run")
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", type=str, default="smoke")
    return ap.parse_args()


def main():
    args = parse_args()

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

    # write report
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


if __name__ == "__main__":
    main()
