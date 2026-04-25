"""
FML auto-research: factorial validation using /Users/pranjal/deepest/ package.

Goals:
  1. Family invariance  : IF works across Poisson, Logit, Gaussian
  2. Dimension sensitivity: d=4 vs d=8 vs d=16 (curse of dim)
  3. N scaling           : bias/variance trade-off as N grows
  4. Target flexibility  : ATE, subgroup mean, pointwise CATE

Each config: M=30 Monte Carlo reps, parallel across 10 workers.

Acceptance criteria per config (from deepest/evals/eval_06_coverage.py):
    coverage ∈ [0.90, 0.99]
    SE ratio (emp/reported) ∈ [0.7, 1.5]
    |bias| < 0.1 · SE

Report:
    Coverage matrix (family × dim)
    N-scaling at best-dim (d=4)
    Target-flexibility at d=4

Honest limitations kept in view:
  - My earlier sweep failed at d=16, N<200k.  Auto-research confirms whether
    the deepest package fixes this via better defaults (n_folds=50,
    3-way split, ridge-Λ) or whether d=16 still fails.
  - We stay in simulation only (no real H&M data) per user instruction.
"""

import sys
sys.path.insert(0, "/Users/pranjal/deepest/src")

import numpy as np
import torch
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import beta as beta_dist

from deep_inference import structural_dml

# ======================================================================
# DGPs for 3 families.  Each returns (X, T, Y, mu_star, alpha*, beta*)
# ======================================================================
def dgp_poisson(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    a = 1.3 + 0.4 * X[:, 0]
    if d >= 2:
        b = -0.4 + 0.3 * np.tanh(X[:, 0]) + 0.2 * X[:, 1]
    else:
        b = -0.4 + 0.3 * np.tanh(X[:, 0])
    T = rng.choice([0.0, 1.0], size=n, p=[0.5, 0.5]).astype(np.float32)
    mu = np.exp(a + b * T).astype(np.float32)
    Y = rng.poisson(mu).astype(np.float32)
    return X, T, Y, float(b.mean()), a, b


def dgp_logit(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    a = -0.3 * X[:, 0]
    if d >= 2:
        b = 0.5 + 0.4 * np.tanh(X[:, 0]) + 0.3 * X[:, 1]
    else:
        b = 0.5 + 0.4 * np.tanh(X[:, 0])
    T = rng.choice([0.0, 1.0], size=n, p=[0.5, 0.5]).astype(np.float32)
    eta = a + b * T
    p = 1 / (1 + np.exp(-eta))
    Y = (rng.uniform(size=n) < p).astype(np.float32)
    return X, T, Y, float(b.mean()), a, b


def dgp_linear(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    a = 0.5 * X[:, 0]
    if d >= 2:
        b = -0.5 + 0.3 * np.tanh(X[:, 0]) + 0.2 * X[:, 1]
    else:
        b = -0.5 + 0.3 * np.tanh(X[:, 0])
    T = rng.standard_normal(n).astype(np.float32)
    Y = (a + b * T + 0.5 * rng.standard_normal(n)).astype(np.float32)
    return X, T, Y, float(b.mean()), a, b


DGPS = {"poisson": dgp_poisson, "logit": dgp_logit, "linear": dgp_linear}


def cp_ci(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


# ======================================================================
# Single MC replicate
# ======================================================================
def run_one(payload):
    family, n, d, n_folds, seed = payload
    X, T, Y, mu_star, _, _ = DGPS[family](n, d, seed)
    try:
        result = structural_dml(
            Y=Y, T=T, X=X, family=family,
            hidden_dims=[64, 32],
            epochs=200,
            lr=0.01,
            patience=10,
            n_folds=n_folds,
            verbose=False,
            store_data=False,
        )
        covered = bool(result.ci_lower <= mu_star <= result.ci_upper)
        return dict(
            family=family, n=n, d=d, n_folds=n_folds, seed=seed,
            mu_star=mu_star,
            mu_hat=float(result.mu_hat),
            mu_naive=float(result.mu_naive),
            se=float(result.se),
            ci_lo=float(result.ci_lower),
            ci_hi=float(result.ci_upper),
            covered=covered,
            failed=False,
        )
    except Exception as e:
        return dict(family=family, n=n, d=d, n_folds=n_folds, seed=seed,
                    mu_star=mu_star, failed=True, error=str(e))


def run_config(label, family, n, d, n_folds, M):
    print(f"\n{'='*78}")
    print(f"[{label}]  family={family}  d={d}  n={n}  n_folds={n_folds}  M={M}"
          .center(78))
    print(f"{'='*78}")
    payloads = [(family, n, d, n_folds, 7000 + m * 31) for m in range(M)]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(run_one, p) for p in payloads]
        for i, f in enumerate(as_completed(futs)):
            results.append(f.result())
            if (i + 1) % max(1, M // 4) == 0:
                ok = [r for r in results if not r["failed"]]
                if ok:
                    mu_f = np.array([r["mu_hat"] for r in ok])
                    mu_s = np.array([r["mu_star"] for r in ok])
                    bias = (mu_f - mu_s).mean()
                    cov = np.mean([r["covered"] for r in ok])
                    print(f"  {i+1:3d}/{M}  t={time.time()-t0:4.0f}s  "
                          f"bias={bias:+.4f}  cov={cov:.3f}")
    return summarize(label, results)


def summarize(label, results):
    ok = [r for r in results if not r["failed"]]
    fail = [r for r in results if r["failed"]]
    if not ok:
        print(f"  ALL FAILED.  First error: {fail[0]['error'] if fail else '?'}")
        return None
    mu_f = np.array([r["mu_hat"] for r in ok])
    mu_s = np.array([r["mu_star"] for r in ok])
    mu_n = np.array([r["mu_naive"] for r in ok])
    se = np.array([r["se"] for r in ok])
    covered = np.array([r["covered"] for r in ok])

    bias_fml = (mu_f - mu_s).mean()
    bias_naive = (mu_n - mu_s).mean()
    emp_sd_fml = (mu_f - mu_s).std(ddof=1)
    emp_sd_naive = (mu_n - mu_s).std(ddof=1)
    mean_se = se.mean()
    se_ratio = mean_se / emp_sd_fml if emp_sd_fml > 0 else float("nan")
    rmse = np.sqrt(((mu_f - mu_s) ** 2).mean())
    cov = covered.mean()
    k_cov = int(covered.sum())
    lo, hi = cp_ci(k_cov, len(ok))

    # pass criteria from eval_06_coverage.py
    pass_cov = 0.90 <= cov <= 0.99
    pass_se = 0.7 <= se_ratio <= 1.5
    pass_bias = abs(bias_fml) < 0.1 * mean_se
    overall = all([pass_cov, pass_se, pass_bias])
    verdict = "PASS" if overall else (
        "PARTIAL" if pass_cov else "FAIL"
    )

    print(f"\n  [SUMMARY {label}]   ({len(ok)}/{len(results)} ok, {len(fail)} fail)")
    print(f"    µ*                     = {mu_s.mean():+.5f}  (±{mu_s.std():.5f} across reps)")
    print(f"    µ̂_FML mean / bias     = {mu_f.mean():+.5f}  /  {bias_fml:+.5f}")
    print(f"    µ̂_naive mean / bias   = {mu_n.mean():+.5f}  /  {bias_naive:+.5f}")
    print(f"    emp SD µ̂_FML          = {emp_sd_fml:.5f}")
    print(f"    mean reported SE       = {mean_se:.5f}   SE/emp_SD = {se_ratio:.3f}")
    print(f"    RMSE µ̂_FML            = {rmse:.5f}")
    print(f"    |bias_FML| / mean_SE   = {abs(bias_fml)/mean_se:.3f}  (PASS if <0.1)")
    print(f"    95% coverage           = {cov:.3f}  CP-CI [{lo:.3f}, {hi:.3f}]")
    print(f"    VERDICT: {verdict}  "
          f"(cov {'PASS' if pass_cov else 'FAIL'}, "
          f"SE ratio {'PASS' if pass_se else 'FAIL'}, "
          f"bias {'PASS' if pass_bias else 'FAIL'})")

    return dict(
        label=label, family=ok[0]["family"], n=ok[0]["n"], d=ok[0]["d"],
        n_folds=ok[0]["n_folds"], M=len(ok), failures=len(fail),
        mu_star_mean=float(mu_s.mean()),
        bias_fml=float(bias_fml), bias_naive=float(bias_naive),
        emp_sd_fml=float(emp_sd_fml), mean_se=float(mean_se),
        se_ratio=float(se_ratio), rmse=float(rmse),
        coverage=float(cov), cp_ci_lo=float(lo), cp_ci_hi=float(hi),
        pass_coverage=pass_cov, pass_se=pass_se, pass_bias=pass_bias,
        verdict=verdict,
    )


# ======================================================================
# Main sweep
# ======================================================================
if __name__ == "__main__":
    M = 30
    all_results = []
    t_all = time.time()
    print(f"\nFML AUTO-RESEARCH")
    print(f"M={M} MC reps per config, 10 workers, deep_inference pkg\n")

    # ------ Block 1: Family invariance at (d=4, n=20k) ------
    all_results.append(run_config("FAM-poisson-d4-n20k",
                                  "poisson", 20_000, 4, 30, M))
    all_results.append(run_config("FAM-logit-d4-n20k",
                                  "logit", 20_000, 4, 30, M))
    all_results.append(run_config("FAM-linear-d4-n20k",
                                  "linear", 20_000, 4, 30, M))

    # ------ Block 2: Dimension sensitivity (poisson, n=20k) ------
    # d=4 already covered above
    all_results.append(run_config("DIM-poisson-d8-n20k",
                                  "poisson", 20_000, 8, 30, M))
    all_results.append(run_config("DIM-poisson-d16-n20k",
                                  "poisson", 20_000, 16, 30, M))
    all_results.append(run_config("DIM-poisson-d2-n20k",
                                  "poisson", 20_000, 2, 30, M))

    # ------ Block 3: N scaling (poisson, d=4) ------
    all_results.append(run_config("N-poisson-d4-n5k",
                                  "poisson", 5_000, 4, 20, M))
    all_results.append(run_config("N-poisson-d4-n10k",
                                  "poisson", 10_000, 4, 25, M))
    # n=20k already covered above
    all_results.append(run_config("N-poisson-d4-n50k",
                                  "poisson", 50_000, 4, 50, M))

    # ------ Block 4: n_folds ablation (poisson, d=4, n=20k) ------
    all_results.append(run_config("FOLDS-poisson-d4-n20k-folds5",
                                  "poisson", 20_000, 4, 5, M))
    all_results.append(run_config("FOLDS-poisson-d4-n20k-folds50",
                                  "poisson", 20_000, 4, 50, M))

    # ------ Report matrix ------
    print("\n" + "=" * 100)
    print("AUTO-RESEARCH SUMMARY TABLE".center(100))
    print("=" * 100)
    hdr = (f"  {'label':<36}{'fam':>9}{'d':>4}{'n':>8}{'folds':>7}"
           f"{'bias':>9}{'|b|/SE':>9}{'SEratio':>9}{'cov':>7}"
           f"{'CP-CI':>14}{'verdict':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for s in all_results:
        if s is None:
            continue
        print(f"  {s['label']:<36}{s['family']:>9}{s['d']:>4}{s['n']:>8,}"
              f"{s['n_folds']:>7}"
              f"{s['bias_fml']:>+9.4f}"
              f"{abs(s['bias_fml'])/s['mean_se']:>9.3f}"
              f"{s['se_ratio']:>9.3f}"
              f"{s['coverage']:>7.3f}"
              f" [{s['cp_ci_lo']:.2f},{s['cp_ci_hi']:.2f}]"
              f"{s['verdict']:>10}")

    # save
    with open("/tmp/qpoisson_if/autoresearch_results.json", "w") as f:
        json.dump([s for s in all_results if s], f, indent=1, default=str)
    print(f"\n[saved] /tmp/qpoisson_if/autoresearch_results.json")
    print(f"[total time] {time.time()-t_all:.0f}s")

    # count passes
    passes = sum(1 for s in all_results if s and s["verdict"] == "PASS")
    partials = sum(1 for s in all_results if s and s["verdict"] == "PARTIAL")
    fails = sum(1 for s in all_results if s and s["verdict"] == "FAIL")
    n_valid = sum(1 for s in all_results if s)
    print(f"\nHEADLINE: {passes}/{n_valid} PASS, {partials} PARTIAL, "
          f"{fails} FAIL")
