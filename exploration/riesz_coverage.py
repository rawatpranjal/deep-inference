"""Known-truth coverage study for the ported riesz_inference (core package).

Verifies that src/deep_inference/riesz reproduces the prototype's validated coverage on
linear and logit DGPs. Writes a markdown summary. Run out of band (slow):

    uv run python exploration/riesz_coverage.py
"""

import time

import numpy as np

from deep_inference import riesz_inference

M = 40
OUT = "exploration/results_riesz_coverage.md"


def linear_dgp(seed, n=1500):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    e = 1.0 / (1.0 + np.exp(-(0.8 * X[:, 0])))
    T = (rng.random(n) < e).astype(float)
    beta = 0.5 + 0.3 * X[:, 1]
    alpha = 1.0 + 0.5 * X[:, 0]
    Y = alpha + beta * T + rng.standard_normal(n)
    return Y, T, X, float(beta.mean())


def logit_dgp(seed, n=2000):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    e = 1.0 / (1.0 + np.exp(-(0.7 * X[:, 0])))
    T = (rng.random(n) < e).astype(float)
    beta = 0.8 + 0.4 * X[:, 1]
    alpha = 0.3 * X[:, 0]
    p1 = 1.0 / (1.0 + np.exp(-(alpha + beta)))
    p0 = 1.0 / (1.0 + np.exp(-(alpha)))
    p_obs = 1.0 / (1.0 + np.exp(-(alpha + beta * T)))
    Y = (rng.random(n) < p_obs).astype(float)
    return Y, T, X, float((p1 - p0).mean())


def study(name, dgp, outcome):
    ests, ses, covers, truths = [], [], [], []
    t0 = time.time()
    for m in range(M):
        Y, T, X, truth = dgp(seed=m)
        r = riesz_inference(
            Y,
            T,
            X,
            outcome=outcome,
            n_folds=5,
            n_repeats=2,
            max_epochs=150,
            patience=20,
            seed=1000 + m,
        )
        ests.append(r.mu_hat)
        ses.append(r.se)
        covers.append(r.ci_lower <= truth <= r.ci_upper)
        truths.append(truth)
        print(
            f"[{name}] {m + 1}/{M} mu={r.mu_hat:.4f} se={r.se:.4f} "
            f"truth={truth:.4f} cov={covers[-1]} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    ests, ses, truths = np.array(ests), np.array(ses), np.array(truths)
    emp_se = float(ests.std(ddof=1))
    return {
        "name": name,
        "mean_truth": float(truths.mean()),
        "mean_est": float(ests.mean()),
        "bias": float((ests - truths).mean()),
        "emp_se": emp_se,
        "mean_se": float(ses.mean()),
        "se_ratio": float(ses.mean() / emp_se),
        "coverage": float(np.mean(covers)),
        "M": M,
    }


def main():
    rows = [study("linear", linear_dgp, "linear"), study("logit", logit_dgp, "logit")]
    lines = [
        "# RieszNet core port: known-truth coverage",
        "",
        f"Ported `riesz_inference`, M={M} replications per family.",
        "",
        "| family | truth | mean est | bias | emp SE | mean SE | SE-ratio | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['mean_truth']:.4f} | {r['mean_est']:.4f} | "
            f"{r['bias']:+.4f} | {r['emp_se']:.4f} | {r['mean_se']:.4f} | "
            f"{r['se_ratio']:.2f} | {r['coverage'] * 100:.0f}% |"
        )
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(text)
    print("\n" + text)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
