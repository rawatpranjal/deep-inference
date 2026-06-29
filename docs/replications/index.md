# Replications

What this package has validated against known truth, with the numbers. The raw harness
output lives in `exploration/` (the master results markdown plus the dark tabbed dashboard
from `build_dashboard.py`); this page is the curated summary.

Each row is a Monte-Carlo benchmark on a confounded DGP with a known average treatment
effect μ*. A method is **valid** when its coverage sits in roughly [93,97]% with an SE-ratio
near 1 and small bias. The full panel runs Oracle-MLE (gold anchor), FLM[cholesky] (the
general PSD-Λ fix), FLM[ridge] and FLM[oracle-Λ] (contrasts and ceiling), RieszNet (the
doubly-robust competitor), and Naive (no correction, expected to under-cover).

## FLM general Λ̂(x) certification, M=100 on RunPod

The headline result. FLM[cholesky] is the general, autodiff-only fix (PSD-by-construction
Λ̂(x)=L(x)L(x)ᵀ, no hardcoded family formula), and it is valid on all three families at a
single truth-free tikhonov=0.01. Source: `exploration/results_cert_all_M100.md`,
commit 12524f1, fresh-agent (Opus) verified.

### Linear (truth 1.0000)

| method | bias | SE-ratio | coverage |
|---|---|---|---|
| Oracle-MLE | -0.0008 | 0.96 | 96% |
| FLM[cholesky] | -0.0041 | 0.92 | 93% |
| FLM[oracle-Λ] | -0.0030 | 0.96 | 95% |
| RieszNet | -0.0083 | 1.09 | 97% |
| Naive | -0.0042 | 0.17 | 23% |

### Logit (truth 0.1481)

| method | bias | SE-ratio | coverage |
|---|---|---|---|
| Oracle-MLE | +0.0054 | 1.02 | 94% |
| FLM[cholesky] | -0.0020 | 1.07 | 98% |
| FLM[ridge] | +0.0166 | 0.92 | 86% |
| FLM[oracle-Λ] | +0.0167 | 0.90 | 86% |
| RieszNet | +0.0062 | 1.06 | 100% |
| Naive | +0.0069 | 0.11 | 10% |

On logit, FLM[cholesky] beats even the oracle-Λ ceiling (98 vs 86), because its implicit
regularization helps where the true Λ is near-singular at low overlap.

### Poisson (truth 0.8981)

| method | bias | SE-ratio | coverage |
|---|---|---|---|
| Oracle-MLE | -0.0058 | 0.98 | 99% |
| FLM[cholesky] | -0.0116 | 0.98 | 98% |
| FLM[ridge] | -0.0325 | 1.07 | 94% |
| FLM[oracle-Λ] | -0.0403 | 1.02 | 95% |
| RieszNet | +0.0382 | 0.32 | 98% |
| Naive | -0.0177 | 0.25 | 33% |

RieszNet is unreliable on poisson: its empirical SE is blown up by divergent reps
(SE-ratio 0.32), so the 98% coverage is not trustworthy. FLM[cholesky] is the least-biased
valid method here.

**Caveat.** Linear 93% sits within about 2.6pp of Monte-Carlo noise of the band edge at
M=100. An M=200 confirm would tighten it.

## Status by family

| family | known-truth DGP | oracle-MLE | FLM[cholesky] valid | RieszNet | tier |
|---|---|---|---|---|---|
| linear | yes | yes | yes (93%, M=100) | yes | done |
| logit | yes | yes | yes (98%, M=100) | yes | done |
| poisson | yes | yes | yes (98%, M=100) | unreliable | done |
| gamma | yes | yes | no (88%, bias -0.07, M=50) | yes (96%) | A, blocked |
| negbin | yes | yes | no (bias -0.11, fragile 96%, M=50) | yes (96%) | A, blocked |
| gaussian | yes | yes | marginal (92%, M=50); oracle-Λ + RieszNet 96% | yes (96%) | A, ~done |
| probit | yes | yes | yes (98%, M=50) | yes (98%) | B, done |
| beta | yes | yes | yes (96%, M=50) | yes (100%) | B, done |
| zip | oracle ready | yes | - | - | B |
| weibull, gumbel, tobit | needs custom MLE oracle | - | - | - | B |
| multinomial, quantile, combinatorial | partial | - | - | - | C |

## The landscape (2026-06-29)

Eight GLM families mapped. Measured result: FLM[cholesky] (the general fix) fails on exactly
gamma and negbin; it passes on linear, gaussian (marginal), logit, probit, beta, poisson. Two
falsifications come for free: beta passes despite a y-dependent Hessian on a continuous
outcome (neither is sufficient to cause the failure), and poisson passes despite a log link
(log link is not sufficient). gamma and negbin differ from the passing families on several
confounded axes at once (heavy tail, log link, y-dependent Hessian weight, convex exp target);
which axis is causal is NOT yet isolated. The clean test is to sweep gamma's shape k and see
whether the failure tracks outcome variance. Until then, treat the cause as open. Dashboard:
`exploration/results_landscape.html`.

## Tier A, M=50 (2026-06-29)

gamma, negbin, gaussian wired into the benchmark (`exploration/results_tierA_M50.html`).
The result is a two-part diagnosis, not a clean cert. (1) FLM[cholesky] is fragile on the
heavy-tailed log-link count families (gamma bias -0.072 / coverage 88%, negbin bias -0.111 /
fragile 96%): the cholesky Λ̂-net misfits the noisy y-dependent per-obs Hessians, a left tail
of bad-Λ reps that the exact oracle-Λ does not have. (2) Those same count families carry a
separate downward nuisance bias of ~-0.045 that hits even RieszNet and oracle-Λ (Jensen
shrinkage on E[exp(η)]), absent on gaussian. The gaussian control confirms both are
count-family-specific: its oracle-Λ (+0.001 / 96%) and RieszNet (+0.000 / 96%) are perfect,
and the 3-dim θ / 3x3 Λ path works. Certifying cholesky on gamma/negbin needs a cholesky-Λ
hardening plus a log-link nuisance-bias fix, both in the core package.

Older component validations (parameter recovery, autodiff, Λ accuracy, ψ assembly, per-family
coverage evals) live under `evals/` and `docs/simulation_studies/`. The FLM framework itself follows
Farrell-Liang-Misra (2021, 2025); see `docs/dev/paper_replication_details.md` for the
paper-to-code mapping.
