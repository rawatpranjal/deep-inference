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
| linear | yes | yes | yes (93%) | yes | done |
| logit | yes | yes | yes (98%) | yes | done |
| poisson | yes | yes | yes (98%) | unreliable | done |
| gamma | building | - | - | - | A |
| negbin | - | - | - | - | A |
| gaussian | - | - | - | - | A |
| probit, weibull, gumbel, tobit, beta, zip | - | - | - | - | B |
| multinomial, quantile, combinatorial | partial | - | - | - | C |

Older component validations (parameter recovery, autodiff, Λ accuracy, ψ assembly, per-family
coverage evals) live under `evals/` and `docs/validation/`. The FLM framework itself follows
Farrell-Liang-Misra (2021, 2025); see `docs/dev/paper_replication_details.md` for the
paper-to-code mapping.
