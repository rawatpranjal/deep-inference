# Replications

This page records what the package has validated against both published paper benchmarks and
known-truth synthetic DGPs. Each section sets the package value against the paper's published
value, side by side.

```{toctree}
:hidden:

did_candidates
```

Scoping in progress: [DID + deep-learning replication candidates](did_candidates.md) surveys
difference-in-differences / observational-panel methods that could be added next.

## Part 1 -- Paper replications (package vs paper)

These run the package's own estimators on the published papers' exact simulation DGPs and
set the result against the paper's reported numbers.

### Farrell, Liang and Misra (2021) -- Monte-Carlo ATE coverage

FLM2021 Section 6 DGP: partially-linear model Y = mu0(X) + tau(X)*T + eps, n=10000, d=20
covariates, true ATE = E[tau(X)]; run through `structural_dml(family='linear')` with hidden
architecture [60,30,20] (the paper's MC architecture 2), M=200 replications. Two designs:
randomized (constant propensity p=0.5) and observational (logistic propensity).

**Randomized (constant propensity p=0.5)**

| Quantity | Package | Paper |
|---|---|---|
| Bias | -0.00199 | -0.00032 |
| Interval length | 0.078 | 0.079 |
| Coverage | 0.955 | 0.951 |

**Observational (logistic propensity)**

| Quantity | Package | Paper |
|---|---|---|
| Bias | -0.01733 | 0.00011 |
| Interval length | 0.081 | 0.079 |
| Coverage | 0.840 | 0.946 |

The randomized design matches the paper (0.955 vs 0.951). The observational design
under-covers (0.840 vs 0.946) because the linear default collapses the covariate-varying
Lambda(x) to a global-mean matrix when the propensity e(x) varies -- the documented
SE-undercount; see `docs/notes/flm_lambda_se_undercount.md`. At n=10000 the gap is
bias-driven.

Reproduce: `python exploration/replicate_papers.py --mode flm --full`

### RieszNet (Chernozhukov et al. 2022) -- IHDP average treatment effect

RieszNet Section 5.1: 1000 IHDP semi-synthetic datasets (747 obs, binary T, 25 confounders);
the package's RieszNet (`deep_inference.riesz.riesz_inference`) evaluated over N=50 of the
public Dragonnet IHDP realizations. MAE = mean|theta_hat - ATE| over realizations.

| Quantity | Package | Paper |
|---|---|---|
| MAE | 0.124 | 0.11 |
| Coverage | 0.98 | 0.95 |

MAE (Table 1) is directly comparable. The coverage row is NOT directly comparable: the paper
redraws T per propensity 'True' from NPCI for its coverage figure, while the harness uses the
IHDP CSVs' original confounded T. N=50 is also below the paper's 1000 realizations.

Reproduce: `python exploration/replicate_papers.py --mode riesz --full`

### Caveats (scale)

These Part 1 runs are scaled down from the papers. FLM uses M=200 replications and RieszNet
N=50 datasets, both below the papers' 1000, so the numbers carry Monte-Carlo noise (coverage
SE roughly 1.5pp at M=200, ~3pp at N=50) and small gaps to the paper values should not be
over-read. The RieszNet IHDP run uses the public Dragonnet mirror, which ships only the first
50 of the 1000 semi-synthetic realizations; N=50 is the mirror ceiling, not a chosen
subsample, and the full-1000 NPCI generation would need the R `NPCI` package.

## Part 2 -- Known-truth Monte-Carlo benchmark (package vs known truth)

The sections below compare the package against a known-truth synthetic DGP rather than a
paper's published numbers. A method is **valid** when its coverage sits in roughly [93,97]%
with an SE-ratio near 1 and small bias. The raw harness output lives in `exploration/` (the
master results markdown plus the dark tabbed dashboard from `build_dashboard.py`); this page
is the curated summary.

The full panel runs Oracle-MLE (gold anchor), FLM[cholesky] (the general PSD-Lambda fix),
FLM[ridge] and FLM[oracle-Lambda] (contrasts and ceiling), RieszNet (the doubly-robust
competitor), and Naive (no correction, expected to under-cover).

### FLM general Lambda-hat(x) certification, M=100 on RunPod

The headline result. FLM[cholesky] is the general, autodiff-only fix (PSD-by-construction
Lambda-hat(x)=L(x)L(x)^T, no hardcoded family formula), and it is valid on all three families
at a single truth-free tikhonov=0.01. Source: `exploration/results_cert_all_M100.md`,
commit 12524f1, fresh-agent (Opus) verified.

#### Linear (truth 1.0000)

| method | bias | SE-ratio | coverage |
|---|---|---|---|
| Oracle-MLE | -0.0008 | 0.96 | 96% |
| FLM[cholesky] | -0.0041 | 0.92 | 93% |
| FLM[oracle-Lambda] | -0.0030 | 0.96 | 95% |
| RieszNet | -0.0083 | 1.09 | 97% |
| Naive | -0.0042 | 0.17 | 23% |

#### Logit (truth 0.1481)

| method | bias | SE-ratio | coverage |
|---|---|---|---|
| Oracle-MLE | +0.0054 | 1.02 | 94% |
| FLM[cholesky] | -0.0020 | 1.07 | 98% |
| FLM[ridge] | +0.0166 | 0.92 | 86% |
| FLM[oracle-Lambda] | +0.0167 | 0.90 | 86% |
| RieszNet | +0.0062 | 1.06 | 100% |
| Naive | +0.0069 | 0.11 | 10% |

On logit, FLM[cholesky] beats even the oracle-Lambda ceiling (98 vs 86), because its implicit
regularization helps where the true Lambda is near-singular at low overlap.

#### Poisson (truth 0.8981)

| method | bias | SE-ratio | coverage |
|---|---|---|---|
| Oracle-MLE | -0.0058 | 0.98 | 99% |
| FLM[cholesky] | -0.0116 | 0.98 | 98% |
| FLM[ridge] | -0.0325 | 1.07 | 94% |
| FLM[oracle-Lambda] | -0.0403 | 1.02 | 95% |
| RieszNet | +0.0382 | 0.32 | 98% |
| Naive | -0.0177 | 0.25 | 33% |

RieszNet is unreliable on poisson: its empirical SE is blown up by divergent reps
(SE-ratio 0.32), so the 98% coverage is not trustworthy. FLM[cholesky] is the least-biased
valid method here.

**Caveat.** Linear 93% sits within about 2.6pp of Monte-Carlo noise of the band edge at
M=100. An M=200 confirm would tighten it.

### Status by family

| family | known-truth DGP | oracle-MLE | FLM[cholesky] valid | RieszNet | tier |
|---|---|---|---|---|---|
| linear | yes | yes | yes (93%, M=100) | yes | done |
| logit | yes | yes | yes (98%, M=100) | yes | done |
| poisson | yes | yes | yes (98%, M=100) | unreliable | done |
| gamma | yes | yes | no (88%, bias -0.07, M=50) | yes (96%) | A, blocked |
| negbin | yes | yes | no (bias -0.11, fragile 96%, M=50) | yes (96%) | A, blocked |
| gaussian | yes | yes | marginal (92%, M=50); oracle-Lambda + RieszNet 96% | yes (96%) | A, ~done |
| probit | yes | yes | yes (98%, M=50) | yes (98%) | B, done |
| beta | yes | yes | yes (96%, M=50) | yes (100%) | B, done |
| zip | oracle ready | yes | - | - | B |
| weibull, gumbel, tobit | needs custom MLE oracle | - | - | - | B |
| multinomial, quantile, combinatorial | partial | - | - | - | C |

### The landscape (2026-06-29)

Eight GLM families mapped. Measured result: FLM[cholesky] (the general fix) fails on exactly
gamma and negbin; it passes on linear, gaussian (marginal), logit, probit, beta, poisson. Two
falsifications come for free: beta passes despite a y-dependent Hessian on a continuous
outcome (neither is sufficient to cause the failure), and poisson passes despite a log link
(log link is not sufficient). gamma and negbin differ from the passing families on several
confounded axes at once (heavy tail, log link, y-dependent Hessian weight, convex exp target);
which axis is causal is NOT yet isolated. The clean test is to sweep gamma's shape k and see
whether the failure tracks outcome variance. Until then, treat the cause as open. Dashboard:
`exploration/results_landscape.html`.

### Tier A, M=50 (2026-06-29)

gamma, negbin, gaussian wired into the benchmark (`exploration/results_tierA_M50.html`).
The result is a two-part diagnosis, not a clean cert. (1) FLM[cholesky] is fragile on the
heavy-tailed log-link count families (gamma bias -0.072 / coverage 88%, negbin bias -0.111 /
fragile 96%): the cholesky Lambda-hat-net misfits the noisy y-dependent per-obs Hessians, a
left tail of bad-Lambda reps that the exact oracle-Lambda does not have. (2) Those same count
families carry a separate downward nuisance bias of ~-0.045 that hits even RieszNet and
oracle-Lambda (Jensen shrinkage on E[exp(eta)]), absent on gaussian. The gaussian control
confirms both are count-family-specific: its oracle-Lambda (+0.001 / 96%) and RieszNet
(+0.000 / 96%) are perfect, and the 3-dim theta / 3x3 Lambda path works. Certifying cholesky
on gamma/negbin needs a cholesky-Lambda hardening plus a log-link nuisance-bias fix, both in
the core package.

Older component validations (parameter recovery, autodiff, Lambda accuracy, psi assembly,
per-family coverage evals) live under `evals/` and `docs/simulation_studies/`. The FLM
framework itself follows Farrell-Liang-Misra (2021, 2025); see
`docs/dev/paper_replication_details.md` for the paper-to-code mapping.
