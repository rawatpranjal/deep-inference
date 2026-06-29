# deep-inference roadmap

## CURRENT GOAL — paper-vs-package replication ledger for FLM and RieszNet

**The directive (frozen).** Build a Replications section in the RTD docs and a root
`replications.md` that replicate the published Monte-Carlo simulations of (1) Farrell, Liang,
Misra (2021, 2025) and (2) RieszNet (Chernozhukov et al. 2022). Run THIS package's own
estimators on each paper's DGP and present **Package-vs-Paper** tables side by side, in the
econirl `replications.html` style: an intro ledger line, then one section per replicated
simulation, each a `Quantity | Package | Paper` table plus the exact reproduce command. The
prior Tier-A cert goal is paused (parked in BACKLOG); its in-flight state stays in STATUS.

**Deliverables.**
1. Restructure `docs/replications/index.md` into the paper-ledger style. Lead line:
   "Each section sets the package value against the paper's published value, side by side."
   Keep the existing known-truth MC benchmark as a clearly-labelled second part. Delete
   nothing (project rule 2).
2. `replications.md` at repo root — the working source-of-truth markdown the harness writes
   and the RTD page curates.
3. `exploration/replicate_papers.py` — the harness that runs the package on each paper's
   exact DGP, with the paper's published numbers transcribed (quoted from the docling source)
   as the Paper column, saving a timestamped report under `exploration/`.

**Scope (minimum, both papers).**
- FLM: at least one Monte-Carlo table — the ATE / CI-coverage simulation in the FLM2021 Monte
  Carlo section (`references/FLM2021_docling.md`) — package coverage, RMSE, bias vs the
  paper's reported values, via `inference()` / `structural_dml()`.
- RieszNet: at least the IHDP binary-treatment ATE experiment (Section 5,
  `references/RieszNet2022_docling.md`) — bias, RMSE, coverage, CI length vs the paper's
  table, via the package's RieszNet estimator.

**Done when.** Both papers have at least one paper-simulation replicated with a
Package-vs-Paper table; every Paper-column number is a verbatim transcription from the paper
text on disk (`references/*_docling.md`, quotable span); every Package-column number comes
from a saved harness run whose report path is shown; `docs/replications/index.md` builds in
the RTD toctree; and a fresh Opus agent confirms the paper numbers are correctly quoted and
the package numbers are from real output, not cooked.

**Acceptance (quantitative).** For the headline quantity of each replicated table (coverage),
the package value lands within a stated, honest band of the paper's value — or the gap is
reported plainly with its cause. No truth-tuning. Show every number (SHOW ME THE FACTS).

**Acceptance (qualitative).** econirl style (intro ledger line, per-section paragraph + table
+ reproduce command). Nothing deleted from the existing replications page. Fresh-agent (Opus)
verified.

**Success condition (narrower than the whole roadmap).** Goal is met once the three
deliverables exist, both papers have one replicated Package-vs-Paper table backed by a saved
run, the RTD page builds, and the fresh-agent verification passes. Then `/autoloop` returns
to the paused Tier-A cert.

---

The larger benchmark this milestone serves: one harness that, for every estimator family,
runs the full method panel on a known-truth DGP and logs every metric to one place. Extends
the existing 3-family spike (`exploration/spike.py`) to the whole family library.

**Method panel (per family, per replication).**

| Method | Role |
|--------|------|
| Oracle-MLE | gold-standard anchor (correctly-specified parametric fit + delta SE) |
| FLM[cholesky] | the general fix, PSD-by-construction Λ̂(x)=L(x)L(x)ᵀ |
| FLM[ridge] | general entry-wise Λ̂ contrast (breaks PSD at low overlap) |
| FLM[oracle-Λ] | ceiling diagnostic, true Λ(x) injected (optional, bespoke per DGP) |
| RieszNet | doubly-robust competitor (TMLE-targeted Riesz representer) |
| Naive | plug-in g(1,X)-g(0,X), no correction |

**Metrics (M replications).** mean estimate, bias, empirical SE, mean estimated SE,
SE-ratio, coverage. Truth is μ* = ATE = E[g(1,X) - g(0,X)] on the natural scale,
Monte-Carlo or closed-form per DGP. Valid = coverage 93-97% with SE-ratio near 1.

**Acceptance.** Each family in scope appears as one `### <name> DGP` section in the master
results markdown, FLM[cholesky] hits the validity band, and a fresh agent confirms no
truth-tuning and no hardcoded family closed-form in the cholesky path.

**Per-family build contract.** Supply seven things; the harness does the rest.
1. confounded DGP generator `gen(n, rng) -> (Y, T, X)`
2. custom `loss(y, t, theta)` (autodiff gives score and Hessian, no hand formulas)
3. ATE `target_fn(x, theta, t_tilde)` on the natural scale
4. true μ* (Monte-Carlo from the DGP, or closed form)
5. correctly-specified oracle-MLE `oracle(Y, T, X) -> (mu, se)` (statsmodels GLM where it fits)
6. RieszNet outcome spec (link + regression loss + TMLE fluctuation) — the bottleneck
7. optional oracle-Λ class (ceiling only, skippable)

Items 2 and 3 make FLM[cholesky] and FLM[ridge] work with zero extra code (already general
via autodiff). The generic FLM wrapper template is `flm_poisson` at `exploration/spike.py:620`.
Register in the `DGPS` dict at `exploration/spike.py:656` plus the supporting `DEFAULT_SPECS`
/ `DGP_BASE_SEED` / `TRUTH_FN` / `*_TIKHONOV` maps.

**Rollout tiers (ranked).**
- **[CRITICAL] Tier A — canonical-link GLMs.** gamma, negbin, gaussian. Clean canonical TMLE
  (log-mean fluctuation like poisson, identity like linear) and a statsmodels oracle, so the
  full principled panel including RieszNet drops in cheaply. Do this first.
- **Tier B — non-canonical link.** probit, weibull, gumbel, tobit, beta, zip. FLM+Oracle
  run everywhere for free. RieszNet targeting is bespoke, so run it best-effort and clearly
  mark families where no defensible fluctuation is included.
- **Tier C — multi-dim θ or non-ATE target.** multinomial_logit, quantile, combinatorial.
  Different target functional, so FLM-only for now. The RieszNet representer here is a
  research item, not plumbing.

**Logging (one place).** Each family is one `### <name> DGP` section in a master results
markdown. `exploration/build_dashboard.py` renders one dark-HTML tab per section, no edits
needed. Chunk-merge (`--rep-offset` / `--dump-raw` / `--from-raw`) keeps M exact across
disjoint sub-60min cloud chunks, verified byte-identical to a single run.

**Decisions settled for this milestone.**
1. Scope and order: Tier A first (gamma, negbin, gaussian), gamma leading.
2. M and platform: smoke at M=3 local, validate at M=50 local, certify at M=100 chunked on
   RunPod (the cert standard).

**Still open (deferred to when Tier B starts).** RieszNet policy on the non-canonical
families (probit, weibull, tobit, beta, zip, quantile): skip-and-mark, force generic
mean-scale targeting as an honest under-covering contrast, or derive correct targeting per
family. Not a blocker for Tier A, whose families all have a canonical TMLE form.

**Key files.** `exploration/spike.py` (registry + method panel + chunking),
`exploration/build_dashboard.py` (tabbed HTML), `src/deep_inference/lambda_/estimate.py`
(general cholesky/ridge Λ), `src/deep_inference/__init__.py` (`inference()` entry point).

## Done log (replication ledger goal)

- **repl-02-riesz-leg** (2026-06-29, VERIFIED, merged to docs/rebuild-rtd). `riesz` mode of
  `exploration/replicate_papers.py` replicates the RieszNet (Chernozhukov et al. 2022) IHDP
  binary-ATE experiment via the package's `riesz_inference` (outcome='linear'), over N=50 of
  the IHDP-1000 semi-synthetic realizations (the public Dragonnet mirror ships 50). **Package
  vs Paper:** MAE **0.124** vs paper **0.110** (Table 1 DR RieszNet, directly comparable,
  close); coverage **0.98** vs paper **0.95** (NOT directly comparable - the paper redraws T
  per propensity 'True', the harness uses the CSVs' original confounded T, so MAE is the
  comparable headline). No leakage: the estimator sees only (Y,T,X); mu0/mu1 score MAE/coverage
  only. Fresh-Opus verified the IHDP column mapping and the verbatim RieszNet-not-ForestRiesz
  transcription (MAE 0.110 line 219, coverage 0.95 line 227). Report:
  `exploration/replicate_papers_riesz_full.md`.

- **repl-01-flm-leg** (2026-06-29, VERIFIED, merged to docs/rebuild-rtd). `exploration/replicate_papers.py`
  replicates the FLM2021 Section-6 Monte Carlo (d=20, linear outcome, n=10000, M=200) via
  `structural_dml(family='linear')`, arch 2 = [60,30,20]. **Package vs Paper:** randomized
  Bias -0.0020 / IL 0.078 / Coverage **0.955** (paper -0.00032 / 0.079 / **0.951**) — clean
  match. Observational Bias -0.0173 / IL 0.081 / Coverage **0.840** (paper 0.00011 / 0.079 /
  **0.946**) — honest gap. The gap is the documented linear two-way Λ(x)-collapse
  (`docs/notes/flm_lambda_se_undercount.md`): a flat-Λ undercounts the SE and leaves a small
  residual confounding bias, firing ONLY when e(x) varies; the randomized arm (constant e=0.5)
  is a clean 95.5% control. Scratchpad finding: at n=10000 the under-coverage is bias-driven
  (bias ≈ 0.85·SE), and the 3-way Λ path does not fix it (bias worsens). Fresh-Opus verified
  DGP fidelity + verbatim Table 6/7 transcription (two passes each), no truth leakage. Reports:
  `exploration/replicate_papers_flm_full.md`, `exploration/replicate_papers_flm_smoke.md`.

## STATUS

linear, logit, poisson wired and certified at M=100 on RunPod (commit 12524f1).
FLM[cholesky] valid on all three (linear 93%/0.92, logit 98%/1.07, poisson 98%/0.98),
no truth-tuning, pure autodiff, fresh-agent verified. RieszNet reliable on linear/logit,
unreliable on poisson (SE-ratio 0.32, divergent reps). Next is Tier A (gamma, negbin,
gaussian). The three open decisions above are unresolved.

**gamma (Tier A) wired and run at M=50 (2026-06-29).** FLM[cholesky] FAILS the band:
bias -0.072, SE-ratio 0.76, coverage 88% (`exploration/results_gamma_M50.md`). Localized by
the oracle ladder to Λ-ESTIMATION: with exact Λ ([[1,e],[e,e]]) the estimator is centered
(median 0.928 vs truth 0.898, coverage 90%), but the cholesky Λ̂-net misfits gamma's noisy
`y/μ`-weighted per-obs Hessians (CV~0.7 at shape k=2) on ~8/50 reps, producing a left tail
(estimates down to 0.26) that biases low and inflates the SE. RieszNet is valid on gamma
(96%, SE-ratio 1.04). So gamma exposes that the general cholesky path needs hardening for
noisy y-dependent Hessians. Next: a legitimate cholesky-net robustness improvement (more
regularization / better early-stop, NOT tikhonov-tuned-to-truth) in
`src/deep_inference/lambda_/estimate.py`, verified to NOT regress linear/logit/poisson.

**negbin (Tier A) wired and run at M=50 (2026-06-29).** Richer than gamma
(`exploration/results_negbin_M50.md`). FLM[cholesky] bias -0.111, SE-ratio 0.88, coverage 96%
- but the 96% is a FRAGILE pass, not clean: the estimator is biased low (12% of truth) with a
big left tail (11/50 reps below 0.7, down to 0.24), and the wide CIs cover by luck. TWO
effects, separable by the oracle ladder. (a) A baseline downward bias of ~-0.045 on the
heavy-tailed exp-mean functional, present even with exact Λ (-0.047) and in RieszNet (-0.041)
but NOT in the parametric Oracle-MLE (+0.008); it grows with outcome variance across the
log-link families (poisson -0.012, gamma -0.072, negbin -0.111). Hypothesis: neural-nuisance
shrinkage on the convex E[exp(η)] (Jensen), an undersmoothing/nuisance-bias issue, NOT a Λ
issue. (b) The cholesky-Λ left tail ON TOP (cholesky -0.111 vs oracle-Λ -0.047), the same
noisy y-dependent Hessian failure gamma showed. So the cholesky-Λ failure is GENERAL (gamma +
negbin), and the heavy-tailed count families expose a second, deeper nuisance-bias effect.

**gaussian (Tier A) the control, M=50 (2026-06-29).** Decisive
(`exploration/results_gaussian_M50.md`). oracle-Λ bias +0.0014 / SE-ratio 1.03 / coverage 96%
and RieszNet +0.0003 / 0.96 / 96% are BOTH essentially perfect. cholesky is mild (+0.051 /
0.76 / 92%, no left tail, min 0.630). This isolates the two effects: (a) the neural nuisance
bias is COUNT-FAMILY-SPECIFIC, gaussian's RieszNet is unbiased, confirming Jensen shrinkage on
E[exp(η)] for the log-link families, not a general neural problem; (b) the 3-dim θ / 3x3 Λ
path works with exact Λ (oracle-Λ 96%), so θ_dim>2 is fine. Consolidated dashboard of all
three at `exploration/results_tierA_M50.html`.

**Tier A verdict.** Two separable problems block certifying cholesky on the count families.
1. Cholesky-Λ noisy-Hessian fragility: severe on gamma/negbin (noisy y-dependent TARGET-block
   weight), mild on gaussian (noisy σ-block orthogonal to the target). Fix = harden the
   cholesky Λ̂-net. 2. Neural-nuisance shrinkage bias on E[exp(η)], log-link heavy-tailed
   only, hits RieszNet and oracle-Λ too (so NOT a Λ issue). Fix = undersmooth / bias-correct
   the θ̂ net for log-link families. gaussian is essentially done (oracle-Λ + RieszNet valid,
   cholesky needs the same Λ-hardening). Both fixes touch the core package and are the
   user's call to greenlight.

**Landscape mapped: probit + beta (Tier B), M=50 (2026-06-29).** Both PASS cholesky cleanly.
probit (bounded link like logit) bias -0.010 / SE-ratio 1.16 / coverage 98%, no left tail;
RieszNet 98% (`exploration/results_probit_M50.md`). beta (bounded CONTINUOUS, y-dependent
Hessian) bias -0.0004 / 1.02 / 96%, tight spread; RieszNet 100% (`results_beta_M50.md`).
**What is measured: cholesky fails on exactly gamma and negbin among the 8 families; it
passes on linear, gaussian (marginal), logit, probit, beta, poisson.** Two falsifications:
beta passes despite a y-dependent Hessian on a continuous outcome (so neither is sufficient to
cause the failure), and poisson passes despite a log link (so log link is not sufficient).
gamma + negbin differ from the passing families on SEVERAL confounded axes at once (heavy
tail, log link, y-dependent Hessian weight, the convex exp target). Which axis is operative is
NOT isolated. To get a causal answer, vary one axis at a time, e.g. sweep gamma's shape k
(outcome variance / weight noise) and see whether the failure tracks it. The narrative about
"noisy weight severity + outcome variance" is an untested hypothesis, not a result. Consolidated dashboard `exploration/results_landscape.html`. Known
ceiling artifact: `OracleProbitLambda` blows up under tikhonov 1e-8 (near-singular probit
Fisher at extreme η), a diagnostic-row bug, not a method failure. Cheap GLM families still
unmapped: zip (`ZeroInflatedPoisson` oracle available). Expensive (custom MLE oracle):
weibull, gumbel, tobit. Research (different target): multinomial, quantile, combinatorial.

All of this lives on branch `night/general-lambda-perfect-scores`. **main is untouched,
pending the user's go-ahead for the squash-merge** (one commit, user as author). Before
declaring the cert done: run an existing eval as a regression check (changes are additive,
cholesky-only early-stop, `max_condition` default 100 unchanged), fresh-agent verify the
numbers were not tuned to truth, then merge. linear 93% sits within ~2.6pp Monte-Carlo
noise of the band edge at M=100, so it is the one thin spot.

**Lineage (folded from the old STATUS.md, 2026-06-27).** This goal followed a positioning
phase: an 86-paper causal-deep-learning corpus (2016-2026) and `docs/dev/competitors.md`
mapping PyWhy / EconML / DoubleML / CausalML against this package on 16 method families. The
white space identified was torch-native estimators with IF standard errors across causal AND
structural-econ targets, which is what the build-next shortlist in BACKLOG draws from.

## BACKLOG

- **[PAUSED — prior CURRENT GOAL]** Certify Tier A (gamma, negbin, gaussian) cholesky on the
  cross-family benchmark at M=100. Blocked by two separable fixes (cholesky-Λ noisy-Hessian
  hardening + log-link nuisance-bias correction), both in the core package, user's call to
  greenlight. Full diagnosis in STATUS and the body above. Resume after the replication ledger.
- Tier A families: gamma, negbin, gaussian (full panel incl. canonical RieszNet).
- Tier B families: probit, weibull, gumbel, tobit, beta, zip (FLM+Oracle, RieszNet best-effort).
- Tier C families: multinomial_logit, quantile, combinatorial (FLM-only).
- Master results store that accumulates a section per family across runs and platforms.
- Ridge as a first-class TRACKED candidate, not just a contrast (it is general, no
  closed-form exploitation). Add a ridge row to the LINEAR table too (currently cholesky
  vs flat vs oracle only). [user-requested 2026-06-28]
- Promote `exploration/build_dashboard.py` to THE canonical sim-study dashboard, the
  standard output for every simulation study here, not a one-off. [user-requested 2026-06-28]
- Harden RieszNet poisson (divergent reps blow up the empirical SE).
- From `docs/dev/competitors.md` shortlist: orthogonal DR deep learners, Deep IV / DeepGMM,
  representation-learning CATE with valid CIs, deep survival CATE, doubly-robust DiD.
