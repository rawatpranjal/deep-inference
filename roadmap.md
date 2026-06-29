# deep-inference roadmap

## CURRENT GOAL — certify Tier A (gamma, negbin, gaussian) on the cross-family benchmark

**The directive (frozen).** Wire gamma, negbin, and gaussian into `exploration/spike.py`'s
`DGPS` registry and certify FLM[cholesky] valid on each, rendering each as a tab in the
master dashboard. This is the first milestone of the larger cross-family benchmark below;
Tier B and Tier C stay in BACKLOG until this lands.

**Done when.** Each of the three families is registered with a confounded DGP, a custom
autodiff loss, an ATE `target_fn`, a Monte-Carlo true μ*, a statsmodels GLM oracle-MLE, and
a canonical-TMLE RieszNet outcome, and `exploration/build_dashboard.py` shows it as a tab.

**Acceptance (quantitative).** At M=100, on all three families: FLM[cholesky] coverage in
[93,97]%, SE-ratio in [0.9,1.1], |bias| small; Naive under-covers (proves the correction is
needed); the panel shows Oracle-MLE / FLM[cholesky] / FLM[ridge] / FLM[oracle-Λ] / RieszNet
/ Naive side by side with the full metric set.

**Acceptance (qualitative).** No truth-tuning. The cholesky path stays pure autodiff with no
hardcoded family closed-form (see [[memory]]). One truth-free tikhonov across families.
Fresh-agent (Opus) verified.

**Sequencing.** gamma first (log link, Gamma-deviance TMLE), smoke at M=3 local, then M=100
cert chunked on RunPod. negbin and gaussian follow the same pattern. The FLM[cholesky/ridge]
and oracle-MLE legs are cheap; the per-family work is the RieszNet TMLE fluctuation.

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

All of this lives on branch `night/general-lambda-perfect-scores`. **main is untouched,
pending the user's go-ahead for the squash-merge** (one commit, user as author). Before
declaring the cert done: run an existing eval as a regression check (changes are additive,
cholesky-only early-stop, `max_condition` default 100 unchanged), fresh-agent verify the
numbers were not tuned to truth, then merge. linear 93% sits within ~2.6pp Monte-Carlo
noise of the band edge at M=100, so it is the one thin spot.

## BACKLOG

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
