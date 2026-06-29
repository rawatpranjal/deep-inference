# deep-inference roadmap

## CURRENT GOAL — cross-family estimator benchmark

One harness that, for every estimator family, runs the full method panel on a known-truth
DGP and logs every metric to one place. Extends the existing 3-family spike
(`exploration/spike.py`) to the whole family library.

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

**Open decisions (need the user's call).**
1. Family scope and order. Tier A first as a clean milestone, or all families at once.
2. RieszNet policy on the hard families. Skip-and-mark where targeting is non-canonical,
   force generic mean-scale targeting as an honest under-covering contrast, or derive correct
   targeting per family (research-heavy).
3. M and platform. RunPod cert standard is M=100 chunked; M=50 local validates wiring first;
   M=200 tightens the coverage SE.

**Key files.** `exploration/spike.py` (registry + method panel + chunking),
`exploration/build_dashboard.py` (tabbed HTML), `src/deep_inference/lambda_/estimate.py`
(general cholesky/ridge Λ), `src/deep_inference/__init__.py` (`inference()` entry point).

## STATUS

linear, logit, poisson wired and certified at M=100 on RunPod (commit 12524f1).
FLM[cholesky] valid on all three (linear 93%/0.92, logit 98%/1.07, poisson 98%/0.98),
no truth-tuning, pure autodiff, fresh-agent verified. RieszNet reliable on linear/logit,
unreliable on poisson (SE-ratio 0.32, divergent reps). Next is Tier A (gamma, negbin,
gaussian). The three open decisions above are unresolved.

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
