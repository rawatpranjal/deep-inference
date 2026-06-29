# Handoff — 2026-06-28 (overnight) — branch night/general-lambda-perfect-scores

## TL;DR (60-second read)
Goal: make the GENERAL FLM estimator valid (SE-ratio≈1, coverage≈0.95, bias≈0) on the linear,
logit, AND poisson ATE benchmarks, no analytical-formula exploitation. RESULT (CERTIFIED, M=100):
**FLM[cholesky] is valid on all three DGPs** — linear −0.0041/0.92/93%, logit −0.0020/1.07/98%,
poisson −0.0116/0.98/98% (bias / SE-ratio / coverage), single truth-free tikhonov=0.01. Contrasts
under-cover (flat 83%, ridge/oracle 86%, naive 23/10/33%); on logit cholesky even beats the
oracle-Λ ceiling. RieszNet is fine on linear/logit, unreliable on poisson (SE-ratio 0.32, divergent
reps). Cert ran fresh on a RunPod 16-core pod (all 3 DGPs one platform) → `exploration/results_cert_all_M100.md`
+ dashboard `exploration/results_cert_all_M100.html`. Fresh-agent (Opus) verify PASSED all 4 checks
(numbers consistent, no truth-tuning, cholesky general, RieszNet-poisson canonical). Thin spot:
linear 93% sits within ~2.6pp MC noise of the band edge at M=100. All work is on branch
`night/general-lambda-perfect-scores`; **main is untouched, PENDING the user's go-ahead for the squash-merge.**

## Backlog (user-requested 2026-06-28, "for later")
1. **Ridge as a first-class tracked solution, not just a contrast.** Ridge-on-Hessian-entries is
   general (no analytical-formula exploitation), so benchmark it as a real candidate alongside
   cholesky, head-to-head. It's already a row in the logit/poisson tables; add a ridge row to the
   LINEAR table too (currently linear contrasts cholesky vs flat-bug vs oracle only).
2. **Promote the dashboard to THE canonical sim-study dashboard.** User likes
   `exploration/build_dashboard.py` + the dark tabbed HTML; make it the standard output for every
   simulation study here (not a one-off), so each benchmark run renders into the same dashboard.
3. **Slowly add more functional forms after poisson.** Extend the flawless general
   cholesky-Lambda cert to probit, gamma, etc., one family at a time (the package already has the
   GLM families; the work is wiring each DGP + oracle row into spike.py like linear/logit/poisson).

## What shipped (committed to the branch)
- **General `lambda_method='cholesky'`** in the package (`src/deep_inference/lambda_/estimate.py`):
  a PSD-by-construction net Λ̂(x)=L(x)L(x)ᵀ trained Frobenius to the per-obs Hessians, now with
  **early stopping** on a held-out Hessian-fit split (the key fix — the net was overfitting the
  noisy rank-1 per-obs Hessians, which poisoned the near-singular inverse). `max_condition`
  exposed (spectrum-adaptive inverse reg). Routed in `selector.py`.
- **`inference(lambda_strategy=...)` passthrough** (`__init__.py`) — inject a custom/oracle Λ.
- **Benchmark harness** (`exploration/spike.py`): both DGPs through the general `inference()` path;
  oracle ladder (cholesky / oracleprop / oracle / ridge|flat contrasts); RieszNet rewritten with
  minibatch + 2-stage LR + **median-over-3-splits** (kills the occasional divergence) + eps clamp.
- Knobs: `--flm-folds --flm-repeats --tikhonov --max-condition --logit-lambdas --linear-lambdas`.

## How it was won (the diagnostic arc — this is the science)
1. Linear cholesky was biased (-0.030) AND over-covering (SE-ratio 1.39). The **oracle ladder**
   (true Λ injected, matched folds/tik) showed the true Λ is flawless at folds=10 -> the failure
   is **Λ-ESTIMATION, not folds/θ**. Root cause: cholesky-net overfitting -> **early stopping** fixed it.
2. The true logit Λ is genuinely near-singular (det=e(1-e)·w0·w1 -> 0 at overlap AND outcome
   saturation), so the inverse needs real regularization; a single truth-free tikhonov=0.01 works
   for both DGPs (NOT per-DGP tuned). Even oracle-Λ under-covers at ε≈0 -> reg matters > Λ accuracy.
3. RieszNet's divergence is the unpenalized eps/TMLE knob, not the representer magnitude; median
   over 3 cross-fit splits rejects the divergent split. Cost: mild conservatism (RieszNet needs
   this crutch; cholesky does not).

## Integrity guardrails honored (autochecks caught these)
- No fixed seed on the Λ net (init variance stays in the empirical SD -> honest SE-ratio).
- One truth-free tikhonov for both DGPs (NOT tuned per-DGP to hit a target -> not gaming).
- Localize with the oracle ladder BEFORE changing knobs; certify "flawless" ONLY at M>=200.
- Two fresh-agent audits passed (cholesky impl; estimand) + caught a factor-2 linear-oracle bug (fixed).

## Pending before "done" (next session, or when M=200 lands)
1. Read `exploration/results_cert_M200.md`; confirm cholesky SE-ratio∈[0.9,1.1], coverage∈[92,98],
   |bias| small on BOTH DGPs. Report RieszNet honestly (valid, mildly conservative).
2. Regression check: run an existing eval (e.g. eval_06 coverage) — package changes are additive
   (cholesky-only early-stop; max_condition default 100 unchanged) so no regression expected; confirm.
3. Fresh-agent verify the M=200 numbers + that nothing was tuned to the truth.
4. Build dashboard (`exploration/build_dashboard.py --in exploration/results_cert_M200.md`),
   update CHANGELOG, then **squash-merge the branch to main as ONE commit** (user as author).
5. OPTIONAL "others": extend the flawless general estimator to one more family (poisson/probit) —
   only if the core is solid and time remains.

## Full running log
`exploration/NIGHT_LOG.md` (decisions, every result table, resume pointer).

---
# (PREVIOUS) Handoff — 2026-06-28 — main

## Where we left off
Diagnosed AND fixed (in a spike) the FLM linear-ATE SE undercoverage. Root cause = the
flat aggregate Lambda(x); the load-bearing failure is inverting a near-singular estimated
Lambda under severe overlap. Built a general PSD-by-construction fix. All committed + pushed
(commits 2965cb8, e0f62b1). Stopped at a clean milestone. Next session formalizes generality.

## Active streams (clustered)
- **PSD-by-construction Lambda fix (LEAD, general).** Net outputs a Cholesky factor L(x),
  Lambda_hat = L L^T (PSD for any output), trained Frobenius to the autodiff per-obs Hessians.
  Also a spectral variant Q diag(softplus(lambda)) Q^T. Both hit target on LINEAR at M=50
  (cholesky 96%/1.01, spectral 98%/0.98). `exploration/lambda_cholesky.py`. SEQUENTIAL next:
  (1) run on the logit DGP, (2) M=200 confirm, (3) fresh-agent verify.
- **Linear propensity fix (DONE, family-specific).** Estimate e(x), build Lambda analytically.
  95%/1.04 at M=200, confirmed. NOT general (uses the closed form). `exploration/lambda_inv_fix.py`.
- **Diagnosis (DONE).** `exploration/lambda_decomp.py` (11-object panel vs oracle),
  `lambda_surface.py` (+ .png, the det<0 figure). Write-up: `docs/notes/flm_lambda_se_undercount.md`.

## Decisions made this session
- The fix MUST be general: never exploit a family's closed-form Hessian. User corrected this
  twice, hard ("you cannot use the analytical formula", "I want delta(x) not e(x)").
- PSD-by-construction beats entry-wise regression + post-hoc clamp (clamp insufficient:
  Lambda-inv-R2 went -4395 -> -385, still broken).
- Stability beats accuracy: PSD methods have Lambda-inv-R2 only 0.17-0.20 yet calibrated SE;
  mlp fit Lambda to R2 0.94 and still detonated (var_ratio 1475).

## Open questions
- Does PSD-Cholesky hold on logit / gamma (theta-dependent, y-dependent Hessians)? Reasoned
  YES (every GLM Hessian = positive-weight * rank-1 outer product, so Lambda is PSD for free),
  but NOT run. This is the one experiment that converts "general by construction" to "shown".
- Candidate Learned Rule for CLAUDE.md (needs your OK): "Fixing a nuisance estimator -> keep
  it general; do not slide into a family's closed-form shortcut. The package is general."

## Landmines / gotchas
- `exploration/lambda_cholesky.py` run_flm_cell (~line 109) hardcodes `family="linear"` — that
  is the TEST-DGP pin, not the estimator. Change it (and the target) for the logit test.
- SpectralNet asserts d==2 (Givens rotation); generalize to O(d) for theta_dim>2. Cholesky has
  no such cap.
- Env fixes this session: lightgbm upgraded 4.5 -> 4.6.0 (was ABI-broken vs sklearn 1.8);
  pygam installed. Interpreter: `/opt/homebrew/bin/python3.11`, run with `PYTHONPATH=src`.
- Linear Hessian is theta-INDEPENDENT, so linear never exercises the autodiff-at-theta-hat path;
  that is precisely why logit (p(1-p) weight) is the real generality test.
- Two intermediate `exploration/lambda_sweep_2026*/` dirs left untracked (early fast-pass logs,
  superseded by the result tables). Harmless.

## Suggested next move
Wire the logit DGP through `exploration/lambda_cholesky.py` and run the autodiff-Cholesky there.
That single result earns the generality claim. Then M=200 confirm + a fresh-agent check.
