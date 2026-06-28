# Handoff — 2026-06-28 — main

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
