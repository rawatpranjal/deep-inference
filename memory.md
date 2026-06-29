# deep-inference memory

Long-term decisions and learnings. A line earns a place only if a future session would get
it wrong without it.

## The nuisance-estimator fix must stay GENERAL

When fixing a nuisance estimator (the Λ(x) curvature estimator), never slide into a family's
closed-form Hessian or propensity. The package is general. The user corrected this twice,
hard ("you cannot use the analytical formula", "I want Λ(x) not e(x)"). The cholesky path
must be pure autodiff with no hardcoded family formula. Closed-form oracle-Λ classes exist
only as ceiling diagnostics, never as the shipped estimator.

## Why the general Λ̂(x) fix works (the science, why we trust it)

- **PSD-by-construction beats entry-wise regression + post-hoc clamp.** Regressing Hessian
  entries independently breaks PSD, and clamping eigenvalues after the fact is insufficient
  (Λ-inv-R² went -4395 to -385, still broken). The cholesky net outputs L(x) and forms
  Λ̂=L(x)L(x)ᵀ, PSD for any output. Every GLM Hessian is a positive-weight rank-1 outer
  product, so the true Λ is PSD for free.
- **Stability beats accuracy.** mlp fit Λ to R² 0.94 and still detonated (var-ratio 1475).
  PSD methods have Λ-inv-R² only 0.17-0.20 yet give calibrated SE. The load-bearing object
  is Λ⁻¹, not Λ.
- **The cholesky net was overfitting** the noisy rank-1 per-obs Hessians, which poisoned the
  near-singular inverse. Early stopping on a held-out Hessian-fit split is the fix.
- **Regularization matters more than Λ accuracy at low overlap.** The true logit Λ is
  genuinely near-singular (det = e(1-e)·w0·w1 to 0 at overlap AND outcome saturation), so the
  inverse needs real regularization. Even the oracle Λ under-covers at tikhonov approximately 0.
  A single truth-free tikhonov=0.01 works across linear/logit/poisson, NOT tuned per-DGP.
- **FLM2021's own MC corroborates the Λ-collapse (2026-06-29).** Replicating the paper's
  Section-6 DGP through `structural_dml(family='linear')` at n=10000, M=200: the randomized arm
  (constant e=0.5) matches the paper cleanly (coverage 0.955 vs 0.951, IL 0.078 vs 0.079), but
  the observational arm under-covers (0.840 vs 0.946). At n=10000 the under-coverage is
  bias-driven (residual confounding bias ≈ 0.85·SE), not pure SE-miscalibration, and
  `three_way=True` does NOT fix it. The randomized arm is the clean control that rules out an
  outcome-model / ATE / transcription bug. Harness: `exploration/replicate_papers.py`.

## Methodology discipline (how to certify honestly)

- **Localize before changing knobs.** Use the oracle ladder (inject true Λ, match folds/tik)
  to prove whether a failure is Λ-estimation vs folds/θ BEFORE touching hyperparameters.
- Certify "valid" only at M>=200 (or report M=100 with the Monte-Carlo noise band stated).
- No fixed seed on the Λ net, so init variance stays in the empirical SD and the SE-ratio is
  honest. One truth-free regularization setting across DGPs, never tuned to hit a target.
- RieszNet's divergence is the unpenalized eps/TMLE knob, not the representer magnitude.
  Median over 3 cross-fit splits rejects the divergent split, at the cost of mild conservatism.
  cholesky needs no such crutch.

## Gotchas that bite

- Interpreter is `/opt/homebrew/bin/python3.11`, run with `PYTHONPATH=src`. lightgbm must be
  4.6.0 (4.5 was ABI-broken vs sklearn 1.8); pygam is installed.
- The linear Hessian is θ-independent, so linear never exercises the autodiff-at-θ̂ path.
  logit (with its p(1-p) weight) is the real test of cholesky generality, not linear.
