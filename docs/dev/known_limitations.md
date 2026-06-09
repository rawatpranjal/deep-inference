# Known limitations & triage notes

**Read this before assuming a coverage FAIL is a bug.** Most coverage shortfalls in this
package are finite-sample / regularization-bias *signatures*, not implementation errors.
The math (loss/score/Hessian for all families, leak-free cross-fitting, IF assembly) is
hand-verified; the core validated path (`structural_dml` + eval_06/09/12) holds ~95-96%.
Each entry below: **symptom → diagnosis → candidate fixes → status**.

---

## Variance estimator toggle (`variance=`)

`inference()` and `structural_dml()` take `variance='pooled'` (default) or `'within_fold'`.

- **`pooled` (default) — the FLM/DML prescription.** Ψ̂ = sample variance of the influence
  function ψ centered at the *global* mean, SE = √(Ψ̂/n). Source: FLM2025 §3.2 (line 265,
  *"the asymptotic variance Ψ = V[ψ(Y,T,X,θ⋆,Λ)] can be consistently estimated by …"*) and
  FLM2021 §4 (line 365, *"replacing the sample first moments … with second moments"*).
  **Use this for anything you report.**
- **`within_fold` — legacy variant.** Centers ψ within each cross-fit fold. Slightly
  *smaller* SE (differs only by the between-fold variation of fold means; ~0.5–1.5% at
  K=50, n=5000). Kept for backward comparison; **not** the paper's estimator. The paper's
  "within-fold" language refers to cross-fitting the *nuisances* (θ̂ₖ, Λ̂ₖ on Iₖᶜ), not the
  variance centering — the two were previously conflated.
- **Triage:** if `structural_dml` and `inference` disagree on SE, check the `variance` arg —
  both call one shared SE function and agree exactly at the same setting (asserted by a test).

---

## Coverage shortfalls (finite-sample signatures, not necessarily bugs)

### General rule: 3-way-split models need n ≥ 8000
Any model with `hessian_depends_on_y=True` (e.g. quantile) or a Regime-C 3-way split loses
~40% of data to nuisance/Λ estimation → finite-sample off-centering at n=5000. Validated:
multinomial 88%→98% and quantile 92%→94% on n=5000→8000. **Before calling a coverage FAIL a
bug, re-run at n ≥ 8000, M ≥ 50.**

### SIM2 — CNN encoder + Logit: 85% coverage @ n=10000
- **Symptom:** coverage 85% (target 95%), SE ratio 0.83 (SEs ~17% too small), z-mean −0.63
  (estimates ~0.6 SD low), |bias| 0.015, M=20.
- **Diagnosis:** textbook regularization-bias / slow-nuisance-rate signature. The first-order
  IF correction removes *first-order* bias, but a heavily-regularized CNN on image data at
  n=10000 has not entered the asymptotic regime — the nuisance-error product (θ̂, Λ̂) remainder
  has not shrunk to o(n^−1/2), leaving residual bias (the z-mean) *and* under-estimated
  cross-fit variance (SE ratio < 1). This is the FLM regularization-bias story; the paper
  already discusses it.
- **Candidate fixes (in order of expected payoff):**
  1. **Repeated cross-fitting + median aggregation.** FLM2025 (line ~998): *"cross fitting
     introduces extra variation … which might be reduced with repeated splitting and median
     aggregation."* Directly targets the SE-too-small. Cheapest principled fix — try first.
  2. **Larger n (n-sweep).** Image+CNN needs more data; run n = 10k / 25k / 50k and show
     coverage → 95%. This *proves* it is finite-sample, not a defect.
  3. **Better Λ̂(x) + confirm 3-way split.** Try `lambda_method='lgbm'` vs `'ridge'`; a poor
     Λ̂ inflates the SE-ratio error. Confirm the Regime-C 3-way split is actually engaged.
  4. **Undersmooth / train the CNN nuisance harder** (more epochs, less regularization) to
     cut the residual bias (z-mean), accepting some variance cost — classic undersmoothing
     for valid inference.
  5. If it survives (1)–(4), it is a **genuine finite-sample limitation at the tested n** —
     document it (already done). Not every DGP hits nominal coverage at every n.
- **Status:** DOCUMENTED limitation (paper discusses CNN undercoverage as regularization
  bias). Triage = try (1)+(2) before treating it as a defect.

### eval_14 — neural 2×2 DiD: OVERALL FAIL @ M=25, n=5000
- **Symptom:** z_mean +0.51 (|z_mean| < 0.3 FAIL), coverage FAIL; only M=25, n=5000.
- **Diagnosis:** same 3-way-split finite-sample off-centering; M=25 is also too small to
  estimate coverage precisely. `handoff.md` called it "validated" prematurely.
- **Fix / triage:** re-run at n=8000, M=50 (per the n≥8000 rule). If z_mean centers and
  coverage enters 90-99%, it was finite-sample. If it persists, investigate the neural 2×2
  Λ calibration specifically.
- **Status:** needs re-validation (close-out Batch 6). Do NOT claim "validated" while the
  report reads FAIL.
