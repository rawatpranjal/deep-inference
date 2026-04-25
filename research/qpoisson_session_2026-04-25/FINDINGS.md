# FML Validation — Running Findings Log

**Goal**: Deliver honest 95% CIs for per-cell Poisson event-study coefficients using FLM's deep-learning framework on raw three-tower embeddings (d=128 no-PCA) as applied to clustered cell subgroups.

**Metric gates** (per `deepest/evals/eval_06_coverage.py`):
- Coverage ∈ [0.90, 0.99]
- SE ratio ∈ [0.7, 1.5]
- |bias| < 0.1 · SE

---

## Prior cranks (completed earlier in session)

| # | Setup | avg cov | RMSE vs sat | Headline |
|---|---|---|---|---|
| 1 | N-sweep d=16 Poisson, N=5k→200k | 0.84–0.92 | — | **Coverage never hits 0.95 at d=16** (curse of dim) |
| Diagnose | Oracle FML at N=50k | 0.940 | — | **IF formula is correct** — all failure is DNN nuisance |
| 1 (rob) | 10 IF fixes × 3 DGPs, M=20, 600 runs | F1 trim_e010 wins | — | **ψ-winsorize BAD**, Λ-trim ε=0.10 good |
| 2 (rob) | DGP stress | — | — | **Imbalanced T (p<0.2)** catastrophic, overdispersion conservative |
| 3 (rob) | 9 architectures | ≈ | — | **Architecture barely matters** (within MC noise) |
| 4 (rob) | ε-sweep 0.0→1.0 | flat plateau 0.92–0.96 | — | ε∈[0.05, 0.5] all OK, ε≥1 over-trims |
| 5 | Cluster K∈{5,10,20}, d=4 | **0.947** | 0.44 | **FML-subgroup dominates saturated 100%** at d=4 |
| 6 | 40 cross-cells, d=8 | **0.9488** | 0.44 | 40/40 cells FML beats sat; sat 65% cells <0.70 coverage |
| 7 | 40 cross-cells, **d=128 no-PCA** | **0.870** | 0.29 | 40/40 beats sat; 19/40 cells ≥0.90 coverage |
| 8 | d=128 scale-up (n=100k, deeper DNN) | 0.48 | 0.39 | **Worse than crank 7** — bigger DNN overfit |

## Current best recipe at d=128 (crank 7 baseline)

- DNN: hidden=(128, 64) ReLU
- n=40,000, K-folds=5
- Λ-trim ε=0.10, no winsorize
- LR=2e-3, patience=25, batch=1024

**Delivers**: avg per-cell coverage 0.87, FML RMSE 0.29× saturated, 40/40 cells dominate.
**Remaining gap**: 0.87 < 0.95 target; 2/40 cells have coverage <0.70.

## Literature fixes to try (in priority order)

1. **Activation match** — truth is Tanh, ReLU approximator has fixed mis-specification bias
2. **Additive structure** β(z_u, z_j) = f(z_u) + g(z_j) — halves effective dim
3. **Bilinear low-rank** β = Σ_k φ_k(z_u)·ψ_k(z_j) — matches user-item interaction
4. **Deep ensemble** — variance reduction via seed averaging
5. **Sieve with regularization** — structural constraints on the DNN
6. **Higher-order IF (Robins 2008)** — theoretical rate extension beyond Neyman orthogonality
7. **TMLE iterations** (Wei 2023) — targeted fluctuation to solve IF equation exactly

---

## Fresh cranks (this run)

(updates appended below as each crank completes)

### C9_tanh_matched  (t=2026-04-24T22:12:12, wall=53s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=1, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.8600**
- avg per-cell \|bias\|: 0.0220
- avg per-cell RMSE: 0.0609
- median SE ratio: 0.849
- cells with coverage ≥0.95: 6/40
- cells with coverage ≥0.90: 19/40
- cells with coverage <0.70: 5/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 5 |
| 0.70–0.85 | 10 |
| 0.85–0.90 | 6 |
| 0.90–0.95 | 13 |
| 0.95–0.99 | 0 |
| ≥0.99 | 6 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=29 | -0.265 | -0.0388 | 0.600 | 0.1085 |
| c=11 | -0.273 | -0.0426 | 0.667 | 0.0762 |
| c=12 | -0.385 | +0.0270 | 0.667 | 0.0829 |
| c=27 | -0.377 | +0.0676 | 0.667 | 0.0824 |
| c=30 | -0.338 | -0.0429 | 0.667 | 0.1058 |

---

### C10_additive  (t=2026-04-24T22:13:38, wall=86s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=1, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.8717**
- avg per-cell \|bias\|: 0.0203
- avg per-cell RMSE: 0.0600
- median SE ratio: 0.796
- cells with coverage ≥0.95: 8/40
- cells with coverage ≥0.90: 18/40
- cells with coverage <0.70: 1/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 1 |
| 0.70–0.85 | 14 |
| 0.85–0.90 | 7 |
| 0.90–0.95 | 10 |
| 0.95–0.99 | 0 |
| ≥0.99 | 8 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=29 | -0.265 | -0.0368 | 0.600 | 0.1116 |
| c=4 | -0.343 | +0.0360 | 0.733 | 0.0862 |
| c=18 | -0.424 | +0.0014 | 0.733 | 0.0739 |
| c=27 | -0.377 | +0.0599 | 0.733 | 0.0825 |
| c=30 | -0.338 | -0.0146 | 0.733 | 0.0811 |

---

### C11_bilinear_r8  (t=2026-04-24T22:14:58, wall=80s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=1, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.8283**
- avg per-cell \|bias\|: 0.0258
- avg per-cell RMSE: 0.0638
- median SE ratio: 0.803
- cells with coverage ≥0.95: 2/40
- cells with coverage ≥0.90: 10/40
- cells with coverage <0.70: 4/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 4 |
| 0.70–0.85 | 17 |
| 0.85–0.90 | 9 |
| 0.90–0.95 | 8 |
| 0.95–0.99 | 0 |
| ≥0.99 | 2 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=27 | -0.377 | +0.0790 | 0.600 | 0.0946 |
| c=29 | -0.265 | -0.0312 | 0.600 | 0.0879 |
| c=12 | -0.385 | +0.0392 | 0.667 | 0.1004 |
| c=38 | -0.369 | -0.0107 | 0.667 | 0.0736 |
| c=14 | -0.462 | +0.0750 | 0.733 | 0.0919 |

---

### C12_tanh_ensemble_5  (t=2026-04-24T22:18:05, wall=187s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=5, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.9000**
- avg per-cell \|bias\|: 0.0143
- avg per-cell RMSE: 0.0542
- median SE ratio: 0.858
- cells with coverage ≥0.95: 9/40
- cells with coverage ≥0.90: 25/40
- cells with coverage <0.70: 2/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 2 |
| 0.70–0.85 | 7 |
| 0.85–0.90 | 6 |
| 0.90–0.95 | 16 |
| 0.95–0.99 | 0 |
| ≥0.99 | 9 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=27 | -0.377 | +0.0500 | 0.667 | 0.0746 |
| c=29 | -0.265 | -0.0127 | 0.667 | 0.0834 |
| c=38 | -0.369 | +0.0081 | 0.733 | 0.0676 |
| c=6 | -0.397 | +0.0408 | 0.800 | 0.0584 |
| c=12 | -0.385 | +0.0075 | 0.800 | 0.0568 |

---

### C13_tanh_bigger_n  (t=2026-04-24T22:19:35, wall=90s)

**Setup**: d=128 no-PCA, 40 cross cells, n=80,000, ensemble_r=1, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.7167**
- avg per-cell \|bias\|: 0.0326
- avg per-cell RMSE: 0.0613
- median SE ratio: 0.691
- cells with coverage ≥0.95: 1/40
- cells with coverage ≥0.90: 4/40
- cells with coverage <0.70: 19/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 19 |
| 0.70–0.85 | 14 |
| 0.85–0.90 | 3 |
| 0.90–0.95 | 3 |
| 0.95–0.99 | 0 |
| ≥0.99 | 1 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=8 | -0.287 | -0.0340 | 0.533 | 0.0801 |
| c=11 | -0.273 | -0.0587 | 0.533 | 0.0882 |
| c=12 | -0.385 | +0.0257 | 0.533 | 0.0699 |
| c=14 | -0.462 | +0.0733 | 0.533 | 0.1025 |
| c=27 | -0.377 | +0.0581 | 0.533 | 0.0747 |

---

## Session ranking

| config | avg cov | cells ≥ 0.90 | cells < 0.70 | avg RMSE | SE ratio | wall (s) |
|---|---|---|---|---|---|---|
| C12_tanh_ensemble_5 | 0.900 | 25/40 | 2/40 | 0.0542 | 0.858 | 187 |
| C10_additive | 0.872 | 18/40 | 1/40 | 0.0600 | 0.796 | 86 |
| C9_tanh_matched | 0.860 | 19/40 | 5/40 | 0.0609 | 0.849 | 53 |
| C11_bilinear_r8 | 0.828 | 10/40 | 4/40 | 0.0638 | 0.803 | 80 |
| C13_tanh_bigger_n | 0.717 | 4/40 | 19/40 | 0.0613 | 0.691 | 90 |

Total wall time: 496s


## Round 2 cranks  (t=2026-04-24T22:20:56)

Building on C12 Tanh-ensemble-5 (0.900 avg coverage).  Round 2 combines best ingredients: ensemble, additive, paired hyperparams for bigger N.

### C14_tanh_ensemble_10  (t=2026-04-24T22:26:13, wall=318s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=10, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.6800**
- avg per-cell \|bias\|: 0.0590
- avg per-cell RMSE: 0.0860
- median SE ratio: 0.791
- cells with coverage ≥0.95: 2/40
- cells with coverage ≥0.90: 6/40
- cells with coverage <0.70: 16/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 16 |
| 0.70–0.85 | 11 |
| 0.85–0.90 | 7 |
| 0.90–0.95 | 4 |
| 0.95–0.99 | 0 |
| ≥0.99 | 2 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=9 | -0.426 | +0.1300 | 0.200 | 0.1469 |
| c=8 | -0.425 | +0.1277 | 0.267 | 0.1404 |
| c=28 | -0.436 | +0.1041 | 0.333 | 0.1418 |
| c=29 | -0.421 | +0.1264 | 0.333 | 0.1458 |
| c=30 | -0.500 | +0.1136 | 0.333 | 0.1471 |

---

### C15_additive_ensemble_5  (t=2026-04-24T22:31:43, wall=330s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=5, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.5850**
- avg per-cell \|bias\|: 0.0757
- avg per-cell RMSE: 0.0916
- median SE ratio: 0.946
- cells with coverage ≥0.95: 0/40
- cells with coverage ≥0.90: 4/40
- cells with coverage <0.70: 21/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 21 |
| 0.70–0.85 | 7 |
| 0.85–0.90 | 8 |
| 0.90–0.95 | 4 |
| 0.95–0.99 | 0 |
| ≥0.99 | 0 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=8 | -0.425 | +0.1534 | 0.067 | 0.1609 |
| c=9 | -0.426 | +0.1594 | 0.067 | 0.1642 |
| c=28 | -0.436 | +0.1378 | 0.133 | 0.1470 |
| c=30 | -0.500 | +0.1675 | 0.133 | 0.1790 |
| c=10 | -0.489 | +0.1344 | 0.200 | 0.1443 |

---

### C16_tanh_ens5_n80k_tuned  (t=2026-04-24T22:42:18, wall=635s)

**Setup**: d=128 no-PCA, 40 cross cells, n=80,000, ensemble_r=5, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.6367**
- avg per-cell \|bias\|: 0.0477
- avg per-cell RMSE: 0.0715
- median SE ratio: 0.706
- cells with coverage ≥0.95: 1/40
- cells with coverage ≥0.90: 4/40
- cells with coverage <0.70: 23/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 23 |
| 0.70–0.85 | 8 |
| 0.85–0.90 | 5 |
| 0.90–0.95 | 3 |
| 0.95–0.99 | 0 |
| ≥0.99 | 1 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=10 | -0.489 | +0.1133 | 0.267 | 0.1311 |
| c=9 | -0.426 | +0.1199 | 0.333 | 0.1408 |
| c=11 | -0.442 | +0.1020 | 0.333 | 0.1339 |
| c=36 | -0.415 | +0.0803 | 0.333 | 0.0954 |
| c=8 | -0.425 | +0.1022 | 0.400 | 0.1280 |

---

### C17_tanh_ens5_patience50  (t=2026-04-24T22:48:09, wall=351s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=5, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.7333**
- avg per-cell \|bias\|: 0.0521
- avg per-cell RMSE: 0.0784
- median SE ratio: 0.853
- cells with coverage ≥0.95: 3/40
- cells with coverage ≥0.90: 12/40
- cells with coverage <0.70: 16/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 16 |
| 0.70–0.85 | 7 |
| 0.85–0.90 | 5 |
| 0.90–0.95 | 9 |
| 0.95–0.99 | 0 |
| ≥0.99 | 3 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=8 | -0.425 | +0.1109 | 0.267 | 0.1244 |
| c=9 | -0.426 | +0.1116 | 0.267 | 0.1300 |
| c=10 | -0.489 | +0.1075 | 0.333 | 0.1256 |
| c=30 | -0.500 | +0.1177 | 0.333 | 0.1519 |
| c=28 | -0.436 | +0.0700 | 0.400 | 0.1148 |

---

### C18_tanh_ens15  (t=2026-04-24T22:58:09, wall=600s)

**Setup**: d=128 no-PCA, 40 cross cells, n=40,000, ensemble_r=15, M=15 MC reps

**Headline**:
- avg per-cell coverage: **0.5867**
- avg per-cell \|bias\|: 0.0765
- avg per-cell RMSE: 0.0962
- median SE ratio: 0.843
- cells with coverage ≥0.95: 2/40
- cells with coverage ≥0.90: 3/40
- cells with coverage <0.70: 22/40

**Coverage histogram**:

| bin | n cells |
|---|---|
| <0.70 | 22 |
| 0.70–0.85 | 10 |
| 0.85–0.90 | 5 |
| 0.90–0.95 | 1 |
| 0.95–0.99 | 0 |
| ≥0.99 | 2 |

**Worst 5 cells**:

| cell | µ_c* | bias | cov | rmse |
|---|---|---|---|---|
| c=36 | -0.415 | +0.1369 | 0.067 | 0.1519 |
| c=8 | -0.425 | +0.1417 | 0.133 | 0.1534 |
| c=28 | -0.436 | +0.1314 | 0.133 | 0.1390 |
| c=29 | -0.421 | +0.1534 | 0.133 | 0.1671 |
| c=30 | -0.500 | +0.1729 | 0.133 | 0.1832 |

---

## Round 2 ranking

| config | avg cov | cells ≥ 0.90 | cells < 0.70 | RMSE | SE ratio |
|---|---|---|---|---|---|
| C17_tanh_ens5_patience50 | 0.733 | 12/40 | 16/40 | 0.0784 | 0.853 |
| C14_tanh_ensemble_10 | 0.680 | 6/40 | 16/40 | 0.0860 | 0.791 |
| C16_tanh_ens5_n80k_tuned | 0.637 | 4/40 | 23/40 | 0.0715 | 0.706 |
| C18_tanh_ens15 | 0.587 | 3/40 | 22/40 | 0.0962 | 0.843 |
| C15_additive_ensemble_5 | 0.585 | 4/40 | 21/40 | 0.0916 | 0.946 |


## Round 3: Stability check  (t=2026-04-24T22:59:09)

Rerun C12 config (Tanh ensemble-5) with 5 different seed offsets to measure MC noise at M=15.  Then run M=60 to get a tight estimate of true coverage.


## Round 3: Stability check (v4) — CRITICAL FINDING

### S1: C12 config run with 5 different seed offsets, M=15 each

| seed_offset | avg_cov | cells ≥ 0.90 | cells < 0.70 |
|---|---|---|---|
| 10,000 | 0.7483 | 11/40 | 14/40 |
| 20,000 | 0.8000 | 12/40 | 12/40 |
| 30,000 | 0.9017 | 24/40 | 0/40 |
| 40,000 | 0.9267 | 30/40 | 0/40 |
| 50,000 | 0.9050 | 20/40 | 0/40 |

**MC noise at M=15**: mean = **0.856**, SD = **0.078**, range = [0.748, 0.927]

### S2: high-M=60 test

**avg_cov = 0.7729** (11/40 cells ≥ 0.90, 13/40 < 0.70)

### S3: second M=60 test — FAILED (disk full, not a method issue)

---

## Revised honest conclusion

The round-1 C12 result of 0.900 coverage was **one lucky seed set within a high-variance
distribution**.  True coverage of Tanh-ensemble-5 at d=128 no-PCA is approximately
**0.77–0.86**, not 0.90.  The S1 mean (0.856) and S2 (0.773) bracket the truth.

**Implications:**
- **FML-subgroup at d=128 no-PCA does NOT hit nominal 95%.** Empirical coverage is
  in the 0.77–0.86 range, so CIs labeled "95%" are actually ~85% honest.
- **But it still dominates saturated GLM by a huge margin** (saturated was 0.088).
  The RMSE ratio (FML 0.29–0.44× saturated) holds across all seed sets.
- **Any ranking between IF-fix variants at M=15 has ±0.15 noise**, so small differences
  between configs (e.g., Tanh vs ReLU, or ensemble-5 vs ensemble-10) cannot be
  distinguished at this MC budget.  Need M ≥ 60 for reliable comparison.
- **Dimension matters more than architecture.**  At d=8 we had solid 0.95 coverage;
  at d=128 we are stuck around 0.80.  Reducing d remains the highest-leverage fix.

## What to recommend for the paper

1. **If the paper needs honest 95% CIs**, project embeddings to d≤8 via supervised dimension
   reduction (NOT PCA since user rules that out — use a learned bottleneck).
2. **If the paper accepts 80-85% empirical coverage**, deploy FML-subgroup at d=128 directly
   and acknowledge the calibration gap.  It still beats saturated GLM on every metric.
3. **Always run M≥60 MC reps** to validate any coverage claim — M=15 is too noisy
   to distinguish configs that differ by less than 0.15 in coverage.
