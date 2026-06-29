# Autoresearch: FLM linear under-coverage

## Objective
Get FLM (the package, `structural_dml` family='linear') to valid 95% CIs on the
known-truth linear ATE DGP.

## Oracle (frozen)
- DGP: `exploration/spike.py` linear DGP. ATE = E[b(X)] = 1.0 (exact).
- Metric: Monte Carlo coverage of the 95% CI over M reps; co-metric SE-ratio =
  mean(reported SE) / empirical SD of the point estimate.
- Iteration config: n=2000, folds=20, M=50 (fast, ~3-5 min local, 11 cores).
- TARGET: coverage in [93%, 97%] AND SE-ratio in [0.92, 1.08].
- Final winner re-confirmed at M=200 (binomial SE drops ~4% -> ~2%).
- Venue: LOCAL. Bounded: ~8 iterations, stop+report best if it plateaus.

## Baseline (established before loop)
M=50, n=2000, folds=20, linear:
| condition | bias | emp SD | reported SE | SE-ratio | coverage |
|---|---|---|---|---|---|
| FLM n_repeats=1 | +0.014 | 0.0587 | 0.0457 | 0.78 | 84% |
| FLM n_repeats=5 | +0.013 | 0.0543 | 0.0469 | 0.86 | 92% |
| FLM n_repeats=10 | +0.013 | 0.0543 | 0.0469 | 0.86 | 92% |
| Oracle (MLE+delta) | -0.001 | 0.0548 | 0.0535 | 0.98 | 98% |
| RieszNet (no repeats) | -0.011 | 0.0540 | 0.0544 | 1.01 | 94% |

Localization (from the sweep): at n_repeats=5 the bias is small (+0.013) and the
point estimate is as stable as the oracle (emp SD 0.0543 vs 0.0548). The defect is
specifically that the REPORTED SE (0.0469) reads ~14% below the true SD (0.0543).
So the fix must target the SE computation, not the point estimator.

## Approaches tried (and why they did/didn't work)
- (baseline) Single-split IF SE, folds=20, n_repeats=1 -> 84%, SE-ratio 0.78. FAIL (SE too small).
- Repeated cross-fitting at FOLDS=20, n_repeats=5/10 -> 92%, SE-ratio 0.86. PARTIAL.
  NOTE: this plateau was an artifact of folds=20, not a real ceiling.
- folds=50 alone (n_repeats=1) -> 90%, SE-ratio 0.82. The package docstring says K>=50
  needed for stable SE; baseline wrongly used 20. Real lever.
- folds=50 + n_repeats=5 -> 94%, SE-ratio 0.875 at M=50. CANDIDATE (coverage in target).
  Both levers stack: folds 20->50 gives 84->90; adding repeats gives 90->94.

## Metric log (per iteration)
| iter | hypothesis | config | SE-ratio | coverage(M=50) | verdict |
|---|---|---|---|---|---|
| 0 | baseline | folds=20 rep=1 | 0.78 | 84% | FAIL |
| 1 | documented folds | folds=50 rep=1 | 0.82 | 90% | below |
| 1 | folds + repeats | folds=50 rep=5 | 0.875 | 94% | CANDIDATE |

### iter 2 (M=200 confirmation, n=2000, folds=50)
| condition | bias | SE-ratio | coverage |
|---|---|---|---|
| rep=1 | +0.013 | 0.879 | 92% |
| rep=5 | +0.012 | 0.915 | 94% |  <- candidate confirmed
| rep=10 | +0.012 | 0.923 | 92% |

Candidate (folds=50, rep=5) at M=200: coverage 94% (in target), SE-ratio 0.915 (just
under the 0.92 co-target). Residual: ~8% SE undercount + small +0.012 bias (~0.24 SD).
Both are finite-sample at n=2000 (n=5000 untested: superlinear cost in the linear path
makes it impractically slow on CPU).

### iter 3 (full metric suite, M=200, n=2000)
| config | bias | SE-ratio | cov | aR2 | aCorr | bR2 | bCorr |
|---|---|---|---|---|---|---|---|
| baseline f20 r1 h32 | +0.012 | 0.878 | 90% | 0.94 | 0.97 | 0.51 | 0.81 |
| candidate f50 r5 h32 | +0.012 | 0.915 | 94% | 0.94 | 0.97 | 0.52 | 0.81 |
| f50 r5 h64,32 | +0.017 | 0.884 | 90% | 0.95 | 0.98 | 0.56 | 0.81 |
| f50 r5 h64,32 ep300 | +0.017 | 0.884 | 90% | 0.95 | 0.98 | 0.56 | 0.81 |

Findings: (1) UNDERSMOOTHING BACKFIRES -- bigger net recovers nuisances better
(alpha/beta R2 up, RMSE down) but coverage DROPS 94->90, bias up, SE-ratio down.
Small net [32] is the calibration sweet spot. (2) alpha(X) recovered well (R2~0.94),
beta(X) only moderate (R2~0.52) because beta=1+0.5*X0 has little heterogeneity; the
TARGET E[beta]=1 is still hit (bias +0.012). (3) Loop converged: folds/repeats/net/
epochs exhausted; only larger n remains (impractical, superlinear cost).

## CONVERGED RESULT
Best = folds=50, n_repeats=5, hidden=[32]: coverage 94% (IN TARGET [93,97]),
SE-ratio 0.915 (just under 0.92 co-target), bias +0.012. The ~8% SE undercount is a
finite-sample residual at n=2000 that the tractable levers do not close.

## Fresh adversarial verify (separate Opus agent, 2026-06-27)
VERDICT: PARTIALLY SUPPORTED. All mechanics correct (metric/recovery code, DGP, theta_hat
column labeling alpha=[:,0]/beta=[:,1], out-of-fold, no softened oracle, no seed overfit;
the M=50 84% baseline was just rep-count noise -- M=200 baseline is 90%). Honest caveats it
forced:
- The candidate's 94% is a POINT estimate; at M=200 the binomial band is +/-1.7% (~[90.6,97.3]),
  so it cannot be statistically distinguished from ~92%.
- The SE-ratio co-target (0.92) is genuinely MISSED (0.915).
- Under-coverage is part SE-undercount AND part bias (+0.012 ~= 0.24 SD); not a pure SE defect.
- "Finite-sample" is a reasonable hypothesis but UNCONFIRMED (n=5000 never run, too slow).
- Nuance: ep300==ep200 because patience defaults to ~10 -> early stop capped both; "more
  epochs" was effectively a no-op test.

## Status: CONVERGED, NOT STRICTLY SOLVED
Big improvement (folds=50 + n_repeats=5: 84/90% -> 94% point, SE-ratio 0.78 -> 0.915) but the
frozen oracle (coverage [93,97] AND SE-ratio [0.92,1.08]) is NOT cleanly met. Tractable config
levers exhausted (bigger nets HURT coverage). Strict closure needs larger n (impractical here,
superlinear cost) or a methodological change (RieszNet-style targeting already hits 94% at n=2000).
