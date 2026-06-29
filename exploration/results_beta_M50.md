# Spike: FLM vs RieszNet (known-truth ATE)

### Beta DGP (lambdas=cholesky,ridge, n=2000, folds=5)  (truth = 0.0859, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.0866 | +0.0007 | 0.0094 | 0.0098 | 1.04 | 98% |
| FLM[cholesky] | 0.0855 | -0.0004 | 0.0105 | 0.0106 | 1.02 | 96% |
| FLM[ridge] | 0.0894 | +0.0034 | 0.0091 | 0.0080 | 0.88 | 92% |
| RieszNet | 0.0830 | -0.0030 | 0.0240 | 0.0235 | 0.98 | 100% |
| Naive | 0.0847 | -0.0012 | 0.0133 | 0.0015 | 0.11 | 12% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
