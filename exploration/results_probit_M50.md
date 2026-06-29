# Spike: FLM vs RieszNet (known-truth ATE)

### Probit DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 0.1666, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.1662 | -0.0004 | 0.0223 | 0.0186 | 0.83 | 88% |
| FLM[cholesky] | 0.1568 | -0.0098 | 0.0445 | 0.0515 | 1.16 | 98% |
| FLM[ridge] | 0.1802 | +0.0136 | 0.0236 | 0.0198 | 0.84 | 84% |
| FLM[oracle] | 24.7630 | +24.5963 | 173.6438 | 18.6687 | 0.11 | 84% |
| RieszNet | 0.1699 | +0.0033 | 0.0248 | 0.0306 | 1.23 | 98% |
| Naive | 0.1670 | +0.0004 | 0.0286 | 0.0036 | 0.13 | 12% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
