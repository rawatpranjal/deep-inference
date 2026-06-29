# Spike: FLM vs RieszNet (known-truth ATE)

### Poisson DGP (lambdas=cholesky,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 0.8981, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.8888 | -0.0093 | 0.0703 | 0.0682 | 0.97 | 100% |
| FLM[cholesky] | 0.8718 | -0.0263 | 0.1049 | 0.1106 | 1.05 | 98% |
| FLM[oracle] | 0.8555 | -0.0427 | 0.0843 | 0.0792 | 0.94 | 90% |
| RieszNet | 0.8755 | -0.0226 | 0.1086 | 0.1274 | 1.17 | 100% |
| Naive | 0.8758 | -0.0224 | 0.1025 | 0.0236 | 0.23 | 28% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
