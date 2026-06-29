# Spike: FLM vs RieszNet (known-truth ATE)

### Negbin DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 0.8981, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9058 | +0.0076 | 0.0772 | 0.0929 | 1.20 | 96% |
| FLM[cholesky] | 0.7869 | -0.1112 | 0.1890 | 0.1670 | 0.88 | 96% |
| FLM[ridge] | 0.8625 | -0.0356 | 0.0943 | 0.1029 | 1.09 | 96% |
| FLM[oracle] | 0.8512 | -0.0469 | 0.1170 | 0.1211 | 1.04 | 96% |
| RieszNet | 0.8566 | -0.0415 | 0.0767 | 0.1016 | 1.33 | 96% |
| Naive | 0.9277 | +0.0295 | 0.1276 | 0.0200 | 0.16 | 20% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
