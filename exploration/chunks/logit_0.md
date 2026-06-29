# Spike: FLM vs RieszNet (known-truth ATE)

### Logit DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 0.1481, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.1543 | +0.0062 | 0.0224 | 0.0220 | 0.98 | 92% |
| FLM[cholesky] | 0.1494 | +0.0013 | 0.0560 | 0.0563 | 1.01 | 98% |
| FLM[ridge] | 0.1640 | +0.0159 | 0.0220 | 0.0198 | 0.90 | 88% |
| FLM[oracle] | 0.1693 | +0.0212 | 0.0265 | 0.0238 | 0.90 | 86% |
| RieszNet | 0.1530 | +0.0049 | 0.0307 | 0.0399 | 1.30 | 100% |
| Naive | 0.1576 | +0.0095 | 0.0293 | 0.0030 | 0.10 | 12% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
