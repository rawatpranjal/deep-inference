# Spike: FLM vs RieszNet (known-truth ATE)

### Gamma DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 0.8981, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9039 | +0.0057 | 0.0784 | 0.0732 | 0.93 | 92% |
| FLM[cholesky] | 0.8258 | -0.0724 | 0.1623 | 0.1226 | 0.76 | 88% |
| FLM[ridge] | 0.9220 | +0.0239 | 0.0938 | 0.0761 | 0.81 | 88% |
| FLM[oracle] | 0.9141 | +0.0160 | 0.1080 | 0.0949 | 0.88 | 90% |
| RieszNet | 0.8612 | -0.0369 | 0.0858 | 0.0889 | 1.04 | 96% |
| Naive | 0.8912 | -0.0070 | 0.1085 | 0.0210 | 0.19 | 26% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
