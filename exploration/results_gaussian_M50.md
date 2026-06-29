# Spike: FLM vs RieszNet (known-truth ATE)

### Gaussian DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 1.0000, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 1.0069 | +0.0069 | 0.0535 | 0.0534 | 1.00 | 96% |
| FLM[cholesky] | 1.0511 | +0.0511 | 0.1085 | 0.0828 | 0.76 | 92% |
| FLM[ridge] | 1.0165 | +0.0165 | 0.0535 | 0.0432 | 0.81 | 90% |
| FLM[oracle] | 1.0014 | +0.0014 | 0.0636 | 0.0655 | 1.03 | 96% |
| RieszNet | 1.0003 | +0.0003 | 0.0648 | 0.0619 | 0.96 | 96% |
| Naive | 0.9952 | -0.0048 | 0.0670 | 0.0122 | 0.18 | 20% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
