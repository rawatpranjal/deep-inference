# Spike: FLM vs RieszNet (known-truth ATE)

### Linear DGP (lambdas=cholesky,flat,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 1.0000, M = 100)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9992 | -0.0008 | 0.0555 | 0.0533 | 0.96 | 96% |
| FLM[cholesky] | 0.9959 | -0.0041 | 0.0632 | 0.0582 | 0.92 | 93% |
| FLM[flat] | 1.0135 | +0.0135 | 0.0602 | 0.0457 | 0.76 | 83% |
| FLM[oracle] | 0.9970 | -0.0030 | 0.0612 | 0.0586 | 0.96 | 95% |
| RieszNet | 0.9917 | -0.0083 | 0.0560 | 0.0609 | 1.09 | 97% |
| Naive | 0.9958 | -0.0042 | 0.0718 | 0.0120 | 0.17 | 23% |

### Logit DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 0.1481, M = 100)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.1535 | +0.0054 | 0.0215 | 0.0220 | 1.02 | 94% |
| FLM[cholesky] | 0.1461 | -0.0020 | 0.0525 | 0.0564 | 1.07 | 98% |
| FLM[ridge] | 0.1647 | +0.0166 | 0.0215 | 0.0197 | 0.92 | 86% |
| FLM[oracle] | 0.1648 | +0.0167 | 0.0264 | 0.0237 | 0.90 | 86% |
| RieszNet | 0.1542 | +0.0062 | 0.0418 | 0.0444 | 1.06 | 100% |
| Naive | 0.1550 | +0.0069 | 0.0271 | 0.0029 | 0.11 | 10% |

### Poisson DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 0.8981, M = 100)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.8923 | -0.0058 | 0.0698 | 0.0682 | 0.98 | 99% |
| FLM[cholesky] | 0.8865 | -0.0116 | 0.1161 | 0.1139 | 0.98 | 98% |
| FLM[ridge] | 0.8657 | -0.0325 | 0.0762 | 0.0812 | 1.07 | 94% |
| FLM[oracle] | 0.8578 | -0.0403 | 0.0773 | 0.0788 | 1.02 | 95% |
| RieszNet | 0.9364 | +0.0382 | 0.5386 | 0.1725 | 0.32 | 98% |
| Naive | 0.8805 | -0.0177 | 0.0957 | 0.0238 | 0.25 | 33% |

Pass if FLM and RieszNet both ~truth, coverage 90-97%, SE ratio ~1; Naive should under-cover (shows the correction is necessary).
