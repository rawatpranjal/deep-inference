# Cross-family landscape: where FLM[cholesky] works vs fails

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

### Gamma DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 0.8981, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9039 | +0.0057 | 0.0784 | 0.0732 | 0.93 | 92% |
| FLM[cholesky] | 0.8258 | -0.0724 | 0.1623 | 0.1226 | 0.76 | 88% |
| FLM[ridge] | 0.9220 | +0.0239 | 0.0938 | 0.0761 | 0.81 | 88% |
| FLM[oracle] | 0.9141 | +0.0160 | 0.1080 | 0.0949 | 0.88 | 90% |
| RieszNet | 0.8612 | -0.0369 | 0.0858 | 0.0889 | 1.04 | 96% |
| Naive | 0.8912 | -0.0070 | 0.1085 | 0.0210 | 0.19 | 26% |

### Negbin DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 0.8981, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9058 | +0.0076 | 0.0772 | 0.0929 | 1.20 | 96% |
| FLM[cholesky] | 0.7869 | -0.1112 | 0.1890 | 0.1670 | 0.88 | 96% |
| FLM[ridge] | 0.8625 | -0.0356 | 0.0943 | 0.1029 | 1.09 | 96% |
| FLM[oracle] | 0.8512 | -0.0469 | 0.1170 | 0.1211 | 1.04 | 96% |
| RieszNet | 0.8566 | -0.0415 | 0.0767 | 0.1016 | 1.33 | 96% |
| Naive | 0.9277 | +0.0295 | 0.1276 | 0.0200 | 0.16 | 20% |

### Gaussian DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 1.0000, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 1.0069 | +0.0069 | 0.0535 | 0.0534 | 1.00 | 96% |
| FLM[cholesky] | 1.0511 | +0.0511 | 0.1085 | 0.0828 | 0.76 | 92% |
| FLM[ridge] | 1.0165 | +0.0165 | 0.0535 | 0.0432 | 0.81 | 90% |
| FLM[oracle] | 1.0014 | +0.0014 | 0.0636 | 0.0655 | 1.03 | 96% |
| RieszNet | 1.0003 | +0.0003 | 0.0648 | 0.0619 | 0.96 | 96% |
| Naive | 0.9952 | -0.0048 | 0.0670 | 0.0122 | 0.18 | 20% |

### Probit DGP (lambdas=cholesky,ridge,oracle, n=2000, folds=5)  (truth = 0.1666, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.1662 | -0.0004 | 0.0223 | 0.0186 | 0.83 | 88% |
| FLM[cholesky] | 0.1568 | -0.0098 | 0.0445 | 0.0515 | 1.16 | 98% |
| FLM[ridge] | 0.1802 | +0.0136 | 0.0236 | 0.0198 | 0.84 | 84% |
| FLM[oracle] | 24.7630 | +24.5963 | 173.6438 | 18.6687 | 0.11 | 84% |
| RieszNet | 0.1699 | +0.0033 | 0.0248 | 0.0306 | 1.23 | 98% |
| Naive | 0.1670 | +0.0004 | 0.0286 | 0.0036 | 0.13 | 12% |

### Beta DGP (lambdas=cholesky,ridge, n=2000, folds=5)  (truth = 0.0859, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.0866 | +0.0007 | 0.0094 | 0.0098 | 1.04 | 98% |
| FLM[cholesky] | 0.0855 | -0.0004 | 0.0105 | 0.0106 | 1.02 | 96% |
| FLM[ridge] | 0.0894 | +0.0034 | 0.0091 | 0.0080 | 0.88 | 92% |
| RieszNet | 0.0830 | -0.0030 | 0.0240 | 0.0235 | 0.98 | 100% |
| Naive | 0.0847 | -0.0012 | 0.0133 | 0.0015 | 0.11 | 12% |

