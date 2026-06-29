# Tier A cross-family benchmark (M=50, n=2000)

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
