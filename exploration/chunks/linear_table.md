### Linear DGP (lambdas=cholesky,flat,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 1.0000, M = 100)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9992 | -0.0008 | 0.0555 | 0.0533 | 0.96 | 96% |
| FLM[cholesky] | 0.9947 | -0.0053 | 0.0580 | 0.0586 | 1.01 | 95% |
| FLM[flat] | 1.0118 | +0.0118 | 0.0559 | 0.0456 | 0.82 | 87% |
| FLM[oracle] | 0.9982 | -0.0018 | 0.0626 | 0.0580 | 0.93 | 92% |
| RieszNet | 0.9917 | -0.0083 | 0.0560 | 0.0609 | 1.09 | 97% |
| Naive | 0.9958 | -0.0042 | 0.0718 | 0.0120 | 0.17 | 23% |

