# PSD-by-construction (Cholesky) general Lambda(x)

M=50 n=2000 folds=20. cholesky = net outputs L(x), Lambda=LL^T, trained Frobenius to per-obs Hessians (general). vs flat bug, lgbm (general regress+invert), oracle.

| cell | SE-ratio | cov | bias | var_ratio | Linv-R2 | L-R2 |
|---|---|---|---|---|---|---|
| flat | 0.808 | 88% | +0.0141 | 0.558 | -0.113 | -0.003 |
| lgbm | 0.777 | 84% | +0.0176 | 0.530 | -0.043 | 0.626 |
| cholesky | 1.010 | 96% | -0.0015 | 1.036 | 0.169 | 0.855 |
| spectral | 0.975 | 98% | +0.0025 | 1.014 | 0.197 | 0.866 |
| spectral_floor@0.1 | 0.969 | 94% | +0.0039 | 0.992 | 0.116 | 0.858 |
| oracle | 1.085 | 96% | +0.0101 | 1.162 | 1.000 | 1.000 |
