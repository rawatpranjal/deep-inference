# FLM linear: stable-Lambda fix study

M=200 folds=20 epochs=100. truth=1.0. prop:* = estimate e(x) directly then form Lambda=2[[1,e],[e,e]] analytically (structured, exactly invertible). ridge@/rf = regress Hessian entries + invert (the unstable baseline). oracle = true e(x).

| cell | SE-ratio | cov | bias | Var(psi_h)/Var(psi*) | Linv-R2 | psi-R2 | L-R2 | beta-R2 | fails |
|---|---|---|---|---|---|---|---|---|---|
| flat/th=net/n=2000 | 0.867 | 89% | +0.0141 | 0.563 | -0.117 | 0.534 | -0.003 | 0.511 | 0 |
| oracle/th=net/n=2000 | 1.030 | 95% | +0.0002 | 1.151 | 1.000 | 0.853 | 1.000 | 0.511 | 0 |
| prop:logit_ridge/th=net/n=2000 | 1.041 | 95% | +0.0001 | 1.201 | 0.936 | 0.824 | 0.991 | 0.511 | 0 |
