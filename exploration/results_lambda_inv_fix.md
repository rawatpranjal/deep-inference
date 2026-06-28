# FLM linear: stable-Lambda fix study

M=50 folds=20 epochs=100. truth=1.0. prop:* = estimate e(x) directly then form Lambda=2[[1,e],[e,e]] analytically (structured, exactly invertible). ridge@/rf = regress Hessian entries + invert (the unstable baseline). oracle = true e(x).

| cell | SE-ratio | cov | bias | Var(psi_h)/Var(psi*) | Linv-R2 | psi-R2 | L-R2 | beta-R2 | fails |
|---|---|---|---|---|---|---|---|---|---|
| oracle/th=net/n=2000 | 1.085 | 96% | +0.0101 | 1.162 | 1.000 | 0.855 | 1.000 | 0.509 | 0 |
| prop:logit_ridge/th=net/n=2000 | 1.124 | 96% | +0.0108 | 1.217 | 0.939 | 0.822 | 0.992 | 0.509 | 0 |
| prop:lpm_ols/th=net/n=2000 | 0.913 | 100% | -0.0107 | 415.076 | -629.045 | -398.418 | 0.974 | 0.509 | 0 |
| prop:histgb/th=net/n=2000 | 1.081 | 100% | +0.0062 | 2.390 | 0.120 | 0.041 | 0.878 | 0.509 | 0 |
| prop:mlp/th=net/n=2000 | 1.176 | 96% | +0.0097 | 1.384 | 0.575 | 0.715 | 0.960 | 0.509 | 0 |
| ridge@100/th=net/n=2000 | 0.429 | 96% | -0.0167 | 20.357 | -513.659 | -19.230 | 0.935 | 0.426 | 0 |
| rf/th=net/n=2000 | 1.142 | 98% | +0.0186 | 2.388 | -0.105 | -0.329 | 0.781 | 0.426 | 0 |
| prop:logit_ridge/th=net/n=1000 | 0.991 | 92% | +0.0004 | 1.238 | 0.886 | 0.768 | 0.982 | 0.356 | 0 |
| prop:logit_ridge/th=net/n=4000 | 1.018 | 94% | -0.0084 | 1.148 | 0.957 | 0.872 | 0.995 | 0.639 | 0 |
| prop:logit_ridge/th=net/n=8000 | 1.110 | 98% | -0.0035 | 1.089 | 0.986 | 0.912 | 0.998 | 0.745 | 0 |
