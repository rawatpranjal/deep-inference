# FLM linear SE decomposition (full observability)

M=30 n=2000 folds=20 epochs=100 reps=1. truth mu=1.0. SE-ratio = mean(reported SE)/empirical SD. Var ratio = (SE_reported/SE_efficient)^2.

| cell | SE-ratio | cov | bias | Var(psi_h)/Var(psi*) | psi-R2 | psi-corr | corr-R2 | Linv-R2p | L-R2p | L-frob | score-R2 | resid-R2 | beta-R2 | alpha-R2 | fails |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flat/th=net | 0.832 | 87% | +0.0300 | 0.560 | 0.521 | 0.719 | 0.498 | -0.114 | -0.003 | 0.782 | 0.901 | 0.912 | 0.498 | 0.937 | 0 |
| ridge@1/th=net | 0.380 | 90% | +0.1886 | 169.612 | -168.538 | 0.371 | -173.873 | -19807.208 | 0.945 | 0.159 | 0.880 | 0.892 | 0.417 | 0.920 | 0 |
| ridge@10/th=net | 0.344 | 87% | +0.1408 | 381.651 | -379.652 | 0.385 | -397.587 | -1168.830 | 0.945 | 0.161 | 0.880 | 0.892 | 0.417 | 0.920 | 0 |
| ridge@100/th=net | 0.394 | 97% | -0.0537 | 29.613 | -28.734 | 0.559 | -29.263 | -70.815 | 0.936 | 0.192 | 0.880 | 0.892 | 0.417 | 0.920 | 0 |
| ridge@1000/th=net | 0.756 | 77% | +0.0305 | 0.507 | 0.627 | 0.795 | 0.606 | 0.017 | 0.644 | 0.477 | 0.880 | 0.892 | 0.417 | 0.920 | 0 |
| rf/th=net | 1.139 | 97% | +0.0318 | 2.033 | -0.042 | 0.719 | -0.102 | -0.076 | 0.782 | 0.332 | 0.880 | 0.892 | 0.417 | 0.920 | 0 |
| oracle/th=net | 1.075 | 93% | +0.0248 | 1.184 | 0.836 | 0.932 | 0.812 | 1.000 | 1.000 | 0.000 | 0.901 | 0.912 | 0.498 | 0.937 | 0 |
| flat/th=oracle | 1.103 | 100% | +0.0159 | 0.531 | 0.558 | 0.744 | 0.544 | -0.114 | -0.003 | 0.782 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| ridge@1000/th=oracle | 1.065 | 100% | +0.0150 | 0.473 | 0.671 | 0.831 | 0.661 | 0.017 | 0.644 | 0.477 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| oracle/th=oracle | 1.114 | 100% | +0.0239 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |

Reading: oracle/th=net is the decisive cell (does true Lambda(x) restore SE-ratio?). 
Objects whose R2 moves together with SE-ratio/Var-ratio locate the defect; flat-R2 objects are exonerated. th=oracle rows remove the net's beta-recovery as a confound.
