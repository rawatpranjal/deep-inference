# RieszNet IHDP Replication

Config: N=3, n_folds=5, n_repeats=1, max_epochs=80, patience=15
Wall time: 27.9s

### RieszNet IHDP ATE replication (N=3 of the paper's 1000 semi-synthetic datasets)
| Quantity | Package | Paper |
|---|---|---|
| MAE | 0.077 | 0.11 |
| Coverage | 1.00 | 0.95 |

NOTE: Data = Dragonnet IHDP-1000 CSVs (Shi et al. 2019). N=3 < the paper's 1000. CI coverage uses the CSVs' original confounded T; the paper's coverage figure (Figure 2, line 227) redraws T per propensity 'True' from NPCI -- these coverage numbers are not directly comparable. The MAE comparison (Table 1) is directly comparable.