# RieszNet IHDP Replication

Config: N=50, n_folds=5, n_repeats=2, max_epochs=200, patience=20
Wall time: 1883.3s

### RieszNet IHDP ATE replication (N=50 of the paper's 1000 semi-synthetic datasets)
| Quantity | Package | Paper |
|---|---|---|
| MAE | 0.124 | 0.11 |
| Coverage | 0.98 | 0.95 |

NOTE: Data = Dragonnet IHDP-1000 CSVs (Shi et al. 2019). N=50 < the paper's 1000. CI coverage uses the CSVs' original confounded T; the paper's coverage figure (Figure 2, line 227) redraws T per propensity 'True' from NPCI -- these coverage numbers are not directly comparable. The MAE comparison (Table 1) is directly comparable.