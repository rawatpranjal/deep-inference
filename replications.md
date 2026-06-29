# replications.md

Raw harness output that `docs/replications/index.md` curates. Numbers here are copied verbatim
from the saved report files. Curated RTD page: `docs/replications/index.md`. Harness:
`exploration/replicate_papers.py`.

## FLM2021 Monte-Carlo ATE coverage

Config: M=200, n=10000, epochs=80, n_folds=5, hidden_dims=[60,30,20] (architecture 2).
True ATE tau_bar = 1.623214. Wall time: 816.1s.

**Randomized (constant propensity p=0.5)**

| Quantity | Package | Paper |
|---|---|---|
| Bias | -0.00199 | -0.00032 |
| Interval length | 0.078 | 0.079 |
| Coverage | 0.955 | 0.951 |

**Observational (logistic propensity)**

| Quantity | Package | Paper |
|---|---|---|
| Bias | -0.01733 | 0.00011 |
| Interval length | 0.081 | 0.079 |
| Coverage | 0.840 | 0.946 |

Source: `exploration/replicate_papers_flm_full.md`

## RieszNet IHDP ATE replication

Config: N=50, n_folds=5, n_repeats=2, max_epochs=200, patience=20. Wall time: 1883.3s.
Data: Dragonnet IHDP-1000 CSVs (Shi et al. 2019). N=50 < the paper's 1000.

| Quantity | Package | Paper |
|---|---|---|
| MAE | 0.124 | 0.11 |
| Coverage | 0.98 | 0.95 |

Coverage note: the paper redraws T per propensity 'True' from NPCI; the harness uses the
CSVs' original confounded T. These coverage numbers are not directly comparable. MAE (Table 1)
is directly comparable.

Source: `exploration/replicate_papers_riesz_full.md`
