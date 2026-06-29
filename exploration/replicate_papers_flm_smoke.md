# FLM2021 Monte Carlo Replication

Config: M=5, n=10000, epochs=80, n_folds=5, hidden_dims=[60,30,20] (architecture 2)

True ATE tau_bar = 1.623214
Wall time: 19.8s
Note: paper across-architecture coverage range for d=20 approx 0.93-0.96

### FLM2021 Monte Carlo -- Randomized (constant propensity p=0.5) (d=20, linear outcome, n=10000, M=5)
| Quantity | Package | Paper |
|---|---|---|
| Bias | 0.01160 | -0.00032 |
| Interval length | 0.078 | 0.079 |
| Coverage | 1.000 | 0.951 |

### FLM2021 Monte Carlo -- Observational (logistic propensity) (d=20, linear outcome, n=10000, M=5)
| Quantity | Package | Paper |
|---|---|---|
| Bias | -0.00843 | 0.00011 |
| Interval length | 0.081 | 0.079 |
| Coverage | 1.000 | 0.946 |
