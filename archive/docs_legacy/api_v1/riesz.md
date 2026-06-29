# RieszNet Module

`deep_inference.riesz` is the automatic-debiasing procedure (Chernozhukov et al. 2022). For the
concepts and when to use it, see the [RieszNet procedure](../inference/riesznet.md) page. This
page is the reference for the importable functions.

## `riesz_inference`

```python
from deep_inference import riesz_inference

riesz_inference(
    Y, T, X,
    outcome='linear',     # 'linear', 'logit', 'poisson', 'gamma', 'negbin', 'probit'
    n_folds=5,
    n_repeats=3,
    max_epochs=400,
    patience=30,
    seed=0,
    nb_dispersion=3.0,    # only used when outcome='negbin'
    alpha=0.05,           # 0.05 gives a 95% CI
    store_data=True,
    verbose=False,
) -> InferenceResult
```

Estimates the binary-treatment ATE contrast `E[g(1, X) - g(0, X)]` with the doubly-robust
moment, cross-fit over `n_folds` folds and combined across `n_repeats` splits with the
median-DML rule.

```{list-table}
:header-rows: 1

* - Argument
  - Meaning
* - `Y`, `T`, `X`
  - Outcomes `(n,)`, treatment `(n,)`, covariates `(n, d)`. For a binary `T` the estimand is the ATE.
* - `outcome`
  - The outcome model for the regression head. One of `linear`, `logit`, `poisson`, `gamma`, `negbin`, `probit`. An unsupported value raises `ValueError`.
* - `n_folds`
  - Cross-fitting folds `K`.
* - `n_repeats`
  - Independent split repeats, median-DML combined.
* - `max_epochs`, `patience`
  - Per-fold training schedule.
* - `nb_dispersion`
  - The negative-binomial size `r`, used only when `outcome='negbin'`.
* - `alpha`
  - CI level. `0.05` gives a 95% interval.
```

Returns an [`InferenceResult`](inference.md) with the usual `mu_hat`, `se`, `ci_lower`,
`ci_upper`, `psi_values` (the doubly-robust moment from the first repeat), `theta_hat` (the
learned representer `a(Z)` as an `(n, 1)` tensor), and a `diagnostics` dict that includes
`procedure='riesznet'`, `mu_naive`, `mean_abs_representer`, and the per-repeat estimates.

```python
import numpy as np
from deep_inference import riesz_inference

rng = np.random.default_rng(0)
n = 1500
X = rng.standard_normal((n, 3))
e = 1 / (1 + np.exp(-(0.8 * X[:, 0])))
T = (rng.random(n) < e).astype(float)        # confounded binary treatment
Y = 1.0 + 0.5 * X[:, 0] + (0.5 + 0.3 * X[:, 1]) * T + rng.standard_normal(n)

result = riesz_inference(Y, T, X, outcome='linear', n_folds=5, n_repeats=3)
print(result.summary())
```

## `RieszNet`

```python
from deep_inference import RieszNet
```

The underlying `torch.nn.Module`: a shared trunk (3 ELU layers, width 200) with a linear Riesz
head and a regression head (2 ELU layers, width 100), plus an unpenalized targeted-regularization
parameter `eps`. Most users do not construct this directly; `riesz_inference` builds, trains, and
cross-fits it. It is exposed for inspection and for custom training loops.

## See also

- [RieszNet procedure](../inference/riesznet.md): concepts, architecture, the doubly-robust moment.
- [Influence Function](../inference/influence_function.md): the default procedure.
- [Replications](../replications/index.md): RieszNet versus the influence-function method and the oracle, per family.
