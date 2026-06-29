# Loading Data & Pre-Estimation

Before you call any estimator, your data has to be in the right shape and the right type. This
page is the checklist. Get this right and everything downstream just works; get it wrong and you
get a shape error or, worse, a silently wrong answer.

## The three arrays

Every entry point takes the same three inputs.

```{list-table}
:header-rows: 1

* - Array
  - Shape
  - What it is
* - `Y`
  - `(n,)`
  - The outcome, one value per observation.
* - `T`
  - `(n,)`
  - The treatment or policy variable whose effect you want.
* - `X`
  - `(n, d)`
  - The covariates. The network maps `X` to the heterogeneous parameters `θ(X)`.
```

Here `n` is the number of observations and `d` is the number of covariates. `X` is
two-dimensional even when you have a single covariate: use `X.reshape(-1, 1)`, not a flat
`(n,)` array.

## Types and a few rules

- Everything is numeric and is cast to float internally. Pass NumPy arrays.
- `Y` is a float array even for binary or count outcomes. For a 0/1 logit outcome use
  `Y.astype(float)`, not booleans.
- No missing values. Drop or impute `NaN`s before you call the estimator.
- `T` must vary. If `T` is constant there is no treatment effect to identify.

```python
import numpy as np

n = 2000
X = np.random.randn(n, 5)        # (n, d)
T = np.random.randn(n)           # (n,)
Y = (X[:, 0] + 0.5 * T + np.random.randn(n))  # (n,), float
```

## Special encodings

Most models use the plain `(n,)` / `(n, d)` layout above. Two need more structure:

- **Multinomial logit.** `T` packs the alternative-specific attributes as `(n, J*K)` for `J`
  alternatives and `K` attributes, and `Y` is the chosen alternative index `0, 1, ..., J-1` as a
  float. See the [Multinomial Logit page](models/multinomial.md).
- **Difference-in-differences.** The `did()` entry point takes group and period indicators (and
  optionally unit and time identifiers for the panel variant) instead of a single `T`. See the
  API reference.

## Pre-estimation choices

Two choices you make at call time matter for the validity of the interval.

### Randomized or observational?

If `T` was randomly assigned and you know its distribution, you are in **Regime A**: the package
can compute the nuisance object exactly instead of estimating it. Tell it so:

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

result = inference(Y, T, X, model='logit', target='beta',
                   is_randomized=True, treatment_dist=Normal(0, 1))
```

If `T` is observational (most field data), leave these off and the package estimates the
nuisance object (Regime C). The full story is in [Estimation](estimation/index.md) and the
[Overview regime table](overview.md).

### How many folds?

Cross-fitting splits the data; more folds means each model trains on more data. The package
default is `n_folds=50`. For linear average-treatment-effect coverage the recommendation is
`n_folds >= 50` together with `n_repeats >= 5`. Fewer folds tends to under-cover.

## Do you need to standardize `X`?

The network trains better when covariates are on comparable scales. If your `X` columns span
very different magnitudes, standardize them (subtract the mean, divide by the standard
deviation) before fitting. The target `E[β(X)]` is unaffected by standardizing `X`, but training
is more stable.

## Next

Once your arrays are ready, pick a [model](models/index.md) and run it. The
[Quick Start](quickstart.md) has a complete runnable example.
