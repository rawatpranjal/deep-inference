# Influence-Function Procedure

This is the default procedure and the one the whole package is built around. It is the Farrell,
Liang, Misra correction. This page explains the one object it produces per observation, how that
object becomes a point estimate and a standard error, and the two knobs that change the variance.

## The influence function, in words

For each observation `i` the procedure builds a single number `ψ_i`, the **influence function
value**. Read it as "the target, plus the correction for the bias that regularization put into
observation `i`". Concretely,

$$\psi_i = \underbrace{H_i}_{\text{plug-in target}} \;-\; \underbrace{H_\theta \,\Lambda(x_i)^{-1}\, \ell_\theta(i)}_{\text{bias correction}}$$

where `H_i` is the target evaluated at the fitted parameters (the naive, biased piece),
`ℓ_θ(i)` is the score (gradient of the loss in `θ`), `Λ(x_i)` is the expected Hessian from
[Estimation](../estimation/index.md), and `H_θ` is the Jacobian of the target in `θ`. You do not
assemble this by hand; the engine does it. The point is that the second term is exactly the
first-order bias, so subtracting it removes the bias to first order.

## From `ψ` to an estimate and an interval

The debiased point estimate is just the average of the influence values, and the standard error
is the sample standard deviation of those values divided by `√n`. Both entry points
(`structural_dml()` and `inference()`) use the same code for this step,
[`compute_se_ci`](https://github.com/rawatpranjal/deep-inference/blob/main/src/deep_inference/engine/variance.py).

$$\hat\mu = \frac{1}{n}\sum_{i=1}^n \psi_i, \qquad
\widehat{SE} = \sqrt{\hat\Psi / n}, \qquad
\text{CI}_{95\%} = \hat\mu \pm 1.96\,\widehat{SE}$$

The only thing that varies is how the variance `Ψ̂` is computed.

### Variance estimator (the `variance=` argument)

```{list-table}
:header-rows: 1

* - `variance=`
  - Formula
  - Notes
* - `'pooled'` (default)
  - $\hat\Psi = \frac{1}{n-1}\sum_i (\psi_i - \hat\mu)^2$
  - Bessel-corrected sample variance centered at the global mean. This is the variance the paper prescribes (FLM2025 §3.2; FLM2021 §4).
* - `'within_fold'` (legacy)
  - $\hat\Psi = \frac{1}{K}\sum_k \operatorname{mean}_{i \in \text{fold } k}(\psi_i - \hat\mu_k)^2$
  - Mean of per-fold variances, each centered at its own fold mean. Drops the between-fold component, so it is always smaller than pooled and gives narrower intervals. Use `'pooled'`.
```

### Repeated cross-fitting (the `n_repeats=` argument)

Cross-fitting splits the data randomly, so a single split is a little noisy. Setting
`n_repeats > 1` repeats the whole procedure on fresh splits and combines them with the median
rule of Chernozhukov et al. (2018),
[`median_dml_aggregate`](https://github.com/rawatpranjal/deep-inference/blob/main/src/deep_inference/engine/variance.py):

$$\hat\mu = \operatorname{median}_r(\hat\mu_r), \qquad
\widehat{SE}^2 = \operatorname{median}_r\!\big(\widehat{SE}_r^2 + (\hat\mu_r - \hat\mu)^2\big)$$

The `(μ̂_r − μ̂)²` term folds the variation across splits into the standard error, so the SE can
only widen, never average away. This is the lever that fixes residual under-coverage. The
package recommendation for linear ATE coverage is `n_folds >= 50` with `n_repeats >= 5`.

## What you get back

The result object (`DMLResult` or `InferenceResult`) exposes:

```{list-table}
:header-rows: 1

* - Attribute
  - Meaning
* - `mu_hat`
  - Debiased point estimate (mean of `ψ`)
* - `se`
  - Standard error
* - `ci_lower`, `ci_upper`
  - 95% confidence interval bounds
* - `psi_values`
  - The `(n,)` influence values themselves
* - `theta_hat`
  - The `(n, d_θ)` fitted heterogeneous parameters
* - `diagnostics`
  - Regime, `n_folds`, `lambda_method`, min Hessian eigenvalue, correction ratio, and more
```

Call `result.summary()` for a statsmodels-style printout.

## See also

- [Estimation](../estimation/index.md) for how `Λ(x)` is obtained, which is what makes or breaks the standard error.
- [RieszNet](riesznet.md) for the alternative, learned correction.
- [Theory: the FLM framework](../theory/flm.md) for the derivation and the formal guarantees.
