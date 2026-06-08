# 5. Three Regimes for Lambda

The expected Hessian $\Lambda(X) = \mathbb{E}[\nabla_\theta^2 \ell(Y, T, \theta)
\mid X]$ from the [IF correction](04_influence_function.md) depends on how the
model and data are structured. Farrell, Liang, and Misra (2025) identify three
cases that determine how $\Lambda$ is obtained and how many sample splits are
needed.

| Regime | When | Lambda method | Cross-fitting |
|--------|------|---------------|---------------|
| A | RCT, known $F_T$ | Compute (MC integration) | 2-way |
| B | Linear model | Analytic (closed-form) | 2-way |
| C | Observational + nonlinear | Estimate (ridge) | 3-way |

## Regime A — Randomized experiment with known treatment distribution

If $T$ is randomly assigned with known distribution $F_T$, and the Hessian does
not depend on $Y$, then $\Lambda(X)$ can be computed via Monte Carlo integration:

$$
\Lambda(X) = \int \nabla_\theta^2 \ell(y, t, \theta) \, dF_T(t).
$$

This is the simplest case and requires only two-way cross-fitting. Typical
examples include A/B tests where treatment assignment probabilities are known by
design, and randomized pricing experiments where the price distribution is
controlled by the researcher. The `inference()` API accepts `is_randomized=True`
and a `treatment_dist` argument to trigger this regime.

## Regime B — Linear model

For linear models the Hessian is constant ($\nabla_\theta^2 \ell = 2$), so
$\Lambda(X) = 2\,\mathbb{E}[T T' \mid X]$, which can be estimated analytically from
the data without a separate estimation step. This avoids three-way splitting
entirely, making the method fast and reliable even at moderate sample sizes. The
package detects Regime B automatically when `model='linear'` is specified.

## Regime C — Observational data with nonlinear model

In most applied settings — including the H&M application — the Hessian depends on
$\theta$ (e.g. through $p(1-p)$ in logit or the softmax probabilities in
multinomial logit). Since $\theta$ is estimated, we must **estimate** $\Lambda$ via
ridge regression, which requires a three-way sample split. The package handles the
splitting automatically.

```{warning}
For Regime C, use ridge (the default), `aggregate`, or `lgbm` for $\Lambda$
estimation. **Never use `mlp`**: it attains the highest correlation with the
oracle Hessian but produces only ~67% coverage, because high correlation does not
guarantee a low-variance estimate. Valid inference needs both.
```

## Regime selection in code

The regime is selected automatically based on the model and data:

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

# Regime A: randomized experiment with known F_T
result = inference(Y, T, X, model='logit', target='beta',
                   is_randomized=True, treatment_dist=Normal(0, 1))

# Regime B: linear model (auto-detected)
result = inference(Y, T, X, model='linear', target='beta')

# Regime C: observational + nonlinear (auto-detected)
result = inference(Y, T, X, model='logit', target='beta')
```
