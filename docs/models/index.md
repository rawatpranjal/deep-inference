# Models

A **model** says how the outcome `Y` is generated from the linear index `η = α(X) + β(X)·T`.
Picking the model is the same decision as picking a generalized linear model: you choose it by
the type of your outcome. The neural network then learns the heterogeneous parameters
`θ(X) = (α(X), β(X))`; the model supplies the loss, the link, and the derivatives that the
inference machinery needs.

Every model page below follows the same layout, so once you have read one you can navigate any
of them:

1. **Source papers** that introduce or use the model.
2. **When to use** it, with concrete examples.
3. **Model**: the data-generating process and link function.
4. **Loss / likelihood**: the negative log-likelihood that is minimized.
5. **Score and Hessian**: the gradient and curvature, and whether they depend on `θ` or `Y`.
6. **Target**: the estimand you get a confidence interval for.
7. **Influence function**: the residual, weight, and score that build the correction.
8. **Algorithm and regime**: which of the three regimes applies and why.
9. **Usage**: runnable Python.
10. **Evidence**: recovery and coverage numbers, with the source.
11. **Diagnostics and pitfalls**.
12. **References and API links**.

## Pick your model by outcome type

```{list-table}
:header-rows: 1

* - Outcome
  - Model
  - Page
* - Continuous, unbounded
  - Linear
  - [Linear](linear.md)
* - Binary (0/1)
  - Logit
  - [Logit](logit.md)
* - Counts
  - Poisson
  - [Poisson](poisson.md)
* - Overdispersed counts
  - Negative Binomial
  - [Negative Binomial](negbin.md)
* - Positive, skewed
  - Gamma
  - [Gamma](gamma.md)
* - Time-to-event
  - Weibull
  - [Weibull](weibull.md)
* - Extreme values
  - Gumbel
  - [Gumbel](gumbel.md)
* - Censored
  - Tobit
  - [Tobit](tobit.md)
* - Discrete choice (3+ options)
  - Multinomial Logit
  - [Multinomial Logit](multinomial.md)
```

```{note}
More model pages (Gaussian, Probit, Beta, Zero-Inflated Poisson, Quantile, Combinatorial, and
the Difference-in-Differences family) are being written to the same template and will appear
here as they land. Until then, see the [API reference](../api/families.md) for their signatures
and the [GLM formula table](../guide/index.md) for their loss and Hessian.
```

```{toctree}
:maxdepth: 1
:caption: Models

linear
logit
poisson
negbin
gamma
weibull
gumbel
tobit
multinomial
```
