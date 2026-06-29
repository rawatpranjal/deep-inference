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
* - Continuous, with modelled variance
  - Gaussian
  - [Gaussian](gaussian.md)
* - Binary (0/1), logistic link
  - Logit
  - [Logit](logit.md)
* - Binary (0/1), normal link
  - Probit
  - [Probit](probit.md)
* - Counts
  - Poisson
  - [Poisson](poisson.md)
* - Overdispersed counts
  - Negative Binomial
  - [Negative Binomial](negbin.md)
* - Counts with excess zeros
  - Zero-Inflated Poisson
  - [Zero-Inflated Poisson](zip.md)
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
* - Bounded in (0,1)
  - Beta
  - [Beta](beta.md)
* - Discrete choice (3+ options)
  - Multinomial Logit
  - [Multinomial Logit](multinomial.md)
* - Quantiles
  - Quantile
  - [Quantile](quantile.md)
* - Multiple treatments
  - Combinatorial
  - [Combinatorial](combinatorial.md)
* - Before/after, treated/control
  - Difference-in-Differences
  - [Difference-in-Differences](did.md)
```

## Which regime does each model land in?

The regime (covered in [Estimation](../estimation/index.md)) follows from one fact: does the
Hessian depend on the parameters? Linear-style squared-error losses have a parameter-free
Hessian and use the fast two-way path (Regime B); everything nonlinear estimates the curvature
on a three-way split (Regime C), unless the treatment is randomized with a known law (Regime A).

```{list-table}
:header-rows: 1

* - Regime
  - Models
* - B (analytic, 2-way)
  - Linear, Difference-in-Differences
* - C (estimated, 3-way)
  - Logit, Probit, Poisson, Negative Binomial, ZIP, Gamma, Weibull, Gumbel, Tobit, Beta, Gaussian, Multinomial, Quantile, Combinatorial
* - A (computed, randomized)
  - Any model when the treatment law is known
```

```{toctree}
:maxdepth: 1
:caption: Models

linear
gaussian
logit
probit
poisson
negbin
zip
gamma
weibull
gumbel
tobit
beta
multinomial
quantile
combinatorial
did
```
