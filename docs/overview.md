# Overview

This page is the map. Read it once, top to bottom, and you will know what the package does,
why it exists, and where every other page fits. Nothing here assumes you have used the package
before.

## What problem does this solve?

You have data on an outcome `Y`, a treatment or policy variable `T`, and a vector of covariates
`X`. You believe the effect of `T` on `Y` is **heterogeneous**: it differs from person to
person, depending on `X`. You want two things at once:

1. A flexible model of that heterogeneity. A neural network is good at this, because it can
   learn the shape of the effect `β(X)` without you guessing a functional form.
2. A valid confidence interval for a summary you care about, for example the average effect
   `μ* = E[β(X)]`. "Valid" means a 95% interval that actually contains the truth 95% of the
   time.

The catch is that you cannot have both for free. A neural network is regularized (early
stopping, weight decay, dropout), and that regularization introduces **bias** into any summary
you read off the fitted network. If you ignore the bias and just take the standard deviation of
the network's predictions, your confidence intervals are far too narrow. They cover the truth
far less than 95% of the time.

```{list-table} The whole motivation in one table
:header-rows: 1

* - Method
  - Coverage (target 95%)
  - SE ratio (target 1.0)
* - Naive neural network
  - 8%
  - 0.27
* - **Influence-function corrected**
  - **95%**
  - **1.08**
```

The "Naive" row is what you get if you treat the network's output like a regression coefficient.
Eight percent coverage means the interval you call "95%" is wrong nineteen times out of twenty.
The corrected row is what this package delivers.

## How does it fix the bias?

There are two procedures in the box. Both correct the regularization bias; they arrive at the
correction differently.

```{mermaid}
flowchart LR
    D["Data: Y, T, X"] --> NN["Neural net learns theta(X) = (alpha(X), beta(X))"]
    NN --> NAIVE["Naive summary: mean of beta(X), biased"]
    NN --> CORR["Add the influence-function correction term"]
    CORR --> VALID["Debiased estimate plus valid 95% CI"]
```

- **Influence-function procedure (Farrell, Liang, Misra).** This is the default. It computes a
  per-observation correction term built from the model's score and its Hessian, then averages
  the corrected quantity. The correction is exactly the amount of bias the regularization
  injected, removed. See [Inference](inference/index.md).
- **RieszNet procedure (Chernozhukov et al.).** This learns the correction directly with a
  second neural-network head, without you deriving any Hessian by hand. It is the
  "automatic debiasing" route. See [the RieszNet page](inference/riesznet.md).

Both rest on the same idea: machine learning and economic structure are complements, not
substitutes. The network supplies the flexible heterogeneity; the structure (a known loss, a
known target) supplies the object you can do honest inference on.

> "The central idea is that machine learning methods and economic structure are complements,
> not substitutes. Machine learning methods alone predict well, but extrapolate nonsensically.
> Economic structure alone can produce robust inference, but may miss important heterogeneity
> that is visible in the data."
> (Farrell, Liang, Misra, 2021)

## What can you model?

The outcome can be almost any common type. Each one is a **model** (the package historically
calls these "families"), and each has its own detailed page under [Models](models/index.md).

```{list-table}
:header-rows: 1

* - Outcome type
  - Model
* - Continuous, unbounded
  - Linear, Gaussian
* - Binary (0/1)
  - Logit, Probit
* - Counts (0,1,2,...)
  - Poisson, Negative Binomial, Zero-Inflated Poisson
* - Positive, skewed
  - Gamma, Weibull
* - Censored
  - Tobit
* - Extreme values
  - Gumbel
* - Bounded in (0,1)
  - Beta
* - Discrete choice (3+ options)
  - Multinomial Logit
* - Quantiles
  - Quantile regression
* - Multiple treatments
  - Combinatorial
* - Before/after, treated/control
  - Difference-in-Differences, Panel Fixed-Effects DiD
```

## What can you estimate?

The thing you want a confidence interval for is the **target**. The default target is the
average effect `E[β(X)]`, but the package ships many economic functionals, and you can supply
your own (autodiff handles the derivatives).

Average marginal effect, price elasticity, willingness to pay, consumer welfare,
dose-response, expected profit, tail probability, conditional variance, combinatorial
treatment effects, and custom targets. Each is covered on its own page; targets are summarized
in the [API reference](api/targets.md).

## The three regimes (read this before you run anything serious)

The correction needs one nuisance object, the expected curvature of the loss, written `Λ(x)`.
How the package gets `Λ(x)` depends on your data, and that choice is called the **regime**.

```{list-table}
:header-rows: 1

* - Regime
  - When it applies
  - How `Λ(x)` is obtained
* - A
  - Randomized experiment with a known treatment distribution
  - Computed by Monte-Carlo integration (no estimation)
* - B
  - Linear / squared-error model
  - Closed form (the Hessian does not depend on the parameters)
* - C
  - Observational data, nonlinear model
  - Estimated with a separate regression, on a three-way data split
```

You usually do not pick the regime by hand. The package detects it from your inputs. The full
explanation is in [Estimation](estimation/index.md) and [Theory](theory/index.md).

## How this documentation is organized

Read the pages in menu order. They are written to be read linearly.

```{list-table}
:header-rows: 1

* - Section
  - What it gives you
* - [Quick Start](quickstart.md)
  - Install, then a runnable example you can paste and run
* - [Loading Data & Pre-Estimation](loading_data.md)
  - The exact shapes `Y`, `T`, `X` must have, and what to do before fitting
* - [Models](models/index.md)
  - One detailed page per outcome type
* - [Estimation](estimation/index.md)
  - How the network is fit and how `Λ(x)` is obtained (the machinery)
* - [Inference](inference/index.md)
  - How standard errors and intervals are formed; the two procedures
* - [Guide](guide/index.md)
  - Practical decisions: which model, which `Λ` method, how to read diagnostics
* - [Theory](theory/index.md)
  - The formal walkthrough, in order, with the theorems
* - [Replications](replications/index.md)
  - What we have validated against known truth, with the numbers
* - [Simulation Studies](simulation_studies/index.md)
  - Head-to-head method comparisons (coverage, SE ratio, bias)
* - [API Reference](api/index.md)
  - Every function and result object
* - [References](references/index.md)
  - The papers, annotated
```

## Who is this for?

Applied economists and data scientists who fit structural or causal models and need defensible
standard errors, not just point predictions. You should be comfortable with a generalized linear
model and with running Python. You do not need to know anything about influence functions going
in; that is what the [Theory](theory/index.md) section is for.
