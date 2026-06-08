# 1. Setup

```{note}
This is the first of seven theory pages. They build on each other — read them
top to bottom. Use the **Next** button at the bottom of each page to follow the
phases in order.
```

This section lays out the Farrell–Liang–Misra (FLM) framework as a practitioner
would encounter it. We describe the structural model, define target functionals,
explain why naive neural network inference fails, present the influence function
correction, and classify problems into three "regimes" that determine how a key
quantity — the expected Hessian $\Lambda(x)$ — is handled.

## The structural model

Following Farrell, Liang, and Misra (2021, 2025), we observe $n$ i.i.d.
observations $(Y_i, T_i, X_i)$ where

- $Y_i$ is the **outcome** (chosen product, sales quantity, duration),
- $T_i$ is the **treatment** or "action" variable (price, product attributes, dosage),
- $X_i \in \mathbb{R}^{d_x}$ are **individual covariates** that drive heterogeneity.

A *structural model* specifies a loss function $\ell(y, t, \theta)$ parameterized
by structural parameters $\theta \in \mathbb{R}^{d_\theta}$. The true parameters
for individual $i$ solve

$$
\theta^*(X_i) = \arg\min_\theta \; \mathbb{E}[\ell(Y, T, \theta) \mid X = X_i].
$$

The key move of the framework is to let the structural parameters be **functions
of the covariates** $X$, rather than fixed constants. A deep neural network learns
the map $X \mapsto \theta^*(X)$, capturing rich heterogeneity while the structural
loss $\ell$ preserves economic interpretability.

## Concrete example (H&M application)

In the H&M application, $Y_i \in \{0, 1, \ldots, J-1\}$ is the chosen product from
a set of $J$ alternatives, $T_i \in \mathbb{R}^{J \times K}$ contains the
attributes of each alternative (log-price, style embeddings), and
$X_i \in \mathbb{R}^{64}$ is the consumer's learned embedding. The structural
parameters

$$
\theta^*(X_i) = \big(\beta_{\text{price}}(X_i),\, \beta_{\text{style},1}(X_i),\, \ldots,\, \beta_{\text{style},5}(X_i)\big)
$$

are the consumer's taste parameters, and the loss is the multinomial logit
negative log-likelihood:

$$
\ell(y, t, \theta) = -V_{y} + \log \sum_{j=0}^{J-1} e^{V_j}, \qquad V_j = x_j' \theta.
$$

## Available structural models

The framework is **not** limited to discrete choice. The table below lists the
structural models currently available in the `deep-inference` package, along with
their loss functions and parameter dimensions.

| Model | Loss $\ell(y,t,\theta)$ | Link | $d_\theta$ |
|-------|-------------------------|------|------------|
| Linear | $(y - \alpha - \beta t)^2$ | Identity | 2 |
| Logit | $\log(1+e^\eta) - y\eta$ | Logistic | 2 |
| Poisson | $e^\eta - y\eta$ | Log | 2 |
| Gamma | $y/\mu + \log \mu$ | Log | 2 |
| Multinomial | $-V_y + \log\sum_j e^{V_j}$ | Softmax | $(J\!-\!1)+K$ |
| Custom | Any differentiable $\ell(y,t,\theta)$ | Any | User-specified |

Here $\eta = \alpha + \beta t$ and $\mu = g^{-1}(\eta)$. The package includes over
a dozen families (Weibull, Gumbel, Tobit, NegBin, Probit, Beta, ZIP, and others).
Custom losses can be supplied via the `loss=` argument — see
[Available Models and Targets](07_models_and_targets.md).
