# 2. Target Functionals

We are usually not interested in $\theta^*(X_i)$ for every individual, but rather
in some **summary**: an *average* of a function of $\theta^*$ across the
population,

$$
\mu^* = \mathbb{E}[H(X, \theta^*(X), \tilde{t})],
$$

where $H$ is the *target function* and $\tilde{t}$ is an evaluation point. This
$\mu^*$ is the scalar we want a valid confidence interval for.

## Common targets

| Target | $H(x, \theta, \tilde{t})$ | Interpretation |
|--------|---------------------------|----------------|
| Average parameter | $\theta_k$ | $\mathbb{E}[\beta_{\text{price}}]$: mean price sensitivity |
| Average marginal effect | $g'(\alpha + \beta \tilde{t}) \cdot \beta$ | Mean marginal effect at $\tilde{t}$ |
| Elasticity | $\beta \cdot \tilde{t} \cdot (1-p)$ | Price elasticity of demand |
| Dose-response | $g(\alpha + \beta \tilde{t})$ | Predicted outcome at $\tilde{t}$ |
| Profit | $\tilde{t} \cdot g(\alpha + \beta \tilde{t})$ | Expected revenue |
| DID / ATE | $g(\theta'\tilde{t}_1) - g(\theta'\tilde{t}_0)$ | Treatment effect contrast |
| Custom | Any $H(x, \theta, \tilde{t})$ | User-specified |

## The Jacobian requirement

The Jacobian $H_\theta = \partial H / \partial \theta$ is required for the
influence function correction introduced in
[The Influence Function Correction](04_influence_function.md). For built-in
targets, closed-form Jacobians are provided. For custom targets, the package
computes them via **automatic differentiation** — the user supplies only the
target function $H$.

```{tip}
Choosing a target is an economic modeling decision, not a statistical one. The
same fitted neural network $\hat\theta(X)$ can answer many questions — average
price sensitivity, elasticity at a chosen price, expected profit — simply by
swapping the target $H$.
```
