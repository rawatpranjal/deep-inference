# RieszNet Procedure

RieszNet is the second inference procedure. It reaches the same debiased estimate as the
[influence-function procedure](influence_function.md), but it **learns** the bias correction with
a neural network instead of deriving it from the model's Hessian. This is the "automatic
debiasing" idea of Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022).

```{admonition} Status
:class: note
The conceptual design below reflects the procedure as implemented. The importable
`riesz_inference(...)` entry point and its validated coverage numbers are documented in full
once the core port lands; this page is being filled out alongside that work. The working
reference implementation lives at `exploration/spike.py`.
```

## The one idea: the Riesz representer

Every linear target (average treatment effect, average marginal effect, a policy value) can be
written as the expectation of the outcome model `g` against a special weighting function called
the **Riesz representer**, written `α(Z)`. For the average treatment effect the representer has a
familiar closed form,

$$\alpha_0(Z) = \frac{T}{p_0(X)} - \frac{1-T}{1 - p_0(X)},$$

where `p_0(X)` is the propensity score. The debiasing term is then `α(Z)·(Y − g(Z))`: it adds
back exactly what a plug-in estimate of the target leaves out.

The influence-function procedure constructs this weight analytically from the model. RieszNet
instead trains a network to produce `α(Z)` directly, so you never derive a Hessian or a
propensity by hand. That is the entire selling point: it generalizes to targets where the
representer has no convenient closed form.

## The architecture

RieszNet is one network with a shared trunk and two heads.

```{mermaid}
flowchart TD
    Z["Inputs (T, X)"] --> TRUNK["Shared trunk: 3 ELU layers, width 200"]
    TRUNK --> RH["Riesz head: linear, outputs a(Z)"]
    TRUNK --> GH["Regression head: 2 ELU layers, width 100, outputs g(Z)"]
    RH --> LOSS["Combined loss"]
    GH --> LOSS
```

- The **regression head** fits the outcome model `g(Z)` with the appropriate loss (squared error
  for linear, cross-entropy for logit, the Poisson or Gamma negative log-likelihood, and so on).
- The **Riesz head** is linear on the shared representation and outputs the representer `a(Z)`.
- A small unpenalized parameter `ε` implements targeted regularization, the step that buys
  coverage.

## The loss and the moment

Training minimizes one combined objective,

$$\mathcal{L} = \underbrace{\text{RegLoss}(g)}_{\text{fit the outcome}} \;+\; \lambda_1\,\underbrace{\mathbb{E}\big[a(Z)^2 - 2\,m(a)\big]}_{\text{learn the representer}} \;+\; \lambda_2\,\underbrace{\text{TargetedReg}(\varepsilon)}_{\text{coverage}},$$

with `λ1 = 0.1`, `λ2 = 1`. The middle term is the Riesz loss; minimizing it makes `a(Z)` the
representer without ever writing the representer down. Inference then uses the doubly-robust
moment

$$\psi = \big(\tilde g_1 - \tilde g_0\big) + a(Z)\,\big(Y - \tilde g\big),$$

averaged over a cross-fit. "Doubly robust" means the estimate stays valid if **either** the
outcome model `g` or the representer `a` is estimated well, not necessarily both.

## When to use it

- You want debiasing without deriving model-specific curvature.
- You want an independent, doubly-robust check on the influence-function result. In the
  [Replications](../replications/index.md) tables the two procedures are run side by side on
  known-truth data; where both are valid they agree, and the page flags the cases where RieszNet
  is unreliable (for example heavy-tailed count outcomes, where its empirical standard error is
  inflated by divergent repeats).

## See also

- [Influence Function](influence_function.md), the default procedure.
- [Replications](../replications/index.md) for the head-to-head numbers.
- The RieszNet paper, annotated, in [References](../references/index.md).
