# Theory

The mathematical foundations, in two tracks. This is the section to read slowly. The
[Estimation](../estimation/index.md) and [Inference](../inference/index.md) sections tell you how
to run the methods; this section is the why, the derivations and the guarantees.

```{toctree}
:maxdepth: 1
:caption: Theory

flm
riesznet
```

- **[The FLM Framework](flm.md)** is the influence-function theory, worked through linear and
  logit: the enriched structural model, why naive plug-in inference fails, the influence-function
  correction and where it comes from, the three regimes for the expected Hessian, and the formal
  convergence and normality guarantees.
- **[RieszNet and Automatic Debiasing](riesznet.md)** is the Riesz-representer theory: the
  debiasing term, the mixed-bias (double-robustness) property, the Riesz loss that learns the
  representer without an analytic formula, targeted regularization, and the bridge showing that
  the FLM correction and the RieszNet correction are the same object.

## The core insight

**Machine learning and economic structure are complements, not substitutes.**

- **ML alone** fits data well but extrapolates nonsensically and cannot answer causal questions.
- **Structure alone** provides interpretability but misses heterogeneity.
- **Combined**, ML learns the heterogeneity patterns $\theta(X)$ while the structure keeps the
  economics, and the target, valid.

> "The central idea is that machine learning methods and economic structure are complements, not
> substitutes. Machine learning methods alone predict well, but extrapolate nonsensically.
> Economic structure alone can produce robust inference, but may miss important heterogeneity
> that is visible in the data."
> (Farrell, Liang, Misra, 2021)

The papers are collected, annotated, on the [References](../references/index.md) page.
