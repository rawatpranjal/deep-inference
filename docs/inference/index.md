# Inference

[Estimation](../estimation/index.md) gives you a fitted network and the nuisance object `Λ(x)`.
**Inference** is the next step: turning those into a debiased point estimate, a standard error,
and a confidence interval you can defend.

This package ships two procedures. They target the same number and, when both are valid, they
agree. They differ in how they build the bias correction.

```{list-table}
:header-rows: 1

* - Procedure
  - How the correction is built
  - When to reach for it
* - [Influence Function](influence_function.md)
  - Analytic, from the model's score and Hessian
  - The default. Works for every model in the package.
* - [RieszNet](riesznet.md)
  - Learned by a second neural-network head
  - When you want automatic debiasing with no hand-derived Hessian, or a doubly-robust check.
```

Both produce the same kind of result object, with `mu_hat`, `se`, `ci_lower`, `ci_upper`, and
`psi_values` (the per-observation influence scores). The mechanics of forming the standard error
from `psi_values`, including the pooled variance estimator and the median-of-splits rule for
repeated cross-fitting, are on the [Influence Function](influence_function.md) page.

```{toctree}
:maxdepth: 1
:caption: Inference

influence_function
riesznet
```
