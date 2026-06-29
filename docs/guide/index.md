# Guide

The other sections explain what each piece is. This section is about the decisions you actually
face when you sit down with your own data: which model, which method for the nuisance object
`Λ(x)`, how to read the diagnostics, and what the common worked applications look like.

## The decision flow

The [decision flowchart](flowchart.md) walks you from "what does my outcome look like" to a
concrete configuration in a few yes/no steps. Start there if you are unsure which model and
regime apply.

## Choosing the `Λ` method (nonlinear models)

For nonlinear models the expected Hessian `Λ(x)` is estimated, and the estimator matters for
coverage. The package default is `lambda_method='ridge'`, validated at 96% coverage.

```{list-table}
:header-rows: 1

* - Method
  - Correlation with oracle
  - Coverage
  - Use it?
* - `ridge` (default)
  - 0.508
  - 96%
  - Yes, the safe default.
* - `lgbm`
  - 0.978
  - 96%
  - Good accuracy alternative.
* - `aggregate`
  - 0.000
  - 95%
  - Ignores `X`-dependence; valid but blunt.
* - `rf`
  - 0.904
  - ~90%
  - Moderate.
* - `mlp`
  - 0.997
  - 67%
  - No. High correlation but invalid standard errors.
```

```{warning}
The highest correlation with the oracle (`mlp`, 0.997) gives the **worst** coverage (67%).
Correlation of the `Λ` estimate with the truth is not what makes inference valid; the variance
of the estimate matters too. Use `ridge` unless you have checked an alternative yourself.
```

## Reading the diagnostics

After a run, inspect `result.diagnostics`:

```{list-table}
:header-rows: 1

* - Diagnostic
  - Healthy range
  - What it tells you
* - `min_lambda_eigenvalue`
  - `> 1e-4`
  - The Hessian is invertible; below this the correction is unstable.
* - correction ratio
  - roughly 0.1 to 50
  - How large the bias correction was relative to the naive piece.
* - mean condition number
  - small (single digits)
  - How close `Λ(x)` is to singular at low overlap.
```

## Worked applications

```{toctree}
:maxdepth: 1
:caption: Guide

flowchart
pricing
continuous_treatment
multimodal
```

- [Pricing and elasticity](pricing.md): heterogeneous price response, the Dubé and Misra setting.
- [Continuous treatment](continuous_treatment.md): dose-response with a continuous `T`.
- [Multimodal inputs](multimodal.md): high-dimensional `X` such as images or text embeddings.
