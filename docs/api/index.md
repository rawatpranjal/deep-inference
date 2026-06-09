# API Reference

Complete API documentation for `deep-inference`.

```{toctree}
:maxdepth: 2
:caption: API Reference

inference
families
targets
lambda
models
metrics
```

## Quick Reference

### Three Entry Points

| API | Use Case |
|-----|----------|
| `structural_dml()` | Production, 13 GLM families, fixed target E[β] |
| `inference()` | Flexible targets, regime detection, RCT support |
| `did()` | Difference-in-differences (exact / neural / panel_fe) |

### Main Entry Points

```python
from deep_inference import structural_dml, inference, did

# Legacy API (production-ready)
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50
)

# New API (flexible)
from deep_inference.lambda_.compute import Normal

result = inference(
    Y=Y, T=T, X=X,
    model='logit',
    target='ame',              # Flexible target
    is_randomized=True,        # Regime A
    treatment_dist=Normal(0, 1)
)

# Difference-in-differences (method auto-selected from arguments)
result = did(Y, group, post)                              # exact 2x2
result = did(Y, group, post, X=X)                         # neural, E[tau(X)]
result = did(Y, group, post, X=X, unit=unit, time=time)   # panel two-way FE
```

### Available Families

```python
from deep_inference import FAMILY_REGISTRY
print(list(FAMILY_REGISTRY.keys()))
# ['linear', 'logit', 'poisson', 'tobit', 'negbin', 'gamma', 'gumbel', 'weibull',
#  'gaussian', 'probit', 'beta', 'zip', 'multinomial_logit']
```

### Family Classes

```python
from deep_inference import (
    LinearFamily, LogitFamily, PoissonFamily, TobitFamily,
    NegBinFamily, GammaFamily, GumbelFamily, WeibullFamily,
    MultinomialLogitFamily,
)
from deep_inference.families import (
    GaussianFamily, ProbitFamily, BetaFamily, ZIPFamily,
)
```

## Module Overview

### structural_dml

The main entry point. Trains a structural neural network with influence function-based inference.

```python
from deep_inference import structural_dml

result = structural_dml(
    Y,                      # Outcome variable (n,)
    T,                      # Treatment variable (n,)
    X,                      # Covariates (n, d)
    family='linear',        # Statistical family
    hidden_dims=[64, 32],   # Network architecture
    epochs=100,             # Training epochs
    n_folds=50,             # Cross-fitting folds
    lr=0.01,               # Learning rate
    batch_size=64,         # Mini-batch size
    weight_decay=1e-4,     # L2 regularization
    verbose=False          # Print progress
)
```

### DMLResult

The result object returned by `structural_dml`:

| Attribute | Description |
|-----------|-------------|
| `mu_hat` | Debiased point estimate of E[beta(X)] |
| `mu_naive` | Naive (biased) estimate |
| `se` | Standard error |
| `ci_lower` | Lower bound of 95% CI |
| `ci_upper` | Upper bound of 95% CI |
| `theta_hat` | Estimated parameters (n, theta_dim) |
| `psi` | Influence scores (n,) |
| `diagnostics` | Dict with training diagnostics |

### families

Statistical families defining loss functions, gradients, Hessians, and influence scores.

### targets

Target functionals for inference. Built-in: `AverageParameter`/`AverageBeta`, `AME`.
Economic: `Elasticity`, `WTP`, `ConsumerWelfare`. Counterfactual: `DoseResponse`,
`Profit`, `TailProbability`, `ConditionalVariance`. Combinatorial: `MultiTreatmentATE`.
Multinomial: `ChoiceProbabilityTarget`, `MultinomialAME`. Plus `CustomTarget` /
`target_from_fn` for arbitrary differentiable functionals.

### lambda

Lambda estimation strategies: `ComputeLambda` (Regime A), `AnalyticLambda` (B), `EstimateLambda` (C).

### models

Structural models mapping covariates to parameters: `StructuralNet` (the network
backbone) plus `LinearModel`, `LogitModel`, `MultinomialLogitModel`,
`CombinatorialModel`, `QuantileModel`, `DiDModel` (saturated 2x2 DiD), and
`FEPanelDiDModel` (two-way fixed-effects panel DiD). `CustomModel` /
`model_from_loss` build a model from any loss function.

### metrics

Helper functions for computing coverage and SE ratios.
