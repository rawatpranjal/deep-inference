# Targets Module

The `targets` module defines what quantity you want to estimate. A **target** is a functional H(θ) that maps the structural parameters to a scalar.

## Overview

| Target | Formula | Use Case |
|--------|---------|----------|
| `AverageParameter` | E[θ_j] | Average treatment effect |
| `AME` | E[p(1-p)·β] | Marginal effect on probability |
| `Elasticity` | E[(1-p)·β·t̃] (logit) | Price elasticity of demand |
| `WTP` | E[-θ_attr / θ_price] | Willingness to pay |
| `ConsumerWelfare` | E[log(1+e^V)/\|β_price\|] | Expected consumer surplus |
| `DoseResponse` | E[G(θ, t̃)] | Average predicted outcome at treatment level |
| `Profit` | E[t̃ · G(θ, t̃)] | Expected revenue at price level |
| `TailProbability` | E[P(Y > c \| θ, t̃)] | Exceedance/risk probability |
| `ConditionalVariance` | E[Var(Y \| θ, t̃)] | Model-implied outcome variance |
| `MultiTreatmentATE` | E[G(θ, t) - G(θ, t₀)] | Combinatorial treatment effect |
| `ChoiceProbabilityTarget` | P(Y=j \| W, X) | Multinomial choice probability |
| `MultinomialAME` | ∂P(Y=j)/∂x_{jk} | Multinomial marginal effect |
| `CustomTarget` | User-defined h(x,θ,t̃) | Any custom functional |

---

## Built-in Targets

### AverageParameter

The default target: average of a specific parameter across the population.

```python
from deep_inference.targets import AverageParameter

# Target: E[β(X)] where β is the second parameter (index=1)
target = AverageParameter(param_index=1, theta_dim=2)
```

**Formula:**
```
H(θ) = (1/n) Σ θ_j(x_i)
```

**Jacobian (closed-form):**
```
∂H/∂θ = [0, ..., 1/n, ..., 0]  (1 at position j)
```

### AME (Average Marginal Effect)

For logit models, the marginal effect on probability (not log-odds).

```python
from deep_inference.targets import AME

# Target: E[p(1-p)·β] evaluated at t_tilde
target = AME(param_index=1, model_type='logit')
```

**Formula (logit):**
```
H(θ, t̃) = (1/n) Σ σ'(α_i + β_i·t̃) · β_i
        = (1/n) Σ p_i(1-p_i) · β_i
```

Where p_i = σ(α_i + β_i·t̃).

**Jacobian (closed-form for logit):**
```
∂H/∂α = p(1-p)(1-2p)·β / n
∂H/∂β = p(1-p)[1 + (1-2p)·β·t̃] / n
```

### Elasticity

Price elasticity of demand — the key object in personalized pricing.

```python
from deep_inference.targets import Elasticity

# logit (default); also 'poisson', 'gamma', 'negbin'
target = Elasticity(model_type='logit')
```

**Formula (logit):**
```
H(θ, t̃) = (1 - p)·β·t̃    where p = σ(α + β·t̃)
```

**Formula (log-link: poisson/gamma/negbin):**
```
H(θ, t̃) = β·t̃
```

**Jacobian (closed-form):**
```
# logit
∂H/∂α = -p(1-p)·β·t̃
∂H/∂β = (1-p)·t̃ - p(1-p)·β·t̃²
# log-link
∂H/∂α = 0
∂H/∂β = t̃
```

**Usage:**
```python
result = inference(Y, T, X, model='logit', target='elasticity', t_tilde=1.0)
```

*Reference: Dubé & Misra (2022, JPE)*

### WTP (Willingness To Pay)

The marginal rate of substitution between an attribute and price: how much a consumer would pay for a one-unit improvement in the attribute.

```python
from deep_inference.targets import WTP

# Ratio of two coefficients in theta
target = WTP(attribute_index=1, price_index=2)
```

**Formula:**
```
H(θ) = -θ_attribute / θ_price
```

**Jacobian (closed-form delta method):**
```
∂H/∂θ_attr  = -1 / θ_price
∂H/∂θ_price =  θ_attr / θ_price²
```

Returns `NaN` (and falls back to `None` Jacobian) when the price coefficient is within `1e-8` of zero. `t_tilde` is not used.

**Usage:**
```python
result = inference(Y, T, X, model='logit', target='wtp')
```

*Reference: Dubé & Misra (2022, JPE)*

### ConsumerWelfare

Expected consumer surplus from logit demand (Small & Rosen 1981 logsum), scaled by the marginal utility of income.

```python
from deep_inference.targets import ConsumerWelfare

target = ConsumerWelfare(price_coef_index=1)
```

**Formula (binary logit):**
```
H(θ, t̃) = log(1 + exp(V)) / |β_price|    where V = α + β_price·t̃
```

Implemented with `softplus(V)` for numerical stability. The Jacobian is left to autodiff (`jacobian()` returns `None`) because it involves derivatives of `softplus` and `|β_price|`.

**Usage:**
```python
result = inference(Y, T, X, model='logit', target='welfare', t_tilde=0.0)
```

*Reference: Small & Rosen (1981); Dubé & Misra (2022, JPE)*

### DoseResponse

Average predicted outcome at a given treatment level — the fundamental counterfactual target.

```python
from deep_inference.targets import DoseResponse

# Target: E[σ(α + β·t̃)] at treatment level t̃=1.0
target = DoseResponse(model_type='logit')
```

**Formula (logit):**
```
H(θ, t̃) = σ(α + β·t̃)
```

**Jacobian (closed-form):**
```
∂H/∂α = p(1-p)
∂H/∂β = p(1-p)·t̃
```

Where p = σ(α + β·t̃). Supports `logit`, `linear`, and `poisson` model types.

**Usage:**
```python
from deep_inference import inference
result = inference(Y, T, X, model='logit', target='dose_response', t_tilde=1.0)
```

*Reference: Colangelo & Lee (2026, JBES)*

### Profit

Expected revenue per consumer at a given price level: price × purchase probability.

```python
from deep_inference.targets import Profit

target = Profit(model_type='logit')
```

**Formula (logit):**
```
H(θ, t̃) = t̃ · σ(α + β·t̃)
```

**Jacobian (closed-form):**
```
∂H/∂α = t̃·p(1-p)
∂H/∂β = t̃·p(1-p)·t̃
```

**Usage:**
```python
result = inference(Y, T, X, model='logit', target='profit', t_tilde=2.0)
```

*Reference: Dubé & Misra (2023, JPE)*

### TailProbability

Probability that the outcome exceeds a threshold: P(Y > c | θ, t̃).

```python
from deep_inference.targets import TailProbability

# P(Y > 5 | Poisson rate) at exposure level t̃=1.0
target = TailProbability(threshold=5, model_type='poisson')
```

**Formula (Poisson):**
```
H(θ, t̃) = 1 - Σ_{k=0}^{c} e^{-λ}·λ^k/k!   where λ = exp(α + β·t̃)
```

**Formula (logit, c=0):**
```
H(θ, t̃) = σ(α + β·t̃)    (identical to DoseResponse)
```

**Formula (linear/Gaussian):**
```
H(θ, t̃) = 1 - Φ((c - μ)/σ)   where μ = α + β·t̃
```

Closed-form Jacobian for logit and linear; autodiff for Poisson.

*Reference: Melnychuk & Feuerriegel (2026, ICLR)*

### ConditionalVariance

Model-implied variance of outcomes — captures heterogeneity in risk across covariates.

```python
from deep_inference.targets import ConditionalVariance

target = ConditionalVariance(model_type='logit')
```

**Formula (logit):**
```
H(θ, t̃) = p(1-p)   where p = σ(α + β·t̃)
```

**Jacobian (closed-form):**
```
∂H/∂α = p(1-p)(1-2p)
∂H/∂β = p(1-p)(1-2p)·t̃
```

**Usage:**
```python
result = inference(Y, T, X, model='logit', target='conditional_variance', t_tilde=0.0)
```

*Reference: Melnychuk & Feuerriegel (2026, ICLR)*

### MultiTreatmentATE

Average treatment effect for combinatorial experiments with multiple binary treatments.

```python
from deep_inference.targets import MultiTreatmentATE
from deep_inference.models.combinatorial import CombinatorialModel

model = CombinatorialModel(n_treatments=3, link='gen_sigmoid_ii')
target = MultiTreatmentATE(
    model=model,
    treatment=[1, 0, 1],   # Apply treatments 1 and 3
    control=[0, 0, 0],     # vs no treatment
)
```

**Formula:**
```
H(θ) = G(θ, t) - G(θ, t₀)
```

Where G is the structural link function from `CombinatorialModel`. Jacobian computed via autodiff.

*Reference: Ye et al. (2025, Management Science)*

---

### ChoiceProbabilityTarget

For multinomial logit models: probability of choosing a specific alternative.

```python
from deep_inference.targets.choice_probability import ChoiceProbabilityTarget

# P(Y=j | W, X) for alternative j=1
target = ChoiceProbabilityTarget(
    alternative=1,        # Which alternative (0-indexed)
    n_alternatives=3,     # Total alternatives J
    n_attributes=2        # Attributes per alternative K
)
```

**Formula:**
```
H = P(Y=j|W,X) = softmax(V)[j]
V_j = alpha_j + x'_j * beta
```

**Jacobian (closed-form):**
```
∂P_j/∂alpha_m = P_{m+1}(delta_{j,m+1} - P_j)
∂P_j/∂beta_k = P_j(x̃_{jk} - x̄_pk)
```

Where $\bar{x}_{pk} = \sum_j P_j x_{jk}$ is the probability-weighted mean attribute.

### MultinomialAME

Average Marginal Effect for multinomial logit: how a unit change in attribute $k$ affects the probability of choosing alternative $j$.

```python
from deep_inference.targets.choice_probability import MultinomialAME

# dP(Y=1)/dx_{1,0}: effect of attribute 0 on probability of alternative 1
target = MultinomialAME(
    alternative=1,        # Which alternative
    attribute=0,          # Which attribute
    n_alternatives=3,     # Total alternatives J
    n_attributes=2        # Attributes per alternative K
)
```

**Formula:**
```
∂P_j/∂x_{jk} = beta_k * P_j * (1 - P_j)
```

This generalizes the familiar binary logit AME to the multinomial context.

---

## Custom Targets

Define any target function and the Jacobian is computed via autodiff.

### CustomTarget

```python
from deep_inference.targets import CustomTarget
import torch

def my_target(x, theta, t_tilde):
    """
    Custom target function.

    Args:
        x: Covariates (unused for average targets)
        theta: (theta_dim,) parameter vector
        t_tilde: Evaluation point for treatment

    Returns:
        Scalar value
    """
    alpha, beta = theta[0], theta[1]
    return torch.sigmoid(alpha + beta * t_tilde)

target = CustomTarget(h_fn=my_target)
```

### Example: Average Prediction

```python
import torch
from deep_inference import inference

def avg_prediction(x, theta, t_tilde):
    """E[P(Y=1|T=t̃)] = E[σ(α + β·t̃)]"""
    return torch.sigmoid(theta[0] + theta[1] * t_tilde)

result = inference(
    Y, T, X,
    model='logit',
    target_fn=avg_prediction,
    t_tilde=0.0  # Prediction at T=0
)
```

### Example: Counterfactual Comparison

```python
def treatment_effect(x, theta, t_tilde):
    """E[P(Y=1|T=1) - P(Y=1|T=0)]"""
    alpha, beta = theta[0], theta[1]
    p1 = torch.sigmoid(alpha + beta * 1.0)
    p0 = torch.sigmoid(alpha + beta * 0.0)
    return p1 - p0

result = inference(
    Y, T, X,
    model='logit',
    target_fn=treatment_effect
)
```

---

## Target Protocol

All targets implement this interface:

```python
class Target(Protocol):
    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """Compute target value for a single observation."""
        ...

    def jacobian(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """Compute ∂h/∂θ. Falls back to autodiff if not implemented."""
        ...
```

### Implementing Custom Targets

```python
from deep_inference.targets import BaseTarget
import torch

class MyTarget(BaseTarget):
    def h(self, x, theta, t_tilde):
        # Your target computation
        return theta[0] ** 2 + theta[1] * t_tilde

    def jacobian(self, x, theta, t_tilde):
        # Optional: closed-form Jacobian (faster than autodiff)
        return torch.tensor([2 * theta[0], t_tilde])
```

---

## Using Targets with inference()

### Built-in Target Strings

```python
# Average beta (log-odds for logit)
result = inference(Y, T, X, model='logit', target='beta')

# Average marginal effect
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)

# Economic targets
result = inference(Y, T, X, model='logit', target='elasticity', t_tilde=1.0)
result = inference(Y, T, X, model='logit', target='wtp')
result = inference(Y, T, X, model='logit', target='welfare', t_tilde=0.0)
```

Recognized `target=` strings include: `beta`, `average_slope`, `ame`, `elasticity`,
`wtp`, `welfare`, `dose_response`, `profit`, `tail_probability`,
`conditional_variance`, `qte`, `tau`, `att`, `fe_effect`.

### Custom Target Functions

```python
def my_target(x, theta, t_tilde):
    return theta[0] + theta[1] * t_tilde

result = inference(Y, T, X, model='logit', target_fn=my_target, t_tilde=1.0)
```

---

## Autodiff Jacobian

When you provide a custom target function, the package automatically computes the Jacobian via PyTorch autodiff:

```python
# Internally, this happens:
theta.requires_grad_(True)
h_value = target_fn(x, theta, t_tilde)
jacobian = torch.autograd.grad(h_value, theta)[0]
```

This enables arbitrary differentiable targets without manual derivatives.
