# Continuous Treatment Guide

This tutorial demonstrates dose-response analysis with continuous treatments using `deep-inference`, following Colangelo & Lee (2026).

## Motivation

**Problem:** Many treatments are continuous (advertising spend, drug dosage, price levels). How do we:
1. Estimate the dose-response curve $E[Y(t)]$ at arbitrary treatment levels?
2. Get valid confidence intervals at each point?
3. Compute treatment elasticities?

**Key insight:** `deep-inference` handles continuous treatments natively. Every model family (linear, logit, Poisson, etc.) takes a scalar $T$ and estimates $\theta(X) = (\alpha(X), \beta(X))$ where $\beta$ is the treatment coefficient. The influence function framework provides valid inference at any evaluation point $\tilde{t}$.

## When to Use

Use this approach when:
- Treatment is continuous (not binary)
- You want heterogeneous dose-response, not just ATE
- Examples: advertising spend, drug dosage, pricing, subsidy levels

## Connection to Colangelo & Lee (2026)

Colangelo & Lee (2026) develop double debiased machine learning for nonparametric inference with continuous treatments. They show that deep neural networks (citing FLM 2021) are valid first-stage nuisance estimators. `deep-inference` provides a complementary *structural* approach: instead of nonparametrically estimating $E[Y|T=t,X]$, we model the data-generating process parametrically with heterogeneous coefficients $\theta(X)$.

**Trade-off:**

| Approach | Flexibility | Interpretability | Extrapolation |
|----------|-------------|-------------------|---------------|
| Nonparametric (C&L) | High | Low | Poor |
| Structural DNN (FLM) | Medium | High | Good |

The structural approach works best when you have a reasonable economic model (e.g., Poisson for counts, logit for binary choice).

## Example: Advertising and Sales

### Step 1: Generate Data

Poisson model: $Y \sim \text{Poisson}(\exp(\alpha(X) + \beta(X) \cdot T))$

```python
import numpy as np

np.random.seed(42)
n = 2000

# Firm characteristics
X = np.random.randn(n, 3)

# Heterogeneous parameters
alpha_true = 2.0 + 0.3 * X[:, 0]       # Baseline log-sales
beta_true = 0.1 + 0.05 * X[:, 0]       # Ad effectiveness (positive)

# Advertising spend (continuous treatment)
T = np.random.exponential(2.0, n)

# Sales (count outcome)
mu = np.exp(alpha_true + beta_true * T)
Y = np.random.poisson(mu).astype(float)
```

### Step 2: Estimate Average Treatment Effect

```python
from deep_inference import inference

# Average β (marginal effect in log-sales)
result_beta = inference(
    Y, T, X,
    model='poisson',
    target='beta',
    epochs=100,
    n_folds=50,
)

print(f"E[β(X)]: {result_beta.mu_hat:.4f}")
print(f"95% CI: [{result_beta.ci_lower:.4f}, {result_beta.ci_upper:.4f}]")
# True value: E[β(X)] = 0.1
```

### Step 3: Dose-Response via Elasticity

For a Poisson (log-link) model, the elasticity at treatment level $t$ is simply $\beta \cdot t$:

```python
# Elasticity at different spending levels
for t in [1.0, 2.0, 3.0, 5.0]:
    result = inference(
        Y, T, X,
        model='poisson',
        target='elasticity',
        t_tilde=float(t),
        epochs=100,
        n_folds=50,
    )
    print(f"t={t:.1f}: elasticity={result.mu_hat:.4f} +/- {result.se:.4f}")
```

### Step 4: Custom Dose-Response Target

For the predicted outcome $E[Y(t)] = \exp(\alpha + \beta \cdot t)$ at a given level:

```python
import torch

def dose_response(x, theta, t_tilde):
    """E[Y(t)] = exp(alpha + beta * t)"""
    return torch.exp(theta[0] + theta[1] * t_tilde)

result_dr = inference(
    Y, T, X,
    model='poisson',
    target_fn=dose_response,
    t_tilde=2.0,
    epochs=100,
    n_folds=50,
)

print(f"E[Y(2.0)]: {result_dr.mu_hat:.4f}")
print(f"95% CI: [{result_dr.ci_lower:.4f}, {result_dr.ci_upper:.4f}]")
```

## Multiple Evaluation Points

Trace out the full dose-response curve by evaluating at multiple $\tilde{t}$ values:

```python
t_values = np.linspace(0.5, 5.0, 10)
results = []

for t in t_values:
    r = inference(Y, T, X, model='poisson', target='elasticity',
                  t_tilde=float(t), epochs=100, n_folds=50)
    results.append({
        't': t,
        'elasticity': r.mu_hat,
        'se': r.se,
        'ci_lower': r.ci_lower,
        'ci_upper': r.ci_upper,
    })

# Plot with confidence bands
import matplotlib.pyplot as plt

ts = [r['t'] for r in results]
etas = [r['elasticity'] for r in results]
ci_lo = [r['ci_lower'] for r in results]
ci_hi = [r['ci_upper'] for r in results]

plt.fill_between(ts, ci_lo, ci_hi, alpha=0.2)
plt.plot(ts, etas, 'b-o')
plt.xlabel('Treatment level (t)')
plt.ylabel('Elasticity')
plt.title('Dose-Response Elasticity with 95% CI')
plt.show()
```

## Supported Model Families

All families support continuous treatment natively:

| Family | Link | Elasticity Formula | Use Case |
|--------|------|-------------------|----------|
| Linear | Identity | $\beta \cdot t$ | Continuous outcomes |
| Logit | Logit | $(1-p) \cdot \beta \cdot t$ | Binary outcomes |
| Poisson | Log | $\beta \cdot t$ | Count data |
| Gamma | Log | $\beta \cdot t$ | Positive continuous |
| NegBin | Log | $\beta \cdot t$ | Overdispersed counts |

## References

- Colangelo, K. & Lee, Y.-Y. (2026). Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments.
- Farrell, M. H., Liang, T., & Misra, S. (2021). Deep Neural Networks for Estimation and Inference. *Econometrica*, 89(1).
