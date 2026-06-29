# Personalized Pricing Guide

This tutorial demonstrates how to use `deep-inference` for personalized pricing analysis with heterogeneous demand, following Dube & Misra (2022, *Journal of Political Economy*).

## Motivation

**Problem:** Firms set prices, but consumers respond differently based on their characteristics. How do we:
1. Estimate heterogeneous price elasticities $\eta(X)$?
2. Compute personalized optimal prices?
3. Measure welfare implications of price discrimination?

**Approach:** Use structural deep learning to estimate heterogeneous demand parameters $\theta(X) = (\alpha(X), \beta(X))$ from a logit demand model, then compute economic targets with valid inference.

## When to Use

Use this approach when:
- You have binary purchase data (buy/no-buy)
- Treatment is price or a continuous instrument
- You want heterogeneous price elasticities, not just average effects
- You need valid confidence intervals on economic quantities

## Mathematical Setup

### Logit Demand Model

$$P(\text{buy} \mid X, P) = \sigma(\alpha(X) + \beta(X) \cdot P)$$

where $\beta(X) < 0$ is the heterogeneous price sensitivity.

### Three Economic Targets

| Target | Formula | Interpretation |
|--------|---------|----------------|
| **Elasticity** | $\eta = (1-p) \cdot \beta \cdot P$ | % change in demand per % change in price |
| **WTP** | $-\beta_{\text{attr}} / \beta_{\text{price}}$ | Dollar value of a one-unit attribute improvement |
| **Consumer Surplus** | $\log(1 + e^V) / \lvert\beta_{\text{price}}\rvert$ | Expected utility in monetary units (Small & Rosen, 1981) |

## Example: Heterogeneous Price Sensitivity

### Step 1: Generate Data

```python
import numpy as np
from scipy.special import expit

np.random.seed(42)
n = 2000

# Consumer characteristics
X = np.random.randn(n, 3)

# Heterogeneous parameters
alpha_true = 1.0 + 0.3 * X[:, 0]       # Base utility
beta_true = -0.5 - 0.2 * X[:, 0]       # Price sensitivity (always negative)

# Price (treatment)
T = np.random.uniform(0.5, 3.0, n)

# Purchase probability and outcome
p_true = expit(alpha_true + beta_true * T)
Y = np.random.binomial(1, p_true).astype(float)
```

### Step 2: Estimate Price Elasticity

```python
from deep_inference import inference

# Elasticity at price point P=2.0
result_elast = inference(
    Y, T, X,
    model='logit',
    target='elasticity',
    t_tilde=2.0,
    epochs=100,
    n_folds=50,
)

print(f"Average elasticity at P=2.0: {result_elast.mu_hat:.4f}")
print(f"SE: {result_elast.se:.4f}")
print(f"95% CI: [{result_elast.ci_lower:.4f}, {result_elast.ci_upper:.4f}]")
```

### Step 3: Compute Consumer Welfare

```python
# Consumer surplus at current price P=2.0
result_welfare = inference(
    Y, T, X,
    model='logit',
    target='welfare',
    t_tilde=2.0,
    epochs=100,
    n_folds=50,
)

print(f"Consumer surplus: {result_welfare.mu_hat:.4f}")
print(f"95% CI: [{result_welfare.ci_lower:.4f}, {result_welfare.ci_upper:.4f}]")
```

### Step 4: Dose-Response Curve

Evaluate elasticity at multiple price points to trace out the demand curve:

```python
prices = np.linspace(0.5, 3.0, 6)
elasticities = []

for p in prices:
    r = inference(Y, T, X, model='logit', target='elasticity', t_tilde=float(p),
                  epochs=100, n_folds=50)
    elasticities.append((p, r.mu_hat, r.se))
    print(f"P={p:.1f}: elasticity={r.mu_hat:.4f} +/- {r.se:.4f}")
```

## Optimal Pricing

With estimated elasticity $\hat{\eta}$, the Lerner markup rule gives:

$$\frac{P - MC}{P} = -\frac{1}{\hat{\eta}}$$

For a firm with marginal cost $MC$:

```python
MC = 1.0  # marginal cost
eta_hat = result_elast.mu_hat  # estimated elasticity

# Optimal markup: P* = MC * eta / (1 + eta)
P_optimal = MC * eta_hat / (1 + eta_hat)
print(f"Optimal price: {P_optimal:.4f}")
```

## Uniform vs. Personalized Pricing

The key insight from Dube & Misra (2022): personalized pricing based on heterogeneous $\beta(X)$ can increase both profits and consumer welfare relative to uniform pricing, when firms can identify consumers with high willingness to pay.

## References

- Dube, J.-P. & Misra, S. (2022). Personalized Pricing and Consumer Welfare. *Journal of Political Economy*, 131(1).
- Farrell, M. H., Liang, T., & Misra, S. (2021). Deep Neural Networks for Estimation and Inference. *Econometrica*, 89(1).
- Small, K. A. & Rosen, H. S. (1981). Applied Welfare Economics with Discrete Choice Models. *Econometrica*, 49(1).
