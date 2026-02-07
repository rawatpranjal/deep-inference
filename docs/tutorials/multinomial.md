# Multinomial Logit (Conditional Logit) Tutorial

The multinomial logit model handles discrete choice among $J \geq 3$ alternatives with heterogeneous preferences.

## Training Configuration

Multinomial logit requires more patient training than binary models due to 3-way cross-fitting splitting:

```python
result = structural_dml(Y, T, X, family='multinomial_logit',
                        n_alternatives=3, n_attributes=2,
                        patience=50, epochs=300)
```

**Key settings:**
- `patience=50` (default 10 is too aggressive for 3-way splitting)
- `epochs=300` (needs more epochs to converge)
- `n >= 8000` recommended for valid coverage

## When to Use

Use the multinomial logit model when:
- Individual chooses one from $J \geq 3$ discrete alternatives
- Alternatives have measurable attributes (price, quality, etc.)
- Parameters vary across individuals (heterogeneous preferences)
- Examples: transportation mode choice, brand selection, product choice

## Mathematical Setup

### Data Generating Process

$$V_{ij} = \alpha_j(W) + X'_{ij} \cdot \beta(W)$$

$$P(Y=j \mid W, X) = \frac{\exp(V_{ij})}{\sum_m \exp(V_{im})}$$

Where:
- $W$ = individual characteristics (covariates, NN input)
- $X_{ij}$ = attributes of alternative $j$ for individual $i$
- $\alpha_j(W)$ = alternative-specific intercepts ($\alpha_0 = 0$, normalized)
- $\beta(W)$ = attribute coefficients (common across alternatives)

### Estimand

$$\mu^* = E[\beta_k(W)]$$

The average attribute coefficient across the population. For example, the average price sensitivity.

### Loss Function

$$L(Y, X, \theta) = -\log P(Y=y \mid W, X) = -\log \text{softmax}(V)[y]$$

Categorical cross-entropy loss.

### Influence Score Components

| Component | Formula |
|-----------|---------|
| Score $\partial\ell/\partial\alpha_j$ | $p_j - \mathbf{1}\{Y=j\}$ (for $j=1,\ldots,J-1$) |
| Score $\partial\ell/\partial\beta_k$ | $\sum_j (p_j - \mathbf{1}\{Y=j\}) x_{jk}$ |
| Hessian (Fisher) | $H_{\alpha\alpha}[j,m] = p_{j+1}(\delta_{jm} - p_{m+1})$ |
| Hessian $H_{\beta\beta}$ | $\sum_j p_j (x_j - \bar{x}_p)(x_j - \bar{x}_p)'$ |

Note: The Hessian is the **Fisher information** (does not depend on $Y$), but depends on $\theta$ through the softmax probabilities.

## Data Encoding

Multinomial logit has a unique data layout compared to other families:

| Variable | Shape | Description |
|----------|-------|-------------|
| `W` (passed as `X`) | `(n, d_w)` | Individual characteristics (NN input) |
| `T` | `(n, J*K)` | Packed alternative attributes |
| `Y` | `(n,)` | Chosen alternative index (float: 0, 1, ..., J-1) |

**Parameter vector:** $\theta = [\alpha_1, \ldots, \alpha_{J-1}, \beta_1, \ldots, \beta_K]$, so `theta_dim = (J-1) + K`.

**Treatment encoding:** Alternative attributes are packed as `(n, J*K)` — each row contains the $K$ attributes for all $J$ alternatives concatenated. Inside the model, this is reshaped to `(n, J, K)`.

## Complete Example

```python
import numpy as np
import torch
from deep_inference import structural_dml

# === DGP Setup ===
J = 3   # alternatives
K = 2   # attributes per alternative
d_w = 3 # individual characteristics
n = 8000

np.random.seed(42)

# Individual characteristics W ~ N(0, I)
W = np.random.normal(0, 1, (n, d_w))

# True parameter functions (heterogeneous in W[0])
# alpha_0 = 0 (reference), alpha_1 = 0.5 + 0.2*W[0], alpha_2 = -0.3 - 0.1*W[0]
# beta_1 = -0.8 - 0.2*W[0], beta_2 = 0.5 + 0.1*W[0]
a0 = [0.0, 0.5, -0.3]
a1 = [0.0, 0.2, -0.1]
b0 = [-0.8, 0.5]
b1 = [-0.2, 0.1]

alphas = np.column_stack([a0[j] + a1[j] * W[:, 0] for j in range(J)])
betas = np.column_stack([b0[k] + b1[k] * W[:, 0] for k in range(K)])

# Alternative attributes X ~ N(0, 1)
X_alt = np.random.normal(0, 1, (n, J, K))

# Utilities V_ij = alpha_j + x'_ij * beta
from scipy.special import softmax
V = alphas.copy()
for j in range(J):
    V[:, j] += np.sum(X_alt[:, j, :] * betas, axis=1)
probs = softmax(V, axis=1)

# Sample choices
Y = np.array([np.random.choice(J, p=probs[i]) for i in range(n)]).astype(float)

# Pack T: (n, J, K) -> (n, J*K)
T = X_alt.reshape(n, -1)

# True target: mu* = E[beta_1(W)] = b0[0] = -0.8
mu_true = b0[0]
print(f"True mu* = {mu_true}")
print(f"Y distribution: {[f'{(Y==j).mean():.1%}' for j in range(J)]}")
print(f"Shapes: Y={Y.shape}, T={T.shape}, W={W.shape}")

# === Run Inference ===
result = structural_dml(
    Y=Y, T=T, X=W,
    family='multinomial_logit',
    n_alternatives=J,
    n_attributes=K,
    hidden_dims=[64, 32],
    epochs=300,
    patience=50,
    n_folds=50,
    lr=0.01
)

print(result.summary())
```

## Expected Results

From [Eval 09: Multinomial Logit](../validation/eval_09.md):

### Parameter Recovery

| Component | RMSE | Correlation | Status |
|-----------|------|-------------|--------|
| $\alpha_1$ | 0.08 | 0.90 | PASS |
| $\alpha_2$ | 0.12 | 0.78 | PASS |
| $\beta_1$ | 0.09 | 0.88 | PASS |
| $\beta_2$ | 0.10 | 0.85 | PASS |

### Coverage (M=50)

| Metric | Value | Status |
|--------|-------|--------|
| Coverage | 98% | PASS |
| SE Ratio | 0.97 | PASS |
| z-mean | 0.14 | PASS |
| z-std | 0.96 | PASS |

## Alternative Targets

### Average Attribute Coefficient (Default)

```python
# Default: E[beta_k(W)] for the first beta
result = structural_dml(Y, T, X, family='multinomial_logit',
                        n_alternatives=3, n_attributes=2, patience=50)
```

### Choice Probability

Using the `inference()` API with `ChoiceProbabilityTarget`:

```python
from deep_inference import inference
from deep_inference.targets.choice_probability import ChoiceProbabilityTarget

# P(Y=j | W, X) for alternative j=1
target = ChoiceProbabilityTarget(alternative=1, n_alternatives=3, n_attributes=2)
result = inference(Y, T, X, model='multinomial_logit', target=target)
```

### Multinomial Average Marginal Effect

```python
from deep_inference.targets.choice_probability import MultinomialAME

# dP(Y=j)/dx_{jk}: marginal effect of attribute k on probability of choosing j
target = MultinomialAME(alternative=1, attribute=0, n_alternatives=3, n_attributes=2)
result = inference(Y, T, X, model='multinomial_logit', target=target)
```

## Real-World Applications

### Transportation Mode Choice

```python
# Y = chosen mode (0=car, 1=bus, 2=train)
# T = packed attributes: [cost_car, time_car, cost_bus, time_bus, cost_train, time_train]
# X = individual characteristics (income, age, ...)
# Target: E[beta_cost(W)] = average price sensitivity

result = structural_dml(Y, T, X, family='multinomial_logit',
                        n_alternatives=3, n_attributes=2, patience=50)
```

### Brand Choice

```python
# Y = chosen brand (0, 1, ..., J-1)
# T = packed attributes: [price, quality] for each brand
# X = consumer demographics
# Target: E[beta_price(W)] = average price elasticity

result = structural_dml(Y, T, X, family='multinomial_logit',
                        n_alternatives=J, n_attributes=2, patience=50)
```

### Market Entry

```python
# Y = chosen market (0, 1, ..., J-1)
# T = packed market characteristics: [size, competition] per market
# X = firm characteristics
# Target: E[beta_size(W)] = average market size sensitivity

result = structural_dml(Y, T, X, family='multinomial_logit',
                        n_alternatives=J, n_attributes=2, patience=50)
```

## Key Takeaways

1. **patience=50 is essential**: The default patience=10 triggers early stopping too aggressively with 3-way splitting (only 60% of data for training)
2. **n >= 8000 for coverage**: 3-way splitting + 4D theta requires more data than binary logit; n=5000 gives only ~88% coverage
3. **3-way splitting (Regime C)**: The Fisher Hessian depends on $\theta$ (through softmax probabilities), requiring 3-way cross-fitting
4. **Fisher information Hessian**: Does not depend on $Y$, only on the softmax probabilities — this is theoretically correct for the expected Hessian
5. **correction_ratio ~70-90 is normal**: Much larger than binary logit (~2) due to higher-dimensional theta and more complex loss surface
6. **$\alpha_2$ is hardest to recover**: The weakest signal (slope -0.1) needs n >= 10000 for reliable correlation > 0.7

## Reference

Hetzenecker, S. & Osterhaus, C. (2024). "Deep Learning for Heterogeneous Parameters in Discrete Choice Models." *arXiv:2408.09560*.
