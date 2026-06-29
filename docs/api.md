# API Reference

Complete API documentation for `deep-inference`. This page consolidates every module into one reference.

The package exposes four top-level entry points.

```python
from deep_inference import structural_dml, inference, did, riesz_inference
```

| Entry point | When to use |
|-------------|-------------|
| `structural_dml()` | Production workhorse, 13 GLM families, fixed target E[beta(X)] |
| `inference()` | Flexible targets, regime detection, RCT support, custom loss |
| `did()` | Difference-in-differences (exact / neural / panel two-way FE) |
| `riesz_inference()` | Automatic debiasing via Riesz representer (binary ATE) |

---

## Entry points

### `structural_dml()` - legacy API

The production-ready API supporting 13 GLM families. Trains a structural neural network with
influence-function-based inference and returns `DMLResult`.

```python
from deep_inference import structural_dml

result = structural_dml(
    Y,                      # (n,) outcomes
    T,                      # (n,) treatments
    X,                      # (n, d) covariates
    family='linear',        # Family name: 'linear', 'logit', 'poisson', etc.
    target=None,            # Target variant (e.g., 'ame' for logit)
    hidden_dims=[64, 32],   # Network architecture
    epochs=100,             # Training epochs
    n_folds=50,             # Cross-fitting folds
    lr=0.01,                # Learning rate
    lambda_method='ridge',  # Lambda estimation: ridge (default), aggregate, lgbm
    verbose=False           # Print progress
)
```

**Supported families:**

| Family | Model | theta_dim | Notes |
|--------|-------|-----------|-------|
| `linear` | Y = alpha + beta*T + eps | 2 | OLS-equivalent |
| `logit` | P(Y=1) = sigmoid(alpha + beta*T) | 2 | Binary outcomes |
| `poisson` | Y ~ Pois(exp(alpha + beta*T)) | 2 | Count data |
| `gamma` | Y ~ Gamma(k, exp(alpha + beta*T)) | 2 | Positive continuous |
| `gumbel` | Y ~ Gumbel(alpha + beta*T, sigma) | 2 | Extreme values |
| `tobit` | Y = max(0, alpha + beta*T + sigma*eps) | 3 | Censored |
| `negbin` | Y ~ NegBin(exp(alpha + beta*T), r) | 2 | Overdispersed counts |
| `weibull` | Y ~ Weibull(k, exp(alpha + beta*T)) | 2 | Survival/duration |
| `gaussian` | Y ~ N(alpha + beta*T, sigma) | 3 | Explicit heteroskedasticity |
| `probit` | P(Y=1) = Phi(alpha + beta*T) | 2 | Binary, normal link |
| `beta` | Y ~ Beta(mu*phi, (1-mu)*phi) | 2 | Fractional outcomes |
| `zip` | Y ~ ZeroInflated-Poisson(pi, lambda) | 4 | Zero-inflated counts |
| `multinomial_logit` | P(Y=j) = softmax(alpha_j + X'_j*beta) | (J-1)+K | Discrete choice |

**Example:**

```python
import numpy as np
from deep_inference import structural_dml

np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)
Y = X[:, 0] + 0.5 * T + np.random.randn(n)

result = structural_dml(Y=Y, T=T, X=X, family='linear', n_folds=50, epochs=100)
print(result.summary())
```

Output:

```
==============================================================================
                            Structural DML Results
==============================================================================
Family:           Linear               Target:           E[beta]
No. Observations: 2000                 No. Folds:        50
==============================================================================
                  coef     std err         z     P>|z|      [0.025    0.975]
------------------------------------------------------------------------------
     E[beta]    0.5012      0.0234    21.410    0.000     0.4553    0.5471
==============================================================================
Diagnostics:
  Min Lambda eigenvalue:    0.998765
  Mean condition number:    1.00
  Correction ratio:         0.0523
------------------------------------------------------------------------------
```

**Lambda method guidance:**

| Method | Coverage | Default | Notes |
|--------|----------|---------|-------|
| `'ridge'` | 96% | Yes | Safe default, validated coverage |
| `'aggregate'` | 95% | No | Stable, ignores X-dependence |
| `'lgbm'` | 96% | No | High accuracy alternative |
| `'mlp'` | 67% | No | AVOID: invalid SEs despite high correlation |

Ridge requires heavy regularization (alpha=1000) to produce stable Lambda estimates.

---

### `inference()` - flexible API

The new API with flexible targets and automatic regime detection. Accepts built-in model strings,
custom loss functions, built-in and custom target functionals, and an explicit regime override.

```python
from deep_inference import inference

result = inference(
    Y,                      # (n,) outcomes
    T,                      # (n,) treatments
    X,                      # (n, d) covariates
    # Model specification (choose one):
    model='logit',          # Built-in: 'linear', 'logit', 'multinomial_logit', 'quantile', 'did', 'did_fe'
    loss=None,              # OR custom loss function
    theta_dim=None,         # Required if custom loss
    # Target specification (choose one):
    target='beta',          # Built-in target string (see list below)
    target_fn=None,         # OR custom target function
    t_tilde=None,           # Evaluation point (default: mean(T))
    # Regime settings:
    is_randomized=False,    # True for RCTs
    treatment_dist=None,    # Known F_T (e.g., Normal(0, 1))
    lambda_method=None,     # Override auto-detection
    # Cross-fitting:
    n_folds=50,
    n_repeats=1,
    # Network:
    hidden_dims=[64, 32],
    epochs=100,
    lr=0.01,
    ridge=1e-4,
    verbose=False
)
```

**Recognized `target=` strings:**

`beta`, `average_slope`, `ame`, `elasticity`, `wtp`, `welfare`, `dose_response`, `profit`,
`tail_probability`, `conditional_variance`, `qte`, `tau`, `att`, `fe_effect`.

**Built-in target quick reference:**

| String | Formula | Use case |
|--------|---------|----------|
| `'beta'` | E[beta(X)] | Average treatment effect (log-odds for logit) |
| `'ame'` | E[p(1-p)*beta] | Average marginal effect (probability scale) |
| `'elasticity'` | E[(1-p)*beta*t] | Price elasticity of demand |
| `'wtp'` | E[-theta_attr / theta_price] | Willingness to pay |
| `'welfare'` | E[log(1+exp(V)) / abs(beta_price)] | Consumer surplus (logsum) |
| `'dose_response'` | E[G(theta, t)] | Average predicted outcome |
| `'profit'` | E[t * G(theta, t)] | Expected revenue at price level |
| `'tail_probability'` | E[P(Y > c | theta, t)] | Exceedance probability |
| `'conditional_variance'` | E[Var(Y | theta, t)] | Model-implied variance |

**Three regimes (auto-detected):**

| Regime | Condition | Lambda method | Cross-fitting |
|--------|-----------|---------------|---------------|
| A | RCT with known F_T | Compute via Monte Carlo | 2-way |
| B | Linear model | Analytic closed-form | 2-way |
| C | Observational and nonlinear | Estimate via neural net | 3-way |

**Regime A example (randomized experiment):**

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

result = inference(
    Y, T, X,
    model='logit',
    target='beta',
    is_randomized=True,
    treatment_dist=Normal(mean=0.0, std=1.0)
)
print(result.diagnostics['regime'])   # 'A'
```

**Custom target function example:**

```python
import torch

def avg_prediction(x, theta, t_tilde):
    """E[P(Y=1 | T=t_tilde)] = E[sigma(alpha + beta*t_tilde)]"""
    return torch.sigmoid(theta[0] + theta[1] * t_tilde)

result = inference(Y, T, X, model='logit', target_fn=avg_prediction, t_tilde=0.0)
```

The Jacobian of any `target_fn` is computed automatically via PyTorch autodiff.

---

### `did()` - difference-in-differences

A single entry point for difference-in-differences with influence-function inference. The method
is auto-selected from the arguments but can be overridden with `method=`.

| `method` | Call signature | What it does |
|----------|----------------|--------------|
| `'exact'` | `did(Y, group, post)` | Closed-form 2x2, beta = mu_11 - mu_10 - mu_01 + mu_00; SE = HC0 |
| `'neural'` | `did(Y, group, post, X=X)` | Heterogeneous 2x2; network learns tau(X); target E[tau(X)] |
| `'panel_fe'` | `did(Y, group, post, X=X, unit=u, time=t)` | Two-way FE panel DiD; target E[tau(X)] |

Auto-selection rule (`method='auto'`, the default): `unit` and `time` given implies `'panel_fe'`;
else `X` given implies `'neural'`; else implies `'exact'`.

**Closed-form 2x2 (`method='exact'`):**

The estimand is the group x post interaction beta. For the saturated cell-mean loss the expected
Hessian is Lambda = diag(p_00, p_01, p_10, p_11), so the influence function is closed-form and
`SE = sqrt(Var(psi)/n)` equals the HC0 robust OLS SE of `Y = alpha + gamma*G + lambda*T + beta*(G*T) + u`
to machine precision. Use `use_bessel=False` (the default); `use_bessel=True` applies an (n-1) correction.

```python
import numpy as np
from deep_inference import did

rng = np.random.default_rng(0)
n = 2000
G = (rng.random(n) < 0.5).astype(float)
P = (rng.random(n) < 0.5).astype(float)
Y = 1.0 + 0.5*G + 0.4*P + 0.6*(G*P) + rng.standard_normal(n)

r = did(Y, G, P)
print(r.mu_hat, r.se)   # beta-hat and HC0-equivalent SE
```

**Heterogeneous 2x2 (`method='neural'`):**

Pass covariates `X` and the network learns covariate-varying coefficients of
`Y = alpha(X) + gamma(X)*G + lambda(X)*P + tau(X)*(G*P) + eps`. The IF machinery returns
E[tau(X)]. Regime B (analytic Lambda = E[WW' | X], two-way cross-fitting).

```python
r = did(Y, G, P, X=X, n_folds=50, epochs=200)
tau_x = r.predict_theta(X_new)[:, 3]   # conditional effect tau(X_new)
```

**Two-way FE panel (`method='panel_fe'`):**

Y and treatment D = group*post (or an explicit `D=`) are residualized by unit and time fixed
effects, then the network learns tau(X) in `Y_tilde_it = D_tilde_it * tau(X_it) + eps_it`.
Target E[tau(X)], Regime B (Lambda = E[D_tilde^2 | X]). Works for continuous and binary outcomes.
For binary Y this is a fixed-effects linear probability model with heteroskedasticity-robust IF SE.

```python
r = did(Y, D=D, X=X, unit=unit, time=time)
tau_x = r.predict_theta(X_new)[:, 0]   # conditional effect tau(X_new)
```

```{note}
For `panel_fe`, the standard error assumes errors are independent across observations.
Serial or within-unit correlation would require cluster-robust SEs, which are not yet implemented.
```

The neural methods are also reachable via the general API: `inference(..., model='did', target='tau')`
and `inference(..., model='did_fe', target='fe_effect')`.

---

### Algorithm overview

**Cross-fitting (K-fold):**

```
For k = 1 to K:
    Train: Fit theta-hat(x) on folds != k
    [If 3-way: Fit Lambda-hat(x) on a separate fold]
    Eval: Compute psi on fold k

Aggregate: mu-hat = mean(psi),  SE = std(psi) / sqrt(n)
```

**Influence function:**

The influence function corrects for regularization bias:

```
psi(z) = H(theta-hat) - H_theta * Lambda(x)^{-1} * ell_theta(z, theta-hat)
```

Where H(theta) is the target functional, H_theta is its Jacobian, Lambda(x) = E[ell_theta_theta | X=x]
is the conditional Hessian, and ell_theta is the score (gradient of loss).

**Expected performance:**

| Method | Coverage | SE ratio |
|--------|----------|----------|
| Naive (no correction) | 10-30% | much less than 1 |
| Influence function | ~95% | ~1.0 |

**Network architecture guidelines:**

| Sample size | Recommended hidden dims |
|-------------|------------------------|
| n < 1,000 | `[32, 16]` |
| 1,000 to 10,000 | `[64, 32]` |
| 10,000 to 100,000 | `[128, 64, 32]` |
| n > 100,000 | `[256, 128, 64]` |

**Cross-fitting folds:**

| Use case | K |
|----------|---|
| Quick exploration | 10-20 |
| Production | 50 |
| Very large data | 20-50 |

---

## Result objects

### `DMLResult`

Returned by `structural_dml()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu_hat` | float | Debiased estimate of E[beta(X)] |
| `mu_naive` | float | Naive (biased) estimate |
| `se` | float | Standard error |
| `ci_lower` | float | Lower 95% CI bound |
| `ci_upper` | float | Upper 95% CI bound |
| `theta_hat` | ndarray | Estimated theta(x) for all observations, shape (n, theta_dim) |
| `psi` | ndarray | Influence function values, shape (n,) |
| `diagnostics` | dict | Training diagnostics (min lambda eigenvalue, condition number, correction ratio) |

Both `DMLResult` and `InferenceResult` provide a `summary()` method that produces a
statsmodels-style table with a header (family, target, sample size, folds, date), a
coefficient table (estimate, SE, z-statistic, p-value, 95% CI), and diagnostics.

```python
print(result.summary())
print(result.mu_hat)    # point estimate
print(result.se)        # standard error
print(result.ci_lower)  # lower 95% CI
print(result.ci_upper)  # upper 95% CI
```

### `InferenceResult`

Returned by `inference()`, `did()`, and `riesz_inference()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu_hat` | float | Point estimate |
| `se` | float | Standard error |
| `ci_lower` | float | Lower 95% CI |
| `ci_upper` | float | Upper 95% CI |
| `psi_values` | Tensor | Influence function values |
| `theta_hat` | Tensor | Estimated theta(x) |
| `diagnostics` | dict | Regime, lambda method, and other diagnostics |

For `riesz_inference`, `theta_hat` holds the learned representer `a(Z)` as an (n, 1) tensor,
and `diagnostics` includes `procedure='riesznet'`, `mu_naive`, `mean_abs_representer`, and
per-repeat estimates.

---

## Families

Statistical families defining the loss function, score (gradient), Hessian, and the per-observation
target and its Jacobian. All families inherit from `BaseFamily`.

### Factory and registry

```python
from deep_inference import get_family, FAMILY_REGISTRY

family = get_family("logit")
print(list(FAMILY_REGISTRY.keys()))
# ['linear', 'logit', 'poisson', 'tobit', 'negbin', 'gamma', 'gumbel', 'weibull',
#  'gaussian', 'probit', 'beta', 'zip', 'multinomial_logit']
```

### `BaseFamily` interface

All families implement:

| Method | Signature | Description |
|--------|-----------|-------------|
| `loss` | `(y, t, theta) -> (n,)` | Per-observation loss (NLL) |
| `gradient` | `(y, t, theta) -> (n, theta_dim)` | Score: gradient w.r.t. theta |
| `hessian` | `(y, t, theta) -> (n, theta_dim, theta_dim)` | Hessian w.r.t. theta |
| `hessian_depends_on_theta()` | `() -> bool` | Whether Hessian depends on theta (controls 2-way vs 3-way split) |
| `per_obs_target` | `(theta, t) -> (n,)` | Per-observation target h(theta) |
| `per_obs_target_gradient` | `(theta, t) -> (n, theta_dim)` | Gradient of target |

```python
import torch
from deep_inference import LinearFamily

family = LinearFamily()
n = 100
y = torch.randn(n)
t = torch.randn(n)
theta = torch.randn(n, 2)   # [alpha, beta]

loss = family.loss(y, t, theta)                       # (n,)
grad = family.gradient(y, t, theta)                   # (n, 2)
hess = family.hessian(y, t, theta)                    # (n, 2, 2)
h    = family.per_obs_target(theta, t)                # (n,)
dh   = family.per_obs_target_gradient(theta, t)       # (n, 2)
```

### GLM family formulas

| Family | Link | Loss (NLL) | Gradient | Hessian weight | theta_dim |
|--------|------|------------|----------|----------------|-----------|
| Linear | Identity | (y-mu)^2 | -2(y-mu)*[1,t] | 2 (constant) | 2 |
| Logit | Logit | log(1+exp(eta)) - y*eta | (p-y)*[1,t] | p(1-p) | 2 |
| Poisson | Log | lambda - y*log(lambda) | (lambda-y)*[1,t] | lambda | 2 |
| NegBin | Log | -lgamma(y+r) + ... | r(mu-y)/(r+mu)*[1,t] | r*mu*(r+y)/(r+mu)^2 | 2 |
| Gamma | Log | y/mu + log(mu) | (1-y/mu)*[1,t] | y/mu | 2 |
| Weibull | Log | -log(k)+k*log(lam)-(k-1)*log(y)+z^k | k(1-z)*[1,t] | k^2*z | 2 |
| Gumbel | Identity | z + exp(-z) | (-1/sigma)*(1-exp(-z))*[1,t] | exp(-z)/sigma^2 | 2 |
| Tobit | Identity | Censored normal NLL | Autodiff (Mills ratio) | Autodiff | 3 |
| Probit | Phi(eta) | -y*log(Phi) - (1-y)*log(1-Phi) | Mills ratio | Autodiff | 2 |
| Beta | Logit | lgamma(mu*phi)+lgamma((1-mu)*phi)-... | Digamma terms | Autodiff | 2 |
| Gaussian | Identity | (y-mu)^2/(2*sigma^2) + log(sigma) | [(mu-y)/sigma^2, t*(mu-y)/sigma^2, 1-(y-mu)^2/sigma^2] | Depends on sigma | 3 |
| ZIP | Mixed | Mixture pi + (1-pi)*Poisson | Autodiff | Autodiff | 4 |
| MultinomialLogit | Softmax | -log softmax(V)[y] | (p_j - 1{Y=j})*[1,x] | p_j*(delta_jm - p_m) blocks | (J-1)+K |

Where: eta = alpha + beta*t, mu = g^{-1}(eta), z = (y/lambda)^k (Weibull) or (y-mu)/sigma (Gumbel),
r = 1/overdispersion, Phi = normal CDF.

### Family classes

**`LinearFamily`**

```python
from deep_inference import LinearFamily

family = LinearFamily()
# Model: Y = alpha(X) + beta(X)*T + epsilon
# Loss: MSE
# Hessian: constant (2-way splitting)
```

**`LogitFamily`**

```python
from deep_inference import LogitFamily

family = LogitFamily(target='beta')   # log-odds target (default)
family = LogitFamily(target='ame')    # average marginal effect
# Model: P(Y=1) = sigmoid(alpha(X) + beta(X)*T)
# Hessian: depends on theta (3-way splitting)
```

**`PoissonFamily`**

```python
from deep_inference import PoissonFamily

family = PoissonFamily()
# Model: Y ~ Poisson(exp(alpha(X) + beta(X)*T))
# Hessian: depends on theta
```

**`TobitFamily`**

```python
from deep_inference import TobitFamily

family = TobitFamily(target='latent')     # effect on latent Y* (default)
family = TobitFamily(target='observed')   # effect on observed E[Y]
# Model: Y = max(0, alpha(X) + beta(X)*T + sigma(X)*eps)
# theta_dim: 3 (alpha, beta, gamma=log(sigma))
```

**`NegBinFamily`**

```python
from deep_inference import NegBinFamily

family = NegBinFamily(overdispersion=0.5)
# Model: Y ~ NegBin(mu, r) where mu = exp(alpha + beta*T)
```

**`GammaFamily`**

```python
from deep_inference import GammaFamily

family = GammaFamily(shape=1.0)
# Model: Y ~ Gamma(shape, exp(alpha + beta*T))
```

**`GumbelFamily`**

```python
from deep_inference import GumbelFamily

family = GumbelFamily(scale=1.0)
# Model: Y ~ Gumbel(alpha + beta*T, scale)
```

**`WeibullFamily`**

```python
from deep_inference import WeibullFamily

family = WeibullFamily(shape=1.0)
# Model: Y ~ Weibull(shape, exp(alpha + beta*T))
```

**`GaussianFamily`** (import from `deep_inference.families`)

```python
from deep_inference.families import GaussianFamily

family = GaussianFamily()
# Model: Y ~ N(alpha + beta*T, sigma)
# theta_dim: 3 (alpha, beta, log_sigma)
```

**`ProbitFamily`** (import from `deep_inference.families`)

```python
from deep_inference.families import ProbitFamily

family = ProbitFamily()
# Model: P(Y=1) = Phi(alpha + beta*T)
# Hessian: autodiff
```

**`BetaFamily`** (import from `deep_inference.families`)

```python
from deep_inference.families import BetaFamily

family = BetaFamily()
# Model: Y ~ Beta(mu*phi, (1-mu)*phi)
# Hessian: autodiff
```

**`ZIPFamily`** (import from `deep_inference.families`)

```python
from deep_inference.families import ZIPFamily

family = ZIPFamily()
# Model: Y ~ ZeroInflated-Poisson(pi, lambda)
# theta_dim: 4
# Hessian: autodiff
```

**`MultinomialLogitFamily`**

```python
from deep_inference import MultinomialLogitFamily

family = MultinomialLogitFamily(
    n_alternatives=3,
    n_attributes=2,
    target='beta',
    target_idx=0
)
# Model: P(Y=j) = softmax(alpha_j(W) + X'_ij * beta(W))
# theta_dim = (J-1) + K = 4
# Hessian: Fisher information (depends on theta, 3-way splitting)
```

**Using a family instance with `structural_dml`:**

```python
from deep_inference import structural_dml, LogitFamily

family = LogitFamily(target='ame')
result = structural_dml(Y, T, X, family=family)
```

---

## Targets

The `targets` module defines what quantity to estimate. A target is a functional H(theta) that
maps structural parameters to a scalar. Targets implement the `Target` protocol.

### `Target` protocol

```python
class Target(Protocol):
    def h(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """Compute target value for a single observation."""
        ...

    def jacobian(self, x: Tensor, theta: Tensor, t_tilde: Tensor) -> Tensor:
        """Compute dh/dtheta. Falls back to autodiff if not implemented (returns None)."""
        ...
```

### Target overview

| Target | Formula | Use case |
|--------|---------|----------|
| `AverageParameter` | E[theta_j] | Average treatment effect |
| `AME` | E[p(1-p)*beta] | Marginal effect on probability |
| `Elasticity` | E[(1-p)*beta*t] (logit) | Price elasticity of demand |
| `WTP` | E[-theta_attr / theta_price] | Willingness to pay |
| `ConsumerWelfare` | E[log(1+exp(V)) / abs(beta_price)] | Expected consumer surplus |
| `DoseResponse` | E[G(theta, t)] | Average predicted outcome at treatment level |
| `Profit` | E[t * G(theta, t)] | Expected revenue at price level |
| `TailProbability` | E[P(Y > c | theta, t)] | Exceedance/risk probability |
| `ConditionalVariance` | E[Var(Y | theta, t)] | Model-implied outcome variance |
| `MultiTreatmentATE` | E[G(theta, t) - G(theta, t_0)] | Combinatorial treatment effect |
| `ChoiceProbabilityTarget` | P(Y=j | W, X) | Multinomial choice probability |
| `MultinomialAME` | dP(Y=j)/dx_{jk} | Multinomial marginal effect |
| `CustomTarget` | User-defined h(x, theta, t) | Any custom functional |

### `AverageParameter`

Average of a specific parameter across the population. Default target for most families.

```python
from deep_inference.targets import AverageParameter

target = AverageParameter(param_index=1, theta_dim=2)
```

Formula: `H(theta) = (1/n) sum theta_j(x_i)`

Jacobian (closed-form): `dH/dtheta = [0, ..., 1/n, ..., 0]` with 1 at position j.

### `AME` (Average Marginal Effect)

For logit models, the marginal effect on probability (not log-odds).

```python
from deep_inference.targets import AME

target = AME(param_index=1, model_type='logit')
```

Formula (logit): `H(theta, t) = (1/n) sum sigma'(alpha_i + beta_i*t) * beta_i = (1/n) sum p_i*(1-p_i)*beta_i`

Jacobian (closed-form for logit):

```
dH/dalpha = p*(1-p)*(1-2p)*beta / n
dH/dbeta  = p*(1-p)*[1 + (1-2p)*beta*t] / n
```

### `Elasticity`

Price elasticity of demand. Key object for personalized pricing.

```python
from deep_inference.targets import Elasticity

target = Elasticity(model_type='logit')   # also 'poisson', 'gamma', 'negbin'
```

Formula (logit): `H(theta, t) = (1 - p)*beta*t` where `p = sigma(alpha + beta*t)`

Formula (log-link: poisson/gamma/negbin): `H(theta, t) = beta*t`

Jacobian (logit):

```
dH/dalpha = -p*(1-p)*beta*t
dH/dbeta  = (1-p)*t - p*(1-p)*beta*t^2
```

Usage: `result = inference(Y, T, X, model='logit', target='elasticity', t_tilde=1.0)`

Reference: Dube and Misra (2022, JPE).

### `WTP` (Willingness To Pay)

Marginal rate of substitution between an attribute and price.

```python
from deep_inference.targets import WTP

target = WTP(attribute_index=1, price_index=2)
```

Formula: `H(theta) = -theta_attribute / theta_price`

Jacobian (closed-form delta method):

```
dH/dtheta_attr  = -1 / theta_price
dH/dtheta_price =  theta_attr / theta_price^2
```

Returns NaN and falls back to `None` Jacobian when the price coefficient is within 1e-8 of zero.
`t_tilde` is not used.

Usage: `result = inference(Y, T, X, model='logit', target='wtp')`

Reference: Dube and Misra (2022, JPE).

### `ConsumerWelfare`

Expected consumer surplus from logit demand (Small and Rosen 1981 logsum), scaled by the marginal
utility of income.

```python
from deep_inference.targets import ConsumerWelfare

target = ConsumerWelfare(price_coef_index=1)
```

Formula (binary logit): `H(theta, t) = log(1 + exp(V)) / abs(beta_price)` where `V = alpha + beta_price*t`

Implemented with `softplus(V)` for numerical stability. Jacobian is computed via autodiff
(`jacobian()` returns `None`) because it involves derivatives of `softplus` and `|beta_price|`.

Usage: `result = inference(Y, T, X, model='logit', target='welfare', t_tilde=0.0)`

Reference: Small and Rosen (1981); Dube and Misra (2022, JPE).

### `DoseResponse`

Average predicted outcome at a given treatment level.

```python
from deep_inference.targets import DoseResponse

target = DoseResponse(model_type='logit')
```

Formula (logit): `H(theta, t) = sigma(alpha + beta*t)`

Jacobian (closed-form, where p = sigma(alpha + beta*t)):

```
dH/dalpha = p*(1-p)
dH/dbeta  = p*(1-p)*t
```

Supports `logit`, `linear`, and `poisson` model types.

Usage: `result = inference(Y, T, X, model='logit', target='dose_response', t_tilde=1.0)`

Reference: Colangelo and Lee (2026, JBES).

### `Profit`

Expected revenue per consumer at a given price level.

```python
from deep_inference.targets import Profit

target = Profit(model_type='logit')
```

Formula (logit): `H(theta, t) = t * sigma(alpha + beta*t)`

Jacobian (closed-form):

```
dH/dalpha = t*p*(1-p)
dH/dbeta  = t*p*(1-p)*t
```

Usage: `result = inference(Y, T, X, model='logit', target='profit', t_tilde=2.0)`

Reference: Dube and Misra (2023, JPE).

### `TailProbability`

Probability that the outcome exceeds a threshold.

```python
from deep_inference.targets import TailProbability

target = TailProbability(threshold=5, model_type='poisson')
```

Formula (Poisson): `H(theta, t) = 1 - sum_{k=0}^{c} exp(-lambda)*lambda^k/k!` where `lambda = exp(alpha + beta*t)`

Formula (logit, c=0): `H(theta, t) = sigma(alpha + beta*t)` (identical to DoseResponse)

Formula (linear/Gaussian): `H(theta, t) = 1 - Phi((c - mu)/sigma)` where `mu = alpha + beta*t`

Closed-form Jacobian for logit and linear; autodiff for Poisson.

Reference: Melnychuk and Feuerriegel (2026, ICLR).

### `ConditionalVariance`

Model-implied variance of outcomes, capturing heterogeneity in risk across covariates.

```python
from deep_inference.targets import ConditionalVariance

target = ConditionalVariance(model_type='logit')
```

Formula (logit): `H(theta, t) = p*(1-p)` where `p = sigma(alpha + beta*t)`

Jacobian (closed-form):

```
dH/dalpha = p*(1-p)*(1-2p)
dH/dbeta  = p*(1-p)*(1-2p)*t
```

Usage: `result = inference(Y, T, X, model='logit', target='conditional_variance', t_tilde=0.0)`

Reference: Melnychuk and Feuerriegel (2026, ICLR).

### `MultiTreatmentATE`

Average treatment effect for combinatorial experiments with multiple binary treatments.

```python
from deep_inference.targets import MultiTreatmentATE
from deep_inference.models.combinatorial import CombinatorialModel

model = CombinatorialModel(n_treatments=3, link='gen_sigmoid_ii')
target = MultiTreatmentATE(
    model=model,
    treatment=[1, 0, 1],   # apply treatments 1 and 3
    control=[0, 0, 0],     # vs no treatment
)
```

Formula: `H(theta) = G(theta, t) - G(theta, t_0)` where G is the structural link from `CombinatorialModel`.

Jacobian computed via autodiff.

Reference: Ye et al. (2025, Management Science).

### `ChoiceProbabilityTarget`

For multinomial logit models: probability of choosing a specific alternative.

```python
from deep_inference.targets.choice_probability import ChoiceProbabilityTarget

target = ChoiceProbabilityTarget(
    alternative=1,         # which alternative (0-indexed)
    n_alternatives=3,      # total alternatives J
    n_attributes=2         # attributes per alternative K
)
```

Formula: `H = P(Y=j | W, X) = softmax(V)[j]` where `V_j = alpha_j + x'_j * beta`

Jacobian (closed-form):

```
dP_j/dalpha_m = P_{m+1}*(delta_{j,m+1} - P_j)
dP_j/dbeta_k  = P_j*(x_tilde_{jk} - x_bar_{pk})
```

Where `x_bar_{pk} = sum_j P_j * x_{jk}` is the probability-weighted mean attribute.

### `MultinomialAME`

Average Marginal Effect for multinomial logit: how a unit change in attribute k affects the
probability of choosing alternative j.

```python
from deep_inference.targets.choice_probability import MultinomialAME

target = MultinomialAME(
    alternative=1,         # which alternative
    attribute=0,           # which attribute
    n_alternatives=3,      # total alternatives J
    n_attributes=2         # attributes per alternative K
)
```

Formula: `dP_j/dx_{jk} = beta_k * P_j * (1 - P_j)`

This generalizes the familiar binary logit AME to the multinomial context.

### `CustomTarget` and `target_from_fn`

Define any target function; the Jacobian is computed via autodiff.

```python
from deep_inference.targets import CustomTarget
import torch

def my_target(x, theta, t_tilde):
    """
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

**Subclassing `BaseTarget` for a closed-form Jacobian (faster than autodiff):**

```python
from deep_inference.targets import BaseTarget
import torch

class MyTarget(BaseTarget):
    def h(self, x, theta, t_tilde):
        return theta[0] ** 2 + theta[1] * t_tilde

    def jacobian(self, x, theta, t_tilde):
        return torch.tensor([2 * theta[0], t_tilde])
```

**Counterfactual comparison example:**

```python
def treatment_effect(x, theta, t_tilde):
    """E[P(Y=1|T=1) - P(Y=1|T=0)]"""
    alpha, beta = theta[0], theta[1]
    p1 = torch.sigmoid(alpha + beta * 1.0)
    p0 = torch.sigmoid(alpha + beta * 0.0)
    return p1 - p0

result = inference(Y, T, X, model='logit', target_fn=treatment_effect)
```

**Autodiff Jacobian (how it works internally):**

```python
theta.requires_grad_(True)
h_value = target_fn(x, theta, t_tilde)
jacobian = torch.autograd.grad(h_value, theta)[0]
```

---

## Lambda strategies

The `lambda_` module handles estimation of `Lambda(x) = E[d^2 ell / dtheta^2 | X=x]`, the
conditional expected Hessian of the loss. This is the load-bearing object for valid influence
function inference.

### Regime overview

| Regime | Condition | Lambda method | Cross-fitting |
|--------|-----------|---------------|---------------|
| A | RCT with known F_T | Compute via Monte Carlo | 2-way |
| B | Linear model | Analytic closed-form | 2-way |
| C | Observational and nonlinear | Estimate via neural net | 3-way |

### Regime detection

```python
from deep_inference.lambda_ import detect_regime, Regime

regime = detect_regime(
    model=my_model,
    is_randomized=True,
    has_treatment_dist=True
)
# Returns Regime.A, Regime.B, or Regime.C
```

Decision logic:

```
if is_randomized AND treatment_dist provided:
    -> Regime A (ComputeLambda)
elif model is linear:
    -> Regime B (AnalyticLambda)
else:
    -> Regime C (EstimateLambda)
```

### `ComputeLambda` (Regime A)

For randomized experiments where T is independent of X and has known distribution F_T.

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal, Bernoulli, Uniform

# Normal treatment: T ~ N(mu, sigma^2)
result = inference(Y, T, X, model='logit', is_randomized=True, treatment_dist=Normal(mean=0.0, std=1.0))

# Binary treatment: T ~ Bernoulli(p)
result = inference(Y, T, X, model='logit', is_randomized=True, treatment_dist=Bernoulli(p=0.5))

# Uniform treatment: T ~ Uniform(a, b)
result = inference(Y, T, X, model='logit', is_randomized=True, treatment_dist=Uniform(low=-1.0, high=1.0))
```

How it works:

```
Lambda(x, theta) = E_T[d^2 ell / dtheta^2 | theta]
                ~= (1/M) sum d^2 ell(y, t_m, theta) / dtheta^2
```

where `t_m ~ F_T` are Monte Carlo samples. No neural network for Lambda needed. 2-way
cross-fitting (faster). More stable than estimation.

**Custom treatment distribution:**

```python
from deep_inference.lambda_.compute import TreatmentDistribution
import torch

class MyDistribution(TreatmentDistribution):
    def sample(self, n: int) -> torch.Tensor:
        return torch.randn(n) * 2 + 1   # N(1, 4)
```

### `AnalyticLambda` (Regime B)

For linear models, Lambda has a closed-form solution. Automatically selected for the linear model.

Formula (linear): `Lambda = E[[1, T]' [1, T]] = [[1, E[T]], [E[T], E[T^2]]]`

```python
result = inference(Y, T, X, model='linear')   # AnalyticLambda selected automatically
```

### `EstimateLambda` (Regime C)

For observational data with nonlinear models. Lambda(x) is regressed from sample Hessians.

```python
result = inference(Y, T, X, model='logit')   # EstimateLambda with method='ridge' (default)
```

How it works:
1. Compute sample Hessians `d^2 ell / dtheta^2` for training data.
2. Fit a regression model to predict Lambda-hat(x) from covariates X.
3. Predict Lambda-hat for the evaluation fold.

This requires 3-way splitting to avoid bias:
- Fold A: train theta-hat(x).
- Fold B: train Lambda-hat(x) using theta-hat from Fold A.
- Fold C: evaluate psi using both.

**Available estimation methods:**

| Method | Correlation with oracle | Coverage | Default | Notes |
|--------|------------------------|----------|---------|-------|
| `ridge` | 0.508 | 96% | Yes | Recommended, validated coverage |
| `aggregate` | 0.000 | 95% | No | Stable when Hessian does not depend on X |
| `lgbm` | 0.978 | 96% | No | High accuracy alternative |
| `mlp` | 0.997 | 67% | No | AVOID: invalid SEs |

Note: MLP achieves the highest correlation with oracle Lambda but produces 67% coverage instead
of 95%. High correlation does not guarantee valid inference. The variance of Lambda estimates matters.

### `LambdaStrategy` protocol

All strategies implement:

```python
class LambdaStrategy(Protocol):
    requires_theta: bool               # Does fit() need theta-hat?
    requires_separate_fold: bool       # 3-way cross-fitting?

    def fit(self, X, T, Y, theta_hat, model) -> None:
        """Fit the strategy (if needed)."""
        ...

    def predict(self, X, theta_hat) -> Tensor:
        """Return Lambda-hat(x) matrices of shape (n, d_theta, d_theta)."""
        ...
```

### Strategy selection API

```python
from deep_inference.lambda_ import select_lambda_strategy

strategy = select_lambda_strategy(
    model=my_model,
    is_randomized=True,
    treatment_dist=Normal(0, 1),
    lambda_method=None   # auto-detect
)
```

To manually override:

```python
result = inference(Y, T, X, model='logit', lambda_method='estimate')
```

### Diagnostics

```python
result = inference(Y, T, X, model='logit', verbose=True)
print(result.diagnostics['regime'])         # 'A', 'B', or 'C'
print(result.diagnostics['lambda_method'])  # 'ComputeLambda', 'AnalyticLambda', or 'EstimateLambda'
```

### When to use each regime

| Scenario | Regime |
|----------|--------|
| A/B test with known assignment probability | A |
| Survey experiment with normal dosage T ~ N(0, 1) | A |
| Linear regression with heterogeneous coefficients | B |
| Observational logit (health outcomes, etc.) | C |
| Custom nonlinear loss with unknown treatment selection | C |

---

## Models

Neural network architectures for structural estimation. Each model maps covariates X to
structural parameters theta(X) and defines the loss, score, and Hessian.

### `StructuralNet`

The backbone network. Input -> linear -> ReLU -> ... -> theta_dim output.

```
Input (d features)
    |
Linear(d, hidden_dims[0])
    |
ReLU + Dropout
    |
Linear(hidden_dims[0], hidden_dims[1])
    |
ReLU + Dropout
    |
...
    |
Linear(hidden_dims[-1], theta_dim)
    |
Output (theta_dim parameters per observation)
```

```python
from deep_inference.models import StructuralNet
import torch

net = StructuralNet(
    input_dim=10,
    hidden_dims=[64, 32],
    theta_dim=2,
    dropout=0.1
)
X = torch.randn(100, 10)
theta = net(X)   # (100, 2)
```

**Training configuration:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dims` | `[64, 32]` | Hidden layer sizes |
| `epochs` | `100` | Training epochs |
| `lr` | `0.01` | Learning rate |
| `batch_size` | `64` | Mini-batch size |
| `weight_decay` | `1e-4` | L2 regularization |
| `dropout` | `0.1` | Dropout rate |

**Custom training loop:**

```python
import torch.nn as nn
from deep_inference.models import StructuralNet
from deep_inference import LinearFamily

net = StructuralNet(input_dim=10, hidden_dims=[64, 32], theta_dim=2)
family = LinearFamily()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(100):
    theta = net(X_tensor)
    loss = family.loss(Y_tensor, T_tensor, theta).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Built-in structural models

The `inference()` API accepts these model strings:

| String | Class | theta_dim | Notes |
|--------|-------|-----------|-------|
| `'linear'` | `Linear` | 2 | Constant expected Hessian (Regime B) |
| `'logit'` | `Logit` | 2 | Softmax sigmoid output |
| `'multinomial_logit'` | `MultinomialLogit` | (J-1)+K | McFadden conditional logit |
| `'quantile'` | `Quantile` | 2 | Smoothed check function |
| `'did'` | `DiDModel` | 4 | Saturated 2x2 DiD |
| `'did_fe'` | `FEPanelDiDModel` | 1 | Two-way FE panel DiD |

All built-in models are exported from `deep_inference`:

```python
from deep_inference import Linear, Logit, MultinomialLogit, Quantile, CombinatorialModel, DiDModel, FEPanelDiDModel
```

### `MultinomialLogitModel`

McFadden conditional logit for discrete choice with J alternatives and K attributes.

```python
from deep_inference.models.multinomial import MultinomialLogitModel

model = MultinomialLogitModel(n_alternatives=3, n_attributes=2)
# theta_dim = (J-1) + K = 4
# theta = [alpha_1, ..., alpha_{J-1}, beta_1, ..., beta_K]
# Hessian: Fisher information (no Y dependence, depends on theta)
# Requires 3-way cross-fitting (Regime C, or Regime A if randomized)
```

Data encoding:
- `W` = individual characteristics (NN input to get theta(W))
- `T` = packed alternative attributes `(n, J*K)`, unpacked to `(J, K)` inside model
- `Y` = chosen alternative index `(n,)` as float (0, 1, ..., J-1)

Example DGP (J=3, K=2, d_w=3): target mu* = E[beta_1(W)] = -0.8.

### `CombinatorialModel`

For multi-treatment combinatorial experiments with binary treatment vectors `T in {0,1}^m`.

```python
from deep_inference.models.combinatorial import CombinatorialModel

model = CombinatorialModel(n_treatments=3, link='gen_sigmoid_ii')
# theta_dim = m + 2 = 5 for gen_sigmoid_ii
```

**Four link functions:**

| Link | Formula | theta_dim | Description |
|------|---------|-----------|-------------|
| `multiplicative` | theta_0 * prod(1 + theta_k * t_k) | m+1 | Product interaction |
| `sigmoid` | a/(1+exp(-(theta_0 + sum theta_k*t_k))) + b | m+1 | Bounded response with fixed scale |
| `gen_sigmoid_i` | theta_{m+1} * sigma(sum theta_k*t_k) | m+1 | Flexible scale, no intercept |
| `gen_sigmoid_ii` | theta_{m+1} * sigma(theta_0 + sum theta_k*t_k) | m+2 | Most flexible (recommended) |

Hessian uses Fisher information: `2*G_theta*G_theta'`. `hessian_depends_on_theta = True`,
`hessian_depends_on_y = False`. Requires 3-way cross-fitting (Regime C). For Regime A, compute
Lambda via Monte Carlo:

```python
import torch
t_samples = torch.randint(0, 2, (1000, 3)).float()
Lambda = model.compute_lambda_integral(theta, t_samples)
```

**Usage with `MultiTreatmentATE`:**

```python
from deep_inference.targets import MultiTreatmentATE

model = CombinatorialModel(n_treatments=3, link='gen_sigmoid_ii')
target = MultiTreatmentATE(model=model, treatment=[1, 0, 1])
```

Reference: Ye et al. (2025, Management Science), DeDL: Debiased Deep Learning for Combinatorial
Experiments.

### `DiDModel`

Saturated 2x2 DiD model. Network learns `[alpha(X), gamma(X), lambda_coef(X), tau(X)]`.

```python
from deep_inference import DiDModel
```

Accessible via `inference(..., model='did', target='tau')`.

### `FEPanelDiDModel`

Two-way fixed-effects panel DiD. Outcomes and treatment are residualized by unit and time FE
before the network learns `tau(X)` in the within-transformed regression.

```python
from deep_inference import FEPanelDiDModel
```

Accessible via `inference(..., model='did_fe', target='fe_effect')`.

### `CustomModel` and `model_from_loss`

Build a structural model from any differentiable loss function.

```python
from deep_inference import CustomModel
```

---

## RieszNet module

`deep_inference.riesz` implements automatic debiasing via the Riesz representer
(Chernozhukov et al. 2022, ICML). Estimates the binary-treatment ATE contrast
`E[g(1, X) - g(0, X)]` using the doubly-robust moment, cross-fit over K folds and combined
across R repeats with the median-DML rule.

### `riesz_inference`

```python
from deep_inference import riesz_inference

result = riesz_inference(
    Y, T, X,
    outcome='linear',     # 'linear', 'logit', 'poisson', 'gamma', 'negbin', 'probit'
    n_folds=5,
    n_repeats=3,
    max_epochs=400,
    patience=30,
    seed=0,
    nb_dispersion=3.0,    # only used when outcome='negbin'
    alpha=0.05,           # 0.05 gives a 95% CI
    store_data=True,
    verbose=False,
) -> InferenceResult
```

**Arguments:**

```{list-table}
:header-rows: 1

* - Argument
  - Meaning
* - `Y`, `T`, `X`
  - Outcomes `(n,)`, treatment `(n,)`, covariates `(n, d)`. For a binary `T` the estimand is the ATE.
* - `outcome`
  - Outcome model for the regression head. One of `linear`, `logit`, `poisson`, `gamma`, `negbin`, `probit`. An unsupported value raises `ValueError`.
* - `n_folds`
  - Cross-fitting folds K.
* - `n_repeats`
  - Independent split repeats, median-DML combined.
* - `max_epochs`, `patience`
  - Per-fold training schedule.
* - `nb_dispersion`
  - The negative-binomial size r, used only when `outcome='negbin'`.
* - `alpha`
  - CI level. `0.05` gives a 95% interval.
```

Returns `InferenceResult` with `mu_hat`, `se`, `ci_lower`, `ci_upper`, `psi_values`
(doubly-robust moment from the first repeat), `theta_hat` (the learned representer `a(Z)` as
an `(n, 1)` tensor), and a `diagnostics` dict that includes `procedure='riesznet'`, `mu_naive`,
`mean_abs_representer`, and per-repeat estimates.

**Example:**

```python
import numpy as np
from deep_inference import riesz_inference

rng = np.random.default_rng(0)
n = 1500
X = rng.standard_normal((n, 3))
e = 1 / (1 + np.exp(-(0.8 * X[:, 0])))
T = (rng.random(n) < e).astype(float)   # confounded binary treatment
Y = 1.0 + 0.5 * X[:, 0] + (0.5 + 0.3 * X[:, 1]) * T + rng.standard_normal(n)

result = riesz_inference(Y, T, X, outcome='linear', n_folds=5, n_repeats=3)
print(result.summary())
```

### `RieszNet`

```python
from deep_inference import RieszNet
```

The underlying `torch.nn.Module`. Architecture: shared trunk (3 ELU layers, width 200) with a
linear Riesz head and a regression head (2 ELU layers, width 100), plus an unpenalized
targeted-regularization parameter `eps`. Most users do not construct this directly.
`riesz_inference` builds, trains, and cross-fits it. It is exposed for inspection and for
custom training loops.

---

## Metrics

Helper functions for computing inference quality metrics.

### `compute_coverage`

```python
from deep_inference import compute_coverage

covered = compute_coverage(mu_true, ci_lower, ci_upper)
# Returns bool: True if mu_true falls within [ci_lower, ci_upper]
```

### `compute_se_ratio`

```python
from deep_inference import compute_se_ratio

se_ratio = compute_se_ratio(estimated_se, empirical_se)
# Returns float: estimated_se / empirical_se (target: 1.0)
```

### Key metrics

| Metric | Formula | Target |
|--------|---------|--------|
| `bias` | E[mu-hat] - mu* | 0 |
| `variance` | Var(mu-hat) | - |
| `rmse` | sqrt(Bias^2 + Var) | Small |
| `empirical_se` | sqrt(Var) | - |
| `se_ratio` | SE-hat / SE_emp | 1.0 |
| `coverage` | P(mu* in CI) | 95% |

### Validation targets

| Metric | Valid range | Interpretation |
|--------|-------------|----------------|
| Coverage | 93-97% | CI contains true value |
| SE ratio | 0.9-1.2 | SE properly calibrated |
| min(lambda eigenvalue) | > 1e-4 | Hessian well-conditioned |

### Monte Carlo validation example

```python
import numpy as np
from deep_inference import structural_dml

M = 100       # number of simulations
N = 2000      # sample size
MU_TRUE = 0.5

results = []
for m in range(M):
    np.random.seed(m)
    X = np.random.randn(N, 10)
    T = np.random.randn(N)
    Y = X[:, 0] + MU_TRUE * T + np.random.randn(N)

    result = structural_dml(Y, T, X, family='linear', verbose=False)
    covered = result.ci_lower <= MU_TRUE <= result.ci_upper
    results.append({'mu_hat': result.mu_hat, 'se': result.se, 'covered': covered})

coverage = np.mean([r['covered'] for r in results])
se_ratio = np.mean([r['se'] for r in results]) / np.std([r['mu_hat'] for r in results])

print(f"Coverage: {coverage:.1%}")   # Target: 95%
print(f"SE Ratio: {se_ratio:.2f}")   # Target: 1.0
```

### Interpreting results

Good results:

```
Coverage: 95.0%   [PASS - in 93-97% range]
SE Ratio: 1.02    [PASS - close to 1.0]
RMSE:     0.032   [Low bias and variance]
```

Warning signs:

```
Coverage: 30%     [FAIL - severe undercoverage]
SE Ratio: 0.27    [FAIL - SE underestimated 4x]
```

Common causes of poor coverage: naive method (no IF correction), too few folds (K < 20),
insufficient training epochs, or model misspecification.
