# Guide

The other sections explain what each piece is and how the math works. This section is about
the decisions you actually face when you sit down with your own data: which model, which
method for the nuisance object $\Lambda(x)$, how to read what the package tells you
afterward, and what the common applications look like end to end.

Read the sections in order the first time. If you already know which application fits your
problem, jump to the corresponding worked example at the bottom.

---

## Choosing a model

### The decision flowchart

Work through the diagram from top to bottom. Each diamond is a yes/no question about your
data. The shaded boxes show the `family` string, target string, and API call you end up
with.

```{mermaid}
flowchart TD
    subgraph START [" "]
        A([I have data Y, T, X])
    end

    A --> Q1{What type of<br/>outcome Y?}

    Q1 -->|Continuous<br/>unbounded| LINEAR[family='linear']
    Q1 -->|Binary 0/1| LOGIT[family='logit']
    Q1 -->|Count 0,1,2...| COUNT{Overdispersed?}
    Q1 -->|Positive<br/>continuous| POS{Distribution shape?}
    Q1 -->|Censored| TOBIT[family='tobit']

    COUNT -->|No| POISSON[family='poisson']
    COUNT -->|Yes| NEGBIN[family='negbin']

    POS -->|Multiplicative errors| GAMMA[family='gamma']
    POS -->|Extreme values| GUMBEL[family='gumbel']
    POS -->|Time-to-event| WEIBULL[family='weibull']

    LINEAR --> T1{Target functional?}
    LOGIT --> T2{Target functional?}
    POISSON --> T3[target='beta']
    NEGBIN --> T3
    GAMMA --> T3
    GUMBEL --> T3
    WEIBULL --> T3
    TOBIT --> T4{Target functional?}

    T1 -->|Average parameter| BETA_LIN[target='beta']
    T1 -->|Custom function| CUSTOM[target_fn=...]

    T2 -->|Average parameter| BETA_LOG[target='beta']
    T2 -->|Average marginal effect| AME[target='ame']
    T2 -->|Custom function| CUSTOM

    T4 -->|Latent Y*| LATENT[target='latent']
    T4 -->|Observed Y| OBS[target='observed']

    BETA_LIN --> R_B[Regime B<br/>Analytic Lambda]
    BETA_LOG --> R1{Treatment<br/>randomized?}
    AME --> R1
    T3 --> R1
    LATENT --> R_C
    OBS --> R_C
    CUSTOM --> R_C[Regime C<br/>Estimate Lambda]

    R1 -->|Yes + known distribution| R_A[Regime A<br/>Compute Lambda]
    R1 -->|No / observational| R_C

    R_A --> API_A["inference(...,<br/>is_randomized=True,<br/>treatment_dist=Normal())"]
    R_B --> API_B["structural_dml(...,<br/>family='linear')"]
    R_C --> API_C["structural_dml(...,<br/>family='logit')"]

    classDef question fill:#fff3cd,stroke:#856404
    classDef model fill:#d4edda,stroke:#155724
    classDef target fill:#cce5ff,stroke:#004085
    classDef regime fill:#f8d7da,stroke:#721c24
    classDef api fill:#e2e3e5,stroke:#383d41

    class Q1,COUNT,POS,T1,T2,T4,R1 question
    class LINEAR,LOGIT,POISSON,NEGBIN,GAMMA,GUMBEL,WEIBULL,TOBIT model
    class BETA_LIN,BETA_LOG,AME,LATENT,OBS,T3,CUSTOM target
    class R_A,R_B,R_C regime
    class API_A,API_B,API_C api
```

### Decision 1: which model family

Choose based on what your outcome variable looks like.

| Outcome type | Family | Link function | Example use cases |
|---|---|---|---|
| Continuous, unbounded | `linear` | Identity | Wages, prices, test scores |
| Binary (0/1) | `logit` | Logit | Purchase decisions, clicks, conversions |
| Count (0, 1, 2, ...) | `poisson` | Log | Website visits, order counts |
| Overdispersed count | `negbin` | Log | Insurance claims, rare events |
| Positive, continuous | `gamma` | Log | Expenditures, durations |
| Extreme values | `gumbel` | Identity | Maximum temperatures, flood levels |
| Time-to-event | `weibull` | Log | Survival times, equipment failure |
| Censored | `tobit` | Identity | Demand with stockouts |
| Binary, normal latent error | `probit` | Probit | Structural binary choice |
| Bounded in (0,1) | `beta` | Logit | Proportions, shares |
| Discrete choice (3+ options) | `multinomial_logit` | Softmax | Product choice, mode choice |
| Quantile | `quantile` | Identity | Distributional treatment effects |

Every family has its own detailed page under [Models](models/index.md). If you are unsure,
start with the model that matches the marginal distribution of your outcome.

### How to check for overdispersion

For count data, compare the variance to the mean. A ratio much above 1.5 suggests negative
binomial is the safer choice.

```python
overdispersion_ratio = np.var(Y) / np.mean(Y)

if overdispersion_ratio > 1.5:
    print("Use family='negbin'")
else:
    print("Use family='poisson'")
```

### Decision 2: which target functional

The target is the economic quantity you want a confidence interval for. Choosing the wrong
one is the most common source of misinterpretation.

**`target='beta'` (the default).** Estimates the average structural parameter
$\mu^* = \mathbb{E}[\beta(X)]$. For logit, this is on the log-odds scale. For Poisson, it
is on the log-counts scale. Use this when you want the average slope in the space where the
model is linear.

```python
result = structural_dml(Y, T, X, family='logit')
# result.mu_hat estimates E[beta(X)], in log-odds units
```

**`target='ame'` (average marginal effect).** For binary outcomes, this converts to the
probability scale.

$$\text{AME} = \mathbb{E}\!\left[\frac{\partial P(Y=1 \mid X, T)}{\partial T}\bigg|_{T=\tilde{t}}\right]$$

You need to supply the evaluation point `t_tilde`. Use this when you want results in
probability-point units rather than log-odds units.

```python
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)
# result.mu_hat estimates the AME at T=0
```

**`target_fn=...` (custom target).** Define any differentiable function of $\theta(x)$ and
$\tilde{t}$. The package computes the Jacobian via autodiff. Use this for economic
functionals that the built-in targets do not cover.

```python
import torch

def my_target(x, theta, t_tilde):
    """Predicted probability at treatment level t_tilde."""
    alpha, beta = theta[..., 0], theta[..., 1]
    eta = alpha + beta * t_tilde
    return torch.sigmoid(eta)

result = inference(Y, T, X, model='logit', target_fn=my_target, t_tilde=0.5)
```

The built-in targets include elasticity, willingness to pay, consumer welfare, dose
response, expected profit, tail probability, conditional variance, multinomial choice
probabilities, and combinatorial treatment effects. They are listed in
[API: Targets](api.md).

### Decision 3: regime and Lambda strategy

The three regimes determine how the expected Hessian $\Lambda(x) = \mathbb{E}[\ell_{\theta\theta} \mid X=x]$
is obtained. The package detects the regime automatically from your inputs in most cases.
The full explanation is in [Estimation](estimation/index.md).

| Regime | Condition | Lambda strategy | Cross-fitting |
|---|---|---|---|
| A | RCT with known $F_T$ | Computed via Monte Carlo integration | 2-way |
| B | Linear / squared-error model | Analytic closed form | 2-way |
| C | Observational data, nonlinear model | Estimated with a separate regression | 3-way |

**Regime A: randomized experiments.** When treatment is randomized and you know the
distribution, you pass `is_randomized=True` and supply the distribution object. The package
integrates over $F_T$ to form $\Lambda(x)$ without needing a separate data fold.

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

result = inference(
    Y, T, X,
    model='logit',
    target='beta',
    is_randomized=True,
    treatment_dist=Normal(mean=0, std=1),
)
```

**Regime B: linear models.** The squared-error Hessian does not depend on $\theta(x)$, so
$\Lambda$ is available in closed form. You do not need to do anything special. Calling
`structural_dml` with `family='linear'` triggers this automatically.

```python
result = structural_dml(Y, T, X, family='linear')
# Analytic Lambda, 2-way cross-fitting
```

**Regime C: observational, nonlinear.** For logit, Poisson, and every other nonlinear
family under observational data, the package estimates $\Lambda(x)$ from a third data
split. The default estimator is ridge regression. See the next section for how to choose
among the available estimators.

```python
result = structural_dml(Y, T, X, family='logit')
# Ridge-estimated Lambda by default, 3-way cross-fitting
```

### Four common paths through the flowchart

These are the patterns that cover the large majority of real applications.

**Path 1: logistic demand, observational.** Binary purchase decisions with heterogeneous
price sensitivity.

```python
result = structural_dml(
    Y=Y, T=T, X=X,
    family='logit',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
)
print(result.summary())
# mu_hat = average price sensitivity E[beta(X)]
```

Flowchart path: binary outcome to `logit`, to `target='beta'`, observational, Regime C,
`structural_dml()`.

**Path 2: RCT with binary outcome.** An A/B test where the treatment assignment mechanism
is known.

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Bernoulli

result = inference(
    Y=Y, T=T, X=X,
    model='logit',
    target='ame',
    t_tilde=0.0,
    is_randomized=True,
    treatment_dist=Bernoulli(p=0.5),
)
print(result.summary())
# mu_hat = average marginal effect on P(Y=1)
```

Flowchart path: binary outcome to `logit`, to `target='ame'`, randomized, Regime A,
`inference()`.

**Path 3: continuous outcome, linear model.** The workhorse case for wage or earnings
regressions.

```python
result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
)
print(result.summary())
# mu_hat = E[beta(X)], the average treatment effect
```

Flowchart path: continuous outcome to `linear`, to `target='beta'`, Regime B,
`structural_dml()`.

**Path 4: custom loss and custom target.** For non-standard models where you supply your
own negative log-likelihood and target function.

```python
import torch
from deep_inference import inference

def custom_loss(y, t, theta):
    alpha, beta = theta[..., 0], theta[..., 1]
    mu = alpha + beta * t
    return (y - mu) ** 2

def custom_target(x, theta, t_tilde):
    return theta[..., 1]

result = inference(
    Y=Y, T=T, X=X,
    loss_fn=custom_loss,
    target_fn=custom_target,
    theta_dim=2,
)
```

Flowchart path: custom loss to `target_fn=...`, Regime C, `inference()`.

### Parameter reference

These are the parameters you will set most often. The full signature is in
[API Reference](api.md).

| Parameter | Type | Default | What it controls |
|---|---|---|---|
| `family` | str | `'linear'` | Model family: `linear`, `logit`, `poisson`, etc. |
| `target` | str | `'beta'` | Target functional: `beta`, `ame`, `latent`, `observed` |
| `target_fn` | callable | None | Custom target function (Jacobian computed by autodiff) |
| `hidden_dims` | list | `[64, 32]` | Neural network hidden layer widths |
| `epochs` | int | 100 | Training epochs |
| `n_folds` | int | 50 | Number of cross-fitting folds |
| `lambda_method` | str | `'ridge'` | Lambda estimator: `ridge`, `lgbm`, `aggregate` |
| `is_randomized` | bool | False | Whether treatment is randomized (triggers Regime A) |
| `treatment_dist` | object | None | Treatment distribution for Regime A |
| `t_tilde` | float | 0.0 | Evaluation point for AME and custom targets |

---

## Choosing the Lambda method

This section applies only to Regime C (observational, nonlinear models). If your model is
linear or your treatment is randomized, the Lambda method is fixed automatically and you can
skip this section.

For Regime C the package must estimate $\Lambda(x) = \mathbb{E}[\ell_{\theta\theta} \mid X=x]$
from data. You have several estimators to choose from. The key finding is that
**correlation with the oracle $\Lambda$ is not what makes inference valid**.

```{list-table}
:header-rows: 1

* - Method
  - Correlation with oracle
  - Coverage
  - Use it?
* - `ridge` (default)
  - 0.508
  - 96%
  - Yes. The safe validated default.
* - `lgbm`
  - 0.978
  - 96%
  - Yes. Good accuracy alternative.
* - `aggregate`
  - 0.000
  - 95%
  - Valid, but ignores $X$-dependence entirely.
* - `rf`
  - 0.904
  - ~90%
  - Moderate. Not recommended as a first choice.
* - `mlp`
  - 0.997
  - 67%
  - No. High correlation but invalid standard errors.
```

```{warning}
The highest correlation with the oracle (`mlp`, 0.997) gives the worst coverage (67%).
Correlation of the $\Lambda$ estimate with the truth is not what makes inference valid. The
variance of the estimate matters too, and the MLP overfits the training fold in a way that
inflates the correction term. Use `ridge` unless you have verified an alternative on your
own data.
```

The package emits a warning if you select `lambda_method='mlp'`:

```
UserWarning: Lambda method 'mlp' can produce invalid standard errors despite high
correlation with oracle. Consider method='ridge' (default) or 'lgbm' for validated coverage.
```

### Why ridge works despite low correlation

Ridge regression with a heavy regularization penalty (`alpha=1000`) pulls the Lambda
estimates toward their mean. That shrinks the per-observation variance of the correction
term, which stabilizes the influence-function average even though the point estimates are
far from oracle. The validated setting is:

```python
result = structural_dml(
    Y, T, X,
    family='logit',
    lambda_method='ridge',   # default
    ridge_alpha=1000.0,      # default; do not lower this
)
```

Lowering `ridge_alpha` toward 1 can be catastrophic. At `alpha=1.0` the bias is 66 and the
SE ratio is 0.5. The default of 1000 is intentional.

### When to use lgbm instead

If you want a Lambda estimator that tracks the oracle more closely (correlation 0.978 vs
0.508 for ridge) and still gives validated coverage, use `lgbm`. It requires heavy
regularization of its own:

```python
result = structural_dml(
    Y, T, X,
    family='logit',
    lambda_method='lgbm',
)
```

The package sets conservative LGBM defaults internally: 20 trees, max depth 2, 150 minimum
child samples per node, and L1 and L2 penalties of 5. Without these constraints LGBM can
produce negative eigenvalues in $\hat\Lambda$, which breaks the Cholesky solve downstream.

### When to use aggregate

The `aggregate` method ignores covariate dependence and collapses $\Lambda(x)$ to a single
global-mean matrix. It gives 95% coverage by construction (because the aggregated matrix is
always PSD and the resulting correction is well-conditioned), but it gives up all the
efficiency gain from knowing how curvature varies with $X$. Use it as a sanity check or
when $n$ is very small and you cannot afford a third data split.

```python
result = structural_dml(Y, T, X, family='logit', lambda_method='aggregate')
```

---

## Reading the diagnostics

After a run, inspect `result.diagnostics`. These are the numbers that tell you whether the
influence-function correction worked correctly.

```{list-table}
:header-rows: 1

* - Diagnostic
  - Healthy range
  - What it tells you
* - `min_lambda_eigenvalue`
  - greater than 1e-4
  - The estimated $\Lambda(x)$ is invertible everywhere. Below this threshold the
    correction is numerically unstable.
* - `correction_ratio`
  - roughly 0.1 to 50
  - How large the bias correction was relative to the naive plug-in. A value near zero
    means the network was nearly unbiased to begin with. A very large value means the
    correction dominated and you should check convergence.
* - `mean_condition_number`
  - single digits preferred
  - How close $\hat\Lambda(x)$ is to singular at low-overlap covariate values. A large
    condition number suggests near-zero eigenvalues in some regions.
* - `corr_alpha`, `corr_beta`
  - above 0.7
  - Cross-fold correlation between the network's predicted $\hat\alpha(X)$ and
    $\hat\beta(X)$ and the truth, when truth is known. Below 0.7 suggests the network
    has not converged.
```

### Accessing the diagnostics in code

```python
result = structural_dml(Y, T, X, family='logit', epochs=100, n_folds=50)

diag = result.diagnostics
print(f"Min Lambda eigenvalue: {diag['min_lambda_eigenvalue']:.6f}")
print(f"Correction ratio:      {diag['correction_ratio']:.4f}")
print(f"Mean condition number: {diag['mean_condition_number']:.2f}")
```

You can also call `result.summary()`, which prints these diagnostics alongside the point
estimate, SE, and confidence interval in a statsmodels-style table.

---

## Troubleshooting

### Low SE ratio (SE ratio much below 1.0)

The SE ratio is the ratio of the estimated standard error to the empirical standard
deviation of the influence scores. A ratio below 0.7 means your confidence intervals are
too narrow and your coverage will be below the nominal 95%.

The most common causes, in order of likelihood, are below.

**Wrong Lambda method.** The most common culprit. Switch from `mlp` to `ridge` or `lgbm`.
The MLP estimator achieves only 67% coverage on the benchmark logit DGP despite its high
oracle correlation. See the Lambda method table above.

**Network has not converged.** Increase `epochs` and `patience`. A network that has not
found a good minimum leaves large residuals in the score, which inflates the correction
term and can either over-correct or under-correct depending on sign. Check `corr_beta` in
`result.diagnostics`. Values below 0.7 indicate poor convergence.

**Too few folds.** With `n_folds` below 20, the cross-fitting estimator has high variance.
The documented recommendation is `n_folds=50`. You can also increase `n_repeats` (repeated
cross-fitting with median aggregation) to stabilize the SE estimate further.

**Low overlap (propensity near 0 or 1).** For logit, the Hessian weight $p(1-p)$
approaches zero near the boundary of the probability simplex. This makes $\Lambda(x)$
near-singular in some regions. Check `min_lambda_eigenvalue`. If it is below 1e-4, trim
observations with extreme predicted probabilities or use the Cholesky Lambda estimator,
which adds a small Tikhonov term by default.

### Lambda instability (min eigenvalue near zero or negative)

This appears as numerical errors during the Cholesky solve or as exploding influence
scores.

For logit, the fundamental issue is that $\det\Lambda(x) = 4p(1-p) \to 0$ when $p \to 0$
or $p \to 1$. Strategies to address this, in order of preference, are below.

Use the Cholesky Lambda path. The Cholesky factorization adds a small Tikhonov term by
default, which keeps the smallest eigenvalue bounded below.

Trim extreme predictions. Before fitting, remove or reweight observations where the
network predicts probabilities below 0.01 or above 0.99.

Use `lambda_method='aggregate'`. The aggregate estimator always produces a PSD matrix by
construction. You trade efficiency for stability.

### Coverage below 90% on a nonlinear model

For the logit DGP, the ridge Lambda achieves 86% coverage rather than the target 96% on
one specific benchmark (see [Replications](replications/index.md)). The Cholesky path
reaches 98% on the same DGP. If you see coverage below 90%, switching to the Cholesky
Lambda estimator is the recommended fix.

The coverage discrepancy traces to how $\Lambda(x)$ is inverted. Ridge estimates the
entries of $\Lambda(x)$ well in Frobenius norm, but the inverse $\Lambda(x)^{-1}$ can be
poorly estimated at low-overlap values where $\Lambda$ is near-singular. The Cholesky path
is more numerically stable in exactly those regions.

### High bias (estimate far from oracle)

If your estimate is far from the oracle even after convergence, check whether `n_folds` is
high enough and `n_repeats` is at least 5. With `n_folds=20` the cross-fitting estimator
systematically under-covers at $n=2000$. The recommended minimum is `n_folds=50`, matching
the `eval_06_coverage.py` settings documented in [Simulation Studies](simulation_studies.md).

If bias persists, confirm the model family is correct. A linear model fit to binary data
produces biased estimates because the squared-error loss does not match the Bernoulli
likelihood. The influence-function correction removes regularization bias in the network,
not model misspecification bias.

### `mlp` Lambda warning

If you see:

```
UserWarning: Lambda method 'mlp' can produce invalid standard errors...
```

this is not a soft suggestion. The MLP Lambda estimator has 67% coverage on the documented
benchmark. Switch to `ridge` (the default) or `lgbm`.

---

## Worked application: personalized pricing

This application follows Dube and Misra (2022, *Journal of Political Economy*). The setting
is a firm that wants to measure heterogeneous price sensitivity, compute optimal prices
per customer segment, and measure the welfare implications of price discrimination.

For the theoretical derivation of the elasticity, willingness-to-pay, and consumer surplus
targets, see [Theory](theory/index.md). For the `elasticity`, `welfare`, and `wtp` target
API, see [API: Targets](api.md).

### When to use this approach

Use this application pattern when you have binary purchase data (buy or no-buy), treatment
is price or a continuous instrument, and you want heterogeneous price elasticities with
valid confidence intervals rather than average effects only.

### Mathematical setup

The structural model is logit demand:

$$P(\text{buy} \mid X, P) = \sigma(\alpha(X) + \beta(X) \cdot P),$$

where $\beta(X) < 0$ is the heterogeneous price sensitivity and $\sigma$ is the logistic
function. The neural network learns $\alpha(X)$ and $\beta(X)$ jointly.

The three economic targets of interest are below.

| Target | Formula | Interpretation |
|---|---|---|
| Elasticity | $\eta = (1-p) \cdot \beta \cdot P$ | Percent change in demand per percent change in price |
| WTP | $-\beta_\text{attr} / \beta_\text{price}$ | Dollar value of a one-unit attribute improvement |
| Consumer surplus | $\log(1 + e^V) / \lvert\beta_\text{price}\rvert$ | Expected utility in monetary units (Small and Rosen, 1981) |

### Step 1: generate data

```python
import numpy as np
from scipy.special import expit

np.random.seed(42)
n = 2000

# Consumer characteristics
X = np.random.randn(n, 3)

# Heterogeneous parameters
alpha_true = 1.0 + 0.3 * X[:, 0]    # Base utility
beta_true = -0.5 - 0.2 * X[:, 0]    # Price sensitivity (negative)

# Price (continuous treatment)
T = np.random.uniform(0.5, 3.0, n)

# Purchase outcome
p_true = expit(alpha_true + beta_true * T)
Y = np.random.binomial(1, p_true).astype(float)
```

### Step 2: estimate price elasticity

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

### Step 3: compute consumer welfare

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

### Step 4: trace the demand curve across prices

Evaluate elasticity at multiple price points to map out how demand responds along the full
range of observed prices. Each call to `inference` with a different `t_tilde` gives you a
point estimate and confidence interval at that price level.

```python
prices = np.linspace(0.5, 3.0, 6)
elasticities = []

for p in prices:
    r = inference(
        Y, T, X,
        model='logit',
        target='elasticity',
        t_tilde=float(p),
        epochs=100,
        n_folds=50,
    )
    elasticities.append((p, r.mu_hat, r.se))
    print(f"P={p:.1f}: elasticity={r.mu_hat:.4f} +/- {r.se:.4f}")
```

### Optimal pricing from estimated elasticity

With the estimated elasticity $\hat\eta$, the Lerner markup rule gives the
profit-maximizing price for a firm with marginal cost $MC$:

$$\frac{P - MC}{P} = -\frac{1}{\hat\eta}$$

```python
MC = 1.0            # marginal cost
eta_hat = result_elast.mu_hat

# Optimal markup: P* = MC * eta / (1 + eta)
P_optimal = MC * eta_hat / (1 + eta_hat)
print(f"Optimal price: {P_optimal:.4f}")
```

### Uniform versus personalized pricing

The key empirical insight in Dube and Misra (2022): personalized pricing based on
heterogeneous $\beta(X)$ can increase both profits and consumer welfare relative to uniform
pricing, when firms can identify consumers with high willingness to pay. The
individual-level $\hat\beta(X_i)$ from `result.theta_hat[:, 1]` is the object you use to
segment customers.

---

## Worked application: continuous treatment

This application follows Colangelo and Lee (2026). The setting is a dose-response problem:
treatment $T$ is continuous (advertising spend, drug dosage, subsidy level) and you want to
estimate the average causal effect at any arbitrary treatment level, with valid confidence
intervals throughout.

For the relationship between this structural approach and the nonparametric double-debiased
machine learning approach of Colangelo and Lee (2026), see [Theory](theory/index.md).

### When to use this approach

Use this pattern when treatment is continuous rather than binary, you want the heterogeneous
dose-response curve rather than just an average treatment effect, and you have a reasonable
economic model for the outcome process (Poisson for counts, logit for binary choice, and
so on).

### Structural versus nonparametric: a comparison

Two published approaches handle continuous treatments.

| Approach | Flexibility | Interpretability | Extrapolation |
|---|---|---|---|
| Nonparametric (Colangelo-Lee 2026) | High | Low | Poor |
| Structural DNN (Farrell-Liang-Misra) | Medium | High | Good |

The structural approach works best when you have a defensible model for the link function.
The nonparametric approach works best when you do not want to commit to a functional form.
The two methods are complementary: you can use nonparametric estimates as a specification
check on the structural estimates.

### Example: advertising spend and sales

The data-generating process is Poisson: sales are a random count with mean
$\exp(\alpha(X) + \beta(X) \cdot T)$, where $T$ is advertising spend.

**Step 1: generate data.**

```python
import numpy as np

np.random.seed(42)
n = 2000

# Firm characteristics
X = np.random.randn(n, 3)

# Heterogeneous parameters
alpha_true = 2.0 + 0.3 * X[:, 0]    # Baseline log-sales
beta_true = 0.1 + 0.05 * X[:, 0]    # Ad effectiveness (positive)

# Advertising spend (continuous, right-skewed)
T = np.random.exponential(2.0, n)

# Sales (count outcome)
mu = np.exp(alpha_true + beta_true * T)
Y = np.random.poisson(mu).astype(float)
```

**Step 2: estimate the average treatment effect.**

```python
from deep_inference import inference

# Average beta: marginal effect of ad spend on log-sales
result_beta = inference(
    Y, T, X,
    model='poisson',
    target='beta',
    epochs=100,
    n_folds=50,
)

print(f"E[beta(X)]: {result_beta.mu_hat:.4f}")
print(f"95% CI: [{result_beta.ci_lower:.4f}, {result_beta.ci_upper:.4f}]")
# True value: E[beta(X)] = 0.1
```

**Step 3: estimate elasticity at different spending levels.**

For a Poisson (log-link) model, the elasticity at treatment level $t$ is $\beta \cdot t$.

```python
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

**Step 4: custom dose-response target.**

For the predicted outcome $E[Y(t)] = \exp(\alpha + \beta \cdot t)$ at a given level, you
can define the target function directly.

```python
import torch

def dose_response(x, theta, t_tilde):
    """E[Y(t)] = exp(alpha + beta * t)."""
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

### Tracing the full dose-response curve

Evaluate at many points and plot the resulting curve with confidence bands.

```python
import matplotlib.pyplot as plt

t_values = np.linspace(0.5, 5.0, 10)
results = []

for t in t_values:
    r = inference(
        Y, T, X,
        model='poisson',
        target='elasticity',
        t_tilde=float(t),
        epochs=100,
        n_folds=50,
    )
    results.append({
        't': t,
        'elasticity': r.mu_hat,
        'se': r.se,
        'ci_lower': r.ci_lower,
        'ci_upper': r.ci_upper,
    })

ts    = [r['t'] for r in results]
etas  = [r['elasticity'] for r in results]
ci_lo = [r['ci_lower'] for r in results]
ci_hi = [r['ci_upper'] for r in results]

plt.fill_between(ts, ci_lo, ci_hi, alpha=0.2)
plt.plot(ts, etas, 'b-o')
plt.xlabel('Treatment level (t)')
plt.ylabel('Elasticity')
plt.title('Dose-Response Elasticity with 95% CI')
plt.show()
```

### Supported families for continuous treatment

All families accept a scalar continuous $T$ natively. The elasticity formula varies by link
function.

| Family | Link | Elasticity formula | Use case |
|---|---|---|---|
| `linear` | Identity | $\beta \cdot t$ | Continuous, unbounded outcomes |
| `logit` | Logit | $(1-p) \cdot \beta \cdot t$ | Binary outcomes |
| `poisson` | Log | $\beta \cdot t$ | Count data |
| `gamma` | Log | $\beta \cdot t$ | Positive continuous outcomes |
| `negbin` | Log | $\beta \cdot t$ | Overdispersed counts |

---

## Worked application: multimodal covariates

This application shows how to use $X$ that comes from neural-network embeddings rather than
tabular columns. Modern text and image encoders (BERT, ResNet, CLIP) produce covariate
vectors with 384 to 2048 dimensions. The package handles high-dimensional $X$ directly
because the structural network treats $X$ as its input regardless of dimension.

### When to use this approach

Use this pattern when your covariates come from unstructured data (job postings, product
images, research abstracts) that a language model or vision model has already encoded into
a fixed-length vector. You believe those embeddings capture heterogeneity in the treatment
effect that tabular proxies would miss.

### A practical note on dimensionality

For very high-dimensional embeddings (384 dimensions or more), use PCA to reduce to
approximately 64 dimensions before passing to the package. The rough rule is $n/\dim > 50$,
meaning you need at least 50 observations per covariate dimension for stable estimation.
At $n=2000$ with 64 dimensions, that ratio is 31, which is workable. At 384 dimensions, it
is 5, which is not.

### Gallery of three scenarios

| Model | Outcome Y | Treatment T | Covariates X |
|---|---|---|---|
| `linear` | Log wages | Years of experience | Job description embeddings (64-dim) |
| `logit` | Purchase (0/1) | Discount percent | Product image embeddings (64-dim) |
| `poisson` | Citation count | Open Access (0/1) | Abstract embeddings (64-dim) |

### Example 1: wages and job description embeddings

A labor economist studies how returns to experience vary across job types. The covariate
$X$ is a 64-dimensional PCA projection of BERT embeddings of job descriptions.

$$Y_i = \alpha(X_i) + \beta(X_i) \cdot T_i + \varepsilon_i$$

where $Y$ is log hourly wage, $T$ is years of experience, and $X$ is the job embedding.

```python
import numpy as np
from deep_inference import structural_dml

# X_embeddings: shape (n, 64), job description embeddings (PCA-reduced from BERT)
# T: years of experience
# Y: log hourly wage

result = structural_dml(
    Y=Y, T=T, X=X_embeddings,
    family='linear',
    hidden_dims=[128, 64],
    epochs=150,
    n_folds=50,
)

print(result.summary())

# Distribution of individual-level effects
beta_hat = result.theta_hat[:, 1]
print(f"Effect range: [{beta_hat.min():.3f}, {beta_hat.max():.3f}]")
```

In a simulation calibrated to this scenario, the network recovers the shape of $\beta(X)$
with correlations between 0.4 and 0.9 depending on the signal-to-noise ratio of the
embedding dimensions.

### Example 2: discount effectiveness and product image embeddings

An e-commerce firm studies whether a 10-percent discount is more effective for some
products than others. The hypothesis is that premium-looking products may be hurt by
discounts (quality signaling), while value-oriented products benefit.

```python
result = structural_dml(
    Y=Y_purchase, T=T_discount, X=X_image_embeddings,
    family='logit',
    hidden_dims=[128, 64],
    epochs=150,
    n_folds=50,
)

# Identify which products respond positively to discounts
beta_hat = result.theta_hat[:, 1]
discount_sensitive = beta_hat > np.median(beta_hat)
print(f"Products to discount: {discount_sensitive.sum()} / {len(beta_hat)}")
```

Products where the estimated $\hat\beta(X_i)$ is high are the ones where discounts increase
purchase probability. Products where $\hat\beta(X_i)$ is low or negative are the ones where
discounts backfire.

### Example 3: open-access citation advantage and abstract embeddings

A bibliometrician studies the Open Access citation advantage. The hypothesis is that
technical papers behind paywalls benefit more from open access than papers that are already
widely circulated.

```python
result = structural_dml(
    Y=Y_citations, T=T_open_access, X=X_abstract_embeddings,
    family='poisson',
    hidden_dims=[128, 64],
    epochs=150,
    n_folds=50,
)

print(result.summary())
print(f"Citation multiplier: {np.exp(result.mu_hat):.2f}x")

# Which papers benefit most from open access?
beta_hat = result.theta_hat[:, 1]
top_beneficiaries = np.argsort(beta_hat)[-100:]
```

The citation multiplier for a Poisson model is $\exp(\hat\mu)$, because the log-link means
$\hat\beta$ is on the log-counts scale. A coefficient of 0.3 corresponds to a 1.35x
citation lift from open access on average.

### Loading real embeddings

Replace simulated embeddings with real ones using the patterns below. In every case, reduce
dimensionality with PCA before passing to `structural_dml`.

**Text embeddings via sentence-transformers:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_raw = encoder.encode(texts)   # shape (n, 384)

pca = PCA(n_components=64)
X = pca.fit_transform(X_raw)    # shape (n, 64)

result = structural_dml(Y, T, X, family='linear', epochs=150, n_folds=50)
```

**Image embeddings via torchvision:**

```python
import torch
from torchvision import models
from sklearn.decomposition import PCA

resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

X_raw = resnet(images).squeeze().numpy()   # shape (n, 2048)

pca = PCA(n_components=64)
X = pca.fit_transform(X_raw)              # shape (n, 64)

result = structural_dml(Y, T, X, family='logit', epochs=150, n_folds=50)
```

**Multimodal embeddings via CLIP:**

```python
import clip
from sklearn.decomposition import PCA

model, preprocess = clip.load("ViT-B/32")
X_images = model.encode_image(images)
X_texts  = model.encode_text(texts)
X_raw    = torch.cat([X_images, X_texts], dim=1).numpy()   # shape (n, 1024)

pca = PCA(n_components=64)
X = pca.fit_transform(X_raw)                               # shape (n, 64)

result = structural_dml(Y, T, X, family='poisson', epochs=150, n_folds=50)
```

### Summary of takeaways for embedding covariates

High-dimensional $X$ from pre-trained encoders works out of the box as long as you keep the
effective dimension manageable with PCA. The neural network inside the package learns which
embedding directions drive heterogeneity in $\beta(X)$ without you specifying them. The
influence-function correction provides valid confidence intervals for the population-average
effect even when $X$ is 64-dimensional.

For a runnable gallery that exercises all three scenarios with simulated embeddings, run:

```bash
python tutorials/06_multimodal_gallery.py
```

This prints point estimates, confidence intervals, HTE distribution tables, ASCII
histograms of $\hat\beta(X)$, and policy insights for each scenario.

---

## See also

- [Overview](overview.md): what the package does and why
- [Quick Start](quickstart.md): installation and a first runnable example
- [Models](models/index.md): one detailed page per outcome type
- [Estimation](estimation/index.md): how the network is fit and how $\Lambda(x)$ is obtained
- [Inference](inference/index.md): how standard errors and intervals are formed
- [Theory](theory/index.md): the formal walkthrough with theorems
- [Replications](replications/index.md): validated benchmarks with numbers
- [API Reference](api.md): every function and result object
