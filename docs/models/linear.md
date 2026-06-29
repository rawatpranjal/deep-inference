# Linear Model

The linear model is the simplest and most numerically stable option in this package. Use it
whenever your outcome is continuous and unbounded. It is also the best starting point when you
are learning the package, because its Hessian is parameter-free, the influence-function math
reduces to a moment computation, and coverage is well-validated at M=100.

## Source Papers

The linear formulation here follows Farrell, Liang, and Misra (2021, 2025). The full
transcribed papers are in [References](../references/index.md).

## When to Use

Choose the linear model when:

- the outcome is continuous with no natural lower or upper bound (wages, test scores, prices,
  consumption expenditure),
- you are comfortable assuming that the conditional mean of $Y$ is approximately linear in $T$
  at each value of $X$, and
- you want the fastest, most stable path to a confidence interval.

It is not appropriate for binary, count, or strictly positive outcomes. For those, see
[Logit](logit.md), [Probit](probit.md), or the Gaussian page.

## Model

The structural model is

$$
Y = \alpha(X) + \beta(X) \cdot T + \varepsilon, \qquad
\mathbb{E}[\varepsilon \mid X, T] = 0,
$$

where $\alpha(X)$ and $\beta(X)$ are learned by a neural network with two output heads. The
linear predictor is $\eta = \alpha(X) + \beta(X) \cdot T$, and the identity link gives
$\mu = \eta$. The parameter vector is $\theta = (\alpha, \beta)$, so $\theta_{\dim} = 2$.

## Loss / Likelihood

The training loss is squared error:

$$
\ell(y, t, \theta) = \bigl(y - \alpha - \beta t\bigr)^2.
$$

This is not the negative log-likelihood of a Gaussian with known variance. It is the raw
squared-error objective. The residual scale is not part of $\theta$ here. If you want to
model the variance as well, use the [Gaussian model](gaussian.md) instead.

## Score and Hessian

The gradient of the loss with respect to $\theta = (\alpha, \beta)$ is

$$
\nabla_\theta \ell = -2\,(y - \mu)
\begin{pmatrix} 1 \\ t \end{pmatrix}.
$$

The Hessian is

$$
\nabla^2_\theta \ell = 2
\begin{pmatrix} 1 & t \\ t & t^2 \end{pmatrix}.
$$

Two properties of this Hessian determine almost everything downstream.

First, the Hessian does not depend on $\theta$ at all. The matrix entries are functions of
$t$ only. This means $\Lambda(x) = \mathbb{E}[\nabla^2_\theta \ell \mid X = x]$ equals
$2\,\mathbb{E}[TT^\top \mid X = x]$, which can be estimated from the data without knowing
$\hat\theta$ first.

Second, the Hessian does not depend on $y$ either. This distinction matters for which regime
the package enters. In the source, both `hessian_depends_on_theta = False` and
`hessian_depends_on_y = False` (see `src/deep_inference/models/linear.py` and
`src/deep_inference/families/linear.py`).

## Target

The default estimand is the average treatment effect

$$
\mu^* = \mathbb{E}[\beta(X)].
$$

This is the population-average slope of $Y$ with respect to $T$, integrated over the
covariate distribution. The target Jacobian is $H_\theta = (0,\; 1)$, selecting the second
component of $\theta$.

## Influence Function

The influence function for observation $i$ is

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i)^\top\, \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i)).$$

For the default $E[\beta]$ target, the plug-in is $H(\hat\theta(X_i)) = \hat\beta(X_i)$, the
per-observation slope (not the constant $\mu^*$), and $H_\theta = (0,\; 1)$ is the unit
selector picking the $\beta$ component. The score is
$\ell_\theta = -2(Y_i - \mu_i)(1, T_i)^\top$, the gradient from the Score and Hessian section
evaluated at $\hat\theta(X_i)$. The correction enters with a minus sign, which removes the
regularization bias the network injected into $\hat\beta(X_i)$. A plus sign would double the
bias and is wrong.

Because the Hessian is constant, $\Lambda(x) = 2\,\mathbb{E}[TT^\top \mid X = x]$ can be
estimated by regressing $TT^\top$ on $X$. No third data split is needed, since the Lambda
estimator is independent of $\hat\theta$.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$, the sample standard deviation of the $\psi_i$ divided by
$\sqrt{n}$.

## Algorithm and Regime

The linear model uses **Regime B** (closed-form analytic Lambda, 2-way cross-fitting).

The reason is that $\nabla^2_\theta \ell$ does not depend on $\theta$. The cross-fitting
engine checks `hessian_depends_on_theta` and, when it is `False`, branches into the 2-way
path automatically. You do not need to set anything manually.

Under 2-way cross-fitting, the data are split into $K$ folds. For each held-out fold, the
network is trained on the remaining folds and used to compute $\hat\theta$ and influence
scores on the held-out observations. Lambda is estimated on the same two splits because it
does not require a separate, independent view of the data.

Regime A (Monte Carlo integral over the known treatment distribution) applies only when the
treatment is randomized and its distribution is known. Regime C (3-way split with an
estimated Lambda) applies to nonlinear models. The linear model never needs either.

## Usage

**Legacy API** with `structural_dml`:

```python
import numpy as np
from deep_inference import structural_dml

np.random.seed(42)
n = 2000
X = np.random.randn(n, 3)
T = np.random.randn(n)
beta_x = 1.0 + 0.5 * X[:, 0]
Y = X[:, 0] + beta_x * T + np.random.randn(n)

result = structural_dml(
    Y, T, X,
    family='linear',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)

print(f"Estimate: {result.mu_hat:.4f}")
print(f"SE:       {result.se:.4f}")
print(f"95% CI:   [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

**New API** with `inference`:

```python
from deep_inference import inference

result = inference(
    Y, T, X,
    model='linear',
    target='beta',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
)
print(result.summary())
```

Use at least `n_folds=50`. The influence function estimate uses each fold's held-out
predictions, and with fewer folds each network sees materially less training data per fit.

## Evidence

**Parameter recovery** (n=5000, one seed, source:
[Eval 01](../simulation_studies/eval_01.md)):

| metric | value |
|--------|-------|
| RMSE($\alpha$) | 0.036 |
| RMSE($\beta$) | 0.045 |
| Corr($\alpha$) | 0.994 |
| Corr($\beta$) | 0.998 |
| status | PASS |

**Coverage** (M=100 replications, truth $\mu^*=1.0000$, source:
[Replications](../replications/index.md)):

| method | bias | SE-ratio | coverage |
|--------|------|----------|----------|
| Oracle-MLE | -0.0008 | 0.96 | 96% |
| FLM[cholesky] | -0.0041 | 0.92 | 93% |
| FLM[oracle-Lambda] | -0.0030 | 0.96 | 95% |
| RieszNet | -0.0083 | 1.09 | 97% |
| Naive (no correction) | -0.0042 | 0.17 | 23% |

FLM[cholesky] reaches 93% at M=100. That figure sits about 2.6 percentage points from the
lower edge of the 93-97% target band, so a larger Monte Carlo would tighten the estimate.
The oracle-Lambda ceiling is 95%, confirming that the gap is in Lambda estimation rather than
in the influence-function assembly. The replications page notes this caveat explicitly.

The Naive row (23% coverage, SE-ratio 0.17) shows how badly an uncorrected network performs.
A regularized network has far less variance in its predictions than an unbiased estimator
would, so the naive standard error is severely underestimated.

## Diagnostics and Pitfalls

**SE ratio below 0.9.** The most common cause is too few folds or too few epochs. Try
`n_folds=50` and `epochs=200` first.

**Lambda instability.** This is rare for the linear model, because Lambda estimation reduces
to a moment condition. If `diagnostics['min_lambda_eigenvalue']` is near zero, check whether
$T$ has near-zero conditional variance given $X$, or whether the network outputs essentially
constant $T$ predictions.

**Identically zero influence scores.** This can happen when the network memorizes training
data and held-out residuals are all near zero. Reduce weight decay or increase $n$.

**The `lambda_method` argument does not apply here.** Lambda is computed analytically from
the constant Hessian, so no ridge or MLP estimator is invoked. You will not see the mlp
warning on this family.

## References and API

- [References index](../references/index.md) for the FLM 2021 and FLM 2025 paper
  transcripts.
- [API: families](../api/families.md) for `LinearFamily` and the full `structural_dml`
  signature.
