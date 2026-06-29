# Weibull Model

The Weibull model handles positive duration or time-to-event outcomes where the hazard
rate is monotone over time. The treatment shifts the log-scale of the distribution, and
the influence function delivers a valid confidence interval for the average shift across
the covariate distribution.

## Source Papers

The structural inference framework is Farrell, Liang, and Misra (2021, 2025). See
[References](../references/index.md) for the full annotated bibliography. The Weibull
accelerated failure time model and its parametric hazard interpretation are covered in
any graduate econometrics or biostatistics textbook.

## When to Use

Use the Weibull model when:

- The outcome is a strictly positive duration (time to failure, subscription length,
  survival time, tenure).
- The hazard rate is expected to be monotone over time (decreasing, constant, or
  increasing).
- You want to estimate how a treatment shifts the log-scale parameter heterogeneously
  across covariates.

Concrete examples where Weibull is the right choice:

- Equipment reliability: time to failure for machinery. Older machines fail at
  increasing rates, giving $k > 1$.
- Customer churn: subscription duration where new subscribers are most at risk early
  on, giving $k < 1$.
- Clinical trials: patient survival time after an intervention.

Do not use Weibull when outcomes include zeros or negative values. The package raises a
`ValueError` if any entry of `Y` is non-positive. If you have zero durations (e.g., a
machine that fails on the first day), consider adding a small constant or using a
different model. Do not use Weibull when the hazard rate is non-monotone (for example,
bathtub-shaped); the Weibull family cannot capture that shape with a single fixed $k$.

## Model

The data-generating process is:

$$Y \sim \text{Weibull}(k, \lambda), \quad \lambda = e^\eta, \quad \eta = \alpha(X) + \beta(X) \cdot T$$

The neural network learns two functions of the covariates:

- $\alpha(X)$: the baseline log-scale.
- $\beta(X)$: the treatment effect on the log-scale.

So `theta_dim = 2` and $\theta(X) = [\alpha(X), \beta(X)]$.

The shape parameter $k > 0$ is fixed at initialization (default `shape=2.0`) and is not
estimated by the network. You pass it as a constructor argument.

Interpretation of $k$:

- $k < 1$: decreasing hazard (early-life or infant-mortality pattern)
- $k = 1$: constant hazard (the distribution reduces to an exponential)
- $k > 1$: increasing hazard (wear-out, aging, or maturation pattern)

The mean and variance of $Y$ given $\lambda$ are:

$$E[Y] = \lambda \,\Gamma\!\left(1 + \tfrac{1}{k}\right), \qquad \text{Var}(Y) = \lambda^2 \left[\Gamma\!\left(1 + \tfrac{2}{k}\right) - \Gamma\!\left(1 + \tfrac{1}{k}\right)^2\right]$$

Because the model uses a log link, $\beta(X)$ is the effect on $\log \lambda$, not on
$\lambda$ directly. An increase of 1 unit in $\beta$ multiplies the expected duration by
$e^\beta$.

## Loss / Likelihood

Define $z = (y/\lambda)^k$, the standardized duration raised to the power $k$. The Weibull
negative log-likelihood is:

$$\ell(y, \theta) = -\log k + k \log \lambda - (k-1)\log y + z$$

Substituting $z = (y e^{-\eta})^k$:

$$\ell(y, \theta) = -\log k + k\eta - (k-1)\log y + \left(\frac{y}{e^\eta}\right)^k$$

The term $-\log k$ is constant with respect to $\theta$ and does not affect the gradient
or Hessian. The terms involving $\theta$ are $k\eta$ (which penalizes large $\lambda$) and
$z$ (which penalizes small $\lambda$). The tension between these two terms defines the MLE.

Source: `WeibullFamily.loss` in
`src/deep_inference/families/weibull.py`.

## Score and Hessian

Define $z = (y/\lambda)^k$ as before.

The gradient of the loss with respect to $\eta = \log \lambda$ is:

$$\frac{\partial \ell}{\partial \eta} = k - k \cdot k \cdot z \cdot \frac{1}{k} \ldots$$

Working it out from $k\eta - (y e^{-\eta})^k$:

$$\frac{\partial \ell}{\partial \eta} = k - k \cdot z = k(1 - z)$$

The score with respect to $\theta = [\alpha, \beta]$ follows by the chain rule, since
$\partial\eta/\partial\alpha = 1$ and $\partial\eta/\partial\beta = t$:

$$\nabla_\theta \ell = k(1 - z) \cdot \begin{bmatrix} 1 \\ t \end{bmatrix}$$

The second derivative with respect to $\eta$ (the Hessian weight) is:

$$w = \frac{\partial^2 \ell}{\partial \eta^2} = k^2 z$$

The full per-observation Hessian matrix is:

$$H(\theta, y, t) = w \cdot \begin{bmatrix} 1 & t \\ t & t^2 \end{bmatrix} = k^2 z \cdot \begin{bmatrix} 1 & t \\ t & t^2 \end{bmatrix}$$

**Key structural facts:**

- The Hessian weight $w = k^2 z = k^2 (y/\lambda)^k$ depends on $\theta$ through
  $\lambda = e^{\alpha + \beta T}$.
- It also depends on $y$ directly through the ratio $y/\lambda$.
- Both dependencies mean $E[H \mid X]$ cannot be computed in closed form without knowledge
  of $\theta$, so Lambda must be estimated from data (Regime C).
- The Hessian is computed in closed form (no autodiff). The formula is validated against
  autodiff in [Eval 02](../simulation_studies/eval_02.md).
- `WeibullFamily.hessian_depends_on_theta()` returns `True`, which tells the engine to
  use the three-way split.

## Target

The default target is:

$$\mu^* = E[\beta(X)]$$

This is the average effect of the treatment on the log-scale parameter $\log \lambda$.
Because the model uses a log link, $e^{\mu^*}$ is the (average) multiplicative change in
expected duration. For small effects, $\mu^* \approx$ the percentage change divided by 100.

## Influence Function

The per-observation influence function contribution is:

$$\psi_i = H(\hat\theta(X_i)) - \begin{bmatrix} 0 & 1 \end{bmatrix} \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i))$$

where:

- $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation plug-in (the fitted slope), and
  $[0, 1]$ is the unit selector picking the $\beta$ component.
- $\ell_\theta(Y_i, T_i, \hat\theta(X_i)) = k(1 - z_i) \cdot [1, T_i]^\top$ is the score at
  observation $i$.
- $\Lambda(X_i) = E[\nabla^2_\theta \ell \mid X = X_i]$ is the expected Hessian (estimated from
  a held-out fold).
- The correction is subtracted (minus sign), which removes the regularization bias. A plus
  sign would double the bias and is wrong.

The score is zero when $z_i = 1$ (the outcome equals the fitted scale, $y_i = \lambda_i$).
Observations with $y_i \gg \lambda_i$ or $y_i \ll \lambda_i$ carry the largest correction.

The point estimate is $\hat\mu = (1/n) \sum_i \psi_i$, and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

**Regime C (three-way split).** The reason is that the Hessian weight $w = k^2 z$ depends
on both $\theta$ (through $\lambda$) and $y$. This means $\Lambda(X) = E[H \mid X]$ must be
estimated from data rather than computed analytically.

The three-way split works as follows:

1. Split the data into three folds.
2. Use fold 1 to train the structural network $\hat{\theta}(X)$.
3. Use fold 2 to estimate $\hat{\Lambda}(X)$ by fitting a secondary model that predicts
   per-observation Hessian entries given $X$.
4. Use fold 3 to assemble the influence function $\psi_i$ using $\hat{\theta}$ and
   $\hat{\Lambda}$ trained on the other folds.

This rotation is performed across all cross-fitting folds. The `three_way_theta_frac`
parameter controls the fraction of training data used for step 2 vs step 3.

The package detects the regime automatically from
`WeibullFamily.hessian_depends_on_theta()`.

## Usage

```python
import numpy as np
from deep_inference import structural_dml

np.random.seed(42)
n = 2000
X = np.random.randn(n, 5)
T = np.random.randn(n)

# True heterogeneous parameters
alpha_true = 3.0 + 0.3 * X[:, 0]
beta_true = 0.5 + 0.2 * X[:, 0]     # E[beta(X)] = 0.5 (true target)
k = 2.0                               # increasing hazard

# Generate Weibull durations: Y ~ Weibull(k, lambda)
lam = np.exp(alpha_true + beta_true * T)
Y = np.random.weibull(k, size=n) * lam

print(f"True mu* = {beta_true.mean():.4f}")
print(f"Min duration: {Y.min():.4f},  Max: {Y.max():.4f}")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='weibull',
    shape=2.0,              # fixed shape k (not learned)
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)
print(result.summary())

# Diagnostics
print(f"Min lambda eigenvalue: {result.diagnostics.get('min_lambda_eigenvalue', 'N/A')}")
print(f"Correction ratio: {result.diagnostics.get('correction_ratio', 'N/A')}")
```

To fit an exponential model ($k = 1$, constant hazard):

```python
result = structural_dml(Y, T, X, family='weibull', shape=1.0)
```

## Evidence

From [Eval 01: Parameter Recovery](../simulation_studies/eval_01.md) (n=5000, seed=42):

| Family | RMSE($\alpha$) | RMSE($\beta$) | Corr($\alpha$) | Corr($\beta$) | Status |
|--------|---------------|--------------|----------------|---------------|--------|
| weibull | 0.860 | 0.007 | 1.000 | 1.000 | PASS |

The network achieves a correlation of 1.000 with the oracle $\beta(X)$ function across
covariates. The RMSE($\alpha$) of 0.860 is large in absolute terms but reflects the large
scale of the intercept in the DGP; the correlation of 1.000 confirms the network learned
the correct shape.

Coverage benchmarking (FLM[cholesky] vs Oracle-MLE) has not yet been run for the Weibull
family. A custom MLE oracle is needed (Weibull MLE is not a standard GLM and requires a
separate implementation). The family is listed as Tier B, pending that infrastructure. See
[Replications](../replications/index.md).

## Diagnostics and Pitfalls

**Strictly positive outcomes required.** The package raises `ValueError: Y contains
non-positive values` if any $Y \leq 0$. Check your data before fitting.

**Shape $k$ is not learned.** The network learns $[\alpha(X), \beta(X)]$ but not $k$.
Misspecifying $k$ is a form of model misspecification. Bias in point estimates arises when
the true hazard shape is far from the assumed $k$. If you do not know $k$, try fitting
with $k \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$ and compare the out-of-sample Weibull NLL.
You can also estimate $k$ by maximum likelihood on a held-out set before using this package.

**Hessian weight can be extreme.** The weight $w = k^2 z = k^2 (y/\lambda)^k$ grows
without bound when $y \gg \lambda$ (the observed duration is far above the fitted scale).
A few outlier observations can dominate the Lambda estimate, making it noisy. Check
`result.diagnostics['min_lambda_eigenvalue']`. Values near zero indicate instability.
Increasing `n_folds` helps by using more data for each Lambda estimate.

**Clamping in source.** The source code clamps $\eta$ to $[-20, 20]$ and $y$ to
$[\,10^{-10}, \infty)$. If predicted $\eta$ values consistently hit these bounds, the
network is failing to fit the data. Try more epochs, a smaller learning rate, or verify
that `Y` is in the right units.

**Log-scale interpretation.** The estimated $\hat{\mu}^* = E[\hat{\beta}(X)]$ is on the
log-scale. To convert to a multiplicative effect on expected duration, compute
$\exp(\hat{\mu}^*)$. For example, $\hat{\mu}^* = 0.5$ implies the treatment multiplies
expected duration by $e^{0.5} \approx 1.65$.

**Censoring (right-censoring).** The standard Weibull NLL here assumes all durations are
fully observed. Right-censored survival data requires a modified likelihood that this
package does not implement out of the box. If you have censored observations, the naive
application of this family will produce biased estimates.

## References and API

- [References](../references/index.md) covers FLM 2021 and FLM 2025 in full.
- [API: Families](../api/families.md) documents `WeibullFamily` and its constructor
  arguments (`shape`, and all inherited arguments).
- Source:
  `src/deep_inference/families/weibull.py`
