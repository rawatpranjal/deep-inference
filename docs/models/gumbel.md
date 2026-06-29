# Gumbel Model

The Gumbel model (Type I extreme value distribution) handles outcomes that represent the
maximum of a large number of independent random variables. The treatment shifts the
location parameter of the distribution, and the influence function delivers a valid
confidence interval for the average shift across the covariate distribution.

## Source Papers

The structural inference framework is Farrell, Liang, and Misra (2021, 2025). See
[References](../references/index.md) for the full annotated bibliography. The Gumbel
distribution for block maxima is classical extreme value theory; the canonical reference
is Gumbel (1958).

## When to Use

Use the Gumbel model when:

- The outcome is the maximum (or minimum, after negation) drawn from a large pool of
  independent random variables.
- You believe the distribution of the outcome follows a Type I extreme value shape (with
  a heavier right tail than a normal, and a characteristic double-exponential CDF).
- You want to estimate how a treatment shifts the location of those extremes
  heterogeneously across covariates.

Concrete examples:

- Environmental risk: annual maximum flood heights, where the maximum of daily
  precipitation records over a year follows an extreme-value distribution.
- Structural engineering: the maximum stress a material experiences under repeated
  random loads, relevant for safety margin estimation.
- Auction economics: the winning bid (maximum of independent private valuations) when
  you want to estimate how auction characteristics shift the distribution of maxima.

To model minima instead of maxima, negate $Y$ before fitting: use `Y_fit = -Y` and
interpret results accordingly.

Do not use the Gumbel model when the hazard rate interpretation of Weibull is more
natural (Weibull is better suited to duration/survival data). The Gumbel is for the
distribution of extreme values across a cross-section, not for time-to-event data.

## Model

The data-generating process is:

$$Y \sim \text{Gumbel}(\mu, \sigma), \quad \mu = \alpha(X) + \beta(X) \cdot T$$

The distribution uses an **identity link**: the location parameter $\mu$ is directly the
linear predictor $\eta = \alpha(X) + \beta(X) \cdot T$. This is different from Weibull,
which uses a log link.

The neural network learns two functions:

- $\alpha(X)$: the baseline location.
- $\beta(X)$: the treatment effect on the location.

So `theta_dim = 2` and $\theta(X) = [\alpha(X), \beta(X)]$.

The scale parameter $\sigma > 0$ is fixed at initialization (default `scale=1.0`) and is
not learned. You pass it as a constructor argument.

The mean and variance of $Y$ given $\mu$ and $\sigma$ are:

$$E[Y] = \mu + \sigma \gamma_E, \qquad \text{Var}(Y) = \frac{\pi^2 \sigma^2}{6}$$

where $\gamma_E \approx 0.5772$ is the Euler-Mascheroni constant. Because the model uses
an identity link, $\beta(X)$ is a direct shift of the location parameter, and therefore
a direct shift of the mean (up to the constant $\sigma \gamma_E$).

## Loss / Likelihood

Define the standardized residual:

$$z = \frac{y - \mu}{\sigma}$$

The Gumbel negative log-likelihood is:

$$\ell(y, \theta) = z + e^{-z}$$

This is the standard Gumbel NLL derived from the density $f(y) = (1/\sigma) \exp(-(z + e^{-z}))$.
The loss has a distinctive shape: it is nearly linear for large positive $z$ (outcomes far
above the location) and grows rapidly for large negative $z$ (outcomes far below the
location), reflecting the heavier left tail of the Gumbel distribution.

Source: `GumbelFamily.loss` in
`src/deep_inference/families/gumbel.py`.

## Score and Hessian

The gradient of the loss with respect to the location $\mu$ is:

$$\frac{\partial \ell}{\partial \mu} = -\frac{1}{\sigma}\left(1 - e^{-z}\right)$$

The score with respect to $\theta = [\alpha, \beta]$ follows by the chain rule, since
$\partial \mu/\partial \alpha = 1$ and $\partial \mu/\partial \beta = t$:

$$\nabla_\theta \ell = -\frac{1}{\sigma}\left(1 - e^{-z}\right) \cdot \begin{bmatrix} 1 \\ t \end{bmatrix}$$

The score is zero when $z = 0$, meaning $y = \mu$ (the outcome falls exactly on the
fitted location). For large positive $z$, the score saturates at $-1/\sigma$ (the
exponential term vanishes). For large negative $z$, the score grows without bound because
$e^{-z}$ dominates.

The second derivative with respect to $\mu$ (the Hessian weight) is:

$$w = \frac{\partial^2 \ell}{\partial \mu^2} = \frac{e^{-z}}{\sigma^2}$$

The full per-observation Hessian matrix is:

$$H(\theta, y, t) = w \cdot \begin{bmatrix} 1 & t \\ t & t^2 \end{bmatrix} = \frac{e^{-z}}{\sigma^2} \cdot \begin{bmatrix} 1 & t \\ t & t^2 \end{bmatrix}$$

**Key structural facts:**

- The Hessian weight $w = e^{-z}/\sigma^2$ depends on $\theta$ through $\mu = \alpha + \beta T$
  (which enters $z = (y-\mu)/\sigma$).
- It also depends on $y$ through the same $z$.
- When $y \ll \mu$ (outcome far below the location), $z$ is large and negative, $e^{-z}$
  is very large, and observations receive very high curvature weight.
- When $y \gg \mu$ (outcome far above the location), $e^{-z} \approx 0$ and those
  observations contribute almost nothing to the Lambda estimate.
- Both dependencies mean $E[H \mid X]$ cannot be computed in closed form without
  knowledge of $\theta$. Lambda must be estimated from data (Regime C).
- The Hessian is computed in closed form (no autodiff). Validated against autodiff in
  [Eval 02](../simulation_studies/eval_02.md).
- `GumbelFamily.hessian_depends_on_theta()` returns `True`.

## Target

The default target is:

$$\mu^* = E[\beta(X)]$$

This is the average effect of the treatment on the Gumbel location parameter $\mu$. Because
the model uses an identity link, $\mu^*$ is also the average shift in $E[Y]$ (the mean of
the Gumbel distribution shifts by exactly $\mu^*$ per unit of $T$, regardless of scale).

## Influence Function

The per-observation influence function contribution is:

$$\psi_i = H(\hat\theta(X_i)) - \begin{bmatrix} 0 & 1 \end{bmatrix} \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i))$$

where:

- $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation plug-in (the fitted location
  slope), and $[0, 1]$ is the unit selector picking the $\beta$ component.
- $\ell_\theta(Y_i, T_i, \hat\theta(X_i)) = -(1/\sigma)(1 - e^{-z_i}) \cdot [1, T_i]^\top$ is the
  score at observation $i$.
- $\Lambda(X_i) = E[\nabla^2_\theta \ell \mid X = X_i]$ is the expected Hessian (estimated on
  a held-out fold).
- The correction is subtracted (minus sign), which removes the regularization bias. A plus
  sign would double the bias and is wrong.

The point estimate is $\hat\mu = (1/n) \sum_i \psi_i$, and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

**Regime C (three-way split).** The Hessian weight $w = e^{-z}/\sigma^2$ depends on both
$\theta$ (through $\mu$) and $y$ (through $z = (y-\mu)/\sigma$). Neither the structural
model class nor a closed-form expression for $E[H \mid X]$ removes this joint dependence.
Lambda must be estimated from a held-out fold.

The three-way split:

1. Use fold A to train the structural network $\hat{\theta}(X)$.
2. Use fold B to estimate $\hat{\Lambda}(X)$ by fitting a secondary model that predicts
   per-observation Hessian entries given $X$.
3. Use fold C to assemble $\psi_i$ using $\hat{\theta}$ and $\hat{\Lambda}$ from the
   other folds.

This rotation is performed across all cross-fitting folds (with `n_folds` controlling
granularity). The package detects the regime automatically from
`GumbelFamily.hessian_depends_on_theta()`.

## Usage

```python
import numpy as np
from deep_inference import structural_dml

np.random.seed(42)
n = 2000
X = np.random.randn(n, 5)
T = np.random.randn(n)

# True heterogeneous parameters
alpha_true = 5.0 + 0.5 * X[:, 0]
beta_true = 1.0 + 0.3 * X[:, 0]     # E[beta(X)] = 1.0 (true target)
sigma = 1.0

# Generate Gumbel outcomes: mu = alpha + beta * T
mu = alpha_true + beta_true * T
Y = np.random.gumbel(mu, sigma)

print(f"True mu* = {beta_true.mean():.4f}")
print(f"Outcome range: [{Y.min():.2f}, {Y.max():.2f}]")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='gumbel',
    scale=1.0,              # fixed scale (not learned)
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)
print(result.summary())

# Diagnostics
print(f"Min lambda eigenvalue: {result.diagnostics.get('min_lambda_eigenvalue', 'N/A')}")
```

To model minima rather than maxima, negate $Y$ before fitting:

```python
result = structural_dml(-Y, T, X, family='gumbel', scale=1.0)
# The estimated mu* is the average treatment effect on the location of -Y,
# which equals minus the effect on the location of the minimum.
```

## Evidence

From [Eval 01: Parameter Recovery](../simulation_studies/eval_01.md) (n=5000, seed=42):

| Family | RMSE($\alpha$) | RMSE($\beta$) | Corr($\alpha$) | Corr($\beta$) | Status |
|--------|---------------|--------------|----------------|---------------|--------|
| gumbel | 0.063 | 0.063 | 0.975 | 0.999 | PASS |

The network achieves a correlation of 0.999 with the oracle $\beta(X)$ function across
covariates.

Coverage benchmarking (FLM[cholesky] vs Oracle-MLE) has not yet been run for the Gumbel
family. The family requires a custom Gumbel MLE oracle (not a standard GLM) and is listed
as Tier B, pending that infrastructure. See [Replications](../replications/index.md).

## Diagnostics and Pitfalls

**Scale $\sigma$ is not learned.** The network learns $[\alpha(X), \beta(X)]$ but not
$\sigma$. The scale affects the magnitude of the Hessian weight $w = e^{-z}/\sigma^2$ and
therefore the confidence interval width. Misspecifying $\sigma$ inflates or deflates the
standard error. Use domain knowledge or estimate $\sigma$ from residuals in a preliminary
OLS pass before passing it to this package.

**High curvature from below-location observations.** The Hessian weight $w = e^{-z}/\sigma^2$
grows rapidly when $y \ll \mu$ (outcome far below the fitted location). A small number of
such observations can dominate the Lambda estimate and make it noisy or numerically
unstable. Check `result.diagnostics['min_lambda_eigenvalue']`. If it is near zero,
consider whether your data contains extreme lower-tail outliers, or try a larger $\sigma$
(which shrinks $w$ uniformly).

**Asymmetric score.** The score $-(1/\sigma)(1 - e^{-z})$ is bounded from below at
$-1/\sigma$ (as $z \to +\infty$) but grows without bound as $z \to -\infty$. This means
the influence function correction is dominated by observations where $y \ll \mu$. If your
data has extreme left-tail values, they will have outsized influence on the estimate.

**Identity vs log link.** Unlike Weibull or Poisson, the Gumbel uses an identity link.
The target $\mu^* = E[\beta(X)]$ is already on the outcome scale (units of $Y$). There
is no need to exponentiate the estimate.

**Modeling minima.** If you want to model the minimum of a large pool, negate $Y$. The
minimum of a set of IID variables follows a Gumbel distribution for the minimum (which
is just the reflection of the maximum distribution). By fitting on $-Y$, the model
estimates the treatment effect on the location of $-Y$, and you recover the effect on
the minimum of $Y$ by negating the result.

**No zeros or negatives needed.** Unlike Weibull, Gumbel places no constraint on the
sign of $Y$. The Gumbel distribution is supported on all of $\mathbb{R}$.

## References and API

- [References](../references/index.md) covers FLM 2021 and FLM 2025 in full.
- [API: Families](../api/families.md) documents `GumbelFamily` and its constructor
  arguments (`scale`, and all inherited arguments).
- Source:
  `src/deep_inference/families/gumbel.py`
