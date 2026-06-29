# Tobit Model

The Tobit model handles continuous outcomes that are censored at a boundary, typically
zero. The data has a point mass at the boundary and a continuous distribution above it.
The model separates the probability of crossing the boundary from the magnitude of the
uncensored outcome, and the influence function delivers a valid confidence interval for
the average treatment effect on the latent (uncensored) scale.

## Source Papers

The structural inference framework is Farrell, Liang, and Misra (2021, 2025). See
[References](../references/index.md) for the full annotated bibliography. The censored
regression (Tobit) model originates in Tobin (1958) and is covered in Amemiya (1984)
and most graduate econometrics textbooks.

## When to Use

Use the Tobit model when:

- The outcome is continuous and left-censored at a lower bound (most commonly zero).
- Many observations pile up at the lower bound, but those above it take a range of
  positive values.
- You believe the censoring reflects a corner solution: the individual would like to
  choose a negative value but is constrained to zero.

Concrete examples:

- Labor supply: hours worked per week, which cannot be negative. Non-working individuals
  are assigned zero, but their desired hours might be negative (they would pay not to
  work if they could). The latent $Y^*$ captures desired hours.
- Charitable donations: amount donated in a solicitation. Many households give nothing
  (censored at zero), but those who give take continuous positive values.
- Durable goods expenditure: spending on a car in a given year. Many households spend
  nothing (censored), while buyers spend varying amounts.

**Key distinctions:**

- Tobit is NOT for binary (0/1) outcomes. If your outcome is only 0 or 1, use logit or
  probit. Tobit requires that the uncensored observations take a range of continuous
  values above the boundary.
- Tobit is NOT for non-negative outcomes where zero is a natural observed value (like
  count data). If zero is a genuine observation (not a censored negative), use Poisson,
  Negative Binomial, or a zero-inflated model.
- The package currently implements left-censoring at zero. For right-censoring or
  two-sided censoring, see the Diagnostics section below.

## Model

The latent data-generating process is:

$$Y^* = \alpha(X) + \beta(X) \cdot T + \sigma(X) \cdot \varepsilon, \quad \varepsilon \sim N(0, 1)$$

The observed outcome is:

$$Y = \max(0, Y^*)$$

The neural network learns three functions:

- $\alpha(X)$: the baseline mean of the latent outcome.
- $\beta(X)$: the treatment effect on the latent outcome.
- $\gamma(X)$: the log-scale of the residual standard deviation.

So `theta_dim = 3` and $\theta(X) = [\alpha(X), \beta(X), \gamma(X)]$, where
$\sigma(X) = \exp(\gamma(X))$ keeps $\sigma$ strictly positive and unconstrained during
optimization.

The probability that an observation is uncensored (i.e., $Y > 0$) is:

$$P(Y > 0 \mid X, T) = \Phi\!\left(\frac{\mu}{\sigma}\right), \quad \mu = \alpha(X) + \beta(X) \cdot T$$

where $\Phi$ is the standard normal CDF.

## Loss / Likelihood

The Tobit likelihood has two cases. Define $\mu = \alpha + \beta T$ and $\sigma = e^\gamma$.

For **uncensored** observations ($Y > 0$), the observation contributes the normal density:

$$\ell_{\text{uncensored}} = \gamma + \frac{(Y - \mu)^2}{2\sigma^2}$$

This is the standard Gaussian NLL (without the constant $\frac{1}{2}\log(2\pi)$).

For **censored** observations ($Y = 0$), the observation contributes the probability of
being at or below zero:

$$\ell_{\text{censored}} = -\log \Phi\!\left(-\frac{\mu}{\sigma}\right)$$

The combined loss for observation $i$ is:

$$\ell_i = \mathbf{1}[Y_i = 0] \cdot \ell_{\text{censored}} + \mathbf{1}[Y_i > 0] \cdot \ell_{\text{uncensored}}$$

Source: `TobitFamily.loss` in
`src/deep_inference/families/tobit.py`.

## Score and Hessian

Both the score and the Hessian are computed via **autodiff** (PyTorch autograd). The
`TobitFamily` class does not define a closed-form `gradient` or `hessian` method; both
return `None`, which signals the engine to fall back to automatic differentiation.

The analytic gradient is well-known from the censored regression literature. For uncensored
observations, the gradient with respect to $\theta = [\alpha, \beta, \gamma]$ involves:

$$\nabla_\theta \ell_{\text{uncensored}} \propto \left[-\frac{Y - \mu}{\sigma^2}, \; -\frac{(Y-\mu)T}{\sigma^2}, \; 1 - \frac{(Y-\mu)^2}{\sigma^2}\right]$$

For censored observations, the gradient involves the inverse Mills ratio
$\lambda_M = \phi(z)/\Phi(-z)$ where $z = \mu/\sigma$ and $\phi$ is the standard normal PDF:

$$\nabla_\theta \ell_{\text{censored}} \propto \left[\frac{\lambda_M}{\sigma}, \; \frac{\lambda_M T}{\sigma}, \; \frac{z \lambda_M}{1}\right]$$

These expressions are intricate because they combine different forms across censored and
uncensored observations, with the boundary determined by the data. The package uses
autodiff to avoid hand-coding errors in these mixed-case derivatives.

**Key structural facts:**

- The Hessian depends on $\theta$ through $\mu = \alpha + \beta T$ and $\sigma = e^\gamma$.
- The Hessian depends on $y$ through the censoring indicator $\mathbf{1}[Y = 0]$ and
  through the residual $(Y - \mu)$ in the uncensored term.
- Both dependencies mean Lambda must be estimated from data (Regime C).
- `TobitFamily.hessian_depends_on_theta()` returns `True`.

## Target

Two targets are available.

**Latent effect (default):**

$$\mu^*_{\text{latent}} = E[\beta(X)]$$

This is the average partial effect of $T$ on the latent outcome $Y^*$. It answers the
question: on average, how much does $T$ shift the desired level of $Y^*$ across the
population?

**Observed effect:**

$$\mu^*_{\text{obs}} = E\!\left[\beta(X) \cdot \Phi\!\left(\frac{\mu(X)}{\sigma(X)}\right)\right]$$

This is the average partial effect on the observed (censored) outcome $E[Y \mid X, T]$.
It weights the latent effect $\beta(X)$ by the probability of being uncensored at each
covariate value. Use this target when you care about the effect on the censored outcome
itself (for example, the average change in observed spending, not in desired spending).

Select the target by passing `target='latent'` (default) or `target='observed'` to
`structural_dml`.

## Influence Function

For the latent target, the per-observation influence function contribution is:

$$\psi_i = H(\hat\theta(X_i)) - \begin{bmatrix} 0 & 1 & 0 \end{bmatrix} \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i))$$

where:

- $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation plug-in (the fitted latent
  slope), and $[0, 1, 0]$ is the unit selector picking the $\beta$ component from the
  three-dimensional $\theta = [\alpha, \beta, \gamma]$.
- $\ell_\theta(Y_i, T_i, \hat\theta(X_i))$ is the $3$-dimensional score vector at observation
  $i$ (computed via autodiff).
- $\Lambda(X_i) = E[\nabla^2_\theta \ell \mid X = X_i]$ is the $3 \times 3$ expected Hessian
  (estimated on a held-out fold, also via autodiff for each per-obs Hessian entry).
- The correction is subtracted (minus sign), which removes the regularization bias. A plus
  sign would double the bias and is wrong.

The point estimate is $\hat\mu = (1/n) \sum_i \psi_i$, and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

For the observed target, the gradient $\nabla_\theta h$ is more complex (it involves
$\Phi(z)$ and $\phi(z)$ derivatives) and is also computed analytically; see
`TobitFamily.per_obs_target_gradient`.

## Algorithm and Regime

**Regime C (three-way split).** The Hessian depends on both $\theta$ (through $\mu$ and
$\sigma$) and $y$ (through the censoring indicator and the residual). Lambda cannot be
derived analytically and must be estimated from data.

The three-way split:

1. Use fold A to train the structural network $\hat{\theta}(X) = [\hat\alpha(X), \hat\beta(X), \hat\gamma(X)]$.
2. Use fold B to estimate $\hat{\Lambda}(X)$ by fitting a secondary model on the
   per-observation Hessian entries (computed via autodiff at the $\hat\theta$ from fold A).
3. Use fold C to assemble $\psi_i$ using $\hat\theta$ and $\hat\Lambda$ from the
   other folds.

With `theta_dim = 3`, the Lambda matrix is $3 \times 3$, requiring fitting 6 unique
entries per observation. This is more expensive than the $2 \times 2$ case for families
with `theta_dim = 2`. Use at least `n_folds = 50` to ensure stable Lambda estimates with
this many entries.

## Usage

**Latent effect (default):**

```python
import numpy as np
from deep_inference import structural_dml

np.random.seed(42)
n = 2000
X = np.random.randn(n, 5)
T = np.random.randn(n)

# True heterogeneous parameters
alpha_true = 0.5 + 0.3 * X[:, 0]
beta_true = 0.3 + 0.2 * X[:, 0]    # E[beta(X)] = 0.3 (true latent target)
sigma = 1.0

# Generate Tobit data: Y = max(0, Y*)
Y_star = alpha_true + beta_true * T + sigma * np.random.randn(n)
Y = np.maximum(0, Y_star)

censored_rate = (Y == 0).mean()
print(f"True mu* (latent) = {beta_true.mean():.4f}")
print(f"Censored at 0: {censored_rate * 100:.1f}%")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='tobit',             # default target is 'latent'
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)
print(result.summary())
```

**Observed effect:**

```python
result_obs = structural_dml(
    Y=Y, T=T, X=X,
    family='tobit',
    target='observed',          # effect on observed E[Y], not latent Y*
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)
print(result_obs.summary())
```

**Inspecting the three learned parameters:**

```python
# theta_hat is (n, 3): columns are [alpha, beta, gamma]
alpha_hat = result.theta_hat[:, 0]
beta_hat  = result.theta_hat[:, 1]
gamma_hat = result.theta_hat[:, 2]
sigma_hat = np.exp(gamma_hat)

print(f"Mean sigma_hat: {sigma_hat.mean():.4f}  (true sigma = {sigma:.4f})")
print(f"Mean beta_hat:  {beta_hat.mean():.4f}   (true E[beta] = {beta_true.mean():.4f})")
```

## Evidence

From [Eval 01: Parameter Recovery](../simulation_studies/eval_01.md) (n=5000, seed=42):

| Family | RMSE($\alpha$) | RMSE($\beta$) | Corr($\alpha$) | Corr($\beta$) | Status |
|--------|---------------|--------------|----------------|---------------|--------|
| tobit | 0.042 | 0.024 | 0.999 | 0.998 | PASS |

The network achieves a correlation of 0.998 with the oracle $\beta(X)$ function across
covariates, and 0.999 for $\alpha(X)$.

Coverage benchmarking (FLM[cholesky] vs Oracle-MLE) has not yet been run for the Tobit
family. The family requires a custom Tobit MLE oracle (not a standard GLM) and is listed
as Tier B, pending that infrastructure. See [Replications](../replications/index.md).

## Diagnostics and Pitfalls

**Check the censoring rate.** Very high censoring (above 70-80%) leaves little uncensored
variation for the network to learn $\beta(X)$ from. Very low censoring (below 5%) means
almost no one hits the boundary, and the Tobit model is nearly equivalent to a Gaussian
with little value added. Check with `(Y == 0).mean()` before fitting.

**$\sigma(X) = \exp(\gamma(X))$ must stay in a reasonable range.** After fitting, inspect
`np.exp(result.theta_hat[:, 2])`. Values near zero (very small $\sigma$) indicate the
network is driving the variance to zero, which is implausible for most real data. Values
that are enormous indicate the network is failing to fit the location and compensating with
a large variance. Both suggest underfitting or a misconfigured learning rate.

**Inverse Mills ratio numerical instability.** The loss for censored observations contains
$-\log \Phi(-\mu/\sigma)$. When $\mu/\sigma$ is very large and positive (nearly all
probability mass above zero), $\Phi(-\mu/\sigma)$ approaches zero and its log diverges.
The source clamps $\Phi(-z)$ from below at $10^{-10}$. If a large fraction of
observations approach this boundary, the gradient signal for censored observations is
unreliable.

**theta_dim = 3 and the 3x3 Lambda.** The $3 \times 3$ Lambda matrix has 6 unique entries
to estimate per observation (by symmetry). This is three times as many as a 2-parameter
family. You need more data per fold to estimate it stably. Use `n_folds >= 50` (the
default) and `n > 1000`.

**Autodiff for score and Hessian.** Unlike Weibull or Gumbel, there is no closed-form
gradient or Hessian in the package. Both are computed via PyTorch autograd. This is
slightly slower but means the package automatically handles the correct derivative of the
piecewise loss (the gradient switches form at $Y = 0$).

**Right-censoring.** The package implements only left-censoring at zero. For right-censored
survival data (where you observe a minimum of the event time and a censoring time, and
some observations are censored from above), this family is not directly applicable. A
common workaround: for data censored above at a known constant $c$, you can reframe the
left-censoring model by fitting on $c - Y$ (reflecting the data). For general
right-censoring with an event indicator, a custom loss function passed via `loss_fn`
is the right approach.

**Two-sided censoring.** Data censored both from above and below (such as top-coded income
surveys) requires a custom extension not available in this family.

**Latent vs observed target.** The latent target $E[\beta(X)]$ answers "what would be the
average effect if no one were censored?" The observed target $E[\beta(X) \Phi(\mu/\sigma)]$
answers "what is the average effect on what we actually see?" Both are valid research
questions, but they differ numerically whenever any censoring occurs.

## References and API

- [References](../references/index.md) covers FLM 2021 and FLM 2025 in full.
- [API: Families](../api/families.md) documents `TobitFamily` and its constructor
  arguments (`target`, and all inherited arguments).
- Source:
  `src/deep_inference/families/tobit.py`
