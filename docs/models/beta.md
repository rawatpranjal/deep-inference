# Beta Model

The Beta model handles outcomes that are rates, proportions, or shares bounded strictly
in the open interval $(0, 1)$. The treatment shifts the logit of the mean proportion
heterogeneously across covariates, and the influence function delivers a valid confidence
interval for the average shift across the covariate distribution.

## Source Papers

The structural inference framework is Farrell, Liang, and Misra (2021, 2025). See
[References](../references/index.md) for the full annotated bibliography. Beta regression
as a parametric model for proportions is introduced in Ferrari and Cribari-Neto (2004),
which is cited in the source file. The precision-parametrized Beta is the standard form
used in `betareg` in Stata and R.

## When to Use

Use the Beta model when:

- The outcome is a rate, proportion, or share bounded strictly between 0 and 1.
- The outcome is continuously distributed in that interval (not just 0 and 1).
- You want to model the heterogeneous effect of a treatment on the mean proportion.

Concrete examples:

- Market share: what fraction of category spending goes to brand $j$, as a function of
  price (treatment) and household characteristics (covariates).
- Test pass rates: what fraction of students in a school pass an exam, as a function of
  school resources (treatment) and demographics (covariates).
- Portfolio allocation: what fraction of wealth a household allocates to equities, as a
  function of income shocks (treatment) and risk preferences (covariates).
- Utilization rate: what fraction of available capacity is used, as a function of
  demand pressure (treatment) and operational characteristics (covariates).

**Critical distinctions:**

- Beta is NOT for binary (0 or 1 only) outcomes. If each observation is either 0 or 1,
  use logit or probit. Beta is for proportions that take a continuum of values in $(0, 1)$.
- Beta is NOT for outcomes that can equal exactly 0 or exactly 1. The Beta density is
  zero at the endpoints. If your data has exact zeros or ones (for example, a market share
  of exactly 0 for a non-entrant), a zero-one-inflated extension is needed.
- Beta is NOT for proportions computed from a small number of trials. If your outcome is
  the fraction of successes in 5 trials, it takes only 6 possible values (0/5, 1/5, ...,
  5/5), and binomial regression is more appropriate.
- The logit link of Beta gives a coefficient that is on the logit (log-odds) scale, not
  the probability scale. This is the same scale as the logit model for binary outcomes.

## Model

The data-generating process is:

$$Y \sim \text{Beta}(\mu\phi,\; (1-\mu)\phi), \quad \mu = \sigma(\eta), \quad \eta = \alpha(X) + \beta(X) \cdot T$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic sigmoid.

The neural network learns two functions:

- $\alpha(X)$: the baseline logit of the mean proportion.
- $\beta(X)$: the treatment effect on the logit of the mean proportion.

So `theta_dim = 2` and $\theta(X) = [\alpha(X), \beta(X)]$.

The precision parameter $\phi > 0$ is fixed at initialization (default `precision=1.0`)
and is not learned. You pass it as a constructor argument. The variance of $Y$ given $\mu$
is:

$$\text{Var}(Y) = \frac{\mu(1-\mu)}{1 + \phi}$$

Higher $\phi$ means lower variance (observations cluster tightly around $\mu$). At
$\phi = 1$, the variance equals $\mu(1-\mu)/2$, which is half the maximum variance of a
Bernoulli with the same mean.

The two shape parameters of the Beta distribution are $a = \mu\phi$ and $b = (1-\mu)\phi$.
The distribution concentrates near $\mu$ as $\phi$ grows.

## Loss / Likelihood

Let $a = \mu\phi$ and $b = (1-\mu)\phi$. The Beta negative log-likelihood (up to the
constant $\log\Gamma(\phi)$ that does not depend on $\theta$) is:

$$\ell(y, \theta) = \log\Gamma(a) + \log\Gamma(b) - (a-1)\log y - (b-1)\log(1-y)$$

Substituting $a = \mu\phi$ and $b = (1-\mu)\phi$ and $\mu = \sigma(\eta)$:

$$\ell(y, \theta) = \log\Gamma(\mu\phi) + \log\Gamma((1-\mu)\phi) - (\mu\phi - 1)\log y - ((1-\mu)\phi - 1)\log(1-y)$$

Both $\log y$ and $\log(1-y)$ appear in the loss, so outcomes near 0 or 1 produce very
large (negative or positive) contributions. This is why exact zeros and ones are excluded.

Source: `BetaFamily.loss` in
`src/deep_inference/families/beta.py`.

## Score and Hessian

Let $\psi(\cdot)$ denote the digamma function $\psi(z) = d\log\Gamma(z)/dz$.

The gradient of the loss with respect to the linear predictor $\eta$ is:

$$\frac{\partial \ell}{\partial \eta} = \phi \cdot [\psi(a) - \psi(b) - \log y + \log(1-y)] \cdot \mu(1-\mu)$$

The factor $\mu(1-\mu) = \sigma(\eta)(1-\sigma(\eta))$ is the derivative of the sigmoid,
connecting the loss to the linear predictor through the chain rule. The factor
$[\psi(a) - \psi(b) - \log y + \log(1-y)]$ is the difference between the digamma-based
expected log-ratio and the observed log-ratio.

The score with respect to $\theta = [\alpha, \beta]$ is:

$$\nabla_\theta \ell = \frac{\partial \ell}{\partial \eta} \cdot \begin{bmatrix} 1 \\ t \end{bmatrix}$$

This closed-form gradient is implemented in `BetaFamily.gradient` using
`torch.digamma`.

The Hessian is computed via **autodiff**. The `hessian()` method returns `None`, signaling
the engine to use PyTorch autograd. The analytic Hessian involves the trigamma function
$\psi'(\cdot) = d^2\log\Gamma/dz^2$ and is more complex to implement reliably than the
gradient. Autodiff handles it correctly.

**Key structural facts:**

- The Hessian depends on $\theta$ through $\mu = \sigma(\alpha + \beta T)$.
- The Hessian also depends on $y$ through $\log y$ and $\log(1-y)$ in the gradient, whose
  derivatives with respect to $\theta$ appear in the Hessian.
- However, under the Beta distribution the expected Hessian $E[H \mid X, T]$ is a
  deterministic function of $\theta$ (because $E[\log Y]$ and $E[\log(1-Y)]$ under
  Beta$(a, b)$ are $\psi(a) - \psi(a+b)$ and $\psi(b) - \psi(a+b)$, respectively, which
  depend only on $\mu$ and $\phi$). This means the oracle Lambda is deterministic given
  $\theta$, but since $\theta$ is not known it must still be estimated.
- `BetaFamily.hessian_depends_on_theta()` returns `True`, which puts the model in Regime C.

## Target

The default target is:

$$\mu^* = E[\beta(X)]$$

This is the average treatment effect on the **logit** of the mean proportion. An increase
of $\beta$ by 1 unit corresponds to a multiplicative change of $e^\beta$ in the odds
$\mu/(1-\mu)$. For small effects and $\mu$ near 0.5, $\mu^*$ is approximately the average
change in the proportion itself (since the logit derivative at 0.5 is 1).

To report on the probability scale, you would use a custom target functional. The `ame`
target (average marginal effect) is available in the `inference()` API with `model='logit'`
for binary outcomes, but Beta does not yet expose a built-in AME target through
`structural_dml`. Use a custom `target_fn` for this.

## Influence Function

The per-observation influence function contribution is:

$$\psi_i = H(\hat\theta(X_i)) - \begin{bmatrix} 0 & 1 \end{bmatrix} \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i))$$

where:

- $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation plug-in (the fitted logit slope),
  and $[0, 1]$ is the unit selector picking the $\beta$ component.
- $\ell_\theta(Y_i, T_i, \hat\theta(X_i)) = (\partial\ell/\partial\eta)_i \cdot [1, T_i]^\top$ is
  the score, with $\partial\ell/\partial\eta$ involving the digamma function (computed in closed
  form).
- $\Lambda(X_i) = E[\nabla^2_\theta \ell \mid X = X_i]$ is the $2 \times 2$ expected Hessian
  (estimated on a held-out fold via autodiff).
- The correction is subtracted (minus sign), which removes the regularization bias. A plus
  sign would double the bias and is wrong.

The point estimate is $\hat\mu = (1/n) \sum_i \psi_i$, and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

**Regime C (three-way split).** The Hessian depends on $\theta$ (through
$\mu = \sigma(\alpha + \beta T)$), so Lambda must be estimated rather than computed
analytically.

The three-way split:

1. Use fold A to train the structural network $\hat{\theta}(X) = [\hat\alpha(X), \hat\beta(X)]$.
2. Use fold B to estimate $\hat{\Lambda}(X)$ by fitting a secondary model on the
   per-observation Hessian entries (computed via autodiff at $\hat\theta$ from fold A).
3. Use fold C to assemble $\psi_i$ using $\hat\theta$ and $\hat\Lambda$ from the
   other folds.

The package detects the regime automatically from
`BetaFamily.hessian_depends_on_theta()`.

## Usage

```python
import numpy as np
from deep_inference import structural_dml
from scipy.stats import beta as beta_dist
from scipy.special import expit

np.random.seed(42)
n = 2000
X = np.random.randn(n, 5)
T = np.random.randn(n)

# True heterogeneous parameters (on the logit scale)
alpha_true = 0.0 + 0.5 * X[:, 0]
beta_true  = 0.3 + 0.2 * X[:, 0]    # E[beta(X)] = 0.3 (true target)
phi = 5.0

# Mean on the probability scale
mu_true = expit(alpha_true + beta_true * T)

# Generate Beta outcomes with shape parameters a = mu*phi, b = (1-mu)*phi
a = mu_true * phi
b = (1 - mu_true) * phi
Y = beta_dist.rvs(a, b)

print(f"True mu* = {beta_true.mean():.4f}   (logit scale)")
print(f"Outcome range: [{Y.min():.4f}, {Y.max():.4f}]")
print(f"Any exact zeros: {(Y == 0).any()},  Any exact ones: {(Y == 1).any()}")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='beta',
    precision=phi,          # fixed precision phi (not learned)
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)
print(result.summary())

# Diagnostics
print(f"Min lambda eigenvalue: {result.diagnostics.get('min_lambda_eigenvalue', 'N/A')}")
```

To use the default precision ($\phi = 1$):

```python
result = structural_dml(Y, T, X, family='beta')
```

To inspect the fitted mean proportions:

```python
from scipy.special import expit
eta_hat = result.theta_hat[:, 0] + result.theta_hat[:, 1] * T
mu_hat  = expit(eta_hat)
print(f"Fitted mean proportion: {mu_hat.mean():.4f}")
```

## Evidence

The Beta family has been benchmarked on a Monte Carlo study at M=50 replications. From
[Replications](../replications/index.md):

| Method | Coverage | Status |
|--------|----------|--------|
| FLM[cholesky] | 96% (M=50) | PASS |
| RieszNet | 100% (M=50) | PASS |

FLM[cholesky] achieves 96% coverage and RieszNet achieves 100% coverage at M=50. Beta
passes despite having a $y$-dependent Hessian weight on a continuous outcome. The
replications index notes this as one of the falsification results: "beta passes despite a
y-dependent Hessian on a continuous outcome (neither is sufficient to cause the failure)."

Parameter recovery has not been reported for Beta in [Eval 01](../simulation_studies/eval_01.md),
which covers nine other families. The coverage benchmark above is the primary evidence of
correct implementation.

## Diagnostics and Pitfalls

**Outcomes must be strictly in $(0, 1)$.** The loss involves $\log y$ and $\log(1-y)$,
which diverge at 0 and 1. The source clamps $y$ to $[10^{-7}, 1 - 10^{-7}]$ internally,
but if many observations fall exactly at 0 or 1, the model is mis-specified and results
are unreliable. Check with `(Y == 0).any()` and `(Y == 1).any()`.

**Precision $\phi$ is not learned.** The network learns $[\alpha(X), \beta(X)]$ but not
$\phi$. Misspecifying $\phi$ affects the curvature of the loss and therefore the magnitude
of the Lambda matrix and the confidence intervals. The point estimate of $\mu^*$ is
relatively robust to $\phi$ misspecification, but the standard error is not.

To choose $\phi$: start with $\phi = 5$ or $\phi = 10$ for typical proportions, and
verify by checking that `result.theta_hat[:, 1].std()` is not trivially small (which
would indicate the network is not detecting heterogeneity). You can also compare
out-of-sample NLL across several $\phi$ values on a validation set.

**Digamma numerical stability.** The gradient uses `torch.digamma(a)` and
`torch.digamma(b)` where $a = \mu\phi$ and $b = (1-\mu)\phi$. The digamma function is
numerically unstable for very small arguments (near zero). This occurs when $\phi$ is
small AND $\mu$ is near 0 or 1. A symptom is exploding gradients during training. Raise
$\phi$ or clip $\mu$ further from the boundary.

**Hessian via autodiff.** Unlike Weibull or Gumbel, the Hessian is not available in
closed form and is computed via PyTorch autograd. This is slower but more robust. The
autodiff Hessian is computed at the current $\hat\theta$ values on the Lambda estimation
fold, not analytically.

**Logit-scale interpretation.** The target $\mu^* = E[\beta(X)]$ is on the logit scale.
A coefficient of 0.3 does NOT mean a 30 percentage-point change in the proportion. For
example, if the baseline is $\mu = 0.5$ ($\alpha = 0$), then $\beta = 0.3$ shifts the
logit from 0 to 0.3, which corresponds to a shift in the mean proportion from 0.5 to
$\sigma(0.3) \approx 0.574$, a change of about 7.4 percentage points.

**Confusion with logit.** Beta and logit both use a logit link and both produce a
coefficient $\beta$ that is on the log-odds scale. The difference is the outcome
distribution. Logit assumes $Y \in \{0, 1\}$ (Bernoulli). Beta assumes $Y \in (0, 1)$
continuously. Using logit on a continuous proportion misspecifies the loss; using Beta on
a binary outcome also misspecifies the loss. Make sure you know which case applies.

**Very high or very low mean proportion.** If $\mu$ is near 0 or 1 for most observations,
one of $a = \mu\phi$ or $b = (1-\mu)\phi$ is near zero. The digamma function is large
in magnitude near zero, making the gradient noisy. Consider centering or rescaling the
outcome, or using a different precision.

## References and API

- [References](../references/index.md) covers FLM 2021, FLM 2025, and Ferrari and
  Cribari-Neto (2004) in full.
- [API: Families](../api/families.md) documents `BetaFamily` and its constructor
  arguments (`precision`, `target`, and all inherited arguments).
- Source:
  `src/deep_inference/families/beta.py`
