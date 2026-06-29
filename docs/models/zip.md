# Zero-Inflated Poisson Model

The Zero-Inflated Poisson (ZIP) model is for count outcomes that have far more zeros than a plain
Poisson rate can produce. It treats the data as a mixture of two processes. With some probability a
unit is a structural zero (it can never have a positive count), and otherwise the count comes from
an ordinary Poisson process. This page covers the model, its mixture likelihood, the two target
choices it supports, and how its inference machinery is wired. It is honest that coverage has not
yet been benchmarked for this family in this repository.

## Source Papers

The estimator and its inference theory come from Farrell, Liang, and Misra, whose general
structural framework lets a neural network learn the heterogeneity $\theta(X)$ while an
influence-function correction restores valid confidence intervals. The annotated reading list is in
the [References](../references/index.md) section. The two framework papers are Farrell, Liang, Misra
(2021), "Deep Neural Networks for Estimation and Inference" (Econometrica), and the extended
Farrell, Liang, Misra (2025) working paper. The zero-inflation parameterization itself follows
Lambert (1992), the standard reference for ZIP regression (it is the model behind Stata's `zip`
command).

## When to Use

Use the Zero-Inflated Poisson model when the outcome is a count and there are many more zeros than
a single Poisson rate would generate. The tell is that even after fitting a Poisson, the model
predicts far fewer zeros than you observe. Three concrete cases.

- Number of doctor visits in a year, where a large share of people never visit at all (a structural
  zero) while the rest follow a count process.
- Number of insurance claims per policy, where many policyholders never file.
- Number of purchases of a product, where most of the population is simply not in the market.

If the excess spread is a heavy right tail rather than a glut of zeros, prefer the
[Negative Binomial model](negbin.md). If the zero count is consistent with a single Poisson rate,
the plain [Poisson model](poisson.md) is enough.

## Model

Each observation is a mixture. With probability $\pi$ the unit is a structural zero. With
probability $1 - \pi$ the count is drawn from a Poisson with rate $\lambda$. Both the rate and the
zero-inflation probability depend on the covariates $X$ and the treatment $T$.

$$\lambda = \exp(\eta_\lambda), \qquad \eta_\lambda = \alpha(X) + \beta(X)\cdot T$$

$$\pi = \sigma(\eta_\pi), \qquad \eta_\pi = \gamma(X) + \delta(X)\cdot T$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the logistic function. The count distribution is

$$P(Y = 0) = \pi + (1 - \pi)\,e^{-\lambda}, \qquad P(Y = k) = (1 - \pi)\,\frac{\lambda^k e^{-\lambda}}{k!} \;\text{ for } k > 0.$$

The network learns four functions at once, $\theta(X) = (\alpha(X), \beta(X), \gamma(X), \delta(X))$,
so $\texttt{theta\_dim} = 4$. The four parameters are, in order, the baseline log-rate $\alpha(X)$,
the treatment effect on the log-rate $\beta(X)$, the baseline logit of the zero-inflation
probability $\gamma(X)$, and the treatment effect on that logit $\delta(X)$.

## Loss / Likelihood

The loss is the Zero-Inflated Poisson negative log-likelihood. It splits on whether the observed
count is zero.

For $Y = 0$:

$$L = -\log\!\big(\pi + (1 - \pi)\,e^{-\lambda}\big).$$

For $Y > 0$:

$$L = -\log(1 - \pi) + \lambda - Y\log\lambda + \log\Gamma(Y + 1).$$

The implementation clamps $\eta_\lambda$ to $[-20, 10]$ before exponentiating, clamps $\pi$ to
$[10^{-7}, 1 - 10^{-7}]$, and floors the zero-probability at $10^{-10}$ inside the logarithm, all
for numerical safety.

## Score and Hessian

Unlike Poisson, the Negative Binomial, or Gamma, the ZIP family does not ship closed-form
derivatives. Its `gradient` and `hessian` methods return `None`, which tells the package to compute
the score and the Hessian by automatic differentiation of the loss above. The source comment is
explicit that this is a mixture model with complex cross-derivatives, so autodiff is both cleaner
and equally accurate. The package declares that the Hessian depends on $\theta$, through both
$\lambda$ and $\pi$, and it also depends on $Y$ through the zero indicator that selects which branch
of the loss applies. The residual used for diagnostics is $Y - E[Y]$, where the ZIP mean is
$E[Y] = (1 - \pi)\,\lambda$.

## Target

The family supports two estimands, selected by the `target` argument on the `ZIPFamily` class.

- `target='beta'` (the default). The average treatment effect on the log-rate,
  $\mu^* = E[\beta(X)]$. The per-observation target is $m(\theta) = \beta$, with Jacobian
  $H_\theta = [0, 1, 0, 0]$ over $\theta = (\alpha, \beta, \gamma, \delta)$.
- `target='delta'`. The average treatment effect on the logit of the zero-inflation probability,
  $\mu^* = E[\delta(X)]$. The per-observation target is $m(\theta) = \delta$, with Jacobian
  $H_\theta = [0, 0, 0, 1]$.

## Influence Function

For either target the per-observation influence function has the same shape.

$$\psi_i = m(\hat\theta(X_i)) - H_\theta(X_i)^\top \, \Lambda(X_i)^{-1} \, \ell_\theta(Y_i, T_i, \hat\theta(X_i)),$$

where the plug-in $m(\hat\theta(X_i))$ is $\hat\beta(X_i)$ or $\hat\delta(X_i)$ depending on the
chosen target, $H_\theta$ is the corresponding constant unit selector ($[0, 1, 0, 0]$ for
$\beta$, $[0, 0, 0, 1]$ for $\delta$), $\Lambda(X) = E[\nabla^2_\theta \ell \mid X]$ is the
expected $4 \times 4$ Hessian, and $\ell_\theta$ is the autodiff score. The first term is the
naive plug-in. The second term is the correction, entered with a minus sign. It reweights the
score by the inverse curvature and removes the regularization bias.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

- Regime C (observational, the default for ZIP). The Hessian depends on $\theta$ (through both
  $\lambda$ and $\pi$) and on $Y$, so $\Lambda(X)$ is not constant and has no closed form. The
  package estimates it on a three-way data split (one part to fit $\theta(X)$, one to fit
  $\Lambda(X)$, one to evaluate). The chain is nonlinear mixture loss, then the Hessian depends on
  $\theta$, then $\Lambda$ must be estimated, then a three-way split.
- Regime A (randomized experiment). With a known treatment distribution $F_T$, the package can
  compute $\Lambda(X)$ by Monte-Carlo integration over $F_T$ on a two-way split. Pass
  `is_randomized=True` and a `treatment_dist`.

Regime detection is automatic. The full account is in [Estimation](../estimation/index.md).

## Usage

The primary entry point is `structural_dml`. For the default log-rate target, pass the family name.

```python
import numpy as np
from scipy.special import expit
from deep_inference import structural_dml

# Generate zero-inflated count data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

alpha_true = 1.0 + 0.2 * X[:, 0]
beta_true = 0.3 + 0.1 * X[:, 0]
gamma_true = -0.5 + 0.1 * X[:, 0]   # logit of zero inflation
delta_true = 0.0

lam = np.exp(alpha_true + beta_true * T)
pi = expit(gamma_true + delta_true * T)
is_structural_zero = np.random.binomial(1, pi).astype(bool)
Y = np.where(is_structural_zero, 0.0, np.random.poisson(lam).astype(float))
mu_true = beta_true.mean()

print(f"True mu* = {mu_true:.6f}")
print(f"Share of zeros = {(Y == 0).mean():.3f}")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='zip',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)

print(result.summary())
```

To target the zero-inflation effect instead, build the family object with `target='delta'`. The
`ZIPFamily` class lives in the `families` submodule (it is not re-exported at the top level).

```python
from deep_inference import structural_dml
from deep_inference.families import ZIPFamily

family = ZIPFamily(target='delta')   # mu* = E[delta(X)]
result = structural_dml(Y=Y, T=T, X=X, family=family, n_folds=50)
```

Note on the flexible `inference()` API. It exposes built-in models named "linear", "logit",
"multinomial_logit", "quantile", "did", and "did_fe" only, so `inference(model='zip')` is not
available. Reach ZIP through `structural_dml(family='zip')`.

## Evidence

Autodiff correctness is validated. In [Eval 02](../simulation_studies/eval_02.md), the ZIP family
is in the autodiff-only group (it has no closed-form derivatives to compare against), and both the
gradient-norm and Hessian-symmetry checks pass, marked PASS. ZIP is also covered by the package
integration part of that eval, which exercises the full `family.gradient()` and `family.hessian()`
paths.

Coverage has not been benchmarked for ZIP in this repository. The
[Replications](../replications/index.md) page lists ZIP as "oracle ready" in tier B, with the
FLM (cholesky) coverage and the RieszNet coverage both shown as not yet run (a dash). So the
parameter-recovery and autodiff components are validated, but there is no Monte-Carlo coverage
number for the Zero-Inflated Poisson yet. Treat the intervals as provisional until that benchmark
exists.

## Diagnostics and Pitfalls

- Not yet benchmarked for coverage. The autodiff machinery is correct, but no Monte-Carlo coverage
  study exists for ZIP in this repository, so the confidence intervals are not yet certified.
- Lambda method. The package defaults to `lambda_method='ridge'`, the validated safe choice on the
  other nonlinear families. Avoid `lambda_method='mlp'`, which produces invalid standard errors
  despite high correlation with the true $\Lambda$. The package warns if you select it.
- Hessian stability. The $4 \times 4$ Hessian can be near-singular when $\pi$ is close to 0 or 1, or
  when $\lambda$ is small. After a run, read `result.diagnostics['min_lambda_eigenvalue']` and
  confirm it stays above about $10^{-4}$.
- Make sure the zeros really are inflated. If a plain Poisson already predicts about the right
  number of zeros, the extra two parameters ($\gamma$ and $\delta$) are unidentified noise, and the
  simpler [Poisson model](poisson.md) will be more stable.

## References and API

The full API for every family, including the `ZIPFamily` class with its `target` argument and its
mixture `loss`, is in the [Families API reference](../api/families.md). The framework papers and the
Lambert (1992) zero-inflation reference are annotated in [References](../references/index.md).
