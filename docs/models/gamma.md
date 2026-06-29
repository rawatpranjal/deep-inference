# Gamma Model

The Gamma model is for outcomes that are continuous and strictly positive with a long right tail,
where the spread grows with the level. Costs, claim amounts, durations, and incomes all have this
shape. A linear model is wrong for them because it allows negative predictions and assumes constant
variance. The Gamma uses a log link to keep the mean positive and lets the variance scale with the
square of the mean. This page covers the model, its deviance loss, and the inference machinery, and
it is honest that the corrected estimator is not yet certified on this family.

## Source Papers

The estimator and its inference theory come from Farrell, Liang, and Misra. The Gamma family is one
instance of their general structural setup, where a neural network learns the heterogeneity
$\theta(X)$ and an influence-function correction removes the regularization bias. The annotated
reading list, with links to the transcripts, is in the [References](../references/index.md) section.
The two papers that matter here are Farrell, Liang, Misra (2021), "Deep Neural Networks for
Estimation and Inference" (Econometrica), and the extended Farrell, Liang, Misra (2025) working
paper that this package follows for the structural-parameter setup.

## When to Use

Use the Gamma model when the outcome is continuous, strictly positive, right-skewed, and its
variance increases with its mean. Three concrete cases.

- Healthcare costs as a function of a treatment, with age and comorbidities in $X$. Costs are
  positive, heavily right-skewed, and more variable for sicker patients.
- Insurance claim amounts as a function of the deductible, with policyholder demographics in $X$.
- Time durations, such as length of hospital stay or time to a repeat purchase.

If the outcome can be zero or negative, the Gamma is not appropriate. If the outcome is a count
rather than a continuous amount, use [Poisson](poisson.md) or
[Negative Binomial](negbin.md) instead.

## Model

Each observation is drawn from a Gamma distribution whose mean $\mu$ depends on the covariates $X$
and the treatment $T$ through a log link.

$$Y \mid X, T \sim \text{Gamma}(k, \mu/k), \qquad \mu = \exp(\eta), \qquad \eta = \alpha(X) + \beta(X)\cdot T$$

Under this parameterization the mean is $E[Y] = \mu$ and the variance is $\text{Var}[Y] = \mu^2 / k$,
where $k$ is the shape parameter. A larger shape means less variance relative to the mean. The shape
is a fixed hyperparameter (the default is `shape=2.0`), not something the network learns. The
network learns $\theta(X) = (\alpha(X), \beta(X))$, so $\texttt{theta\_dim} = 2$. Here $\alpha(X)$ is
the baseline log-mean and $\beta(X)$ is the per-unit effect of the treatment on the log-mean.

## Loss / Likelihood

The loss is the Gamma deviance, dropping the constants that do not depend on the parameters.

$$L(Y, T, \theta) = \frac{Y}{\mu} + \log\mu, \qquad \mu = \exp(\alpha + \beta T)$$

The implementation clamps the linear predictor to $[-10, 10]$ before exponentiating, and floors
$\mu$ at $10^{-6}$ in the divisions, both for numerical safety.

## Score and Hessian

The score and Hessian have closed forms, so the package uses them directly.

| Component | Formula |
|-----------|---------|
| Residual $r$ | $1 - Y/\mu$ |
| Hessian weight $W$ | $Y/\mu$ |
| Score $\nabla_\theta \ell$ | $(1 - Y/\mu)\cdot[1, T] = r\cdot[1, T]$ |

The full Hessian is

$$\nabla^2_\theta \ell = \frac{Y}{\mu} \begin{bmatrix} 1 & T \\ T & T^2 \end{bmatrix}.$$

The fact that drives the regime is that the Hessian weight $W = Y/\mu$ depends on $\theta$, through
$\mu = \exp(\alpha + \beta T)$, and it also depends on $Y$. A Hessian that moves with both $\theta$
and $Y$ is what forces the observational regime below.

## Target

The default estimand is the average effect of the treatment on the log-mean across the covariate
distribution.

$$\mu^* = E[\beta(X)]$$

With a log link, $\beta(X)$ is an effect on the log-mean, so $\exp(\beta)$ is the multiplicative
effect of a one-unit change in $T$ on $E[Y]$. The per-observation target is $m(\theta) = \beta$, and
its Jacobian with respect to $\theta = (\alpha, \beta)$ is the constant vector $H_\theta = [0, 1]$.

## Influence Function

For the average-$\beta$ target the per-observation influence function is

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i)^\top \, \Lambda(X_i)^{-1} \, \ell_\theta(Y_i, T_i, \hat\theta(X_i)),$$

where the plug-in $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation slope,
$H_\theta = [0, 1]$ is the target Jacobian (the unit selector picking $\beta$),
$\Lambda(X) = E[\nabla^2_\theta \ell \mid X]$ is the expected Hessian, and
$\ell_\theta = (1 - Y/\mu)[1, T]$ is the score from the table above. The first term is the naive
plug-in. The second term is the correction, entered with a minus sign. It reweights the score by
the inverse curvature $\Lambda^{-1}$ and removes the regularization bias the network injected into
$\beta(X)$.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

- Regime C (observational, the default for Gamma). The Hessian weight $W = Y/\mu$ depends on both
  $\theta$ and $Y$, so $\Lambda(X)$ is not a fixed constant and has no closed form. The package
  estimates it with a separate regression on a three-way data split (one part to fit $\theta(X)$,
  one to fit $\Lambda(X)$, one to evaluate). The chain is nonlinear loss, then Hessian depends on
  $\theta$, then $\Lambda$ must be estimated, then a three-way split.
- Regime A (randomized experiment). With a known treatment distribution $F_T$, the package can
  compute $\Lambda(X)$ by Monte-Carlo integration over $F_T$ on a two-way split. Pass
  `is_randomized=True` and a `treatment_dist`.

Regime detection is automatic. The full account is in [Estimation](../estimation/index.md).

## Usage

The primary entry point is `structural_dml`. The default `lambda_method='ridge'` is the validated
safe choice.

```python
import numpy as np
from deep_inference import structural_dml

# Generate Gamma-distributed positive outcomes
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

alpha_true = 2.0 + 0.3 * X[:, 0]
beta_true = 0.5 + 0.2 * X[:, 0]
mu_true = beta_true.mean()

mu = np.exp(alpha_true + beta_true * T)
shape = 2.0
Y = np.random.gamma(shape, mu / shape, size=n)

print(f"True mu* = {mu_true:.6f}")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='gamma',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)

print(result.summary())
```

To set the shape explicitly, build the family object and pass it in. The `GammaFamily` class is
exported at the top level of the package.

```python
from deep_inference import structural_dml, GammaFamily

family = GammaFamily(shape=2.0)   # higher shape means lower variance
result = structural_dml(Y=Y, T=T, X=X, family=family, n_folds=50)
```

Note on the flexible `inference()` API. It exposes built-in models named "linear", "logit",
"multinomial_logit", "quantile", "did", and "did_fe" only, so `inference(model='gamma')` is not
available. Reach Gamma through `structural_dml(family='gamma')`.

## Evidence

Component validations pass. Parameter recovery is reported in
[Eval 01](../simulation_studies/eval_01.md), where Gamma gives RMSE on $\alpha$ of 0.039, RMSE on
$\beta$ of 0.028, correlation with the true $\alpha(X)$ of 0.997, and correlation with the true
$\beta(X)$ of 0.999, marked PASS. Autodiff agreement with the closed-form score and Hessian is in
[Eval 02](../simulation_studies/eval_02.md), with a score error of $2.22\times10^{-16}$ and a
Hessian error of $4.44\times10^{-16}$, marked PASS.

End-to-end coverage is not certified, and the [Replications](../replications/index.md) page is
explicit about it. Gamma is listed as "A, blocked". The corrected FLM estimator (cholesky) is not
valid here. At M of 50 it shows a bias of about -0.072 and only 88 percent coverage, driven by a
left tail of bad-$\Lambda$ replications where the cholesky $\Lambda$-net misfits the noisy,
$Y$-dependent per-observation Hessians. On top of that, the
[Replications](../replications/index.md) page reports a separate downward nuisance bias of about
-0.045 that is specific to the log-link count and positive families and hits even RieszNet and the
oracle-$\Lambda$ ceiling. It attributes that bias to Jensen shrinkage on $E[\exp(\eta)]$. The
doubly-robust competitor RieszNet reaches 96 percent coverage on Gamma. Until the cholesky
$\Lambda$ hardening and the log-link nuisance-bias fix land in the core package, treat the corrected
Gamma confidence intervals as not yet certified, and prefer RieszNet if you need defensible
intervals on this family today.

## Diagnostics and Pitfalls

- Not yet certified. As the Evidence section states, the corrected estimator is blocked on this
  family at 88 percent coverage. Read the [Replications](../replications/index.md) page before
  relying on the intervals, and cross-check against RieszNet.
- Lambda method. The package defaults to `lambda_method='ridge'`, the validated safe choice. Avoid
  `lambda_method='mlp'`, which produces invalid standard errors despite high correlation with the
  true $\Lambda$. The package warns if you select it.
- Hessian stability. The weight $W = Y/\mu$ can be noisy because it carries the raw outcome $Y$ in
  the numerator. After a run, read `result.diagnostics['min_lambda_eigenvalue']` and confirm it
  stays above about $10^{-4}$.
- Confirm the outcome is strictly positive. The deviance divides by $\mu$ and takes a logarithm, so
  zeros or negatives are outside the model. If the data has exact zeros, the Gamma is the wrong
  family.

```python
assert (Y > 0).all(), "Gamma requires strictly positive outcomes"
print(f"Min lambda eigenvalue: {result.diagnostics.get('min_lambda_eigenvalue', 'N/A')}")
```

## References and API

The full API for every family, including the `GammaFamily` class with its `shape` argument and its
`loss`, `gradient`, and `hessian` methods, is in the [Families API reference](../api/families.md).
The papers behind the method are annotated in [References](../references/index.md).
