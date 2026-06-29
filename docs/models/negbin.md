# Negative Binomial Model

The Negative Binomial model is for count outcomes that are overdispersed, meaning the variance is
larger than the mean. A plain Poisson assumes the variance equals the mean, so when a few units
generate far more events than the rest, Poisson standard errors come out too small. The Negative
Binomial adds a single extra parameter to absorb that extra spread. This page covers the model,
its likelihood, and the inference machinery, and it is honest about the fact that the corrected
estimator is not yet certified on this family.

## Source Papers

The estimator and its inference theory come from Farrell, Liang, and Misra. The Negative Binomial
family is one instance of their general structural setup, where a neural network learns the
heterogeneity $\theta(X)$ and an influence-function correction removes the regularization bias.
The annotated reading list, with links to the transcripts, is in the
[References](../references/index.md) section. The two papers that matter here are Farrell, Liang,
Misra (2021), "Deep Neural Networks for Estimation and Inference" (Econometrica), and the extended
Farrell, Liang, Misra (2025) working paper that this package follows for the structural-parameter
setup.

## When to Use

Use the Negative Binomial model when the outcome is a count (a non-negative integer) and the
variance exceeds the mean, so the Poisson model underfits the spread. Three concrete cases.

- Doctor visits per year where a small set of heavy users drives a long right tail.
- Insurance claims filed per policy, which cluster (most customers file few, some file many).
- Species observed in an ecological survey, where counts are typically overdispersed.

Decide between Poisson and Negative Binomial by comparing the sample variance with the sample
mean. If the variance is close to the mean, use [Poisson](poisson.md). If the variance clearly
exceeds the mean, use the Negative Binomial. If the excess is driven specifically by a glut of
zeros rather than a heavy tail, consider the [Zero-Inflated Poisson model](zip.md).

## Model

Each observation is drawn from a Negative Binomial distribution whose mean $\mu$ depends on the
covariates $X$ and the treatment $T$ through a log link.

$$Y \mid X, T \sim \text{NegBin}(\mu, r), \qquad \mu = \exp(\eta), \qquad \eta = \alpha(X) + \beta(X)\cdot T$$

The variance is $\text{Var}(Y) = \mu + \mu^2 / r$, where $r = 1/\text{overdispersion}$ is the
dispersion parameter. As $r$ grows large, the $\mu^2/r$ term vanishes and the model collapses back
to Poisson. The dispersion is a fixed hyperparameter (the default is `overdispersion=0.5`, so
$r = 2$), not something the network learns. The network learns
$\theta(X) = (\alpha(X), \beta(X))$, so $\texttt{theta\_dim} = 2$. Here $\alpha(X)$ is the baseline
log-mean and $\beta(X)$ is the per-unit effect of the treatment on the log-mean.

## Loss / Likelihood

The loss is the full Negative Binomial (NB2) negative log-likelihood.

$$L(Y, T, \theta) = -\log\Gamma(Y+r) + \log\Gamma(r) + \log\Gamma(Y+1) + (r+Y)\log(r+\mu) - Y\log\mu - r\log r$$

with $\mu = \exp(\alpha + \beta T)$. The implementation keeps all of these terms, including the
gamma-function constants, so the reported value is a proper likelihood. The linear predictor is
clamped to $[-20, 20]$ before exponentiating, and $\mu$ is floored at $10^{-10}$ inside the
logarithm, both for numerical safety.

## Score and Hessian

The score and Hessian have closed forms, so the package uses them directly. Write the dispersion
in either form, $r = 1/\text{overdispersion}$ or equivalently $\text{overdispersion} = 1/r$.

| Component | Formula |
|-----------|---------|
| Residual $r_{\text{res}}$ | $(Y - \mu)/(1 + \text{overdispersion}\cdot\mu)$ |
| Score $\nabla_\theta \ell$ | $\dfrac{r(\mu - Y)}{r + \mu}\,[1, T]$ |
| Hessian weight $W$ | $\dfrac{r\,\mu\,(r + Y)}{(r + \mu)^2}$ |

The full Hessian is

$$\nabla^2_\theta \ell = W \begin{bmatrix} 1 & T \\ T & T^2 \end{bmatrix}, \qquad W = \frac{r\,\mu\,(r + Y)}{(r + \mu)^2}.$$

Two facts drive the regime. The Hessian weight $W$ depends on $\theta$ through $\mu = \exp(\eta)$,
and it also depends on $Y$ through the $(r + Y)$ factor. This is the observed-information Hessian,
not the expected-information one. (Replacing $Y$ by its mean $\mu$ would give the simpler expected
weight $r\mu/(r+\mu)$, which drops the $Y$-dependence, but the implementation uses the observed
form.) A Hessian that moves with both $\theta$ and $Y$ is what forces the observational regime
below.

## Target

The default estimand is the average effect of the treatment on the log-mean across the covariate
distribution, the same target as Poisson.

$$\mu^* = E[\beta(X)]$$

With a log link, $\beta(X)$ is a semi-elasticity, so a unit increase in $T$ changes the expected
count by approximately $100 \cdot \beta$ percent. The per-observation target is $m(\theta) = \beta$,
and its Jacobian with respect to $\theta = (\alpha, \beta)$ is the constant vector $H_\theta = [0, 1]$.

## Influence Function

For the average-$\beta$ target the per-observation influence function is

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i)^\top \, \Lambda(X_i)^{-1} \, \ell_\theta(Y_i, T_i, \hat\theta(X_i)),$$

where the plug-in $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation slope,
$H_\theta = [0, 1]$ is the target Jacobian (the unit selector picking $\beta$),
$\Lambda(X) = E[\nabla^2_\theta \ell \mid X]$ is the expected Hessian, and
$\ell_\theta = \frac{r(\mu - Y)}{r + \mu}[1, T]$ is the score from the table above. The first
term is the naive plug-in. The second term is the correction, entered with a minus sign. It
reweights the score by the inverse curvature $\Lambda^{-1}$ and removes the regularization bias
the network injected into $\beta(X)$.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

- Regime C (observational, the default for Negative Binomial). The Hessian weight depends on both
  $\theta$ and $Y$, so $\Lambda(X)$ is not a fixed constant and has no closed form. The package
  estimates it with a separate regression on a three-way data split (one part to fit $\theta(X)$,
  one to fit $\Lambda(X)$, one to evaluate). The chain is nonlinear loss, then Hessian depends on
  $\theta$, then $\Lambda$ must be estimated, then a three-way split.
- Regime A (randomized experiment). With a known treatment distribution $F_T$, the package can
  compute $\Lambda(X)$ by Monte-Carlo integration over $F_T$ on a two-way split instead. Pass
  `is_randomized=True` and a `treatment_dist`.

Regime detection is automatic. The full account of the regimes is in
[Estimation](../estimation/index.md).

## Usage

The primary entry point is `structural_dml`. The default `lambda_method='ridge'` is the validated
safe choice.

```python
import numpy as np
from deep_inference import structural_dml

# Generate overdispersed count data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

alpha_true = 1.0 + 0.2 * X[:, 0]
beta_true = 0.3 + 0.1 * X[:, 0]
mu = np.exp(alpha_true + beta_true * T)

r = 2.0  # dispersion parameter
p = r / (r + mu)
Y = np.random.negative_binomial(r, p).astype(float)
mu_true = beta_true.mean()

print(f"True mu* = {mu_true:.6f}")
print(f"Mean = {Y.mean():.2f}, Variance = {Y.var():.2f}, ratio = {Y.var()/Y.mean():.2f}")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='negbin',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)

print(result.summary())
```

To set the dispersion explicitly, build the family object and pass it in. The `NegBinFamily` class
is exported at the top level of the package.

```python
from deep_inference import structural_dml, NegBinFamily

family = NegBinFamily(overdispersion=0.5)  # r = 1 / 0.5 = 2.0
result = structural_dml(Y=Y, T=T, X=X, family=family, n_folds=50)
```

Note on the flexible `inference()` API. It exposes built-in models named "linear", "logit",
"multinomial_logit", "quantile", "did", and "did_fe" only, so `inference(model='negbin')` is not
available. Reach the Negative Binomial through `structural_dml(family='negbin')`.

## Evidence

Component validations pass. Parameter recovery is reported in
[Eval 01](../simulation_studies/eval_01.md), where the Negative Binomial gives RMSE on $\alpha$ of
0.059, RMSE on $\beta$ of 0.061, correlation with the true $\alpha(X)$ of 0.985, and correlation
with the true $\beta(X)$ of 0.938, marked PASS. Autodiff agreement with the closed-form score and
Hessian is in [Eval 02](../simulation_studies/eval_02.md), with a score error of
$3.33\times10^{-16}$ and a Hessian error of $5.55\times10^{-16}$, marked PASS.

End-to-end coverage is a different story, and the [Replications](../replications/index.md) page is
explicit about it. The Negative Binomial is listed as "A, blocked". The corrected FLM estimator
(cholesky) is not valid here. At M of 50 it shows a bias of about -0.111 and only a fragile 96
percent coverage, driven by a left tail of bad-$\Lambda$ replications where the cholesky
$\Lambda$-net misfits the noisy, $Y$-dependent per-observation Hessians. On top of that, the
[Replications](../replications/index.md) page reports a separate downward nuisance bias of about
-0.045 that is specific to the log-link count families and hits even RieszNet and the
oracle-$\Lambda$ ceiling. It attributes that bias to Jensen shrinkage on $E[\exp(\eta)]$. The
doubly-robust competitor RieszNet reaches 96 percent coverage on this family. Until the cholesky
$\Lambda$ hardening and the log-link nuisance-bias fix land in the core package, treat the
corrected Negative Binomial confidence intervals as not yet certified, and prefer RieszNet if you
need defensible intervals on this family today.

## Diagnostics and Pitfalls

- Not yet certified. As the Evidence section states, the corrected estimator is blocked on this
  family. Read the [Replications](../replications/index.md) page before relying on the intervals,
  and cross-check against RieszNet.
- Lambda method. The package defaults to `lambda_method='ridge'`, the validated safe choice. Avoid
  `lambda_method='mlp'`. It correlates highly with the true $\Lambda$ but produces invalid
  standard errors. The package warns if you select it.
- Hessian stability. The weight involves $\mu = \exp(\eta)$ and the observed count $Y$, so it can
  be noisy. After a run, read `result.diagnostics['min_lambda_eigenvalue']` and confirm it stays
  above about $10^{-4}$.
- Pick the dispersion deliberately. Larger overdispersion means less information per observation
  and wider intervals. The weight $W$ approaches $1/\text{overdispersion}$ for large $\mu$, so
  high-count observations get relatively less weight than they would under Poisson.

```python
mean_y, var_y = Y.mean(), Y.var()
print(f"Dispersion ratio: {var_y / mean_y:.2f}")
if var_y / mean_y > 1.5:
    print("Overdispersion detected -> NegBin is appropriate")

print(f"Min lambda eigenvalue: {result.diagnostics.get('min_lambda_eigenvalue', 'N/A')}")
```

## References and API

The full API for every family, including the `NegBinFamily` class with its `overdispersion`
argument and its `loss`, `gradient`, and `hessian` methods, is in the
[Families API reference](../api/families.md). The papers behind the method are annotated in
[References](../references/index.md).
