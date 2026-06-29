# Poisson Model

The Poisson model is for count outcomes, the non-negative integers $0, 1, 2, \ldots$, when you
believe the treatment shifts the rate at which events happen and that shift differs across the
covariates $X$. This page walks through the model end to end, from the data-generating process
to the exact standard-error machinery, so that you can run it and read every diagnostic without
guessing.

## Source Papers

The estimator and its inference theory come from Farrell, Liang, and Misra. The Poisson family
is one concrete instance of their general structural setup, where a neural network learns the
heterogeneity and an influence-function correction restores valid confidence intervals. The
annotated reading list, with links to the transcripts, is in the
[References](../references/index.md) section. The two papers that matter here are Farrell, Liang,
Misra (2021), "Deep Neural Networks for Estimation and Inference" (Econometrica), and the
extended Farrell, Liang, Misra (2025) working paper that this package follows for the
structural-parameter setup.

## When to Use

Use the Poisson model when the outcome is a count, a non-negative integer with no fixed upper
bound, and the variance is roughly equal to the mean (equidispersion). Three concrete cases.

- Patent counts as a function of research-and-development spending, with firm size and industry
  in $X$.
- Doctor visits per year as a function of insurance generosity, with age and health status in
  $X$.
- Traffic accidents at an intersection as a function of the speed limit, with traffic volume and
  road design in $X$.

If the variance is much larger than the mean (overdispersion, which is common in real count
data), the Poisson standard errors will be too small. In that case move to the
[Negative Binomial model](negbin.md). If there are far more zeros than a Poisson rate can
produce, move to the [Zero-Inflated Poisson model](zip.md).

## Model

Each observation is drawn from a Poisson distribution whose rate $\lambda$ depends on the
covariates $X$ and the treatment $T$ through a log link.

$$Y \mid X, T \sim \text{Poisson}(\lambda), \qquad \lambda = \exp(\eta), \qquad \eta = \alpha(X) + \beta(X)\cdot T$$

The log link, $\lambda = \exp(\eta)$, guarantees the rate is positive for any value of the
linear predictor $\eta$. The neural network learns the pair of functions
$\theta(X) = (\alpha(X), \beta(X))$, so the parameter vector has dimension $\texttt{theta\_dim} = 2$.
Here $\alpha(X)$ is the baseline log-rate and $\beta(X)$ is the per-unit effect of the treatment
on the log-rate.

## Loss / Likelihood

The loss is the Poisson negative log-likelihood, dropping the $\log(y!)$ term that does not
depend on the parameters.

$$L(Y, T, \theta) = \lambda - Y \log \lambda, \qquad \lambda = \exp(\alpha + \beta T)$$

In the implementation the linear predictor is clamped to $[-20, 20]$ before exponentiating, and
a tiny $10^{-10}$ is added inside the logarithm, both for numerical safety. Neither changes the
loss in the region where the optimizer operates.

## Score and Hessian

The score is the gradient of the loss with respect to $\theta$, and the Hessian is its second
derivative. Both have closed forms for Poisson, so the package uses them directly rather than
autodiff.

| Component | Formula |
|-----------|---------|
| Residual $r$ | $Y - \lambda$ |
| Hessian weight $W$ | $\lambda$ |
| Score $\nabla_\theta \ell$ | $(\lambda - Y)\cdot[1, T] = -r\cdot[1, T]$ |

The full Hessian is

$$\nabla^2_\theta \ell = \lambda \begin{bmatrix} 1 & T \\ T & T^2 \end{bmatrix}.$$

The single fact that drives everything downstream is this. The Hessian weight is
$W = \lambda = \exp(\alpha + \beta T)$, so it depends on $\theta$. It does not depend on $Y$. A
weight that moves with $\theta$ means the curvature of the loss is not a fixed constant, and that
is exactly what forces the observational regime described below. High-count observations get
larger weight $\lambda$, so they carry more influence in the curvature.

## Target

The default estimand is the average effect of the treatment on the log-rate across the covariate
distribution.

$$\mu^* = E[\beta(X)]$$

With a log link, $\beta(X)$ is a semi-elasticity, $\partial \log E[Y] / \partial T = \beta(X)$, so
a unit increase in $T$ changes the expected count by approximately $100 \cdot \beta$ percent. If
$\hat{\mu} = 0.05$, then on average a one-unit increase in treatment raises the expected count by
about 5 percent. The per-observation target is $m(\theta) = \beta$, and its Jacobian with respect
to $\theta = (\alpha, \beta)$ is the constant vector $H_\theta = [0, 1]$.

## Influence Function

The influence function is the per-observation quantity whose sample average is the debiased
estimate, and whose sample variance gives the standard error. For the average-$\beta$ target it is

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i)^\top \, \Lambda(X_i)^{-1} \, \ell_\theta(Y_i, T_i, \hat\theta(X_i)),$$

where the plug-in $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation slope,
$H_\theta = [0, 1]$ is the target Jacobian (the unit selector picking $\beta$),
$\Lambda(X) = E[\nabla^2_\theta \ell \mid X]$ is the expected Hessian (the nuisance object the
regime has to supply), and $\ell_\theta = (\lambda - Y)[1, T]$ is the score from the table above.
The first term is the naive plug-in. The second term is the correction, entered with a minus
sign. It is built from the score (which is large exactly where the network mis-fits) reweighted
by the inverse curvature $\Lambda^{-1}$, and it removes the regularization bias the network
injected into $\beta(X)$.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

The regime is the rule for how $\Lambda(X)$ is obtained, and it is set by the structure of the
Hessian.

- Regime C (observational, the default for Poisson). The Hessian weight
  $\lambda = \exp(\alpha + \beta T)$ depends on $\theta$, so $\Lambda(X)$ is not a fixed constant
  and cannot be read off in closed form. The package estimates it with a separate regression on a
  three-way data split (one part to fit $\theta(X)$, one to fit $\Lambda(X)$, one to evaluate).
  This is the chain nonlinear loss, then Hessian depends on $\theta$, then $\Lambda$ must be
  estimated, then a three-way split.
- Regime A (randomized experiment). If you ran a randomized experiment and the treatment
  distribution $F_T$ is known, the package can instead compute $\Lambda(X)$ by Monte-Carlo
  integration over $F_T$, with no estimation and a two-way split. Pass `is_randomized=True` and a
  `treatment_dist`.

The detection is automatic from your inputs. The broader explanation of the three regimes is in
[Estimation](../estimation/index.md).

## Usage

The primary entry point is `structural_dml`. The default `lambda_method='ridge'` is the
validated safe choice and needs no extra configuration.

```python
import numpy as np
from deep_inference import structural_dml

# Generate count data
np.random.seed(42)
n = 2000
X = np.random.randn(n, 10)
T = np.random.randn(n)

alpha_true = 1.0 + 0.2 * X[:, 0]
beta_true = 0.3 + 0.1 * X[:, 0]
lam = np.exp(alpha_true + beta_true * T)
Y = np.random.poisson(lam).astype(float)
mu_true = beta_true.mean()

print(f"True mu* = {mu_true:.6f}")
print(f"Mean count = {Y.mean():.2f}, max count = {Y.max()}")

result = structural_dml(
    Y=Y, T=T, X=X,
    family='poisson',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)

print(result.summary())
```

If your data came from a randomized experiment, switch to Regime A so $\Lambda(X)$ is computed
rather than estimated.

```python
from deep_inference.lambda_.compute import Normal

result = structural_dml(
    Y=Y, T=T, X=X,
    family='poisson',
    is_randomized=True,
    treatment_dist=Normal(0, 1),
)
```

Note on the flexible `inference()` API. It exposes built-in models named "linear", "logit",
"multinomial_logit", "quantile", "did", and "did_fe" only, so `inference(model='poisson')` is not
available. Reach Poisson through `structural_dml(family='poisson')`, or through `inference()` with
a custom `loss` and `target_fn` if you need a non-default target.

## Evidence

Component validations for Poisson pass cleanly. Parameter recovery is reported in
[Eval 01](../simulation_studies/eval_01.md), where Poisson gives RMSE on $\alpha$ of 0.014, RMSE
on $\beta$ of 0.030, correlation with the true $\alpha(X)$ of 0.998, and correlation with the true
$\beta(X)$ of 0.972, marked PASS. Autodiff agreement with the closed-form score and Hessian is in
[Eval 02](../simulation_studies/eval_02.md), with a score error of $1.74\times10^{-9}$ and a
Hessian error of $2.57\times10^{-9}$, marked PASS.

The end-to-end coverage benchmark is the Poisson row of the
[Replications](../replications/index.md) page, run at M of 100 on a confounded DGP with true
$\mu^*$ of 0.8981. The numbers below are copied from that page.

| Method | Bias | SE-ratio | Coverage |
|--------|------|----------|----------|
| Oracle-MLE | -0.0058 | 0.98 | 99% |
| FLM (cholesky) | -0.0116 | 0.98 | 98% |
| FLM (ridge) | -0.0325 | 1.07 | 94% |
| FLM (oracle-$\Lambda$) | -0.0403 | 1.02 | 95% |
| RieszNet | +0.0382 | 0.32 | 98% |
| Naive | -0.0177 | 0.25 | 33% |

The corrected FLM estimator (cholesky) is the least-biased valid method here, at 98 percent
coverage. The Naive row, no correction, covers the truth only 33 percent of the time, which is
the failure the package exists to fix. The [Replications](../replications/index.md) page flags
RieszNet as unreliable on Poisson, because its empirical standard error is inflated by a few
divergent replications (SE-ratio 0.32), so its 98 percent is not trustworthy.

## Diagnostics and Pitfalls

- Lambda method. The package defaults to `lambda_method='ridge'`, which is the validated safe
  choice. Avoid `lambda_method='mlp'`. It correlates highly with the true $\Lambda$ but produces
  invalid standard errors, because a high correlation does not control the variance of the
  $\Lambda$ estimates. The package emits a warning if you select it.
- Hessian stability. The weight $\lambda = \exp(\alpha + \beta T)$ can be small when the predicted
  rate is low, which makes $\Lambda(X)$ close to singular. After a run, read
  `result.diagnostics['min_lambda_eigenvalue']` and check it stays above about $10^{-4}$. Tikhonov
  regularization is applied to the inversion to keep it stable.
- Check for overdispersion before trusting the model. Compare the sample variance with the sample
  mean. If the variance exceeds the mean by more than about half, the Poisson assumption is
  violated and the standard errors will be too small. Move to the
  [Negative Binomial model](negbin.md).

```python
print(f"Mean: {Y.mean():.2f}, Variance: {Y.var():.2f}")
if Y.var() > 1.5 * Y.mean():
    print("Overdispersion detected, consider family='negbin'")

diagnostics = result.diagnostics
print(f"Min lambda eigenvalue: {diagnostics.get('min_lambda_eigenvalue', 'N/A')}")
```

## References and API

The full API for every family, including the `PoissonFamily` class and its `loss`, `gradient`,
and `hessian` methods, is in the [Families API reference](../api/families.md). The papers behind
the method are annotated in [References](../references/index.md).
