# Gaussian Model

The Gaussian model fits a heteroskedastic normal regression. The network learns both the
conditional mean $\mu(X, T)$ and the conditional standard deviation $\sigma(X)$
simultaneously. Use it when you believe the residual variance differs across observations,
or when you want a full-likelihood baseline comparable to a Gaussian GLM.

## Source Papers

The Gaussian formulation follows Farrell, Liang, and Misra (2021, 2025). The full
transcribed papers are in [References](../references/index.md).

## When to Use

Choose the Gaussian model when:

- the outcome is continuous and you want to model its variance as well as its mean,
- you suspect that different individuals or groups have systematically different noise
  levels, and
- you are willing to accept a slightly more expensive computation (the Hessian is $3 \times 3$
  and depends on $\sigma$, so Regime C applies).

**Gaussian vs Linear.** If you fix $\sigma$ to a constant, the Gaussian model collapses to
the linear model up to a constant rescaling of the loss. The difference matters when
$\sigma(X)$ varies across units. The Gaussian model weights each observation by
$1/\sigma^2$, which is efficient when that heteroskedasticity is real. The linear model
weights all observations equally and is faster and more stable. Prefer Gaussian only when
heteroskedasticity is substantively important to your application.

## Model

The structural model is

$$
Y \mid X, T \;\sim\; \mathcal{N}\!\bigl(\mu(X, T),\; \sigma(X)^2\bigr),
$$

where

$$
\mu(X, T) = \alpha(X) + \beta(X) \cdot T, \qquad \sigma(X) = \exp\!\bigl(\gamma(X)\bigr).
$$

The network outputs three values per observation: $\theta = (\alpha, \beta, \gamma)$, so
$\theta_{\dim} = 3$. The log-scale parameterization $\gamma = \log \sigma$ ensures
$\sigma > 0$ without any explicit constraint. The implementation clamps $\gamma$ to
$[-10, 10]$ for numerical safety (see `src/deep_inference/families/gaussian.py`).

## Loss / Likelihood

The negative log-likelihood, dropping constants, is

$$
\ell(y, t, \theta)
= \frac{(y - \mu)^2}{2\sigma^2} + \log \sigma
= \frac{(y - \mu)^2\,e^{-2\gamma}}{2} + \gamma,
$$

where $\mu = \alpha + \beta t$ and $\sigma = e^\gamma$.

## Score and Hessian

The gradient of the loss with respect to $\theta = (\alpha, \beta, \gamma)$ is

$$
\nabla_\theta \ell =
\begin{pmatrix}
(\mu - y)/\sigma^2 \\[4pt]
t\,(\mu - y)/\sigma^2 \\[4pt]
1 - (y - \mu)^2/\sigma^2
\end{pmatrix}.
$$

The Hessian is a $3 \times 3$ matrix. Writing $r = (\mu - y)/\sigma^2$ and
$s = (y - \mu)^2/\sigma^2$, the entries are

$$
\nabla^2_\theta \ell =
\begin{pmatrix}
1/\sigma^2 & t/\sigma^2 & -2r \\[4pt]
t/\sigma^2 & t^2/\sigma^2 & -2rt \\[4pt]
-2r & -2rt & 2s
\end{pmatrix}.
$$

The Hessian depends on $\theta$ through $\sigma^2 = e^{2\gamma}$. It also depends on $y$
through $r$ and $s$. Both dependencies are present, placing this model in Regime C.

In the source, `hessian_depends_on_theta = True`
(see `src/deep_inference/families/gaussian.py`).

## Target

The default estimand is

$$
\mu^* = \mathbb{E}[\beta(X)],
$$

the population-average slope of $\mu(X, T)$ with respect to $T$. The third component
$\gamma$ is a nuisance. The target Jacobian selects
$H_\theta = \partial \mu^*/\partial \theta = (0,\; 1,\; 0)$.

## Influence Function

The influence function for observation $i$ is

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i)^\top\, \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i)).$$

For the default $E[\beta]$ target, the plug-in is $H(\hat\theta(X_i)) = \hat\beta(X_i)$ (the
per-observation slope, not the constant $\mu^*$) and $H_\theta = (0,\; 1,\; 0)$ is the unit
selector picking the $\beta$ component from the three-dimensional
$\theta = (\alpha, \beta, \gamma)$. The score $\ell_\theta$ is the three-dimensional gradient
from the Score and Hessian section, and $\Lambda(x)$ is the $3 \times 3$ expected Hessian at
covariate value $x$. The correction enters with a minus sign, removing the regularization bias
in $\hat\beta(X_i)$.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

Because the Hessian depends on both $\theta$ and $y$, the Lambda estimator in Regime C
trains on per-observation Hessian samples computed from a separate fold of data, then
predicts $\hat\Lambda(x)$ at new $x$ values. The PSD-by-construction cholesky approach
($\hat\Lambda = LL^\top$ where $L$ is output by a separate network) ensures positive
definiteness in all reps.

## Algorithm and Regime

The Gaussian model uses **Regime C** (estimated Lambda, 3-way cross-fitting).

The flag `hessian_depends_on_theta = True` triggers the 3-way path automatically. Under
3-way cross-fitting, three disjoint splits are used in each outer fold. The first split
trains the network $\hat\theta$. The second forms per-observation Hessian samples and trains
the Lambda estimator. The third computes influence scores and contributes to the final
estimate. The outer folds rotate so every observation appears in the third role exactly once.

## Usage

**Legacy API** with `structural_dml`:

```python
import numpy as np
from deep_inference import structural_dml

np.random.seed(42)
n = 2000
X = np.random.randn(n, 3)
T = np.random.randn(n)
# Heteroskedastic: noise scale varies with X
sigma_x = np.exp(0.3 * X[:, 0])
beta_x = 0.5 + 0.2 * X[:, 0]
Y = X[:, 0] + beta_x * T + sigma_x * np.random.randn(n)

result = structural_dml(
    Y, T, X,
    family='gaussian',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)

print(f"Estimate: {result.mu_hat:.4f}")
print(f"SE:       {result.se:.4f}")
print(f"95% CI:   [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

The new `inference()` entry point maps the model names "linear", "logit",
"multinomial_logit", "quantile", "did", and "did_fe" only, so `inference(model='gaussian')`
is not available. Reach the Gaussian model through `structural_dml(family='gaussian')` as
above.

If the data are actually homoskedastic, the Gaussian model still recovers the correct
$\mu^*$. It is not wrong to use it in that case; it is just slower.

## Evidence

**Parameter recovery** (n=5000, one seed, source:
[Eval 01](../simulation_studies.md)):

| metric | value |
|--------|-------|
| RMSE($\alpha$) | 0.030 |
| RMSE($\beta$) | 0.040 |
| Corr($\alpha$) | 0.994 |
| Corr($\beta$) | 0.998 |
| status | PASS |

**Coverage** (M=50 replications, source: [Replications](../replications/index.md)):

| method | coverage | notes |
|--------|----------|-------|
| FLM[cholesky] | 92% | marginal |
| oracle-Lambda | 96% | ceiling |
| RieszNet | 96% | valid |

The 92% cholesky figure is marginal relative to the 93-97% target band. The oracle-Lambda
ceiling at 96% confirms that the mean and variance networks recover well and that the
influence-function assembly is correct. The gap is in Lambda estimation. The replications
page classifies this as "approximately done" rather than fully certified, because M=50 adds
about 3 percentage points of Monte-Carlo noise on its own. These numbers are not yet
benchmarked at M=100.

## Diagnostics and Pitfalls

**Gamma hitting the clamp.** The log-scale output $\gamma$ is clamped to $[-10, 10]$ in the
source, giving $\sigma \in (e^{-10}, e^{10})$. If training loss is erratic, check whether
$\gamma$ predictions are regularly hitting the boundary.

**3x3 Lambda near-singular.** The $3 \times 3$ Hessian can become ill-conditioned when
$\sigma$ is very small (making $1/\sigma^2$ large) or when $T$ has low conditional variance
given $X$. Check `diagnostics['min_lambda_eigenvalue'] > 1e-4` after every run.

**Use linear as a sanity check.** Running `family='linear'` first gives you a
coverage-validated baseline. If Gaussian gives a substantially different point estimate, the
heteroskedastic weighting is doing real work. If they agree, linear is simpler.

**Lambda method.** The default `lambda_method='ridge'` applies here. Avoid
`lambda_method='mlp'`. The mlp lambda method produces invalid standard errors on nonlinear
families despite high correlation with the oracle; the package emits a warning if you choose
it. See the [lambda method recommendations](../overview.md) for full coverage numbers.

## References and API

- [References index](../references/index.md) for the FLM 2021 and FLM 2025 paper
  transcripts.
- [API: families](../api.md) for `GaussianFamily` and the full `structural_dml`
  signature.
