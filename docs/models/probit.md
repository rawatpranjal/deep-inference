# Probit Model

The probit model handles binary outcomes (0 or 1) with a normal-CDF link instead of the
logistic sigmoid. Use it when your theoretical setup assumes a normally distributed latent
error, or when you need a binary model that is directly comparable to a classical probit
regression.

## Source Papers

The probit formulation follows Farrell, Liang, and Misra (2021, 2025). The full transcribed
papers are in [References](../references/index.md).

## When to Use

Choose the probit model when:

- the outcome takes exactly two values, 0 and 1,
- you are modeling the probability of an event and prefer the normal latent-error
  assumption over the logistic one, and
- you want coefficients that are directly comparable to a classical probit.

**Probit vs Logit.** Both models handle binary outcomes. The difference is the shape of the
link function. Probit uses $\Phi(\eta)$ (normal CDF), while logit uses $\sigma(\eta)$
(logistic sigmoid). The two curves are very similar over the range $\eta \in [-2, 2]$, but
the logit link has heavier tails. Coefficients differ by a scale factor of roughly 1.6
(probit coefficients are smaller). Coverage is validated at 98% for both families (M=50);
pick the one whose latent-error assumption fits your application.

The probit Hessian is computed via autodiff because the closed-form expression involves the
Mills ratio and its derivative, which is more error-prone to implement by hand than to
differentiate automatically. This means the package uses Regime C regardless of whether
treatment is randomized.

## Model

The structural model is

$$
P(Y = 1 \mid X, T) = \Phi\!\bigl(\alpha(X) + \beta(X) \cdot T\bigr),
$$

where $\Phi$ is the standard normal CDF. The linear predictor is
$\eta = \alpha(X) + \beta(X) \cdot T$ and the predicted probability is $p = \Phi(\eta)$.
The parameter vector is $\theta = (\alpha, \beta)$, so $\theta_{\dim} = 2$.

## Loss / Likelihood

The training loss is the negative log-likelihood of a Bernoulli with probability
$p = \Phi(\eta)$:

$$
\ell(y, t, \theta) =
-y \log \Phi(\eta) - (1 - y) \log\!\bigl(1 - \Phi(\eta)\bigr),
\qquad \eta = \alpha + \beta t.
$$

Probabilities are clamped to $[10^{-7},\, 1-10^{-7}]$ to prevent log-of-zero
(see `src/deep_inference/families/probit.py`).

## Score and Hessian

Let $\phi = \Phi'$ denote the standard normal PDF. The gradient of the loss is

$$
\nabla_\theta \ell =
\frac{\phi(\eta)}{\Phi(\eta)\bigl(1-\Phi(\eta)\bigr)}\,
\bigl(\Phi(\eta) - y\bigr)
\begin{pmatrix} 1 \\ t \end{pmatrix}.
$$

The fraction $\phi(\eta) / [\Phi(\eta)(1-\Phi(\eta))]$ is the inverse Mills ratio evaluated
at $\eta$. At prediction time the gradient simplifies, but the raw formula above is what
enters the influence function correction.

The Hessian depends on $\theta$ through $\eta$ and therefore on $\Phi(\eta)$ and
$\phi(\eta)$. In the source, `hessian()` returns `None` and autodiff computes the Hessian
for each observation (see `src/deep_inference/families/probit.py`). The flag
`hessian_depends_on_theta = True` is set explicitly.

Because the loss depends on $y$ and the Hessian is the second derivative of the loss, the
per-observation Hessian also depends on $y$. This rules out Regime A (which requires
y-independence of the Hessian).

## Target

The default estimand is the average log-index effect

$$
\mu^* = \mathbb{E}[\beta(X)].
$$

This is the population-average change in the normal-index $\eta$ per unit increase in $T$.
It is on the index scale (like a probit coefficient), not the probability scale.

An alternative target is the **average marginal effect** (AME) on the probability scale:

$$
\mu^*_{\text{AME}} = \mathbb{E}[\phi(\eta) \cdot \beta(X)].
$$

Pass `target='ame'` to select it.

## Influence Function

The influence function for observation $i$ is

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i)^\top\, \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i)).$$

For the default `beta` target, the plug-in is $H(\hat\theta(X_i)) = \hat\beta(X_i)$ (the
per-observation index slope, not the constant $\mu^*$) and $H_\theta = (0,\; 1)$ is the unit
selector picking the $\beta$ component. The score $\ell_\theta$ is the Mills-ratio gradient from
the Score and Hessian section, evaluated at $\hat\theta(X_i)$. The correction enters with a
minus sign, removing the regularization bias in $\hat\beta(X_i)$.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

$\Lambda(x) = \mathbb{E}[\nabla^2_\theta \ell \mid X = x]$ is the $2 \times 2$ expected
Hessian, estimated via the cholesky Lambda network on a separate data split.

## Algorithm and Regime

The probit model uses **Regime C** (estimated Lambda, 3-way cross-fitting).

Regime C applies because both `hessian_depends_on_theta = True` and the Hessian depends on
$y$ (via the loss's second derivative). This means you cannot compute $\Lambda(x)$ without
$\hat\theta(x)$, and you cannot estimate $\Lambda(x)$ on the same split that produced
$\hat\theta$ without inducing overfitting into the influence scores.

Under 3-way cross-fitting, the engine uses three disjoint splits per outer fold. The first
split trains the network $\hat\theta$. The second forms per-observation Hessian samples
(via autodiff) and trains the Lambda estimator. The third computes influence scores using
both $\hat\theta$ and $\hat\Lambda$, contributing to the final estimate. The outer folds
rotate so every observation appears in the third role exactly once.

**Note on the `lambda_method` argument.** The default `lambda_method='ridge'` applies. The
same mlp warning that exists for logit applies here: `lambda_method='mlp'` can produce
invalid standard errors. The cholesky Lambda estimator (used by the new `inference()` API
by default) is the recommended path for probit.

## Usage

**Legacy API** with `structural_dml`:

```python
import numpy as np
from scipy.stats import norm
from deep_inference import structural_dml

np.random.seed(42)
n = 2000
X = np.random.randn(n, 3)
T = np.random.randn(n)
alpha_x = 0.5 + 0.3 * X[:, 0]
beta_x = -0.4 + 0.4 * X[:, 0]
eta = alpha_x + beta_x * T
p_true = norm.cdf(eta)
Y = np.random.binomial(1, p_true).astype(float)

result = structural_dml(
    Y, T, X,
    family='probit',
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
"multinomial_logit", "quantile", "did", and "did_fe" only, so `inference(model='probit')` is
not available. Reach probit through `structural_dml(family='probit')` as above.

**AME target.** The average-marginal-effect target is available through the families API:

```python
from deep_inference import ProbitFamily, structural_dml

result = structural_dml(Y, T, X, family=ProbitFamily(target='ame'))
```

## Evidence

**Parameter recovery.** Not benchmarked in `evals/eval_01.py` at the time of writing
(eval_01 covers linear, gaussian, logit, and seven other families but not probit). Recovery
benchmarks for probit are not yet recorded in [Eval 01](../simulation_studies/eval_01.md).

**Coverage** (M=50 replications, source: [Replications](../replications/index.md)):

| method | coverage |
|--------|----------|
| FLM[cholesky] | 98% |
| RieszNet | 98% |

Both methods pass at M=50. The replications page reports probit in the "B, done" tier,
meaning the benchmark is complete and the result is certified. The probit cholesky result
(98%) is one of the strongest in the family suite, notably better than the 93% on linear at
M=100. The replications page notes that probit passes the cholesky path despite using
autodiff for the Hessian rather than a closed-form expression.

## Diagnostics and Pitfalls

**Inverse Mills ratio near zero.** The factor $\phi(\eta) / [\Phi(\eta)(1-\Phi(\eta))]$
grows large when $|\eta|$ is large (probability near 0 or 1). This inflates the gradient
norm, which can destabilize training. The probabilities are clamped at $10^{-7}$ in the
source, which limits but does not eliminate the inflation. If the network is predicting
extreme probabilities, adding more regularization (higher weight decay or fewer epochs) helps.

**Autodiff Hessian cost.** Because the probit Hessian is computed via autodiff rather than a
closed-form expression, the Lambda-estimation step is slower than for logit. The cost grows
with the network depth and width. At `hidden_dims=[64, 32]` the overhead is small, but it
is noticeable at larger architectures.

**Coefficient scale.** Probit coefficients $\beta$ are on the normal-index scale. To convert
to log-odds (logit scale), multiply by approximately 1.6. When comparing probit and logit
estimates, always report the scale you are using.

**min_lambda_eigenvalue.** Check `diagnostics['min_lambda_eigenvalue'] > 1e-4` after every
run. If the eigenvalue is near zero, the Lambda network is fitting poorly, which inflates
the influence scores and underestimates the variance.

## References and API

- [References index](../references/index.md) for the FLM 2021 and FLM 2025 paper
  transcripts.
- [API: families](../api/families.md) for `ProbitFamily` and the full `structural_dml`
  signature.
