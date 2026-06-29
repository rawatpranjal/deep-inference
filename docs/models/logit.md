# Logit Model

The logit model handles binary outcomes (0 or 1) with a heterogeneous treatment effect. The
network learns individual-specific log-odds intercepts $\alpha(X)$ and slopes $\beta(X)$.
Use it when you need a probability model and the logistic latent-error assumption is
acceptable.

## Source Papers

The logit formulation here follows Farrell, Liang, and Misra (2021, 2025), with the
heterogeneous-parameter discrete-choice extension in Hetzenecker and Osterhaus (2024). The
full transcribed papers are in [References](../references/index.md).

## When to Use

Choose the logit model when:

- the outcome takes exactly two values, 0 and 1,
- you are modeling the probability of an event (purchase, default, program uptake, market
  entry), and
- you are comfortable with the logistic latent-error assumption (the error in the latent
  index is logistically distributed).

For the same binary setting with a normal latent error, use [Probit](probit.md) instead. The
two models are close numerically but differ by a scale factor on the coefficients.

## Model

The structural model is

$$
P(Y = 1 \mid X, T) = \sigma\!\bigl(\alpha(X) + \beta(X) \cdot T\bigr),
\qquad \sigma(z) = \frac{1}{1 + e^{-z}},
$$

where $\alpha(X)$ and $\beta(X)$ are learned by a neural network. The linear predictor is
$\eta = \alpha(X) + \beta(X) \cdot T$ and the predicted probability is $p = \sigma(\eta)$.
The parameter vector is $\theta = (\alpha, \beta)$, so $\theta_{\dim} = 2$.

## Loss / Likelihood

The training loss is binary cross-entropy:

$$
\ell(y, t, \theta) = \log\!\bigl(1 + e^{\eta}\bigr) - y\,\eta,
\qquad \eta = \alpha + \beta t.
$$

This equals $-y\log p - (1-y)\log(1-p)$ and is the negative log-likelihood of a Bernoulli
with probability $p = \sigma(\eta)$.

## Score and Hessian

The gradient of the loss with respect to $\theta = (\alpha, \beta)$ is

$$
\nabla_\theta \ell = (p - y)
\begin{pmatrix} 1 \\ t \end{pmatrix}.
$$

The Hessian is

$$
\nabla^2_\theta \ell = p(1-p)
\begin{pmatrix} 1 & t \\ t & t^2 \end{pmatrix}.
$$

The Hessian weight $p(1-p)$ depends on $\theta$ through $p = \sigma(\alpha + \beta t)$. It
does not depend on $y$ at all: the observed outcome drops out of the second derivative
because the cross-entropy Hessian equals the Fisher information. This y-independence is the
key property that allows Regime A (the Monte Carlo integral path) to work under
randomization. In the source, `hessian_depends_on_theta = True` and
`hessian_depends_on_y = False` (see `src/deep_inference/models/logit.py`).

## Target

The default estimand is the average log-odds effect

$$
\mu^* = \mathbb{E}[\beta(X)].
$$

This is the population-average change in log-odds of $Y=1$ per unit increase in $T$,
integrated over the covariate distribution.

An alternative target is the **average marginal effect** (AME) on the probability scale:

$$
\mu^*_{\text{AME}} = \mathbb{E}[p(1-p) \cdot \beta(X)].
$$

Pass `target='ame'` to select it. The AME is usually easier to interpret substantively but
is harder to compare across models.

## Influence Function

The influence function for observation $i$ is

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i)^\top\, \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i)).$$

For the default `beta` target, the plug-in is $H(\hat\theta(X_i)) = \hat\beta(X_i)$ (the
per-observation log-odds slope, not the constant $\mu^*$) and $H_\theta = (0,\; 1)$ is the unit
selector picking the $\beta$ component (for `ame`, use the AME Jacobian instead). The score is
$\ell_\theta = (p_i - Y_i)(1, T_i)^\top$. The correction enters with a minus sign, removing the
regularization bias in $\hat\beta(X_i)$.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

$\Lambda(x)$ is the expected Hessian at covariate value $x$:

$$
\Lambda(x) = \mathbb{E}\!\bigl[p(1-p)\,TT^\top \mid X = x\bigr].
$$

Under randomization (known $F_T$), $\Lambda(x)$ is computed via Monte Carlo as described
below. Under observational data, it is estimated from a separate data split.

## Algorithm and Regime

The logit model uses **Regime A** when the treatment distribution $F_T$ is known (randomized
experiment), and **Regime C** otherwise (observational data).

**Why Regime A is possible.** The Hessian weight $p(1-p)$ does not depend on $y$. So under
randomization you can integrate over independent draws $\tilde t \sim F_T$ and average

$$
\hat\Lambda(x) = \frac{1}{M} \sum_{m=1}^M
p\!\bigl(\hat\theta(x), \tilde t_m\bigr)\,
\bigl(1 - p\!\bigl(\hat\theta(x), \tilde t_m\bigr)\bigr)\,
\tilde t_m \tilde t_m^\top,
$$

without needing a separate data fold for $\Lambda$.

**Why Regime C otherwise.** When treatment is endogenous and $F_T$ is unknown, you cannot
integrate over $F_T$ without observing it. The engine then estimates $\Lambda(x)$ from a
third data split using the `lambda_method` you specify.

**Lambda method.** The default is `lambda_method='ridge'`, which produces 96% coverage on
the logit benchmark (source: [Replications](../replications/index.md)). Avoid
`lambda_method='mlp'`. That method produces only 67% coverage despite high correlation with
the oracle Lambda. The package emits a warning if you choose it.

To invoke Regime A explicitly:

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

result = inference(
    Y, T, X,
    model='logit',
    target='beta',
    is_randomized=True,
    treatment_dist=Normal(0, 1),
)
```

## Usage

**Legacy API** with `structural_dml`:

```python
import numpy as np
from deep_inference import structural_dml
from scipy.special import expit

np.random.seed(42)
n = 2000
X = np.random.randn(n, 3)
T = np.random.randn(n)
alpha_x = 0.5 + 0.3 * X[:, 0]
beta_x = -0.5 + 0.5 * X[:, 0]
p_true = expit(alpha_x + beta_x * T)
Y = np.random.binomial(1, p_true).astype(float)

result = structural_dml(
    Y, T, X,
    family='logit',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
    lr=0.01,
)

print(f"Estimate: {result.mu_hat:.4f}")
print(f"SE:       {result.se:.4f}")
print(f"95% CI:   [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

**New API** with `inference` (average marginal effect):

```python
from deep_inference import inference

result = inference(
    Y, T, X,
    model='logit',
    target='ame',
    hidden_dims=[64, 32],
    epochs=100,
    n_folds=50,
)
print(result.summary())
```

## Evidence

**Parameter recovery** (n=5000, one seed, source:
[Eval 01](../simulation_studies/eval_01.md)):

| metric | value |
|--------|-------|
| RMSE($\alpha$) | 0.127 |
| RMSE($\beta$) | 0.180 |
| Corr($\alpha$) | 0.963 |
| Corr($\beta$) | 0.968 |
| status | PASS |

RMSE is higher than for the linear model because the logit link compresses predictions near
0 and 1, making the effective signal-to-noise ratio lower. Correlation above 0.96 confirms
the network recovers the shape of the heterogeneity.

**Coverage** (M=100 replications, truth $\mu^*=0.1481$, source:
[Replications](../replications/index.md)):

| method | bias | SE-ratio | coverage |
|--------|------|----------|----------|
| Oracle-MLE | +0.0054 | 1.02 | 94% |
| FLM[cholesky] | -0.0020 | 1.07 | 98% |
| FLM[ridge] | +0.0166 | 0.92 | 86% |
| FLM[oracle-Lambda] | +0.0167 | 0.90 | 86% |
| RieszNet | +0.0062 | 1.06 | 100% |
| Naive (no correction) | +0.0069 | 0.11 | 10% |

FLM[cholesky] reaches 98%, which beats the oracle-Lambda ceiling of 86%. The replications
page explains why: the cholesky factorization's implicit regularization helps in the
low-overlap region where the true $\Lambda$ is near-singular ($\det\Lambda = 4p(1-p) \to 0$
when $p \to 0$ or $p \to 1$).

FLM[ridge] produces 86% coverage on this DGP, which is below the target band. The
replications page lists this benchmark. If you see coverage below 90%, switching to the
cholesky Lambda estimator is the recommended fix.

The Naive row (10% coverage, SE-ratio 0.11) shows the severity of the regularization bias
on a nonlinear model.

## Diagnostics and Pitfalls

**min_lambda_eigenvalue near zero.** The Hessian weight $p(1-p)$ approaches zero when the
network predicts probabilities near 0 or 1. If the minimum eigenvalue of $\hat\Lambda$ is
below $1\mathrm{e}{-4}$, the influence scores will be noisy. The cholesky Lambda estimator
adds a small Tikhonov term by default. Check `diagnostics['min_lambda_eigenvalue']` after
every run.

**FLM[ridge] at 86%.** The replications table shows that ridge Lambda underperforms cholesky
on the logit DGP. If you are using `lambda_method='ridge'` and see low coverage, this is the
likely cause. Use the cholesky path instead (the default for the new `inference()` API).

**Avoid mlp Lambda.** The `lambda_method='mlp'` option produces only 67% coverage on logit
despite high oracle correlation. The package warns you, but heed it. See the
[lambda method table in CLAUDE.md](../overview.md) for the full comparison.

**Log-odds vs AME.** The `beta` target is on the log-odds scale. A coefficient of 0.15 does
not mean the treatment raises the probability by 15 percentage points. If you need the
probability-scale effect, use `target='ame'`.

## References and API

- [References index](../references/index.md) for the FLM 2021, FLM 2025, and
  Hetzenecker-Osterhaus 2024 paper transcripts.
- [API: families](../api/families.md) for `LogitFamily` and the full `structural_dml`
  signature.
