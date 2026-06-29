# Quantile Regression

The quantile model estimates distributional treatment effects: instead of asking "what is
the average effect of $T$ on $Y$?", it asks "what is the effect of $T$ on the $\tau$-th
quantile of $Y$?" Read this page if you care about tails, medians, or any specific
percentile of the conditional distribution.

## Source Papers

Smoothed quantile regression is a standard technique in the quantile regression
literature; the smoothing approach used here makes the check function twice differentiable
so the Hessian-based influence function correction from Farrell, Liang, Misra (2021) can
apply directly. Koenker, R. and Bassett, G. (1978), "Regression Quantiles," *Econometrica*,
is the original reference for the check function and its population properties.

The influence-function correction framework follows Farrell, Liang, Misra (2021), "Deep
Neural Networks for Estimation and Inference," *Econometrica*.

## When to Use

Use this model when:

- Your outcome $Y$ is continuous and you want effects at a specific quantile of its
  conditional distribution, not just the conditional mean.
- You suspect heteroskedasticity: the treatment may shift the spread of outcomes in
  addition to the mean.
- You want to study distributional effects such as the effect of a job training program
  on the lower quartile of earnings, or the effect of a drug on the 90th percentile
  of blood pressure.
- Your DGP is a location-shift model (variance of $Y$ does not depend on $T$) and you
  want a robustness check against mean regression.

For symmetric, homoskedastic outcomes, the median ($\tau = 0.5$) equals the mean
regression target and the two models give the same answer. For skewed or heavy-tailed
distributions, quantile regression is more informative and more robust.

Do not use this model if your outcome is binary or count-valued; it technically applies
but mean regression (logit, Poisson) is more natural. If you want only the median and
your data are symmetric, the linear model is simpler and avoids the smoothing parameter.

## Model

The model specifies the $\tau$-th conditional quantile of $Y$ as a linear function of
treatment $T$ with covariate-varying coefficients:

$$Q_\tau(Y \mid T, X) = \alpha(X) + \beta(X)\, T$$

where $\alpha(X)$ is the quantile-specific intercept and $\beta(X)$ is the quantile
treatment effect (QTE) at level $\tau$. Both vary with covariates $X$ via the neural
network. The parameter vector is $\theta(X) = [\alpha(X), \beta(X)]$, so
$\theta_{\text{dim}} = 2$.

The quantile level $\tau \in (0, 1)$ and the smoothing parameter $\varepsilon > 0$ are
set at model initialization. Defaults are $\tau = 0.5$ (median regression) and
$\varepsilon = 0.01$.

## Loss / Likelihood

The standard check function $\rho_\tau(u) = u(\tau - \mathbf{1}\{u < 0\})$ has a kink
at zero and is not differentiable there, so the Hessian does not exist at the true
parameter. The package uses a smooth approximation:

$$\tilde{\rho}_\tau(u) = \tau\, u + \varepsilon\, \log\!\left(1 + e^{-u/\varepsilon}\right)$$

where $u = y - \alpha - \beta t$ is the residual and $\varepsilon > 0$ is the smoothing
parameter.

Why smooth? As $\varepsilon \to 0$, $\tilde{\rho}_\tau$ converges to $\rho_\tau$
uniformly. For any $\varepsilon > 0$, $\tilde{\rho}_\tau$ is twice differentiable
everywhere, so the Hessian exists and the FLM influence-function correction applies. The
smoothing introduces an approximation in the theta estimate of order $\varepsilon$; the
default $\varepsilon = 0.01$ is small enough to be negligible in practice.

The `logaddexp(0, -u/eps)` formula is used in the implementation for numerical stability.

## Score and Hessian

Let $u = y - \alpha - \beta t$ be the residual and define
$s_\varepsilon(u) = \sigma(u/\varepsilon) = 1/(1 + e^{-u/\varepsilon})$, the sigmoid
evaluated at $u/\varepsilon$.

**Score:**

$$\ell_\theta = \frac{\partial \tilde{\rho}_\tau}{\partial \theta}
= \bigl(1 - \tau - s_\varepsilon(u)\bigr)\, [1,\; t]^\top$$

At the true $\theta^*$, the score has zero expectation when
$P(Y \leq \alpha^* + \beta^* T \mid X, T) = \tau$, which is precisely the defining
condition for the $\tau$-th quantile. The sigmoid term $s_\varepsilon(u)$ is a smooth
approximation to the indicator $\mathbf{1}\{u > 0\}$.

**Hessian:**

$$\ell_{\theta\theta} = \frac{\partial^2 \tilde{\rho}_\tau}{\partial \theta^2}
= \frac{s_\varepsilon(u)\,(1 - s_\varepsilon(u))}{\varepsilon}\,
\begin{pmatrix} 1 & t \\ t & t^2 \end{pmatrix}$$

The scalar weight $w(u) = s_\varepsilon(u)(1 - s_\varepsilon(u))/\varepsilon$ depends on
the residual $u = y - \alpha - \beta t$, which depends on both $\theta = [\alpha, \beta]$
and on $y$. This means the Hessian depends on both $\theta$ and $Y$, which forces the
model into Regime C always (see below).

The weight $w(u)$ is largest when $u \approx 0$ (the residual is near zero) and shrinks
to zero when $|u|/\varepsilon$ is large. Intuitively, observations near the quantile
boundary get the most weight in the curvature estimate.

## Target

The default target is the average quantile treatment effect:

$$\mu^* = E[\beta(X)]$$

For a location-shift model $Y = \alpha(X) + \beta(X)T + \sigma\varepsilon$ where $\varepsilon$
is symmetric, the QTE equals the ATE at every quantile because treatment shifts the
distribution without changing its shape. The validation DGP in
`evals/dgp_quantile.py` is a location-shift model with $\mu^* = E[\beta(X)] = 0.3$,
independent of $\tau$.

The target Jacobian is $H_\theta = e_1 = [0, 1]^\top$, the unit vector picking out the
$\beta$ component. This is the same constant vector as in the linear model with the same
two-parameter layout.

## Influence Function

The per-observation influence function is

$$\psi_i = H(\hat\theta(X_i)) - e_1^\top\, \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i))$$

where the plug-in $H(\hat\theta(X_i)) = \hat\beta(X_i)$ is the per-observation QTE,
$e_1 = [0, 1]^\top$ is the unit selector picking the $\beta$ component, $\ell_\theta$ is the
score from the Score and Hessian section, and
$\Lambda(X) = E[\ell_{\theta\theta}(Y, T, \theta(X)) \mid X]$ is estimated in the Lambda step of
three-way cross-fitting. The correction enters with a minus sign.

The point estimate is $\hat{\mu} = \frac{1}{n}\sum_i \psi_i$ with SE $\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

The quantile model is **always Regime C**. Both `hessian_depends_on_theta = True` and
`hessian_depends_on_y = True` because the Hessian weight
$w = s_\varepsilon(u)(1-s_\varepsilon(u))/\varepsilon$ depends on the residual
$u = y - \alpha - \beta t$, which involves both $\theta$ and $y$.

There is no randomization regime (Regime A) shortcut: even with a randomized treatment
and known $F_T$, you still need to integrate out $Y$ to get $\Lambda(X) = E[H \mid X]$,
which requires its own estimation step.

Three-way cross-fitting proceeds as follows:

1. **First split (theta training).** On a subset of the complement fold, fit the neural
   network to learn $\hat{\theta}(X) = [\hat{\alpha}(X), \hat{\beta}(X)]$ by minimizing
   the smoothed check loss.
2. **Second split (Lambda estimation).** On another subset, compute the per-observation
   Hessian $h_i = \ell_{\theta\theta}(Y_i, T_i, \hat{\theta}(X_i))$ using the fitted
   $\hat{\theta}$ from step 1. Then regress $h_i$ on $X_i$ to get $\hat{\Lambda}(X)$.
   Ridge regression (`lambda_method='ridge'`, `ridge_alpha=1000.0`) is the default.
3. **Held-out fold.** Assemble the influence function
   $\hat{\psi}_i = \hat{\beta}(X_i) - e_1^\top \hat{\Lambda}(X_i)^{-1} \ell_\theta(Y_i, T_i, \hat{\theta}(X_i))$
   and average across folds.

## Usage

```python
import numpy as np
from deep_inference import inference

np.random.seed(42)
n = 8000

# DGP: location-shift model with confounded treatment
X = np.random.uniform(-2, 2, n).reshape(-1, 1)
alpha_true = 1.0 + 0.5 * X[:, 0]
beta_true  = 0.3 + 0.2 * X[:, 0]
T = 0.5 * X[:, 0] + np.random.normal(0, 0.5, n)   # confounded
Y = alpha_true + beta_true * T + 0.5 * np.random.normal(0, 1, n)
# mu* = E[beta*(X)] = 0.3

result = inference(
    Y=Y, T=T, X=X,
    model='quantile',
    target='qte',       # E[beta(X)], the average QTE
    tau=0.5,            # median regression
    smooth_eps=0.01,    # smoothing parameter
    n_folds=20,
    epochs=200,
    patience=50,
    lr=0.01,
)
print(result.summary())
print(f"mu_hat = {result.mu_hat:.4f}  se = {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

**Different quantile levels:**

```python
# 75th-percentile QTE
result_75 = inference(Y, T, X, model='quantile', target='qte', tau=0.75)

# 25th-percentile QTE
result_25 = inference(Y, T, X, model='quantile', target='qte', tau=0.25)

print(f"Q25 QTE: {result_25.mu_hat:.4f}")
print(f"Q50 QTE: {result.mu_hat:.4f}")
print(f"Q75 QTE: {result_75.mu_hat:.4f}")
```

**Accessing the model class directly:**

```python
from deep_inference.models.quantile import QuantileModel
import torch

model = QuantileModel(tau=0.5, smooth_eps=0.01)
y  = torch.tensor(1.5)
t  = torch.tensor(0.5)
th = torch.tensor([1.0, 0.3])

print("loss  :", model.loss(y, t, th).item())
print("score :", model.score(y, t, th))
print("hess  :\n", model.hessian(y, t, th))
```

## Evidence

From `evals/reports/eval_12_coverage_n8000_20260219_175808.txt`
(M=50, n=8000, $\tau=0.5$, $\varepsilon=0.01$, ridge Lambda, n_folds=20):

True $\mu^* = 0.300$, mean estimate $\hat{\mu} = 0.293$.

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 94.0% (47/50) | 90-99% | PASS |
| SE Ratio | 1.053 | 0.7-1.5 | PASS |
| Bias | -0.007 | less than 0.05 | PASS |
| z-score mean | -0.166 | (-0.3, 0.3) | PASS |
| z-score std | 1.047 | 0.7-1.5 | PASS |

**EVAL 12: PASS**

All five criteria pass. The SE ratio of 1.053 indicates a slight overestimation of SE
(conservative), which is typical for ridge-estimated Lambda.

## Diagnostics and Pitfalls

**Always Regime C, always three-way split.** There is no shortcut. Budget data
accordingly: the eval at n=8000 and n_folds=20 took roughly 3 hours on 4 CPU cores.
Use n >= 5000 as a practical minimum; the eval showed 94% coverage at n=8000.

**Smoothing parameter $\varepsilon$ is a bias-variance tradeoff.** Smaller $\varepsilon$
(say 0.001) is closer to the true quantile loss but produces a spikier Hessian that is
harder to estimate. Larger $\varepsilon$ (say 0.1) is smoother but introduces more bias
in the theta estimate. The default $\varepsilon = 0.01$ worked well in the eval DGP.
If you change $\varepsilon$, also budget more data for the Lambda estimation step.

**n_folds=20 vs the recommended 50.** The eval used n_folds=20 rather than the package
recommendation of 50. Coverage was 94.0%, near the lower bound. Increasing to n_folds=50
is expected to bring coverage closer to 95%.

**Location-shift vs location-scale.** In the eval DGP, the treatment shifts the
distribution without changing its shape, so the QTE equals the ATE at every $\tau$.
If your DGP is location-scale (i.e., $\text{Var}(Y)$ also depends on $T$), the QTE
and ATE differ and the choice of $\tau$ changes the estimand meaningfully.

**Near-zero Hessian weight.** The weight $w = s_\varepsilon(u)(1-s_\varepsilon(u))/\varepsilon$
is small when $|u|/\varepsilon$ is large (large residuals or extreme observations). If
Lambda is near-singular, the package applies Tikhonov regularization via the
`tikhonov_scale` parameter. Check `result.diagnostics['min_lambda_eigenvalue']`; it
should exceed $10^{-4}$.

## References and API

**Papers:**

- Koenker, R. and Bassett, G. (1978). "Regression Quantiles." *Econometrica*, 46(1), 33-50.
- Farrell, M., Liang, T., and Misra, S. (2021). "Deep Neural Networks for Estimation and
  Inference." *Econometrica*.

**API:**

- `inference(Y, T, X, model='quantile', target='qte', tau=0.5, smooth_eps=0.01)`
- Model class: `deep_inference.Quantile`
  (`src/deep_inference/models/quantile.py`)
- DGP helper and oracle formulas: `evals/dgp_quantile.py`
- Eval: `evals/eval_12_quantile.py`, report at
  `evals/reports/eval_12_coverage_n8000_20260219_175808.txt`
