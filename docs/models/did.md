# Difference-in-Differences

The DiD family in this package covers three estimators: a closed-form 2x2 case, a
heterogeneous neural 2x2 case, and a two-way fixed-effects panel case. All three return
an `InferenceResult` with a valid influence-function standard error. A single entry
point `did()` auto-selects the right estimator from the arguments you pass.

## Source Papers

The saturated 2x2 DiD regression and two-way FE panel DiD follow the standard DiD
literature. Angrist, J. D. and Pischke, J. S. (2009), *Mostly Harmless Econometrics*,
Chapter 5 covers both specifications. The influence-function methodology that produces
valid standard errors follows Farrell, Liang, Misra (2021), "Deep Neural Networks for
Estimation and Inference," *Econometrica*.

The closed-form 2x2 estimator is proved in `src/deep_inference/_did_closed.py` to equal
the HC0 robust OLS standard error of the saturated cell-mean regression to machine
precision.

## When to Use

Use DiD when you have a binary group indicator $G$ (treated vs control) and a binary
time indicator $P$ (pre vs post period), and you are willing to invoke the parallel
trends assumption: absent the treatment, the treated group's outcome would have evolved
the same way as the control group's outcome.

Choose the sub-method based on your data:

- `did(Y, group, post)`, the closed-form `exact` method, has no covariates and estimates
  one average treatment effect for all individuals.
- `did(Y, group, post, X=X)`, the `neural` method, lets the treatment effect $\tau(X)$
  vary with covariates and targets $E[\tau(X)]$.
- `did(Y, group, post, X=X, unit=unit, time=time)`, the `panel_fe` method, absorbs unit
  and time fixed effects and targets $E[\tau(X)]$ in a panel.

Do not use the closed-form `exact` method if you believe $\tau$ is heterogeneous. Do not
use `neural` if you have a genuine panel (the same units observed over time); use
`panel_fe`, which absorbs unit-level confounders.

## Model

**Saturated 2x2 DiD (methods `exact` and `neural`).**

The saturated regression is

$$Y_i = \alpha(X_i) + \gamma(X_i)\,G_i + \lambda(X_i)\,P_i + \tau(X_i)\,(G_i \cdot P_i) + \varepsilon_i$$

where $G_i \in \{0,1\}$ is the group indicator and $P_i \in \{0,1\}$ is the post-period
indicator. In the `neural` method, all four coefficients $[\alpha, \gamma, \lambda, \tau]$
vary with covariates $X_i$ via the neural network. In the `exact` method, they are
constants estimated from cell means.

The per-observation treatment variable packed into `T` is $t_i = [G_i, P_i, G_i P_i]$
(dimension 3). The design vector with intercept is $W_i = [1, G_i, P_i, G_i P_i]$
(dimension 4). The structural parameter vector is
$\theta_i = [\alpha, \gamma, \lambda, \tau]$, so `theta_dim = 4`.

**Panel FE DiD (method `panel_fe`).**

The two-way FE estimator removes unit-specific effects $\mu_i$ and time-specific effects
$\nu_t$ by within-transforming. Define $\tilde{Y}_{it} = Y_{it} - \bar{Y}_i - \bar{Y}_t + \bar{Y}$
and $\tilde{D}_{it} = D_{it} - \bar{D}_i - \bar{D}_t + \bar{D}$ as the within-transformed
outcome and treatment indicator. After this transformation, the model reduces to an
intercept-free regression through the origin:

$$\tilde{Y}_{it} = \tau(X_{it})\,\tilde{D}_{it} + \varepsilon_{it}$$

There is only one structural parameter per observation: $\tau(X_{it}) = \theta_0(X_{it})$,
so `theta_dim = 1`. The neural network maps covariates $X_{it}$ to the scalar effect.

## Loss / Likelihood

Both the saturated 2x2 and the panel FE model use squared-error loss.

**Saturated 2x2:**

$$\ell(y, t, \theta) = \frac{1}{2}(y - W^\top\theta)^2, \quad W = [1, G, P, G \cdot P]^\top$$

**Panel FE:**

$$\ell(\tilde{y}, \tilde{d}, \theta) = \frac{1}{2}(\tilde{y} - \tilde{d}\,\theta_0)^2$$

The factor $\frac{1}{2}$ is cosmetic and cancels in the gradient.

## Score and Hessian

**Saturated 2x2:**

$$\ell_\theta = (W^\top\theta - y)\, W, \qquad \ell_{\theta\theta} = W W^\top$$

The Hessian $W W^\top$ is constant: it does not depend on $\theta$ or $Y$.

**Panel FE:**

$$\ell_\theta = (\tilde{d}\,\theta_0 - \tilde{y})\,\tilde{d}, \qquad \ell_{\theta\theta} = \tilde{d}^2$$

The scalar Hessian $\tilde{d}^2$ is also constant in $\theta$ and $Y$.

Both Hessians are theta-independent, so $\Lambda(X) = E[\ell_{\theta\theta} \mid X]$ can
be estimated without any additional data split. This places both models in Regime B.

## Target

The target is $\mu^* = E[\tau(X)]$, the population average treatment effect (equal to the
ATT under parallel trends).

For the saturated 2x2, this is `AverageParameter(param_index=3)`, which picks out
$\theta_3 = \tau$.

For the panel FE, this is `AverageParameter(param_index=0)`, which picks out $\theta_0 = \tau$
(the only parameter).

For the closed-form `exact` method, the estimand is the cell-mean DiD contrast:

$$\beta = \mu_{11} - \mu_{10} - \mu_{01} + \mu_{00}, \qquad \mu_{gp} = E[Y \mid G=g, P=p]$$

## Influence Function

**Closed-form 2x2.** For observation $i$ in cell $(g, p)$, the influence function is

$$\psi_i = \hat\beta + s_{g_i, p_i}\,\frac{Y_i - \hat\mu_{g_i, p_i}}{\hat p_{g_i, p_i}}$$

where $s_{gp} \in \{+1, -1\}$ is $+1$ for cells $(0,0)$ and $(1,1)$ and $-1$ for cells
$(0,1)$ and $(1,0)$, $\hat\mu_{gp}$ is the cell mean, and $\hat p_{gp} = n_{gp}/n$ is
the cell proportion. Then $\hat\beta = \text{mean}(\psi_i)$ and
$\hat\sigma^2 = \text{var}(\psi_i)/n$ (HC0 denominator). The package docstring proves
this equals the HC0 robust OLS SE of the saturated regression to machine precision.

**Neural 2x2 and Panel FE.** The FLM influence function applies with the constant
Hessian:

$$\psi_i = H(\hat\theta(X_i)) - e_\tau^\top\, \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i))$$

where the plug-in $H(\hat\theta(X_i)) = \hat\tau(X_i)$ is the per-observation treatment effect,
$e_\tau$ is the unit selector picking the $\tau$ component of $\theta$, and $\ell_\theta$ is the
squared-error score. The correction enters with a minus sign. The point estimate is
$\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is $\mathrm{sd}(\psi)/\sqrt{n}$, which is
heteroskedasticity-robust.

## Algorithm and Regime

All three DiD methods use Regime B (analytic Lambda, two-way cross-fitting). The
Hessian is constant in $\theta$ and $Y$, so Lambda requires no additional estimation
step beyond averaging:

- Saturated 2x2: $\hat\Lambda = \frac{1}{n}\sum_i W_i W_i^\top$, the sample average of
  $WW^\top$.
- Panel FE: $\hat\Lambda(X) = E[\tilde{D}^2 \mid X]$, estimated by ridge regression
  without an intercept (the within-transform absorbed the intercept; the `panel_fe` model
  signals this via `analytic_intercept=False`).

No three-way split is needed. Each fold uses a two-way split: one half fits $\hat\theta(X)$
and the other half computes the influence function.

The `did()` function inspects its arguments to auto-select the method:

1. `unit` and `time` both provided: selects `panel_fe`.
2. `X` provided but no panel ids: selects `neural`.
3. Neither: selects `exact`.

Override with `method='exact'`, `method='neural'`, or `method='panel_fe'`.

## Usage

**Closed-form 2x2 (method `exact`):**

```python
import numpy as np
from deep_inference import did

np.random.seed(42)
n = 2000
G = (np.random.random(n) < 0.5).astype(float)   # treated group
P = (np.random.random(n) < 0.5).astype(float)   # post period

# True cell means: beta* = 3.1 - 2.0 - 1.5 + 1.0 = 0.6
cell_means  = {(0,0): 1.0, (0,1): 1.5, (1,0): 2.0, (1,1): 3.1}
cell_sigmas = {(0,0): 1.2, (0,1): 1.0, (1,0): 1.4, (1,1): 1.8}
Y = np.array([
    cell_means[(int(G[i]), int(P[i]))]
    + cell_sigmas[(int(G[i]), int(P[i]))] * np.random.normal()
    for i in range(n)
])

result = did(Y, group=G, post=P)
print(result.summary())
print(f"DiD estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
```

**Heterogeneous neural 2x2 (method `neural`):**

```python
from deep_inference import did
import numpy as np

np.random.seed(42)
n = 5000
X = np.random.normal(0, 1, (n, 3))
G = (np.random.random(n) < 0.5).astype(float)
P = (np.random.random(n) < 0.5).astype(float)

# True tau(X) = 0.5 + 0.3*X[:,0], so E[tau] = 0.5
tau_true   = 0.5 + 0.3 * X[:, 0]
alpha_true = 1.0 + 0.2 * X[:, 0]
gamma_true = 0.3 + 0.1 * X[:, 0]
lam_true   = 0.2 - 0.1 * X[:, 0]
Y = (alpha_true + gamma_true*G + lam_true*P + tau_true*(G*P)
     + np.random.normal(0, 1, n))

result = did(Y, group=G, post=P, X=X, n_folds=50, epochs=200)
print(f"E[tau(X)]: {result.mu_hat:.4f} +/- {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

# theta_hat has four columns: [alpha, gamma, lambda, tau]
tau_hat = result.theta_hat[:, 3]
print(f"tau(X) mean: {tau_hat.mean():.4f}  (true {tau_true.mean():.4f})")
```

**Panel FE DiD (method `panel_fe`):**

```python
from deep_inference import did
import numpy as np

np.random.seed(42)
N_units, N_periods = 400, 6
n = N_units * N_periods

unit = np.repeat(np.arange(N_units), N_periods)
time = np.tile(np.arange(N_periods), N_units)
X    = np.random.normal(0, 1, (n, 2))

G_unit = (np.arange(N_units) % 2 == 1).astype(float)
G    = G_unit[unit]               # half the units treated
Post = (time >= 3).astype(float)  # last three periods post
D    = G * Post                   # treatment indicator

tau_true = 0.5 + 0.3 * X[:, 0]
unit_fe  = 0.5 * unit / N_units
time_fe  = 0.3 * time / N_periods
Y = unit_fe + time_fe + tau_true * D + np.random.normal(0, 1, n)

result = did(Y, group=G, post=Post, X=X, unit=unit, time=time,
             n_folds=50, epochs=200)
print(f"E[tau(X)]: {result.mu_hat:.4f} +/- {result.se:.4f}")

# theta_hat is (n, 1); column 0 is tau(X)
print(f"tau(X) sample mean: {result.theta_hat[:, 0].mean():.4f}")
```

**Via `inference()` directly:**

```python
from deep_inference import inference
import numpy as np

# Saturated 2x2 neural: pack [G, P, G*P] into T
T_did = np.column_stack([G, P, G * P])
result = inference(Y, T=T_did, X=X, model='did', target='tau', t_tilde=0.0)

# Panel FE: residualize first
from deep_inference.utils import residualize_fixed_effects
Y_tilde = residualize_fixed_effects(Y, unit, time)
D_tilde = residualize_fixed_effects(D, unit, time)
result  = inference(Y_tilde, T=D_tilde, X=X, model='did_fe', target='fe_effect',
                    t_tilde=0.0)
```

## Evidence

All numbers are read directly from eval reports under `evals/reports/`.

**Closed-form 2x2** (`evals/reports/eval_13_did_20260608_140151.txt`,
M=500, n=2000, $\beta^* = 0.6$):

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 96.4% (482/500) | 90-99% | PASS |
| SE Ratio | 0.969 | 0.7-1.5 | PASS |
| Bias | -0.0004 | less than 0.05 | PASS |
| z-score mean | -0.003 | (-0.3, 0.3) | PASS |
| z-score std | 0.969 | 0.7-1.5 | PASS |

**EVAL 13: PASS**

**Neural 2x2** (`evals/reports/eval_14_did_nn_20260608_142239.txt`,
M=25, n=5000, $E[\tau^*] = 0.5$):

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 96.0% (24/25) | 90-99% | PASS |
| SE Ratio | 0.847 | 0.7-1.5 | PASS |
| Bias | +0.026 | less than 0.05 | PASS |
| z-score mean | +0.51 | (-0.3, 0.3) | FAIL |
| z-score std | 0.850 | 0.7-1.5 | PASS |

**EVAL 14: FAIL** (z-score mean 0.51 exceeds the 0.3 threshold). Parameter recovery,
autodiff, and Lambda estimation all pass individually. M=25 is a small number of
replications, so the z-score mean estimate is noisy. Coverage is 96.0% but the
off-center z-score distribution needs further investigation at larger M.

**Panel FE** (`evals/reports/eval_15_panel_fe_20260609_060522.txt`,
M=50, 400 units, 6 periods):

| Outcome | Coverage | SE Ratio | Bias | Status |
|---------|---------|---------|------|--------|
| Continuous | 98.0% (49/50) | 1.019 | -0.002 | PASS |
| Binary (LPM) | 94.0% (47/50) | 1.040 | -0.003 | PASS |

**EVAL 15: PASS** (both continuous and binary outcomes).

## Diagnostics and Pitfalls

**Parallel trends is untestable.** All three DiD methods invoke parallel trends: absent
the treatment, the treated group would have evolved like the control group. This is an
identifying assumption you must argue for from your research design. The package enforces
no pre-trend tests; add them yourself if needed.

**`exact` vs `neural`.** The `exact` estimator assumes $\tau$ is the same for all
individuals. If $\tau$ varies with $X$, the `exact` method returns an OLS-weighted
average that depends on cell sizes and may not equal $E[\tau(X)]$. The `neural` method
directly targets $E[\tau(X)]$ with a heteroskedasticity-robust SE.

**Panel FE and serial correlation.** The panel FE standard error assumes independence
across observations. If there is within-unit serial correlation (the common case in
panels), cluster-robust SEs are needed. The package does not implement cluster-robust
SEs; if serial correlation is present, the reported SE underestimates uncertainty.

**Neural 2x2 z-score centering.** The z-score mean of 0.51 at M=25, n=5000 (Eval 14) is
a soft warning. Coverage is 96.0% and bias is small (+0.026), but the off-center z-score
distribution suggests finite-sample bias in the SE estimator at this configuration. Use
larger n or M for stronger guarantees.

**theta_hat layout for neural 2x2.** `result.theta_hat` has four columns ordered as
$[\alpha, \gamma, \lambda, \tau]$ (0-indexed). Access the heterogeneous treatment effect
with `result.theta_hat[:, 3]`. For panel FE, `result.theta_hat` has one column, accessed
as `result.theta_hat[:, 0]`.

**Panel FE binary outcomes (LPM).** The panel FE method handles binary $Y$ as a linear
probability model after within-transformation. The IF standard error is
heteroskedasticity-robust, which correctly handles the $p(1-p)$ variance structure. Eval
15 confirmed 94.0% coverage on binary outcomes.

## References and API

**Papers:**

- Angrist, J. D. and Pischke, J. S. (2009). *Mostly Harmless Econometrics.* Princeton
  University Press. Chapter 5 covers the saturated DiD regression and two-way FE panel
  DiD.
- Farrell, M., Liang, T., and Misra, S. (2021). "Deep Neural Networks for Estimation
  and Inference." *Econometrica*.

**API:**

- `did(Y, group, post)`: closed-form 2x2 (`src/deep_inference/_did_closed.py`)
- `did(Y, group, post, X=X)`: heterogeneous neural 2x2,
  model class `deep_inference.DiDModel` (`src/deep_inference/models/did.py`)
- `did(Y, group, post, X=X, unit=unit, time=time)`: two-way panel FE,
  model class `deep_inference.FEPanelDiDModel` (`src/deep_inference/models/panel_fe.py`)
- `inference(Y, T, X, model='did', target='tau')`: direct new API for neural 2x2
- `inference(Y, T, X, model='did_fe', target='fe_effect')`: direct new API for panel FE
- Eval 13: `evals/eval_13_did.py` (closed-form)
- Eval 14: `evals/eval_14_did_nn.py` (neural 2x2)
- Eval 15: `evals/eval_15_panel_fe.py` (panel FE)
