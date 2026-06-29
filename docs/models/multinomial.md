# Multinomial Logit (Conditional Logit / McFadden)

The multinomial logit model handles discrete choice among $J \geq 3$ alternatives when
individuals have measurable heterogeneous preferences. Read this page if you have a choice
outcome (mode of transport, brand, product) and you want valid confidence intervals for
population-average preference coefficients.

## Source Papers

Hetzenecker, S. and Osterhaus, C. (2024). "Deep Learning for the Estimation of
Heterogeneous Parameters in Discrete Choice Models." *arXiv:2408.09560*.
Section 2 of this paper is the direct source for the conditional logit setup: utility
$V_{ij} = \alpha_j(W) + x'_{ij}\beta(W)$, the score and Fisher information Hessian
formulas, and the three-way cross-fitting requirement. The abstract describes their Monte
Carlo evidence that "regular robust standard errors lead to invalid inference results,
showing the need for the influence function approach."

The core influence-function methodology follows Farrell, Liang, Misra (2021),
"Deep Neural Networks for Estimation and Inference," *Econometrica*.

## When to Use

Use this model when:

- Each individual chooses exactly one option from $J \geq 3$ discrete alternatives.
- The alternatives have observable, measurable attributes (price, travel time, quality)
  that vary across individuals and alternatives.
- You believe that preference coefficients (price sensitivity, brand loyalty) vary across
  individuals as a function of their observable characteristics $W$.
- Examples: transportation mode choice (car, bus, train), brand choice among competing
  products, market entry choice across markets.

Do not use this model for binary ($J=2$) outcomes. That belongs to the logit model. Do not
use it when alternatives have no attribute data and you only have alternative-specific
intercepts per individual; a standard fixed-parameter multinomial logit is more appropriate
in that case.

## Model

This is McFadden's conditional logit model, extended to allow all parameters to vary with
individual characteristics.

Individual $i$ faces $J$ alternatives. Alternative $j$ has systematic utility

$$V_{ij} = \alpha_j(W_i) + x'_{ij}\,\beta(W_i)$$

where $W_i \in \mathbb{R}^{d_w}$ are individual characteristics (demographics, income),
$x_{ij} \in \mathbb{R}^K$ are the $K$ attributes of alternative $j$ for individual $i$
(price, quality), $\alpha_j(W_i)$ is the alternative-specific intercept, and
$\beta(W_i) \in \mathbb{R}^K$ is the attribute coefficient vector, common across alternatives.

Individual $i$ picks the alternative that maximizes their utility, so the choice probability is

$$P(Y_i = j \mid W_i, X_i) = \frac{\exp(V_{ij})}{\sum_{m=0}^{J-1} \exp(V_{im})}$$

The normalization $\alpha_0 = 0$ is imposed (alternative 0 is the reference). The neural
network maps $W_i$ to the full parameter vector

$$\theta(W_i) = [\alpha_1(W_i), \ldots, \alpha_{J-1}(W_i),\; \beta_1(W_i), \ldots, \beta_K(W_i)]$$

so $\theta_{\text{dim}} = (J-1) + K$.

**Data layout.** This model uses an encoding different from scalar-treatment models.

| Variable | Shape | Contents |
|----------|-------|----------|
| $W$ (passed as `X`) | $(n,\, d_w)$ | Individual characteristics, the neural network input |
| $T$ | $(n,\, J \times K)$ | Alternative attributes, packed row-by-row |
| $Y$ | $(n,)$ | Chosen alternative index as float: $0, 1, \ldots, J-1$ |

The packing order for $T$: row $i$ is the $K$ attributes of alternative 0, then of
alternative 1, up to alternative $J-1$, all concatenated. Inside the model this row is
reshaped to $(J, K)$.

## Loss / Likelihood

The loss is categorical cross-entropy, the negative log-likelihood of the chosen
alternative:

$$\ell(Y_i, T_i, \theta) = -\log P(Y_i = y_i \mid W_i, X_i) = \log\!\sum_{m} \exp V_{im} - V_{i,y_i}$$

This is the standard multinomial log-likelihood. The `torch.logsumexp` trick is used for
numerical stability.

## Score and Hessian

**Score.** Define the per-alternative residual $r_{ij} = p_{ij} - \mathbf{1}\{Y_i = j\}$,
where $p_{ij} = P(Y_i = j \mid W_i, X_i)$. The gradient splits into the $\alpha$ block
and the $\beta$ block.

For $j = 1, \ldots, J-1$:

$$\frac{\partial \ell}{\partial \alpha_j} = r_{ij}$$

For $k = 1, \ldots, K$:

$$\frac{\partial \ell}{\partial \beta_k} = \sum_{j=0}^{J-1} r_{ij}\, x_{ijk}$$

All $J$ alternatives contribute to the $\beta$ gradient, including the reference alternative
$j=0$ (whose intercept is fixed at zero, but whose attributes still enter the utility and
the beta gradient).

**Hessian.** The Hessian equals the Fisher information matrix. It does not depend on $Y$,
only on $\theta$ through the softmax probabilities. Define $\bar{x}_i = \sum_j p_{ij} x_{ij}$
as the probability-weighted mean attribute. Then:

$$H_{\alpha\alpha}[j, m] = p_{i,\,j+1}\,(\delta_{jm} - p_{i,\,m+1}), \quad j, m = 0, \ldots, J-2$$

$$H_{\alpha\beta}[j, \,:] = p_{i,\,j+1}\,(x_{i,\,j+1} - \bar{x}_i)$$

$$H_{\beta\beta} = \sum_{j=0}^{J-1} p_{ij}\,(x_{ij} - \bar{x}_i)(x_{ij} - \bar{x}_i)'$$

Hetzenecker and Osterhaus (2024, Section 2) write this as the matrix $\dot{G}_i$ with
diagonal entries $\dot{g}_{kk} = P_j(1 - P_j)$ and off-diagonal entries
$\dot{g}_{k,l} = -P_j P_m$, which is exactly the formula above.

The $y$-independence is important: because the Hessian is the Fisher information, the
expected Hessian $\Lambda(W) = E[\ell_{\theta\theta} \mid W]$ does not require integrating
over $Y$, only over the treatment distribution.

## Target

The default target is the average attribute coefficient:

$$\mu^* = E[\beta_1(W)]$$

the first attribute coefficient. The parameter vector layout is
$[\alpha_1, \ldots, \alpha_{J-1}, \beta_1, \ldots, \beta_K]$, so $\beta_1$ sits at
0-indexed position $J-1$. In the `inference()` API, pass `target='beta'` and optionally
`target_idx=k` (0-indexed) to select the $(k+1)$-th attribute coefficient.

For a price sensitivity example with $K=2$ where the first attribute is price,
$\mu^* = E[\beta_{\text{price}}(W)]$ is the average price sensitivity across the population.

Other available targets:

- **Choice probability** $P(Y = j \mid W, X_{\text{new}})$ via
  `ChoiceProbabilityTarget(alternative=j, n_alternatives=J, n_attributes=K)`.
- **Multinomial AME** $\partial P(Y=j)/\partial x_{jk}$ via
  `MultinomialAME(alternative=j, attribute=k, n_alternatives=J, n_attributes=K)`.

## Influence Function

For target $\mu^* = E[\beta_k(W)]$, the per-observation plug-in is
$H(\hat\theta(W_i)) = \hat\beta_k(W_i)$ and the target Jacobian is the constant unit selector
$H_\theta = e_{(J-1)+k}$ of dimension $\theta_{\text{dim}}$, which picks out the $\beta_k$
component. The per-observation influence function is

$$\psi_i = H(\hat\theta(W_i)) - e_{(J-1)+k}^\top\, \Lambda(W_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(W_i))$$

where $\ell_\theta$ is the multinomial score from the Score and Hessian section, and the
correction enters with a minus sign. The point estimate is $\hat{\mu} = \frac{1}{n}\sum_i \psi_i$
and the standard error is $\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

**Regime A (randomized experiment with known $F_T$).** The Hessian depends on $\theta$
through the softmax probabilities, but it does not depend on $Y$. So $\Lambda(W) =
E[\ell_{\theta\theta}(T, \theta(W)) \mid W]$ can be computed by Monte Carlo integration
over the treatment distribution, without any additional data split. Pass
`is_randomized=True` and `treatment_dist=...` to `inference()`. This uses two-way
cross-fitting.

**Regime C (observational data).** In observational settings, $\Lambda(W)$ must be
estimated because it depends on $\theta(W)$, which is itself estimated from data.
This requires three-way cross-fitting. Hetzenecker and Osterhaus (2024, Section 2.3)
describe this explicitly: "the complement $S^c_s$ is split into two pieces to first
estimate $\hat{\delta}_s(w_i)$ using the first piece, and then $\hat{\Lambda}_s(w_i)$
using the second piece together with the fixed functions $\hat{\delta}_s(w_i)$."

The three splits reduce the effective training data to roughly 60% of each fold's
complement. The package auto-selects Regime A or C from your inputs. For observational
data (the common case), it falls back to Regime C with ridge Lambda estimation.

**patience=50 is non-negotiable.** With the three-way split, each training run has only
about 60% of the complement data for fitting $\theta$. With the default `patience=10`,
early stopping fires at roughly 15-20 epochs, far before the loss converges. The
`structural_dml` path auto-bumps patience to 50 for `family='multinomial_logit'`. Pass
`patience=50` explicitly when using the `inference()` API.

## Usage

**Legacy API:**

```python
import numpy as np
from scipy.special import softmax
from deep_inference import structural_dml

J, K, d_w = 3, 2, 3
n = 8000

np.random.seed(42)
W = np.random.normal(0, 1, (n, d_w))

# True parameters (heterogeneous in W[0])
a0, a1 = [0.0, 0.5, -0.3], [0.0, 0.2, -0.1]
b0, b1 = [-0.8, 0.5], [-0.2, 0.1]

alphas = np.column_stack([a0[j] + a1[j] * W[:, 0] for j in range(J)])
betas = np.column_stack([b0[k] + b1[k] * W[:, 0] for k in range(K)])

X_alt = np.random.normal(0, 1, (n, J, K))
V = alphas.copy()
for j in range(J):
    V[:, j] += (X_alt[:, j, :] * betas).sum(axis=1)
probs = softmax(V, axis=1)
Y = np.array([np.random.choice(J, p=probs[i]) for i in range(n)]).astype(float)
T = X_alt.reshape(n, -1)   # pack to (n, J*K)

# mu* = E[beta_1(W)] = b0[0] = -0.8

result = structural_dml(
    Y=Y, T=T, X=W,
    family='multinomial_logit',
    n_alternatives=J,
    n_attributes=K,
    hidden_dims=[64, 32],
    epochs=300,
    patience=50,    # critical: default patience=10 is too aggressive
    n_folds=50,
    lr=0.01,
)
print(result.summary())
```

**New API (`inference()`):**

```python
from deep_inference import inference

result = inference(
    Y=Y, T=T, X=W,
    model='multinomial_logit',
    target='beta',
    n_alternatives=J,
    n_attributes=K,
    patience=50,
    epochs=300,
    n_folds=50,
)
```

**Choice probability and AME targets:**

```python
from deep_inference.targets.choice_probability import ChoiceProbabilityTarget, MultinomialAME

# P(Y=1 | W, X_new)
cp_target = ChoiceProbabilityTarget(alternative=1, n_alternatives=3, n_attributes=2)
result = inference(Y, T, W, model='multinomial_logit', target=cp_target,
                   n_alternatives=3, n_attributes=2, patience=50)

# dP(Y=1)/dx_{1,0}: marginal effect of attribute 0 on P(Y=1)
ame_target = MultinomialAME(alternative=1, attribute=0, n_alternatives=3, n_attributes=2)
result = inference(Y, T, W, model='multinomial_logit', target=ame_target,
                   n_alternatives=3, n_attributes=2, patience=50)
```

## Evidence

From `docs/simulation_studies/eval_09.md` (Eval 09, J=3, K=2, $d_w=3$):

**Parameter recovery (n=10000, single run):**

| Component | RMSE | Correlation | Status |
|-----------|------|-------------|--------|
| $\alpha_1$ | 0.08 | 0.90 | PASS |
| $\alpha_2$ | 0.12 | 0.78 | PASS |
| $\beta_1$ | 0.09 | 0.88 | PASS |
| $\beta_2$ | 0.10 | 0.85 | PASS |

Autodiff validation: score and Hessian match closed-form formulas to machine precision
(max error $4.44 \times 10^{-16}$, PASS).

**Coverage (M=50, n=8000, $\mu^* = -0.8$):**

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 98% | 90-99% | PASS |
| SE Ratio | 0.97 | 0.7-1.5 | PASS |
| Bias | -0.006 | less than 0.05 | PASS |
| z-score mean | 0.14 | (-0.3, 0.3) | PASS |
| z-score std | 0.96 | 0.7-1.5 | PASS |

**EVAL 09: PASS** (source: `docs/simulation_studies/eval_09.md`)

## Diagnostics and Pitfalls

**patience=50 is essential.** The default patience=10 triggers early stopping at roughly
15-20 epochs with three-way splitting, fatal for coverage. Symptom: very high
correction_ratio (above 100) and inflated SE. The `structural_dml` path handles this
automatically; `inference()` requires you to pass `patience=50`.

**n >= 8000 for valid coverage.** Three-way splitting and a higher-dimensional theta
($J-1+K$ parameters instead of 2) each reduce effective sample size. The eval showed
88% coverage at n=5000. Use at least 8000 observations.

**Large correction_ratio is normal.** A correction_ratio in the range 70 to 90 is expected
for multinomial logit, versus roughly 2 for binary logit. This reflects the higher
dimensionality of the problem and is not a sign of failure.

**$\alpha_2$ is the hardest to recover.** In the eval DGP, the slope on $\alpha_2(W)$ is
-0.1 (weak signal). Recovery correlation is 0.78 and n=10000 was needed for reliable
recovery. If your alpha heterogeneity is weak, you may need larger samples.

**Near-singular Lambda.** If choice probabilities are near 0 or 1, the Lambda matrix can
become near-singular and its inverse explodes. Check
`result.diagnostics['min_lambda_eigenvalue']`; it should exceed $10^{-4}$. Add
regularization via `tikhonov_scale` if needed.

## References and API

**Papers:**

- Hetzenecker, S. and Osterhaus, C. (2024). "Deep Learning for the Estimation of
  Heterogeneous Parameters in Discrete Choice Models." *arXiv:2408.09560*. The model
  setup is in Section 2 and the three-way splitting procedure is in Section 2.3.
- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior."
  *Frontiers in Econometrics.*

**API:**

- Legacy: `structural_dml(Y, T, X, family='multinomial_logit', n_alternatives=J, n_attributes=K)`
- New: `inference(Y, T, X, model='multinomial_logit', target='beta', n_alternatives=J, n_attributes=K)`
- Model class: `deep_inference.MultinomialLogit`
  (`src/deep_inference/models/multinomial.py`)
- Legacy family: `deep_inference.MultinomialLogitFamily`
  (`src/deep_inference/families/multinomial.py`)
- Target classes: `ChoiceProbabilityTarget`, `MultinomialAME`
  (`src/deep_inference/targets/choice_probability.py`)
- Simulation study: `evals/eval_09_multinomial.py`, results at
  `docs/simulation_studies/eval_09.md`
