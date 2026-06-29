# Combinatorial Multi-Treatment Model

The combinatorial model estimates the causal effect of any bundle of binary treatments
when you have only observed a subset of all possible bundles. Read this page if you
run multiple simultaneous experiments (A/B tests, feature rollouts) and want to infer
the effect of combinations you have not directly measured.

## Source Papers

Ye, Z., Zhang, Z., Zhang, D. J., Zhang, H., and Zhang, R. (2025). "Deep Learning Based
Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence."
*Management Science*. This paper is the direct source for the DeDL (debiased deep learning)
framework implemented here. The structural link functions, the influence function (Proposition
2 and Equation 5 in the paper), and the Lambda formula
$\Lambda(x) = 2E[G_\theta(\theta(x),T)\,G_\theta(\theta(x),T)' \mid X=x]$ all come from
this paper. The abstract states the goal: "Without observing the outcomes of all treatment
combinations, how to estimate the causal effect of any treatment combination and identify
the optimal treatment combination?"

The influence-function framework follows Farrell, Liang, Misra (2021), "Deep Neural
Networks for Estimation and Inference," *Econometrica*.

## When to Use

Use this model when:

- You have $m$ binary treatments ($T \in \{0,1\}^m$) and want to estimate the causal
  effect of any specific bundle $t$ relative to a control $t_0$.
- You have not observed (or cannot observe) all $2^m$ possible treatment combinations.
  For example, with $m=3$ treatments you have $2^3=8$ possible bundles but may have
  only run 4 of them.
- You believe the outcome follows a structured functional form that depends on
  treatments through a known link function $G(\theta, t)$.
- Platform or e-commerce context: users are assigned bundles of features (push
  notifications, UI changes, price promotions) and you want to estimate the average
  effect of applying any bundle.

Do not use this model when treatments are continuous. For a single binary or continuous
treatment, the logit or linear model is simpler. If all $2^m$ combinations are observed
and you have no structural assumption, a fully non-parametric approach may be more
reliable. The model is specifically designed for the partial-factorial regime where the
structural assumption is needed for extrapolation to unobserved bundles.

## Model

The outcome follows a structural regression:

$$Y_i = G(\theta(X_i),\; T_i) + \varepsilon_i, \qquad E[\varepsilon_i \mid X_i, T_i] = 0$$

where $T_i \in \{0,1\}^m$ is the vector of binary treatment indicators,
$\theta(X_i) \in \mathbb{R}^{d_\theta}$ are covariate-varying structural parameters
learned by the neural network, and $G$ is a known link function chosen by the researcher.

**Why not assume linear additivity?** A naive approach approximates the bundle effect as
the sum of individual effects: $\mu(t) \approx \sum_k \tau_k t_k$. This misses
interactions: the joint effect of two treatments can be larger or smaller than the sum of
their individual effects. Ye et al. (2025) demonstrate on a large video-sharing platform
that DeDL "significantly outperforms other benchmarks to accurately estimate and infer the
average treatment effect (ATE) of any treatment combination" when interactions are present.
The structural link function $G$ captures these interactions through its nonlinear form.

**Four link functions.** The package implements the four specifications from DeDL2025
(read from `src/deep_inference/models/combinatorial.py`):

| Name | Formula for $G(\theta, t)$ | $\theta$ layout | $d_\theta$ |
|------|---------------------------|-----------------|------------|
| `multiplicative` | $\theta_0\,\prod_{k=1}^m (1 + \theta_k t_k)$ | $[\theta_0, \theta_1, \ldots, \theta_m]$ | $m+1$ |
| `sigmoid` | $a\,\sigma(\theta_0 + \sum_k \theta_k t_k) + b$ | $[\theta_0, \theta_1, \ldots, \theta_m]$ | $m+1$ |
| `gen_sigmoid_i` | $\theta_{m+1}\,\sigma(\sum_k \theta_k t_k)$ (no intercept) | $[\theta_1,\ldots,\theta_m,\,\theta_{m+1}]$ | $m+1$ |
| `gen_sigmoid_ii` | $\theta_{m+1}\,\sigma(\theta_0 + \sum_k \theta_k t_k)$ | $[\theta_0,\theta_1,\ldots,\theta_m,\,\theta_{m+1}]$ | $m+2$ |

`gen_sigmoid_ii` is the default and recommended choice. It is the most flexible: the
scale parameter $\theta_{m+1}$ and the intercept $\theta_0$ give two degrees of freedom
beyond the treatment coefficients. `multiplicative` is natural when effects multiply
(e.g., demand shifts). `sigmoid` is appropriate when the outcome is bounded. `gen_sigmoid_i`
lacks an intercept in the exponent, making it less flexible than `gen_sigmoid_ii` with no
benefit in practice.

## Loss / Likelihood

The loss is mean squared error:

$$\ell(y, t, \theta) = (y - G(\theta, t))^2$$

This is a regression loss. The link function $G$ supplies the structural restriction; the
neural network learns the parameter vector $\theta(X)$ that indexes $G$, not $G$ itself.

## Score and Hessian

**Score.** The score $\ell_\theta = -2(y - G(\theta, t))\,G_\theta(\theta, t)$ is
computed via PyTorch autodiff (the `score()` method returns `None`, signaling the engine
to use autodiff). The closed form varies by link function, so autodiff is the clean path.

**Hessian.** Expanding the second derivative:

$$\ell_{\theta\theta} = 2\,G_\theta G_\theta^\top - 2(y - G)\,G_{\theta\theta}$$

Under correct specification, $E[y - G(\theta^*(X), T) \mid X, T] = 0$, so the second
term has zero expectation. The expected Hessian is therefore

$$\Lambda(x) = 2\,E\bigl[G_\theta(\theta(x), T)\,G_\theta(\theta(x), T)^\top \;\big|\; X=x\bigr]$$

This matches Ye et al. (2025, Equation read from paper): "$\Lambda(x) := 2E[G_\theta(\theta(x),T)G_\theta(\theta(x),T)'\mid X=x]$."

The implementation returns $2G_\theta G_\theta^\top$ as the per-observation Hessian, which
is the Fisher information for regression models. It does not depend on $y$
(`hessian_depends_on_y=False`), but it does depend on $\theta$ through $G_\theta$
(`hessian_depends_on_theta=True`).

## Target

The target is the average treatment effect of a bundle $t$ relative to a control $t_0$:

$$\mu^*(t) = E[G(\theta(X), t) - G(\theta(X), t_0)]$$

The control $t_0$ defaults to $\mathbf{0}$ (no treatment). For example, with $m=3$
treatments, `treatment=[1,1,0]` estimates the average effect of activating treatments
1 and 2 simultaneously versus no treatment.

The target Jacobian is

$$H_\theta(x, \theta) = G_\theta(\theta(x), t) - G_\theta(\theta(x), t_0)$$

This matches Ye et al. (2025): "$H_\theta(x,\theta(x);t,t_0) := G_\theta(\theta(x),t) - G_\theta(\theta(x),t_0)$."
It is computed via autodiff.

## Influence Function

The influence function follows Ye et al. (2025, Proposition 2, Equation 5). For target
$\mu^*(t) = E[G(\theta(X),t) - G(\theta(X),t_0)]$:

$$\psi_i = H(\hat\theta(X_i)) - H_\theta(X_i, \hat\theta(X_i))^\top\, \Lambda(X_i)^{-1}\, \ell_\theta(Y_i, T_i, \hat\theta(X_i))$$

where the plug-in $H(\hat\theta(X_i)) = G(\hat\theta(X_i),t) - G(\hat\theta(X_i),t_0)$ is the
per-observation bundle contrast, $H_\theta$ is the target Jacobian above, and
$\ell_\theta = 2G_\theta(\theta,T)(G(\theta,T)-y)$ is the MSE score. The correction enters with a
minus sign. Ye et al. call this the "debiasing term" because it removes the regularization bias
introduced by the neural network.

The point estimate is $\hat\mu = (1/n)\sum_i \psi_i$ and the standard error is
$\mathrm{sd}(\psi)/\sqrt{n}$.

## Algorithm and Regime

Because `hessian_depends_on_theta=True` (through $G_\theta$), the regime depends on
whether treatments are randomly assigned.

**Regime A (randomized experiment).** In the DeDL2025 platform A/B test setting,
treatment combinations are randomly assigned. Lambda can then be computed by Monte Carlo
integration without any additional data split:

$$\hat{\Lambda}(x) = \frac{2}{M} \sum_{s=1}^{M} G_\theta(\hat\theta(x), T_s)\, G_\theta(\hat\theta(x), T_s)^\top, \quad T_s \sim \nu(\cdot \mid x)$$

This is the `compute_lambda_integral()` method in `CombinatorialModel`. Pass
`is_randomized=True` and `treatment_dist=...` to `inference()`. Two-way cross-fitting
applies (no extra data split for Lambda).

**Regime C (observational data).** If treatment assignment is observational, Lambda must
be estimated on a separate data split via three-way cross-fitting, the same procedure as
the multinomial logit.

For the platform A/B testing context (the main motivation of DeDL2025), Regime A is the
appropriate choice.

## Usage

```python
import numpy as np
import torch
from deep_inference import inference, CombinatorialModel
from deep_inference.targets.multi_treatment import MultiTreatmentATE

np.random.seed(42)
n = 5000
m = 3  # three binary treatments

# Covariates
X = np.random.normal(0, 1, (n, 2))

# Randomly assign treatment combinations (Regime A)
T = np.random.randint(0, 2, (n, m)).astype(float)

# True parameters: gen_sigmoid_ii with known theta
theta_true_np = np.array([0.5, 1.0, 0.8, 0.6, 2.0])  # [th0, th1, th2, th3, scale]
model = CombinatorialModel(n_treatments=m, link='gen_sigmoid_ii')
theta_t = torch.tensor(theta_true_np, dtype=torch.float32)

G_vals = np.array([
    model.G(theta_t, torch.tensor(T[i], dtype=torch.float32)).item()
    for i in range(n)
])
Y = G_vals + np.random.normal(0, 0.3, n)

# Target: ATE of bundle [1,1,0] vs control [0,0,0]
target = MultiTreatmentATE(model=model, treatment=[1, 1, 0], control=[0, 0, 0])

result = inference(
    Y=Y, T=T, X=X,
    loss=model.loss,
    target_fn=target.h,
    theta_dim=model.theta_dim,
    hessian_depends_on_theta=True,
    hessian_depends_on_y=False,
    is_randomized=True,   # platform A/B: treatment is randomly assigned
    n_folds=20,
    epochs=200,
    patience=50,
    lr=0.01,
)
print(f"ATE [1,1,0] vs [0,0,0]: {result.mu_hat:.4f} +/- {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

**Multiple targets (one run, multiple bundles):**

```python
bundles = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,1,1]]
for bundle in bundles:
    target_b = MultiTreatmentATE(model=model, treatment=bundle, control=[0,0,0])
    res = inference(Y, T, X, loss=model.loss, target_fn=target_b.h,
                    theta_dim=model.theta_dim,
                    hessian_depends_on_theta=True, hessian_depends_on_y=False,
                    is_randomized=True, n_folds=20, epochs=100, patience=50)
    print(f"  {bundle}: ATE={res.mu_hat:.4f} se={res.se:.4f}")
```

**Using the multiplicative link:**

```python
model_mult = CombinatorialModel(n_treatments=3, link='multiplicative')
# theta_dim = m+1 = 4, layout [theta0, theta1, theta2, theta3]
# G = theta0 * (1 + theta1*t1) * (1 + theta2*t2) * (1 + theta3*t3)
```

## Evidence

From `evals/reports/eval_11_20260208_120420.txt` (Eval 11):

| Part | Test | Result | Status |
|------|------|--------|--------|
| Link functions | All four links match oracle (max error $< 3.8 \times 10^{-7}$) | 4/4 | PASS |
| Oracle ATEs | MC integral matches analytic for 5 bundles (max error $< 2.5 \times 10^{-4}$) | 5/5 | PASS |
| ATE Jacobians | Max error 0.051 (threshold 0.01) | 0/3 | FAIL |
| Lambda integral | Error 0.00, min eigenvalue 0.007, shape $(5 \times 5)$ | 3/3 | PASS |

**EVAL 11: FAIL** due to Part 3 (Jacobian accuracy). The link function computation,
oracle ATEs, and Lambda integral are all validated and passing. The ATE Jacobian
$H_\theta = G_\theta(\theta,t) - G_\theta(\theta,t_0)$ has a maximum error of roughly
0.05 against the expected value, exceeding the 0.01 threshold. Frequentist coverage
(end-to-end Monte Carlo) has not yet been benchmarked in this repository.

Until the Jacobian issue is resolved, treat the debiasing correction for this model as
provisional. The point estimates (plug-in $G(\theta,t) - G(\theta,t_0)$) are unaffected;
it is the variance estimate and SE that may be inaccurate.

## Diagnostics and Pitfalls

**Jacobian accuracy (known open issue).** The ATE Jacobian autodiff path has a max error
of roughly 0.05 in Eval 11. This enters the influence function correction directly. An
inaccurate $H_\theta$ biases the debiasing term and can shift coverage. Monitor
`result.diagnostics` for unexpectedly large correction_ratio or SE.

**Link function choice.** `gen_sigmoid_ii` (the default) is the most flexible because it
has both an intercept $\theta_0$ in the exponent and a separate scale $\theta_{m+1}$.
`gen_sigmoid_i` has no intercept and is strictly less expressive; prefer `gen_sigmoid_ii`.
`multiplicative` is the right choice when the economic model implies that each treatment
multiplies the baseline outcome (e.g., each promotion multiplies demand by a factor).

**Near-singular Fisher Hessian.** The Hessian $2G_\theta G_\theta^\top$ is rank-1 when
$\theta$ is a scalar, and can be near-singular when some $G_\theta$ components are near
zero (e.g., when $\theta_{m+1} \cdot \sigma'(\cdot)$ is very small at saturation).
Check `result.diagnostics['min_lambda_eigenvalue']`; it should exceed $10^{-4}$.

**Partial factorial design.** The structural assumption is what makes the model useful in
the partial factorial regime: it provides the parametric restriction needed to extrapolate
to unobserved bundles. If you observe all $2^m$ combinations, a non-parametric estimator
(one estimate per bundle) is more reliable. The structural model is best when you have
good subject-matter knowledge of the link function.

**Regime A only with known treatment distribution.** The `compute_lambda_integral` path
requires that you know the distribution $\nu(T \mid X)$ from which treatment bundles are
drawn. In a randomized platform experiment this is known by design. In observational data
you do not know it, and must fall back to Regime C (three-way split, estimated Lambda).

## References and API

**Paper:**

- Ye, Z., Zhang, Z., Zhang, D. J., Zhang, H., and Zhang, R. (2025). "Deep Learning
  Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical
  Evidence." *Management Science*. Proposition 2 and Equation 5 give the DeDL influence
  function. The Lambda formula is in the body of the paper where $\Lambda(x) := 2E[G_\theta G_\theta' \mid X=x]$ is derived.

**API:**

- Model class: `deep_inference.CombinatorialModel`
  (`src/deep_inference/models/combinatorial.py`)
- Target class: `deep_inference.MultiTreatmentATE`
  (`src/deep_inference/targets/multi_treatment.py`)
- `inference(Y, T, X, loss=model.loss, target_fn=target.h, theta_dim=model.theta_dim, hessian_depends_on_theta=True, hessian_depends_on_y=False)`
- Eval: `evals/eval_11_combinatorial.py`, report at
  `evals/reports/eval_11_20260208_120420.txt`
