# Theory: RieszNet and Automatic Debiasing

```{note}
This is the second theory track, the **automatic-debiasing** route of Chernozhukov,
Newey, Quintas-Martinez, and Syrgkanis (RieszNet, 2022) and the adversarial Riesz work
of Chernozhukov, Newey, and Singh (2020). It is a separate path to the same correction
the [FLM page](flm.md) builds by hand. Read the FLM page first. This page is the **why**,
not the how. For running RieszNet in the package see [Inference](../inference/index.md),
for the pipeline see [Estimation](../estimation/index.md), and for the papers see
[References](../references/index.md).
```

The FLM track corrects regularization bias with a term assembled from the model's score
and its expected Hessian, $-H_\theta\,\Lambda(x)^{-1}\,\ell_\theta$. The automatic-debiasing
track reaches the same correction from the other direction. It writes the bias correction
as $\alpha(Z)\,(Y - g(Z))$, where $\alpha$ is a single learned function called the Riesz
representer, and then learns $\alpha$ directly from a loss, never differentiating the model
and never inverting a matrix. The last section shows the two corrections are the same object.
The payoff of having both is practical. The FLM route can go unstable exactly where the
representer route does not, and where the two intervals agree you can trust the answer.

The running example is the same one that closes the FLM page, the average treatment effect,
because there the representer is the inverse-propensity weight everyone knows. The logit
marginal effect and the linear slope are the same story written with a different functional.

---

## 1. Averages of a regression as linear functionals

The setting is a target that is an average of a regression. Let $g_0(Z) = \mathbb{E}[Y \mid Z]$
be a regression function, where $Z$ collects the treatment and the covariates, and let the
parameter of interest be

$$
\theta_0 = \mathbb{E}\big[m(W; g_0)\big],
$$

where $m(W; g)$ is a moment functional that is **linear in $g$**. The linearity is the only
structural requirement, and it covers the objects the package targets.

- **Average treatment effect.** $m(w; g) = g(1, z) - g(0, z)$, so $\theta_0$ is the mean
  contrast of the regression at treatment one versus zero.
- **Average derivative (and the logit AME).** $m(w; g) = \partial g(t, z)/\partial t$ averaged
  over a weight, so $\theta_0$ is a mean marginal effect.

These match the FLM targets when the structural parameter is written through the conditional
mean. The average parameter $\mathbb{E}[\beta]$ in the linear binary-treatment model is the
average treatment effect, and the logit average marginal effect is an average derivative of the
choice probability.

---

## 2. The Riesz representer

Because $g \mapsto \mathbb{E}[m(W; g)]$ is linear in $g$, the Riesz representation theorem
applies. If the functional is also continuous in mean square, that is
$\mathbb{E}[m(W; g)] \le C\,\|g\|$ with $\|g\| = \sqrt{\mathbb{E}[g(Z)^2]}$, then there exists a
unique square-integrable function $\alpha_0(Z)$ such that

$$
\mathbb{E}\big[m(W; g)\big] = \mathbb{E}\big[\alpha_0(Z)\,g(Z)\big] \qquad \text{for every } g.
$$

This $\alpha_0(Z)$ is the **Riesz representer** of the target. Two facts about it are
load-bearing. First, the representer has a known closed form in the standard cases even though
we will not need that form to estimate it.

- **Average treatment effect.** $\displaystyle \alpha_0(Z) = \frac{T}{p_0(X)} - \frac{1 - T}{1 - p_0(X)}$,
  where $p_0(X) = \Pr(T = 1 \mid X)$ is the propensity score. This is the inverse-propensity
  weight. For binary $T$ it simplifies to $\alpha_0(Z) = (T - p_0(X)) / \{p_0(X)(1 - p_0(X))\}$.
- **Average derivative.** $\displaystyle \alpha_0(Z) = -\,\frac{\partial f_0(T, X)/\partial t}{f_0(Z)}$,
  the negative log-density derivative, where $f_0$ is the joint density.

Second, the representer exists if and only if $\theta_0$ has a finite semiparametric variance
bound. So assuming $\alpha_0$ exists is the same as assuming the question is answerable at the
$\sqrt{n}$ rate at all. CNSS (2020) note the existence is "equivalent to the semiparametric
variance bound for $\theta_0$ being finite." The overlap condition shows up here too. For the
average treatment effect the representer is square-integrable only if
$\mathbb{E}[\{p_0(Z)(1 - p_0(Z))\}^{-1}] < \infty$, the same $p(1-p)$ in the denominator that
appears as $\det \Lambda(x)$ on the FLM page.

---

## 3. Why the debiasing term is $\alpha(Z)(Y - g(Z))$

The plug-in estimator $\mathbb{E}_n[m(W; \hat g)]$ inherits the regularization bias of $\hat g$,
the same first-order plug-in bias the FLM page diagnoses. Debiased machine learning removes it
by adding one term to the moment, and the representer is exactly what that term is built from.
The debiased (orthogonal) score is

$$
\psi(W; g, \alpha, \theta) = m(W; g) + \alpha(Z)\big(Y - g(Z)\big) - \theta.
$$

The added piece $\alpha(Z)(Y - g(Z))$ is the representer weighting the regression residual.
CNSS (2020) derive it as the nonparametric influence function of $\mathbb{E}[m(W; g)]$, added to
the identifying moment, and show the result is orthogonal "model free." The construction does
not assume $\hat g$ equals the true conditional mean, only that it lands in the function class.

### The mixed-bias property and double robustness

The reason this particular term works is the **mixed bias property**. Evaluate the score at the
truth $\theta_0$ but at arbitrary first-stage functions $g$ and $\alpha$, and the bias factors,

$$
\mathbb{E}\big[\psi(W; g, \alpha, \theta_0)\big]
= -\,\mathbb{E}\big[\big(\alpha(Z) - \alpha_0(Z)\big)\big(g(Z) - g_0(Z)\big)\big].
$$

The bias is the inner product of the two estimation errors. Three things follow. First, it is
zero whenever **either** $\alpha = \alpha_0$ **or** $g = g_0$, which is double robustness. Second,
its size is bounded by the product $\|\hat\alpha - \alpha_0\|\,\|\hat g - g_0\|$ of the two
mean-square errors, so if the regression is learned well the representer can be learned loosely,
and the reverse. Third, whenever that product converges faster than $1/\sqrt{n}$,

$$
\sqrt{n}\big(\hat\theta - \theta_0\big) \xrightarrow{d} N\!\big(0,\ \mathbb{E}[\psi_0(W)^2]\big),
$$

and $\mathbb{E}[\psi_0(W)^2]$ is the semiparametric efficiency bound. This is the same
product-rate, Neyman-orthogonality argument as FLM Theorem 2, restated for the representer
parameterization. The two factors needing $o_P(n^{-1/4})$ each is the typical sufficient case.
Cross-fitting (and optionally double cross-fitting, with $\hat g$ and $\hat\alpha$ on different
folds) keeps the own-observation bias out, exactly as on the FLM page.

---

## 4. Learning the representer without a formula: the Riesz loss

The closed forms in Section 2 are nice but fragile. Plug-in inverse-propensity weighting divides
by $\hat p(1 - \hat p)$, which is unstable when fitted propensities approach $0$ or $1$. The
representer route avoids the division by learning $\alpha_0$ as the minimizer of a loss that never
references the analytic form. The key result (Chernozhukov et al., 2021) is that the representer
minimizes

$$
L(\alpha) = \mathbb{E}\big[\alpha(Z)^2 - 2\,m(W; \alpha)\big].
$$

Why this loss recovers $\alpha_0$ is a one-line completion of the square. Add the constant
$\mathbb{E}[\alpha_0(Z)^2]$, which does not depend on $\alpha$, and use the defining identity
$\mathbb{E}[m(W; \alpha)] = \mathbb{E}[\alpha_0\,\alpha]$ from Section 2,

$$
L(\alpha) + \mathbb{E}\big[\alpha_0(Z)^2\big]
= \mathbb{E}\big[\alpha(Z)^2 - 2\,\alpha_0(Z)\,\alpha(Z) + \alpha_0(Z)^2\big]
= \mathbb{E}\big[\big(\alpha_0(Z) - \alpha(Z)\big)^2\big].
$$

So minimizing $L$ is, up to a constant offset, minimizing the square error to the true representer.
The benefits are twofold, as the authors put it. There is no need to derive an analytic form, and
the estimate trades off bias and variance against the representer itself rather than against an
intermediate propensity or density that then gets inverted. The only thing the loss needs is the
ability to evaluate the moment $m(W; \alpha)$ on candidate functions, which the user already
supplies by naming the target. For the average treatment effect, $m(W; \alpha) = \alpha(1, X) - \alpha(0, X)$,
so the loss is computed entirely from forward evaluations of $\alpha$, with no propensity in a
denominator anywhere.

The adversarial Riesz line (CNSS, 2020) gives the same loss its minimum-distance form. With a
dictionary $b(x)$, let $M = \mathbb{E}[m(W; b)]$ and $G = \mathbb{E}[b\,b']$, and learn linear
coefficients $\rho$ for $\alpha = b'\rho$ by minimizing $\rho' G \rho - 2 M'\rho$ plus an $\ell_1$
penalty. Expanding $\mathbb{E}[\alpha^2] = \rho' G \rho$ and $\mathbb{E}[m(W; \alpha)] = M'\rho$
shows this is exactly the empirical $L(\alpha)$ restricted to linear functions. RieszNet replaces
the dictionary with a neural network. The statistical guarantee is a critical-radius bound
(RieszNet Theorem 2.1, from Chernozhukov et al., 2021), which yields fast $L_2$ rates for the
representer over neural-network and forest function classes, the rates the product-bound of
Section 3 then consumes.

---

## 5. Targeted regularization

There is a second device that makes the plug-in estimate itself debiased, so the correction term
vanishes by construction rather than only in expectation. Augment the regression with the
representer as an extra, unpenalized input,

$$
\tilde g(Z) = g(Z) + \varepsilon\,\alpha(Z),
$$

where $\varepsilon$ is a single scalar that is **not** penalized. Fit $\varepsilon$ by least
squares along with everything else. Its first-order condition is

$$
\frac{\partial}{\partial \varepsilon}\,\mathbb{E}_n\big[(Y - \tilde g(Z))^2\big] = 0
\quad\Longrightarrow\quad
\mathbb{E}_n\big[(Y - \tilde g(Z))\,\alpha(Z)\big] = 0.
$$

But $\mathbb{E}_n[(Y - \tilde g)\,\alpha]$ is precisely the debiasing correction. So with the
corrected regression $\tilde g$, the correction is **identically zero in sample**, and the plug-in
average $\mathbb{E}_n[m(W; \tilde g)]$ already equals the doubly robust estimate. This is the idea
of Bang and Robins (2005) and the targeted maximum likelihood framework, where a least-favorable
direction is added to the regression and its coefficient zeroes the score. RieszNet automates the
choice of that direction. The direction is the learned representer $\alpha$, fit jointly rather
than in a post-processing step, so no case-by-case derivation of an efficient influence function is
needed.

---

## 6. The multitask architecture

RieszNet packs the regression and the representer into one network with a shared trunk. The
motivation is Lemma 3.1 of the paper. To estimate the average moment of $g_0$ it suffices to
estimate a regression that conditions only on the value of the representer,
$g_0(Z) = h_0(\alpha_0(Z))$. This generalizes the classic fact that for the average treatment
effect it is enough to condition on the propensity score and treatment (Rosenbaum and Rubin, 1983),
the same insight Dragonnet (Shi et al., 2019) used. So a feature representation rich enough to
express the representer is rich enough for the regression too.

```{mermaid}
flowchart TD
    Z["Input Z = (T, X)"] --> Trunk["Shared representation f_1(Z; w_1:k)<br/>(k hidden layers)"]
    Trunk --> RR["Riesz head<br/>alpha(Z) = inner product of f_1 and beta"]
    Trunk --> Reg["Regression head<br/>g(Z) = f_2(f_1; w_(k+1):d)"]
    Reg --> Tilde["Targeted layer<br/>g_tilde = g + eps * alpha<br/>(eps unpenalized)"]
    RR --> Loss["Combined loss"]
    Reg --> Loss
    Tilde --> Loss
    Loss --> Out["L_sq + lambda_1 * L_Riesz + lambda_2 * L_targeted + R(w)"]
```

The trunk $f_1(Z; w_{1:k})$ is shared. The Riesz head is linear in the final features,
$\alpha(Z) = \langle f_1(Z; w_{1:k}), \beta\rangle$, and is trained by the Riesz loss of Section 4.
The regression head $g(Z) = f_2(f_1(Z; w_{1:k}); w_{(k+1):d})$ branches off the same trunk and is
trained by square loss, and a targeted layer adds $\varepsilon\,\alpha(Z)$. Because the trunk feeds
both heads, the representer loss shapes the features the regression uses, and the regression loss
in turn lets the trunk explain more of $Y$ even when the representer is nearly constant. The full
objective is

$$
\mathbb{E}_n\big[(Y - g(Z))^2\big]
\; + \; \lambda_1\,\mathbb{E}_n\big[\alpha(Z)^2 - 2\,m(W; \alpha)\big]
\; + \; \lambda_2\,\mathbb{E}_n\big[(Y - \tilde g(Z))^2\big]
\; + \; R(w),
$$

where $R$ is a weight penalty that crucially does not touch $\varepsilon$. The three pieces are the
regression square loss, the Riesz loss, and the targeted-regularization loss. As $\lambda_1 \to 0$
the second term drops to the one-step Bang and Robins form, and as $\lambda_2 \to 0$ the model
approaches the two-step TMLE behavior where $\varepsilon$ is fit after a fixed regression.

---

## 7. The bridge to FLM: one correction, two routes

The two tracks compute the same correction. The clean way to see it is the binary-treatment
average treatment effect, the example that closes the [FLM page](flm.md). There the FLM correction
was computed to be

$$
- H_\theta\,\Lambda(x)^{-1}\,\ell_\theta = \frac{T - p(x)}{p(x)(1 - p(x))}\,\big(Y - g_0(Z)\big),
$$

with $g_0 = \alpha + \beta t$ and $\det \Lambda(x) = p(x)(1 - p(x))$. The weight
$(T - p)/\{p(1-p)\}$ is exactly the Riesz representer $\alpha_0(Z)$ of the average treatment effect
from Section 2. So the FLM correction reads

$$
\psi = \beta^\star + \alpha_0(Z)\big(Y - g_0(Z)\big) = m(W; g_0) + \alpha_0(Z)\big(Y - g_0(Z)\big),
$$

which is identically the RieszNet orthogonal score of Section 3. The algebra was checked term by
term against FLM Appendix C.1. The takeaway is mechanical. The score $\ell_\theta = T_1(g_0 - Y)$
already carries the regression residual $Y - g_0$, and the curvature inverse $\Lambda(x)^{-1}$
supplies the overlap weighting $1/\{p(1-p)\}$ that turns that residual into the representer.
Differentiating the model and inverting the Hessian (FLM) and learning a weight from the Riesz loss
(RieszNet) are two routes to the one quantity $\alpha_0(Z)(Y - g(Z))$. Neither is more correct.

### Why this matters: the load-bearing object is the curvature inverse

The equivalence holds only when $\Lambda(x)$ is right, and that is where inference can break. In the
observational, nonlinear regime (the logit case, Regime C on the [FLM page](flm.md)) $\Lambda(x)$ is
estimated by regressing the per-observation Hessian on $X$ and then inverted. The inverse is the
fragile step. At low overlap $\det \Lambda(x) = p(x)(1 - p(x)) \to 0$, so a small error in $\Lambda$
becomes a large error in $\Lambda^{-1}$, and a regressor can fit $\Lambda$ well while its inverse
explodes.

This repository documents the concrete failure. When the covariate-varying curvature $\Lambda(x)$ is
collapsed to a single global-average matrix (the two-way path that replaces the propensity $e(x)$
with its mean $\bar e$), the implied representer is wrong, the influence values are too tightly
clustered, the standard error comes out too small, and coverage falls below nominal. Injecting the
correct $\Lambda(x)$ into the pipeline restores the variance ratio to one. That is the same statement
as the bridge above, read in reverse. Get the curvature wrong and you get the representer wrong. The
diagnosis and the fix (estimate the scalar propensity directly and form $\Lambda(x)$ analytically,
or use a $\Lambda$ estimator that stays invertible) are summarized on the
[Replications](../replications/index.md) page.

### What RieszNet buys you

RieszNet never forms $\Lambda(x)^{-1}$. It learns $\alpha(Z)$ from the Riesz loss, a
well-conditioned minimization with no analytic Hessian and no matrix inversion, so the unstable step
is simply absent. That makes it a useful second opinion rather than a replacement. When the
influence-function interval and the RieszNet interval agree, the curvature was estimated well and the
answer is trustworthy. When they disagree, the [Replications](../replications/index.md) page is the
place to see which to trust for a given outcome family, since RieszNet has its own weak spots (it is
unreliable on heavy-tailed counts, for instance).

---

## Where to go next

- [Inference](../inference/index.md) for how to run RieszNet in the package and read its result.
- [Estimation](../estimation/index.md) for the cross-fitting pipeline both procedures share.
- [The FLM Framework](flm.md) for the score-and-Hessian route this page is the dual of.
- [References](../references/index.md) for the RieszNet, adversarial Riesz, and FLM papers.
