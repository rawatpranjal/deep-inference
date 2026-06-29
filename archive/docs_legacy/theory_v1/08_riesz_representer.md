# 8. The Riesz Representer: Why the Two Procedures Agree

By now you have seen two ways to debias the same estimate. The
[influence-function correction](04_influence_function.md) subtracts
$H_\theta\,\Lambda(X)^{-1}\,\ell_\theta$, built from the model's score and its expected Hessian.
The [RieszNet procedure](../inference/riesznet.md) instead learns a weight $\alpha(Z)$ and
subtracts $\alpha(Z)(Y - g(Z))$. This page explains why these are the same correction, written
two ways, and why that matters in practice.

Read this page after the influence-function and three-regimes pages. It assumes you know what
$\Lambda(X)$ is.

## The debiasing term has a unique form

Fix a target that is linear in the regression function $g$, for example the average treatment
effect $\mu = \mathbb{E}[g(1, X) - g(0, X)]$. A plug-in estimate (just average the fitted
$\hat g$) is biased by regularization. The theory of debiased machine learning says there is a
unique extra term that removes that bias to first order, and it always takes the form

$$
\alpha_0(Z)\,\bigl(Y - g(Z)\bigr),
$$

where $\alpha_0(Z)$ is the **Riesz representer** of the target. The representer is the function
that satisfies $\mathbb{E}[m(W; g)] = \mathbb{E}[\alpha_0(Z)\, g(Z)]$ for every $g$, where
$m(W; g)$ is the moment that defines the target. For the average treatment effect the representer
has a closed form,

$$
\alpha_0(Z) = \frac{T}{p_0(X)} - \frac{1 - T}{1 - p_0(X)},
$$

with $p_0(X) = P(T = 1 \mid X)$ the propensity score. This is the inverse-propensity weight you
may already know from causal inference.

## Two ways to get the same correction

The influence-function and RieszNet procedures differ only in how they produce $\alpha_0$.

```{list-table}
:header-rows: 1

* - Procedure
  - How it forms the correction
* - Influence function
  - Computes $H_\theta\,\Lambda(X)^{-1}\,\ell_\theta$ analytically. With the correct $\Lambda(X)$ this product **is** the representer-weighted residual.
* - RieszNet
  - Learns $\alpha(Z)$ directly with a neural-network head by minimizing the Riesz loss, then forms $\alpha(Z)(Y - g(Z))$.
```

The key fact is that $H_\theta\,\Lambda(X)^{-1}\,\ell_\theta$ and $\alpha_0(Z)(Y - g(Z))$ are the
same object when $\Lambda(X)$ is correct. The score $\ell_\theta$ already contains the residual
$Y - g(Z)$, and the curvature inverse $\Lambda(X)^{-1}$ supplies exactly the overlap weighting
$1 / (p(X)(1 - p(X)))$ that turns that residual into the representer. The correction weight
$(T - e(X)) / (e(X)(1 - e(X)))$, with $e(X)$ the propensity, is algebraically the Riesz
representer for the average effect. Neither procedure is more "correct"; they are two routes to
one quantity.

## Why this matters: the load-bearing object is the curvature, not the model

This equivalence is not just elegant. It tells you where inference can go wrong. Because the two
corrections coincide only when $\Lambda(X)$ is right, **estimating $\Lambda(X)$ well is the whole
game** for the influence-function procedure in the observational regime.

This repository documents exactly this failure and fix. When the covariate-varying curvature
$\Lambda(X)$ is collapsed to a single global-average matrix, the propensity $e(X)$ is replaced by
its mean, the representer is wrong, and the standard error is too small (coverage drops below
nominal). The cure is to estimate $\Lambda(X)$ as a function of $X$ in a way that stays positive
definite and invertible even where overlap is poor, which is what the `cholesky` and `ridge`
Lambda methods do. The [Replications](../replications/index.md) page shows the influence-function
method with a correct $\Lambda(X)$ and RieszNet landing on the same coverage, family by family,
because they are estimating the same correction.

## What RieszNet buys you

If the two agree, why have both? Because RieszNet sidesteps the hard step. It never forms
$\Lambda(X)^{-1}$, which is the part that goes unstable at low overlap. It learns $\alpha(Z)$
directly from the Riesz loss, so it needs no analytic Hessian and no matrix inversion. That makes
it a useful second opinion: when the influence-function interval and the RieszNet interval agree,
you can be more confident the curvature was estimated well. When they disagree, the
[Replications](../replications/index.md) page is the place to see which one to trust for a given
outcome family (RieszNet is unreliable on heavy-tailed counts, for instance).

## References

See [References](../references/index.md) for the RieszNet paper (Chernozhukov, Newey,
Quintas-Martinez, Syrgkanis 2022), the adversarial Riesz learning paper, and the FLM framework
papers. The propensity-collapse diagnosis is recorded in the project notes and summarized on the
[Replications](../replications/index.md) page.
