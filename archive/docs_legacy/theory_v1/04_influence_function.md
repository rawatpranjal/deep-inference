# 4. The Influence Function Correction

The influence function (IF) correction of Farrell, Liang, and Misra (2025,
Theorem 2) removes the regularization bias derived on the
[previous page](03_why_naive_fails.md). For each observation $i$, define

$$
\psi_i = H_i - H_{\theta,i} \cdot \Lambda_i^{-1} \cdot \ell_{\theta,i},
$$

where

- $H_i = H(X_i, \hat{\theta}(X_i), \tilde{t})$ is the target evaluated at the estimated parameters,
- $H_{\theta,i} = \left.\partial H / \partial \theta\right|_{\hat{\theta}(X_i)} \in \mathbb{R}^{1 \times d_\theta}$ is the **target Jacobian**,
- $\Lambda_i = \left.\mathbb{E}[\nabla_\theta^2 \ell(Y, T, \theta) \mid X = X_i]\right|_{\hat{\theta}(X_i)} \in \mathbb{R}^{d_\theta \times d_\theta}$ is the **expected Hessian** of the loss conditional on covariates,
- $\ell_{\theta,i} = \nabla_\theta \ell(Y_i, T_i, \hat{\theta}(X_i)) \in \mathbb{R}^{d_\theta}$ is the **score** (gradient of the loss at the observed data point).

The debiased estimator and its standard error are

$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^n \psi_i, \qquad
\widehat{\text{SE}} = \frac{\text{sd}(\psi_1, \ldots, \psi_n)}{\sqrt{n}}.
$$

## Intuition

The correction term $H_{\theta} \Lambda^{-1} \ell_\theta$ is an "adjustment for
what the neural net got wrong."

- The score $\ell_\theta$ measures how far observation $i$'s data is from the fitted model.
- The Hessian $\Lambda$ converts this into parameter-space error.
- The Jacobian $H_\theta$ maps this to the target.

When the neural net fits perfectly (scores near zero), the correction vanishes.
When it makes systematic errors (regularization bias), the correction kicks in.

## Neyman orthogonality

The influence function $\psi$ satisfies the **Neyman orthogonality** condition:

$$
\left.\frac{\partial}{\partial \theta} \mathbb{E}[\psi(Y, T, \theta, \Lambda)]\right|_{\theta = \theta^*,\, \Lambda = \Lambda^*} = 0.
$$

The correction term $H_\theta \Lambda^{-1} \ell_\theta$ is constructed precisely so
that first-order perturbations in $\hat{\theta}$ around $\theta^*$ cancel. This is
why the debiased estimator $\hat{\mu} = \bar{\psi}$ achieves $\sqrt{n}$-consistency
**despite the neural network's slow convergence rate**: the estimation error in
$\hat{\theta}$ enters only through a second-order remainder, which is negligible at
the $\sqrt{n}$ scale. (The formal rate conditions are stated on the
[Theorems and Convergence Rates](influence_functions.md) page.)

## Connection to classical statistics

The IF $\psi$ is a semiparametric influence function in the tradition of Hampel
(1974) and Newey (1994). The specific form arises from the "one-step correction"
or "debiasing" approach used widely in semiparametric inference (Chernozhukov et
al., 2018). The Neyman orthogonality property is central to the broader program of
orthogonal statistical learning (Foster & Syrgkanis, 2023); related automatic
debiasing approaches include RieszNet (Chernozhukov et al., 2022). The
contribution of Farrell, Liang, and Misra (2021, 2025) is showing that this
correction remains valid when the first-stage nuisance (the structural parameters)
is estimated by a **deep neural network**.
