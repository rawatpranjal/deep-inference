# 8. Theorems and Convergence Rates

```{note}
This is the final, formal page of the theory section. The earlier pages build the
intuition; this page states the two main guarantees, the neural network
convergence rate (Theorem 1) and the asymptotic normality of the debiased
estimator (Theorem 2), together with the supporting applications.
```

## Notation recap

Recall from [Setup](01_setup.md) and [Target Functionals](02_targets.md): the
enriched structural model makes the parameters functions of covariates,

$$
\theta^\star(\cdot) = \arg\min_{\theta \in \mathcal{F}} \mathbb{E}[\ell(Y, T, \theta(X))],
$$

and the second-stage parameter of interest is the population summary

$$
\mu^\star = \mathbb{E}[H(X, \theta^\star(X), \tilde{t})].
$$

## Theorem 1: Convergence rate of the network

The parameter functions are estimated by

$$
\hat{\theta}(\cdot) = \arg\min_{\theta \in \mathcal{F}_{\text{dnn}}} \frac{1}{n} \sum_{i=1}^{n} \ell(y_i, t_i, \theta(x_i)).
$$

For smooth parameter functions with $p$ derivatives and $d_c$ continuous
covariates, the deep network achieves the minimax-optimal rate

$$
\|\hat{\theta}_k - \theta^\star_k\|^2_{L_2(X)} = O\!\left(n^{-\frac{p}{p+d_c}} \log^8 n\right).
$$

> **Theorem 1 (FLM 2021).** "Under smoothness assumptions, the neural network
> estimator achieves the minimax optimal rate
> $$\|\hat{\theta}_k - \theta^\star_k\|^2_{L_2} = O_P\!\left(n^{-\frac{p}{p+d_c}} \log^8 n\right)$$
> where $p$ is the smoothness of $\theta^\star(\cdot)$ and $d_c$ is the dimension
> of continuous covariates."

This rate is slower than $\sqrt{n}$, which is exactly why naive inference fails
([Why Naive Inference Fails](03_why_naive_fails.md)) and why the influence function
correction is needed.

## Theorem 2: Asymptotic normality of the debiased estimator

Recall the influence function from [The Influence Function
Correction](04_influence_function.md):

$$
\psi(y, t, x, \theta, \Lambda) = H(x, \theta(x); \tilde{t}) - H_\theta(x, \theta(x); \tilde{t})\, \Lambda(x)^{-1}\, \ell_\theta(y, t, \theta(x)),
$$

with cross-fitted estimator

$$
\hat{\mu} = \frac{1}{K} \sum_{k=1}^{K} \frac{1}{|I_k|} \sum_{i \in I_k} \psi\big(y_i, t_i, \hat{\theta}_k(x_i), \hat{\Lambda}_k(x_i)\big).
$$

Under the rate condition $\|\hat{\theta} - \theta^\star\|_{L_2} = o_P(n^{-1/4})$,

$$
\sqrt{n}(\hat{\mu} - \mu^\star) \xrightarrow{d} N(0, \Psi).
$$

> **Theorem 2 (FLM 2021).** "Under rate conditions
> $\|\hat{\theta} - \theta^\star\|_{L_2} = o_P(n^{-1/4})$, the cross-fitted
> estimator satisfies
> $$\sqrt{n}(\hat{\mu} - \mu^\star) \xrightarrow{d} N(0, \Psi)$$
> where $\Psi = E[\psi_0(W)^2]$ and $\psi_0$ is the efficient influence function."

> **Neyman Orthogonality (FLM 2021).** "The influence function $\psi$ satisfies
> Neyman orthogonality, meaning first-order errors in nuisance estimation have no
> first-order effect on the target estimator. This is why the bias scales as
> $O(\delta^2)$ rather than $O(\delta)$."

**Critical rate condition.** The $n^{-1/4}$ threshold comes from the product-rate
requirement:

> "The product of nuisance estimation errors must satisfy
> $\|\hat{\theta} - \theta^\star\| \cdot \|\hat{\Lambda} - \Lambda^\star\| = o_P(n^{-1/2})$"
> FLM (2021), Theorem 2 conditions.

Because Theorem 1 gives each factor an $o_P(n^{-1/4})$ rate, their product is
$o_P(n^{-1/2})$, and the bias of $\hat\mu$ is negligible at the $\sqrt{n}$ scale.

## Worked application: binary choice with heterogeneity

**Model.**

$$
P[Y=1 \mid X=x, R=r] = G\big(\theta_1^\star(x_d, x_a) + \theta_2^\star(x_d)\, r\big),
$$

where $G(u) = 1/(1 + e^{-u})$ is the logit function.

**Average marginal effect.**

$$
\text{AME}(\tilde{r}) = \mathbb{E}\big[G(\theta^\star(X)'\tilde{r}_1)\,(1 - G(\theta^\star(X)'\tilde{r}_1))\,\theta_2^\star(X)\big].
$$

**Optimal personalized pricing.** Solve $\dfrac{d\Pi(r)}{dr} = 0$, where expected
profits are

$$
\Pi(r) = L\big[P(r)\,(M(1-D(r))r - D(r)) + (1-P(r))\,M r_0\big].
$$

## Applications and related work

### Personalized pricing (Dubé & Misra, 2023)

Dubé & Misra (2023, *JPE*) apply the FLM framework to personalized pricing with
heterogeneous demand. By estimating $\beta(X)$ (price sensitivity) as a function of
consumer characteristics, they compute:

- **Price elasticities** $\eta(X) = (1-p)\,\beta(X)\,P$, how responsive each consumer is to price changes;
- **Optimal personalized prices** via the Lerner markup rule $\frac{P-MC}{P} = -1/\eta$;
- **Consumer welfare** using the Small & Rosen (1981) logsum formula $CS = \log(1 + e^V)/|\beta_{\text{price}}|$.

`deep-inference` implements all three as built-in targets: `Elasticity`, `WTP`,
`ConsumerWelfare`. See the [Pricing Tutorial](../guide/pricing.md).

### Continuous treatment inference (Colangelo & Lee, 2026)

Colangelo & Lee (2026) develop double debiased ML for nonparametric inference with
continuous treatments, citing FLM DNNs as valid nuisance estimators that achieve
the required convergence rates. `deep-inference` complements their nonparametric
approach with a structural alternative: all model families natively support
continuous $T$, enabling dose-response analysis with economic structure. See the
[Continuous Treatment Tutorial](../guide/continuous_treatment.md).
