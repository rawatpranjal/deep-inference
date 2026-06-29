# Theory: The FLM Framework

```{note}
This is the long-form mathematical track for the influence-function method of
Farrell, Liang, and Misra (FLM). It is meant to be read top to bottom. It is the
**why**, not the how. For how to run the pipeline see [Estimation](../estimation/index.md),
for how a standard error and interval are formed see [Inference](../inference/index.md),
and for the papers see [References](../references/index.md). The automatic-debiasing
dual (RieszNet) has its own track on the [RieszNet page](riesznet.md).
```

The framework has one job. It lets a deep neural network learn rich heterogeneity in
a structural parameter, and still returns a confidence interval you can defend. The
two goals pull against each other. A network that fits well is regularized, and
regularization biases any summary computed from it. The influence-function correction
is what removes that bias to first order, so the interval is valid even though the
network converges slower than $\sqrt{n}$. The whole page builds to that one correction
and the conditions under which it works.

Two running examples carry the exposition, both written as generalized linear models
with a linear index $\eta = \alpha(x) + \beta(x)\,t$.

- **Linear** (identity link). The loss is $\ell(y,t,\theta) = (y - \alpha - \beta t)^2$.
  This is the partially-heterogeneous regression and, with a binary $t$, the conditional
  average treatment effect.
- **Logit** (logistic link). The loss is $\ell(y,t,\theta) = \log(1 + e^{\eta}) - y\,\eta$.
  This is heterogeneous binary choice, the workhorse of demand and pricing.

These two are enough to show every moving part, because they sit on opposite sides of
the one distinction that organizes the method. The linear Hessian does not depend on
the parameter, the logit Hessian does. The package handles richer models (Poisson,
Gamma, and the multinomial choice model used in the H and M application) with the same
machinery. The multinomial case is a vector-parameter generalization, where the scalar
curvature $p(1-p)$ becomes a block built from softmax probabilities $p_j(\delta_{jm} - p_m)$.
It is covered in the [model docs](../models/multinomial.md) and is not developed further here.

---

## 1. The enriched structural model

The starting point is an ordinary parametric structural model. We observe i.i.d. data
$W = (Y, T, X)$ where $Y$ is an outcome, $T$ is the policy-relevant variable (a price,
a dose, a treatment), and $X \in \mathbb{R}^{d_X}$ are covariates. A structural model is
a per-observation loss $\ell(y, t, \theta)$ with parameters $\theta \in \mathbb{R}^{d_\theta}$,
and the homogeneous model fixes a single $\theta^\star$ by

$$
\theta^\star = \arg\min_{\theta} \; \mathbb{E}\big[\ell(Y, T, \theta)\big].
$$

The loss carries the economics. Whatever $\theta$ means (a price sensitivity, a slope,
an elasticity), that meaning is preserved. The drawback is rigidity. A single $\theta^\star$
forces the same relationship between $Y$ and $T$ on everyone, and that is exactly the
heterogeneity an applied question usually cares about.

FLM (2025) enrich the model by letting the parameters be **functions of the covariates**.
The true first stage is

$$
\theta^\star(\cdot) = \arg\min_{\theta(\cdot) \in \mathcal{F}} \; \mathbb{E}\big[\ell(Y, T, \theta(X))\big],
$$

so that for each "type" $X = x$ there is its own parameter vector $\theta^\star(x)$. FLM are
emphatic that these are **not nuisance functions**. In their words the learned heterogeneity
is "actually the most interesting part," and because $\theta^\star(x)$ keeps the interpretation
of the original $\theta$, it is directly usable for policy. A deep network estimates the map
$x \mapsto \theta^\star(x)$ by minimizing the empirical analogue,

$$
\hat{\theta}(\cdot) = \arg\min_{\theta(\cdot) \in \mathcal{F}_{\text{dnn}}} \; \frac{1}{n}\sum_{i=1}^{n} \ell\big(y_i, t_i, \theta(x_i)\big).
$$

The structure is "baked into" the network. Only a line or two of code differs from a
standard regression network, because the final layer is forced to emit the parameter
vector $\theta(x)$ and the structural loss is applied on top of it, rather than letting
the network predict $Y$ directly.

### The GLM index form, used throughout

Both running examples are the conditional-mean model of FLM Appendix B. Write
$T_1 = (1, t)'$ and let $G$ be the inverse link, so $\mathbb{E}[Y \mid x, t] = G(\alpha^\star(x) + \beta^\star(x)\,t)$.
Set $\theta = (\alpha, \beta)$ and define

$$
G_0(x, t) = G\big(\alpha^\star(x) + \beta^\star(x)\,t\big), \qquad
\dot{G}_0(x, t) = \frac{dG(u)}{du}\bigg|_{u = \alpha^\star(x) + \beta^\star(x)\,t}.
$$

For the **linear** case $G(u) = u$ and $\dot{G}_0 \equiv 1$. For the **logit** case
$G(u) = 1/(1 + e^{-u})$ and $\dot{G}_0 = G(1-G) = p(1-p)$, where $p = G_0(x,t)$ is the
fitted choice probability. This single distinction, whether $\dot{G}_0$ is a constant or
a function of $\theta$, is what later splits the problem into regimes.

---

## 2. The target functional

We rarely want $\theta^\star(x)$ for every individual. We want a population summary,

$$
\mu^\star = \mathbb{E}\big[H(X, \theta^\star(X), \tilde{t})\big],
$$

for a known function $H$ that may depend on a policy-relevant evaluation point $\tilde{t}$.
The function $H$ encodes the economic question. The same fitted network answers many
questions by swapping $H$. Two cases carry this page.

**Average parameter.** $H(x, \theta, \tilde{t}) = \beta$ gives $\mu^\star = \mathbb{E}[\beta^\star(X)]$,
the mean price sensitivity or, in the linear binary-treatment model, the average treatment
effect $\mathbb{E}[\beta^\star(X)]$ where $\beta^\star(x)$ is the conditional average treatment
effect. Its Jacobian is constant, $H_\theta = (H_\alpha, H_\beta) = (0, 1)$.

**Average marginal effect (AME).** For the logit, the marginal effect of $t$ on the
choice probability at $\tilde{t}$ is $\dot{G}(\alpha + \beta\tilde{t})\,\beta$, so

$$
\mu^\star = \mathbb{E}\big[\dot{G}(\alpha^\star(X) + \beta^\star(X)\tilde{t})\,\beta^\star(X)\big]
= \mathbb{E}\big[p(1-p)\,\beta^\star(X)\big].
$$

This is the object FLM compute in their interest-rate application, the "fully flexible
average marginal effects in a logistic regression." Its Jacobian $H_\theta$ is not constant,
and for the built-in targets the package supplies it in closed form, otherwise by automatic
differentiation. The Jacobian is required by the correction in Section 4. Choosing $H$ is an
economic decision, not a statistical one.

---

## 3. Why naive plug-in inference fails

The obvious estimator trains the network, plugs in, and averages,

$$
\hat{\mu}_{\text{naive}} = \frac{1}{n}\sum_{i=1}^{n} H\big(X_i, \hat{\theta}(X_i), \tilde{t}\big),
\qquad
\widehat{\text{SE}}_{\text{naive}} = \frac{\operatorname{sd}(H_1, \ldots, H_n)}{\sqrt{n}}.
$$

The naive standard error treats $\hat{\theta}$ as if it were the truth. It is not, and the gap
does not wash out. Expand the estimator's error around the truth,

$$
\sqrt{n}\big(\hat{\mu}_{\text{naive}} - \mu^\star\big)
= \underbrace{\frac{1}{\sqrt{n}}\sum_{i=1}^{n}\big[H(X_i,\theta^\star(X_i),\tilde t) - \mu^\star\big]}_{\text{sampling noise, } O_P(1)}
\; + \; \underbrace{\frac{1}{\sqrt{n}}\sum_{i=1}^{n} H_{\theta,i}\big(\hat{\theta}(X_i) - \theta^\star(X_i)\big)}_{\text{first-order plug-in bias}}
\; + \; o_P(1).
$$

The first term is ordinary sampling variability and is what the naive standard error
measures. The second term is the problem. The deep network converges in $L_2$ at the
nonparametric rate of Theorem 1 (Section 6), which is **slower than $\sqrt{n}$**. So
$\frac{1}{\sqrt{n}}\sum_i H_{\theta,i}(\hat{\theta}_i - \theta^\star_i)$ does not vanish.
Worse, $\hat{\theta} - \theta^\star$ is not mean zero. Weight decay, dropout, and early
stopping all shrink $\hat{\theta}$ toward zero by design, so the bias term has a systematic
sign and pulls the estimate toward zero. The naive interval is then centered in the wrong
place and is too narrow, because its width ignores this term entirely.

The package documents how severe this is in practice. Naive 95 percent intervals cover
the truth only 0 to 20 percent of the time, the naive standard error runs 3 to 10 times too
small, and the miss persists at very large $n$. It is not a small-sample artifact. The
regularization that makes the network generalize is precisely what breaks naive inference.
The correction in the next section is built to cancel the first-order bias term exactly.

---

## 4. The influence-function correction

For each observation define the influence value

$$
\psi_i = H_i - H_{\theta,i}\,\Lambda(X_i)^{-1}\,\ell_{\theta,i},
$$

which is the exact object the package assembles in
`src/deep_inference/engine/assembler.py` as $\psi = H - H_\theta\,\Lambda^{-1}\,\ell_\theta$.
The four pieces are

- $H_i = H(X_i, \hat{\theta}(X_i), \tilde{t})$, the target at the fitted parameters,
- $H_{\theta,i} = \partial H / \partial \theta$ evaluated at $\hat{\theta}(X_i)$, the $1 \times d_\theta$ **target Jacobian**,
- $\ell_{\theta,i} = \nabla_\theta \ell(Y_i, T_i, \hat{\theta}(X_i))$, the **score**, the loss gradient at the observed data point,
- $\Lambda(X_i) = \mathbb{E}[\nabla^2_\theta \ell(Y, T, \theta(X)) \mid X = X_i]$ evaluated at $\hat\theta(X_i)$, the **expected conditional Hessian** of the loss.

The debiased estimator and its standard error are then sample moments of $\psi$,

$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n}\psi_i, \qquad
\widehat{\Psi} = \frac{1}{n}\sum_{i=1}^{n}\big(\psi_i - \hat\mu\big)^2, \qquad
\widehat{\text{SE}} = \sqrt{\widehat{\Psi}/n}.
$$

FLM (2025) note that this has "precisely the same form as its parametric counterpart, but
appropriately generalized." For an ordinary parametric two-step problem the influence
function is $H(x,\theta;\tilde t) - H_\theta(x,\theta;\tilde t)\,\Lambda^{-1}\,\ell_\theta(y,t,\theta)$
with $\Lambda = \mathbb{E}[\ell_{\theta\theta}]$ (Newey and McFadden, 1994). The enriched version
is identical, except every piece is conditional on $X = x$, and $\Lambda(x)$ is the conditional
Hessian rather than a single matrix.

### Intuition for the three pieces

Read the correction term right to left. The score $\ell_\theta$ measures how far observation
$i$'s data sits from what the fitted model expects. The curvature inverse $\Lambda(x)^{-1}$
converts that residual gradient into an error in parameter space, the same way an inverse
Hessian turns a gradient into a Newton step. The Jacobian $H_\theta$ maps that parameter
error into target space. When the network fits a point well its score is near zero and the
correction is near zero. When the network is systematically off (regularization bias) the
score is systematically nonzero and the correction kicks in to absorb it.

### The GLM score and Hessian, worked

For the conditional-mean model of Section 1, FLM Appendix B.2 gives the score and Hessian
in closed form,

$$
\ell_\theta(y, t, \theta(x)) = T_1\big(G_0(x,t) - y\big), \qquad
\Lambda(x) = \mathbb{E}\big[\dot{G}_0(x, T)\,T_1 T_1' \mid X = x\big].
$$

The score is the linear-index direction $T_1 = (1,t)'$ scaled by the fitted residual
$G_0 - y$. Substituting the two running examples:

- **Linear** ($\dot{G}_0 \equiv 1$). The score is $T_1(\alpha + \beta t - y)$, the residual
  times $(1, t)'$. The conditional Hessian
  $\Lambda(x) = \mathbb{E}[T_1 T_1' \mid X = x]$, with entries $1$, $\mathbb{E}[T\mid x]$, and
  $\mathbb{E}[T^2\mid x]$, does **not** depend on $\theta$. (The package's squared-error loss carries a factor of $2$,
  which is immaterial here. A constant rescaling of the loss multiplies both $\Lambda$ and
  $\ell_\theta$ by the same constant, and it cancels in $\Lambda^{-1}\ell_\theta$.)
- **Logit** ($\dot{G}_0 = p(1-p)$). The score is $T_1(p - y)$. The conditional Hessian
  $\Lambda(x) = \mathbb{E}[p(1-p)\,T_1 T_1' \mid X = x]$ **does** depend on $\theta$, through the
  fitted probability $p = G_0(x,t)$.

For a scalar treatment this $\Lambda(x)$ is the $2\times 2$ matrix with entries
$\lambda_k(x) = \mathbb{E}[\dot{G}_0(x,T)\,T^k \mid X = x]$ for $k \in \{0,1,2\}$,

$$
\Lambda(x) = \begin{pmatrix} \lambda_0(x) & \lambda_1(x) \\ \lambda_1(x) & \lambda_2(x) \end{pmatrix},
\qquad \det \Lambda(x) = \lambda_0(x)\lambda_2(x) - \lambda_1(x)^2 .
$$

Invertibility of $\Lambda(x)$ is the identification condition. For the identity link it
requires a positive conditional variance of $T$ given $X$, and for the logistic link it
holds whenever $P[Y=1 \mid x, t]$ is bounded away from $0$ and $1$. This is the enriched
analogue of the familiar parametric requirement that $\Lambda^{-1}$ exist, now imposed for
every type $x$.

### Why the minus sign removes the bias: Neyman orthogonality

The correction is not just a plausible adjustment. The minus sign and the $\Lambda(x)^{-1}$
are tuned so that the score $\psi$ is **Neyman orthogonal**, meaning its derivative with
respect to the first-stage parameter is zero at the truth,

$$
\frac{\partial}{\partial \theta}\,\mathbb{E}\big[\psi(Y, T, X, \theta, \Lambda)\big]\bigg|_{\theta = \theta^\star,\,\Lambda = \Lambda^\star} = 0.
$$

The reason it holds is worth seeing, because it explains why the slow first-stage rate is
tolerated. Take the directional derivative of $\mathbb{E}[\psi]$ at $\theta^\star$ in a
direction $h(x)$. Whenever the derivative falls on the Jacobian $H_\theta$ or on $\Lambda^{-1}$,
the resulting term still carries a factor of $\ell_\theta$, and FLM Assumption 3(ii) is the
first-order condition $\mathbb{E}[\ell_\theta(Y, T, \theta^\star(X)) \mid X, T] = 0$, so those
terms vanish in expectation. Two terms survive,

$$
\frac{\partial}{\partial \theta}\mathbb{E}\big[H\big]\big|_{\theta^\star} = \mathbb{E}\big[H_\theta\, h\big],
\qquad
-\frac{\partial}{\partial \theta}\mathbb{E}\big[H_\theta\,\Lambda^{-1}\,\ell_\theta\big]\big|_{\theta^\star}
= -\,\mathbb{E}\big[H_\theta\,\Lambda^{-1}\,\ell_{\theta\theta}\,h\big].
$$

Now use the definition $\mathbb{E}[\ell_{\theta\theta} \mid X] = \Lambda(x)$. The second term
becomes $-\mathbb{E}[H_\theta\,\Lambda^{-1}\,\Lambda\, h] = -\mathbb{E}[H_\theta\,h]$, which
cancels the first exactly. The expected Hessian $\Lambda$ is precisely the matrix that turns
$\Lambda^{-1}\ell_{\theta\theta}$ into the identity, so the curvature inverse is doing the
canceling. That is the content of the minus sign.

The consequence is the engine of the whole method. Because the first-order term in
$\hat{\theta} - \theta^\star$ is gone, the estimation error in the network enters
$\hat{\mu}$ only through a second-order remainder. The bias of $\hat\mu$ is
$O(\|\hat\theta - \theta^\star\|^2)$ rather than $O(\|\hat\theta - \theta^\star\|)$, and a
quantity that converges slower than $\sqrt{n}$ has a square that converges faster than
$1/\sqrt{n}$. The slow nonparametric rate becomes negligible after squaring. FLM put the
same point in semiparametric language. The correction makes the score orthogonal, double
machine learning then needs only weak $L_2$ rate conditions on the first stage, and almost
nothing else is required of the network.

---

## 5. The three regimes for $\Lambda(x)$

Everything above treats $\Lambda(x)$ as known. In practice it is the second nuisance object,
and how it is obtained is the practical heart of the method. FLM call $\Lambda(x)$ "a nuisance
in the truest sense," required only because we use an influence function as a tool. How you
get it depends on one question, does $\dot{G}_0$ (the Hessian weight) depend on the unknown
parameter, and a second, is the treatment randomized.

```{list-table}
:header-rows: 1

* - Regime
  - When
  - How $\Lambda(x)$ is obtained
  - Cross-fitting
* - A
  - Randomized $T$ with known $F_T$, Hessian free of $Y$
  - Compute by integrating over $F_T$
  - Two-way
* - B
  - Linear (Hessian constant in $\theta$)
  - Analytic from the data
  - Two-way
* - C
  - Observational and nonlinear
  - Estimate by regressing the per-observation Hessian on $X$
  - Three-way
```

**Regime A (randomized).** If $T$ is assigned with a known distribution $F_T$ and the Hessian
does not depend on $Y$, then $\Lambda(x) = \int \nabla^2_\theta \ell(y, t, \theta)\,dF_T(t)$ can
be computed by Monte Carlo integration with no separate estimation step. FLM note that under
randomization $\Lambda(x)$ "can often be computed or estimated more simply, though it may remain
a function of $x$." This needs only two-way cross-fitting.

**Regime B (linear).** This is the linear running example. Because $\dot{G}_0 \equiv 1$, the
Hessian $\Lambda(x) = \mathbb{E}[T_1 T_1' \mid X]$ does not involve $\theta$ at all. Its
regressand, $T_1 T_1'$, is observed data, so $\Lambda(x)$ is a plain conditional second moment
estimated directly, and three-way splitting is not needed. FLM state it plainly, "For linear
models, three splits are not necessary."

**Regime C (observational, nonlinear).** This is the logit running example and the typical
applied case. Because $\dot{G}_0 = p(1-p)$ depends on $\theta$, the regressand of
$\Lambda(x) = \mathbb{E}[\dot{G}_0(x,T)\,T_1 T_1' \mid X]$ is itself a function of the unknown
$\theta^\star(x)$ and must be estimated first. FLM make the reason for the extra split explicit,
"the regressand of $\Lambda(x)$ depends on the unknown $\theta^\star(x)$ through $\dot{G}_0(x,T)$,
which must be estimated in a first step." So $I_k^c$ is split in two, $\hat\theta_k$ and
$\hat\Lambda_k$ are fit on separate halves, and the procedure uses three-way cross-fitting.
In implementation the column of per-observation second derivatives
$\ell_{\theta\theta}(y_i, t_i, \hat\theta(x_i))$ is simply regressed nonparametrically on $x_i$
to obtain $\hat\Lambda(x_i)$.

The load-bearing object in Regime C is $\Lambda(x)^{-1}$, not $\Lambda(x)$. At low overlap the
matrix is near singular ($\det \Lambda \to 0$), so a regressor that fits $\Lambda$ well can still
produce a wildly unstable inverse. This is the source of the standard-error undercount documented
in this project, and it is also the exact place where the RieszNet dual sidesteps the problem by
never forming an inverse. That story is told on the [RieszNet page](riesznet.md). For which
$\Lambda$ estimator to choose, and why the package defaults to ridge, see
[Estimation](../estimation/index.md) and [Inference](../inference/index.md).

---

## 6. Formal guarantees

Two theorems make the method rigorous. The first bounds how fast the network learns
$\theta^\star(x)$, the second turns that rate into a valid interval.

### Theorem 1: convergence rate of the structural network

Under a Lipschitz-plus-curvature condition on the loss (Assumption 1) and smoothness of the
parameter functions (Assumption 2, each $\theta^\star_k$ in a Holder ball $W^{p,\infty}$ over the
continuous covariates), the structured deep ReLU network with width $J \asymp n^{d_c/2(p+d_c)}\log^2 n$
and depth $L \asymp \log n$ achieves, for each coordinate $k$,

$$
\big\|\hat{\theta}_k - \theta^\star_k\big\|^2_{L_2(X)} = O\!\left(n^{-\frac{p}{p + d_c}}\,\log^8 n\right),
$$

where $p$ is the smoothness and $d_c$ is the number of continuously distributed covariates. This
is the FLM (2025) generalization of FLM (2021, Theorem 1) from prediction to structural
parameters. The rate is the minimax nonparametric rate up to logs, and it is **slower than
$\sqrt{n}$**, which is exactly why Section 3's naive interval fails and why the correction of
Section 4 is needed. Discrete covariates do not enter $d_c$ and do not slow the rate, an
advantage FLM stress for deep networks.

### Theorem 2: asymptotic normality of the debiased estimator

Recall the cross-fitted estimator over $K$ folds,

$$
\hat\mu = \frac{1}{K}\sum_{k=1}^{K}\frac{1}{|I_k|}\sum_{i \in I_k}
\psi\big(y_i, t_i, x_i, \hat\theta_k(x_i), \hat\Lambda_k(x_i)\big),
$$

where $\hat\theta_k$ and $\hat\Lambda_k$ are fit out of fold. Under FLM Assumption 3 (the loss is
thrice differentiable, the first-order condition $\mathbb{E}[\ell_\theta \mid X, T] = 0$ holds,
$\Lambda(x)$ is invertible with bounded inverse, $\mu^\star$ is pathwise differentiable and $H$ is
thrice differentiable, and the relevant quantities have more than four finite moments), together
with the rate conditions

$$
\big\|\hat\theta_k - \theta^\star_k\big\|_{L_2(X)} = o_P\!\big(n^{-1/4}\big), \qquad
\big\|[\hat\Lambda]_{k_1,k_2} - [\Lambda]_{k_1,k_2}\big\|_{L_2(X)} = o_P\!\big(n^{-1/4}\big),
$$

and $\hat\Lambda(x_i)$ uniformly invertible, the score is Neyman orthogonal and

$$
\sqrt{n}\big(\hat\mu - \mu^\star\big) \xrightarrow{d} N\big(0, \Psi\big),
\qquad \Psi = \mathbb{V}\big[\psi(Y, T, X, \theta^\star, \Lambda)\big],
$$

with $\widehat{\Psi}$ from Section 4 a consistent estimator. The $n^{-1/4}$ threshold is the
familiar product-rate logic. Neyman orthogonality makes the bias scale as the **product** of the
two nuisance errors, and requiring each of $\|\hat\theta - \theta^\star\|$ and
$\|\hat\Lambda - \Lambda\|$ to be $o_P(n^{-1/4})$ makes that product $o_P(n^{-1/2})$, negligible
at the $\sqrt{n}$ scale. Theorem 1 supplies the rate for $\hat\theta$, and the same nonparametric
machinery supplies it for $\hat\Lambda$. Both running examples satisfy the conditions, with the
caveat that Regime B (linear) needs only the $\hat\theta$ rate because $\Lambda$ is not separately
estimated, while Regime C (logit) needs both.

A note on efficiency. When the structural model comes from a likelihood or exponential family and
the two stages contain all the information, FLM conjecture (Remark 5.1) that the influence function
matches the efficient one, so the interval is not just valid but as tight as semiparametrics allows.
In partially linear models (where the slope is forced constant) efficiency holds only under
homoskedasticity, because the FLM score does not impose the constancy that the efficient score for
that submodel would.

### Why cross-fitting

The reason the splits appear in the theorem is that the remainder must stay negligible. Fitting
$\hat\theta_k$ and $\hat\Lambda_k$ on data disjoint from where $\psi$ is evaluated removes the
"own-observation" dependence that would otherwise leave an $O_P(n^{-1/2})$ term in the expansion.
FLM (2021) showed that for post-DNN average treatment effects splitting can even be avoided under
the same assumptions, but the general structural result keeps it. The mechanics of the K-fold loop,
the pooled variance, and repeated cross-fitting are pipeline details, covered in
[Estimation](../estimation/index.md) and [Inference](../inference/index.md).

---

## 7. The binary-treatment case, and the bridge to automatic debiasing

It is worth closing on the one example that ties everything together and connects this track to
the [RieszNet track](riesznet.md). Take the linear binary-treatment model, $G(u) = u$ and
$t \in \{0, 1\}$, with the average treatment effect target $H = \beta$ (so $H_\alpha = 0$,
$H_\beta = 1$). Because $T$ is binary, $\mathbb{E}[T \mid x] = \mathbb{E}[T^2 \mid x] = p(x)$, the
propensity score, so

$$
\Lambda(x) = \begin{pmatrix} 1 & p(x) \\ p(x) & p(x) \end{pmatrix},
\qquad \det \Lambda(x) = p(x)\big(1 - p(x)\big).
$$

The determinant is the propensity overlap, and the standard assumption that $p(x)$ stays away
from $0$ and $1$ is exactly what keeps $\Lambda(x)^{-1}$ well behaved. Now carry out the
correction. With the score $\ell_\theta = T_1(g_0 - y)$ where $g_0 = \alpha + \beta t$, a short
calculation (FLM C.1, "by adding and subtracting $p(x)$ and using $(1-t)t = 0$") gives

$$
- H_\theta\,\Lambda(x)^{-1}\,\ell_\theta = \frac{T - p(x)}{p(x)\big(1 - p(x)\big)}\,\big(Y - g_0\big),
\qquad\text{so}\qquad
\psi = \beta^\star + \frac{T - p(x)}{p(x)(1 - p(x))}\big(Y - g_0\big).
$$

This is the doubly robust influence function of Hahn (1998), the workhorse AIPW estimator. The
weight $(T - p)/(p(1-p))$ is the inverse-propensity correction, and it is the Riesz representer of
the average treatment effect. In other words, the FLM correction $-H_\theta\,\Lambda^{-1}\,\ell_\theta$
**is** a representer-weighted residual, with the curvature inverse $\Lambda^{-1}$ supplying the
overlap weighting $1/(p(1-p))$. That identity is the entire content of the
[RieszNet page](riesznet.md), where the same correction is learned directly instead of assembled
from a score and a Hessian, and where the failure mode of estimating $\Lambda(x)^{-1}$ at low
overlap is examined in full.

---

## Where to go next

- [Estimation](../estimation/index.md) for the pipeline that fits $\hat\theta(x)$, estimates
  $\Lambda(x)$, and assembles $\psi$.
- [Inference](../inference/index.md) for turning $\psi$ into a standard error and a confidence
  interval, and for choosing between the two procedures.
- [RieszNet and Automatic Debiasing](riesznet.md) for the dual route that learns the correction
  directly.
- [References](../references/index.md) for FLM (2021, 2025) and the surrounding literature.
