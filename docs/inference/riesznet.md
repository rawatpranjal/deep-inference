# RieszNet Procedure

RieszNet is the second inference procedure. It reaches the same debiased estimate as the
[influence-function procedure](influence_function.md), but it **learns** the bias correction with
a neural network instead of deriving it from the model's Hessian (the "automatic debiasing" idea
of Chernozhukov, Newey, Quintas-Martinez, Syrgkanis 2022). It is available as
`riesz_inference(...)`; see the [API reference](../api.md) for the full signature.

This page is the practical how-to. For the full derivation (the Riesz representer, the
double-robustness property, why the Riesz loss recovers the representer, and why this equals the
FLM correction), read [Theory: RieszNet](../theory/riesznet.md).

## How it works, in brief

RieszNet trains one network with two heads: a regression head for the outcome model $g(Z)$ and a
linear Riesz head for the representer $a(Z)$. It then debiases with the doubly-robust moment

$$\psi = (\tilde g_1 - \tilde g_0) + a(Z)\,(Y - \tilde g),$$

averaged over a cross-fit, where $\tilde g = g + \varepsilon a$ is the targeted-regularized
regression. "Doubly robust" means the estimate stays valid if either the outcome model $g$ or the
representer $a$ is learned well. The payoff for implementation is that you never derive a Hessian
or a propensity by hand. That is all you need to run it; the why is in [Theory](../theory/riesznet.md).

## When to use it

- You want debiasing without deriving model-specific curvature.
- You want an independent, doubly-robust check on the influence-function result. In the
  [Replications](../replications/index.md) tables the two procedures are run side by side on
  known-truth data. Where both are valid they agree, and the page flags the cases where RieszNet
  is unreliable (for example heavy-tailed count outcomes, where its empirical standard error is
  inflated by divergent repeats).

## Usage

```python
import numpy as np
from deep_inference import riesz_inference

rng = np.random.default_rng(0)
n = 1500
X = rng.standard_normal((n, 3))
e = 1 / (1 + np.exp(-(0.8 * X[:, 0])))     # confounded binary treatment
T = (rng.random(n) < e).astype(float)
Y = 1.0 + 0.5 * X[:, 0] + (0.5 + 0.3 * X[:, 1]) * T + rng.standard_normal(n)

result = riesz_inference(Y, T, X, outcome='linear', n_folds=5, n_repeats=3)
print(result.summary())
```

`outcome` is one of `linear`, `logit`, `poisson`, `gamma`, `negbin`, `probit`. The estimand is
the ATE contrast `E[g(1, X) - g(0, X)]`. Arguments are documented on the [API page](../api.md).

## Evidence

The ported `riesz_inference` was run on known-truth data, M=40 replications per family
(`exploration/riesz_coverage.py`). The interval should cover the truth about 95% of the time with
a standard-error ratio near 1.

```{list-table}
:header-rows: 1

* - Outcome
  - Truth
  - Mean estimate
  - Bias
  - SE-ratio
  - Coverage
* - linear
  - 0.5012
  - 0.5052
  - +0.0041
  - 0.96
  - 95%
* - logit
  - 0.1812
  - 0.1807
  - -0.0006
  - 0.91
  - 92%
```

Both are valid (the logit 92% sits within Monte-Carlo noise of the band at M=40, with bias
essentially zero). These reproduce the prototype's larger M=100 study reported on the
[Replications](../replications/index.md) page (linear 97%, logit 100%), where RieszNet is run side
by side with the influence-function method and the oracle. The unit tests in
`src/deep_inference/tests/test_riesz.py` check the same recovery on every commit.

## See also

- [Influence Function](influence_function.md), the default procedure.
- [Theory: RieszNet](../theory/riesznet.md) for the full derivation.
- [Replications](../replications/index.md) for the head-to-head numbers.
- [API Reference](../api.md) for the function signatures.
