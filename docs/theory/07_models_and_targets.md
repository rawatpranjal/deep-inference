# 7. Available Models and Targets

The `deep-inference` package provides a growing library of built-in model families
(see [Setup](01_setup.md)) and target functionals (see
[Target Functionals](02_targets.md)). Together, the choice of **model** (the loss
$\ell$) and **target** (the functional $H$) fully specify an inference problem.

## Custom losses

For models not in the built-in library, users can supply a custom loss function.
The package then determines automatically whether the Hessian depends on $\theta$
(triggering Regime C and three-way splitting) and computes all required
derivatives — score, Hessian, and target Jacobian — via automatic differentiation.
The user needs only the loss function and the target.

```python
import torch
from deep_inference import inference

def my_loss(y, t, theta):
    """Custom structural loss (Poisson with log-link)."""
    mu = torch.exp(theta[0] + theta[1] * t)  # log-link
    return mu - y * torch.log(mu)            # Poisson NLL

result = inference(
    Y, T, X,
    loss=my_loss, theta_dim=2,
    target_fn=lambda x, th, tt: th[1],
)
```

## Putting it together

The three earlier phases combine into one recipe:

1. Pick a **model** — the structural loss $\ell$ ([Setup](01_setup.md)).
2. Pick a **target** — the functional $H$ ([Target Functionals](02_targets.md)).
3. The package fits $\hat\theta(X)$, applies the
   [influence function correction](04_influence_function.md), selects the
   [Lambda regime](05_three_regimes.md) automatically, and returns $\hat\mu$ with a
   valid standard error and confidence interval.

```{seealso}
The formal guarantees behind this recipe — the neural network convergence rate
(Theorem 1) and the asymptotic normality of the debiased estimator (Theorem 2) —
are stated on the final page, [Theorems and Convergence
Rates](influence_functions.md).
```
