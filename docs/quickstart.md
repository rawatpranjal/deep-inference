# Quick Start

Install the package, run one example, and read the output. By the end of this page you will have
a debiased estimate and a valid confidence interval, and you will understand every line that
produced them.

## Install

Requires Python 3.10+ and PyTorch 2.0+.

```bash
pip install deep-inference
```

Or from source:

```bash
git clone https://github.com/rawatpranjal/deep-inference
cd deep-inference
pip install -e .
```

Optional extras: `deep-inference[plotting]` for the plot helpers, `[docs]` to build this site,
`[dev]` for the test suite, `[all]` for everything. Check the install:

```python
from deep_inference import structural_dml
print("deep-inference ready")
```

## The whole thing in one example

We simulate data where we know the answer, so you can see that the interval covers it. The
treatment effect is heterogeneous, `β(X) = 0.5 + 0.3·X₁`, so its average is `E[β(X)] = 0.5`.
That `0.5` is the truth we want to recover.

```python
import numpy as np
import torch
from deep_inference import structural_dml

np.random.seed(42)
torch.manual_seed(42)

n = 2000
X = np.random.randn(n, 5)           # 5 covariates
T = np.random.randn(n)              # continuous treatment

beta = 0.5 + 0.3 * X[:, 1]         # heterogeneous effect; mean is 0.5
alpha = 0.2 * X[:, 0]              # baseline
Y = alpha + beta * T + np.random.randn(n)   # continuous outcome

result = structural_dml(
    Y=Y, T=T, X=X,
    family='linear',               # continuous outcome
    hidden_dims=[64, 32],          # network shape
    epochs=100,
    n_folds=50,                    # cross-fitting folds
    lr=0.01,
)

print(result.summary())
```

This example sets `epochs=100`. The default is `200`; for a well-behaved linear model 100 is
plenty, and you can raise it if the diagnostics suggest underfitting.

## What you get back

Running the above prints (date and time will differ; this is real output from the example):

```text
==============================================================================
                            Structural DML Results
==============================================================================
Family:           Linear               Target:           E[beta]
No. Observations: 2000                 No. Folds:        50
==============================================================================
                  coef     std err         z     P>|z|      [0.025    0.975]
------------------------------------------------------------------------------
     E[beta]    0.5044      0.0240    20.982  0.000      0.4573    0.5515
==============================================================================
Diagnostics:
  Min Lambda eigenvalue:    1.914079
  Mean condition number:    1.06
  Correction ratio:         43.2249
  Pct regularized:          0.0%
------------------------------------------------------------------------------
```

Read it line by line:

- **`coef` 0.5044** is the debiased point estimate of `E[β(X)]`. The truth is `0.5`. It nailed it.
- **`std err` 0.0240** is the standard error from the influence-function variance.
- **`[0.025  0.975]` = [0.4573, 0.5515]** is the 95% confidence interval. It contains `0.5`, as it should.
- **Min Lambda eigenvalue** well above `1e-4` means the Hessian was stable and invertible.
- **Correction ratio 43** says the bias correction was sizeable relative to the naive piece; this is the part that fixes coverage.

## Why bother: naive versus debiased

The whole point is that skipping the correction gives you a wrong interval. Compare the two
numbers the run produces:

```python
print(f"Naive estimate:    {result.mu_naive:.4f}")   # 0.4817
print(f"Debiased estimate: {result.mu_hat:.4f}")      # 0.5044
```

On this run the naive average of the network's `β(X)` is `0.4817`, biased low; the debiased
estimate is `0.5044`, on the truth. The bias is small here because the model is linear and
well-behaved; on nonlinear models the naive interval can cover the truth as little as 8% of the
time. See the [Overview](overview.md) motivation table and the
[Simulation Studies](simulation_studies.md) for the full picture.

## A few of the knobs

```{list-table}
:header-rows: 1

* - Argument
  - Default
  - What it does
* - `family`
  - (required)
  - The outcome model: `'linear'`, `'logit'`, `'poisson'`, and more. See [Models](models/index.md).
* - `hidden_dims`
  - `[64, 32]`
  - Network width per layer.
* - `epochs`
  - `200`
  - Training epochs.
* - `n_folds`
  - `50`
  - Cross-fitting folds. Use `>= 50` for coverage.
* - `n_repeats`
  - `1`
  - Repeat the split and median-combine. Use `>= 5` to tighten coverage.
* - `lr`
  - `0.01`
  - Learning rate.
```

## The flexible API

`structural_dml()` is the production entry point for the average effect `E[β]`. For other
targets and for randomized experiments, use `inference()`:

```python
from deep_inference import inference

# Average marginal effect on the probability scale (logit), evaluated at T=0
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)

# Any custom differentiable target; autodiff supplies the Jacobian
import torch
def avg_prediction(x, theta, t_tilde):
    return torch.sigmoid(theta[0] + theta[1] * t_tilde)
result = inference(Y, T, X, model='logit', target_fn=avg_prediction, t_tilde=0.0)

# Randomized experiment with a known treatment law: Lambda is computed, not estimated
from deep_inference.lambda_.compute import Normal
result = inference(Y, T, X, model='logit', target='beta',
                   is_randomized=True, treatment_dist=Normal(0.0, 1.0))
```

## Next

- [Loading Data & Pre-Estimation](loading_data.md): the exact shapes and types your inputs need.
- [Models](models/index.md): pick the right model for your outcome.
- [Inference](inference/index.md): how the standard errors are formed, and the RieszNet alternative.
