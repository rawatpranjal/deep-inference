# 3. Why Naive Inference Fails

To estimate $\mu^*$, we train a neural network $\hat{\theta}(X)$ by minimizing
$\sum_i \ell(Y_i, T_i, \hat{\theta}(X_i))$ with sample splitting, and then compute

$$
\hat{\mu} = \frac{1}{n}\sum_i H(X_i, \hat{\theta}(X_i), \tilde{t}).
$$

The **naive** standard error treats $\hat{\theta}$ as if it were the truth and
computes $\text{SE}_{\text{naive}} = \text{sd}(H_i) / \sqrt{n}$. This is wrong,
because $\hat{\theta}$ is itself estimated with error, and neural network
regularization introduces systematic bias.

## The problem is quantitatively severe

In our simulations, naive 95% confidence intervals achieve only **0–20% actual
coverage**, the naive SE is **3–10× too small**, and the bias is toward zero
because regularization shrinks $\hat{\theta}$.

This is *not* a small-sample problem; it persists at $n = 50{,}000$. The
regularization bias is a **feature** of neural network training (it prevents
overfitting) but becomes a **bug** for inference.

In code, the difference is stark:

```python
# Naive SE (wrong): ignores estimation uncertainty
se_naive = np.std(H_values) / np.sqrt(n)

# IF SE (correct): accounts for neural network estimation
se_if = result.se  # from influence function

print(f"Naive: {se_naive:.4f}, IF: {se_if:.4f}, Ratio: {se_if/se_naive:.1f}x")
# Typical output for logit: Naive=0.009, IF=0.026, Ratio=2.8x
```

## Why, formally

To see why the naive SE fails, decompose the naive estimator's error:

$$
\sqrt{n}\left(\hat{\mu}_{\text{naive}} - \mu^*\right)
= \underbrace{\frac{1}{\sqrt{n}}\sum_{i=1}^n \left[H_i - \mu^*\right]}_{\text{sampling noise}}
+ \underbrace{\frac{1}{\sqrt{n}}\sum_{i=1}^n H_{\theta,i}\left(\hat{\theta}_i - \theta^*_i\right)}_{\text{regularization bias}}
+ o_P(1).
$$

The first term is standard sampling variability and vanishes at rate $\sqrt{n}$.
The second term captures the bias introduced by the neural network's
regularization: because $\hat{\theta}$ is systematically shrunk toward zero (by
weight decay, dropout, and early stopping), $\hat{\theta}_i - \theta^*_i$ is **not
mean-zero**, and this term does not vanish.

The influence function correction on the [next page](04_influence_function.md) is
designed precisely to cancel this second term.
