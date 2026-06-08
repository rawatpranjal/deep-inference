# 6. The Multinomial Choice Model

The H&M application uses a multinomial logit (conditional logit) model (McFadden,
1974; Berry, Levinsohn & Pakes, 1995; Train, 2009). Consumer $i$ chooses from $J$
alternatives, each described by $K$ attributes $x_{ij} \in \mathbb{R}^K$. Utilities
are

$$
V_{ij} = x_{ij}' \theta(X_i), \qquad
P(Y_i = j \mid X_i) = \frac{e^{V_{ij}}}{\sum_{m=1}^J e^{V_{im}}}.
$$

## Heterogeneity without a distributional assumption

The FLM framework estimates $\theta(X_i)$ for each individual using
individual-level data and neural networks, **rather than assuming a parametric
distribution for heterogeneity**. This is the key departure from mixed/random-
coefficient logit, which integrates over an assumed taste distribution. See also
Hetzenecker & Osterhaus (2024) for a related approach to heterogeneous discrete
choice.

## Connection to two-tower embeddings

The connection between two-tower embeddings and the FLM framework is natural:

- The **consumer embedding** $X_i$ captures the latent heterogeneity dimensions.
- The **neural network** maps $X_i \to \theta(X_i)$, the consumer's taste parameters.
- The **item attributes** $T_i$ include prices and (PCA-reduced) item embeddings.

This is what lets the package recover a full taste vector
$\theta(X_i) = (\beta_{\text{price}}(X_i), \beta_{\text{style},1}(X_i), \ldots)$
for every consumer, and then average any target $H$ over the population to obtain
$\mu^*$ with a valid confidence interval.

```{seealso}
For implementation details (data encoding, the score/Hessian blocks, and sample
size requirements for valid multinomial coverage), see the
[Multinomial tutorial](../tutorials/multinomial.md).
```
