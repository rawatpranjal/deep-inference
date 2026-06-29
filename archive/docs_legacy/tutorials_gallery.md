# Gallery

Eight validated examples demonstrating `deep-inference` across model families.

---

## At a Glance

| # | Model | Outcome | Treatment | Covariates | Result |
|---|-------|---------|-----------|------------|--------|
| 1 | **Linear** | Log wages | Experience (years) | Job embeddings (64-dim) | CI covers true |
| 2 | **Logit** | Purchase (0/1) | Discount (%) | Product embeddings (64-dim) | CI covers true |
| 3 | **Poisson** | Citations | Open Access (0/1) | Abstract embeddings (64-dim) | CI covers true |
| 4 | **Tobit** | Donation ($) | Match ratio | Donor demographics | CI covers true |
| 5 | **Gamma** | Claim amount ($) | Deductible ($) | Policyholder features | CI covers true |
| 6 | **Weibull** | Months to churn | Discount offered | Customer profile | CI covers true |
| 7 | **Multinomial Logit** | Transport mode | Travel time (min) | Commuter attributes | CI covers true |
| 8 | **Gaussian** | Part diameter (mm) | Machine speed | Sensor readings | CI covers true |

All eight achieve valid 95% CI coverage with influence function correction.

---

## 1. Linear: Wage Returns to Experience

**Question:** How does experience affect wages? Does the effect vary by job type?

$$Y_i = \alpha(X_i) + \beta(X_i) \cdot T_i + \varepsilon_i$$

```python
from deep_inference import structural_dml

result = structural_dml(
    Y=wages, T=experience, X=job_embeddings,
    family='linear',
    epochs=200, n_folds=50
)

print(result.summary())
```

**Validation:** Corr(true, estimated) = 0.985

---

## 2. Logit: Discount Effectiveness

**Question:** Do discounts increase purchases? Which products respond most?

$$P(Y_i = 1) = \sigma(\alpha(X_i) + \beta(X_i) \cdot T_i)$$

```python
result = structural_dml(
    Y=purchased, T=discount_pct, X=product_embeddings,
    family='logit',
    epochs=200, n_folds=50
)

print(result.summary())

# Who should get discounts?
beta_hat = result.theta_hat[:, 1]
high_responders = beta_hat > np.median(beta_hat)
```

**Validation:** Corr(true, estimated) = 0.421

---

## 3. Poisson: Open Access Citation Advantage

**Question:** Does Open Access increase citations? Which papers benefit most?

$$Y_i \sim \text{Poisson}(\exp(\alpha(X_i) + \beta(X_i) \cdot T_i))$$

```python
result = structural_dml(
    Y=citations, T=open_access, X=abstract_embeddings,
    family='poisson',
    epochs=200, n_folds=50
)

print(result.summary())

# Citation multiplier
print(f"OA multiplier: {np.exp(result.mu_hat):.2f}x")
```

**Validation:** Corr(true, estimated) = 0.709

---

## 4. Tobit: Charitable Donations

**Question:** Does employer matching increase donations? By how much, accounting for the zero-censoring mass?

$$Y_i^* = \alpha(X_i) + \beta(X_i) \cdot T_i + \varepsilon_i, \quad Y_i = \max(0, Y_i^*)$$

Many donors give \$0. The Tobit model separates the propensity to give from the amount.

```python
result = structural_dml(
    Y=donation_amount, T=match_ratio, X=donor_features,
    family='tobit',
    epochs=200, n_folds=50
)

print(result.summary())

# Fraction of donors at zero
print(f"Zero mass: {(donation_amount == 0).mean():.1%}")
```

**Validation:** eval_01 recovery PASS (RMSE < 0.15, Corr > 0.8). See [Tobit tutorial](tobit.md).

---

## 5. Gamma: Insurance Claims

**Question:** Do higher deductibles reduce claim severity? Which policyholders are most price-sensitive?

$$Y_i \sim \text{Gamma}\bigl(\text{shape},\; \exp(\alpha(X_i) + \beta(X_i) \cdot T_i)\bigr)$$

Claim amounts are strictly positive and right-skewed — the Gamma family handles this naturally.

```python
result = structural_dml(
    Y=claim_amount, T=deductible, X=policyholder_features,
    family='gamma',
    epochs=200, n_folds=50
)

print(result.summary())

# Percent change in expected claim per $100 deductible increase
print(f"Elasticity: {result.mu_hat:.4f}")
```

**Validation:** eval_01 recovery PASS. See [Gamma tutorial](gamma.md).

---

## 6. Weibull: Customer Churn

**Question:** Does offering a discount extend customer lifetime? Who benefits most from retention offers?

$$Y_i \sim \text{Weibull}\bigl(k,\; \exp(\alpha(X_i) + \beta(X_i) \cdot T_i)\bigr)$$

Subscription durations are positive and often right-skewed with hazard rates that change over time.

```python
result = structural_dml(
    Y=months_subscribed, T=discount_offered, X=customer_profile,
    family='weibull',
    epochs=200, n_folds=50
)

print(result.summary())

# Who retains longest with discount?
beta_hat = result.theta_hat[:, 1]
best_targets = beta_hat > np.percentile(beta_hat, 75)
```

**Validation:** eval_01 recovery PASS. See [Weibull tutorial](weibull.md).

---

## 7. Multinomial Logit: Transportation Mode Choice

**Question:** How does travel time affect mode choice (car, bus, train)? Which commuters are most responsive?

$$P(Y_i = j \mid W_i, X_i) = \frac{\exp(V_{ij})}{\sum_m \exp(V_{im})}, \quad V_{ij} = \alpha_j(W_i) + x'_{ij} \cdot \beta(W_i)$$

With J=3 alternatives and K=2 attributes (time, cost), the model estimates heterogeneous preferences.

```python
result = structural_dml(
    Y=chosen_mode, T=alternative_attributes, X=commuter_features,
    family='multinomial_logit',
    n_alternatives=3, n_attributes=2,
    epochs=300, patience=50, n_folds=50
)

print(result.summary())
```

**Validation:** eval_09 coverage 98% (SE ratio 0.97). See [Multinomial tutorial](multinomial.md).

---

## 8. Gaussian: Manufacturing Quality Control

**Question:** Does machine speed affect part diameter? Is the variance also heterogeneous?

$$Y_i \sim \mathcal{N}\bigl(\alpha(X_i) + \beta(X_i) \cdot T_i,\; \sigma^2(X_i)\bigr)$$

Unlike `family='linear'`, the Gaussian family estimates heteroscedastic variance $\sigma(X)$ as a third parameter.

```python
result = structural_dml(
    Y=part_diameter, T=machine_speed, X=sensor_readings,
    family='gaussian',
    epochs=200, n_folds=50
)

print(result.summary())

# Estimated noise level
sigma_hat = np.exp(result.theta_hat[:, 2])
print(f"Mean sigma: {sigma_hat.mean():.4f}")
```

**Validation:** eval_01 recovery PASS (sigma recovery RMSE < 0.05). See the [full family list](index.md).

---

## Run It Yourself

```bash
# Full gallery with validation output
python tutorials/06_multimodal_gallery.py
```

Individual tutorials are available for each family:
[Linear](linear.md) | [Logit](logit.md) | [Poisson](poisson.md) | [Tobit](tobit.md) | [Gamma](gamma.md) | [Weibull](weibull.md) | [Multinomial](multinomial.md) | [Gumbel](gumbel.md) | [NegBin](negbin.md)

See [Multimodal Tutorial](multimodal.md) for detailed code with real embedding examples (BERT, ResNet, CLIP).
