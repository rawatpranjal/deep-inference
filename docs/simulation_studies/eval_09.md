# Eval 09: Multinomial Logit

Comprehensive validation of the multinomial logit (conditional logit / McFadden) implementation.

## Configuration

| Parameter | Value (Recovery) | Value (Coverage) |
|-----------|------------------|------------------|
| Sample Size (n) | 10000 | 8000 |
| Alternatives (J) | 3 | 3 |
| Attributes (K) | 2 | 2 |
| Simulations (M) | - | 50 |
| Epochs | 300 | 300 |
| Patience | 50 | 50 |
| Cross-fitting Folds | 50 | 50 |

## DGP: Heterogeneous Conditional Logit

```
J = 3 alternatives, K = 2 attributes, d_w = 3

V_ij = alpha_j(W) + X'_ij * beta(W)
P(Y=j | W, X) = softmax(V)[j]

alpha_0 = 0 (reference)
alpha_1(W) = 0.5 + 0.2*W[0]
alpha_2(W) = -0.3 - 0.1*W[0]
beta_1(W) = -0.8 - 0.2*W[0]
beta_2(W) = 0.5 + 0.1*W[0]

True mu* = E[beta_1(W)] = -0.8
```

## Test 1: Parameter Recovery

| Component | RMSE | Correlation | Status |
|-----------|------|-------------|--------|
| alpha_1 | 0.08 | 0.90 | PASS |
| alpha_2 | 0.12 | 0.78 | PASS |
| beta_1 | 0.09 | 0.88 | PASS |
| beta_2 | 0.10 | 0.85 | PASS |

## Test 2: Autodiff Validation

Score and Hessian computed via autodiff match oracle closed-form formulas.

| Metric | Value | Status |
|--------|-------|--------|
| Max score error | 4.44e-16 | PASS |
| Max Hessian error | 4.44e-16 | PASS |

## Test 3: Lambda Estimation

Monte Carlo integration for E[H | W=w] matches oracle.

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Relative Frobenius error | < 0.15 | < 0.15 | PASS |
| Min eigenvalue | > 1e-4 | > 1e-4 | PASS |
| Non-PSD count | 0 | 0 | PASS |

## Test 4: Coverage (M=50)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Coverage | 98% | 90-99% | PASS |
| SE Ratio | 0.97 | 0.7-1.5 | PASS |
| Bias | -0.006 | < 0.05 | PASS |
| z-score Mean | 0.14 | (-0.3, 0.3) | PASS |
| z-score Std | 0.96 | 0.7-1.5 | PASS |

**EVAL 09: PASS**

## Key Findings

- **patience=50 essential**: Default patience=10 triggers early stopping at ~15-20 epochs, fatal for 3-way split training
- **n >= 8000 required**: 3-way splitting reduces effective training data to 60%; n=5000 gives only 88% coverage
- **correction_ratio ~70-90 is normal**: Much larger than binary logit (~2) due to higher-dimensional theta
- **alpha_2 is hardest**: Weakest signal (slope -0.1) requires most data for reliable recovery

## Run Command

```bash
python3 -m evals.eval_09_multinomial 2>&1 | tee evals/reports/eval_09_$(date +%Y%m%d_%H%M%S).txt
```
