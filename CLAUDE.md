# deep-inference

# Core Rules

**Always follow these rules:**

1. **Always `git push` when done** - Push changes after completing work
2. **Never remove content** - Content removed is content lost
   - If something seems outdated, find a new home for it
   - Archive to `archive/` rather than delete
3. **Update `CHANGELOG.md`** after making changes - 1-2 line summary per day

---

Influence function validation for neural network inference.
Implements Farrell, Liang, Misra (2021, 2025) approach.

## Rules for Claude

- **SHOW ME THE FACTS** - Report EVERY metric, EVERY statistic, EVERY number computed. No summaries. No opinions. No hiding. Raw data only. The user does not trust you - prove yourself with transparency.
- **BENCHMARKING** - The goal is to benchmark the Neural Network against the Oracle. Show ALL methods side by side: Oracle (Naive SE), Oracle (Delta SE), NN Naive, NN IF. Show estimates, SEs, CIs, coverage, bias for EVERY method. This is what benchmarking means.
- **VERBOSE STDOUT REPORTS** - All validation runs MUST capture full stdout to a report file. Use `tee` to save output: `python3 -m evals.run_all 2>&1 | tee evals/evals_report.txt`. Never summarize - the raw output IS the proof.
- **ALWAYS SAVE EVAL REPORTS** - When running any eval, ALWAYS save output to `evals/reports/` and show the full path:
  ```bash
  python3 -m evals.eval_01_theta 2>&1 | tee evals/reports/eval_01_$(date +%Y%m%d_%H%M%S).txt
  ```
  After running, ALWAYS print: `Report saved to: /Users/pranjal/deepest/evals/reports/<filename>`
- **ALWAYS SHOW PATHS** - In every chat, always show the full paths of:
  1. The `.py` file being run or modified
  2. The `.txt` report file generated
  Example: "Running `/Users/pranjal/deepest/evals/eval_03_lambda.py` → Report: `/Users/pranjal/deepest/evals/reports/eval_03_20260113_212445.txt`"
- **ALWAYS SHOW THE CODE** - When implementing new families, features, or making changes, ALWAYS show the full code in your response. Don't just describe what you're doing - show the actual implementation. The user wants to see the code.
- **RUTHLESS EVALS** - Evals are firewalls. They MUST be brutal. We WANT to see FAIL when implementation is wrong. Tight tolerances, multiple test cases, no mercy. A passing eval that misses bugs is worse than useless.
- Show all the statistics.
- NO overrides, placeholders, or deviating from the plan - no matter how hard it gets
- Follow the plan exactly as specified
- If something doesn't work, fix it properly instead of using workarounds

## What This Is

Neural nets output structural parameters directly (NOT DML).
Influence functions correct for regularization bias.

Target: μ* = E[β(X)] with valid 95% confidence intervals.

## Package Structure

```
src/deep_inference/
├── __init__.py              # Main APIs: structural_dml(), inference(), did()
├── _did_closed.py           # Closed-form 2x2 DiD (did(method='exact'))
├── core/
│   ├── algorithm.py         # Legacy DML algorithm
│   ├── autodiff.py          # Gradient/Hessian computation
│   └── lambda_estimator.py  # Legacy Lambda estimation
├── families/                 # 13 GLM families (legacy API)
│   ├── base.py              # BaseFamily protocol
│   ├── linear.py, logit.py, poisson.py, gamma.py
│   ├── gumbel.py, tobit.py, negbin.py, weibull.py
│   ├── multinomial.py       # Conditional logit (McFadden)
├── models/                   # NEW: Structural models
│   ├── base.py              # StructuralModel protocol, CustomModel, model_from_loss()
│   ├── linear.py, logit.py  # Built-in models
│   ├── multinomial.py       # Conditional logit (McFadden) J alternatives
│   ├── combinatorial.py     # Multi-treatment model (DeDL2025, 4 link functions)
│   ├── quantile.py          # Quantile regression (smoothed check function)
│   ├── did.py               # Saturated 2x2 DiD model, DiDModel (did(method='neural'))
│   ├── panel_fe.py          # Two-way FE panel DiD, FEPanelDiDModel (did(method='panel_fe'))
│   └── structural_net.py    # Neural network θ(x)
├── targets/                  # NEW: Target functionals
│   ├── base.py              # Target protocol, BaseTarget, CustomTarget, target_from_fn()
│   ├── average_param.py     # E[θ_j] target (AverageParameter, AverageBeta)
│   ├── marginal_effect.py   # AME target
│   ├── elasticity.py        # Price elasticity of demand (DM2023)
│   ├── wtp.py               # Willingness to pay -β_attr/β_price (DM2023)
│   ├── welfare.py           # Consumer surplus / logsum (Small-Rosen 1981, DM2023)
│   ├── dose_response.py     # E[G(θ, t̃)] average predicted outcome (CL2026)
│   ├── profit.py            # E[t̃·G(θ, t̃)] expected revenue (DM2023)
│   ├── tail_probability.py  # E[P(Y > c | θ, t̃)] exceedance prob (GDR2026)
│   ├── conditional_variance.py # E[Var(Y | θ, t̃)] model variance (GDR2026)
│   ├── multi_treatment.py   # E[G(θ,t) - G(θ,t₀)] combinatorial ATE (DeDL2025)
│   └── choice_probability.py # Multinomial choice probability + AME
├── lambda_/                  # NEW: Lambda strategies
│   ├── base.py              # LambdaStrategy protocol
│   ├── compute.py           # ComputeLambda (Regime A: randomized)
│   ├── analytic.py          # AnalyticLambda (Regime B: linear)
│   ├── estimate.py          # EstimateLambda (Regime C: observational)
│   └── selector.py          # detect_regime(), select_lambda_strategy()
├── engine/                   # NEW: Cross-fitting engine
│   ├── crossfit.py          # CrossFitter, run_crossfit()
│   ├── assembler.py         # compute_psi() - influence function assembly
│   └── variance.py          # SE and CI computation
├── autodiff/                 # NEW: Autodiff utilities
│   ├── jacobian.py          # vmap Jacobians for targets
│   ├── hessian.py           # Hessian computation via autodiff
│   └── score.py             # Score (gradient) computation via autodiff
└── utils/
    ├── linalg.py            # Linear algebra helpers (PSD projection, inversion)
    ├── fe.py                # Two-way FE within-transform for panel DiD
    ├── formatting.py        # statsmodels-style summary formatting
    └── result_mixin.py      # Shared predict/plot methods for result objects

archive/deep_inference_v1/    # Old implementation (MC tools, DGPs, etc.)
```

## Quick Start

### Legacy API: `structural_dml()` (production-ready)

```python
import numpy as np
from deep_inference import structural_dml

# Generate data
np.random.seed(42)
n = 1000
X = np.random.randn(n, 5)
T = np.random.randn(n)
Y = X[:, 0] + 0.5 * T + np.random.randn(n)

# Run influence function inference
result = structural_dml(Y, T, X, family='linear', epochs=50, n_folds=50)
print(f"Estimate: {result.mu_hat:.4f} +/- {result.se:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### New API: `inference()` (flexible targets & regimes)

```python
from deep_inference import inference
from deep_inference.lambda_.compute import Normal

# Same data as above...

# Flexible target: Average Marginal Effect
result = inference(Y, T, X, model='logit', target='ame', t_tilde=0.0)

# Custom target function (Jacobian computed via autodiff)
import torch
def my_target(x, theta, t_tilde):
    return torch.sigmoid(theta[0] + theta[1] * t_tilde)

result = inference(Y, T, X, model='logit', target_fn=my_target, t_tilde=0.0)

# Randomized experiment (Regime A: compute Λ instead of estimate)
result = inference(Y, T, X, model='logit', target='beta',
                   is_randomized=True, treatment_dist=Normal(0, 1))
```

### Three Regimes

| Regime | When | Lambda Method | Cross-Fitting |
|--------|------|---------------|---------------|
| A | RCT with known F_T | Compute (MC integration) | 2-way |
| B | Linear model | Analytic (closed-form) | 2-way |
| C | Observational, nonlinear | Estimate (neural net) | 3-way |

## Monte Carlo Tools (Archived)

MC validation tools are archived in `archive/deep_inference_v1/`:
- `dgp.py` - 8 DGPs
- `run_mc.py` - Monte Carlo entry point
- `metrics.py` - Metrics computation
- `logging.py` - JSON logging

For validation studies, use scripts in `prototypes/`.

## Comprehensive Logging

Every run generates TWO log files in `--log-dir` (default: `logs/`):

**Full paths (example with `--log-dir logs`):**
```
/Users/pranjal/deepest/logs/mc_run_20260105_132821.log          # JSON (all data)
/Users/pranjal/deepest/logs/mc_run_20260105_132821_readable.txt # Human summary
```

**File descriptions:**
1. **`.log`** - Machine-readable JSON containing ALL data (for AI/programmatic analysis)
2. **`_readable.txt`** - Human-readable summary (for quick review)

### Log Contents (JSON structure)

```json
{
  "meta": {"generated": "...", "version": "1.0", "framework": "FLM"},
  "config": {"M": 500, "N": 2000, "n_folds": 50, "epochs": 100, ...},
  "dgp": {"alpha_star": "...", "beta_star": "...", "mu_true": -0.168},
  "phase1_recovery": {"naive": {"rmse_alpha": X, "rmse_beta": X, ...}},
  "phase2_inference": {"naive": {"coverage": X, "se_ratio": X, ...}},
  "phase3_diagnostics": {"influence": {"correction_ratio": X, ...}},
  "training_quality": {"influence": {"val_loss": X, "train_val_gap": X, ...}},
  "validation_scorecard": {"coverage_pass": true, "overall_grade": "PASS"},
  "raw_data": [...],  // All M*2 simulation results
  "timing": {"total_seconds": X, "per_sim_seconds": X}
}
```

### Console Output Tables

The run prints 4 tables:
1. **PHASE 1: PARAMETER RECOVERY** - RMSE_α, RMSE_β, Corr_α, Corr_β
2. **PHASE 2: INFERENCE** - μ*, Bias, Var, SE(emp), SE(est), Ratio, Coverage
3. **PHASE 3: TRAINING DIAGNOSTICS** - Grad→0, β_std, R_corr, min(Λ), cond(Λ)
4. **TRAINING QUALITY** - ValLoss, TrainLoss, Gap, BestEpoch, Overfit%

Plus a **VALIDATION SCORECARD SUMMARY** with pass/fail grades.

### Key Metrics to Check

| Metric | Target | What it Tests |
|--------|--------|---------------|
| Coverage | 93-97% | Valid confidence intervals |
| SE Ratio | 0.9-1.2 | SE calibration |
| R_corr | 0.1-1.0 | IF correction magnitude |
| min(Λ) | > 1e-4 | Hessian stability |
| train_val_gap | ≈ 0 | No overfitting |
| Violation Rate | 30-70% | IF is necessary |

### Reading Logs Programmatically

```python
import json

# Load full report
with open("logs/mc_run_20260105_132821.log") as f:
    report = json.load(f)

# Access specific sections
print(report["phase2_inference"]["influence"]["coverage"])  # 0.95
print(report["validation_scorecard"]["overall_grade"])       # "PASS"

# Get all raw simulation results
for sim in report["raw_data"]:
    print(f"Sim {sim['sim_id']}: {sim['method']} coverage={sim['covered']}")
```

## Key Files

- `src/deep_inference/` - Main package (structural_dml API)
- `evals/` - Ground truth validation scripts (see below)
- `archive/deep_inference_v1/CLAUDE.md` - Detailed simulation study spec
- `references/` - Academic papers
- `paper/` - Our paper (LaTeX)
- `prototypes/` - Experiments and validation scripts
- `archive/deepstats_v1/` - Older implementation (v1)
- `archive/deep_inference_v1/` - Previous implementation (v2, MC tools)

## Evals Folder (Ground Truth Validation)

The `evals/` folder contains isolated validation scripts for EVERY mathematical object in Theorem 2:

```
evals/
├── dgp.py           # Canonical DGP: Heterogeneous Logistic Demand (Regime C)
├── eval_01_theta.py # Parameter recovery: θ̂(x) vs θ*(x)
├── eval_02_autodiff.py # Score & Hessian: autodiff vs calculus formulas
├── eval_03_lambda.py   # Lambda estimation: Λ̂(x) vs E[ℓ_θθ|X=x]
├── eval_04_jacobian.py # Target Jacobian: H_θ autodiff vs chain rule
├── eval_05_psi.py      # Influence function assembly: ψ package vs Oracle
├── eval_06_coverage.py # Frequentist coverage: Monte Carlo validation
├── eval_09_multinomial.py # Multinomial logit: recovery + autodiff + lambda + coverage
├── eval_10_new_targets.py # New targets: DoseResponse, Profit, ConditionalVariance Jacobians + coverage
├── eval_11_combinatorial.py # Combinatorial model: link functions, ATEs, Jacobians, Lambda integral
├── eval_12_quantile.py     # Quantile model: recovery, autodiff, lambda, coverage (M=50)
├── dgp_multinomial.py  # Multinomial logit DGP with oracle formulas
├── dgp_quantile.py     # Quantile DGP: location-shift model with oracle formulas
├── common/
│   └── metrics.py      # Standardized thresholds + validation functions
└── run_all.py          # Run all evals, produce full report
```

**Run all evals:**
```bash
python3 -m evals.run_all 2>&1 | tee evals/evals_report.txt
```

**Quick mode (faster, smaller samples):**
```bash
python3 -m evals.run_all --quick 2>&1 | tee evals/evals_report_quick.txt
```

**Eval 06 Settings (NON-NEGOTIABLE):**
- **M=50** replications minimum (detects systematic failure)
- **n=5000** samples (matches eval_01)
- **epochs=200**, **patience=50** (proper convergence)
- **lambda_method='ridge'** (default, validated 96% coverage)
- **t_tilde=0.0** (must match DGP's mu_true() definition)

## How to do E2E User Runs

**Benchmarking = comparing NN to Oracle. Show EVERYTHING. No opinions. Only facts.**

### Step 1: Set Config

```python
# DGP
A0, A1 = 1.0, 0.3      # alpha(X) = A0 + A1*X
B0, B1 = 0.5, 0.2      # beta(X) = B0 + B1*X
MU_TRUE = 0.5          # E[beta(X)] = B0 (since E[X]=0)

# Oracle MC
M_ORACLE = 100         # replications (fast)
N = 1000               # sample size

# NN (single run only - MC too slow)
NN_SEED = 42
EPOCHS = 100
N_FOLDS = 50
HIDDEN_DIMS = [64, 32]
LR = 0.01
# lambda_method='ridge' is now the default (validated 96% coverage)
```

### Step 2: Run Oracle MC (M=100)

```python
oracle_results = []
for seed in range(1, M_ORACLE+1):
    np.random.seed(seed)
    X = np.random.normal(0, 1, N)
    T = np.random.normal(0, 1, N)
    p = expit((A0 + A1*X) + (B0 + B1*X)*T)
    Y = np.random.binomial(1, p).astype(float)

    # Fit oracle logistic regression
    X_design = np.column_stack([np.ones(N), X, T, X*T])
    model = sm.Logit(Y, X_design).fit(disp=0)
    b0, b1 = model.params[2], model.params[3]
    cov = model.cov_params()

    # Compute estimates and SEs
    mu = b0 + b1*X.mean()
    se_naive = sqrt(cov[2,2] + X.mean()**2*cov[3,3] + 2*X.mean()*cov[2,3])
    se_delta = sqrt(se_naive**2 + b1**2*(X.var()/N))

    # Store: seed, mu, se_naive, se_delta, CI, covers, bias
```

### Step 3: Run NN (single run)

```python
np.random.seed(NN_SEED)
# Generate data...

nn = structural_dml(
    Y=Y, T=T, X=X.reshape(-1,1),
    family='logit',
    # lambda_method='ridge' is default (validated 96% coverage)
    epochs=EPOCHS, n_folds=N_FOLDS,
    hidden_dims=HIDDEN_DIMS, lr=LR
)

# Extract:
# - mu_naive = nn.theta_hat[:,1].mean()
# - se_naive = nn.theta_hat[:,1].std() / sqrt(N)
# - mu_hat = nn.mu_hat (IF corrected)
# - se = nn.se (IF)
# - All diagnostics from nn.diagnostics
```

### Step 4: Report EVERYTHING

**ORACLE MC TABLE** (all M rows):
```
Seed  mu_hat    SE_naive  SE_delta  CI_naive           CI_delta           Cov_N  Cov_D  Bias
1     0.58114   0.07715   0.07722   [0.43, 0.73]       [0.43, 0.73]       True   True   0.081
2     0.54577   0.08005   0.08084   [0.39, 0.70]       [0.39, 0.70]       True   True   0.046
...
```

**ORACLE SUMMARY**:
```
Mean estimate: X.XXXXXX
Empirical SE: X.XXXXXX
Mean bias: X.XXXXXX

Naive SE: Mean=X.XX, Ratio=X.XX, Coverage=XX/100=XX%
Delta SE: Mean=X.XX, Ratio=X.XX, Coverage=XX/100=XX%
```

**NN RESULTS** (single run):
```
mu_naive: X.XXXXXX
mu_hat: X.XXXXXX
se_naive: X.XXXXXX
se (IF): X.XXXXXX
CI_naive: [X.XX, X.XX]
CI_IF: [X.XX, X.XX]
Covers_naive: True/False
Covers_IF: True/False
```

**NN DIAGNOSTICS**:
```
min_lambda_eigenvalue: X.XXXXXX
correction_ratio: X.XX
Corr(alpha): X.XXXX
Corr(beta): X.XXXX
```

**FINAL COMPARISON**:
```
Method          Estimate  SE        CI_lo     CI_hi     Covers  Bias      Coverage(MC)
Oracle_Naive    X.XXXXX   X.XXXXX   —         —         —       X.XXXXX   XX%
Oracle_Delta    X.XXXXX   X.XXXXX   —         —         —       X.XXXXX   XX%
NN_Naive        X.XXXXX   X.XXXXX   X.XXXXX   X.XXXXX   T/F     X.XXXXX   N/A
NN_IF           X.XXXXX   X.XXXXX   X.XXXXX   X.XXXXX   T/F     X.XXXXX   N/A
```

### Key Expectations

| Method | Coverage | SE Ratio | Notes |
|--------|----------|----------|-------|
| Oracle_Naive | ~95-98% | ~1.0 | Gold standard |
| Oracle_Delta | ~95-98% | ~1.0 | Accounts for Var(X̄) |
| NN_Naive | **LOW** (~0-20%) | **<<1** (~0.15) | Overconfident - BAD |
| NN_IF | **~95%** | **~1.0** | Matches Oracle - GOOD |

### Report Location

Full benchmark: `tutorials/02_logit_oracle.ipynb`

## GLM Family Formulas

| Family | Link | Loss (NLL) | Gradient | Hessian Weight | θ_dim |
|--------|------|------------|----------|----------------|-------|
| **Linear** | Identity | `(y-μ)²` | `-2(y-μ)·[1,t]` | `2` (constant) | 2 |
| **Gaussian** | Identity | `(y-μ)²/(2σ²) + log(σ)` | `[(μ-y)/σ², t(μ-y)/σ², 1-(y-μ)²/σ²]` | Depends on σ | 3 |
| **Logit** | Logit | `log(1+exp(η)) - y·η` | `(p-y)·[1,t]` | `p(1-p)` | 2 |
| **Poisson** | Log | `λ - y·log(λ)` | `(λ-y)·[1,t]` | `λ` | 2 |
| **NegBin** | Log | `-lgamma(y+r) + ...` | `r(μ-y)/(r+μ)·[1,t]` | `rμ(r+y)/(r+μ)²` | 2 |
| **Gamma** | Log | `y/μ + log(μ)` | `(1-y/μ)·[1,t]` | `y/μ` | 2 |
| **Weibull** | Log | `-log(k) + k·log(λ) - (k-1)log(y) + zᵏ` | `k(1-z)·[1,t]` | `k²z` | 2 |
| **Gumbel** | Identity | `z + exp(-z)` | `(-1/σ)(1-exp(-z))·[1,t]` | `exp(-z)/σ²` | 2 |
| **Tobit** | Identity | Censored normal NLL | Autodiff (Mills ratio) | Autodiff | 3 |
| **Probit** | Φ(η) | `-y·log(Φ) - (1-y)·log(1-Φ)` | Mills ratio | Autodiff | 2 |
| **Beta** | Logit | `lgamma(μφ) + lgamma((1-μ)φ) - ...` | Digamma terms | Autodiff | 2 |
| **ZIP** | Mixed | Mixture: π + (1-π)·Poisson | Autodiff | Autodiff | 4 |
| **MultinomialLogit** | Softmax | `-log softmax(V)[y]` | `(p_j - 1{Y=j})·[1,x]` | `p_j(δ_{jm}-p_m)` blocks | (J-1)+K |
| **Quantile** | Identity | `τ·u + ε·log(1+exp(-u/ε))` | `(1-τ-σ(u/ε))·[1,t]` | `σ(u/ε)(1-σ(u/ε))/ε` | 2 |

Where: `η = α + β·t`, `μ = g⁻¹(η)`, `z = (y/λ)^k` (Weibull) or `(y-μ)/σ` (Gumbel), `r = 1/overdispersion`, `Φ` = normal CDF

## Lambda Method Recommendations

For `structural_dml()` with nonlinear models (logit, poisson, etc.), the package defaults to `lambda_method='ridge'`:

| Method | Corr w/ Oracle | Coverage | Speed | Default | Notes |
|--------|----------------|----------|-------|---------|-------|
| **ridge** | 0.508 | **96%** | Fast | **Yes** | Safe default, validated coverage |
| **aggregate** | 0.000 | 95% | Fast | No | Ignores X-dependence |
| **lgbm** | 0.978 | 96% | Fast | No | Good accuracy alternative |
| **mlp** | 0.997 | **67%** | Slow | No | **AVOID** - invalid SEs |
| **rf** | 0.904 | ~90% | Medium | No | Moderate accuracy |

### WARNING: Avoid MLP for Lambda Estimation

The package now warns when `lambda_method='mlp'` is used:

```
UserWarning: Lambda method 'mlp' can produce invalid standard errors
despite high correlation with oracle. Consider method='ridge' (default)
or 'lgbm' for validated coverage.
```

**Key finding:** MLP achieves highest correlation (0.997) but produces **67% coverage** instead of 95%. This is because high correlation doesn't guarantee valid inference - the variance of the Lambda estimates matters too.

### Package Defaults (as of v0.2)

```python
# src/deep_inference/core/algorithm.py
lambda_method: str = 'ridge',    # Was 'mlp'
ridge_alpha: float = 1000.0,     # Was 1.0

# src/deep_inference/core/lambda_estimator.py
method: Literal[...] = 'ridge',  # Was 'mlp'
ridge_alpha: float = 1000.0,     # Was 1.0

# src/deep_inference/lambda_/estimate.py
method: Literal[...] = 'ridge',  # Was 'aggregate'
ridge_alpha: float = 1000.0,     # Was 1.0
```

### Ridge regularization (critical!)

Ridge requires **heavy regularization** (α=1000) to produce stable Lambda estimates:

```python
Ridge(alpha=1000.0)  # Pull predictions toward mean
```

Without this (old default α=1.0), ridge produces catastrophic failure: Bias=66, SE ratio=0.5.

### LGBM regularization (if using lgbm)

LGBM requires **heavy regularization** to produce stable Lambda estimates:

```python
LGBMRegressor(
    n_estimators=20,        # Very few trees
    max_depth=2,            # Very shallow
    min_child_samples=150,  # Many samples per leaf
    reg_alpha=5.0,          # Strong L1
    reg_lambda=5.0,         # Strong L2
)
```

Without this, LGBM can produce negative eigenvalues → numerical instability.

### Coverage Results (M=50)

```
Method         Corr    Coverage   SE Ratio       Bias   Status
---------------------------------------------------------------
ridge          0.508      96.0%      0.914    -0.0064     PASS
aggregate      0.000      95.0%      1.000    -0.0102     PASS
lgbm           0.978      96.0%      0.999    -0.0093     PASS
mlp            0.997      67.0%      0.650    -0.0150     FAIL
```

**Conclusion**: Use ridge (default), aggregate, or lgbm. **Never use mlp** for production.

## Paper Transcriptions

Docling-transcribed (markdown with equations), 22 papers:

**Core Framework:**
- `references/FLM2021_docling.md` - FLM 2021 DNN Estimation & Inference (*Econometrica*)
- `references/FLM2023_docling.md` - FLM 2023 Deep Learning for Heterogeneity (earlier revision)
- `references/FLM2025_docling.md` - FLM 2025 Deep Learning for Individual Heterogeneity

**Applications:**
- `references/DM2023_docling.md` - Dubé & Misra 2023 Personalized Pricing (*JPE*)
- `references/HO2024_multinomial_docling.md` - Hetzenecker & Osterhaus 2024 Multinomial Choice
- `references/CL2026_docling.md` - Colangelo & Lee 2026 Continuous Treatments (*JBES*)
- `references/CLMZ2024_docling.md` - Chen, Liu, Ma, Zhang 2024 General Treatment Effects (*J Econometrics*)
- `references/DeDL2025_docling.md` - Ye et al. 2025 Combinatorial Experiments (*Management Science*)

**Automatic Debiasing / Riesz:**
- `references/RieszNet2022_docling.md` - Chernozhukov et al. 2022 RieszNet (*ICML*)
- `references/CNSS2020_adversarial_riesz_docling.md` - Chernozhukov, Newey, Singh 2022 (*Econometrica*)
- `references/IN_glr_docling.md` - CNQS 2021 Automatic Debiased ML via Neural Nets for GLR
- `references/HH2025_docling.md` - Hines & Hines 2025 Moment-Constrained Debiasing (*CLeaR*)

**DNN Architecture + IF:**
- `references/Dragonnet2019_docling.md` - Shi, Blei, Veitch 2019 DragonNet (*NeurIPS*)
- `references/TDA2025_docling.md` - Li, McCoy et al. 2025 Targeted Deep Architectures
- `references/DeepLTMLE2024_docling.md` - Shirakawa et al. 2024 Deep LTMLE (*ICML*)
- `references/DNASE2024_docling.md` - Liu et al. 2024 DNA-SE (*ICML*)
- `references/CLearner2025_docling.md` - Cai et al. 2025 C-Learner

**Theory:**
- `references/YCY2025_docling.md` - Yan, Chen, Yao 2025 Overparameterized NN Semiparametric
- `references/Metzger2022_docling.md` - Metzger 2022 Adversarial Estimators
- `references/FS2023_docling.md` - Foster & Syrgkanis 2023 Orthogonal Statistical Learning (*Ann Stat*)

**Frontier:**
- `references/GDR2026_docling.md` - Melnychuk & Feuerriegel 2026 GDR-Learners (*ICLR*)
- `references/NNES2025_docling.md` - Nguyen 2025 NNES for DDC Models (JMP)

Legacy plaintext: `references/FLM2021.txt`, `references/FLM2025.txt`

Cross-reference report: `references/verification_report.md`

## Standardized Eval Metrics

Central thresholds in `evals/common/metrics.py`:

| Category | Metric | Threshold | Tests |
|----------|--------|-----------|-------|
| Coverage | coverage | 90-99% | Valid CIs |
| Coverage | se_ratio | 0.7-1.5 | SE calibration |
| Coverage | abs_bias | < 0.05 | Unbiasedness |
| Coverage | z_mean | (-0.3, 0.3) | z-score centering |
| Coverage | z_std | 0.7-1.5 | z-score spread |
| Recovery | rmse | < 0.3 | θ̂ accuracy |
| Recovery | correlation | > 0.7 | θ̂ manifold |
| Autodiff | error | < 1e-6 | Precision |
| Lambda | frobenius | < 0.15 | Λ̂ accuracy |
| Lambda | min_eig | > 1e-4 | Stability |
| Lambda | non_psd | = 0 | PSD guarantee |
| Psi | correlation | > 0.9 | ψ accuracy |
| Psi | rmse | < 0.1 | ψ scale |

Functions: `validate_coverage_run()`, `validate_recovery_run()`, `validate_autodiff_run()`, `validate_lambda_run()`, `validate_psi_run()`, `format_validation_table()`

## Multinomial Logit (Conditional Logit / McFadden)

**Paper**: Hetzenecker & Osterhaus (2024) — "Deep Learning for Heterogeneous Parameters in Discrete Choice Models" (arXiv 2408.09560)

### Model

```
P(Y=j | W, X) = exp(V_ij) / Σ_m exp(V_im)
V_ij = α_j(W) + x'_ij · β(W)

α_0 = 0 (reference), θ = [α_1,...,α_{J-1}, β_1,...,β_K]
```

Data encoding:
- `W` = individual characteristics (NN input → θ(W))
- `T` = packed alternative attributes: `(n, J*K)`, unpacked to `(J, K)` inside model
- `Y` = chosen alternative index: `(n,)` as float (0, 1, ..., J-1)
- `theta_dim` = (J-1) + K

### Usage

```python
# Legacy API
result = structural_dml(Y, T, X, family='multinomial_logit',
                        n_alternatives=3, n_attributes=2)

# New API
result = inference(Y, T, X, model='multinomial_logit', target='beta')
```

### DGP (eval_09)

J=3, K=2, d_w=3. Heterogeneous α_j(W) = a0_j + a1_j*W[0], β_k(W) = b0_k + b1_k*W[0].
Target: μ* = E[β_1(W)] = -0.8.

### Key Properties

| Property | Value |
|----------|-------|
| `hessian_depends_on_theta` | True (softmax probs) |
| `hessian_depends_on_y` | False (Fisher information) |
| Regime (randomized) | A (compute Λ via MC) |
| Regime (observational) | C (estimate Λ, 3-way split) |

## Results Document (Automated Pipeline)

**Generator:** `/Users/pranjal/Dropbox/deep-aesthetics/final-analysis/generate_results.py`
**Output:** `/Users/pranjal/Dropbox/deep-aesthetics/paper/results.tex` + `results.pdf`
**Figures:** `/Users/pranjal/Dropbox/deep-aesthetics/final-analysis/results_figures/` (22 PDFs)

### Usage

```bash
cd /Users/pranjal/Dropbox/deep-aesthetics/final-analysis

# Full pipeline (all sections, all figures, compile PDF)
python3 generate_results.py

# Selective sections
python3 generate_results.py --sections eda hedonic event_item

# Figures only (no LaTeX/PDF)
python3 generate_results.py --figures-only

# Skip compilation
python3 generate_results.py --no-compile
```

### Architecture (8 layers, ~1500 lines)

1. **Imports + Constants** — paths, colors
2. **Path Registry** — 30+ file paths (txt, json, npy, parquet)
3. **Parsers** (8) — regex-based extraction from output .txt files
4. **Figure Generators** (22) — matplotlib, saved as PDF
5. **LaTeX Table Generators** — f-string templates
6. **Section Functions** (8) — each returns (figs, latex)
7. **TEX Assembly + Compilation** — pdflatex x2
8. **CLI** — argparse entry point

### 8 Sections

| Section | Figures | Tables | Key Sources |
|---------|---------|--------|-------------|
| Data | 1 | 1 | ref_prices.npy |
| EDA | 3 | 0 | item/user embeddings, alpha arrays |
| Embeddings | 4 | 2 | 08_train.txt, 21_train.txt, 24_comparison.txt |
| Choice | 3 | 2 | fit_logs, biz_apps, alpha .npy |
| Hedonic | 4 | 4 | ablation x3, price_indices x3, residuals |
| Event (Items) | 3 | 2 | event_study JSON x2, comparison x2 |
| Event (Users) | 3 | 4 | user JSON x2, comparison x2 |
| Scorecard | 1 | 1 | aggregated from other sections |

- Auto-generated — do NOT hand-edit `results.tex`
- Re-run after any analysis pipeline changes
- For evaluation only — not part of the paper

## References

- Farrell, Liang, Misra (2021): Econometrica
- Farrell, Liang, Misra (2025): Extended theory
- Hetzenecker, Osterhaus (2024): arXiv 2408.09560
