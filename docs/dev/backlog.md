# Expanding deep-inference: 10 High-Impact Causal Inference Models

Research on 10 causal inference models for expanding the deep-inference package, including:
- (a) Structural loss
- (b) Common targets
- (c) Influence function representation
- (d) FOC and Hessians

Plus paper links, package links, and implementation notes.

---

## **Priority 1: Essential Causal Inference Methods (Highest Impact)**

### **1. ATE/CATE - Double/Debiased Machine Learning**

| Resource | Link | Notes |
|----------|------|-------|
| **DoubleML Python Package** | https://github.com/DoubleML/doubleml-for-py | Official implementation of Chernozhukov et al. (2018) |
| **DoubleML Documentation** | https://docs.doubleml.org/ | Comprehensive user guide and API reference |
| **DoubleML R Package** | https://github.com/DoubleML/doubleml-for-r | CRAN: https://cran.r-project.org/package=DoubleML |
| **Original Paper (PDF)** | https://arxiv.org/abs/1801.09197 | Chernozhukov et al. (2018), *Econometrica* |
| **JMLR Paper on DoubleML** | https://www.jmlr.org/papers/v23/21-0862.html | Bach et al. (2022) on package implementation |

**Key Insight:** The influence function for ATE is $\psi(Z) = \frac{A}{\pi(X)}(Y - \mu_1(X)) - \frac{1-A}{1-\pi(X)}(Y - \mu_0(X)) + \mu_1(X) - \mu_0(X) - \theta$

| Component | Details |
|-----------|---------|
| **Structural Loss** | Augmented IPW (AIPW) or Efficient Influence Function (EIF) approach |
| **Common Targets** | ATE: $E[Y(1) - Y(0)]$; CATE: $E[Y(1) - Y(0) \| X=x]$; ATT; Heterogeneous treatment effects |
| **IF Representation** | $\psi(Z) = \frac{A}{\pi(X)}(Y - \mu_1(X)) - \frac{1-A}{1-\pi(X)}(Y - \mu_0(X)) + \mu_1(X) - \mu_0(X) - \theta$ |
| **FOC/Hessian** | Score: $\nabla_\theta \ell = -(\psi(Z) - \theta)$; Hessian: $\nabla^2_\theta \ell = I$ (identity, constant) |

**Implementation Notes:** This is the "hello world" of causal inference. The EIF combines outcome regression ($\mu_a(X)$) and propensity score ($\pi(X)$) estimation. The Riesz representer for ATE is exactly the inverse probability weight.

---

### **2. Deep Learning for Individual Heterogeneity (FLM Framework)**

| Resource | Link | Notes |
|----------|------|-------|
| **Farrell-Liang-Misra (2021) Paper** | https://arxiv.org/abs/2010.14694 | Deep neural networks for estimation and inference |
| **PyTorch Implementation** | https://github.com/rmmomin/causal-ml-auto-inference | Community implementation of FLM |
| **PDF of Paper** | https://tyliang.github.io/Tengyuan.Liang/pdf/preprint-Farrell-Liang-Misra-20.pdf | Direct PDF access |

**Key Insight:** The influence function correction is $\psi_i = H_i - H_{\theta,i} \cdot \Lambda_i^{-1} \cdot \ell_{\theta,i}$

---

### **3. Partial Linear Model (Robinson, 1988)**

| Resource | Link | Notes |
|----------|------|-------|
| **DoubleML PLR Implementation** | https://docs.doubleml.org/ | `DoubleMLPLR` class |
| **Original Robinson Paper** | Search via JSTOR/IDEAS | *Econometrica* 1988 |

| Component | Details |
|-----------|---------|
| **Structural Loss** | $Y = \theta A + g(X) + \epsilon$ where $g(X)$ is nonparametric nuisance |
| **Common Targets** | Average treatment effect $\theta$; average partial effects; policy effects |
| **IF Representation** | $\psi(Z) = \frac{(A - E[A\|X])(Y - E[Y\|X] - \theta(A - E[A\|X]))}{E[\text{Var}(A\|X)]}$ |
| **FOC/Hessian** | Score involves residuals from both treatment and outcome models; Hessian depends on conditional variance |

**Implementation Notes:** The "partialling out" estimator is semiparametric efficient. The influence function shows the double-residual structure—residualize both treatment and outcome, then regress residual-outcome on residual-treatment.

---

### **4. LATE/IV Methods**

| Resource | Link | Notes |
|----------|------|-------|
| **DoubleML PLIV/IIVM** | https://docs.doubleml.org/ | Partially linear and interactive IV models |
| **Abadie (2003) Bootstrap** | Search via *Econometrica* | Bootstrap for weak instruments |

| Component | Details |
|-----------|---------|
| **Structural Loss** | Wald estimator framework: $Y = \beta D + \epsilon$, with $Z$ as instrument |
| **Common Targets** | LATE for compliers: $E[Y(1) - Y(0) \| D(1) > D(0)]$; Marginal Treatment Effects (MTE) |
| **IF Representation** | $\psi(Z) = \frac{Z}{\pi(X)}(Y - \mu_1(X)) - \frac{1-Z}{1-\pi(X)}(Y - \mu_0(X))$ for instrument, scaled by first-stage |
| **FOC/Hessian** | Two-stage: first-stage (compliance) and second-stage (outcome) scores |

**Implementation Notes:** The MTE framework extends LATE to continuous instruments via derivatives of outcome with respect to propensity score. The influence function for LATE is more complex due to the ratio structure (Wald = reduced form / first stage).

---

### **5. Marginal Treatment Effects (MTE)**

| Resource | Link | Notes |
|----------|------|-------|
| **grmpy Python Package** | https://github.com/OpenSourceEconomics/grmpy | Generalized Roy Model estimation with MTE |
| **Heckman-Vytlacil (2005)** | https://arxiv.org/pdf/2404.03235 | Structural equations, treatment effects |

| Component | Details |
|-----------|---------|
| **Structural Loss** | Selection equation: $D = I[Z'\gamma > V]$; Outcome: $Y = D \cdot Y_1 + (1-D) \cdot Y_0$ |
| **Common Targets** | $MTE(X, u_D) = E[Y_1 - Y_0 \| X, U_D = u_D]$; Policy effects as weighted integrals |
| **IF Representation** | Derivative-based: $\frac{\partial E[Y\|X, P(Z)=p]}{\partial p}$ at point $p = u_D$ |
| **FOC/Hessian** | Requires estimation of derivative of conditional expectation; involves local polynomial smoothing |

**Key Insight:** MTE is identified by $\frac{\partial E[Y|X, P(Z)=p]}{\partial p}$ at point $p = u_D$

**Implementation Notes:** The MTE is identified by the derivative of the outcome with respect to the propensity score. This connects directly to the FLM framework—neural networks can estimate $E[Y|X, P(Z)]$ and automatic differentiation computes the derivative.

---

## **Priority 2: Advanced Causal & Statistical Methods (High Impact)**

### **6. Causal Mediation Analysis**

| Resource | Link | Notes |
|----------|------|-------|
| **DeepMed Python Package** | https://pypi.org/project/deepmed/ | `pip install DeepMed` |
| **DeepMed R Package** | https://github.com/siqixu/DeepMed | Original R implementation |
| **DeepMed Paper (NeurIPS 2022)** | Search NeurIPS proceedings | Xu, Liu, Liu (2022) |
| **Tchetgen Tchetgen (2013)** | https://pmc.ncbi.nlm.nih.gov/articles/PMC4710381/ | Semiparametric theory for mediation |

| Component | Details |
|-----------|---------|
| **Structural Loss** | Sequential ignorability: Natural Direct Effect (NDE) and Natural Indirect Effect (NIE) |
| **Common Targets** | NDE, NIE, controlled direct effects; proportion mediated |
| **IF Representation** | $\psi^{NDE} = \frac{A}{\pi(X)}(M - \eta_1(X))(Y - \mu_{1M}(X,M)) + \mu_{1M}(X,M) - \mu_{0M}(X,M) - \theta^{NDE}$ |
| **FOC/Hessian** | Three nuisance functions: mediator model, outcome model (both treatments), propensity score |

**Implementation Notes:** DeepMed (NeurIPS 2022) shows how to use neural networks for cross-fitting mediation analysis with valid IF-based inference. The EIF involves products of nuisance errors, making it sensitive to estimation quality.

---

### **7. Quantile Treatment Effects / Distribution Regression**

| Resource | Link | Notes |
|----------|------|-------|
| **Counterfactual R Package** | https://github.com/bmelly/discreteQ | Chernozhukov, Fernández-Val, Melly |
| **Vignette (PDF)** | https://ifs.org.uk/sites/default/files/output_url_files/CWP641717.pdf | Counterfactual analysis in R |
| **Original Paper (PDF)** | https://www.econstor.eu/bitstream/10419/79512/1/746173318.pdf | *Econometrica* 2013 |
| **MIT Open Access** | https://dspace.mit.edu/bitstream/handle/1721.1/95960/Chernozhukov_Inference%20on.pdf | Inference on counterfactual distributions |

| Component | Details |
|-----------|---------|
| **Structural Loss** | Check function: $\rho_\tau(Y - q_\tau(X))$ where $\rho_\tau(u) = u(\tau - I[u<0])$ |
| **Common Targets** | Quantile treatment effects (QTE); unconditional QTE (via RIF regression) |
| **IF Representation** | RIF: $\psi(Y, F) = q_\tau + \frac{\tau - I[Y \leq q_\tau]}{f(q_\tau)}$ |
| **FOC/Hessian** | Score: $\tau - I[Y \leq q_\tau]$; Hessian involves density at quantile (sparsity function) |

**Implementation Notes:** The Recentered Influence Function (RIF) approach allows regression of quantiles on covariates. For treatment effects, we need the difference of RIFs between treatment arms. The density estimation at the quantile is the challenging nuisance.

---

### **8. Survival Analysis / Cox Proportional Hazards**

| Resource | Link | Notes |
|----------|------|-------|
| **PyTMLE (Python)** | https://www.medrxiv.org/content/10.1101/2025.07.02.25330730v1 | Targeted ML estimation for survival |
| **Martingale Residuals (R)** | https://www.rdocumentation.org/packages/rms/versions/8.1-0/topics/residuals.cph | `residuals.cph` function |
| **Cox Diagnostics** | https://www.sthda.com/english/wiki/cox-model-assumptions | Schoenfeld, martingale residuals |

| Component | Details |
|-----------|---------|
| **Structural Loss** | Partial likelihood: $\prod_i \left(\frac{\exp(\theta'X_i)}{\sum_{j:Y_j \geq Y_i} \exp(\theta'X_j)}\right)^{\Delta_i}$ |
| **Common Targets** | Log hazard ratio $\theta$; survival probabilities $S(t\|X)$; restricted mean survival time |
| **IF Representation** | Score residual: $\Delta_i (X_i - \bar{X}(Y_i; \theta))$ where $\bar{X}$ is risk-set average |
| **FOC/Hessian** | Score: martingale residual; Hessian: observed information (time-varying) |

**Implementation Notes:** The Cox model is semiparametric with time-varying nuisance (baseline hazard). The influence function involves martingale theory. For the FLM framework, we need time-varying covariates and competing risks extensions.

---

### **9. Sample Selection / Heckman Model**

| Resource | Link | Notes |
|----------|------|-------|
| **StatsModels Heckman** | https://zgcharaf.medium.com/adjusting-for-selection-bias-in-credit-scoring-models-using-python-762f0880f802 | `sm.heckman.Heckman` |
| **PyHeckman Implementation** | http://dspace.mit.edu/bitstream/handle/1721.1/2144/SWP-1793-15720588.pdf | MIT lecture notes with code |
| **Selection Bias Guide** | https://medium.com/@gorfein1/unraveling-selection-bias-a-guide-to-robust-causal-inference-e08298e2255e | Python examples |

| Component | Details |
|-----------|---------|
| **Structural Loss** | Two-stage: Selection $D = I[Z'\gamma + \eta > 0]$; Outcome $Y = X'\beta + \epsilon$ (observed only if $D=1$) |
| **Common Targets** | Treatment effects corrected for selection; policy effects on selected population |
| **IF Representation** | IF for Heckman two-stage involves inverse Mills ratio correction |
| **FOC/Hessian** | Bivariate normal likelihood; selection bias correction term |

**Implementation Notes:** The influence function for Heckman's estimator is unbounded—small deviations from normality cause large bias. Robust versions exist. For FLM, we can estimate the selection equation and outcome equation with neural networks, with IF correction for the two-stage structure.

---

### **10. Dynamic Treatment Regimes / Off-Policy Evaluation**

| Resource | Link | Notes |
|----------|------|-------|
| **DRL Paper (JMLR)** | https://jmlr.csail.mit.edu/papers/volume21/19-827/19-827.pdf | Kallus & Uehara (2020) |
| **Interpretable OPE (ICML 2020)** | https://arxiv.org/abs/2002.03478 | Gottesman et al., influence functions for FQE |
| **OPE with Optimal Policies** | https://arxiv.org/html/2505.13809v2 | Recent IF characterization |

| Component | Details |
|-----------|---------|
| **Structural Loss** | Fitted Q-evaluation: $Q^\pi(S,A) = R + \gamma E[Q^\pi(S', \pi(S'))]$ |
| **Common Targets** | Value of policy $\pi$; regret compared to optimal; safe policy improvement |
| **IF Representation** | Doubly robust for RL: $\psi = \frac{\pi(A\|S)}{\mu(A\|S)}(R + \gamma V(S') - Q(S,A)) + Q(S,A) - V(S)$ |
| **FOC/Hessian** | Bellman residual; value function approximation |

**Implementation Notes:** Influence functions for OPE in RL are actively researched. The doubly robust estimator combines importance sampling and Q-function approximation. For continuous actions, we need Riesz representers for the Bellman operator.

---

### **11. Average Derivative Estimation**

| Resource | Link | Notes |
|----------|------|-------|
| **Powell-Stock-Stoker (1989)** | http://dspace.mit.edu/bitstream/handle/1721.1/2144/SWP-1793-15720588.pdf | Density-weighted ADE |
| **Stoker (1986)** | Search *Econometrica* | Original average derivative |
| **Smoothness Adaptive ADE** | https://sticerd.lse.ac.uk/dps/em/em529.pdf | Cattaneo, Crump, Jansson |

| Component | Details |
|-----------|---------|
| **Structural Loss** | $Y = g(X'\beta) + \epsilon$; target is $\beta$ (up to scale) |
| **Common Targets** | Average derivative $\delta = E[\nabla g(X'\beta)]$; weighted ADE for identification |
| **IF Representation** | $\psi(Z) = \bar{v}(W)\frac{\partial \gamma_0(W)}{\partial W} - \beta + \alpha(X)[Y - \gamma_0(W)]$ |
| **FOC/Hessian** | Weighted derivative of conditional expectation; involves density weighting |

**Implementation Notes:** The influence function involves integration by parts to convert derivative of regression to regression of derivative. The weight function $\bar{v}(W)$ depends on the density of $W$.

---

## **Priority 3: Major Python/R Packages for Implementation**

| Package | Language | Link | Models Covered |
|---------|----------|------|---------------|
| **DoubleML** | Python/R | https://docs.doubleml.org/ | ATE, PLR, PLIV, IRM, IIVM |
| **EconML** | Python | https://github.com/py-why/EconML | CATE, IV, policy learning |
| **CausalML** | Python | https://github.com/uber/causalml | Meta-learners (T, S, X, R) |
| **DeepMed** | Python/R | https://pypi.org/project/deepmed/ | Mediation analysis |
| **grmpy** | Python | https://github.com/OpenSourceEconomics/grmpy | MTE, Generalized Roy |
| **GenericML** | R | https://github.com/mwelz/GenericML | Heterogeneous treatment effects |
| **Counterfactual** | R | https://github.com/bmelly/discreteQ | Quantile/distribution regression |
| **PyTMLE** | Python | https://www.medrxiv.org/content/10.1101/2025.07.02.25330730v1 | Survival, competing risks |

---

## **Foundational Theory Papers (PDF Links)**

| Paper | Authors | Link |
|-------|---------|------|
| Large Sample Estimation and Hypothesis Testing | Newey & McFadden (1994) | https://statweb.rutgers.edu/ztan/material/newey-mcfadden.pdf |
| Double/Debiased Machine Learning | Chernozhukov et al. (2018) | https://arxiv.org/abs/1801.09197 |
| Deep Neural Networks for Estimation and Inference | Farrell, Liang, Misra (2021) | https://arxiv.org/abs/2010.14694 |
| Inference on Counterfactual Distributions | Chernozhukov, Fernández-Val, Melly (2013) | https://www.econstor.eu/bitstream/10419/79512/1/746173318.pdf |
| Semiparametric Estimation of Weighted Average Derivatives | Powell, Stock, Stoker (1989) | http://dspace.mit.edu/bitstream/handle/1721.1/2144/SWP-1793-15720588.pdf |

---

## **Summary: Recommended Implementation Order**

1. **Start here:** ATE/CATE with DoubleML framework
2. **Add deep learning:** FLM framework for individual heterogeneity
3. **Extend to IV:** LATE and partial linear IV models
4. **Add distributional effects:** Quantile treatment effects
5. **Add dynamic settings:** Survival analysis and OPE
6. **Advanced:** MTE, mediation, and selection models

All these models have established influence function representations and can be integrated into the deep-inference package using the FLM framework's three-regime approach (Lambda estimation via ridge regression with cross-fitting).

---

## **Summary Table: Implementation Priority**

| Priority | Model | Key Papers | Complexity | Impact Score |
|----------|-------|-----------|------------|--------------|
| 1 | ATE/CATE | Chernozhukov et al. (2018), Farrell et al. (2021) | Low | ★★★★★ |
| 2 | Partial Linear | Robinson (1988), Chernozhukov et al. (2018) | Low | ★★★★★ |
| 3 | LATE/IV | Imbens & Angrist (1994), Abadie (2003) | Medium | ★★★★★ |
| 4 | MTE | Heckman & Vytlacil (2005), Carneiro et al. (2011) | High | ★★★★☆ |
| 5 | Mediation | Tchetgen Tchetgen (2013), DeepMed (2022) | High | ★★★★☆ |
| 6 | Survival/Cox | Cox (1972), Reid & Crepeau (1985) | Medium | ★★★★☆ |
| 7 | Quantile Effects | Firpo (2007), Chernozhukov et al. (2013) | Medium | ★★★★☆ |
| 8 | Heckman Selection | Heckman (1979), Zhelonkin et al. (2016) | High | ★★★☆☆ |
| 9 | Average Derivative | Stoker (1986), Ai & Chen (2007) | High | ★★★☆☆ |
| 10 | OPE/RL | Gottesman et al. (2020), Kallus & Uehara (2022) | Very High | ★★★★☆ |

---

## **Critical Implementation Notes**

**For all models, the key insight from FLM applies:**
- Use neural networks to estimate nuisance functions (conditional expectations, densities, propensity scores)
- Compute influence functions via automatic differentiation
- Apply cross-fitting to avoid overfitting bias
- Use the IF-corrected standard errors, not naive plug-in SEs

**Regime Classification for New Models:**
- **Regime A (RCT):** ATE with known randomization, LATE with known instrument assignment
- **Regime B (Linear):** Partial linear model with identity link (Hessian constant)
- **Regime C (General):** Nonlinear models (logit, Cox, quantile, selection models) requiring 3-way splitting and ridge Lambda estimation

**Missing Models (Lower Priority or No IF Yet):**
- Causal forests (Wager & Athey) - uses bootstrap not IF
- Continuous-time survival (intensity models) - theoretically complex
- Network interference - active research area, IF representations not standardized

---

## **Backlog Add (2026-06-09): Doubly-Robust DiD — Sant'Anna & Zhao (2020), Callaway & Sant'Anna (2021)**

Natural extension of the already-shipped `did()` family (closed-form 2×2, neural,
two-way FE). Upgrades the panel DiD from FE/closed-form to a **doubly-robust,
semiparametrically efficient** EIF-based estimator. SEs come natively from the
empirical IF variance — no bootstrapping.

**Build order — 2×2 first, event-study second.**

### Step 1 — DR-DiD 2×2 (Sant'Anna & Zhao 2020)
Panel-data EIF for the ATT (τ), baseline covariates X, treatment D∈{0,1},
outcome change ΔY = Y_t − Y_{t−1}:

ψ(W) = (D/E[D])·(ΔY − m_{0,Δ}(X) − τ) − (p(X)(1−D))/(E[D](1−p(X)))·(ΔY − m_{0,Δ}(X))

| Component | Details |
|-----------|---------|
| Nuisances | p(X)=P(D=1\|X) propensity; m_{0,Δ}(X)=E[ΔY\|D=0,X] control outcome-evolution |
| Target | ATT τ; doubly robust (consistent if either p or m_{0,Δ} correct) |
| IF / SE | Var̂(τ̂) = (1/N²)Σ ψ̂_i² — single empirical IF variance, no bootstrap |
| Panel advantage | EIF does NOT depend on m_{1,1}, m_{1,0}; within-unit differencing absorbs time-invariant confounders → tighter efficiency bound than repeated cross-sections |
| Repeated cross-section variant | Needs four marginal outcome models m_{d,t}(X) + λ sample-proportion term; defer |

Slots into the existing IF assembler (`engine/assembler.py`) — nuisances p(X),
m_{0,Δ}(X) are exactly the FLM-style cross-fitted nets we already train.

### Step 2 — Staggered event-study (Callaway & Sant'Anna 2021)
Compute a separate IF for each group-time ATT(g,t), then **aggregate the influence
functions** to derive event-study parameters and uniform confidence bands. Builds
directly on Step 1's per-(g,t) DR estimator.

**Refs:** Sant'Anna & Zhao (2020) *J. Econometrics* (DR-DiD); Callaway & Sant'Anna
(2021) *J. Econometrics* (multiple periods); R pkg `DRDID` / `did` for cross-checks.
Already flagged in `handoff.md` as the DiD next-extension.
