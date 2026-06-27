# Competitive landscape for deep-inference

**Status.** Internal decider doc. Landscape as of 2026-06-27. Maintenance dates and feature claims for other packages are taken from each package's own documentation (fetched on this date) and the maintenance survey collected in the planning thread. Cells I could not confirm against a package's own docs are marked unverified rather than asserted. Characterizations of individual corpus papers (DragonNet, DeepIV, DeepGMM, RieszNet) are grounded in their full text on disk under `literature/causal-deep-learning-2016-2026/markdown_docling`, not abstracts.

**What this doc is for.** Before expanding `deep-inference` from an FLM influence-function tool into a general deep causal and statistical inference workhorse, we need to know where the white space actually is, so the build targets gaps instead of re-implementing EconML. This is the field map and the gap argument. The build sequence that follows from it lives in [backlog.md](./backlog.md), and the honest list of current weaknesses lives in [known_limitations.md](./known_limitations.md). This doc is the why, those are the what and the caveats.

---

## TL;DR, the gap in one paragraph

The mature packages split into two camps. The honest-SE camp (EconML, DoubleML) gives rigorous asymptotic confidence intervals, but the machine learning is plug-in (you pass an sklearn or Keras model as a nuisance), and the estimand is a low-dimensional treatment-effect projection (ATE, a linear or forest CATE, LATE). The deep-native camp (CausalML's DragonNet and CEVAE, plus the research repos behind TARNet, CFRNet, DeepIV) ships real neural estimators, but the uncertainty is bootstrap or absent, and the product is an ITE or uplift point prediction. Almost nothing sits in the intersection, which is torch-native estimators where the structural parameter is itself the network output, carrying honest influence-function standard errors, across both causal targets and structural-econ functionals (willingness-to-pay, consumer welfare, elasticity, profit, dose-response). That intersection is what `deep-inference` already occupies for the FLM family, and it is the boundary the expansion should defend.

---

## The ecosystem map

**PyWhy** (the umbrella, [pywhy.org](https://www.pywhy.org)) ties three projects under one banner.

- [**DoWhy**](https://github.com/py-why/dowhy) is an end-to-end pipeline, not an estimator zoo. Its value is the four-step discipline (model the graph, identify the estimand, estimate, refute) plus the GCM module for structural causal models. For heterogeneous effects it largely wraps EconML. The differentiator is identification and refutation testing, not the estimators.
- [**EconML**](https://www.pywhy.org/EconML/) (Microsoft) is the heterogeneous-treatment-effect workhorse and the strongest direct competitor. Deep dive below.
- [**causal-learn**](https://github.com/py-why/causal-learn) is the discovery arm (PC, GES, and friends). Different problem from ours, listed for completeness.

[**DoubleML**](https://docs.doubleml.org) is the orthogonal-moments econometrics camp. Honest standard errors are the whole point. Nuisances are plug-in machine learning.

[**CausalML**](https://github.com/uber/causalml) (Uber) is uplift-first and is the one mainstream package that ships native deep estimators (DragonNet, CEVAE).

[**causallib**](https://github.com/BiomedSciAI/causallib) (IBM) is a tidy sklearn-style library for the classic estimators (IPW, standardization, doubly-robust, TMLE) plus a survival module and a strong evaluation suite.

Below these sit the research-grade deep repos (TARNet and CFRNet, DragonNet, CEVAE, DeepIV, the longitudinal CRN and Causal Transformer lines). They are method papers with code, not maintained libraries, and almost none of them carry valid confidence intervals.

---

## Per-package profiles

**EconML** (PyWhy / Microsoft, active through 2025, MIT). Native CATE and DR estimators: `LinearDML`, `SparseLinearDML`, `CausalForestDML`, `NonParamDML`, `KernelDML`, `DRLearner`, `LinearDRLearner`, `ForestDRLearner`, the metalearners `SLearner` / `TLearner` / `XLearner` / `DomainAdaptationLearner`, and the orthogonal forests `DMLOrthoForest` / `DROrthoForest`. Native IV: `OrthoIV`, `DMLIV`, `NonParamDMLIV`, the `DRIV` family, and `SieveTSLS`. Time-varying treatment via `DynamicDML`. Policy learning via `PolicyTree` / `PolicyForest` / `DRPolicy*`. Inference is genuinely honest, since the docs offer both asymptotic (OLS, debiased-lasso, subsampled honest forest) and bootstrap. The deep-learning role is wrap (you pass a Keras or torch model as `model_y` / `model_t`). EconML once shipped a Keras `DeepIV`, but it is legacy and not featured in the current reference. No survival outcomes, no structural-econ targets.

**DoubleML** (Bach, Chernozhukov, et al., active through 2025, BSD). Native model classes: `DoubleMLPLR`, `DoubleMLLPLR`, `DoubleMLPLPR` (panel), `DoubleMLPLIV`, `DoubleMLIRM`, `DoubleMLAPO` / `DoubleMLAPOS`, `DoubleMLIIVM`, `DoubleMLDIDMulti` (staggered adoption), `DoubleMLSSM` (sample selection), `RDFlex` (RDD). Every model reports `coef`, `std err`, and confidence intervals from Neyman-orthogonal scores, which is the gold standard for valid SEs. IV via PLIV and IIVM. The machine-learning role is wrap any sklearn-style learner as a nuisance, with no native deep estimators. The models overview page does not list quantile, policy, or survival classes (some quantile and policy classes exist in the package history, so those rows are marked unverified).

**CausalML** (Uber, v0.15.x released early 2025, Apache-2.0). Meta-learners (S, T, X, R, DR, DRIV), the uplift forest family (KL, Euclidean, chi-square, plus IDDP, IT, CIT, CTS), `2SLS` IV, TMLE for the ATE, matching and IPTW, and crucially the native neural estimators CEVAE and DragonNet. The deep-learning role is native, but inference is bootstrap-style, not influence-function honest SEs. Uplift is the core competence. No survival, no longitudinal, no structural-econ targets.

**causallib** (IBM, v0.10 in 2025, Apache-2.0). The classic estimators behind a clean sklearn-style API: IPW, standardization (direct outcome models), doubly-robust / AIPW, TMLE, plus a survival module and a well-regarded evaluation suite. The machine-learning role is wrap any fit-predict estimator. Standard errors are not its headline, since it leans on bootstrap and the evaluation suite. No IV, policy, or quantile in the headline API.

**DoWhy** (PyWhy, active through 2025, MIT). The pipeline and refutation layer plus GCM for structural causal models. Estimation either uses simple built-ins (IPW, regression, stratification, matching, 2SLS / Wald IV, frontdoor) or delegates heterogeneity to EconML. The deep-learning role is wrap. Its real contribution is identification and the refutation tests, which none of the others package as cleanly.

---

## Coverage matrix

Rows are method families. Columns are the five mature packages plus `deep-inference` as it stands today.

Legend. `native` ships a dedicated native estimator. `wrap` is covered only by wrapping your own model into a generic method. `part` is partial, a related class, or unverified against the package's own docs. `no` is absent.

| Method family | DoWhy | EconML | DoubleML | CausalML | causallib | deep-inference (today) |
|---|---|---|---|---|---|---|
| ATE / ATT | native | native | native | native | native | native |
| CATE / ITE (heterogeneity) | wrap | native | part | native | part | part (theta(X) native, ITE-as-product partial) |
| Orthogonal / debiased (DR, DML) | wrap | native | native | native | native | native (IF-debiased, FLM) |
| IV / LATE / endogeneity | native | native | native | native | no | no |
| DiD / event study | part | no | native | no | no | native (exact, neural, panel-FE) |
| Quantile / QTE | no | no | part | no | no | native |
| Survival CATE (time-to-event) | no | no | no | no | native | part (Weibull GLM only, not causal) |
| Longitudinal / time-varying | no | native | part | no | part | part (panel-FE static, no dynamics) |
| Policy learning / OPE | no | native | part | part | no | no |
| Uplift | no | part | no | native | no | no |
| Continuous / dose-response | part | native | part | part | no | native |
| Structural-econ targets (WTP, welfare, elasticity, profit) | no | no | no | no | no | native |
| Discrete choice (multinomial logit) | no | no | no | no | no | native (McFadden) |
| Conformal / distribution-free UQ | no | no | no | no | no | no |
| Causal discovery | part (GCM) | no | no | no | no | no |
| Network / interference | no | no | no | no | no | no |

Two rows are empty across the whole field. Conformal / distribution-free uncertainty and network / interference are open for everyone, which makes them low-priority differentiators (hard, and nobody is asking for them in a workhorse yet). The rows that matter are the ones where `deep-inference` is `no` or `part` but a competitor is `native`, and the row where `deep-inference` is the only `native`.

---

## The deep-learning angle, specifically

This is the load-bearing section. The question is not who covers IV but who covers IV with a deep estimator and honest SEs. On that axis the field is mostly empty, and our literature corpus ([searchable_catalog.csv](../../literature/causal-deep-learning-2016-2026/searchable_catalog.csv), 86 papers) already holds the methods that fill it. Each family below maps to real rows in that catalog.

- **Deep IV with valid inference.** DeepIV (arXiv 1612.09596), DeepGMM (1905.12495), DFIV (2010.07154), regularized DeepIV (2403.04236), and orthogonality-constrained DeepIV (2506.02790). EconML and DoubleML do IV with honest SEs but classical nuisances. CausalML does 2SLS. DeepIV itself includes frequentist data-splitting and dropout-Bayesian inference (read confirms a dedicated inference section), so the missing piece is a maintained package, not the method. Nobody ships a maintained deep IV estimator with confidence intervals. The `iv` and `deepiv` tags cover eight papers.
- **Deep survival CATE.** SurvITE (2110.14001), BITES (2201.03448), SurvCaus (2203.15672), and the 2026 deep survival learner (2604.10398). causallib has a survival module but it is classical and not a treatment-effect learner. The `survival` tag covers five papers.
- **Longitudinal / time-varying treatment.** The time-series deconfounder (1902.00450), adversarially balanced representations (2002.04083), the Causal Transformer (2204.07258), and longitudinal TMLE with a transformer (2404.04399). EconML's `DynamicDML` is the only mainstream entry and it is not deep. The `longitudinal` tag covers nine papers.
- **Automatic debiasing / Riesz.** Riesz regression (2104.14737), RieszNet and ForestRiesz (2110.03031), and moment-constrained debiasing (2409.19777). RieszNet already delivers asymptotically normal estimates with confidence intervals at near-nominal coverage (the paper reports 95 to 96 percent on IHDP using the doubly-robust moment), so its inference story already matches ours, but it lives only as research code with no maintained-library home. This is the most natural extension of our influence-function machinery. The `auto-dml` and `riesz` tags cover these.
- **Representation CATE with valid CIs.** TARNet and CFRNet (1606.03976), DragonNet (1906.02120). DragonNet's targeted regularization buys good asymptotic properties, but the paper evaluates point-estimate error and not interval coverage, and CausalML ships it as a point estimator. The gap is the same architecture with honest standard errors. The `representation` and `dragonnet` tags cover more than thirteen papers.

The pattern is consistent. Where a competitor exists, it is either honest-but-not-deep (EconML, DoubleML) or deep-but-not-honest (CausalML). The corpus gives us the deep estimators, and the FLM influence-function layer is exactly the honest-SE machinery they lack.

---

## Where deep-inference sits today

Verified against the package's actual exports in `src/deep_inference/__init__.py` and the package tree.

**What it uniquely has.** Influence-function standard errors on a parameter the network outputs directly (the FLM construction, not plug-in DML). A three-regime cross-fitting engine that adapts the Lambda strategy to randomized, linear, or observational settings. Structural-econ targets as first-class objects: willingness-to-pay, consumer welfare, elasticity, profit, dose-response, tail probability, conditional variance. Thirteen GLM families with closed-form properties. A native multinomial logit (McFadden) and native difference-in-differences (exact 2x2, neural saturated, two-way panel FE). No other package combines deep structural parameters, honest SEs, and economic functionals.

**What it lacks.** No instrumental variables. No causal survival learner (the Weibull family is a GLM, not a time-to-event treatment-effect model). No longitudinal dynamics (panel FE is static within-transform). No policy learning or off-policy evaluation. No uplift. No conformal or bootstrap option. No causal discovery. No interference or network effects.

---

## Implications for the expansion

The build-next shortlist is the set of families that are both absent or only partial in `deep-inference`, and a real competitor gap rather than a solved problem, and paper-backed in our corpus. Ranked by leverage, cross-linked to [backlog.md](./backlog.md):

1. **Orthogonal / DR deep learners and Riesz** (`auto-dml`, `riesz`). Closest to what we already do, fills the DR/DML row natively with a deep estimator, and RieszNet has no production home. Highest leverage, lowest new-assumption cost.
2. **Deep IV / DeepGMM** (`iv`, `deepiv`). The most-requested missing family for any marketplace, pricing, or ads setting with endogenous treatment. Honest SEs on a deep IV estimator is genuine white space.
3. **Representation CATE with valid CIs** (`representation`, `dragonnet`). Take the architecture CausalML ships as a point estimator and add our influence-function layer. Direct, defensible, differentiating.
4. **Deep survival CATE** (`survival`). Churn, time-to-conversion, lifetime. causallib's survival is classical, so a deep treatment-effect survival learner is open.
5. **Longitudinal / time-varying basics** (`longitudinal`). Higher cost, but EconML's `DynamicDML` is the only mainstream entry and it is not deep.

Defer policy learning, uplift, conformal, discovery, and interference. Policy and uplift are well-served by EconML and CausalML respectively, and the rest are open-for-everyone rows that do not play to the influence-function strength.

---

## Sources

- PyWhy umbrella, [pywhy.org](https://www.pywhy.org). DoWhy, [github.com/py-why/dowhy](https://github.com/py-why/dowhy). EconML reference, [pywhy.org/EconML](https://www.pywhy.org/EconML/reference.html). causal-learn, [github.com/py-why/causal-learn](https://github.com/py-why/causal-learn).
- DoubleML model classes, [docs.doubleml.org](https://docs.doubleml.org/stable/guide/models.html).
- CausalML methodology, [causalml.readthedocs.io](https://causalml.readthedocs.io/en/latest/methodology.html). Repo, [github.com/uber/causalml](https://github.com/uber/causalml).
- causallib, [github.com/BiomedSciAI/causallib](https://github.com/BiomedSciAI/causallib).
- Literature corpus, [searchable_catalog.csv](../../literature/causal-deep-learning-2016-2026/searchable_catalog.csv) and [SEARCH.md](../../literature/causal-deep-learning-2016-2026/SEARCH.md), 86 papers, 2016-2026.
- Current `deep-inference` surface, `src/deep_inference/__init__.py` (`__all__`) and the package tree.
