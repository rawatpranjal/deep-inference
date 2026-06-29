# DID + deep-learning replication candidates (scoping menu)

A scope-only survey: difference-in-differences and observational-panel methods built on deep
learning and the FLM / RieszNet (influence-function, automatic-debiasing) machinery, and which
ones this package could replicate. This document is a **menu** to pick from. It implements
nothing. Pick options below and they become the next goal.

Every candidate paper here was downloaded with `websource` into
`references/did_scoping/` and read from the on-disk full text (PDF plus a `tomd` markdown
conversion). No paper is cited from an abstract; each entry carries a verbatim span with its
file path so the claim is checkable.

---

## 1. What the package already does for DID

Three estimators behind one entry point `deep_inference.did(...)` (`src/deep_inference/__init__.py:1030`),
all targeting the average effect `E[tau(X)]` via the FLM influence function:

| Method | Call | Model | Estimand | Regime |
|---|---|---|---|---|
| `exact` | `did(Y, group, post)` | closed-form 2x2 (`_did_closed.py`) | `beta = mu11-mu10-mu01+mu00`, SE = saturated-OLS HC0 | none (closed form) |
| `neural` | `did(Y, group, post, X=X)` | `DiDModel` (`models/did.py`): `Y = alpha(X)+gamma(X)G+lambda(X)P+tau(X)·GP`, theta=[a,g,l,tau], target `E[tau(X)]` | heterogeneous 2x2 ATT | B (2-way, analytic `Lambda=E[WW'|X]`) |
| `panel_fe` | `did(Y, group, post, X=X, unit=u, time=t)` | `FEPanelDiDModel` (`models/panel_fe.py`): within-transform `Ytilde = Dtilde·tau(X)`, target `E[tau(X)]`, HC-robust SE, binary (LPM) ok | two-way FE panel ATT | B (2-way, `Lambda=E[Dtilde^2|X]`) |

Validation lives in `evals/eval_13_did.py`, `evals/eval_14_did_nn.py`, the DGP
`evals/dgp_did_nn.py`, and `src/deep_inference/tests/test_did.py`.

**Two structural facts that decide what replicates cleanly.**

1. **Everything is 2x2 (or within-transformed panel).** There is no staggered, multi-period,
   or group-time `ATT(g,t)` machinery. The package's DiD is the canonical two-group two-period
   design with covariate-heterogeneous effects.
2. **The current DiD eval assigns treatment INDEPENDENTLY of X.** `dgp_did_nn.py` states it
   plainly: "group G and period P are assigned independently of the covariates X (only the
   OUTCOME means depend on X)." Under that design `E[WW'|X]` is constant, so the Regime-B
   aggregate `Lambda` is exactly right. **Selection on X (the propensity to be treated depends
   on covariates) is not yet tested.** This is the crux for the DR-DiD literature, whose entire
   point is robustness under exactly that selection. It also connects to the open Lambda-collapse
   finding for the FLM observational case (`docs/notes/flm_lambda_se_undercount.md`): when the
   group propensity `e(X)` varies, the two-way aggregate `Lambda` collapses it to a constant.

---

## 2. Candidate papers (downloaded and read)

### C1. Sant'Anna & Zhao (2020), Doubly Robust Difference-in-Differences Estimators
`references/did_scoping/arXiv 1812.01723.pdf` (Journal of Econometrics 219:101-122). Code: R package `DRDID`.

- **Estimand.** Unconditional ATT under conditional parallel trends, `tau = E[Y1(1)-Y1(0) | D=1]`.
- **Deep learning?** No. Nuisances are logistic PS + OLS outcome regression. The DR moment is
  estimator-agnostic, so neural nuisances are a stated future extension.
- **Riesz / DR?** Yes. The classical doubly-robust DiD score (their eq. 3.1/3.7); the IPW weight
  `pi(X)(1-D)/(1-pi(X))` is the DiD Riesz representer in disguise.
- **Simulation.** n=1000, 10,000 reps, true ATT=0, four Kang-Schafer DGPs crossing whether the
  PS and OR working models are correctly specified. **Treatment is selected on covariates:**
  `D = 1{p(Z) >= U}`. Metrics: avg/median bias, RMSE, coverage, CI length. Verbatim
  (`arXiv 1812.01723/01-front-matter-1.md:1135`): "We consider sample size n equal to 1000. For
  each design, we conduct 10,000 Monte Carlo simulations. We compare the various DID estimators
  for the ATT in terms of average bias, median bias, root mean square error (RMSE), empirical
  95% coverage probability, the average length of a 95% confidence interval ...".
- **Target table (Table 1, panel, DR-DiD row `tau_dr,p`).** DGP1 (both models correct): bias
  -0.001, RMSE 0.106, coverage 0.947, CIL 0.412. DGP2 (OR right, PS wrong): coverage 0.947.
  DGP3 (OR wrong, PS right): coverage 0.942. DGP4 (both wrong): coverage 0.308 (DR breaks, as
  expected; the improved `tau_dr,p_imp` row is 0.274). TWFE `tau_fe`: coverage ~0.00 in every DGP ("severely biased ... almost zero
  coverage", `:1226`), because it rules out covariate-specific trends.

### C2. Bach, Klaassen, Kueck, Mattes & Spindler (2025), Sensitivity Analysis for DiD using Riesz Representation
`references/did_scoping/arXiv 2510.09064.pdf` (arXiv preprint). Code: `DoubleML` Python package.

- **Estimand.** DiD ATT (2x2 and staggered), plus a sensitivity bound for parallel-trends
  violations from an unobserved confounder.
- **Deep learning?** No (ridge / logistic / random forest nuisances), but the construction is
  exactly RieszNet-shaped.
- **Riesz / DR?** Yes, and this is the load-bearing reason to include it: it writes the DiD ATT
  Riesz representer in closed form. Verbatim (`arXiv 2510.09064/01-front-matter.md:644`):
  `alpha(W) = D/p - (1-D)·m(X,A)/(p·(1-m(X,A)))`, with `m(X,A)=E[D|X,A]`, `p=P(D=1)`. That is the
  DR-DiD weight of C1, stated as a Riesz representer a RieszNet can target directly.
- **Simulation.** Extends the C1 DGP1 with an unobserved confounder; true ATT=5;
  n in {500,...,50000}; 10,000 reps; linear learners. Point-estimation table at n=500: the
  short estimator covers near nominal; the standard DML interval's coverage of the true ATT
  decays from 0.768 (n=500) toward 0.000 as n grows under the parallel-trends violation (that
  decay is the sensitivity story, not a defect).

### C3. Lan, Chang, Dillon & Syrgkanis (2025), A Meta-learner for Heterogeneous Effects in DiD
`references/did_scoping/arXiv 2502.04699.pdf` (arXiv / ICML-affiliated).

- **Estimand.** The CATT function `theta_0(X) = E[Y1(1)-Y1(0) | D=1, X]` (a function, not a
  scalar aggregate).
- **Deep learning?** Yes. Neural nets are one of the second-stage meta-learners (with XGBoost,
  linear) and an optional nuisance learner.
- **Riesz / DR?** Yes. A Neyman-orthogonal conditional moment turned into a convex
  doubly-robust loss; nuisances `pi(W)=P(D=1|W)` and `g0(W)=E[DeltaY|D=0,W]`.
- **Simulation.** 2x2 pre/post, treatment `D ~ Binomial(p)` with `p` a nonlinear function of
  observed W and unobserved U, true CATT `theta_0 = (1/2)·W1·1(W2>0)`, parallel trends conditional
  on full W (`arXiv 2502.04699/02-part-2.md:332`). 100 reps. **Metric is MSE of the CATT
  function, no coverage / SE.** Table 1 best: Neural-Net-DR 0.10, XGBoost-DR 0.04 (CATT MSE).

### C4. Zhang (2025), Continuous Difference-in-Differences with Double/Debiased ML
`references/did_scoping/arXiv 2408.10509.pdf` (arXiv preprint). No public code found.

- **Estimand.** Dose-specific ATT curve `ATT(d) = E[Yt(d)-Yt(0) | D=d]` for a CONTINUOUS dose.
- **Deep learning?** No (random forest nuisances; NN named as possible). Neyman-orthogonal DML
  score, kernel-smoothed over `d` (non-root-N, `sqrt(Nh)`).
- **Simulation.** p=100 covariates, continuous dose confounded by X, true `ATT(d)=-0.5 d^2`,
  evaluated at d=0.9; n in {2000,10000}; 500 reps. Panel coverage 0.918 (n=2000), 0.930 (n=10000).
- Needs a conditional-density nuisance `f_h(d|X)` and a kernel bandwidth schedule.

### C5. Callaway & Sant'Anna (2021), Difference-in-Differences with Multiple Time Periods
`references/did_scoping/arXiv 1803.09015.pdf` (Journal of Econometrics 225:200-230). Code: R `did`, Python `csdid`.

- **Estimand.** Group-time `ATT(g,t)` with staggered adoption, plus event-study / overall
  aggregations. DR estimand (their eq. 2.4), consistent if PS or OR is right.
- **Deep learning?** No (logit/probit PS, parametric OR).
- **Simulation.** The Monte Carlo lives in a Supplementary Appendix that is NOT in the arXiv
  source on disk (`arXiv 1803.09015/01-front-matter-1.md:206` points to the external supp PDF),
  so the DGP and numbers would need that file before a replication.
- **Requires multiple periods and staggered timing.** The canonical 2x2 is a degenerate special
  case; the load-bearing object is the multi-cohort structure.

### C6. Chernozhukov, Newey, Singh & Syrgkanis (2023), Auto-Debiased ML for Dynamic Treatment Effects and Nested Functionals
`references/did_scoping/arXiv 2203.13887.pdf` (arXiv preprint).

- **Estimand.** `E[Y(tau)]` for a sequence of treatments over time, via M recursive nested
  regressions and a recursive Riesz representer. **Not a DiD paper** (no parallel trends, no
  ATT framing) and **no simulation section at all** (pure theory). Verbatim
  (`arXiv 2203.13887/01-front-matter.md:28`): "the multiply robust formula for the dynamic
  treatment regime ... can be re-stated in terms of a recursive Riesz representer characterization
  of nested mean regressions." Included only as the frontier the package's DiD is a small special
  case of; there is nothing here to replicate.

### Not downloaded
- **Chang (2020), Double/debiased ML for DiD** (Econometrics Journal 23:177-191) is paywalled
  (OUP returned HTTP 403 via `websource`); no arXiv/working-paper PDF resolved. It is the
  canonical Neyman-orthogonal DML-DiD reference and would be worth a manual pull if its sim is
  wanted as a target.

---

## 3. Clean-fit scoping (per candidate)

| # | Paper | Verdict | Package seam it would use |
|---|---|---|---|
| C1 | Sant'Anna-Zhao DR-DiD | **Drop-in to small-lift** | `did(method='neural')` as-is on the C1 DGP; the only new code is the DGP generator and a TWFE baseline row |
| C2 | Riesz-DiD representer | **Small-lift to medium** | new DiD target in `riesz/inference.py` using the closed-form `alpha(W)`; reuses the RieszNet engine + `engine/variance.py` |
| C3 | Meta-learner CATT | **Small-lift, different metric** | `did(method='neural')` outputs `theta_hat[:,3]=tau(X)`; score CATT-MSE on a test split. No coverage to match |
| C4 | Continuous-dose DiD | **Research item** | new score + kernel bandwidth + conditional-density nuisance; reuses `engine/crossfit.py` only |
| C5 | Callaway-Sant'Anna staggered | **Research item** | new multi-period `ATT(g,t)` model + aggregation; current 2x2 is a special case. Also needs the external supp DGP |
| C6 | Dynamic nested Riesz | **Not a target** | recursive sequential Riesz infra that does not exist; no simulation to match |

**The honest risk on C1 (read before picking).** The package's `did(method='neural')` defaults
to Regime B with an aggregate `Lambda`, which is exact only when group assignment is independent
of X. C1's treatment is selected on X, so `E[WW'|X]` varies with the group propensity. By the
same mechanism documented for the FLM observational case, the default path may under-cover. So
C1 is not a guaranteed pass; it is the cleanest existing test of whether the package's DiD
survives selection on X. Running it with the covariate-varying `Lambda` path (Regime C) is the
intended fix and the interesting comparison. Either outcome is a real, publishable-internally
result.

---

## 4. Ranked options (pick and choose)

Each is independent. Effort: S = a day-ish (DGP + run existing code), M = a new target/leg in
the package, L = new machinery.

**Option A. [S] Replicate Sant'Anna-Zhao DR-DiD Table 1 with the package's neural DiD.**
What it proves: whether the package's heterogeneous 2x2 DiD recovers ATT=0 with ~95% coverage
under selection on X, head to head with the canonical DR-DiD benchmark (their `tau_dr,p`:
coverage 0.947 on DGP1). Concrete target: C1 Table 1, panel, DGP1 and DGP3 (the two with a
correct outcome model the network can learn), metrics bias / RMSE / coverage / CIL, plus a TWFE
baseline row to reproduce the ~0 coverage. Front-load this: it is the cheapest, uses
`did(method='neural')` unchanged, and stress-tests the open Lambda-collapse question on a real
benchmark. Highest value per unit effort.

**Option B. [M] RieszNet-for-DiD via the closed-form representer.**
What it proves: that the package's automatic-debiasing (`riesz/`) module extends from the ATE to
the DiD ATT, the literal "deep learning + diff-in-diff + Riesz" target. Concrete: add a DiD ATT
functional whose Riesz representer is C2's `alpha(W)=D/p-(1-D)m(X)/(p(1-m(X)))`, learn `m(X)` (or
the adversarial Riesz loss), and replicate point-estimation + coverage on the C1/C2 DGP (true
ATT=5 in C2, or ATT=0 in C1). The representer is closed-form and well-conditioned (no Hessian
inversion, unlike the logit Lambda path), so this is the cleanest new-leg option. Pairs naturally
with Option A (same DGP, different estimator), giving a package-internal DR-DiD vs RieszNet-DiD
panel.

**Option C. [S/M] Heterogeneous CATT recovery on the meta-learner DGP.**
What it proves: the package's neural DiD recovers the heterogeneous effect function `tau(X)`,
not just its mean. Concrete: run `did(method='neural')` on C3's DGP, report CATT-MSE against
their Neural-Net-DR 0.10 / XGBoost-DR 0.04. Caveat: the comparison metric is MSE of the CATT
function, not coverage, so it exercises a different strength than the package's inference focus.

**Option D. [L] Staggered multi-period DiD (Callaway-Sant'Anna).**
What it proves: coverage of the modern staggered-DiD literature (group-time `ATT(g,t)`,
event-study aggregation). Big lift: a new multi-period model, the aggregation layer, and the
external supplementary DGP. High payoff (this is where applied DiD has moved), but it is a
research item, not a replication you bolt on.

**Option E. [L] Continuous-dose DiD (Zhang 2025).**
What it proves: DiD for a continuous treatment dose. Needs net-new machinery (kernel smoothing
over the dose, a conditional-density nuisance, the `ATT(d)` curve). Research item; no public code.

**Not recommended as a target.** C6 (dynamic nested Riesz) has no simulation and is not DiD;
keep it as background on where the Riesz machinery generalizes.

### Suggested pairing
Options A and B on the shared Sant'Anna-Zhao DGP give a single package-vs-paper table with both
the influence-function DR-DiD and the RieszNet-DiD next to the paper's `tau_dr,p` row, which is
the most direct "does our deep-learning DiD match the literature" deliverable for the least new
code. C optionally adds the heterogeneity (CATT) angle. D and E are larger research bets to be
scoped separately if the staggered or continuous-dose settings matter.

---

## Verifier note (2026-06-29)

**Verdict: PASS** (one flagged numeric misattribution, qualitatively harmless).

Fresh adversarial verification against the goal's four standards. All five checks below.

- **Check 1 (papers on disk):** PASS. All six candidate PDFs and their `tomd` markdown dirs
  exist under `references/did_scoping/` (C1 1812.01723, C2 2510.09064, C3 2502.04699,
  C4 2408.10509, C5 1803.09015, C6 2203.13887). No abstract-only citations.
- **Check 2 (quoted spans real):** PASS with one flag.
  - C1 span at `arXiv 1812.01723/01-front-matter-1.md:1135` ("We consider sample size n equal
    to 1000. For each design, we conduct 10,000 Monte Carlo simulations...") grepped out verbatim.
    Table 1 cells verified: DGP1 `tau_dr,p` bias -0.001 / RMSE 0.106 / cover 0.947 / CIL 0.412
    (line 1199, matches), DGP2 cover 0.947 (matches), DGP3 cover 0.942 (line 1211, matches),
    TWFE ~0 coverage and the ":1226" "almost zero coverage" span (matches).
  - **FLAG (minor):** the doc's DGP4 `tau_dr,p` coverage 0.274 is the WRONG ROW. The `tau_dr,p`
    row at line 1211 has DGP4 coverage **0.308**; the value **0.274** is the `tau_dr,p_imp`
    (improved, eq. 3.7) row at line 1213. The doc explicitly states it is citing the `tau_dr,p`
    row, so this one cell is misattributed. Qualitatively harmless (both 0.308 and 0.274 show DR
    breaking under DGP4, and DGP4 is not a recommended target -- Option A targets DGP1/DGP3).
    Recommend changing 0.274 to 0.308 (or relabel that cell as the `_imp` row).
  - C2 representer at `arXiv 2510.09064/01-front-matter.md:644` confirmed on disk:
    `alpha(W) = D/p - (1-D)/p * m(X,A)/(1-m(X,A))`, with `m(X,A)=E[D|X,A]`, `p=P(D=1)`.
  - C3 DGP at `arXiv 2502.04699/02-part-2.md:330-332` confirmed: `theta_0 = (1/2) W1 1(W2>0)`,
    conditional-parallel-trends DGP, `D ~ Binomial(p)` with p nonlinear in W and U.
  - C6 span at `arXiv 2203.13887/01-front-matter.md:28` confirmed verbatim ("recursive Riesz
    representer characterization of nested mean regressions").
- **Check 3 (package seams real):** PASS. `did()` at `__init__.py:1030` dispatches `method` in
  {auto, exact, neural, panel_fe} (lines 55-81). `_did_closed.py` exists. `DiDModel`
  (`models/did.py:33`) has the saturated form `Y = alpha + gamma*G + lambda*P + tau*(G*P)`,
  theta=[alpha,gamma,lambda,tau], target `AverageParameter(param_index=3)` = E[tau(X)].
  `FEPanelDiDModel` at `models/panel_fe.py:31`. `riesz/inference.py` exists (Option B valid).
  `evals/dgp_did_nn.py` lines 4-5 state verbatim "group G and period P are assigned independently
  of the covariates X (only the OUTCOME means depend on X)", line 18 confirms constant E[WW'|X].
  All cited eval/test files exist. No phantom seam.
- **Check 4 (honesty of verdicts):** PASS. `DiDModel.hessian_depends_on_theta = False`
  (`models/did.py:42`, "KEY: enables Regime B") confirms the central caveat that the neural DiD
  defaults to aggregate-Lambda Regime B, exact only under treatment independent of X, so the
  selection-on-X DGP may under-cover. C5 "MC in external supplementary not on disk" is true
  (`arXiv 1803.09015/...:206-208,226` point to an external supp URL; no supp file in the dir).
  C6 "no simulation section, not a DiD paper" is true (zero hits for monte carlo / simulation
  across all C6 md files). No drop-in overclaim; the C1 risk is disclosed in-doc.
- **Check 5 (nothing deleted):** PASS. `git diff docs/rebuild-rtd -- docs/replications/index.md`
  is 9 insertions, 0 deletions (toctree + scoping pointer only). No Part-1/Part-2 content removed.

Net: standards (a) full-text-on-disk + greppable spans, (b) real seams, (c) nothing deleted,
(d) inventory matches code all hold. The single defect flagged (DGP4 coverage cell pulled from
the `_imp` row, 0.274) has been CORRECTED to the `tau_dr,p` value 0.308.
