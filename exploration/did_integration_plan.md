# Deep-learning DiD integration plan (FLM and RieszNet)

Dev research note (working area, not the published RTD site). Scope-only: how to bring
deep-learning difference-in-differences estimators into this package along BOTH integration
spines, the FLM influence-function path and the RieszNet automatic-debiasing path, and which
paper simulations to replicate once each is built. It implements nothing. The build sequence at
the end is the order to actually code, later.

The emphasis is deep-learning DiD: the neural network is the nuisance learner (FLM) or the Riesz
representer (RieszNet). Classical DiD (logit / OLS / Lasso nuisances) is the no-neural subset,
kept only as the baseline each paper reports and a sanity anchor.

Companion: the candidate menu in `docs/replications/did_candidates.md`. Every paper cited here
has on-disk full text under `references/did_scoping/` (the six original candidates plus Chang and
two new arXiv finds) or in the local collections `Code/mldid/papers/` and
`Code/applied-science-new/causal-machine-learning-panels/papers/` (49-paper curated wing with
per-paper summaries and a `_curriculum.md` taxonomy).

---

## 0. The one object both spines target

Every DiD-ATT estimator in this literature is debiasing the SAME linear functional. The DiD ATT
is `theta_0 = E[ (D/p)·(g(1,X) - g(0,X)) ]` on the time-difference `DeltaY` (with
`g(d,X)=E[DeltaY | D=d, X]`, `p=P(D=1)`), and its Riesz representer is the IPW-DiD weight

    alpha(W) = D/p - (1-D)·m(X)/(p·(1-m(X))),   m(X) = E[D | X].

Three papers we read are three faces of this one object:

- Sant'Anna-Zhao (2020) write it as the doubly-robust DiD moment (`tau_dr`): plug-in plus the
  `alpha·(DeltaY - g)` correction.
- Chang (2020) writes the same correction as a Neyman-orthogonal score. Abadie's (2005) plug-in
  score `phi = (Y(1)-Y(0))/P(D=1)·(D-g0(X))/(1-g0(X)) - theta0` has a non-zero Gateaux derivative
  in the propensity `g0`, so he adds an orthogonalizing term to kill the first-order bias
  (`references/did_scoping/chang_2020_dml_did.md:107`).
- Bach et al. (2025) write it as the Riesz representer verbatim, `alpha(W) = D/p -
  (1-D)·m(X,A)/(p·(1-m(X,A)))` (`references/did_scoping/arXiv 2510.09064/01-front-matter.md:644`).

The FLM spine forms the correction as `H_theta · Lambda^{-1} · l_theta` (the influence function);
the RieszNet spine forms it as `alpha·(DeltaY - g)` with `alpha` learned directly. Same target,
two debiasing devices. That is why one DGP can exercise both legs side by side.

---

## 1. The two spines, concretely

### FLM spine (influence function / Lambda)
Three things plugged into the existing engine:
- a `StructuralModel` (`models/base.py`): the `theta(X)` network and its loss. For 2x2 DiD this
  is `DiDModel` (`models/did.py`), `Y = alpha(X)+gamma(X)G+lambda(X)P+tau(X)·GP`, `theta_dim=4`.
- a `Target` (`targets/base.py`): the functional. For ATT it is `AverageParameter(param_index=3)`,
  which is `E[tau(X)]`.
- a Lambda regime (`lambda_/`): `analytic.py` (Regime B, `Lambda=E[WW'|X]` when treatment is
  independent of X) or `estimate.py` (Regime C, the cholesky/ridge net that estimates `Lambda(x)`).
  `engine/crossfit.py` orchestrates, `engine/assembler.py` builds `psi`, `engine/variance.py` gives
  the SE. Entry point `inference()` / `did()`.
- The crux for DiD is the Lambda regime. Under selection on X (the propensity `e(X)=E[G|X]`
  varies, which is the whole DR-DiD setting), `E[WW'|X]` varies with X, so the default Regime-B
  aggregate Lambda is wrong and Regime C is required. This is the documented Lambda-collapse risk
  (`docs/notes/flm_lambda_se_undercount.md`); it is also the single most important thing the DiD
  replications will measure.
- Deep-learning content: the `theta(X)` net is the neural nuisance. Classical is the same with a
  parametric or Lasso fit.

### RieszNet spine (automatic debiasing / Riesz representer)
- a linear functional `m(W;g)` and its Riesz representer `alpha(W)`. `riesz/model.py`
  (`RieszNet`, a multitask net with a shared trunk, a regression head `g` and a linear Riesz head
  `alpha`) and `riesz/inference.py` (`riesz_inference`, the combined regression + Riesz +
  TMLE-targeting loss `riesz_loss_term`, then the DR moment and `engine/variance.py` SE).
- For DiD: the package's `riesz_inference` already estimates the ATE contrast `g(1,X)-g(0,X)` on
  `Y`. The DiD ATT is the same contrast on `DeltaY` with the representer above, so the change is
  (i) feed `DeltaY` as the outcome, (ii) target the ATT-weighted moment `E[(D/p)(g(1,X)-g(0,X))]`
  whose representer is the IPW-DiD weight. The representer is closed-form and well-conditioned (no
  matrix inversion, unlike the logit Lambda path), so this is the cleaner of the two new legs.
- Deep-learning content: the shared trunk and both heads are neural by construction; this is
  RieszNet exactly, retargeted from ATE to ATT.

---

## 2. Per-candidate integration (the six plus Chang)

### Chang (2020), Double/Debiased ML for DiD  [NEW, on disk]
`references/did_scoping/chang_2020_dml_did.md` (Econometrics Journal 23:177-191). The DML-DiD
bridge: Abadie (2005) ATT, many controls (`p` can exceed `N`), three Neyman-orthogonal scores
(repeated outcomes, repeated cross-sections, multilevel treatment). Verbatim (`:107`): the plug-in
score is not orthogonal in `g0`; the new scores add a zero-mean adjustment to orthogonalize.
- FLM path: this score is the FLM influence function for the ATT. Build a `StructuralModel` whose
  nuisance is the propensity `g0(X)=P(D|X)` (or use the `DiDModel` saturated form whose `Lambda`
  encodes `g0`); target `E[tau]`. Regime C, because `g0(X)` varies. Seams: `models/`,
  `lambda_/estimate.py`, `engine/`.
- RieszNet path: direct. The representer is the same IPW-DiD weight `(D-g0(X))/(1-g0(X))/p`. Learn
  `g0` (or the representer) with a net via `riesz/`.
- Replication target: Section 4 / Figure 1, true `theta0=3`, `N=200`, `p=300` controls,
  Logit-Lasso plus random-forest nuisances. Show the orthogonal estimator centered and normal
  while Abadie's plug-in is biased. Deep-learning variant: swap Lasso/RF for the neural nuisance.
  The paper reports it as a histogram, so a bias / coverage table is the package's value-add.
- Effort: S/M (reuses the spine, a new high-dim nuisance DGP).

### C1. Sant'Anna-Zhao (2020), Doubly Robust DiD
`references/did_scoping/arXiv 1812.01723.pdf`. ATT under conditional parallel trends; treatment
selected on covariates `D=1{p(Z)>=U}`; n=1000, 10,000 reps, true ATT=0.
- FLM path: `did(method='neural')` as-is, but Regime C (selection on X). The flagship test of
  whether the package's DiD survives selection on X.
- RieszNet path: the DR moment is `alpha·(DeltaY - g)`; the RieszNet-DiD leg above replicates it.
- Replication target: Table 1 panel, DR row `tau_dr,p`. DGP1 coverage 0.947 / RMSE 0.106, DGP3
  0.942, DGP4 0.308 (DR breaks), TWFE about 0 throughout. Package run on DGP1 and DGP3 (the
  outcome-correct cells a net can learn).
- Effort: S (FLM leg) plus M (RieszNet leg).

### C2. Bach et al. (2025), Sensitivity Analysis for DiD using Riesz Representation
`references/did_scoping/arXiv 2510.09064.pdf`. The source of the closed-form DiD representer
(section 0). Also extends it to staggered `ATT(g,t)` (`:931`) and to a sensitivity bound.
- FLM path: not the natural home (this is a Riesz paper); the representer feeds the RieszNet leg.
- RieszNet path: this is the enabling reference. It hands the package the exact representer to put
  in `riesz/inference.py`. Implement it, replicate their point-estimation table (true ATT=5, n in
  {500..50000}); the sensitivity bound is an optional later layer.
- Replication target: their Table 1 point estimates / coverage on the S-Z-extended DGP.
- Effort: M (representer leg) plus L (sensitivity layer, optional).

### C3. Lan, Chang, Dillon & Syrgkanis (2025), Meta-learner for Heterogeneous DiD
`references/did_scoping/arXiv 2502.04699.pdf`. CATT function `theta_0(X)`, neural-net DR
meta-learner; parallel trends conditional on full W; metric is CATT-MSE (no coverage).
- FLM path: `did(method='neural')` already outputs `tau(X) = theta_hat[:,3]`, the CATT. Score
  CATT-MSE on a held-out grid. The package adds the inference (CI for `E[tau(X)]`) the paper omits.
- RieszNet path: their orthogonal pseudo-outcome is the conditional version of the same DR moment;
  a conditional Riesz representer (their Section 4) is the heterogeneous analogue, a research
  extension of the RieszNet-DiD leg.
- Replication target: their Table 1 CATT-MSE, Neural-Net-DR 0.10, XGBoost-DR 0.04. The deep-learning
  (Neural-Net) learner is the headline.
- Effort: S (FLM CATT-MSE) plus L (conditional Riesz).

### C5. Callaway & Sant'Anna (2021), DiD with Multiple Time Periods
`references/did_scoping/arXiv 1803.09015.pdf`. Group-time `ATT(g,t)`, staggered timing, DR
estimand, event-study aggregation. Requires multi-period machinery the package does not have.
- FLM path: each `ATT(g,t)` is an `E[tau_{g,t}(X)]` the FLM engine can target, but the multi-cohort
  indexing, the comparison-group choice (never-treated vs not-yet-treated), and the event-study
  aggregation are net-new orchestration around the engine.
- RieszNet path: Bach (C2) gives the staggered representer `alpha(W)` for `ATT(g,t)` (`:931`), so a
  RieszNet-per-(g,t) is defined; the aggregation layer is still new.
- Replication target: the Monte Carlo is in an external Supplementary Appendix not on disk
  (`arXiv 1803.09015/01-front-matter-1.md:206`); fetch it before committing to numbers.
- Effort: L (multi-period model plus aggregation).

### C4. Zhang (2025), Continuous DiD with Double/Debiased ML
`references/did_scoping/arXiv 2408.10509.pdf`. Dose-curve `ATT(d)`, continuous treatment, kernel
smoothing over `d`, conditional-density nuisance.
- FLM path: needs a continuous-dose `Target` with a kernel bandwidth and a conditional density
  `f_h(d|X)` nuisance; the existing `dose_response.py` target is an average potential outcome, not
  a conditional-on-treated-at-d ATT, so it does not drop in.
- RieszNet path: a continuous-treatment Riesz representer exists in the broader Auto-DML literature
  but is not stated for this estimand here; mark "no off-the-shelf representer yet".
- Replication target: their Table 1, panel coverage 0.918 (n=2000), 0.930 (n=10000), at d=0.9.
- Effort: L (new kernel/density machinery).

### C6. Chernozhukov, Newey, Singh & Syrgkanis (2023), Dynamic / Nested Auto-DML
`references/did_scoping/arXiv 2203.13887.pdf`. Recursive Riesz representer for sequential
treatments. Not a DiD paper, no simulation. Frontier only: it shows the Riesz machinery
generalizes to nested functionals, of which a panel DiD is a small special case. No replication
target. Both paths: "not yet" (would need sequential/recursive Riesz infrastructure).

---

## 3. The shared anchor (first build, exercises both spines)

The cheapest high-value deliverable is one DGP run by both new legs against one paper row:

- DGP: Sant'Anna-Zhao DGP1 (selection on X, true ATT=0, n=1000).
- Leg 1 (FLM): `did(method='neural')` forced to Regime C (estimate `Lambda(x)`).
- Leg 2 (RieszNet): the new DiD-ATT representer in `riesz/inference.py`.
- Baselines: TWFE (must reproduce about 0 coverage) and the closed-form `did(method='exact')`.
- Paper column: `tau_dr,p` coverage 0.947 / RMSE 0.106.

One package-vs-paper table with FLM-DiD, RieszNet-DiD, TWFE, and the paper's DR row settles three
questions at once: does the package's deep-learning DiD cover under selection on X, does the
Regime-C Lambda path fix the collapse, and does RieszNet-DiD match the influence-function leg.

---

## 4. The deep-learning DiD landscape

Two paradigms, only one of which is the package's spine.

### Paradigm 1: orthogonal / DR-DML DiD (the FLM and RieszNet wheelhouse)
Every method here has a Neyman-orthogonal score (the FLM influence function) and a Riesz
representer; neural nuisances make it deep-learning. Beyond the seven above, the local collection
adds: `zimmert_2020_efficient_did` (efficient DML-DiD), `semenova_2023_hte_dynamic_panels`
(orthogonal high-dim dynamic-panel HTE), `zhao_cui_2025_semiparametric_iv_did` (IV-DiD when
conditional parallel trends fails), `clarke_polselli_2025_dml_panel_fe` (DML with unit/time FE),
`akbari_2025_triple_difference` (DML triple-difference), `hatamyar_2023_mldid_staggered` (ML-DiD
staggered). All live in `Code/mldid/papers/` with summaries; the
`causal-machine-learning-panels/papers/_curriculum.md` taxonomy (sections 01-02) organizes them.
These are the natural follow-ons once the anchor leg exists.

### Paradigm 2: representation / latent-variable counterfactuals (NOT the FLM/RieszNet spine)
Deep, but a different paradigm: they learn balanced or latent representations of counterfactual
trajectories, with no influence-function or Riesz inference. `bica_2020_counterfactual_balanced`
(CRN, domain-adversarial RNN), `melnychuk_2022_causal_transformer` (Causal Transformer),
`poulos_zeng_2021_rnn_counterfactual` (propensity-weighted RNN), `elbouchattaoui_2024_cdvae`
(latent-confounder CDVAE). These do not slot into the package's two spines; they are a separate
estimator family. Honest verdict: out of scope for an FLM/RieszNet integration, listed so the
boundary is explicit.

### New arXiv finds (subagent triage, 2026-06-29)
The deep-learning-specific DiD field is thin; the subagent found two genuinely-new neural DiD
papers (both now downloaded and read):
- `arXiv 2509.24259` (Sun & Xiao 2025), Network DR-DiD: GNN nuisances inside a doubly-robust DiD
  for treatment-spillover settings; sim in Appendix E (`:54`, `tau_DATT(0)=0.2`, `tau_DATT(1)=0.4`,
  Table 3). Fits Paradigm 1 (DR moment plus neural nuisance) but in a network-interference setting
  that is a research extension, not a clean first target.
- `arXiv 2505.20536` (CoDEAL, 2025): FFNN plus autoencoder panel imputation; no valid CI, so
  Paradigm-2-adjacent.
Lower-priority DML-compatible (not neural-specific) finds recorded for completeness:
`arXiv 2406.16234` (EIF DiD for longitudinal time-varying treatments), `2503.11375` (DiD-meets-SC),
`2512.00296` (continuous stochastic-policy DiD), `2603.04080` (staggered DR with time-varying X).
Bottom line from the search: the package's existing FLM/RieszNet machinery already targets the
mainstream; the novelty is applying neural nuisances and a neural representer to it, which the
anchor build does.

---

## 5. Build sequence (the order to code, later)

1. The anchor (S+M). RieszNet-DiD ATT representer in `riesz/` plus FLM-DiD Regime-C Lambda, both on
   the Sant'Anna-Zhao DGP1, against `tau_dr,p` (Section 3). Front-load: it stands up both spines
   and answers the selection-on-X / Lambda-collapse question in one table.
2. Chang DML-DiD (S/M). Same spines, high-dimensional `p>N` nuisance DGP (theta0=3, N=200, p=300),
   neural nuisance as the deep-learning variant. Reproduce the bias-killing histogram as a
   bias/coverage table.
3. Lan CATT heterogeneity (S, plus L for conditional Riesz). `tau(X)` recovery plus CATT-MSE vs the
   meta-learner's Neural-Net-DR 0.10.
4. Staggered DiD, Callaway-Sant'Anna (L). Multi-period `ATT(g,t)` plus event-study aggregation;
   pull the external supplementary DGP first.
5. Continuous-dose DiD, Zhang (L). Kernel / conditional-density target.

Frontier, not scheduled: network GNN DR-DiD (`2509.24259`), the representation/latent paradigm
(CRN, Causal Transformer, CDVAE), and the dynamic nested Riesz (`2203.13887`).

The first two items are the deep-learning DiD core the user asked for, each exercising both the FLM
and the RieszNet path on a published benchmark.
