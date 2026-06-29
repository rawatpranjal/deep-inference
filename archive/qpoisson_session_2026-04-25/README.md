# Quasi-Poisson FML Validation Session — 2026-04-24/25

Self-contained research artifacts from a single session validating the
Farrell-Liang-Misra framework for Poisson event-study coefficients. ~3,500
Monte Carlo replicates across 8 cranks. All scripts are standalone and
re-runnable.

## Headline findings (full detail in `FINDINGS.md`)

| Setting | Empirical 95% coverage | vs saturated GLM |
|---|---|---|
| d=4 cluster subgroups (K∈{5,10,20}) | **0.95** (PASS) | dominates 32/35 cells |
| d=8, 40 cross cells | **0.95** (PASS) | sat avg = 0.55, FML 40/40 wins |
| d=128 no-PCA, 40 cross cells, Tanh ensemble-5 | **~0.80** (true coverage at M=60) | sat avg = 0.09, FML 40/40 wins |

**Key results:**
- IF formula is correct (Oracle test at M=200 gives coverage 0.940).
- ψ-winsorize is harmful; Λ-trim ε=0.10 is the right stabilization.
- Architecture barely matters within MC noise (32→256 hidden).
- M=15 MC reps has SD ≈ 0.08 — cannot distinguish configs differing < 0.15.
- d=128 no-PCA stuck at ~0.80 because of curse-of-dim DNN nuisance rate.

## Path to 0.95 at d=128 (untested in this session)

1. **Supervised bottleneck** — train R^128 → R^8 encoder jointly with Poisson loss.
2. **Iterative TMLE targeting** — proper fluctuation submodel, not 1-step shift.
3. **Higher-order IF** — Argañaraz-Escanciano 2024 / Robins 2008.

## File map

- `FINDINGS.md` — running log of all cranks with per-config coverage tables
- `sim_if.py` — initial QPoisson IF analytical verification
- `sim_event_study*.py` — saturated cell event study, IF vs bootstrap vs jackknife
- `sim_flm_dml.py` — full FML pipeline with DNN first stage
- `flm_scaling.py` — N-scaling sweep (5k → 200k)
- `flm_diagnose.py` — Oracle FML test, DNN rate analysis
- `flm_rescue.py` — d-reduction tests, Tanh-matched, 3-way split
- `autoresearch.py` — family × dim × N grid via deep_inference
- `flm_robustness.py` — 10 IF-fix robustness crank
- `flm_crank{2,3,4,5,6,7,8}*.py` — DGP stress, architecture, ε-sweep, clustering, cross-cells, full d=128
- `flm_autoresearch_v{2,3,4}.py` — ensemble + structured + stability
- `flm_*_results.json`, `*.log` — raw outputs

Source: Farrell, Liang, Misra (2021/2025), "Deep Learning for Individual
Heterogeneity," arXiv:2010.14694. Influence-function eq. B.4 for GLM case.
