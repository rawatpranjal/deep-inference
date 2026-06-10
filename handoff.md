# Handoff — 2026-06-09 — main

## Where we left off
Mid coverage-fix program. A fast *directional* SIM2 sweep (R0 baseline vs R1a deeper
training; folds=5/M=12/threads=8) is RUNNING in background (job `b0cfmmt45` →
`evals/reports/sim_02_R*fast_*.txt`). Even this trimmed config has run >2h without
finishing — CNN Monte Carlo on CPU is the binding constraint.

## Active streams (clustered)
- **Pristine close-out — DONE, pushed.** Batches 0-5 (editable-install fix; multinomial
  `inference()` estimand bug E[α₂]→E[β₁]; repo hygiene — untracked 53 copyrighted PDFs +
  211 data files, sealed leak paths; API coherence + decision tree; doc sync), Batch 3
  (variance → FLM **pooled** default + `within_fold` toggle; removed dead inversion `ridge`
  knob), Part A (repeated cross-fitting median DML + `n_repeats`/`three_way_theta_frac`/
  `dropout`/`weight_decay` knobs; CLI-ified sim_02 / run_all / eval_14). 39 fast tests green.
- **Coverage-fix program — IN FLIGHT (SEQUENTIAL, compute-bound).** Target: SIM2 CNN 85%
  under-coverage. Hypothesis from the eval_06 firewall: the z_mean<0 bias is driven by
  nuisance UNDER-TRAINING → training depth (epochs/patience) is the prime lever, `n_repeats`
  the secondary SE lever. NEXT: read R0fast vs R1afast summary; if R1a's z_mean→0 and
  coverage↑, confirm at full M=20/folds=20/n=10k, then add `n_repeats` if SE-ratio<0.8.
- **eval_14 DiD + SIM3 — PENDING (task 5).** eval_14: re-run M=50/n=8000 (only z_mean
  fails; coverage 96% already ok). SIM3: confirm it's over-conservative (100%, not a bug);
  relabel harness so >99% ≠ "FAIL".
- **Finalize — PENDING (task 6).** Bake winning configs as defaults; regenerate run_all;
  update known_limitations/CHANGELOG; commit.

## Decisions made this session
- Variance: FLM prescribes POOLED (sample var of ψ at global mean) → now default;
  within_fold kept as toggle. structural_dml SE shifted within-fold→pooled (widens, safe).
- API: keep 3 entry points + document decision tree (Option B); NOT deprecating.
- Copyrighted PDFs: removed from HEAD only (`git rm --cached`), NOT history-rewritten (user
  chose option a; no force-push) — they remain in git history.
- DR-DiD (Sant'Anna-Zhao 2020 + Callaway-Sant'Anna 2021) logged to `docs/dev/backlog.md`.

## Open questions
- Compute: SIM2 CNN MC is brutally slow on CPU. May need GPU, far smaller configs, or to
  accept "n-sweep convergence as proof" rather than chase 95% at n=10k.
- Clean eval_06 at VALIDATED settings (epochs=200/patience=50) to confirm no-regression, or
  trust byte-identical unit tests + theory? (Deferred; SIM2 R0 is the de-facto check.)

## Landmines / gotchas
- **Light eval ≠ valid firewall.** eval_06 at reduced epochs (150 vs validated 200)
  reintroduces the z_mean<0 regularization bias and tanks coverage to 80% — independent of
  any code change. Firewalls MUST use validated epochs/patience.
- **eval_06 default drifted** to n=8000/lgbm (`evals/eval_06_coverage.py:174,177`) → ~135
  min/run. CLAUDE.md still says n=5000/ridge. Use explicit light args for quick checks.
- **n_repeats>1**: returned `psi_values`/diagnostics come from the FIRST repeat only; only
  scalar mu_hat/se/CI are median-aggregated. `dropout` is a no-op when a `network_factory`
  is supplied (SIM2 CNN) → use `weight_decay`/epochs/patience for SIM2 undersmoothing.
- Package is editable (`pip install -e .` on python3.11). Always run with `python3.11`.

## Suggested next move
When `b0cfmmt45` lands: compare R0fast vs R1afast z_mean/coverage. If deeper training pulls
z_mean toward 0, training depth is the fix — confirm at full settings, then `n_repeats` for SE.
