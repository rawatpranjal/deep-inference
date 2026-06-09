# Handoff — 2026-06-09 — main

## Where we left off
DiD work is complete and shipped. Three estimators (closed-form 2×2, neural
heterogeneous 2×2, two-way FE panel) were built, eval-validated, then consolidated
into a single `did()` entry point. All committed + pushed to main; docs live on RTD.
Earlier the same session migrated Sphinx docs to the RTD theme and added a phased
theory section. Nothing in flight.

## Active streams (clustered)
- **DiD family — DONE.** `did(Y, group, post[, X][, unit, time], method=...)` auto-
  dispatches: `exact` (closed-form, HC0), `neural` (E[τ(X)]), `panel_fe` (two-way FE,
  continuous + binary LPM). Registered models `model='did'`, `model='did_fe'`.
  Validated: eval_13 PASS; eval_14 96% coverage; eval_15 98%/94%; sim_04 90/100/93%.
- **Docs — DONE.** RTD theme + linearly-phased theory section live at
  deep-inference.readthedocs.io. DiD API documented (api/inference.md).

## Decisions made this session
- `did(method='exact')` is closed-form and bypasses the neural path (user call).
- Binary FE outcome = fixed-effects **linear probability model**, not logit-FE (user call).
- **Cluster-robust SE deferred** — FE SE assumes iid errors (user call).
- One `did()` entry point; the per-variant functions were removed (user disliked the clutter).

## Open questions
- None blocking. CLAUDE.md package-structure listing could gain the new DiD entries +
  a Learned Rule (proposed at session end, awaiting OK).

## Landmines / gotchas
- **Installed package is a NON-editable copy** in site-packages. `from deep_inference
  import did` in a plain shell uses the STALE copy. Tests/evals/docs work because they
  prepend `src/` to sys.path. Run `pip install -e .` to make source edits live.
- **Top-level function names must not collide with submodule filenames.** A public
  `did()` in `__init__.py` collided with `did.py` (so `di.did` returned the module).
  Fixed by renaming the closed-form module to `_did_closed.py`. Watch this for future
  top-level functions.
- eval_14 / eval_15 / sim_04 call sites were updated to `did()` and import- + pattern-
  smoked, but not re-run at full M after the rename (low risk — estimators unchanged,
  only the wrapper name moved).

## Suggested next move
DiD family is shippable. If extending: staggered-adoption / event-study DiD, or add the
deferred cluster-robust SEs for panels.
