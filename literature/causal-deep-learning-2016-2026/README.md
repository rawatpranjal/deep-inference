# Causal Deep Learning Papers, 2016-2026

Generated on 2026-06-27 for a high-recall treatment-effect and neural semiparametric causal-inference pass.

## Contents

- `manifest.csv` / `manifest.json`: 86 curated records with DOI/arXiv/source fields.
- `papers.txt`: batch input for `websource`.
- `downloads/`: local PDFs downloaded by `websource` (ignored by the repo-wide `downloads/` gitignore rule).
- `download_status.csv`: retrieval status after the `websource` run.
- `search_notes.md`: source queries, inclusion rules, and unresolved checks.

## Scope

Included papers use deep neural, representation, adversarial, transformer, graph, variational, KAN, or neural semiparametric machinery for treatment effects, counterfactual outcomes, uplift, IV, longitudinal causal inference, or closely related potential-outcome/semiparametric targets.

The strict current-date ten-year window starts on 2016-06-27. Two May/June 2016 foundational counterfactual-representation entries are retained and tagged `foundational-boundary` because the practical literature window is 2016-2026.

## Counts By Year

- 2016: 3
- 2017: 2
- 2018: 7
- 2019: 4
- 2020: 13
- 2021: 7
- 2022: 15
- 2023: 8
- 2024: 13
- 2025: 10
- 2026: 4

## Most Common Tags

- ite: 34
- local-reference: 17
- representation: 13
- semiparametric: 9
- longitudinal: 9
- iv: 8
- neural: 8
- neural-networks: 8
- counterfactual-regression: 7
- adversarial: 7
- continuous-treatment: 6
- deep-learning: 5
- survival: 5
- deepiv: 4
- hte: 4
- uplift: 4
- causal-inference: 4
- counterfactual: 3
- variational: 3
- deep-inference: 3

## Boundary Entries

- 2016-05-12: Learning Representations for Counterfactual Inference
- 2016-06-13: Estimating individual treatment effect: generalization bounds and algorithms

## Retrieval Status

- downloaded: 79
- missing: 7

## Rebuild

```bash
python3 build_manifest.py
websource papers.txt -o downloads -v --limit 8 --max-downloads 1 --timeout 30
python3 build_manifest.py --sync-downloads
```

The manifest is curated, while arXiv title/date/author metadata is refreshed from `https://export.arxiv.org/api/query` when available.
