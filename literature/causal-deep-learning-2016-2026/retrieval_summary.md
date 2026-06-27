# Retrieval Summary

- Manifest records: 86
- PDFs on disk: 79
- PDF bytes on disk: 131921294
- Manifest status `downloaded`: 79
- Manifest status `missing`: 7
- Docling Markdown files: 79
- Docling status `ok`: 79
- arXiv source rows attempted: 78
- arXiv source status `extracted`: 75
- arXiv source status `pdf-returned`: 3
- TeX-like files retained: 476
- Text-like source files retained: 830

## Runs

- `websource_run_20260627.txt`: main `websource papers.txt` run. It downloaded the arXiv/OpenReview records and attempted DOI-backed records; the run was manually interrupted after repeated OpenAlex/Semantic Scholar 429 backoff on DOI failures.
- `websource_missing_retry_20260627.txt`: title-only retry for unresolved records with OpenAlex/Semantic Scholar excluded from search sources. It found the public arXiv copy of Counterfactual Propagation, then was interrupted when DOI expansion again entered OpenAlex backoff.

Sci-Hub was not used. Both runs passed `--mirrors-file ./no-scihub-mirrors.txt`.

## Searchable Text Runs

- `docling_extract_20260627.txt`: serial `extract --backend docling` run over the 79 local PDFs. The Docling report in `markdown_docling/extract-report.json` records 79 ok results.
- `fetch_arxiv_sources_20260627.txt`: public arXiv `e-print` retrieval for the 78 arXiv-backed records. Text-like files were extracted to `tex_sources/extracted/`; raw archives and binary figures were discarded.
- `searchable_catalog.csv` / `searchable_catalog.json`: joins each manifest record to its Docling Markdown path and arXiv source directory when available.

## Missing Public PDFs

- 2018: Learning Causal Structures for Individualized Treatment Effects. Note: websource may find a public author copy; quick DOI/arXiv checks did not verify one.
- 2022: Deep Learning for Counterfactual Inference and Treatment Effect Estimation. DOI `10.1201/9781003028543-7`.
- 2022: Review of Deep Learning Methods for Individual Treatment Effect Estimation with Automatic Hyperparameter Optimization. DOI `10.36227/techrxiv.20448768`.
- 2025: Individualized treatment rules based on adaptive transfer-dragonnet. DOI `10.1007/s11222-025-10704-9`.
- 2025: Treatment Effect Estimation in Survival Analysis Using Copula-Based Deep Learning Models for Causal Inference. DOI `10.3390/axioms14060458`.
- 2025: Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence. DOI `10.1287/mnsc.2024.04625`. Note: Published DOI is 10.1287/mnsc.2024.04625; SSRN DOI is used as the first public retrieval route.
- 2026: Individualized treatment effect estimation with compromised adversarial nets. DOI `10.1007/s00180-025-01705-3`. Note: Crossref returned publication year 2026; no precise date was available in the quick query.
