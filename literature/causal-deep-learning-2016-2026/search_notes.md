# Search Notes

## Sources Used

- arXiv API: `https://export.arxiv.org/api/query` for exact ID validation and broad title/query discovery.
- Crossref API: `https://api.crossref.org/works` for DOI-backed non-arXiv and publisher records.
- OpenReview API/PDF endpoint for GANITE.
- Existing local `references/*_docling.md` titles in this repo for deep-inference-specific coverage.
- `websource` batch mode for public PDF retrieval.

OpenAlex and Semantic Scholar were attempted during discovery but returned HTTP 429 in this session, so they were not treated as authoritative for this pack.

## Discovery Queries

- arXiv: `individual treatment effect` + `deep` / `neural`.
- arXiv: `heterogeneous treatment effects` + `neural`.
- arXiv: `counterfactual regression` + `treatment`.
- arXiv exact-title checks for CFR, TARNet, CEVAE, Dragonnet, DeepIV, DeepGMM, DFIV, VCNet, RieszNet, C-Learner, Deep LTMLE, GDR, TDA, and related local-reference titles.
- Crossref title queries for GANITE, SITE, adaptive transfer-Dragonnet, deep counterfactual inference chapters/reviews, and survival/copula variants.

## Inclusion Rules

- Keep method papers and surveys/reviews when the title/abstract route is explicitly causal treatment-effect, potential-outcome, counterfactual-outcome, uplift, IV, longitudinal deconfounding, or semiparametric causal inference with neural/deep components.
- Keep repo-local deep-inference papers even when they are framed as semiparametric inference rather than ITE/CATE, because this repository implements that branch.
- Exclude purely classical causal ML, purely predictive healthcare, generic graph causality without treatment-effect estimands, and application-only Dragonnet uses unless they introduce a method.

## Known Unresolved / Manual Checks

- `Learning Causal Structures for Individualized Treatment Effects` (SITE) is indexed because it is a named deep ITE method, but this pass did not verify a public PDF or DOI route. It remains in `papers.txt` as a title query for `websource`.
- Publisher DOI entries may require manual/public author-copy lookup if `websource` cannot find an open PDF route.

## Manifest Totals

- Total records: 86
- arXiv-backed records: 78
- DOI-backed records: 9
