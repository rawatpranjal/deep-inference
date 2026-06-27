#!/usr/bin/env python3
"""Build the causal deep learning literature pack.

The curated list is intentionally explicit: the goal is a reproducible folder,
not a one-off search transcript.  arXiv metadata is refreshed when the network is
available; static fallbacks keep the artifact buildable offline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parent
START_DATE = date(2016, 1, 1)
STRICT_TEN_YEAR_START = date(2016, 6, 27)
CUTOFF_DATE = date(2026, 6, 27)


FIELDS = [
    "key",
    "title",
    "authors",
    "year",
    "date",
    "venue",
    "doi",
    "arxiv_id",
    "url",
    "download_target",
    "tags",
    "inclusion",
    "source",
    "local_reference",
    "retrieval_status",
    "pdf_path",
    "notes",
]


@dataclass
class Entry:
    key: str
    title: str
    year: int
    date: str
    tags: str
    inclusion: str
    source: str
    authors: str = ""
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    download_target: str = ""
    local_reference: str = ""
    retrieval_status: str = "pending"
    pdf_path: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, str | int]:
        return {field_name: getattr(self, field_name) for field_name in FIELDS}


def keyify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:72]


def arxiv_entry(
    arxiv_id: str,
    title: str,
    year: int,
    published: str,
    tags: str,
    inclusion: str,
    *,
    authors: str = "",
    doi: str = "",
    venue: str = "arXiv",
    local_reference: str = "",
    notes: str = "",
) -> Entry:
    base_id = arxiv_id.split("v", 1)[0]
    return Entry(
        key=keyify(title),
        title=title,
        authors=authors,
        year=year,
        date=published,
        venue=venue,
        doi=doi,
        arxiv_id=base_id,
        url=f"https://arxiv.org/abs/{base_id}",
        download_target=base_id,
        tags=tags,
        inclusion=inclusion,
        source="arXiv API / curated query set",
        local_reference=local_reference,
        notes=notes,
    )


def non_arxiv_entry(
    title: str,
    year: int,
    published: str,
    tags: str,
    inclusion: str,
    source: str,
    *,
    authors: str = "",
    venue: str = "",
    doi: str = "",
    url: str = "",
    download_target: str = "",
    local_reference: str = "",
    notes: str = "",
) -> Entry:
    return Entry(
        key=keyify(title),
        title=title,
        authors=authors,
        year=year,
        date=published,
        venue=venue,
        doi=doi,
        arxiv_id="",
        url=url or (f"https://doi.org/{doi}" if doi else ""),
        download_target=download_target or doi or title,
        tags=tags,
        inclusion=inclusion,
        source=source,
        local_reference=local_reference,
        notes=notes,
    )


ENTRIES: list[Entry] = [
    arxiv_entry(
        "1605.03661",
        "Learning Representations for Counterfactual Inference",
        2016,
        "2016-05-12",
        "ite; representation; counterfactual-regression; foundational-boundary",
        "Foundational counterfactual representation paper; included as a 2016 boundary item.",
        notes="Strict current-date 10-year cutoff starts 2016-06-27; retained for field coverage.",
    ),
    arxiv_entry(
        "1606.03976",
        "Estimating individual treatment effect: generalization bounds and algorithms",
        2016,
        "2016-06-13",
        "ite; cfr; tarnet; representation; foundational-boundary",
        "Foundational TARNet/CFR generalization bounds paper; included as a 2016 boundary item.",
        notes="Strict current-date 10-year cutoff starts 2016-06-27; retained for field coverage.",
    ),
    arxiv_entry(
        "1612.09596",
        "Counterfactual Prediction with Deep Instrumental Variables Networks",
        2016,
        "2016-12-30",
        "iv; deepiv; counterfactual",
        "Deep IV network for counterfactual prediction.",
    ),
    arxiv_entry(
        "1705.08821",
        "Causal Effect Inference with Deep Latent-Variable Models",
        2017,
        "2017-05-24",
        "ite; cevae; latent-confounding; variational",
        "CEVAE latent-variable deep causal model for treatment effects.",
    ),
    arxiv_entry(
        "1706.05966",
        "Deep Counterfactual Networks with Propensity-Dropout",
        2017,
        "2017-06-19",
        "ite; counterfactual; propensity; neural",
        "Deep counterfactual network architecture with propensity-dropout.",
    ),
    non_arxiv_entry(
        "Generative Adversarial Nets for Individualized Treatment Effects",
        2018,
        "2017-11-27",
        "ite; ganite; adversarial; openreview",
        "GANITE paper with a confirmed public OpenReview PDF.",
        "OpenReview API / public PDF",
        authors="Jinsung Yoon; James Jordon; Mihaela van der Schaar",
        venue="ICLR 2018",
        url="https://openreview.net/forum?id=ByKWUeWA-",
        download_target="https://openreview.net/pdf?id=ByKWUeWA-&download=1",
    ),
    arxiv_entry(
        "1803.00149",
        "Deep Learning for Causal Inference",
        2018,
        "2018-03-01",
        "ite; review; deep-learning",
        "Early survey/tutorial treatment of deep learning for causal inference.",
    ),
    arxiv_entry(
        "1808.07804",
        "Transfer Learning for Estimating Causal Effects using Neural Networks",
        2018,
        "2018-08-23",
        "ite; transfer-learning; neural",
        "Neural transfer learning for causal effect estimation.",
    ),
    arxiv_entry(
        "1809.09953",
        "Deep Neural Networks for Estimation and Inference",
        2018,
        "2018-09-26",
        "deep-inference; semiparametric; neural-networks; local-reference",
        "DNN estimation and inference theory directly relevant to this repo.",
        local_reference="references/FLM2021_docling.md",
    ),
    arxiv_entry(
        "1810.00656",
        "Perfect Match: A Simple Method for Learning Representations For Counterfactual Inference With Neural Networks",
        2018,
        "2018-10-01",
        "ite; matching; representation; neural",
        "Representation learning for counterfactual inference with neural matching.",
    ),
    arxiv_entry(
        "1810.11010",
        "Heterogeneous Treatment Effect Estimation through Deep Learning",
        2018,
        "2018-10-25",
        "hte; ite; neural",
        "Direct heterogeneous treatment effect estimation with deep learning.",
    ),
    non_arxiv_entry(
        "Learning Causal Structures for Individualized Treatment Effects",
        2018,
        "2018-01-01",
        "ite; site; representation; unresolved-pdf",
        "Named SITE method; retained in index, but no verified public PDF route was found in this pass.",
        "Curated named-method check",
        venue="IJCAI 2018",
        download_target="Learning Causal Structures for Individualized Treatment Effects",
        notes="websource may find a public author copy; quick DOI/arXiv checks did not verify one.",
    ),
    arxiv_entry(
        "1901.09036",
        "Orthogonal Statistical Learning",
        2019,
        "2019-01-25",
        "orthogonal-learning; semiparametric; local-reference",
        "Orthogonal statistical learning with machine learning nuisances.",
        local_reference="references/FS2023_docling.md",
    ),
    arxiv_entry(
        "1902.00450",
        "Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders",
        2019,
        "2019-02-01",
        "longitudinal; deconfounder; time-series; hidden-confounding",
        "Deep latent-factor treatment-effect estimation over time.",
    ),
    arxiv_entry(
        "1905.12495",
        "Deep Generalized Method of Moments for Instrumental Variable Analysis",
        2019,
        "2019-05-29",
        "iv; gmm; deepgmm",
        "Deep GMM approach for instrumental-variable causal analysis.",
    ),
    arxiv_entry(
        "1906.02120",
        "Adapting Neural Networks for the Estimation of Treatment Effects",
        2019,
        "2019-06-05",
        "ite; dragonnet; targeted-regularization",
        "Dragonnet / targeted regularization for treatment effect estimation.",
        local_reference="references/Dragonnet2019_docling.md",
    ),
    arxiv_entry(
        "2001.04754",
        "Learning Overlapping Representations for the Estimation of Individualized Treatment Effects",
        2020,
        "2020-01-14",
        "ite; overlap; representation",
        "Representation learning focused on overlap for ITE estimation.",
    ),
    arxiv_entry(
        "2001.10652",
        "Treatment effect estimation with disentangled latent factors",
        2020,
        "2020-01-29",
        "ite; disentanglement; latent-factors",
        "Disentangled latent factors for treatment-effect estimation.",
    ),
    arxiv_entry(
        "2002.04083",
        "Estimating Counterfactual Treatment Outcomes over Time Through Adversarially Balanced Representations",
        2020,
        "2020-02-10",
        "longitudinal; counterfactual-outcomes; adversarial; representation",
        "Adversarially balanced representations for counterfactual outcomes over time.",
    ),
    arxiv_entry(
        "2004.03036",
        "Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments",
        2020,
        "2020-04-06",
        "continuous-treatment; dml; semiparametric; local-reference",
        "Continuous-treatment inference with debiased machine learning.",
        local_reference="references/CL2026_docling.md",
    ),
    arxiv_entry(
        "2004.05013",
        "Estimating Individual Treatment Effects through Causal Populations Identification",
        2020,
        "2020-04-10",
        "ite; causal-populations",
        "Causal-population identification approach to ITE estimation.",
    ),
    arxiv_entry(
        "2004.14954",
        "On Deep Instrumental Variables Estimate",
        2020,
        "2020-04-30",
        "iv; deepiv; theory",
        "Analysis of deep instrumental-variable estimates.",
    ),
    arxiv_entry(
        "2008.09858",
        "Hi-CI: Deep Causal Inference in High Dimensions",
        2020,
        "2020-08-22",
        "ite; high-dimensional; deep-learning",
        "High-dimensional deep causal inference method.",
    ),
    arxiv_entry(
        "2008.13620",
        "Estimating Individual Treatment Effects with Time-Varying Confounders",
        2020,
        "2020-08-27",
        "ite; longitudinal; time-varying-confounding",
        "ITE estimation with time-varying confounders.",
    ),
    arxiv_entry(
        "2009.07055",
        "Causal Inference of General Treatment Effects using Neural Networks with A Diverging Number of Confounders",
        2020,
        "2020-09-15",
        "general-treatment-effects; neural-networks; high-dimensional; local-reference",
        "General treatment effects with neural networks and many confounders.",
        local_reference="references/CLMZ2024_docling.md",
    ),
    arxiv_entry(
        "2010.07154",
        "Learning Deep Features in Instrumental Variable Regression",
        2020,
        "2020-10-14",
        "iv; dfiv; representation",
        "Deep feature learning for instrumental-variable regression.",
    ),
    arxiv_entry(
        "2010.14694",
        "Deep Learning for Individual Heterogeneity",
        2020,
        "2020-10-28",
        "individual-heterogeneity; deep-inference; local-reference",
        "Automatic inference framework for individual heterogeneity.",
        doi="10.47004/wp.cem.2021.2921",
        local_reference="references/FLM2023_docling.md; references/FLM2025_docling.md",
    ),
    arxiv_entry(
        "2011.00041",
        "Adapting Neural Networks for Uplift Models",
        2020,
        "2020-10-30",
        "uplift; neural; ite",
        "Neural network adaptation for uplift modeling.",
    ),
    arxiv_entry(
        "2103.07861",
        "VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments",
        2021,
        "2021-03-14",
        "continuous-treatment; vcnet; targeted-regularization",
        "VCNet and functional targeted regularization for continuous treatment effects.",
    ),
    arxiv_entry(
        "2104.14737",
        "Automatic Debiased Machine Learning via Riesz Regression",
        2021,
        "2021-04-30",
        "auto-dml; riesz; semiparametric; local-reference",
        "Automatic debiasing via Riesz regression.",
        local_reference="references/IN_glr_docling.md",
    ),
    arxiv_entry(
        "2105.05146",
        "A Twin Neural Model for Uplift",
        2021,
        "2021-05-11",
        "uplift; neural; ite",
        "Twin-neural uplift model.",
    ),
    arxiv_entry(
        "2106.02881",
        "Graph Infomax Adversarial Learning for Treatment Effect Estimation with Networked Observational Data",
        2021,
        "2021-06-05",
        "network; graph-neural; adversarial; ite",
        "Graph infomax adversarial learning for networked treatment-effect estimation.",
        doi="10.1145/3447548.3467302",
    ),
    arxiv_entry(
        "2110.03031",
        "RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests",
        2021,
        "2021-10-06",
        "riesznet; auto-dml; neural-networks; local-reference",
        "RieszNet automatic debiased ML with neural nets.",
        local_reference="references/RieszNet2022_docling.md",
    ),
    arxiv_entry(
        "2110.04442",
        "A Primer on Deep Learning for Causal Inference",
        2021,
        "2021-10-09",
        "survey; deep-learning; causal-inference",
        "Primer/survey; included as a map of the field.",
    ),
    arxiv_entry(
        "2110.14001",
        "SurvITE: Learning Heterogeneous Treatment Effects from Time-to-Event Data",
        2021,
        "2021-10-26",
        "survival; hte; time-to-event; neural",
        "Heterogeneous treatment effects for survival outcomes.",
    ),
    arxiv_entry(
        "2005.05099",
        "Counterfactual Propagation for Semi-Supervised Individual Treatment Effect Estimation",
        2020,
        "2020-05-11",
        "ite; semi-supervised; counterfactual",
        "Semi-supervised counterfactual propagation for ITE estimation.",
        doi="10.1007/978-3-030-67658-2_31",
        venue="Lecture Notes in Computer Science",
    ),
    arxiv_entry(
        "2201.03448",
        "BITES: Balanced Individual Treatment Effect for Survival data",
        2022,
        "2022-01-05",
        "survival; ite; representation",
        "Balanced ITE estimation for survival data.",
    ),
    arxiv_entry(
        "2201.08559",
        "Individual Treatment Effect Estimation Through Controlled Neural Network Training in Two Stages",
        2022,
        "2022-01-21",
        "ite; neural-training",
        "Two-stage controlled neural network training for ITE.",
    ),
    arxiv_entry(
        "2203.15672",
        "SurvCaus : Representation Balancing for Survival Causal Inference",
        2022,
        "2022-03-29",
        "survival; representation; causal-inference",
        "Representation balancing for survival causal inference.",
    ),
    arxiv_entry(
        "2204.07258",
        "Causal Transformer for Estimating Counterfactual Outcomes",
        2022,
        "2022-04-14",
        "longitudinal; transformer; counterfactual-outcomes",
        "Transformer model for counterfactual outcome estimation.",
    ),
    arxiv_entry(
        "2204.10022",
        "Scalable Sensitivity and Uncertainty Analysis for Causal-Effect Estimates of Continuous-Valued Interventions",
        2022,
        "2022-04-21",
        "continuous-treatment; uncertainty; sensitivity",
        "Sensitivity and uncertainty analysis for continuous interventions.",
    ),
    arxiv_entry(
        "2204.10495",
        "Adversarial Estimators",
        2022,
        "2022-04-22",
        "adversarial; semiparametric; local-reference",
        "Adversarial estimation for semiparametric targets.",
        local_reference="references/Metzger2022_docling.md",
    ),
    arxiv_entry(
        "2206.01022",
        "Learning Disentangled Representations for Counterfactual Regression via Mutual Information Minimization",
        2022,
        "2022-06-02",
        "counterfactual-regression; disentanglement; representation",
        "Mutual-information-minimizing disentangled representations for counterfactual regression.",
    ),
    arxiv_entry(
        "2206.01900",
        "Estimating counterfactual treatment outcomes over time in complex multiagent scenarios",
        2022,
        "2022-06-04",
        "longitudinal; multiagent; counterfactual-outcomes",
        "Counterfactual treatment outcomes over time in multiagent scenarios.",
    ),
    arxiv_entry(
        "2206.10261",
        "Interpretable Deep Causal Learning for Moderation Effects",
        2022,
        "2022-06-21",
        "moderation-effects; interpretable; neural",
        "Interpretable deep causal learning for moderation effects.",
    ),
    arxiv_entry(
        "2207.04049",
        "Learning Causal Effects on Hypergraphs",
        2022,
        "2022-07-07",
        "hypergraph; network; causal-effects",
        "Causal effect learning on hypergraphs.",
    ),
    arxiv_entry(
        "2207.09920",
        "DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation",
        2022,
        "2022-07-19",
        "ite; descn; cross-network",
        "Deep entire-space cross network for ITE.",
    ),
    arxiv_entry(
        "2207.11251",
        "Variational Temporal Deconfounder for Individualized Treatment Effect Estimation from Longitudinal Observational Data",
        2022,
        "2022-07-23",
        "longitudinal; deconfounder; variational; ite",
        "Variational temporal deconfounder for longitudinal ITE.",
    ),
    arxiv_entry(
        "2208.08544",
        "Estimating individual treatment effects under unobserved confounding using binary instruments",
        2022,
        "2022-08-17",
        "iv; ite; unobserved-confounding",
        "ITE estimation under unobserved confounding using binary instruments.",
    ),
    non_arxiv_entry(
        "Review of Deep Learning Methods for Individual Treatment Effect Estimation with Automatic Hyperparameter Optimization",
        2022,
        "2022-08-01",
        "review; ite; hyperparameter-optimization",
        "Review paper found via Crossref.",
        "Crossref API / curated query set",
        doi="10.36227/techrxiv.20448768",
        venue="TechRxiv",
    ),
    non_arxiv_entry(
        "Deep Learning for Counterfactual Inference and Treatment Effect Estimation",
        2022,
        "2022-01-01",
        "book-chapter; counterfactual-inference; treatment-effects",
        "Book chapter found via Crossref; included for coverage.",
        "Crossref API / curated query set",
        doi="10.1201/9781003028543-7",
        venue="Chapman and Hall/CRC",
    ),
    arxiv_entry(
        "2303.04201",
        "DR-VIDAL -- Doubly Robust Variational Information-theoretic Deep Adversarial Learning for Counterfactual Prediction and Treatment Effect Estimation on Real World Data",
        2023,
        "2023-03-07",
        "ite; doubly-robust; variational; adversarial",
        "Doubly robust variational adversarial model for counterfactual prediction and treatment effects.",
    ),
    arxiv_entry(
        "2305.15984",
        "Dynamic Inter-treatment Information Sharing for Individualized Treatment Effects Estimation",
        2023,
        "2023-05-25",
        "ite; multiple-treatments; information-sharing",
        "Dynamic inter-treatment information sharing for ITE.",
    ),
    arxiv_entry(
        "2305.19742",
        "Reliable Off-Policy Learning for Dosage Combinations",
        2023,
        "2023-05-31",
        "dosage; off-policy; continuous-treatment",
        "Reliable off-policy learning for dosage combinations.",
    ),
    arxiv_entry(
        "2307.03315",
        "Assisting Clinical Decisions for Scarcely Available Treatment via Disentangled Latent Representation",
        2023,
        "2023-07-06",
        "ite; clinical; scarce-treatment; disentanglement",
        "Clinical scarce-treatment method using disentangled latent representations.",
    ),
    arxiv_entry(
        "2309.13884",
        "Estimating Treatment Effects Under Heterogeneous Interference",
        2023,
        "2023-09-25",
        "interference; network; treatment-effects",
        "Treatment effects under heterogeneous interference.",
    ),
    arxiv_entry(
        "2310.10559",
        "Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data",
        2023,
        "2023-10-16",
        "longitudinal; vae; counterfactual-regression",
        "Dynamic VAE for longitudinal counterfactual regression.",
    ),
    arxiv_entry(
        "2311.08434",
        "Uplift Modeling based on Graph Neural Network Combined with Causal Knowledge",
        2023,
        "2023-11-14",
        "uplift; graph-neural; causal-knowledge",
        "GNN uplift modeling with causal knowledge.",
    ),
    arxiv_entry(
        "2312.10570",
        "Adversarially Balanced Representation for Continuous Treatment Effect Estimation",
        2023,
        "2023-12-17",
        "continuous-treatment; adversarial; representation",
        "Adversarial balanced representation for continuous treatment effects.",
    ),
    arxiv_entry(
        "2401.06557",
        "Treatment-Aware Hyperbolic Representation Learning for Causal Effect Estimation with Social Networks",
        2024,
        "2024-01-12",
        "social-network; hyperbolic; representation; treatment-effects",
        "Treatment-aware hyperbolic representations for social-network causal effects.",
    ),
    arxiv_entry(
        "2403.00178",
        "Causal Graph ODE: Continuous Treatment Effect Modeling in Multi-agent Dynamical Systems",
        2024,
        "2024-02-29",
        "continuous-treatment; graph-ode; multiagent",
        "Graph ODE model for continuous treatment effects in multiagent dynamics.",
    ),
    arxiv_entry(
        "2403.04236",
        "Regularized DeepIV with Model Selection",
        2024,
        "2024-03-07",
        "iv; deepiv; model-selection",
        "Regularized DeepIV with model selection.",
    ),
    arxiv_entry(
        "2403.06489",
        "Graph Neural Network with Two Uplift Estimators for Label-Scarcity Individual Uplift Modeling",
        2024,
        "2024-03-11",
        "uplift; graph-neural; label-scarcity",
        "GNN uplift estimators for label scarcity.",
    ),
    arxiv_entry(
        "2404.04399",
        "Longitudinal Targeted Minimum Loss-based Estimation with Temporal-Difference Heterogeneous Transformer",
        2024,
        "2024-04-05",
        "longitudinal; tmle; transformer; local-reference",
        "Deep LTMLE with temporal-difference heterogeneous transformer.",
        local_reference="references/DeepLTMLE2024_docling.md",
    ),
    arxiv_entry(
        "2405.03130",
        "Deep Learning for Causal Inference: A Comparison of Architectures for Heterogeneous Treatment Effect Estimation",
        2024,
        "2024-05-06",
        "survey; hte; architecture-comparison",
        "Architecture comparison for deep HTE estimation.",
    ),
    arxiv_entry(
        "2405.09493",
        "C-Learner: Constrained Learning for Causal Inference",
        2024,
        "2024-05-15",
        "c-learner; constrained-learning; semiparametric; local-reference",
        "Constrained learning for causal inference; includes neural instantiations.",
        local_reference="references/CLearner2025_docling.md",
    ),
    arxiv_entry(
        "2405.15505",
        "Revisiting Counterfactual Regression through the Lens of Gromov-Wasserstein Information Bottleneck",
        2024,
        "2024-05-24",
        "counterfactual-regression; information-bottleneck; wasserstein",
        "Counterfactual regression revisited with Gromov-Wasserstein information bottleneck.",
    ),
    arxiv_entry(
        "2406.00535",
        "Causal Contrastive Learning for Counterfactual Regression Over Time",
        2024,
        "2024-06-01",
        "longitudinal; contrastive-learning; counterfactual-regression",
        "Contrastive learning for counterfactual regression over time.",
    ),
    arxiv_entry(
        "2408.02045",
        "DNA-SE: Towards Deep Neural-Nets Assisted Semiparametric Estimation",
        2024,
        "2024-08-04",
        "semiparametric; deep-inference; neural-networks; local-reference",
        "Deep neural-nets assisted semiparametric estimation.",
        local_reference="references/DNASE2024_docling.md",
    ),
    arxiv_entry(
        "2408.05428",
        "Generalized Encouragement-Based Instrumental Variables for Counterfactual Regression",
        2024,
        "2024-08-10",
        "iv; encouragement; counterfactual-regression",
        "Encouragement-based IV setup for counterfactual regression.",
    ),
    arxiv_entry(
        "2408.09560",
        "Deep Learning for the Estimation of Heterogeneous Parameters in Discrete Choice Models",
        2024,
        "2024-08-18",
        "heterogeneous-parameters; discrete-choice; local-reference",
        "Deep learning for heterogeneous parameters in discrete choice.",
        local_reference="references/HO2024_multinomial_docling.md",
    ),
    arxiv_entry(
        "2409.19777",
        "Automatic debiasing of neural networks via moment-constrained learning",
        2024,
        "2024-09-29",
        "auto-dml; moment-constrained-learning; neural-networks; local-reference",
        "Moment-constrained neural debiasing for semiparametric targets.",
        local_reference="references/HH2025_docling.md",
    ),
    non_arxiv_entry(
        "Deep Learning-Based Causal Inference for Large-Scale Combinatorial Experiments: Theory and Empirical Evidence",
        2025,
        "2025-10-15",
        "combinatorial-experiments; causal-inference; deep-learning; local-reference",
        "Deep learning based causal inference for large-scale combinatorial experiments.",
        "Crossref API / local reference",
        doi="10.1287/mnsc.2024.04625",
        venue="Management Science",
        download_target="10.2139/ssrn.4375327",
        local_reference="references/DeDL2025_docling.md",
        notes="Published DOI is 10.1287/mnsc.2024.04625; SSRN DOI is used as the first public retrieval route.",
    ),
    arxiv_entry(
        "2504.02694",
        "Semiparametric Counterfactual Regression",
        2025,
        "2025-04-03",
        "counterfactual-regression; semiparametric",
        "Semiparametric counterfactual regression.",
    ),
    arxiv_entry(
        "2504.19089",
        "Semiparametric M-estimation with overparameterized neural networks",
        2025,
        "2025-04-27",
        "semiparametric; overparameterized; neural-networks; local-reference",
        "Semiparametric M-estimation with overparameterized neural networks.",
        local_reference="references/YCY2025_docling.md",
    ),
    arxiv_entry(
        "2505.01995",
        "Extended Fiducial Inference for Individual Treatment Effects via Deep Neural Networks",
        2025,
        "2025-05-04",
        "ite; fiducial-inference; neural-networks",
        "Extended fiducial inference for ITE via DNNs.",
    ),
    arxiv_entry(
        "2506.02790",
        "Orthogonality-Constrained Deep Instrumental Variable Model for Causal Effect Estimation",
        2025,
        "2025-06-03",
        "iv; orthogonality; deepiv",
        "Orthogonality-constrained deep IV model for causal effect estimation.",
    ),
    arxiv_entry(
        "2507.12435",
        "Targeted Deep Architectures: A TMLE-Based Framework for Robust Causal Inference in Neural Networks",
        2025,
        "2025-07-16",
        "tmle; targeted-deep-architecture; neural-networks; local-reference",
        "TMLE-based targeted deep architectures.",
        local_reference="references/TDA2025_docling.md",
    ),
    arxiv_entry(
        "2509.22467",
        "CausalKANs: interpretable treatment effect estimation with Kolmogorov-Arnold networks",
        2025,
        "2025-09-26",
        "ite; kan; interpretable",
        "Treatment-effect estimation with Kolmogorov-Arnold networks.",
    ),
    arxiv_entry(
        "2509.22953",
        "GDR-learners: Orthogonal Learning of Generative Models for Potential Outcomes",
        2025,
        "2025-09-26",
        "potential-outcomes; generative-model; orthogonal-learning; local-reference",
        "Orthogonal learning of generative potential-outcome models.",
        local_reference="references/GDR2026_docling.md",
    ),
    non_arxiv_entry(
        "Individualized treatment rules based on adaptive transfer-dragonnet",
        2025,
        "2025-01-01",
        "ite; dragonnet; transfer-learning",
        "Adaptive transfer-Dragonnet treatment-rule paper found via Crossref.",
        "Crossref API / curated query set",
        doi="10.1007/s11222-025-10704-9",
        venue="Statistics and Computing",
    ),
    non_arxiv_entry(
        "Treatment Effect Estimation in Survival Analysis Using Copula-Based Deep Learning Models for Causal Inference",
        2025,
        "2025-06-01",
        "survival; copula; deep-learning; causal-inference",
        "Survival treatment-effect paper found via Crossref.",
        "Crossref API / curated query set",
        doi="10.3390/axioms14060458",
        venue="Axioms",
    ),
    arxiv_entry(
        "2604.10398",
        "Estimating heterogeneous treatment effects with survival outcomes via a deep survival learner",
        2026,
        "2026-04-12",
        "survival; hte; deep-survival",
        "Deep survival learner for HTE with survival outcomes.",
    ),
    arxiv_entry(
        "2605.07065",
        "Causal EpiNets: Precision-corrected Bounds on Individual Treatment Effects using Epistemic Neural Networks",
        2026,
        "2026-05-08",
        "ite; uncertainty; epistemic-neural-networks",
        "Epistemic neural network bounds on ITE.",
    ),
    arxiv_entry(
        "2606.18969",
        "Balanced Twins: Causal Inference on Time Series with Hidden Confounding",
        2026,
        "2026-06-17",
        "time-series; hidden-confounding; representation",
        "Time-series causal inference with hidden confounding.",
    ),
    non_arxiv_entry(
        "Individualized treatment effect estimation with compromised adversarial nets",
        2026,
        "2026-01-01",
        "ite; adversarial; gan",
        "Adversarial-net ITE paper found via Crossref.",
        "Crossref API / curated query set",
        doi="10.1007/s00180-025-01705-3",
        venue="Computational Statistics",
        notes="Crossref returned publication year 2026; no precise date was available in the quick query.",
    ),
]


def fetch_arxiv_metadata(ids: Iterable[str]) -> dict[str, dict[str, str | int]]:
    ids = list(ids)
    if not ids:
        return {}
    url = "https://export.arxiv.org/api/query?" + urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids) + 5}
    )
    request = Request(url, headers={"User-Agent": "deep-inference-literature-pack/1.0"})
    root = ET.fromstring(urlopen(request, timeout=45).read())
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out: dict[str, dict[str, str | int]] = {}
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.findtext("atom:id", namespaces=ns).rsplit("/", 1)[-1]
        base_id = arxiv_id.split("v", 1)[0]
        title = " ".join((entry.findtext("atom:title", namespaces=ns) or "").split())
        published = (entry.findtext("atom:published", namespaces=ns) or "")[:10]
        authors = [
            " ".join((author.findtext("atom:name", namespaces=ns) or "").split())
            for author in entry.findall("atom:author", ns)
        ]
        out[base_id] = {
            "title": title,
            "date": published,
            "year": int(published[:4]),
            "authors": "; ".join(author for author in authors if author),
        }
    return out


def existing_arxiv_metadata() -> dict[str, dict[str, str | int]]:
    path = ROOT / "manifest.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, str | int]] = {}
    for row in rows:
        arxiv_id = row.get("arxiv_id")
        if not arxiv_id:
            continue
        out[arxiv_id] = {
            "title": row.get("title") or "",
            "date": row.get("date") or "",
            "year": int(row.get("year") or 0),
            "authors": row.get("authors") or "",
        }
    return out


def refresh_arxiv(entries: list[Entry]) -> None:
    ids = [entry.arxiv_id for entry in entries if entry.arxiv_id]
    try:
        metadata = fetch_arxiv_metadata(ids)
    except Exception as exc:  # pragma: no cover - network fallback
        print(f"warning: arXiv metadata refresh failed: {exc}", file=sys.stderr)
        metadata = existing_arxiv_metadata()
    missing = sorted(set(ids) - set(metadata))
    if missing and len(metadata) == len(ids) - len(missing):
        print(f"warning: using static metadata for arXiv ids: {', '.join(missing)}", file=sys.stderr)
    elif missing:
        raise RuntimeError(f"arXiv metadata missing for: {', '.join(missing)}")
    for entry in entries:
        if not entry.arxiv_id:
            continue
        meta = metadata.get(entry.arxiv_id)
        if not meta:
            continue
        entry.title = str(meta["title"])
        entry.key = keyify(entry.title)
        entry.date = str(meta["date"])
        entry.year = int(meta["year"])
        entry.authors = str(meta["authors"])


def validate(entries: list[Entry]) -> None:
    seen: dict[str, str] = {}
    for entry in entries:
        entry_date = date.fromisoformat(entry.date)
        if not (START_DATE <= entry_date <= CUTOFF_DATE):
            raise ValueError(f"{entry.key}: date outside 2016-01-01..{CUTOFF_DATE}: {entry.date}")
        for label, value in (
            ("key", entry.key),
            ("title", entry.title.casefold()),
            ("arxiv_id", entry.arxiv_id),
            ("doi", entry.doi.casefold()),
            ("url", entry.url.casefold()),
        ):
            if not value:
                continue
            compound = f"{label}:{value}"
            if compound in seen:
                raise ValueError(f"duplicate {label}: {value} in {entry.key} and {seen[compound]}")
            seen[compound] = entry.key
        if not entry.download_target:
            raise ValueError(f"{entry.key}: missing download target")


def write_csv(entries: list[Entry], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.as_dict())


def write_json(entries: list[Entry], path: Path) -> None:
    path.write_text(
        json.dumps([entry.as_dict() for entry in entries], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_papers_txt(entries: list[Entry], path: Path) -> None:
    lines = [
        "# causal deep learning papers, 2016-2026",
        "# Non-comment lines are fed directly to websource.",
        "# Generated by build_manifest.py.",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"# {entry.key}",
                f"# {entry.year} | {entry.title}",
                f"# tags: {entry.tags}",
                entry.download_target,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(entries: list[Entry], path: Path) -> None:
    by_year = Counter(entry.year for entry in entries)
    by_status = Counter(entry.retrieval_status for entry in entries)
    major_tags = Counter()
    for entry in entries:
        for tag in entry.tags.split("; "):
            major_tags[tag] += 1
    strict_boundary = [
        entry for entry in entries if date.fromisoformat(entry.date) < STRICT_TEN_YEAR_START
    ]
    lines = [
        "# Causal Deep Learning Papers, 2016-2026",
        "",
        f"Generated on {CUTOFF_DATE.isoformat()} for a high-recall treatment-effect and neural semiparametric causal-inference pass.",
        "",
        "## Contents",
        "",
        f"- `manifest.csv` / `manifest.json`: {len(entries)} curated records with DOI/arXiv/source fields.",
        "- `papers.txt`: batch input for `websource`.",
        "- `downloads/`: local PDFs downloaded by `websource` (ignored by the repo-wide `downloads/` gitignore rule).",
        "- `download_status.csv`: retrieval status after the `websource` run.",
        "- `search_notes.md`: source queries, inclusion rules, and unresolved checks.",
        "",
        "## Scope",
        "",
        "Included papers use deep neural, representation, adversarial, transformer, graph, variational, KAN, or neural semiparametric machinery for treatment effects, counterfactual outcomes, uplift, IV, longitudinal causal inference, or closely related potential-outcome/semiparametric targets.",
        "",
        "The strict current-date ten-year window starts on 2016-06-27. Two May/June 2016 foundational counterfactual-representation entries are retained and tagged `foundational-boundary` because the practical literature window is 2016-2026.",
        "",
        "## Counts By Year",
        "",
    ]
    for year in sorted(by_year):
        lines.append(f"- {year}: {by_year[year]}")
    lines.extend(["", "## Most Common Tags", ""])
    for tag, count in major_tags.most_common(20):
        lines.append(f"- {tag}: {count}")
    if strict_boundary:
        lines.extend(["", "## Boundary Entries", ""])
        for entry in strict_boundary:
            lines.append(f"- {entry.date}: {entry.title}")
    if by_status:
        lines.extend(["", "## Retrieval Status", ""])
        for status, count in sorted(by_status.items()):
            lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "```bash",
            "python3 build_manifest.py",
            "websource papers.txt -o downloads -v --limit 8 --max-downloads 1 --timeout 30",
            "python3 build_manifest.py --sync-downloads",
            "```",
            "",
            "The manifest is curated, while arXiv title/date/author metadata is refreshed from `https://export.arxiv.org/api/query` when available.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_search_notes(entries: list[Entry], path: Path) -> None:
    arxiv_count = sum(1 for entry in entries if entry.arxiv_id)
    doi_count = sum(1 for entry in entries if entry.doi)
    lines = [
        "# Search Notes",
        "",
        "## Sources Used",
        "",
        "- arXiv API: `https://export.arxiv.org/api/query` for exact ID validation and broad title/query discovery.",
        "- Crossref API: `https://api.crossref.org/works` for DOI-backed non-arXiv and publisher records.",
        "- OpenReview API/PDF endpoint for GANITE.",
        "- Existing local `references/*_docling.md` titles in this repo for deep-inference-specific coverage.",
        "- `websource` batch mode for public PDF retrieval.",
        "",
        "OpenAlex and Semantic Scholar were attempted during discovery but returned HTTP 429 in this session, so they were not treated as authoritative for this pack.",
        "",
        "## Discovery Queries",
        "",
        "- arXiv: `individual treatment effect` + `deep` / `neural`.",
        "- arXiv: `heterogeneous treatment effects` + `neural`.",
        "- arXiv: `counterfactual regression` + `treatment`.",
        "- arXiv exact-title checks for CFR, TARNet, CEVAE, Dragonnet, DeepIV, DeepGMM, DFIV, VCNet, RieszNet, C-Learner, Deep LTMLE, GDR, TDA, and related local-reference titles.",
        "- Crossref title queries for GANITE, SITE, adaptive transfer-Dragonnet, deep counterfactual inference chapters/reviews, and survival/copula variants.",
        "",
        "## Inclusion Rules",
        "",
        "- Keep method papers and surveys/reviews when the title/abstract route is explicitly causal treatment-effect, potential-outcome, counterfactual-outcome, uplift, IV, longitudinal deconfounding, or semiparametric causal inference with neural/deep components.",
        "- Keep repo-local deep-inference papers even when they are framed as semiparametric inference rather than ITE/CATE, because this repository implements that branch.",
        "- Exclude purely classical causal ML, purely predictive healthcare, generic graph causality without treatment-effect estimands, and application-only Dragonnet uses unless they introduce a method.",
        "",
        "## Known Unresolved / Manual Checks",
        "",
        "- `Learning Causal Structures for Individualized Treatment Effects` (SITE) is indexed because it is a named deep ITE method, but this pass did not verify a public PDF or DOI route. It remains in `papers.txt` as a title query for `websource`.",
        "- Publisher DOI entries may require manual/public author-copy lookup if `websource` cannot find an open PDF route.",
        "",
        "## Manifest Totals",
        "",
        f"- Total records: {len(entries)}",
        f"- arXiv-backed records: {arxiv_count}",
        f"- DOI-backed records: {doi_count}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def sync_downloads(entries: list[Entry], downloads_dir: Path) -> None:
    pdfs = [path for path in downloads_dir.glob("*.pdf") if path.stat().st_size > 0]
    used: set[Path] = set()
    for entry in entries:
        candidates: list[Path] = []
        if entry.arxiv_id:
            compact = entry.arxiv_id.replace(".", "")
            candidates = [
                path
                for path in pdfs
                if entry.arxiv_id in path.name or compact in normalize(path.name)
            ]
        if not candidates:
            title_norm = normalize(entry.title)
            key_norm = normalize(entry.key)
            target_norm = normalize(entry.download_target)
            candidates = [
                path
                for path in pdfs
                if path not in used
                and (
                    key_norm in normalize(path.stem)
                    or normalize(path.stem) in title_norm
                    or (target_norm and normalize(path.stem).replace("pdf", "") in target_norm)
                    or (target_norm and target_norm in normalize(path.stem))
                )
            ]
        if candidates:
            chosen = sorted(candidates, key=lambda path: (path in used, len(path.name), path.name))[0]
            used.add(chosen)
            entry.retrieval_status = "downloaded"
            entry.pdf_path = str(chosen.relative_to(ROOT))
        else:
            entry.retrieval_status = "missing"
            entry.pdf_path = ""


def write_status(entries: list[Entry], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "year", "title", "download_target", "retrieval_status", "pdf_path", "notes"],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "key": entry.key,
                    "year": entry.year,
                    "title": entry.title,
                    "download_target": entry.download_target,
                    "retrieval_status": entry.retrieval_status,
                    "pdf_path": entry.pdf_path,
                    "notes": entry.notes,
                }
            )


def write_retrieval_summary(entries: list[Entry], path: Path) -> None:
    counts = Counter(entry.retrieval_status for entry in entries)
    pdfs = [pdf for pdf in (ROOT / "downloads").glob("*.pdf") if pdf.stat().st_size > 0]
    total_bytes = sum(pdf.stat().st_size for pdf in pdfs)
    lines = [
        "# Retrieval Summary",
        "",
        f"- Manifest records: {len(entries)}",
        f"- PDFs on disk: {len(pdfs)}",
        f"- PDF bytes on disk: {total_bytes}",
        f"- Manifest status `downloaded`: {counts.get('downloaded', 0)}",
        f"- Manifest status `missing`: {counts.get('missing', 0)}",
        "",
        "## Runs",
        "",
        "- `websource_run_20260627.txt`: main `websource papers.txt` run. It downloaded the arXiv/OpenReview records and attempted DOI-backed records; the run was manually interrupted after repeated OpenAlex/Semantic Scholar 429 backoff on DOI failures.",
        "- `websource_missing_retry_20260627.txt`: title-only retry for unresolved records with OpenAlex/Semantic Scholar excluded from search sources. It found the public arXiv copy of Counterfactual Propagation, then was interrupted when DOI expansion again entered OpenAlex backoff.",
        "",
        "Sci-Hub was not used. Both runs passed `--mirrors-file ./no-scihub-mirrors.txt`.",
        "",
    ]
    missing = [entry for entry in entries if entry.retrieval_status != "downloaded"]
    if missing:
        lines.extend(["## Missing Public PDFs", ""])
        for entry in missing:
            parts = [f"- {entry.year}: {entry.title}"]
            if entry.doi:
                parts.append(f"DOI `{entry.doi}`")
            if entry.notes:
                parts.append(f"Note: {entry.notes.rstrip('.')}")
            lines.append(". ".join(parts) + ".")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build(args: argparse.Namespace) -> list[Entry]:
    entries = [Entry(**entry.as_dict()) for entry in ENTRIES]
    if args.refresh_arxiv:
        refresh_arxiv(entries)
    entries.sort(key=lambda entry: (entry.year, entry.date, entry.title.casefold()))
    if args.sync_downloads:
        sync_downloads(entries, ROOT / "downloads")
    validate(entries)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-refresh-arxiv",
        dest="refresh_arxiv",
        action="store_false",
        help="Use static arXiv metadata instead of refreshing title/author/date fields.",
    )
    parser.add_argument(
        "--sync-downloads",
        action="store_true",
        help="Scan downloads/*.pdf and update retrieval status fields.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate only; do not rewrite files.",
    )
    parser.set_defaults(refresh_arxiv=True)
    args = parser.parse_args()

    entries = build(args)
    if args.check:
        print(f"ok: {len(entries)} entries validated")
        return 0

    write_csv(entries, ROOT / "manifest.csv")
    write_json(entries, ROOT / "manifest.json")
    write_papers_txt(entries, ROOT / "papers.txt")
    write_readme(entries, ROOT / "README.md")
    write_search_notes(entries, ROOT / "search_notes.md")
    write_status(entries, ROOT / "download_status.csv")
    write_retrieval_summary(entries, ROOT / "retrieval_summary.md")
    print(f"wrote {len(entries)} entries to {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
