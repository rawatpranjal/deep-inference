# H&M Fashion Demand Application

Companion code for "A Practitioner's Guide to Deep Learning for Individual Heterogeneity" (Rawat & Misra, 2026).

## Pipeline

| Script | Purpose | Inputs | Outputs |
|--------|---------|--------|---------|
| `00_prep_data.py` | Load H&M data, PCA embeddings, sample choice sets | Pre-trained embeddings | `data/{Y,T,X}.npy` |
| `01_train_embeddings.py` | Two-tower InfoNCE training (reference) | Raw H&M data | Embeddings |
| `02_fit_choice_model.py` | Custom MLE for reference comparison | `data/` | MLE estimates |
| `03_inference.py` | IF-corrected inference via `deep_inference` | `data/` | Results |
| `04_compare_methods.py` | Naive SE vs IF SE vs Bootstrap | `data/` | Comparison tables |
| `05_counterfactuals.py` | Pricing experiments with uncertainty | `data/` + results | Counterfactual tables |
| `06_simulation_study.py` | MC validation of IF coverage | None (DGP) | Coverage report |
| `07_generate_figures.py` | Paper figures | All results | `figures/` |

## Quick Start

```bash
# Step 1: Prepare data (requires pre-trained embeddings from deep-aesthetics)
python 00_prep_data.py

# Step 2: Run inference
python 03_inference.py

# Step 3: Compare methods
python 04_compare_methods.py

# Step 4: Simulation validation
python 06_simulation_study.py
```

## Prerequisites

```bash
pip install deep-inference numpy torch scikit-learn
```
