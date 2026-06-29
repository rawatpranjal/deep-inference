# Theory

Mathematical foundations of the Farrell-Liang-Misra framework for deep learning with valid inference.

This section is a **linear walkthrough**. The eight phase pages below build on each other, so
read them top to bottom. Page 8 connects the influence-function correction to the Riesz
representer (and to the RieszNet procedure), and the final reference page collects the formal
theorems and convergence rates.

```{toctree}
:maxdepth: 1
:caption: Theory (read in order)

01_setup
02_targets
03_why_naive_fails
04_influence_function
05_three_regimes
06_multinomial
07_models_and_targets
08_riesz_representer
influence_functions
```

## Overview

This section explains the theoretical foundations of `deep-inference`, specifically the enriched structural model approach from Farrell, Liang, and Misra. It walks from the structural setup, through the failure of naive inference, to the influence function correction that restores valid confidence intervals, and closes with the formal guarantees.

## Key References

The papers behind this framework, with one-line annotations and links, are collected on the
[References](../references/index.md) page. The load-bearing ones for this section are
Farrell, Liang, Misra (2021, 2025) for the influence-function framework and Chernozhukov et al.
(2022) for the Riesz-representer view in the [last page](08_riesz_representer.md).

## The Core Insight

**Machine learning and economic structure are complements, not substitutes.**

- **ML alone** fits data well but extrapolates nonsensically and can't answer causal questions
- **Structure alone** provides interpretability but misses heterogeneity
- **Combined**: ML learns heterogeneity patterns $\theta(X)$ while structure ensures valid economics

> "The central idea is that machine learning methods and economic structure are complements, not substitutes. Machine learning methods alone predict well, but extrapolate nonsensically... Economic structure alone can produce robust inference, but may miss important heterogeneity that is visible in the data."
> — Farrell, Liang, Misra (2021)
