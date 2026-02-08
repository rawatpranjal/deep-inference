# Theory

Mathematical foundations of the Farrell-Liang-Misra framework for deep learning with valid inference.

```{toctree}
:maxdepth: 2
:caption: Theory

influence_functions
```

## Overview

This section explains the theoretical foundations of `deep-inference`, specifically the enriched structural model approach from Farrell, Liang, and Misra.

## Key References

- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*
- Hetzenecker, Osterhaus (2024): "Deep Learning for Heterogeneous Parameters in Discrete Choice Models" *arXiv 2408.09560*
- Momin (2025): "Heterogeneous Treatment Effects Using Deep Neural Networks" *SSRN 5149650*
- Chen, Liu, Ma, Zhang (2024): "Causal Inference of General Treatment Effects using Neural Networks" *Journal of Econometrics*
- Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022): "RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets" *ICML 2022*

## The Core Insight

**Machine learning and economic structure are complements, not substitutes.**

- **ML alone** fits data well but extrapolates nonsensically and can't answer causal questions
- **Structure alone** provides interpretability but misses heterogeneity
- **Combined**: ML learns heterogeneity patterns $\theta(X)$ while structure ensures valid economics

> "The central idea is that machine learning methods and economic structure are complements, not substitutes. Machine learning methods alone predict well, but extrapolate nonsensically... Economic structure alone can produce robust inference, but may miss important heterogeneity that is visible in the data."
> — Farrell, Liang, Misra (2021)
