# Theory

Mathematical foundations of the Farrell-Liang-Misra framework for deep learning with valid inference.

This section is a **linear walkthrough** — the seven phase pages below build on
each other, so read them top to bottom. The final page collects the formal
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
influence_functions
```

## Overview

This section explains the theoretical foundations of `deep-inference`, specifically the enriched structural model approach from Farrell, Liang, and Misra. It walks from the structural setup, through the failure of naive inference, to the influence function correction that restores valid confidence intervals, and closes with the formal guarantees.

## Key References

### Core Framework
- Farrell, Liang, Misra (2021): "Deep Neural Networks for Estimation and Inference" *Econometrica*
- Farrell, Liang, Misra (2025): "Deep Learning for Individual Heterogeneity" *Working Paper*

### Applications
- Dubé, Misra (2023): "Personalized Pricing and Consumer Welfare" *Journal of Political Economy*
- Hetzenecker, Osterhaus (2024): "Deep Learning for Heterogeneous Parameters in Discrete Choice Models" *arXiv 2408.09560*
- Colangelo, Lee (2026): "Double Debiased ML Nonparametric Inference with Continuous Treatments" *JBES*
- Momin (2025): "Heterogeneous Treatment Effects and Counterfactual Policy Targeting Using DNNs" *SSRN 5149650*
- Chen, Liu, Ma, Zhang (2024): "Causal Inference of General Treatment Effects using Neural Networks" *Journal of Econometrics*
- Ye, Zhang, Zhang, Zhang, Zhang (2025): "Deep-Learning-Based Causal Inference for Large-Scale Combinatorial Experiments" *Management Science*

### Automatic Debiasing / Riesz Representation
- Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022): "RieszNet and ForestRiesz" *ICML*
- Chernozhukov, Newey, Singh (2022): "Automatic Debiased Machine Learning of Causal and Structural Effects" *Econometrica*
- Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2021): "Automatic Debiased ML via Neural Nets for GLR" *Working Paper*
- Hines, Hines (2025): "Automatic Debiasing of Neural Networks via Moment-Constrained Learning" *CLeaR*

### DNN Architecture + Influence Functions
- Shi, Blei, Veitch (2019): "Adapting Neural Networks for the Estimation of Treatment Effects (DragonNet)" *NeurIPS*
- Li, McCoy et al. (2025): "Targeted Deep Architectures for Estimation and Inference" *arXiv 2507.12435*
- Shirakawa et al. (2024): "Deep Longitudinal Targeted Minimum Loss-based Estimation" *ICML*
- Liu et al. (2024): "DNA-SE: Towards Deep Neural-Nets Assisted Semiparametric Estimation" *ICML*
- Cai, Fonseca, Hou, Namkoong (2025): "C-Learner: Constrained Learning for Causal Inference" *arXiv 2405.09493*

### Theory
- Yan, Chen, Yao (2025): "Overparameterized Neural Networks in Semiparametric Inference" *arXiv 2504.19089*
- Metzger (2022): "Adversarial Estimators" *arXiv 2204.10495*
- Foster, Syrgkanis (2023): "Orthogonal Statistical Learning" *Annals of Statistics*

### Frontier
- Melnychuk, Feuerriegel (2026): "GDR-Learners: Generalized Doubly Robust Learners" *ICLR*
- Nguyen (2025): "Neural Network Estimation and Simulation for DDC Models" *Georgetown JMP*

## The Core Insight

**Machine learning and economic structure are complements, not substitutes.**

- **ML alone** fits data well but extrapolates nonsensically and can't answer causal questions
- **Structure alone** provides interpretability but misses heterogeneity
- **Combined**: ML learns heterogeneity patterns $\theta(X)$ while structure ensures valid economics

> "The central idea is that machine learning methods and economic structure are complements, not substitutes. Machine learning methods alone predict well, but extrapolate nonsensically... Economic structure alone can produce robust inference, but may miss important heterogeneity that is visible in the data."
> — Farrell, Liang, Misra (2021)
