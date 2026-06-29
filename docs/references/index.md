# References

The papers behind the package, grouped by role, each with a one-line note on why it matters
here. The links go to the transcribed markdown in the repository.

## Core framework

- Farrell, Liang, Misra (2021), "Deep Neural Networks for Estimation and Inference", *Econometrica*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/FLM2021_docling.md). The foundational result: neural-network plug-ins are biased, the influence-function correction restores valid inference.
- Farrell, Liang, Misra (2025), "Deep Learning for Individual Heterogeneity", working paper. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/FLM2025_docling.md). The extended theory this package follows for the structural-parameter setup.

## Applications

- Dubé, Misra (2023), "Personalized Pricing and Consumer Welfare", *Journal of Political Economy*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/DM2023_docling.md). Heterogeneous price elasticity and the welfare targets (elasticity, profit, consumer surplus).
- Hetzenecker, Osterhaus (2024), "Deep Learning for Heterogeneous Parameters in Discrete Choice Models", arXiv 2408.09560. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/HO2024_multinomial_docling.md). The multinomial-logit (conditional logit) setup.
- Colangelo, Lee (2026), "Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments", *JBES*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/CL2026_docling.md). Continuous-treatment dose-response.
- Chen, Liu, Ma, Zhang (2024), "Causal Inference of General Treatment Effects using Neural Networks", *Journal of Econometrics*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/CLMZ2024_docling.md).
- Ye et al. (2025), "Deep-Learning-Based Causal Inference for Large-Scale Combinatorial Experiments", *Management Science*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/DeDL2025_docling.md). The combinatorial multi-treatment model.

## Automatic debiasing and Riesz representation

- Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2022), "RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests", *ICML*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/RieszNet2022_docling.md). The basis for the [RieszNet procedure](../inference/riesznet.md).
- Chernozhukov, Newey, Singh, Syrgkanis (2022), "Automatic Debiased Machine Learning of Causal and Structural Effects", *Econometrica*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/CNSS2020_adversarial_riesz_docling.md).
- Chernozhukov, Newey, Quintas-Martinez, Syrgkanis (2021), "Automatic Debiased ML via Neural Nets for Generalized Linear Regression", working paper. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/IN_glr_docling.md).
- Hines, Hines (2025), "Automatic Debiasing of Neural Networks via Moment-Constrained Learning", *CLeaR*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/HH2025_docling.md).

## DNN architecture and influence functions

- Shi, Blei, Veitch (2019), "Adapting Neural Networks for the Estimation of Treatment Effects (DragonNet)", *NeurIPS*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/Dragonnet2019_docling.md).
- Li, McCoy et al. (2025), "Targeted Deep Architectures for Estimation and Inference", arXiv 2507.12435. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/TDA2025_docling.md).
- Shirakawa et al. (2024), "Deep Longitudinal Targeted Minimum Loss-based Estimation", *ICML*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/DeepLTMLE2024_docling.md).
- Liu et al. (2024), "DNA-SE: Towards Deep Neural-Nets Assisted Semiparametric Estimation", *ICML*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/DNASE2024_docling.md).
- Cai, Fonseca, Hou, Namkoong (2025), "C-Learner: Constrained Learning for Causal Inference and Semiparametric Statistics", arXiv 2405.09493. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/CLearner2025_docling.md).

## Theory

- Yan, Chen, Yao (2025), "Overparameterized Neural Networks in Semiparametric Inference", arXiv 2504.19089. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/YCY2025_docling.md).
- Metzger (2022), "Adversarial Estimators", arXiv 2204.10495. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/Metzger2022_docling.md).
- Foster, Syrgkanis (2023), "Orthogonal Statistical Learning", *Annals of Statistics*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/FS2023_docling.md).

## Frontier

- Melnychuk, Feuerriegel (2026), "GDR-Learners: Generalized Doubly Robust Learners for Causal Inference", *ICLR*. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/GDR2026_docling.md).
- Nguyen (2025), "Neural Network Estimation and Simulation for Dynamic Discrete Choice Models", Georgetown JMP. [transcript](https://github.com/rawatpranjal/deep-inference/blob/main/references/NNES2025_docling.md).
