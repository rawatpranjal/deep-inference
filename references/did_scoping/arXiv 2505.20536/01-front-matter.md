<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2505.20536.pdf
backend: pdftotext
part: 1/3
-->

# Front Matter

<!-- pages: 1-30 -->

Covariate-Adjusted Deep Causal Learning for
                                                             Heterogeneous Panel Data Models


arXiv:2505.20536v1 [stat.ML] 26 May 2025
                                                              Guanhao Zhou, Yuefeng Han, and Xiufan Yu
                                                                           University of Notre Dame


                                                                                      Abstract

                                                    This paper studies the task of estimating heterogeneous treatment effects in causal
                                                panel data models, in the presence of covariate effects. We propose a novel Covariate-
                                                Adjusted DEep CAusal Learning (CoDEAL) for panel data models, that employs
                                                flexible model structures and powerful neural network architectures to cohesively deal
                                                with the underlying heterogeneity and nonlinearity of both panel units and covariate
                                                effects. The proposed CoDEAL integrates nonlinear covariate effect components (param-
                                                eterized by a feed-forward neural network) with nonlinear factor structures (modeled
                                                by a multi-output autoencoder) to form a heterogeneous causal panel model. The
                                                nonlinear covariate component offers a flexible framework for capturing the complex
                                                influences of covariates on outcomes. The nonlinear factor analysis enables CoDEAL to
                                                effectively capture both cross-sectional and temporal dependencies inherent in the data
                                                panel. This latent structural information is subsequently integrated into a customized
                                                matrix completion algorithm, thereby facilitating more accurate imputation of missing
                                                counterfactual outcomes. Moreover, the use of a multi-output autoencoder explicitly
                                                accounts for heterogeneity across units and enhances the model interpretability of the
                                                latent factors. We establish theoretical guarantees on the convergence of the estimated
                                                counterfactuals, and demonstrate the compelling performance of the proposed method
                                                using extensive simulation studies and a real data application.

                                           Keywords: causal panel data models, counterfactual estimation, heterogeneous treatment
                                           effects, matrix completion, missing not at random, multi-output autoencoders, nonlinear
                                           factor models


                                                                                          1

1     Introduction
    Causal inference in panel data settings has attracted growing interest in recent years, with
broad applications across various fields such as economics (Clarke et al., 2024), life science
(Helske and Tikka, 2024), political science (Imai and Kim, 2021), and social science (Imbens,
2024). Causal panel data analysis is inherently a missing data problem due to the fundamental
problem of causal inference: for each unit and time period, we could only observe either the
treated or the untreated outcome, but never both. The panel data structure brings unique
challenges to causal analysis. Different units may get exposed to the treatment at various time
periods, forming a block-wise non-random missing pattern (Athey and Imbens, 2022). The
presence of both temporal and cross-sectional dependencies could bias the estimation if not
properly addressed (Sun and Abraham, 2021), and unit-specific heterogeneity can confound
causal interpretation (Millimet and Bellemare, 2023), especially if additional covariates are
present.
    To estimate causal effects, one natural approach is to first impute the missing coun-
terfactual, and then examine the difference between the imputed counterfactuals and the
actual observed outcomes. Athey et al. (2021) formulated the imputation of missing coun-
terfactuals as a matrix completion task with nuclear norm minimization. It is important
to note that standard matrix completion approaches are not directly suitable for causal
imputation (Choi and Yuan, 2024). Most conventional matrix completion methods operate
under the assumption that missing entries are missing at random, which enables estimation
of missing values without explicitly modeling the missingness mechanism. However, in causal
panel data settings where different units receive treatment at different times, missing entries
correspond to counterfactual outcomes that are systematically unobserved due to treatment
assignments, invalidating the missing-at-random assumption. As a result, applying standard
matrix completion blindly without properly accounting for this structured, non-random
missingness can lead to biased estimation (Agarwal et al., 2023).
    In recent years, a growing body of work has focused on developing causal matrix completion
methods that handle missing-not-at-random panel designs. Besides the optimization-based
causal matrix completion (Athey et al., 2021; Choi and Yuan, 2024), another line of research


                                               2

leverages factor models (Bai, 2003, 2009) to capture the latent low-rank structure of observed
entries. Most factor-model-based causal matrix completion methods in the literature (Xu,
2017; Bai and Ng, 2021; Agarwal et al., 2023; Xiong and Pelger, 2023; Yan and Wainwright,
2024) posit a linear factor structure. Linear factor analysis helps capture cross-sectional
and temporal dependencies to some extent (Han et al., 2024b,a; Yu et al., 2024b), but is
often limited in capturing complex nonlinear relationships, motivating the generalization to
nonlinear factor models (Yalcin and Amemiya, 2001).
   Another key aspect in panel data analysis is to account for unit-specific heterogeneity
(Millimet and Bellemare, 2023; Semenova et al., 2023). Most relevant works in the literature
still focus on the overall average treatment effects across units (Athey et al., 2021; Athey
and Imbens, 2025), as opposed to the unit-specific heterogeneous treatment effects. Pooled
estimates may mask important variations across units. Units in panel data often differ in
characteristics and respond to the treatment differently (Wager and Athey, 2018; Nandy
et al., 2023), which may bias the estimates of treatment effects, motivating a critical need to
estimate unit-specific heterogeneous treatment effects and adjust for the covariate effects if
possible.
   In this paper, we develop Covariate-Adjusted DEep CAusal Learning (CoDEAL) for
panel data models, a deep-learning-based causal learning method for estimating heterogeneous
treatment effects. CoDEAL captures covariate effects with a deep neural network regression,
and models the covariate-adjusted outcomes via a nonlinear factor structure. Our contributions
can be summarized as follows.

• CoDEAL provides a causal matrix completion method that addresses the non-random
  missingness mechanism in causal panel data analysis. By leveraging nonlinear factor analysis,
  CoDEAL exploits both cross-sectional and temporal dependencies. This latent
  structural information is subsequently integrated into the matrix completion procedure,
  thereby facilitating more accurate imputation of missing counterfactual outcomes.

• CoDEAL employs flexible model structures and powerful neural network architectures to
  cohesively deal with the underlying heterogeneity and nonlinearity of both panel
  units and covariate effects. Algorithmically, we begin with a feed-forward deep neural


                                              3

  network (DNN) to flexibly remove nonlinear covariate influences, followed by a multi-output
  autoencoder (AE) to recover nonlinear latent factors and explicitly take the unit-specific
  heterogeneity into account.

• CoDEAL offers a unifying framework for heterogeneous causal panel models in the
  presence of covariates. By combining nonlinear covariate adjustment with deep latent
  factor models, CoDEAL generalizes beyond traditional linear factor models and unifies a
  class of factor-model-based causal matrix completion approaches, e.g., Xu (2017); Bai and
  Ng (2021); Agarwal et al. (2023); Xiong and Pelger (2023); Yan and Wainwright (2024).
  Furthermore, with different choices of factor models and neural network architectures, the
  proposed framework can be further generalized to accommodate various data types such as
  spatial and tensor data.

Related Work. (a) An overview of causal learning methods in panel data models.
One traditional approach to estimate treatment effects in causal panel models is the difference-
in-differences (Imai and Kim, 2021; Athey and Imbens, 2022; Wing et al., 2024) that compares
the average changes in outcomes over time between the treated and untreated units under a
parallel trend assumption. Alternatively, one can estimate the causal effects by first imputing
the missing counterfactual outcomes, and then examining the difference between the imputed
counterfactuals and the actual observed outcomes. Following this strand, two approaches
have emerged in recent causal panel literature: the uncounfoundedness-based approach (also
known as horizontal regression) (Rosenbaum and Rubin, 1983; Imbens and Rubin, 2015) and
the synthetic control method (also known as vertical regression) (Abadie and Gardeazabal,
2003; Abadie et al., 2010, 2015; Doudchenko and Imbens, 2016; Abadie, 2021). The close
connections between the difference-in-differences, unconfoundedness-based methods, synthetic
control approaches, and causal matrix completion estimators were revealed by Athey et al.
(2021). Despite their distinct appearances, Athey et al. (2021) showed that the linear versions
of all four estimators can be characterized as solutions to the same optimization problem
with the exact same objective function, but subject to different restrictions on parameters of
that optimization. We refer readers to Arkhangelsky and Imbens (2024) for a comprehensive
review of past and recent progress on causal panel models.


                                               4

(b) Causal matrix completion and deep matrix completion. A surge of interest in
the recent past has led to new research on causal matrix completion methods capable of
addressing missing-not-at-random panel data, e.g., optimization-based methods (Athey et al.,
2021; Choi and Yuan, 2024) and factor-model-based methods (Xu, 2017; Bai and Ng, 2021;
Agarwal et al., 2023; Xiong and Pelger, 2023; Yan and Wainwright, 2024). There has been a
handful of deep-learning-based methods for matrix completion, e.g., Fan and Cheng (2018);
Radhakrishnan et al. (2022); Xiu and Shen (2024); Fan et al. (2024). However, most of them
assume missing-at-random, and therefore, not directly suitable for causal imputation in panel
data models.


2       Methodology
2.1      Problem Setup

Heterogeneous Treatment Effects 1 in Causal Panel Data Models. We begin by
setting up the causal learning problem in panel data. Consider a panel data setting with N
units over T periods. Let [n] denote the set {1, 2, . . . , n}. For each unit i ∈ [N ] and time point
t ∈ [T ], there are two potential outcomes, Yit (1) and Yit (0), representing the outcomes the
i-th unit would experience at time t under treatment and control, respectively. The treatment
assignment is captured by a binary indicator variable Wit ∈ {0, 1}, where Wit = 1 indicates
that the unit i receives the treatment at time t and Wit = 0 otherwise. Suppose we also have
access to some unit-specific covariates Xi ∈ RP that could potentially affect the observations
{Yit , t ∈ [T ]}. We denote the matrix forms of the observed outcomes, treatment indicators,
and covariates by Y = (Yit ) ∈ RN ×T , Ω = (Wit ) ∈ RN ×T , and X = (X1 , . . . , XN )⊤ ∈ RN ×P ,
respectively. Let Y(1) = (Yit (1)) ∈ RN ×T and Y(0) = (Yit (0)) ∈ RN ×T be the matrices of
potential outcomes of the treated and untreated, respectively. Our aim2 in this work is to
estimate the unit-specific (potentially heterogeneous) average treatment effects on the treated
(ATT), τi⋆ = Et [Yit (1) − Yit (0) | Wit = 1] for each treated unit i.
    1
     Here, the “heterogeneous treatment effect” refers to the heterogeneity across units.
    2
     Aligning with relevant works in the literature (Athey et al., 2021; Athey and Imbens, 2025), we focus
on the unit-specific average treatment effects on the treated (ATT) as our primary estimand of interest to
demonstrate our proposed model and algorithm. The proposed method can be also be used to many other
estimands, such as the overall (non-unit-specific) ATT E(i,t) [Yit (1) − Yit (0) | Wit = 1], the unit-specific average
treatment effects (ATE) Et [Yit (1) − Yit (0)], and the overall (non-unit-specific) ATE E(i,t) [Yit (1) − Yit (0)].


                                                          5

Staggered Adoption Design. This work focuses on staggered adoption design (Athey
and Imbens, 2022), where treatments can be rolled out at different times across various units
and are irreversible once initiated. We assume there is at least one never-treated unit in the
panel. Details about staggered design are in Appendix A.

Causal Assumptions. To ensure the validity of the estimators’ interpretation, we adopt the
following assumptions throughout the analysis. (i) (Stable Unit Treatment Value Assumption
(SUTVA).) The potential outcomes for a given unit at a particular time depend solely on the
treatment administered to that unit at that time, and are unaffected by treatments assigned
to other units or at other time points. (ii) (Static Treatment Effects.) The treatment effects
are non-dynamic, i.e., do not change over time.

Challenges and Limitations of Existing Methods. The fundamental problem of causal
inference implies that Yit (0) and Yit (1) cannot be observed simultaneously. The panel data
forms two partially observed data matrices Y(0) and Y(1), where the missing pattern is
determined according to the treatment assignments. To estimate the treatment effects, one
natural idea is to estimate the unobserved counterfactuals, in other words, to impute the
missing potential outcomes. Since our focus is the unit-specific ATT and {Yit (1) : Wit = 1}
is observed, it suffices to estimate {Yit (0) : Wit = 1} to impute missing elements in Y(0). Let
{Ybit (0) : Wit = 1} denote imputed values. The unit-specific ATT can then be estimated by
τbi = t:Wit =1 [Yit (1) − Ybit (0)]/ t Wit for each treated unit i.
       P                            P

   There are three main limitations of existing methods that this work aims to overcome in
the task of modeling unit-specific ATT and imputing missing potential outcomes. First, the
vast majority of matrix completion methods rely on the assumption of missing at random, an
assumption that is typically violated in staggered adoption designs, thereby leading to biased
estimates. Second, existing matrix-completion-based causal panel models mostly focus on the
overall (non-unit-specific) ATT, without adequately accounting for unit-level heterogeneity.
Third, many existing methods often lack the capacity to disentangle the causal impact of
treatment from confounding covariate effects and to model the underlying complex nonlinear
dependencies, limiting both the precision of counterfactual predictions and the interpretability
of causal analysis.

                                               6

2.2    Covariate-Adjusted Deep Causal Learning (CoDEAL) for Panel
       Data Models

   To tackle the above challenges, we propose the CoDEAL method that (i) leverages the
staggered adoption structure to fully exploit both cross-sectional and temporal dependencies
for more accurate imputation of missing potential outcomes, (ii) models the unit-specific
heterogeneity explicitly to accommodate variation in treatment effects across units, and
(iii) involves flexible model structures and employs powerful neural network architectures to
efficiently capture complex nonlinear relationships.
   For each unit i ∈ [N ] at time t ∈ [T ], we model the observed outcome by

                       Yit = ϕ⋆i (F⋆t ) + gt⋆ (Xi ) + τi⋆ · 1{Wit = 1} + εit ,            (1)

where F⋆t ∈ RK is a K-dimension vector of latent factors, ϕ⋆i (·) : RK → R is a potentially
nonlinear factor loading function, gt⋆ (·) : RP → R is a potentially nonlinear function that
captures how covariates influence the outcome, and 1{·} is an indicator function, and εit is
the idiosyncratic noise. CoDEAL employs a nonlinear factor model (Yalcin and Amemiya,
2001) to characterize an underlying low-rank structure of the covariate-adjusted outcomes.
By letting ϕ⋆i (·) take different forms, CoDEAL unifies a class of factor-model-based causal
matrix completion approaches, e.g., (Xu, 2017; Bai and Ng, 2021; Agarwal et al., 2023; Xiong
and Pelger, 2023; Yan and Wainwright, 2024). As shown by (1), CoDEAL accommodates
heterogeneous but non-dynamic treatment effects. The covariate effects can be heterogeneous
across units and time-varying over periods. Here, we allow N and T to be large and diverging,
while P and K are fixed and small.
   For clarity, we first present the learning procedures using the simplest form of a 2 × 2
staggered adoption (also known as the four-block design) in Section 2.2.1, and generalize the
algorithm to the more general staggered adoption design in Section 2.2.2. More details about
the four-block and staggered adoption design as well as their graphical illustrations are in
