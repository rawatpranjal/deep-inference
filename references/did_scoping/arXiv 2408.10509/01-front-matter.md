<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2408.10509.pdf
backend: pdftotext
part: 1/6
-->

# Front Matter

<!-- pages: 1-50 -->

Continuous difference-in-differences with double/debiased machine
                                                                                              learning

                                                                                    Lucas Zheng Zhang ∗


arXiv:2408.10509v6 [econ.EM] 12 Dec 2025
                                                                University of California, Los Angeles, Dept. of Economics;
                                                                            Bates White Economic Consulting

                                                                                         lucaszz@g.ucla.edu

                                                                                This Version: December 16, 2025


                                                                                                Abstract

                                                     This paper extends difference-in-differences to settings with continuous treatments. Specif-
                                                 ically, the average treatment effect on the treated (ATT) at any level of treatment intensity is
                                                 identified under a conditional parallel trends assumption. Estimating the ATT in this framework
                                                 requires first estimating infinite-dimensional nuisance parameters, particularly the conditional
                                                 density of the continuous treatment, which can introduce substantial bias. To address this
                                                 challenge, we propose estimators for the causal parameters under the double/debiased machine
                                                 learning framework and establish their asymptotic normality. Additionally, we provide consis-
                                                 tent variance estimators and construct uniform confidence bands based on a multiplier bootstrap
                                                 procedure. To demonstrate the effectiveness of our approach, we revisit a previous study on
                                                 the 1983 Medicare Prospective Payment System reform, reframing it as a DiD with continuous
                                                 treatment and non-parametrically estimating its effects.
                                                     Keywords: Difference-in-differences, causal inference, continuous treatment, machine learn-
                                                 ing


                                              ∗
                                                The author would like to express his gratitude to Denis Chetverikov, Andres Santos, and Rosa Matzkin for
                                           their generous time and extremely helpful discussions, which have led to substantial improvements to this paper.
                                           Additionally, he extends his thanks to Jinyong Hahn, Zhipeng Liao, Shuyang Sheng, and participants at the UCLA
                                           econometrics proseminars for their valuable comments and suggestions. Furthermore, the author is grateful to
                                           Kathleen McGarry, Daron Acemoglu, Amy Finkelstein, and the National Bureau of Economic Research for facilitating
                                           access to the data source in Acemoglu and Finkelstein (2008). Finally, the author thanks the editor and the anonymous
                                           referees for their thorough review and constructive feedback.


                                                                                                    1

1    Introduction

Difference-in-differences (DiD) is one of the most widely used research designs in empirical work.
While conventional DiD settings typically focus on binary or discrete multi-valued treatments, there
is growing interest in extending DiD to continuous treatments. The motivation for continuous DiD
is clear: the treatment group rarely receives interventions at a constant level, and treatment effects
can vary with the intensity or “dose” of the treatment. Thus, rather than comparing treated and
control groups before and after an intervention at an aggregate level, one can further investigate
how outcomes vary across different treatment intensities within the treated group.

    Continuous treatments are prevalent in many empirical settings. For instance, individuals may
experience varying levels of exposure to policy interventions, marketing campaigns, or environmen-
tal pollutants, all of which can be modeled as continuous treatments. Several recent studies have
explored DiD with continuous treatments, including Zeng et al. (2022) on the impact of shutting
down online advertising sites, Cook et al. (2023) on racial discrimination in public accommodations,
and Ananat et al. (2022) on the effects of the expanded child tax credit.

    Despite its widespread use in empirical research, the theoretical foundation for continuous DiD
remains relatively underdeveloped, particularly in comparison to the extensive body of literature on
DiD with binary or discrete treatments (see Roth et al. (2023), de Chaisemartin and D’Haultfoeuille
(2023), Callaway (2023) for recent overviews). A few recent studies have begun to bridge this gap,
notably de Chaisemartin et al. (2022), D’Haultfoeuille et al. (2023), and Callaway et al. (2024). For
instance, D’Haultfoeuille et al. (2023) extend the change-in-changes model of Athey and Imbens
(2006) to accommodate continuous treatments, while de Chaisemartin et al. (2022) examine the
average slope of stayers in the continuous DiD setting. Our paper is closely related to Callaway et al.
(2024), which studies continuous DiD in the commonly used two-way fixed effect (TWFE) regression
framework. Callaway et al. (2024) demonstrate that, under TWFE, the regression parameter of
interest can be decomposed as weighted integrals of either the average treatment parameters across
treatment intensities with potentially negative weights or average causal responses with selection
bias but nonnegative weights. They also provide data-driven non-parametric estimators for these
causal parameters that are rate optimal.

    In this paper, we focus on the average treatment effect on the treated (ATT) for any given
continuous treatment intensity. Although this parameter is one of several investigated in Callaway
et al. (2024), our primary contribution is to incorporate covariates non-parametrically into both
the identification and estimation procedures. Specifically, we modify the parallel trends assumption
in Callaway et al. (2024) by conditioning on covariates in a manner analogous to the “conditional
parallel trends” assumption used in DiD for binary or discrete treatments; see Heckman et al. (1997,
1998), Abadie (2005), Chang (2020), and Sant’Anna and Zhao (2020), for example. As noted in


                                                  2

Abadie (2005), an unconditional parallel trends assumption can be restrictive if covariates that
influence outcome dynamics have different distributions across treatment and control groups. By
conditioning on such covariates, we obtain a more robust framework for identifying and estimating
the ATT in continuous treatment settings.

   We first establish identification results analogous to those in Abadie (2005), adapted to the
continuous treatment setting. Based on these identification results, a naive estimator for the ATT
can be constructed in two steps. First, one estimates several nuisance parameters from the identi-
fication results, including the conditional density of the continuous treatment. In the second step,
the nuisance estimates are substituted into a simple average to obtain the estimator of the causal
parameter. However, for potentially high-dimensional controls, while one may employ machine
learning methods to estimate the nuisance parameters, doing so can introduce substantial bias
in the causal parameter estimation (see Chernozhukov et al. (2018) and the references therein).
Moreover, reusing the same sample for both nuisance and causal parameter estimation can result
in additional overfitting bias. To address these concerns, we adopt the double/debiased machine
learning (DML) framework studied in Chernozhukov et al. (2018), which uses orthogonalization
and cross-fitting to reduce the influence of nuisance parameter estimation on causal estimates.

   Previous studies have adopted similar strategies in related settings. For instance, Chang (2020)
considers the DML framework for DiD with binary or discrete treatments, and Sant’Anna and Zhao
(2020) proposes efficient doubly robust estimators for DiD with binary treatment. We contribute to
and extend this literature to the continuous treatment setting. In particular, in place of the usual
propensity score for the treated group, our setting requires the conditional density of the continuous
treatment, which poses additional difficulties for directly applying DML methods, often involving
only conditional mean functions as the nuisance parameters. To circumvent this, we introduce
an approximate causal parameter AT Th using a kernel function. As the kernel bandwidth shrinks,
AT Th converges to the true AT T . Importantly, by focusing on AT Th , we can replace the conditional
density with a conditional mean, which allows us to apply the existing DML results. We then derive
orthogonal scores for both panel and repeated cross-sectional cases and construct corresponding
DML estimators. Building on Chernozhukov et al. (2014a,b), Chernozhukov et al. (2018), and Fan et
al. (2022), we establish the asymptotic normality of these estimators and show that the asymptotic
bias becomes negligible under an appropriate undersmoothing kernel bandwidth. Additionally, we
provide consistent variance estimators via cross-fitting and develop uniform confidence bands for
the treatment curve using a multiplier bootstrap procedure. The results from our carefully designed
simulation studies suggest that our estimators perform well.

   To illustrate the usefulness of our method, we revisit Acemoglu and Finkelstein (2008), which
examines the impact of the 1983 Medicare payment system (PPS) reform on the healthcare industry.
Since the PPS reform affected hospitals with varying proportions of Medicare inpatients differently,


                                                  3

the share of Medicare inpatients can be interpreted as a continuous treatment variable. This makes
Acemoglu and Finkelstein (2008) an exemplary case for applying our methods. Thus, we non-
parametrically estimate the ATTs of the PPS reform in a continuous DiD context, providing a
more detailed understanding of the effects of this policy reform. In particular, contrasting with the
linear estimates from Acemoglu and Finkelstein (2008), our results suggest significant heterogeneity
in the impact of the PPS reform across hospitals with different shares of Medicare inpatients.

    We note that the kernel smoothing has been previously considered in the causal inference lit-
erature with continuous treatment. For example, Kennedy et al. (2017) studies average potential
outcomes under a continuous treatment, proposing a doubly robust signal and a two-step estima-
tion procedure involving a pseudo-outcome and local kernel linear regression. Along similar lines,
Semenova and Chernozhukov (2021) employs series methods to establish uniform asymptotic re-
sults. Hettinger et al. (2025) recently adopted a similar framework as Kennedy et al. (2017) to
establish identification and estimation results on the average dose effect on treated. This causal
parameter differs from ours in that it relies on different sets of parallel trends assumptions and it is
an average dose-response on the entire treated group, akin to the average potential outcome. It is
important to emphasize that while Kennedy et al. (2017) and Hettinger et al. (2025) also employ
the kernel techniques, their approach differs from ours in non-trivial ways. We use kernels primarily
to approximate the original causal parameters, facilitating the construction of orthogonal scores,
after which the final estimation proceeds as a simple average; see Bibaut and van der Laan (2017)
for a more general discussion on this method. This contrasts with their approach, which uses kernel
regressions to estimate the conditional mean of a pseudo-outcome. In this respect, our work is also
related to Kallus and Zhou (2018), Su et al. (2019), and Colangelo and Lee (2025), all of which
consider continuous treatments and employ kernel-based moment functions to study the average
potential outcomes and partial effects.

    The remainder of this paper is organized as follows. Section 2 introduces continuous DiD
and demonstrates the identification of the causal parameter. Section 3 provides the orthogonal
scores. In Section 4, we present our estimators and establish their asymptotic properties. Section
5 showcases the simulation results, followed by a detailed empirical example in Section 6. Section
7 concludes.


2    Setup and Identification

In this section, we formally set up the difference-in-differences with continuous treatment following
Abadie (2005) and Callaway et al. (2024). First, using the potential outcome notation (e.g. Rubin
(1974)), let Yi,t (0) denote the potential outcome of individual i in period t when receiving no


                                                   4

treatment, and similarly let Yi,t (d) denote the potential outcome of individual i in period t when
receiving treatment with intensity d.

    The treatment variable D is modeled as a random variable with a mixture distribution: a
probability mass at 0 and a continuous distribution on an interval [dL , dH ] excluding 0. Specifically,
the control group consists of individuals who receive treatment D = 0, and we need a relatively large
number of individuals in the control group so that the comparison with the treated is meaningful.
On the other hand, the treated individuals can receive varied treatments, each with a potentially
different treatment dose/intensity D = d ∈ [dL , dH ]. We restrict our attention to the two-period
(t − 1, t) models and suppress the time notation in treatment Di in the panel setting. Let Xi denote
the set of individual-level covariates. We make the following assumptions:

Assumption 2.1 (Panel). The observed data {Yi,t−1 , Yi,t , Di , Xi }N
                                                                    i=1 are independently and iden-
tically distributed.

Assumption 2.2 (Repeated Cross-Sections). (a) For each individual i in the pooled sample, Ti is
a time indicator = 1 if observation i belongs to the post-treatment sample and = 0 otherwise, and
Yi = (1 − Ti )Yi,t−1 + Ti Yi,t ; (b) (D, X) ⊥ T and the following holds: (i) conditional on T = 0, data
are i.i.d. from the distribution of (Yt−1 , D, X); (ii) conditional on T = 1, data are i.i.d. from the
distribution of (Yt , D, X).

Assumption 2.3 (Support). (a) The support of D is {0} ⊔ [dL , dH ] with 0 < dL < dH < ∞;
(b) there exists a constant 0 < κ < 12 such that, almost surely, κ < P (D = 0|X) < 1 − κ and
fD|X (d|X) > κ for all d ∈ [dL , dH ].

Assumption 2.4 (No Anticipation). Yt = Yt (D), Yt−1 = Yt−1 (0).

Assumption 2.5 (Conditional Parallel Trends). For all d ∈ [dL , dH ], the following holds

                       E[Yt (0) − Yt−1 (0)|X, D = d] = E[Yt (0) − Yt−1 (0)|X, D = 0].                          (2.1)


    Assumptions 2.1 and 2.2 are analogous to those in the DiD literature with a discrete treatment.
While Assumption 2.1 requires a balanced panel, Assumption 2.2 allows for repeated cross-sections
but imposes stationarity of (D, X) and hence rules out compositional changes.1 Assumption 2.3 is
the strong overlap assumption, ensuring sufficient support for both treated and untreated individ-
uals, which is crucial for identification. Assumption 2.4 formalizes the requirement that there is no
anticipated treatment effect prior to the treatment. Assumption 2.5, a generalization of the discrete
case in Heckman et al. (1997, 1998), is the key identifying condition for the causal parameter. This
assumption essentially states that, conditional on covariates, the unobserved counterfactual trend
   1
     For DiD with compositional changes, see Hong (2013), Zimmert (2020), and Sant’Anna and Zhao (2025) for
detailed discussions in the discrete treatment setting, and Haddad et al. (2024) in the continuous treatment setting.


                                                         5

of the treated at each given treatment intensity is the same as the observed trend of the control
group.

   Next, we describe our target parameter. The causal parameter we are interested in is the average
treatment effect on the treated (ATT for short) at any given treatment intensity d ∈ [dL , dH ]:

                                 AT T (d) := E[Yt (d) − Yt (0)|D = d].                            (2.2)

The interpretation of this parameter is analogous to the cases with discrete treatment variables:
the expected effect of treatment with intensity d for those who actually received treatment with
intensity d. See also Callaway et al. (2024) Section 3 for a comprehensive discussion on (2.2) and
an alternative parallel trends assumption under which the average treatment effect AT E(d) :=
E[Yt (d) − Yt (0)] can be identified. The following theorem presents the main results of this section,
in which we establish the identification of AT T (d) for both panel and repeated cross-sectional
settings.

Theorem 2.1 (Identification of ATT). (a) (Panel) If Assumptions 2.1, 2.3, 2.4, and 2.5 hold,
then, for any d ∈ [dL , dH ],

                                                                             fD|X (d|X)
                                                                                          
            AT T (d) = E[Yt − Yt−1 |D = d] − E (Yt − Yt−1 )1{D = 0}                          ;    (2.3)
                                                                         fD (d)P (D = 0|X)

(b) (Repeated Cross-Sections) if Assumptions 2.2, 2.3, 2.4, and 2.5 hold, then, for any d ∈ [dL , dH ],

                                                                      fD|X (d|X)
                                                                                 
                          T −λ                 T −λ
            AT T (d) = E          Y D =d −E            Y 1{D = 0}                                 (2.4)
                         λ(1 − λ)             λ(1 − λ)            fD (d)P (D = 0|X)

where λ := P (T = 1).

   With Theorem 2.1, one can build estimators for AT T (d) using the estimated sample analogs. For
potentially high-dimensional covariates, machine learning methods can be employed to estimate the
nuisance parameters, including the conditional density fD|X (d|X) and the conditional probability
P (D = 0|X). However, the use of machine learning methods can often result in non-trivial first-
order biases in the estimation of the causal parameter, see e.g. Chernozhukov et al. (2018) and
references therein for a detailed discussion. Therefore, we consider alternative estimating equations
that reduce the influence of the nuisance parameters.


                                                  6

3       Orthogonal Scores

In this section, we focus on the panel case for illustration as the repeated cross-sectional case only
requires minor modifications. We begin by introducing Neyman orthogonality. Let θ0 (d) ∈ Θ ⊂ R
be the low-dimensional parameter of interest, e.g., AT T (d), and let ρ0 (d) ∈ H(d) denote the true
low-dimensional nuisance parameters, e.g., ρ0 (d) = fD (d). The true infinite-dimensional nuisance
parameters η0 (d) ∈ T (d) include fD|X (d|X) and P (D = 0|X) with the estimated η̂(d) in the
realization set TN (d) ⊂ T (d) with high probability.2 Let Z be the observable random vector,
e.g. Z = (Yt−1 , Yt , D, X) in the panel setting, and let ψ : (Z, θ(d), ρ(d), η(d)) 7→ R denote a
score function.3 With these notations, following Chernozhukov et al. (2018) and Chang (2020),
we formally define the Neyman orthogonality with respect to the infinite-dimensional nuisance
parameters.

Definition 1 (Neyman Orthogonality). A score ψ satisfies the Neyman orthogonality condition at
(θ0 (d), ρ0 (d), η0 (d)) with respect to a nuisance realization set TN (d) ⊂ T (d) if (a) θ0 (d) satisfies the
moment condition

                                       EP [ψ(Z, θ0 (d), ρ0 (d), η0 (d))] = 0;                                   (3.1)

(b) for r ∈ [0, 1) and η(d) ∈ TN (d), the Gateaux (directional) derivative satisfies

                         ∂r EP [ψ(Z, θ0 (d), ρ0 (d), η0 (d) + r(η(d) − η0 (d)))]|r=0 = 0.                       (3.2)


    In the above definition, (a) says that ψ identifies the parameter of interests while (b) ensures
the first-order bias from estimating the infinite-dimensional nuisance parameters is zero. Recall
that in the panel case,

                                                                                  fD|X (d|X)
                                                                                               
               θ0 (d) = AT T (d) = E[∆Y |D = d] − E ∆Y 1{D = 0}                                   .             (3.3)
                                                                              fD (d)P (D = 0|X)

where ∆Y := Yt − Yt−1 . First, given the continuous nature of the treatment intensity, θ0 (d) cannot
be estimated non-parametrically at root-N rate. This relates to a class of non-regular parameters
involving continuous treatment variables; see Galvao and Wang (2015), Kennedy et al. (2017), Su et
al. (2019), Semenova and Chernozhukov (2021), Fan et al. (2022), and Colangelo and Lee (2025) for
example. Moreover, a score based on the above expression does not satisfy Neyman orthogonality,
and an adjustment term has to be added.
    2
      New infinite-dimensional nuisance parameters can arise when constructing the orthogonal scores. We also ex-
plicitly index the nuisance parameters and nuisance function spaces by treatment intensity d.
    3
      We say ψ is a score function if at the true nuisance parameters (ρ0 (d), η0 (d)) and the true θ0 (d), the moment
condition E[ψ(Z, θ0 (d), ρ0 (d), η0 (d)] = 0 holds.


                                                          7

   To this end, we approximate the non-regular AT T (d) with a family of smoothed regular pa-
rameters that are tractable. We note that this approach has been discussed extensively in Bibaut
and van der Laan (2017) and Colangelo and Lee (2025), and specifically we rely on the following
observation (e.g., Fan et al. (1996)):

                                                                             1 u
                    fD|X (d|x) = lim E[Kh (D − d)|X = x],        Kh (u) :=     K               (3.4)
                                 h→0                                         h   h

where K(·) is a kernel function. Replacing E[∆Y |D = d] and fD|X (d|x) by their kernel counterparts,
we can define AT Th (d) as follows:
                                                                           
                              Kh (D − d)                     E[Kh (D − d)|X]
             AT Th (d) :=E ∆Y              − E ∆Y 1{D = 0}
                                fD (d)                      fD (d)P (D = 0|X)
                                                                               
                              Kh (D − d)P (D = 0|X) − 1{D = 0}E[Kh (D − d)|X]
                        =E ∆Y                                                     ,            (3.5)
                                              fD (d)P (D = 0|X)

which is an expression that consists of only conditional expectations. Notably, it can be shown that

                                         AT T (d) = lim AT Th (d),
                                                    h→0

which suggests that we can work with AT Th (d) instead. In particular, define the bias Bh (d) :=
AT Td − AT Th (d), one can show that Bh (d) = O(h2 ), and we defer the formal result to the next
section. For notation simplicity, we now formally define AT Th (d) in both settings.

Definition 2 (Panel).
                                                                                      
                              Kh (D − d)P (D = 0|X) − 1{D = 0}E[Kh (D − d)|X]
             AT Th (d) = E ∆Y                                                                  (3.6)
                                              fD (d)P (D = 0|X)

where ∆Y = Yt − Yt−1 .

Definition 3 (Repeated Cross-Sections).
                                                                              
                             λ Kh (D − d)P (D = 0|X) − 1{D = 0}E[Kh (D − d)|X]
              AT Th (d) =E Y                                                                   (3.7)
                                               fD (d)P (D = 0|X)

              T −λ
where Y λ := λ(1−λ) Y with λ = P (T = 1).

   Our goal is to construct scores that satisfy Neyman orthogonality for each h, and then take
the limit as h → 0. The next lemma presents such scores. To simplify the expressions, denote:
g(X) := P (D = 0|X); fh (d|X) := E[Kh (D − d)|X]; E∆Y (X) := E[∆Y |X, D = 0]; EλY (X) :=
   T −λ           
E λ(1−λ) Y X, D = 0 with λ = P (T = 1).


                                                    8

Lemma 3.1. Define (a) for the panel setting,
                                                                                        
                    (1)  Kh (D − d)g(X) − 1{D = 0}fh (d|X)
                   ψh :=                                                     ∆Y − E∆Y (X) − AT Th (d),             (3.8)
                                     fD (d)g(X)

and (b) for the repeated cross-sectional setting,
                                                                              
           (2)         Kh (D − d)g(X) − 1{D = 0}fh (d|X)    T −λ
          ψh :=                                                     Y − EλY (X) − AT Th (d).                       (3.9)
                                   fD (d)g(X)              λ(1 − λ)

                                 (1)                                 (2)                              (1)      (1)
Suppose there exist Mh                 ∈ L1 (PYt−1 ,Yt ,D,X ) and Mh       ∈ L1 (PY,T,D,X ) such that |ψh | ≤ Mh     and
  (2)    (2)                               (1)    (2)
|ψh | ≤ Mh almost surely. Then the scores ψh and ψh satisfy Neyman orthogonality defined in
(1).

     The proof is provided in the appendix, where we construct the adjustment term and verify
the Neyman orthogonality conditions from Definition 1. We also provide an alternative derivation
                 (1)       (2)
showing ψh and ψh as the efficient influence functions for the smoothed parameter AT Th (d) using
the method proposed in Hines et al. (2022). The assumption on the existence of integrable functions
    (1)            (2)
Mh and Mh is mild and it justifies interchanging expectation and differentiation. For simplicity,
                                         (1)        (2)
we omit superscripts on ψh                     and ψh     whenever the context is clear. The infinite-dimensional
nuisance parameters in these new scores include fh (d|X), g(X), E∆Y (X), and EλY (X), with the
latter two introduced by the adjustment terms. Notably, the estimating moments for AT Th (d)
based on these orthogonal scores remain robust to the first-order biases introduced by the nuisance
estimates. In the next section, we construct DML estimators of AT T (d) using these scores and
establish their asymptotic properties.


4         Estimation and Inference

As mentioned in the introduction, constructing DML estimators involves two main steps. In the
previous section, we established scores that satisfy Neyman orthogonality (Lemma 3.1). These
scores are then used alongside a cross-fitting procedure, further reducing estimation bias. With
these key components in place, we construct DML estimators following the procedure proposed by
Chernozhukov et al. (2018).

     First, we partition the sample IN into K ≥ 2 disjoint subsets {Ik }K
                                                                        k=1 of equal size n = N/K.
For each k ∈ {1, · · · , K}, we use the auxiliary sample Ikc := IN \ Ik to estimate the nuisance
parameters. We then compute sample averages according to (3.8) and (3.9) using these estimates,
evaluated at Ik , to obtain AT
                            [  T k (d). Finally, we average across the K estimates to obtain the final
estimator AT
          [  T (d). We note that at each k = 1, · · · , K, the nuisance parameters and AT
                                                                                       [  T k (d) are

                                                                 9

estimated using disjoint subsamples, which reduces the overfitting bias and significantly simplifies
the asymptotic analysis. Moreover, since K is fixed, it does not affect the asymptotic properties of
the estimator. In practice, we recommend using K = 5 as a rule of thumb and leave the optimal
choice of K to future research. The detailed algorithms are deferred to Appendix A.

   Next, we outline the regularity conditions required to establish the asymptotic properties of our
DML estimators. We focus on the panel case and present the analogous results for the repeated
cross-sections in Appendix B. For notational simplicity, let D denote a closed sub-interval of (dL , dH )
whose boundary points can be chosen arbitrarily close to dL and dH , and let X and ∆Y denote the
supports of X and ∆Y , respectively.

Assumption 4.6 (Kernel). The kernel function K(·) satisfies: (a) K(·) is bounded and differ-
entiable; (b) K(u)du = 1, uK(u)du = 0, 0 < u2 K(u)du < ∞. Moreover, for notation
             R           R                     R

simplicity, define Kh (u) := h−1 K(u/h).

Assumption 4.7 (Bounds and Smoothness, Panel). (a) There exist constants c > 0 and 0 < C <
                    0 (d) > c, |Y                           0                         0
∞ such that supd∈D fD            t−1 | < C, |Yt | < C, c < fh (d|X) < C ∀d ∈ D, and |E∆Y (X)| < C
                    0 (d) ∈ C 2 (D) and sup      2 0                 0            2
almost surely; (b) fD                      d∈D |∂d fD (d)| < ∞; (c) fD|X (d|x) ∈ C (D) ∀x ∈ X and
                 0
supd,x∈D,X |∂d2 fD|X (d|x)| < ∞; (d) f∆Y,D (t, d) ∈ C 2 (∆Y) and supt,d∈∆Y,D |∂t2 f∆Y,D (t, d)| < ∞.

Assumption 4.8 (Rates, Panel). (a) The kernel bandwidth h = hN → 0 satisfies N h → ∞ and
√
  N h5 = o(1); (b) there exists a sequence εN → 0 such that h−1 ε2N = o(1); (c) with probability tend-
ing to 1, ∥fˆh (d|X)−f 0 (d|X)∥P,2 ≤ h−1/2 εN , ∥ĝ(X)−g0 (X)∥P,2 ≤ εN , ∥Ê∆Y (X)−E 0 (X)∥P,2 ≤ εN ;
                      h                                                               ∆Y
(d) with probability tending to 1, κ < ĝ(X) < 1 − κ and c < fˆh (d|X) < C almost surely, and
∥Ê∆Y (X)∥P,∞ < C.

   The kernel function is central to our analysis. In addition to its well-established theoretical
properties for estimating the density fD (d), we also use it to approximate the point mass at D = d
and the conditional density fD|X (d|X). Assumption 4.6 imposes the standard regularity conditions
on the kernel function, which are essential for establishing the asymptotic normality of our estima-
tor. Assumption 4.7 requires smoothness and boundedness of the outcome variable and relevant
distributions, while Assumption 4.8 specifies conditions on the kernel bandwidth and the quality
of the non-parametric nuisance estimators.

Remark 4.1. Assumption 4.8 (a) and (b) give h = o(N −1/5 ) and h = ω(ε2N ). However, consistency
of our variance estimator additionally requires h−2 ε2N + h−3 N −1 = o(1), which imposes a more
restrictive lower bound on h. Moreover, whereas the standard DML literature assumes the nuisance
estimators to converge at rate εN = o(N −1/4 ), we allow the conditional density fˆh to converge at
a slower rate h−1/2 εN . This relaxation does not contradict the existing DML results for regular
parameters; our target parameter is non-regular and cannot be estimated non-parametrically at

                                                   10

√
    N rate because of the continuous treatment. Finally, although our results assume a deterministic
kernel bandwidth, they should extend to data-driven choices (for example, the adaptive procedure
in Bibaut and van der Laan (2017)), which we leave to future work.

     The following lemma characterizes the bias of using kernels to approximate AT T (d).

Lemma 4.2 (Bias of AT Th (d), Panel). Suppose Assumptions 4.6, 4.7, 4.8 hold. Then Bh (d) :=
AT T (d) − AT Th (d) satisfies Bh (d) = O(h2 ) for any d ∈ D.

     The proof is given in the companion supplement. This lemma suggests that, for an under-
smoothing bandwidth, the bias does not affect the asymptotic distribution of our estimators. The
next theorem is the main result of this section that establishes the asymptotic normality of our
estimator for AT T (d).

Theorem 4.2 (Asymptotic Normality, Panel). Suppose assumptions 2.1, 2.3, 2.4, 2.5, 4.6, 4.7,
and 4.8 hold. Then, for d ∈ D, if εN = o(N −1/4 ),

                                   AT
                                   [  T (d) − AT T (d)
                                              √             →d   N (0, 1)
                                      σN (d)/ N

where

                                                                                           2
                                                                                          
       2               (1)             0               θ0h (d)
      σN (d) := E     ψh (Z, θ0h (d), fD (d), η0 (d)) − 0      Kh (D − d) − E[Kh (D − d)]        (4.1)
                                                       fD (d)

                                                 (1)
for θ0h (d) := AT Th (d) defined in (3.6) and ψh defined in (3.8).

     The proof builds on the DML framework of Chernozhukov et al. (2018), modified to accommo-
date kernel smoothing. The asymptotic variance has two components, both depending inversely
on the kernel bandwidth h: one arising from the kernels in the orthogonal score ψh , and the other
from the linear expansion of the estimator with respect to the kernel density estimator fˆD (d).
Since h is a function of the sample size N under our assumptions, we index the asymptotic variance
by N to reflect this dependence. Therefore, our estimator AT [ T (d) attains a convergence rate of
√                                                       √
  N h, which, though slower than the parametric rate N , is comparable to the optimal rate for
one-dimensional non-parametric regression estimation.

    Next, following Chernozhukov et al. (2018) and Chang (2020), we consider a cross-fitted variance
                                                         T (d) and En,k f (Zi ) := n−1
                                                                                       P
estimator. For notation simplicity, denote θ̂h (d) := AT
                                                      [                                     f (Zi ) as
                                                                                          i∈Ik
the empirical average of a function f evaluated at Zi ’s in the subsample Ik . For the panel case,


                                                       11

define
                    K
                                                                                                2
                                                                                               
                  1 X           (1)                                θ̂h (d)
       2
     σ̂N (d) :=       En,k     ψh (Z, θ̂h (d), fˆk (d), η̂k (d)) −         Kh (D − d) − fˆk (d)    .   (4.2)
                  K
                    k=1
                                                                   fˆk (d)


   Then, with this variance estimator, the 1−α confidence interval can be constructed as [AT
                                                                                           [   T (d)−
               √                              √
z1−α/2 σ̂N (d)/ N , AT
                    [  T (d) + z1−α/2 σ̂N (d)/ N ] where z1−α/2 denotes the 1 − α/2-th quantile of the
standard normal random variable. The following theorem establishes the consistency of the cross-
fitted variance estimator.

Theorem 4.3 (Consistency of Variance Estimator, Panel). Suppose the conditions of Theorem 4.2
hold and assume that h−2 ε2N + h−3 N −1 = o(1). Then, for d ∈ D,

                                            2        2
                                          σ̂N (d) = σN (d) + op (1)

        2 (d) is defined in (4.2) and σ 2 (d) is defined in (4.1).
where σ̂N                              N


   Alternatively, we can consider a multiplier bootstrap procedure to construct confidence intervals.
Such procedure has been discussed extensively in recent studies, see, e.g., Chernozhukov et al.
(2014b), Belloni et al. (2017), Su et al. (2019), Cattaneo and Jansson (2021), Fan et al. (2022), and
Colangelo and Lee (2025). First, we make the following assumption on the multiplier.

Assumption 4.9 (Sub-exponential Multiplier). The random variable ξ satisfies: (a) ξ has a sub-
exponential distribution; (b) E[ξ] = V ar(ξ) = 1; (c) ξ is independent of (Yt−1 , Yt , D, X) for the
panel case and independent of (Y, T, D, X) for the repeated cross-sectional case.

   In practice, let {ξi }N
                         i=1 be an i.i.d. sequence of random variables that satisfies Assumption 4.9.
Then for each b = 1, · · · , B, we independently draw such a sequence {ξi }N
                                                                           i=1 and construct estimates
based on the following expression. For the panel case, define

                                    K X
                                1   X         Kh (Di − d)ĝk (Xi ) − 1{Di = 0}fˆh,k (d|Xi )
                  AT
                  [  T (d)∗b :=             ξi
                                N
                                    k=1 i∈Ik
                                                              fˆk (d)ĝk (Xi )
                                                                  
                                             × ∆Yi − Ê∆Y,k (Xi ) .                                    (4.3)

Let ĉα denote the α-th quantile of {AT [   T (d)∗b − AT
                                                       [ T (d)}B
                                                               b=1 , a 1 − α confidence interval can be
constructed as [AT T (d) − ĉ1−α/2 , AT T (d) − ĉα/2 ].
                [                    [

   Moreover, we can establish valid uniform inference results based on the bootstrap estimator
proposed here. The following assumption strengthens Assumption 4.8.


                                                      12

Assumption 4.10 (Uniform Inference Rates, Panel). (a) The kernel bandwidth h = hN → 0
                        √
satisfies N h → ∞ and N h5 = o(1); (b) there exists a sequence εN → 0 such that h−1 ε2N = o(1);
(c) with probability tending to 1, sup ∥fˆh (d|X) − f 0 (d|X)∥P,2 ≤ h−1/2 εN , ∥ĝ(X) − g0 (X)∥P,2 ≤
                                         d∈D                      h
                  0 (X)∥
εN , ∥Ê∆Y (X) − E∆Y     P,2 ≤ εN ; (d) with probability tending to 1, κ < ĝ(X) < 1 − κ and
     ˆ                                            (1)
c < fh (d|X) < C almost surely ∀d ∈ D, sup     |fˆ (d)| < C, sup
                                                      d∈D     D       ∥∂d fˆh (d|X)∥P,∞ < C, and
                                                                                 d∈D
∥Ê∆Y (X)∥P,∞ < C.

   This assumption differs from the pointwise case in two key ways. First, we require that, uni-
formly over D, the nuisance estimator fˆh (d|X) remains bounded and has rate h−1/2 εN . Second,
we assume that the estimated density and conditional density to have bounded derivatives with
probability tending to 1, ensuring that the score functions are Lipschitz continuous on D. These
additional assumptions are mild and can be enforced during estimation procedures. With these
modified assumptions, the linear expansion of the bootstrap estimators holds uniformly over d ∈ D.

Theorem 4.4 (Uniform Linear Expansion, Panel). Suppose assumptions 2.1, 2.3, 2.4, 2.5, 4.6,
4.7, 4.9, and 4.10 hold. Then, for d ∈ D, if εN = o(N −1/4 ),

        AT
        [  T (d) − AT
                    [ T (d)∗
               N
                    "                                                                          #
           1 X˙       (1)               0               θ 0h (d)                             
        =         ξi ψh (Zi , θ0h (d), fD (d), η0 (d)) − 0       Kh (Di − d) − E[Kh (D − d)]
           N                                            fD (d)
                i=1
              (1)
         +R         (d)                                                                            (4.4)

where ξ˙i := ξi − 1 and supd∈D |R(1) (d)| = op ((N h)−1/2 ).

   This theorem is the basis for establishing uniform inference theory using the multiplier bootstrap
estimator. We consider the following procedure, see Chernozhukov et al. (2014b) and Fan et al.
(2022) for example, to establish valid uniform confidence bands.

   1. Construct AT
                [  T (d) and σ̂N (d) on a finite grid of values d ∈ D̄ ⊂ D.

   2. For each b = 1, · · · , B, draw an i.i.d. sequence of multipliers {ξ}N
                                                                           i=1 from a N (1, 1) distribu-
                                     ∗
      tion, and construct AT T (d) for all d ∈ D̄.
                              [
                                    b

   3. Compute ĉ(1 − α), which we denote as the (1 − α)-th quantile of
                                   (           √                                )B
                                                   N |AT
                                                      [  T (d) − AT[T (d)∗b |
                                        max                                            .
                                        d∈D̄               σ̂N (d)
                                                                                 b=1


                                                         13

    4. For all d ∈ D, construct the 1 − α uniform confidence band as
                                                  √                                     √
                     [AT
                      [  T (d) − ĉ(1 − α)σ̂N (d)/ N ,      AT
                                                            [  T (d) + ĉ(1 − α)σ̂N (d)/ N ].


With Assumption 4.10, we can easily adapt our proof of Theorem 4.3 to establish the uniform
consistency of our cross-fitted variance estimator (4.2) over D. Then, with Theorem 4.4, we can
show that the proposed uniform confidence band achieves asymptotic coverage of 1 − α, using
results from Chernozhukov et al. (2014a) (Proposition 3.2 and Theorem 3.2) and Chernozhukov et
al. (2014b) (Corollary 3.1). Since this argument is well established in the literature, e.g., see the
discussion of Theorem 4.2 in Fan et al. (2022), we do not include the formal theoretical discussion
here. Instead, we focus on presenting the new results in Theorem 4.4 and defer its proof to the
appendix.

Remark 4.2. A natural extension of our framework is to develop a test for the conditional parallel
trends assumption, akin to the approach in Callaway and Sant’Anna (2018), Section 4, which
examines differences between the not-yet-treated and the never-treated in the pre-treatment period.
Extending such a test to the continuous treatment setting requires a multi-period generalization
of the methods considered in this paper, and we suspect that stronger parallel trends assumptions
would be necessary for a valid test. While our companion study, Haddad et al. (2024), proposes
estimators that could aid in this analysis, a formal testing procedure remains an open question.
Additionally, drawing on insights from Sant’Anna and Zhao (2020), we recognize that more efficient
estimators may exist in the repeated cross-sectional settings than those considered in this paper
(Appendix B), and we defer a detailed investigation of such estimators to Haddad et al. (2024).


5     Simulation

Data-generating process (a) p = 100 dimensional covariates X ∼ N (0.2, Σ), where Σ has
variances 1 on the diagonal and covariances 0.1 off-diagonal; (b) the control group propensity score
follows P (D = 0|X) = 1/(1 + exp(−X ′ γ)), with γj = 0.5j −2 ; (c) for D > 0, the continuous
treatment is generated as D = (1 + exp(X ′ α))−1 + V , where V ⊥ X, V ∼ Beta(2, 2), αj = 0.3j −2 ;
(d) the potential outcomes are given by Yt−1 (0) = ϵ1 , Yt (0) = Yt−1 (0) + X ′ β + 1 + ϵ2 , Yt (D) =
Yt (0)−0.5D2 +ϵ3 , where βj = 0.5/j for j = 1, · · · , 6 and 0 otherwise, and (ϵ1 , ϵ2 , ϵ3 ) ∼ N (0, I3 ). For
the panel setting, the generated data are (Yi,t−1 , Yi,t , Xi , Di ), with Yt−1 = Yt−1 (0) and Yt = 1{D >
0}Yt (D) + 1{D = 0}Yt (0). Additionally, for the repeated cross-sectional setting, the generated data
are (Yi , Ti , Xi , Di ), with time indicator T ∼ Bern(0.5) and Y = T Yt + (1 − T )Yt−1 , Yt−1 = Yt−1 (0),
Yt = 1{D > 0}Yt (D) + 1{D = 0}Yt (0).

    In our simulations, the nuisance parameters P (D = 0|X), fh (d|X) = E[Kh (D − d)|X], E[Yt −

                                                      14

                       T −λ            
Yt−1 |X, D = 0], and E λ(1−λ) Y X, D = 0 are estimated non-parametrically using random forests
each with 200 trees of maximum depth 20. Throughout our simulations, we also use an under-
smoothing kernel bandwidth h = 1.06σ̂D̃ N −1/4 , where σ̂D̃ is the estimated standard deviation of
positive treatment intensities. We consider sample sizes N = 2000 and 10000 for both panel and
repeated cross-sectional settings, and we conduct B = 500 simulations in each setting. The DGP
implies the true AT T (d) = −0.5d2 , and we focus on a specific treatment intensity d = 0.9. Notably,
the continuous treatment variable is dependent on the correlated high-dimensional covariates in a
nonlinear way. Additionally, the DGPs suggest that the effective sample size should be small at
the target intensity, which adds another layer of difficulty for estimation.

    Despite these challenges, the simulation results suggest that our estimators perform well. The
histograms of these simulation estimates are shown in Figure 1, where the red lines indicate the
true ATT. We see that as the sample size increases, both bias and variance decrease. The simu-
lation estimates appear to follow a normal distribution in each case, which is consistent with our
asymptotic theory. Moreover, in Table 1, we report the bias, the standard deviation of estimated
ATTs (Std), the root-mean-squared error (RMSE), the average standard deviations (AVSE), and
the coverage probability of 95 percent confidence intervals. In both settings, bias, standard devia-
tion, and RMSE decrease as the sample size increases. The standard deviations of the simulation
estimates are very close to the average estimated standard errors, suggesting that our variance
estimators perform well. The coverage of the estimated confidence intervals is close to 95 percent,
although there is a slight under-coverage in the panel setting.

                             Table 1: Monte Carlo simulation results.

Setting and Sample Size            Bias            Std         RMSE            AVSE        Coverage
panel, n=2000                    -0.0720         0.2725        0.2819          0.2524       0.9180
panel, n=10000                   -0.0198         0.1261        0.1277          0.1262       0.9300
cross-sections, n = 2000         -0.0340         0.5745        0.5755          0.5578       0.9440
cross-sections, n = 10000         0.0290         0.2754        0.2769          0.2710       0.9500


6     Empirical Example

6.1   Background

The Medicare Prospective Payment System (PPS) reform, introduced in 1983, shifted Medicare
hospital reimbursements from a full-cost model to a fixed payment per diagnosis. However, for


                                                 15

Figure 1: The simulation results, true AT T (d) = −0.405.


                           16

the first three years, capital costs continued to be reimbursed based on actual expenses.4 This
created a relative increase in labor costs for hospitals treating Medicare inpatients. Acemoglu
and Finkelstein (2008) highlights this feature, showing that the PPS reform significantly increased
hospitals’ capital-labor ratios and encouraged technology adoption.

    Theoretically, Acemoglu and Finkelstein (2008) predicts that PPS reform would lead to a higher
capital-labor ratio and, if capital-labor substitution is sufficiently elastic, an increased demand for
capital and technology. Since only hospitals with Medicare inpatients were affected, these effects
likely varied with Medicare inpatient share. To test these predictions, Acemoglu and Finkelstein
(2008) uses data from the 1980–1986 Annual American Hospital Association (AHA) survey, which
provides hospital information including expenditures, employment, and technology adoption. Their
baseline specification is a linear regression:

                                                 ′
                               Yi,t = αi + γt + Xi,t η + β · (Di · Postt ) + εi,t ,                        (6.1)

where Yi,t is the capital-labor ratio or total number of medical facilities for hospital i in year t, Di is
the pre-reform Medicare inpatient share, and Postt is a treatment-timing indicator. Xi,t represents
covariates, and αi and γt are hospital and year fixed effects, respectively. Acemoglu and Finkelstein
(2008) argues that β captures the causal effect of PPS reform on capital-labor ratios and technology
adoption, relying on a parallel trends assumption: in the absence of the PPS reform, hospitals with
different shares Di should have experienced similar changes in outcomes over time.

    Recent work by Callaway et al. (2024) examines the same empirical setting in detail and finds
suggestive evidence that the parallel trends assumption may be too strong. This underscores the
importance of incorporating covariates to improve the plausibility of the identifying assumption.
By conditioning on covariates, our approach refines the parallel trends assumption, ensuring that
hospitals are compared based on more similar characteristics. In this way, our analysis complements
Callaway et al. (2024), offering an alternative perspective on the effects of the PPS reform.


6.2    Setup as a continuous DiD

Regression (6.1) resembles a Two-Way Fixed Effects (TWFE) design but differs in that Di is
continuous. As shown by Callaway et al. (2024), with continuous treatment, the coefficient β
in (6.1) can be viewed as a weighted average of AT T (d) with possible negative weights, which
complicates interpretation.5 Our continuous DiD framework addresses this by reframing Acemoglu
and Finkelstein (2008)’s design as follows:
   4
     As noted in Acemoglu and Finkelstein (2008), Medicare’s capital cost reimbursements remained unchanged until
1991 due to delays.
   5
     See Proposition 10 in Callaway et al. (2024). They do not incorporate covariates, but the issue persists.


                                                       17

 (a) No Treatment Pre-PPS: Before the PPS reform, no hospital was treated.

 (b) Control Group: Hospitals with Di = 0 (no Medicare patients) are the control.

  (c) Treatment Group: Hospitals with positive Medicare shares (treatment intensities) Di > 0.

 (d) Outcomes: Y includes the capital-labor ratio or measures of technological adoption.

  (e) Covariates: X includes number of beds, metro status, private status, number of medical staff,
       and state dummies.6 For the capital-labor ratio, we also add binary indicators of specialized
       capital equipments (CT, MRI, etc.).

  (f) Conditional parallel trends:

                         E[Yt (0) − Yt−1 (0)|X, D = d] = E[Yt (0) − Yt−1 (0)|X, D = 0],

       i.e., absent the PPS reform, hospitals with share D = d would have experienced similar
       changes over time as hospitals with no Medicare inpatients (shares D = 0), conditional on
       hospital-specific covariates X determined before the PPS reform.

We identify the causal effect at intensity d as:

                                    AT T (d) = E[Yt (d) − Yt (0)|D = d].

Unlike the constant β in (6.1), the causal effect curve AT T (d) can be used to study the policy
impact at a much more granular level. For example, if the PPS reform raised the capital-labor
ratio, AT T (d) should be positive for all d > 0. Moreover, AT T (d) should increase in d if the impact
of PPS reform is larger for hospitals with higher shares of Medicare inpatients. We apply our panel
estimator and, for comparability with Acemoglu and Finkelstein (2008), average pre-treatment
outcomes (Yt−1 ) over 1980–1983 and post-treatment outcomes (Yt ) over 1984–1986 (capital-labor
ratio) or 1984–1985 (technology adoption). Our data source is the cleaned data file from Acemoglu
and Finkelstein (2008).


6.3    Results

First, we examine the results for capital-labor ratio. All estimated ATTs are positive, mirroring the
findings in Acemoglu and Finkelstein (2008) and suggesting that the PPS reform led to an increase
in the capital-labor ratio. For comparison, Acemoglu and Finkelstein (2008) reports an estimate of

   6
     We exclude some additional characteristics in Acemoglu and Finkelstein (2008)—e.g., general, short-term, or
federal status—to avoid conditioning on PPS exemption criteria.


                                                      18

1.13, which exceeds most of our estimates. Moreover, our estimates vary across treatment intensities
and do not exhibit a strictly increasing trend, contradicting the theoretical prediction that hospitals
with higher Medicare shares would see greater increases in the capital-labor ratio. At low and high
treatment intensities, we note that the small effective sample sizes lead to noisier estimates, as
reflected in the wider confidence intervals. For completeness, an effect curve estimated without
covariates using a kernel method shows a similar pattern to our DML estimates.

    Next, we present evidence of increased technological adoption following the PPS reform. Specif-
ically, we consider the total number of specialized medical facilities per hospital as a proxy for
technological adoption. All of our estimated ATTs for this outcome are positive, aligning with
Acemoglu and Finkelstein (2008)’s prediction that the PPS reform would incentivize technolog-
ical adoption. The estimated treatment curve initially rises with treatment intensity but then
declines at higher intensities, again diverging from the theoretical prediction that hospitals with
larger Medicare shares would invest more. For comparison, an effect curve estimated using a ker-
nel method without covariates again shows a similar pattern to our DML estimates. As with the
capital-labor ratio, estimates are especially noisy where data are sparse, an issue amplified by our
undersmoothing bandwidth.

Remark 6.3. We apply 5-fold cross-fitting, shuffling the data before sample splitting to pre-
vent over-representation in subsamples. A second-order Gaussian kernel with an undersmoothing
bandwidth h = 1.06 × σ̂D̃ N −1/4 is used to estimate both the density fD (d) and the conditional
mean E[Kh (D − d)|X] (see Silverman (2018)). The infinite-dimensional nuisance parameters are
estimated using the Random Forest (RF) from the Python scikit-learn package, with 200 trees of
maximum depth 20 and fixed minimum leaf size 5. The RF is chosen for its flexibility to accom-
modate both continuous and discrete covariates, though other ML methods, such as deep neural
networks, can be similarly employed. The standard errors are obtained from the cross-fitted esti-
mator defined in (4.2) and used to construct the 95-percent pointwise confidence intervals and the
bootstrap uniform confidence bands. For the bootstrap CIs, we use Gaussian multipliers {ξi }N
                                                                                            i=1
drawn from a normal distribution with E[ξi ] = V ar[ξi ] = 1, with B = 1000 repetitions.


7    Conclusion

This paper studies difference-in-differences models with continuous treatments. Our identification
results are based on a conditional parallel trends assumption, allowing researchers to account for
covariates non-parametrically. Under the double/debiased machine learning framework, we develop
non-parametric estimators for the average treatment effect on the treated at each continuous treat-
ment intensity and establish their asymptotic properties. Monte Carlo simulations demonstrate

                                                  19

Figure 2: AT
          [  T (d), panel data


             20

that our estimators perform well despite the highly non-linear relationship between the continu-
ous treatment and the high-dimensional covariates. To demonstrate the empirical relevance of our
methodology, we re-examine the research questions posed in Acemoglu and Finkelstein (2008) by
applying our estimators to their dataset and obtaining new empirical insights. The extension of
difference-in-differences models to the continuous treatment setting has important implications for
empirical research. Our methods provide researchers with new tools for examining the impacts of
continuous treatment variables.
