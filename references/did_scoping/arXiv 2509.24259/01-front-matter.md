<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2509.24259.pdf
backend: pdftotext
part: 1/7
-->

# Front Matter

<!-- pages: 1-50 -->

Difference-in-Differences Under Network
                                                                                 Interference


arXiv:2509.24259v1 [stat.ME] 29 Sep 2025
                                                                       Zhiguo Xiao                    Kuan Sun ∗

                                                                                 September 30, 2025


                                                                                        Abstract

                                                  This paper develops doubly robust estimators for direct (DATT) and spillover (SATT)
                                           average treatment effects on the treated in network-based difference-in-differences (DID)
                                           designs. Unlike standard DID methods, the proposed approach explicitly accounts for
                                           treatment spillovers and high-dimensional network confounding from complex unit depen-
                                           dencies in networks. It introduces a novel identification condition where conditional parallel
                                           trends hold only after adjusting for high-dimensional network confounders. The estimators
                                           are shown to be consistent and asymptotically normal as network size increases, leveraging
                                           graph neural networks (GNNs) to handle nuisance functions. Simulation studies and an
                                           empirical application on U.S. county-level mask mandates’ impact on COVID-19 transmis-
                                           sion confirm their finite-sample performance, addressing limitations of conventional DID
                                           that ignore network interference.

                                                  Keywords: Difference-in-differences; Network spillover; Doubly robust; Graph neural
                                           networks.


                                              ∗
                                               School of Management, Fudan University, Shanghai, 200433, China. (Address correspondence to:
                                           zhiguo xiao@fudan.edu.cn). Zhiguo Xiao acknowledges financial support from the National Natural Science
                                           Foundation of China (Grant Number: 72232002).


                                                                                             1

1     Introduction
Difference-in-differences (DID) methods are widely used for policy evaluation with obser-
vational data. In its canonical form with covariates, DID relies on the conditional parallel
trends assumption (CPTA), which posits that, in the absence of treatment, treated and
comparison groups with identical covariates information would have followed similar trends
in potential outcomes over time (see Roth et al. 2023 [25] ). This assumption is typically
justified under the stable unit treatment value assumption (SUTVA), which rules out in-
terference between units. However, in many real world applications, such as in social,
economic, or epidemiological contexts, units are interconnected through networks. In such
settings, the CPTA becomes ambiguous, as a unit’s potential outcome may depend not only
on its own treatment status but also on the treatment status of its neighbors. Moreover, to
restore the credibility of the CPTA under network interference, it is necessary to condition
on high-dimensional network confounders. This network-mediated interference generates
two distinct sources of bias: (1) spillover bias, which stems from causal effects propagated
through the network, and (2) confounding bias, which arises due to the endogenous struc-
ture of the network itself.
    Two-way fixed-effects (TWFE) regressions are the most common implementation of
DID methods in panel data settings. The panel data literature on peer effects in networks
remains sparse. The existing studies, such as Bramoullé (2020) [3] , typically extend TWFE
regressions with only a very low-dimensional set of controls: an individual’s own covari-
ates, the number of immediate neighbors, and the neighbors’ average characteristics. This
strategy is rather restricted for two reasons. First, the parsimonious control set implicitly
presumes that only first-order connections matter, thereby overlooking confounding that
may arise from higher-order network links. Second, reducing neighbors’ characteristics to
simple averages cannot capture the complex, potentially nonlinear channels through which
these attributes affect outcomes.
    To accomodate the network effects, in this paper we decompose the average treatment
effects on the treated (ATT) into two components: the direct average treatment effects on
the treated (DATT) and the spillover average treatment effects on the treated (SATT).


                                             1

We develop nonparametric estimation and inference procedures for both DATT and SATT
under a new set of network-based conditional parallel trends assumption. To eliminate
spillover bias, we adapt the exposure mapping framework to delineate the subsets of units
whose untreated outcomes are expected to follow parallel paths. To remove confounding
bias, we impose the parallel trends assumption conditional on the entire covariate matrix
X and the full adjacency matrix A, thereby avoiding ad hoc restrictions to low-order neigh-
borhoods. We further demonstrate, both analytically and in simulations, that conventional
DID estimators which ignore either treatment spillovers or network confounder can suffer
substantial bias and lead to invalid inference.
   Our primary contribution is to provide a theoretical foundation for difference-in-differences
estimators that accommodate both treatment and confounder interference in observational
networks. We extend the approximate-neighborhood-interference (ANI) framework of Le-
ung (2022 [22] , 2024 [23] ) to panel and repeated cross-sectional data with staggered treatment
adoption. Moreover, we enrich the emerging double/debiased-machine-learning DID liter-
ature by replacing parametric first-step models with graph-neural-network (GNN) learners
that exploit the full adjacency matrix. We also demonstrate that the doubly robust DATT
and SATT estimator exhibits asymptotic normality as the network size increases. Notably,
this network data structure yields a distinct variance-covariance matrix. For variance es-
timation, we employ the network heteroskedasticity and autocorrelation consistent (HAC)
estimator developed by Kojevnikov et al. (2021) [20] .
   Most DID methodology continues to impose SUTVA, thus excluding spillovers, for
example, the augmented IPW estimator of Sant’Anna and Zhao (2020) [26] and the multi-
period heterogeneity frameworks of de Chaisemartin and D’Haultfoeuille (2020) [10] , Sun and
Abraham (2021) [29] , and Callaway and Sant’Anna (2021) [5] . A small but growing strand
relaxes SUTVA by introducing limited interference. Butts (2021) [4] and Fiorini (2024) [14]
modify two-way fixed-effects (TWFE) specifications to allow local spatial spillovers; Het-
tinger et al. (2023) [17] and Lee et al. (2023) [21] use specific exposure mappings to mo-
tivate outcome regression (OR), inverse probability weighting (IPW), and doubly robust
(DR) estimators; Shahn et al. (2022) derive structural-nested mean models under clus-
tered/network interference; and Xu (2023) [33] adopts a design-based approach with ANI,


                                               2

focusing solely on outcome interference and ignoring neighbor covariate effects. We con-
tribute to this literature by developing a DID framework that simultaneously accommodates
network interference arising from both treatment assignment and confounding variables,
yielding a more comprehensive and flexible structure for causal inference with panel and
repeated cross-sectional data under interference.
    The remainder of the paper is organized as follows. Section 2 introduces the modeling
framework, provides motivation, and defines the causal estimands of interest. Section 3
presents the main identification assumptions and examines the bias of the naive DID esti-
mator in the presence of treatment and confounder interference, motivating the construction
of a doubly robust estimand. Section 4 outlines the estimation procedure, including the use
of graph neural networks (GNNs) for first-step nuisance function estimation. Section 5 es-
tablishes the large sample properties of the proposed estimators, including consistency and
asymptotic normality under the ANI framework, and introduces a HAC variance estimator
adapted to network dependence. Section 6 reports results from a comprehensive simula-
tion study and Section 7 applies the method to evaluate the impact of U.S. county-level
mask-mandate policy on COVID-19 transmission. Section 8 concludes.


2     Problem Setup
Let the population of units be Nn “ t1, . . . , nu. We represent the undirected network by an
nˆn binary adjacency matrix A. A link between units i and j is indicated by Aij “ Aji “ 1,
while self-ties are excluded by setting Aii “ 0. The graph distance ℓA pi, jq is the length of
the shortest path connecting nodes i and j (taken as 8 if no path exists). For each node
i, its K-neighborhood is N pi, Kq “ t j : ℓA pi, jq ď K u whose size is npi, Kq “ |N pi, Kq|.
We call the nodes in N pi, 1qztiu the neighbors of i and those in N pi, Kqztiu with K ą 1 its
higher-order neighbors; the degree of node i is npi, 1q, the number of its direct neighbors.
    Units are indexed by i P Nn and time periods are indexed by t “ t1, . . . , T u. Yit denotes
the observed outcome. Dit denotes the treatment, with its realized value dit P t0, 1u. Xi
is a vector of pre-treatment covariates — such as age, geographic location, or socioeco-
nomic status — which may influence both treatment assignment and potential outcomes.


                                               3

The potential outcome is Yit pdt q, where dt “ pdit , d´i,t q, with d´i,t being the treatment
assignments of all other units at time t. Thus the vector dt “ pdit , d´i,t q represents the full
treatment assignment at time t. We assume that the potential outcome is determined by


                                 Yit pdt q “ hit pdit , d´i,t , X, A, εt q ,                          (1)


where hit is an unknown function, X “ pX1 , . . . , Xn q1 , εt “ pε1t , . . . , εnt q1 , with εit ’s being
unobservable random errors related to the variation of the potential outcomes. We also
assume the following treatment assignment mechanism:

                                                  `         ˘
                                         Dit “ lit X, A, ν t ,                                        (2)


where lit is an unknown function, ν t “ pν1t , . . . , νnt q1 , with νit ’s being unobservable random
errors related to the variation of the treatment assignment.
    This setup captures potential spillovers and local interactions: an individual’s outcome
may depend not only on their own treatment but also on the treatments and characteristics
of neighbors in the network. In a standard DID setup, researchers often treat the treat-
ment assignment as given or quasi-exogenous. However, when treatment is suspected to
be endogenous or correlated with underlying characteristics, it can be useful to explicitly
model the treatment assignment function like (2). In this extended DID settings, incor-
porating a propensity score model offers two main advantages. First, when dealing with
high-dimensional covariates or complex network structures, balancing treatment and con-
trol groups based solely on the outcome model becomes challenging. A propensity score
model allows researchers to flexibly model the treatment assignment mechanism, using ma-
chine learning tools such as random forests, neural networks, or graph neural networks,
thereby improving the accuracy of causal effect estimation. Second, within a doubly robust
DID framework, the inclusion of a propensity score model provides robustness: consistent
estimation and valid inference can still be achieved even if either the outcome model or the
treatment model is misspecified.
    For ease of exposition, we focus on the two-period scenario, i.e., t “ 1, 2, in the following
analysis. The results for multi-period settings are discussed in the Appendix.

                                                     4

2.1    Motivation

Under SUTVA, the parallel trends assumption serves as the core identification condition
in standard DID analysis. It states that, in the absence of treatment, the average outcome
paths of the treatment and control groups would have followed the same trend over time.
Formally, for untreated potential outcomes Yi2 p0q, this implies:


                 ErYi2 p0q ´ Yi1 p0q | Di “ 1s “ ErYi2 p0q ´ Yi1 p0q | Di “ 0s.             (3)


   A stronger and more flexible version is the conditional parallel trends assumption, which
allows for systematic differences in observed covariates Xi . It posits that, conditional on
Xi , the potential outcome paths of the treated and control units would have remained
parallel in the absence of treatment. That is, for all relevant values of x,


        E rYi2 p0q ´ Yi1 p0q | Di “ 1, Xi “ xs “ E rYi2 p0q ´ Yi1 p0q | Di “ 0, Xi “ xs .   (4)


   To relax the SUTVA assumption and allow for network interference, the existing litera-
ture introduces the concept of effective treatment or exposure mapping, where each unit’s
outcomes are depend not only on their own treatment status but also on the treatment
received by others in their network. As formalized by Manski (2013) [24] and Aronow and
Samii (2017) [2] , this approach defines a low-dimensional exposure vector:


                             Ti “ pDi , Gi q “ pDi , gpi, D ´i , Aqq ,                      (5)


where gp¨q summarizes the expose to peer treatment based on the network structure A.
The individual treatment Di is separated from the exposure term to distinguish the direct
treatment effect and spillover effects in the potential outcomes framework. In parallel,
covariate exposure is captured through a low-dimensional control vector:


                                       Wi “ qpi, X, Aq,                                     (6)

which aggregates relevant covariate information from i’s neighborhood. A commonly used


                                                5

example of such mappings is
                            ˜                      ¸              ˜   řn          ¸
                                    n
                                                                       j“1 Aij Xj
                                    ÿ
                     Ti “    Di ,         Aij Dj       ,   Wi “   Xi , řn           ,               (7)
                                    j“1                                  j“1 Aij

where the second element of Ti captures the total number of treated neighbors, and the
second element of Wi represents the average covariate value among them. Motivated by
the use of low-dimensional exposure mappings in conventional cross-sectional studies to
address network interference, we can immediately extend the parallel trends assumption to
settings with network interference. Specifically, we assume that, conditional on network-
adjusted covariates Wi , the evolution of untreated potential outcomes is comparable across
units with and without exposure to treatment. Formally, the assumption is stated as:


E rYi2 p0, 0q ´ Yi1 p0, 0q | Di “ 1, Gi “ g, Wi s “ E rYi2 p0, 0q ´ Yi1 p0, 0q | Di “ 0, Gi “ 0, Wi s .
                                                                                                    (8)
   As specified in (7), the treatment vector D reduces to two sufficient statistics: an
indicator Di for the unit’s own treatment and the count of its treated neighbors Gi —the
former pinpoints the direct effect, while the latter captures spillovers. Likewise, Wi is
summarized by the unit’s covariates and those of its immediate neighbors. Consistent
with most exposure mappings literature, this construction depends only on D N pi,1q and on
X N pi,1q , thereby ruling out interference beyond the first-order neighborhood. Essentially,
the assumption (8) states that, conditional on a unit’s own covariates and those of its
immediate neighbors, the untreated potential outcome trend of treated and control units
with no treated neighbors would have evolved in parallel. However, the assumption that
the summary statistics Ti and Wi can be correctly specified is difficult to justify (Sävje
2024 [27] ). In contrast, our model (1) and (2) is considerably less restrictive — we do not
require the correct specification of a low-dimensional function Ti of (D, A) to capture
treatment interference, nor do we require the correct specification of a low-dimensional
function Wi based on(D, A) to summarize confounder interference.


                                                           6

2.2    Causal estimands of interest

We consider conditional ATT estimands that are indexed by exposure mappings following
the DID literature. Let Ti “ pDi , Gi q “ pDi , gpi, D´i , Aqq, where the function gp¨q takes
values in a finite set G of possible exposure levels. For each sample size n, let Mn Ď Nn
denote a selected subset of units and its size is denoted by mn , i.e., mn “ |Mn |.
   In terms of the individual treatment, we first establish its causal estimand given a
specific level of the neighborhood treatment. The definition of direct average treatment
effect on the treated (DATT) is:

                           1 ÿ
           τ DAT T pgq “          E rYi2 p1, gq ´ Yi2 p0, gq | Di “ 1, Gi “ g, X, As ,       (9)
                           mn iPM
                                   n


for g P G. This denotes the direct average treatment effect on the treated when the
neighborhood treatment is set to level g while adjusting for high-dimensional network con-
founders. We restrict the comparison to a subpopulation Mn in order to ensure overlap
assumption, as further discussed below.
   Next, define the overall DATT, denoted by τ DAT T , as the average treatment effect on
the treated aggregated over the distribution of the neighborhood treatment among treated
units, which is
                                  ÿ
                     τ DAT T “             τ DAT T pgqP pGi “ g | Di “ 1, X, Aq .           (10)
                                  gPG

   We now define the spillover effects for treated units, i.e., the SATT. Specifically, we
consider the SATT of having the neighborhood treatment set to level g versus 0, when the
individual treatment is d, is defined as

                             1 ÿ
          τ SAT T pg; dq “          E rYi2 pd, gq ´ Yi2 pd, 0q | Di “ 1, Gi “ g, X, As .    (11)
                             mn iPM
                                       n


   The overall SATT effect when the individual treatment equals d is then given by

                                   ÿ
                   τ SAT T pdq “           τ SAT T pg; dqP pGi “ g | Di “ 1, X, Aq.         (12)
                                   gPG


   The direct effects τ DAT T pgq in (9) and spillover effects τ SAT T pg; dq in (11) compare po-

                                                      7

tential outcomes for treated units under fixed values of individual and neighborhood treat-
ment. In contrast, the overall DATT in (10) and SATT in (12) average these treatment
effects over the distribution of the neighborhood treatment among treated units. Unlike
previous studies that consider averages over hypothetical interventions (e.g., Bernoulli as-
signments or general stochastic interventions), our ATT estimands fix the treatment status
of the treated unit and average over the observed neighborhood treatment distribution.
This allows us to identify the total ATT for units who are treated and are also exposed to
other units’ treatment:

           1 ÿ ÿ
τ AT T “              E rYi2 p1, gq ´ Yi2 p0, 0q | Di “ 1, Gi “ g, X, As P pGi “ g | Di “ 1, X, Aq,
           mn iPM gPG
                 n
                                                                                              (13)
Then, it is straightforward to show that this is equal to the sum of the overall DATT and
SATT effects:

        1 ÿ ÿ
τ AT T “          E rYi2 p1, gq ´ Yi2 p0, gq | Di “ 1, Gi “ g, X, As P pGi “ g | Di “ 1, X, Aq
       mn iPM gPG
             n

       1 ÿ ÿ
     `            E rYi2 p0, gq ´ Yi2 p0, 0q | Di “ 1, Gi “ g, X, As P pGi “ g | Di “ 1, X, Aq
       mn iPM gPG
                  n


      “ τ DAT T ` τ SAT T p0q.                                                               (14)


   This formula shows that the overall ATT for treated units under interference consists of
two parts: the direct treatment effect (DATT) capturing how their own treatment changes
outcomes, and the spillover effect (SATT) reflecting how exposure to treated neighbors
affects them.
   In the main body of this paper, we develop a general framework for identifying the
DATT (i.e., τ DAT T pgq). The identification of the SATT follows a parallel logic and is
discussed in the Appendix.


                                                 8

3     Identification of DATT
First, we outline a set of commonly used assumptions for identifying our key causal estimand
DATT.

Assumption 1 (Locality of Exposure Mapping). There exists a fixed neighborhood
size K such that a unit’s exposure mapping depends only on the treatment assignments and
network structure within its K-neighborhood. Specifically, for any treatment vectors d, d1
and network structures A, A1 , we have:
                                                            $
                                                             NA pi, Kq “ NA1 pi, Kq,
                                                            ’
                                                            ’
                                                            ’
                                                            ’
                                                            ’
                                                            &
               Gpi, d´i , Aq “ Gpi, d1´i , A1 q     if       A NA pi,Kq
                                                                        “ A  1N A1 pi,Kq
                                                                                         ,   (15)
                                                            ’
                                                            ’
                                                            ’
                                                            %dN A pi,Kq    1N pi,Kq
                                                            ’
                                                                        “ d´iA1       .
                                                            ’
                                                              ´i


    This assumption ensures that exposure mapping is determined by the local network
structure and treatment assignments within the K-neighborhood. This restriction is modest
and consistent with the assumptions underlying most exposure mappings in prior literature.

Example 1. The following exposure mapping satisfies Assumption 1:
                                            #                          +
                                                n
                                                ÿ
                                   Gi “ 1             Aij Dj ą 0 ,
                                                j“1

where Gi indicates whether unit i has at least one treated neighbor, based on the adjacency
matrix A and the treatment vector D.

    This mapping allows us to define DATT effect, τ DAT T p1q, comparing treated and un-
treated units with treated neighbors, and SATT effect, τ SAT T p1; 1q and τ SAT T p1; 0q measure
how having at least one treated neighbor affects outcomes for treated and untreated indi-
viduals, respectively, holding own treatment status fixed.

Example 2. A more general exposure mapping that satisfies Assumption 1 is:
                                             ˜                     ¸
                                                  n
                                                  ÿ
                                      Gi “                Aij Dj       .
                                                  j“1


                                                      9

    This form represents one of the most commonly used exposure mappings, leveraging
local treatment aggregation to facilitate the analysis of peer effects in networked settings.
Additional examples of exposure mappings under network interference can be found in the
literature, including Aronow and Samii (2017) [2] , Sävje et al. (2021) [28] , and Eckles et al.
(2017) [11] .

Assumption 2 (No Anticipation). Treatment occurs only in period 2, and all units
remain untreated and unaffected by any spillover effects prior to this point.


                                   Yi1 pdi,2 , d´i,2 q “ Yi1 p0, 0q.                        (16)


    This assumption implies that the potential outcomes in the pre-treatment period is
the same as it would be in the absence of both treatment and spillovers. It extends the
standard no-anticipation assumption by additionally ruling out any spillover effects in the
pre-treatment period, under the premise that no units are treated at that time.
    We now introduce the core assumption for identifying the DATT:

Assumption 3 (Network Conditional Parallel Trends). For each unit i P Mn ,


         E pYi2 p0, gq | Di “ 1, Gi “ g, X, Aq ´ E pYi1 p0, 0q | Di “ 1, Gi “ g, X, Aq

       “E pYi2 p0, gq | Di “ 0, Gi “ g, X, Aq ´ E pYi1 p0, 0q | Di “ 0, Gi “ g, X, Aq .     (17)


    Although the Network Conditional Parallel Trends (NCPT) assumption shares concep-
tual roots with the standard conditional parallel trends assumption (4), our framework
introduces two critical innovations. First, beyond conditioning on individual covariates xi ,
we incorporate the full covariate matrix X and network structure A. This generaliza-
tion enables the use of network-derived covariate functions—such as centrality measures
or positional characteristics—rather than relying solely on individual-level attributes. Sec-
ond, while traditional parallel trends assumptions compare potential outcome trends across
treatment groups absent treatment, our NCPT assumption explicitly addresses interference.
By controlling for spillover exposure through the exposure mapping, we isolate the direct
effect under the assumption that potential outcomes evolve similarly across exposure groups

                                                  10

when spillover effects are accounted for.
   In essence, our assumption simultaneously accommodates both treatment interference
and confounding interference. To highlight the practical implications of this distinction,
we subsequently demonstrate how the naive difference-in-differences estimator becomes
biased—whether targeting the conventional average treatment effect on the treated (ATT)
or our proposed direct average treatment effect on the treated (DATT)—when interference
in treatment assignment and confounder structure is neglected.


3.1    The bias of the naive DID estimator under treatment inter-
       ference

In this part, we examine the case where only treatment interference is present, exclud-
ing the influence of confounding interference. In the following subsection, we extend the
analysis to incorporate confounding interference. Under SUTVA, the potential outcomes
for ATT depend only on the individual’s own treatment status, denoted as Yi2 pdi q, and
are unaffected by the treatment assignments of other units. The standard ATT under the
SUTVA assumption is given by:

                                      1 ÿ
                       τ SU T V A “          E rYi2 p1q ´ Yi2 p0q | Di “ 1, xi s            (18)
                                      mn iPM
                                             n


In the naive DID framework, several covariate-adjusted estimators for the ATT have been
proposed, such as the outcome regression estimator, the inverse probability weighting esti-
mator, and the doubly robust DID estimator. All these estimators consistently estimates
the following quantity:

                     1 ÿ
           τ obs “          tErYi2 ´ Yi1 | Di “ 1, xi s ´ ErYi2 ´ Yi1 | Di “ 0, xi su .     (19)
                     mn iPM
                            n


   If the conditional parallel trends assumption (4) holds, then τ obs and τ SU T V A are identi-
cal. However, if SUTVA is violated, we cannot obtain a clean τ obs because the second-period
outcome Yi2 would be influenced by other individuals’ treatment statuses and, thus, these
estimators would clearly not estimate the quantity τ SU T V A . Moreover, they also fail to con-

                                                   11

sistently estimate the direct average treatment effect on the treated τ DAT T pgq or τ DAT T ,
since they compare changes over time between treated and control units based solely on
their own treatment status Di , while disregarding potential variation in exposure due to
the neighborhood exposure.
   We next present a proposition that characterizes the discrepancy between τ obs and
τ DAT T , and identifies two primary sources of bias contributing to the difference.

Proposition 1. Suppose Assumption 1, 2 and 3 holds for any g P G, @i. An unbiased
estimator targeting τ obs does not imply unbiasedness for τ DAT T , the resulting bias equals

                   1 ÿ ÿ” `                                    ˘     `                               ˘ı
τ obs ´ τ DATT “              E Yi2 ´ Yi1 | Di “ 0, Gi “ g, xi ´ E Yi2 ´ Yi1 | Di “ 0, Gi “ g 1 , xi
                   mn iPM gPG
                         n
                      ”                                                 ı
                     ¨ PpGi “ g | Di “ 0, xi q ´ PpGi “ g | Di “ 1, xi q .               (20)


   Proposition 1 characterizes the bias that arises when interference is mistakenly ignored.
This result parallels the discussion in Forastiere et al. (2021) [15] , which also considers
interference over networks but under an unconfoundedness framework. However, in our
setting, the bias of the ATT-type estimator vanishes under weaker conditions than those
required in prior work. Specifically, it is sufficient for the neighborhood treatment Gi to
have no effect on outcome changes among control units only, or for the individual and
neighborhood treatments pDi , Gi q to be conditionally independent given covariates Xi .
   There are several main sources of dependence between Di and Gi , including: (i) un-
observed neighborhood-level confounders not captured by Xi , and (ii) peer influence in
treatment uptake (see Forastiere et al., 2021) [15] . We now examine the bias of the naive
DID estimator in the presence of confounder interference.


3.2    The bias of the naive DID under confounder interference

We are concerned with bias that arises when the parallel trends assumption fails to hold
conditional on individual covariates Xi , but becomes valid when conditioning additionally
on a vector of neighborhood-level covariates Ui P U. A typical example of Ui is the network-
                                                       řn
                                                             Aij Xj
weighted average of neighbors’ covariates, such as Ui “ řj“1
                                                          n         .
                                                          j“1 Aij


                                                12

   In what follows, we assume a network parallel trends assumption holds conditional on
the enriched covariate set pXi , Ui q, where Ui captures aggregated information from unit i’s
neighbors:


ErYi2 p0, gq ´ Yi1 p0, gq | Di “ 1, Xi , Ui s “ ErYi2 p0, gq ´ Yi1 p0, gq | Di “ 0, Xi , Ui s,   @g P G.
                                                                                                    (21)
   We present a proposition that characterizes the discrepancy between τ obs and τ DATT
under confounder interference.

Proposition 2. Suppose Assumption(1),(2) and (21) holds for any g P G, @i. An unbiased
estimator targeting τ obs does not imply unbiasedness for τ DAT T , the resulting bias equals

                        1 ÿ ÿÿ” `                                                ˘
     τ obs ´ τ DATT “                   E Yi2 ´ Yi1 | Di “ 0, Gi “ g, Ui “ u, xi
                        mn iPM gPG uPU
                              n
                            `                         1         1
                                                                     ˘ı
                        ´ E Yi2 ´ Yi1 | Di “ 0, Gi “ g , Ui “ u , xi
                           ”
                          ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q
                                                                                        ı
                             ´ PpUi “ u | Di “ 0, Gi “ g, xi q ¨ PpGi “ g | Di “ 0, xi q .          (22)


If we further assume that Di and Gi are conditionally independent given Xi , then the bias
simplifies to:

                                1 ÿ ÿ” `                                     ˘
             τ obs ´ τ DATT “               E Yi2 ´ Yi1 | Di “ 0, Ui “ u, xi
                                mn iPM uPU
                                       n
                                      `                                ˘ı
                                  ´ E Yi2 ´ Yi1 | Di “ 0, Ui “ u1 , xi
                                   ”                                                 ı
                                  ¨ PpUi “ u | Di “ 1, xi q ´ PpUi “ u | Di “ 0, xi q .             (23)


   Proposition 2 implies that if we mistakenly assume no interference and condition only on
an individual’s own covariates, the resulting bias is a combination of two sources: treatment
interference bias and confounder interference bias. In contrast, even if we assume that the
individual treatment Di is independent of the neighborhood exposure Gi given a subset of
covariates Xi —thus effectively ruling out treatment interference—bias may still arise due
to unmeasured neighborhood-level confounders.

                                                   13

    To address this, conditioning on a simplified summary measure of first-order neighbors’
covariates—as described in (21), such as their average covariate values—can help alleviate
this confounder interference bias. Nevertheless, this method still fails to capture important
structural heterogeneity in the network relationships.
    Our network conditional parallel trends assumption as (3) requires conditioning on the
entire adjacency matrix A and the full covariate matrix X. This implies a much stricter
version of parallel trends, as we assume that only units with isomorphic network positions
and identical covariates exhibit parallel counterfactual trajectories. As shown in Figure
1, units 3 and 4 share identical individual-level confounders as well as the same first-
order neighborhood confounder information. This configuration satisfies an analogue of
the parallel trends assumption as (21). However, our method does not require the parallel
trends assumption to hold specifically between units 3 and 4. In fact, in this example, units
3 and 4 are not isomorphic (they would have been if unit 2 and unit 3 were not connected).
Instead, our method requires the parallel trends assumption to hold among units with
greater similarity in confounding information, enabling more accurate estimation of causal
effects.


3.3        The doubly robust estimand

Under our network parallel trends assumption, the causal estimand of interest for both
DATT and SATT can be transformed into an identifiable estimand using one of three
commonly used strategies in the literature: outcome regression, inverse probability weight-
ing, and doubly robust methods. These approaches are widely discussed in works such
as Abadie (2005) [1] , Wooldridge (2009) [31] , and Sant’Anna and Zhao (2020) [26] . Among
them, doubly robust methods are particularly appealing due to their robustness to model
misspecification. Moreover, doubly robust frameworks are naturally compatible with mod-
ern machine learning techniques, allowing researchers to flexibly model high-dimensional
network covariates while still maintaining valid inference.
    To formally construct the doubly robust estimand, we first define the outcome regression
function as


                                             14

                                           X2 : High income

               X1 : High income        1            2         5   X5 : High income


               X3 : Low income         3            4   X4 : Low income


                                            Untreated
                                            Treated

                    Figure 1: Conditional Parallel Trends on X and A
Note: Figure 1 illustrates the logic underlying the network conditional parallel trends assump-
tion, highlighting the necessity of conditioning on both individual covariates X and the network
adjacency structure A. Initially, if we condition only on traditional individual covariates Xi , units
3 and 4 would appear to satisfy parallel trends because they share similar characteristics (e.g.,
both having low income).
Extending the conditioning set to include simple first-order
                                                      ´    řn neighborhood
                                                                        ¯     information—such as
                                                             j“1 Aij Xj
the average covariates of neighbors, defined by Wi “ Xi , n Aij —may still suggest that units
                                                            ř
                                                              j“1
3 and 4 are comparable, since their aggregated neighbor profiles appear similar. However, when we
fully condition on the entire adjacency matrix A along with the covariates X, it becomes clear that
units 3 and 4 do not satisfy the network conditional parallel trends assumption. This discrepancy
arises because deeper network features—such as global connectivity patterns, centrality, or indirect
pathways—are now captured. Therefore, failing to fully account for the detailed network structure
encoded in A can result in biased comparisons.


                       µt,dg pi, X, Aq “ E pYit | Di “ d, Gi “ g, X, Aq .                        (24)

We also follow Imbens (2000) [18] to define the generalized propensity score regression as


                           pdg pi, X, Aq “ PpDi “ d, Gi “ g | X, Aq.                             (25)


Here we focus on the panel data case with t “ 1, 2, leaving the derivations for the multiple-
period panel and the repeated cross-section cases to the Appendix. Let ∆Yi “ Yi2 ´ Yi1
and ∆µdg pi, X, Aq “ µ2,dg pi, X, Aq ´ µ1,dg pi, X, Aq. Define

                                                  1 ÿ dr
                                     τ dr pgq “          τ pgq,                                  (26)
                                                  mn iPM i
                                                        n


                                                   15

where
                ˆ                                                       ˙
                                   p1 ´ Di q1tGi “ gu ¨ p1g pi, X, Aq
   τidr pgq “       Di 1tGi “ gu ´                                          p∆Yi ´ ∆µ0g pi, X, Aqq .
                                            1 ´ p1g pi, X, Aq
                                                                                                  (27)
Remark. The estimand τ dr pgq defined above bears a close resemblance to the doubly
robust score for panel data DID models developed by Sant’Anna and Zhao (2020) [26] .
Their results demonstrate that estimators based on this doubly robust structure achieve
semiparametric efficiency under standard regularity conditions. Our work generalizes this
framework by (i) incorporating spillover effects and (ii) accounting for network confounding
– features absent in their original formulation.
   Beyond efficiency considerations in conventional panel settings (the focus of Sant’Anna
and Zhao), we investigate whether valid inference persists when combining the doubly
robust score with double/debiased machine learning (DML) techniques. This approach
parallels that of Chang (2020) [6] , who examines DML-based inference in standard DID
frameworks. Our contribution adapts this methodology to settings with dependence struc-
tures induced by network interference, while establishing a central limit theorem that en-
sures asymptotically valid inference even with flexibly estimated, high-dimensional nuisance
parameters.

Proposition 3. Suppose Assumptions (1)-(3) hold. If either the conditional outcome mean
model or the propensity score model is correctly specified, then τ DAT T pgq “ τ dr pgq.

   Proposition 3 establishes that, under mild assumptions on the exposure mapping and
conditional trends, our primary target–the causal estimand τ DAT T pgq–is identified (i.e.,
expressible in terms of observable quantities) provided that either the outcome regression
model or the propensity score model is correctly specified. Consequently, our proposed
estimand τ dr pgq is doubly robust, yielding valid inference even under misspecification of
one of the two models. Relative to approaches relying solely on outcome regression or
inverse probability weighting, the doubly robust estimand imposes weaker assumptions
and demonstrates greater reliability in practice.


                                                   16

4      Estimation

4.1      Network DR-DID estimator

To estimate the doubly robust estimand τ dr pgq, we adopt a standard plug-in approach that
leverages machine learning-based estimators for the relevant nuisance components. Specif-
ically, let p̂dg pi, X, Aq denote the estimated generalized propensity score, µ̂t,dg pi, X, Aq de-
note the estimated outcome regression for t “ 1, 2, and ∆µ̂dg pi, X, Aq “ µ̂2,dg pi, X, Aq ´
µ̂1,dg pi, X, Aq. We then propose the following DR-DID estimator for τ dr pgq that allows for
network interference:

                                                       1 ÿ dr
                                         τ̂ dr pgq “          τ̂ pgq,
                                                       mn iPM i
                                                             n


where each τ̂idr pgq is defined as:
                  ˆ                                                       ˙
                                       p1 ´ Di q1tGi “ gup̂1g pi, X, Aq
    τ̂idr pgq “       pDi 1tGi “ guq ´                                        p∆Yi ´ ∆µ̂0g pi, X, Aqq .
                                              1 ´ p̂1g pi, X, Aq
                                                                                                     (28)

    The validity of the Network DR-DID estimator τ̂ dr pgq hinges on accurately estimating
the nuisance components, the propensity score and the outcome regression function. These
components are traditionally estimated using parametric models, such as logistic regres-
sion or linear outcome regression. However, such models often lack the flexibility needed
to capture the complex dependencies and nonlinear interactions that arise in networked
data, especially when spillovers effects or network confounding are present. To address
this, we propose using Graph Neural Networks (GNNs) to estimate these nuisance func-
tions. GNNs are a class of nonparametric machine learning models designed specifically for
graph-structured data. They incorporate both individual-level covariates (node features)
and network structure (neighborhood edges) to generate learned representations of each
unit. Through iterative local averaging—commonly referred to as message passing—GNNs
extract information from a unit’s neighbors to improve prediction. This allows GNNs to
flexibly approximate high-dimensional, nonlinear relationships in both the outcome and


                                                        17

treatment assignment mechanisms, without requiring explicit model specification or man-
ual feature construction. As a result, plugging GNNs estimator into the doubly robust
framework can improve both the robustness and efficiency of causal inference in complex
network settings.
   In the next subsection, we provide a brief overview of the GNN architecture used to
estimate the nuisance components.


4.2    GNNs estimator for nuisance functions

GNNs are deep learning models designed to model graph-structured data. A standard
GNNs architecture consists of nested, parameterized, vector-valued functions called neurons
                                                                               plq
arranged in L layers. The embedding of the i-th node at layer l, denoted hi , is updated
via the following message-passing architecture for layers l “ 1, . . . , L:

                             ´        ´                         ¯¯
                      plq      pl´1q    pl´1q pl´1q
                     hi “ Φ0l hi , Φ1l hi , thj     : j P N piqu ,                       (29)


where Φ0l p¨q and Φ1l p¨q are parameterized, vector-valued functions. The embedding is ini-
            p0q
tialized as hi “ xi , thus initially incorporating only node features. As layers progress, the
embeddings incorporate increasingly richer neighborhood information. This architecture
endows GNNs with several essential properties. Permutation invariance ensures that the
aggregation of neighbor embeddings is insensitive to the order in which they appear, a criti-
cal feature given that graph neighborhoods are inherently unordered. Due to the unordered
nature of graph neighborhoods, the estimation functions p1g pi, X, Aq and ∆µ0g pi, X, Aq
must be permutation invariant in the features of i’s neighbors. This ensures that the esti-
mated values are not sensitive to arbitrary ordering of the neighborhood set. As discussed in
Leung (2024) [23] , such invariance allows us to reduce from a collection of neighbor-specific
functions to a single symmetric function over the neighborhood multiset. Neighborhood
                                                                                           pLq
scope is controlled via the number of layers L, such that the final node embedding hi
reflects information from the node’s L-hop neighborhood. The scalability of GNNs arises
from the fact that the learnable functions Φ0l p¨q and Φ1l p¨q depend solely on the dimension
of node features and are independent of the graph size. As a result, GNNs can be deployed


                                               18

efficiently across networks of varying scales, from small graphs to large-scale systems. These
structural properties make GNNs particularly effective for accurately modeling complex de-
pendencies in observational data, thereby improving the estimation of nuisance functions
such as propensity scores or conditional outcome regressions. By incorporating rich rela-
tional structure in a robust and scalable manner, GNNs estimator can more effectively ad-
just for confounding, particularly when outcomes or treatments exhibit network-dependent
relationship. The specific choices of Φ0l p¨q and Φ1l p¨q define the architectural variants of
the GNN and thus influence both model expressiveness and computational behavior. There
exist various GNN embedding architectures, including the Graph Convolutional Network
(GCN) [19] , the Graph Isomorphism Network (GIN) [32] , and the Principal Neighborhood Ag-
gregation (PNA) network [9] , each differing in how neighborhood information is aggregated
and combined.
              prop          µ
   We define FGNN  pLq and FGNN pLq as classes of L-layer graph neural networks used to
estimate the generalized propensity score and the outcome regression function, respectively.
             prop             µ
For any f P FGNN  pLq or f P FGNN pLq, we let f pi, X, Aq denote its output for node i,
corresponding to the final-layer embedding hi . The nuisance estimators fˆGNN
                                                                          prop
                                                                               and fˆGNN
                                                                                     µ
                                            pLq


are obtained via empirical risk minimization:

                                             n
                                             ÿ
                fˆGNN
                  prop
                       P arg      min
                                  prop
                                                   ℓlog p1tDi “ d, Gi “ gu, f pi, X, Aqq ,
                               f PFGNN pLq
                                             i“1


                                                        ÿ
                      fˆGNN
                        µ
                            P arg        min
                                         µ
                                                                    ℓsq p∆Yi , f pi, X, Aqq ,
                                    f PFGNN pLq
                                                   i:Di “0, Gi “g

where ℓlog py, ŷq “ ´y ŷ ` logp1 ` eŷ q is the logistic loss and ℓsq py, ŷq “ 0.5py ´ ŷq2 is the
squared-error loss.
   The estimated functions are then used to define the nuisance components:
                                                  ´                 ¯
                                              exp fˆGNN
                                                     prop
                                                          pi, X, Aq
                           p̂dg pi, X, Aq “         ´                 ¯,
                                                      ˆ prop
                                            1 ` exp fGNN pi, X, Aq


                                    µ̂0g pi, X, Aq “ fˆGNN
                                                       µ
                                                           pi, X, Aq.


                                                         19

    The estimated nuisance functions p̂dg pi, X, Aq and µ̂0g pi, X, Aq are then plugged into
the doubly robust score defined in (28) to deliver τ̂idr pgq, i “ 1, . . . , mn , which are averaged
to obtain the estimator τ̂ dr pgq for the estmation of τ dr pgq, and consequently the DATT
τ DAT T pgq.


5     Asymptotic theory

5.1     Limiting distribution

We first discuss the limiting distribution of τ̂ dr pgq as n Ñ 8. While our analysis treats
`                                                                `     ˘
  X, A, εt , ν t q as random, the asymptotic theory conditions on X, A to avoid imposing
additional assumptions on their underlying dependence structure. Define:


         ˆ                                                    ˙
                        p1 ´ Di q1tGi “ gu p1g pi, X, Aq
ϕi pgq “ Di 1tGi “ gu ´                                           p∆Yi ´ ∆u0g pi, X, Aqq ´ τ DAT T pgq,
                                1 ´ p1g pi, X, Aq
                                                                                               (30)

and                                      ˜                      ¸
                                               1 ÿ
                             σn2 “ Var       ?       ϕi pgq X, A .                             (31)
                                              mn iPM
                                                      n


The following assumptions are required to guarantee the validity of our asymptotic analysis.
They ensure that the estimators are well-defined and converge properly as the sample size
increases.

Assumption 4 (Approximate Neighborhood Interference). For each sample size
n P N there exist non-negative functions γn psq and ηn psq, defined on R` , satisfying


                                 sup maxtγn psq, ηn psqu ÝÝÝÑ 0,
                                 nPN                      sÑ8


                                                 20

such that for every individual i P Nn and period t,
       ”ˇ                          ` N pi,sq N pi,sq N pi,sq N pi,sq ˘ˇ ˇ          ı
      E ˇhit pDt , X, A, εt q ´ hit Dt      ,X      ,A      , εt      ˇ ˇ Dt , X, A ď γn psq,
                        ”ˇ                                                                 ı
                                                                         N pi,sq ˘ˇˇ ˇ
                                                                                     ˇ
                       E ˇlit pX, A, ν t q ´ lit XN pi,sq , AN pi,sq , ν t
                                                `
                                                                                       X, A ď ηn psq.

   Assumption 4 indicates a uniform, distance-based decay of interference within the net-
work. Specifically, outcome models hit and the propensity score model lit are asymptotically
insensitive to information originating beyond an s-step neighborhood of the focal node. The
bounding functions γn psq and ηn psq converge to zero uniformly in n, ensuring that remote
nodes exert a vanishing influence as s Ñ 8. Hence, observations from a single, expansive
network can be treated as only weakly dependent, permitting the application of classical
asymptotic theorem. Leung (2022) [22] demonstrated that the ANI assumption is satisfied
by a range of interference structures, such as the linear-in-means model with endogenous
peer effects.

Assumption 5 (Moments and Overlap). (a) There exist constants M ă 8 and p ą 4
such that, for every sample size n P N, every individual i P Mn , every period t, and every
treatment vector dt P t0, 1un ,

                                “              ˇ     ‰
                               E |Yi2 pdt q| p ˇ X, A ď M           a.s.


   (b) For every unit i P Mn , each treatment status d P t0, 1u, and each exposure level
g P G, there exists a constant ε ą 0 such that


                                    ε ă pdg pi, X, Aq ă 1 ´ ε.


   Assumption 5(a) bound p-th moments of the potential outcomes, which is a standard
regularity condition, see the double machine learning literature (e.g., Chernozhukov et
al., 2018 [7] ; Farrell, 2018 [12] ; Farrell et al., 2021 [13] ). By contrast, Assumption 5(b) is
conceptually more restrictive because it links the exposure mapping, network structure, and
treatment distribution. For simplicity, I assume that the overlap condition holds for every
unit in the population. However, under certain specifications of the exposure mapping,

                                                  21

this assumption might not always be valid. If violations appear, one can (i) redefine or
coarsen the exposure mapping, (ii) trim units with near-zero or near-one propensities, or
(iii) restrict inference to a subpopulation where credible overlap holds.

Assumption 6 (GNN Convergence Rates). For every g P G and d P t0, 1u, let p̂dg pi, X, Aq
and ∆µ̂dg pi, X, Aq be the first–stage GNN estimators of pdg pi, X, Aq and ∆µdg pi, X, Aq, re-
spectively. Suppose the following conditions hold:

 (a)
                              1 ÿ`                                 ˘2
                                     p̂dg pi, X, Aq ´ pdg pi, X, Aq “ op p1q,
                              mn iPM
                                      n

                         1    ÿ`                                ˘2
                                ∆µ̂dg pi, X, Aq ´ ∆µdg pi, X, Aq “ op p1q.
                         mn iPM
                                 n


 (b)
              "                         *1{2 "                         *1{2
                  1 ÿ                 2        1 ÿ                   2
                                                                                `     ˘
                         pp̂dg ´ pdg q                p∆µ̂dg ´ ∆µdg q       “ op n´1{2 .
                  mn iPM                       mn iPM
                         n                            n


 (c)
              "                                                                *
       1 ÿ p̂dg pi, X, Aq ´ 1pDi “ dq1pGi “ gq                                      `     ˘
                                               p∆udg pi, X, Aq´∆ûdg pi, X, Aqq “ op n´1{2 .
       mn iPM          1 ´ p̂dg pi, X, Aq
             n


   These regularity conditions are well-established in the double machine learning liter-
ature. The validity of these assumptions are verified for convolutional neural networks
(CNNs) in the i.i.d. setting by both Farrell (2021) [13] and Ghasempour et al. (2024) [16] .
Extending to network data, Wang et al. (2024) [30] establish analogous n´1{2 -rate con-
vergence results for GNNs, under certain architectural constraints. A recent advance by
Leung (2024) [23] strengthens the required independence structure through an approximate
conditional-independence assumption (his Assumption 8). Under that assumption he shows
that, for both the propensity-score model and the outcome-regression model,

                  1 ÿ”                                            ˘ ı2
                         ft pi, X, Aq ´ ft i, XN pi,Lq , AN pi,Lq
                                          `                                `     ˘
                                                                       “ op n´1{2 ,
                  mn iPM
                          n

                                           `                      ˘
where the neighborhood-restricted target ft i, XN pi,Lq , AN pi,Lq can be consistently esti-

                                                 22

mated by an L-layer GNN. Leung further shows that choosing L — log n (or any slowly
diverging sequence) is sufficient for the network to achieve the required approximation error
while keeping the effective model complexity low. Together, these results place GNNs on
essentially the same theoretical footing as classical machine-learning estimators for semi-
parametric causal inference.
   To establish a central limit theorem for our main term, we require that the sequence
tϕi pgquni“1 be ψ-dependent (as in Definition C.1 by Kojevnikov et al., 2021 [20] ). This as-
sumption restricts how quickly a specific dependence measure decays in relation to the
growth rate of network neighborhoods. To formalize this, we first define the s-neighborhood
boundary of node i as
                               N B pi, sq “ tj P Nn : ℓpi, jq “ su,

and its kth moment by
                                                    n
                                                 1ÿ B
                                  δnB ps; kq “         |N pi, sq|k .
                                                 n i“1

Next, we introduce

                                         n
                                    1ÿ
                     ∆n ps, m; kq “         max |N pi, mqzN pj, s ´ 1q|k ,
                                    n i“1 jPN B pi,sq

which captures, on average, the maximal expansion of a node’s m-neighborhood beyond
the ps ´ 1q-neighborhood of any node on its s-boundary. Based on this, we define

                                                            ´       α ¯1´1{α
                     cn ps, m; kq “ inf ∆n ps, m; kαq1{α δnB s;              ,
                                   αą1                             α´1

which is a quantity that essentially measures the network density. Finally, we set

                       ´                                                         ¯
           ψn psq “ max γn ps{2q ` ηn ps{2q r1 ` npi, Kq ` Λn pi, s{2q npi, s{2qs ,     (32)
                     iPNn


where Λn pi, s{2q is a constant defined in the subsequent assumption. ψn psq provides a
bound on the covariance between ϕg piq and ϕg pjq when the network distance ℓA pi, jq is at
most s.

Assumption 7 (Weak Dependence for CLT). (a) The dependence coefficients are

                                                  23

uniformly bounded. Specifically, supnPN maxsě1 ψn psq ă 8 almost surely. (b) Let p ą 4
in assumption 5(b), for some sequence vn Ñ 8 and for each k P t1, 2u, the following
conditions hold:

                    8
                1 ÿ
                k{2
                      cn ps, vn ; kq ψn psq1´p2`kq{p Ñ 0,       n3{2 ψn pvn q1´1{p Ñ 0,
               n s“0

and
                                  8
                                  ÿ
                        lim sup         δnB ps; 2q1{2 γn ps{2q1´2{p ă 8   a.s.
                          nÑ8     s“0

   The quantity ψn psq measures the degree of dependence between pairs of observations
Yit and Yjt across different individuals i ‰ j at the same time period t. As discussed
in Kojevnikov et al. (2021) [20] , many network-dependent processes satisfy ψ-dependence.
Moreover, Leung (2024, Appendix C) [23] derives the rate of ψn psq under the ANI assumption
(4) for observational data. This enables the use of robust inferential procedures despite the
presence of approximate local network dependencies in the observational setting. The first
two parts of Condition (b) in Assumption (7) coincide with Condition ND in Kojevnikov
et al. (2021) [20] . The third part is an analogous requirement that guarantees the linear
expansion of the doubly robust ATT estimator under network dependence. Leung (2024) [23]
demonstrates that both polynomial and exponential neighborhood-growth patterns satisfy
all three components of Condition (b).

Theorem 1. Under Assumptions 1-7, the Network DR-DID estimator τ̂ dr pgq has asymp-
totically normal distribution centered around τ DAT T pgq. Specifically,

                              ? `                        ˘ d
                        σn´1{2 mn τ̂ dr pgq ´ τ DAT T pgq ÝÑ Np0, 1q.


5.2    Variance Estimation

We now focus on the variance estimator for large-sample inference. To estimate the asymp-
totic variance, we utilize the network HAC (heteroskedasticity and autocorrelation consis-
tent) estimator as described by Kojevnikov et al. (2021) [20] :


                                                   24

                    1 ÿ ÿ ` dr                    ˘`                    ˘
           σ̂ 2 “              τ̂i pgq ´ τ̂ dr pgq τ̂jdr pgq ´ τ̂ dr pgq 1tℓA pi, jq ď Bn u.   (33)
                    mn iPM jPM
                            n     n


   We adopt the uniform-kernel variance estimator and choose the bandwidth as

                              $Q
                                  1        U                log n
                                     LpAq   , if LpAq ă 2           ,
                              ’
                              ’
                                 2`γ
                              &
                                                          log δ̄pAq
                          Bn “ Q                                                               (34)
                              ’         1 U
                              % “LpAq‰ 2`γ , otherwise,
                              ’

                                                               1ř
where r¨s denotes rounding up to the nearest integer; δ̄pAq “         Aij is the network’s
                                                               n i,j
average degree; LpAq is the average path length; and γ ą 0 is a fixed positive constant.
   Thus, the bandwidth adapts to the network’s size and density while accounting for
first-stage estimation error. This bandwidth rule builds on the scheme proposed by Leung
(2022) [22] , Leung (2024) [23] . The next theorem states the asymptotic properties of σ̂ 2 .
Because we condition on pX, Aq, σ̂ 2 is not guaranteed to be consistent—exactly as in
Leung (2022) [22] . Nevertheless, the same argument shows it is typically asymptotically
conservative. We introduce the required notation below:

                           ␣                                                              (
                Jn ps, mq “ pi, j, k, lq P Nn4 : k P N pi, mq, l P N pj, mq, ℓA pi, jq “ s .

Assumption 8 (Weak Dependence for σ̂). (a) For some ϵ P p0, 1q and a bandwidth
    Bn Ñ 8, limnÑ8 n1 8
                     ř                           1´ϵ
                       s“0 cn ps, Bn ; 2q ψn psq     “ 0 a.s.
          řn                 `? ˘
 (b) n1    i“1 npi, Bn q “ op  n .
          řn            2
                                 `? ˘
 (c) n1    i“1 npi, Bn q     “ Op n .
       řn ˇ           ˇ            ` ˘
 (d)      ˇJn ps, Bn qˇ ψn psq “ op n2 .
          s“0

   This assumption regulates the growth rate of the neighborhood size and the bandwidth
Bn , ensuring that the estimator σ̂ 2 remains consistent and well-behaved in large samples by
balancing bias and variance. Assumption 8(a) corresponds to the first part of Assumption
7(b). Parts (b)–(d) align with Assumptions 7(b)–(d) in Leung (2022) [22] , which serve to
characterize the bias properties of the variance estimator. These conditions are satisfied
under both polynomial and exponential neighborhood growth network.

                                                    25

Theorem 2. Define ϕ˚i pgq by replacing τ DAT T pgq in the definition of ϕi pgq with τiDAT T pgq.
Let


      1 ÿ ÿ ˚
    σ̂˚2 “        ϕ pgqϕ˚j pgq1 tℓA pi, jq ď Bn u  and
      mn iPM jPM i
            n   n

      1 ÿ ÿ ` DAT T                          ˘`                       ˘
 Rn “              τi      pgq ´ τ DAT T pgq τjDAT T pgq ´ τ DAT T pgq 1 tℓA pi, jq ď Bn u .
      mn iPM jPM
                 n    n


Under Assumption 8 and the assumptions of Theorem 1, we have that

                                                                 ˇ 2        ˇ p
                          σ̂ 2 “ σ̂˚2 ` Rn ` op p1q        and   ˇσ̂˚ ´ σn2 ˇ Ý
                                                                              Ñ 0.


      This extends Proposition 4.1 of Kojevnikov et al. (2021) [20] and Theorem 4 of Leung
(2022) [22] to accommodate doubly robust ATT estimators. Note that Rn is a HAC es-
timator of the variance of the unit-level contrasts τiDAT T pgq, in which case σ̂ 2 would be
asymptotically conservative.


6       Simulations
In this simulation, we demonstrate the finite-sample performance of the estimators proposed
for DATT. For the data generating process, we simulated a network A comprising 2000
individuals based on a random geometric graph model, which defines the adjacency matrix
A by setting
                                      Aij “ 1 t}ρi ´ ρj } ď rn u ,

where the positions tρi , ρj uni“1 are independently and uniformly drawn from the unit square
                                                             a
r0, 1s2 , and the radius parameter rn is specified as rn “ 5{pπnq. The simulated random
geometric graph has an average path length of approximately 39.4.
      We consider a two-period panel data structure, with the outcome equation for the first


                                                      26

period generated as:
                                     řn                     řn
                                      j“1 Aij Xj              j“1 Aij ϵj
                       Ypre,i “ 0.5 ` řn         ` Xi ` ϵi ` řn          ,                  (35)
                                        j“1 Aij                j“1 Aij


where tXi uni“1 are i.i.d. draws from a discrete uniform distribution on t0, 0.25, 0.5, 0.75, 1u,
and tϵi uni“1 are i.i.d. Np0, 1q random variables. The treatment variable Di is generated
according to the following equation:
               #        řn           řn                      řn              +
                         j“1 Aij Dj   j“1 Aij X j              j“1 Aij ν j
        Di “ 1 0.5 ` 1.5 řn         ` řn          ´ Xi ` νi ` řn           ą0 ,
                              A
                           j“1 ij          A
                                        j“1 ij                  j“1 Aij


where the error terms tνi uni“1 are i.i.d. as Np0, 1q. The outcome equation for the post-
treatment period is defined as
                             řn              řn                     řn
                              j“1 Aij Yj      j“1 Aij Xj             j“1 Aij µj
          Ypost,i “ 0.5 ` 0.8 řn         ` 10 řn         ` Xi ` µi ` řn
                                j“1 Aij         j“1 Aij                j“1 Aij


The error terms tui u are also i.i.d. as standard normal. The true value of the estimand
τ DAT T is zero under this design.
   We compare two estimators: GNNs and nonparametric generalized linear model (NGLM)
estimators. The GNNs are implemented using the PNA architecture [9] , with the number
                                  plq      plq
of layers L P t1, 2, 3u. Both ϕ0 and ϕ1 are single-layer multilayer perceptrons (MLPs)
with hidden dimension H P t1, 3, 5u. As for NGLM, we apply polynomial basis expansions
of degree 1, 2, or 3 to estimate the nuisance functions. The degree of the polynomial
plays a role analogous to the number of layers L in GNNs, as both determine the order of
neighborhood effects captured by the model.
   Table 1 presents simulation results based on 1000 replications for the random geometric
graph. The upper panel reports results with nuisance parameters estimated by the GNNs
method, and the lower panel reports results with nuisance parameters estimated the NGLM
method employing polynomial sieve methods, where the polynomial order is also indicated
by L. For convenience, we refer to the former as the GNN method and the latter as the
                               dr                                      dr
NGLM method. The row labeled τ̂GN N reports the average value of the τ̂GN N estimates,


                                                 27

                       Table 1: Simulation Results for GNNs and NGLM

                        L“1                          L“2                        L“3
      n                  2000                        2000                       2000
  # treated              1105                        1105                       1105
     H          1          3       5        1         3         5        1       3         5
      dr
    τ̂GN N    0.0469    0.0652   0.0110   0.0366     0.0128   0.0012   0.0192   0.0172   0.0320
     SE       0.1564    0.1766   0.1787   0.1556     0.1854   0.1674   0.1588   0.1511   0.2502
     CI       0.8080    0.9440   0.9540   0.8140     0.9660   0.8800   0.8160   0.8360   0.8660
   SE IID     0.0979    0.1079   0.1108   0.0972     0.1161   0.1136   0.1009   0.0937   0.2049
   CI IID     0.5968    0.6934   0.6442   0.6332     0.5675   0.6955   0.5841   0.5980   0.5321
   τ̂NdrGLM             0.0836                       0.0764                     0.0774
       SE                0.098                       0.098                      0.098


and the row labeled τ̂NdrGLM reports the average value of the τ̂NdrGLM estimates, both of
which also reflect the bias due to the fact that the true parameter τ DAT T is zero. “SE”
denotes standard error constructed by the HAC estimator. “CI” displays the coverage rate
of confidence interval constructed with the HAC variance estimator. “SE IID” denotes
standard error computed under the assumption of independence and identical distribution,
with “CI IID” being the coverage rate of confidence interval constructed with the i.i.d.
variance estimator.
   The bias results presented in the first row of Table 1 demonstrate that the GNNs
method provides reliable causal estimates across all specifications of L and H. Notably,
GNNs with L “ 2 layers consistently outperform other configurations, achieving the lowest
bias regardless of the hidden dimension H. Furthermore, for a fixed number of layers, bias
tends to decrease as the hidden dimension increases.
   The HAC standard errors are substantially larger than those computed under the i.i.d.
assumption, suggesting the presence of both heteroskedasticity and autocorrelation in the
error terms. Coverage rates generally improve with larger hidden dimensions, and our pro-
posed method produces more accurate confidence intervals compared to those derived from
i.i.d. standard errors. However, as is common with HAC-type estimators, our confidence
intervals exhibit a slight degree of undercoverage.
   The NGLM method also delivers reliable causal estimates for all choices of polynomial
order, though the magnitude of its bias is larger than that of the GNNs method. This
suggests that GNNs capture a different function of pX, Aq than the Wi variables alone,


                                                28

one that better adjusts for confounding effects.


7     An Application
We employ the method proposed in this paper to assess the impact of mask mandate
policy on the spread of COVID-19 in the US. Our analysis is based on a balanced panel
constructed from data used by Chernozhukov et al. (2021b) [8] , which consist of 2,510 US
counties observed weekly from April 1, 2020 to December 2, 2020. A total of 736 counties
remained untreated throughout the study period and thus serve as the control group. To
accommodate variation in treatment timing across the remaining counties and maintain a
clean 2 ˆ 2 DID design, we select a subset of counties for the treatment group. Specifically,
we focus on the 343 counties that adopted mask mandates in week 28—the week with the
highest number of implementations. We define the pre-treatment period as all weeks before
week 28 (t “ 1) and the post-treatment period as week 28 onward (t “ 2).
    In this study, Yit denotes the logarithm of reported COVID-19 cases in county i at time
period t. The main treatment variable Dit represents a policy indicator for mask mandates.
The set of control variables Xi includes measures of foot traffic to K-12 schools and colleges
(sourced from SafeGraph), along with other policy indicators such as stay-at-home orders
and bans on gatherings of more than 50 persons, as well as the weekly growth rate in
COVID-19 testing. We construct an adjacency matrix based on the geographic distance
matrix between counties identified by their FIPS codes, where a link is assumed to exist
between two counties if the distance between them is less than 400 kilometers, and no link
otherwise.
    For illustration, we consider the estimation of the direct treatment effect τ DATT p1q. We
compare two approaches: our proposed Network DR-DID and the DR-DID of Sant’Anna
and Zhao (2020) [26] . Both estimators target a direct effect of mask mandate policy, but
differ in how they account for network-related interference. Our estimator conditions on
having at least one treated neighbor and explicitly controls for the network confounding
spillovers, while the DR-DID method assumes no treatment and confounding spillovers.
    As shown in Table 2, both estimation methods yield significantly negative causal effect


                                              29

                           Table 2: Comparison of ATT Estimates

                Method                   ATT Estimate            Standard Error
                Network DR-DID             -0.7021˚˚˚                0.2983
                DR-DID                     -0.9363˚˚˚                0.2586
    Notes: Robust standard errors are reported in parentheses.
    ˚˚˚ p ă 0.01, ˚˚ p ă 0.05, ˚ p ă 0.1


values, indicating that the mask mandate policy effectively and significantly reduced the
number of COVID-19 cases. It is important to note that the magnitude of the Network
DR-DID estimate is smaller than that of the conventional DR-DID estimate. This finding
aligns with our intuition that the protective effect of wearing a mask diminishes when
one’s neighbors also wear masks, compared to scenarios where neighbors do not. This
discrepancy suggests that traditional estimates may suffer from bias due to unaccounted
spillover effects and network confounders.


8     Conclusion
In this article, we develop doubly robust estimators for the Direct Average Treatment Effect
on the Treated (DATT) and the Spillover Average Treatment Effect on the Treated (SATT)
in network-based DID designs, where conditional parallel trends hold after adjusting for
high-dimensional network confounders. The proposed estimators remain consistent for
the DATT (or SATT) under the condition that either the propensity score model or the
outcome regression model is correctly specified. We establish their large-sample properties
and demonstrate that, under mild regularity conditions, the doubly robust estimators are
asymptotically normal as the network size increases. The practical utility of our method is
illustrated through Monte Carlo simulations and an empirical application.
    Our findings can be extended to several other settings of practical relevance. First, the
network-based analytical framework developed in this study can be adapted to alterna-
tive identification strategies in panel data settings, particularly those relying on sequential
conditional independence assumptions. Second, while our analysis focuses on contem-
poraneous treatment effects, incorporating both dynamic treatment effects and spillover

                                              30

effects simultaneously would introduce additional methodological challenges. Finally, the
current framework assumes a static network structure, whereas real-world networks of-
ten exhibit dynamic evolution. Extending the causal inference framework to account for
network dynamics—such as by modeling network formation or selection processes over
time—represents a promising direction for future research.


                                           31
