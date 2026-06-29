<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1803.09015.pdf
backend: pdftotext
part: 1/7
-->

# Front Matter 1

<!-- pages: 1-45 -->

Difference-in-Differences with Multiple Time Periods∗

                                                                        Brantly Callaway†                  Pedro H. C. Sant’Anna‡

                                                                                             December 1, 2020

arXiv:1803.09015v4 [econ.EM] 1 Dec 2020
                                                                                                  Abstract

                                                    In this article, we consider identification, estimation, and inference procedures for treatment effect
                                                parameters using Difference-in-Differences (DiD) with (i) multiple time periods, (ii) variation in treat-
                                                ment timing, and (iii) when the “parallel trends assumption” holds potentially only after conditioning
                                                on observed covariates. We show that a family of causal effect parameters are identified in stag-
                                                gered DiD setups, even if differences in observed characteristics create non-parallel outcome dynamics
                                                between groups. Our identification results allow one to use outcome regression, inverse probability
                                                weighting, or doubly-robust estimands. We also propose different aggregation schemes that can be
                                                used to highlight treatment effect heterogeneity across different dimensions as well as to summarize
                                                the overall effect of participating in the treatment. We establish the asymptotic properties of the
                                                proposed estimators and prove the validity of a computationally convenient bootstrap procedure to
                                                conduct asymptotically valid simultaneous (instead of pointwise) inference. Finally, we illustrate the
                                                relevance of our proposed tools by analyzing the effect of the minimum wage on teen employment from
                                                2001–2007. Open-source software is available for implementing the proposed methods.

                                                JEL: C14, C21, C23, J23, J38.
                                                Keywords: Difference-in-Differences, Dynamic Treatment Effects, Doubly Robust, Event Study, Vari-
                                                ation in Treatment Timing, Treatment Effect Heterogeneity, Semi-Parametric.

                                             ∗
                                               First complete version: March 23, 2018. A previous version of this paper has been circulated with the title “Difference-
                                          in-Differences with Multiple Time Periods and an Application on the Minimum Wage and Employment”. We thank the
                                          Editor, the Associate Editor, two anonymous referees, Stéphane Bonhomme, Carol Caetano, Sebastian Calonico, Xiaohong
                                          Chen, Clément de Chaisemartin, Xavier D’Haultfoeuille, Bruno Ferman, John Gardner, Andrew Goodman-Bacon, Federico
                                          Gutierrez, Sukjin Han, Hugo Jales, Andrew Johnston, Vishal Kamat, Qi Li, Tong Li, Jason Lindo, Catherine Maclean, Matt
                                          Masten, Magne Mogstad, Tom Mroz, Aureo de Paula, Jonathan Roth, Donald Rubin, Bernhard Schmidpeter, Yuya Sasaki,
                                          Na’Ama Shenhav, Tymon Sloczyński, Sebastian Tello-Trillo, Alex Torgovitsky, Jeffrey Wooldridge, Haiqing Xu and several
                                          seminar and conference audiences for comments and suggestions. Code to implement the methods proposed in the paper is
                                          available in the R package did which is available on CRAN.
                                             †
                                               Department of Economics, University of Georgia. Email: brantly.callaway@uga.edu
                                             ‡
                                               Department of Economics, Vanderbilt University. Email: pedro.h.santanna@vanderbilt.edu

                                                                                                       1

1    Introduction
Difference-in-Differences (DiD) has become one of the most popular research designs used to evaluate
causal effects of policy interventions. In its canonical format, there are two time periods and two groups:
in the first period no one is treated, and in the second period some units are treated (the treated group),
and some units are not (the comparison group). If, in the absence of treatment, the average outcomes
for treated and comparison groups would have followed parallel paths over time (which is the so-called
parallel trends assumption), one can estimate the average treatment effect for the treated subpopulation
(ATT) by comparing the average change in outcomes experienced by the treated group to the average
change in outcomes experienced by the comparison group. Methodological extensions of DiD methods
often focus on this standard two periods, two groups setup; see, e.g., Heckman et al. (1997, 1998), Abadie
(2005), Athey and Imbens (2006), Qin and Zhang (2008), Bonhomme and Sauder (2011), de Chaisemartin
and D’Haultfœuille (2017), Botosaru and Gutierrez (2018), Callaway et al. (2018), and Sant’Anna and
Zhao (2020).1
    Many DiD empirical applications, however, deviate from the canonical DiD setup and have more than
two time periods and variation in treatment timing. In this article, we provide a unified framework for
average treatment effects in DiD setups with multiple time periods, variation in treatment timing, and
when the parallel trends assumption holds potentially only after conditioning on observed covariates. We
concentrate our attention on DiD with staggered adoption, i.e., to DiD setups such that once units are
treated, they remain treated in the following periods.
    The core of our proposal relies on separating the DiD analysis into three separate steps: (i) identifi-
cation of policy-relevant disaggregated causal parameters; (ii) aggregation of these parameters to form
summary measures of the causal effects; and (iii) estimation and inference about these different target
parameters. Our approach allows for estimation and inference on interpretable causal parameters al-
lowing for arbitrary treatment effect heterogeneity and dynamic effects, thereby completely avoiding the
issues of interpreting results of standard two-way fixed effects (TWFE) regressions as causal effects in
DiD setups as pointed out by Borusyak and Jaravel (2017), de Chaisemartin and D’Haultfœuille (2020),
Goodman-Bacon (2019), Sun and Abraham (2020), and Athey and Imbens (2018). In addition, it adds
transparency and objectivity to the analysis (Rubin (2007, 2008)), and allows researchers to exploit a
variety of estimation methods to answer different questions of interest.
    The identification step of the analysis provides a blueprint for the other steps. In this paper, we pay
particular attention to the disaggregated causal parameter that we call the group-time average treatment
effect, i.e., the average treatment effect for group g at time t, where a “group” is defined by the time period
when units are first treated. In the canonical DiD setup with two periods and two groups, these parameters
reduce to the ATT which is typically the parameter of interest in that setup. An attractive feature of the
group-time average treatment effect parameters is that they do not directly restrict heterogeneity with
respect to observed covariates, the period in which units are first treated, or the evolution of treatment
effects over time. As a consequence, these easy-to-interpret causal parameters can be directly used for
learning about treatment effect heterogeneity, and/or to construct many other more aggregated causal

    1
      See Section 6 of Athey and Imbens (2006) and Theorem S1 in de Chaisemartin and D’Haultfœuille (2017) for notable
exceptions that cover multiple periods and multiple groups.

                                                          2

parameters. We view this level of generality and flexibility as one of the main advantages of our proposal.
   We provide sufficient conditions related to treatment anticipation behavior and conditional parallel
trends under which these group-time average treatment effects are nonparametrically point-identified.
A unique feature of our framework is that it shows how researchers can flexibly incorporate covariates
into the staggered DiD setup with multiple groups and multiple periods. This is particularly important
in applications in which differences in observed characteristics create non-parallel outcome dynamics
between different groups – in this case, unconditional DiD strategies are generally not appropriate to
recover sensible causal parameters of interest (Heckman et al., 1997, 1998; Abadie, 2005). We propose
three different types of DiD estimands in staggered treatment adoption setups: one based on outcome
regressions (Heckman et al., 1997, 1998), one based on inverse probability weighting (Abadie, 2005),
and one based on doubly-robust methods (Sant’Anna and Zhao, 2020). We provide versions of these
estimands both for the case with panel data and for the case with repeated cross sections data. To the
best of our knowledge, this paper is the first to show how one can allow for covariate-specific trends across
groups in DiD setups with variation in treatment timing. Our results also highlight that, in practice, one
can rely on different types of parallel trends assumptions and allow some types of treatment anticipation
behavior; our proposed estimands explicitly reflect these assumptions.
   Our framework acknowledges that in some applications there may be many group-time average treat-
ment effects and researchers may want to aggregate them into different summary causal effect measures.
This characterizes the aggregation step of the analysis. We provide ways to aggregate the potentially
large number of group-time average treatment effects into a variety of intuitive summary parameters and
discuss specific aggregation schemes that can be used to highlight different sources of treatment effect
heterogeneity across groups and time periods. In particular, we consider aggregation schemes that deliver
a single overall treatment effect parameter with similarities to the ATT in the two period and two group
case as well as partial aggregations that highlight heterogeneity along certain dimensions such as (a) how
average treatment effects vary with length of exposure to the treatment (event-study-type estimands); (b)
how average treatment effects vary across treatment groups; and (c) how cumulative average treatment
effects evolve over calendar time. We also provide a formal discussion of the costs and benefits of balanc-
ing the sample in “event time” when analyzing dynamic treatment effects. Overall, our setup makes it
clear that, in general, the “best” aggregation scheme is application-specific as it depends on the type of
question one wants to answer.
   Given that our identification results are constructive, we propose easy-to-use plug-in type (parametric)
estimators for the causal parameters of interest. Although the outcome regression, inverse probability
weighting and doubly-robust estimands are equivalent from the identification point of view, they suggest
different types of DiD estimators one can use in practice. Here, we note that using doubly-robust
estimators can be particularly attractive as they rely on less stringent modeling conditions than the
outcome regression and the inverse probability weighting procedures.
   In order to conduct asymptotically valid inference, we justify the use of a computationally conve-
nient multiplier-type bootstrap procedure. This approach can be used to obtain simultaneous confidence
bands for the group-time average treatment effects. Unlike commonly used pointwise confidence bands,
our simultaneous confidence bands asymptotically cover the entire path of the group-time average treat-
ment effects with fixed probability and take into account the dependency across different group-time

                                                     3

average treatment effect estimators. Thus, our proposed confidence bands are arguably more suitable for
visualizing the overall estimation uncertainty than more traditional pointwise confidence intervals.
   We illustrate the practical relevance of our proposal by analyzing the effect of the minimum wage on
teen employment. Here, we follow much empirical work on the effects of the minimum wage and exploit
having access to panel data and variation in treatment timing across states (e.g., Card and Krueger
(1994); Neumark and Wascher (2000, 2008); Dube et al. (2010), among many others) in order to estimate
the effect of the minimum wage on employment. Interestingly, in our setup, using our approach leads
to qualitatively different results than results from the TWFE estimator. This suggests that, at least
in certain applications, using methods that are robust to treatment effect heterogeneity can lead to
meaningful differences relative to more standard TWFE regressions.

   Recent Related Literature: This paper is related to the recent and emerging literature on het-
erogeneous treatment effects in DiD and/or event studies with variation in treatment timing; see, e.g.,
de Chaisemartin and D’Haultfœuille (2020), Goodman-Bacon (2019), Imai et al. (2018), Borusyak and
Jaravel (2017), Athey and Imbens (2018) and Sun and Abraham (2020). All these papers present, among
other things, some negative results about the interpretation of parameters associated with standard
TWFE linear regression specifications; see also Laporte and Windmeijer (2005), Wooldridge (2005a),
Chernozhukov et al. (2013), and Gibbons et al. (2018) for earlier related results based on (one-way) fixed-
effect estimators. Our proposed procedure completely bypasses the pitfalls highlighted in these papers as
we clearly separate the identification, aggregation and estimation/inference steps of the analysis.
   These aforementioned papers also propose alternative DiD estimators that do not suffer from the
pitfalls associated with TWFE. Among these, perhaps the closest to our proposal are those of de Chaise-
martin and D’Haultfœuille (2020), and Sun and Abraham (2020), though several major differences are
worth stressing.
   de Chaisemartin and D’Haultfœuille (2020) is focused on recovering an instantaneous treatment effect
measure, while we pay particular attention to treatment effect dynamics. In fact, our framework allows one
to form families of different aggregate parameters in a unified manner. Second, while we pay particular
attention to the role played by pre-treatment covariates, de Chaisemartin and D’Haultfœuille (2020)
mainly focus on unconditional DiD designs. On the other hand, the setup in de Chaisemartin and
D’Haultfœuille (2020) is more general than ours as we consider staggered adoption designs and they
allow for more general treatment selection. Nonetheless, we note that the unconditional versions of our
parallel trends assumptions are weaker than the one in de Chaisemartin and D’Haultfœuille (2020), even
if one were to specialize their setup to staggered adoption designs.
   Sun and Abraham (2020) proposes a parameter, cohort-specific average treatment effects, that trans-
lates our group-time average treatment effects from calendar time into event time. Sun and Abraham
(2020) proposes regression-based estimators of these parameters that have similar properties to our esti-
mators in the specific case of staggered treatment adoption under an unconditional version of the parallel
trends assumption. However, our approach is more general in several respects. First, we allow for parallel
trends assumptions to hold after conditioning on covariates, and it is not clear how to adapt the regression
based estimators in Sun and Abraham (2020) to this case. Second, we consider a wide variety of possible
aggregations of group-time average treatment effects where Sun and Abraham (2020) focuses particularly

                                                     4

on the event study type of aggregation. Third, we make use of simultaneous inference procedures that
explicitly account for potential multiple-testing problems; Sun and Abraham (2020) focuses on pointwise
inference. On the other hand, we do not have any results highlighting the pitfalls associated with using
TWFE specifications with leads and lags of treatment indicators to conduct causal inference; these are
unique to Sun and Abraham (2020).
    We also note that Athey and Imbens (2018) considers a staggered treatment adoption setup similar
to ours. However, the starting point of Athey and Imbens (2018) is an assumption that the treatment
adoption date is fully randomized which is stronger than our parallel trends assumptions. We also note
that Athey and Imbens (2018) abstracts away from the important role played by covariates in the DiD
analysis and does not consider aggregation schemes to summarize treatment effect heterogeneity like
we do. On the other hand, we stress that the main focus of their paper is on providing design-based
inference procedures for staggered DiD setups with random treatment dates. Their design-based inference
procedures complement our sampling-based inference procedures.

    Organization of the paper: The remainder of this article is organized as follows. Section 2 presents
our main identification results. We discuss our different aggregation schemes in Section 3. Estimation
and inference procedures for the treatment effects of interest are presented in Section 4. We revisit the
effect of minimum wage on employment in Section 5. Section 6 concludes. Proofs as well as additional
methodological results are reported in the Appendix. In the Supplementary Appendix, we present proofs
for the results when only repeated cross-sections data is available, provide additional details about the
empirical application, and present a small scale Monte Carlo simulation to illustrate the finite sample
properties of our proposed estimators.2

2       Identification
2.1     Setup
We first introduce the notation we use throughout the article. We consider the case with T periods and
denote a particular time period by t where t = 1, . . . , T . In a canonical DiD setup, T = 2 and no one
is treated in period t = 1. Let Di,t be a binary variable equal to one if unit i is treated in period t and
equal to zero otherwise. We make the following assumption about the treatment process:

Assumption 1 (Irreversibility of Treatment). D1 = 0 almost surely (a.s.). For t = 2, . . . , T ,

                                          Dt−1 = 1 implies that Dt = 1 a.s..

    Assumption 1 states that no one is treated at time t = 1, and that once a unit becomes treated,
that unit will remain treated in the next period.3 This assumption is also called staggered treatment

    2
     Supplementary Appendix is available at https://pedrohcgs.github.io/files/Callaway_SantAnna_2020_supp.pdf.
    3
     In applications, it can be the case that some units are already treated by the first time period. In our case, we would
drop these units; this is analogous to the case with two time periods. The reason to drop these units is that untreated
potential outcomes are never observed for this group which will imply that treatment effects are not identified for this group
nor are they useful as a comparison group under a parallel trends assumption.

                                                              5

adoption in the literature. We interpret this assumption as if units do not “forget” about the treatment
experience.4
    Define G as the time period when a unit first becomes treated. Under Assumption 1, for all units
that eventually participate in the treatment, G defines which “group” they belong to. If a unit does not
participate in any time period, we arbitrarily set G = ∞. We define Gg to be a binary variable that is
equal to one if a unit is first treated in period g (i.e., Gi,g = 1{Gi = g}) and define C to be a binary
variable that is equal to one for units that do not participate in the treatment in any time period (i.e.,
Ci = 1{Gi = ∞} = 1−Di,T ). Let ḡ = maxi=1,··· ,n Gi be the maximum G in the dataset. Next, denote the
generalized propensity score as pg,s (X) = P (Gg = 1|X, Gg + (1 − Ds ) (1 − Gg ) = 1). Note that pg,s (X)
indicates the probability of being first treated at time g, conditional on pre-treatment covariates X and
on either being a member of group g (in this case, Gg = 1) or a member of the “not-yet-treated” group by
time s (in this case, (1−Ds )(1−Gg ) = 1). Many of our results use a specialized version of this generalized
propensity score, and, henceforth, we define pg (X) = pg,T (X) = P (Gg = 1|X, Gg + C = 1) which is the
probability of being first treated in period g conditional on covariates and either being a member of group
g or not participating in the treatment in any time period. Let G = supp(G)\ {ḡ} ⊆ {2, 3, . . . , T } denote
the support of G excluding ḡ.5 Likewise, let X = supp(X) ⊆ Rk denote the support of the pre-treatment
covariates. Finally, for a generic δ ≥ 0, let Gδ = G∩ {2 + δ, 3 + δ, . . . , T }.
    Next, we set up the potential outcomes framework. Here, we combine the dynamic potential outcomes
framework of Robins (1986, 1987) with the multi-stage treatment adoption setup discussed by Heckman
et al. (2016); see also Sianesi (2004). Let Yi,t (0) denote unit i’s untreated potential outcome at time t
if they remain untreated through time period T ; i.e., if they were not to participate in the treatment
across all available time periods. For g = 2, . . . , T , let Yi,t (g) denote the potential outcome that unit i
would experience at time t if they were to first become treated in time period g. Note that our potential
outcomes notation accounts for potential dynamic treatment selection, though it also accommodates (pre-
specified) treatment regimes (Murphy et al., 2001; Murphy, 2003). The observed and potential outcomes
for each unit i are related through
                                                        T
                                                        X
                                    Yi,t = Yi,t (0) +         (Yi,t (g) − Yi,t (0)) · Gi,g                          (2.1)
                                                        g=2

In other words, we only observe one potential outcome path for each unit. For those that do not participate
in the treatment in any time period, observed outcomes are untreated potential outcomes in all periods.
For units that do participate in the treatment, observed outcomes are the unit-specific potential outcomes
corresponding to the particular time period when that unit adopts the treatment.
    We also impose the following random sampling assumption.

Assumption 2 (Random Sampling). {Yi,1 , Y,i2 , . . . Yi,T , Xi , Di,1 , Di,2 , . . . , Di,T }ni=1 is independent and
identically distributed (iid).

    Assumption 2 implies that we have access to panel data; our results extend essentially immediately
   4
      See Han (2020), de Chaisemartin and D’Haultfœuille (2020) and Bojinov et al. (2020) for alternative setups where
treatment can “turn off”.
    5
      When there is a “never treated” set of units with G = ∞, G only excludes this group. When such “never-treated”
group is not available, we exclude the latest-treated group as there will be no valid untreated comparison group for them.

                                                               6

to the case with repeated cross sections data and this case is developed in Appendix B. Here, we note
that Assumption 2 allows us to view all potential outcomes as random. Furthermore, it does not impose
restrictions between potential outcomes and treatment allocation, nor does it restrict the time series
dependence of the observed random variables. On the other hand, Assumption 2 imposes that each
unit i is randomly drawn from a large population of interest. For an alternative design-based inference
approach, see Athey and Imbens (2018).
   Henceforth, to keep the notation more concise, we will suppress the unit index i in our notation.

2.2      The Group-Time Average Treatment Effect Parameter
Given that different potential outcomes cannot be observed for the same unit at the same time, researchers
often focus on identifying and estimating some average causal effects. For instance, in the canonical DiD
setup with two time periods, the most popular treatment effect parameter of interest is the average
treatment effect on the treated, denoted by6

                                           AT T = E[Y2 (2) − Y2 (0)|G2 = 1].

In this paper, we consider a natural generalization of the AT T that is suitable to setups with multiple
treatment groups and multiple time periods. More precisely, we use the average treatment effect for units
who are members of a particular group g at a particular time period t, denoted by

                                         AT T (g, t) = E[Yt (g) − Yt (0)|Gg = 1],

as the main building block of our framework. We call this causal parameter the group-time average
treatment effect.
   Note that the AT T (g, t) does not impose any restriction on treatment effect heterogeneity across
groups or across time. Thus, focusing on the family of AT T (g, t)’s allow us to analyze how average
treatment effects vary across different dimensions in a unified manner. For instance, by fixing a group g
and varying time t, one is able to highlight how average treatment effects evolve over time for that specific
group. By doing this for different groups, we can have a better understanding about how treatment effect
dynamics vary across groups. In addition, as we discuss in Section 3, one can build on the AT T (g, t)’s
to form more aggregated causal parameters that are constructed to answer specific questions like: (a)
What was the average effect of participating in the treatment across all groups that participated in the
treatment by time period T ? (b) Are average treatment effects heterogeneous across groups? (c) How do
average treatment effects vary by length of exposure to the treatment? (d) How do cumulative average
treatment effects evolve over calendar time? We view this level of generality and flexibility as one of the
main advantages of our framework that first focuses on the family of AT T (g, t)’s.

2.3      Identifying Assumptions
In order to identify the AT T (g, t) and their functionals, we impose the following assumptions.

   6
       Existence of expectations is assumed throughout.

                                                            7

Assumption 3 (Limited Treatment Anticipation). There is a known δ ≥ 0 such that

      E[Yt (g)|X, Gg = 1] = E[Yt (0)|X, Gg = 1] a.s. for all g ∈ G, t ∈ {1, . . . , T } such that t < g − δ.

   Assumption 3 restricts anticipation of the treatment for all “eventually treated” groups. When δ = 0,
it imposes a “no-anticipation” assumption, see, e.g., Abbring and van den Berg (2003) and Sianesi (2004).
This is likely to be the case when the treatment path is not a priori known and/or when units are not
the ones who “choose” treatment status. However, Assumption 3 also allows for anticipation behavior, as
long as we have a good understanding about the anticipation horizon δ. For instance, if units anticipate
treatment by one period, Assumption 3 would hold with δ = 1; see, e.g., Laporte and Windmeijer (2005)
and Malani and Reif (2015) for the importance of accounting for potential anticipation behavior. Note
that, under Assumption 3, AT T (g, t) = 0 for all pre-treatment periods such that t < g − δ.
   Next, we consider two alternative assumptions that impose restrictions on the evolution of untreated
potential outcomes.

Assumption 4 (Conditional Parallel Trends based on a “Never-Treated” Group). Let δ be as defined in
Assumption 3. For each g ∈ G and t ∈ {2, . . . , T } such that t ≥ g − δ,

                      E[Yt (0) − Yt−1 (0)|X, Gg = 1] = E[Yt (0) − Yt−1 (0)|X, C = 1] a.s..

Assumption 5 (Conditional Parallel Trends based on “Not-Yet-Treated” Groups). Let δ be as defined
in Assumption 3. For each g ∈ G and each (s, t) ∈ {2, . . . , T } × {2, . . . , T } such that t ≥ g − δ and
t + δ ≤ s < ḡ,

                  E[Yt (0) − Yt−1 (0)|X, Gg = 1] = E[Yt (0) − Yt−1 (0)|X, Ds = 0, Gg = 0] a.s..

   Assumptions 4 and 5 are two different conditional parallel trends assumptions that generalize the
two-period parallel trends assumption to the case where there are multiple time periods and multiple
treatment groups; see, e.g., Heckman et al. (1997, 1998), Abadie (2005) and Sant’Anna and Zhao (2020).
Both assumptions hold after conditioning on covariates X. This can be important in many applications in
economics particularly in cases where there are covariate specific trends in outcomes over time and when
the distribution of covariates is different across groups. For example, Heckman et al. (1997) motivates
conditional parallel trends assumptions in the context of evaluating a job training program. For evaluating
job training programs, the distribution of observed covariates such as age, employment history, and years
of education is often quite different between individuals who participate in job training and those that do
not. When the path of labor market outcomes (in the absence of participating in job training) depends
on these covariates, a conditional parallel trends becomes more plausible than an unconditional parallel
trends assumption. In fact, ignoring the presence of covariate-specific trends can result in important
biases when evaluating causal effects of policy interventions using unconditional DiD methods.
   Assumptions 4 and 5 differ from each other depending on the comparison group one is willing to use
in a given application. More specifically, Assumption 4 states that, conditional on covariates, the average
outcomes for the group first treated in period g and for the “never-treated” group would have followed
parallel paths in the absence of treatment. Assumption 5 imposes conditional parallel trends between

                                                        8

group g and groups that are “not-yet-treated” by time t + δ.7 Importantly, both of these assumptions
allow for covariate-specific trends and do not restrict the relationship between treatment timing and the
potential outcomes, Yt (g)’s. Thus, they are weaker than the randomization-based assumption made by
Athey and Imbens (2018). We also note that the unconditional versions of Assumptions 4 and 5 are
weaker than the parallel trends assumption imposed by de Chaisemartin and D’Haultfœuille (2020) and
Sun and Abraham (2020) as they impose fewer restrictions on the evolution of Yt (0) in pre-treatment
periods; see, e.g., Marcus and Sant’Anna (2020) for a comparison.
    In our view, practitioners may favor Assumption 4 with respect to Assumption 5 when there is a
sizable group of units that do not participate in the treatment in any period, and, at the same time,
these units are similar enough to the “eventually treated” units. When a “never-treated” group of units
is not available or “too small”, researchers may favor Assumption 5 as it allows one to use more groups
as valid comparison units, which potentially leads to more informative inference procedures. However, it
is important to stress that favoring Assumption 5 with respect to Assumption 4 also involves potential
drawbacks. For instance, in the absence of treatment anticipation (δ = 0), Assumption 4 does not restrict
observed pre-treatment trends across groups, whereas Assumption 5 does; see, e.g., Marcus and Sant’Anna
(2020). Not restricting pre-treatment trends may be particularly meaningful in applications where the
economic environment during the “early-periods” was potentially different from the “later-periods.” In
these cases, the outcomes of different groups may evolve in a non-parallel manner during “early-periods”,
perhaps because the groups were exposed to different shocks, while trends become parallel in the “later-
periods.” We recommend taking these trade-offs into account when deciding which conditional parallel
trends assumption is more appropriate for a given application.8
    The final identifying assumption we impose is an overlap condition.

Assumption 6 (Overlap). For each t ∈ {2, . . . , T }, g ∈ G, there exist some ε > 0 such that P (Gg = 1) >
ε and pg,t (X) < 1 − ε a.s..

    Assumption 6 extends the overlap assumption in Heckman et al. (1997, 1998), Abadie (2005), and
Sant’Anna and Zhao (2020) to the multiple groups and multiple periods setup. It states that a positive
fraction of the population starts treatment in period g, and that, for all g and t, the generalized propensity
score is uniformly bounded away from one. Assumption 6 rules out “irregular identification”, see, e.g.,
Khan and Tamer (2010).

Remark 1. Note that Assumption 3 and Assumption 4 (Assumption 5) are intrinsically connected. For
instance, when one imposes the “no-anticipation” condition (so that δ = 0), Assumption 4 would then
impose conditional parallel trends only for post-treatment periods t ≥ g. If one allows for anticipation
behavior (so that δ > 0), Assumption 4 would then impose conditional parallel trends in some pre-
treatment periods, too. In fact, the parallel trends assumptions become stronger as one increases δ. To

    7
      Athey and Imbens (2006) and de Chaisemartin and D’Haultfœuille (2017) also consider using “not-yet-treated” units
as comparison groups in related DiD procedures.
    8
      It may be tempting to use statistical pre-tests to select between different versions of the parallel trends assumption.
However, the results of Roth (2020) show that such a practice can lead to important distortions when conducting inference.
Thus, we do not recommend following this path, but instead recommend taking the context of the application into account
in order to choose the appropriate parallel trends assumption.

                                                             9

the best of our knowledge, this trade-off between the strength of these assumptions has not been noticed
before.

Remark 2. In some applications, practitioners may not be comfortable with using “never-treated” units
as part of the comparison group because they behave very differently from the other “eventually treated”
units. In these cases, practitioners could drop all “never-treated” units from the analysis and proceed with
Assumption 5.

2.4   Nonparametric Identification of the Group-Time Average Treatment Effects
In this section, we show that the family of group-time average treatment effects are nonparametrically
point-identified under the aforementioned assumptions. Furthermore, we show that one can use outcome
regression (OR), inverse probability weighting (IPW), or doubly robust (DR) estimands to recover the
AT T (g, t)’s. In addition, we also highlight the roles played by Assumption 3 and by Assumptions 4 and
5 when forming these different estimands.
   Before formalizing all the results, we need to introduce some additional notation. Let mnev
                                                                                           g,t,δ (X) =
E [Yt − Yg−δ−1 |X, C = 1] and mny
                               g,t,δ (X) = E [Yt − Yg−δ−1 |X, Dt+δ = 0, Gg = 0]. These are population
outcome regressions for the never-treated group and for the “not-yet-treated” by time t + δ group. Let

                                                    pg (X) C
                                                                          

                 nev
                                    Gg           1 − pg (X)               
           AT Tipw                  E [Gg ] −  pg (X) C   (Yt − Yg−δ−1 ) ,
                     (g, t; δ) = E                                                                (2.2)
                                                                           
                                                 E
                                                    1 − pg (X)
                                                                      
                 nev                  Gg                     nev
                                                                     
           AT Tor (g, t; δ) = E              Yt − Yg−δ−1 − mg,t,δ (X) ,                            (2.3)
                                     E [Gg ]
                                                    pg (X) C
                                                                                      

                 nev
                                    Gg           1 − pg (X)                   nev
                                                                                        
           AT Tdr                   E [Gg ] −  pg (X) C   Yt − Yg−δ−1 − mg,t,δ (X)  .
                     (g, t; δ) = E                                                                (2.4)
                                                                                       
                                                 E
                                                    1 − pg (X)

Analogously, let

                                          pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )
                                                                                         

       ny                Gg                       1 − pg,t+δ (X)                         
  AT Tipw                E [Gg ] −  pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )   (Yt − Yg−δ−1 ) ,
          (g, t; δ) = E                                                                              (2.5)
                                                                                          
                                        E
                                                     1 − pg,t+δ (X)
                                                              
       ny                  Gg     
                                                       ny
  AT Tor (g, t; δ) = E              Yt − Yg−δ−1 − mg,t,δ (X) ,                                        (2.6)
                          E [Gg ]
                                          pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )
                                                                                                   

       ny                Gg                       1 − pg,t+δ (X)                                 
                                                                               Yt − Yg−δ−1 − mny (X)  .
  AT Tdr  (g, t; δ) = E            −                                                       g,t,δ
                         E [Gg ]          pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )                         
                                        E
                                                     1 − pg,t+δ (X)
                                                                                                      (2.7)

   With some abuse of notation, we write ḡ − δ = ∞ for any non-negative δ whenever ḡ = ∞.

                                                    10

Theorem 1. Let Assumptions 1, 2, 3 and 6 hold.
   (i) If Assumption 4 holds, then, for all g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ} and t ≥ g − δ,

                                           nev                 nev                 nev
                        AT T (g, t) = AT Tipw  (g, t; δ) = AT Tor  (g, t; δ) = AT Tdr  (g, t; δ) .

   (ii) If Assumption 5 holds, then, for all g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ} and g−δ ≤ t < ḡ−δ,

                                            ny                 ny                 ny
                         AT T (g, t) = AT Tipw (g, t; δ) = AT Tor (g, t; δ) = AT Tdr (g, t; δ) .

   Theorem 1 is the first main result of this paper. It provides powerful identification results that
extend the DiD identification results based on the outcome regression approach of Heckman et al. (1997,
1998), the IPW approach of Abadie (2005), and the DR approach of Sant’Anna and Zhao (2020) to
the multiple-periods, multiple groups setup. In other words, Theorem 1 says that, from an identification
point of view, one can recover the AT T (g, t)’s by exploiting different parts of the data generating process:
the OR approach only relies on modeling the conditional expectation of the outcome evolution for the
comparison groups, the IPW approach relies on modeling the conditional probability of being in group
g, whereas the DR approach exploits both OR and IPW components.
   In order to extend the results of Heckman et al. (1997, 1998), Abadie (2005), and Sant’Anna and Zhao
(2020) to the multiple groups, multiple periods framework, we have to address two different challenges: one
associated with an appropriate reference time period and one associated with an appropriate comparison
group. Theorem 1 highlights how a solution to these challenges is directly connected to the limited
anticipation and the conditional parallel trends assumptions. More specifically, Theorem 1 says that we
can use the time period t = g − δ − 1 as an appropriate reference time period under Assumption 3 and
either Assumption 4 or 5. This is the most recent time period when untreated potential outcomes are
observed for units in group g. Interestingly, the more treatment anticipation is allowed (i.e., the higher
δ is), the further back in time one needs to go.9 Theorem 1 also suggests that the choice of comparison
group is directly tied to the conditional parallel trends assumption one makes: under Assumption 4, one
can use “never treated” units as a fixed comparison group for all “eventually treated” units; whereas,
under Assumption 5, one can use the “not-yet-treated by time t + δ” units as a valid comparison group
for those who are first treated at time g. In this latter case, Theorem 1 also highlights that when all units
eventually gets treated (ḡ < ∞), one is only able to identify the AT T (g, t)’s for time periods before the
last treated group “effectively” starts their treatment, i.e., t < ḡ − δ. In this case, one can not identify
the AT T (g, t) for the last treated cohort, too.
   Finally, we note that when pre-treatment covariates play no role in identification (i.e., Assumptions
3, 4, and 5 hold unconditionally on X), (2.2)-(2.4) collapse to

                            nev
                        AT Tunc (g, t; δ) = E[Yt − Yg−δ−1 |Gg = 1] − E[Yt − Yg−δ−1 |C = 1],                   (2.8)

and (2.5)-(2.7) collapse to

                          ny
                      AT Tunc (g, t; δ) = E[Yt − Yg−δ−1 |Gg = 1] − E[Yt − Yg−δ−1 |Dt+δ = 0].                  (2.9)

   9
       As mentioned in Remark 1, as one allows δ to increase, Assumptions 4 and 5 becomes more restrictive.

                                                            11

These expressions for AT T (g, t) clearly resemble the one for AT T in the canonical two-periods and two-
groups case. As in that case, the average effect of participating in the treatment for units in group g is
identified by taking the path of outcomes (i.e., the change in outcomes between the most recent period
before they were affected by the treatment and the current period) actually experienced by that group
and adjusting it by the path of outcomes experienced by a comparison group. Under the parallel trends
assumption, this latter path is the path of outcomes that units in group g would have experienced if they
had not participated in the treatment.

Remark 3. From (2.8) one can see that when Assumptions 3 and 4 hold unconditionally and there is
no-anticipation, the AT T (g, t) parameter can be obtained by first subsetting the data to only contain
observations at time t and g − 1, from units with either Gg = 1 or C = 1, and then, using only the
observations of this subset, running the (population) linear regression

                   Y = α1g,t + α2g,t · Gg + α3g,t · 1 {T = t} + β g,t · (Gg × 1 {T = t}) + g,t .           (2.10)

It is then easy to verify that β g,t = AT T (g, t). Note that one would need to consider different partitions
of the data to characterize different AT T (g, t) in terms of regression parameters. Alternatively, one could
use the interacted two-way fixed effects regression proposed by Sun and Abraham (2020).

Remark 4. When covariates are available, the β̃ g,t coefficient of the population linear regression

               Y = α̃1g,t + α̃2g,t · Gg + α̃3g,t · 1 {T = t} + β̃ g,t · (Gg × 1 {T = t}) + γ̃ · X + ˜g,t

that uses the same subset of data as in Remark 3 is, in general, not equal to AT T (g, t) unless one is
willing to assume (i) homogeneous (in X) treatment effects, i.e., E[Yt (g) − Yt (0)|Gg = 1, X] = E[Yt (g) −
Yt (0)|Gg = 1] a.s., and (ii) rule-out covariate-specific trends, i.e., for E [Yt − Yt−1 |X, G] = E[Yt − Yt−1 |G]
a.s. for all groups and time periods; see, e.g., Sloczyński (2018) for a related discussion. The characteri-
zations of AT T (g, t) discussed in Theorem 1 do not rely on these restrictive conditions.

Remark 5. Although the IPW, OR, and DR based estimands presented in Theorem 1 are identical from
an identification standpoint, this is not the case when one wants to estimate and make inference about
the AT T (g, t). As we discuss in Section 4, DiD estimators based on the DR estimands (2.4) and (2.7)
usually enjoy additional robustness against model-misspecifications when compared to the IPW and OR
estimands.

Remark 6. Theorem 1 suggests that we can identify AT T (g, t) only for groups in Gδ ⊆ G which can
involve dropping some “early treated” groups due to anticipation effects. When δ = 0, i.e. when there is
no anticipation, Gδ = G. Theorem 1 also suggests that we can identify AT T (g, t) only until t = T − δ
because of potential treatment anticipation behavior. In applications where some units are known to never
participate in the treatment (including periods after time period T ), however, we can identify AT T (g, t)
up to t = T by using these units as a valid comparison group for all time periods t = T − δ + 1, . . . , T ,
provided that an appropriate parallel trends assumption is satisfied.

Remark 7. From Theorem 1 it is clear that pre-treatment covariates play a prominent role in our anal-
ysis. Importantly, Assumptions 4 and 5 suggest that researchers should include pre-treatment covariates

                                                          12

that are potentially associated with the outcome evolution of Y (0) during post-treatment periods. We ex-
plicitly rule out incorporating post-treatment covariates as they can potentially be affected by the treatment;
see, e.g., Wooldridge (2005b), for a related discussion under the unconfoundedness setup.

3    Summarizing Group-Time Average Treatment Effects
The previous section shows that we can identify the AT T (g, t)’s by restricting treatment anticipation
behavior and imposing a conditional parallel trends assumption. In many applications, the AT T (g, t)’s
can be the ultimate causal parameters of interest. They can be used to highlight treatment effect hetero-
geneity across different groups g, at different points in time t, and across different lengths of treatment
exposure, e = t − g. In other situations, however, researchers may want to combine these different
AT T (g, t)’s to form more aggregated causal parameters. For instance, if the number of groups and time
periods is relatively large, it may be challenging to interpret many group-time average treatment effects.
    In this section, we consider different aggregation schemes for the AT T (g, t)’s that allow researchers
to form a variety of summary measures of the causal effects of a given policy. Our aggregation schemes
are of the form
                                              T
                                             XX
                                       θ=              w (g, t) · AT T (g, t),                           (3.1)
                                             g∈G t=2

where w (g, t) are carefully-chosen (known or estimable) weighting functions specified by the researcher
such that θ can be used to address a well-posed empirical/policy question. Difference choices of w (g, t)
allows researchers to highlight different types of treatment effect heterogeneity. We pay particular at-
tention to aggregations that result in a single overall treatment effect summary parameter as well as to
aggregations related to understanding dynamic effects as is commonly done in event-study analysis. Of
course, many other aggregated parameters of the type (3.1) can be easily constructed following our frame-
work. We illustrate this point by also summarizing heterogeneity with respect to group or by calendar
time.
    Before proceeding with the discussion on how to construct these different aggregated parameters, it is
worth revisiting the two most popular treatment effect summary measures used by practitioners in DiD
setups. These are based on the “static” and “dynamic” two-way fixed effects (TWFE) linear regression
specifications

                          Yi,t = αt + αg + βDi,t + i,t ,                                                (3.2)
                                              −2
                                              X                          L
                                                                         X
                          Yi,t = αt + αg +          δeanticip · Di,t
                                                                 e
                                                                     +               e
                                                                               βe · Di,t + vi,t ,        (3.3)
                                             e=−K                        e=0

respectively, where αt is a time fixed effect, αg is a group fixed effect, i,t and vi,t are error terms,
 e = 1 {t − G = e} is an an indicator for unit i being e periods away from initial treatment at time
Di,t         i
t, and K and L are positive constants. The parameter of interest in the static TWFE specification is
β, which, in applications, is typically interpreted as an overall effect of participating in the treatment
across groups and time periods. In the dynamic TWFE specification, practitioners usually focus on the
βe , e ≥ 0, and these parameters are typically interpreted as measuring the effect of participating in the

                                                          13

treatment at different lengths of exposure to the treatment.
   Despite the popularity of these specifications, recent research has shown that one must be very careful
in attaching a causal interpretation to these aggregated parameters. For instance, Borusyak and Jaravel
(2017), Goodman-Bacon (2019), de Chaisemartin and D’Haultfœuille (2020), and Athey and Imbens
(2018) have shown that, in general, β recovers a weighted average of some underlying treatment effect
parameters but some of the weights on these parameters can be negative. This can potentially lead to
particularly problematic cases such as the effect of the treatment being positive for all units, but the
TWFE estimation resulting in estimates of β that are negative. Even in cases where the weights are
not negative, the weights on underlying treatment effect parameters are entirely driven by the TWFE
estimation strategy and are sensitive to the size of each group, the timing of treatment, and the total
number of time periods (see Theorem 1 in Goodman-Bacon (2019)). The results in this section can be
used in exactly the same setup to identify a single interpretable average treatment effect parameter and,
thus, provide a way to circumvent the issues with the more common approach.
   As discussed by Goodman-Bacon (2019), the “negative weight problem” associated with β arises when
treatment effects evolve over time. Thus, one may wonder if such problems would still be present when
considering more general, dynamic specifications such as (3.3). Sun and Abraham (2020) shows that this
is still the case as the βe0 s associated with (3.3) do not recover easy-to-interpret causal parameters and
still generally suffer from the same sorts of “negative weighting problems.” In contrast to this, we provide
a simple way to directly aggregate our group-time average treatment effects into average treatment effects
across different lengths of exposure to the treatment.

3.1     Aggregations to Highlight Treatment Effect Heterogeneity
Next, we discuss several partial aggregations of the group-time average treatment effects in order to sum-
marize different dimensions of treatment effect heterogeneity. Although there are additional possibilities,
we focus our discussion below on how to answer three particular questions: (a) How does the effect of
participating in the treatment vary with length of exposure to the treatment? (b) Do groups that are
treated earlier have, on average, higher/lower average treatment effects than groups that are treated
later? (c) What is the cumulative average treatment effect of the policy across all groups until some
particular point in time? Throughout this section, to avoid notation clutter, we assume that units do not
anticipate treatment, i.e., we consider the case where Assumption 3 holds with δ = 0. We also assume
that a “never treated” group is available.

3.1.1    How do average treatment effects vary with length of exposure to the treatment?

One of the most popular questions that arises in DiD setups with multiple time periods concerns treatment
effect dynamics: How does the effect of participating in the treatment vary with length of exposure to the
treatment? For instance, do average treatment effects increase/decrease with elapsed treatment time?
Indeed, answering this type of question is often the main motivation for using the event study regression
in (3.3), though, as we mentioned above, that sort of regression may not be suitable for such a task. In
this section, we propose an aggregation scheme that is suitable to highlight treatment effect heterogeneity
with respect to length of exposure to the treatment that does not suffer from the drawbacks associated

                                                    14

                             Table 1: Weights on AT T (g, t) for Aggregated Parameters

        Parameter                                                      w(g, t)

             θes (e)           wees (g, t) = 1{g + e ≤ T }1{t − g = e}P (G = g|G + e ≤ T )
          bal
         θes  (e, e0 )     wees,bal (g, t) = 1{g + e0 ≤ T }1{t − g = e}P (G = g|G + e0 ≤ T )
            θsel (g̃)            wg̃s (g, t) = 1{t ≥ g}1 {g = g̃}/ (T − g + 1)
                                 wt̃c (g, t) = 1{t ≥ g}1 t = t̃ P (G = g|G ≤ t)
                                                         
               θc (t̃)
          θccumu (t̃)     wt̃c,cumu (g, t) = 1{t ≥ g}1 t ≤ t̃ P (G = g|G ≤ t)
                                                         

                  O              O
                                      (g, t) = 1{t ≥ g}P (G = g|G ≤ T )/ g∈G Tt=2 1{t ≥ g}P (G = g|G ≤ T )
                                                                          P    P
                 θW             wW
                  O            O
                 θsel         wsel (g, t) = 1{t ≥ g}P (G = g|G ≤ T )/ (T − g + 1)
        Notes: This table provides expressions for the weights on each AT T (g, t) (as in Equation 3.1) for each
        parameter discussed in this section. In all cases except for θccumu (t̃), the weights are all non-negative and sum to
        one. For θccumu (t̃), the
                                 weights are all non-negative but sum up to t̃ − 1 (rather than one), but this is just a
        reflection of θccumu t̃ being a cumulative treatment effect measure.

with the event study regression in (3.3).
    Let e denote event-time, i.e., e = t − g denotes the time elapsed since treatment was adopted. Recall
that G denotes the time period that a unit is first treated. Thus, a way to aggregate the AT T (g, t)’s to
highlight treatment effect heterogeneity with respect to e is
                                        X
                            θes (e) =         1{g + e ≤ T }P (G = g|G + e ≤ T )AT T (g, g + e).                             (3.4)
                                        g∈G

This is the average effect of participating in the treatment e time periods after the treatment was adopted
across all groups that are ever observed to have participated in the treatment for exactly e time periods.
Here, the “on impact” average effect of participating in the treatment occurs for e = 0. θes (e) is the
natural target for event study regressions that are common in applied work, though it completely avoids
the pitfalls associated with the dynamic TWFE specification in (3.3).10
    In event study regressions, it is common to plot βe across different values of e and to interpret
differences as being due to treatment effect dynamics. Similarly, one can plot θes (e) across different
e’s to better understand treatment effect dynamics. When doing so, it is important to be aware that
these comparisons may include compositional changes that can complicate the interpretation of these
parameters (note that the same complications arise for event study regressions as well). To see this, for

   10
      Many of the parameters in this section involve expressions that have similar components as the one for θes (e) in (3.4),
and it is worth mentioning a few extra details for this case that are common to the other expressions below. The term
involving the indicator function, 1{g + e ≤ T }, limits consideration to identified group-time average treatment effects. The
summation over groups with group specific weights, in this case given by P (G = g|G + e ≤ T ), calculates an average,
weighted by group size, of AT T (g, t)’s that are involved in a particular aggregation. In addition, it is straightforward to
show that θes (e) can be written in the form of θ in Equation (3.1). Throughout this section, we have written each parameter
of interest in its most intuitive form. Weights for each parameter in this section corresponding to the form of the weights in
Equation (3.1) are provided in Table 1.

                                                                15

0 ≤ e1 < e2 ≤ T − 2, consider the difference between θes (e2 ) and θes (e1 ) which is given by
                           X
 θes (e2 ) − θes (e1 ) =         1{g + e1 ≤ T }P (G = g|G + e1 ≤ T )(AT T (g, g + e2 ) − AT T (g, g + e1 ))              (3.5)
                                                                    |                  {z                 }
                           g∈G
                                                                                          dynamic effect for group g
                               X
                           +         1{g + e2 ≤ T }(P (G = g|G + e2 ≤ T ) − P (G = g|G + e1 ≤ T ))AT T (g, g + e2 )
                                                   |                      {z                     }
                               g∈G
                                                                            differences in weights
                               X
                           −         1{T − e2 < g ≤ T − e1 }P (G = g|G + e1 ≤ T )AT T (g, g + e2 ).
                                     |         {z          }
                               g∈G
                                     different composition of groups

From the above decomposition it becomes clear that comparing θes (e) at two different values of e provides
a weighted average of the dynamic effect of participating in the treatment – the first component on the
right-hand side of (3.5) – plus two extra undesirable terms. Both of these undesirable terms are due
to different compositions of groups at different event times.11 The first term arises because the weights
at each length of exposure differ due to the changing composition of groups at each event time. The
second term comes directly from different compositions of groups at each length of exposure. These two
additional terms may prevent one from interpreting the differences in θes (e) across different values of e
as being actual dynamic effects of participating in the treatment unless one is willing to impose that
AT T (g, g + e) does not vary with g for any e ≥ 0; i.e., that dynamic effects are common across groups.12
However, this sort of homogeneity condition may be deemed too strong in many applications.
    A simple alternative causal parameter that can be used to highlight treatment effect dynamics with
respect to length of exposure to the treatment and does not suffer from the issue of compositional changes
highlighted in (3.5) arises from “balancing” the groups with respect to event time, i.e., to only aggregate
the AT T (g, t)’s for a fixed set of groups that are exposed to the treatment for at least some particular
number of time periods and thereby circumvent the issue of compositional changes across different values
of e. In particular, for some event time e0 with 0 ≤ e ≤ e0 ≤ T − 2, let
                                            X
                          bal
                         θes  (e; e0 ) =          1{g + e0 ≤ T }AT T (g, g + e)P (G = g|G + e0 ≤ T ).                    (3.6)
                                            g∈G

                               bal (e; e0 ) is very similar to θ (e) except that it calculates the average group-
Notice that the definition of θes                               es
time average treatment effect for units whose event time is equal to e and who are observed to participate
in the treatment for at least e0 periods. In this case, since the composition of groups is the same across
                                                                                         bal (e; e0 ) across
all values of e, the additional terms in (3.5) do not show up at all and differences in θes
different values of e cannot be due to differences in the composition of groups at different values of e. As
an example, when one is interested in analyzing the evolution of treatment effects up to 5 periods after
treatment was implemented, one can set e0 = 5 and, this way, the same groups of units will be used when
           bal (0; 5), θ bal (1; 5), . . . , θ bal (5; 5).
computing θes           es                    es
    The price one pays for “balancing” the groups with respect to event time is that fewer groups are
used to compute these event-study-type estimands, which can lead to less informative inference. Thus, in

   11
      The composition changes mentioned here arise due to the staggered adoption of the treatment. For example, when
T = 3, groups 2 and 3 both show up in the expression for θes (0), but only group 2 shows up in the expression for θes (1).
   12
      If AT T (g, g + e) does not vary with g for any e ≥ 0, it is straightforward to show that the last two terms of (3.5) sum
up to 0.

                                                                       16

                                                                                                     bal
practice, one should consider this “robustness” versus “efficiency” trade-off when choosing between θes
and θes .
                        bal (e; e0 ) closely resembles the empirical practice of only reporting event-study-
Remark 8. We note that θes
type coefficients for the event periods that do not suffer from compositional changes, see, e.g., McCrary
(2007) and Bailey and Goodman-Bacon (2015). An important caveat is that our proposed event-study-type
           bal (e; e0 ) are not based on dynamic TWFE specifications akin to (3.3), and therefore bypass
estimands θes
the pitfalls associated with (3.3) highlighted by Sun and Abraham (2020).

3.1.2       How do average treatment effects vary across groups?

It is also straightforward to aggregate our group-time average treatment effects to understand hetero-
geneity in the effect of participating in the treatment across groups. Although understanding this sort
of heterogeneity is relatively less common in applied work than trying to understand dynamic effects as
discussed above, there are still a number of cases in economics where understanding this sort of hetero-
geneity may be of interest. For example, work on the effect of graduating during a recession on labor
market outcomes (Oreopoulos et al. (2012)) or the effect of job displacement across the business cycle
(Farber (2017)) are related to heterogeneous effects across groups. More generally, these parameters are
useful for understanding if the effect of participating in the treatment was larger for groups that are
treated earlier relative to groups that are treated later. In addition, in the next section, these parameters
will be the building block for our main measure of the overall effect of participating in the treatment. To
consider heterogeneous effects across groups, we consider the following parameter
                                                                T
                                                       1       X
                                     θsel (g̃) =                 AT T (g̃, t).                          (3.7)
                                                    T − g̃ + 1
                                                               t=g̃

θsel (g̃) is the average effect of participating in the treatment among units in group g̃, across all their
post-treatment periods.

3.1.3       What is the cumulative average treatment effect of the policy across all groups until
            time t̃?

In some applications, researchers may want to construct an aggregated target parameter to highlight
treatment effect heterogeneity with respect to calendar time. In economics, for example, researchers
might wish to study heterogeneous treatment effects across the business cycle. The average effect of
participating in the treatment in time period t (across groups that have adopted the treatment by period
t) is given by
                                          X
                              θc (t̃) =         1{t̃ ≥ g}P (G = g|G ≤ t̃)AT T (g, t)                    (3.8)
                                          g∈G

An extension to this parameter is to think about the cumulative effect of participating in the treatment
up to some particular time period. For instance, in active labor market applications, policy makers may
want to know the cumulative average effect of a given training program on earnings from the year that
the first group of people were trained until year t̃. This would provide a measure of the cumulative

                                                          17

earnings gains induced by the training program. Alternatively, in health applications, researchers may
want to measure how many COVID-19 cases have been averted by shelter-in-place orders up to day t̃.
To consider the cumulative effect, consider the following parameter

                                                                 t̃
                                                                 X
                                             θccumu
                                                        
                                                       t̃ =            θc (t).                         (3.9)
                                                                 t=2

θccumu t̃ can be interpreted as the cumulative average treatment effect among the units that have been
         

treated by time t̃.

3.2   Aggregations into Overall Treatment Effect Parameters
Finally in this section, we consider some ideas for aggregating group time average treatment effects into
an overall effect of participating in the treatment. One very simple idea is to just average all of the
identified group-time average treatment effects together; i.e., to consider the parameter
                                        T
                            O     1 XX
                           θW =        1{t ≥ g}AT T (g, t)P (G = g|G ≤ T )                           (3.10)
                                  κ
                                    g∈G t=2

            P       PT
where κ =     g∈G t=2 1{t ≥ g}P (G = g|G ≤ T ) (which ensures that the weights on AT T (g, t) in the sec-
                          O is a weighted average of each AT T (g, t) putting more weight on AT T (g, t)’s
ond term sum up to one). θW
with larger group sizes. Unlike β in the TWFE regression specification (3.2), this simple combination of
AT T (g, t)’s immediately rules out troubling issues due to negative weights; as a particular example, when
the effect of participating in the treatment is positive for all units, this aggregated parameter cannot be
negative.
   That being said, just requiring positive weights is a very minimal requirement of a reasonable overall
                                                          O is that it systematically puts more weight
treatment effect parameter. For example, one drawback of θW
on groups that participate in the treatment for longer. Instead, we suggest the following parameter as a
general-purpose summary of the average effect of participating in the treatment
                                              X
                                     O
                                    θsel =          θsel (g)P (G = g|G ≤ T )                         (3.11)
                                              g∈G

where θsel (g) is the average effect of participating in the treatment for units in group g as defined in
                       O first computes the average effect for each group (across all time periods) and
Equation (3.7) above. θsel
then averages these effects together across groups to summarize the overall average effect of participating
                         O is the average effect of participating in the treatment experienced by all
in the treatment. Thus, θsel
units that ever participated in the treatment. In this respect, its interpretation is the same as the ATT in
the canonical DiD setup with two periods and two groups. This is an attractive property for a summary
measure of the overall effect of participating in the treatment in the context of multiple time periods and
variation in treatment timing.
   Working by analogy, one can also define overall treatment effect parameters by averaging θes (e) across

                                                            18

all event times or θc (t) across all time periods, i.e.,
                                          T −2                                     T
                                      1 X                                      1 X
                             O
                            θes =         θes (e)                    θcO =         θc (t)             (3.12)
                                    T −1                                     T −1
                                          e=0                                     t=2

                                                                                                  O
In our view, the appeal of these aggregations is likely to be somewhat more limited than that of θsel
                                                          O is complicated by the issue of the changing
in most applications. For example, the interpretation of θes
composition of groups across different values of e discussed above (similar arguments apply to θcO as well).
As before, one can circumvent the issue of the changing composition of groups by balancing the sample
with respect to event time. A (local) single summary parameter is given by

                                                                e0
                                         O,bal 0          1 X bal
                                        θes   (e ) =          θes (e, e0 )                            (3.13)
                                                       e0 + 1
                                                                e=0

This is the average effect of participating in the treatment over the first e0 periods of exposure to the
treatment. This is also a reasonable alternative overall treatment effect parameter, but it should also be
noted that it is local to groups that participated in the treatment for at least e0 periods.
    As a final comment, in general, none of the overall effect parameters considered in this section are
equal to each other except in the special case where AT T (g, t) is the same for all groups and all time
periods. In that case, all of the aggregated parameters, including β from the TWFE regression, are equal
to each other.

4    Estimation and Inference
So far we have focused on the identification and aggregation stages of the analysis. In this section,
we show how one can build on these results to form estimators for and conduct inference about the
group-time average treatment effects and their summary measures described in Section 3. Given that the
AT T (g, t)’s are the main building blocks of our analysis, we start with them.
    First, it is important to notice that our identification results in Theorem 1 are constructive and
suggest a simple and intuitive two-step estimation strategy to estimate the AT T (g, t)’s. In the first step,
one estimates the nuisance functions for each group g and time period t — pg (x) and/or mnev
                                                                                         g,t,δ (X) if
one relies on Assumption 4, and pg,t+δ (x) and/or mny
                                                   g,t,δ (X) if one relies on Assumption 5. In the second
step, one plugs the fitted values of these estimated nuisance functions into the sample analogue of the
considered AT T (g, t) estimand to obtain estimates of the group-time average treatment effect.
    A natural question that then arises is which type of approach one should use in practice: the out-
come regression, inverse probability weighting, or the doubly-robust one. Although these three differ-
ent approaches are equivalent from the identification perspective, this is not the case from the esti-
mation/inference perspective. The OR approach requires researchers to correctly model the outcome
evolution of the comparison group to estimate the group-time average treatment effects. This approach
is explicitly connected with the conditional parallel trends assumption required in DiD analysis as this
condition is usually expressed in terms of conditional expectations. The IPW approach, on the other
hand, avoids explicitly modeling the outcome evolution of the comparison group and therefore does not

                                                           19

rely on putative model restrictions directly tied to the parameter of interest. Instead, the IPW approach
requires one to correctly model the conditional probability of unit i being in group g given their covari-
ates X and that they are either in group g or in an appropriate comparison group. The DR approach
combines both the OR and IPW approaches as it relies on modeling both the outcome evolution and the
propensity score. However, it only requires one to correctly specify either (but not necessarily both) the
outcome evolution for the comparison group or the propensity score model (Sant’Anna and Zhao, 2020).
Thus, the DR approach enjoys additional robustness against model misspecifications when compared to
the OR and IPW approaches. In addition, the DR approach potentially allows one to use a broader set
of estimation methods such as those that involve penalization and some types of model selection, see, e.g.
Belloni et al. (2017).
    Given these attractive robustness features associated with the DR approach, in this section we consider
estimators of the DR form; the discussion on how to proceed with the OR and IPW approaches is
analogous and therefore omitted. We also focus on parametric estimators for the nuisance functions. We
consider this case mainly for its practical appeal which is especially true in applications where the number
of covariates is fairly large and the number of observations is only moderate.13
    More concisely, let

                     nev                 h                                                   i
               AT
               [  T dr (g, t; δ) = En   bgtreat − w
                                        w                                  b nev
                                                  bgcomp,nev Yt − Yg−δ−1 − m         X; bnev
                                                                                        β            ,                    (4.1)
                                                                             g,t,δ        g,t,δ
                     ny               h                                                     i
                                        bgtreat − w                       b ny
                                                  bgcomp,ny Yt − Yg−δ−1 − m            bny
                                                           
                AT
                [  T dr (g, t; δ) = En w                                    g,t,δ X; βg,t,δ        ,                      (4.2)

where
                                        pbg (X; πbg ) C                       pbg,t+δ (X; πbg,t+δ ) (1 − Dt+δ ) (1 − Gg )
              Gg                       1 − pbg (X; π bg )                                1 − pbg,t+δ (X; π
                                                                                                         bg,t+δ )
  bgtreat =
  w                    bgcomp,nev =
                     , w                                      bgcomp,ny =
                                                            , w                                                           ,
            En [Gg ]                       pbg (X; π
                                                   bg ) C                                    bg,t+δ ) (1 − Dt+δ ) (1 − Gg )
                                                                                 pbg,t+δ (X; π
                                    En                                     En
                                          1 − pbg (X; πbg )                                1 − pbg,t+δ (X; π
                                                                                                           bg,t+δ )

                                         mnev        bnev         b ny        bny
with pbg (·; π
             bg ), pbg,t+δ (·; π
                               bg,t+δ ), b g,t,δ (·; βg,t,δ ) and m g,t,δ (·; βg,t,δ ) being (parametric) estimators of pg (·),
                                                                                                                   nev
pg,t+δ (·), mnev (·) and mny (·), respectively, and for a generic Z, En [Z] = n−1 n Zi . AT
                                                                                                   P
             g,t,δ           g,t,δ                                                                  i=1
                                                                                                              [  T dr (g, t; δ)
           nev
and AT
    [  T dr (g, t; δ) are our proposed DR DiD estimators for AT T (g, t) when one invokes Assumption
4 and Assumption 5, respectively. These estimators extend the DR DiD estimators of Sant’Anna and
Zhao (2020) from the two periods, two groups setup to the multiple groups, multiple periods setup while
allowing for possible treatment anticipation. In addition, these estimators are of the Hájek (1971)-type
and their associated weights are guaranteed to sum up to one in finite samples. As illustrated by Busso
et al. (2014), this usually leads to improved finite sample properties.
    With the estimators for the AT T (g, t)’s in hand, one can use the analogy principle and combine these

   13
     Alternatively, one could adopt a fully nonparametric approach. Let f (x) be a generic notation for the nuisance
functions. From Newey (1994), Chen et al. (2003), Ai and Chen (2003, 2007, 2012), and Chen et al. (2008), onecan see
                                                                                                                    
                                                                                                g (x) − g (x)kH = op n−1/4
that the use of nonparametric first-step estimators gb (x) of g (x) is warranted provided that kb
for a pseudo-metric k·kH , H being a vector space of functions. However, when the dimension of X is moderate or large, as
                                                                       g (x) − g (x)kH = op n−1/4 can be rather stringent
is often the case in empirical applications, conditions ensuring that kb
due to the so-called “curse of dimensionality”.

                                                              20

to estimate the summarized average treatment effect parameters discussed in Section 3.

Remark 9. In applications with limited covariate overlap (i.e., with propensity scores sufficiently close
to one), IPW and DR estimators may lead to imprecise (irregular) inference procedures, see, e.g., Khan
and Tamer (2010). In such cases, provided that one is comfortable with (parametric) extrapolation and
is sufficiently confident that the outcome regression working models are correctly specified, relying on the
OR estimation approach may lead to more informative inferences. Alternatively, one may choose to trim
extreme propensity score estimates though proceeding in this manner would change the target parameter;
i.e., we would not be recovering the AT T (g, t)’s; see, e.g., Crump et al. (2009) and Yang and Ding (2018)
for related discussion in other contexts. In the rest of the paper, we abstract from these points.

4.1    Asymptotic Theory for Group-Time Average Treatment Effects
Next, we derive the asymptotic properties of our DR DiD estimators for the AT T (g, t)’s. To simplify
exposition, we focus on the case with a never-treated comparison group as in (4.1); results that come from
using the not-yet-treated group as the comparison group as in (4.2) follow from symmetric arguments
and are therefore omitted. We also note that the theoretical results in this section are justified within
the large n, fixed T paradigm.
                p
   Let kZk = trace (Z 0 Z) denote the Euclidean norm of Z and set W = (Y1 , . . . , YT , X, D1 , . . . , DT ).
                                0
For a generic κnev      0   nev
                g,t = πg , βg,t,δ , let

         hdr,nev W ; κnev        treat
                                       (W ) − wgcomp,nev (W ; πg ) Yt − Yg−δ−1 − mnev       nev
                                                                                                
          g,t         g,t , δ = wg                                                g,t,δ X; βg,t,δ    ,

where the normalized weights wgtreat (W ) and wgcomp,nev (W ; πg ) are given by

                    
                         Gg                                          pg (X; πg ) C           pg (X; πg ) C
       wgtreat (W ) =           ,          wgcomp,nev (W ; πg ) =                         E                   .   (4.3)
                        E [Gg ]                                     1 − pg (X; πg )         1 − pg (X; πg )

Let g (·) be a generic notation for pg (·) and mnev      g,t,δ (·). With some abuse of notation, let g (·; γ) be a
                                                         
                                             nev     nev
generic notation for pg (·; πg ) and mg,t,δ ·; βg,t,δ . The vector of pseudo-true parameters is given by
                        0                                                        . nev
 ∗,nev           ∗,nev 0
       = πg∗0 , βg,t,δ      . Finally, let ḣdr,nev      nev = ∂ hdr,nev W ; κnev
                                                              
κg,t                                         g,t    W ; κg,t         g,t       g,t   ∂κg,t .

Assumption 7. (i) g (x; γ) is a parametric model for g (x), where γ ∈ Θ ⊂ Rk , Θ being compact; (ii)
g (X; γ) is a.s. continuous at each γ ∈ Θ; (iii) there exists a unique pseudo-true parameter γ ∗ ∈ int (Θ);
(iv) g (X; γ) is a.s. twice continuously differentiable in a neighborhood of γ ∗ , Θ∗ ⊂ Θ; (v) the estimator
b is strongly consistent for γ ∗ and satisfies the following linear expansion:
γ
                                                           n
                                    √           ∗  1 X
                                           γ−γ )= √
                                        n (b           lg,t (Wi ; γ ∗ ) + op (1) ,
                                                    n
                                                          i=1

                                                          ∗                     ∗             ∗ 0
                                                                                                 
where lg,t (·; γ) is a k × 1 vector
                                h such that E [lg,t (W ; γ )] = 0, E lg,t (W
                                                                           i ; γ ) lg,t (W ; γ ) exists and is
positive definite and lims→0 E supγ∈Θ∗ :kγ−γ ∗ k≤s klg (W ; γ) − lg (W ; γ ∗ )k2 = 0. In addition, (vi) for some
ε > 0 and all g ∈ G, 0 ≤ pg (X; πg ) ≤ 1 − ε a.s., for all π ∈ int (Θps ), where Θps denotes the parameter
space of πg .

                                                           21

                                                                       h                        i
                                                                                      ∗,nev  2
Assumption 8. For each g ∈ G and t = {2, . . . , T − δ}, assume that E hnev g,t W ; κ g,t  , δ    < ∞
     h                     i
                                        ∗                             ∗,nev
and E supκ∈Γ∗ ḣnev
                g,t (W ; κ) < ∞, where Γ is a small neighborhood of κg,t .

   Assumptions 7-8 are standard in the literature, see e.g. Abadie (2005), Wooldridge (2007), Bonhomme
and Sauder (2011), Graham et al. (2012), and Sant’Anna and Zhao (2020). Assumption 7 requires that the
                                                                                                   √
first-step estimators are based on smooth parametric models and that the estimated parameters admit n-
asymptotically linear representations, whereas Assumption 8 imposes some weak integrability conditions.
Under mild moment conditions, these requirements are fulfilled when one adopts linear/nonlinear outcome
regressions or logit/probit models, for example, and estimates the unknown parameters by (nonlinear)
least squares, quasi-maximum likelihood, or other alternative estimation methods, see e.g. Chapter 5 in
van der Vaart (1998), Wooldridge (2007), Graham et al. (2012) and Sant’Anna and Zhao (2020). In
other words, Assumptions 7-8 allow for flexible parametric specifications of the nuisance functions and
accommodate different estimation methods.                                                                 
   In what follows, we write wgtreat = wgtreat (W ), wgcomp (πg ) = wgcomp,nev (W ; πg ), and mnev    nev
                                                                                               g,t,δ βg,t,δ =
                                                                          0
mnev     X; β nev    to minimize notation. For a generic κnev = π 0 , β nev 0 , define
 g,t,δ       g,t,δ                                        g,t      g g,t,δ

        dr,nev                 treat,nev        nev        comp,nev             nev        est,nev             nev
       ψg,t,δ  (Wi ; κnev
                      g,t ) = ψg,t,δ     (Wi ; βg,t,δ ) − ψg,t,δ    (Wi ; πg , βg,t,δ ) − ψg,t,δ   (Wi ; πg , βg,t,δ ),     (4.4)

with

                treat,nev                                                     nev
                                nev
                                      ) = wgtreat · Yt − Yg−δ−1 − mnev
                                                                                   
             ψg,t,δ       (W ; βg,t,δ                                 g,t,δ βg,t,δ
                                                                                                  nev
                                              − wgtreat · E wgtreat · Yt − Yg−δ−1 − mnev
                                                                                                       
                                                                                           g,t,δ βg,t,δ     ,
          comp,nev                                                                  nev
                                nev
                                      ) = wgcomp (πg ) · Yt − Yg−δ−1 − mnev
                                                                                         
         ψg,t,δ      (W ; πg , βg,t,δ                                       g,t,δ βg,t,δ
                                                                                                              nev
                                              − wgcomp (πg ) · E wgcomp (πg ) · Yt − Yg−δ−1 − mnev
                                                                                                                   
                                                                                                       g,t,δ βg,t,δ     ,

and
                                                                   0
                     est,nev
                    ψg,t                nev
                             (W ; πg , βg,t,δ      or,nev
                                              ) = lg,t     nev
                                                          βg,t,δ           dr,nev,1
                                                                        · Mg,t,δ    + lgps,nev (πg )0 · Mg,t,δ
                                                                                                         dr,nev,2
                                                                                                                  ,
       or,nev
where lg,t    (·) is the asymptotic linear representation of the estimator for the outcome evolution of the
comparison groups as described in Assumption 7(iv), lgps,nev (·) is defined analogously for the generalized
propensity score, and

             dr,nev,1
                                 wgtreat − wgcomp (πg ) · ṁnev    nev
                                                                       
            Mg,t,δ    =E                                    g,t,δ βg,t,δ    ,
             dr,nev,2
                      = E αgps,nev (πg ) · Yt − Yg−δ−1 − mnev        nev
                                                                                      
            Mg,t,δ                                            g,t,δ βg,t,δ    · ṗg (πg )
                           − E αgps,nev (πg ) · wgcomp (πg ) · Yt − Yg−δ−1 − mnev            nev
                                                                                                              
                                                                                      g,t,δ βg,t,δ    · ṗg (πg ) ,
                                        .
with ṁnev
       g,t,δ   β nev
                 g,t,δ   = ∂mnev X; β nev
                             g,t,δ   g,t,δ
                                                nev , ṗ (π ) = ∂p (X; π )/ ∂π , and
                                              ∂βg,t,δ   g  g      g     g     g

                                                                           ,                  
                                                            C                   pg (X; πg ) C
                                 αgps,nev (πg ) =                           E                    .
                                                    (1 − pg (X; πg ))2         1 − pg (X; πg )

                                               dr,nev                                                          nev
   Finally, let AT Tt≥(g−δ) and AT
                                [  T t≥(g−δ) denote the vector of AT T (g, t) and AT
                                                                                  [  T dr (g, t; δ), respec-

                                                                22

tively, for all g ∈ Gδ , t ∈ {2, . . . T − δ} such that t ≥ g − δ. Analogously, let Ψdr,nev
                                                                                     t≥(g−δ) denote the collection
    dr,nev
of ψg,t,δ  across all g ∈ Gδ , t ∈ {2, . . . T − δ} such that t ≥ g − δ. Consider the following claim:

                           For each g ∈ Gδ , t ∈ {2, . . . T − δ} such that t ≥ g − δ,
                           ∃πg∗ ∈ Θps : P pg (X; πg∗ ) = pg (X) = 1 or
                                                                   
                                                                                                            (4.5)
                                                                                  
                              ∗,nev                             ∗,nev
                           ∃βg,t,δ  ∈ Θreg : P mnev
                                                  g,t,δ    X; βg,t,δ    = m nev
                                                                            g,t,δ (X)  = 1.

Claim (4.5) says that either the working parametric model for the generalized propensity score is correctly
specified, or the working outcome regression model for the comparison group is correctly specified.
                                                                        dr,nev
   The next theorem establishes the joint limiting distribution of AT
                                                                   [  T t≥(g−δ) .

Theorem 2. Under Assumptions 1-4, 6-8, for each g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ} and
t ≥ g − δ, provided that (4.5) is true,
                                                                  n
                  √          nev                            1 X dr,nev
                      n(AT
                        [  T dr (g, t; δ) − AT T (g, t)) = √   ψg,t,δ (Wi ; κ∗,nev
                                                                             g,t ) + op (1).
                                                             n
                                                                 i=1

Furthermore, as n → ∞,
                                  √          dr,nev                    d
                                      n(AT
                                        [  T t≥(g−δ) − AT Tt≥(g−δ) ) −
                                                                     → N (0, Σ)

where Σ = E[Ψdr,nev       dr,nev      0
             t≥(g−δ) (W )Ψt≥(g−δ) (W ) ].

   Theorem 2 provides the influence function for estimating the vector of group-time average treatment
effects, AT Tt≥(g−δ) , as well as its limiting distribution. Importantly, Theorem 2 emphasizes the DR
                  nev
property of AT
             [  T dr (g, t; δ): it recovers the AT T (g, t) provided that either the propensity score working
model or outcome regression working model for the “never treated” is correctly specified.
   In order to conduct inference, one can show that the sample analogue of Σ is a consistent estimator
for Σ, which leads directly to standard errors and pointwise confidence intervals. Instead of following this
route, we propose to use a simple multiplier bootstrap procedure to conduct asymptotically valid inference.
Our proposed bootstrap leverages the asymptotic linear representations derived in Theorem 2 and inherits
important advantages. First, it is easy to implement and very fast to compute. Each bootstrap iteration
simply amounts to “perturbing” the influence function by a random weight V , and it does not require
re-estimating the propensity score in each bootstrap draw. Second, in each bootstrap iteration, there are
always observations from each group. This can be a real problem with the traditional empirical bootstrap
where there may be no observations from a particular group in some particular bootstrap iteration.
Third, computation of simultaneously (in g and t) valid confidence bands is relatively straightforward.
This is particularly important since researchers are likely to use confidence bands to visualize estimation
uncertainty about AT T (g, t) . Unlike pointwise confidence bands, simultaneous confidences bands do not
suffer from multiple-testing problems and are guaranteed to cover all AT T (g, t)’s with a probability at
least 1 − α. Finally, we note that our proposed bootstrap procedure can be readily modified to account
for clustering, see Remark 10 below.
    To proceed, let Ψb dr,nev (W ) denote the sample-analogue of Ψdr,nev (W ), where population expec-
                       t≥(g−δ)                                    t≥(g−δ)
tations are replaced by their empirical analogue, and the true nuisance functions and their derivatives
are replaced by their estimators. Let {Vi }ni=1 be a sequence of iid random variables with zero mean,

                                                         23

unit variance, and finite third moment, independent of the original sample {Wi }ni=1 . A popular exam-
                                                                     √                      √
ple involves iid Bernoulli variates {Vi } with P (V = 1 − κ) = κ/ 5 and P (V = κ) = 1 − κ/ 5, where
      √      
κ=      5 + 1 /2, as suggested by Mammen (1993).
                    ∗,dr,nev                           dr,nev
    We define AT
               [  T t≥(g−δ) , a bootstrap draw of AT
                                                  [  T t≥(g−δ) , via

                                      ∗,dr,nev       dr,nev      h                 i
                                 AT
                                 [  T t≥(g−δ) = AT
                                                [                     b dr,nev (W ) .
                                                   T t≥(g−δ) + En V · Ψ                                                 (4.6)
                                                                        t≥(g−δ)

The next theorem establishes the asymptotic validity of the multiplier bootstrap procedure proposed
above.

Theorem 3. Under the assumptions of Theorem 2
                         √
                                                       
                                   ∗,dr,nev      dr,nev   d
                           n AT T t≥(g−δ) − AT T t≥(g−δ) → N (0, Σ),
                              [             [
                                                                               ∗

                                             d
where Σ is as in Theorem 2, and → denotes weak convergence (convergence in distribution) of the bootstrap
                                             ∗
law in probability, i.e., conditional on the original sample {Wi }ni=1 . Additionally, for any continuous
functional Γ(·),14

                                       √
                                                                       
                                                  ∗,dr,nev       dr,nev     d
                               Γ           n AT
                                              [ T t≥(g−δ) − AT
                                                            [  T t≥(g−δ)    → Γ (N (0, Σ)) .
                                                                               ∗

    We now describe a practical bootstrap algorithm to compute studentized confidence bands that cover
AT T (g, t) simultaneously over all t ≥ g − δ with a pre-specified probability 1 − α in large samples.
This is similar to the bootstrap procedures used in Kline and Santos (2012), Belloni et al. (2017) and
Chernozhukov et al. (2018) in different contexts.
                                                                                   ∗,dr,nev
Algorithm 1. 1) Draw a realization of {Vi }ni=1 . 2) Compute AT [  T t≥(g−δ) as in (4.6), denote its (g, t)-
               ∗
element as AT
           [  T (g, t) , and form a bootstrap draw of its limiting distribution as
                                                          √     ∗                    
                                           R̂∗ (g, t) =    n AT
                                                              [ T (g, t) − AT
                                                                           [  T (g, t) .

3) Repeat steps 1-2 B times. 4) Compute a bootstrap estimator of the main diagonal of Σ1/2 such as
the bootstrap interquartile range normalized by the interquartile range of the standard normal distribu-
      b 1/2 (g, t) = (q0.75 (g, t) − q0.25 (g, t)) / (z0.75 − z0.25 ) , where qp (g, t) is the pth sample quantile of the
tion, Σ
R̂∗ (g, t) in the B draws, and zp is the pth quantile of the standard normal distribution. 5) For each boot-
strap draw, compute t−testt≥(g−δ) = max(g,t) R̂∗ (g, t) Σ   b (g, t)−1/2 . 5) Construct cb1−α as the empirical
(1 − a)-quantile of the B bootstrap draws of t−testt≥(g−δ) . 6) Construct the bootstrapped simultaneous
                                                                                        b (g, t)−1/2 /√n].
                                                                   nev
confidence band for AT T (g, t), t ≥ (g − δ) , as C
                                                  b (g, t) = [AT
                                                              [  T dr (g, t; δ) ± cb1−α Σ

    The next corollary to Theorem 3 states that the simultaneous confidence band for AT T (g, t) described
in Algorithm 1 has correct asymptotic coverage.

Corollary 1. Under the assumptions of Theorem 2, for any 0 < α < 1, as n → ∞,
                                                                               
              P AT T (g, t) ∈ C
                              b (g, t) ∀t ∈ {2, . . . , T } , g ∈ Gδ : t ≥ g − δ → 1 − α,

  14
       Since the number of periods T is fixed, Γ(·) should be interpreted as a continuous functional between Euclidean spaces.

                                                                  24

where C
      b (g, t) is as defined in Algorithm 1.

Remark 10. In DiD applications, it is common to use “cluster-robust” inference procedures; see, e.g.,
Wooldridge (2003) and Bertrand et al. (2004). However, we note that the choice of whether to cluster
or not is usually not obvious, and depends on the kind of uncertainty one is trying to reflect; see, e.g.,
Abadie et al. (2017) for a discussion in a cross-sectional setup.15 In the case that one wishes to account
for clustering to reflect “cluster-based” sampling uncertainty, we note that this can be done in a straight-
forward manner using a small modification of the multiplier bootstrap described above, provided that the
number of cluster is “large.” More precisely, instead of drawing observation-specific V ’s, one simply needs
to draw cluster-specific V ’s; see, e.g., Sherman and Le Cessie (2007), Kline and Santos (2012), Cheng
et al. (2013), and MacKinnon and Webb (2018, 2020). If the number of clusters is “small,” however,
the application of the aforementioned bootstrap procedure is not warranted.16

Remark 11. In Algorithm 1 we have required an estimator for the main diagonal of Σ. However, we
note that if one takes Σ
                       b (g, t) = 1 for all (g, t), the result in Corollary 1 continues to hold. However, the
resulting “constant width” simultaneous confidence band may be of larger length; see, e.g., Montiel Olea
and Plagborg-Møller (2018) and Freyberger and Rai (2018).

Remark 12. The above results focus on making inference about AT T (g, t)’s in (effective) post-treatment
periods t ≥ g − δ. Although the limited anticipation condition in Assumption 3 implies that AT T (g, t) = 0
for all t < g − δ regardless of the group g, it is common practice to also estimate these pre-treatment
parameters and use them to assess the credibility of the underlying identifying assumptions. Note that
our DiD estimands (2.2) - (2.7) can be easily adjusted to include these by simply replacing the “long
differences” (Yt − Yg−δ−1 ) with the “short differences” (Yt − Yt−1 ) for all t < g − δ. All our results
                                           dr,nev
continues to hold when one augments AT [ T t≥(g−δ) to also include these estimates for the AT T (g, t)’s in
the pre-treatment periods t < g − δ.

4.2    Asymptotic Theory for Summary Parameters
Assume, for simplicity, that Assumption 3 holds with δ = 0. In this section, we discuss how one can
estimate and make inference about the summary measures of the casual effects discussed in Section 3.
More concisely, we consider parameters of the form of θ as defined in (3.1), which covers all of the
aggregated parameters discussed in Section 3.
    Given the discussion in Section 4.1, a natural way to estimate θ is to use the plug-in type estimators

                                               X T
                                                X                      nev
                                        θ̂ =             w
                                                         b (g, t) AT
                                                                  [  T dr (g, t; 0) ,
                                               g∈G t=2

   15
      The formal results in Abadie et al. (2017) focus on the cross section case and rely on additional functional form
restrictions that we do not impose in this paper. Fully extending the results of Abadie et al. (2017) to the semiparametric
panel data case is beyond the scope of our paper.
   16
      In such cases, provided that one is comfortable imposing additional functional form assumptions, one could use alter-
native procedures such as Conley and Taber (2011) and Ferman and Pinto (2019). Extending these proposals to our setup
is beyond the scope of this paper though.

                                                              25

      b (g, t) are estimators for w (g, t) such that for all g ∈ G and t = 2, . . . , T ,
where w
                                                                       n
                               √                              1 X w
                                      b (g, t) − w (g, t)) = √
                                   n (w                          ξg,t (Wi ) + op (1) ,
                                                               n
                                                                      i=1

                                 w (W)0 finite and positive definite. Estimators based on the sample
       w               w            
with E ξgt (W) = 0 and E ξgt (W)ξgt
analogue of the weights discussed in Section 3 satisfy this condition.
    Let
                                T                                                                         
                                                        dr,nev
                                                               (Wi ; κ∗,nev
                               XX
                  lw (Wi ) =                w (g, t) · ψg,t,0         g,t   ) + ξ w
                                                                                  g,t (W i ) · AT T (g, t)   ,
                               g∈G t=2
       dr,nev
where ψg,t,δ  are as defined in (4.4).
    The following result follows immediately from Theorem 2, and can be used to conduct asymptotically
valid inference for the summary causal parameters θ.

Corollary 2. Under the assumptions of Theorem 2,
                                                                 n
                                        √                1 X w
                                            n(θ̂ − θ) = √        l (Wi ) + op (1)
                                                          n
                                                             i=1
                                                                 h       i
                                                      d
                                                      → N 0, E lw (W )2
                                                      −

    Corollary 2 implies that one can construct standard errors andh confidence
                                                                           i   intervals for summary
treatment effect parameters based on a consistent estimator of E lw (W )2 or by using a bootstrap
procedure like the one in Algorithm 1. The main advantage of using the bootstrap procedure akin
to Algorithm 1 is that inference procedures would be robust against multiple-testing problems. This
                                                      bal (e; e0 ), θ (g̃), and θ t̃ , as practitioners would
                                                                                    
is particularly attractive when considering θes (e), θes             sel         c
probably analyze how these parameters differ across event-times e, groups g̃, and calendar-time t̃.

Remark 13. As discussed in Remark 10, the validity of the “cluster-robust” multiplier bootstrap procedure
relies on the number of clusters being “large.” In some applications such a condition may be more plausible
when analyzing the aggregated parameter θ than when analyzing the AT T (g, t) themselves.

5    The Effect of Minimum Wage Policy on Teen Employment
In this section, we illustrate the empirical relevance of our proposed methods. To do this, we apply our
methods to study the effect of the minimum wage on teen employment. The main goal of this section
is to compare results arising from using a TWFE specification (as is most common in applications) to
results coming from our proposed method. We think that this comparison is important in order to get
a sense of whether the theoretical limitations of TWFE discussed in recent work end up translating into
meaningful differences in applications. Moreover, one might expect that understanding the effect of a
minimum wage change on employment is a challenging case for TWFE as the effect of the minimum wage
may be dynamic (Meer and West (2016)) and the timing of minimum wage changes varies across states.
Unlike TWFE, the approach that we have proposed in the current paper is robust to these challenges.

                                                               26

   By far the most common approach to trying to understand the effect of the minimum wage on em-
ployment is to exploit variation in the timing of minimum wage increases across states. Our identification
strategy follows this approach. In particular, we consider a time period from 2001-2007 where the federal
minimum wage was flat at $5.15 per hour. We focus on county level teen employment in states whose
minimum wage was equal to the federal minimum wage at the beginning of the period. Some of these
states increased their minimum wage over this period – these become treated groups. In particular, we
define groups by the time period when a state first increased its minimum wage. Others did not increase
their minimum wage – these are the untreated group. This setup allows us to have more data than local
case study approaches. On the other hand, it also allows us to have cleaner identification (state-level
minimum wage policy changes) than in studies with more periods; the latter setup is more complicated
than ours particularly because of the variation in the federal minimum wage over time. It also allows
us to check for internal consistency of identifying assumptions – namely whether or not the identifying
assumptions hold in periods before particular states raised their minimum wages.
   We use county level data on teen employment and other county characteristics. County level teen
employment comes from the Quarterly Workforce Indicators (QWI), as in Dube et al. (2016); see Dube
et al. (2016) for a detailed discussion of this dataset. Other pre-treatment county characteristics come from
the 2000 County Data Book. These include county population in 2000, the fraction of the population that
is white, educational characteristics from 1990, median income in 1997, and the fraction of the population
below the poverty level in 1997. After dropping ten states due to their minimum wage being higher than
the federal minimum wage in 2000, seven other states for lack of data on teen employment, and four
other states in the Northern census region, our final sample includes county-level data from 29 states.
We provide additional details on constructing the data in the Supplementary Appendix.
   Summary statistics for county characteristics are provided in Table 2. There are some notable dif-
ferences in county characteristics between counties in states that increased their minimum wage and in
states that did not increase their minimum wage. Treated counties are much less likely to be in the South.
They also have much higher population (on average 94,000 compared to 53,000 for untreated counties).
The proportion of white residents is higher in treated counties (on average, 89% compared to 83% for
untreated counties). There are smaller differences in the fraction with high school degrees and the poverty
rate though the differences are both statistically significant. Treated counties have a somewhat higher
fraction of high school graduates and a somewhat lower poverty rate.

5.1   Results
In the following we discuss different sets of results using different identification strategies. In particular,
we consider the cases in which one would assume that the parallel trends assumption would hold un-
conditionally, and when it holds only after controlling on observed characteristics X. In the main text,
we consider the case where never-treated counties are the comparison group and where we do not allow
for any anticipation effects (i.e., δ = 0). We provide results using the not-yet-treated counties as the
comparison group and allowing for one year anticipation in the Supplementary Appendix; results from
those cases are quite similar to the ones presented here.
   The first set of results comes from using the unconditional parallel trends assumption to estimate the

                                                      27

                                Table 2: Summary Statistics for Main Dataset

                                     Treated Counties       Untreated Counties        Diff.    P-val on Diff.
          Midwest                           0.59                    0.34              0.25           0.00
          South                             0.27                     0.59             -0.32          0.00
          West                               0.14                    0.07              0.07          0.00
          Population (1000s)                94.32                   53.43             40.89          0.00
          White                              0.89                    0.83              0.06          0.00
          HS Graduates                      0.59                     0.55              0.04          0.00
          Poverty Rate                      0.13                     0.16             -0.03          0.00
          Median Inc. (1000s)               33.91                   31.89              2.02          0.00
        Notes: Summary statistics for counties located in states that raised their minimum wage between Q2 of
       2003 and Q1 of 2007 (treated) and states whose minimum wage was effectively set at the federal minimum
       wage for the entire period (untreated). The sample consists of 2284 counties. Sources: Quarterly Workforce
       Indicators and 2000 County Data Book

effect of raising the minimum wage on teen employment. The results for group-time average treatment
effects are reported in Panel (a) of Figure 1 along with a uniform 95% confidence band. All inference
procedures use clustered bootstrapped standard errors at the county level, and account for the autocorre-
lation of the data. The plot contains pre-treatment estimates that can be used to “pre-test” the parallel
trends assumption as well as treatment effect estimates in post-treatment periods.
   The group-time average treatment effect estimates provide support for the view that increasing the
minimum wage led to a reduction in teen employment. For 5 out of 7 group-time average treatment ef-
fects, there is a clear statistically significant negative effect on employment. The other two are marginally
insignificant (and negative). The group-time average treatment effects range from 2.3% lower teen em-
ployment to 13.6% lower teen employment. The simple average (weighted only by group size) is 5.2%
lower teen employment, and the average effect of a minimum wage increase across all groups that in-
                                                             O above) is 3.9% lower teen employment
creased their minimum wage (corresponding to an estimate of θsel
(see Panel (a) of Table 3). A two-way fixed effects model with a post treatment dummy variable also
provides similar results, indicating 3.7% lower teen employment due to increasing the minimum wage. In
light of the literature on the minimum wage these results are not surprising as they correspond to the
types of regressions that tend to find that increasing the minimum wage decreases employment; see the
discussion in Dube et al. (2010).
   As in Meer and West (2016), there also appears to be a dynamic effect of increasing the minimum wage.
For Illinois (the only state in the group that first raised its minimum wage in 2004), teen employment
is estimated to be 3.4% lower on average in 2004 than it would have been if the minimum wage had
not been increased. In 2005, teen employment is estimated to be 7.1% lower; in 2006, 12.5% lower; and
in 2007, 13.6% lower. For states first treated in 2006, there is a small effect in 2006: 2.3% lower teen
employment; however, it is larger in 2007: 7.1% lower teen employment.
   Panel (a) of Table 3 reports aggregated treatment effect measures. First, we consider how the effect
of increasing the minimum changes by the amount of time that the policy has been in place. These
parameters paint largely the same picture as the group-time average treatment effects. The effect of

                                                          28

                              Figure 1: Minimum Wage Group-Time Average Treatment Effects

                                                                       ●       Pre−Treatment
                                                                       ●       Post−Treatment

                            (a) Unconditional Parallel Trends                                             (b) Conditional Parallel Trends
               Group 2004                                                                    Group 2004

         0.2                                                                           0.2
