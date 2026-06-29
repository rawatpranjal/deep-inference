<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2510.09064.pdf
backend: pdftotext
part: 1/2
-->

# Front Matter

<!-- pages: 1-50 -->

Sensitivity Analysis for Treatment Effects in
                                                   Difference-in-Differences Models using Riesz
                                                                  Representation


arXiv:2510.09064v1 [econ.EM] 10 Oct 2025
                                              Philipp Bach* a , Sven Klaassenb,c , Jannis Kueckd , Mara Mattesd , and Martin
                                                                                 Spindlerb,c
                                                     a
                                                         School of Business & Economics, Freie Universität Berlin, Boltzmannstr. 20, 14195 Berlin, Germany
                                            b Chair of Statistics with Application in Business Analytics, University of Hamburg Business School, Moorweidenstr. 18,

                                                                                      20148 Hamburg, Hamburg, Germany
                                                                   c Economic AI, Nürnberger Str. 262 A, 93059 Regensburg, Bayern, Germany
                                           d Düsseldorf Institute for Competition Economics, Heinrich Heine University Düsseldorf, Universitätstr. 1, 20225 Düsseldorf,

                                                                                        Nordrhein-Westphalen, Germany


                                                                                                   Abstract
                                                      Difference-in-differences (DiD) is one of the most popular approaches for empirical re-
                                                  search in economics, political science, and beyond. Identification in these models is based on
                                                  the conditional parallel trends assumption: In the absence of treatment, the average outcome
                                                  of the treated and untreated group are assumed to evolve in parallel over time, conditional
                                                  on pre-treatment covariates. We introduce a novel approach to sensitivity analysis for DiD
                                                  models that assesses the robustness of DiD estimates to violations of this assumption due to
                                                  unobservable confounders, allowing researchers to transparently assess and communicate
                                                  the credibility of their causal estimation results. Our method focuses on estimation by Double
                                                  Machine Learning and extends previous work on sensitivity analysis based on Riesz Rep-
                                                  resentation in cross-sectional settings. We establish asymptotic bounds for point estimates
                                                  and confidence intervals in the canonical 2 × 2 setting and group-time causal parameters
                                                  in settings with staggered treatment adoption. Our approach makes it possible to relate
                                                  the formulation of parallel trends violation to empirical evidence from (1) pre-testing, (2)
                                                  covariate benchmarking and (3) standard reporting statistics and visualizations. We provide
                                                  extensive simulation experiments demonstrating the validity of our sensitivity approach and
                                                  diagnostics and apply our approach to two empirical applications.

                                           Keywords: Sensitivity Analysis, Difference-in-differences, Double Machine Learning, Riesz
                                           Representation, Causal Inference
                                              * Corresponding author: philipp.bach@fu-berlin.de. This version: October 13, 2025


                                                                                                        1

1   Introduction
    Identification of causal effects in difference-in-differences (DiD) models fundamentally relies
on the parallel trends assumption. For example, in the canonical 2 × 2 design with two periods
and two groups, it is assumed that, in the absence of treatment, the expected potential outcomes
of both groups would have followed parallel trends over time. In practice, however, this
assumption is often only plausible after conditioning on observed pre-treatment confounders.
Empirical researchers typically select these covariates based on domain knowledge or economic
reasoning relevant to the context of the study. The treatment assignment in empirical DiD
studies often arises from the decisions of individual units or groups, such as states or countries
choosing to adopt certain policies, in response to economic considerations and other factors,
some of which may be unobserved. Empirical researchers try to model these decision processes
by accounting for pre-treatment covariates in order to justify parallel trends conditionally on
these characteristics. However, identification of the Average Treatment Effect on the Treated
(ATT) fails if these variables do not adequately account for all relevant confounding information:
The reported ATT estimate will be contaminated by the bias from unobserved self-selection into
the treatment groups.
    In such settings, it seems natural to question the validity of the conditional parallel trend
assumption: Is the researcher really able to account for all systematic selection mechanisms that
make certain types of individuals more likely to receive (or rather choose) the treatment than
others? Quantifying the implications of such violations can help to assess the robustness of
causal findings: If the parallel trend assumption were violated, what would be the resulting bias
for the causal estimate? Would this bias be sufficient to substantially change the conclusions of
the causal analysis, for example changing the significance of an effect estimate? Such sensitivity
considerations are useful for an appropriate interpretation and transparent communication of
causal results according to the plausible strength of the parallel trend violation. Building on
previous work by Chernozhukov et al. (2024) and Cinelli and Hazlett (2020), we develop a new
approach for sensitivity analysis in DiD models with panel data exploiting the Riesz representa-
tion for the ATT in the canonical 2 × 2 DiD setting and group-time specific average treatment
effects, ATT (g, t) in multi-period settings with staggered adoption. Our sensitivity approach
helps to quantify the implications from omitting one or several pre-treatment confounders in
terms of the corresponding explanatory power for the treatment assignment and the observed
difference in outcomes over time. Consequently, our framework makes it possible to bound the
bias from omitting pre-treatment confounders, adjust the ATT estimators accordingly and to
compute critical values, which are also known as “robustness values” (Cinelli and Hazlett, 2020;
Chernozhukov et al., 2024) or “breakdown” values (Rambachan and Roth, 2023).
    Our approach builds on the doubly robust ATT estimator introduced by Zimmert (2018),
Chang (2020) and Sant’Anna and Zhao (2020), which is compatible with Machine Learning
(ML) nuisance estimators in the Double/Debiased Machine Learning (DML) framework (Cher-
nozhukov et al., 2018). We derive the Riesz representation for the ATT in the canonical 2 × 2 DiD
setting to establish the bias bounds from parallel trends violation and, thus, extend prior work
on cross-sectional data by Chernozhukov et al. (2024) and Cinelli and Hazlett (2020). To relate
our sensitivity approach to the common practice of pre-testing in event studies, we also present
results for group-time specific average treatment effects in multi-period settings with staggered

                                                2

treatment adoption (Callaway and Sant’Anna, 2021).
    We consider a canonical 2 × 2 DiD setting with two treatment groups and two periods, in
which the conditional parallel trend assumption would be satisfied only if we had access to
observed pre-treatment confounders Xi and one or several unobserved confounding variables Ai .
In general, not taking into account the confounding through Ai will lead to a systematic bias in
the estimation of the ATT, which is our target causal parameter θ0 . This corresponds to a situation
in which researchers worry about the comparability of the treated and control group prior to
treatment onset due to differences in pre-treatment characteristics. These would translate into
non-parallel trends of the potential outcomes without treatment over the considered evaluation
period. A prominent example where individual characteristics play an important role to establish
parallel trends is the famous Lalonde data as re-analyzed in Smith and Todd (2005). In this
example, the individual characteristics of participants in the National Supported Work (NSW)
experiment are not only important to predict whether individuals enter the sample of the treated
or control group, but also for the change of the outcome variable (earnings). Smith and Todd
(2005) state that accounting for these time-invariant confounders in a DiD model is crucial to
reduce the bias from selection into the treatment groups. In our first empirical example in
Section 6, we apply the suggested sensitivity approach to obtain bias bounds on the ATT in
a reevaluation of the data used by Smith and Todd (2005) and LaLonde (1986). The second
empirical application is a replication of Draca et al. (2011), demonstrating the use of pre-testing
information for scenarios of parallel trend violations.
    Our theoretical results show that the asymptotic bias resulting from violations of parallel
trends is closely related to the explanatory power of the unobserved confounding variables Ai
for the treatment assignment probability and the outcome difference over time in addition to Xi .
We provide asymptotic bounds on the causal parameter θ0 and corresponding (1 − a)-confidence
limits if we have access to the observed data only, i.e., Yi,t , Di,t , Xi . For example, this reflects
a setting where measurements for pre-treatment confounding are imperfect proxies, such as
variables for educational achievement and income capturing confounding from socio-economic
status.
    A critical step in sensitivity analysis is to define plausible and realistic scenarios of identifica-
tion violations. A scenario specifies explicit numeric values for the sensitivity parameters, which
enter the asymptotic bias formula that is obtained using the Riesz representation for the causal
parameters. In our approach, the sensitivity parameters quantify the strength of the parallel
trend violation in terms of the explanatory power of Ai for the treatment assignment and the
difference in outcomes. We propose three ways to formulate parallel trends violation scenarios:
(1) Include information from pre-testing, which is a common practice in event studies with
access to pre-treatment periods, (2) exploiting knowledge from leaving-out known and observed
pre-treatment confounders (so-called benchmarking) and (3) standard reporting measures and
visualizations such as contour plots. In addition to two empirical applications, we also provide
evidence on the validity of our sensitivity framework in systematic simulation studies. We would
like to highlight that, to the best of our knowledge, the simulation results are the first thorough
numerical experiments underscoring the validity of sensitivity analysis based on Riesz represen-
tation (Chernozhukov et al., 2024). Our results demonstrate that bias bounds for the confidence
intervals achieve near-to-nominal empirical coverage. Moreover, in our simulation experiments,


                                                   3

we compare the performance of the sensitivity bounds for the ATT to that of the (infeasible)
oracle estimator for the ATT, which would be obtained if one had access to all observed and
unobserved confounders. Our results show that having knowledge on the explanatory power of
the unobserved pre-treatment confounder Ai with regard to the treatment status and outcome
difference is equivalent to directly having access to the unobserved confounders, already with
moderate sample size. We share an open source implementation of our DiD sensitivity analysis
with additional documentation and examples through the DoubleML package for Python (Bach
et al., 2022, 2024b).1
    The remainder of this paper is structured as follows. In Section 2, we briefly review the
existing literature on difference-in-differences with a focus on violations of the parallel trend as-
sumption, doubly robust estimation, and sensitivity analysis. Section 3 introduces the difference-
in-differences setting and the major idea of our sensitivity approach for DiD. We do so by
considering a motivation example, reviewing the underlying ideas of Chernozhukov et al. (2024)
and then presenting our new approach in the canonical 2 × 2 DiD model. Section 4 extends
this framework for sensitivity analysis to the multi-period DiD model with staggered adoption.
Section 5 provides simulation studies and Section 6 provides two empirical applications of our
proposed framework. Finally, Section 7 concludes and provides an outlook on future research.

2    Related Literature
    Difference-in-differences is probably the most frequently used approach to causal inference
with observational data, see for example recent textbooks by de Chaisemartin and D’Haultfœuille
(2023), Huber (2023) and Cunningham (2021). The recent econometric literature has vastly
been impacted by innovative developments in the difference-in-differences literature: New
approaches have addressed limitations of classical estimation procedures such as two-way
fixed effects (TWFE) in settings with multiple treatment groups and heterogeneous effects, for
example including work by Goodman-Bacon (2021), Borusyak et al. (2024), Sun and Abraham
(2021), De Chaisemartin and d’Haultfoeuille (2020), Athey and Imbens (2022) and Callaway
and Sant’Anna (2021). Recent surveys are available in Roth et al. (2023), De Chaisemartin
and d’Haultfoeuille (2023), and Callaway (2023). A practice-oriented guide to difference-in-
differences is provided by Baker et al. (2025). Our study builds on the doubly robust estimator
for the ATT in the canonical 2 × 2 setting as suggested by Sant’Anna and Zhao (2020) and on
group-time specific average treatment effects, ATT (g, t), in multi-period designs with staggered
adoption as considered by Callaway and Sant’Anna (2021). The latter has been a seminal
study to overcome typical limitations of traditional TWFE estimators by recognizing that causal
parameters in complex DiD designs can be modeled as aggregations of possibly many 2-by-
2 comparisons. Moreover, new estimation approaches have been suggested building on the
property of double robustness or Neyman orthogonality. Orthogonality makes it possible
to utilize machine learning (ML) for estimation in the double machine learning framework
(Chernozhukov et al., 2018; Sant’Anna and Zhao, 2020; Chang, 2020; Zimmert, 2018).
    For a long time, empirical economists and econometricians have recognized the importance
of the parallel trend assumption for the causal interpretation of empirical DiD results, which is
assumed to hold conditionally or unconditionally on observable characteristics. In the canonical
    1 More information on DoubleML available at https://docs.doubleml.org.


                                                     4

setting with one pre- and one post-treatment period, this assumption states that, in expectation,
the potential outcomes under no treatment for the treatment and control group develop in
parallel over time. However, the parallel trend assumption is untestable. A common practice
that is suitable if observations from pre-treatment periods are available, is so-called pre-testing.
For example, pre-testing is recommended as part of Pedro Sant’Anna’s DiD checklist,2 Roth
et al. (2023) and in the conclusion of Baker et al. (2025). It is worth noting that the parallel
trend assumption makes a statement on the development of the expected potential outcome
of the treated group in the post-treatment period if the treatment had not occurred, which is
a fundamentally unobservable or counterfactual quantity. The idea of pre-testing is to collect
evidence on observable counterparts of this unknown average potential outcome from pre-
treatment periods: If researchers have access to observations prior to the treatment, it is possible
to assess whether the observed average difference in the outcome of the treated and control
group are the same in these periods. As there is no treatment effective prior to the actual
assignment (by ruling out anticipatory effects), the observed outcome difference should be zero.
Otherwise, a significant effect would indicate a violation of the parallel trends assumption in the
pre-treatment period under consideration. Pre-testing serves as a plausibility check rather than a
proper statistical test of the (untestable) parallel trends assumption. There is no guarantee that
evidence suggesting parallel trends prior to the treatment actually correspond to parallel trends
over the considered post-treatment period. The same is true vice versa: Significant pre-trends do
not necessarily mean that parallel trends are actually violated after the treatment occurred. In the
end, the conclusions from pre-testing exercises have to be interpreted in the specific context of an
empirical analysis (Baker et al., 2025). Recent work by Freyaldenhoven et al. (2019); Bilinski and
Hatfield (2018); Roth (2022); Kahn-Lang and Lang (2020) addresses limitations of pre-testing due
to low power. Moreover, Roth (2022) point to a risk of selection bias, which arises if researchers
only evaluate data for which no pre-treatment violations can be rejected.
    Sensitivity analysis with regard to parallel trend violations is listed as a recommended
step in DiD analysis in Baker et al. (2025). Despite their relevance, sensitivity approaches to
parallel trend violations have only recently been developed. A frequently used approach with
a focus on pre-testing has been introduced by Rambachan and Roth (2023) who extend prior
work by Manski and Pepper (2018).3 Rambachan and Roth (2023) develop finite-sample and
uniformly valid asymptotic bounds when the parallel trends assumption is relaxed. Unlike the
point identification of causal parameters under the exact parallel trends assumption, partial
identification of a set of causal parameters is obtained under a “bounded differential trends”
assumption (de Chaisemartin and D’Haultfœuille, 2023, p. 153). The bounds can be based
on user-provided specifications on the shape of the parallel trends violation resulting in a
set of different post-treatment trends ∆. The choice of ∆ can be motivated from smoothness
assumptions on the differences in trends, pre-testing and their combinations. From a practical
perspective, it is appealing that the knowledge gathered from pre-testing can be exploited, for
example relative to a multiple of the strongest pre-treatment difference in the parallel-trends in
two consecutive periods. By default, this implements a linear extrapolation of pre-treatment
   2Available at https://psantanna.com/DiD/checklist.png.
   3 For example, the approach of Rambachan and Roth (2023) has been used in Callaway (2023), Baker et al. (2025),

Chiu et al. (2025).


                                                        5

violations over the treatment evaluation periods, which for example can be used to point down
so-called breakdown values that correspond to a reduction of the reported effects to zero.
     An earlier approach to sensitivity analysis with regard to violations of parallel trends is
provided by Keele et al. (2019), who adapt previous sensitivity analysis by Rosenbaum and Silber
(2009) and Rosenbaum (2002) to DiD designs. In addition, Freyaldenhoven et al. (2019) develop
an approach for identification of the causal parameter in a linear panel setting under violations
of the parallel trends assumption. They require an additional identification assumption on the
confounding relationships between the unobserved confounders, an additional covariate and
the outcome variable: The confounding variables are assumed to affect the additional covariate
in a similar way as the outcome, but the treatment variable is not allowed to have an impact
on the auxiliary variable. Huber and Oeß (2024) develop a joint test for unconfoundedness and
conditional parallel trend in a DiD setting based on a testing idea introduced in Huber and
Kueck (2023).
     In contrast, our approach builds on sensitivity analysis based on Riesz representation as estab-
lished in Chernozhukov et al. (2024). A detailed comparison of the framework of Chernozhukov
et al. (2024) to Rosenbaum’s approach in cross-sectional settings is provided in Appendix E
of Chernozhukov et al. (2024), which similarly applies for the difference-in-differences setting
considered here and in Keele et al. (2019). In terms of estimation, the methodology in Keele et al.
(2019) focuses on a matching approach, whereas we build on the doubly robust estimators of
Sant’Anna and Zhao (2020) and Callaway and Sant’Anna (2021).
     Our approach extends the current literature on difference-in-differences and sensitivity to
parallel trends violations in various regards. It provides a new set of tools for analyzing the
robustness of DiD estimation results to the existence of unobserved pre-treatment confounders in
settings with and without pre-treatment periods. Rambachan and Roth (2023) establish bounds
based on a user-provided description of parallel trends violations through a specification of
smoothness conditions or relative magnitudes to pre-testing violations, leading to a possibly
very flexible set ∆. Our bias bounds are motivated by the additional explanatory power of
omitted pre-treatment confounding variables relative to the observed variables Xi . In analogy
to the framework of Chernozhukov et al. (2024), we distinguish two different models: A long
and a short model, with corresponding values for the identified (long and short) parameters.
Accordingly, the model that contains both observed confounders Xi and unobserved confounders
Ai is referred to as the long model. This model would have access to all confounding variables
that are sufficient to establish conditional parallel trends and, hence, correspond to what is often
called an oracle model. In contrast, the short model only has access to the observable data,
i.e., Xi , Di,t and Yi,t and thus omits Ai . A difference to the sensitivity approach by Rambachan
and Roth (2023) is that their framework does not assume the existence of such an oracle model.
Intuitively, our bias bounds are obtained from a systematic comparison of the long and short
parameters as identified by the corresponding models using their Riesz representation, whereas
Rambachan and Roth (2023) base their bias bounds on user-provided specification of possible
parallel trend violations through ∆, which can be challenging for applied researchers. From
our point of view, we consider the existence of a long model as plausible and intuitive to
applied researchers in many cases, as it often serves as the starting point for identification
in DiD models in empirical studies. For example, a common reason for violations to causal


                                                 6

identification in economic applications is the existence of socio-economic status (SES). Usually,
empirical researchers employ possibly imperfect proxy measures for SES such as educational
attainment, occupation, and income. The sensitivity parameters that we will use to derive
the bias from parallel trend violations are defined as (nonparametric) partial R2 values, for
example quantifying the share of the residual variation in the outcome difference over time that
could be explained by SES in addition to the included and imperfect proxy variables. When
applied researchers postulate identification under conditional parallel trends, we believe that
this modeling approach often explicitly or implicitly assumes the existence of such an oracle
model.
    Another difference in our approach compared to Rambachan and Roth (2023) is the inferential
framework, which is built on previous work by Chernozhukov et al. (2024) and Cinelli and
Hazlett (2020). Chernozhukov et al. (2024) provide a general framework for analyzing the
omitted variable bias in cross-sectional settings for a wide class of target parameters which can
be characterized by a so-called Riesz representation (Chernozhukov et al., 2022b, 2021, 2022a).
Accordingly, the estimator of interest is obtained as a solution to a moment equation which can
be represented by a linear functional containing two terms: The conditional expectation of the
outcome variable and the so-called Riesz representer. The latter models the relationship between
the covariates and the treatment variable, such as the Horvitz-Thompson transform in augmented
inverse probability weighting or Frisch-Waugh-Lovell partialling out of the covariates from the
treatment variable in a partially linear regression model. We review the general sensitivity
framework of Chernozhukov et al. (2018) in Section 3. Building on the initial work on linear
regression by Cinelli and Hazlett (2020), Chernozhukov et al. (2024) establish asymptotic bounds
on the omitted variable bias and coverage guarantees for confidence bounds in a variety of
causal models.
    As described before, the bias bounds for the causal parameter and confidence intervals are a
function of the sensitivity parameters, which characterize the strengths of the unobserved con-
founding relationships. An appealing feature based on this modeling approach is that it makes it
possible to obtain bounds on the ATT parameter even if no pre-treatment periods are available
for pre-testing. In these settings, it would be very difficult to plausibly specify the set of possible
parallel trend violations ∆ in the approach by Rambachan and Roth (2023). In contrast, our
approach allows to leverage the explanatory power of one, several or all observed pre-treatment
confounders in our framework to inform violation scenarios. In so-called benchmarking, these
variables are left out from the model, which is used to compute empirically grounded values
for the sensitivity parameter in the bias formula. Finally, in analogy to the breakdown values in
Rambachan and Roth (2023), our approach is compatible with the standard reporting statistics
of Cinelli and Hazlett (2020) and Chernozhukov et al. (2024), which inform researchers of how
strong unobserved confounding would have to be to cause a reduction of the reported estimate
to zero.
    Our paper contributes to the existing literature on sensitivity analysis in difference-in-
difference models in various regards. First, we establish new results on the asymptotic bias from
parallel trends violation for the ATT in the canonical 2 × 2 DiD setting as well as on ATT (g, t)
parameters in multi-period settings with staggered adoption. Our bounds quantify the bias
as a function of the explanatory power of the unobserved pre-treatment confounders in terms


                                                  7

ot the treatment status and the outcome difference. Second, we propose practical approaches
to parameterize the parallel trend violation scenarios. Our approach is compatible with the
common practice of pre-testing, but remains applicable even if no-pretreatment periods are
available. To the best of our knowledge, our third contribution is the first systematic evalua-
tion of sensitivity analysis based on Riesz representation, providing supportive evidence and
facilitating the interpretation of its properties, such as the empirical distribution of bias bounds,
their comparison to oracle (long) estimates and the empirical coverage of confidence sensitivity
bounds. Finally, we demonstrate the use of our sensitivity approach in a 2 × 2 DiD setting based
on LaLonde (1986) and Smith and Todd (2005) and a multi-period setting in a reassessment of
Draca et al. (2011).

3 Identification, Estimation and Sensitivity Analysis for the ATT in the
  Canonical Difference-in-Differences Design
    In this section, we motivate and introduce our sensitivity approach in the canonical 2 × 2 DiD
setting with two periods and two treatment groups (treated and control). In Section 4, we will
extend the sensitivity analysis to group-time average treatment effects in multi-period settings
with staggered adoption.

3.1    Motivating Example
Before introducing the formal sensitivity framework in the following sections, we briefly illustrate
the main ideas in a motivation example. We consider the common situation that researchers


Figure 1: Point estimate and two-sided 90%-confidence intervals for the ATT obtained from the short (blue) and long
model (orange). The one-sided sensitivity bounds for the point estimator and 90%-confidence interval of the short
model as obtained from the oracle setting are colored in green. The data is simulated using a modified version of a
data generating process adapted from Sant’Anna and Zhao (2020), more information in Section 5. n = 2, 500.


                                                        8

face in empirical applications of difference-in-differences: Given a set of observed pre-treatment
confounders Xi , the researcher might worry about unobserved confounding, which is not
accounted for by Xi . As a consequence, the conditional parallel trend violation would be violated
leading to a possibly substantial bias of the causal estimate. We can imagine two different models:
A feasible model (denoted as the “short” model in Chernozhukov et al. (2024)) with access only
to the observed pre-treatment confounders Xi and an infeasible (or “long”) model, which would
account for Xi and Ai . The short and the long model identify the short parameter θs and the long
parameter θ0 , which is the true target parameter, respectively. The target parameter in the 2 × 2
DiD setting is the ATT defined as

                                 θ0 = E [Yi,1 (1) − Yi,0 (0)| Di = 1] .

Here, the treatment variable Di,t = 1 indicates that unit i is treated before time t (otherwise
Di,t = 0). Since Di,0 = 0 for all i, we define Di := Di,1 . Furthermore, Yi,t (0) denotes the potential
outcome of unit i at time t if the unit did not receive treatment up to time t and analogously
Yi,t (1) denotes the potential outcome of unit i at time t if the unit did receive treatment. Our goal
is therefore to make statements on the expected bias of the short parameter as compared to the
true ATT,

                                       bias(θ0 , θs ) = θ0 − θs .

As we will show later, the magnitude of this bias will depend on the explanatory power of
the unobserved pre-treatment covariates additionally to the observed Xi . Figure 1 illustrates
the consequences of a parallel trend violation due to omitting the unobserved pre-treatment
confounder Ai in the short model. The results show the DML point estimates for the ATT and
two-sided 90%-confidence intervals obtained from the long (colored orange) and short model
(blue) in a simulated data example. The example is based on a modified version of a data
generating process (DGP) from Sant’Anna and Zhao (2020), which is further explained in Section
5. In practice, researchers would not be able to know whether the estimated ATT, θbs ≈ 5.338, is
close to its true value or not, which is θ0 = 5.0 in the example. Applying our suggested approach
for sensitivity analysis would result in a lower bound of the ATT of around θ̂− = 5.059, which is
substantially closer to the true value. Moreover, the standard 90%-confidence intervals for the
point estimate from the short model do not cover the true ATT. In contrast, we can observe that
the one-sided 90%-confidence sensitivity bounds derived from our sensitivity approach exhibit
coverage of the true effect. The example illustrates that effectively using sensitivity analysis
helps to improve the quality of causal statements in DiD studies when researchers are worried
about the validity of the parallel trends assumption. As we will later emphasize in our empirical
examples, a key ingredient to sensitivity analysis is the proper definition of plausible scenarios
of parallel trends violations. In the motivating example, we employed the population values for
the underlying sensitivity parameters, which we calibrated in our data generating process.

3.2   The Canonical DiD Setting
We will rely on a notation similar to Sant’Anna and Zhao (2020). Let Yi,t be the outcome of
interest for unit i observed at time t ∈ {0, 1}. The observed outcome for unit i at time t is


                                                   9

determined by the treatment status,4

                                      Yi,t = Di,t Yi,t (1) + (1 − Di,t )Yi,t (0).                                     (1)

Further, let Xi be a vector of observed pre-treatment covariates for unit i. Moreover, one or
multiple pre-treatment confounders, Ai , are unobserved. Throughout, we work in a panel setting
and assume that the data Wi = (Yi,0 , Yi,1 , Di , Xi , Ai ) are i.i.d. across units i ∈ {1, . . . , n}. Again,
the target parameter of interest is the average treatment effect on the treated (ATT)

                                        θ0 = E [Yi,1 (1) − Yi,0 (0)| Di = 1] .

It is useful to define the first difference in observed outcomes over time, ∆Yi := Yi,1 − Yi,0 .
In the following, we may occasionally abstract from the index i to keep the notation simple.
Estimation of the ATT parameter can be based on different approaches, including inverse
probability weighting, outcome regression or doubly robust approaches (Sant’Anna and Zhao,
2020; Abadie, 2005). Our sensitivity approach builds on double machine learning (DML). We
define the nuisance parameters m( x, a), denoting the propensity score, and g(d, x, a), denoting
the outcome difference conditional on treatment status d and covariates ( x, a) as

                                    m( x, a) := P( D = 1| X = x, A = a)
                                   g(d, x, a) := E[∆Y | D = d, X = x, A = a].

    Identification of the ATT is based on the following standard assumptions.

Assumption 1 (Parallel Trends Conditionally on X and A).

              E [Y1 (0) − Y0 (0)| D = 1, X, A] = E [Y1 (0) − Y0 (0)| D = 0, X, A]                P − a.s.

Assumption 2 (Overlap). There exists an ϵ > 0, such that for p = P( D = 1), we have

                           p > ϵ and P( D = 1| X, A) = m( x, a) ≤ 1 − ϵ              P − a.s.

    Assumption 1 states that, in expectation, the change in the potential outcomes without
treatment would be the same in the treated and untreated group, conditional on all observed and
unobserved pre-treatment covariates X and A. Identification is therefore compatible with time
trends that are specific to the values of X and A. Assumption 2 is a commonly imposed overlap
assumption that requires a positive fraction of treated individuals and that the propensity score
is restricted to values strictly smaller than 1.

3.3    Estimation and Inference: Double Machine Learning
We focus on estimation using Double/Debiased Machine Learning (DML) as generally estab-
lished in Chernozhukov et al. (2018) and adapted to estimation of the ATT in the 2 × 2 DiD setting
   4 Note that the “switching” Equation (1) for the observed outcome incorporates a no-anticipation assumption

stating that Yi,t−1 (1) = Yi,t−1 (0) which excludes an effect of the treatment variable prior to the realization of Di = 1
in period t, i.e. Di,t = 1, see for example Callaway (2023).


                                                           10

by Chang (2020), Zimmert (2018) and Sant’Anna and Zhao (2020). DML relies on three key ingre-
dients: (1) Neyman orthogonality, (2) high-quality machine learners, and (3) sample splitting.
Under these three requirements, the DML estimator is consistent and asymptotically normal.
Asymptotically valid confidence bounds are provided by Chernozhukov et al. (2018). The DML
estimator for the ATT is the solution to the orthogonal moment condition and corresponds to the
doubly robust estimator in Sant’Anna and Zhao (2020)

                                         D     D − m( X, A)
                     ψ(W, θ, η ) = −       θ+                 (Y1 − Y0 − g(0, X, A)) ,                                 (2)
                                         p    p(1 − m( X, A))

with nuisance components η = (m, g) being estimated by machine learning. For the sake of
notational simplicity, we abstract from in-sample normalization, which is commonly imple-
mented as a finite-sample adjustment. The score function and sensitivity results for the case with
in-sample normalization are presented in Appendix A. Similarly, we abstract from a dedicated
notation to highlight out-of-sample predictions, but rather assume that all nuisance predictions
are obtained from hold-out partitions from the data to safeguard against overfitting-induced
bias (Chernozhukov et al., 2018).

3.4 A General Framework for Sensitivity Analysis based on Riesz Represen-
    tation
Before we establish new bias bounds for the point estimator of the ATT and confidence intervals
in the canonical DiD setting in Section 3.5, we first give a brief review of the general approach for
omitted variable bias under violations of the unconfoundedness assumption in cross-sectional
settings as established by Chernozhukov et al. (2024).5 Chernozhukov et al. (2024) extend
previous results on sensitivity analysis for the average treatment effect (ATE) in a linear regression
model in Cinelli and Hazlett (2020) to a class of estimators that can be characterized as a
linear functional of the conditional expectation of the outcome variable and a so-called Riesz
representer,

                                     θ0 := E[M(W, g)] = E[ g(W )α(W )].

In the cross-sectional setting, g(W ) generally refers to the conditional expectation of the outcome
variable. W denotes the data. Furthermore, α(W ) is the so-called Riesz representer (RR). The
Riesz representer plays a key role for debiasing and implements Neyman orthogonality either
through a known analytical expression or through an approximation by an automatic estimation
algorithm (Chernozhukov et al., 2021, 2022b).

Example 1 (Motivating sensitivity analysis in a partially linear regression model). As an illus-
tration, we briefly review the leading example of a partially linear regression model (PLR) from Section
2 in Chernozhukov et al. (2024). In this setting, the causal parameter is the coefficient for a continuous
treatment variable D in the PLR model, Y = α + θ0 D + f ( X, A) + ν with ν being an error term and
f (·) being a possibly nonlinear function to account for the effect of confounding variables. In this example,
   5 Unconfoundedness is also referred to as selection-on-observables, conditional ignorability or conditional exogeneity.


                                                           11

we have

    θ0 = E[M(W, g)] = E[Y (d + 1) − Y (d)] = E [E[Y | D = d + 1, X, A] − E[Y | D = d, X, A]]
                    = E [ g(d + 1, X, A) − g(d, X, A)] ,

with Y (d) being the potential outcome for the treatment variable being equal to D = d and g( D, X, A) =
E[Y | D, X, A]. Hence, E[M(W, g)] describes the average treatment effect from increasing D by one unit.

   According to Chernozhukov et al. (2024), θ0 can be identified by exploiting the Riesz repre-
sentation E[ g(W )α(W )].

Example 1 (continued). In the PLR example, the Riesz representation implements the famous Frisch-
Waugh-Lovell partialling out of the covariates from the treatment variable,

                                                  D − E[ D | X, A]
                                     α (W ) =                        .
                                                ( D − E[ D | X, A])2
Hence, under regularity conditions (Chernozhukov et al., 2024), the ATE is identified by

                                                      D − E[ D | X, A]
                                                                       
                         θ0 = E (E[Y | D, X, A])                             .
                                                    ( D − E[ D | X, A])2

    Identification of θ0 is feasible only if X and A were observed, which, however, is infeasible in
an empirical analysis. Hence, Chernozhukov et al. (2024) establish asymptotic bias bounds on
the resulting omitted variable bias quantifying the deviation of the short model from the long
model. The corresponding quantities for the short model are defined as gs (W s ) and αs (W s ) with
W s = (Y, D, X ). Accordingly, the bias can be expressed as

                         bias(θ0 , θs ) = E[ g(W )α(W )] − E[ gs (W s )αs (W s )].

Importantly, α(W s ) is the projection of α given the short data,

                                          αs = E[α(W )|W s ].

Chernozhukov et al. (2024) show that the magnitude of the bias depends on the values of the
sensitivity parameters, which quantify the difference of the short from the long quantities,

                                      bias2 (θ0 , θs ) = ρ2 CY2 CD
                                                                 2 2
                                                                   S ,                               (3)

with S2 = E[(Y − gs )2 ]E[α2s ] being a scaling factor that can be estimated empirically and ρ2 =
Cor2 ( g − gs , α − αs ) quantifying the correlation between the residual confounding in the outcome
regression and the Riesz representer. For conservative bounds, ρ is set to a value of ρ = 1.
Importantly, the sensitivity parameters CY2 and CD     2 quantify the proportion of residual variance


                                                    12

in Y and α, respectively, that can be attributed to A,

                                                           E[( g(W ) − gs (W s ))2 ]
                             CY2 = RY2 − gs ∼ g− gs =
                                                            E[(Y − gs (W s ))2 ]

                                                      
                                          E α s (W s ) 2
                                   1−                
                              2        E   α ( W  ) 2
                             CD :=                 ,
                                    E α s (W s ) 2
                                                
                                         E α (W ) 2

                  Var(E[V |W ])            E[Var(V |W )]
with R2V ∼W =        Var(V )
                                = Var(V )−Var (V )
                                                         denoting the nonparametric R2 (Chernozhukov
et al., 2024, P. 8). The interpretation of the sensitivity parameter CY2 is often similar for different
causal models and parameters. However, since CD               2 refers to the interpretation of the Riesz

representer α, which differs across causal models and parameters, the direct interpretation of CD       2

has to be adapted accordingly.

Example 1 (continued). In the PLR example, CY2 can be interpreted as a nonparametric partial R2
measuring the proportion of residual variation in the outcome Y that can be explained by A after taking
into account the explanatory power of the short model,

                                                      E[Var(Y | X ) − Var(Y | X, A)]
                         CY2 = RY2 − gs ∼ g− gs =                                    .
                                                            E[Var(Y | X )]
            2 in the PLR can be interpreted as the proportion of residual variance in the treatment D
Similarly, CD
explained by A,
                                                                           2 
                                      E        E[ D | X, A] − E[ D | X ]
                                           
                                  2
                                 CD =                            2              .
                                               E D − E[ D | X ]
                                                 


   According to Theorem 4 in Chernozhukov et al. (2024), the DML plug-in estimators for
the lower and upper bounds of the target parameter, θ̂± = θ̂s ± |ρ|CY CD Ŝ are asymptotically
normally distributed and centered around their population counterparts, θ± . Moreover, the
result gives rise to a coverage property of the corresponding asymptotic one-sided confidence
bounds, ℓ− and u+ , such that P(θ− ≥ ℓ− ) → 1 − a and P(θ+ ≤ u+ ) → 1 − a given a significance
level a. Evaluating the bias formula in Equation (3) with the oracle values for the sensitivity
parameters ρ, CY2 , and CD 2 , it is possible to identify the absolute value of the bias. However,

these values are generally unknown and researchers have to find plausible, ideally empirically
grounded choices to obtain bounds on the bias. These can be obtained from domain expertise
and/or benchmarking exercises that take advantage of the explanatory relationships between Y,
D and the observed covariates X in the data.


                                                           13

3.5 Sensitivity Analysis for the Canonical Difference-in-Differences Model
The results in Chernozhukov et al. (2024) refer to identification under the unconfoundedness
assumption in cross-sectional setting. In this section, we extend their work to sensitivity analysis
with regard to violations of the conditional parallel trend assumption and focus on the case of
panel data. Extending our sensitivity approach to repeated cross-sectional data as considered in
Sant’Anna and Zhao (2020) would be straightforward. This case is covered in the user guide of
the DoubleML library for Python with an implementation also provided by DoubleML.6 Appendix
A also presents the Riesz representer in the case of in-sample normalization, which is commonly
implemented in practice. We first derive the Riesz representation for the ATT in the canonical
2 × 2 DiD setting. Importantly, other than in the cross-sectional data settings, the conditional
expectation in the DiD model with panel data refers to the difference in the outcome over time,
∆Y,

                                g(d, X, A) := E[∆Y | D = d, X, A],

where g(W ) = E[∆Y | D, X, A] and W = (∆Y, X, D, A). The Riesz representer for the ATT is
given by

                                          D (1 − D ) m( X, A)
                               α (W ) =     −                    ,
                                          p    p    1 − m( X, A)

with m( X, A) = E[ D | X, A] and p = P( D = 1). Hence, the ATT corresponds to
            h            i
     θ0 = E M(W, g)
                                           
               D
        =E       ( g(1, X, A) − g(0, X, A))
               p
                                                             D (1 − D ) m( X, A)
                                                                                 
               D
        =E       ( g(1, X, A) − g(0, X, A)) + E g(0, X, A)     −
               p                                             p       p  1 − m( X, A)
                             D (1 − D ) m( X, A)
                                                      
        = E g( D, X, A)        −
                             p        p    1 − m( X, A)
            h              i
        = E g ( W ) α (W )

with M(W, g) := D/p( g(1, X, A) − g(0, X, A)). Details are provided in Appendix A. It is worth
noting that θ0 is only identified if we would observe A because Assumption 1 is assumed to
only hold conditionally on X and A. In practice, we can only work with the observed data
W s = (∆Y, X, D ). Hence, the short parameter θs is given by
                                                                   
                   h
                          s
                                i       D
           θs = E M(W , gs ) = E          ( gs (1, X ) − gs (0, X )) = E [ gs (W s )αs (W s )] ,
                                        p
  6 More information available at https://docs.doubleml.org/stable/guide/sensitivity.html.


                                                 14

with gs (W s ) := E[∆Y | D, X ]. Consequently, we can apply the sensitivity methodology from
Chernozhukov et al. (2024) and obtain the resulting bounds for the bias

                                       |θ0 − θs |2 ≤ S2 C∆Y
                                                         2   2
                                                            CD ,

with S2 := E[(∆Y − gs (W s ))2 ]E[α2s (W s )]. The interpretation of the corresponding sensitivity
parameters

                               2      E[( g − gs )2 ]
                              C∆Y :=                  = R2∆Y − gs ∼ g− gs
                                     E[(∆Y − gs )2 ]
                                2    E[α2 ] − E[α2s ]   1 − R2α∼αs
                               CD :=                  =              ,
                                         E[α2s ]          R2α∼αs

leads to interesting results. The strength of confounding generated in the outcome regression
  2 directly takes the form
C∆Y

                                               E   ∆Y                     E   ∆Y
                                                                                         
                  2                        Var   [    | D, X, A ]   − Var   [    | D, X ]
                C∆Y  = R2∆Y − gs ∼ g− gs =                                                  ,
                                                Var ∆Y − Var E[∆Y | D, X ]
                                                                                 

which measures the proportion of residual variation in the differenced outcomes ∆Y that can be
explained by the unobserved pre-treatment confounders A in addition to the observed covariates
X. This directly refers to the effect of violating the conditional parallel trend assumption.
Furthermore, the strength of confounding generated in the treatment CD  2 can be rewritten as

                                       h         i        h        i
                                         m( X,A)            m( X )
                                    E 1−m(X,A) − E 1−m(X )
                               2
                              CD  =            h          i          .
                                                  m( X )
                                             E 1− m ( X )

This shows that CD 2 can be interpreted as the relative increase in the average odds of entering the

treatment group due to A after taking into account X. In the implementation and application of
our sensitivity framework, we use a modified version of CD      2 that is bounded by 0 and 1 (Bach

et al., 2024a),
                                               h          i       h         i
                                                  m( X,A)            m( X )
                                     2       E 1−m(X,A) − E 1−m(X )
                          e2 := CD =
                          C                                                   .
                            D          2
                                                       h
                                                          m( X,A)
                                                                   i
                                  1 + CD             E   1−m( X,A)


Here, C e2 can be interpreted as the relative decrease in the average odds of being in the treatment
         D
group due to only observing X but not A. Again, technical details are provided in Appendix A.
    The sensitivity parameters emphasize the role of the pre-treatment confounders for iden-
tification in difference-in-differences. In order to identify the ATT, it is crucial to account for
pre-treatment confounding such that the parallel evolution of the expected potential outcome
under no treatment can be justified. The bias from missing information in this regard by not
observing A is proportional to the share of the variation in the outcome difference ∆Y that can be

                                                  15

attributed to the unobserved pre-treatment confounding through A. For the treatment variable,
the confounding variables help to better separate between treated and untreated individuals,
                                                    e2 . The larger the explanatory power of A for
as reflected by the odds ratio in the definition of C D
selection into the treatment group, the larger will be the corresponding bias for the ATT.

4     Sensitivity Analysis in the Multi-Period Difference-in-Differences with
      Staggered Adoption
    In this section, we establish sensitivity analysis for causal parameters in the multi-period DiD
setting with staggered adoption as, for example, considered by Callaway and Sant’Anna (2021).

4.1    Identification in Difference in Differences with Staggered Adoption
In the following, we use a notation and identifcation assumptions that are based on Callaway and
Sant’Anna (2021). In their paper, Callaway and Sant’Anna (2021) show that group-time specific
causal effect parameters in a multi-period setting with staggered adoption can be expressed
as binary comparisons of the considered treatment and control group. In our presentation,
we slightly adjust the notation to better fit into the common naming conventions in the Dou-
ble/Debiased Machine Learning literature, sometimes slightly abusing notation. The framework
is a generalization of the previously presented 2 × 2 DiD setting. As before, we focus on panel
data to abstract from complex notation in the case of repeated cross-sectional data.
     We consider n observational units at time t = 1, . . . , T . The treatment status for unit i in
period t is indicated by the binary variable Di,t . In settings with staggered treatment adoption,
it is common to define treatment groups according to the first period after treatment receipt,
which requires additional notation. We focus on the case of an absorbing treatment status, i.e., if
individual i is treated first in period g, the individual will remain treated until the final period
                     g
T . The variable Gi is an indicator variable that takes value one if i is treated for the first time in
            g
period g, Gi = 1{ Gi = g} with Gi referring to the first post-treatment period. In the setting with
absorbing treatment, we have Di,t = 1, ∀t ≥ Gi almost surely (cf. Assumption 1 in Callaway
and Sant’Anna (2021)). If individuals are never exposed to the treatment, we define Gi = ∞ and
  g
Gi = 0, ∀t = 1, . . . , T . We define Yi,t (0) as the potential outcome of individual i in period t if no
treatment has been assigned until period T . We summarize the assumptions as follows.

Assumption 3 (Panel Data). We assume that the data (Yi,1 , . . . , Yi,T , Xi , Ai , Di,1 , . . . , Di,T )in=1 are
independent and identically distributed (iid).

Assumption 4 (Irreversibility of Treatment). It holds Di1 = 0 a.s. and Di,t−1 = 1 implies Di,t = 1
a.s. for all t = 2, . . . , T .

   The target causal parameters are defined in terms of differences in potential outcomes. The
potential outcome of an individual i that has been treated first in period g can be evaluated in
period t by
                                                    T
                                 Yi,t = Yi,t (0) + ∑ (Yi,t (g) − Yi,t (0)) Gi .
                                                                             g
                                                                                                              (4)
                                                   g=2


                                                         16

   As a measure of the average causal effect of the treatment, it is common to define a group-
time average treatment effect parameter, ATT (g, t). This target parameter quantifies the average
change of the potential outcomes due to being treated first in period g as evaluated in period t,
                                                                 g
                           ATT (g, t) := E Yi,t (g) − Yi,t (0)| Gi = 1 .
                                                                     
                                                                                               (5)

In the 2 × 2 DiD setting, the counterfactual average outcome for the treated group (which would
have realized had the group not received the treatment) is estimated based on the information
from the untreated group. This is valid under the conditional parallel trend assumption, which
ensures that conditional on the pre-treatment covariates, the expected average outcome difference
over time is the same for the treated and untreated. However, in multi-period DiD settings
with staggered adoption, there is no longer one unique definition of the control group, whose
information can be exploited for identification of the causal parameters. To characterize the
control groups in line with the literature, we define an indicator variable Ci,t , which depends on
whether never treated or not yet treated units are used for comparison.
                             (nev)      (nev)
                           Ci,t      ≡ Ci        := 1{ Gi = ∞}    (never treated),
                                         (nyt)
                                       Ci,t      := 1{ Gi > t}   (not yet treated).

As a consequence, the parallel trend assumption will be adapted to the control group under
consideration. To account for anticipation effects, we will first introduce the limited anticipation
assumption of Callaway and Sant’Anna (2021) using an anticipation parameter δ.
                                                                                                                g
Assumption 5 (Limited Treatment Anticipation). There is a known δ > 0 such that E[Yi,t (g)| Xi , Ai , Gi =
                            g
1] = E[Yi,t (0)| Xi , Ai , Gi = 1] a.s. for all g ∈ G , t ∈ {1, . . . , T } such that t < g − δ.

Assumption 6 (Parallel Trends Conditionally on X and A). Let δ be defined in Assumption 5. For
each g ∈ G and t ∈ {2, . . . , T } such that t ≥ g − δ:

   a. Never treated control group:
                                              g                                            (nev)
          E[Yi,t (0) − Yi,t−1 (0)| Xi , Ai , Gi = 1] = E[Yi,t (0) − Yi,t−1 (0)| Xi , Ai , Ci       = 1] a.s.

   b. Not yet treated control group
                                              g                                            (nyt)
          E[Yi,t (0) − Yi,t−1 (0)| Xi , Ai , Gi = 1] = E[Yi,t (0) − Yi,t−1 (0)| Xi , Ai , Ci,t+δ = 1]    a.s.

    Assumption 6 a. assumes that, in the absence of treatment, the expected outcome for the
group treated first in period g would have evolved in parallel over the time periods considered
as compared to the group that never received the treatment. This assumption is qualitatively
different from Assumption 6 b., which imposes parallel trends of the group treated first in
period g as compared to groups that have not yet been exposed to the treatment at time t + δ
(Callaway and Sant’Anna, 2021). Whether never treated or not yet treated are used as a control
group depends on the empirical context of an analysis and the overall (causal) research question.

                                                        17

Often, only one of these groups is available or relevant for causal evaluations. In addition to the
previous assumptions, an overlap assumption has to be made in the multi-period DiD setting to
achieve identification of the group-time specific causal parameters.
Assumption 7 (Overlap). For each time period t = 2, . . . , T and g ∈ G , there exists a ϵ > 0 such that
     g                      g                     g       (nyt)
P( Gi = 1) > 0 and P( Gi = 1| Xi , Ai , Gi + Ci,t                  = 1) < 1 − ϵ a.s.

   Identification of the long parameters, i.e., the ATT (g, t) in Equation (5), is granted by As-
sumptions 4 to 7 (Callaway and Sant’Anna, 2021). Again, we note that identification of these
target parameters require accounting for X and A in order to justify that the trends in the aver-
age outcomes between the defined control and treatment groups develop in parallel over the
evaluation period under consideration.

4.2 Machine-Learning based Estimation of Group-time Average Causal Pa-
    rameters
It is possible to extend the machine-learning based estimation of the ATT parameter presented
in Section 3.3 to the muli-period DiD model with staggered adoption as presented by Callaway
and Sant’Anna (2021). Note that the corresponding nuisance functions depend on the control
group used for the estimation of the target parameter. By slight abuse of notation we use the
                                                      (nev)            (nyt)
same notation for both control groups Ci,t                    and Ci,t         . More specifically, the control group only
depends on δ for not yet treated units.
                                                                                         (·)
                   g0,g,tpre ,teval ,δ ( Xi , Ai ) := E[Yi,teval − Yi,tpre | Xi , Ai , Ci,t              = 1],         (6)
                                                                                              eval + δ
                                                              g
                     m0,g,teval +δ ( Xi , Ai ) := P Gi = 1| Xi , Ai , max( Gg , Ci,teval ) = 1 ,
                                                                                                                 
                                                                                                                       (7)
                                                         
with nuisance elements η0 = g0,g,tpre ,teval , m0,g,teval . Here, g0,g,tpre ,teval ,δ (·) denotes the population
outcome change regression function for the control group that is specified to evaluate the causal
effect for the group treated first in g over the pre-period tpre and evaluation period teval . tpre is
also denoted as the base period and often specified as the last period before a group received the
treatment for the first time, tpre = g − δ − 1. Furthermore, m0,g,teval +δ ( Xi , Ai ) is the generalized
propensity score. For notational purposes, we will omit the subscripts g, tpre , teval , δ and refer to
the corresponding functions by the simplified versions

                                       g0 (0, Xi , Ai ) ≡ g0,g,tpre ,teval ,δ ( Xi , Ai )
                                         m0 ( Xi , Ai ) ≡ m0,g,teval +δ ( Xi , Ai ).

Note that for estimation of the causal parameters in DiD models, it suffices to estimate the
conditional expectation of the outcome differences for the specified control group as indicated
by conditioning on Ci,teval+δ = 1 in Equation (6). However, for sensitivity analysis, we require
additional estimation of the conditional outcome difference for the group treated first in period
g, which we define as
                                                                                         g
                                g0 (1, Xi , Ai ) ≡ E[Yi,teval − Yi,tpre | Xi , Ai , Gi = 1]

                                                                  18

machine-learning based estimation of the ATT (g, t) parameters can be based on the multi-period
analog of the Neyman-orthogonal score for the ATT in the canonical DiD setting in Equation (2),
7

                                         Gg             Gg − m( X, A)                                      
             ψ(W, θ, η ) = −                     θ +                         Yteval − Ytpre − g ( 0, X, A )   .    (8)
                                        E[ G g ]     E[ Gg ](1 − m( X, A))

4.3      Sensitivity Analysis in a Multi-Period DiD Setting
To extend the sensitivity framework from Section 3.5, we derive the Riesz representer for the
multi-period DiD setting with staggered adoption. In this setting, the moment equation is

                                                             Gg
                                                                     · max( Gg , C (·) ),
                                                         
                        M(W, g) = g(1, X, A) − g(0, X, A) ·                                                        (9)
                                                            E[ G g ]

where max( Gg , C (·) ) is an indicator that takes value one if an individual belongs to the treated
or the specified control group. Including this indicator ensures that only observations from the
relevant treatment and control groups are used for identification. When we later present the
sensitivity parameters, it is important to account only for variation in the data that is relevant for
the target causal parameter. This gives rise to the Riesz representer for the ATT (g, t)

                                  Gg        m( X, A)(1 − Gg )
                                                               
                    α (W ) =            −                         · max( Gg , C (·) ).
                                 E[ G ] E[ G ](1 − m( X, A))
                                     g         g


To have a compact presentation of the sensitivity parameters we refer to the outcome difference
that is estimated for evaluation of the ATT ( g, t) parameter by ∆tpre ,teval Y = Yteval − Ytpre .
   Accordingly, the sensitivity parameters are given by
     2
    C∆Y = R2∆tpre ,t          Y − gs ∼ g − gs =
                       eval


    Var E[∆tpre ,teval Y | D, X, A, max( Gg , C (·) ) = 1] − Var E[∆tpre ,teval Y | D, X, max( Gg , C (·) ) = 1]
                                                                                                                
                                                                                                                   ,
         Var ∆tpre ,teval Y | max( Gg , C (·) ) = 1 − Var E[∆tpre ,teval Y | D, X, max( Gg , C (·) ) = 1]
                                                                                                         


and, for the selection into the treatment groups,
                     h                               i    h                              i
                        m( X,A)                              m( X )
                    E 1−m(X,A) |max( Gg , C (·) ) = 1 − E 1−m(X ) |max( Gg , C (·) ) = 1
              2
            CD  =                      h                            i                      .
                                          m( X )
                                      E 1−m(X ) |max( G , C ) = 1
                                                        g   (·)

     7Again, we abstract from in-sample normalization in our presentation. The score function and Riesz representa-

tion with in-sample normalization are presented in Appendix A.


                                                                 19

As before, it is useful to use the transformed version C        e2 .
                                                                 D
                            h                                    i     h                                       i
                               m( X,A)            g , C (·) ) = 1 − E      m( X )
                   2      E             | max ( G                                    | max ( G g , C (·) ) = 1
      e2 := CD =
      C
                              1−m( X,A)                                   1− m ( X )
                                                                                                                 .
        D            2
                                                h
                                                    m( X,A)
                                                                                        i
               1 + CD                        E                 |max( Gg , C (·) ) = 1
                                                       1−m( X,A)

The sensitivity parameters quantify the explanatory power of the unobserved confounders for
the considered outcome difference and the treatment assignment, which is analguous to their
interpretation in the 2 × 2 case. However, we have to restrict attention to only those observations
in the data that are relevant for the binary comparison in the multi-period DiD setting. This is
done by conditioning on the indicator max( Gg , C (·) ).
    C∆Y2 is then measuring the share of the residual variation in the difference of the outcome

variable over the period tpre to teval for the group that received treatment first in period g,
∆tpre ,teval Y, which can be explained by accounting for the unobserved pre-treatment confounders
A. Ce2 measures the relative decrease in the odds ratio to receive the treatment for the first time
       D
in period g due to not accounting for the explanatory power of A.

4.4    Determining Scenarios with Parallel Trend Violations
The previous sections provided a bias formula for causal parameters in difference-in-differences
models to quantify violations of the conditional parallel trend assumption. Once we plug in
values for CD 2 and C2 , we obtain an upper and lower bound for the ATT and the ATT ( g, t )
                       ∆Y
parameters, respectively. Moreover, we can exploit Theorem 4 of Chernozhukov et al. (2024),
which makes it possible to establish coverage guarantees of the confidence bounds. Heuristically
speaking, plugging in the population values for the sensitivity parameters in the bias formula in
Equation (3) is asymptotically equivalent to having direct access to the A in the oracle model.
     In empirical applications, researchers cannot use the long data W = (Y, D, X, A), nor can
they know the population values for the sensitivity parameters. As they are concerned with the
presence of unobserved pre-treatment confounders A, they are faced with choosing values for
CD2 , C2 , and ρ that are plausible for their specific empirical data setting and, ideally, close to the
       ∆Y
population values of the sensitivity parameters.8 We denote the choice of specific values for CD       2

and C∆Y 2 as parametrizing a violation scenario of the conditional parallel trend assumption (or shorter,

simply violation scenarios), for which the asymptotic bias is estimated. We suggest four practical
ways, in which empirical researchers can obtain plausible values for CD        2 and C 2 : First, domain
                                                                                      ∆Y
expertise can help to quantify the plausible explanatory power of unobserved pre-treatment
confounders for ∆tpre ,teval Y and Gg . Second, standard reporting and visualizations can serve as
informative measures, which might be complementary to domain expertise. Third, information
from pre-treatment periods might be exploited, following the rationale of pre-testing in event
studies. Fourth, so-called benchmarking, i.e., leaving out observed pre-treatment confounding
variables, helps to obtain empirically grounded choices for CD   2 and C2 . In the end, the definition
                                                                           ∆Y
of one or more violation scenarios can be the result of combining these four procedures.
   8 It is generally recommended to start with the most conservative value ρ = 1.


                                                           20

Standard Reporting and Visualization
Cinelli and Hazlett (2020) and Chernozhukov et al. (2024) develop a suite of standard reporting
measures in their sensitivity framework, which can also be applied in difference-in-differences
models. The so-called robustness values RV and RVa indicate critical scenarios of violations of
the identifying assumptions: In the difference-in-differences setting, RV corresponds to the
minimum strength of violating the conditional parallel trends assumption that suffices to set
the ATT or, respectively, the ATT (g, t) estimate equal to zero or an alternatively chosen null
hypothesis. RVa accounts for estimation uncertainty and reports the minimum violation scenario
that is sufficient to make the causal parameter non-significant, hence RV ≥ RVa in general. Note
that the robustness values always refer to symmetric violation scenarios, C∆Y 2 = C e2 . The bias
                                                                                     D
bounds for the point estimate or the confidence limits can be illustrated in contour plots, as for
                                                                                          2 and
example presented in Figure 8 in Section 6. A contour line indicates all combinations of C∆Y
Ce2 that correspond to the same value of the bias-adjusted estimate.
  D

Pre-testing
Pre-testing is a common practical procedure to assess violations of parallel trends in difference-
in-differences models with multiple periods. Whereas no formal statistical test exists for testing a
violation of parallel trends over the relevant time periods for estimation of the causal parameter,
the underlying idea is to provide evidence from pre-treatment periods. In absence of an actual
treatment, a significant causal effect for one or several pre-treatment comparisons might cast
doubt on the validity of conditional parallel trends in the relevant treatment periods. However,
not rejecting the null hypothesis of a zero effect in the pre-treatment period does not serve as
evidence for the validity of the identification assumptions. In the end, the conclusions from
pre-testing evidence are dependent on the context of the empirical application.
    It is possible to use pre-testing information to empirically support the choice of sensitivity
parameters CD  2 and C2 . Here, we focus on sensitivity analysis for the average treatment effect
                        ∆Y
for one treatment group that has received the treatment first in period g being evaluated in the
first post-treatment period, i.e., ATT (g, g) with pre-treatment period g − 1. Suppose, we have
access to Tpre pre-treatment periods, i.e, we have periods 1, . . . , Tpre , g, g + 1, . . . , T . Pre-testing
makes it possible to obtain Tpre − 1 placebo effects θ̂2 , . . . , θ̂Tpre . Because there is no treatment
in the placebo periods and anticipatory effects are ruled out by assumption, any measured
pre-testing effect is the result of a parallel trends violation (Roth, 2022; Rambachan and Roth,
2023). Hence, it is possible to bound the bias in the pre-treatment periods by

                          |θ0,t − θ̂t |2 = |0 − θ̂t |2 = |θ̂t |2 ,   ∀t ∈ {2, . . . , Tpre }.

The robustness value for each pre-treatment period t, RVt , indicates the values C∆Y2 and Ce2 that
                                                                                             D
would suffice to reduce (or rather correct) the bias from a conditional parallel trend violation to
zero. In our application in Section 6, we employ a conservative rule that takes the maximum


                                                           21

robustness values from all pre-treatment periods (or a k-fold multiple thereof, k > 0):9
                                      2,pre      2,pre
                                    C∆Y       = CeD      = max{ RV2 , . . . , RVTpre }.

Benchmarking: Leaving out known pre-treatment confounders
An additional way to obtain specific values for the sensitivity parameters, which is applicable
also if no pre-treatment periods are available, is so-called benchmarking. Leaving out observed
confounding variables has previously suggested for example in Cinelli and Hazlett (2020) and
(Chernozhukov et al., 2024, Appendix F) who build on prior work by Imbens (2003), Altonji
et al. (2005), and Oster (2019). Adapting the idea of leaving-out-confounders from the cross-
sectional setting, we can mimic the omission of unobserved pre-treatment confounders by leaving
out one or multiple variables from X. Whereas there is no formal guarantee that the actual
confounding variable A shares the same explanatory power with the benchmarking variable
X j , these modeling exercises are very useful for empirical researchers: In many cases, domain
experts and empirical researchers know about the role of the most important pre-treatment
covariates, which are crucial to justify the conditional parallel trend assumption. Leaving these
variables out can be helpful to judge the plausibility of violation scenarios based on the empirical
evidence.
      Benchmarking works as follows: Let X j denote the benchmarking pre-treatment covariates,
leaving only X− j = X \ { X j } for estimation of the nuisance functions g(·) and m(·). Then it is
possible to recompute the sensitivity parameters that correspond to the change in the estimate
of the ATT or, respectively, ATT (g, t) that is caused by removing X j from the proxy model.
We denote these calibrated sensitivity parameters as C      e2,bench and C2,bench . It is necessary to
                                                             D            ∆Y
incorporate a correction factor κ that adjusts for the change in the residual variation that is
left after removing X j from X in the proxy model, see also Appendix F of Chernozhukov et al.
(2024).

5    Simulation Study
     In this section, we report the results of a simulation study to assess the finite-sample perfor-
mance of our sensitivity framework. To the best of our knowledge, these are the first systematic
simulation results for sensitivity analysis based on Riesz representation. We extend a data
generating process (DGP) from Sant’Anna and Zhao (2020) for the canonical 2 × 2 DiD setting
in terms of an unknown confounder A. As we will clarify in the following, we implement a
confounding setting with known population values for the sensitivity parameters ρ, C∆Y       2 , and

Ce2 . This makes it possible to assess the empirical performance of our sensitivity approach
  D
in four regards: First, given the oracle values for the sensitivity parameters, we can estimate
the lower and upper bias bounds for the ATT as obtained from Equation (3), θ̂− and θ̂+ , and
compare them to the true parameter value, θ0 . In the scenario considered, the parallel trend
violation implements an upward bias of the short ATT parameter, such that we focus on the
    9 It might be that the sign of the pre-treatment estimates is different than that of the suspected bias. In that case,

the rule might be adjusted to select the maximum among all pre-treatment violations in the suspected direction of
the bias.


                                                             22

evaluation of the lower bias bound. Second, we can evaluate the empirical performance of
the robustness values RV and RVa , which are expected to be close to the oracle values of C∆Y    2 ,

and Ce2 .10 Third, we can evaluate the performance of the lower and upper bias bounds for the
       D
                                                                                                  2
confidence limits, ℓ̂− and û+ . Evaluating the sensitivity bounds at the population values for C∆Y
and C e2 , the one-sided bounds are expected to cover the true ATT in 1 − a percent of the data
       D
realizations. Lastly, we can compare the bias bounds to the long estimate of the ATT, θ̂long , which
is only feasible in a simulation setting. Unlike in real data analysis, we can use the simulated
pre-treatment confounder A to compute the DML estimate that would be obtained from the
long data. This is informative to assess the usability of the sensitivity bounds according to the
sensitivity parameters as compared to having access to the unobserved pre-treatment confounder
A.

5.1    Data Generating Process
We consider an adapted version of the Monte Carlo simulation considered in Sant’Anna and
Zhao (2020). Let X = ( X1 , X2 , X3 , X4 , X5 ) T ∼ N (0, Σ), where Σ corresponds to the identity
                                             e j − En [ Z
matrix. For j = 1, 2, 3, 4, 5, define Zj = ( Z          e j ])/Var( Z
                                                                    e j ), where

                          e1 = exp(0.5 · X1 )                 e2 = 10 +         X2
                          Z                                   Z
                                                                           1 + exp( X1 )
                                                   3
                                          X · X3
                                 
                          e3 =
                          Z          0.6 + 1                  e4 = (20 + X2 + X4 )2
                                                              Z
                                            25
                          e5 = X5
                          Z

For generic V = (V1 , V2 , V3 , V4 , V5 ) T , define

                          f reg (V ) = 210 + 27.4 · V1 + 13.7 · (V2 + V3 + V4 )
                           f ps (V ) = 0.75 · (−V1 + 0.5 · V2 − 0.25 · V3 − 0.1 · V4 ).

Using only the observed pre-treatment confounders X or, respectively, Z in the population model
would basically implement the DGP from Sant’Anna and Zhao (2020). However, we extend the
simulation design in terms of an unobserved pre-treatment confounder A, which is uniformly
distributed over an interval (−1, 1), i.e., A ∼ U (−1, 1). A enters the equation of the propensity
score and the outcome difference regression in an additive way11

                                                    exp( f ps ( Z ))
                                     p( Z, A) =                        + γA · A
                                                  1 + exp( f ps ( Z ))
                                           D = 1{ p( Z, A) ≥ U },
  10We implemented a symmetric parallel trend violation scenario, such that the population values are calibrated to
 2 =C
C∆Y   e2 = 0.1, which implements a nominal level for the robustness values RV and RVa of 0.1 .
       D
  11Additivity in the propensity score helps to compute the population values for the Riesz representer. To ensure

that p( Z, A) ∈ (0, 1), we impose an additional clipping such that 0.1 ≤ p( Z, A) ≤ 0.9.


                                                         23

          n              θ̂s               θ̂−                θ̂+             θ̂long             RVθ =5
          500      5.301 (0.424)     5.001 (0.430)     5.600 (0.423)      4.997 (0.413)       0.134 (0.092)
          1000     5.303 (0.289)     5.004 (0.290)     5.602 (0.290)      4.995 (0.284)       0.113 (0.072)
          5000     5.306 (0.129)     5.005 (0.129)     5.606 (0.129)      5.001 (0.126)       0.101 (0.040)
          10000    5.305 (0.091)     5.005 (0.091)     5.605 (0.091)      5.001 (0.090)       0.101 (0.029)
          50000    5.303 (0.041)     5.003 (0.041)     5.604 (0.041)      4.999 (0.040)       0.101 (0.013)

Table 1: Average simulation results for point estimation based on 10, 000 replications and θ0 = 5.0. Estimation and
bias bounds for ATT, standard deviations in brackets. θ̂s : DML estimate for ATT under parallel trend violation
(short model); θ̂− and θ̂+ : Lower and upper bound for ATT according to bias adjustment with population sensitivity
parameters; θ̂long : DML estimate as obtained from using long data, including pre-treatment confounder A; RVθ =5 :
Robustness value with null hypothesis θ = 5, nominal value in the simulation design is 0.1.


with U ∼ U [0, 1]. The outcome Y is generated as

                           Y0 (0) = f reg ( Z ) + β A · A + ε 0
                          Y1 ( D ) = D · θ · ( Z5 + 1) + f reg ( Z ) + β A · A + ε 1 ( D ),

where ε 0 , ε 1 ( D ) ∼ N (0, σε2 ) and θ ∈ R.
    We parametrize the causal model such that for the ATT we have θ0 = 5. Simulation studies
for sensitivity analysis are characterized by a specific challenge. We would like to implement a
specific confounding scenario, which is in line with the previously presented theoretic framework.
To do so, we calibrate the values for γ A and β A based on a super-population model with
1, 000, 000 observations, for which we can compute the long and short model. Accordingly,
we can compute the population-level values of the sensitivity parameters C        e2 , C2 , and ρ.
                                                                                   D     ∆Y
These values are then used as the evaluated parallel trend violation scenario in the empirical
application of the sensitivity framework, such that we can measure the performance of the
corresponding bias bounds at these population values. We consider settings with sample
size n ∈ {500, 100, 5000, 50000} and report results from R = 10, 000 simulation repetitions.
We consider specification for the outcome regression and propensity scores that are linear in
the covariates, i.e., Z = Z,    e which corresponds to DGP 1 in Sant’Anna and Zhao (2020). For
estimation, we use unpenalized linear and logistic regression learners. Hence, the resulting bias
of the ATT estimate will be only the consequence of the parallel trends violation and not reflect
misspecification bias.

5.2    Results
Table 1 shows the average results for the point estimation and bias bounds for the ATT as
obtained from R = 10, 000 simulation repetitions for different sample sizes n. In all settings,
the short estimate of the ATT, θ̂s , exhibits an upward bias that results from omitting A from the
model. Applying the bias formula in Equation (3) according to the definitions of the sensitivity
parameters presented in the previous sections leads to a lower bound, θ̂− , that is very close
to the true value θ0 = 5.0. In all settings, the robustness value is close to its nominal level of
RVθ =5 = 0.1. In the setting with the smallest sample size, n = 500, the robustness value is slightly

                                                         24

 n               ℓ̂s              ℓ̂−              ℓ̂long       θ0 ≥ ℓ̂s   θ0 ≥ ℓ̂−     θ0 ≥ ℓ̂long    RVθ =5,a=0.1
 500       4.688 (0.438)    4.388 (0.444)     4.395 (0.417)      0.768       0.926        0.928       0.019 (0.046)
 1000      4.877 (0.290)    4.578 (0.291)     4.577 (0.284)      0.660       0.927        0.932       0.022 (0.041)
 5000      5.117 (0.129)    4.817 (0.129)     4.816 (0.126)      0.178       0.923        0.929       0.044 (0.036)
 10000     5.172 (0.091)    4.872 (0.091)     4.870 (0.090)      0.029       0.921        0.925       0.058 (0.029)
 50000     5.244 (0.041)    4.943 (0.041)     4.941 (0.040)      0.000       0.919        0.932       0.082 (0.013)

Table 2: Average simulation results for confidence limits for 10, 000 replications and θ0 = 5.0. Estimation and bias
bounds for lower confidence limit, standard deviations in brackets. ℓ̂s : Lower limit of one-sided 90%-confidence
interval from DML inference under parallel trend violation (short model); ℓ̂− : Lower one-sided confidence bias
bound at nominal level 90%; ℓ̂long : Lower limit of one-sided 90%-confidence interval from DML inference using long
data, including pre-treatment confounder A; RVθ =5,a=0.1 : Robustness value accounting for estimation uncertainty at
significance level 0.1 and null hypothesis θ = 5, nominal level in the design is 0.1.


too optimistic suggesting to use conservative settings in small-sample settings. This result can
be explained by some numerical instabilities in small samples, which is partly reflected by the
higher standard variation. We recommend considering estimation uncertainty when interpreting
robustness values, as also reflected by low values for the robustness values RVθ =5,a=0.1 in these
settings, cf. Table 2. Interestingly, comparing the lower bound, θ̂− , to the oracle estimator θ̂long
reveals that using the oracle confounding scenario is almost equivalent to directly using the
unobserved pre-treatment confounder A. In small samples, the bias bounds are slightly more
variable than the oracle estimate. However, with increasing sample size, this difference becomes
negligible.
    Table 2 shows the average results for estimation and sensitivity bounds of the lower con-
fidence limit. The results illustrate that, in line with the expectations, the empirical coverage
of the one-sided confidence interval [ℓ̂s , ∞) for the ATT is below the nominal level of 90% and
approaches 0.00 as estimation uncertainty diminishes in larger samples. In contrast, the lower
confidence bias bounds, ℓ̂− , cover the true value of the ATT in 91.9% to 92.7% of the cases,
approaching a nominal coverage of 90%. In larger sample settings, the lower bound of the
confidence limit gets closer to the true value for the ATT.12 The robustness value RVθ =5,a=0.1
appears to be conservative in small samples but approaches the nominal value of 10% in larger
samples. Comparing the performance of the lower confidence bias bound to that of the lower
confidence corresponding to the long point estimator reveals that their performance is similar in
terms of empirical coverage and variability. The average value for ℓ̂− is very close to ℓ̂long , with
differences becoming smaller in larger samples. Moreover, increasing the sample size leads to
smaller variability of the bias bounds, such that the corresponding standard deviations approach
those of the oracle confidence bound in moderate and large samples.
    To get more insight on the distribution of the lower bounds for the point estimate and the
lower 90% confidence limit, we provide histograms of their standardized versions in Figure
2. The histograms show that the lower bounds θ̂− (Panel (i)) and ℓ̂− (Panel (ii)) are normally
  12 Note that the population lower bound, θ
                                             − , evaluated in the population parallel trend violation scenario is
approximating the true ATT, θ0 (subject to numerical differences).


                                                        25

                                                                    (i)


                                                                   (ii)


                                            θ̂−,r −θ0                                      ℓ̂−,r −θ0
                            θ −,r = q
Figure 2: Histograms of (i) ê                                   and (ii) êℓ−,r = q                              . Results from R = 10, 000
                                        1  R                                           1  R
                                        R ∑1 ( θ̂−,r − θ̂− )                           R ∑1 ( ℓ̂−,r − ℓ̂−,r )
                                                             2                                                2

simulation repetitions.


distributed with θ̂− being close to the true ATT, on average. The distribution of the lower bias
bounds is rather dispersed in small data settings, with the approximation of a standard normal
distribution becoming better with larger sample sizes. As indicated by the solid green vertical
line, the 90th percentile of the empirical distribution of ℓ̂− is close to the true ATT (red dashed
line), which provides evidence on the close-to-nominal level empirical coverage of the sensitivity
confidence bounds with decreasing estimation uncertainty. Figure 3 illustrates the empirical
distribution of the DML estimate (short model), θ̂s , and the lower bound, θ̂− , for settings with
increasing sample size in a density plot. The figure illustrates that with larger sample size, the
lower bias bound (right panel) concentrates around the true value, whereas the DML estimate
exhibits a bias irrespective of the reduced estimation uncertainty.
    Figure 4 shows the histogram of the standardized robustness values. The results for the
small sample settings with n = 500 and n = 1000 show that the estimation of the RV might
be complicated by numerical instabilities, as it is often estimated to be 0. With moderate and
larger samples, the RV approaches its nominal value and exhibits an empirical distribution that
is similar to the standard normal distribution. More results can be found in Appendix B.

                                                                   26

     Figure 3: Density plots for DML estimate, θ̂s , and lower bound, θ̂− , based on 10, 000 simulation repetitions.


6     Empirical Application
6.1     Sensitivity Analysis for the LaLonde Data
We apply the framework for sensitivity analysis to the famous LaLonde data. In his influential
study, LaLonde (1986) evaluated the causal effect of participation in the National Supported
Work Demonstration (NSW) program on earnings combining two different data sources. First,
he used data from a field experiment, where individuals were randomly assigned to participate
in a job training. Second, he constructs additional data sets from the Current Population Survey
(CPS)13 and the Panel Study of Income Dynamics (PSID). The idea of using these additional data
sets was to mimic an observational study, for which the results could be compared to those of an
experimental evaluation. The so-called Lalonde critique casts doubts on the validity of (at that
time state-of-the-art) observational causal inference techniques and sparked an intense debate
on the use of observational data in the economics and econometrics literature. Important studies
include Dehejia and Wahba (1999), Dehejia and Wahba (2002) (henceforth DW), Heckman et al.
(1997), and Smith and Todd (2005), among others. A comprehensive survey and reevaluation
has recently been provided by Imbens and Xu (2024), which includes state-of-the-art causal
estimators and new data sets. Imbens and Xu (2024) also perform sensitivity analysis using
the framework of Cinelli and Hazlett (2020) for linear regression under the unconfoundedness
assumption.
    We perform sensitivity analysis on the ATT in the 2 × 2 DiD setting building on the previous
work by Smith and Todd (2005), which was also evaluated using the doubly robust ATT estimator
in Sant’Anna and Zhao (2020). Also Huber and Oeß (2024) use the Lalonde-PSID sample for
their joint test for unconfoundedness and parallel trends.
    13We follow Smith and Todd (2005) and focus on using the CPS samples.


                                                           27

     Figure 4: Histograms of standardized robustness values RVθ =5 from R = 10, 000 simulation repetitions.


    Smith and Todd (2005) employ a matching DiD estimator to analyze three different data sets:
(1) The original Lalonde-CPS sample, (2) a modified version of the Lalonde-CPS data used in
DW, and (3) a refined version of the DW data that focuses on individuals from an early phase
of the field experiment. The appealing feature of the LaLonde data is the availability of an
experimental benchmark for observational causal estimates. This makes it possible to run two
different types of a quasi-observational causal evaluation: First, following LaLonde (1986) and
DW, the individuals that were actually assigned to participate in the job training are considerd
as a treatment group and the control group is composed from the CPS data. Second, it is possible
to evaluate a placebo effect: Those individuals who were not assigned to the treatment in the
experiment are considered as a pseudo-treated group, for which the ATT is evaluated against
the control group from the CPS data. The placebo analysis is useful to quantify the so-called
“evaluation bias” (Smith and Todd, 2005, P. 320) that originates from systematic differences in
the experimental and observational samples. Because the pseudo-treated group did not actually
receive the treatment, a nonzero and possibly significant ATT estimate only reflects selection
into the experimental sample (θ0 = 0 in this case). In their results, Smith and Todd (2005) and
Sant’Anna and Zhao (2020) report the evaluation bias as θ̂ θ̂ , where θ̂Exp is the ATT estimate
                                                                       Exp
from the original experimental evaluation in LaLonde (1986).14
    The rationale for the placebo analysis in Smith and Todd (2005) is that the previously reported
observational estimates in DW are not only reflecting the causal effect, which is estimated
according to their flexibly specified propensity score matching estimators, but also affected
by such sample selection bias. To account for systematic differences in the experimental and
observational samples related to the measurement of the outcome variable and accounting for
local labor market characteristics, Smith and Todd (2005) suggest to use matching difference-in-
differences estimators. Accordingly, Smith and Todd (2005) conclude that DiD estimators are
better able to adjust for these time-invariant factors.
  14A bias of 1 would indicate that a reported ATT estimate is to 100% reflecting a bias due to selection into the

experimental sample.


                                                       28

                        Samples             LaLonde               DW         Early RA
                        ATT, θ0                0                    0            0
                        Exp. Benchmark        886                1, 794       2, 748
                        # Treated (Placebo)   425                 260          142
                                             (1) Point estimation
                        Spec. g()                  ADW         ADW            ADW
                        Spec. m()                   DW          DW             DW
                        Learner g()                Ridge      Lin. Reg.      Lin. Reg.
                        Learner m()                Ridge      Log. Reg.      Log. Reg.

                        θ̂                          -692          301           -326
                        Std. err.                    414          487            592
                        Eval. bias (%)               -78           17            -12
                                           (2) Sensitivity Analysis
                        RV (%)                     0.008         0.003         0.003
                        RVα (%)                    0.000         0.000         0.000

Table 3: DML DiD estimation results for placebo analysis for different data sets considered in Sant’Anna and Zhao
(2020) and Smith and Todd (2005). Size of control group: 15, 992. DW: Model specification as suggested in DW;
ADW: Augmented DW specification of outcome difference regression and propensity score in Sant’Anna and Zhao
(2020), Lin.Reg = Linear regression, Log. Reg. = Logistic regression; More details provided in Appendix C.


    Moreover, Smith and Todd (2005) emphasize that the functional form specification for the
propensity score plays an important role to replicate results in DW, which is in line with the
results in Sant’Anna and Zhao (2020). Sant’Anna and Zhao (2020) use linear and logistic
regression based on manually constructed specifications including polynomial and interaction
terms as motivated by DW. Table 3 show the point estimates for the DML DiD ATT in the placebo
analysis according to the preferred choices for modeling the nuisance components. We chose the
learner specification that performed best in terms of the predictive performance for the nuisance
functions g() and m(). To address overlap issues, which are known to be a major challenge in the
LaLonde data evaluation, we calibrated the learners using isotonic regression (van der Laan et al.,
2024, 2025; Klaassen et al., 2025; Ballinari, 2024). More results, including those from other learner
choices, are available in Appendix C. The DML DiD estimators and the evaluation bias in Panel
(1) of Table 3 are all in the range of the results reported in Sant’Anna and Zhao (2020) (Table 3),
with slightly reduced standard errors. Overall, the estimation of the ATT is very variable leading
to non-significant estimates, due the number of pseudo-treated individuals being very small as
compared to the control group. As a consequence the robustness values RV displayed in Panel
(2) of Table 3 are very close to zero pointing at non-robust effects. However, we estimate the bias
bounds for the ATT in the sample with the actual treated based on the placebo settings, i.e., we
      2 and C
set C∆Y        e2 to the corresponding robustness values from the placebo analysis. The upper
                D
bias bounds for the point estimates and confidence bounds are shown in Figure 5. It is possible
to see that in the LaLonde CPS and early randomized samples, the upper sensitivity bounds for

                                                       29

Figure 5: ATT estimates and two-sided 90%-confidence levels (blue), experimental benchmarks from LaLonde (1986)
(dahsed lines), one-sided bias bounds for the ATT with one-sided upper 90%-confidence bound for LaLonde data
sets (with actually treated group). The bias bounds are evaluated at the robustness values from the placebo analysis
presented in Table 3.


the ATT are now closer to the experimental benchmark. In all cases, the confidence sensitivity
bounds cover the experimental benchmarks.

6.2    The Impact of Minimum Wages on Firm Profitability
As a second empirical example, we apply sensitivity analysis to a study by Draca et al. (2011)
who evaluate the causal effect of the introduction of the national minimum wage (NWM) in
the UK on firm profitability in 1999. Unlike the original study, we only consider the case of a
balanced panel, i.e., we drop observations that are not observed during the entire study period
from 1994 to 2002. The unit of observation i at time t is a firm. In contrast to the original study
with data of up to 771 firms, our balanced panel data set contains 337 with 57 treated firms. Draca
et al. (2011) define the treatment based on pre-treatment wages: Firms that are expected not to be
affected or less affected according to average wages in the time before the NWM introduction
are categorized as untreated. Firms with lower average wages prior to the introduction of the
NWM are considered as treated. In line with the original study, we consider the effect of the
NWM introduction on two outcome variables, the log average wage at the firm (ln_avwage) and
the firm’s profit margin defined as the profit to sales ratio (net_pcm). In this section, we focus
on the analysis with regard to net_pcm and provide the results on average wages in Appendix
C. The original study employs two-way fixed effects regression. In our analysis, we focus on
the previously presented DML DiD estimator in the multi-period setting. Due to the different
sample compositions, our results deviate slightly from the original study. In our replication
of Draca et al. (2011), we base identification on the conditional parallel trends assumption,


                                                        30

Figure 6: ATT (g, t) estimates and 95% (pointwise) confidence intervals associated with the 1999-2000 NWM
introduction on the net profit margin.


which we assume to hold after accounting for pre-treatment information on a firm’s industry
(2-digit industry classification, sic2), the government office region of workplace (gorwk), the
share of part-time workers (ptwk), the share of female workers (female), and the share of union
members (unionmem) by the three-digit industry classification. The latter three of these variables
are time variant whereas there is no variation in industry and region information over time.
For the time-varying variables, we condition on the pre-treatment level. Figure 6 illustrate the
ATT (g, t) estimates as obtained from double machine learning using a random forest learner
for the propensity score and the outcome difference regression. We estimate an effect of the
NWM introduction on the net profit margin in the first post-treatment period of −0.021 (95%
confidence interval [−0.041, −0.001]), which is in line with the results in Draca et al. (2011).
    We apply our sensitivity approach to the ATT (g, t) parameters and show the resulting
robustness values in Figure 7. The RVs for the post-treatment ATT (g, t) estimates range between
6.5% for the third post-treatment period to 9.6% in the second post-treatment period. To set
these values into context, we can exploit the information from pre-testing, for which we expect
non-significant effects and, thus, small RV values under valid conditional parallel trends (cf.
Section 4.4). The effect estimates for the pre-treatment periods are relatively close to zero and not


                                                   31

Figure 7: Robustness values RV and RVa=0.1 for the ATT (g, t) parameters evaluating the 1999-2000 NWM introduc-
tion the net profit margin.


significant. The maximum robustness value from pre-testing is 2.56% as obtained for the last pre-
treatment period. Compared to the post-treatment RV values , the pre-testing RV is relatively
small. Note that the pre-testing coefficient also has a different sign than the post-treatment
ATT (g, t) estimate, which might, hence, result in a rather conservative parallel trend violation
scenario.
    An appealing feature of the study by Draca et al. (2011) is the availability of a rich set of
pre-treatment confounding variables, which can be exploited for benchmarking exercises and,
thus, inform plausible parallel trend violation scenarios. Following the general benchmark-
ing procedure described in Appendix F of Chernozhukov et al. (2024), we can compute gain
statistics from leaving out covariates. The corresponding values for the sensitivity parameters
are presented in Table 4. In many cases, we find that omitting the covariates, as for example
the information on industry classification, has considerably more explanatory power for the
treatment status than for the difference in outcomes. This is in line with the general intuition that


                                                      32

                             Y             Scenario              e2
                                                                 C          2
                                                                           C∆Y      |ρ|
                                                                   D
                             net_pcm       Pretest    0.0256           0.0256    1.0000
                                           Benchmark
                                           - ptwk     0.0972           0.0085 0.1299
                                           - sic2     0.1429           0.0010 1.0000
                                           - unionmen 0.0620           0.0010 1.0000
                                           - female   0.0309           0.0010 1.0000
                                           - gorwk    0.0416           0.0173 0.1171

Table 4: PT violation scenarios based on pre-testing and benchmarking pre-treatment covariates. We enforce a
minimum value of 0.0010 for benchmarking scenarios.


information on a firm’s industry is likely related to the average wage level (prior to the NMW
introduction), which is used to define the treatment status in Draca et al. (2011).
    The previous sensitivity exercises have resulted in a set of parallel trend violation scenarios.
As a next step, we consider the critical parallel trend violation that would lead to a substantial
change in the causal results, i.e., reduce the (negative) effect estimate of the NWM introduction
to zero. In the following, we focus on the sensitivity bounds for the ATT (g, t) in the first
post-treatment period. For this parameter, we obtain robustness values RV = 8.76% and
RVa=0.1 = 1.81. Hence, an unobserved pre-treatment confounder A, which could explain 8.76%
of the residual variation in the conditional expected outcome difference and lower the odds ratio
for the treatment status according as explained by the oracle model by 8.76% would be required
in order to set the effect to zero.
    As a next step, we can estimate the upper sensitivity bounds for the ATT (g, t) parameter in
the first post-treatment period. To do so, we use an additional visualization of the upper bound
for the causal parameter in the different parallel trend violation scenarios through a contour plot,
presented in Figure 8. Note that we enforced |ρ|= 1 for the indicated scenarios in the contour
plot to maintain the comparability of the different scenarios.15 As a consequence, the evaluated
settings are worst-case scenarios with a maximum correlation of the confounding variation
in the treatment assignment and outcome difference. The contour plot shows that the upper
bound for the ATT (g, t) parameter in the first post-treatment period is negative in all scenarios
considered. A parallel trend violation corresponding to the strongest pre-treatment violation
would result in a reduction of the causal effect (in absolute values) to −0.015. The strongest
(conservative) benchmarking scenario corresponds to leaving out the share of part-time workers
by the three-digit industry classification. A parallel trend violation that is comparably strong
in terms of the explanatory power for the treatment assignment and the considered outcome
difference would induce an adjustment of the ATT (g, t) parameter to −0.014.
    Moreover, it is possible to vary the strength in terms of a k-fold multiple of the pretesting
scenario sensitivity parameters, which would be similar to the type of results commonly reported
in applications of the approach by Rambachan and Roth (2023). Figure 9 presents the upper
bound of the ATT (g, t) parameter in the first post-treatment period and the one-sided upper
  15 ρ operates as a scaling factor in the bias formula in Equation (3).


                                                          33

Figure 8: Contour plot illustrating the upper bound of the ATT (g, t) estimate for the first post-treatment period in
different scenarios of parallel trends violations. The indicated scenarios illustrate the values from pretesting and
selected benchmarking settings, with ρ = 1 being enforced.


confidence bound at a nominal level of 90%. The figure illustrates that the reported causal
effect in the first post-treatment would become non-significant if the conditional parallel trend
violation was comparable to the pre-testing scenario.
    The previous sensitivity results point at a rather robust effect in the considered scenarios. Of
course, the overall conclusion from our empirical analyses have to be set into the context of the
study. Draca et al. (2011) study a country-wide introduction of a minimum wage, which is a
policy affecting all companies operating in the UK. Differences in the probability to be affected
by the NMW introduction and the change in the profitability over the considered time horizon
might be related to firm characteristics, such as industry and characteristics of the workers. We
find that benchmarking against observed variables in the data point at a rather robust effect.
However, considering estimation uncertainty, the results in Figure 9 show that the effect quickly
becomes non-significant in our pre-testing sensitivity exercises.


                                                         34

Figure 9: ATT (g, t) for the first post-treatment period with one-sided 90% confidence bounds according to different
PT violation scenarios based on pre-testing, dependent variable net_pcm.


7    Conclusion and Outlook
    Our study contributes to the existing and quickly growing literature on difference-in-differences,
causal machine learning and sensitivity analysis. In the previous section, we presented a new
approach to sensitivity analysis with regard to violations of the conditional parallel trend assump-
tion in common difference-in-differences models. In many empirical applications, researchers are
concerned with potential violations of the parallel trend assumptions and a corresponding bias
of the causal parameter estimate. To assess the robustness of causal estimation results due to such
violations, researchers can use our suggested sensitivity approach and obtain asymptotic bounds
for the point estimates and confidence intervals for the target parameters. Specific violation
scenarios in terms of a set of values for the sensitivity parameters in the asymptotic bias formula
can be based on pre-testing evidence, benchmarking analyses, standard reporting statistics and
domain expertise. In addition to the theoretical results on Riesz representation in the canonical
and multi-period DiD setting with staggered adoption, we provide new evidence on the validity
of Riesz-representation-based sensitivity analysis in a simulation study. Moreover, we provide
an open source implementation of our approach in DoubleML for Python and we demonstrate
the application of DiD sensitivity analysis in two empirical examples.
    There are various extensions worth to explore in future research. A natural extension of the
current approach is to aggregations of the ATT (g, t) parameters in the multi-period difference-in-
differences setting as considered in Callaway and Sant’Anna (2021), For example, often the treat-
ment effect is evaluated relative to the treatment receipt in event studies. Riesz-representation
based sensitivity analysis for aggregated effects is non-trivial and, hence, left for future re-


                                                        35

search. Further extensions might be interesting for recently considered difference-in-differences
models such as models with continuous treatments. Finally, empirical application of Riesz-
representation-based sensitivity analysis is important to address empirical challenges of these
new techniques for causal analysis.


                                               36
