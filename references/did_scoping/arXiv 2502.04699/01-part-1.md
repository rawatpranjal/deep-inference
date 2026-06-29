<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2502.04699.pdf
backend: pdftotext
part: 1/2
-->

# Part 1

<!-- pages: 1-32 -->

A Meta-learner for Heterogeneous Effects in Difference-in-Differences

                                                                        Hui Lan 1 Haoge Chang 2 Eleanor Dillon 3 Vasilis Syrgkanis 4

                                                                    Abstract                                      such as conditional exogeneity, which rules out unobserved
                                                                                                                  confounding. Panel data consist of repeated observations of
                                                We address the problem of estimating heteroge-
                                                                                                                  the same units over time, which allows researchers to control

arXiv:2502.04699v2 [stat.ML] 26 Apr 2025
                                                neous treatment effects in panel data, adopting the
                                                                                                                  for certain types of unobserved, time-invariant characteris-
                                                popular Difference-in-Differences (DiD) frame-
                                                                                                                  tics. Due to its flexibility and robustness in handling non-
                                                work under the conditional parallel trends assump-
                                                                                                                  experimental data, the DiD approach has gained significant
                                                tion. We propose a novel doubly robust meta-
                                                                                                                  traction in empirical research, especially in the evaluation
                                                learner for the Conditional Average Treatment
                                                                                                                  of policy interventions (e.g. Thome et al., 2024, etc.), labor
                                                Effect on the Treated (CATT), reducing the esti-
                                                                                                                  market changes (e.g. Card & Krueger, 1994; Rossin-Slater
                                                mation to a convex risk minimization problem in-
                                                                                                                  et al., 2013; Pierce & Schott, 2016, etc.), environmental
                                                volving a set of auxiliary models. Our framework
                                                                                                                  regulations (e.g. Gao et al., 2020, etc.), and public health
                                                allows for the flexible estimation of the CATT,
                                                                                                                  (e.g. Finkelstein et al., 2012; Dimick & Ryan, 2014, etc.).
                                                when conditioning on any subset of variables of
                                                interest using generic machine learning. Leverag-                 Despite several recent methodological advances in the DiD
                                                ing Neyman orthogonality, our proposed approach                   literature (Roth et al., 2023; Chiu et al., 2023), most state-
                                                is robust to estimation errors in the auxiliary mod-              of-the-art approaches are still only able to generate aver-
                                                els. As a generalization to our main result, we                   age causal effects, or at best group average causal effects
                                                develop a meta-learning approach for the estima-                  for predefined subpopulations. On the contrary, in many
                                                tion of general conditional functionals under co-                 empirical applications, especially on large-scale datasets
                                                variate shift. We also provide an extension to the                that stem from digital platforms, practitioners are interested
                                                instrumented DiD setting with non-compliance.                     in treatment effect heterogeneity for personalized decision
                                                Empirical results demonstrate the superiority of                  making. The estimation of heterogeneous treatment effects
                                                our approach over existing baselines.                             has gained considerable attention in recent years due to its
                                                                                                                  potential to uncover variation in how different subpopula-
                                                                                                                  tions respond to an intervention. Motivated by the success
                                           1. Introduction                                                        of machine learning techniques in learning complex tasks,
                                                                                                                  many studies have employed them in learning heteroge-
                                           Difference-in-Differences estimators have become a foun-               neous treatment effects, see for instance Shalit et al., 2017;
                                           dational tool for causal inference in economics (Roth et al.,          Shi et al., 2019; Künzel et al., 2019; Nie & Wager, 2021;
                                           2023), social sciences (Chiu et al., 2023) and healthcare              Oprescu et al., 2019; Kennedy, 2023, etc. However, estimat-
                                           (Wang et al., 2024) for evaluating causal effects of policy in-        ing heterogeneous treatment effects for panel data remains
                                           terventions or treatments when both pre- and post-treatment            relatively unexplored in literature.
                                           outcomes are observed. In contrast to cross-sectional data,
                                           having panel data enables researchers to work with different           In this paper, we explore the estimation of heterogeneous
                                           assumptions that are often considered more plausible in ap-            treatment effects of a binary treatment using panel data
                                           plication. Due to the non-random assignment of treatments,             under the canonical parallel trends condition used in DiD
                                           estimating the causal effect of a treatment or intervention            setups (e.g. Ashenfelter & Card, 1984; Card & Krueger,
                                           in observational studies often requires strong assumptions,            1994, etc.). The parallel trends assumption posits that, in the
                                                                                                                  absence of treatment, the treated and control units would
                                              *Part of this work is done during an internship at Microsoft        have followed similar trends over time. Recent research
                                           Research. **Vasilis Syrgkanis and Hui Lan are Supported by NSF
                                           Award IIS-2337916. 1 Institute of Computational and Mathematical       has explored different approaches in addressing limitations
                                           Engineering, Stanford University, Stanford, USA 2 Department           of traditional methods (e.g. Roth et al., 2023). One line of
                                           of Economics, Columbia University 3 Microsoft Research, New            work focuses on relaxing the unconditional parallel trends
                                           England 4 Department of Management Science and Engineering,            assumption by taking into account systematic differences
                                           Stanford University, Stanford, USA. Correspondence to: Hui Lan         in the time trends due to other (observed) characteristics
                                           <huilan@stanford.edu>.
                                                                                                                  through the conditional parallel trends condition (e.g. Heck-

                                                                                                              1

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

man et al., 1997; Sant’Anna & Zhao, 2020, etc.). Another               for which only a subset of covariates is available.
line of research tackles the challenges of estimating average
                                                                       We demonstrate using synthetic and semi-synthetic exper-
treatment effects under treatment effect heterogeneity over
                                                                       iments that the proposed meta-learner outperforms prior
time for multi-period settings (e.g. Sun & Abraham, 2021;
                                                                       baselines. Finally, we applied our method on a real-world
Callaway & Sant’Anna, 2021, etc.). This paper synthesizes
                                                                       case study on the effects of raising minimum wage on teen
the insights from these two lines of works, and extends the
                                                                       employment. Our flexible doubly robust meta-learner au-
framework to incorporate heterogeneous treatment effects
                                                                       tomatically identified dimensions and patterns of hetero-
across any dimension, in a flexible manner.
                                                                       geneity that had not been highlighted in prior literature. In
We propose a doubly robust estimation framework for the                particular, our method uncovered that the county population
conditional average treatment effect on the treated (CATT),            plays a significant role on the magnitude of the treatment ef-
and show that the mean squared error (MSE) of the learned              fect of raising the minimum wage on teen employment and
model is robust to the estimation error of auxiliary models            even though this effect can be quite large and negative for
that need to be estimated. While there are doubly robust esti-         small counties, it becomes negligible and close to zero on
mators proposed for unconditional ATT with panel data (e.g.            large counties. We developed an out-of-sample validation
Sant’Anna & Zhao, 2020; Callaway & Sant’Anna, 2021),                   pipeline and showcased that the patterns of heterogeneity
there does not exist one for the heterogeneous effect. In              identified by our methodology are statistically significant.
contrast to the conditonal average treatment effect (CATE),
the asymmetry of the CATT allows our proposed method to                2. Problem Statement
avoid estimating a conditional outcome model under treat-
ment, which can be hard to learn given a unbalanced dataset            We consider the standard setup in the DiD framework. We
with a small number of treated units. We also draw the con-            observe a balanced panel with n units and T periods. We
nection to the literature on debiasing under covariate shift           denote time by t = 0, ..., T − 1. The units are assumed to
(Chernozhukov et al., 2023), and provide an extension of               be an i.i.d sample from a superpopulation. For each unit i,
our main result to a unifying framework for general condi-             we observe a time series of outcomes {Yit }Tt=1 , a time se-
tional functionals, encompassing many widely encountered               ries of binary treatment status {Dit }Tt=1 , and time-invariant
empirical problems such as conditional prediction powered              covariates Wi . For simplicity, we restrict our discussion to
inference under co-variate shift, heterogeneous long-term              T = 2 periods in this section and Section 3. We discuss
effects via surrogates based on historical data and heteroge-          extensions to the multi time period setting in Section 5.
neous treatment effects tailored to target sub-populations.
                                                                       We adopt the potential outcomes framework and assume for
Moreover, we extend our main result to the case of a binary
                                                                       unit i at time t, the outcome is generated as:
instrument (or exposure to treatment) with two-sided non-
compliance, and provide a doubly robust estimator for the                        Yi,t = Di,t Yi,t (1) + (1 − Di,t )Yi,t (0)
conditional local average treatment effect of the exposed.
Similar to Ogburn et al., 2015, Semenova & Chernozhukov,               where Yi,t (d) denotes the potential outcome at time t under
2021 and Oprescu et al., 2019, we consider a framework                 treatment d. For brevity of notation, we may drop the unit
that allows for the conditional parallel trends assumption to          subscript i. We assume that both the treated and untreated
condition on a high dimensional set of observed covariates,            groups are untreated at t = 0, and the treated group becomes
denoted as W . This conditioning strengthens the plausi-               treated at t = 1, while the control group remains untreated.
bility of the assumptions and improves the robustness of               Our target estimand is the conditional average treatment
the resulting estimators. Our focus is on the estimation               effect on the treated (CATT), conditioning on any subset X
of the average treatment effect on the treated (ATT) while             of the covariates W :
conditioning on any subset, X, of the covariates W . Es-
                                                                                 θ0 (X) = E[Y1 (1) − Y1 (0)|D = 1, X].
timating the projection of heterogeneous treatment effects
onto a subset of covariates is particularly advantageous for
interpretation, when the goal is to uncover heterogeneity              2.1. Assumptions and Identification
with respect to a set of key features that are of most interest.       Panel data allows us to disentangle unobserved confounding
For instance, in medical applications, we might have high-             to some degree by leveraging both cross-sectional and time-
dimensional imaging data that can be used to predict the               series variations. In this section, we focus on the conditional
outcome, while we are only interested in understanding how             parallel trends assumption that is commonly employed in
the treatment effect is modified by other features such as             the empirical literature to identify treatment effects for panel
age, bone density, etc. Furthermore, this framework can be             data. This assumption posits that the untreated outcome will
helpful for decision making when trying to leverage the find-          evolve in parallel for both the treated and untreated group,
ings to deploy a personalized policy on a larger population            for units with the same observed characteristics W .

                                                                   2

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Assumption 2.1 (Conditional Parallel Trends).                        since the parallel trends assumption crucially does not make
                                                                     any restriction that the trends under treatment are condition-
             E[Y1 (0) − Y0 (0)|D1 = 1, W ]                           ally parallel between treated and control units. Therefore
              = E[Y1 (0) − Y0 (0)|D1 = 0, W ]                        E[g1 (W ) | X] ̸= E[Y1 (1)−Y0 (1) | X], which subsequently
                                                                     implies that τ0 (X) ̸= θ0 (X). This difference will be more
Conditioning on covariates makes the assumption more plau-           pronounced for datasets where there is a big difference in
sible, as it allows the treatment assignment to depend on any        the covariate distribution between the treated and un-treated
baseline trends that are predictable from the observed covari-       groups.
ates. A practical motivation comes from the abundance of             Thus, when X ⊂ W , the statistical problem that we need to
pre-treatment outcome data (for time periods before t = 0).          solve based on the identification formula in Proposition 2.3
It could be reasonable to condition on the full outcome his-         is inherently different than the statistical problem of estimate
tory to try to account for cases where the magnitude of the          a CATE. Hence, we need to develop novel meta-learners,
growth (or decline) through time might depend on the base            specifically for the CATT, that enjoy local robustness prop-
outcome level. For instance, employees with a higher salary          erties analogous to the robustness properties of methods that
usually receive higher pay raises through time.                      have been developed for the CATE in prior work (Nie & Wa-
Assumption 2.2 (No-anticipation Assumption).                         ger, 2021; Foster & Syrgkanis, 2023; Oprescu et al., 2019;
                                                                     Kennedy, 2023). Our main result will be a doubly-robust
            E[Y0 (0) − Y0 (1)|D1 = 1, W ] = 0                        meta learner for the CATT.
                                                                     The simplest plug-in meta-learning approach for the CATT
In practical applications, Assumption 2.1 is imposed with
                                                                     is to construct an estimate ĝ0 of the baseline growth model
the full set of covariates W for plausibility, as we expect
                                                                     g0 using generic ML techniques (since it corresponds to
more covariates to able to capture more confounding. How-
                                                                     the regression problem of predicting the difference Y1 − Y0
ever, we might only be interested in the heterogeneity of the
                                                                     from covariate W , using samples only from the control
treatment effect in a smaller and interpretable subset of the
                                                                     population, i.e., D = 0) and then estimate a CATT model
covariates X ⊂ W .
                                                                     by learning a second-stage regression model that predicts
Proposition 2.3. Under Assumptions 2.1 and 2.2, the CATT,            the label Y1 −Y0 −ĝ0 (W ) from covariates X, using samples
θ0 (X), can be identified as:                                        only from the treated population, i.e., D = 1.
        θ0 (X) = E[Y1 − Y0 − g0 (X)|D = 1, X],                       It is well-known (Chernozhukov et al., 2018) that using
                                                                     ML estimators in a plug-in manner may cause large esti-
where g0 (x) := E[Y1 (0) − Y0 (0)|D = 0, W = w].                     mation bias due to, for example, regularization and model
                                                                     mis-speification. A doubly-robust estimator alleviates this
3. DR-Learner for CATT                                               concern as it is less sensitive to errors in the baseline growth
                                                                     model ĝ0 , and allows for consistent estimation under weaker
In the special case when W = X, the statistical problem that         statistical conditions.
results from Proposition 2.3 is identical to the estimation of
the conditional average treatment effect under conditional           To present our main result, we need to present a set of
ignorability with outcomes Y1 − Y0 (even though, the result-         preliminary definitions and assumptions. To avoid ill-posed
ing statistical model can only be interpreted as a CATT, due         extrapolations between the treated and untreated groups, we
to the one-sided nature of the parallel trends assumption).          need the following overlap condition:
For discussion, see Appendix C.                                      Assumption 3.1 (Sufficient Overlap). For all W , there exist
                                                                     c > 0 such that c ≤ P(D = 1|W ) ≤ 1 − c.
However, when X ⊂ W , this equivalence no longer holds
and prior approaches for CATE estimation under conditional           A key concept related to robustness is that of Neyman or-
exogeneity is no longer applicable and can lead to biased            thogonality:
results even in the limit of infinite samples. For instance,
                                                                     Definition 3.2 (Conditional Neyman Othogonality). Let
the simplest identification formula for the CATE and its
                                                                     m(Z; θ, η) be a moment for the target estimand θ(·) with
accompanying estimation estimation strategy, the T -Learner,
                                                                     nuisance functions η = (η1 , η2 , . . . ). Such moment is Ney-
would estimate the statistical model:
                                                                     man orthogonal if the directional derivatives with respect to
            τ0 (X) = E[g1 (W ) − g0 (W ) | X],                       all nuisance functions η is zero when evaluated at the true
                                                                     nuisances, i.e.
where gd (W ) = E[Y1 − Y0 | D = d, W ]. However, under
the conditional parallel trends assumption it is no longer the                      ∂η E[m(Z; θ0 , η)|W ]          =0
                                                                                                            η=η0
case that E[Y1 − Y0 | D = 1, W ] = E[Y1 (1) − Y0 (1) | W ],

                                                                 3

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Lemma 3.3 (Doubly Robust CATT on Subspace of Covari-                  & Syrgkanis, 2024). Moreover, as we show next in our
ates). Under Assumptions 2.1, 2.2 and 3.1, the true CATT θ0           main estimation theorem, this loss-based estimator enjoys
is a solution to the following conditional moment equation:           double robustness properties, in that it leads to fast rates for
                                                                  the CATT if the product of the estimation rates for π̂ and ĝ
        D − π0 (W )                                                   decays fast enough.
 E                      (∆Y − g0 (W )) − Dθ(X) X = 0
       (1 − π0 (W ))
                                                                      In the theorem below, we use θ̂ to denote a generic estimator
where ∆Y = Y1 − Y0 , g0 (W ) = E[∆Y |D = 0, W ],                      that achieves small excess risk with respect to the plug-in
π0 (W ) = P(D = 1|W ). Moreover, this moment is con-                  loss L(θ; π̂, ĝ), where π̂, ĝ are nuisance estimates, con-
ditionally Neyman orthogonal with respect to all nuisance             structed from an auxiliary dataset (sample-splitting). Note
functions (i.e. π(W ) and g(W )).                                     that the problem of achieving a small excess risk with re-
                                                                      spect to a given loss is a standard statistical learning theory
Remark 3.4. Comparing with the DR-learner (Kennedy,
                                                                      problem and hence many ML techniques can be invoked
2023; Chernozhukov et al., 2017) for conditional average
                                                                      to provide such a guarantee. Hence, our theorem accom-
treatment effect (CATE), we note that, by refocusing on the
                                                                      modates estimators resulting from a variety of CATT ML
CATT, our proposed moment condition no longer requires
                                                                      estimators, such as empirical risk minimization on the em-
the estimation of the conditional expectation of the outcome
                                                                      pirical loss, gradient boosted forests or neural networks.
∆Y for the treated group w.r.t the high dimensional W .
This can be especially advantageous in practical settings             Theorem 3.6 (CATT Rates). Let π̂, ĝ be estimates of the
where there are only a small number of treated units in the           nuisance functions,
                                                                                       p constructed using an auxiliary dataset.
panel, making the estimation of the conditional expectation           Let ∥θ∥D=1 = E[θ(X)2 |D = 1] denote the L2 norm over
of the treated units difficult. Moreover, we show that simply         the treated population. Let θ̂ be the result of any estimation
regressing the CATE pseudo-outcome as in the DR-learner               process using n samples, satisfying w.p. 1 − δ
for CATE will give a biased estimate when the treated and                                                             2
                                                                                  L(θ̂; π̂, ĝ) − inf L(θ; π̂, ĝ) ≤ Rn,δ
control groups have very different distributions. For more                                      θ∈Θ

details, please refer to Appendix C.                                  Suppose Assumptions 2.1, 2.2, and 3.1. If the hypothesis
                                                                      space Θ is convex or is well specified (i.e. θ0 ∈ Θ), then θ̂
The next key insight of our paper is that the Neyman or-
                                                                      satisfies w.p. 1 − δ:
thogonal moment restriction from Lemma 3.3 can be turned
into a loss minimization problem and models that satisfy the            ∥θ̂(X) − θ∗ (X)∥2D=1 ≤
conditional moment restrictions can be equivalently viewed                         h                                      i2 
                                                                      4 2                                 π0 (W )−π̂(W )
                                                                      ρ Rn,δ + β E E (ĝ(W ) − g0 (W ))
as minimizers of a strongly convex loss function. This in-                                                   1−π̂(W )      X
sight is crucial in order to turn the statistical problem into
a statistical learning theory problem and subsequently into           where ρ = P(D = 1) and β = ρ22c2 and
meta-learning estimation strategy, which will allow for the
use of generic ML methods for the estimation of θ0 .                              θ∗ ∈ arg min ∥θ(X) − θ0 (X)∥2D=1
                                                                                            θ∈Θ
Proposition 3.5. Consider the incomplete squared loss:
                          h                   i                       Lagged Dependent Outcome Alternate Assumption: In
        L(θ; π0 , g0 ) = E Dθ(X)2 − 2Yb θ(X)                          Appendix B, we also provide an extension of our approach
                                                                      under the lagged dependent outcome assumption, which
                                                                    posits that the past outcomes capture sufficient information
where Yb (π0 , g0 ) = D−π   0 (W )
                         1−π0 (W ) (∆Y − g0 (W )). Under              to disentangle future outcomes and treatment assignment.
the same assumptions as in Lemma 3.3, the minimizer of                This assumption is commonly used to model time series
L(θ; π0 , g0 ) over any hypothesis space Θ is equivalent to           data and is also used in estimating treatment effects (e.g.
the solution to the best-projection problem of the CATT               Angrist & Pischke, 2009; Antonelli et al., 2024, etc.).
among the treated:
                                                                      DiD with Instruments: As a further extension, we con-
                                      2
            min E[(θ(X) − θ0 (X)) | D = 1]                            sider the setting of estimating heterogeneous effects from
             θ∈Θ                                                      panel data with a binary instrument Z. This has applica-
                                                                      tions in policy evaluation where the exposure to the pol-
Note that this is a convex loss function, which suggests              icy does not perfectly determine treatment receipt due to
computational tractability and fast statistical learning rates        non-compliance (Gerber & Green, 2012), and we are only
and allows it to be efficiently solved using any standard opti-       willing to assume parallel trends on the exposure and not the
mization solver. Another advantage of the loss minimization           chosen treatment. Thus, policy exposure can be interpreted
approach is that the out-of-sample loss can be used as a met-         as the instrument. In Appendix A, we present a meta-learner
ric for model selection over different function classes (Lan          for this IV-DID setup.

                                                                  4

                              A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Table 1. MSE (mean ± standard deviation) Over 100 Simulations. Each row represent a different meta-learner, and columns represent the
different nuisance function classes.

                                          Linear
                                                      Lasso (CV)        Ridge (CV)     Random Forest     Best
                                        Regression
                    Neural Net (OR)     0.12 ± 0.02   0.12 ± 0.02       0.12 ± 0.02    0.38 ± 0.18       0.12 ± 0.02
                    Neural Net (DR)     0.1 ± 0.02    0.1 ± 0.03        0.1 ± 0.02     0.14 ± 0.04       0.1 ± 0.02
                    XGBoost (OR)        0.09 ± 0.02   0.09 ± 0.02       0.09 ± 0.02    0.31 ± 0.16       0.09 ± 0.02
                    XGBoost (DR)        0.04 ± 0.01   0.04 ± 0.01       0.04 ± 0.02    0.06 ± 0.03       0.04 ± 0.01

4. General Conditional Functionals Under                                their analysis to the case of conditional functionals and pro-
   Covariate Shift                                                      vide a doubly robust meta-learning strategy for any such
                                                                        conditional linear functional problem under covariate shift.
In this section, we show that the CATT estimation problem
under conditional parallel trends can be viewed as a special            We further motivate this setup with several other empirically
case of a much more broad statistical estimation problem                prevalent examples from the machine learning and causal
which can capture many other empirical problems beyond                  inference literature.
heterogeneous effects in DiD analysis.                                  Example 4.2 (Conditional prediction powered inference).
                                                                        In settings where prediction is a central task, it is often desir-
In particular, we consider the following estimation problem.
                                                                        able to leverage predictive models to improve the efficiency
Consider data consisting of Z, which contains covariates
                                                                        and accuracy of statistical inference. Consider some high-
W drawn from a target distribution Dt . Let Et [·] denote the
                                                                        dimensional features or covariates W , some labels Y , and
expectation with respect to the distribution Dt . The goal is
                                                                        a simulation model g(W ) for the predictive task E[Y | W ].
to estimate a conditional linear functional Et [m(Z; g0 )|X]
                                                                        An example of the prediction powered inference frame-
of the regression function g0 (W ) = E[Y |W ], where X is a
                                                                        work of (Angelopoulos et al., 2023), asks to estimate Et [Y ].
subset of W , m is a linear moment functional of g0 and the
                                                                        However, we might only have labeled data on a smaller or
expectation is taken with respect to the target distribution.
                                                                        slightly different sub-population Ds . In this case, we can
On the other hand, labels for the target variable Y of the
                                                                        use the simulation model and instead target the statistical
regression function are available only on data where the
                                                                        estimand Et [g(W )], using the labeled data only for debias-
covariates are drawn from a different source distribution,
                                                                        ing the simulation model. Our work extends this setting to
i.e., (Y, W ) ∼ Ds . Let Es [·] denote the expectation with
                                                                        allow for the estimation of conditional means with respect
respect to Ds . We assume that there is only covariate drift
                                                                        to a subset of the covariates X, in the target distribution, i.e.
and no concept drift, i.e.
                                                                        θ(X) = Et [g(W ) | X] and in a setting where the covariate
Assumption 4.1 (No concept drift). g0 (W ) = Es [Y |W ] =               shift density ratio is unknown (prior work considers only
Et [Y |W ] = E[Y |W ].                                                  the case of a known covariate shift). In many applications,
                                                                        labels might be expensive to obtain and are only available
Let E denote the indicator variable of whether the sample               for a small subpopulation which can be a different covariate
stems from the target distribution environment. We can then             distribution from the whole population. This setting fits into
rewrite the statistical estimand as:                                    the framework with m(Z; g) = g(W ).
               θ(X) = E[m(Z; g)|E = 1, X]                               Example 4.3 (Heterogeneous long-term effects from short-
                                                                        -term experiments using historical data). Here we consider
                                                                        settings where we have run a short-term experiment, where
For instance, in the case of the CATT problem in the DiD set-
                                                                        a treatment D was randomized over a population of users
ting, the moment is m(Z; g) = Y1 (1) − Y0 (1) − g(W ) and
                                                                        drawn from Dt and our goal is to estimate the effect of
the outcome regression g(W ) = E[Y1 (0) − Y0 (0)|W, D =
                                                                        D on a long-term outcome Y . However, we want to esti-
0] is learned based on the covariate distribution of the un-
                                                                        mate that effect without the need to wait for the long-term
treated units, while the estimand is the conditional func-
                                                                        effect to materialize, but solely based on short-term data.
tional E[m(Z; g)|X, D = 1] = E[Y1 − Y0 − g(W )|X, D =
                                                                        A typical technique used in this setting is the surrogate
1], which is a conditional expectation taken over the covari-
                                                                        approach, where we assume that the long-term outcome
ate distribution of the treated units, conditioning on a subset
                                                                        Y , is not directly affected by the treatment D, but is af-
X of W . Since the label for the regression function g(W )
                                                                        fected indirectly through some short-term or ”surrogate”
is Y1 (0) − Y0 (0), it is only available for the untreated group.
                                                                        post-treatment outcomes S, i.e. Y (d) = Y (S(d)). Under
Debiasing techniques for unconditional functionals under                this assumption, it can be shown that the long-term effect
covariate shift were analyzed in the prior work of Cher-                can be identified by measuring the effect of the treatment
nozhukov et al., 2023. In this paper, we substantially extend           on the predicted long term outcome, based on the surro-

                                                                    5

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

gates and other potentially pre-treatment covariates X. For          nuisance functions π(W ), α(W ) and g(W ):
any set of pre-treatment co-variates X, we can identify the               h
CATE as θ(X) = E[Y (1) − Y (0) | X] = E[g(W ) | D =                     E E · (m(Z; g) − θ(X))+
1, X] − E[g(W ) | D = 0, X], where W = (S, X) and
g(W ) = E[Y | W ]. The function g(W ) can be learned                                     π(W )                     i
                                                                           (1 − E) ·             α(W )(Y − g(W )) X = 0
using historical data where we have access to short-term                               1 − π(W )
signals S, characteristics X and long term outcomes Y .
                                                                     where π(W ) = P(E = 1|W ) and α(W ) is the conditional
However, the historical covariate distribution Ds can po-
                                                                     Riesz representer of Es [m(Z; g) | X].
tentially be different from the distribution Dt . Since the
treatment is randomized, we can write the target estimand            In the CATT application the Riesz Representer is −1. In
as:                                                                  Example 4.2, the Riesz representer is 1. In Example 4.3 the
                                                                 Riesz representer is q(W )  1−q(W )
                                                                                            π + 1−π , where q(W ) = P(D =
                           D 1−D
         θ(X) = Es g(W )     −       X                               1 | W, E = 1). In Example 4.4 the Riesz representer
                           π   1−π                                                       D               1−D
                                                                     α(D, W ) is P(D=1|E=1,W    ) − P(D=0|E=1,W ) .

where π = P(D = 1) = P(D = 1 | W ).This setting                     As in Proposition 3.5, this conditional moment can be turned
                                                falls
                                            D   1−D                  into a convex doubly robust loss minimization problem.
in the framework with m(Z; g) = g(X)        π − 1−π     .
                                                                                                 h                    i
Example 4.4 (CATE with covariate shift). Consider the                          L(θ; π, g) = E Eθ(X)2 − 2Yb θ(X)
case of estimating the CATE τ0 (X) under conditional exo-
geneity. Many times we want to understand the projection             where Yb = Em(Z; g) + (1−E)π(W           )
                                                                                                     1−π(W ) α(W )(Y − g(W )).
of the CATE on a subset of variables W and over some                 Note that in this loss function the variable Y is always mul-
target population Dt over which we will deploy our per-              tiplied by 1 − E and therefore it respects the constraint that
sonalized policy. However, we might want to use a big-               outcomes Y are only available in the source environment.
ger population Ds to train our CATE model, so as to in-              The double robustness property of the loss will make the re-
crease accuracy. In this setting, the target statistical esti-       sulting estimand robust to estimation errors in the nuisance
mand can be written as Et [g(1, W ) − g(0, W ) | X], where           functions. Analogous to Theorem 3.6, we can prove fast
g(D, W ) = E[Y | D, W ]. This lies in the framework with             statistical learning rates, for the resulting estimator based
m(Z; g) = g(1, W ) − g(0, W ).                                       on this doubly robust loss. It is easy to verify that for the
                                                                     CATT setting this loss coincides with the loss in Section 3.
We provide a debiasing framework for this problem. Before
presenting the main results, we state the necessary defini-
tions and assumptions.                                               5. Extension to Multi-Period Setting
Definition 4.5 (Conditional Riesz Representer). The Con-             In the multiple time period setting, we observe the outcomes
ditional Riesz Representer of a continuous linear functional         for each unit for time periods t = 0, 1, . . . , T . Moreover,
m(Z; g) on X, with respect to some function g(W ), is the            assume that no unit is treated at period 0. Consider first the
square-integrable random variable α(X) such that:                    case where all treated units are treated at period G = 1 and
                                                                     we assume the conditional parallel trends assumption that
                                                                     for all t ≥ 1, E[Yt (0) − Y0 (0) | D = 1, W ] = E[Yt (0) −
          Es [m(Z; g)|X] = Es [α(W )g(W )|X]
                                                                     Y0 (0) | D = 0, W ]. Note that in this case, we can treat
                 ∀g(W ) s.t. E[g(W )2 ] < ∞                          the distance ∆ ∈ {0, . . . , T − 1} of a target period t from
                                                                     the initial treatment time period 1 as a random variable.
Assumption 4.6 (Sufficient Overlap Under Covariate Shift).           We can also denote with Ypost (0) as the random variable
For all W , there exist c > 0 such that c ≤ P(E = 1|W ) ≤            corresponding to the post-treatment period outcome we are
1 − c.                                                               looking at. Then we can equivalently write:

Theorem 4.7 (Neyman Orthogonal Moments for General                   E[Ypost (1) − Ypost (0) | X, ∆ = t] = E[Yt (1) − Yt (0) | X]
Conditional Funcitonals under Covariate Shift). Suppose
that Assumptions 4.1 and 4.6 hold. Consider a nuisance               Thus we can treat the distance from treatment ∆, as yet
regression function g0 (W ) = E[Y |W ], and target estimand          another covariate in our framework and make it part of
θ(X) = Et [m(Z; g0 )|X] = E[m(Z; g0 )|E = 1, X], where               X. This way, we can flexibly estimate treatment effect
m(Z; g) is a continuous linear functional of g. The true             heterogeneity as a function of the distance from the initial
solution θ0 (X) satisfies the following conditional moment           treatment period and let ML methods select the best model
restriction that is Neyman orthogonal with respect to all the        on how distance from initial treatment changes the effect.

                                                                 6

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Table 2. MSE (mean ± standard deviation) Over 100 Simulations of Imbalanced Dataset. Each row represent a different meta-learner, and
columns represent the different nuisance function classes.

                                             Linear
                                                         Lasso (CV)       Ridge (CV)    Random Forest    Best
                                           Regression
                Neural Net (OR)            0.22 ± 0.06   0.21 ± 0.06      0.21 ± 0.06   0.4 ± 0.15       0.21 ± 0.05
                Neural Net (DR)            0.18 ± 0.07   0.18 ± 0.05      0.18 ± 0.05   0.24 ± 0.07      0.18 ± 0.05
                Neural Net (CATE OR)       0.27 ± 0.08   0.27 ± 0.08      0.27 ± 0.08   0.51 ± 0.16      0.27 ± 0.08
                Neural Net (CATE DR)       0.22 ± 0.07   0.22 ± 0.07      0.21 ± 0.07   0.33 ± 0.11      0.21 ± 0.07
                XGBoost (OR)               0.21 ± 0.06   0.21 ± 0.06      0.21 ± 0.06   0.34 ± 0.11      0.21 ± 0.06
                XGBoost (DR)               0.12 ± 0.03   0.12 ± 0.03      0.12 ± 0.03   0.18 ± 0.06      0.12 ± 0.03
                XGBoost (CATE OR)          0.27 ± 0.08   0.27 ± 0.08      0.27 ± 0.08   0.51 ± 0.16      0.27 ± 0.08
                XGBoost (CATE DR)          0.15 ± 0.05   0.15 ± 0.04      0.15 ± 0.05   0.34 ± 0.13      0.15 ± 0.04

       Table 3. MSE (mean ± standard deviation) over 100 semi-synthetic datasets generated from the Minimum Wage dataset.

                                            Linear
                                                         Lasso (CV)      Ridge (CV)     Random Forest   Best
                                          Regression
                 XGBoost (OR)             1.97 ± 0.04    2.02 ± 0.05     1.96 ± 0.04    2.08 ± 0.09     2.07 ± 0.09
                 XGBoost (DR)             1.91 ± 0.04    1.88 ± 0.04     1.91 ± 0.04    1.8 ± 0.09      1.8 ± 0.09
                 XGBoost (CATE OR)        2.72 ± 0.06    2.71 ± 0.07     2.73 ± 0.06    3.4 ± 0.36      2.69 ± 0.07
                 XGBoost (CATE DR)        2.73 ± 0.06    2.7 ± 0.06      2.73 ± 0.07    3.47 ± 0.3      2.66 ± 0.07
                 Linear (OR)              1.96 ± 0.04    2.01 ± 0.04     1.96 ± 0.04    2.07 ± 0.08     2.06 ± 0.08
                 Linear (DR)              1.92 ± 0.04    1.89 ± 0.04     1.92 ± 0.04    1.83 ± 0.07     1.83 ± 0.07
                 Linear (CATE OR)         2.78 ± 0.05    2.76 ± 0.05     2.78 ± 0.05    3.14 ± 0.38     2.76 ± 0.05
                 Linear (CATE DR)         2.8 ± 0.05     2.75 ± 0.05     2.8 ± 0.05     3.13 ± 0.36     2.71 ± 0.05

Next consider the more general setting of a staggered roll-            where the datasets are generated from known data generat-
out, i.e. each treated unit (D = 1) is treated at some period          ing processes that satisfy the identifying assumptions de-
G = [1, T ] and remains treated after that period. We denote           scribed in Section 2.1. The data has 20 covariates, and
the never-treated group as G = ∞. In this setting, we can              the CATT learners look at the projection onto 5 covariates.
make the parallel trends assumption that for all g ∈ [1, T ]           We report the mean MSE (mean square error) between the
and for all t ≥ g E[Yt (0) − Y0 (0) | D = 1, W, G = g] =               predicted CATT and the true CATT on covariates of the
E[Yt (0) − Y0 (0) | D = 0, W, G = ∞]. Similarly, we can                treated units of a held out test set. We compare our results
incorporate heterogeneity as a function of the initial treat-          with the following baseline models, here gd (W ) = E[Y1 −
ment period G and the distance ∆ to the treatment period,              Y0 |W, D = d], g(W, D) = g1 (W )D + g0 (W )(1 − D), and
by making these variables as part of our heterogeneity set             π(W ) = P(D = 1|W ):
X: for all g ∈ [1, T ] and t ∈ [g, T ]:
                                                                          • Outcome regression (OR) learner: θ(X) = E[Y1 −
  E[Ypost (1) − Ypost (0) | D = 1, X, ∆ = ℓ, G = g]                         Y0 − g0 (W )|D = 1, X]
            = E[Yg+ℓ (1) − Yg+ℓ (0) | D = 1, X, G = g]
                                                                          • CATE outcome regression learner:             θ(X)     =
Prior work of Callaway & Sant’Anna, 2021 considers het-                     E[g1 (W ) − g0 (W )|D = 1, X]
erogeneity with respect to G and ∆, albeit in a fully non-
parametric manner and does not provide a method for model                 • CATE
                                                                                DR-learner: θ(X) = E[g(W, 1) − g(W, 0) +
selection, so as to uncover in a more data-driven manner the                    D      1−D
                                                                              π(W ) − 1−π(W )    (Y − g(W, D))|D = 1, X]
functional form of this heterogeneity. We note that the prior
work of Callaway & Sant’Anna, 2021 also considers doubly
                                                                       We considered three different final-stage models for the
robust estimation and inference on weighted averages of
                                                                       CATT: neural net, XGBoost, and linear models, to fit the
these heterogeneous effect models across different values of
                                                                       meta-learners. Simulation results are presented in Table 1.
G and ∆, which we do not discuss in this work.
                                                                       The results of linear models can be found in the Appendix 6.
                                                                       The columns represent different ML methods that are used
6. Experiments and Results                                             to learn the outcome regression. The propensity function,
                                                                       i.e. P(D = 1|W ), is always fitted using logistic regression.
6.1. Fully Synthetic Data
                                                                       The ”Best” column, represents using the ML method that
First, to compare the results of our proposed method with              achieved the lowest out-of-sample MSE for the outcome
other baselines, we conducted fully simulated experiments              regression. In the Appendix, we also provide results that

                                                                 7

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

investigate the performance of our DiD CATT method and
the baselines even when the parallel trends assumption is
violated. The results in Appendix E suggest that the dou-
bly robust estimator reduces the MSE, as compared to the
baselines, even under the violation of parallel trends.
Moreover, we also consider unbalanced datasets, where
the size of the control group is much larger than that of the
treated group. In particular, the propensities of each unit was
lowered by a factor of 10. The results are presented in Table
2. We see that while all models suffered in performance, our
proposed doubly robust model still outperforms the other
meta-learners as it leverages the asymmetry of the CATT
definition to be more robust to unbalanced settings. Notably,
the proposed learner out-performs the doubly robust CATE
learner as discussed in Remark 3.4.
                                                                      Figure 2. Calibration plot for CATT of minimum wage with respect
6.2. Minimum Wage Case Study                                          to log county population.

                                                                      As a preliminary evaluation, we tested the performance of
                                                                      our methods on semi-synthetic data generated from this
                                                                      dataset. The semi-synthetic data was generated by boot-
                                                                      strapping the samples in the dataset and applying a chosen
                                                                      function for treatment assignment and treatment effect to
                                                                      compute the post-treatment outcomes. Since this dataset is
                                                                      rather low dimensional, we did not experiment with neural
                                                                      networks. The mean MSE with respect to the true CATT
                                                                      function is reported in Table 3. The results show that the
                                                                      proposed method out-performs outcome regression learners
                                                                      as well as the CATE learners.
                                                                      We then applied our method to the original real dataset. Fig-
                                                                      ure 1 shows the CATT prediction over different values of
                                                                      log county population. The three models used (linear re-
                                                                      gression, XGBoost, and kernel ridge regression) all showed
Figure 1. Predicted CATT with respect to log county population.       some extend of positive trends. This suggests that raising
                                                                      minimum wage might have a smaller negative effect on teen
                                                                      employment for counties with a larger population.
We applied our proposed approach to the minimum wage
                                                                      Validating CATT: Since we do not have access to ground
dataset that is also studied in Callaway & Sant’Anna, 2021
                                                                      truth treatment effects for real datasets, we need a way to
and Callaway, 2023. This dataset studies the effect of mini-
                                                                      validate the heterogeneity that is picked up by the model
mum wage changes on teen employment during the period
                                                                      is not due to noise. One approach is through calibration.
2001–2007. The outcome variable of interest is the log
                                                                      The first step of the procedure is quantile binning the CATT
of county-level teen employment, while the treatment vari-
                                                                      predictions on a held out validation set. Next, the CATT
able is defined as a binary indicator representing whether
                                                                      predictions on a held out test set will be put into the bins
a county’s minimum wage exceeds the federal minimum
                                                                      according to the tresholds. The group average treatment
wage. The dataset includes covariates such as county pop-
                                                                      effect on the treated (GATT) for each bin is calculated as
ulation and average annual pay, which serve as controls to
                                                                      the mean of the heterogeneous model predictions, as well
account for differences in local economic conditions. For
                                                                      as calculating the unconditional ATT for each group (i.e.
the ease of interpretation, we focus only on the raise in
                                                                      conditioning on the empty set, we get θ = E[Yb ]/E[D]).
minium wage at year 2004 as the treatment. In our analysis,
                                                                      If the heterogeneity is indeed significant, we expect the
we treat years after treatment assignment time (2004) as an
                                                                      calibration plots line up in the 45◦ line with non-overlapping
additional covariate to control for, as discussed in Section 5.
                                                                      confidence intervals.
Employment rates as well as other covariates from before
2003 are also used as the covariates.                                 Figure 2 presents the calibration plot for the doubly robust

                                                                  8

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

CATT learner realized using XGBoost. While we see that              Chernozhukov, V., Newey, M., Newey, W. K., Singh, R.,
the GATT for the lowest quantile has a larger confidence in-          and Srygkanis, V. Automatic debiased machine learning
tervals, the highest most two quantiles have non-overlapping          for covariate shifts. arXiv preprint arXiv:2307.04527,
confidence intervals. This suggests that there is significant         2023.
heterogeneity between high and low populations. Together
with Figure 1, the results seem to suggest that the treatment       Chiu, A., Lan, X., Liu, Z., and Xu, Y. What to do (and not
effect for counties with large populations is close to zero.          to do) with causal panel analysis under parallel trends:
                                                                      Lessons from a large reanalysis study. arXiv preprint
                                                                      arXiv:2309.15983, 2023.
Impact Statement
                                                                    Daw, J. R. and Hatfield, L. A. Matching and regression to
This paper presents work whose goal is to advance the field
                                                                      the mean in difference-in-differences analysis. Health
of Causal Inference and Machine Learning. There are many
                                                                      services research, 53(6):4138–4156, 2018.
potential societal consequences of our work, none which we
feel must be specifically highlighted here.                         Dimick, J. B. and Ryan, A. M. Methods for evaluat-
                                                                      ing changes in health care policy: the difference-in-
References                                                            differences approach. Jama, 312(22):2401–2402, 2014.
Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I.,       Finkelstein, A., Taubman, S., Wright, B., Bernstein, M.,
  and Zrnic, T. Prediction-powered inference. Science, 382            Gruber, J., Newhouse, J. P., Allen, H., Baicker, K., and
  (6671):669–674, 2023.                                               Oregon Health Study Group, t. The oregon health insur-
Angrist, J. D. and Pischke, J.-S. Mostly harmless econo-              ance experiment: evidence from the first year. The Quar-
  metrics: An empiricist’s companion. Princeton university            terly journal of economics, 127(3):1057–1106, 2012.
  press, 2009.
                                                                    Foster, D. J. and Syrgkanis, V. Orthogonal statistical learn-
Antonelli, J., Rubinstein, M., Agniel, D., Smart, R., Stuart,         ing. The Annals of Statistics, 51(3):879–908, 2023.
  E., Cefalu, M., Schell, T., Eagan, J., Stone, E., Griswold,
  M., et al. Autoregressive models for panel data causal            Gao, Y., Li, M., Xue, J., and Liu, Y. Evaluation of effec-
  inference with application to state-level opioid policies.          tiveness of china’s carbon emissions trading scheme in
  arXiv preprint arXiv:2408.09012, 2024.                              carbon mitigation. Energy Economics, 90:104872, 2020.

Ashenfelter, O. C. and Card, D. Using the longitudinal              Gerber, A. and Green, D. Field Experiments: Design, Anal-
  structure of earnings to estimate the effect of training            ysis, and Interpretation. W. W. Norton, 2012. ISBN
  programs, 1984.                                                     9780393979954. URL https://books.google.
                                                                      com/books?id=yxEGywAACAAJ.
Callaway, B. Difference-in-differences for policy evaluation.
  Handbook of Labor, Human Resources and Population                 Heckman, J. J., Ichimura, H., and Todd, P. E. Matching as
  Economics, pp. 1–61, 2023.                                          an econometric evaluation estimator: Evidence from eval-
                                                                      uating a job training programme. The review of economic
Callaway, B. and Sant’Anna, P. H. Difference-in-differences
                                                                      studies, 64(4):605–654, 1997.
  with multiple time periods. Journal of econometrics, 225
  (2):200–230, 2021.                                                Kennedy, E. H. Towards optimal doubly robust estimation
Card, D. and Krueger, A. B. Minimum wages and em-                     of heterogeneous causal effects. Electronic Journal of
  ployment: A case study of the fast-food industry in                 Statistics, 17(2):3008–3049, 2023.
  new jersey and pennsylvania. The American Economic
                                                                    Künzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. Met-
  Review, 84(4):772–793, 1994. ISSN 00028282. URL
                                                                      alearners for estimating heterogeneous treatment effects
  http://www.jstor.org/stable/2118030.
                                                                      using machine learning. Proceedings of the national
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,            academy of sciences, 116(10):4156–4165, 2019.
  Hansen, C., and Newey, W. Double/debiased/neyman ma-
  chine learning of treatment effects. American Economic            Lan, H. and Syrgkanis, V. Causal q-aggregation for cate
  Review, 107(5):261–265, 2017.                                       model selection. In International Conference on Artificial
                                                                      Intelligence and Statistics, pp. 4366–4374. PMLR, 2024.
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
  Hansen, C., Newey, W., and Robins, J. Double/debiased             Miyaji, S. Instrumented difference-in-differences with
  machine learning for treatment and structural parameters,          heterogeneous treatment effects.      arXiv preprint
  2018.                                                              arXiv:2405.12083, 2024.

                                                                9

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Nie, X. and Wager, S. Quasi-oracle estimation of hetero-                methods to evaluate policy effects with staggered adop-
  geneous treatment effects. Biometrika, 108(2):299–319,                tion: an application to medicaid and hiv. arXiv preprint
  2021.                                                                 arXiv:2402.12576, 2024.

Ogburn, E. L., Rotnitzky, A., and Robins, J. M. Doubly                Wang, G., Hamad, R., and White, J. S. Advances in
  robust estimation of the local average treatment effect              difference-in-differences methods for policy evaluation
  curve. Journal of the Royal Statistical Society Series B:            research. Epidemiology, 35(5):628–637, 2024.
 Statistical Methodology, 77(2):373–396, 2015.

Oprescu, M., Syrgkanis, V., and Wu, Z. S. Orthogonal
  random forest for causal inference. In International Con-
  ference on Machine Learning, pp. 4932–4941. PMLR,
  2019.

Pierce, J. R. and Schott, P. K. The surprisingly swift decline
  of us manufacturing employment. American Economic
  Review, 106(7):1632–1662, 2016.

Rossin-Slater, M., Ruhm, C. J., and Waldfogel, J. The
  effects of california’s paid family leave program on moth-
  ers’ leave-taking and subsequent labor market outcomes.
  Journal of Policy Analysis and Management, 32(2):224–
  245, 2013.

Roth, J., Sant’Anna, P. H., Bilinski, A., and Poe, J. What’s
  trending in difference-in-differences? a synthesis of the
  recent econometrics literature. Journal of Econometrics,
  235(2):2218–2244, 2023.

Sant’Anna, P. H. and Zhao, J. Doubly robust difference-in-
  differences estimators. Journal of econometrics, 219(1):
  101–122, 2020.

Semenova, V. and Chernozhukov, V. Debiased machine
  learning of conditional average treatment effects and other
  causal functions. The Econometrics Journal, 24(2):264–
  289, 2021.

Shalit, U., Johansson, F. D., and Sontag, D. Estimating
  individual treatment effect: generalization bounds and
  algorithms. In International conference on machine learn-
  ing, pp. 3076–3085. PMLR, 2017.

Shi, C., Blei, D., and Veitch, V. Adapting neural networks
  for the estimation of treatment effects. Advances in neural
  information processing systems, 32, 2019.

Sun, L. and Abraham, S. Estimating dynamic treatment
  effects in event studies with heterogeneous treatment ef-
  fects. Journal of econometrics, 225(2):175–199, 2021.

Syrgkanis, V., Lei, V., Oprescu, M., Hei, M., Battocchi, K.,
  and Lewis, G. Machine learning estimation of heteroge-
  neous treatment effects with instruments. Advances in
  Neural Information Processing Systems, 32, 2019.

Thome, J. C., Rebeiro, P. F., Spieker, A. J., and Shep-
  herd, B. E. Understanding difference-in-differences

                                                                 10

                           A Meta-learner for Heterogeneous Effects in Difference-in-Differences

A. DiD with Instruments
As an extension, we consider a widely encountered setting of estimating heterogeneous treatment effects from panel data
with a binary instrument Z, with two sided non-compliance. In this setting, the target estimand is the conditional local
average treatment effect among the exposed (CLATT) in the second (post-treatment) time period:
                                 θ0 (X) = E[Y1 (1) − Y1 (0) | D1 (1) > D1 (0), Z = 1, X]
First, we consider the natural conditional extensions, which allows for more heterogeneity and flexibility, of the parallel
trends assumptions stated in Miyaji, 2024.
Assumption A.1 (No carryover assumption). Let d = (d0 , d1 ) denote the treatment path, then Y0 (d, z) = Y0 (d0 , z) and
Y1 (d, z) = Y1 (d1 , z).

This assumption requires that the outcome is only affected by the current treatment, in other words, there is no carry over
effects from previous treatments.
Assumption A.2 (Exclusion restriction for potential outcomes). For all t, Yt (d, z) = Yt (d)

This is the standard exclusion restriction assumption for instrumental variables that the instrument only affects the outcome
through the treatment.
Assumption A.3 (Monotonicity Assumption). P(D1 (1) ≥ D1 (0)) = 1 or P(D1 (1) ≤ D1 (0)) = 1

This assumption requires that the effect of the instrument is monotone - that it either increases treatment adoption or
decreases treatment adoption, but not both. This assumption is needed for identification under two-sided non-compliance.
Assumption A.4 (No anticipation in treatment). D0 (1) = D0 (0) for all units with Z = 1

Similar to the standard no-anticipation assumption that the treatment assignment in the second period should not have an
affect on the outcome in the first period, here we assume that the exposure event that happens at the second period should
not have an anticipatory effect on the treatment adoption in the first period.
Assumption A.5 (CPTA in Treatment).
                                              E[D1 (0) − D0 (0)|Z = 0, W ]
                                               = E[D1 (0) − D0 (0)|Z = 1, W ]

Here we no longer require the instrument to be independent with the potential outcome of the treatments, but instead require
that the trend under no exposure is (mean) independent to the exposure.
Assumption A.6 (CPTA in Outcome).
                                          E[Y1 (D1 (0)) − Y0 (D0 (0))|Z = 0, W ]
                                          = E[Y1 (D1 (0)) − Y0 (D0 (0))|Z = 1, W ]
Assumption A.7 (Sufficient Overlap in Instrument). For all W , there exist c > 0 such that c ≤ P(Z = 1|W ) ≤ 1 − c.

Moreover if the instrument does not have any effects on the treatment, the local average treatment effect will also not be
identified. Hence, we need the following assumption to low-bound the effects of the instrument on the treatment.
Assumption A.8 (Strong Instrument under PTA). There exist cz > 0 such that

                                 E[D1 − D0 − E[D1 − D0 | Z = 0, W ] | Z = 1, X] ≥ cz

Proposition A.9. Under Assumptions A.2, A.3, A.4, A.5, A.6, and A.8, the CLATE, θ0 (W ), can be identified as:
                                           E[Y1 − Y0 − E[Y1 − Y0 | Z = 0, W ] | Z = 1, X]
                              θ0 (W ) =
                                          E[D1 − D0 − E[D1 − D0 | Z = 0, W ] | Z = 1, X]

In the special case that W = X under the parallel trends assumption, the problem is again equivalent to the standard IV
problem, and Syrgkanis et al., 2019 proposed a doubly robust algorithm for estimating the heterogeneous LATE. For more
discussion, see Appendix C.

                                                             11

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Lemma A.10 (Doubly Robust Conditional Moment Restriction for CLATE). Under Assumptions A.2, A.3, A.4, A.5, and
A.6, A.7, the true CLATE is a solution to the following conditional moment equation:
                                 h                                                   i
                               E Zb {(∆Y − gY (W )) − (∆Y − gD (W ))θ(X)} X = 0

where ∆S = S1 − S0 for S = Y or D, gS (W ) = E[S1 − S0 |Z = 0, W ], and Zb = Z−P(Z=1|W )
                                                                             1−P(Z=1|W ) . Moreover, this moment
is Neyman orthogonal with respect to all nuisance functions.
Proposition A.11. Consider the incomplete squared loss:

                             LIV (θ; π0 , g0,Y , g0,D )
                                  h                                              i
                             = E Zb (∆D − g0,D (W ))θ(X)2 − 2(∆Y − g0,Y (W ))θ(X)

where∆S = S1 − S0 for S = Y or D, g0,S (W ) = E[S1 − S0 |D = 0, W ], and Z     b = Z−π0 (W ) for π0 (W ) = P(Z = 1|W ).
                                                                                    1−π0 (W )
Under the same assumptions as in Lemma A.10, the minimizer of L(θ; π0 , g0,Y , g0,D ) over any hypothesis space Θ is
equivalent to the solution to the best-projection problem of the CATT among the treated:

                                      min E[(θ(X) − θ0 (X))2 | Z = 1, D(1) > D(0)]
                                      θ∈Θ

Theorem A.12 (CLATE   p Rates). Let π̂, ĝD , ĝY be estimates of the nuisance functions, constructed using an auxiliary dataset.
Let ∥θ∥D=1,CM = E[θ(X)2 |D = 1, D(1) > D(0)] denote the L2 norm over the compliers among the treated population.
Let θ̂ be the result of any estimation process using n samples, satisfying w.p. 1 − δ:
                                                                           
                                                                                   2
                                           E LIV (θ̂; η̂) − inf LIV (θ; η̂) ≤ Rn,δ
                                                            θ∈Θ

Define the nuisance errors to be:
                                         "                                           2 # 21
                                              π0 (W ) − π(W )
                       Error(π, gD ) := E E                     (g0,D (W ) − gD (W )) X
                                                 1 − π(W )
                                         "                                           2 # 12
                                              π0 (W ) − π(W )
                       Error(π, gY ) := E E                     (g0,Y (W ) − gY (W )) X
                                                 1 − π(W )

Suppose Assumptions A.2, A.3, A.4, A.5, and A.6, A.7 are satisfied. Moreover, assume that there exist finite constant B such
that |θ(X)| ≤ B for all X with positive measure and all θ ∈ Θ. If the hypothesis space Θ is convex or is well specified (i.e.
                                                                  chk
θ0 ∈ Θ), and Error(π, gD ) is sufficiently small (Error(π, gD ) ≤ 8B 2 ), then θ satisfies, w.p. 1 − δ:

                                                                                              !
                2                         4             2              max(4B 2 , 2)
∥θ(X) − θ∗ (X)∥D=1,CM ≤                 2              Rn +                 2                 (Error(π, ĝY ) + Error(π, ĝD ))
                              hk − 8Bc Error(π, gD )            c hk − 8Bc Error(π, gD )

where h = P(Z = 1), k = P(D(1) > D(0)|Z = 1), and

                                              θ∗ ∈ arg min ∥θ(X) − θ0 (X)∥2Θ
                                                        θ∈Θ

B. Alternate Assumptions: Lagged Dependent Outcome
The parallel trends assumption guards against linear, time invariant, additive confounding. However, this may be unrealistic
in practice. For instance, it might be sensible for the increment with respect to time to depend on the initial level of the
outcome, i.e. Y1 (0) − Y0 (0) ∝ Y0 (0). One may also be interested in the natural extension to the parallel trends assumptions
that also accounts for more complicating confounding pattern. An popular alternative to model panel data is through the
lagged dependent variable assumption.

                                                              12

                           A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Definition B.1 (Outcome Support). Let Ydt denote the support of the outcome at time t for the cohort with treatment
assignment d.
Assumption B.2 (Lagged Dependent Outcome with Covariates). E[Y1 (0)|Y0 (0) = y, D = 1, W ] = E[Y1 (0)|Y0 (0) =
y, D = 0, W ] for all y ∈ Y00 , and x ∈ X.

Note that this assumption may be seen as a special case of Assumption 2.1, where the conditioning variable also includes
the pre-treatment outcome as well as the observed covariates. This assumption might be more convincing in some practical
applications. For instance, in wage studies, current income is generally believed to he highly dependent on past income.
However, in cases where the distribution of outcome is significantly different between the treated and untreated groups, this
assumption might lead an increase in bias due to matching the pre-treatment outcomes, as shown in (Daw & Hatfield, 2018).
Assumption B.3 (Overlap in Pre-treatment Outcome). Y10 ⊆ Y00

Note this is a testable assumption, and one can also perform a diagnostic test to ensure that the treated and control outcome
distributions have sufficient overlap.
Proposition B.4. Under Assumptions B.2 and B.3, the CATT, θ0 (W ), can be identified as:

                   θ0 (X) = E[Y1 (1) − Y1 (0)|D = 1, X] = E[Y1 |D = 1, X] − E[g(Y0 , W )|D = 1, X]

where g(y, W ) = E[Y1 |D = 0, Y0 = y, W ]

Proof.
                                           Z
                   E[Y1 (0)|D = 1, X] =            E[Y1 (0)|Y0 = y, D = 1, W ]p(Y0 = y, W |D = 1, X)dy
                                             Y
                                           Z 10
                                       =           E[Y1 (0)|Y0 = y, D = 0, W ]p(Y0 = y, W |D = 1, X)dy
                                             Y10
                                           Z
                                       =           E[Y1 |Y0 = y, D = 0, W ]p(Y0 = y, W |D = 1, X)dy
                                             Y10
                                       = E[g(Y0 , X)|D = 1, X]

Similarly, we may also replace the parallel assumptions for IV-CATT by the conditional lagged dependent variable
assumptions for both the outcome and treatment.
Assumption B.5 (Lagged Treatment).

                                    E[D1 (0)|Z = 0, W, D0 ] = E[D1 (0)|Z = 1, W, D0 ]

Assumption B.6 (Lagged Outcome).

                                E[Y1 (D1 (0))|Z = 0, W, Y0 ] = E[Y1 (D1 (0))|Z = 1, W, Y0 ]

Under the lagged outcome framework, we need a slightly different notion of strong instruments as in the PTA framework.
Assumption B.7 (Strong Instrument under Lagged Outcome). There exist cz > 0 such that

                                      |E[D1 − E[D1 |Z = 1, W, D0 ]|Z = 1, X]| ≥ cz

Proposition B.8. Under Assumptions A.2, A.3, A.4, B.5, and B.6, the CLATE, θ0 (W ), can be identified as:

                                                E[Y1 − E[Y1 | Z = 0, Y0 , W ] | Z = 1, X]
                                  θ0 (X) =
                                               E[D1 − E[D1 | Z = 0, D0 , W ] | Z = 1, X]

                                                              13

                           A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Unifying the two assumptions: First we observe that both Proposition 2.3 and B.4 shares the same general form:

                                             θ0 (X) = E[S − g(V )|D = 1, X]

where S is an observed outcome random variable and g(V ) is a nuisance function that is the conditional expectation
E[S|D = 0, V ] on some covariates V , which is a superset of X. Under the parallel trends assumption, S = Y1 − Y0 and
V = W , and under the lagged outcome assumption, S = Y1 and V = [W, Y0 ]. Thus, we see that the results in Section 3
can be generalized to the lagged dependent variable assumption. For IV-DID, similarly to the standard DiD case, when
considering X ⊂ W , we can rewrite the identification in Proposition A.9 so that the CLATE is the solution as:

                                                    E[SY − gY (VY ) | Z = 1, X]
                                         θ0 (X) =
                                                    E[SD − gD (VD ) | Z = 1, X]

where Sr is an observed outcome random variable for r = Y, D, and gr (V ) is a nuisance function that is the conditional
expectation E[Sr |Z = 0, V ] on some covariates V , which is a superset of X.

C. Conditioning on the full set of W
C.1. Standard DID
Here, we consider the case where X = W , and are interested in estimating:

                                         θ0 (W ) = E[Y1 (1) − Y1 (0)|D1 = 1, W ]

Leveraging the no-anticipation and (conditional) parallel trends assumption, we can identify this as:

                               θ0 (W ) = E[Y1 − Y0 |D1 = 1, W ] − E[Y1 − Y0 |D1 = 0, W ]

Note that this shares the same form of identification with the conditional average treatment effect (CATE) under conditional
ignorability, but with the differences as the outcome:

                           CAT E = E[Y (1) − Y (0)|W ] = E[Y |W, D = 1] − E[Y |W, D = 0]

In other words, the meta-learners of CATT can be constructed the same way that was constructed for CATE using the
difference in outcome. For instance, the doubly-robust pseudo-outcome can be constructed as:
                                                                            
                                                           D     1−D
                      Y DR = g(1, W ) − g(0, W ) +            −                  (Y1 − Y0 − g(D, W ))
                                                         π(W ) 1 − π(W )

where g(D, W ) is an estimator for the conditional expectation E[Y1 − Y0 |W, D], and π(W ) is an estimator of the propensity
P(D = 1|W ). The nuisance functions g(D, W ) and π(W ) may be estimated using any ML methods.
As in (Kennedy, 2023), the doubly robust learner (DR-learner) can be constructed as θ(W ) = E[Y DR |W ]. Note that due
to the asymmetry of the parallel trends assumption, this estimator gives the conditional average treatment effect of the
treated. If we further assume that the treatment effects are also mean independent of the treatment conditional on W (i.e.
E[Y1 (1) − Y1 (0)|D = 1, W ] = E[Y1 (1) − Y1 (0)|D = 0, W ]), then the CATE estimator on the difference in the outcomes
identifies the conditional average treatment effects.
However, this CATE pseudo-outcome will give the biased estimate of the CATT when projecting on a subset of covariates, i.e.
θCAT E (X) = E[Y DR |X] where X ⊂ W . Here we see that θCAT E (X) = E[E[Y DR |W ]|X] = E[E[Y1 (1) − Y1 (0)|D =
1, W ]|X] ̸= E[E[Y1 (1) − Y1 (0)|D = 1, W ]|D = 1, X] = E[Y1 (1) − Y1 (0)|D1 = 1, X]. This difference will be more
pronounced for datasets where there is a big difference in the covariate distribution between the treated and un-treated
groups.

                                                            14

                           A Meta-learner for Heterogeneous Effects in Difference-in-Differences

C.2. IV-DID
In this section, we show that when conditioning on the full set of variables W , the CLATE can be estimated by the DR-IV
learner in (Syrgkanis et al., 2019). By the standard LATE identification argument we can write:

                                                      E[(Y1 (1) − Y1 (0)) 1{D1 (1) > D1 (0)} | Z = 1, W ]
    E[Y1 (1) − Y1 (0) | D1 (1) > D1 (0), Z = 1, W ] =
                                                                P(D1 (1) > D1 (0) | Z = 1, W )
                                                      E[(Y1 (D1 (1)) − Y1 (D1 (0))) 1{D1 (1) > D1 (0)} | Z = 1, W ]
                                                    =
                                                                      E[D1 (1) − D1 (0) | Z = 1, W ]
                                                      E[Y1 (D1 (1)) − Y1 (D1 (0)) | Z = 1, W ]
                                                    =
                                                          E[D1 (1) − D1 (0) | Z = 1, W ]

Under the parallel trends assumption in the outcome, the numerator is identified as:

          E[Y1 (D1 (1)) − Y1 (D1 (0)) | Z = 1, W ] = E[Y1 − Y0 | Z = 1, W ] − E[Y1 (D(0)) − Y0 | Z = 1, W ]
                                                   = E[Y1 − Y0 | Z = 1, W ] − E[Y1 − Y0 | Z = 0, W ]

Moreover, under the parallel trends assumption in the treatment, the denominator is identified as:

                E[D1 (1) − D1 (0) | Z = 1, W ] = E[D1 − D0 | Z = 1, W ] − E[D1 (0) − D0 | Z = 1, W ]
                                                = E[D1 − D0 | Z = 1, W ] − E[D1 − D0 | Z = 1, W ]

For brevity of notation, let Y denote Y1 − Y0 and let D denote D1 − D0 Thus, under the PTA assumptions the effect is
identified as:
                                                E[Y | Z = 1, W ] − E[Y | Z = 0, W ]
                                    θ0 (W ) =
                                                E[D | Z = 1, W ] − E[D | Z = 0, W ]

Moreover, note that we can also write these quantities as conditional covariances.
In particular, let α(W ) = E[Y | Z = 1, W ] − E[Y | Z = 0, W ] and γ(W ) = E[Y | Z = 0, W ]. Without loss of generality
we can write:

            E[Y | Z, W ] = Z(E[Y | Z = 1, W ] − E[Y | Z = 0, W ]) + E[Y | Z = 0, W ] = Zα(W ) + γ(W )

Thus we can write:

                          Y = Zα(W ) + γ(W ) + ϵ,                               E[ϵ | Z, W ] = 0

Then we have:

                                      Cov(Y, Z | W ) = E[Ỹ Z̃ | W ] = E[Y Z̃ | W ]

where Ỹ = Y − E[Y | W ] and Z̃ = Z − π0 (W ) = Z − E[Z | W ].
Moreover, note that:

                       E[Y Z̃ | W ] = E[α(W )Z Z̃] + E[γ(W )Z̃ | W ] + E[ϵZ̃ | W ]
                                   = α(W )Var(Z | W ) + γ(W )E[Z̃ | W ] + E[E[ϵ | Z, W ]Z̃ | W ]
                                   = α(W )Var(Z | W )

Thus we have:

                  Cov(Y, Z | W ) = E[Y Z̃ | W ] = (E[Y | Z = 1, W ] − E[Y | Z = 0, W ]) Var(Z | W )

                                                            15

                              A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Similarly, we can derive:

                  Cov(D, Z | W ) = E[DZ̃ | W ] = (E[D | Z = 1, W ] − E[D | Z = 0, W ]) Var(Z | W )

Thus we have deduced that we can equivalently identify the conditional LATE among the exposed as:

                        E[Y Z̃ | W ]   Cov(Y, Z | W )    (E[Y | Z = 1, W ] − E[Y | Z = 0, W ]) Var(Z | W )
           θ0 (W ) =                 =                =
                        E[DZ̃ | W ]    Cov(D, Z | W )    (E[D | Z = 1, W ] − E[D | Z = 0, W ]) Var(Z | W )
                        E[Y | Z = 1, W ] − E[Y | Z = 0, W ]
                      =
                        E[D | Z = 1, W ] − E[D | Z = 0, W ]

Let α̂ be an estimate of:

                         a0 (W ) := E[Y Z̃ | W ] = (E[Y | Z = 1, W ] − E[Y | Z = 0, W ]) Var(Z | W )

and β̂ an estimate of:

                         β0 (W ) := E[DZ̃ | W ] = (E[D | Z = 1, W ] − E[D | Z = 0, W ]) Var(Z | W )

and let θ̂ = α̂/β̂. Then we can construct the random variable

                                                                   (Y − θ̂(W )D)Z̃
                                              Ŷ (ĝ) = θ̂(W ) +
                                                                       β̂(W )

and the moment equation for the conditional LATT is:

                                                   ϕ = E[Ŷ (ĝ) − θ(W )|W ]

Similar to the standard DiD case, projecting onto a lower dimensional subset of covariates will give a biased estimated of
the CLATE.

D. Proofs
D.1. Identification
Proof of Proposition 2.3.

                  E[Y1 (1) − Y1 (0)|D = 1, X]
               = E[Y1 (1) − Y0 (1)|D = 1, X] − E[Y1 (0) − Y0 (1)|D = 1, X]
               = E[Y1 − Y0 |D = 1, X] − E[Y1 (0) − Y0 (0)|D = 1, X]                                    (By Assumption 2.2)
               = E[Y1 − Y0 |D = 1, X] − E{E[Y1 (0) − Y0 (0)|D = 1, W ]|D = 1, X}
               = E[Y1 − Y0 |D = 1, X] − E{E[Y1 (0) − Y0 (0)|D = 0, W ]|D = 1, X}                       (By Assumption 2.1)
               = E[Y1 − Y0 − E[Y1 (0) − Y0 (0)|D = 0, W ]|D = 1, X]

Proof of Proposition A.9. Here we want to show that the CLATE, θ0 (X), can be identified as:

                                            E[Y1 − Y0 − E[Y1 − Y0 | Z = 0, W ] | Z = 1, X]
                                θ0 (X) =
                                           E[D1 − D0 − E[D1 − D0 | Z = 0, W ] | Z = 1, X]

                                                               16

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

We first analyze the denominator:

          E[D1 − D0 − E[D1 − D0 | Z = 0, X] | Z = 1, X]
      = E[D1 (1) − D0 (1) − E[D1 (0) − D0 (0) | Z = 0, W ] | Z = 1, X]
      = E[D1 (1) − D1 (0) + D1 (0) − D0 (1) − E[D1 (0) − D0 (0) | Z = 0, W ] | Z = 1, X]
      = E[D1 (1) − D1 (0) + D1 (0) − D0 (0) − E[D1 (0) − D0 (0) | Z = 0, W ] | Z = 1, X]                (By Assumption A.4)
      = E[D1 (1) − D1 (0) + E[D1 (0) − D0 (0)|Z = 1, W ] − E[D1 (0) − D0 (0) | Z = 0, W ] | Z = 1, X]
      = E[D1 (1) − D1 (0) | Z = 1, X]                                                                   (By Assumption A.5)
      = P(D1 (1) > D1 (0)|Z = 1, X)                                                                     (By Assumption A.3)

Now we analyze the numerator:

   E[Y1 − Y0 − E[Y1 − Y0 | Z = 0, W ] | Z = 1, X]
= E[Y1 (D(1)) − Y0 (D(1)) − E[Y1 (D(0)) − Y0 (D(0)) | Z = 0, W ] | Z = 1, X]
= E[Y1 (D(1)) − Y1 (D(0)) + Y1 (D(0)) − Y0 (D(0)) − E[Y1 (D(0)) − Y0 (D(0)) | Z = 0, W ] | Z = 1, X]
                                                                                             (By Assumption A.4)
= E[Y1 (D(1)) − Y1 (D(0)) + E[Y1 (D(0)) − Y0 (D(0))|Z = 1, W ] − E[Y1 (D(0)) − Y0 (D(0)) | Z = 0, W ] | Z = 1, X]
= E[Y1 (D(1)) − Y1 (D(0)) | Z = 1, X]                                                                   (By Assumption A.6)
= E[(D(1) − D(0))(Y1 (1) − Y1 (0)) | Z = 1, X]                                                          (By Assumption A.3)
= E[Y1 (1) − Y1 (0) | Z = 1, D(1) > D(0), X]P(D(1) > D(0)|Z = 1, X)

Thus, combining them, we get:

           E[Y1 − Y0 − E[Y1 − Y0 | Z = 0, W ] | Z = 1, X]
                                                          = E[Y1 (1) − Y1 (0) | Z = 1, D(1) > D(0), X]
          E[D1 − D0 − E[D1 − D0 | Z = 0, W ] | Z = 1, X]

D.2. Orthogonal Moments
Proof of Lemma 3.3. First, we show that the true CATT function θ0 (X) = E[Y1 (1) − Y1 (0)|D = 1, X] is the solution to
the moment:

                                                                                                    
                                                          D − π0 (W )
                  E [m(Z; θ0 , g0 , π0 )|X] = E                              (∆Y − g0 (W )) − Dθ0 (X) X = 0
                                                         (1 − π0 (W ))

where Z = (W, D, Y ), ∆Y = Y1 − Y0 , g0 (W ) = E[∆Y |D = 0, W ], π0 (W ) = P(D = 1|W ). First, since this
moment is conditioned on X, we can multiply by any functions of X. Thus, we can divide by the propensity with X, i.e.
γ0 (X) = P(D = 1|X), which is bounded away from zero:

                                                                                        
                                            D − π0 (W )
                                  E                            (∆Y − g0 (W )) − Dθ0 (X) X = 0
                                           (1 − π0 (W ))
                                                                    ⇕
                                                                                            
                                      D − π0 (W )                                 D
                         E                                     (∆Y − g0 (W )) −        θ0 (X) X = 0
                                  (1 − π0 (W ))γ0 (X)                           γ0 (X)

                                                                    17

                                 A Meta-learner for Heterogeneous Effects in Difference-in-Differences
                       h                    i
                             D
The latter term is E       γ0 (X) θ0 (X)   X = E[θ0 (X)|D = 1, X] = θ0 (X). Now we consider the first term:
                                                                 
                             D − π0 (W )
                       E                        (∆Y − g0 (W )) X
                         (1 − π0 (W ))γ0 (X)
                                                                           
                           D          (1 − D)π0 (W )
                 = E             −                        (∆Y − g0 (W )) X
                         γ0 (X) (1 − π0 (W ))γ0 (X)
                                                                                             
                         D                                   (1 − D)π0 (W )
                 = E          (∆Y − g0 (W )) X − E                             (∆Y − g0 (W )) X
                       γ0 (X)                             (1 − π0 (W ))γ0 (X)
                                                                                                
                                                              (1 − D)π0 (W )
                 = E [∆Y − g0 (W ) | D = 1, X] − E E                            (∆Y − g0 (W )) W X
                                                           (1 − π0 (W ))γ0 (X)
                                                                          
                                     π0 (W )
                 = θ0 (X) − E E              (∆Y − g0 (W )) D = 0, W X
                                     γ0 (X)
                                                                         
                                 π0 (W )
                 = θ0 (X) − E            E [∆Y − g0 (W ) | D = 0, W ] X
                                 γ0 (X)
                 = θ0 (X)

Thus, the moment condition is satisfied for the true CATT θ0 (X). Now, we show that the moment is Neyman orthogonal with
respect to all nuisance functions. It suffices to show that the directional derivative with respect to all the nuisance functions
are 0 when evaluated at the true nuisance and target functions. Recall that the directional derivative of a functional m(Z; f )
                                                                                                     d
with respect to the function f (W ) in the direction of ∆f (W ) is defined as: ∂f E[m(z; f )][∆f ] = dt E[m(z; f + t · ∆f )]         .
                                                                                                                               t=0

First, we look the directional derivative with respect to the outcome regression g(W ):
                                                                                         
                                                                   D − π0 (W )
                       ∂g E[m(Z; θ, g, π)|X][∆g]|θ0 ,g0 ,π0 = E                 ∆g(W ) X
                                                                   1 − π0 (W )
                                                                                             
                                                                       D − π0 (W )
                                                            = E E                    W ∆g(W ) X
                                                                       1 − π0 (W )
                                                                                            
                                                                   π0 (W ) − π0 (W )
                                                            = E                      ∆g(W ) X = 0
                                                                      1 − π0 (W )

Now, we look at the the directional derivative with respect to the outcome regression π(W ):
                                                                                                                 
                                               −(1 − π0 (W ))∆π(W ) + (D − π0 (W ))∆π(W )
 ∂π E[m(Z; θ, g, π)|X][∆π]|θ0 ,g0 ,π0 = E                                                          (∆Y − g0 (W )) X
                                                                  (1 − π0 (W ))2
                                                                                   
                                               ∆π(W )(D − 1)
                                      = E                           (∆Y − g0 (W )) X
                                                (1 − π0 (W ))2
                                                                                          
                                                   ∆π(W )(D − 1)
                                      = E E                            (∆Y − g0 (W )) W X
                                                    (1 − π0 (W ))2
                                                                                               
                                                    ∆π(W )
                                      = E −                      E [∆Y − g0 (W )) | D = 0, W ] X = 0
                                                  1 − π0 (W )

Thus, we have shown that this moment is Neyman orthogonal with respect to all nuisances.

Proof of Theorem 4.7. First, we show that the true estimand θ0 (X) = Esource [m(Z; g0 )|X] satisfies the following condi-
tional moment restriction
                                                                                                             
       h
          DR
                                  i                                 (1 − E)π0 (W )
     E m (Z; θ0 , g0 , π0 , α0 ) X = E E(m(Z; g0 )) − θ0 (X)) +                    α0 (W )(Y − g0 (W )) X = 0
                                                                      1 − π0 (W )

                                                                  18

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

where π0 (W ) = P(E = 1|W ) and α(W ) is the Riesz representer of Es [m(Z; g)|X]. Similar to the earlier the proof of
Lemma 3.3, we can divide both sides of the moment equation by γ0 (X) = P(E = 1|X) since it is bounded away from 0.
So it is equivalent to show:
                                                                                               
                         E                              (1 − E)π0 (W )
                  E           (m(Z; g0 )) − θ0 (X)) +                     α0 (W )(Y − g0 (W )) X = 0
                       γ0 (X)                         (1 − π0 (W ))γ0 (X)

First, let’s look at the first term:
                                                         
                              E
                       E           (m(Z; g0 )) − θ0 (X)) X = E [(m(Z; g0 )) − θ0 (X)) | E = 1, X] = 0
                           γ0 (X)

Thus, it remains to show that the second term also has conditional expectation of 0.
                                                                                   
                                        (1 − E)π0 (W )
                                  E                        α0 (W )(Y − g0 (W )) X
                                      (1 − π0 (W ))γ0 (X)
                                                                                       
                                            (1 − E)π0 (W )
                              = E E                            α0 (W )(Y − g0 (W )) W X
                                          (1 − π0 (W ))γ0 (X)
                                                                                     
                                          π0 (W )
                              = E E               α0 (W )(Y − g0 (W )) E = 0, W X
                                          γ0 (X)
                                                                                     
                                      π0 (W )
                              = E             α0 (W )E [(Y − g0 (W )) | E = 0, W ] X = 0
                                      γ0 (X)

Now, we proceed to show that the moment mDR (Z; θ, g, π, α) is Neyman orthogonal. First, we look at the directional
derivative with respect to the nuisance g(W ).

                             ∂g E[mDR (Z; θ, g, π, α)|X][∆g]|θ0 ,g0 ,π0 ,α0
                                                                                           
                                                             (1 − E)π0 (W )
                           = ∂g E[Em(Z; g)|X][∆g]|g0 − E                    α0 (W )∆g(W ) X
                                                               1 − π0 (W )

We first look at the first term:

       ∂g E[Em(Z; g)|X][∆g]             = ∂g E[γ0 (W )E[m(Z; g)|E = 1, X]|X][∆g]
                                   g0                                              g0

                                        = ∂g E[γ0 (X)E[α0 (W )g(W )|E = 1, X]|X][∆g]           (By the Definition of α0 (W ))
                                                                                        g0
                                        = E[γ0 (X)E[α0 (W )∆g(W )|E = 1, X]|X]
                                        = E[Eα0 (W )∆g(W )|W ]

Putting this back, we get:

                                 ∂g E[mDR (Z; θ, g, π)|X][∆g]|θ0 ,g0 ,π0 ,α0
                                                                                          
                                                        (1 − E)π0 (W )
                               = E Eα0 (W )∆g(W ) −                        α0 (W )∆g(W ) X
                                                          1 − π0 (W )
                                                                               
                                                          (1 − E)π0 (W )
                               = E α0 (W )∆g(W ) E −                            X
                                                            1 − π0 (W )
                                                                                  
                                                           (1 − E)π0 (W )
                               = E α0 (W )∆g(W )E E −                           W X =0
                                                             1 − π0 (W )

                                                               19

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Next, we look at the derivative with respect to π(W ):

                 ∂π E[mDR (Z; θ, g, π, α)|X][∆π]|θ0 ,g0 ,π0 ,α0
                                                                                                  
                       (1 − E)(1 − π(W ))∆π(W ) + (1 − E)π(W )∆π(W )
               = E                                                            α0 (W )(Y − g0 (W )) X
                                            (1 − π0 (W ))2
                                                                
                     (1 − E)∆π(W )
               = E                    α0 (W )(Y − g0 (W )) X
                      (1 − π0 (W ))2
                                                                     
                        (1 − E)∆π(W )
               = E E                      α0 (W )(Y  − g0 (W  ))  W   X
                         (1 − π0 (W ))2
                                                                       
                          ∆π(W )
               = E E                 α0 (W )(Y − g0 (W )) E = 0, W X
                        1 − π0 (W )
                                                                       
                       ∆π(W )
               = E               α0 (W )E [(Y − g0 (W ) | E = 0, W ] X = 0
                     1 − π0 (W )

Lastly, we show that the directional derivative with respect to α(W ) is equal to 0.

                                  ∂α E[mDR (Z; θ, g, π, α)|X][∆α]|θ0 ,g0 ,π0 ,α0
                                                                                
                                      (1 − E)π0 (W )
                                = E                  ∆α(W )(Y − g0 (W )) X
                                        1 − π0 (W )
                                                                                  
                                         (1 − E)π0 (W )
                                = E E                   ∆α(W )(Y − g0 (W )) W X
                                           1 − π0 (W )
                                = E [E [π0 (W )∆α(W )(Y − g0 (W )) | E = 0, W ] | X]
                                = E [π0 (W )∆α(W )E [Y − g0 (W ) | E = 0, W ] | X] = 0

Proof of Lemma A.10. First we show that the true CLATE, θ0 (X) = E[Y1 (1) − Y1 (0) | Z = 1, D(1) > D(0), X] , is the
solution to the following moment equation:
                                                 h                                              i
          E mDR (Z; θ0 , g0,Y , g0,D , π0 )|X = E Z
                                            
                                                   b {(∆Y − g0,Y (W )) − (∆D − g0,D (W ))θ(X)} X = 0

where ∆S = S1 −S0 for S = Y or D, g0,S (W ) = E[S1 −S0 |Z = 0, W ], and Zb = Z−π        0 (W )
                                                                                    1−π0 (W ) with π0 (W ) = P(Z = 1|W ).
We can apply same trick as in the other orthogonality proofs to divide by γ0 (X) = P(Z = 1|X). We first consider the first
term:
            "                               #
                 Zb
          E          (∆Y − g0,Y (W )) X
              γ0 (X)
                                                         
                   Z − π0 (W )
       = E                         (∆Y − g0,Y (W )) X
              (1 − π0 (W ))γ0 (X)
                                                                     
                   Z        (1 − Z)π0 (W )
       = E             −                          (∆Y − g0,Y (W )) X
                γ0 (X) (1 − π0 (W ))γ0 (X)
                                                                                                        
                   Z                                    π0 (W )          1−Z
       = E               (∆Y − g0,Y (W )) X − E                 E                     (∆Y − g0,Y (W )) W X
                γ0 (X)                                   γ0 (X)      (1 − π0 (W )))
                                                                                              
                                                 π0 (W )
       = E [∆Y − g0,Y (W ) | Z = 1, X] − E               E [(∆Y − g0,Y (W )) | Z = 0, W ] X
                                                 γ0 (X)
       = E [∆Y − g0,Y (W ) | Z = 1, X]

                                                             20

                           A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Similarly, for the second term:
                      "                              #          "                           #
                           Zb                                        Zb
                    E          (∆D − g0,D (W ))θ(X) X = θ0 (X)E          (∆D − g0,D (W )) X
                        γ0 (X)                                    γ0 (X)
                                                            = θ0 (X)E [∆D − g0,D (W ) | Z = 1, X]
By the definition of θ0 (X), this shows that it is a solution to the doubly robust moment equation. Now, we proceed to show
that the moment mDR (Z; θ, gY , gD , π) is Neyman orthogonal. First, we look at the directional derivative with respect to
the nuisance gY (W ).
                                                                            h                i
         ∂gY E[mDR (Z; θ, gY , gD , π)|X][∆gY ]                        = −E b Z∆gY (W ) X
                                                    θ0 ,g0,Y ,g0,D ,π0
                                                                                                    
                                                                              Z − π0 (W )
                                                                       = −E               ∆gY (W ) X
                                                                              1 − π0 (W )
                                                                                                           
                                                                                 Z − π0 (W )
                                                                       = −E E                  W ∆gY (W ) X = 0
                                                                                 1 − π0 (W )
Similarly,
                                                                       h                   i
      ∂gD E[mDR (Z; θ, gY , gD , π)|X][∆gD ]                      = E b Z∆gD (W )θ0 (X) X
                                               θ0 ,g0,Y ,g0,D ,π0
                                                                                                      
                                                                               Z − π0 (W )
                                                                  = θ0 (X)E E                W ∆gD (W ) X = 0
                                                                               1 − π0 (W )
Lastly, we check the directional derivative with respect to π(W ):

         ∂π E[mDR (Z; θ, gY , gD , π)|X][∆π]
                                               θ0 ,g0,Y ,g0,D ,π0
                                                                                                                
             −(1 − π0 (W )∆π(W ) + (Z − π0 (W )∆π(W )
       = E                                                        {(∆Y   − g0,Y (W )) − (∆D  − g0,D (W ))θ(X)} X
                              (1 − π0 (W ))2
                                                                                     
             (Z − 1)∆π(W )
       = E                     {(∆Y  −  g0,Y (W ))  −   (∆D    −  g0,D (W ))θ(X)}   X
              (1 − π0 (W ))2
                                                                                           
                (Z − 1)∆π(W )
       = E E                      {(∆Y − g0,Y (W )) − (∆D − g0,D (W ))θ(X)} W X
                 (1 − π0 (W ))2
                                                                                              
                 (∆π(W )
       = E E                 {(∆Y − g0,Y (W )) − (∆D − g0,D (W ))θ(X)} Z = 0, W X
                1 − π0 (W )
                                                                                                         
              (∆π(W )
       = E               {E [∆Y − g0,Y (W )|Z = 0, W ] − E [∆D − g0,D (W )|Z = 0, W ] θ(X)} X = 0
             1 − π0 (W )

D.3. Losses
Proof of Proposition 3.5. Note that the true CATT θ0 satisfies the conditional moment restrictions in Lemma 3.3, which
imply that:
                                               E[Dθ0 (X) | X] = E[Ŷ | X]
Hence, the loss L(θ; π0 , g0 ) at any function θ can be simplified as:
                                                       h                  i
                                     L(θ; π0 , g0 ) = E Dθ(X)2 − 2Yb θ(X)
                                                       h                        i
                                                    = E Dθ(X)2 − 2E[Yb | X]θ(X)
                                                    = E Dθ(X)2 − 2E[Dθ0 (X) | X]θ(X)
                                                                                   

                                                    = E Dθ(X)2 − 2Dθ0 (X)θ(X)
                                                                              

                                                            21

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Note that when the loss is evaluated at θ0 , then it takes the value E[−Dθ0 (X)2 ]. Moreover, note that minimizing L(θ; π0 , g0 )
is equivalent to minimizing the difference L(θ; π0 , g0 ) − L(θ0 ; π0 , g0 ), which in turn simplifies to:

                          E Dθ(X)2 − 2Dθ0 (X)θ(X) + Dθ0 (X)2 = E D(θ(X) − θ0 (X))2
                                                                                

Hence, minimizing L(θ; π0 , g0 ) over any space Θ is equivalent to minimizing over Θ the loss function:

                                                E (θ(X) − θ0 (X))2 | D = 1
                                                                         

Proof of Proposition A.11. Note that the true CATT θ0 satisfies the conditional moment restrictions in Lemma A.10, which
imply that:

                               E[Z(∆D
                                 b    − gD (W ))θ0 (X) | X] = E[Z(∆Y
                                                                b    − gY (W )) | X]

Let η0 denote the set of nuisance functions. The loss LIV (θ; η0 ) at any function θ can be simplified as:
                                  h                                                     i
                                   Z(∆D − gD (W ))θ(X)2 − 2 b
                  LIV (θ; η0 ) = E b                       Z(∆D − gD (W ))θ0 (X)θ(X)
                                  h                                                          i
                               = E Z(∆D
                                    b   − gD (W ))θ(X)2 − 2E[Z(∆D
                                                              b     − gD (W ))θ0 (X) | X]θ(X)
                                  h                                    i
                                   Z(∆D − gD (W )) θ(X)2 − 2θ0 (X)θ(X)
                               = E b

                                                                         Z(∆D − gD (W ))θ0 (X)2 ]. Moreover, note that
Note that when the loss is evaluated at θ0 , then it takes the value E[− b
minimizing LIV (θ; η0 ) is equivalent to minimizing the difference LIV (θ; η0 ) − LIV (θ0 ; η0 ), which in turn simplifies to:
          h                                                 i
         E Z(∆D
            b      − gD (W )) θ(X)2 − 2θ0 (X)θ(X) + θ0 (X)2
          h                                   i
       = E Z(∆D
            b      − gD (W ))(θ(X) − θ0 (X))2
                                                                
                  (1 − Z)π0 (W )                                 2
       = E Z−                      (∆D − gD (W ))(θ(X) − θ0 (X))
                    1 − π0 (W )
       = E Z(∆D − gD (W ))(θ(X) − θ0 (X))2 − E π0 (W )(θ(X) − θ0 (X))2 E [(∆D − gD (W )) | Z = 0, W ]
                                                                                                   

       = E Z(∆D − gD (W ))(θ(X) − θ0 (X))2
                                            

       = E (∆D − gD (W ))(θ(X) − θ0 (X))2 |Z = 1 P(Z = 1)
                                                 

       = E E [(∆D − gD (W ))|Z = 1, X] (θ(X) − θ0 (X))2 |Z = 1 P(Z = 1)
                                                              

       = E P(D1 (1) > D1 (0)|Z = 1, X)(θ(X) − θ0 (X))2 |Z = 1 P(Z = 1)
                                                            
                                                                            (See the proof of Proposition A.9)
                                           2
                                                   
       = E (D1 (1) > D1 (0))(θ(X) − θ0 (X)) |Z = 1 P(Z = 1)
       = E (θ(X) − θ0 (X))2 |Z = 1, D(1) > D(0) P(Z = 1)P(D(1) > D(0)|Z = 1)
                                               

Hence, minimizing LIV (θ; η0 ) over any space Θ is equivalent to minimizing over Θ the loss function:

                                        E (θ(X) − θ0 (X))2 | Z = 1, D(1) > D(0)
                                                                              

D.4. Rates
Before proving Theorem 3.6, we first present some auxiliary Lemmas.

                                                               22

                              A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Lemma D.1. Let η = (π, g) denote the set of nuisance functions, and let η0 be the true nuisance functions. Consider the
loss defined in Proposition 3.5. Then, we have that for all θ1 , θ2 , η1 and η2 ,
                                                                           s 
                                                                               h                      i2 
                |L(θ1 ; η1 ) − L(θ2 ; η1 ) − L(θ2 ; η1 ) + L(θ2 ; η2 )| ≤ 2 E E Yb (η1 ) − Yb (η2 ) X     ∥θ1 − θ2 ∥

.

Proof of Lemma D.1.

        |L(θ1 ; η1 ) − L(θ2 ; η1 ) − L(θ2 ; η1 ) + L(θ2 ; η2 )|
           h                                                       i h                                           i
      = E D(θ12 (X) − θ22 (X)) + 2Yb (η1 )(θ2 (X) − θ1 (X) − E D(θ12 (X) − θ22 (X)) + 2Yb (η2 )(θ2 (X) − θ1 (X))
           h                                         i
      = E 2 Yb (η1 ) − Yb (η2 ) (θ1 (X) − θ2 (X))
             h h                                              ii
      = 2 E E Yb (η1 ) − Yb (η2 ) (θ1 (X) − θ2 (X)) X
         s 
                  h                    i2 
      ≤ 2 E E Y (η1 ) − Y (η2 ) X
                    b         b              ∥θ1 (X) − θ2 (X)∥

We then show that the bias in the pseudo-outcome Yb is equal to the product of the biases in the nuisance functions.
Lemma D.2. Let η = (π, g) denote the set of nuisance functions, and let η0 be the true nuisance functions. Consider the
pseudo-outcome defined in Proposition 3.5. Then we have:
                                                                                           
                                                                         π0 (W ) − π̂(W )
                          E[Yb (η0 ) − Yb (η̂)|X] = E (ĝ(W ) − g0 (W ))                  X
                                                                            1 − π̂(W )

Proof of Lemma D.2.
