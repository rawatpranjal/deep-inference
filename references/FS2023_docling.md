## Orthogonal Statistical Learning

Dylan J. Foster Microsoft Research dylanfoster@microsoft.com

## Abstract

We provide non-asymptotic excess risk guarantees for statistical learning in a setting where the population risk with respect to which we evaluate the target parameter depends on an unknown nuisance parameter that must be estimated from data. We analyze a two-stage sample splitting meta-algorithm that takes as input arbitrary estimation algorithms for the target parameter and nuisance parameter. We show that if the population risk satisfies a condition called Neyman orthogonality , the impact of the nuisance estimation error on the excess risk bound achieved by the meta-algorithm is of second order. Our theorem is agnostic to the particular algorithms used for the target and nuisance and only makes an assumption on their individual performance. This enables the use of a plethora of existing results from machine learning to give new guarantees for learning with a nuisance component. Moreover, by focusing on excess risk rather than parameter estimation, we can provide rates under weaker assumptions than in previous works and accommodate settings in which the target parameter belongs to a complex nonparametric class. We provide conditions on the metric entropy of the nuisance and target classes such that oracle rates of the same order as if we knew the nuisance parameter are achieved.

## Contents

| 1 Introduction   | 1 Introduction                                                      | 1 Introduction                                                      |   4 |
|------------------|---------------------------------------------------------------------|---------------------------------------------------------------------|-----|
|                  | 1.1                                                                 | Related work . . . . . . . . . . . . . . . . . . .                  |   8 |
|                  | 1.2                                                                 | Organization . . . . . . . . . . . . . . . . . . .                  |   9 |
| 2                | Framework: Statistical Learning with a Nuisance Component           | Framework: Statistical Learning with a Nuisance Component           |  10 |
| 3                | Orthogonal Statistical Learning                                     | Orthogonal Statistical Learning                                     |  11 |
|                  | 3.1                                                                 | Fast Rates Under Strong Convexity . . . . . . .                     |  13 |
|                  | 3.2                                                                 | Beyond Strong Convexity: Slow Rates . . . . .                       |  16 |
|                  | 3.3                                                                 | Example: Treatment Effect Estimation . . . . .                      |  17 |
|                  |                                                                     | 3.3.1 Residualized Loss (R-Loss) . . . . . . .                      |  17 |
|                  |                                                                     | 3.3.2 Doubly-Robust Loss (DR-Loss) . . . . .                        |  20 |
|                  | 3.4                                                                 | Example: Policy Learning . . . . . . . . . . . .                    |  21 |
|                  | 3.5                                                                 | Discussion . . . . . . . . . . . . . . . . . . . . .                |  22 |
| 4                | Instantiating the Main Results: Plug-In Empirical Risk Minimization | Instantiating the Main Results: Plug-In Empirical Risk Minimization |  23 |
|                  | 4.1                                                                 | Fast Rates for Plug-In Empirical Risk Minimization                  |  25 |
|                  | 4.2                                                                 | Slow Rates and Variance Penalization . . . . .                      |  26 |

Vasilis Syrgkanis Stanford University vsyrgk@stanford.edu

| 6 Discussion                                                                                        | 6 Discussion                                                                                        | 6 Discussion                                                                                        | 30    |
|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-------|
| I Experiments                                                                                       | I Experiments                                                                                       | I Experiments                                                                                       | 39    |
| A Experiments                                                                                       | A Experiments                                                                                       | A Experiments                                                                                       | 39    |
|                                                                                                     | A.1                                                                                                 | Conditional Average Treatment Effect Estimation                                                     | 40    |
|                                                                                                     | A.2                                                                                                 | Policy Learning . . . . . . . . . . . . . . . . . . .                                               | 44    |
| II Additional Results                                                                               | II Additional Results                                                                               | II Additional Results                                                                               | 48    |
| B Additional Algorithms                                                                             | B Additional Algorithms                                                                             | B Additional Algorithms                                                                             | 48    |
| C Orthogonal Statistical Learning: User-Friendly Tools                                              | C Orthogonal Statistical Learning: User-Friendly Tools                                              | C Orthogonal Statistical Learning: User-Friendly Tools                                              | 49    |
| D Construction of Orthogonal Losses                                                                 | D Construction of Orthogonal Losses                                                                 | D Construction of Orthogonal Losses                                                                 | 52    |
|                                                                                                     | D.1 Orthogonal Loss Construction: Examples                                                          | . . . .                                                                                             | 54    |
| E Sufficient Conditions for Theorems and : Single Index Losses                                      | E Sufficient Conditions for Theorems and : Single Index Losses                                      | E Sufficient Conditions for Theorems and : Single Index Losses                                      | 55    |
|                                                                                                     | E.1                                                                                                 | Fast Rates . . . . . . . . . . . . . . . . . . . . . .                                              | 55    |
|                                                                                                     | E.2                                                                                                 | Slow Rates . . . . . . . . . . . . . . . . . . . . .                                                | 58    |
|                                                                                                     | E.3                                                                                                 | Proofs . . . . . . . . . . . . . . . . . . . . . . . .                                              | 58    |
| F Sufficient Conditions for Oracle Rates: Further Results                                           | F Sufficient Conditions for Oracle Rates: Further Results                                           | F Sufficient Conditions for Oracle Rates: Further Results                                           | 61    |
| F.1 Oracle Rates for Square Losses with Misspecification                                            | F.1 Oracle Rates for Square Losses with Misspecification                                            | F.1 Oracle Rates for Square Losses with Misspecification                                            | 61    |
| F.2 Oracle Rates for Generic Lipschitz Losses                                                       | F.2 Oracle Rates for Generic Lipschitz Losses                                                       | . . . .                                                                                             | 62    |
| G Additional Applications                                                                           | G Additional Applications                                                                           | G Additional Applications                                                                           | 63    |
|                                                                                                     | G.1                                                                                                 | Policy Learning: Further Examples . . . . . . . .                                                   | 64    |
| Correction                                                                                          | Correction                                                                                          | Correction                                                                                          | 66    |
| G.2 Domain Adaptation and Sample Bias G.3 Missing Data . . . . . . . . . . . . . . . . .            | G.2 Domain Adaptation and Sample Bias G.3 Missing Data . . . . . . . . . . . . . . . . .            | G.2 Domain Adaptation and Sample Bias G.3 Missing Data . . . . . . . . . . . . . . . . .            |       |
|                                                                                                     |                                                                                                     | . . .                                                                                               | 67    |
|                                                                                                     | Plug-in                                                                                             | Empirical Risk Minimization: Further Results                                                        | 69    |
| H.1 Plug-in Empirical Risk Minimization:                                                            | H.1 Plug-in Empirical Risk Minimization:                                                            | Examples                                                                                            | 69    |
| H.1.1 Proofs . .                                                                                    | H.1.1 Proofs . .                                                                                    | . . . . . . . . . . . . . . . . . .                                                                 | 71    |
|                                                                                                     |                                                                                                     | H.2.1 Proof of Theorem 8 . . . . . . . . . . . .                                                    | 75    |
|                                                                                                     | Proofs                                                                                              | for Main Results                                                                                    | 80    |
| J Proofs from Section 3                                                                             | J Proofs from Section 3                                                                             | J Proofs from Section 3                                                                             |       |
| I Preliminaries                                                                                     | I Preliminaries                                                                                     | I Preliminaries                                                                                     | 81    |
|                                                                                                     |                                                                                                     |                                                                                                     | 81    |
| J.1 Omitted Proofs for Main Results J.2 Proofs for Examples . . . . . . .                           | J.1 Omitted Proofs for Main Results J.2 Proofs for Examples . . . . . . .                           | . . . . . . . . . . . . . . . . . .                                                                 | 81 82 |
| K Technical Lemmas for Constrained M -Estimators K.1 Proofs of Lemmas for Constrained M -Estimators | K Technical Lemmas for Constrained M -Estimators K.1 Proofs of Lemmas for Constrained M -Estimators | K Technical Lemmas for Constrained M -Estimators K.1 Proofs of Lemmas for Constrained M -Estimators | 86    |

| L                                    | Proofs from Section 4                | Proofs from Section 4                |   91 |
|--------------------------------------|--------------------------------------|--------------------------------------|------|
|                                      | L.1                                  | Proof of Theorem 3 . . . . . .       |   91 |
|                                      | L.2                                  | Proof of Theorem 4 . . . . . .       |   92 |
| M Proofs from Section 5 and Appendix | M Proofs from Section 5 and Appendix | M Proofs from Section 5 and Appendix |   94 |
|                                      | M.1                                  | Notation . . . . . . . . . . . .     |   94 |
|                                      | M.2                                  | Preliminaries . . . . . . . . .      |   94 |
|                                      | M.3                                  | Overview of Proofs . . . . . .       |   95 |
|                                      | M.4                                  | Skeleton Aggregation . . . . .       |   97 |
|                                      | M.5                                  | Rates for Specific Algorithms        |   98 |
|                                      | M.6                                  | Proofs for Oracle Rates . . .        |  102 |

## 1 Introduction

Predictive models based on modern machine learning methods are becoming increasingly widespread in policy making, with applications in healthcare, education, law enforcement, and business decision making. Most problems that arise in policy making, such as attempting to predict counterfactual outcomes for different interventions or optimizing policies over such interventions, are not pure prediction problems, but rather are causal in nature. It is important to address the causal aspect of these problems and build models that have a causal interpretation.

A common paradigm in the search of causality is that to estimate a model with a causal interpretation from observational data-that is, data not collected via randomized trial or via a known treatment policy-one typically needs to estimate many other quantities that are not of primary interest, but that can be used to de-bias a purely predictive machine learning model by formulating an appropriate loss. One example of such a nuisance parameter is the propensity for taking an action under the current policy, which can be used to form unbiased estimates for the reward for new policies, but is typically unknown in datasets that do not come from controlled experiments.

To make matters more concrete, let us walk through an example for which certain variants have been well-studied in machine learning (Dud´ ık et al., 2011; Swaminathan and Joachims, 2015a; Nie and Wager, 2021; Kallus and Zhou, 2018). Suppose a decision maker wants to estimate the causal effect of some treatment T ∈ { 0 , 1 } on an outcome Y as a function of a set of observable features X ; the causal effect will be denoted as θ ( X ). Typically, the decision maker has access to data consisting of tuples ( X i , T i , Y i ), where X i is the observed feature for sample i , T i is the treatment taken, and Y i is the observed outcome. Due to the partially observed nature of the problem, one needs to create unbiased estimates of the unobserved outcome. A standard approach is to make an unconfoundedness assumption (Rosenbaum and Rubin, 1983) and use the so-called doubly-robust formula, which is a combination of direct regression and inverse propensity scoring. Let Y i ( t ) denote the potential outcome for treatment t in sample i , and let f 0 ( t, x i ) := E [ Y i ( t ) | x i ] and p 0 ( t, x i ) := E [1 { T = t } | x i ]. If ( Y i (0) , Y i (1)) ⊥ T i | X i , then the following is an unbiased estimator for the conditional mean potential outcome (given covariates):

<!-- formula-not-decoded -->

Given such an estimator, we can estimate the treatment effect by running a regression between the unbiased estimates and the features, i.e. solve min θ ∈ Θ ∑ i ( ̂ Y (1) -̂ Y (0) -θ ( X i )) 2 over a target parameter class Θ. In the population limit, with infinite samples, this corresponds to finding a parameter θ ( x ) that minimizes the population risk E [ ( ̂ Y i (1) -̂ Y i (0) -θ ( X )) 2 ] . Similarly, if the decision maker is interested in policy optimization rather than estimating treatment effects, they can use these unbiased estimates to solve min θ ∈ Θ ∑ i ( ̂ Y i (0) -̂ Y i (1)) · θ ( X i ) over a policy space Θ of functions mapping features to { 0 , 1 } . However, when dealing with observational data, the functions f 0 and p 0 are not known, and must be estimated if we wish to evaluate the proxy labels ̂ Y ( t ). Since these functions are only used as a means to learn the target parameter θ , we may regard them as nuisance parameters. The goal of the learner is to estimate a target parameter that achieves low population risk when evaluated at the true nuisance parameters as opposed to the estimated nuisance parameters, since only then does the model have a causal interpretation.

This phenomenon is ubiquitous in causal inference and motivates us to formulate the abstract problem of statistical learning with a nuisance component : Given n i.i.d. examples from a distribution D , a learner is interested in finding a target parameter ̂ θ ∈ Θ so as to minimize a population risk

function L D : Θ × G → R . The population risk depends not just on the target parameter, but also on a nuisance parameter whose true value g 0 ∈ G is unknown to the learner. The goal of the learner is to produce an estimate that has small excess risk evaluated at the unknown true nuisance parameter:

<!-- formula-not-decoded -->

Depending on the application, such an excess risk bound can take different interpretations. For many settings, such as treatment effect estimation, it is closely related to mean squared error, while in policy optimization it typically corresponds to regret. Following the tradition of statistical learning theory (Vapnik, 1995; Bousquet et al., 2004), we make excess risk the primary focus of our work, independent of the interpretation. We develop algorithms and analysis tools that generically address (2), then apply these tools to a number of applications of interest.

The problem of statistical learning with a nuisance component is strongly connected to the wellstudied semiparametric inference problem (Levit, 1976; Ibragimov and Has'Minskii, 1981; Pfanzagl, 1982; Bickel, 1982; Klaassen, 1987; Robinson, 1988; Bickel et al., 1993; Newey, 1994; Robins and Rotnitzky, 1995; Ai and Chen, 2003; van der Laan and Dudoit, 2003; van der Laan and Robins, 2003; Ai and Chen, 2007; Tsiatis, 2007; Kosorok, 2008; van der Laan and Rose, 2011; Ai and Chen, 2012; Chernozhukov et al., 2022a; Belloni et al., 2017; Chernozhukov et al., 2018a), which focuses on providing so-called ' √ n -consistent and asymptotically normal' estimates for a low-dimensional target parameter θ 0 (which may be expressed as a population risk minimizer or a solution to estimating equations) in the presence of a typically nonparametric nuisance parameter. Unlike the semiparametric inference problem, statistical learning with a nuisance component does not require a well-specified model, nor a unique minimizer of the population risk. Moreover, we do not ask for parameter recovery or asymptotic inference (e.g., asymptotically valid confidence intervals). Rather, we are content with an excess risk bound, regardless of whether there is an underlying true parameter to be identified. As a consequence, we provide guarantees even in the presence of misspecification, and when the target parameter belongs to a large, potentially nonparametric class. For example, one line of previous work gives semiparametric inference guarantees when the nuisance parameter is a neural network (Chen and White, 1999; Farrell et al., 2021); by focusing on excess risk we can give guarantees for the case where the target parameter is a neural network.

The case where the target parameter belongs to an arbitrary class has not been addressed at the level of generality we consider in the present work, but we mention some prior work that goes beyond the low-dimensional/parametric setup for special cases. Athey and Wager (2017) and Zhou et al. (2023) give guarantees based on metric entropy of the target class for the specific problem of treatment policy learning. For estimation of treatment effects, various nonparametric classes have been used for the target class on a case by case basis, including kernels (Nie and Wager, 2021), random forests (Athey et al., 2019; Oprescu et al., 2019; Friedberg et al., 2020), and high-dimensional linear models (Chernozhukov et al., 2017, 2018b). Other results allow for fairly general choices for the target parameter class in specific statistical models (Rubin and van der Laan, 2005, 2007; D´ ıaz and van der Laan, 2013; van der Laan and Luedtke, 2014; Kennedy et al., 2017, 2019; K¨ unzel et al., 2019). Our work unifies these directions into a single framework, and our general tools lead to improved or refined results when specialized to many of these individual settings.

Our approach is to reduce the problem of statistical learning with a nuisance component to the standard formulation of statistical learning. We build on a recent thread of research on semiparametric inference known as 'double' or 'debiased' machine learning (Chernozhukov et al., 2022a, 2017, 2018a,c,b), which leverages sample splitting to provide inference guarantees under

Meta-Algorithm 1 (Two-Stage Estimation with Sample Splitting) .

Input : Sample set S = z 1 , . . . , z n .

- Split S into subsets S 1 = z 1 , . . . , z ⌊ n/ 2 ⌋ and S 2 = S \ S 1 .
- Return ̂ θ , the output of Alg(Θ , S 2 ; ̂ g ) .
- Let ̂ g be the output of Alg( G , S 1 ) .

weak assumptions on the estimator for the nuisance parameter. Rather than directly analyzing particular algorithms and models for the target parameter (e.g., regularized regression, gradient boosting, or neural network estimation), we assume a black-box guarantee for the excess risk in the case where a nuisance value g ∈ G is fixed. Our main theorem asks only for the existence of an algorithm Alg(Θ , S ; g ) that, for any given nuisance parameter g and data set S , achieves low excess risk with respect to the population risk L D ( θ, g ), i.e. with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Likewise, we assume the existence of a black-box algorithm Alg( G , S ) to estimate the nuisance component g 0 from the data, with the required estimation guarantee varying from problem to problem.

Given access to the two black-box algorithms, we analyze a simple sample splitting meta-algorithm for statistical learning with a nuisance component, presented as Meta-Algorithm 1. We can now state the main question addressed in this paper: When is the excess risk achieved by sample splitting robust to nuisance component estimation error?

In more technical terms, we seek to understand when the two-stage sample splitting meta-algorithm achieves an excess risk bound with respect to g 0 , in spite of error in the estimator ̂ g output by the first-stage algorithm. Robustness to nuisance estimation error allows the learner to use more complex models for nuisance estimation and-under certain conditions on the complexity of the target and nuisance parameter classes-to learn target parameters whose error is, up to lower order terms, as good as if the learner had known the true nuisance parameter in advance. Such a guarantee is referred to as achieving an oracle rate in semiparametric inference.

Overview of results. We use Neyman orthogonality (Neyman, 1959, 1979), a key tool in inference in semiparametric models (Newey, 1994; van der Vaart, 2000; Robins et al., 2008; Zheng and van der Laan, 2010; Belloni et al., 2017; Chernozhukov et al., 2018a), to provide oracle rates for statistical learning with a nuisance component. We show that if the population risk satisfies a functional analogue of Neyman orthogonality, the estimation error of ̂ g has a second order impact on the overall excess risk (relative to g 0 ) achieved by ̂ θ . To gain some intuition, Neyman orthogonality is weaker condition than double-robustness, albeit similar in flavor, (see, e.g., Chernozhukov et al. (2022a)) and is satisfied by both the treatment effect loss and the policy learning loss described in the introduction. In more detail, our variant of the Neyman orthogonality condition asserts that a functional cross-derivative of the loss vanishes when evaluated at the optimal target and nuisance parameters. Prior work provides a number of means through which to construct Neyman orthogonal losses whenever certain moment conditions are satisfied by the data generating process (Chernozhukov et al., 2018a, 2022a, 2018b). Indeed, orthogonal losses can be constructed in settings including treatment effect estimation, policy learning, missing and censored data problems, estimation of structural econometric models, and game-theoretic models.

We identify two regimes of excess risk behavior:

1. Fast rates. When the population risk is strongly convex with respect to the prediction of the target parameter (e.g., the treatment effect estimation loss), then typically so-called fast rates (e.g., rates of order of O (1 /n ) for parametric classes) are optimal if the true nuisance parameter is known. Letting R G denote the estimation error of the nuisance component, in this setting we show that orthogonality implies that the first stage error has an impact on the excess risk of the order of R 4 G (in particular, n -1 / 4 -RMSE rates for the nuisance suffice when the target is parametric).
2. Slow rates. Absent any strong convexity of the population risk (e.g., for the treatment policy optimization loss), typically slow rates (e.g. rates of order O (1 / √ n ) for parametric classes) are optimal if the true nuisance parameter is known. For this setting, we show that the impact of nuisance estimation error is of the order R 2 G so, once again, n -1 / 4 RMSE rates for the nuisance suffice when the target is parametric.

To make the conditions above concrete for arbitrary classes, we give conditions on the relative complexity of the target and nuisance classes-quantified via metric entropy -under which the sample splitting meta-algorithm achieves oracle rates, assuming the two black-box estimation algorithms are instantiated appropriately. This allows us to extend several prior works beyond the parametric regime to complex nonparametric target classes. Our technical results extends the works of Yang and Barron (1999); Rakhlin et al. (2017), which provide minimax optimal rates without nuisance components and utilize the technique of aggregation in designing optimal algorithms.

The flexibility of our approach allows us to instantiate our framework with any machine learning model and algorithm of interest for both nuisance and target parameter estimation, and to utilize the vast literature on generalization bounds in machine learning to establish refined (e.g., data-dependent or dimension-independent) rates for several classes of interests. For instance, our approach allows us to leverage recent work on size-independent generalization error of neural networks.

Moving beyond black-box results, we use our main theorems as a starting point to provide sharp analyses for certain general-purpose statistical learning algorithms for target estimation in the presence of nuisance parameters. First, we provide a new analysis for empirical risk minimization with plug-in estimation of nuisance parameters, wherein we extend the classical local Rademacher complexity analysis of empirical risk minimization (Koltchinskii and Panchenko, 2000; Bartlett et al., 2005) to account for the impact of the nuisance error (leveraging orthogonality). Second, in the slow rate regime we give a new analysis of variance-penalized empirical risk minimization with plug-in nuisance estimation, which allows us to recover and extend several prior results in the literature on policy learning. Our result improves upon the variance-penalized risk minimization approach of Maurer and Pontil (2009) by replacing the dependence on the metric entropy at a fixed approximation level with the critical radius , which is related to the entropy integral.

As a consequence of focusing on excess risk, we obtain oracle rates under weaker assumptions on the data generating process than in previous works. Notably, we obtain guarantees even when the target parameter is misspecified and the target parameters are not identifiable. For instance, for sparse high-dimensional linear classes, we obtain optimal prediction rates with no restricted eigenvalue assumptions. We highlight the applicability of our results to four settings of primary importance in the literature: 1) estimation of heterogeneous treatment effects from observational data, 2) offline policy optimization, 3) domain adaptation, 4) learning with missing data. For each of these applications, our general theorems allow for the use of arbitrary estimators for the nuisance and target parameter classes and provide robustness to the nuisance estimation error.

## 1.1 Related work

General frameworks for learning/inference with nuisance parameters. The work of van der Laan and Dudoit (2003) and subsequent refinements and extensions (van der Laan et al., 2006, 2007) develops cross-validation methodology for a similar risk minimization setting in which the target risk parameter depends on an unknown nuisance parameter. van der Laan and Dudoit (2003) analyze a cross-validation meta-algorithm in which the learner simultaneously forms a nuisance parameter estimator and a set of candidate target parameter estimators using a set of training samples, then selects a final estimate for the target parameter by minimizing an empirical loss over a validation set. The train and validation splits may be chosen in a general fashion that encompasses K -fold and Monte Carlo validation. They provide finite-sample oracle rates for the excess risk in the case where the target parameter belongs to a finite class (in particular, rates of the type log | Θ | /n for a class of square losses and √ log | Θ | /n for general losses), and also extend these guarantees to linear combinations of basis functions via pointwise ε -nets (in our language, such classes are parametric ). Overall, our approach offers several new benefits:

- By completely splitting nuisance estimation and target estimation into separate stages and taking advantage of orthogonality, we can provide meta-theorems on robustness that are invariant to the choice of learning algorithm both for the first and second stage, which obviates the need to assume the target class is finite or admits a linear representation (Section 3).
- When we do specialize to algorithms such as ERM and variants, we can provide finite-sample guarantees for rich classes of target parameters in terms of sharp learning-theoretic complexity measures such as local Rademacher complexity and empirical metric entropy (Section 4). In particular, we can provide conditions under which oracle rates are attained under very general complexity assumptions on the target and nuisance parameters (Section 5).

The methodology of van der Laan and Dudoit (2003) can be used to directly estimate a target parameter or to select the best of many candidate nuisance estimators in a data-driven fashion. van der Laan et al. (2007) refers to the use of this cross-validation methodology to perform dataadaptive estimation of nuisance parameters as the 'super learner', and subsequent work has advocated for its use for nuisance estimation within a framework for semiparametric inference known as targeted maximum likelihood estimation (TMLE). TMLE (Scharfstein et al., 1999; van der Laan and Rubin, 2006; Zheng and van der Laan, 2010; van der Laan and Rose, 2011) and its more general variant, targeted minimum loss-based estimation, are general frameworks for semiparametric inference which-like our framework-employ empirical risk minimization in the presence of nuisance parameters. TMLE estimates the target parameter by repeatedly minimizing an empirical risk (typically the negative log-likelihood) in order to refine an initial estimate. This approach easily incorporates constraints, and can be used in tandem with the super learning technique. The analysis leverages orthogonality, and is also agnostic to how the nuisance estimates are obtained. However, the main focus of this framework is on the classical semiparametric inference objective; minimizing a population risk is not the end goal as it is here.

Specific instances of risk minimization with nuisance parameters. A number of prior works employ empirical risk minimization with nuisance parameters for specific statistical models (Rubin and van der Laan, 2005, 2007; D´ ıaz and van der Laan, 2013; van der Laan and Luedtke, 2014; Kennedy et al., 2017, 2019; K¨ unzel et al., 2019). These results allow for general choices for the target class and nuisance class (typically subject to Donsker conditions, or with guarantees in the vein of van der Laan and Dudoit (2003)), and the main focus is semiparametric inference rather

than excess risk guarantees.

Nonparametric target parameters. Outside of the risk minimization-based approaches above and the examples in the prequel (Athey et al., 2019; Nie and Wager, 2021; Athey and Wager, 2017; Zhou et al., 2023; Oprescu et al., 2019; Friedberg et al., 2020; Chernozhukov et al., 2017, 2018b), a number of other results also consider inference for nonparametric target parameters in the presence of nuisance parameters. In van der Vaart and van der Laan (2006), the target is a Lipschitz function over [0 , ∞ ) (the marginal survival function) and an estimation rate of n -2 / 3 is given. Wang et al. (2010) consider estimation of smooth nonparametric target parameters in the presence of missing outcomes, and give algorithms based on kernel smoothing. Robins and Rotnitzky (2001); Robins et al. (2008) consider settings where the target parameter is scalar, but the optimal rate is nonparametric due to the presence of complex nuisance parameters.

Sample splitting. While our use of sample splitting is directly inspired by recent use of the technique in double/debiased machine learning (Chernozhukov et al., 2022a, 2018a), the basic technique dates back to the early days of semiparametric inference and it has found use in many other works to remove Donsker conditions for estimation in the presence of nuisance parameters (Bickel, 1982; Klaassen, 1987; van der Vaart, 2000; Robins et al., 2008; Zheng and van der Laan, 2010).

Limitations. Our results are quite general, but there are some applications that go beyond the scope of our framework. For example, while we consider only plug-in estimation for the nuisance parameters, several works attain refined results by using specialized estimators van der Laan and Rubin (2006); Hirshberg and Wager (2021); Chernozhukov et al. (2018c); Ning et al. (2020). While our focus is on methods based on loss minimization, some problems such as nonparametric instrumental variables (Newey and Powell, 2003; Hall et al., 2005; Blundell et al., 2007; Chen and Pouzo, 2009, 2012, 2015; Chen and Christensen, 2018) are more naturally posed in terms of conditional moment restrictions. 1

Another direction where our results leave room for future improvement concerns the reliance on Neyman orthogonality. While Neyman orthogonality is a fairly general condition which allows one to handle many nuisance parameters simultaneously, many problems admit additional structure which can lead to more refined guarantees. For example, in the context of treatment effect estimation, subsequent work of Kennedy (2020) uses the doubly robust structure of the problem to give guarantees that accommodate the case where different nuisance components (regression functions and propensity scores) are estimated at different rates.

## 1.2 Organization

The first part of this paper presents our main results. Section 2 contains technical preliminaries and definitions, and Section 3 presents our main theorems concerning the excess risk of MetaAlgorithm 1. Section 3 also includes basic examples in which we apply these theorems to treatment effect estimation and policy learning.

Our main results are stated at a high level of generality, and consider generic estimation algorithms for the target and nuisance parameters. In the second part of the paper, we make matters more

1 In fact, nonparametric IV can be cast as a special case of the setup in (4), but we do not know of any estimators for this problem that satisfy the conditions required to apply our main theorems.

concrete and focus on specific algorithms. We leverage the main theorems to give explicit bounds based on the statistical capacity of the target and nuisance class. In particular:

- Section 4 ( Plug-in Empirical Risk Minimization ) provides explicit bounds for plug-in empirical risk minimization as the second stage of the meta-algorithm.
- Section 5 ( Sufficient Conditions for Oracle Rates ) considers aggregation based algorithms that go beyond empirical risk minimization, and gives sufficient conditions (as a function of the statistical capacity of the nuisance and target class) under which Meta-Algorithm 1 can be configured such that oracle excess risk bounds are achieved.

We conclude with discussion in Section 6. Additional results are deferred to the appendix, which is split into three parts. Part I contains experiments, and Part II contains supplementary theoretical results, including sufficient conditions for Neyman orthogonality, applications of our main results to specific settings, and further guarantees for specific algorithms and function classes. Part III contains proofs for our main results.

## 2 Framework: Statistical Learning with a Nuisance Component

We work in a learning setting in which observations belong to an abstract set Z . We receive a sample set S := z 1 , . . . , z n where each z t is drawn i.i.d. from an unknown distribution D over Z . Define variable subsets X ⊆ W ⊂ Z ; the restriction X ⊆ W is not strictly necessary but simplifies notation. We focus on learning parameters that come from a target parameter class Θ : X → V 2 and nuisance parameter class G : W → V 1 , where V 1 and V 2 are finite dimensional vector spaces of dimension K 1 and K 2 respectively, equipped with norms ∥·∥ V 1 and ∥·∥ V 2 . Note that since our results are fully non-asymptotic, the classes Θ and G may be taken to grow with n .

Given an example z t ∈ Z , we write w t ∈ W and x t ∈ X to denote the subsets of z t that act as arguments to the nuisance and target parameters respectively. For example, we may write g ( w t ) for g ∈ G or θ ( x t ) for θ ∈ Θ. We assume that the function spaces Θ and G are equipped with pre-norms ∥·∥ Θ and ∥·∥ G respectively, which need to satisfy non-negativity and ∥ 0 ∥ = 0, but not necessarily the triangle inequality nor absolute homogeneity. In our applications, both pre-norms take the form ∥ f ∥ L p ( V , D ) = ( E z ∼D ∥ f ( z ) ∥ p V ) 1 /p for functions f : Z → V , where V ∈ {V 1 , V 2 } .

We measure performance of the target predictor through the real-valued population loss functional L D ( θ, g ), which maps a target predictor θ and nuisance predictor g to a loss. The subscript D in L D denotes that the functional depends on the underlying distribution D . For all of our applications, L D has the following structure, in line the classical statistical learning setting: First define a pointwise loss function ℓ ( θ, g ; z ), then define L D ( θ, g ) := E z ∼D [ ℓ ( θ, g ; z )]. Our general framework does not explicitly assume this structure, however.

Let g 0 ∈ G be the unknown true value for the nuisance parameter. Given the samples S , and without knowledge of g 0 , we aim to produce a target predictor ̂ θ that minimizes the excess risk evaluated at g 0

<!-- formula-not-decoded -->

As discussed in the introduction, we will always produce such a predictor via the sample splitting meta-algorithm (Meta-Algorithm 1), which makes uses of a nuisance predictor ̂ g .

When the infimum in the excess risk is obtained, we use θ ⋆ to denote the corresponding minimizer,

in which case the excess risk can be written as

<!-- formula-not-decoded -->

We occasionally use the notation θ 0 to refer to a particular target parameter with respect to which the second stage satisfies a first-order condition , e.g. D θ L D ( θ 0 , g 0 )[ θ -θ 0 ] = 0 ∀ θ ∈ Θ. If θ 0 ∈ Θ and the population risk is convex, then we can take θ ⋆ = θ 0 without loss of generality, but we do not assume this, and in general we do not assume existence of a such a parameter θ 0 .

Notation. We let ⟨· , ·⟩ denote the standard inner product. ∥·∥ p will denote the ℓ p norm over R d and ∥·∥ σ will denote the spectral norm over R d 1 × d 2 .

Unless otherwise stated, the expectation E [ · ], probability P ( · ), and variance Var( · ) operators will be taken with respect to the underlying distribution D . We define empirical analogues E n [ · ], P n ( · ), and Var n ( · ) with respect to a sample set z 1 , . . . , z n , whose value will be clear from context. For a vector space V with norm ∥·∥ V and function f : Z → V , we define ∥ f ∥ L p ( V , D ) = ( E z ∼D ∥ f ( z ) ∥ p V ) 1 /p for p ∈ (0 , ∞ ), with L p ( ℓ q , D ) referring to the special case where ∥·∥ V = ∥·∥ q . For a sample set S = z 1: n , we define the empirical variant ∥ f ∥ L p ( V ,S ) = ( 1 n ∑ n i =1 ∥ f ( z i ) ∥ p V ) 1 /p . When V = R , we drop the first argument and write L p ( D ) and L p ( S ). We extend these definitions to p = ∞ in the natural way.

For a subset X of a vector space, conv( X ) will denote the convex hull. For an element x ∈ X , we define the star hull via

<!-- formula-not-decoded -->

and adopt the shorthand star( X ) := star( X , 0).

Given functions f, g : X → [0 , ∞ ) where X is any set, we use non-asymptotic bigO notation, writing f = O ( g ) if there exists a numerical constant c &lt; ∞ such that f ( x ) ≤ c · g ( x ) for all x ∈ X and f = Ω( g ) if there is a numerical constant c &gt; 0 such that f ( x ) ≥ c · g ( x ). We write f = ˜ O ( g ) as shorthand for f = O ( g max { 1 , polylog( g ) } ).

## 3 Orthogonal Statistical Learning

In this section we present our main results on orthogonal statistical learning, which state that under certain conditions on the loss function, the error due to estimation of the nuisance component g 0 has higher-order impact on the prediction error of the target component. The results in this section, which form the basis for all subsequent results, are algorithm-independent , and only involve assumptions on properties of the population risk L D . To emphasize the high level of generality, the results in this section invoke the learning algorithms in Meta-Algorithm 1 only through 'rate' functions Rate D ( G , . . . ) and Rate D (Θ , . . . ) which respectively bound the estimation error of the first stage and the excess risk of the second stage.

Definition 1 (Algorithms and Rates) . The first and second stage algorithms and corresponding rate functions are defined as follows:

- a) Nuisance algorithm and rate. The first stage learning algorithm Alg( G , S ) , when given a sample set S from distribution D , outputs a predictor ̂ g for which

<!-- formula-not-decoded -->

with probability at least 1 -δ .

- b) Target algorithm and rate. The second stage learning algorithm Alg(Θ , S ; g ) , when given sample set S from distribution D and any g ∈ G outputs a predictor ̂ θ for which

<!-- formula-not-decoded -->

with probability at least 1 -δ .

We let ̂ Θ denote any function class (fixed a-priori) for which ̂ θ, θ ⋆ ∈ ̂ Θ almost surely. We denote worstcase variants of the rates by Rate D ( G , n, δ ) := sup S : | S | = n Rate D ( G , S, δ ) and Rate D (Θ , n, δ ; ̂ θ, g ) := sup S : | S | = n Rate D (Θ , S, δ ; ̂ θ, g ) .

Observe that if one naively applies the algorithm for the target class using the nuisance predictor ̂ g as a plug-in estimate for g 0 , the rate stated in Definition 1 will only yield a 'pseudo'-excess risk bound of the form

<!-- formula-not-decoded -->

This clearly does not match the desired bound (4), which concerns the excess risk evaluated at g 0 rather than the plug-in estimate ̂ g . The bulk of our work is to show that orthogonality can be used to correct this mismatch.

̸

Definition 1 and subsequent results are stated in terms of a class ̂ Θcontaining ̂ θ , which in general may have ̂ Θ = Θ. This extra level of generality serves two purposes. First, it allows for refined analysis in the case where ̂ Θ ⊂ Θ, which is encountered when using algorithms based on regularization that do not impose hard constraints on, e.g., the norm of the estimator. Second, it permits the use of improper prediction , i.e. ̂ Θ ⊃ Θ, which in some cases is required to obtain optimal rates for misspecified models (Audibert, 2008; Foster et al., 2018).

Recall that for a sample set S = z 1 , . . . , z n , the empirical loss is defined via L S ( θ, g ) = 1 n ∑ n t =1 ℓ ( θ, g ; z t ). Many classical results from statistical learning can be applied to the double machine learning setting by minimizing the empirical loss with plug-in estimates for g 0 , and we can simply cite these results to provide examples of Rate D for the target class Θ. Note however that this structure is not assumed by Definition 1, and we indeed consider algorithms that do not have this form (cf. Section 5). Let us highlight that we allow the function Rate D (Θ , S, δ ; ̂ θ, ̂ g ) to depend on both the target estimator ̂ θ and the nuisance estimator ̂ g ; this extra level of generality is useful for deriving algorithm-specific guarantees (cf. Section 4).

Fast rates and slow rates. The rates presented in this section fall into two distinct categories, which we distinguish by referring to them as either fast rates or slow rates . The meaning of the word 'fast' or 'slow' here is two-fold: First, for fast rates, our assumptions on the loss imply that when the target class Θ is not too large (e.g. a parametric or VC-subgraph class) prediction error rates of order O (1 /n ) are possible in the absence of nuisance parameters. For our slow rate results, the best prediction error rate that can be achieved is O (1 / √ n ), even for small classes. This distinction is consistent with the usage of the term fast rate in statistical learning (Bousquet et al., 2004; Bartlett et al., 2005; Srebro et al., 2010), and we will see concrete examples of such rates for specific classes in later sections (Section 4, Section 5).

The second meaning of 'fast' versus 'slow' refers to the first stage: When estimation error for the nuisance is of order ε , the impact on the second stage in our fast rate results is of order ε 4 , while for our slow rate results the impact is of order ε 2 . The fast rate regime-particularly, the ε 4 -type dependence on the nuisance error-will be the more familiar of the two for readers accustomed to semiparametric inference. While fast rates might at first seem to strictly improve over slow rates,

these results require stronger assumptions on the loss. Our results in Section 5 show that which setting is more favorable will in general depend on the precise relationship between the complexity of the target parameter class and the nuisance parameter class.

## 3.1 Fast Rates Under Strong Convexity

We first present general conditions under which the sample splitting meta-algorithm obtains so-called fast rates for prediction. Our assumptions are stated in terms of directional derivatives with respect to the target and nuisance parameters.

Definition 2 (Directional Derivative) . Let F be a vector space of functions. For a functional F : F → R , we define the derivative operator D f F ( f )[ h ] = d dt F ( f + th ) ∣ ∣ ∣ t =0 for a pair of functions f, h ∈ F . Likewise, we define D k f F ( f )[ h 1 , . . . , h k ] = ∂ k ∂t 1 ...∂t k F ( f + t 1 h 1 + . . . + t k h k ) ∣ ∣ ∣ t 1 = ··· = t k =0 . When considering a functional in two arguments, e.g. L D ( θ, g ) , we write D θ L D ( θ, g ) and D g L D ( θ, g ) to make the argument with respect to which the derivative is taken explicit.

To present our results, we fix a representative θ ⋆ ∈ arg min θ ∈ Θ L D ( θ, g 0 ). In general, the minimizer may not be unique-indeed, by focusing on excess risk, we can provide guarantees even when parameter recovery is impossible. Thus, we assume that a single fixed representative θ ⋆ is used throughout all the assumptions stated in this subsection.

Our first assumption is the starting point for this work, and asserts that the population loss is orthogonal in the sense that the certain pathwise derivatives vanish.

Assumption 1 (Orthogonal Loss) . The population risk L D is Neyman orthogonal :

<!-- formula-not-decoded -->

Note that while Assumption 1 is stated in terms of the risk L D , it is typically satisfied by choosing a particular point-wise loss function whose expectation equals the risk; examples are given in the sequel. The construction of such a point-wise loss is typically achieved by adding a de-biasing correction term to some 'initial' loss, whose minimizer is the target quantity (see Appendix D for details on automated orthogonal loss construction). The de-biasing correction reduces the impact of errors in the nuisance function estimates on the gradient of the loss, and is related to the notion of an efficient influence function in semi-parametric inference (however, our estimand is not necessarily pathwise differentiable, and hence violates the basic premise of most semi-parametric inference theory).

Beyond orthogonality, our main theorem requires three additional assumptions, all of which are fairly standard in the context of fast rates for statistical learning. We require a first-order optimality condition for the target class, and require that the population risk is both smooth and strongly convex with respect to the target parameter.

Assumption 2 (First Order Optimality) . The minimizer for the population risk satisfies the first-order optimality condition:

D

θ

L

D

(

θ

⋆

, g

0

)[

θ

-

θ

⋆

]

≥

0

∀

θ

∈

star(

̂

Θ

, θ

⋆

)

.

(8)

Remark 1. The first-order condition is typically satisfied for models that are well-specified , meaning that there is some variable in z that identifies the target parameter θ 0 . More generally, it suffices to

'almost' satisfy the first-order condition, i.e. to replace (8) by the condition

<!-- formula-not-decoded -->

The first-order condition is also satisfied whenever ̂ Θ is star-shaped around θ ⋆ , i.e. star( ̂ Θ , θ ⋆ ) ⊆ ̂ Θ .

Assumption 3 (Higher-Order Smoothness) . There exist constants β 1 and β 2 such that the following derivative bounds hold:

- a) Second-order smoothness with respect to target. For all θ ∈ ̂ Θ and all ¯ θ ∈ star( ̂ Θ , θ ⋆ ) :

<!-- formula-not-decoded -->

- b) Higher-order smoothness. There exists r ∈ [0 , 1) such that for all θ ∈ star( ̂ Θ , θ ⋆ ) , g ∈ G , and ¯ g ∈ star( G , g 0 ) :

<!-- formula-not-decoded -->

Assumption 4 (Strong Convexity) . The population risk is strongly convex with respect to the target parameter: There exist constants λ, κ &gt; 0 such that for all θ ∈ ̂ Θ and g ∈ G ,

<!-- formula-not-decoded -->

where r ∈ [0 , 1) is as in Assumption 3.

̸

Assumption 3 and Assumption 4 are easily satisfied whenever the loss is obtained by applying a square loss or another smooth, strongly convex loss pointwise to the prediction of the target class; concrete examples are given in Appendix E. For most of our results, we apply these assumptions with r = 0, but the case r &gt; 0 will prove useful for certain settings in which strong L ∞ -type estimation guarantees for the target parameter are available (cf. Example 1). In general, Assumptions 1 to 4 do not imply that θ ⋆ is uniquely identified unless ∥·∥ Θ is a norm. However, if the assumptions are satisfied by two parameters θ ⋆ = ˜ θ ⋆ , we must have ∥ θ ⋆ -˜ θ ⋆ ∥ Θ = 0, meaning convergence in the sense that ∥ ̂ θ -θ ⋆ ∥ Θ → 0 is equivalent for both representatives.

We now state our main theorem concerning fast rates.

Theorem 1. Suppose there exists θ ⋆ ∈ arg min θ ∈ Θ L D ( θ, g 0 ) such that Assumptions 1 to 4, are satisfied. Then the sample splitting meta-algorithm (Meta-Algorithm 1) produces a parameter ̂ θ such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where C 1 ≤ 4 λ and C 2 ≤ 2 ( ( β 2 λ ) 2 1+ r + κ λ ) . In addition,

<!-- formula-not-decoded -->

The majority of the results in this paper concern the special case in which r = 0. In this case, since Rate D ( G , S 1 , δ/ 2) ∝ ∥ ̂ g -g 0 ∥ G , Theorem 1 shows that for Meta-Algorithm 1, the impact of the unknown nuisance parameter on the prediction is of second-order, i.e.

<!-- formula-not-decoded -->

This implies that if the optimal rate without nuisance parameters is of order O ( n -1 ), it suffices to take ∥ ̂ g -g 0 ∥ 2 G = o ( n -1 / 2 ) to achieve the oracle rate.

Proof of Theorem 1. We prove Theorem 1 by performing a Taylor expansion to relate the excess risk at ̂ g to the excess risk at g 0 , employing orthogonality and self-bounding arguments to control cross terms. We abbreviate R Θ := Rate D (Θ , S 2 , δ/ 2; ̂ θ, ̂ g ) and R G := Rate D ( G , S 1 , δ/ 2) to simplify notation.

By a second-order Taylor expansion on the risk at ̂ g , there exists ¯ θ ∈ star( ̂ Θ , θ ⋆ ) such that

<!-- formula-not-decoded -->

Next, using the strong convexity assumption (Assumption 4), we have

<!-- formula-not-decoded -->

Combining these statements, we conclude that

<!-- formula-not-decoded -->

Using the assumed rate for ̂ θ (Definition 1), this implies the inequality

<!-- formula-not-decoded -->

We now apply a second-order Taylor expansion (using the assumed derivative continuity from Assumption 3), which implies that there exists ¯ g ∈ star( G , g 0 ) such that

<!-- formula-not-decoded -->

Using orthogonality of the loss (Assumption 1), this is equal to

<!-- formula-not-decoded -->

We use the second order smoothness assumed in Assumption 3 to upper bound by

<!-- formula-not-decoded -->

Invoking Young's inequality and using that r ∈ [0 , 1), we have that for any constant η &gt; 0, this is at most

<!-- formula-not-decoded -->

Lastly, we use the assumed rate for ̂ g (Definition 1) to bound by

<!-- formula-not-decoded -->

Choosing η = λ β 2 , combining this string of inequalities with (15), and rearranging, we have:

<!-- formula-not-decoded -->

Assumption 2 implies that D θ L D ( θ ⋆ , g 0 )[ ̂ θ -θ ⋆ ] ≥ 0, which establishes the inequality (13).

To derive the inequality (14), we use another Taylor expansion, which implies that there exists ¯ θ ∈ star( ̂ Θ , θ ⋆ ) such that

<!-- formula-not-decoded -->

Using the smoothness bound from Assumption 3, we upper bound this by

<!-- formula-not-decoded -->

We combine (16) with (17) to conclude that L D ( ̂ θ, g 0 ) -L D ( θ ⋆ , g 0 ) is bounded by

<!-- formula-not-decoded -->

The result follows by again using that D θ L D ( θ ⋆ , g 0 )[ ̂ θ -θ ⋆ ] ≥ 0, along with the fact that β 1 /λ ≥ 1 without loss of generality.

There is one issue not addressed by Theorem 1: If the nuisance parameter g 0 were known, the rate for the target parameters would be Rate D (Θ , . . . ; ̂ θ, g 0 ), but the bound in (14) scales instead with Rate D (Θ , . . . ; ̂ θ, ̂ g ). This is addressed in Sections 4 and 5, where-building on Theorem 1-we show that for many standard algorithms, the cost to relate these quantities grows as (Rate D ( G , S 1 , δ/ 2)) 4 , and can be absorbed into the second term in (13) or (14).

## 3.2 Beyond Strong Convexity: Slow Rates

The strong convexity assumption used by Theorem 1 requires curvature only in the prediction space, not the parameter space. This is considerably weaker than what is assumed in prior works on double machine learning (e.g., Chernozhukov et al. (2018b)), and is a major advantage of analyzing prediction error rather than parameter recovery. Nonetheless, in some situations even assuming strong convexity on predictions may be unrealistic. A second advantage of studying prediction is that, while parameter recovery is not possible in this case, it is still possible to achieve low prediction error, albeit with slower rates than in the strongly convex case. We now give guarantees under which these (slower) oracle rates for prediction error can be obtained in the presence of nuisance parameters using Meta-Algorithm 1. As in the prequel, we fix a representative θ ⋆ ∈ arg min θ ∈ Θ L D ( θ, g 0 ) throughout this subsection.

The key technical assumption for next result is universal orthogonality , which informally states that the loss is not simply orthogonal around θ ⋆ , but rather is orthogonal for all θ ∈ Θ.

Assumption 5 (Universal Orthogonality) . For all ¯ θ ∈ star( ̂ Θ , θ ⋆ ) + star( ̂ Θ -θ ⋆ , 0) ,

<!-- formula-not-decoded -->

Universal orthogonality is a strengthening of Assumption 1, which requires that the cross derivative at g 0 vanishes for all ¯ θ ∈ star( ̂ Θ , θ ⋆ ), rather than only at θ ⋆ . It is satisfied for examples including treatment effect estimation (Section 3.3) and policy learning (Section 3.4), and is used implicitly in previous work in these settings (Nie and Wager, 2021; Athey and Wager, 2017). Beyond orthogonality, we require a mild smoothness assumption for the nuisance class.

Assumption 6. The derivatives D 2 g L D ( θ, g ) and D 2 θ D g L D ( θ, g ) are continuous. Furthermore, there exists a constant β such that for all θ ∈ star( ̂ Θ , θ ⋆ ) and ¯ g ∈ star( G , g 0 ) ,

<!-- formula-not-decoded -->

Our main theorem for slow rates is as follows.

Theorem 2. Suppose that there is θ ⋆ ∈ arg min θ ∈ Θ L D ( θ, g 0 ) such that Assumption 5 and Assumption 6 are satisfied. Then with probability at least 1 -δ , the target parameter ̂ θ produced by Meta-Algorithm 1 enjoys the excess risk bound:

<!-- formula-not-decoded -->

For generic Lipschitz losses, the optimal rate for parametric classes-in the absence of nuisance parameters-scales with n -1 / 2 . Without orthogonality, one expects the dependence on nuisance estimation error to scale linearly with ∥ ̂ g -g 0 ∥ G , which would require ∥ ̂ g -g 0 ∥ 2 G = o ( n -1 ) to achieve the oracle rate. Theorem 2 shows that under orthogonality, the impact of nuisance parameter estimation is of lower order, and it suffices that ∥ ̂ g -g 0 ∥ 2 G = o ( n -1 / 2 ). The proof follows similar reasoning to that of Theorem 1; see Appendix J.

## 3.3 Example: Treatment Effect Estimation

To make matters concrete, we now walk through a detailed example in which we specialize our general framework to the well-studied problem of treatment effect estimation. We show how the setup falls in our framework, explain what statistical assumptions are required to apply our main theorems, and show how to interpret the resulting excess risk bounds.

Following, e.g., Robinson (1988); Nie and Wager (2021), we receive examples z = ( X,W,Y,T ) according to the following data generating process:

<!-- formula-not-decoded -->

where X ∈ X and W ∈ W are covariates, T ∈ { 0 , 1 } is the treatment variable, and Y ∈ R is the target variable. The true target parameter is θ 0 : X → R , but we do not necessarily assume that θ 0 ∈ Θ. The functions e 0 : W → [0 , 1] and f 0 : W → R are unknown; we define m 0 ( x, w ) = E [ Y | X = x, W = w ] = θ 0 ( x ) e 0 ( w ) + f 0 ( w ) and take g 0 = { m 0 , e 0 } to be the true nuisance parameter. We set w = ( X,W,T ) and x = ( X ).

## 3.3.1 Residualized Loss (R-Loss)

Following Robinson (1988); Nie and Wager (2021), we consider the residualized square loss

<!-- formula-not-decoded -->

Let us take a moment to interpret the meaning of excess risk under this loss. It is simple to verify that if the true nuisance parameters g 0 = { m 0 , e 0 } are plugged in, then

<!-- formula-not-decoded -->

Thus, if a predictor θ has low risk it, must be good at predicting θ 0 ( X ) whenever there is sufficient variation in the treatment T . In addition, If the model is not well-specified ( θ 0 / ∈ Θ) but Θ is convex, we can still deduce that

<!-- formula-not-decoded -->

so in this case low excess risk implies that we predict nearly as well as the best predictor in class (again, assuming sufficient variation in T ).

Applying the main results: Fast rates. We now apply Theorem 1 to derive oracle excess risk bounds for the residualized loss. Let us consider the seminorms ∥ θ ∥ 2 Θ := E [ ( T -e 0 ( W )) 2 θ 2 ( X ) ] and ∥·∥ G := ∥·∥ L 4 ( ℓ 2 , D ) and r = 0. Establishing the basic orthogonality and first-order conditions required to apply Theorem 1 is a simple exercise (see Appendix J for a full derivation). To establish the smoothness and strong convexity properties Assumption 3 and Assumption 4, we require mild boundedness assumptions, and a lower bound on the coverage parameter

<!-- formula-not-decoded -->

In particular, we have the following result.

Proposition 1. Consider the treatment effect estimation setting with the residualized loss and norms ∥ θ ∥ 2 Θ := E [ ( T -e 0 ( W )) 2 θ 2 ( X ) ] and ∥·∥ G := ∥·∥ L 4 ( ℓ 2 , D ) . Suppose that θ 0 ∈ Θ and | θ 0 ( x ) | ≤ 1 . Then the assumptions of Theorem 1 are satisfied with constants r = 0 , λ = 1 4 , κ = 4 λ -1 re , β 1 = 1 , and β 2 = 4 λ -1 / 2 re . As a result, the sample splitting meta-algorithm (Meta-Algorithm 1) with the residualized loss produces a parameter ̂ θ such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

More generally, whenever Θ is convex, the same conclusion holds with θ 0 replaced by θ ⋆ ∈ arg min θ ∈ Θ L D ( θ, g 0 ) , regardless of whether θ 0 ∈ Θ .

Proposition 1 implies that for any class, oracle rates for excess risk are achievable whenever Rate D ( G , n, δ ) = o ( Rate D (Θ , · · · ) 1 / 4 ) . Interestingly, in the case where the target class Θ is convex, this holds even when the target parameter is arbitrarily misspecified. In addition, the excess risk bound in Proposition 1 has the desirable property that the coverage parameter λ re enters only through the higher-order nuisance error term.

Let us interpret the coverage parameter λ re , which acts as a problem-dependent constant whose value reflects the interaction between the treatment policy and the treatment effect. In general, to lower bound λ re , it suffices to assume that Var( T -e 0 ( W ) | X ) ≥ η for some η &gt; 0, with no further assumptions required on the data distribution or target parameter class. This condition is typically referred to as overlap , since it requires that the treatment is not deterministic for any realization of the covariates, and implies that λ re ≥ η . On the other hand, even if overlap is not satisfied, one can still lower bound the coverage parameter. To do so, we focus on a special case investigated in Chernozhukov et al. (2017) and Chernozhukov et al. (2018b), where Θ is a class of high-dimensional predictors of the form θ ( x ) = ⟨ w,ϕ ( x ) ⟩ , where w ∈ R p and ϕ : X → R p is a fixed featurization; in general, the dimension p may grow with n , with p ≫ n . In this case, note that it suffices that the matrix E [ Var( T -e 0 ( W ) | X ) ϕ ( X ) ϕ ( X ) ⊤ ] satisfies a restricted minimum eigenvalue condition. Hence, a lower bound on λ re generalizes assumptions used in Chernozhukov et al. (2017, 2018b).

Stronger guarantees for specific target classes. The results in the prequel apply to arbitrary target classes, but require that the nuisance estimation algorithm is close in the L 4 norm (i.e., ∥·∥ G = ∥·∥ L 4 ( ℓ 2 , D ) ). For specific target classes (typically, classes with additional structure that facilitates estimation in parameter error), it is possible to provide improved guarantees that scale with weaker L 2 estimation error for the nuisance class. To illustrate the flexibility of Theorem 1 in accommodating such cases, we consider a constrained variant of the R-learner of Nie and Wager (2021) and recover the oracle rates from this work.

Example 1 (Constrained R-Learner) . The R-learner of Nie and Wager (2021) corresponds to a special case of the treatment effect estimation setup in (19) in which the target parameter belongs to a kernel class, and is estimated by minimizing the orthogonal loss (20) with regularization. Specializing the sample splitting meta-algorithm (Meta-Algorithm 1) to this setting, we obtain a constrained variant of their method.

In more detail, consider the treatment effect estimation setting with ∥ θ ∥ 2 Θ := E [ ( T -e 0 ( W )) 2 θ 2 ( X ) ] and ∥·∥ G := ∥·∥ L 2 ( ℓ 2 , D ) . Let H be a reproducing kernel Hilbert space (RKHS) with norm ∥·∥ H and kernel K . Assume that | Y | ≤ 1 almost surely and that treatments satisfy overlap, and consider the constrained target parameter class

<!-- formula-not-decoded -->

where c ≥ 1 is a parameter. For the target estimation algorithm, consider the plug-in empirical risk minimizer

<!-- formula-not-decoded -->

where ̂ g is the nuisance estimator. Assume that the kernel K has eigenvalue decay of the form σ j ∼ j -1 /p for some parameter p ∈ (0 , 1) and that a smoothed version of θ 0 lies in the RKHS for smoothing parameter α ∈ (0 , 1 / 2) (refer to proof in Appendix J.2 for definitions), and choose c ∝ n α/ ( p +(1 -2 α )) . If the nuisance estimation algorithm has ∥ ̂ g -g 0 ∥ L 2 ( ℓ 2 , D ) ≤ Rate D ( G , S 1 , δ ) = ˜ o ( n -1 / 4 ) , then with probability at least 1 -δ , Meta-Algorithm 1 has

<!-- formula-not-decoded -->

where ˜ O ( · ) suppresses dependence on regularity parameters and log( n ) factors. This matches the best known rate for the oracle learner (Nie and Wager, 2021).

This example shows that an O ( n -1 / 4 ) rate in L 2 -error for the nuisances suffices to achieve the optimal rate in the absence of nuisance parameter. The proof leverages a lemma of Mendelson and Neeman (2010), which states that for all θ ∈ H , ∥ θ ∥ L ∞ ( D ) ≲ ∥ θ ∥ p H ∥ θ ∥ 1 -p L 2 ( D ) , where p is the eigenvalue decay parameter. This allows us to establish Assumption 3 and Assumption 4 with respect to L 2 -error for the nuisance parameter, at the cost of incurring exponent r = p , rather than r = 0 as in the generic result (Proposition 1). Moreover, as a consequence of the norm comparison inequality of Mendelson and Neeman (2010), the L 2 ( D )-error bound from our theorem also implies a bound on L ∞ ( D ) error.

Slow rates. We mention in passing that some distributions may simply not satisfy the coverage condition in (21). In this case, we can appeal to Theorem 2 (we show in Appendix J.2 that the residualized loss satisfies the universal orthogonality property), which does not require any lower bounds in the vein of (21), but leads to slower rates. In general, whether the fast rate (Theorem 1)

or slow rate (Theorem 2) will give better results given finite samples will depend on the behavior of the data distribution and target class.

## 3.3.2 Doubly-Robust Loss (DR-Loss)

As an alternative to the residualized loss, for the special case of a binary treatment, we can use the doubly-robust approach described in Section 1. Consider the special case of (24) in which X = W . Recall that that e 0 ( X ) = E [ T | X ] is the treatment propensity, and define f 0 ( t, x ) = E [ Y | T = t, X = x ]. Define

<!-- formula-not-decoded -->

We take g = { f (1 , · ) , f (0 , · ) , e } as the nuisance parameter. Then the doubly robust loss takes the form

<!-- formula-not-decoded -->

One can verify that E [ φ ( f 0 , e 0 ; z ) | X ] = θ 0 ( X ). As a result, whenever the true nuisance parameters g 0 = { f 0 (0 , · ) , f 0 (1 , · ) , e 0 } are plugged in, the oracle excess risk satisfies

<!-- formula-not-decoded -->

and hence is equivalent to L 2 -error. It is straightforward to verify that the doubly-robust loss satisfies the preconditions of Theorem 1, which leads to the following result.

Proposition 2. Consider the treatment effect estimation setting with the doubly-robust loss and norms ∥·∥ Θ := ∥·∥ L 2 ( D ) and ∥·∥ G := ∥·∥ L 4 ( ℓ 2 , D ) . Suppose that θ 0 ∈ Θ and that | θ ( x ) | , ∥ g ( x ) ∥ ∞ ≤ 1 for all θ ∈ Θ , g ∈ G . In addition, assume that | Y | ≤ 1 almost surely, and that η ≤ e ( X ) ≤ 1 -η for all g = { f (0 , · ) , f (1 , · ) , e } ∈ G . Then the assumptions of Theorem 1 are satisfied with constants r = 0 , λ = 1 4 , κ = 0 , β 1 = 1 , and β 2 = 24 η -3 . As a result, the sample splitting meta-algorithm (Meta-Algorithm 1) with the doubly-robust loss produces a parameter ̂ θ such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

This approach was further developed in the subsequent work of Kennedy (2020), who termed it the DR-Learner , and provided improved oracle estimation rates which are doubly-robust with respect to the estimation errors for the propensities and conditional means. A variant of our two-stage algorithm for the doubly robust loss was also explored in the prior work of Oprescu et al. (2019) for the special case where the target estimation algorithm is a Generalized Random Forest.

We note that the explicit dependence on η in the second order term in Proposition 2 can be avoided if one instead re-defines the inverse propensity term a 0 ( T, X ) = T -e 0 ( X ) e 0 ( X ) (1 -e 0 ( X )) as the nuisance function. In this case, the second part of Assumption 3 is satisfied with respect to the product pre-norm: ∥ g ∥ G = ∥ ( f, a ) ∥ = √ ∥ f ∥ L 4 ( ℓ 2 , D ) ∥ a ∥ L 4 ( ℓ 2 , D ) , since we have that:

<!-- formula-not-decoded -->

Applying Theorem 1 with this definition of nuisance pre-norm yields a doubly robust version of Proposition 2, where only products of nuisance estimation rates arise.

## 3.4 Example: Policy Learning

As a second example, we show how to apply our framework to the classical problem of policy learning. Compared to our treatment effect estimation example, losses for this setting do not typically satisfy the strong convexity property, meaning that Theorem 2 is the relevant meta-theorem, and slow rates are to be expected.

In policy learning, we receive examples of the form Z = ( X,T,Y ), where Y ∈ R is an incurred loss, T ∈ T is a treatment vector and X ∈ X is a vector of covariates. The treatment T is chosen based on an unknown, potentially randomized policy which depends on X . Specifically, we assume the following data generating process:

<!-- formula-not-decoded -->

The learner wishes to optimize over a set of treatment policies Θ ⊆ ( X → T ) (i.e., policies take as input covariates X and return a treatment). Their goal is to produce a policy ̂ θ that achieves small regret with respect to the population risk:

<!-- formula-not-decoded -->

This formulation has been extensively studied in statistics (Qian and Murphy, 2011; Zhao et al., 2012; Zhou et al., 2017; Athey and Wager, 2017; Zhou et al., 2023) and machine learning (Beygelzimer and Langford, 2009; Dud´ ık et al., 2011; Swaminathan and Joachims, 2015a; Kallus and Zhou, 2018); in the latter, it is sometimes referred to as counterfactual risk minimization.

The learner does not know the so-called counterfactual outcome function f 0 , so it is treated as a nuisance parameter. Typically, orthogonalization of this nuisance parameter is possible by utilizing the secondary treatment equation in (24) and fitting a parameter for the observational policy e 0 , which is also treated as a nuisance parameter. We can then write the expected counterfactual reward as

<!-- formula-not-decoded -->

for some known loss function ℓ that utilizes the treatment parameter e 0 . Letting g 0 = { f 0 , e 0 } , the learner's goal can be phrased as minimizing the population risk,

<!-- formula-not-decoded -->

over θ ∈ Θ. This formulation clearly falls into our orthogonal statistical learning framework, where the target parameter is the policy θ and the counterfactual outcome f 0 and observed treatment policy e 0 together form the nuisance parameter g 0 := { f 0 , e 0 } . To facilitate the use of estimation for the nuisance components, one typically assumes access to function classes E and F with e 0 ∈ E and f 0 ∈ F (so that G = F × E ), and fits the nuisance parameters via regression over these classes.

We make this discussion concrete for the special case of binary treatments T ∈ { 0 , 1 } , with additional examples in Appendix G.1. To simplify notation, define p 0 ( t, x ) = P [ T = t | X = x ], so that p 0 ( t, x ) = e 0 ( x ) if t = 1 and 1 -e 0 ( x ) if t = 0. Consider the nuisance parameter g = { f (0 , · ) , f (1 , · ) , e } . Then the loss function

<!-- formula-not-decoded -->

has the structure in (27): it evaluates to the true risk (25) whenever the true nuisance parameter is plugged in. This formulation leads to the well-known doubly-robust estimator for the counterfactual outcome (Cassel et al., 1976; Robins et al., 1994; Robins and Rotnitzky, 1995; Dud´ ık et al., 2011). It is straightforward to verify that the resulting population risk is orthogonal with respect to g . We can also obtain an equivalent loss function by subtracting the loss incurred by choosing treatment 0. Define

<!-- formula-not-decoded -->

and set ℓ ( t, g ; Z ) = β ( g ; Z ) · t . This formulation leads to a linear population risk:

<!-- formula-not-decoded -->

This population risk satisfies universal orthogonality, and Theorem 2 can be applied with ∥·∥ G = ∥·∥ L 2 ( ℓ 2 , D ) whenever the nuisance parameters are bounded appropriately. In particular, we have the following corollary of Theorem 2.

Proposition 3. Consider the policy learning setting with binary treatments and norm ∥·∥ G = ∥·∥ L 2 ( ℓ 2 , D ) . Suppose that | Y | ≤ 1 almost surely, and that all g = { f (0 , · ) , f (1 , · ) , e } ∈ G have | f ( t, X ) | ≤ 1 and e ( X ) ∈ [ η, 1 -η ] for some η ∈ (0 , 1 / 2] . Then with probability at least 1 -δ , the target parameter ̂ θ produced by Meta-Algorithm 1 enjoys the excess risk bound:

<!-- formula-not-decoded -->

Note that this bound depends on the overlap parameter η only through the nuisance error term. We mention in passing that explicit dependence on this parameter can be avoided entirely by treating the inverse propensity term a 0 ( T, X ) = T -e 0 ( X ) e 0 ( X ) (1 -e 0 ( X )) as nuisance parameter (see, e.g., Chernozhukov et al. (2021)). In this case, note that Assumption 6 is satisfied with respect to the product pre-norm: ∥ g ∥ G = ∥ ( f, a ) ∥ = √ ∥ f ∥ L 2 ( ℓ 2 , D ) ∥ a ∥ L 2 ( ℓ 2 , D ) , since we have that:

<!-- formula-not-decoded -->

Applying our Theorem 1 with this definition of nuisance pre-norm, yields a doubly robust version of Proposition 3, where only products of nuisance estimation rates arise. Recent follow-up work of Chernozhukov et al. (2021) provides a statistical learning approach for estimating such nuisance functions with respect to mean-squared error, which is based on minimizing an empirical analogue of the risk function E [ a ( T, X ) 2 -2( a (1 , X ) -a (0 , X )) ] .

## 3.5 Discussion

We close by discussing extensions that build on the results presented in this section, as well as additional connections between our results and existing techniques in the literature on semiparametric inference and double machine learning.

Experiments, additional tools, and applications. Part I of the appendix contains an empirical evaluation of the techniques presented in this section, with applications to treatment effect estimation and policy learning. Part II of the appendix contains supplementary theoretical results that build on the development of this section, including user-friendly variants of the main theorems (Appendix C), construction of orthogonal losses (Appendix D), sufficient conditions to apply the main theorems (Appendix E), and further applications (Appendix G).

Construction of orthogonal losses. While orthogonal losses are already known for many problem settings and statistical models (treatment effect estimation, policy learning, regression with missing or censored data, and so on), for new problems one often begins with a loss that is not necessarily orthogonal. In Appendix D, we give a generic approach to construct orthogonal losses, building on a technique from Chernozhukov et al. (2018b).

One the use of cross-fitting. Meta-Algorithm 1 relies on sample splitting. While this strategy is quite general and results in rate-optimal estimates, it can be inefficient, since the target parameter is only estimated using a subset of the data. A more practical alternative is to employ the well-known cross-fitting approach (e.g., Chernozhukov et al. (2018a)), in which we split the data into K folds, obtain estimators using complementary folds, and combine the results. Cross-fitting variants of Meta-Algorithm 1 are given in Appendix B as Meta-Algorithms 2 and 3.

One can show that under fairly genreal assumptions, the analysis in this section remains valid when cross-fitting is employed, and we recommend this in practice. However, compared to the setting considered in Chernozhukov et al. (2018a), in which cross-fitting provably enjoys improved efficiency over basic sample splitting, there is no hope of establishing that cross-fitting improves efficiency at the level of generality considered in the present work. This is because our framework permits the use of arbitrary, potentially nonparametric or high-dimensional estimators which may be biased due to the use of regularization or constraints and-for example-may not be asymptotically linear. As a result, even in the absence of nuisance parameters, there is no guarantee that averaging multiple target estimators obtained from independent sample splits will lead to improved efficiency.

One the use of influence functions. A special case of our framework can be phrased in the language of classical semiparametric inference as follows: If the population risk functional is pathwise differentiable, and one estimates the target by minimizing an estimator for the risk based on influence functions, which will typically lead to a Neyman orthogonal loss and the resulting target estimator will have favorable second-order errors dependence on the error of the nuisance estimator; see Curth et al. (2020) for follow-up work which takes this approach explicitly. However, Neyman orthogonality goes beyond pathwise differentiability (for instance, one can construct orthogonal losses assuming only existence of pathwise derivatives locally at ( θ ⋆ , g 0 ); see Appendix D), and our results apply in settings where influence functions may not exist. Moreover, one can obtain Neyman orthogonal losses without invoking influence functions, hence orthogonal statistical learning is a more flexible framework. See van der Laan and Robins (2003); Tsiatis (2007); Kosorok (2008); Kennedy (2016) for a review of influence functions and semiparametric theory.

## 4 Instantiating the Main Results: Plug-In Empirical Risk Minimization

The results in Section 3 are stated at a high level of generality, and concern generic estimation algorithms for the target and nuisance parameters. In this section we shift our focus to specific algorithms, and instantiate our general tools to provide explicit bounds based on intrinsic properties of the function classes under consideration. In particular, we develop algorithms and analysis for orthogonal statistical learning with M -estimation losses of the form

<!-- formula-not-decoded -->

We analyze one of the most natural and widely used estimation algorithms for the target parameter: plug-in empirical risk minimization (plug-in ERM) . Specifically, recalling that S = S 1 ∪ S 2 , we

define the empirical risk via

<!-- formula-not-decoded -->

where we adopt the convention that | S | = 2 n with S 2 = { z 1 , . . . , z n } to keep notation compact. The plug-in ERM algorithm returns the minimizer plug-in empirical loss obtained by plugging in the first-stage estimate of the nuisance component:

<!-- formula-not-decoded -->

We provide oracle excess risk bounds for the plug-in ERM algorithm (and variants) in terms of statistical standard complexity measures for the target class Θ. The main results in this section show that the impact of ̂ g on the oracle excess risk achieved ERM is of second order, and that classical excess risk bounds carry over up to lower order terms and constant factors. These results are derived by bounding the second-stage Rate D (Θ , S 2 , δ ; ̂ θ, ̂ g ) using (localized) empirical process tools, then appealing to the main theorems (Theorem 1 and Theorem 2).

In the fast rate regime (i.e., for strongly convex losses) we offer a generalization of the local Rademacher complexity analysis of Bartlett et al. (2005) in the presence of an estimated nuisance component and show that the notion of the critical radius of the class Θ still governs rate Rate D (Θ , S 2 , δ ; ̂ θ, ̂ g ). This leads to several applications of our theory to specific target classes, including sparse linear models, neural networks and kernels (Appendix H.1).

In the slow rate regime (i.e., for generic Lipschitz losses), we offer a novel moment-penalized variant of the plug-in ERM algorithm that achieves a rate whose leading term is equal to the critical radius, multiplied by the variance of the population loss evaluated at the optimal target parameter. This offers an improvement over prior variance-penalized ERM approaches (Maurer and Pontil, 2009), whose leading term depends on the metric entropy of the target function class at single scale, and which typically is larger than the critical radius.

Technical preliminaries. To present our main results, we need to introduce additional tools from empirical process theory and statistical learning. For any real-valued function class G , define the localized Rademacher complexity:

<!-- formula-not-decoded -->

where ϵ 1 , . . . , ϵ n are independent Rademacher random variables. Let R n ( G ) denote the non-localized Rademacher complexity (that is, R n ( G , ∞ )). We also make use of the metric entropy of a function class (which is closely related to the Rademacher complexity). We make the mild assumption that Θ and G are separable, so as to ensure that associated empirical processes are measurable (cf. Boucheron et al. (2013, pp 314-315)).

Definition 3 (Metric Entropy) . For any real-valued function class G and sample z 1: n , the empirical metric entropy H p ( G , ε, z 1: n ) is the logarithm of the size of the smallest function class G ′ , such that for any g ∈ G there exists g ′ ∈ G ′ , with ∥ g -g ′ ∥ L p ( z 1: n ) ≤ ε . Moreover H p ( G , ε, n ) will denote the maximal empirical entropy over all possible sample sets z 1 , . . . , z n .

Finally, for a vector-valued function class F , let F| t = { f t : ( f 1 , . . . , f t , . . . , f d ) ∈ F} denote the projection of the class onto the t -th coordinate.

## 4.1 Fast Rates for Plug-In Empirical Risk Minimization

Our first contribution is an extension of the foundational results of Bartlett et al. (2005); Koltchinskii and Panchenko (2000)-which bound the excess risk for empirical risk minimization in terms of local Rademacher complexities-to incorporate misspecification due to nuisance parameter estimation error. A crucial parameter in this approach is the critical radius δ n of a function class G , defined as the smallest solution to the inequality

<!-- formula-not-decoded -->

Classical work shows that in the absence of a nuisance component, if a loss ℓ ( θ ( z ); z ) is Lipschitz in its first argument and satisfies standard assumptions required for fast rates (strong convexity in the first argument), then empirical risk minimization achieves an excess risk bound of order δ 2 n . For the case of parametric classes, δ n = ˜ O ( n -1 / 2 ), leading to the fast ˜ O ( n -1 ) rates for strongly convex losses. For more general classes (cf. Wainwright (2019)) the critical radius is-up to constant factors-equal to the solution to an inequality on the metric entropy of the function class (cf. Appendix H.1.1):

<!-- formula-not-decoded -->

where G ( δ, z 1: n ) := { g ∈ G : ∥ g ∥ L 2 ( z 1: n ) ≤ δ } ; see Appendix H.1 for concrete examples.

Our first theorem in this section extends this result in the presence of a nuisance component and bounds the excess risk of the plug-in ERM algorithm by the critical radius of the target function class Θ (more precisely, the worst-case critical radius for each coordinate of the target class, since we deal with vector-valued function classes).

Theorem 3 (Fast Rates for Plug-In ERM) . Consider a function class Θ : X → R K 2 with R := sup θ ∈ Θ ∥ θ ∥ L ∞ ( ℓ 2 , D ) ∨ 1 . Let δ 2 n = Ω ( R 2 K 2 log(log( n )) n ) be a solution to the equation:

<!-- formula-not-decoded -->

where θ ⋆ t is the projection of θ ⋆ onto coordinate t . Suppose that ℓ ( · , ̂ g ( w ); z ) is L -Lipschitz with respect to the ℓ 2 norm and that the population risk L D satisfies Assumptions 1 to 4 with ∥·∥ Θ = ∥ · ∥ L 2 ( ℓ 2 , D ) and ∥·∥ G arbitrary. Define B 1 := L 2 K 2 2 λ 2 and B 2 := ( β 2 λ ) 2 1+ r + κ λ , and let ̂ θ be the outcome of the plug-in ERM algorithm. Then, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Critically, when r = 0, the dependence on the nuisance estimation error scales as ∥ ̂ g -g 0 ∥ 4 G due to orthogonality, meaning that we can use a complex function class for nuisance estimation without spoiling the rate for the target class. This result is proven in two steps. First, we show that one can take Rate D (Θ , n, δ ; ̂ θ, ̂ g ) ≲ δ n · ∥ ̂ θ -θ ⋆ ∥ Θ + δ 2 n ; this result uses standard empirical process theory tools, and does not leverage orthogonality. Then, we invoke orthogonality through Theorem 1 to derive the final guarantee. See Appendix L for details.

## 4.2 Slow Rates and Variance Penalization

We now turn to the slow rate regime from Section 3.2, where the loss is not necessarily strongly convex in the prediction. We prove upper bounds on the generalization error of a variance penalized version of the plug-in ERM algorithm. Our main result gives a slow rate that scales with the variance of the loss rather than the range, and is robust to nuisance estimation error. The basic algorithm we analyze first estimates the nuisance parameter, then estimates the optimal loss value µ ⋆ := inf θ ∈ Θ L D ( θ, g 0 ) using auxiliary samples, and finally performs plug-in empirical risk minimization with an empirical variance penalty which is centered using the estimate for µ ⋆ . See Algorithm 1 in Appendix B for a full description. To simplify notation, we assume that | S | = 3 n and is partitioned equal splits S = S 1 ∪ S 2 ∪ S 3 . Define the variance of the loss at ( θ ⋆ , g 0 ) via

<!-- formula-not-decoded -->

Theorem 4 (Plug-In ERM with Centered Second Moment Penalization) . Consider the centered second moment-penalized plugin empirical risk minimizer in Algorithm 1:

<!-- formula-not-decoded -->

where ̂ g is the output of Alg( G , S 1 ) and ̂ µ = inf θ ∈ Θ L S 3 ( θ, ̂ g ) . Consider the function class F = { ℓ ( θ ( · ) , ̂ g ( · ); · ) : θ ∈ Θ } , and let R := sup f ∈F ∥ f ∥ L ∞ ( D ) ∨ 1 and f ⋆ := ℓ ( θ ⋆ ( · ) , ̂ g ( · ); · ) . Let δ 2 n ≥ 0 be any solution to the inequality

<!-- formula-not-decoded -->

Suppose that Assumption 5 holds, ℓ ( θ ( x ) , · ; z ) is L -Lipschitz, and Assumption 6 holds with parameter β and ∥·∥ G := ∥·∥ L 2 ( ℓ 2 , D ) . Let C := ( L 2 + βR ) . Then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

As with the previous result, Theorem 4 is proven by first upper bounding Rate D (Θ , n, δ ; ̂ θ, ̂ g ) using empirical process tools, then invoking orthogonality through one of the main theorems (in this case, Theorem 2). The only complication is that the result requires the additional step of relating Rate D (Θ , · · · , ̂ g ) to the function Rate D (Θ , · · · , g 0 ), which entails bounding the variance of the loss at ̂ g in terms of the variance of the loss at g 0 and nuisance estimation error.

Our approach offers an improvement over the rates for empirical variance penalization in Maurer and Pontil (2009), which provides a generalization error bound whose leading term is of the form: √ Var n ( ℓ ( θ ⋆ ( · ) , ̂ g ( · ) , · )) H ∞ ( ℓ ◦ Θ ,n -1 ,z 1: n ) n . The drawback of such a bound is that it evaluates the metric entropy at a fixed approximation level of 1 /n , which can be suboptimal compared to the critical radius. In Appendix H.2, we show that for classes Θ with bounded VC dimension, this guarantee can be further improved as a consequence of our general machinery, and give a bound which scales with the so-called Alexander capacity function .

Discussion. Due to space constraints, applications to specific target classes (sparse linear models, neural networks, kernel classes) are deferred to Appendix H.1.

## 5 Instantiating the Main Results: Sufficient Conditions for Oracle Rates

The previous section developed guarantees for orthogonal statistical learning with a specific algorithm, plug-in empirical risk minimization. While empirical risk minimization is a workhorse of statistical learning, in general it does not attain minimax excess risk for rich function classes, even in the absence of nuisance parameters. In this section we build on the development so far and, by appealing to aggregation techniques, provide algorithms that always attain minimax excess risk up to secondorder dependence on nuisance parameters. Our main results provide sufficient conditions under which oracle rates are achieved which explicitly depend on intrinsic properties of both the target and nuisance parameter classes. In particular, we give sufficient conditions based on the relationship between the metric entropy for the nuisance and target classes.

For any real-valued function class F , we say that the complexity of F is p if for all ε &gt; 0,

<!-- formula-not-decoded -->

where we recall that H 2 is the metric entropy defined in Definition 3. When p = 0, this corresponds to the case of parametric functions (e.g., linear models and VC-subgraph classes), while for p &gt; 0, we recover nonparametric function classes, such as Lipschitz/smooth functions or kernel spaces. We let p 1 and p 2 denote the maximum complexity of any output coordinate projection for the nuisance and target class, respectively. We provide sufficient conditions on the pair ( p 1 , p 2 ) under which the sample splitting meta-algorithm (Meta-Algorithm 1)-with an appropriate choice for the target and nuisance estimator-can achieve oracle rates.

We focus on the important special case of square losses of the form

<!-- formula-not-decoded -->

where Λ and Γ are known functions, and where we recall from Section 2 that x , w are subsets of the data z , and v ⊆ z is an arbitrary auxiliary subset of the data. We assume that the nuisance parameters are defined in terms of regression problems, i.e., that g 0 ( w ) = E [ u | w ] for some known random vector u ⊆ z . This assumption is standard in semiparametric literature (Bickel et al., 1993; Kosorok, 2008; van der Laan and Rose, 2011), and implies that each coordinate t of g 0 may be expressed as the minimizer of a squared loss: g 0 ,t = arg min g t ∈G| t E [ ( g t ( w ) -u t ) 2 ] . In this setting, a sufficient condition for orthogonality is that

<!-- formula-not-decoded -->

where ∇ ζ and ∇ γ denote the gradient of ℓ with respect to the first and second argument, respectively. In the absence of nuisance parameters, minimax optimal rates for excess risk in square loss regression have been characterized for the well-specified setting in which

<!-- formula-not-decoded -->

for some θ 0 ∈ Θ, and for the misspecified setting where this assumption is removed. In the former setting, the minimax rates are of order Θ( n -2 2+ p 2 ) (Yang and Barron, 1999), while in the latter setting the optimal rate is ˜ Θ( n -2 2+ p 2 ∧ 1 p 2 ) (Rakhlin et al., 2017). We show that under orthogonality, the optimal well-specified and misspecified rates can be achieved in the presence of nuisance parameters even when the nuisance class G is larger than the target class Θ, provided it is not too much larger.

This generalizes the large body of results on semiparametric inference (Levit, 1976; Ibragimov and Has'Minskii, 1981; Pfanzagl, 1982; Bickel, 1982; Klaassen, 1987; Robinson, 1988; Bickel et al., 1993; Newey, 1994; Robins and Rotnitzky, 1995; Ai and Chen, 2003; van der Laan and Dudoit, 2003; van der Laan and Robins, 2003; Ai and Chen, 2007; Tsiatis, 2007; Kosorok, 2008; van der Laan and Rose, 2011; Ai and Chen, 2012; Chernozhukov et al., 2022a; Belloni et al., 2017; Chernozhukov et al., 2018a), which show under various assumptions that if the target class is parametric, one can obtain a √ n -consistent estimator for the target if the nuisance estimator converges at a n -1 4 rate.

Our main workhorse for the results in this section is the 'Aggregation of ε -Nets' or 'Skeleton Aggregation' algorithm described in Yang and Barron (1999) and extended to random design in Rakhlin et al. (2017). The Skeleton Aggregation method operates by splitting the samples in two, building an empirical cover for the function class under consideration using the first split, and then aggregating the elements of the cover using the second split. See Appendix M.4 for a full description. This approach is related to sieve-based methods (e.g., (Semenova and Chernozhukov, 2021)), which employ parametric methods to learn a linear combination of basis elements that approximate the target, but an important difference is that Skeleton Aggregation builds the basis in a data-dependent fashion. We use Skeleton Aggregation as-is to provide rates for the first stage, and provide an extension in the presence of nuisance parameters for the second stage, which entails relating Rate D (Θ , n, δ ; ̂ θ, ̂ g ) to Rate D (Θ , n, δ ; ̂ θ, g 0 ).

We caution that the algorithms in this section are only designed to attain the minimax rates for generic square losses of the type in (43) (e.g., vanilla square loss regression), and specific special cases may admit better rates. Deriving minimax lower bounds for specific losses of interest (as in Kennedy (2020)) is an interesting direction for future research.

Assumptions. Since our aim is to provide sufficient conditions based on the metric entropy of the classes Θ and G , which is already quite technical, we assume that all other problem-dependent parameters are constant. This is only for expository purposes.

Assumption 7. The classes are bounded in the sense that for all θ ∈ Θ + star(Θ -Θ) and g ∈ G +star( G-G , 0) the following bounds hold a.s.: a) ⟨ Λ( g ( w ) , v ) , θ ( x ) ⟩ ∈ [ -1 , +1] , b) Γ( g ( w ) , z ) ∈ [ -1 , +1] , c) Λ( g ( w ) , v )Λ( g ( w ) , v ) ⊤ ⪯ I , d) ∥ g ( w ) ∥ ∞ , ∥ θ ( x ) ∥ ∞ ≤ 1 , e) K 1 , K 2 = O (1) , f) ∥ u ∥ ∞ ≤ 1 almost surely, g) the functions { Λ t ( · , v ) } K 2 t =1 and Γ( · , z ) have O (1) -Lipschitz gradients with respect to ℓ 2 , h) the strong convexity condition E [ ⟨ Λ( g 0 ( w ) , v ) , θ ( x ) -θ ⋆ ( x ) ⟩ 2 ] ≥ γ E ∥ θ ( x ) -θ ⋆ ( x ) ∥ 2 2 is satisfied for all θ ∈ Θ+star(Θ -Θ) for some γ = Ω(1) .

Assumption 7 implies that Assumption 3 and Assumption 4 are satisfied with respect to the seminorms ∥ θ ∥ Θ := ( E ⟨ Λ( g 0 ( w ) , v ) , θ ( x ) ⟩ 2 ) 1 / 2 and ∥·∥ G = ∥·∥ L 4 ( ℓ 2 , D ) , with r = 0. Since typical results on minimax oracle rates provide rates for the nuisance g with respect to ∥ · ∥ L 2 ( ℓ 2 , D ) , we assume control on the ratio between these seminorms.

Assumption 8 (Moment Comparison) . There is a constant C 2 → 4 such that

<!-- formula-not-decoded -->

The moment comparison condition has been used in statistics as a minimal assumption for learning without boundedness (Lecu´ e and Mendelson, 2016; Mendelson, 2014; Liang et al., 2015). For example, suppose that each g ∈ G has the form x ↦→⟨ w,x ⟩ for w,x ∈ R d . Then C 2 → 4 ≤ 3 1 / 4 if x is mean-zero gaussian and C 2 → 4 ≤ √ 8 if x follows any distribution that is independent across all

Figure 1: Relationship between first and second stage for oracle rates; well-specified case.

![Image](data:image/png;base64,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)

coordinates and symmetric (via the Khintchine inequality). Moment comparison is also implied by the 'subgaussian class' assumption used in Mendelson (2011); Lecu´ e and Mendelson (2016). 2 We emphasize that the moment constant C 2 → 4 does not enter the leading term in any of our bounds-only the Rate D ( G , . . . , ) term in Theorem 1-and so it does not affect the asymptotic rates under conditions on metric entropy growth of G that we prescribe in the sequel. We also note that this condition is not required for many classes of interest, where direct L 4 estimation rates are available (see discussion in Appendix E). We adopt the condition here because it allows us to develop guarantees for arbitrary classes at the highest possible level of generality.

Main Result. The main theorem for this section provides sufficient conditions for oracle rates in the well-specified setting (45). For extensions of this result to misspecified models, as well as non-strongly convex losses, see Appendix F.

Theorem 5 (Oracle Rates, Well-Specified Case) . Suppose that we are in the well-specified setting (45), and that Assumptions 1 and 2 and Assumptions 7 and 8 are satisfied for the class ̂ Θ defined below. Suppose that the following relationship holds:

<!-- formula-not-decoded -->

Then for appropriate choice of sub-algorithms, the sample splitting meta-algorithm Meta-Algorithm 1 produces a predictor ̂ θ that guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where ˜ O ( · ) hides problem-dependent parameters and log( δ -1 ) terms. This result matches the minimax rate in the absence of nuisance parameters. In particular, when p 2 ≤ 2 it suffices to take ̂ Θ = Θ and use plug-in ERM for stage two, and when p 2 &gt; 2 it suffices to take ̂ Θ = Θ + star(Θ -Θ , 0) and use Skeleton Aggregation for stage two; in both cases, it suffices to use Skeleton Aggregation for stage one.

Theorem 5 is proven by combining the main theorem (Theorem 1) with algorithm-specific upper bounds on Rate D (Θ , · · · ) and Rate D ( G , · · · ). Figure 1 summarizes the sufficient conditions under

2 Suppose G is scalar-valued and let ∥ g ∥ ψ 2 = inf { c &gt; 0 | E exp ( g 2 ( w ) /c 2 ) ≤ 2 } . Then the subgaussian class assumption for our setting asserts that ∥ g -g 0 ∥ ψ 2 ≤ C ∥ g -g 0 ∥ L 2 ( D ) for all g ∈ G .

which Theorem 5 leads to the oracle rate Θ( n -2 2+ p 2 ) (Yang and Barron, 1999). In particular, whenever Θ is a parametric class (i.e. H 2 (Θ , ε, n ) ∝ d 2 log(1 /ε )), it suffices to take p 1 &lt; 2, which recovers the usual setup for semiparametric inference.

## 6 Discussion

This paper initiates the systematic study of prediction error and excess risk guarantees in the presence of nuisance parameters and Neyman orthogonality. Our results highlight that orthogonality is beneficial for learning with nuisance parameters even in the presence of possible model misspecification, and even when the target parameters belong to large nonparametric classes. We also show that many of the typical assumptions used to analyze estimation in the presence of nuisance parameters can be relaxed when excess risk is the focus. There are many promising future directions, including weakening assumptions, obtaining sharper guarantees for specific settings and losses of interest (e.g., doubly-robust guarantees), and analyzing further algorithms for general function classes (along the lines of Sections 4 and 5). We refer to the appendix for additional results, as well as empirical results.

Acknowledgements. We are grateful to the anonymous COLT reviewers and to Xiaohong Chen for pointing out additional related work. Part of this work was completed while DF was an intern at Microsoft Research, New England. DF acknowledges support from the Facebook PhD fellowship and NSF Tripods grant #1740751.

## References

- C. Ai and X. Chen. Efficient estimation of models with conditional moment restrictions containing unknown functions. Econometrica , 71(6):1795-1843, 2003.
- C. Ai and X. Chen. Estimation of possibly misspecified semiparametric conditional moment restriction models with different conditioning variables. Journal of Econometrics , 141(1):5-43, 2007.
- C. Ai and X. Chen. The semiparametric efficiency bound for models of sequential moment restrictions containing unknown functions. Journal of Econometrics , 170(2):442-457, 2012.
- M. Anthony and P. L. Bartlett. Neural network learning: Theoretical foundations . Cambridge University Press, 1999.
- S. Arora, S. Du, W. Hu, Z. Li, and R. Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning , pages 322-332, 2019.
- S. Athey and S. Wager. Efficient policy learning. arXiv preprint arXiv:1702.02896 , 2017.
- S. Athey, J. Tibshirani, and S. Wager. Generalized random forests. The Annals of Statistics , 47(2): 1148-1178, 2019.
8. J.-Y. Audibert. Progressive mixture rules are deviation suboptimal. In Advances in Neural Information Processing Systems , pages 41-48, 2008.

- P. L. Bartlett, O. Bousquet, and S. Mendelson. Local rademacher complexities. The Annals of Statistics , 33(4):1497-1537, 2005.
- P. L. Bartlett, D. J. Foster, and M. Telgarsky. Spectrally-normalized margin bounds for neural networks. Advances in Neural Information Processing Systems (NIPS) , 2017.
- P. L. Bartlett, N. Harvey, C. Liaw, and A. Mehrabian. Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Conference on Learning Theory , 2017.
- A. Belloni, V. Chernozhukov, I. Fern´ andez-Val, and C. Hansen. Program evaluation and causal inference with high-dimensional data. Econometrica , 85(1):233-298, 2017.
- S. Ben-David and R. Urner. On the hardness of domain adaptation and the utility of unlabeled target samples. In International Conference on Algorithmic Learning Theory , pages 139-153. Springer, 2012.
- S. Ben-David, J. Blitzer, K. Crammer, and F. Pereira. Analysis of representations for domain adaptation. In Advances in neural information processing systems , pages 137-144, 2007.
- D. Bertsimas and C. McCord. Optimization over continuous and multi-dimensional decisions with observational data. In Advances in Neural Information Processing Systems , pages 2962-2970, 2018.
- A. Beygelzimer and J. Langford. The offset tree for learning with partial labels. In Proceedings of the 15th ACM SIGKDD international conference on Knowledge discovery and data mining , pages 129-138. ACM, 2009.
- P. J. Bickel. On adaptive estimation. The Annals of Statistics , pages 647-671, 1982.
- P. J. Bickel, C. A. Klaassen, P. J. Bickel, and Y. Ritov. Efficient and adaptive estimation for semiparametric models , volume 4. Johns Hopkins University Press Baltimore, 1993.
- J. Blitzer, K. Crammer, A. Kulesza, F. Pereira, and J. Wortman. Learning bounds for domain adaptation. In Advances in neural information processing systems , pages 129-136, 2008.
- R. Blundell, X. Chen, and D. Kristensen. Semi-nonparametric iv estimation of shape-invariant engel curves. Econometrica , 75(6):1613-1669, 2007.
- S. Boucheron, G. Lugosi, and P. Massart. Concentration inequalities: A nonasymptotic theory of independence . Oxford university press, 2013.
- O. Bousquet. Concentration inequalities for sub-additive functions using the entropy method. In Stochastic inequalities and applications , pages 213-247. Springer, 2003.
- O. Bousquet, S. Boucheron, and G. Lugosi. Introduction to statistical learning theory. In Advanced lectures on machine learning , pages 169-207. Springer, 2004.
- C. M. Cassel, C. E. S¨ arndal, and J. H. Wretman. Some results on generalized difference estimation and generalized regression estimation for finite populations. Biometrika , 63(3):615-620, 1976.
- X. Chen and T. M. Christensen. Optimal sup-norm rates and uniform inference on nonlinear functionals of nonparametric iv regression. Quantitative Economics , 9(1):39-84, 2018.

- X. Chen and D. Pouzo. Efficient estimation of semiparametric conditional moment models with possibly nonsmooth residuals. Journal of Econometrics , 152(1):46-60, 2009.
- X. Chen and D. Pouzo. Estimation of nonparametric conditional moment models with possibly nonsmooth generalized residuals. Econometrica , 80(1):277-321, 2012.
- X. Chen and D. Pouzo. Sieve Wald and QLR inferences on semi/nonparametric conditional moment models. Econometrica , 83(3):1013-1079, 2015.
- X. Chen and H. White. Improved rates and asymptotic normality for nonparametric neural network estimators. IEEE Transactions on Information Theory , 45(2):682-691, 1999.
- V. Chernozhukov, M. Goldman, V. Semenova, and M. Taddy. Orthogonal machine learning for demand estimation: High dimensional causal inference in dynamic panels. arXiv preprint arXiv:1712.09988 , 2017.
- V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey, and J. Robins. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal , 21(1):C1-C68, 2018a.
- V. Chernozhukov, D. Nekipelov, V. Semenova, and V. Syrgkanis. Plug-in regularized estimation of high-dimensional parameters in nonlinear semiparametric models. arXiv preprint arXiv:1806.04823 , 2018b.
- V. Chernozhukov, W. Newey, and J. Robins. Double/de-biased machine learning using regularized riesz representers. arXiv preprint arXiv:1802.08667 , 2018c.
- V. Chernozhukov, W. K. Newey, V. Quintas-Martinez, and V. Syrgkanis. Automatic debiased machine learning via neural nets for generalized linear regression. arXiv preprint arXiv:2104.14737 , 2021.
- V. Chernozhukov, J. C. Escanciano, H. Ichimura, W. K. Newey, and J. M. Robins. Locally robust semiparametric estimation. Econometrica , 90(4):1501-1535, 2022a.
- V. Chernozhukov, W. K. Newey, and R. Singh. De-biased machine learning of global and local parameters using regularized riesz representers. The Econometrics Journal , 2022b.
- C. Cortes, Y. Mansour, and M. Mohri. Learning bounds for importance weighting. In Advances in neural information processing systems , pages 442-450, 2010.
- F. Cucker and S. Smale. On the mathematical foundations of learning. Bulletin of the American Mathematical Society , 39(1):1-49, 2002.
- A. Curth, A. M. Alaa, and M. van der Schaar. Estimating structural target functions using machine learning and influence functions. arXiv preprint arXiv:2008.06461 , 2020.
- T. Dao, G. M. Kamath, V. Syrgkanis, and L. Mackey. Knowledge distillation as semiparametric inference. In International Conference on Learning Representations , 2020.
- H. Daume III and D. Marcu. Domain adaptation for statistical classifiers. Journal of artificial Intelligence research , 26:101-126, 2006.
- B. Delyon and A. Juditsky. On minimax wavelet estimators. Applied and Computational Harmonic Analysis , 3(3):215-228, 1996.

- I. D´ ıaz and M. J. van der Laan. Targeted data adaptive estimation of the causal dose-response curve. Journal of Causal Inference , 1(2):171-192, 2013.
- M. Dud´ ık, J. Langford, and L. Li. Doubly robust policy evaluation and learning. In Proceedings of the 28th International Conference on International Conference on Machine Learning , pages 1097-1104. Omnipress, 2011.
- M. H. Farrell, T. Liang, and S. Misra. Deep neural networks for estimation and inference. Econometrica , 89(1):181-213, 2021.
- D. J. Foster, S. Kale, H. Luo, M. Mohri, and K. Sridharan. Logistic regression: The importance of being improper. Conference on Learning Theory , 2018.
- R. Friedberg, J. Tibshirani, S. Athey, and S. Wager. Local linear forests. Journal of Computational and Graphical Statistics , 30(2):503-517, 2020.
- E. Gin´ e and V. Koltchinskii. Concentration inequalities and asymptotic results for ratio type empirical processes. The Annals of Probability , 34(3):1143-1216, 2006.
- N. Golowich, A. Rakhlin, and O. Shamir. Size-independent sample complexity of neural networks. Conference on Learning Theory , 2018.
- B. S. Graham. Efficiency bounds for missing data models with semiparametric restrictions. Econometrica , 79(2):437-452, 2011.
- A. Guntuboyina and B. Sen. Global risk bounds and adaptation in univariate convex regression. Probability Theory and Related Fields , 163(1-2):379-411, 2015.
- P. Hall, J. L. Horowitz, et al. Nonparametric methods for inference in the presence of instrumental variables. The Annals of Statistics , 33(6):2904-2929, 2005.
- S. Hanneke. Theory of disagreement-based active learning. Foundations and Trends ® in Machine Learning , 7(2-3):131-309, 2014.
- T. Hastie, R. Tibshirani, and M. Wainwright. Statistical learning with sparsity: the lasso and generalizations . CRC press, 2015.
- D. Haussler. Sphere packing numbers for subsets of the boolean n-cube with bounded vapnikchervonenkis dimension. Journal of Combinatorial Theory, Series A , 69(2):217-232, 1995.
- D. A. Hirshberg and S. Wager. Augmented minimax linear estimation. The Annals of Statistics , 49 (6):3206-3227, 2021.
- I. A. Ibragimov and R. Z. Has'Minskii. Statistical estimation: asymptotic theory . Springer-Verlag, New York, 1981.
- J. Jiang and C. Zhai. Instance weighting for domain adaptation in NLP. In Proceedings of the 45th annual meeting of the association of computational linguistics , pages 264-271, 2007.
- N. Kallus and A. Zhou. Policy evaluation and optimization with continuous treatments. In International Conference on Artificial Intelligence and Statistics , pages 1243-1251, 2018.
- E. H. Kennedy. Semiparametric theory and empirical processes in causal inference. In Statistical causal inferences and their applications in public health research , pages 141-167. Springer, 2016.

- E. H. Kennedy. Optimal doubly robust estimation of heterogeneous causal effects. arXiv preprint arXiv:2004.14497 , 2020.
- E. H. Kennedy, Z. Ma, M. D. McHugh, and D. S. Small. Non-parametric methods for doubly robust estimation of continuous treatment effects. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 79(4):1229-1245, 2017.
- E. H. Kennedy, S. Lorch, and D. S. Small. Robust causal inference with continuous instruments using the local instrumental variable curve. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 81(1):121-143, 2019.
- G. Kerkyacharian, O. Lepski, and D. Picard. Nonlinear estimation in anisotropic multi-index denoising. Probability theory and related fields , 121(2):137-170, 2001.
- G. Kerkyacharian, O. Lepski, and D. Picard. Nonlinear estimation in anisotropic multi-index denoising. sparse case. Theory of Probability &amp; Its Applications , 52(1):58-77, 2008.
- C. A. Klaassen. Consistent estimation of the influence function of locally asymptotically linear estimators. The Annals of Statistics , pages 1548-1562, 1987.
- V. Koltchinskii. Oracle Inequalities in Empirical Risk Minimization and Sparse Recovery Problems . Springer, 2011.
- V. Koltchinskii and D. Panchenko. Rademacher processes and bounding the risk of function learning. High Dimensional Probability II , 47:443-459, 2000.
- M. R. Kosorok. Introduction to empirical processes and semiparametric inference. Springer, 2008.
- S. R. K¨ unzel, J. S. Sekhon, P. J. Bickel, and B. Yu. Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences , 116 (10):4156-4165, 2019.
- G. Lecu´ e and S. Mendelson. Learning subgaussian classes: Upper and minimax bounds. In Topics in Learning Theory . Societe Mathematique de France, 2016.
- E. L. Lehmann and G. Casella. Theory of point estimation . Springer Science &amp; Business Media, 2006.
- O. Lepski. Asymptotically minimax adaptive estimation. I: Upper bounds. optimally adaptive estimates. Theory of Probability &amp; Its Applications , 36(4):682-697, 1992.
- B. Y. Levit. On the efficiency of a class of non-parametric estimates. Theory of Probability &amp; Its Applications , 20(4):723-740, 1976.
- T. Liang, A. Rakhlin, and K. Sridharan. Learning with square loss: Localization through offset rademacher complexity. In Proceedings of The 28th Conference on Learning Theory , pages 1260-1285, 2015.
- L. Mackey, V. Syrgkanis, and I. Zadik. Orthogonal machine learning: Power and limitations. In International Conference on Machine Learning , pages 3375-3383, 2018.
- Y. Mansour, M. Mohri, and A. Rostamizadeh. Domain adaptation: Learning bounds and algorithms. In 22nd Conference on Learning Theory (COLT) 2009 , 2009.

- A. Maurer. A vector-contraction inequality for Rademacher complexities. In International Conference on Algorithmic Learning Theory , pages 3-17. Springer, 2016.
- A. Maurer and M. Pontil. Empirical Bernstein bounds and sample variance penalization. In The 22nd Conference on Learning Theory (COLT) , 2009.
- S. Mendelson. Improving the sample complexity using global data. IEEE transactions on Information Theory , 48(7):1977-1991, 2002.
- S. Mendelson. Discrepancy, chaining and subgaussian processes. The Annals of Probability , 39(3): 985-1026, 2011.
- S. Mendelson. Learning without concentration. In Conference on Learning Theory (COLT) , pages 25-39, 2014.
- S. Mendelson and J. Neeman. Regularization in kernel learning. The Annals of Statistics , 38(1): 526-565, 2010.
- W. K. Newey. The asymptotic variance of semiparametric estimators. Econometrica: Journal of the Econometric Society , pages 1349-1382, 1994.
- W. K. Newey and J. L. Powell. Instrumental variable estimation of nonparametric models. Econometrica , 71(5):1565-1578, 2003.
- J. Neyman. Optimal asymptotic tests of composite hypotheses. Probability and statsitics , pages 213-234, 1959.
- J. Neyman. c ( α ) tests and their use. Sankhy¯ a: The Indian Journal of Statistics, Series A , pages 1-21, 1979.
- X. Nie and S. Wager. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika , 108 (2):299-319, 2021.
- Y. Ning, P. Sida, and K. Imai. Robust estimation of causal effects via a high-dimensional covariate balancing propensity score. Biometrika , 107(3):533-554, 2020.
- M. Oprescu, V. Syrgkanis, and Z. S. Wu. Orthogonal random forest for causal inference. In International Conference on Machine Learning , pages 4932-4941, 2019.
- J. Pfanzagl. Contributions to a general asymptotic statistical theory. 1982.
- M. Qian and S. A. Murphy. Performance guarantees for individualized treatment rules. Annals of statistics , 39(2):1180, 2011.
- A. Rakhlin and K. Sridharan. Statistical learning and sequential prediction, 2012. Preprint. Available at http://www.mit.edu/ ∼ rakhlin/courses/stat928/stat928 notes.pdf .
- A. Rakhlin, K. Sridharan, and A. B. Tsybakov. Empirical entropy, minimax regret and minimax risk. Bernoulli , 23(2):789-824, 2017.
- J. Robins, L. Li, E. Tchetgen, and A. van der Vaart. Higher order influence functions and minimax estimation of nonlinear functionals. In Probability and statistics: essays in honor of David A. Freedman , pages 335-421. Institute of Mathematical Statistics, 2008.

- J. M. Robins and A. Rotnitzky. Semiparametric efficiency in multivariate regression models with missing data. Journal of the American Statistical Association , 90(429):122-129, 1995.
- J. M. Robins and A. Rotnitzky. Comment on the Bickel and Kwon article,'Inference for semiparametric models: Some questions and an answer'. Statistica Sinica , 11(4):920-936, 2001.
- J. M. Robins, A. Rotnitzky, and L. P. Zhao. Estimation of regression coefficients when some regressors are not always observed. Journal of the American statistical Association , 89(427): 846-866, 1994.
- P. M. Robinson. Root-n-consistent semiparametric regression. Econometrica: Journal of the Econometric Society , pages 931-954, 1988.
- P. R. Rosenbaum and D. B. Rubin. The central role of the propensity score in observational studies for causal effects. Biometrika , 70(1):41-55, 1983.
- D. Rubin and M. J. van der Laan. A general imputation methodology for nonparametric regression with censored data. 2005.
- D. Rubin and M. J. van der Laan. A doubly robust censoring unbiased transformation. The international journal of biostatistics , 3(1), 2007.
- D. O. Scharfstein, A. Rotnitzky, and J. M. Robins. Rejoinder-Adjusting for nonignorable drop-out using semiparametric nonresponse models. Journal of the American Statistical Association , 94 (448):1135-1146, 1999.
- V. Semenova and V. Chernozhukov. Debiased machine learning of conditional average treatment effects and other causal functions. The Econometrics Journal , 24(2):264-289, 2021.
- S. Shalev-Shwartz and S. Ben-David. Understanding machine learning: From theory to algorithms . Cambridge university press, 2014.
- H. Shimodaira. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of statistical planning and inference , 90(2):227-244, 2000.
- N. Srebro, K. Sridharan, and A. Tewari. Smoothness, low noise and fast rates. In Advances in Neural Information Processing Systems , pages 2199-2207, 2010.
- C. J. Stone. Optimal rates of convergence for nonparametric estimators. The annals of Statistics , pages 1348-1360, 1980.
- C. J. Stone. Optimal global rates of convergence for nonparametric regression. The annals of statistics , pages 1040-1053, 1982.
- A. Swaminathan and T. Joachims. Counterfactual risk minimization: Learning from logged bandit feedback. In International Conference on Machine Learning , pages 814-823, 2015a.
- A. Swaminathan and T. Joachims. The self-normalized estimator for counterfactual learning. In Advances in Neural Information Processing Systems , pages 3231-3239, 2015b.
- A. Tsiatis. Semiparametric theory and missing data . Springer Science &amp; Business Media, 2007.
- A. B. Tsybakov. Pointwise and sup-norm sharp adaptive estimation of functions on the Sobolev classes. The Annals of Statistics , 26(6):2420-2469, 1998.

- M. J. van der Laan and S. Dudoit. Unified cross-validation methodology for selection among estimators and a general cross-validated adaptive epsilon-net estimator: Finite sample oracle inequalities and examples. 2003.
- M. J. van der Laan and A. R. Luedtke. Targeted learning of an optimal dynamic treatment, and statistical inference for its mean outcome. 2014.
- M. J. van der Laan and J. M. Robins. Unified methods for censored longitudinal data and causality . Springer Science &amp; Business Media, 2003.
- M. J. van der Laan and S. Rose. Targeted learning: causal inference for observational and experimental data . Springer Science &amp; Business Media, 2011.
- M. J. van der Laan and D. Rubin. Targeted maximum likelihood learning. The International Journal of Biostatistics , 2(1), 2006.
- M. J. van der Laan, S. Dudoit, and A. van der Vaart. The cross-validated adaptive epsilon-net estimator. Statistics &amp; Decisions. International Mathematical Journal for Stochastic Methods and Models , 24(3):373-395, 2006.
- M. J. van der Laan, E. C. Polley, and A. E. Hubbard. Super learner. Statistical applications in genetics and molecular biology , 6(1), 2007.
- A. van der Vaart. Asymptotic statistics , volume 3. Cambridge university press, 2000.
- A. van der Vaart and M. J. van der Laan. Estimating a survival distribution with current status data and high-dimensional covariates. The International Journal of Biostatistics , 2(1), 2006.
- V. N. Vapnik. The nature of statistical learning theory . Springer, 1995.
- M. J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint . Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2019.
- C. Wang, Q. Wu, M. Weimer, and E. Zhu. Flaml: A fast and lightweight automl library. In MLSys , 2021.
- L. Wang, A. Rotnitzky, and X. Lin. Nonparametric regression with missing outcomes using weighted kernel estimating equations. Journal of the American Statistical Association , 105(491):1135-1146, 2010.
- Y. Yang and A. Barron. Information-theoretic determination of minimax rates of convergence. Annals of Statistics , pages 1564-1599, 1999.
- T. Zhang. Covering number bounds of certain regularized linear function classes. Journal of Machine Learning Research , 2(Mar):527-550, 2002.
- Y. Zhao, D. Zeng, A. J. Rush, and M. R. Kosorok. Estimating individualized treatment rules using outcome weighted learning. Journal of the American Statistical Association , 107(499):1106-1118, 2012.
- W. Zheng and M. J. van der Laan. Asymptotic theory for cross-validated targeted maximum likelihood estimation. 2010.

- X. Zhou, N. Mayer-Hamblett, U. Khan, and M. R. Kosorok. Residual weighted learning for estimating individualized treatment rules. Journal of the American Statistical Association , 112 (517):169-187, 2017.
- Z. Zhou, S. Athey, and S. Wager. Offline multi-action policy learning: Generalization and optimization. Operations Research , 71(1):148-183, 2023.

## Organization of Appendix

The appendix is organized as follows.

- Part I contains experimental results evaluating the performance of orthogonal risk minimization methods for examples considered in Section 3.
- Part II contains supplemental theoretical results: Algorithms omitted from the main body for space (Appendix B), user-friendly variants of the main theorems (Appendix C), construction of orthogonal losses (Appendix D), sufficient conditions to apply the main theorems, (Appendix E), sufficient conditions for oracle rates that extend Section 5 (Appendix F), applications of the main results (Appendix G), further results regarding plug-in empirical risk minimization (generalization guarantees for specific function classes (Appendix H.1), and an application of the guarantees for plug-in variance-penalized ERM developed in Section 4 to VC classes (Appendix H.2).
- Part III contains proofs for the results presented in the main body of the paper: Appendix I contains preliminaries, Appendix J proves the main theorems for the sample-splitting metaalgorithm, Appendix L proves the main results for plug-in empirical risk minimization (building on technical tools developed in Appendix K), and Appendix M proves our results concerning oracle rates.

## Part I

## Experiments

## A Experiments

We conducted experiments to evaluate of the benefit of risk minimization with orthogonal loss functions as compared to non-orthogonal losses, focusing on the main applications considered in Section 3: conditional average treatment effect (CATE) estimation under conditional exogeneity (Section 3.3) and policy learning (Section 3.4).

For both applications, we use synthetic data. Each dataset contains n i.i.d. examples, where each example is a tuple consisting of a d -dimensional random vector of confounders ( X ∈ R d ), a random binary treatment ( T ∈ { 0 , 1 } ), and a random scalar outcome ( Y ∈ R ). These random variables are drawn from a data generating process of the form:

<!-- formula-not-decoded -->

We refer to e 0 ( X ) as the propensity function, b 0 ( X ) as the base response function, and τ 0 ( X ) as the treatment effect function. The dimension d and noise level σ 2 are parameters that vary across experiments.

Inspired by the set of experiments in the prior work of Nie and Wager (2021); Athey and Wager (2017) we consider six variants of the data-generating process in (48) that capture different qualitative challenges for CATE estimation and policy learning.

- In Setup A we consider X ∼ U ([0 , 1]) d , e 0 ( X ) = clip(sin( π X [0] X [1]) , . 2 , . 8), τ 0 ( X ) = . 2 + ( X [0] + X [1]) / 2, and b 0 ( X ) = sin( π X [0] X [1]) + 2 ( X [2] -. 5) 2 + X [3] + . 5 X [4], where clip( x, a, b ) := max { a, min { b, x }} denotes the operator that clips x to the interval [ a, b ]. This setup has a complicated propensity function and base response, but a relatively simple treatment effect function τ 0 ( X ).
- In Setup B , we consider X ∼ U ([ -. 5 , . 5]) d , e 0 ( X ) = . 5, b 0 ( X ) = max { 0 , X [0] + X [1] , X [2] } + max { 0 , X [3] + X [4] } , and τ 0 ( X ) = X [0] + log(1 + exp { X [1] } ). This setup is a randomized trial with a complex base response and treatment effect function.
- In Setup C , we consider X ∼ U ([ -. 5 , . 5]) d , e 0 ( X ) = (1 + exp { X [1] + X [2] } ) -1 , b 0 ( X ) = 2 log(1 + exp { X [0] + X [1] + X [2] } ), and τ 0 ( X ) = 1. This setup has a complex propensity and base response function, but a constant treatment effect.
- In Setup D , we consider X ∼ U ([ -. 5 , . 5]) d , e 0 ( X ) = (1 + exp {-X [0] } +exp {-X [1] } ) -1 , b 0 ( X ) = . 5 max { 0 , X [0] + X [1] + X [2] } + . 5 max { 0 , X [3] + X [4] } , and τ 0 ( X ) = max { 0 , X [0] + X [1]+ X [2] }-max { 0 , X [3]+ X [4] } . In this setup, the response under treatment is independent of the response under control, and hence there is no statistical benefit of jointly learning the two responses.
- In Setup E , we consider X ∼ U ([ -. 5 , . 5]) d , e 0 ( X ) = (1 + exp { 3 X [1] + 3 X [2] } ) -1 , b 0 ( X ) = 5 max { 0 , X [0] + X [1] } , and τ 0 ( X ) = 2 (( X [0] &gt; . 1) ∨ ( X [1] &gt; . 1)) -1. In this setup, we have a very wide range of responses that is largely explained by the confounder. Hence, failing to appropriately center the response around the predictable part can lead to large variance in the learning objective. Moreover, the treatment effect function is discontinuous.
- In Setup F , we consider X ∼ U ([ -. 5 , . 5]) d , e 0 ( X ) = (1 + exp { 3 X [1] + 3 X [2] } ) -1 , b 0 ( X ) = 5 max { 0 , X [0] + X [1] } , τ 0 ( X ) = X [0] + log(1 + exp { X [1] } ). This setup is similar to setup E , but with a smooth treatment effect function.

For each setup, we consider three values for the sample size ( n ∈ { 500 , 1000 , 3000 } ), two values for the dimension of the confounders ( d ∈ { 6 , 12 } ), and three values for the scale of the response noise ( σ ∈ { . 5 , 1 , 2 } ). For each such parameter setting, we run 100 experiments and report average metrics and standard errors for the average metric (for the appropriate metric for each task).

Reproducibility. Code for reproducing the results is available at https://github.com/vsyrgkanis/ orthogonal learning . We note that because the FLAML package used for nuisance model estimation does not allow for control of random seeds for fully reproducible execution, results will have slight variation across multiple executions, albeit not to an extent that affects the qualitative take-aways.

## A.1 Conditional Average Treatment Effect Estimation

We first consider the task of estimating the conditional average treatment effect τ 0 ( x ) (i.e., we have θ 0 = τ 0 ) with respect to the mean squared error metric E [( ̂ θ ( X ) -θ 0 ( X ) ) 2 ] . We use the two orthogonal losses considered in Section 3.3. The first is the residual-on-residual loss:

<!-- formula-not-decoded -->

where g = { q, e } , with the true values of the nuisance parameters given by q 0 ( X ) = E [ Y | X ] and e 0 ( X ) = E [ T | X ]. Second, we consider the doubly-robust loss:

<!-- formula-not-decoded -->

where g = { f, e } , with the true values of the nuisance parameters given by f 0 ( T, X ) = E [ Y | T, X ] and e 0 ( X ) = E [ T | X ]. For each loss, we consider two risk minimization approaches: a semi-crossfitting approach that allows for some sample re-use, and a pure sample splitting approach.

- In the semi-crossfitting approach, we perform model selection using the FLAML (Wang et al., 2021) automated machine learning package, which performs hyperparameter tuning and selection using a broad class of forest based models. Using the model class and hyperparameters chosen by FLAML, we perform cross-fitting: we train a set of the nuisance models (e.g. ̂ e, ̂ q ) on half of the data and compute their values (e.g. ̂ e i := e ( X i ) , ̂ q i := q ( X i )) on the other half, and vice-versa. We then minimize the loss function over θ using all the samples with their computed nuisance values (e.g. 1 n ∑ n i =1 ( Y i -̂ q i -θ ( X i ) ( T i -̂ e i )) 2 for the residual-on-residual loss); this phase is also performed using FLAML.
- In the pure sample-splitting approach, we split the dataset in half and perform hyperparameter tuning, model selection, and model fitting with FLAML on the first half. We then compute nuisance values on the second half. Finally, we estimate the target function ̂ θ , also using the second half of the dataset. This approach corresponds to the meta-algorithm presented in the main body (Meta-Algorithm 1).

We note that while the theoretical results in this paper only concern the pure sample-splitting approach, we expect that the semi-crossfitting approach will also benefit from the orthogonality of the loss, and it has been analyzed in prior literature for estimation of finite-dimensional target parameters (Chernozhukov et al., 2018a).

We refer to the estimation approach that uses the residual-on-residual loss with semi-crossfitting as dml, and refer to the variant with pure sample splitting as dml split. Similarly, we refer to the two variants that use the doubly robust loss as dr and dr split. Finally, as a benchmark we also consider the non-orthogonal loss:

<!-- formula-not-decoded -->

which we apply without any sample splitting or cross-fitting. We refer to this method as the slearner.

For each method, to perform risk minimization for the target parameter θ , we applied the FLAML package, which performs automated machine learning and hyperparameter tuning across many candidate forest models. In order to make the target parameter inherently simpler than then nuisance parameters, we constrained the final stage model to depend only on the first n x &lt; d coordinates of the variables X . For all setups except D , we used n x = 4, while for setup D we used n x = 5 to maintain well-specification of the target model.

Results. The average mean-squared error for the estimated target parameters ̂ θ across the 100 experiments and across the different setups are depicted in Figures 2, 3, 4, 5, 6, 7. We find that the methods that use orthogonal losses either significantly out-perform or are comparable to the non-orthogonal loss in most domains. The superior performance vanishes only in the presence

of small samples and high noise level σ , potentially due to large errors in the nuisance functions. Moreover, we find that for relatively small sample size, the orthogonal losses perform comparably to the oracle method, which uses the true models p 0 , q 0 and minimizes the residual-on-residual loss (denoted by oracle). Moreover, we find that sample-splitting is sufficient to get a substantial boost in performance, and that cross-fitting tends to give small further improvement over sample-splitting.

Figure 2: Setup A. Conditional Average Treatment Effect (CATE) estimation. Mean-Squared-Error (MSE), averaged across 100 experiments, with standard error.

|         |        |                 | oracle                      | dml                         | dml split                         | dr dr split                                       | slearner                                  |
|---------|--------|-----------------|-----------------------------|-----------------------------|-----------------------------------|---------------------------------------------------|-------------------------------------------|
| n = 500 | d = 6  | σ = 0 . 5 σ = 1 | 0.039 (0.003) 0.090 (0.011) | 0.057 (0.003) 0.083 (0.006) | 0.076 (0.004) 0.077 0.124 (0.011) | (0.004) 0.084 (0.005) 0.124 (0.010) 0.322 (0.093) | 0.113 (0.006) 0.137 (0.007) 0.181 (0.012) |
| n = 500 |        | σ = 2           |                             |                             |                                   | 0.092 (0.006)                                     |                                           |
| n = 500 |        |                 | 0.240 (0.047)               | 0.181 (0.031)               | 0.382 (0.074)                     | 0.171 (0.020)                                     |                                           |
| n = 500 |        | σ = 0 . 5       | 0.041 (0.004)               | 0.062 (0.003)               | 0.076 (0.005) 0.076               | (0.003) 0.085 (0.005)                             | 0.107 (0.006)                             |
| n = 500 | d = 12 | σ = 1           | 0.073 (0.006)               | 0.080 (0.005)               | 0.161 (0.026) 0.089               | (0.005) 0.134 (0.014)                             | 0.111 (0.007)                             |
| n = 500 |        | σ = 2           | 0.187 (0.025)               | 0.136 (0.009)               | 0.219 (0.016) 0.149               | (0.010) 0.290 (0.031)                             | 0.208 (0.013)                             |
|         |        | σ = 0 . 5       | 0.025 (0.002)               | 0.055 (0.002)               | 0.062 (0.003)                     | 0.071 (0.003) 0.074 (0.003)                       | 0.118 (0.006)                             |
|         | d = 6  | σ = 1           | 0.070 (0.017)               | 0.061 (0.004)               | 0.082 (0.007) 0.078 (0.004)       | 0.088 (0.004)                                     | 0.116 (0.006)                             |
|         |        | σ = 2           | 0.112 (0.015)               | 0.119 (0.007)               | 0.224 (0.036) 0.102               | (0.008) 0.159 (0.015)                             | 0.155 (0.010)                             |
|         | d = 12 | σ = 0 . 5       | 0.022 (0.001)               | 0.057 (0.003)               | 0.067 (0.004) 0.070 (0.003)       | 0.074 (0.003)                                     | 0.107 (0.006)                             |
|         |        | σ = 1           | 0.058 (0.004)               | 0.062 (0.003)               | 0.088 (0.007) 0.075 (0.004)       | 0.086 (0.006)                                     | 0.125 (0.006)                             |
|         |        | σ = 2           | 0.120 (0.015)               | 0.112 (0.007)               | 0.180 (0.020) 0.118 (0.012)       | 0.146 (0.011)                                     | 0.171 (0.009)                             |
|         |        | σ = 0 . 5       | 0.017 (0.001)               | 0.038 (0.001)               | 0.045 (0.002) 0.073               | (0.002) 0.074 (0.002)                             | 0.115 (0.005)                             |
|         | d = 6  | σ = 1           | 0.034 (0.002)               | 0.048 (0.003)               | 0.053 (0.002) 0.071               | (0.003) 0.070 (0.003)                             | 0.112 (0.005)                             |
|         |        | σ = 2           | 0.075 (0.009)               | 0.064 (0.003)               | 0.079 (0.004) 0.078 (0.004)       | 0.088 (0.007)                                     | 0.142 (0.006)                             |
|         |        | σ = 0 . 5       | 0.017 (0.001)               | 0.049 (0.002)               | 0.052 (0.002) 0.074               | (0.002) 0.076 (0.002)                             | 0.138 (0.004)                             |
|         | d = 12 | σ = 1           | 0.034 (0.003)               | 0.050 (0.002)               | 0.058 (0.003) 0.075               | (0.003) 0.079 (0.003)                             | 0.142 (0.004)                             |
|         |        | σ = 2           | 0.094 (0.012)               | 0.062 (0.003)               | 0.079 (0.005) 0.076               | (0.004) 0.098 (0.007)                             | 0.131 (0.007)                             |

Figure 3: Setup B. Conditional Average Treatment Effect (CATE) estimation. Mean-Squared-Error (MSE), averaged across 100 experiments, with standard error.

|         |        |                 | oracle      | oracle          | dml         | dml             | dml split   | dml split       |                             | dr split                    | slearner    | slearner        |
|---------|--------|-----------------|-------------|-----------------|-------------|-----------------|-------------|-----------------|-----------------------------|-----------------------------|-------------|-----------------|
| n = 500 | d = 6  | σ = 0 . 5 σ = 1 | 0.037 0.100 | (0.002) (0.007) | 0.063       | (0.002)         | 0.075       | (0.002) (0.006) | 0.063 (0.002) 0.088 (0.005) | 0.074 (0.003)               | 0.154       | (0.011) (0.010) |
| n = 500 |        |                 |             |                 | 0.084       | (0.003)         | 0.112       |                 |                             | 0.115 (0.005)               | 0.186       |                 |
| n = 500 |        | σ = 2           | 0.206       | (0.018)         | 0.142       | (0.009)         | 0.240       | (0.023)         | 0.171 (0.019)               | 0.418 (0.143)               | 0.263       | (0.015)         |
| n = 500 |        | σ = 0 . 5       | 0.042       | (0.002)         | 0.065       | (0.001)         | 0.075       | (0.002)         | 0.063 (0.001)               | 0.079 (0.003)               | 0.182       | (0.011)         |
| n = 500 | d = 12 | σ = 1           | 0.090       | (0.005)         | 0.084       | (0.003)         | 0.104       | (0.004)         | 0.096 (0.007)               | 0.137 (0.017)               | 0.188       | (0.010)         |
|         | d = 6  | σ = 2           | 0.224       | (0.026) (0.001) | 0.172       | (0.013)         | 0.283       | (0.038)         | 0.159 (0.007)               | 0.540 (0.207)               | 0.302       | (0.018) (0.011) |
|         |        | σ = 0 . 5 σ = 1 | 0.033 0.072 | (0.008)         | 0.056 0.071 | (0.001) (0.001) | 0.060 0.082 | (0.001) (0.003) | 0.055 (0.001) 0.070 (0.001) | 0.058 (0.001) 0.082 (0.002) | 0.139 0.174 | (0.011)         |
|         |        | σ = 2           | 0.141       | (0.010)         | 0.109       | (0.005)         | 0.155       | (0.012)         | 0.137 (0.018)               | 0.170 (0.012)               | 0.228       | (0.012)         |
|         |        | σ = 0 . 5       | 0.028       | (0.001)         | 0.057       | (0.001)         | 0.064       | (0.001)         | 0.059 (0.001)               | 0.064 (0.001)               | 0.197       | (0.011)         |
|         | d = 12 | σ = 1           | 0.063       | (0.004)         | 0.069       | (0.001)         | 0.088       | (0.003)         | 0.072 (0.002)               | 0.086 (0.002)               | 0.163       | (0.010)         |
|         |        | σ = 2           | 0.155       | (0.021)         | 0.113       | (0.005)         | 0.233       | (0.027)         | 0.138 (0.013)               | 0.181 (0.020)               | 0.228       | (0.012)         |
|         |        | σ = 0 . 5       | 0.021       | (0.001)         | 0.057       | (0.000)         | 0.058       | (0.001)         | 0.057 (0.001)               | 0.056 (0.001)               | 0.109       | (0.009)         |
|         | d = 6  | σ = 1           | 0.040       | (0.003)         | 0.064       | (0.001)         | 0.067       | (0.001)         | 0.063 (0.001)               | 0.066 (0.001)               | 0.146       | (0.011)         |
|         |        | σ = 2           | 0.096       | (0.009)         | 0.074       | (0.002)         | 0.090       | (0.003)         | 0.076 (0.002)               | 0.098 (0.006)               | 0.205       | (0.011)         |
|         |        | σ = 0 . 5       | 0.022       | (0.001)         | 0.058       | (0.001)         | 0.059       | (0.001)         | 0.058 (0.001)               | 0.057 (0.002)               | 0.078       | (0.006)         |
|         | d = 12 | σ = 1           | 0.040       | (0.003)         | 0.063       | (0.001)         | 0.066       | (0.001)         | 0.062 (0.001)               | 0.065 (0.002)               | 0.149       | (0.011)         |
|         |        | σ = 2           | 0.087       | (0.008)         |             | (0.002)         |             |                 | (0.002)                     | 0.090 (0.002)               | 0.143       |                 |
|         |        |                 |             |                 | 0.074       |                 | 0.090       | (0.003)         | 0.075                       |                             |             | (0.010)         |

Figure 4: Setup C. Conditional Average Treatment Effect (CATE) estimation. Mean-Squared-Error (MSE), averaged across 100 experiments, with standard error.

|         |        |                       | oracle            | oracle                  | dml               | dml             | dml split         | dml split               |                                                 | dr split                          | slearner          | slearner                |
|---------|--------|-----------------------|-------------------|-------------------------|-------------------|-----------------|-------------------|-------------------------|-------------------------------------------------|-----------------------------------|-------------------|-------------------------|
| n = 500 | d = 6  | σ = 0 . 5 σ = 1 σ = 2 | 0.012 0.036 0.159 | (0.004) (0.006) (0.029) | 0.033 0.053 0.171 | (0.004) (0.006) | 0.042 0.126 0.225 | (0.004) (0.014) (0.011) | 0.035 (0.004) 0.066 (0.006) 0.167 (0.011) 0.043 | 0.059 (0.006) 0.176 (0.075) 0.288 | 0.238 0.246 0.398 | (0.021) (0.017) (0.026) |
| n = 500 |        |                       |                   |                         |                   | (0.018)         |                   |                         |                                                 | (0.019)                           |                   |                         |
| n = 500 | d = 12 | σ = 0 . 5             | 0.009             | (0.001)                 | 0.033             | (0.004)         | 0.044             | (0.005)                 | (0.005)                                         | 0.060 (0.006)                     | 0.252             | (0.021)                 |
| n = 500 |        | σ = 1                 | 0.030             | (0.005)                 | 0.083             | (0.010)         | 0.108             | (0.011)                 | 0.090 (0.011)                                   | 0.128 (0.013)                     | 0.261             | (0.018)                 |
| n = 500 | d = 6  | σ = 0 . 5             | 0.006             | (0.001)                 | 0.013             | (0.002)         | 0.019             | (0.003)                 | 0.021 (0.004)                                   | 0.037 (0.005)                     | 0.185             | (0.018)                 |
|         |        | σ = 1                 | 0.017             | (0.003)                 | 0.042             | (0.005)         | 0.062             | (0.006)                 | 0.050 (0.006)                                   | 0.070 (0.006)                     | 0.233             | (0.020)                 |
|         |        | σ = 2                 | 0.084             | (0.014)                 | 0.104             | (0.013)         | 0.150             | (0.015)                 | 0.099 (0.008)                                   | 0.168 (0.016)                     | 0.271             | (0.022)                 |
|         |        | σ = 0 . 5             | 0.006             | (0.002)                 | 0.016             | (0.002)         | 0.039             | (0.005)                 | 0.026 (0.003)                                   | 0.032 (0.004)                     | 0.244             | (0.019)                 |
|         | d = 12 | σ = 1                 | 0.022             | (0.004)                 | 0.050             | (0.006)         | 0.073             | (0.007)                 | 0.049 (0.005)                                   | 0.063 (0.006)                     | 0.158             | (0.014)                 |
|         |        | σ = 2                 | 0.073             | (0.014)                 | 0.109             | (0.012)         | 0.177             | (0.018)                 | 0.098 (0.007)                                   | 0.159 (0.009)                     | 0.224             | (0.020)                 |
|         |        | σ = 0 . 5             | 0.002             | (0.000)                 | 0.012             | (0.002)         | 0.015             | (0.002)                 | 0.015 (0.002)                                   | 0.018 (0.003)                     | 0.182             | (0.021)                 |
|         | d = 6  | σ = 1                 | 0.010             | (0.002)                 | 0.021             | (0.003)         | 0.035             | (0.005)                 | 0.027 (0.004)                                   | 0.033 (0.004)                     | 0.203             | (0.020)                 |
|         |        | σ = 2                 | 0.051             | (0.010)                 | 0.063             | (0.006)         | 0.077             | (0.006)                 | 0.068 (0.006)                                   | 0.076 (0.008)                     | 0.185             | (0.017)                 |
|         |        | σ = 0 . 5             | 0.003             | (0.001)                 | 0.008             | (0.001)         | 0.013             | (0.002)                 | 0.017 (0.003)                                   | 0.018 (0.002)                     | 0.161             | (0.019)                 |
|         | d = 12 | σ = 1                 | 0.008             | (0.001)                 | 0.026             | (0.004)         | 0.044             | (0.005)                 | 0.022 (0.003)                                   | 0.040 (0.005)                     | 0.224             | (0.022)                 |
|         |        | σ = 2                 | 0.039             | (0.008)                 | 0.053             | (0.006)         | 0.091             | (0.006)                 | 0.048 (0.005)                                   | 0.076 (0.006)                     | 0.242             | (0.022)                 |

Figure 5: Setup D. Conditional Average Treatment Effect (CATE) estimation. Mean-Squared-Error (MSE), averaged across 100 experiments, with standard error.

|        |             | oracle      | oracle          | dml         | dml             | dml split   | dml split       | dr                          | dr split                    | slearner    | slearner        |
|--------|-------------|-------------|-----------------|-------------|-----------------|-------------|-----------------|-----------------------------|-----------------------------|-------------|-----------------|
| d = 6  | σ = 0 . 5   | 0.088       | (0.003)         | 0.117       | (0.003)         | 0.136       | (0.009)         | 0.120 (0.003)               | 0.129 (0.004)               | 0.135       | (0.001)         |
|        | σ = 1       | 0.143       | (0.005)         | 0.156       | (0.009)         | 0.176       | (0.013)         | 0.145 (0.005)               | 0.202 (0.017)               | 0.140       | (0.001)         |
|        | σ = 2       | 0.274       | (0.022)         | 0.240       | (0.010)         | 0.404       | (0.038)         | 0.295 (0.032)               | 0.373 (0.027)               | 0.152       | (0.006)         |
|        | σ = 0 . 5   | 0.088       | (0.002)         | 0.114       | (0.002)         | 0.126       | (0.003)         | 0.116 (0.003)               | 0.129 (0.005)               | 0.134       | (0.001)         |
| d = 12 | σ = 1 σ = 2 | 0.161 0.246 | (0.008) (0.013) | 0.141 0.266 | (0.004) (0.029) | 0.178 0.366 | (0.007) (0.024) | 0.167 (0.010) 0.262 (0.016) | 0.212 (0.018) 0.375 (0.020) | 0.141 0.146 | (0.001) (0.004) |
|        | σ = 0 . 5   |             | (0.002)         | 0.111       | (0.002)         | 0.117       | (0.003)         | 0.115 (0.003)               | 0.118 (0.003)               | 0.132       | (0.001)         |
| d = 6  | σ = 1       | 0.076 0.109 | (0.004)         | 0.132       | (0.004)         | 0.146       | (0.005)         | 0.138 (0.005)               | 0.157 (0.009)               | 0.137       | (0.001)         |
|        | σ = 2       | 0.206       | (0.018)         | 0.182       | (0.009)         | 0.234       | (0.011)         | 0.183 (0.006)               | 0.263 (0.026)               | 0.142       | (0.001)         |
|        | σ = 0 . 5   | 0.079       | (0.002)         | 0.111       | (0.002)         | 0.115       | (0.002)         | 0.111 (0.002)               | 0.119 (0.003)               | 0.131       | (0.001)         |
| d = 12 | σ = 1       | 0.113       | (0.005)         | 0.137       | (0.004)         | 0.149       | (0.008)         | 0.140 (0.005)               | 0.152 (0.006)               | 0.138       | (0.001)         |
|        | σ = 2       | 0.211       | (0.018)         | 0.178       | (0.006)         | 0.245       | (0.023)         | 0.242 (0.032)               | 0.244 (0.010)               | 0.143       | (0.001)         |
|        | σ = 0 . 5   | 0.058       | (0.002)         | 0.107       | (0.000)         | 0.108       | (0.000)         | 0.107 (0.000)               | 0.107 (0.000)               | 0.127       | (0.001)         |
| d = 6  | σ = 1       | 0.085       | (0.002)         | 0.114       | (0.003)         | 0.117       | (0.003)         | 0.112 (0.002)               | 0.135 (0.005)               | 0.132       | (0.001)         |
|        | σ = 2       | 0.153       | (0.010)         | 0.148       | (0.005)         | 0.162       | (0.005)         | 0.162 (0.014)               | 0.158 (0.005)               | 0.137       | (0.001)         |
|        | σ = 0 . 5   | 0.059       | (0.002)         | 0.109       | (0.001)         | 0.108       | (0.001)         | 0.106 (0.000)               | 0.108 (0.001)               | 0.129       | (0.001)         |
| d = 12 | σ = 1       | 0.091       | (0.003)         | 0.126       | (0.004)         | 0.126       | (0.004)         | 0.123 (0.004)               | 0.127 (0.004)               | 0.131       | (0.001)         |
|        | σ = 2       | 0.143       | (0.009)         | 0.133       | (0.004)         | 0.148       | (0.004)         | 0.143 (0.004)               | 0.157 (0.005)               | 0.139       | (0.000)         |

Figure 6: Setup E. Conditional Average Treatment Effect (CATE) estimation. Mean-Squared-Error (MSE), averaged across 100 experiments, with standard error.

|         |        |                       | oracle                                    | dml                                       | dml split                                 | dr dr split                                                                 | slearner                                  |
|---------|--------|-----------------------|-------------------------------------------|-------------------------------------------|-------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------|
| n = 500 | d = 6  | σ = 0 . 5 σ = 1 σ = 2 | 0.121 (0.004) 0.248 (0.014) 0.660 (0.033) | 0.548 (0.012) 0.567 (0.014) 0.730 (0.024) | 0.595 (0.013) 0.655 (0.017) 0.905 (0.038) | 0.568 (0.013) 0.611 (0.014) 0.583 (0.014) 0.693 (0.019) 0.757 (0.023) 0.894 | 0.978 (0.007) 0.954 (0.009) 0.984 (0.005) |
| n = 500 |        |                       |                                           |                                           |                                           | (0.023)                                                                     |                                           |
| n = 500 | d = 12 | σ = 0 . 5             | 0.138 (0.011)                             | 0.507 (0.012)                             | 0.564 (0.016)                             | 0.544 (0.012) 0.599 (0.017)                                                 | 0.965 (0.011)                             |
| n = 500 |        | σ = 1                 | 0.271 (0.015)                             | 0.548 (0.012)                             | 0.672 (0.027)                             | 0.588 (0.014) 0.644 (0.018)                                                 | 0.984 (0.004)                             |
| n = 500 |        | σ = 2                 | 0.785 (0.073)                             | 0.793 (0.042)                             | 0.897 (0.039)                             | 0.790 (0.032) 0.926 (0.044)                                                 | 0.985 (0.010)                             |
|         |        | σ = 0 . 5             | 0.075 (0.005)                             | 0.450 (0.007)                             | 0.517 (0.012) 0.460                       | (0.011) 0.520 (0.011)                                                       | 0.951 (0.013)                             |
|         | d = 6  | σ = 1                 | 0.190 (0.013)                             | 0.522 (0.011)                             | 0.555 (0.012) 0.517                       | (0.010) 0.601 (0.018)                                                       | 0.969 (0.009)                             |
|         |        | σ = 2                 | 0.428 (0.030)                             | 0.558 (0.015)                             | 0.810 (0.046) 0.588                       | (0.013) 0.710 (0.019)                                                       | 0.985 (0.004)                             |
|         |        | σ = 0 . 5             | 0.076 (0.004)                             | 0.492 (0.007)                             | 0.522 (0.009) 0.522                       | (0.005) 0.548 (0.010)                                                       | 0.976 (0.008)                             |
|         | d = 12 | σ = 1                 | 0.167 (0.009)                             | 0.515 (0.009)                             | 0.554 (0.014)                             | 0.531 (0.007) 0.580 (0.012)                                                 | 0.970 (0.008)                             |
|         |        | σ = 2                 | 0.451 (0.028)                             | 0.579 (0.018)                             | 0.740 (0.030) 0.603                       | (0.014) 0.784 (0.031)                                                       | 0.972 (0.008)                             |
|         |        | σ = 0 . 5             | 0.056 (0.008)                             | 0.475 (0.005)                             | 0.504 (0.007)                             | 0.510 (0.005) 0.529 (0.008)                                                 | 0.905 (0.020)                             |
|         | d = 6  | σ = 1                 | 0.091 (0.004)                             | 0.475 (0.006)                             | 0.515 (0.009)                             | 0.522 (0.005) 0.526 (0.009)                                                 | 0.915 (0.018)                             |
|         |        | σ = 2                 | 0.213 (0.014)                             | 0.510 (0.009)                             | 0.568 (0.013) 0.561                       | (0.009) 0.590 (0.013)                                                       | 0.960 (0.012)                             |
|         |        | σ = 0 . 5             | 0.036 (0.004)                             | 0.494 (0.006)                             | 0.502 (0.007) 0.525                       | (0.005) 0.532 (0.008)                                                       | 0.957 (0.012)                             |
|         |        |                       |                                           | 0.492 (0.007)                             | 0.526                                     | (0.006) 0.532                                                               | 0.945 (0.015)                             |
|         | d = 12 | σ = 1                 | 0.119 (0.008)                             |                                           | 0.514 (0.008)                             | (0.009)                                                                     |                                           |
|         |        | σ = 2                 | 0.215 (0.014)                             | 0.543 (0.010)                             | 0.576 (0.012) 0.550                       | (0.008) 0.575 (0.012)                                                       | 0.967 (0.009)                             |

Figure 7: Setup F. Conditional Average Treatment Effect (CATE) estimation. Mean-Squared-Error (MSE), averaged across 100 experiments, with standard error.

|         |        |                 | oracle      | oracle          | dml         | dml             | dml split   | dml split       | dr                                | dr split              | slearner          | slearner        |
|---------|--------|-----------------|-------------|-----------------|-------------|-----------------|-------------|-----------------|-----------------------------------|-----------------------|-------------------|-----------------|
| n = 500 | d = 6  | σ = 0 . 5 σ = 1 | 0.045 0.104 | (0.002) (0.005) | 0.164 0.180 | (0.010) (0.008) | 0.184 0.242 | (0.012) (0.018) | 0.169 (0.006) 0.196 (0.010) 0.249 | 0.189 (0.008) (0.014) | 0.534 0.547 0.522 | (0.011) (0.009) |
| n = 500 |        |                 |             |                 |             |                 |             |                 |                                   | 0.239                 |                   |                 |
| n = 500 |        | σ = 2           | 0.341       | (0.050)         | 0.251       | (0.014)         | 0.332       | (0.019)         | (0.013)                           | 0.420 (0.048)         |                   | (0.013)         |
| n = 500 |        | σ = 0 . 5       | 0.050       | (0.003)         | 0.171       | (0.008)         | 0.189       | (0.010)         | 0.183 (0.008)                     | 0.216 (0.020)         | 0.568             | (0.007)         |
| n = 500 | d = 12 | σ = 1           | 0.148       | (0.049)         | 0.181       | (0.009)         | 0.231       | (0.017)         | 0.208 (0.013)                     | 0.228 (0.012)         | 0.529             | (0.012)         |
|         | d = 6  | σ = 0 . 5       | 0.036       | (0.002)         | 0.107       | (0.003)         | 0.179       | (0.017)         | 0.130 (0.004)                     | 0.155 (0.006)         | 0.519             | (0.013)         |
|         |        | σ = 1           | 0.087       | (0.007)         | 0.135       | (0.005)         | 0.192       | (0.011)         | 0.157 (0.006)                     | 0.187 (0.012)         | 0.536             | (0.011)         |
|         |        | σ = 2           | 0.177       | (0.022)         | 0.178       | (0.009)         | 0.246       | (0.012)         | 0.178 (0.007)                     | 0.278 (0.017)         | 0.526             | (0.010)         |
|         |        | σ = 0 . 5       | 0.035       | (0.002)         | 0.131       | (0.004)         | 0.161       | (0.009)         | 0.162 (0.005)                     | 0.173 (0.006)         | 0.518             | (0.012)         |
|         | d = 12 | σ = 1           | 0.091       | (0.016)         | 0.154       | (0.011)         | 0.174       | (0.008)         | 0.156 (0.008)                     | 0.181 (0.009)         | 0.518             | (0.013)         |
|         |        | σ = 2           | 0.196       | (0.022)         | 0.171       | (0.007)         | 0.230       | (0.012)         | 0.177 (0.007)                     | 0.310 (0.047)         | 0.521             | (0.012)         |
|         |        | σ = 0 . 5       | 0.026       | (0.001)         | 0.164       | (0.006)         | 0.166       | (0.006)         | 0.185 (0.006)                     | 0.178 (0.007)         | 0.546             | (0.013)         |
|         | d = 6  | σ = 1           | 0.049       | (0.005)         | 0.145       | (0.006)         | 0.151       | (0.006)         | 0.150 (0.005)                     | 0.163 (0.006)         | 0.544             | (0.013)         |
|         |        | σ = 2           | 0.145       | (0.018)         | 0.134       | (0.005)         | 0.163       | (0.008)         | 0.165 (0.007)                     | 0.160 (0.006)         | 0.509             | (0.014)         |
|         |        | σ = 0 . 5       | 0.025       | (0.001)         | 0.124       | (0.005)         | 0.143       | (0.006)         | 0.154 (0.005)                     | 0.157 (0.005)         | 0.530             | (0.013)         |
|         | d = 12 | σ = 1           | 0.043       | (0.003)         | 0.146       | (0.006)         | 0.163       | (0.007)         | 0.171 (0.006)                     | 0.185 (0.007)         | 0.514             | (0.016)         |
|         |        | σ = 2           | 0.118       | (0.015)         | 0.125       | (0.004)         | 0.156       | (0.006)         | 0.151 (0.004)                     | 0.184 (0.007)         | 0.499             | (0.015)         |

## A.2 Policy Learning

Our second set of experiments concern the task of policy learning, where the goal is to minimize the loss L D ( θ ) = -E [ θ ( X ) ( τ 0 ( X ) -c )] (or equivalently, maximize the reward E [ θ ( X ) ( τ 0 ( X ) -c )]), where θ : X → { 0 , 1 } is the treatment policy (the target parameter) and c ∈ R is a pre-defined treatment cost; for all experiments, we use the mean of the heterogeneous treatment effect as the cost (i.e. c := E [ τ 0 ( X )]). We implemented the orthogonal doubly-robust loss (denoted dr) described in Section 3.4 (Eq. (29)):

<!-- formula-not-decoded -->

where g = { f, e } is the nuisance parameter, with f 0 ( T, X ) = E [ Y | T, X ] and e 0 ( X ) := E [ T | X ]. We consider two non-orthogonal losses as benchmarks. The first benchmark loss (denoted direct) is based solely on the regression model f :

<!-- formula-not-decoded -->

while the second benchmark loss (denoted ips), is based solely on the propensity model e :

<!-- formula-not-decoded -->

We consider the same data generating process setups (A-F) as in the CATE estimation setting (in particular, the true nuisance parameters f 0 and e 0 are chosen in the same fashion). As in the CATE estimation experiments, we used FLAML to fit the nuisance and target parameters. We performed hyperparameter tuning, model selection and model fitting on half of each dataset, and performed loss minimization over θ on the other half. For the target parameter (the treatment policy) we searched over binary decision trees of depth at most 2, with a minimum leaf size of at least 20 samples. These trees where constructed using a greedy method, where-starting at the root node-we recursively choose at each node a split that greedily leads to the largest improvement in the target criterion.

Results. We present our results in Figures 8, 9, 10, 11, 12, which display the mean reward (negative of loss) for the policies learned by the approaches above; we omit setup C because the treatment effect is constant, which renders the policy learning problem trivial. Similar to our CATE results, we find that in most settings, the orthogonal loss approach gives comparable results to the non-orthogonal losses, but for setups E and F , the orthogonal loss method significantly outperforms the non-orthogonal losses. As a secondary result, we compare to

- The unrestricted optimal policy θ opt ( X ) = 1 { τ ( X ) -c &gt; 0 } (denoted opt).
- An oracle method (denoted as or) that uses the true nuisance functions and optimizes the oracle objective over the same space of target policies as in dr, direct, and ips.

We find that when the sample size is large and the variance of the outcome is relatively small, the learned policies perform comparably to the oracle policy.

Figure 8: Setup A. Policy Learning. Learned policy value, averaged across 100 experiments, with standard error.

|        |                       | opt               |                         | or                |                         | dr                |                         | direct      |                         | ips               |                         |
|--------|-----------------------|-------------------|-------------------------|-------------------|-------------------------|-------------------|-------------------------|-------------|-------------------------|-------------------|-------------------------|
| d = 6  | σ = 0 . 5 σ = 1 σ = 2 | 0.083 0.083 0.083 | (0.000) (0.000) (0.000) | 0.077 0.077 0.077 | (0.000) (0.000) (0.000) | 0.021 0.005 0.004 | (0.003) (0.003) (0.002) | 0.028 0.012 | (0.002) (0.002) (0.002) | 0.055 0.043 0.033 | (0.002) (0.003) (0.003) |
|        | σ = 0 . 5             | 0.083             | (0.000)                 | 0.077             | (0.000)                 | 0.019             | (0.002)                 | 0.001       |                         |                   | (0.003)                 |
| d = 12 | σ = 1                 | 0.083             | (0.000)                 | 0.077             | (0.000)                 |                   |                         | 0.034       | (0.003)                 | 0.047             | (0.003)                 |
|        | σ = 2                 | 0.083             | (0.000)                 | 0.077             | (0.000)                 | 0.005             | (0.002)                 | 0.012       | (0.002)                 | 0.041 0.027       | (0.002)                 |
|        | σ = 0 . 5             | 0.082             | (0.000)                 | 0.077             |                         | 0.002             | (0.002)                 | 0.004       | (0.002) (0.003)         | 0.060             | (0.002)                 |
| d = 6  | σ = 1                 | 0.082             | (0.000)                 | 0.077             | (0.000) (0.000)         | 0.029 0.013       | (0.002) (0.002)         | 0.038 0.019 | (0.002)                 | 0.053             | (0.002)                 |
|        | σ = 2                 | 0.082             | (0.000)                 | 0.077             | (0.000)                 | 0.005             | (0.002)                 | 0.007       | (0.002)                 | 0.039             | (0.003)                 |
| d = 12 | σ = 0 . 5             | 0.084             | (0.000)                 | 0.078             | (0.000)                 | 0.033             | (0.003)                 | 0.032       | (0.003)                 | 0.060             | (0.002)                 |
|        | σ = 1                 | 0.084             | (0.000)                 | 0.078             | (0.000)                 | 0.009             | (0.002)                 | 0.016       | (0.002)                 | 0.053             | (0.002)                 |
|        | σ = 2                 | 0.084             | (0.000)                 | 0.078             | (0.000)                 | 0.005             | (0.002)                 | 0.004       | (0.002)                 | 0.030             | (0.003)                 |
|        | σ = 0 . 5             | 0.083             | (0.000)                 | 0.078             | (0.000)                 | 0.054             | (0.002)                 | 0.040       | (0.002)                 | 0.068             | (0.002)                 |
| d = 6  | σ = 1                 | 0.083             | (0.000)                 | 0.078             | (0.000)                 | 0.032             | (0.003)                 | 0.033       | (0.003)                 | 0.069             | (0.001)                 |
|        | σ = 2                 | 0.083             | (0.000)                 | 0.078             | (0.000)                 | 0.016             | (0.002)                 | 0.015       | (0.002)                 | 0.063             | (0.001)                 |
|        | σ = 0 . 5             | 0.080             | (0.000)                 | 0.075             | (0.000)                 | 0.041             | (0.003)                 | 0.028       | (0.003)                 | 0.067             | (0.001)                 |
| d = 12 | σ = 1                 | 0.080             | (0.000)                 | 0.075             | (0.000)                 | 0.017             | (0.003)                 | 0.023       | (0.003)                 | 0.066             | (0.001)                 |
|        | σ = 2                 | 0.080             | (0.000)                 | 0.075             | (0.000)                 | 0.001             | (0.002)                 | 0.013       | (0.002)                 | 0.056             | (0.002)                 |

Figure 9: Setup B. Policy Learning. Learned policy value, averaged across 100 experiments, with standard error.

|        |                       | opt               |                 | or          |                         | dr                |                 | direct      |                 | ips               |                 |
|--------|-----------------------|-------------------|-----------------|-------------|-------------------------|-------------------|-----------------|-------------|-----------------|-------------------|-----------------|
| d = 6  | σ = 0 . 5 σ = 1 σ = 2 | 0.135 0.135 0.135 | (0.000) (0.000) | 0.124 0.124 | (0.000) (0.000) (0.000) | 0.078 0.026 0.011 | (0.004) (0.005) | 0.076 0.037 | (0.004) (0.005) | 0.042 0.031 0.005 | (0.005) (0.005) |
|        | σ = 0 . 5             |                   | (0.000)         | 0.124       |                         |                   | (0.004)         | 0.013       | (0.003)         |                   | (0.004)         |
| d = 12 |                       | 0.135             | (0.000)         | 0.124       | (0.000)                 | 0.058             | (0.005)         | 0.064       | (0.005)         | 0.043             | (0.005)         |
|        | σ = 1                 | 0.135             | (0.000)         | 0.124       | (0.000)                 | 0.027             | (0.004)         | 0.032       | (0.005)         | 0.020             | (0.004)         |
|        | σ = 2                 | 0.135             | (0.000)         | 0.124       | (0.000)                 | 0.008             | (0.002)         | 0.004       | (0.002)         | 0.008             | (0.003)         |
|        | σ = 0 . 5             | 0.135             | (0.000)         | 0.124       | (0.000)                 | 0.104             | (0.003)         | 0.097       | (0.002)         | 0.081             | (0.004)         |
| d = 6  | σ = 1                 | 0.135             | (0.000)         | 0.124       | (0.000)                 | 0.058             | (0.005)         | 0.052       | (0.004)         | 0.049             | (0.005)         |
|        | σ = 2                 | 0.135             | (0.000)         | 0.124       | (0.000)                 | 0.015             | (0.004)         | 0.010       | (0.002)         | 0.014             | (0.004)         |
|        | σ = 0 . 5             | 0.136             | (0.000)         | 0.125       | (0.000)                 | 0.094             | (0.004)         | 0.087       | (0.004)         | 0.068             | (0.005)         |
| d = 12 | σ = 1                 | 0.136             | (0.000)         | 0.125       | (0.000)                 | 0.043             | (0.005)         | 0.051       | (0.005)         | 0.035             | (0.004)         |
|        | σ = 2                 | 0.136             | (0.000)         | 0.125       | (0.000)                 | 0.007             | (0.003)         | 0.013       | (0.003)         | 0.014             | (0.003)         |
|        | σ = 0 . 5             | 0.136             | (0.000)         | 0.125       | (0.000)                 | 0.116             | (0.002)         | 0.106       | (0.002)         | 0.108             | (0.003)         |
| d = 6  | σ = 1                 | 0.136             | (0.000)         | 0.126       | (0.000)                 | 0.102             | (0.003)         | 0.096       | (0.003)         | 0.097             | (0.004)         |
|        | σ = 2                 | 0.136             | (0.000)         | 0.126       | (0.000)                 | 0.036             | (0.005)         | 0.050       | (0.004)         | 0.045             | (0.005)         |
|        | σ = 0 . 5             | 0.131             | (0.000)         | 0.121       | (0.000)                 | 0.113             | (0.002)         | 0.107       | (0.001)         | 0.106             | (0.002)         |
| d = 12 | σ = 1                 | 0.131             | (0.000)         | 0.121       | (0.000)                 | 0.091             | (0.004)         | 0.085       | (0.003)         | 0.074             | (0.005)         |
|        | σ = 2                 | 0.131             | (0.000)         | 0.121       | (0.000)                 | 0.028             | (0.005)         | 0.038       | (0.004)         | 0.032             | (0.005)         |

Figure 10: Setup D. Policy Learning. Learned policy value, averaged across 100 experiments, with standard error.

|        |             | opt         |                 | or          |                 |                             | direct                      | ips         |                 |
|--------|-------------|-------------|-----------------|-------------|-----------------|-----------------------------|-----------------------------|-------------|-----------------|
| d = 6  | σ = 0 . 5   | 0.133       | (0.000) (0.000) | 0.079       | (0.001) (0.001) | 0.039 (0.003) 0.015         | 0.033 (0.003)               | 0.045       | (0.003) (0.003) |
|        | σ = 1       | 0.133       |                 | 0.080       |                 | (0.004)                     | 0.011 (0.002)               | 0.023       |                 |
|        | σ = 2       | 0.133       | (0.000)         | 0.079       | (0.001)         | 0.009 (0.003)               | -0.001 (0.002)              | 0.011       | (0.004)         |
| d = 12 | σ = 0 . 5   | 0.133       | (0.000)         | 0.079       | (0.001)         | 0.022 (0.003)               | 0.014 (0.002)               | 0.026       | (0.003)         |
|        | σ = 1 σ = 2 | 0.133 0.133 | (0.000) (0.000) | 0.080 0.079 | (0.001) (0.001) | 0.007 (0.003) 0.005 (0.003) | 0.001 (0.001) 0.001 (0.001) | 0.013 0.005 | (0.003) (0.003) |
|        | σ = 0 . 5   | 0.133       | (0.000)         | 0.080       | (0.001)         | 0.056 (0.003)               | 0.051 (0.003)               | 0.050       | (0.003)         |
| d = 6  | σ = 1       | 0.133       | (0.000)         | 0.081       | (0.000)         | 0.032 (0.003)               | 0.017 (0.003)               | 0.019       | (0.003)         |
|        | σ = 2       | 0.133       | (0.000)         | 0.081       | (0.000)         | 0.012 (0.003)               | 0.003 (0.002)               | 0.011       | (0.003)         |
|        | σ = 0 . 5   | 0.136       | (0.000)         | 0.086       | (0.000)         | 0.051 (0.003)               | 0.039 (0.003)               | 0.050       | (0.003)         |
| d = 12 | σ = 1       | 0.136       | (0.000)         | 0.085       | (0.000)         | 0.024 (0.003)               | 0.011 (0.002)               | 0.027       | (0.003)         |
|        | σ = 2       | 0.136       | (0.000)         | 0.084       | (0.001)         | 0.008 (0.002)               | 0.006 (0.002)               | 0.009       | (0.003)         |
|        | σ = 0 . 5   | 0.135       | (0.000)         | 0.086       | (0.000)         | 0.072 (0.001)               | 0.074 (0.001)               | 0.072       | (0.001)         |
| d = 6  | σ = 1       | 0.135       | (0.000)         | 0.086       | (0.000)         | 0.050 (0.003)               | 0.051 (0.003)               | 0.051       | (0.003)         |
|        | σ = 2       | 0.135       | (0.000)         | 0.086       | (0.000)         | 0.025 (0.004)               | 0.011 (0.002)               | 0.023       | (0.004)         |
|        | σ = 0 . 5   | 0.133       | (0.000)         | 0.083       | (0.000)         | 0.067 (0.002)               | 0.067 (0.002)               | 0.062       | (0.002)         |
| d = 12 | σ = 1       | 0.133       | (0.000)         | 0.082       | (0.000)         | 0.042 (0.003)               | 0.038 (0.003)               | 0.050       | (0.003)         |
|        | σ = 2       | 0.133       | (0.000)         | 0.083       | (0.000)         | 0.008 (0.003)               | 0.005 (0.002)               | 0.013       | (0.003)         |

Figure 11: Setup E. Policy Learning. Learned policy value, averaged across 100 experiments, with standard error.

|        |                       | opt               |                 | or          |                         | dr                |                 | direct      |                         | ips               |                         |
|--------|-----------------------|-------------------|-----------------|-------------|-------------------------|-------------------|-----------------|-------------|-------------------------|-------------------|-------------------------|
| d = 6  | σ = 0 . 5 σ = 1 σ = 2 | 0.463 0.463 0.463 | (0.000) (0.000) | 0.454 0.453 | (0.001) (0.001) (0.001) | 0.392 0.243 0.080 | (0.009) (0.014) | 0.185 0.094 | (0.015) (0.012) (0.010) | 0.068 0.041 0.022 | (0.011) (0.008) (0.008) |
|        | σ = 0 . 5             | 0.460             | (0.000)         | 0.454 0.451 | (0.001)                 | 0.376             | (0.013)         | 0.054       |                         |                   |                         |
| d = 12 |                       |                   | (0.000)         |             |                         |                   | (0.010)         | 0.137       | (0.014)                 | 0.032             | (0.009)                 |
|        | σ = 1                 | 0.460             | (0.000)         | 0.451       | (0.001)                 | 0.227             | (0.016)         | 0.047       | (0.009)                 | 0.030             | (0.007)                 |
|        | σ = 2                 | 0.460             | (0.000)         | 0.453       | (0.001)                 | 0.077             | (0.012)         | 0.031       | (0.008)                 | 0.014             | (0.007)                 |
| d = 6  | σ = 0 . 5             | 0.454             | (0.000)         | 0.449       | (0.001)                 | 0.436             | (0.002)         | 0.284       | (0.014)                 | 0.099             | (0.012)                 |
|        | σ = 1 σ = 2           | 0.454 0.454       | (0.000) (0.000) | 0.449 0.449 | (0.001) (0.001)         | 0.360 0.136       | (0.010) (0.015) | 0.183 0.048 | (0.015) (0.008)         | 0.065 0.034       | (0.010) (0.007)         |
| d = 12 | σ = 0 . 5             | 0.464             | (0.000)         | 0.459       | (0.001)                 | 0.441             | (0.003)         | 0.217       | (0.015)                 |                   | (0.010)                 |
|        |                       |                   |                 |             |                         |                   |                 |             |                         | 0.067             |                         |
|        | σ = 1 σ = 2           | 0.464 0.464       | (0.000) (0.000) | 0.459 0.459 | (0.001) (0.001)         | 0.361 0.088       | (0.010) (0.011) | 0.136 0.053 | (0.013) (0.009)         | 0.038 0.020       | (0.007) (0.007)         |
|        | σ = 0 . 5             | 0.459             | (0.000)         | 0.457       | (0.000)                 | 0.452             | (0.001)         | 0.356       | (0.009)                 | 0.122             | (0.013)                 |
| d = 6  | σ = 1                 | 0.459             | (0.000)         | 0.457       | (0.000)                 | 0.439             | (0.002)         | 0.291       | (0.013)                 | 0.120             | (0.013)                 |
|        | σ = 2                 | 0.459             | (0.000)         | 0.458       | (0.000)                 | 0.350             | (0.010)         | 0.193       | (0.014)                 | 0.092             | (0.012)                 |
|        | σ = 0 . 5             | 0.453             | (0.000)         | 0.451       | (0.000)                 | 0.445             | (0.001)         | 0.347       | (0.012)                 | 0.114             | (0.013)                 |
| d = 12 | σ = 1                 | 0.453             | (0.000)         | 0.451       | (0.000)                 | 0.429             | (0.003)         | 0.259       | (0.014)                 | 0.112             | (0.013)                 |
|        | σ = 2                 | 0.453             | (0.000)         | 0.452       | (0.000)                 | 0.297             | (0.014)         | 0.102       | (0.012)                 | 0.049             | (0.009)                 |

Figure 12: Setup F. Policy Learning. Learned policy value, averaged across 100 experiments, with standard error.

|        |                 | opt         |                 | or          |                 | dr          |                 | direct      |                 | ips          |                 |
|--------|-----------------|-------------|-----------------|-------------|-----------------|-------------|-----------------|-------------|-----------------|--------------|-----------------|
| d = 6  | σ = 0 . 5 σ = 1 | 0.135 0.135 | (0.000) (0.000) | 0.124 0.124 | (0.000) (0.000) | 0.040 0.015 | (0.005) (0.004) | 0.022 0.013 | (0.004) (0.003) | 0.007 -0.000 | (0.003) (0.003) |
|        | σ = 2           | 0.135       | (0.000)         | 0.123       | (0.000)         | 0.005       | (0.003)         | 0.007       | (0.003)         | 0.001        | (0.004)         |
| d = 12 | σ = 0 . 5       | 0.135       | (0.000)         | 0.124       | (0.000)         | 0.034       | (0.005)         | 0.010       | (0.003)         | 0.002        | (0.002)         |
|        | σ = 1           | 0.135       | (0.000)         | 0.124       | (0.000)         | 0.009       | (0.003)         | 0.006       | (0.002)         | 0.003        | (0.003)         |
|        | σ = 2           | 0.135       | (0.000)         | 0.124       | (0.000)         | 0.004       | (0.003)         | 0.003       | (0.002)         | 0.003        | (0.003)         |
|        | σ = 0 . 5       | 0.135       | (0.000)         | 0.124       | (0.000)         | 0.064       | (0.004)         | 0.032       | (0.003)         | 0.011        | (0.003)         |
| d = 6  | σ = 1           | 0.135       | (0.000)         | 0.124       | (0.000)         | 0.030       | (0.004)         | 0.022       | (0.003)         | 0.007        | (0.002)         |
|        | σ = 2           | 0.135       | (0.000)         | 0.124       | (0.000)         | 0.010       | (0.004)         | 0.008       | (0.002)         | 0.003        | (0.003)         |
|        | σ = 0 . 5       | 0.136       | (0.000)         | 0.125       | (0.000)         | 0.058       | (0.005)         | 0.026       | (0.004)         | 0.009        | (0.002)         |
| d = 12 | σ = 1           | 0.136       | (0.000)         | 0.125       | (0.000)         | 0.017       | (0.003)         | 0.013       | (0.003)         | 0.002        | (0.002)         |
|        | σ = 2           | 0.136       | (0.000)         | 0.125       | (0.000)         | 0.005       | (0.002)         | 0.008       | (0.003)         | -0.003       | (0.002)         |
|        | σ = 0 . 5       | 0.136       | (0.000)         | 0.125       | (0.000)         | 0.108       | (0.002)         | 0.060       | (0.004)         | 0.016        | (0.002)         |
| d = 6  | σ = 1           | 0.136       | (0.000)         | 0.126       | (0.000)         | 0.074       | (0.005)         | 0.051       | (0.004)         | 0.015        | (0.002)         |
|        | σ = 2           | 0.136       | (0.000)         | 0.126       | (0.000)         | 0.023       | (0.004)         | 0.026       | (0.003)         | 0.004        | (0.002)         |
|        | σ = 0 . 5       | 0.131       | (0.000)         | 0.120       | (0.000)         | 0.102       | (0.003)         | 0.057       | (0.004)         | 0.012        | (0.002)         |
| d = 12 | σ = 1           | 0.131       | (0.000)         | 0.121       | (0.000)         | 0.060       | (0.005)         | 0.036       | (0.004)         | 0.007        | (0.002)         |
|        | σ = 2           | 0.131       | (0.000)         | 0.121       | (0.000)         | 0.009       | (0.003)         | 0.017       | (0.003)         | -0.000       | (0.001)         |

## Part II

## Additional Results

## B Additional Algorithms

Algorithm 1 contains pseudocode for the variance-penalized plug-in empirical risk minimization method described in Section 4.2, which is omitted from the main body due to space constraints.

Meta-Algorithm 2 and Meta-Algorithm 3 present variants of Meta-Algorithm 1 that employ crossfitting rather than sample splitting, and serve as statistical learning counterparts to the DML1 and DML2 methods described in Chernozhukov et al. (2018a). Meta-Algorithm 3 is specialized to M-estimation losses of the type considered in Section 4, in which L D ( θ, g ) = E [ ℓ ( θ ( x ) , g ( w ); z )] for a point-wise loss function ℓ .

Algorithm 1 (Plug-In ERM with Centered Second Moment Penalization) .

Input : Sample set S = z 1 , . . . , z n .

- Split S into subsets S 1 , S 2 , and S 3 of equal size.
- Let ̂ µ = inf θ ∈ Θ L S 3 ( θ, ̂ g ) .
- Let ̂ g be the output of Alg( G , S 1 ) .
- Return ̂ θ = arg min θ ∈ Θ L S 2 ( θ, ̂ g ) + 36 δ n R -1 ∥ ℓ ( θ ( · ) , ̂ g ( · ); · ) -̂ µ ∥ L 2 ( S 2 ) .

Meta-Algorithm 2 (Two-Stage Estimation with Cross-Fitting (DML1)) .

Input : Sample set S = z 1 , . . . , z n , Number of folds K .

- Let S 1 , . . . , S K be a random K -fold partition of S such that each fold S i has size n/K .
- For i = 1 , . . . , K :
- -Let ̂ g i be the output of Alg( G , S c i ) , where S c i = S \ S i .
- -Let ̂ θ i be the output of Alg(Θ , S i ; ̂ g i ) .
- Return ̂ θ = 1 K ∑ K i =1 ̂ θ i .

Meta-Algorithm 3 (Two-Stage Estimation with Cross-Fitting (DML2)) .

Input : Sample set S = z 1 , . . . , z n , Number of folds K .

- Let S 1 , . . . , S K be a random K -fold partition of S such that each fold S i has size n/K .
- For i = 1 , . . . , K :
- -Let ̂ g i be the output of Alg( G , S c i ) , where S c i = S \ S i .
- Use any algorithm Alg(Θ , S 2 ; { ̂ g i } K i =1 ) that returns ̂ θ which achieves average plug-in excess risk:

<!-- formula-not-decoded -->

We note that most the natural instantiation of the DML2 meta-algorithm is to use a second-stage algorithm that minimizes the average empirical loss across the folds. In particular, assuming the empirical loss takes the form L S ( θ, g ) := ∑ i ∈ S ℓ ( θ ( x i ) , g ( w i ); z i ), one can apply plug-in ERM to the average empirical risk across the folds

<!-- formula-not-decoded -->

The localized Rademacher complexity techniques we develop in this paper can be adapted to this method to provide average excess risk bounds of order δ n/K · ∥ ̂ θ -θ ⋆ ∥ L 2 ( ℓ 2 , D ) + δ 2 n/K , where δ n is as described in Theorem 3. This guarantee can then be combined with our main theorems to achieve oracle excess risk bounds with second order dependence. For example, the follow-up work of Dao et al. (2020) invokes such an analysis in the context of knowledge distillation. Meta-Algorithm 3 is stated in a more general form to allow for second stage algorithms that go beyond plug-in ERM (e.g., penalized ERM variants or aggregation methods).

## C Orthogonal Statistical Learning: User-Friendly Tools

Our main results, Theorem 1 and Theorem 2, give excess risk bounds for Meta-Algorithm 1 (with generic nuisance and target estimators) under Neyman orthogonality. In this section we provide some additional consequences and variants of these results which will prove useful in deriving guarantees for specific estimators.

The first result gives a consequence of Theorem 1 for the case where the target estimator satisfies a certain self-bounding property. Such is the case for plug-in empirical risk minimization.

Lemma 1. Suppose that the conditions of Theorem 1 hold, and that for all g ∈ G ,

<!-- formula-not-decoded -->

for functions ε n ( δ ) and α n ( δ ) . Then the sample splitting meta-algorithm (Meta-Algorithm 1) produces an estimate ̂ θ such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where C 1 and C 2 are defined as in Theorem 1.

Proof of Lemma 1. Theorem 1 implies that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where C 1 and C 2 are as in Theorem 1. By the AM-GM inequality, we can upper bound this by

<!-- formula-not-decoded -->

Rearranging yields (57). To prove (58), we begin by applying, Assumption 3, which gives that

<!-- formula-not-decoded -->

Applying Lemma 2, we have

<!-- formula-not-decoded -->

where C ≤ λ 2 (( β 2 λ ) 2 1+ r + κ λ ) . We further bound this by

<!-- formula-not-decoded -->

where we have applied the AM-GM inequality. It follows that

<!-- formula-not-decoded -->

where we have applied (57) and then used that λ ≤ β 1 to simplify.

The next lemma we provide is a variant of Theorem 1 which gives a bound on the first derivative of the oracle risk for Meta-Algorithm 1.

Lemma 2 (Variant of Theorem 1) . Suppose there exists θ ⋆ ∈ arg min θ ∈ Θ L D ( θ, g 0 ) such that Assumptions 1 to 4, are satisfied. Then the sample splitting meta-algorithm (Meta-Algorithm 1) produces a parameter ̂ θ such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma 2. See (16) within the proof of Theorem 1.

The next lemma (Lemma 3) uses orthogonality to provide a bound on the plug-in excess risk in terms of the target error and nuisance error. Recall that our main theorems give a bound on the oracle excess risk as a function of the plug-in excess risk. Lemma 3 is-in some sense-a converse, upper bounding the plug-in excess risk with the estimation error ∥ θ -θ 0 ∥ Θ under orthogonality. This type of guarantee can be useful as an intermediate result when analyzing the plug-in excess risk for second stage algorithms. In this paper, we use it in the context of Skeleton Aggregation (Appendix M), to control misspecification error for the pseudo-excess risk.

Lemma 3. Suppose that Assumption 1 holds, that

<!-- formula-not-decoded -->

and that the following derivative bounds hold:

- ∀ θ ∈ Θ , ¯ θ ∈ star(Θ , θ 0 ) , g ∈ G :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then for all θ ∈ Θ ,

<!-- formula-not-decoded -->

Proof of Lemma 3. Let θ be fixed. Using a second-order Taylor expansion, there exists ¯ θ ∈ star(Θ , θ 0 ) such that

<!-- formula-not-decoded -->

Using another second-order Taylor expansion, there exists ¯ g ∈ star( G , g 0 ) for which

<!-- formula-not-decoded -->

where the second equality uses the first-order condition and orthogonality. The result now follows from the assumed derivative bounds.

## D Construction of Orthogonal Losses

While orthogonal losses are already known for many problem settings and statistical models (treatment effect estimation, policy learning, regression with missing/censored data, and so on), for new problems we often begin with a loss which is not necessarily orthogonal. A natural question, which we address now, is whether one can modify the loss to satisfy orthogonality so that our main theorems can be applied.

Suppose we begin with a loss ℓ ( θ ( x ) , g ; z ) such that the nuisance and target parameter are specified by the moment equations

<!-- formula-not-decoded -->

where u ⊆ z is a random variable, x ⊆ w , and ∇ ζ denotes the derivative with respect to the first argument. If L D ( θ, g ) = E z [ ℓ ( θ ( x ) , g ( w ); z )] is not orthogonal, we can construct an orthogonal loss using a generalization of a construction in Chernozhukov et al. (2018b). For simplicity, we sketch the approach for the special case where θ 0 is scalar-valued.

To begin, assume that there exists a function a 0 such that for all x ∈ X , we have

<!-- formula-not-decoded -->

Under this assumption, we can expand our nuisance parameters to include a 0 -that is, define ˜ g 0 := { g 0 , a 0 } -and construct a new orthogonal loss:

<!-- formula-not-decoded -->

Letting ˜ L D ( θ, ˜ g ) = E [ ˜ ℓ ( θ ( x ) , ˜ g ; z ) ] be the new population risk, we have the following claim.

Lemma 4. The population risk ˜ L D ( θ, ˜ g ) satisfies Assumption 1 and Assumption 2.

Proof of Lemma 4. The first-order condition (Assumption 2) follows immediately by using that E [ ∇ ζ ℓ ( θ 0 ( x ) , g 0 ; z ) | x ] = 0, and by the assumption that E [ u | w ] = g 0 ( w ). To establish orthogonality with respect to the nuisance parameter g , we compute

<!-- formula-not-decoded -->

where the second equality follows from the definition of a 0 and the final inequality is the law of total expectation. For orthogonality with respect to a , we have

<!-- formula-not-decoded -->

θ , where the final inequality uses that x ⊆ w and E [ u | w ] = g 0 ( w ).

As a first example, in the special case where the loss depends on g 0 only through its evaluation at w (i.e., (61) simplifies to E [ ∇ ζ ℓ ( θ 0 ( x ) , g 0 ( w ); z ) | x ] = 0), then we can take

<!-- formula-not-decoded -->

Of course, to make use of the lemma, we must be able to estimate the new nuisance parameter a 0 . This can be accomplished through an additional plug-in estimation step based on sample splitting: Split S into folds S 1 , S 2 , S 3 , and S 4 of equal size. Estimate ̂ g on S 1 , then obtain an initial estimate ̂ θ init for θ 0 by solving arg min θ ∈ Θ L S 2 ( θ, ̂ g ), where L S 2 denotes the empirical loss over S 2 . Next, use the initial estimator to compute a plug-in estimator ̂ a for a 0 by regressing onto the 'targets' ∇ ζ ∇ γ ℓ ( ̂ θ init ( x ) , ̂ g ( w ); z ) on S 3 ; this requires access to an additional function class A containing a 0 , and the resulting guarantee will have (second-order) dependence on the complexity of this class. Finally, produce the main estimator for the target parameter by solving ̂ θ = arg min θ ∈ Θ ˜ L S 4 ( θ, { ̂ g, ̂ a } ). A full description is given in Algorithm 2.

Algorithm 2 (Plug-In ERM with Automatic Debiasing) .

Input : Sample set S = z 1 , . . . , z n , Function class A containing a 0 (cf. (62))

- Split S into subsets S 1 , S 2 , S 3 , and S 4 of equal size.
- Let ̂ θ init = arg min θ ∈ Θ L S 2 ( θ, ̂ g ) , where L S 2 denotes the empirical loss over S 2 .
- Let ̂ g be the output of Alg( G , S 1 ) .
- Estimate a 0 via ̂ a = arg min a ∈A 1 | S 3 | ∑ z ∈ S 3 ∥ ∥ a ( w ) -∇ ζ ∇ γ ℓ ( ̂ θ init ( x ) , ̂ g ( w ); z ) ∥ ∥ 2 2 .
- Return ̂ θ = arg min θ ∈ Θ ˜ L S 4 ( θ, { ̂ g, ̂ a } ) , where ˜ ℓ is defined as in (63).

The key idea behind this scheme is that the initial estimator ̂ θ init will not be able to take advantage of orthogonality, but its estimation error will only enter the final bound through the error of ̂ a , and thus will only have higher-order impact on the rate. We omit details, which are beyond the scope of the present paper.

This approach is applicable for the problem of estimating utility functions in models of strategic competition, as used in Chernozhukov et al. (2018b); see Appendix D.1 for a detailed example. For some models-including utility function estimationa 0 is a known function of θ 0 and g 0 , so that the extra regression to estimate ̂ a given the initial estimators is not required.

A more general setting where the loss has the form in (62) is as follows. Suppose that all functions g ∈ G are conditionally square-integrable in the sense that for all x , E [ g 2 ( w ) | x ] &lt; ∞ , and suppose there exist functions β 0 , T x such that we can write

<!-- formula-not-decoded -->

where T x ( g ) is a linear operator on g with uniformly bounded operator norm:

̸

<!-- formula-not-decoded -->

By the Riesz-Frechet representation theorem, we can express the operator T x as

<!-- formula-not-decoded -->

where we have used that x ⊆ w to simplify. Hence, we have

<!-- formula-not-decoded -->

so that (62) is satisfied for a 0 induced by the family of Riesz representers for operators T x . This is a variant of the Riesz representer approach presented in Chernozhukov et al. (2022b). In Appendix D.1, we show that this construction recovers the treatment effect estimation example presented in the introduction.

We mention in passing that that another, perhaps more standard, approach to constructing an orthogonal loss is to derive the influence function for the risk function L D using standard calculations from semiparametric theory (van der Laan and Robins, 2003; Tsiatis, 2007; Kosorok, 2008; Kennedy, 2016).

## D.1 Orthogonal Loss Construction: Examples

We now walk through concrete examples of the orthogonal loss construction approach outlined in Appendix D. Both examples use the loss structure in (61).

Estimating utility functions in models of strategic competition. In this setting, we have

<!-- formula-not-decoded -->

where L is the logistic function. The motivation of this problem stems from estimating games of incomplete information, where y is the entry decision of one player, u is the entry decision of the opponent, x is a featurized state of the world, ψ ( x ) is the non-strategic part of the utility of the player and ∆ g is the competitive part of the utility, i.e. the effect of the opponent's entry decision on the player's utility. For this setting, we can take the auxiliary nuisance variable to be

<!-- formula-not-decoded -->

Thus, a 0 ( w ) is a known function of θ 0 and g 0 , and to estimate a 0 it suffices to construct preliminary estimates for θ 0 and g 0 (e.g., using plug-in estimation with the original non-orthogonal loss). We can simplify the final orthogonal loss based our generic construction to

<!-- formula-not-decoded -->

where ˜ g = { g, ˜ θ } and ˜ θ = ( ˜ ψ ( x ) ˜ ∆ ) is a preliminary estimate for θ 0 .

Treatment effect estimation. In this setting, we denote the treatment as d ∈ { 0 , 1 } and w = ( d, x, v ), for some vector of extra control variables v . In this setting, we take z = ( x, w, y ) and w = ( x, v, d ), where v is a vector of extra control variables and d ∈ { 0 , 1 } is a binary treatment. We begin from the moment equation

<!-- formula-not-decoded -->

with g 0 identified by the local moment equation E [ y -g 0 ( x, v, d ) | x, v, d ] = 0. In this case, the Riesz representer takes the form a 0 ( w ) = d -(1 -d ) Pr[ D = d | x,v ] , and the resulting orthogonal loss created using the generic construction takes the form

<!-- formula-not-decoded -->

where we recall ˜ g = { g, a } . Minimizing this over θ is equivalent to minimizing the loss

<!-- formula-not-decoded -->

which is precisely the loss presented in the introduction of the paper.

## E Sufficient Conditions for Theorems and : Single Index Losses

Our setup and main results in Section 3 (Theorem 1 and Theorem 2) are stated at a high level of generality, with abstract assumptions on the structure of the population risk. In this section of the appendix we provide conditions under which these assumptions follow from concrete structural assumptions on the risk. We give sufficient guarantees for general families of loss functions. In particular, the conditions we give here suffice to derive guarantees for the applications considered in Appendix G.

## E.1 Fast Rates

In this section we give a broad class of losses under which the conditions for fast rates in Section 3.1 are satisfied. The population loss L D is defined as the expectation of a point-wise loss ℓ ( ζ, γ ; z ) acting on the predictions of the nuisance and target parameters. We assume existence of functions Φ and Λ such that the loss has the structure

<!-- formula-not-decoded -->

so that

<!-- formula-not-decoded -->

Here we recall from Section 2 that x , w are subsets of the data z , and let v ⊆ z be an auxiliary subset of the data. We also assume existence of functions ϕ ( ζ ) and Γ( γ, z ) such that the partial derivative of Φ may be written as

<!-- formula-not-decoded -->

where ϕ is non-decreasing and Lipschitz. A simple example is square loss regression, where Φ( t, γ, z ) = 1 2 ( t -Γ( γ, z )) 2 .

To provide fast rates of the type in Section 3 for losses with this structure, we let p ≥ 1 be given, define 1 q = 1 -1 p , and consider the norms

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Taking p large leads to a more stringent notion of distance for the nuisance parameter, but weakens the regularity conditions we consider, which depend on the dual parameter q . In particular, we make the following assumptions.

and

Assumption 9 (Sufficient Conditions for Fast Rates: Single Index Losses) . The loss ℓ satisfies the following conditions for parameters ( µ si , T si , τ si , L si , λ si , r si , R si ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let us highlight the condition (81), which involves the dual exponent q . Taking p large strengthens the notion of distance for the nuisance parameter, but weakens the condition (81). For example, with r si = 0, when we take p = 2, (81) becomes an average-case eigenvalue-type condition, but we require L 4 distance on the nuisance parameter. On the other hand, with p = 1, (81) essentially requires that parameter error ( L ∞ ) and L 2 error are equivalent for the target, but leads to L 2 distance for the nuisance parameter, which is more permissive than L 4 . As a concrete example, consider the logistic loss, where Φ( t, γ, z ) = y · log( σ log ( t )) + (1 -y ) · log(1 -σ log ( t )), where the target class y ∈ { 0 , 1 } is a subset of the data z and σ log ( t ) := 1 / (1 + e -t ) is the logistic function, so that ∂ ∂t Φ( t, γ, z ) = σ log ( t ) -y . Observe that the gradient of the loss with respect to the target index value can be written as

<!-- formula-not-decoded -->

Moreover, whenever the arguments to the loss are bounded, the Hessian can be bounded above and below via

<!-- formula-not-decoded -->

and thus, when p = 2, the ratio condition in (81) is implied by a minimum eigenvalue assumption on the conditional covariance matrix E [ Λ( g 0 ( w ) , v )Λ( g 0 ( w ) , v ) ⊤ | x ] .

We now show that these conditions are sufficient to satisfy the assumptions of Theorem 1, and thus guarantee higher order impact from the nuisance parameters. Our main result is as follows.

Lemma 5. If Assumption 9 holds, then the Assumptions 1 to 4 required by Theorem 1 are satisfied with constants r = r si , λ = τ si 4 , κ = 8 τ si ( L 4 si R 2 si λ -1 si ) 1 1+ r si , β 1 = T si and β 2 = µ si √ K 2 √ λ si , and with respect to the norms ∥·∥ Θ and ∥·∥ G defined in Eqs. (75) and (76) .

Combining this lemma with the guarantee from Theorem 1 directly yields an oracle excess risk bound.

Corollary 1. Suppose that there is some θ ⋆ ∈ arg min θ ∈G L D ( θ, g 0 ) such that Assumption 9 is satisfied. The sample splitting meta-algorithm Meta-Algorithm 1 produces a predictor ̂ θ that guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where C 1 = 16 τ si and C 2 = 2 (( 16 µ 2 si K 2 τ 2 si λ si ) 1 1+ r si +32 ( L 4 si R 2 si λ si ) 1 1+ r si ) . Furthermore, with ∥·∥ Θ as in Lemma 5, the following prediction error guarantee is satisfied with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Observe that for both of the bounds in Corollary 1, the only problem-dependent parameters that multiply the target estimation rate Rate D (Θ , · · · ) are T si and τ -1 si . Importantly, this implies that if the more restrictive parameters λ si , L si , µ si , R si , K 2 and so forth are held constant as n grows, they have negligible impact asymptotically, so long as the nuisance parameter can be estimated quickly enough. For the square loss this is particularly desirable, since T si = τ si = 1.

En route to proving Lemma 7, we also prove the following, slightly stronger, smoothness bound, which is used for several results.

Lemma 6. Suppose Assumption 9 holds with r si = 0 . Then for all functions θ ∈ ̂ Θ , ¯ θ ∈ star( ̂ Θ , θ ⋆ ) , and g ∈ G ,

<!-- formula-not-decoded -->

Discussion: Estimation for the first stage. When we work with L 2 error for the target parameter (that is, p = 2), Corollary 1 (with r si = 0) provides guarantees in terms of the L 4 estimation rate for the nuisance parameters, i.e.

<!-- formula-not-decoded -->

Since L 4 error rates are somewhat less common than L 2 (i.e., square loss) estimation rates, let us briefly discuss conditions under which out-of-the box algorithms can be used to give guarantees on the L 4 error.

First, for many nonparametric classes of interest, minimax L p error rates have been characterized and can be applied directly. This includes smooth classes (Stone, 1980, 1982), H¨ older classes (Lepski, 1992; Kerkyacharian et al., 2001, 2008), Besov classes (Delyon and Juditsky, 1996), Sobolev classes (Tsybakov, 1998), and convex regression (Guntuboyina and Sen, 2015)

Second, whenever the G is a linear class or more broadly a parametric class, classical statistical theory (Lehmann and Casella, 2006) guarantees parameter recovery. Up to problem-dependent constants, this implies a bound on the L 4 error as soon as the fourth moment is bounded. This approach also extends to the high-dimesional setting (Hastie et al., 2015).

Last, if the class G has well-behaved moments in the sense that ∥ g -g 0 ∥ L 4 ( ℓ 2 , D ) ≤ C ∥ g -g 0 ∥ L 2 ( ℓ 2 , D ) for all g ∈ G , we can directly appeal to square loss regression algorithms for the first stage; this

is the approach taken in Section 5. This condition is related to the so-called 'subgaussian class' assumption, and both have been explored in recent works (Lecu´ e and Mendelson, 2016; Mendelson, 2014; Liang et al., 2015).

Of course, whenever L ∞ guarantees are available for the target class itself (e.g., for parametric models), we can instead take p = 1, which permits the use of the more standard L 2 distance for the nuisance parameter.

## E.2 Slow Rates

In the single-index setup, assumptions much weaker than Assumption 9 suffice to obtain slow rates via Theorem 2. In particular, the following conditions are sufficient.

Assumption 10 (Sufficient Slow Rate Conditions for Single Index Losses) .

<!-- formula-not-decoded -->

Compared to Assumption 9, the most important difference is that since we require universal orthogonality, the first condition is required to hold for all θ , not just at θ 0 . Assumption 10 has the following immediate consequences.

Lemma 7. If Assumption 10 holds, then Assumption 5 is satisfied and Assumption 6 is satisfied with constant β = β si and with respect to ∥·∥ G = ∥·∥ L 2 ( ℓ 2 , D ) .

Corollary 2. Suppose Assumption 10 holds. Then with probability at least 1 -δ , the target predictor ̂ θ produced by Meta-Algorithm 1 enjoys the excess risk bound

<!-- formula-not-decoded -->

## E.3 Proofs

Proof of Lemma 5. We prove that the assumptions required by Theorem 1 are implied by our conditions one by one.

Assumption 1. From the definition of the directional derivative, the law of iterated expectations, and the fact that X ⊆ W , we have

<!-- formula-not-decoded -->

Assumption 2. This likewise follows immediately by expanding the directional derivative and applying the law of iterated expectation:

<!-- formula-not-decoded -->

We now argue about the remaining assumptions. We will repeatedly invoke the following expression for the second derivative of the population risk.

<!-- formula-not-decoded -->

Let us introduce some additional notation. Let g ∈ G and θ ∈ Θ be the free variables in the statements of Assumption 3 and Assumption 4. We define the following vector- and matrix-valued random variables:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To prove the lemma, it suffices to verify that Assumption 4 and Assumption 3 hold for the norms ∥ θ -θ ′ ∥ 2 Θ = E [ ⟨ W 0 , θ ( x ) -θ ′ ( x ) ⟩ 2 ] and ∥ g -g ′ ∥ G = ∥ g -g ′ ∥ L 2 p ( ℓ 2 , D ) .

Assumption 3(a). Using (84) and (86) we have:

<!-- formula-not-decoded -->

since ∥·∥ L 2 ( D ) ≤ ∥·∥ L p ( D ) for all p ≥ 2. It follows that Assumption 3(a) is satisfied with β 1 = T si .

Assumption 3(b). Define random vectors X 0 = θ ⋆ ( x ), X n = θ ( x ), V 0 = g 0 ( w ), V n = g ( w ) and Σ( w ) = E [ ∇ 2 γγ ∇ ζ i ℓ ( θ ⋆ ( x ) , ¯ g ( w ) , z ) | w ] . Then, invoking the assumed structure for the loss function, we have

<!-- formula-not-decoded -->

Using that x ⊆ w , we have

<!-- formula-not-decoded -->

All that remains is to relate these norms to the norms appearing in the lemma statement.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we have so that Assumption 3(b) is satisfied with β 2 = µ si K 2 √ λ si .

Assumption 4. By (84) and (86), we have

<!-- formula-not-decoded -->

Using H¨ older's inequality, we have that

<!-- formula-not-decoded -->

Hence, by Young's inequality, we have that for any η &gt; 0

<!-- formula-not-decoded -->

Choosing η = (4 L 2 si R si λ -1 / 2 si ) -1 yields

<!-- formula-not-decoded -->

Thus, Assumption 4 is satisfied with λ = τ si 4 and κ = 8 τ si ( L 2 si R si λ -1 / 2 si ) 2 1+ r si .

Proof of Lemma 6. We adopt the same notation as in the proof of Lemma 5. This proof is almost the same as the proof that Assumption 4 is satisfied in that lemma.

<!-- formula-not-decoded -->

Using the AM-GM inequality, this implies the inequality

<!-- formula-not-decoded -->

Proof of Lemma 7. Immediate.

## F Sufficient Conditions for Oracle Rates: Further Results

The results in Section 5 provide sufficient conditions for fast oracle rates under the assumption of strong convexity and a well-specified model, which is not satisfied for all losses used in practice. For example, linear losses used in policy learning (cf. Section 3.4) do not satisfy strong convexity property. In this section we extend the sufficient conditions for oracle rates given in Section 5. First, we give guarantees for strongly convex losses in the presence of misspecification, and then give guarantees for any (potentially non-strongly convex) loss that is Lipschitz in the target prediction θ ( x ). We follow the same notation as in Section 5.

## F.1 Oracle Rates for Square Losses with Misspecification

Here we consider strongly convex losses with the same structure as in Section 5, but consider the case in which the target parameter is misspecified . This setting has been relatively unexplored in recent results on double machine learning (Chernozhukov et al., 2018a; Mackey et al., 2018; Chernozhukov et al., 2018b); this is perhaps not surprising since for many settings, assuming a well-specified model is critical to establish orthogonality. However, for certain settings including treatment effect estimation (Section 3.3), orthogonality can indeed hold even without model correctness. The following theorem shows that we can obtain oracle rates in the presence of both misspecification and nuisance parameters as long as the nuisance class has moderate complexity.

Theorem 6 (Oracle Rates, Misspecified Case) . Suppose that the target class Θ is convex. Suppose that Assumptions 1 and 2 and Assumptions 7 and 8 hold. If the relationship

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 13: Relationship between first and second stage for oracle rates; misspecified case.

![Image](data:image/png;base64,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)

holds, then for appropriate choice of sub-algorithms, the sample splitting meta-algorithm MetaAlgorithm 1 produces a predictor ̂ θ such that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

thereby matching the minimax rate in the absence of nuisance parameters. In particular, it suffices to use Skeleton Aggregation for the first stage and plug-in ERM for the second stage.

Like the previous result, Theorem 6 is proven by combining the main theorem (Theorem 1) with algorithm-specific upper bounds on Rate D (Θ , · · · ) and Rate D ( G , · · · ). Figure 13 summarizes sufficient conditions under which Theorem 6 matches the oracle rate ˜ Θ( n -2 2+ p 2 ∧ 1 p 2 ) (Rakhlin et al., 2017). Comparing to the well-specified case (Theorem 5/Figure 1), we see that in the misspecified case, the condition on the metric entropy for the nuisance parameter is significantly more permissive when the target parameter class Θ is large (i.e. p 2 &gt; 2). For example, if p 2 = 5 then we require p 1 &lt; 12 for oracle rates in the well-specified case, but only require p 1 &lt; 18 in the misspecified case. On the other hand, when p 2 ≤ 2 the conditions on the nuisance metric entropy match the well-specified case. In particular, whenever Θ is a parametric class it again suffices to take p 1 &lt; 2 so that the first stage has an n -1 / 4 -rate.

## F.2 Oracle Rates for Generic Lipschitz Losses

For arbitrary Lipschitz losses (in particular, for the linear loss), the optimal rate in the absence of nuisance parameters is ˜ O ( n -1 2 ∧ 1 p 2 ) under the metric entropy growth assumed in Section 5 (cf. Section 12.8/12.9 of Rakhlin and Sridharan (2012)). Our main theorem for this section shows that this rate is still obtained in the presence of nuisance parameters when the nuisance metric entropy parameter p 1 is not too much larger than p 2 . We make the following regularity assumption on the loss.

Assumption 11. The loss ℓ has absolute value bounded by 1 , the mapping ζ ↦→ ℓ ( ζ, γ ; z ) is O (1) -Lipschitz with respect to ℓ 2 , and the mapping γ ↦→ ℓ ( ζ, γ ; z ) has O (1) -Lipschitz gradients with respect to ℓ 2 . Assumption 5 holds for all g ∈ G when p 1 ≤ 2 , and for all g ∈ G +star( G - G , 0) if p 1 &gt; 2 .

Compared to the oracle rates for square losses, the assumptions required for our main Lipschitz loss theorem are relaxed significantly: We no longer require strong convexity, nor do we require any

Figure 14: Relationship between first and second stage for oracle rates; general loss case.

![Image](data:image/png;base64,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)

type of moment comparison for the nuisance class. On the other hand, we do require the additional universal orthogonality condition from Section 3.2.

Theorem 7. Suppose that Assumption 11 is satisfied. If the relationship

<!-- formula-not-decoded -->

holds, then for appropriate choice of sub-algorithms, the sample splitting meta-algorithm produces a predictor ̂ θ that guarantees that, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

thereby matching the minimax rate in the absence of nuisance parameters.

Theorem 7 is proven as a corollary of Theorem 2, by proving upper bounds on Rate D (Θ , . . . ) and Rate D ( G , · · · ) for plug-in empirical risk minimization and skeleton aggregation, respectively; see Appendix M for the proof. The theorem is summarized in Figure 14. In particular, suppose that our aim is to match the minimax optimal rate in the absence of nuisance parameters, which is ˜ O ( n -1 2 ∧ 1 p 2 ) (Rakhlin and Sridharan, 2012). Whenever Θ is a parametric class (i.e. H 2 (Θ , ε, n ) ∝ d 2 log(1 /ε )), it suffices to take p 1 &lt; 2, as in the well-specified and misspecified square loss setups in Section 5, and as in standard semiparametric results (Newey, 1994; van der Vaart, 2000; Robins et al., 2008; Zheng and van der Laan, 2010; Chernozhukov et al., 2018a). That the condition matches is somewhat interesting given that the final rate in this case is slower than in the square loss setup. Note however that we cannot tolerate p 1 &gt; 2 until p 2 &gt; 2, as compared to our results strongly convex losses, where the admissible value of p 1 is growing for all values of p 2 . This result generalizes the condition given in Athey and Wager (2017) for the specific case of policy learning, which applies only in the parametric setting, and only for a specific loss.

## G Additional Applications

In this section of the appendix we show how three additional families of applications-offline policy learning/optimization, domain adaptation/sample bias correction, and learning with missing data-fall into our general orthogonal statistical learning framework. We also sketch some statistical consequences based on our general algorithmic tools, and show how these results generalize and extend previous work.

## G.1 Policy Learning: Further Examples

In this section we show how to view some additional policy learning models in our framework. As in Section 3.4, we consider the data-generating process in (24), but here we go beyond binary treatments.

Multiple finite treatments. The binary setting above can easily be extended to the case of N possible treatments, analyzed in Zhou et al. (2023). Formally, let T ∈ { ⃗ e 1 , . . . , ⃗ e N } , where ⃗ e i ∈ { 0 , 1 } N is the i -th standard basis vector. We still follow the data generating process (24), but now e 0 : X → ∆ N and f 0 : { 0 , 1 } N ×X → R . To simplify notation, let p 0 ( t, x ) = Pr[ T = t | X = x ] so that p 0 ( t, x ) = e 0 ( x ) t . Then the following loss function is an unbiased estimate of the counterfactual loss:

<!-- formula-not-decoded -->

This formulation leads to the standard extension of the doubly-robust estimator to multiple outcomes Dud´ ık et al. (2011); Zhou et al. (2023). Define an N -dimensional vector-valued function β ( f 0 , e 0 , Z ) to have the t -th coordinate is equal to ℓ ( t, Z ; f 0 , e 0 ). Then, as in the binary case, we can equivalently optimize a population risk that is linear in the target parameter:

<!-- formula-not-decoded -->

This population risk is easily shown to satisfy universal orthogonality.

Counterfactual risk minimization and general continuous treatments. Counterfactual risk minimization (CRM) is a learning framework introduced by Swaminathan and Joachims (2015a). It is mathematically equivalent to the policy learning setup with arbitrary treatment and outcome spaces, but is motivated by a different set real-world learning scenarios and was developed in a parallel line of research in the machine learning literature. To highlight the relationship with policy learning and the applicability of our results to this setting we will present the CRM framework using the notation of policy learning.

In counterfactual risk minimization we receive data Z = ( Y, T, X ) from the policy learning data generating process (24). The goal is to choose a hypothesis θ : X → ∆( T ) (i.e., the policy takes as input covariates and returns a distribution over treatments) that minimizes the population risk:

<!-- formula-not-decoded -->

As in policy learning, we construct an unbiased estimate of this counterfactual loss via inverse propensity scoring. Let p 0 ( t, X ) denote the probability density of treatment t conditional on covariates X and (overloading notation) let θ ( t, x ) denote the density that hypothesis θ assigns to treatment t . Then we can formulate a new risk function that provides an unbiased estimate of the target risk (98):

<!-- formula-not-decoded -->

In the CRM framework the propensity p 0 is assumed to be known (Swaminathan and Joachims, 2015a,b). When the propensity is not known, we can treat it as a nuisance parameter to be estimated from data. However, the loss (99) is not orthogonal to p 0 . We can orthogonalize the population

risk by also constructing an estimate of f 0 (see (24)) by regressing Y on ( T, X ). This leads to an analogue of the doubly robust formulation from the finite treatment setup:

<!-- formula-not-decoded -->

For finite treatments this formulation is mathematically equivalent to the population risk for multiple finite treatments presented in the prequel.

For continuous treatments, the empirical version of the problem (100) may be ill-posed, even if we assume that the propensity p 0 has density ower bounded by some constant (the analogue of the overlap condition). Swaminathan and Joachims (2015a) proposed to regularize the empirical risk via variance penalization. A similar variance penalization approach is also proposed in recent work of Bertsimas and McCord (2018), who consider policy learning over arbitrary treatment spaces. The variance-penalized empirical risk minimization algorithm-in the context of Meta-Algorithm 1-can be seen as a second stage algorithm that achieves a rate whose leading term scales with the variance of the optimal policy rather than some worst-case upper bound on the risk. Hence, it can be used in our framework to achieve variance-dependent excess risk bounds.

Kallus and Zhou (2018) develop alternative algorithms for policy learning with continuous treatments via a kernel smoothing approach. This approach is equivalent to adding noise to a deterministic hypothesis space Π, e.g. θ ( x ) = π ( x ) + ζ for each π ∈ Π, where ζ ∼ N (0 , σ 2 ). In our framework Θ is the space of density functions θ ( t, x ) induced by this construction. The value of a deterministic policy π ∈ Π (or equivalently the value of its corresponding density θ ∈ Θ) is equal to

<!-- formula-not-decoded -->

where K σ is the pdf of a normal distribution with standard deviation σ . This is equivalent to the formulation in Kallus and Zhou (2018), since the empirical version of this risk is the kernel-weighted loss:

<!-- formula-not-decoded -->

though we note that Kallus and Zhou (2018) do not restrict themselves to only the gaussian kernel.

This idea falls into our framework by simply defining Θ to be this space of randomly perturbed policy functions. The resulting analysis in our framework is slightly different than that of Kallus and Zhou (2018), where kernel weighting is invoked to show consistency of the empirical risk, and subsequently optimization of the empirical risk over deterministic policies is analyzed. With our framework, we directly calculate the regret with respect to randomized policies by applying our general theorem. This implies that we enjoy robustness to errors in estimating f 0 and p 0 .

Observe that the rate for the second stage will depend on the amount of randomization σ , since the variance of the empirical risk is governed by σ . Consequently, if one is interested in regret against deterministic strategies, we can invoke Lipschitzness of the reward function f 0 to control the regret added by the extra randomness we are injecting, which would typically be of order σ . We can then choose an optimal σ as a function of the number of samples to trade-off the bias and variance of the regret. If we wish to further optimize dependence on problem-dependent parameters in the resulting rates one can use variance penalization in the kernel-based framework to achieve a regret rate whose leading term scales with the variance of the optimal policy.

## G.2 Domain Adaptation and Sample Bias Correction

Domain adaptation is a widely studied topic in machine learning (Daume III and Marcu, 2006; Jiang and Zhai, 2007; Ben-David et al., 2007; Blitzer et al., 2008; Mansour et al., 2009). The goal is to choose a hypothesis that minimizes a given loss in expectation over a target data distribution, where the target distribution may be different from the distribution of data that is already collected.

We consider a particular instance of domain adaptation called covariate shift, encountered in supervised learning (Shimodaira, 2000). We assume that we have data Z = ( X,Y ), where X are co-variates drawn from some distribution D s with density p s and Y are labels, drawn from some distribution D x conditional on x . Our goal is to choose a hypothesis θ from some hypothesis space Θ, so as to minimize a loss function ℓ ( θ ( x ) , y ) in expectation over a different distribution of co-variates D t with density p t . Both of the densities are unknown, and we solve this issue in the orthogonal statistical learning framework by treating their ratio as a nuisance parameter for an importance-weighted loss function. Let f 0 ( x ) = p t ( x ) p s ( x ) and g 0 = { f 0 } , so that

<!-- formula-not-decoded -->

We assume the hypothesis space satisfies a realizability condition, i.e. there exists θ 0 ∈ Θ such that:

<!-- formula-not-decoded -->

where ∇ ζ corresponds to the gradient with respect to the first input of ℓ . For instance, for the case of the square loss ℓ ( ζ, y ) = ( ζ -y ) 2 , then this condition has the natural interpretation that there exists θ 0 ∈ Θ such that:

<!-- formula-not-decoded -->

Observe that when we treat the density ratio f 0 ( x ) as a nuisance function, the loss function L D is orthogonal. Indeed,

<!-- formula-not-decoded -->

Also, note that this setup fits in to the single index structure from Appendix E by writing L D as the expectation of a new loss ˜ ℓ ( θ ( x ) , g ( w ) , z ) := ℓ ( θ ( x ) , y ) · f ( x ). Focusing on the square loss ℓ ( ζ, y ) = ( ζ -y ) 2 for concreteness, it is simple to show that all of the conditions of Assumption 9 are satisfied with r = 0 as long as we have g ( x ) ≥ η &gt; 0 for all g ∈ G . Corollary 1 then implies that with probability at least 1 -δ , Meta-Algorithm 1 enjoys the bound

<!-- formula-not-decoded -->

Note that whenever Rate D ( G , S 1 , δ/ 2) = o ( Rate D (Θ , S 2 , δ/ 2; ̂ θ, ̂ g ) 1 / 4 ) the dependence on η -1 is negligible asymptotically. Of course, it is also important to develop algorithms for which the rate of the target class does not depend on η -1 . As one example, we can employ the variance-penalized ERM guarantee from Theorem 8. When Θ has VC dimension d , and the variance of the loss at ( θ 0 , g 0 ) and the capacity function τ 0 are bounded, this gives Rate D ( G , S 1 , δ/ 2) = O ( √ d/n ), with η -1 entering only lower-order terms. The final result is that if Rate D ( G , S 1 , δ/ 2) = o ( n -1 / 8 ), we get an excess risk bound for which the dominant term is O ( √ d/n ), with no dependence on η -1 .

Related work. Cortes et al. (2010) gave generalization error guarantees for the important weighted loss (103) in the case where the densities p s and p t are known. At the other extreme, Ben-David and Urner (2012) showed strong impossibility results in the regime where the densities are unknown. Our results lie in the middle, and show that learning with unknown densities is possible in the regime where the weights belong to a nonparametric class that is not much more complex than the target predictor class Θ. We remark in passing that algorithms based on discrepancy minimization Ben-David et al. (2007); Mansour et al. (2009) offer another approach to domain adaptation that does not require importance weights, but these results are not directly comparable to our own.

## G.3 Missing Data

As a final application, we apply our tools to the well-studied problem of regression with missing/censored data (Robins and Rotnitzky, 1995; van der Laan and Robins, 2003; Rubin and van der Laan, 2005; Tsiatis, 2007; Wang et al., 2010; Graham, 2011). In this setting we receive data is generated through standard regression model, but label/target variables are sometimes 'missing' or unobserved. The learner observes whether or not the target is missing for each example, and the conditional probability that the target is missing is treated as an unknown nuisance parameter. As usual, the target is the unknown regression function.

To proceed, we formalize the setting through the following data-generating process for the observed variables ( X,W,T, ˜ Y ):

<!-- formula-not-decoded -->

Here T ∈ { 0 , 1 } is an auxiliary variable (observed by the learner) that indicates whether the target variable is missing, and e 0 : X → [0 , 1] is the unknown propensity for T . The parameter θ 0 : X → R is the true regression function. We make the standard unobserved confounders assumption that X ⊆ W and T ⊥ Y | W . We define h 0 ( w ) = -2 ( E [ Y | W = w ] -θ 0 ( x )) e 0 ( w ) , take g 0 = { h 0 , e 0 } , and use the loss

<!-- formula-not-decoded -->

Observe that this loss has the property that

<!-- formula-not-decoded -->

so that the excess risk relative to θ 0 precisely corresponds to prediction accuracy whenever the true nuisance parameter is plugged in.

Proposition 4. This model satisfies Assumption 1 and Assumption 2 whenever θ 0 ∈ Θ , i.e. we have D θ L D ( θ 0 , { h, e 0 } )[ θ -θ 0 ] = 0 , D e D θ L D ( θ 0 , { h 0 , e 0 } )[ θ -θ 0 , e -e 0 ] = 0 , and D h D θ L D ( θ 0 , { h 0 , e 0 } )[ θ -θ 0 , h -h 0 ] = 0 .

̸

Note that the extra nuisance parameter h is only required here because we consider the general setting in which W = X . Whenever W = X this is unnecessary (and indeed h 0 = 0). This parameter can generally be estimated at a rate no worse than the rate for e 0 and θ 0 (absent nuisance parameters); see Chernozhukov et al. (2018b) for discussion.

As to rates and algorithms, the situation here is essentially the same as that of the domain adaptation example, so we discuss it only briefly. The setup has the single index structure from Appendix E,

and all of the sufficient conditions for fast rates from Assumption 9 are satisfied with r = 0 as long as we have e ( W ) ≥ η &gt; 0 for all e in the nuisance class. Thus, with probability at least 1 -δ , Meta-Algorithm 1 enjoys the bound

<!-- formula-not-decoded -->

As with the previous example, the variance-penalized ERM guarantees from Section 4 can be applied here to provide bounds on Rate D (Θ , . . . ) for which the dominant term in the excess risk does not scale with the inverse propensity range.

Proof of Proposition 4. We first show that the gradient vanishes in the sense of Assumption 2 when evaluated at θ 0 . In particular, for any choice of h we have

<!-- formula-not-decoded -->

Using that E [ T | W ] = e 0 ( W ) and X ⊆ W :

<!-- formula-not-decoded -->

Using that T ∈ { 0 , 1 } :

<!-- formula-not-decoded -->

Using that T ⊥ Y | W and that X ⊆ W :

<!-- formula-not-decoded -->

Using that E [ Y | X ] = θ 0 ( X ):

= 0

.

To establish orthogonality with respect to e , we have

<!-- formula-not-decoded -->

Using that X ⊆ W :

<!-- formula-not-decoded -->

The result follows immediately, using that h 0 ( w ) = -2 ( E [ Y | W = w ] -θ 0 ( x )) e 0 ( w ) .

= 0 .

To establish orthogonality with respect to h , we have

<!-- formula-not-decoded -->

Using that E [ ε 2 | W ] = 0 and X ⊆ W , the expression above is seen to be equal to zero.

## H Plug-in Empirical Risk Minimization: Further Results

## H.1 Plug-in Empirical Risk Minimization: Examples

In this section of the appendix, we instantiate the general plug-in ERM framework from Section 4 to give concrete guarantees for some concrete classes of interest. In all examples we use ˜ O to hide dependence on problem-dependent constants, log n factors, and log( δ -1 ) factors.

High-dimensional linear classes. For our first set of examples, we focus on high-dimensional linear predictors. Chernozhukov et al. (2018b) gave orthogonal/debiased estimation guarantees for high-dimensional predictors using Lasso-type algorithms. Our first example shows how to recover the type of guarantee they gave, and our second example shows that we can give similar guarantees under weaker assumptions by exploiting that we work in the excess risk / statistical learning (rather than parameter estimation) framework.

Example 2 (High-Dimensional Linear Predictors with ℓ 1 Constraint) . Suppose that θ ⋆ ∈ R d is an s -sparse linear function with support set T ⊂ [ d ] and that ∥ θ ⋆ ∥ 1 ≤ 1 and ∥ x ∥ ∞ ≤ 1 almost surely under D . Define the target class via

<!-- formula-not-decoded -->

Given S 2 = x 1: n , define the restricted eigenvalue for the target class as

<!-- formula-not-decoded -->

where X ∈ R n × d has x 1: n as rows. Then under the assumptions of Theorem 3, the empirical risk minimizer guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

For parameter estimation it is well known that restricted eigenvalue or related conditions are required to ensure parameter consistency. For prediction however, such as assumptions are not needed if we are willing to consider inefficient algorithms. The next example shows that ERM over predictors with a hard sparsity constraint obtains the optimal high-dimensional rate for prediction in the presence of nuisance parameters with no restricted eigenvalue assumption .

Example 3 (High-Dimensional Linear Predictors with Hard Sparsity) . Suppose that Θ is a class of high-dimensional linear predictors obeying exact or 'hard' sparsity:

<!-- formula-not-decoded -->

and suppose ∥ x ∥ ∞ ≤ 1 almost surely under D . Then under the assumptions of Theorem 3, the empirical risk minimizer guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Neural networks. We now move beyond the classical linear setting to the case to the case where the target parameters belong to a class of neural networks, a considerably more expressive class of models. Let σ log ( t ) = (1 + e -t ) -1 be the logistic link function and let σ relu ( t ) = max { t, 0 } be the ReLU function. 3

Our first neural network example is inspired by Chen and White (1999); Farrell et al. (2021), who analyzed neural networks for nuisance parameter estimation with parametric target parameters. We depart from their approach by using neural networks to estimate target parameters .

Example 4. Suppose that the target parameters are a class of neural networks Θ = σ log ◦ F , where

<!-- formula-not-decoded -->

and d 0 = d and d L = 1 . Let W = ∑ L i =1 d i d i -1 denote the total number of weights in the network. Under the assumptions of Theorem 3, the empirical risk minimizer guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

The target class in this example is well-suited to estimation of binary treatment effects. Note that in this example our only quantitative assumption on the network weights is that they guarantee boundedness of the output. However, the bound scales linearly with the number of parameters W , and thus may be vacuous for modern overparameterized neural networks. Our next example, which is based on neural networks covering bounds from Bartlett et al. (2017), shows that by making stronger assumptions on the weight matrices we can obtain weaker dependence on the number of parameters. This comes at the price of a slower raten -1 2 vs. n -1 .

Example 5. Suppose that the target parameters are a class of neural networks Θ = σ log ◦ F , where

<!-- formula-not-decoded -->

and ∥ A ∥ 2 , 1 denotes the sum of row-wise ℓ 2 -norms. Suppose ∥ x ∥ 2 ≤ 1 almost surely under D . Under the assumptions of Theorem 3, the empirical risk minimizer guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Let us give a concrete example where the neural network guarantees above enable oracle rates for the target class while using a more flexible parameter class for the nuisance. Suppose the target

3 For vector-valued inputs x we overload σ log ( t ) and σ relu ( t ) to denote element-wise application.

parameters belong to the class in Example 4 with L 2 layers and W 2 weights, and suppose the nuisance parameters also belong to a neural network class, but with L 1 layers and W 1 weights. In the next section we will show that under certain assumptions one can guarantee (Rate D ( G , S 1 , δ/ 2)) 4 = ˜ O (( W 1 L 1 /n ) 2 ) for such a class. In this case, Example 5 shows that we obtain oracle rates whenever W 1 L 1 = o ( √ W 2 L 2 n ), meaning the number of parameters in the nuisance network can be significantly larger than for the target network. Similar guarantees can be derived for Example 4.

Deriving tight generalization bounds for neural networks is an active area of research and there are many more results that can be used as-is to give guarantees for the second stage in our general framework (Golowich et al., 2018; Arora et al., 2019).

Kernels. For our final example, we give rates for some basic kernel classes. These examples were chosen only for concreteness, and the machinery in this section and the subsequent sections can be invoked to give guarantees for more rich and general nonparametric classes.

Example 6 (Gaussian Kernels) . Suppose that Θ ⊂ ([0 , 1] → R ) is unit ball in the reproducing kernel Hilbert space with the Gaussian kernel K ( x, x ′ ) = e -1 2 ( x -x ′ ) 2 . Suppose x is drawn from the uniform distribution over [0 , 1] . Under the assumptions of Theorem 3, the empirical risk minimizer guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Example 7 (Sobolev Spaces) . Suppose the target class is the Sobolev space

<!-- formula-not-decoded -->

and suppose that x is drawn from the uniform distribution on [0 , 1] . Under the assumptions of Theorem 3, the empirical risk minimizer guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

## H.1.1 Proofs

Throughout this section we adopt the shorthand ∥·∥ n, 2 = ∥·∥ L 2 ( z 1: n ) . We first recall some basic technical lemmas which will be used in the proofs for the examples.

Lemma 8 (Mendelson (2002), Lemma 4.5) . For any real-valued function class F with ∥ f ∥ n, 2 ≤ 1 for all f ∈ F and any f ⋆ with ∥ f ⋆ ∥ n, 2 ≤ 1 ,

<!-- formula-not-decoded -->

Lemma 9 (Wainwright (2019), Proposition 14.1) . Let δ n be the minimal solution to

<!-- formula-not-decoded -->

where F ⊆ ( Z → R ) is a star-shaped set with sup f ∈F sup z ∈Z | f ( z ) | ≤ 1 . Then with probability at least 1 -exp( -cnδ 2 n ) over the draw of data z 1: n ,

<!-- formula-not-decoded -->

where ̂ δ n is the minimal solution to

<!-- formula-not-decoded -->

The following result is an immediate consequence of Lemma 15.

## Lemma 10. Define

Then any minimal solution to

We will now establish that

<!-- formula-not-decoded -->

for an appropriate choice of b using the restricted eigenvalue bound. Let

<!-- formula-not-decoded -->

We first claim Θ -θ ⋆ ⊂ C . Indeed, fix θ ∈ Θ and let ∆ = θ -θ ⋆ . Then we have

<!-- formula-not-decoded -->

Rearranging, we get ∥ ∆ T c ∥ 1 ≤ ∥ ∆ T ∥ 1 as desired. Now observe that for any ∆ ∈ C , we have

<!-- formula-not-decoded -->

This implies that Θ( δ, x 1: n ) ⊆ G b for b = 2 √ s √ γ re δ , and as a consequence

<!-- formula-not-decoded -->

We plug this bound into (112) and derive an upper bound of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is an upper bound for the fixed point ̂ δ n for (110).

Proof for Example 2. Let data x 1: n be fixed. Let G = { x ↦→⟨ θ, x ⟩ | ∥ θ ∥ 1 ≤ b } . Under our assumption that ∥ x t ∥ ∞ ≤ 1, Zhang (2002), Theorem 3, implies that

<!-- formula-not-decoded -->

where we have used that star(Θ -θ ⋆ , 0) = Θ -θ ⋆ . Using Lemma 10, we may now take δ n ≤ O (√ s log( d/s ) n log n ) in (37), then combine with Theorem 3 and Theorem 1 to get the result.

Proof for Example 3. Since ∥ θ ∥ 1 ≤ 1 and ∥ x ∥ ∞ ≤ 1, the standard covering number bound for linear classes states that the covering number at scale ε for any fixed sparsity pattern is at most C · s log(1 /ε ). We take the union over all such covers for all ( d s ) ≤ ( ed s ) s sparsity patterns, which implies H 2 (Θ -θ ⋆ , ε, x 1: n ) ∝ s (log( d/s ) + log(1 /ε )). Lemma 8 further implies that

<!-- formula-not-decoded -->

It is now a standard calculation to show that

<!-- formula-not-decoded -->

Thus, via Lemma 9 and Lemma 10, we may take δ n ≤ O (√ s log( d/s ) log n n ) in (37). The final result follows by combining Theorem 3 and Theorem 1.

Proof for Example 4. Since our the target class is bounded, Lemma 8 implies

<!-- formula-not-decoded -->

Recall

<!-- formula-not-decoded -->

Then, since σ relu is 1-Lipschitz and positive-homogeneous, we have H 2 (Θ , ε, x 1: n ) ≤ H 2 ( F , ε, x 1: n ). Recall that for [0 , M ]-valued classes of regressors we can relate the empirical L 2 metric to an empirical L 1 metric for a closely related VC class as follows. Let Y ∼ unif([0 , M ]), let f, f ′ ∈ G , and write

<!-- formula-not-decoded -->

Consequently, we see that the L 2 covering number for F on the distribution P n at scale ε , is at most the size of the L 1 cover of the class F ′ = { ( x, y ) ↦→{ y ≤ f ( x ) } | f ∈ F} on distribution P n × P Y at scale ε 2 /M . Thus, invoking Haussler's L 1 covering number bound for VC classes (Haussler, 1995), we have

<!-- formula-not-decoded -->

where vc( · ) denotes the VC dimension and pdim( · ) denotes the pseudodimension. Using Theorem 14.1 from Anthony and Bartlett (1999) and Theorem 6 from Bartlett et al. (2017), we have

<!-- formula-not-decoded -->

With this bound on the metric entropy we have

<!-- formula-not-decoded -->

Thus, it suffices to take δ n ≤ O (√ LW log W log M log n n ) in (37) and appeal to Theorem 3 and Theorem 1.

Proof for Example 5. As in Example 4, we have H 2 (star(Θ -θ ⋆ ) , ε, x 1: n ) ≤ H 2 ( F , ε, x 1: n ). Theorem 3.3 of Bartlett et al. (2017) implies that under our assumptions,

<!-- formula-not-decoded -->

The result follows by plugging this bound into Lemma 10 and proceeding exactly as in the previous examples.

Proof for Example 6 and Example 7. Note that each target class Θ has range bounded by 1. By examples 14.4 and 14.3 in Wainwright (2019), we may take δ n = c √ log n/n and δ n = cn -1 / 3 in (37) for the gaussian and Sobolev classes respectively. We combine with this with Theorem 3 and Theorem 1.

## H.2 Plug-in Empirical Risk Minimization: Refined Guarantees for VC Classes

In this section of the appendix we use the general tools developed in Section 4 to provide efficient/variance-dependent oracle rates for VC classes with general Lipschitz losses. Our main result shows that for VC classes with dimension d , the excess risk enjoyed by variance penalization grows exactly as O ( √ V ⋆ d/n ) (where V ⋆ , as before, is the variance of the loss at the pair ( θ ⋆ , g 0 )) so long as the nuisance estimator converges at a rate of o ( n -1 / 4 ). The key to our approach is to assume boundedness of the so-called Alexander capacity function , a classical quantity that arises in the study of ratio type empirical processes (Gin´ e and Koltchinskii, 2006).

To be more precise, for this example we assume that Θ is a class of binary predictors with VC dimension d , and let ℓ have the following policy learning structure:

<!-- formula-not-decoded -->

where Γ is a known function. Our goal is to derive a bound for which the leading term only scales with V ⋆ rather than the loss range. Our results depend on a variant of the Alexander capacity function (Gin´ e and Koltchinskii, 2006; Hanneke, 2014). Letting

<!-- formula-not-decoded -->

the capacity function is defined as

<!-- formula-not-decoded -->

When Γ is the unweighted classification loss, this definition recovers the classical definition of the capacity function (Gin´ e and Koltchinskii, 2006; Hanneke, 2014). Beyond boundedness of the capacity function, we make the following assumption.

Assumption 12. Assumption 5 holds along, with the following bounds:

- | Γ( g, z ) | ≤ R almost surely for all g ∈ G , for some R ≥ 1 .
- E [ Γ 2 ( g 0 , z ) | x ] ≥ γ almost surely.
- ( E (Γ( g, z ) -Γ( g 0 , z )) 4 ) 1 / 4 ≤ L ∥ g -g 0 ∥ G for all g ∈ G , for some seminorm ∥·∥ G .
- The first stage algorithm provides an estimation error bound with respect to ∥·∥ G , i.e. ∥ ̂ g -g 0 ∥ G ≤ Rate D ( G , S, δ ) with probability at least 1 -δ over the draw of S .
- Assumption 6 holds with respect to ∥·∥ G with constant β .

Theorem 8. Suppose that Assumption 12 holds, and define τ 0 := sup ε ≥ √ d / n { τ ( ε ) } . Then variancepenalized empirical risk minimizer guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Note that whenever Rate D ( G , S 1 , δ/ 2) = o ( n -1 / 4 ) the asymptotic rate depends only on the variance at θ ⋆ and g 0 , not on the problem-dependent parameters L/R/βγ . Furthermore, whenever the capacity function is constant the asymptotic rate is exactly O ( √ V ⋆ d/n ).

Variance-dependent bounds that obtain the efficient O ( √ V ⋆ d/n ) rate have been the subject of much recent investigation, and there is much interest in understanding when the O ( √ V ⋆ d log n/n ) rate obtained by naive approaches can be improved. To give a brief survey, the seminal empirical variance bound due to Maurer and Pontil (2009) when applied directly to this setting gives a suboptimal O ( √ V ⋆ d log n/n ) rate. Recent work of Athey and Wager (2017) shows that for a specific loss and nuisance parameter setup arising in policy learning, the log n can be replaced with a certain worst-case variance parameter. Our result, Theorem 8 is complementary, and shows that the log n can be replaced by the capacity function for general losses . It appears unlikely that the log n factor can be removed without at least some type of assumption. Indeed the results in Rakhlin et al. (2017) imply that there are indeed VC classes for which the critical radius grows as √ d log n/n in the worst case.

The proof of Theorem 8 can be broken into three parts: First, we apply the previous results of this section to show that the excess risk obtained by variance penalization depends on the critical radius of the class ℓ ◦ Θ. Second, we show that in the absence of first-stage estimation error, the capacity function controls the critical radius. Finally, we show that the impact of nuisance estimation error on the capacity function is of second order.

## H.2.1 Proof of Theorem 8

Define the function class F = { z ↦→ ℓ ( θ ( x ) , ̂ g ( w ); z ) : θ ∈ Θ } . We assume for now that ∥ f ∥ ∞ ≤ 1 for all f ∈ F , that Γ is 1-Lipschitz (i.e. L = R = 1 in the theorem statement) and that E [ Γ 2 ( g 0 , z ) | x ] ≥ γ ; the general case will be handled by rescaling at the end of the proof..

Our starting point is to appeal to Theorem 9 . In particular, let δ n ≥ 0 be any solution to the inequality:

<!-- formula-not-decoded -->

where f ⋆ ( z ) := ℓ ( θ ⋆ ( x ) , ̂ g ( w ); z ). Then if ̂ θ is the outcome of variance-penalized ERM, we have that by Theorem 4, with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Moving to capacity function at g 0 . Per the discussion in the prequel, we focus on bounding the critical radius in the case where F is bounded by 1 and Γ is 1-Lipschitz. We wish to make use of the capacity function, which is defined at g 0 , but the local Rademacher complexity we need to bound is that of F , which evaluates the weight function Γ at ̂ g . To make progress, we show how to use the capacity function defined in the theorem statement to bound the following 'plug-in' variant:

<!-- formula-not-decoded -->

Wefirst show how to relate the L 2 norm at ̂ g to the L 2 norm at g 0 . Define ∥ θ ∥ Θ = √ E Γ( ̂ g, z ) 2 ( θ ( x ) -θ ⋆ ( x )) 2 . Then for any θ ∈ Θ we have

<!-- formula-not-decoded -->

Using AM-GM and boundedness of θ , for any η &gt; 0 this is lower bounded by

<!-- formula-not-decoded -->

Using the Lipschitz assumption and conditional lower bound on Γ, we further lower bound by

<!-- formula-not-decoded -->

Hence, by choosing η = γ/ 2 and rearranging, we get

<!-- formula-not-decoded -->

We proceed to bound the capacity function ̂ τ . Let ε 0 = 2 γ 1 / 2 ∥ ̂ g -g 0 ∥ 2 G . Let ε ≥ ε 0 be fixed and let θ ∈ Θ be any policy with E [ Γ( ̂ g, z ) 2 ( θ ( x ) -θ ⋆ ( x )) 2 ] ≤ ε 2 . Then equation (115) implies that that ∥ θ -θ ⋆ ∥ 2 Θ ≤ 5 ε 2 , and so

<!-- formula-not-decoded -->

To handle the term in the numerator we proceed similar to the proof of (115). We have

<!-- formula-not-decoded -->

Fix any η &gt; 0. We use AM-GM and boundedness of policies to upper bound the second term as

<!-- formula-not-decoded -->

We choose η = γ and recall the definition of ε 0 , which gives

<!-- formula-not-decoded -->

Putting everything together, we get

<!-- formula-not-decoded -->

Thus, for all ε ≥ ε 0 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Bounding the critical radius. For any class F we define F δ = { f ∈ F : ∥ f ∥ 2 ≤ δ } . We work with the following empirical version of the local Rademacher complexity

<!-- formula-not-decoded -->

which has R n ( F , δ ) = E z 1: n [ R n ( F , δ, z 1: n )]. Let the draw of z 1: n be fixed. Invoking Lemma 15, we have

<!-- formula-not-decoded -->

Using that any h ∈ star( F f ⋆ ) δ can be written as r · ( f -f ⋆ ), where ∥ f -f ⋆ ∥ 2 ≤ δ and r ∈ [0 , 1], a simple discretization argument (cf. proof of Lemma 8) shows that

<!-- formula-not-decoded -->

Let us adopt the shorthand v n = sup h ∈ ( Ff ⋆ ) δ ∥ h ∥ n . It follows from the usual symmetrization argument that E v 2 n ≤ δ 2 +2 R n ( F f ⋆ , δ ). Letting α = 0 be fixed, we can summarize our argument so far as

<!-- formula-not-decoded -->

Furthermore, using a change of variables we have

<!-- formula-not-decoded -->

We now handle the covering integral for ( F f ⋆ ) δ . Let

<!-- formula-not-decoded -->

Our approach is to upper bound the empirical L 2 covering number for ( F f ⋆ ) δ by the covering number of the class Θ( δ ) with respect to Hamming error. Let the Hamming error on a set S ′ = { x 1 , . . . , x M } be defined via

̸

We claim that there is a choice for the dataset S ′ = x ′ 1 , . . . , x ′ M such that for all g, g ′ ∈ ( F f ⋆ ) δ , the empirical L 2 error on S 2 is upper bounded by the empirical Hamming error of associated policies θ, θ ′ on S ′ .

<!-- formula-not-decoded -->

Let h = ( f -f ⋆ ) and h ′ = ( f ′ -f ⋆ ) be fixed elements of ( F f ⋆ ) δ , and let θ, θ ′ ∈ Θ( δ ) be such that f ( z ) = Γ( ̂ g, z )( θ ( x ) -θ ⋆ ( x )) and likewise for f ′ and θ ′ . Define Γ i = Γ( ̂ g, z i ), and take S ′ to contain of m i := ⌈ sup θ ∈ Θ( δ ) Γ 2 i ( θ ( x i ) -θ ⋆ ( x i )) 2 δ 2 ⌉ copies of example x i for each i . With this choice, we have

̸

<!-- formula-not-decoded -->

̸

Thus, if we let ε ′ = n 4 Mδ 2 ε 2 , then any ε ′ -cover in Hamming error is an ε -cover in L 2 . Now define

<!-- formula-not-decoded -->

and note that u 2 n ≥ v 2 n by definition. We invoke the following facts.

<!-- formula-not-decoded -->

̸

̸

- Haussler's bound (Haussler, 1995) implies that any class with VC dimension d admits a ε -Hamming error cover of size e ( d +1) ( 2 e ε ) d .

Putting everything together, we have

<!-- formula-not-decoded -->

It follows from the usual symmetrization argument that E [ v 2 n ] ≤ δ 2 +2 R n ( F f ⋆ , δ ). Furthermore, using the concentration bound in (122) (we use the assumed boundedness of elements of F to simplify (122) to the form that appears on this page), there exists a constant C ≥ 1 such that for any s &gt; 0, with probability at least 1 -e -s over the draw of z 1: n ,

<!-- formula-not-decoded -->

Thus, conditioning on this event, we have

<!-- formula-not-decoded -->

where the second inequality uses a change of variables and that δ ≤ ˜ δ .

Now, to summarize our developments so far, we have shown that with probability at least 1 -e -s ,

<!-- formula-not-decoded -->

Using Markov's inequality, we have that with probability at least 1 -e -s , u 2 n ≤ E [ u 2 n ] · e s . Thus, by union bound, with probability at least 1 -2 e -s ,

<!-- formula-not-decoded -->

Integrating out this tail bound, we get that

<!-- formula-not-decoded -->

Using AM-GM and that R n ( F f ⋆ , δ ) ≤ R n (star( F f ⋆ ) , δ ), then rearranging, this implies

<!-- formula-not-decoded -->

We now bound the ratio E z 1: n [ u 2 n ] /δ 2 . We have

<!-- formula-not-decoded -->

Thus, using the relationship between τ and ̂ τ established in the previous section of the proof, if δ &gt; ε 0 we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In particular, we can see from this expression that taking

<!-- formula-not-decoded -->

yields a valid upper bound on the critical radius.

Final bound. Putting together the excess risk bound and the critical radius bound, we have

<!-- formula-not-decoded -->

To handle the square Rademacher complexity term, we recall that since | Γ | ≤ 1, the main result of Haussler (1995) implies that R n ( F ) ≤ √ d n ; since this term is squared in the final bound, its contribution is of lower order.

To deduce the final bound in the general R -bounded L -Lipschitz case we divide the class by ( L + R ), then rescale the final bound (observing that β , γ , and V ⋆ all vary appropriately under the rescaling).

and so for all δ &gt; ε 0 ,

## Part III

## Proofs for Main Results

## I Preliminaries

We invoke the following version of Taylor's theorem and its directional derivative generalization repeatedly.

Proposition 5 (Taylor expansion) . Let a ≤ b be fixed and let f : I → R , where I ⊆ R is an open interval containing a, b . If f is ( k +1) -times differentiable, then there exists c ∈ [ a, b ] such that

<!-- formula-not-decoded -->

Let F : F → R , where F is a vector space of functions. For any g, g ′ ∈ F , if t ↦→ F ( t · g +(1 -t ) · g ′ ) is ( k +1) -times differentiable over an open interval containing [0 , 1] , then there exists ¯ g ∈ conv( { g, g ′ } ) such that

<!-- formula-not-decoded -->

## J Proofs from Section 3

## J.1 Omitted Proofs for Main Results

Proof of Theorem 2. To begin, we use the guarantee for the second stage from Definition 1 and perform straightforward manipulation to show

<!-- formula-not-decoded -->

Using continuity guaranteed by Assumption 6, we perform a second-order Taylor expansion with respect to g for each pair of loss terms in the preceding expression to conclude that there exist g, g ′ ∈ star( G , g 0 ) such that

<!-- formula-not-decoded -->

Using the smoothness promised by Assumption 6:

<!-- formula-not-decoded -->

To relate the two derivative terms, we apply another second-order Taylor expansion (which is possible due to Assumption 6), this time with respect to the target predictor.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ¯ θ ∈ conv( { ̂ θ, θ ⋆ } ). Universal orthogonality immediately implies that

<!-- formula-not-decoded -->

Furthermore, observe that

<!-- formula-not-decoded -->

Since ¯ θ + t ( ̂ θ -θ ⋆ ) ∈ star( ̂ Θ , θ ⋆ ) + star( ̂ Θ -θ ⋆ , 0) for all t ∈ [0 , 1], including t = 0, universal orthogonality (Assumption 5) implies that both terms in the numerator are zero, and hence D 2 θ D g L D ( ¯ θ, g 0 )[ ̂ g -g 0 , ̂ θ -θ ⋆ , ̂ θ -θ ⋆ ] = 0. We conclude that D g L D ( ̂ θ, g 0 )[ ̂ g -g 0 ] = D g L D ( θ ⋆ , g 0 )[ ̂ g -g 0 ]. Using this identity in the excess risk upper bound, we arrive at

<!-- formula-not-decoded -->

## J.2 Proofs for Examples

Proof of Proposition 1. We first verify that the conditions of Theorem 1 are satisfied. To establish orthogonality (Assumption 1) for the propensities e , let θ, θ ′ be fixed. Then we have

<!-- formula-not-decoded -->

To handle the first term, we use that for any x, w ,

<!-- formula-not-decoded -->

Similarly, the second term is handled by using that

<!-- formula-not-decoded -->

To establish orthogonality for the expected value parameter m , for any θ, θ ′ we have

<!-- formula-not-decoded -->

which follows from the assumption E [ ε 2 | X,W ] = 0. Note that both of these orthogonality proofs held for any choice of θ , not just θ 0 , and hence Assumption 1 is satisfied for all θ ⋆ .

Next, we show that Assumption 2 holds whenever the second stage is well-specified (i.e. θ 0 ∈ Θ); the fact that Assumption 2 whenever Θ is convex is immediate. We have

<!-- formula-not-decoded -->

In particular, for any x we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It remains to verify Assumptions 3 and 4. We do so appealing to Lemma 5. Following the notation of Appendix E, we set Λ( g ( w ) , w ) = ( T -e ( W )), Γ( g ( w ) , z ) = ( Y -m ( X,W )), and ϕ ( ζ ) = ζ . With this choice, we can take T si = τ si = L si = R si = 1, and λ si = λ re . To bound the parameter µ si , for ζ ∈ R , γ ∈ R 2 , and z = ( X,W,T ), we write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which has

It follows that ∥∇ 2 γγ ∇ ζ i ℓ ( θ ⋆ ( x ) , g ( w ); z ) ∥ σ ≤ 4 =: µ si whenever | θ ⋆ ( x ) | ≤ 1. As a result Lemma 5 implies that Assumptions 3 and 4 are satisfied with constants λ = 1 4 , κ = 4 λ -1 re , β 1 = 1 and β 2 = 4 λ -1 / 2 re . The result now follows from Theorem 1.

Proof and Details for Example 1. Formally, we consider the following setup (Nie and Wager, 2021). Let H is an RKHS with norm ∥·∥ H and kernel K . We assume that X ⊆ R d is a compact metric space, and that K is a kernel with respect to D . We define T K : L 2 ( D ) → L 2 ( D ) via

<!-- formula-not-decoded -->

Using Mercer's theorem (Cucker and Smale, 2002), there exist eigenfunctions ( ψ j ) ∞ j =1 and eigenvalues ( σ j ) ∞ j =1 such that

<!-- formula-not-decoded -->

The main assumptions are as follows.

1. Eigenvalue decay. There exists p ∈ (0 , 1) such that

<!-- formula-not-decoded -->

2. Approximation. There exists α ∈ (0 , 1 / 2) such that the function θ 0 has

<!-- formula-not-decoded -->

In addition, we assume that K ( x, y ) ≤ 1, that ∥ ψ j ∥ L ∞ ( D ) ≤ A (note that ∥ ψ j ∥ L 2 ( D ) = 1), and that | Y | ≤ 1 almost surely. We also assume overlap, i.e. η ≤ e 0 ( X ) ≤ 1 -η . Throughout the proof, we use ˜ O ( · ) to suppress dependence on G , R , A , η -1 (1 -η ) -1 , and log( n ). Recall that the estimator ̂ θ is obtained via plug-in empirical risk minimization with respect to the constrained function class

<!-- formula-not-decoded -->

̸

for a parameter c ≥ 1. As noted in Eq. (19) of Nie and Wager (2021), if we define θ ⋆ := arg min θ ∈ Θ L D ( θ, g 0 ) (note that θ ⋆ = θ 0 , since Θ is constrained), we have

<!-- formula-not-decoded -->

In particular, by choosing c ∝ n α/ ( p +(1 -2 α )) , the approximation error scales as O ( n -1 -2 α p +(1 -2 α ) ) to θ 0 . In light of this approximation result, is remains to derive an oracle excess risk bound against θ ⋆ for fixed c .

To proceed, we verify that the assumptions required by Theorem 1 (in particular, smoothness and strong convexity) are satisfied. We do so appealing to Lemma 5. Following the notation of Appendix E, we set Λ( g ( w ) , w ) = ( T -e ( W )), Γ( g ( w ) , z ) = ( Y -m ( X,W )), and ϕ ( ζ ) = ζ . With this choice, we can take T si = τ si = L si = R si = 1. It remains to bound µ si and λ si .

- To bound the parameter µ si , for ζ ∈ R , γ ∈ R 2 , and z = ( X,W,T ), we write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which has

It follows that ∥∇ 2 γγ ∇ ζ i ℓ ( θ ⋆ ( x ) , g ( w ); z ) ∥ σ ≤ 4 =: µ si whenever | θ ⋆ ( x ) | ≤ 1.

- To bound λ si , we recall Lemma 5.1 of Mendelson and Neeman (2010), which states that for all θ ∈ H ,

<!-- formula-not-decoded -->

where C p is a constant depending on p , G , and A . In addition, we have ∥ θ ∥ 2 L 2 ( D ) ≤ η -1 (1 -η ) -1 ∥ θ ∥ 2 Θ and ∥ θ ∥ H ≤ c . Hence, we may take λ -1 si = ˜ O ( c 2 p ) and r si = p .

As a result, Lemma 5 implies that Assumptions 3 and 4 are satisfied with constants r = p , λ -1 , β 1 = ˜ O (1), κ = ˜ O ( c 2 p ), and β 2 = ˜ O ( c p ). Theorem 3 now implies that with probability at least 1 -δ , the plug-in empirical risk minimizer satisfies

<!-- formula-not-decoded -->

where δ n is the solution to the fixed point equation in (37). From Lemma 5 of Nie and Wager (2021), we have

<!-- formula-not-decoded -->

This implies that δ 2 n = ˜ O ( c 2 p 1+ p n -1 (1+ p ) ) is a valid fixed point, so we have

<!-- formula-not-decoded -->

Whenever ∥ ̂ g -g 0 ∥ L 2 ( ℓ 2 , D ) ≤ ˜ o ( n -1 / 4 ), this simplifies to

<!-- formula-not-decoded -->

and combining with the approximation result in (118) yields

<!-- formula-not-decoded -->

Choosing c = n α/ ( p +(1 -2 α )) gives excess risk O ( n -1 -2 α p +(1 -2 α ) ).

Proof of Proposition 2. We verify that the conditions of Theorem 1 are satisfied. We do so by showing that Assumption 9 is satisfied and appealing to Lemma 5. In particular, following the notation of Appendix E, we observe that the doubly-robust loss has the structure required by Assumption 9, since we may take Λ( g ( w ) , w ) = 1, Γ( g, z ) = φ ( f, e ; z ), and ϕ ( ζ ) = ζ . Furthermore, it can be written as a square loss of the form

<!-- formula-not-decoded -->

where, for γ = ( a 0 , a 1 , b ) ∈ R 3 , we overload notation and write

<!-- formula-not-decoded -->

To establish orthogonality (Assumption 1), it suffices to show that

<!-- formula-not-decoded -->

We have

<!-- formula-not-decoded -->

and it is straightforward to verify that E [ ∇ γ φ ( f 0 , e 0 ; z ) | X ] = 0 as a consequence of the double robustness property

Next, since L D ( θ, g 0 ) = E [ ( θ ( X ) -θ 0 ( X )) 2 ] +Var( φ ( f 0 , e 0 ; z )), it is immediate that Assumption 2 is satisfied.

Finally, we verify the regularity conditions required to establish Assumptions 3 and 4. With p = 2, it is immediate that we may take T si = τ si = R si = λ si = 1, L si = 0, and r si = 0 for Assumption 9 whenever | θ ( X ) | ≤ 1. To bound the parameter µ si , for ζ ∈ R , γ ∈ R 3 and z = ( X,W,T ) we observe that

<!-- formula-not-decoded -->

Since | ζ | ≤ 1 by assumption, it suffices to bound ∥ ∥ ∇ 2 γ φ ( γ ; z ) ∥ ∥ σ ≤ 3 ∥ ∥ ∇ 2 γ φ ( γ ; z ) ∥ ∥ ∞ . It is straightforward to verify that whenever | f (0 , X ) | , | f (1 , X ) | ≤ 1, | Y | ≤ 1, and η ≤ e ( X ) ≤ 1 -η , we have ∥ ∥ ∇ 2 γ φ ( γ ; z ) ∥ ∥ ∞ ≤ 4 η -3 , so we may take µ si = 24 η -3 . As a result, Lemma 5 implies that Assumptions 3 and 4 are satisfied with constants r = 0, λ = 1 4 , κ = 0, β 1 = 1 and β 2 = 24 η -3 . The result now follows from Theorem 1.

Proof of Proposition 3. We first verify that Assumption 5 is satisfied. Let g , θ , and ¯ θ be arbitrary. Abbreviate f ( t ) ( x ) = f ( t, x ). We have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Similarly, we have

<!-- formula-not-decoded -->

This establishes the universal orthogonality property.

We now verify that Assumption 6 is satisfied with ∥·∥ G = ∥·∥ L 2 ( ℓ 2 , D ) . Consider the function B Z : R 3 → R given by

<!-- formula-not-decoded -->

Assumption 6 holds whenever ∥ ∥ ∇ 2 v B Z ( v ) ∥ ∥ σ is bounded for all Z and all v in the range of G . In particular, one can verify by inspection that ∥∇ 2 v B Z ( v ) ∥ σ ≤ O ( η -3 ) whenever | Y | ≤ 1, all g = { f (0 , · ) , f (1 , · ) , e } ∈ G have | f ( t, X ) | ≤ 1 and e ( X ) ∈ [ η, 1 -η ].

With both assumptions satisfied, the result now follows from Theorem 2.

## K Technical Lemmas for Constrained M -Estimators

In this section of the appendix we give self-contained technical results for M -estimation over general function classes, in the absence of nuisance parameters. The results here serve as a building block for the results of Section 4.

Let F : X → R d be the function class, and let ℓ : R d ×Z → R be the loss. We receive a sample set S = z 1 , . . . , z n drawn from distribution D independently. Let L f denote the random variable ℓ ( f ( x ) , z ) and let

<!-- formula-not-decoded -->

denote the population risk and empirical risk over n samples. The constrained empirical risk minimizer is given by

<!-- formula-not-decoded -->

Additional Notation. To keep notation compact, we adopt the abbreviations ∥ f ∥ p,q := ∥ f ∥ L p ( ℓ q , D ) and ∥ f ∥ n,p,q := ∥ f ∥ L p ( ℓ q ,z 1: n ) . We drop the subscript q for real-valued function classes. We recall that F| t denotes the t th coordinate projection of F and, likewise, for any f ∈ F , the t th coordinate projection is f t ∈ F| t .

The following lemmas provide a vector-valued extension of the analysis of constrained ERM based on local Rademacher complexities given in Wainwright (2019). The key idea behind the extension is to invoke a vector-valued contraction theorem for Rademacher complexity due to Maurer (2016). For completeness we give a include proofs for these lemmas, even though some parts are straightforward adaptations of lemmas in Wainwright (2019).

Lemma 11. Consider a function class F : X → R d , with sup f ∈F ∥ f ∥ ∞ , 2 ≤ 1 and pick any f ⋆ ∈ F . Assume that the loss ℓ is L -Lipschitz in its first argument with respect to the ℓ 2 -norm and let

<!-- formula-not-decoded -->

Then there are universal constants c 1 , c 2 &gt; 0 such that

<!-- formula-not-decoded -->

Moreover, if δ n is any solution to the inequalities

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 12. Consider a vector valued function class F : X → R d with sup f ∈F ∥ f ∥ ∞ , 2 ≤ 1 , and pick any f ⋆ ∈ F . Let δ 2 n ≥ 4 d log(41 log(2 c 2 n )) c 2 n be any solution to the system of inequalities

<!-- formula-not-decoded -->

Moreover, assume that the loss ℓ is L -Lipschitz in its first argument with respect to the ℓ 2 norm. Consider the following event:

<!-- formula-not-decoded -->

Then for some universal constants c 3 , c 4 , Pr[ E 1 ] ≤ c 3 exp( -c 4 nδ 2 n ) .

Lemma 13. Consider a vector valued function class F : X → R d and pick any f ⋆ ∈ F . Let δ n ≥ 0 be any solution to the inequalities

<!-- formula-not-decoded -->

Suppose sup f ∈F ∥ f ∥ ∞ , 2 ≤ 1 . Moreover, assume that the loss ℓ is L -Lipschitz in its first argument with respect to the ℓ 2 norm and also linear, i.e. L f + f ′ = L f + L f ′ and L αf = α L f . Consider the following event:

<!-- formula-not-decoded -->

Then for some universal constants c 3 , c 4 , Pr[ E 1 ] ≤ c 3 exp( -c 4 nδ 2 n ) .

then for each r ≥ δ n ,

Lemma 14. Consider a function class F , with sup f ∈F ∥ f ∥ ∞ ≤ 1 , and pick any f ⋆ ∈ F . Let δ 2 n ≥ 4 d log(41 log(2 c 2 n )) c 2 n be any solution to the inequalities:

<!-- formula-not-decoded -->

Moreover, assume that the loss ℓ is L -Lipschitz in its first argument with respect to the ℓ 2 norm. Then for some universal constants c 5 , c 6 , with probability 1 -c 5 exp( c 6 nδ 2 n ) ,

<!-- formula-not-decoded -->

Hence, the outcome ̂ f of constrained ERM satisfies that with the same probability,

<!-- formula-not-decoded -->

If the loss L f is also linear in f , i.e. L f + f ′ = L f + L f ′ and L αf = α L f , then the lower bound on δ 2 n is not required.

## K.1 Proofs of Lemmas for Constrained M -Estimators

Proof of Lemma 11. By the Lipschitz condition on the loss and the boundedness of the functions, we have ∥L f -L f ⋆ ∥ ∞ ≤ L ∥ f -f ⋆ ∥ ∞ , 2 ≤ 2 L . Moreover,

<!-- formula-not-decoded -->

Thus, by Bousquet's concentration inequality (see Theorem 7.3 of Bousquet (2003) or Theorem 12.5 of Boucheron et al. (2013)) we have that for all u &gt; 0,

<!-- formula-not-decoded -->

for absolute constants c 1 , c 2 &gt; 0. Moreover, by a standard symmetrization argument,

<!-- formula-not-decoded -->

where the second inequality follows from the fact that each summand is non-negative, since we can always choose f = f ⋆ . By invoking the multivariate contraction inequality of Maurer (2016), letting

{ ϵ i,t } 1 ≤ i ≤ n, 1 ≤ t ≤ d be independent Rademacher random variables, we have

<!-- formula-not-decoded -->

where we also used the fact that for any fixed f ⋆ , E [ ϵ i ℓ ( f ⋆ ( x i ) , z i )] = E [ ϵ i,t f ⋆ t ( x i )] = 0. This completes the proof of the first part of the lemma. For the second part, observe that: R ( F| t -f ⋆ t , r ) ≤ R (star( F| t -f ⋆ t ) , r ). Moreover, for any star shaped function class G , the function r → R ( G ,r ) r is monotone non-increasing. Thus for any r ≥ δ n ,

<!-- formula-not-decoded -->

Rearranging yields that R (star( F| t -f ⋆ t ) , r ) ≤ rδ n . Hence, E [ Z n ( r )] ≤ 8 Ldrδ n . This completes the proof of the second part of the lemma.

Proof of Lemma 12. We invoke a peeling argument. Consider the events

<!-- formula-not-decoded -->

for α = 18 / 17. Since sup f ∈F ∥ f -f ⋆ ∥ 2 , 2 ≤ 2sup f ∈F ∥ f ∥ ∞ , 2 ≤ 2, it must be that any f ∈ F with ∥ f -f ⋆ ∥ 2 , 2 ≥ δ n belongs to some S m for m ∈ { 1 , 2 , . . . , M } , where M ≤ log(2 /δ n ) log α ( e ) ≤ 41log(2 /δ n ). Thus by a union bound we have

<!-- formula-not-decoded -->

Moreover, observe that if the event E 1 ∩ S m occurs then there exists a f ∈ F with ∥ f -f ⋆ ∥ 2 , 2 ≤ α m δ n = r m , such that

<!-- formula-not-decoded -->

Thus, by the definition of Z n ( r ), we have

<!-- formula-not-decoded -->

Applying Lemma 11 with r = r m and u = Ldr m δ n , yields that the latter probability is at most c 1 exp( -c 2 n L 2 r 2 m δ 2 n L 2 r 2 m + L 2 dr m δ n ) ≤ c 1 exp( -c 2 n δ 2 n 2 d ), where we used the fact that δ n ≤ r m in the last inequality. Subsequently, taking a union bound over the M events, we have

<!-- formula-not-decoded -->

Since, by assumption on the lower bound on δ n we have log( M ) ≤ log(41 log(2 /δ n )) ≤ log(41 log(2 c 2 n )) ≤ c 2 nδ 2 n 4 d , we get

<!-- formula-not-decoded -->

Proof of Lemma 13. For simplicity, let ∥·∥ = ∥·∥ 2 , 2 . Suppose that there exists a function f ∈ F , with ∥ f -f ⋆ ∥ ≥ δ n , such that

<!-- formula-not-decoded -->

Then we will show that there exists a function f ′ ∈ star( F f ⋆ ), with ∥ f ′ -f ⋆ ∥ = δ n , such that

<!-- formula-not-decoded -->

To do so, we simply choose f ′ to satisfy

<!-- formula-not-decoded -->

Since δ n ∥ f -f ⋆ ∥ ≤ 1 and by the definition of the star hull, we know that f ′ ∈ star( F f ⋆ ). Moreover, by the definition of θ ′ , we also have that ∥ f ′ -f ⋆ ∥ n = δ n . Moreover, by the linearity of the loss L ⋆ f with respect to f , we have:

<!-- formula-not-decoded -->

Thus we have that the probability of event E 1 is upper bounded by the probability of the event

<!-- formula-not-decoded -->

Invoking Lemma 11 with r = δ n and u = Ldδ 2 n , we conclude that the probability of the second event is also at most µ ′ 1 exp( -µ ′ 2 nδ 2 n ), for some universal constants µ ′ 1 , µ ′ 2 .

Proof of Lemma 14. Consider the events:

<!-- formula-not-decoded -->

with Z n ( r ) as defined in Lemma 11. Observe that if (130) is violated, then one of these events must occur. Applying Lemma 11 with r = δ n and u = Ldδ 2 n yields, that event E 0 happens with probability at most c 1 exp( c ′ 2 nδ 2 n ), where c ′ 2 = c 2 / ( L 2 + Ld ). Moreover, applying Lemma 12 we get that Pr[ E 1 ] ≤ c 3 exp( -c 4 nδ 2 n ). Thus by a union bound with probability 1 -c 5 exp( c 6 nδ 2 n ), neither events occur. If the loss L f is linear then we apply Lemma 13 instead of Lemma 12, which does not require a lower bound on δ 2 n .

## L Proofs from Section 4

Let L θ,g be shorthand for the random variable ℓ ( θ ( x ) , g ( w ); z ), and recall that

<!-- formula-not-decoded -->

denote, respectively, the population risk and empirical risk over the n samples in S 2 . We consider the two-stage plugin ERM algorithm

<!-- formula-not-decoded -->

As in Appendix K, we adopt the abbreviations ∥ f ∥ p,q := ∥ f ∥ L p ( ℓ q , D ) and ∥ f ∥ n,p,q := ∥ f ∥ L p ( ℓ q ,S 2 ) ; we drop the subscript q for real-valued function classes.

Throughout this section, we repeatedly make use of the following fact: If δ n solves a fixed point equation such as (37), then δ ′ n := δ n + C √ log(1 /δ ) n does as well. By expanding the radius in this fashion, we can replace success probabilities of the form 1 -e -cnδ 2 n (e.g., in Theorem 3 below) by 1 -δ , as long as δ n is replaced by δ ′ n in the final excess risk bound.

## L.1 Proof of Theorem 3

We prove Theorem 3 for the case where R = 1. The general case follows by observing that ERM over Θ is equivalent to ERM over the class Θ /R with the loss ˜ ℓ ( ζ, g ( w ); z ) := ℓ ( R · ζ, g ( w ); z ), with the problem parameters remapped as L ↦→ LR , β 1 ↦→ β 1 R 2 , β 2 ↦→ β 2 R , κ ↦→ κ , λ ↦→ λR 2 , and δ n ↦→ δ n /R .

Proof of Theorem 3. Since ̂ θ is the outcome of the Plug-In ERM and since θ ⋆ ∈ Θ, we have

<!-- formula-not-decoded -->

Applying Lemma 12, with F = Θ, f ⋆ = θ ⋆ and L · = L · , ̂ g , we know that with probability at least 1 -c 1 exp( -c 2 nδ 2 n ), where c 1 , c 2 &gt; 0 are numerical constants,

<!-- formula-not-decoded -->

Since Assumptions 1 to 4 are satisfied, Lemma 1 (a corollary of Theorem 1) with ε n ( δ ) := 18 LK 2 δ n and α n ( δ ) := 18 LK 2 δ 2 n implies that

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where C 1 ≤ 4 λ and C 2 ≤ 2 ( ( β 2 λ ) 2 1+ r + κ λ ) are as in Theorem 1; note that we have used that α n ( δ ) ≤ ε 2 n ( δ ) and C 1 ≥ 1 to simplify. This establishes the result.

## L.2 Proof of Theorem 4

We first prove a theorem about non-centered ℓ 2 -moment penalization, and then show that this implies Theorem 4.

Theorem 9 (Moment-Penalized Plug-In ERM) . Consider the function class F = { ℓ ( θ ( · ) , ̂ g ( · ); · ) : θ ∈ Θ } , with R := sup f ∈F ∥ f ∥ L ∞ ( D ) and f ⋆ := ℓ ( θ ⋆ ( · ) , ̂ g ( · ); · ) . Let δ 2 n ≥ 0 be any solution to the inequality

<!-- formula-not-decoded -->

and ̂ θ be the second moment-penalized empirical risk minimizer,

<!-- formula-not-decoded -->

Then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof of Theorem 9. We first consider the case where R = 1. We apply Lemma 14 with L f := f ( x ) (i.e., the identity loss). Observe that ∥ f -f ⋆ ∥ 2 ≤ 2 ∥ f -f ⋆ ∥ n, 2 + δ n with probability 1 -c 7 exp( c 8 nδ 2 n ) (via Theorem 4.1 of Wainwright (2019)), and that L f is linear. As a result, we have that for any δ n ≥ 0 that satisfies the conditions of Theorem 9, with probability 1 -c 9 exp( c 10 nδ 2 n ),

<!-- formula-not-decoded -->

This implies that with probability at least 1 -c 9 exp( c 10 nδ 2 n ),

<!-- formula-not-decoded -->

where the second inequality follows by the definition of the moment-penalized algorithm, since

<!-- formula-not-decoded -->

Now, for general values of R , we may apply the reasoning above to the normalized class F /R . In particular, the condition (136) implies that

<!-- formula-not-decoded -->

so we may take δ ′ n := δ n /R as the critical radius for the normalized class. This, combined with (137), implies that probability at least 1 -δ ,

<!-- formula-not-decoded -->

which establishes the result.

Proof of Theorem 4. We first sketch the intuition behind how to deduce the variance-based bound from Theorem 4 from the second moment-based bound from Theorem 9, then give a formal proof. Observe that if the optimal loss µ ⋆ := L D ( θ ⋆ , g 0 ) is zero then the two are equivalent. This motivates the following approach: if one has access to a good preliminary estimate ̂ µ of the value µ ⋆ , then using moment penalization one can always attain a bound that depends on ∥ f ⋆ -̂ µ ∥ L 2 ( D ) = √ Var( f ⋆ ) + O ( | ̂ µ -µ ⋆ | ). The latter is achieved by simply redefining the function class F in Theorem 9 to be the centered class of losses { ℓ ( θ ( · ) , ̂ g ( · ) , · ) -̂ µ : θ ∈ Θ } . This leads to the algorithm in the theorem statement, which penalizes the centered second moment:

<!-- formula-not-decoded -->

As long as the error in the preliminary estimate is vanishing, i.e. | ̂ µ -µ ⋆ | =: ε n → 0, then the impact of this error on the regret is only of second order, since the final regret bound takes the form.

<!-- formula-not-decoded -->

In more detail, using the bound from Theorem 9, we have

<!-- formula-not-decoded -->

which, using the AM-GM inequality, we can bound by

<!-- formula-not-decoded -->

We are nearly ready to apply Theorem 2, but first we must bound the error term ε n and relate the variance on the right-hand side above to the true variance Var[ ℓ ( θ ⋆ ( · ) , g 0 ( · ); · )]. We bound the error ε n for the estimate ̂ µ using vanilla (non-localized) Rademacher complexity and two-sided uniform convergence arguments over the function class F . In particular, using standard arguments, we can guarantee that with probability at least 1 -δ over the draw of S 3 , we have

<!-- formula-not-decoded -->

Finally, we observe that since γ ↦→ ℓ ( θ ( z ) , γ ; z ) is L -Lipschitz for all θ and z , we have | L D ( θ, ̂ g ) -L D ( θ, g 0 ) | = O ( L ∥ g 0 -̂ g ∥ G ) for all θ ∈ Θ. Hence, we have | L D ( θ ⋆ , g 0 ) -inf θ L D ( θ, ̂ g ) | = O ( L ∥ g 0 -̂ g ∥ G ), and

<!-- formula-not-decoded -->

Finally, we observe that

<!-- formula-not-decoded -->

After another application of the AM-GM inequality, this yields

<!-- formula-not-decoded -->

Note that we have not used orthogonality up to this point. Applying Theorem 2 (using Assumption 5 and Assumption 6) gives the final result.

## M Proofs from Section 5 and Appendix F

## M.1 Notation

We state the guarantees in this section using slightly more refined notation than in Section 5 and Appendix F. In particular, we consider both the case where the target and nuisance classes are parametric and where they are nonparametric. For the nuisance parameter class G , the two cases we consider are:

- a) Parametric case. There exists a constant d 1 such that

<!-- formula-not-decoded -->

- b) Nonparametric case. There exists a constant p 1 such that

<!-- formula-not-decoded -->

Likewise, for the target class, the two cases we consider are:

- a) Parametric case. There exists a constant d 2 such that

<!-- formula-not-decoded -->

- b) Nonparametric case. There exists a constant p 2 such that

<!-- formula-not-decoded -->

## M.2 Preliminaries

Definition 4 (Empirical Rademacher Complexity) . For a real-valued function class F and sample set S = z 1 , . . . , z n , the Rademacher complexity is defined via

<!-- formula-not-decoded -->

where ϵ = ϵ 1 , . . . , ϵ n are i.i.d. Rademacher random variables. We define R n ( F ) = sup S ∈Z n R n ( F , S ) .

We require the following technical lemmas. First is the Dudley entropy integral bound; we use the following form from Srebro et al. (2010) (Lemma A.3). In all results that follow, we use C &gt; 0 to denote an absolute constant whose value may change from line to line.

Lemma 15. For any real-valued function class F ⊆ ( Z → R ) , we have

<!-- formula-not-decoded -->

As a consequence, whenever F takes values in [ -1 , +1] the following bounds hold:

- If H 2 ( F , ε, S ) ≤ O ( ε -p ) , then R n ( F , S ) ≤ r n,p , where r n,p satisfies

<!-- formula-not-decoded -->

- If H 2 ( F , ε, S ) ≤ O ( d log(1 /ε )) , then R n ( F , S ) ≤ C · √ d/n .

We also require the following lemma, which controls the rate at which the empirical L 2 metric converges to the population L 2 metric in terms of metric entropy behavior.

Lemma 16. Let F ⊆ ( Z → [ -1 , +1]) , and let S = z 1: n be a collection of samples in Z drawn i.i.d. from D .

- If H 2 ( F , ε, n ) ≤ O ( ε -p ) for some p , then with probability at least 1 -δ , for all f, f ′ ∈ F ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where

- If H 2 ( F , ε, n ) ≤ O ( d log(1 /ε )) , then with probability at least 1 -δ , for all f, f ′ ∈ F ,

<!-- formula-not-decoded -->

Proof of Lemma 16. Using Lemma 8 and Lemma 9 from Rakhlin et al. (2017), it also holds that with probability at least 1 -4 δ , for all f, f ′ ∈ F , we have

<!-- formula-not-decoded -->

where β = (log(1 /δ ) + log log n ) /n , and where r ⋆ ≤ d 2 log( en/d 2 ) in the parametric case, and r ⋆ ≤ R 2 n ( F ) log 3 ( n ) in the general case. The final result follows by applying the Rademacher complexity bounds from Lemma 15.

Remark 2. Technically, the result in Rakhlin et al. (2017) we appeal to in the proof above is stated for [0 , 1] -valued classes, but it may be applied to our [ -1 , +1] -valued setting by shifting and rescaling the class F (i.e., invoking with F ′ := ( F +1) / 2 ). We appeal to the same reasoning throughout this section, shifting regression targets in the same fashion when necessary.

## M.3 Overview of Proofs

We now sketch the high-level approach behind the main results in Section 5 and Appendix F. The idea is to use out-of-the-box learning algorithms for both the nuisance and target stage. However, which algorithm gives an optimal rate will depend on the complexity of G and Θ. Moreover, some of the algorithms we employ for the target class require new analyses based on orthogonality to bound the error due to nuisance parameter estimation.

First stage. Base on our assumptions on the metric entropy, we can obtain rates by appealing to the following generic algorithms.

- Global ERM : For each t , select

<!-- formula-not-decoded -->

- Skeleton Aggregation/Aggregation of ε -nets (Yang and Barron (1999); Rakhlin et al. (2017); see Appendix M.4 for a formal description): For each t , run the Skeleton Aggregation algorithm with the class G| t on the dataset of instance-target pairs ( w 1 , ( u t ) 1 ) , . . . , ( w n , ( u t ) n ). Let ( ̂ g ) t be the result.

Proposition 6 (Rates for first stage, informal) . Suppose that Assumption 7 or Assumption 11 holds. Then Global ERM guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Skeleton Aggregation guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Here the ˜ O notation suppresses log n , log( K 1 ) , and log( δ -1 ) factors.

A precise version of Proposition 6 and detailed description of the algorithms are given in Appendix M.5. Note that the minimax rate is Ω( n -2 2+ p 1 ) (Yang and Barron, 1999), and so Skeleton Aggregation is optimal for all values of p 1 , while Global ERM is optimal only for p 1 ≤ 2. While these are not the only algorithms in the literature for which we have generic guarantees based on metric entropy (other choices include Star Aggregation (Liang et al., 2015) and Aggregation-of-Leaders (Rakhlin et al., 2017)), they suffice for our goal in this section, which is to characterize the spectrum of admissible rates.

In all applications we study, the dimension K 1 is constant. Nevertheless, studying procedures that jointly learn all output dimensions of G and, in particular, deriving the correct statistical complexity when K 1 is large is an interesting direction for future research and may be practically useful.

Second stage. The idea behind the second-stage rates we provide is that the problem of obtaining a target predictor for the second stage can be solved by reducing to the classical square loss regression setting. We map our setting onto square loss regression by defining auxiliary variables X = w and Y = Γ( g 0 ( w ) , w ), and by defining auxiliary predictor classes

<!-- formula-not-decoded -->

With these definitions, our goal to bound the excess risk in, e.g., Theorem 5 can be equivalently stated as producing a predictor ̂ f ∈ F 0 that enjoys a bound on

<!-- formula-not-decoded -->

which is the standard notion of square loss excess risk used in, e.g., Liang et al. (2015); Rakhlin et al. (2017). Defining, ˜ Y = Γ( ̂ g ( w ) , w ), we can apply any standard algorithm for the class F to the dataset ( X 1 , ˜ Y 1 ) , . . . , ( X n , ˜ Y n ). Note however that, due to the use of ̂ g as a plug-in estimate, predictors produced via Meta-Algorithm 1 will-invoking Definition 1-give a guarantee of the form

<!-- formula-not-decoded -->

where ̂ f and the benchmark f belong to F instead of F 0 . The machinery developed in Section 3 relates the left-hand-side of this expression to the oracle excess risk (143). Depending on the setting, more work is required to show that the right-hand-side of (144) is controlled. This challenge is only present in the well-specified setting. The difficulty is that while the original problem (143) is well-specified in this case, the presence of the plug-in estimator ̂ g in (144) introduces additional 'misspecification'. We show for global ERM and Skeleton Aggregation the right-hand-side of the expression is controlled as well, meaning that Rate D (Θ , S 2 , δ/ 2; ̂ θ, ̂ g ) is not much larger than the rate Rate D (Θ , S 2 , δ/ 2; ̂ θ, g 0 ) that would have been achieved if the true value for the nuisance parameter was known. This achieved by exploiting orthogonality once again.

In the misspecified setting, we can simply upper bound the right-hand side of (144) by the worst-case bound sup g ∈G Rate D (Θ , S 2 , δ/ 2; ̂ θ, g ) and get the desired growth. Since the model is misspecified to begin with, any extra misspecification introduced by using the plugin estimate here is irrelevant. To be precise, the algorithm configuration is as follows.

- For stage one, use Skeleton Aggregation (Yang and Barron, 1999; Rakhlin et al., 2017). If p 1 ≤ 2, global ERM can be used instead.
- For stage two, in the misspecified setting with Θ convex, use global ERM.
- For stage two, in the well-specified setting, we use Skeleton Aggregation, with a new analysis to account for the small amount of 'model misspecification' introduced by the plug-in nuisance estimate ̂ g . If p 1 ≤ 2, global ERM can be used instead; this is because skeleton ERM and global ERM are both optimal for p 1 ≤ 2, even in the presence of nuisance parameters.

## M.4 Skeleton Aggregation

Here we briefly describe the Skeleton Aggregation meta-algorithm for real-valued regression (Yang and Barron, 1999; Rakhlin et al., 2017). The setting is as follows: we receive n examples S = ( X 1 , Y 1 ) , . . . , ( X n , Y n ) ∈ ( X × R ) n i.i.d. from a distribution D . For a function class F ⊆ ( X → R ), we define L D ( f ) = E D ( f ( X ) -Y ) 2 . Our goal is to produce a predictor ̂ f S for which the excess risk L D ( ̂ f ) -inf f ∈F L D ( f ) is small.

We call a sharp model selection aggregate any algorithm that, given a finite collection of M functions f 1 , . . . , f M and n i.i.d. samples, returns a convex combination ̂ f = ∑ M i =1 ν i f i for which

<!-- formula-not-decoded -->

with probability at least 1 -δ . One such model selection aggregate is the star aggregation algorithm of Audibert (2008), which produces a 2-sparse convex combination ̂ f with the property (145) whenever | Y | ≤ 1 almost surely and the functions in f 1 , . . . , f M take values in [ -1 , +1].

We use the following variant of skeleton aggregation, following Rakhlin et al. (2017). Given a dataset S = ( X 1 , Y 1 ) , . . . , ( X n , Y n ), we split it into two equal-sized parts S ′ and S ′′ .

- Fix a scale ε &gt; 0, and let N = H ( F , ε, S ′ ). Let { ̂ f i } i ∈ [ N ] be a collection of functions that realize the cover, and assume the cover is proper without loss of generality. 4
- Let ̂ f be the output of the star aggregation algorithm run with the collection { ̂ f i } i ∈ [ N ] on the dataset S ′′ .

4 Any improper ε -cover can be made into a proper 2 ε -cover.

For a simple analysis of this algorithm, see Section 6 of Rakhlin et al. (2017). In general, the algorithm is optimal only in the well-specified setting in which E [ Y | X ] = f ⋆ ( X ) for some f ⋆ ∈ F . We give a more refined analysis in the presence of nuisance parameters in the sequel. As final remark, note that since we use a proper cover and the star aggregate is 2-sparse, the final predictor ̂ f lies in the class F ′ := F +star( F - F , 0)

## M.5 Rates for Specific Algorithms

Given an example z ∈ Z , we define an auxiliary example ( ˜ X, ˜ Y ) via ˜ X ( z ) = w , ˜ Y ( z ) = Γ( ̂ g ( w ) , z ). For the remainder of this section we make use of the auxiliary second-stage dataset ˜ S defined via

<!-- formula-not-decoded -->

We make use of the following auxiliary predictor classes:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we define ˜ ℓ ( ̂ y, y ) = ( ̂ y -y ) 2 and ˜ L ( f ) = E ˜ X, ˜ Y ˜ ℓ ( f ( ˜ X ) , ˜ Y ), where ( ˜ X, ˜ Y ) are sampled from the distribution introduced by drawing z ∼ D , and taking ( ˜ X ( z ) , ˜ Y ( z )). With these definitions, observe that for any f ∈ ̂ F of the form ˜ X ↦→⟨ Λ( ̂ g ( w ) , w ) , θ ( x ) ⟩ , we have

<!-- formula-not-decoded -->

We relate the metric entropy of the auxiliary class ̂ F to that of Θ as follows.

Proposition 7. Under Assumption 7, it holds that

<!-- formula-not-decoded -->

Lemma 17 (Rates for first stage) . Global ERM guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where C p 1 is a constant that depends only on p 1 . Skeleton Aggregation guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof of Lemma 17. In what follows we analyze the algorithms under consideration for the class G| i for a fixed coordinate i . The final result follows by union bounding over coordinates and summing the coordinate-wise error bounds we establish.

Global ERM. When we are either in the parametric case or the nonparametric case with p 1 &lt; 2,

the result is given by Theorem 5.2 of Koltchinskii (2011). See Example 3 and Example 4 that follow the theorem for precise calculations under these assumtions. See also Remark 2.

On the other hand, when p 1 ≥ 2 we apply to the standard Rademacher complexity bound for ERM (e.g. Shalev-Shwartz and Ben-David (2014)), which states that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where ℓ square ( g i , u i ) = ( g i ( w ) -u i ) 2 . The result follows by applying Lipschitz contraction to the Rademacher complexity (using that the class is bounded) and appealing to the Rademacher complexity bound from Lemma 15.

Skeleton Aggregation. We appeal to Section 6 of Rakhlin et al. (2017). See Remark 2.

Lemma 18. Consider the plug-in ERM algorithm for the setting in Section 5, i.e.

<!-- formula-not-decoded -->

Under the assumptions of Theorem 5 and Theorem 6, global ERM guarantees

<!-- formula-not-decoded -->

Proof of Lemma 18. To begin, let ̂ f ( ˜ X ) := 〈 Λ( ̂ g ( w ) , w ) , ̂ θ ( x ) 〉 and observe that we can write ̂ f as the global ERM for the auxiliary dataset ˜ S :

<!-- formula-not-decoded -->

Case p 2 &lt; 2 . In the misspecified case we appeal to Theorem 5.1 in Koltchinskii (2011), using that ̂ f is the global ERM for the class ̂ F . To invoke the theorem, we verify that a) ̂ F takes values in [ -1 , +1] under Assumption 7 (see Remark 2), b) ̂ F inherits convexity from Θ, and c) H 2 ( ̂ F , ε, n ) ≤ H 2 (Θ , ε, n ), following Proposition 7. The theorem (see also the following discussion in Example 3 and Example 4) therefore guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

The result now follows from (149), in particular that ˜ L ( ̂ f ) = L D ( ̂ θ, ̂ g ).

Case p 2 ≥ 2 . We apply the standard Rademacher complexity bound for ERM (e.g. Shalev-Shwartz and Ben-David (2014)), which states that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

where we have applied Lipschitz contraction to the Rademacher complexity (using that the class is bounded). To complete the result, we use that H 2 ( ̂ F , ε, n ) ≤ H 2 (Θ , ε, n ) and appeal to the Rademacher complexity bound from Lemma 15.

Lemma 19. Consider the following variant of the Skeleton Aggregation algorithm: 5

- Split S 2 into equal-sized subsets S ′ and S ′′ .
- Fix a scale ε &gt; 0 , and let N = H 2 (Θ , ε, S ′ ) . Let { θ i } i ∈ [ N ] be a collection of functions that realize the cover, and assume the cover is proper without loss of generality. Define f i = ˜ X ↦→⟨ Λ( ̂ g ( w ) , w ) , θ i ( x ) ⟩ for each i ∈ [ N ] .
- Let ̂ θ ∈ Θ+star(Θ -Θ , 0) =: ̂ Θ realize the output of the star aggregation algorithm using the function class { ̂ f i } i ∈ [ N ] on the dataset { ( ˜ X ( z ) , ˜ Y ( z )) } z ∈ S ′′ .

Under the assumptions of Theorem 5, when the model is well-specified, Skeleton Aggregation guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

so long as K 2 = o ( n p 2 2+ p 2 ∧ 4 p 2 (2+ p 2 ) ) .

Proof of Lemma 19. Let ̂ F = { f i } i ∈ [ N ] . The Skeleton Aggregation algorithm as described outputs a predictor ̂ f ∈ ̂ F +star( ̂ F -̂ F , 0) (see Appendix M.4) such that

<!-- formula-not-decoded -->

Translating back into the language of the lemma statement, recall that we can express each f i via f i = X ↦→⟨ Λ( ̂ g ( w ) , w ) , θ i ( x ) ⟩ , with { θ i } i ∈ [ N ] ⊂ Θ since we have assumed a proper cover. Since this parameterization is linear in θ , there must be some ̂ θ ∈ Θ+star(Θ -Θ , 0) that realizes ̂ f . Using the expression for the risk in (149), this implies

<!-- formula-not-decoded -->

Adding and subtracting from both sides, we rewrite the inequality as

<!-- formula-not-decoded -->

Let ∥ θ -θ ′ ∥ 2 Θ = E [ ⟨ Λ( g 0 ( w ) , w ) , θ ( x ) -θ ′ ( x ) ⟩ 2 ] . Observe that by Lemma 6, we have that for all θ, ¯ θ ,

<!-- formula-not-decoded -->

5 See Appendix M.4 for background.

and by Lemma 5, we have that for all θ and ¯ g ,

<!-- formula-not-decoded -->

Hence, since we have assumed orthogonality, Lemma 3 implies that for all i ,

<!-- formula-not-decoded -->

Furthermore, Assumption 7 implies that ∥ θ i -θ 0 ∥ Θ ≤ ∥ θ i -θ 0 ∥ L 2 ( ℓ 2 , D ) . Plugging this bound back into (151), we have that

<!-- formula-not-decoded -->

for constants C, C ′ , C ′′ &gt; 0. We now invoke Lemma 16 for each of the K 2 output coordinates of the target space separately and union bound, which implies that with probability at least 1 -δ over the draw of S ′ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where in the nonparametric case, and

<!-- formula-not-decoded -->

in the parametric case. Returning to the excess risk, this implies

<!-- formula-not-decoded -->

The cover property implies that min i ∈ [ N ] d 2 2 ,S ′ ( θ i , θ 0 ) ≤ ε 2 . So we are left with

<!-- formula-not-decoded -->

Solving for the balance ε 2 ≍ H 2 ( F ,ε,S ) n , leads the first two terms to be of order d 2 log( en/d 2 ) in the parametric case and n -2 2+ p 2 in the nonparametric case. Thus, in the parametric case, the term U dominates and the final bound is CK 2 d 2 log( en/d 2 ) n + C ′ K 2 log( K 2 log n/δ ) n . In the nonparametric case, our assumption on the growth of K 2 implies that U is of lower order.

## M.6 Proofs for Oracle Rates

Proof of Theorem 5. First we invoke Lemma 17, which along with Assumption 8 implies that depending on whether p 1 &gt; 2, one of either global ERM or skeleton aggregation guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Observe that the assumption that the model is well-specified at ( g 0 , θ 0 ) implies that Assumption 2 is satisfied. We invoke Assumption 9 (implied by Assumption 7) and Corollary 1 to get

<!-- formula-not-decoded -->

We now invoke Lemma 19 using that the model is assumed to be well-specified, which implies that with probability at least 1 -δ , Skeleton Aggregation enjoys

<!-- formula-not-decoded -->

Combining these results, we get

<!-- formula-not-decoded -->

The final result follows by setting p 1 to guarantee that the first term dominates this expression.

We mention in passing that to show that global ERM achieves the desired rate for stage two when p 2 ≤ 2, one can appeal to the rates in Appendix L.

Proof of Theorem 6. As in the proof of Theorem 5, we invoke Lemma 17, which implies that Skeleton Aggregation 6 guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Observe that since Θ is convex, Assumption 2 is satisfied, and we can invoke Assumption 9 (implied by Assumption 7) and Corollary 1 to get

<!-- formula-not-decoded -->

We use global ERM for the second stage. Lemma 18 implies that since the class Θ is convex, with probability at least 1 -δ , global ERM guarantees

<!-- formula-not-decoded -->

This leads to a final guarantee of

<!-- formula-not-decoded -->

and the stated result follows by setting p 1 to guarantee that the first term dominates this expression.

6 Alternatively, global ERM can be applied when p 1 ≤ 2.

Proof of Theorem 7. Lemma 17 implies that either Skeleton Aggregation or global ERM (for p 1 ≤ 2) guarantees that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Theorem 2 guarantees

<!-- formula-not-decoded -->

We use global ERM for the second stage. The standard Rademacher complexity bound for ERM (e.g. Shalev-Shwartz and Ben-David (2014)), states that with probability at least 1 -δ , the excess risk is bounded by the Rademacher complexity of the target class Θ composed with the loss class as follows:

<!-- formula-not-decoded -->

Using Lemma 15 and boundedness of the loss, we have

<!-- formula-not-decoded -->

Since the loss is 1-Lipschitz with respect to ℓ 2 , we have H 2 ( ℓ ◦ Θ , ε, S 2 ) ≤ H 2 (Θ , ε, S 2 ). Under the assumed growth of H 2 (Θ , ε, S 2 ) ∝ ε -p 2 , this gives

<!-- formula-not-decoded -->

This leads to a final guarantee of

<!-- formula-not-decoded -->

The theorem statement follows by setting p 1 so that the first term dominates.