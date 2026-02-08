## AUTOMATIC DEBIASED MACHINE LEARNING OF CAUSAL AND STRUCTURAL EFFECTS

VICTOR CHERNOZHUKOV, WHITNEY K. NEWEY, AND RAHUL SINGH

Abstract. Many causal and structural effects depend on regressions. Examples include policy effects, average derivatives, regression decompositions, average treatment effects, causal mediation, and parameters of economic structural models. The regressions may be high dimensional, making machine learning useful. Plugging machine learners into identifying equations can lead to poor inference due to bias from regularization and/or model selection. This paper gives automatic debiasing for linear and nonlinear functions of regressions. The debiasing is automatic in using Lasso and the function of interest without the full form of the bias correction. The debiasing can be applied to any regression learner, including neural nets, random forests, Lasso, boosting, and other high dimensional methods. In addition to providing the bias correction we give standard errors that are robust to misspecification, convergence rates for the bias correction, and primitive conditions for asymptotic inference for estimators of a variety of estimators of structural and causal effects. The automatic debiased machine learning is used to estimate the average treatment effect on the treated for the NSW job training data and to estimate demand elasticities from Nielsen scanner data while allowing preferences to be correlated with prices and income.

Keywords: Debiased machine learning, causal parameters, structural parameters, regression effects, Lasso, Riesz representation.

## 1. Introduction

Many causal and structural parameters of economic interest depend on regressions, i.e. on conditional expectations or least squares projections. Examples include policy effects, average derivatives, regression decompositions, average treatment effects, causal mediation, and parameters of economic structural models. Often, regressions may be high dimensional, depending on many variables. There may be many covariates for policy effects, average derivatives, and treatment effects, or many prices and covariates in the economic demand for some commodity. This paper is about estimating economic and causal parameters that depend on high dimensional regressions.

Date : October 7, 2021.

The present paper formed the basis of the Fisher-Schultz Lecture given by Victor Chernozhukov at the 2019 European Meeting of the Econometric Society in Manchester. This research was supported by NSF grants 1559172 and 1757140. Helpful comments were provided by the editor G. Imbens, three referees, J. Robins, Y. Zhu, and participants at a 2016 demand workshop at Boston College and a 2018 machine learning and statistical inference workshop at the Banff International Research Station.

Machine learning is a collection of modern, adaptive statistical learning methods for estimating regression functions and other statistical objects. These methods exploit structured parsimony restrictions (such as approximate sparsity) on regressions, together with various forms of regularization and model selection, to enable high quality prediction in high dimensional settings. Key methods include neural nets (deep learning), random forests, and Lasso. The goal of this paper is to deploy these methods to infer causal and structural parameters that depend on regression functions, including policy, derivative, decomposition, and treatment effects as well as economic structural parameters.

Machine learning is different than other methods in ways that are useful in high dimensional settings. For example, Lasso has good properties with very many potential regressors (possibly many more than sample size) when relatively few important regressors give a good approximation but the identity of those few is not known (i.e. the regression is approximately sparse). In contrast, series regression is based on relatively few regressors, often many fewer than the sample size. Lasso and series regression are similar in that they both depend on a few regressors giving a good approximation. They differ in that series regression requires that the identity of the important regressors is known, while with Lasso their identity need not be known. For Lasso, the important regressors just need to be included somewhere among the many potential regressors. This difference is useful in high dimensional settings, where there are potentially very many regressors needed to approximate a function of many variables. Typically, economics and statistics provide little guidance about which regressors are important. With Lasso, such information is not needed, since very many terms can be included among the potential regressors. Other machine learning methods, such as random forests and neural nets, are also well suited to high dimensional regression.

Machine learners provide remarkably good predictions in a variety of settings but are inherently biased. The bias arises from using regularization and/or model selection to control the variance of the prediction. To obtain small mean squared prediction errors, machine learners regularize and/or select among models so that variance and squared bias are approximately equal. Although such equality is good for prediction, it is not good for inference. Confidence intervals based on estimators with approximately equal variance and squared bias will tend to have poor coverage. This inference problem can be even worse when machine learners are plugged into a formula for a causal or structural effect. These formulae often involve averaging over regressor values which reduces variance without affecting as much the bias. Variance could also potentially also be a problem but machine learners control that for prediction purposes.

For causal and structural estimators that plug-in regularized machine learners, the squared bias can shrink slower than the variance, leading to extremely poor confidence interval coverage and estimators that are not root-n consistent. Chernozhukov et al. (2017, 2018) give

Lasso and random forest examples respectively and Chernozhukov et al. (2020) shows that Lasso plug-in estimators are not root-n consistent. Model selection inherent in machine learners also creates inference problems. Model selection creates bias from incorrect model choice under local alternatives, making the usual asymptotic confidence intervals invalid over local alternatives, as shown by Leeb and Potscher (2008a,b). Estimators of parameters of interest obtained by plugging in machine learners can inherit this problem, as pointed out by Belloni, Chernozhukov, and Kato (2015) and Chernozhukov, Hansen, and Spindler (2015) and shown in Chernozhukov et al. (2020).

To reduce regularization and model selection bias we use a Neyman orthogonal moment function where there is no first-order effect of the regression on the expected moment function. The orthogonal moment function is constructed by adding to an identifying moment the nonparametric influence function of the regression on the identifying moment function. This construction is model free, nonparametric, and based on the probability limit of the regression learner for any distribution, as in Chernozhukov et al. (2016, 2020). As a result the orthogonality property is model free, meaning that regression learners have no first order effect on the moments for unrestricted, possibly misspecified, nonparametric distributions. Consequently the standard errors are robust to misspecification because they are constructed from the orthogonal moments while ignoring the presence of the regression learners.

The orthogonal moment function depends on another unknown function ¯ α in addition to the regression. We develop a Lasso minimum distance learner of ¯ α that is automatic and nonparametric, in the sense that it depends only on the identifying moment function and not on the form of ¯ α . The structure of the identifying moment function is used to approximate ¯ α as a linear combination of a dictionary (i.e. basis) of known functions. We use the Lasso learner of ¯ α and a regression learner in the orthogonal moment functions to construct an automatic debiased machine learner (Auto-DML) of parameters of interest. We introduce debiased machine learning estimators for a wide variety of effects, including policy effects, average derivatives, bounds on average equivalent variation, and any other linear function of a regression where debiased machine learners were not previously available. We also allow for the identifying moment functions to be nonlinear in regressions. In addition we give novel estimators of average treatment effects, causal mediation, and regression decomposition.

We allow any regression learner, including neural nets, random forests, Lasso, and other high dimensional learners to be used in the orthogonal moment function. The primary requirement of the regression learner is that the product of mean-square convergence rates for the learner of ¯ α and the regression learner is faster than n -1 / 2 . Under this condition and a few other regularity conditions we show root-n consistency and asymptotic normality of the estimator of the parameter of interest. We give convergence rates for the Lasso learner of ¯ α and combine them with existing convergence rates for regressions to verify conditions

for particular estimators. A learner of ¯ α and large sample theory is given for parameters that depend nonlinearly on regressions as well as parameters that are linear in a regression.

The large sample theory in this paper takes the probability limit of the regression learner and ¯ α to be fixed. It would be straightforward to extend the results to allow the regression limit and ¯ α to change with sample size. Such a change would allow us to accommodate sparse specifications where number of nonzero coefficients in the true regression grows with the sample size but would complicate notation and detail. We choose to work with a fixed regression for simplicity while accommodating high dimensional regressions via approximate sparsity.

We give an application to estimating the treatment effect on the treated of job training from the National Supported Work Demonstration (NSW). For many large sets of covariates, we find similar estimates based on neural net, random forest, and Lasso regressions with the automatic bias correction for each. We also give an application to estimating price elasticities from scanner panel data while allowing endogeneity of prices. We estimate the elasticities from Auto-DML of an average derivative that includes many covariates that account for correlated random effects. We find price elasticities that are much smaller than cross-section elasticities, consistent with though larger t than fixed effects elasticities found in Chernozhukov, Hausman, and Newey (2021). We also find that plug in estimates are similar to the cross-section elasticity estimates, so that debiasing is important in this application.

The estimators of parameters of interest use cross-fitting, as in Chernozhukov et al. (2018), where orthogonal moment functions are averaged over groups of observations, the regression and ¯ α learners use all observations not in the group, and each observation is included in the average over one group. Cross-fitting removes a source of bias and eliminates any need for Donsker conditions for the regression learner. Early work by Bickel (1982), Schick (1986), and Klaassen (1987) used similar sample splitting ideas.

Auto-DML for a general linear functional of a regression, convergence rates, and asymptotic normality results for a Dantzig selector of ¯ α and the regression were given in Chernozhukov, Newey, and Robins (2018). Chernozhukov, Newey, and Singh (2018) gave AutoDML for any regression learner, for nonlinear functions of a regression, and convergence rates for a Lasso learner of ¯ α. The current paper is a revised version of Chernozhukov, Newey, and Singh (2018) with a different title. Chernozhukov, Newey, and Singh (2019) is a revised version of Chernozhukov, Newey, and Robins (2018) and is distinguished from the current paper and previous work in giving and analyzing Auto-DML for local (nonparametric) effects as well as focusing on the Dantzig selector for ¯ α and the regression for global effects. All of these papers make use of model free orthogonal moment functions for regression learners given in Chernozuhkov et al. (2016) and the automatic debiasing in Chernozhukov et al. (2020) builds on this paper. The combined use of cross-fitting and orthogonal moment

functions for debiased machine learning is like Chernozhukov et al. (2018). The Auto-DML in Chernozhukov, Newey, and Robins (2018), Chernozhukov, Newey, and Singh (2018), and here innovates by not requiring an explicit formula for the bias correction that is required in Chernozhukov et al. (2018) and earlier papers.

This work builds upon ideas in classical semi- and nonparametric learning theory with lowdimensional regressions using traditional smoothing methods (Van Der Vaart, 1991; Bickel et al., 1993; Newey 1994; Robins and Rotnitzky, 1995; Van der Vaart, 1998), that do not apply to the current high-dimensional setting. The orthogonal moment functions developed in Chernozhukov et al. (2016) and used here build on previous work on model free orthogonal moment functions. Hasminskii and Ibragimov (1979) and Bickel and Ritov (1988) suggest such estimators for functionals of a density. Newey (1994) develops such scores for densities and regressions from computation of the semiparametric efficiency bound for regular functionals. Doubly robust estimating equations for treatment effects as in Robins, Rotnitzky, and Zhao (1995) and Robins and Rotnitzky (1995) constitute model based orthogonal moment functions and have motivated much subsequent work. Newey, Hsieh, and Robins (1998, 2004) extend model free orthogonal moment functions to any functional of a density or distribution in a low dimensional setting. Model free, orthogonal moments for any learner are given and their general properties derived in Chernozhukov et al. (2016, 2020). We use those model free, orthogonal moment functions for regressions.

This paper also builds upon and contributes to the literature on modern orthogonal/debiased estimation and inference, including Zhang and Zhang (2014), Belloni et al. (2012, 2014a,b), Robins et al. (2013), van der Laan and Rose (2011), Javanmard and Montanari (2014a,b, 2015), Van de Geer et al. (2014), Farrell (2015), Ning and Liu (2017), Chernozhukov et al. (2015), Neykov et al. (2018), Ren et al. (2015), Jankova and Van De Geer (2015, 2016a, 2016b), Bradic and Kolar (2017), Zhu and Bradic (2017a,b). This prior work is about regression coefficients, treatment effects, and semiparametric likelihood models. The objects of interest we consider are different than those analyzed in Cai and Guo (2017). The continuity properties of functionals we consider provide additional structure that we exploit, namely the ¯ α , an object that is not considered in Cai and Guo (2017).

Targeted maximum likelihood was developed by Scharfstein, Rotnitzky, Robins (1999) and Van Der Laan and Rubin (2006). The use of machine learning for these estimators was proposed by Van der Laan and Rose (2011) and large sample theory given by Luedtke and Van Der Laan (2016), Toth and van der Laan (2016), and Zheng et al. (2016). In this paper we give a targeted version of Auto-DML with automatic debiasing that we refer to as Auto-TML. This estimator differs from previous ones in the objects we consider and the use of automatic debiasing in Auto-TML.

Various papers have considered direct estimation of ¯ α for treatment effects, where ¯ α is a Riesz representer that depends on inverse propensity scores. Our work is the first to present a framework for direct estimation of the Riesz representer of a broad class of linear and nonlinear functionals, in a high-dimensional setting, without requiring strong Donsker class assumptions. The earliest reference of which we know is Robins et al. (2007), which gives a linear estimator for ¯ α for only the average treatment effect. Vermeulen and Vansteelandt (2015) base parametric propensity score and regression estimators on double robustness conditions for the average treatment effect. We differ in using a linear approximation to ¯ α , which is restrictive in a parametric setting but is general in high dimensional and/or nonparametric settings. Newey and Robins (2018) present and analyze estimators based on regression splines, while we present and analyze sparse methods for the high-dimensional setting. The Lasso minimum distance learner of ¯ α given in Chernozhukov, Newey, and Singh (2018) and here is a direct estimator of the Riesz representer for a broad class of linear and nonlinear functionals that can be interpreted as being based on orthogonality of the moment functions. Chernozhukov et al. (2020) extends this learner of ¯ α to functions of high dimensional regression quantiles and other objects.

In independent work on treatment effects Avagyan and Vansteelandt (2017) give a model assisted estimator based on regularized first order conditions and Tan (2020) developed a model assisted, multistep method of doubly robust estimation with Lasso type regression learners having standard errors that are robust to misspecification of the regression or propensity score. Smucler, Rotnitzky, and Robins (2019) extended that approach to the linear functionals of a regression considered in Chernozhukov, Newey, and Singh (2018). For treatment effects the estimator we give is single step, allows for any regression learner (e.g. neural nets), is model free, and has correct standard errors if either or both the regression and the propensity score are misspecified. Farrell, Liang, and Misra. (2021) gave a neural nets and model based estimator of the average treatment effect and Wooldridge and Zhu (2020) give a Lasso based debiased machine learner for panel data with correlated random effects that depend on high dimensional regressions. Our results also allow for a neural net regression learner but are model free with specification robust standard error.

Chernozhukov, Newey, and Robins (2018) gave Auto-DML for linear functionals using the Dantzig selector. More recently Hirshberg and Wager (2018) gave estimators for linear functionals based on minimax estimation of sample weights that are consistent for realizations of ¯ α in sample mean square error, rather than a linear approximation to the ¯ α function, in the low dimensional case, using the same orthogonal moment functions considered here. The objects considered by Chernozhukov, Newey, and Robins (2018) include average derivatives. More recently Hirshberg and Wager (2020) gave an average derivative estimator based on debiasing a Lasso regression learner of a single index high dimensional regression and

Rothenhausler and Yu (2019) gave an average derivative estimator using debiased Lasso regression. Singh and Sun (2019) extend the present work to the instrumental variable setting and present estimators of the local average treatment effect, average complier characteristics, and complier counter factual distributions. Previous to the current version of this paper Farbmacher et al. (2020) gave DML (debiased machine learning) for causal mediation. We propose an Auto-DML for causal mediation analysis as an example in Section 5.

In summary, contributions of the paper include the construction of DML for a wide range of interesting policy effects and structural parameters where DML was not previously available. This construction is based on a Lasso minimum distance learner of ¯ α we propose. The debiasing and inference is model free and robust to misspecification and carried out in a single step, unlike previous estimators of average treatment effects. For average treatment and other effects we construct DML for a variety of regression learners, such as neural nets, random forests, or high dimensional methods.

In Section 2 we describe the objects of interest we consider and associated orthogonal moment functions. In Section 3 we give the Lasso learner of ¯ α, the Auto-DML and AutoTML estimators, and a consistent estimator of their asymptotic variance. Section 4 derives mean square convergence rates for the Lasso learner of ¯ α and conditions for root-n consistency and asymptotic normality of Auto-DML and Auto-TML including primitive conditions in examples. Section 5 gives Auto-DML for nonlinear functionals of multiple regressions and as an example develops Auto-DML for causal mediation analysis. Section 6 gives AutoDML for regression decomposition and estimates the average treatment on the treated for the NSW experiment. Section 7 gives Auto-DML estimates of price elasticities that allow for correlated random effects in scanner panel data. Section 8 offers some conclusions and possible extensions.

## 2. Average Linear Effects and Orthogonal Moment Functions

For expositional purposes, in this Section we first consider parameters that depend linearly on a single conditional expectation. To describe such an object, let W denote a data observation, and consider a subvector ( Y, X ′ ) ′ where Y is a scalar outcome with finite second moment and X is a covariate vector. Denote the conditional expectation of Y given X ∈ X as

<!-- formula-not-decoded -->

Let m ( w, γ ) denote a function of the function γ (i.e. a functional of γ ) , where γ denotes a possible conditional expectation function γ : X -→ R , that depends on a data observation w and is linear in γ. We will consider effects of the form

<!-- formula-not-decoded -->

The parameter of interest θ 0 is an expectation of some known formula m ( W,γ ) of a data observation W and a regression γ.

We also give results in later Sections for important parameters having more general forms. In Section 5 we allow m ( W,γ ) to be nonlinear in multiple regressions and propose an estimator of causal effects with mediation. In Section 6 we give estimators of regression decompositions and their properties. These important examples extend the framework of this Section to parameters that are nonlinear in multiple regressions

Several important examples of linear effects are:

Example 1: (Average Policy Effect). An average effect of a counter factual shift in the distribution of regressors from a known F 0 to another known F 1 , when γ 0 does not vary with the distribution of X , is

<!-- formula-not-decoded -->

Here m ( w, γ ) = ∫ γ ( x ) dµ ( x ) which does not depend on w. This policy effect builds on but is different than Stock (1989) in comparing averages over two known distributions rather than the empirical distribution.

Example 2: (Weighted Average Derivative). Here X = ( D,Z ) for a continuously distributed random variable D, γ 0 ( x ) = γ 0 ( d, z ) , ω ( d ) is a pdf, and

<!-- formula-not-decoded -->

where S ( u ) = -ω ( u ) -1 ∂ω ( u ) /∂u is the negative score for the pdf ω ( u ) , the second equality follows by integration by parts, and U is a random variable that is independent of Z with pdf ω ( u ) . This U could be thought of as one simulation draw from the pdf ω ( u ) . Here m ( w, γ ) = S ( u ) γ ( u, x ) where W includes U.

This θ 0 can be interpreted as an average treatment effect on Y of a continuous treatment D in a model where Y = Y ( D ) for a potential outcome stochastic process Y ( d ) that is independent of D conditional on covariates Z. By conditional independence

<!-- formula-not-decoded -->

for ω ( u ) &gt; 0 assuming that the joint pdf of ( D,Z ) is positive where ω ( D ) &gt; 0, as in Chamberlain (1984), Wooldridge (2002), and Blundell and Powell (2004). The E[ Y ( u )] is the average outcome at D = u and is sometimes referred to as the average structural function. Assuming that we can interchange the order of differentiation and integration,

<!-- formula-not-decoded -->

similarly to Imbens and Newey (2009) and Rothenh¨ ausler and Yu (2019), which build on but are different than Powell, Stock, and Stoker (1989). Regarding E[ ∂Y ( u ) /∂u ] as the average treatment effect at u we see that θ 0 is a weighted average treatment effect. Alternatively, θ 0 can be regarded as an average derivative of the average structural function. The averaging over a known pdf ω ( u ) helps fulfill regularity conditions for the Auto-DML developed here that can be used to estimate θ 0 for high dimensional covariates Z.

Example 3: (Average Treatment Effect). In this example X = ( D,Z ) and γ 0 ( x ) = γ 0 ( d, z ), where D ∈ { 0 , 1 } is the treatment indicator and Z are covariates. The object of interest is

<!-- formula-not-decoded -->

If potential outcomes are mean independent of treatment D conditional on covariates Z , then θ 0 is the average treatment effect (Rosenbaum and Rubin, 1983). Here m ( w, γ ) = γ (1 , z ) -γ (0 , z ) .

Example 4: (Average Equivalent Variation Bound). An economic example is a bound on average equivalent variation for heterogenous demand. Here Y is the share of income spent on a commodity and X = ( P 1 , Z ) , where P 1 is the price of the commodity and Z includes income Z 1 , prices of other goods, and other observable variables affecting utility. Let ˇ p 1 &lt; ¯ p 1 be lower and upper prices over which the price of the commodity can change, κ a bound on the income effect, ω ( z ) some weight function, and U a random variable that is uniformly distributed over (ˇ p 1 , ¯ p 1 ) and independent of ( Y, X ) . U can be thought of as one simulation draw from a uniform distribution on (ˇ p 1 , ¯ p 1 ) . The object of interest is

<!-- formula-not-decoded -->

If individual heterogeneity in consumer preferences is independent of X and κ is a lower (upper) bound on the derivative of consumption with respect to income for all individuals, then θ 0 is an upper (lower) bound on the weighted average over consumers of equivalent variation for a change in the price of the first good from ˇ p 1 to ¯ p 1 ; see Hausman and Newey (2016). Here m ( w, γ ) = Λ( u, z ) γ ( u, z ) , where W includes U.

We focus on m ( w, γ ) where there exists a function α 0 ( X ) with E[ α 0 ( X ) 2 ] &lt; ∞ and

<!-- formula-not-decoded -->

By the Riesz representation theorem, existence of such a α 0 ( X ) is equivalent to E[ m ( W,γ )] being a mean-square continuous functional of γ, i.e. E[ m ( W,γ )] ≤ C ‖ γ ‖ for all γ , where ‖ γ ‖ = √ E[ γ ( X ) 2 ] and C &gt; 0 . We will refer to this α 0 ( X ) as the Riesz representer (Rr). Existence of the Rr is equivalent to the semiparametric variance bound for θ 0 being finite,

Table 1. m and Rr for Examples 1-4

| Effect                                                                                        | m ( W,γ )                                                                                       | Riesz Representer                                                                                                                                                              |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Policy Effect Weighted Average Derivative Average Treatment Effect Equivalent Variation Bound | ∫ γ ( x )[ f 1 ( x ) - f 0 ( x )] dx S ( U ) γ ( U,Z ) γ (1 ,Z ) - γ (0 ,Z ) Λ( U,Z ) γ ( U,Z ) | f ( X ) - 1 [ f 1 ( X ) - f 0 ( X )] f ( D &#124; Z ) - 1 ω ( D ) S ( D ) π 0 ( Z ) - 1 D - (1 - π 0 ( Z )) - 1 (1 - D (¯ p 1 - ˇ p 1 ) - 1 f ( P 1 &#124; Z ) - 1 Λ( P 1 ,Z ) |

as stated in Newey (1994) and shown in Hirshberg and Wager (2018) for conditional expectations and in Chernozhukov, Newey, and Singh (2019) more generally for least squares projections. Thus, in assuming existence of α 0 ( X ) we are just assuming that θ 0 has a finite semiparametric variance bound.

Each of Examples 1-4 has such a Rr. Let f ( x ) denote the pdf of X in Example 1, f ( d | z ) the pdf of D conditional on Z in Example 2, π 0 ( z ) = Pr( D = 1 | Z = z ) the propensity score in Example 3, and f ( p 1 | z ) the pdf of P 1 conditional on Z in Example 4. Table 1 summarizes the functional m ( w, γ ) and the Rr in each of the examples:

Equation (2.1) follows in Example 1 by multiplying and dividing by f ( x ) inside the integral, in Example 2 by integration and multiplying and dividing by f ( d | z ), in Example 3 in a standard way for average treatment effects, and in Example 4 by multiplying and dividing by f ( p 1 | z ). For E[ α 0 ( X ) 2 ] &lt; ∞ to hold the denominator must not be too small relative to the numerator in each α 0 ( X ), on average. For instance Example 3 must have E[ { π 0 ( Z )(1 -π 0 ( Z )) } -1 ] &lt; ∞ .

Equation (2.1) implies that the effect of interest can be represented in three different ways, as

<!-- formula-not-decoded -->

where the last equality follows by iterated expectations. Any of these three expressions could be used to estimate θ 0 . We could estimate θ 0 from the first expression using a learner (estimator) of γ 0 . We could also estimate θ 0 from the last expression using a learner of α 0 ( X ) . In addition we could use learners of both γ 0 and α 0 to estimate θ 0 from the middle expression. We focus here on using a learner of γ 0 , though α 0 will be important for the bias correction to follow.

We rely on a regression learner (estimator) ˆ γ of γ 0 to estimate θ 0 . The ˆ γ can be any of a variety of machine learners including neural nets, random forests, Lasso, and other high dimensional methods. All we require is that ˆ γ converge in mean square at a sufficiently fast rate, as specified in Section 4.

Whatever the choice of ˆ γ, estimating θ 0 by plugging ˆ γ into m ( W,γ ) and averaging over observations on W can lead to large biases when ˆ γ involves regularization and/or model

selection, as discussed in the Introduction. For that reason we use an orthogonal moment function for θ 0 , where the regression learner ˆ γ has no first-order effect on the moments. We follow Chernozhukov et al. (2016, 2020) in basing the orthogonal moment function on the probability limit (plim) γ ( F ) of ˆ γ when one observation W has CDF F, where F is unrestricted except for regularity conditions. Here γ ( F ) can be thought of as the plim of ˆ γ under general misspecification, where γ ( F ) need not be the conditional expectation E F [ Y | X ].

The plim γ ( F ) of ˆ γ depends on the learner. For example Lasso, the Dantzig selector, boosting, and other high dimensional methods are based on a sequence of potential regressors X = ( X 1 , X 2 , ... ). These learners have the form

/negationslash

<!-- formula-not-decoded -->

where x = ( x 1 , x 2 , ... ) denotes a possible realization of X . Because each ˆ γ ( X ) is a linear combination of X = ( X 1 , X 2 , ... ) the plim γ ( F ) of ˆ γ will also be a linear combination of X , or at least will be approximated by such a linear combination. Define Γ to be the mean square closure of the set of finite linear combinations of X , i.e. Γ is the set of γ ( X ) such that E[ γ ( X ) 2 ] &lt; ∞ and for every ε &gt; 0 there exists ( β ε j ) ∞ j =1 such that β ε j ′ = 0 for a finite number of j ′ and E[ { γ ( X ) -∑ ∞ j =1 β ε j X j } 2 ] &lt; ε. It will be the case that γ ( F ) ∈ Γ . Because Lasso and other high dimensional methods are being used for least squares prediction of Y it will also be the case that

<!-- formula-not-decoded -->

This γ ( F ) minimizes population least squares criteria over the (mean square closure of) linear combinations of X, i.e. it is the best linear predictor of Y by linear combinations of X. Here γ ( F ) is the infinite dimensional linear regression that is nonparametrically estimated by Lasso and other high dimensional methods.

Neural nets and random forests may have a different γ ( F ). A neural net or random forest is often a nonparametric regression estimator for a finite (but high) dimensional X . In that case

<!-- formula-not-decoded -->

which satisfies equation (2.2) when Γ is the set of all (measurable) functions of X with finite second moment. The plim of Lasso and other high dimensional methods will also be this γ ( F ) if X = ( X 1 , X 2 , ... ) can approximate any function of a fixed set of regressors, but otherwise will not. A third type of learner ˆ γ is one that imposes additivity restrictions on ˆ γ , such as ˆ γ ( X ) = ˆ γ 1 ( X 1 ) + ˆ γ 2 ( X 2 ), allowing for nonparametric learners ˆ γ 1 ( X 1 ) and ˆ γ 2 ( X 2 ) . In that case γ ( F ) will be satisfy equation (2.2) where Γ is the mean square closure of functions that are additive in X 1 and X 2 .

/negationslash

We use the orthogonal moment function from Chernozhukov et al. (2016, 2020) for a regression learner ˆ γ having plim γ ( F ) satisfying equation (2.2) for any linear, closed Γ . The orthogonal moment function is constructed by adding to the identifying moment function m ( w, γ ) -θ the nonparametric influence function of of E[ m ( W,γ ( F ))] . As shown in Newey (1994) the nonparametric influence function of E[ m ( W,γ ( F ))] is

<!-- formula-not-decoded -->

where ¯ γ ( X ) is the solution to equation (2.2) for F = F 0 and ¯ α ∈ Γ satisfies E[ m ( W,γ )] = E[¯ α ( X ) γ ( X )] for all γ ∈ Γ . As in Chernozhukov, Newey, and Singh (2019),

<!-- formula-not-decoded -->

This ¯ α can be thought of as the Riesz representer for the linear functional E[ m ( W,γ )] with domain Γ . Evaluating the nonparametric influence function at possible values γ and α of ¯ γ and ¯ α and adding it to the the identifying moment function gives the orthogonal moment function

<!-- formula-not-decoded -->

The moment function ψ ( w, θ, γ, α ) depends on a possible value α of the unknown function ¯ α as well as a possible value γ of the plim ¯ γ of the regression learner. A learner ˆ α of ¯ α is needed to use this orthogonal moment function to estimate θ 0 . In Section 3 we will describe how to construct ˆ α. In Chernozhukov et al. (2016, 2020) ψ ( w, θ, γ, α ) is shown to be orthogonal without being specific about the form of ˆ α. For exposition we repeat that demonstration here. Consider any γ, α ∈ Γ, representing possible realizations of learners ˆ γ and ˆ α that are in Γ . The well known necessary and sufficient conditions for equation (2.2) with F = F 0 are that E[ α ( X ) { Y -¯ γ ( X ) } ] = 0 for all α ∈ Γ . Therefore

<!-- formula-not-decoded -->

where the second equality follows by equation (2.1) and the third equality by the necessary and sufficient condition for equation (2.3) that E[ { α 0 ( X ) -¯ α ( X ) } γ ( X )] = 0 for all γ ∈ Γ . Here we see that ψ ( w, θ, γ, ¯ α ) 'partials out' γ in the sense that

<!-- formula-not-decoded -->

does not depend on γ . Also equation (2.5) gives an explicit formula showing that the effect of γ and α on E[ ψ ( W,θ,γ, α )] is second order and hence ψ ( W,θ,γ, α ) is orthogonal.

/negationslash

The orthogonality property of ψ ( W,θ,γ, α ) only depends on γ, α ∈ Γ and ¯ γ satisfying equation (2.2). In particular orthogonality does not depend on either ¯ γ being E[ Y | X ] or on ¯ α = α 0 . In this sense orthogonality of ψ ( W,θ,γ, α ) is model free, i.e. nonparametric. Consequently the estimator of θ will be asymptotically normal and standard errors consistent even if either ¯ γ = γ 0 or ¯ α = α 0 or both, which is possible when neither γ 0 ( X ) = E[ Y | X ] nor α 0 ( X ) satisfying equation (2.1) is an element of Γ . This robustness of the standard errors results from the orthogonality of the moments only depending on the ¯ γ limit of the regression estimator, so that the sample average of the estimated orthogonal moment function will be asymptotically equivalent to the sample average at the truth, without any model assumptions.

/negationslash

The orthogonal moment function could also be viewed as the efficient influence function of E[ m ( W, ¯ γ )] which clarifies that the Auto-DML is an efficient semiparametric estimator of E[ m ( W, ¯ γ )]. Viewing ψ ( w, θ, γ, α ) in this way is not useful for debiasing because the results of Chernozhukov et. al. (2016, 2020) already imply model free orthogonality.

The moment function ψ ( w, θ, γ, α ) is doubly robust for estimation of the true parameter θ 0 . Evaluating at θ 0 , ¯ γ, ¯ α and taking the expectation gives

<!-- formula-not-decoded -->

which is zero for ¯ γ = γ 0 or ¯ α = α 0 . Thus E[ ψ ( W,θ 0 , ¯ γ, ¯ α )] = 0, so that the orthogonal moment condition identifies θ 0 , when either ¯ γ ( X ) = E[ Y | X ] or α 0 ( X ) ∈ Γ . These conditions both hold when the regression learner is nonparametric so that Γ is the set of all functions of X with finite second moment. For high dimensional regressions where Γ is the closed linear span of X = ( X 1 , X 2 , ... ) the plim of the learner ˆ γ may not be E[ Y | X ] but the orthogonal moment function still identifies θ 0 when α 0 ( X ) ∈ Γ . That is, θ 0 is identified when α 0 ( X ) can be approximated arbitrarily well in mean square by a linear combination of X. This robustness condition can be interpreted in each of Examples 1-4:

/negationslash

Example 1: For high dimensional ˆ γ, where Γ is the mean square closure of linear combinations of X, E[ ψ ( W,θ 0 , ¯ γ, ¯ α )] = 0 even when ¯ γ ( X ) = E[ Y | X ] if α 0 ( X ) = [ f 1 ( X ) -f 0 ( X )] /f ( X ) ∈ Γ .

/negationslash

Example 2: For high dimensional ˆ γ, where Γ is the mean square closure of linear combinations of X, E[ ψ ( W,θ 0 , ¯ γ, ¯ α )] = 0 even when ¯ γ ( X ) = E[ Y | X ] if α 0 ( X ) = f ( D | Z ) -1 ω ( D ) S ( D ) ∈ Γ .

Example 3: For the average treatment effect where Γ is nonparametric, so that ¯ γ ( X ) = E[ Y | X ] and ¯ α ( X ) = α 0 ( X ) , the orthogonal moment function in equation (2.4) corresponds to the seminal doubly robust moment function of Robins, Rotnitzky, and Zhao (1995). When ˆ γ is high dimensional, with say X = ( DZ, (1 -D ) ˜ Z ) for sequences Z = ( Z 1 , Z 2 , ... ) and ˜ Z = ( ˜ Z 1 , ˜ Z 2 , ... ), with each ˜ Z j a function of Z, the orthogonal moment function is

<!-- formula-not-decoded -->

This orthogonal moment function is different than those previously considered in ¯ α ( X ) being the projection of α 0 ( X ) on Γ rather than α 0 ( X ). Here E[ ψ ( W,θ 0 , ¯ γ, ¯ α )] = 0 if linear combinations of Z and ˜ Z can approximate abitrarily well π 0 ( Z ) -1 and [1 -π 0 ( Z )] -1 respectively, even when ¯ γ ( X ) = E[ Y | X ] .

/negationslash

For brevity we omit further discussion of Example 4 from the paper and refer the interested reader to Chernozhukov, Hausman, and Newey (2021).

## 3. Estimation

To estimate (learn) θ 0 we use cross-fitting where the orthogonal moment function ψ ( w, γ, α, θ ) is averaged over observations different than used to estimate ¯ γ and ¯ α. We assume that the data W i , ( i = 1 , ..., n ) are i.i.d.. Let I /lscript , ( /lscript = 1 , ..., L ), be a partition of the observation index set { 1 , ..., n } into L distinct subsets of about equal size. In practice L = 5 (5-fold) or L = 10 (10-fold) cross-fitting is often used. Let ˆ γ /lscript and ˆ α /lscript be estimators constructed from the observations that are not in I /lscript . We construct the estimator ˆ θ by setting the sample average of ψ ( W i , θ, ˆ γ /lscript , ˆ α /lscript ) to zero and solving for θ. This ˆ θ and an associated asymptotic variance estimator ˆ V have explicit forms

<!-- formula-not-decoded -->

Any regression learner ˆ γ /lscript can be used here as long as its mean-square convergence rate is a power of 1 /n, as assumed in Section 4. Such a convergence rate is available for neural nets (Chen and White, 1999, Schmidt-Heiber, 2020, Farrell, Liang, and Misra, 2021), random forests (Syrgkanis and Zampetakis, 2020), Lasso (Bickel, Ritov, and Tsybakov, 2009), boosting (Luo and Spindler, 2016), and other high dimensional methods. As a result any of these regression learners can be used to construct an Auto-DML ˆ θ from equation (3.1), in conjunction with a learner ˆ α /lscript of ¯ α.

The correctness of ˆ V relies on consistency of the regression learner ˆ γ /lscript . It would be interesting to investigate whether the finite sample approximation could be improved by using a variance estimator that allowed ˆ γ /lscript to not be consistent because the dimension of the regression grows as fast as the sample size, e.g. as in Cattaneo, Jansson, and Newey (2018).

An alternative estimator of θ 0 can be constructed that extends the targeted maximum likelihood approach of Scharfstein, Rotnitzky, and J.M. Robins (1999) and van der Laan and Rubin (2006) to the objects we consider. This Auto-TML estimator is a plug-in estimator based on a regression learner that has been debiased in a direction specific to the object of interest. This estimator is given by

<!-- formula-not-decoded -->

As with other targeted estimators the plug-in form of Auto-TML allows imposition of constraints through m ( W,γ ). In Section 4 we show that this estimator is asymptotically equivalent to ˆ θ .

To describe ˆ α /lscript let b ( x ) = ( b 1 ( x ) , ..., b p ( x )) be a p × 1 dictionary of functions of x, where p can be large, with each b j ( x ) standardized to have mean 0 and standard deviation 1 , to be further discussed in this Section. For convenience we ignore dependence of b ( x ) on the data in the notation. The learner ˆ α /lscript given here is

<!-- formula-not-decoded -->

where n /lscript is the number of observations in I /lscript and r &gt; 0 is a positive scalar. This ˆ α /lscript is used in equation (3.1) to construct ˆ θ and ˆ V .

To explain and motivate ˆ α /lscript it is notationally convenient to drop the /lscript subscript, with the understanding that ˆ α /lscript is computed using only observations not in I /lscript for each /lscript, as in equation (3.3). It is also notationally convenient to drop the 0 mean normalization of b ( x ) and consider ˆ α having the form

<!-- formula-not-decoded -->

where ˆ ρ is a vector of estimated coefficients.

The ˆ α depends on the choice of dictionary b ( x ) and penalty degree r. For the dictionary we require that each b j ( x ) belongs to the set Γ of possible plims of ˆ γ ( x ) discussed in Section 2 and that linear combinations of the dictionary 'span' Γ .

Assumption 1: b ( x ) = ( b 1 ( x ) , ..., b p ( x )) ′ where i) b j ∈ Γ for all j and ii) for any α ∈ Γ and ε &gt; 0 there is p and ρ ∈ R p such that E[ { α ( X ) -b ( X ) ′ ρ } 2 ] &lt; ε.

One key feature of this condition is that each b j ∈ Γ . This feature allows us to use m ( w, γ ) to construct ˆ α and will guarantee that ˆ α ∈ Γ, as required for the orthogonality shown in equation (2.5). Another key feature is that linear combinations of b ( x ) can approximate anything that belongs to Γ . This feature will lead to ˆ α estimating ¯ α. The link imposed by Assumption 1, between the regression learner ˆ γ and the dictionary b ( x ) used to construct ˆ α, is important for the orthogonality property of ψ ( w, γ, α, θ ) and hence for ˆ θ to be asymptotically normal and ˆ V to be a consistent estimator of the asymptotic variance under general misspecification.

Assumption 1 requires that linear combinations of b ( x ) must be able to approximate any γ in the set of possible plims of ˆ γ and that each b j must be a possible plim of ˆ γ . For Lasso and other high dimensional regression learners where X = ( X 1 , X 2 , ... ) Assumption 1 will be satisfied for

<!-- formula-not-decoded -->

Evidently each element b j ( X ) = X j is an element of Γ and the spanning condition is satisfied because any linear combination of X with a finite number of nonzero coefficients will also be a linear combination of b ( x ) for p large.

We emphasize that b ( X ) is required to approximate only the projection ¯ α ( X ) and not α 0 ( X ) . For instance, in the average treatment effect example ¯ α ( X ) is the projection of the difference of inverse propensity scores on the space spanned by X = ( X 1 , X 2 , ... ) which is naturally approximated by linear combinations of X = ( X 1 , ..., X p ) . Assumption 1 does not require that this b ( X ) approximate the inverse propensity score.

For neural nets, random forests, and other learners that nonparametrically estimate E[ Y | X ] , Assumption 1 will require that a linear combination of b ( X ) can approximate any function of X for large enough p. Such a b ( x ) can be formed from low order multivariate powers of components of x , with a full set of approximating functions included as p grows. In applications one may use a variety of nonlinear functions including powers of transformations of X.

The learner ˆ α also depends on the choice of penalty degree r. An important, useful feature of Lasso is that r = A √ ln( p ) /n for a constant A gives the fastest possible mean square convergence rate for Lasso, that optimally trades off bias and variance. In Appendix A, we describe cross-validation and theoretical methods for choosing the choosing r based on data that have proven stable across several different applications. We also provide R code, available upon request, for the construction of ˆ α ( x ) and ˆ θ .

We can motivate ˆ ρ in ˆ α ( x ) = b ( x ) ′ ˆ ρ as being based on the Riesz representation in equation (2.1) and ¯ α satisfying equation (2.3), which imply that for m ( w, b ) = ( m ( w, b 1 ) , ..., m ( w, b p )) ′ ,

<!-- formula-not-decoded -->

where the last equality is satisfied by b j ∈ Γ , which implies E[ b j ( X ) { α 0 ( X ) -¯ α ( X ) } ] = 0 for each j . We see that the cross moments M between the true, unknown ¯ α ( x ) and the dictionary b ( x ) are equal to the expectation of the known vector of functions m ( w, b ) . Also, the second moment matrix G = E[ b ( X ) b ( X ) ′ ] of the dictionary is an expectation of a known function of the data. Estimating M and G enables learning coefficients ρ of the least squares regression of ¯ α ( X ) on b ( X ) , satisfying M = Gρ. We learn ρ using a Lasso minimum distance objective function to allow for large p . Let

<!-- formula-not-decoded -->

be unbiased estimators of M and G. The coefficient estimator is given by

<!-- formula-not-decoded -->

The estimator ˆ ρ can be interpreted as a minimum distance version of Lasso. Here ˆ M is analogous to ∑ n i =1 Y i b ( X i ) /n in Lasso. The objective function in equation (3.7) can be thought of as the Lasso objective with ∑ n i =1 Y i b ( X i ) /n replaced by ˆ M and ∑ n i =1 Y 2 i /n dropped. In this way the objective function is a penalized approximation to the least squares regression of α 0 ( x ) on b ( x ) , where 2 r ‖ ρ ‖ 1 is the penalty. We refer to this as minimum distance Lasso because ˆ M does not have the product form of Lasso regression.

The learner ˆ α ( x ) of ¯ α ( x ) is automatic in being based on ˆ M and ˆ G , neither of which requires knowledge of the form of ¯ α. In particular, ˆ α ( x ) = b ( x ) ′ ˆ ρ does not depend on plugging in nonparametric estimates of components of ¯ α ( x ) . Instead, b ( x ) ′ ˆ ρ is linear in the dictionary b ( x ) and uses the known functional m ( w, γ ) in the construction of ˆ M to obtain the learner ˆ ρ. This automatic nature of ˆ α ( x ) is especially useful for Lasso and other high dimensional regression learners where b ( x ) can be taken to be the first p elements of x = ( x 1 , x 2 , ... ) , and where ¯ α ( x ) is a least squares projection of α 0 ( X ) on Γ , as in Section 2. The projection ¯ α ( x ) will generally not have a simple form that can be learned by plugging in nonparametric learners to an explicit formula. For instance, in the average treatment effect example the projection of the inverse propensity score on the high dimensional regressors ( X 1 , X 2 , ... ) does not have a closed form but is naturally approximated by a linear combination of the first p regressors where b ( X ) = ( X 1 , ..., X p ) ′ .

The learner ˆ α ( x ) = b ( x ) ′ ˆ ρ also avoids inverting a learner of a conditional probability or pdf. The finite sample properties of methods that rely on inverses of learners can be

poor; see Singh and Sun (2019) for recent examples. Instead, ˆ α approximates and learns ¯ α by a linear combination of functions. In this way the ˆ α that we propose here avoids potential instability from inverting a high dimensional estimator. The inverse of a conditional probability or density is present in α 0 ( x ) in all of the examples in this paper. We anticipate that this feature is present quite generally for causal and structural models involving shifts in regressors, because the Rr equation (2.1) involves an expectation with respect to the data distribution rather than the shifted distribution. Thus absence of an inverse of a machine learner in ˆ α may prove to be widely useful. In some economic structural models the linearity of ˆ α in b ( x ) may not be quite as appealing, because inverse densities can have a parametric form and so mitigate the problem of inverting a high dimensional learner. An example is the dynamic discrete choice learner of Chernozhukov et al. (2016, 2020). Also there is more work to be done to see whether this approach has better properties than previously proposed ones in practical settings.

This learner ˆ α ( x ) can be thought of as being based on orthogonality of the moment function with respect to γ. Let τ denote a scalar and b j ( x ) an element of b ( x ). Then by equation (3.6)

<!-- formula-not-decoded -->

Replacing the expectation by a sample average and ¯ α ( X ) by b ( X ) ′ ρ gives

<!-- formula-not-decoded -->

where e j is the jth columin of a p dimensional identity matrix. This sample average is a scaled version of the derivative of objective function in equation (3.7) without the penalty term. The first-order conditions for equation (3.7) will set ˆ ρ so that this object is close to zero, subject to the penalty, i.e. will solve penalized versions of a moment equation. Thus, the Lasso minimum distance learner can be thought of as a method that uses orthogonality of ψ ( W,θ,γ, α ) with respect to γ to learn ¯ α while penalizing to facilitate high dimensional estimation. In Section 6 we use an extension of this approach to construct an Auto-DML when m ( W,γ ) is nonlinear in γ.

To illustrate ˆ α we consider the choice of dictionary and the form of ˆ α for Examples 1-3.

Example 1: If the regression learner ˆ γ is nonparametric the dictionary b ( X ) should also be nonparametric while if ˆ γ is a high dimensional regression the dictionary should be chosen as in equation (3.5). Here m ( w, b ) = ∫ b ( x )[ f 1 ( x ) -f 0 ( x )] dx does not depend on the data observation w and the first order conditions for ˆ ρ imply that for each j ,

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ Here ˆ α /lscript ( X i ) acts to approximately re-weight so that the integral of the basis function b j ( x ) over the policy shift is approximately equal to the sample average of the re-weighted basis function b j ( X i )ˆ α /lscript ( X i ) .

Example 2: The dictionary b ( X ) should be chosen as in Example 1. Also by m ( w, b ) = S ( u ) γ ( u, z ) the first order conditions for ˆ ρ imply that for each j ,

<!-- formula-not-decoded -->

∣ ∣ Here ˆ α /lscript ( X i ) acts approximately as a re-weighting scheme, making the sample average of the score S ( U i ) times the basis function b j ( U i , Z i ) be approximately equal to the sample average of the re-weighted basis function b j ( X i )ˆ α /lscript ( X i ) .

Example 3: The dictionary should be chosen similarly to Example 1. For instance suppose that X = ( DZ, (1 -D ) Z ), where Z = ( Z 1 , Z 2 , ... ) is a sequence or possible covariates. Then the dictionary

<!-- formula-not-decoded -->

would satisfy Assumption 1. The estimator ˆ α /lscript has an interesting form for this dictionary. Note that m ( w, b ) = b (1 , z ) -b (0 , z ) = ( q ( z ) ′ , 0 ′ ) ′ -(0 ′ , q ( z ) ′ ) ′ = ( q ( z ) ′ , -q ( z ) ′ ). Then

Let ˆ ρ 1 /lscript be the estimated coefficients of dq ( z ) and ˆ ρ 0 /lscript be the estimated coefficients of (1 -d ) q ( z ). Then the learner of ¯ α ( X i ) is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ ω 1 /lscripti and ˆ ω 0 /lscripti might be thought of as 'weights.' These weights sum to one if q ( z ) includes a constant but may be negative. The first order conditions for ˆ α are that for each j,

<!-- formula-not-decoded -->

∣ ∣ ∣ ∣ Here ˆ ρ /lscript sets the weights ˆ ω 1 /lscripti and ˆ ω 0 /lscripti to approximately 'balance' the overall sample average with the treated and untreated averages for each element of the dictionary q ( z ) . The constraints of equation (3.9) are like the balancing conditions of Zubizarreta (2015) and Athey,

Imbens, and Wager (2018). The source of these constraints is regularized least squares approximation of ¯ α ( x ) = proj ( π 0 ( z ) -1 d -[1 -π 0 ( z )] -1 (1 -d ) | Z ) by a linear combination of the dictionary b ( x ). The approach of this paper shows that this type of balancing is sufficient to debias any regression learner under regularity conditions in Section 4.

## 4. Large Sample Inference

In this Section, we give mean square convergence rates for the Lasso minimum distance learner of ˆ α and root-n consistency and asymptotic normality results for the learner ˆ θ of the object of interest and its asymptotic variance estimator ˆ V . Let ε n denote a sequence that converges to zero no faster than √ ln( p ) /n and for a random variable a ( W ) let ‖ a ‖ = √ E[ a ( W ) 2 ]

Assumption 2: There exists C &gt; 1 , ξ &gt; 0 such that for each positive integer s ≤ Cε -2 / (2 ξ +1) n there is ¯ ρ with s nonzero elements such that

<!-- formula-not-decoded -->

Here ‖ ¯ α -b ′ ¯ ρ ‖ is the mean square approximation error from using the linear combination b ′ ¯ ρ to approximate ¯ α. This approximate sparsity condition specifies that there is a sparse ¯ ρ , having only s nonzero elements, so that the approximation error is bounded by C ( s ) -ξ . Note that it is not required that ¯ α be equal to linear combination of s terms, i.e. it is not required that ¯ α be strictly sparse. Assumption 2 does allow unknown identity of the elements of b ( x ) that give the approximation rate s -ξ . In this way this condition allows for high dimensional x where statistics and economics do not provide much guidance on which elements of b ( x ) are important.

The ε n in this condition represents a convergence rate for ˆ M and ˆ G that will be no faster than √ ln( p ) /n under the conditions given in the rest of this Section. When s is chosen to be approximately Cε -2 / (2 ξ +1) n , which is the largest s allowed by Assumption 2, s will grow no faster than ( √ n/ ln( p )) 2 / (2 ξ +1) ≤ n 1 / (2 ξ +1) , which grows slower than n. Because p ≥ s is implicitly required by this condition, Assumption 2 puts a quite a weak restriction on p. An important feature of Assumption 2 is that the sparse approximation is based on functions included in the p × 1 dictionary b ( x ). Thus larger values of p give more flexibility and will help Assumption 2 to be satisfied.

Our results will require a convergence rate for ˆ α that is faster than some power of n. Assumption 2 is a natural condition that leads to such a rate. Sufficient conditions for Assumption 2 are well known from the approximation literature when ¯ α ( x ) belongs to a Besov or Holder class of function and linear combinations of b ( x ) can approximate any function of x .

We will also make use of a sparse eigenvalue condition as considered in much of the Lasso literature. Let ρ denote a p × 1 vector, ρ J a J × 1 subvector of ρ, and ρ J c the vector consisting of components of ρ that are not in ρ J . Also for a matrix A let ‖ A ‖ 1 = ∑ i,j | a ij | .

<!-- formula-not-decoded -->

Assumption 3: G = E[ b ( X ) b ( X ) ′ ] has largest eigenvalue bounded uniformly in n and there is C, c &gt; 0 such that for all s ≈ Cε -2 n with probability approaching one

This is a sparse eigenvalue condition that is familiar from the Lasso literature, including Bickel, Ritov, Tsybakov (2009), Belloni and Chernozhukov (2013), and Rudelson and Zhou (2013).

We will work with a dictionary b ( X ) with elements that are uniformly bounded.

Assumption 4 : There is C &gt; 0 such that with probability one sup j | b j ( X ) | ≤ C.

This condition implies a convergence rate of √ ln( p ) /n for ∥ ∥ ∥ ˆ G -G ∥ ∥ ∥ ∞ , where ‖ A ‖ ∞ = max i,j | a ij | for a matrix A = [ a ij ].

Lasso mean square convergence rates are often stated in terms of finite sample bounds. Because the focus of this paper is root-n consistency for ˆ θ and for that we only need convergence at certain powers of n we can simplify the statement of convergence rates without affecting the conditions for ˆ θ by allowing the Lasso regularization value r to shrink slightly slower than ε n . This does lead to approximate sparseness conditions that are strict inequalities on the size of ξ but Bradic et al. (2019) have shown that strict inequalities are necessary for root-n consistent estimation, meaning that there is no loss of generality in these conditions. We also limit the growth of p to be slower than some power of n.

Assumption 5: ε n = o ( r ) , r = o ( n c ε n ) for all c &gt; 0, and there exists C &gt; 0 such that p ≤ Cn C .

We also hypothesize a convergence rate for ˆ M.

<!-- formula-not-decoded -->

∥ ∥ We use this condition to accommodate ˆ M that can depend on the regression learner ˆ γ as needed for Section 5.

Theorem 1: If Assumptions 1 - 6 are satisfied then for all c &gt; 0 ,

<!-- formula-not-decoded -->

This theorem is based on extending Lemmas of Bradic et al. (2019) to allow ε n to shrink slower than √ ln( p ) /n. The extension will be used in Section 5 to obtain convergence rates when ˆ M depends on a nonparametric estimator.

The sparse eigenvalue condition of Assumption 3 seems strong in some settings. It is possible to drop Assumption 3 and Assumption 2 if the following condition is satisfied:

Assumption 7: ¯ α ( X ) = ∑ ∞ j =1 ρ j 0 b j ( X ) , ∑ ∞ j =1 | ρ j 0 | &lt; ∞ , and for C &gt; 0 and ¯ s = C √ n the b j ( x ) corresponding to the largest ¯ s values of | ρ j 0 | are included in b ( x ) .

This condition allows us to drop Assumption 2 because absolute summability of the coefficients ρ 0 j implies a sparse approximation rate of ξ = 1 / 2 . It also allows ˆ G to converge at a rate slower ε n in order to accommodate nonparametric estimation in ˆ G.

Theorem 2: If Assumptions 1 and 5-7 are satisfied and ∥ ∥ ∥ ˆ G -G ∥ ∥ ∥ ∞ = O p ( ε n ) then for all c &gt; 0 ,

<!-- formula-not-decoded -->

This result extends Chatterjee and Javarov (2015) to allow ε n to shrink slower than √ ln( p ) /n. When ε n = √ ln( p ) /n in Assumption 6 this result gives a mean square convergence rate for ˆ α that is faster than n -1 / 4+ c for all c &gt; 0 , without a sparse eigenvalue condition.

We now use these results to obtain root-n consistency and asymptotic normality for the Auto-DML ˆ θ and consistency of its asymptotic variance estimator ˆ V . We impose some additional regularity conditions.

Assumption 8 : There is C &gt; 0 such that with probability one max j ≤ p | m ( W,b j )) | ≤ C.

Under this condition Assumption 6 will be satisfied with ε n = √ ln( p ) /n. This condition will be satisfied under by Assumption 4 in each of Examples 1-3 under conditions of Corollaries 4-6 to follow.

Assumption 9: E[ { Y -¯ γ ( X ) } 2 | X ] and ¯ α ( X ) are bounded.

Weimpose this condition for simplicity; it could be weakened. Wealso impose the following condition.

Assumption 10: E[ m ( W,γ 0 ) 2 ] &lt; ∞ and ∫ [ m ( w, ˆ γ ) -m ( w, ¯ γ )] 2 F W ( dw ) p -→ 0 .

This condition will be implied by existence of C &gt; 0 with | E[ m ( W,γ ) 2 ] | ≤ C ‖ γ ‖ 2 for all γ , which will be satisfied in the examples we consider under regularity conditions to be specified.

Assumption 11: With probability approaching one ˆ γ /lscript ∈ Γ and there is d γ &gt; 0 such that ‖ ˆ γ -¯ γ ‖ = O p ( n -d γ ) and either Assumptions 2 and 3 are satisfied with

<!-- formula-not-decoded -->

or Assumption 7 is satisfied and d γ &gt; 1 / 4 .

This assumption allows ˆ γ to be any learner that converges in mean square at a rate that is some power of n. By Theorem 1, the mean square convergence rate for ˆ α is as close as desired to n -ξ/ (2 ξ +1) . Thus Assumption 11 requires that the product of convergence rates for ˆ α and ˆ γ must go to zero faster than 1 / √ n. This is a rate double robustness condition that appears in earlier low dimensional and high dimensional literatures cited in the introduction. Under Assumptions 2 and 3 a full trade-off in rates between ˆ α and ˆ γ is permitted, since Assumption 11 is satisfied for any ξ if d γ is large enough and for any d γ if ξ is large enough. Under Assumption 7 this trade-off is not present, since d γ &gt; 1 / 4 is required by Assumption 11. Assumption 11 can be dropped if α 0 ( X ) is known and is used in place of ˆ α ( X ) in the construction of ˆ θ in equation (3.1). In that case only mean square consistency of ˆ γ will be required for root-n consistency and asymptotic normality of ˆ θ .

The following gives the large sample inference results for ˆ θ and ˆ V . Define

<!-- formula-not-decoded -->

Here ¯ θ will be the object estimated by ˆ θ when neither of the double robustness conditions ¯ γ ( X ) = E[ Y | X ] nor ¯ α ( X ) ∈ Γ is satisfied.

Theorem 3 : If Assumptions 1-5, and 8-11 are satisfied then √ n ( ˆ θ -¯ θ ) d -→ N (0 , V ) . If in addition Assumption 7 is satisfied then ˆ V p -→ V .

It is possible to construct a consistent estimator of V without Assumption 7 by using a trimmed version of ˆ α /lscript ( x ) but we omit that demonstration to avoid further complicating ˆ V . The conclusion of Theorem 3 implies that asymptotic test statistics and confidence intervals can be formed in the usual manner from ˆ θ and ˆ V . Theorem 3 is proven by using the convergence rate results of Theorem 1 and Theorem 2 to show that the hypotheses of Lemma 15 of Chernozhukuv et al. (2020) are satisfied.

The asymptotic variance V is fixed rather than varying with n because we have chosen to work with i.i.d. data and an approximately sparse regression for simplicity. It would be

straightforward to extend the results to allow the regression to change with sample size in order to accomodate sparse regressions and corresponding variances that change with n .

Under similar conditions as Theorem 3 Auto-TML is also consistent and asymptotically normal.

Corollary 4: If Assumptions 1-5, and 8-11 are satisfied, E[ m ( W,γ ) 2 ] ≤ C ‖ γ ‖ 2 for all γ ∈ Γ , and ¯ α ( X ) = 0 then √ n ( ˜ θ -¯ θ ) d -→ N (0 , V ) .

/negationslash

Most of the conditions of Theorem 3 are quite general, with only Assumptions 8 and 10 pertaining to a particular m ( w, γ ). It is straightforward to specify conditions under which Assumptions 8 and 10 are satisfied for Examples 1-3.

Corollary 5 (Example 1): If Assumptions 1-5, 9, and 11 are satisfied and there is C &gt; 0 such that | [ f 1 ( x ) -f 0 ( x )] /f ( x ) | ≤ C then √ n ( ˆ θ -¯ θ ) d -→ N (0 , V ) . If in addition Assumption 7 is satisfied then ˆ V p -→ V .

The specific regularity condition for the policy effect in Corollary 5 is that the Rr α 0 ( X ) = [ f 1 ( X ) -f 0 ( X )] /f ( x ) be bounded.

Corollary 6 (Example 2): If Assumptions 1-5, 9, and 11 are satisfied and there is C &gt; 0 such that | S ( u ) | ≤ C , f ( D | Z ) -1 ω ( D ) ≤ C then √ n ( ˆ θ -¯ θ ) d -→ N (0 , V ) . If in addition Assumption 7 is satisfied then ˆ V p -→ V .

The regularity conditions for the weighted average derivative in Corollary 6 are that the score S ( u ) is bounded and the Rr α 0 ( X ) = f ( D | Z ) -1 ω ( D ) S ( D ) is also bounded.

Corollary 7 (Example 3): If Assumptions 1, 4-5, 9, and 11 are satisfied and there is C &gt; 0 with π 0 ( Z ) ∈ [ C, 1 -C ] then √ n ( ˆ θ -¯ θ ) d -→ N (0 , V ) . If in addition Assumption 7 is satisfied then ˆ V p -→ V .

The additional condition in Corollary 7 is that the propensity score is bounded away from 0 and 1, an overlap condition that is common in asymptotic theory for estimators of the average treatment effect. Together Corollaries 5-7 demonstrate how simple primitive conditions involving m ( w, γ ) can be specified so that the Auto-DML ˆ θ of an object of interest will be asymptotically normal and the asymptotic variance estimator ˆ V consistent.

## 5. Nonlinear Effects of Multiple Regressions

Some important effects of interest are expectations of nonlinear functions of multiple regressions. Causal mediation analysis is an important example that we consider in this Section. The regression decomposition in Section 6 is another important example. In this Section we give Auto-DML for such effects. Such effects have the form θ 0 = E[ m ( W,γ 0 )] where m ( w, γ ) is nonlinear in a possible value γ of multiple regressions ( γ 1 ( X 1 ) , ..., γ K ( X K )) ′ with regressors X k specific to each regression γ k ( X k ). The corresponding orthogonal moment functions are like those discussed in Section 3 except that the bias correction is a sum of K terms with the k th term being the bias correction for the learner of γ k , as in Newey (1994, p. 1357). The estimated bias corrections are like those of Section 4 with the k th term being the product of a Lasso learner ˆ α k/lscript ( X k ) and the residual Y k -ˆ γ k/lscript ( X k ) . Each ˆ α k/lscript ( X k ) differs from Section 3 in the corresponding ˆ M k/lscript being a derivative evaluated at a preliminary estimator of ¯ γ . Because the construction of ˆ θ is so closely related to that in Section 3 we proceed immediately with its description here and fill in details concerning the orthogonal moment function below.

The Auto-DML of a nonlinear effect is similar to equation (3.1). Specifically it is

<!-- formula-not-decoded -->

where each ˆ α k/lscript ( X ki ) is obtained as follows: For each k let b k ( x k ) = ( b k 1 ( x k ) , ...., b kp ( x k )) ′ be a p × 1 dictionary vector specific to the k th regression γ k ( x k ) and let ˆ γ /lscript,/lscript ′ be the vector of regressions computed from all observations not in either I /lscript or I /lscript ′ . Also let τ denote a scalar, and e k the k th column of the K dimensional identity matrix. Then

(5.2)

<!-- formula-not-decoded -->

/negationslash

where b kj denotes the j th element of the dictionary b k ( x k ) as a function of x k . Thus the ˆ α k/lscript ( X i ) in equation (5.1) is a Lasso minimum distance estimator like that of Section 3 that is specific to ˆ γ k and uses the ˆ M k/lscript from equation (5.2) rather than the one in equation (3.3).

The ˆ M k/lscriptj given here generalizes equation (3.3) to allow for nonlinearity of m ( w, γ ) in γ. The derivative with respect to the scalar τ in ˆ M k/lscriptj is generally simple to compute analytically using the chain rule of calculus, as we will illustrate for causal mediation analysis. When m ( w, γ ) is linear in a single γ this derivative just evaluates m ( W i , γ ) at γ = b j , giving the ˆ M /lscriptj of equation (3.3). As with linear m ( w, γ ) the ˆ M k/lscriptj and the rest of the ˆ θ depends just on m ( w, γ ) and the first step. Thus the ˆ θ in equation (5.1) is automatic, in the same way as the estimator of equation (3.1), in only requiring m ( w, γ ) and the regression residuals Y for its construction.

The ˆ M k/lscriptj given here does depend on a cross-fit regression learner ˆ γ /lscript,/lscript ′ in order to allow for the nonlinearity of m ( w, γ ) in γ. The cross-fitting will make the sample average used in the construction of ˆ M k/lscriptj independent of the regression learner ˆ γ /lscript,/lscript ′ used in its construction. This independence helps ˆ M k/lscriptj to be uniformly consistent over j = 1 , ..., p for large p with only mean square convergence convergence rates for ˆ γ /lscript,/lscript ′ . This feature of the theory helps ˆ θ to be root-n consistent and asymptotically normal for a wide variety of regression learners ˆ γ /lscript,/lscript ′ . This ˆ M k/lscriptj was given in Chernozhukov, Newey, and Singh (2018, p. 17). Multiple cross-fitting has also been used in Newey and Robins (2018) and Kennedy (2020).

The dictionary b k ( x k ) used in the construction of ˆ α k/lscript ( x k ) should be chosen analogously to the b ( x ) in Section 3. Each b kj should be an element of the set Γ k of possible plim's of ˆ γ k . Also linear combinations of b k ( x k ) should be able to approximate any element of Γ k arbitrarily well in mean square. That is, Assumption 1 should be satisfied with Γ k and b k ( x ) replacing Γ and b ( x ) respectively. In particular if ˆ γ k is a high dimensional regression then b ( x ) = ( x k 1 , ..., x kp ) ′ will do. If ˆ γ k is a nonparametric estimator then b k ( x k ) should be chosen so that linear combinations can approximate any function of x k .

An important difference between the Lasso minimum distance learner in Section 3 and each ˆ α k/lscript ( x k ) here is that the penalty size r k must be chosen to be larger than √ ln( p ) /n when m ( w, γ ) depends nonlinearly on γ. The reason for larger r k is that ˆ M k/lscript depends on the machine learner ˆ γ /lscript,/lscript ′ and so will converge at a slower rate, leading to a requirement that r k converge to zero slightly slower than the mean square convergence rate of ˆ γ /lscript,/lscript ′ . A choice of r k proportional to n -1 / 4 will generally suffice for this purpose, since ˆ γ /lscript,/lscript ′ will be required to converge faster than n -1 / 4 .

This estimator will not be doubly robust due to the nonlinearity of m ( w, γ ) in γ ; see Chernozhukov et al. (2016). Nevertheless it will have zero first order bias and so be root-n consistent and asymptotically normal under sufficient regularity conditions. It has zero first

order bias because ˆ α k/lscript ( x k ) will consistently estimate ¯ α k ( x k ) such that ∑ K k =1 ¯ α k ( x )[ y k -¯ γ k ( x k )] is the influence function for E[ m ( W,γ ( F ))] at γ ( F ) = ¯ γ where γ ( F ) =plim(ˆ γ ).

Example 5: (Causal Mediation Analysis) Causal mediation analysis provides an interesting example of a nonlinear function of multiple regressions. This effect allows for intermediate variables, called mediators, that lie between treatment and outcome. In this example there is an outcome variable Y , a treatment indicator D ∈ { 0 , 1 } , and covariates Z similar to the average treatment effect in Example 3. In addition there is a mediation variable that we will denote by Q, where we assume that Q ∈ { 1 , ..., K -1) for an integer K ≥ 3 . Let

<!-- formula-not-decoded -->

The causal mediation effect of Imai, Keele, and Tingley (2010, Theorem 1) is

<!-- formula-not-decoded -->

This effect, or parameter, has the form θ 0 ( d, d ′ ) = E[ m ( W,γ )] for W = ( Y, D, Q, Z ) and

<!-- formula-not-decoded -->

In this example we have X k = ( D,Z ), ( k = 1 , ..., K -1) and X K = ( D,Q,Z ) . To construct the Auto-DML ˆ θ we need to choose the dictionaries b k ( X k ) for each k . We choose

<!-- formula-not-decoded -->

to be a nonparametric dictionary if ˆ γ K is a nonparametric estimator such as a neural net or random forest or choose b K ( D,Q,Z ) to be the leading p regressors used in a high dimension regression learner ˆ γ K . For k ≤ K -1 we choose the same dictionary b k ( X k ) = b 1 ( D,Z ) with

<!-- formula-not-decoded -->

for each k ≤ K -1 . We specify b 1 ( D,Z ) to be a nonparametric dictionary if each ˆ γ k is a nonparametric estimator such as a neural net or random forest or choose b 1 ( D,Z ) to be the leading p regressors used in a high dimension regression learner for each ˆ γ k .

It is straightforward to compute each ˆ M k/lscriptj . Note that for k ≤ K -1 ,

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

/negationslash

/negationslash

<!-- formula-not-decoded -->

We can then compute ˆ α k/lscript ( x ) as in equation (5.2) and ˆ θ for Y ki = 1( Q i = k ) , ( k = 1 , ..., K -1) and Y Ki = Y i as in equation (5.1).

The orthogonal moment function corresponding to this estimator is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Γ K is the set of possible plims of ˆ γ K and Γ 1 is the set of plims of ˆ γ k for k ≤ K -1. This moment function differs from the multiply robust moment function of Tchetgen Tchetgen and Shipster (2012) in imposing the constraint that each γ k and α k are contained in the set Γ k of possible plim's of ˆ γ k . For example, when ˆ γ K is a high dimensional regression estimator γ K and α K must be elements of the mean square span of ( X 1 , X 2 , ... ) similarly to Section 2. It has the multiple robustness feature that for ¯ θ = E[ m ( W, ¯ γ )] and any α = ( α 1 , ..., α K ) ∈ Π K k =1 Γ k ,

<!-- formula-not-decoded -->

shown in Chernozhukov et al. (2020) to be a general feature of orthogonal moment functions constructed from the influence function of E[ m ( W,γ ( F ))]. It also has other multiple robustness features. For α k 0 , ( k = 1 , ..., K ) given in the proof of Corollary 10 in the Appendix, when α k 0 ∈ Γ 1 , ( k ≤ K -1) and α K 0 ∈ Γ K ,

<!-- formula-not-decoded -->

for any γ K ∈ Γ K and γ k ∈ Γ 1 , ( k ≤ K -1) .

We now return to the general learner ˆ θ and give regularity conditions for asymptotic normality and consistent estimation of the asymptotic variance of ˆ θ . For ˜ γ = (˜ γ 1 , ..., ˜ γ K ) ′ ∈ Π K k =1 Γ k and γ k ∈ Γ k let

<!-- formula-not-decoded -->

be the Gateaux derivative of m ( W,γ ) with respect to γ k when it exists. Comparing this definition with equation (5.2) we see that each ˆ M kj/lscript is an average of values of this Gateaux derivative. We impose the following condition on these derivatives.

Assumption 12: There are C, ε &gt; 0 , a kj ( w ) , and A k ( w, γ ) such that for all γ with ‖ γ -¯ γ ‖ ≤ ε, D k ( W,b kj , γ ) exists and for k = 1 , ..., K

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This condition and the use of the cross-fit ˆ γ /lscript,/lscript ′ in ˆ M k/lscript lead to a convergence rate for ˆ M k/lscript . Let M kj = E[ D k ( W,b kj , ¯ γ )] and M k = ( M k 1 , ..., M kp ) , ( j = 1 , ..., p ; k = 1 , ..., K ) .

Lemma 8: If there is 0 &lt; d γ &lt; 1 / 2 such that ‖ ˆ γ k/lscript,/lscript ′ -¯ γ k/lscript,/lscript ′ ‖ = O p ( n -d γ ) , ( k = 1 , ..., K ; /lscript, /lscript ′ = 1 , ...L ), and Assumption 12 is satisfied then

<!-- formula-not-decoded -->

∥ ∥ This result can be utilized to obtain mean square convergence rates for ˆ α k from Theorems 1 and 2. As for linear functionals the limit ¯ α k of the estimators ˆ α k are important for the properties of ˆ θ . Here the ¯ α k are associated with the Gateaux derivatives D k ( W,γ k , ¯ γ ) , ( k = 1 , ..., K ) . The following condition specifies each ¯ α k and specifies the size of the remainder in a linearization using the Gateaux derivatives.

Assumption 13: i) For ( k = 1 , ..., K ) there is ¯ α k ∈ Γ k such that for all γ k ∈ Γ k , E[ D k ( W,γ k , ¯ γ )] = E[¯ α k ( X k ) γ k ( X k )]; ii) ¯ α k ( X k ) and E[ { Y k -¯ γ k ( X k ) } 2 | X k ] are bounded; iii) there are ε, C &gt; 0 such that for all γ ∈ Π K k =1 Γ k with ‖ γ -¯ γ ‖ &lt; ε,

<!-- formula-not-decoded -->

Here each ¯ α k is specified as the Riesz representer for the linear functional E[ D k ( W,γ k , ¯ γ )] on γ k ∈ Γ k as in Newey (1994, equation 4.4). Here the linearization E[ D k ( W,γ k , ¯ γ )] has the role that was fulfilled by the linear functional E[ m ( W,γ )] earlier. Indeed when m ( W,γ ) is linear then m ( W,γ ) will be its Gateaux derivative.

From Lemma 8 we see that the convergence rate for each ˆ M k/lscript is the convergence rate n -d γ of ˆ γ rather than √ ln( p ) /n. Consquently conditions for root-n consistency are different in the nonlinear m ( W,γ ) case than in the linear one. The following condition imposes the rate conditions for a nonlinear functional.

Assumption 14: There is 1 / 4 &lt; d γ &lt; 1 / 2 such that ‖ ˆ γ k -¯ γ k ‖ = O p ( n -d γ ) , ( k = 1 , ..., K ) and for ¯ α = ¯ α k and b ( x ) = b k ( x k ) , either i) Assumptions 2 and 3 are satisfied and d γ (1 + 4 ξ ) / (1 + 2 ξ ) &gt; 1 / 2 or ii) Assumption 7 is satisfied and d γ &gt; 1 / 3 .

The requirement d γ &gt; 1 / 4 given here is familiar for estimators that depend nonlinearly on unknown functions, e.g. Newey (1994).. Condition i) allows d γ to be any rate greater than 1 / 4 if ξ is large enough. Condition ii), which drops the sparse eigenvalue assumption but requires absolute summability of the coefficients of each ¯ α k , requires d γ &gt; 1 / 3 .

The following gives the large sample inference results for ˆ θ and ˆ V . Define

<!-- formula-not-decoded -->

Here ¯ θ will be the object estimated by ˆ θ for ¯ γ =plim(ˆ γ ) .

Theorem 9 : If for Γ = Γ k , b ( x ) = b k ( x k ) , r = r k for ( k = 1 , ..., K ) and ε n = n -d γ Assumptions 1, 4, 5, 10, and 12-14 are satisfied then √ n ( ˆ θ -¯ θ ) d -→ N (0 , V ) . If in addition Assumption 7 is satisfied for ¯ α = ¯ α k and b = b k for each ( k = 1 , ..., K ) then ˆ V p -→ V .

Example 6: It is straightforward to specify regularity conditions for causal mediation that are sufficient for the conditions of Theorem 9 to hold.

Assumption 15: ¯ γ k ( X k ) is bounded ( k = 1 , ..., K ) , there is C &gt; 0 such that Pr( D = d, Q = q | Z ) &gt; C for all d ∈ { 0 , 1 } , q ∈ { 1 , ..., K -1 } , and E[ { Y -¯ γ K ( D,Q,Z ) } 2 | D,Q,Z ] ≤ C.

This condition is used to guarantee that ¯ α k ( X k ) is bounded for each k. For brevity the form of ¯ α k ( X k ) and ψ ( w ) is given in the Appendix

Corollary 10: If for Γ = Γ k , b ( x ) = b k ( x k ) , r = r k for ( k = 1 , ..., K ) and ε n = n -d γ Assumptions 1, 4, 5, 14, and 15 are satisfied and there is C &gt; 0 such that | ˆ γ k ( x k ) | ≤ C for all x k then √ n ( ˆ θ -¯ θ ) d -→ N (0 , V ) . If in addition Assumption 7 is satisfied for ¯ α = ¯ α k and b = b k for each ( k = 1 , ..., K ) then ˆ V p -→ V .

The conditions of this result are simple relative to the general regularity conditions in Assumptions 12 and 13. This simplicity is facilitated by m ( W,γ ) being quadratic in γ. The condition that | ˆ γ k ( x k ) | ≤ C is not strong for k = 1 , ..., K -1 because Y ki ∈ { 0 , 1 } . For k = K this restriction could be imposed by truncating ˆ γ k ( x ) for some C larger than a known bound on γ K ( X k ) without affecting Assumption 14. In this way Corollary 10 provides a quite simple set of conditions for Auto-DML of causal mediation effects.

## 6. Regression Decomposition and the Average Treatment Effect on the Treated

In this Section we consider regression decompositions and the average treatment effect on the treated (ATET). We also give an empirical application of the ATET using Auto-DML.

Example 6: (Regression Decomposition and ATET): The effect of some dummy variable D ∈ { 0 , 1 } on an outcome variable Y is often of interest. Regression analysis can be used to decompose the unconditional effect into an effect conditional on covariates and an effect from a shift in the covariate distribution when D shifts. One such decomposition takes the form

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where γ 0 ( D,Z ) = E[ Y | D,Z ]. We will focus here on the response effect

<!-- formula-not-decoded -->

This θ 0 is the average effect of changing D on the outcome Y conditional on Z, averaged over the subpopulation with D = 1 . One could also consider a corresponding effect on the subpopulation with D = 0 . That could also be estimated using Auto-DML similarly to θ 0 but for brevity we omit this discussion.

This θ 0 is also the ATET when D is a treatment indicator and potential outcomes are mean independent of treatment conditional on covariates Z . Thus the estimator ˆ θ and the asymptotic variance estimator ˆ V we give could be applied for inference for the ATET. We do so in the application given later in this Section.

The key regression functional of interest for θ 0 is

<!-- formula-not-decoded -->

Here α 0 ( X ) is the Rr of a linear effect as in Section 2 with m ( w, γ ) = dγ (0 , z ) . The condition E[ α 0 ( X ) 2 ] &lt; ∞ for a finite semiparametric variance bound is E[1 / { 1 -π 0 ( Z ) } ] &lt; ∞ .

The effect θ 0 = ∆ response = ATET is a special case of the nonlinear effect in Section 5 where γ = ( γ 1 , γ 2 ), Y 1 = Y, X 1 = ( D,Z ) , Y 2 = D, X 2 = 1, and

<!-- formula-not-decoded -->

The orthogonal moment function for this object is

<!-- formula-not-decoded -->

where for notational convenience we let y 1 = y, y 2 = d, and γ 1 = γ . Similarly to Section 2 this moment function is doubly robust in that

<!-- formula-not-decoded -->

if either ¯ γ ( X ) = E[ Y | X ] or α 0 ( X ) ∈ Γ.

An Auto-DML is given by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where n D is the number of treated observations and ˆ α /lscript ( x ) is the Lasso learner of the Rr for m ( w, γ ) = dγ (0 , z ) . Similarly to the ATE in Example 3 we specify the dictionary to be b ( x ) = [ dq ( z ) ′ , (1 -d ) q ( z ) ′ ] ′ , where q ( z ) = ( z 1 , ..., z p/ 2 ) ′ when ˆ γ /lscript is high dimensional and q ( z ) is a vector of approximating functions when ˆ γ /lscript is nonparametric. Then m ( w, b j ) = d · b j (0 , z ) = d · 1( j &gt; p/ 2) q j -p/ 2 ( z ) , so that

<!-- formula-not-decoded -->

Then by block diagonality of ˆ G /lscript and the first block of ˆ M /lscript being zero

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first order conditions for the Lasso coefficients ˆ ρ /lscript 2 are

<!-- formula-not-decoded -->

∣ ∣ The ˆ α /lscript learner sets the 'weights' ˆ ω /lscripti to approximately 'balance' the treated and untreated averages for each element of q ( z ) .

Corollary 11 : If i) there is C &gt; 0 with π 0 ( Z ) &lt; 1 -C and ii) Assumptions 1, 4, 5, and 9, 11 are satisfied then for ¯ θ = E[ D { Y -¯ γ (0 , Z ) } ] / Pr( D = 1) and ψ ( W ) = Pr( D =

<!-- formula-not-decoded -->

If Assumption 7 is also satisfied then ˆ V p -→ V.

<!-- formula-not-decoded -->

As an empirical application, we use the Auto-DML of the ATET to estimate the effect of job training in the National Supported Work Demonstration (NSW), a job training program for disadvantaged workers that operated in the mid-1970s. We follow the empirical strategy of LaLonde (1986) and Dehijia and Wahba (1999), who compare the difference-in-means estimator applied to an experimental data set with various econometric estimators applied to 'quasi-experimental' data sets. The experimental data set consists of the treatment and control groups from a field experiment. A quasi-experimental data set consists of the treatment group from a field experiment and a comparison group from an unrelated national survey.

We use sample selection and variable construction as in Dehijia and Wahba (1999) and Farrell (2015). The outcome Y is earnings in 1978. The treatment D is an indicator of participation in job training. We consider three specifications of covariates Z . We impose common support of the propensity score for the treated and untreated groups based on covariates Z as in Farrell (2015). Specifically, we calculate the range of propensity scores for the treated group, and drop observations in the untreated group whose propensity scores lie outside this range. We implement this procedure for each of the three specifications (inducing three different propensity scores), and ultimately keep the untreated observations that pass all three tests. In estimation, we consider the fully-interacted dictionary b ( D,Z ) = (1 , D, Z, DZ ) for all three specifications of Z .

The covariate specifications are as follows.

- (1) Demographics and earnings, with quadratic terms of continuous variables. In particular, the covariates are: age, education, black indicator, Hispanic indicator, married indicator, 1974 earnings, 1975 earnings, age squared, education squared, 1974 earnings squared, and 1975 earnings squared. This specification is moderately flexible. It is one that an analyst may reasonably implement without knowing the experimental benchmark ex ante. Here dim ( Z ) = 11 and p = dim ( b ( D,Z )) = 24.
- (2) Demographics and earnings, with quadratic terms of continuous variables and constructed indicators. In particular, the covariates are: those in specification 1; unemployed in 1974 indicator, unemployed in 1975 indicator, and no degree indicator. This specification includes some domain knowledge about which signals employers may respond to while making hiring decisions. Note that it does not include conveniently hand-crafted basis functions to get closer to the experimental benchmark. Here dim ( Z ) = 14 and p = dim ( b ( D,Z )) = 30.

|   spec. |   treated |   untreated |   Lasso ATET |   Lasso SE |   RF ATET |   RF SE |   NN ATET |   NN SE |
|---------|-----------|-------------|--------------|------------|-----------|---------|-----------|---------|
|       1 |       185 |         172 |      3022.84 |    1278.54 |   3106.55 | 1327.02 |   2585.19 | 1183.85 |
|       2 |       185 |         172 |      2959.72 |    1253.13 |   3077.26 | 1318.67 |   2606.43 | 1020.56 |
|       3 |       185 |         172 |      2289.65 |     836.19 |   2785.13 |  819.17 |   2504.83 |  770.24 |

Table 2. ATET using NSW treatment and NSW control, by Auto-DML

Table 3. ATET using NSW treatment and PSID comparison, by Auto-DML

|   spec. |   treated |   untreated |   Lasso ATET |   Lasso SE |   RF ATET |   RF SE |   NN ATET |   NN SE |
|---------|-----------|-------------|--------------|------------|-----------|---------|-----------|---------|
|       1 |       185 |         727 |       900.58 |     873.62 |   1521.92 |  977.08 |    197.53 |  946.3  |
|       2 |       185 |         727 |      1466.35 |     882.67 |   1336.66 |  956.22 |   1447.65 |  980.73 |
|       3 |       185 |         727 |      1763.2  |    1026.09 |   2010.53 |  987.73 |   2698.55 | 1036.24 |

Table 4. ATET using NSW treatment and CPS comparison, by Auto-DML

|   spec. |   treated |   untreated |   Lasso ATET |   Lasso SE |   RF ATET |   RF SE |   NN ATET |   NN SE |
|---------|-----------|-------------|--------------|------------|-----------|---------|-----------|---------|
|       1 |       185 |        5904 |       703.21 |     583.23 |   1639.95 |  616.08 |   1686.77 |  611.81 |
|       2 |       185 |        5904 |       971.46 |     583.48 |   1584.12 |  616.33 |   1094.86 |  590.31 |
|       3 |       185 |        5904 |      1358.46 |     614.56 |   1906.62 |  651.77 |   2235.09 |  742.86 |

- (3) A high dimensional specification where the covariates are: those in specification 2; all possible first order interactions, and all polynomials up to order five of the continuous variables (age, education, 1974 earnings, 1975 earnings). This specification was introduced by Farrell (2015). Here dim ( Z ) = 171 and p = dim ( b ( D,X )) = 344.

We estimate the Rr with Lasso minimum distance, and the regression with Lasso minimum distance, random forests (RF), or neural networks (NN). For Lasso minimum distance, we use the tuning procedure described in Appendix A. We use the same settings of random forest as Chernozhukov et al. (2018). We implement a neural network with two hidden layers of eight units each and linear activation. We use L = 5 folds in cross-fitting.

Tables 2, 3, and 4 summarize results for the NSW, PSID, and CPS data sets, respectively. For comparison, LaLonde (1986) reports 1794 (633) by difference-in-means applied to the NSW data, which is the experimental benchmark. Farrell (2015) reports 1737 (869) by group Lasso applied to the PSID data using specification 3. Our corresponding estimate is 1763 (1026), and our other results are broadly consistent. To validate the robustness of our results with respect to the choice of tuning procedure, we report analogous tables using cross validated regularization in Appendix C.

## 7. Panel Average Derivative and Demand Elasticities

In this Section, we apply Auto-DML to estimating demand elasticities while allowing for individual preferences that are correlated with prices and total expenditure. Specifically, we estimate own-price elasticity in a panel data model with correlated random slopes. We apply this approach to Nielsen scanner data.

A panel data model requires double indexing. Let Y it , ( t = 1 , ..., T i , i = 1 , ..., n ), denote the share of total expenditure on some good for household i in time period t . Let X it be a vector of log prices, log expenditure, and covariates. Let ˜ X i = ( X ′ i 1 , ..., X ′ i,T i ) ′ collect observations over all time periods for individual i into one vector. We allow for an unbalanced panel where different households may have different numbers of observations T i as in Wooldridge (2019).

Consider the demand model of Chernozhukov, Hausman, and Newey (2021) given by

<!-- formula-not-decoded -->

The K -dimensional dictionary b 1 ( X it ) is a vector of functions of X it that includes a constant and, for example, powers of log price and log expenditure. B it represents household specific preferences that may vary over time and that may be correlated with regressors from each time period. We assume the conditional mean of B it is time stationary with

<!-- formula-not-decoded -->

where I K is a K -dimensional identity matrix. ˜ H i is a vector of functions of ˜ X i with length that does not depend on T i . This panel model is like that of Chamberlain (1982, 1992), Chernozhukov et al. (2013b), Graham and Powell (2012), and Wooldridge (2019), as further discussed in Chernozhukov, Hausman, and Newey (2021).

We will consider identifying and estimating transformations of β 0 = E[ B it ]. β 0 is interpretable as the average marginal effect of changing b 1 ( X it ). The transformations we consider will be interpretable as average income, own-price, and cross-price elasticities. By law of iterated expectations, our model implies

<!-- formula-not-decoded -->

Combining (7.1), (7.2), and (7.3), we summarize the correlated random effects model as follows.

<!-- formula-not-decoded -->

In summary, the choice of K -dimensional dictionary b 1 ( X it ) in the demand model (7.1) induces a p -dimensional dictionary b it = b ( ˜ X i ) = ( b 1 ( X it ) ′ , [ b 1 ( X it ) ⊗ ( ˜ H i -E[ ˜ H i ])] ′ ) ′ in the correlated random effects model (7.4). In practice, we replace E[ ˜ H i ] with 1 n ∑ n i =1 ˜ H i and set ˜ H i = 1 T i ∑ T i t =1 b 1 ( X it ).

Example 10: Demand elasticities. Denote X it = ( D it , Z it ) where D it is log own price. By the derivation in Chernozhukov et al. (2019) for budget share regressions, an average own-price elasticity is

<!-- formula-not-decoded -->

Own-price elasticity θ ∗ 0 is a smooth transformation of a linear effect θ 0 , which in this case is average derivative. Auto-DML of own-price elasticity is then given by

<!-- formula-not-decoded -->

where ˆ θ is the Auto-DML of average derivative from Example 4. Income elasticity and cross-price elasticity have a similar structure; see Appendix B.1 for details.

For completeness, we present ˆ M /lscript for average derivative using the panel data dictionary b it .

<!-- formula-not-decoded -->

Recall Theorem 4 provides consistency and asymptotic normality guarantees for Auto-DML ˆ θ . A more sophisticated estimator ˆ V of the asymptotic variance of ˆ θ is required that accounts for clustering of observations by household. See the Appendix B.1 for details. Importantly, the cluster structure is also preserved in cross-fitting. Clustering methods for DML were previously used by Chiang et al. (2019) and Chernozhukov, Hausman, and Newey (2021). The consistency of the own-price elasticity ˆ θ ∗ follows from the continuous mapping theorem, and the asymptotic normality of ˆ θ ∗ follows from delta method.

As an empirical application, we apply Auto-DML to estimate own-price elasticity of milk and soda with Nielsen scanner data. The empirical work here is the researchers' own analyses calculated (or derived) based in part on data from Nielsen Consumer LLC and marketing databases provided through the NielsenIQ Datasets at the Kilts Center for Marketing Data Center at The University of Chicago Booth School of Business. The conclusions drawn from the NielsenIQ data are those of the researchers and do not reflect the views of NielsenIQ. NielsenIQ is not responsible for, had no role in, and was not involved in analyzing and preparing the results reported herein.

The data we use are a subset of the Nielsen Homescan Panel as in Burda, Harding, and Hausman (2008, 2012). The data include 1483 households from the Houston-area zip codes for the years 2004-2006. The number of monthly observations for each household ranges from 12 to 36, with some households being added and taken away throughout the three years covered. 609 households are included the entire time. Expenditures include all purchases of the household in each month. The original data had time stamps for purchases. If a household purchased a good more than once in a month, the 'monthly price' is the average price that the household paid (i.e. total amount spent on good/total quantity purchased). We include observations with zero expenditure share as justified in Chernozhukov, Hausman, and Newey (2021). For those observations, Y it = 0 and own price is imputed in the ways described in Chernozhukov, Hausman, and Newey (2021).

We consider 15 groups of goods: bread, butter, cereal, chips, coffee, cookies, eggs, ice cream, milk, orange juice, salad, soda, soup, water, and yogurt. As in Burda, Harding, and Hausman (2008, 2012), we choose these groups because they make up a relatively large proportion of total food expenditure. We consider budget share regressions for two of these goods: milk and soda. Y it is share of expenditure spent on milk (soda) by household i in month t . We take as b 1 ( X it ) the concatenation of the following variables: fourth order polynomial of log expenditure; fourth order polynomial of log price for milk (soda); up to fourth order interactions thereof; and log price of other goods. For ˜ H i , we use the time averages of b 1 ( X it ). Note that K = dim ( b 1 ( X it )) = 42 and p = 1521.

We estimate own-price elasticity according to the procedure outlined previously in this Section. We estimate both the Rr and the regression with Lasso minimum distance. For Lasso minimum distance, we use the tuning procedure described in Appendix A. We use L = 5 folds in cross-fitting. We calculate clustered standard errors by delta method, as described in Appendix B.1.

Table 5 summarizes results for the milk and soda own-price elasticities using Auto-DML. For comparison, the cross sectional estimates for milk and soda elasticities are -1 . 27 (0 . 0163) and -0 . 859 (0 . 00485), respectively (Table 1 of Chernozhukov, Hausman, and Newey 2021) and the corresponding fixed effects estimates are -. 739 ( . 0197) and -. 853 ( . 00517). Our results show that allowing for correlated random coefficients lowers these elasticity estimates, especially the milk elasticity. These results confirm the finding in Table 5 of Chernozhukov, Hausman, and Newey (2019), that panel elasticity estimates allowing for correlation of preferences with prices and total expenditure are much smaller than cross-section estimates for milk. Our own-price elasticity estimates are not as small as their slope fixed effect estimates, which for milk are between -0 . 626 (0 . 00849) and -0 . 496 (0 . 0479) and for soda are between -0 . 805 (0 . 00830) and -0 . 780 (0 . 0235) depending on choice of regularization parameter.

Table 5. Average own-price elasticity, by Auto-DML

| good   |   elasticity |      SE |
|--------|--------------|---------|
| milk   |       -0.645 | 0.00649 |
| soda   |       -0.826 | 0.00379 |

Table 6. Average own-price elasticity, by plug-in

| good   |   elasticity |      SE |
|--------|--------------|---------|
| milk   |       -0.863 | 0.00255 |
| soda   |       -0.863 | 0.00305 |

For further comparison, we report results from the plug-in approach in Table 6. The plugin elasticity estimates are much closer to the cross-section estimates than the Auto-DML estimates. The results of this table confirm the importance of debiasing in this application, with debiased estimates differing from plug-in estimates by much more than the associated standard errors.

## 8. Conclusions

In this paper we have given an automatic method of debiasing a machine learner of a parameter of interest that depends on a high dimensional and/or nonparametric regression. The method only requires the form of the object of interest. The regression learners are allowed to be anything that converges in mean square at a fast enough rate. We have shown root-n consistency and asymptotic normality and given a consistent asymptotic variance estimator for a wide variety of causal and structural estimators, including nonlinear functionals of regression. We have applied these methods to estimate the average treatment effect on the treated in a job training experiment and have found similar results for Lasso, neural nets, and random forests regressions. We also have also estimated a correlated random slopes specification for consumer demand from scanner data and found estimates that are similar to fixed slope effect elasticities.

## A.1. Tuning.

A.1.1. Theoretical Procedure. The estimating equation (3.7) takes as given the value of regularization parameter r L . For practical use, we provide an iterative tuning procedure to empirically determine r L . Due to its iterative nature, the tuning procedure is most clearly stated as a replacement for equation (3.7).

Recall that the inputs to equation (3.7) are observations in I c /lscript , i.e. excluding fold /lscript . The analyst must also specify the p -dimensional dictionary b . For notational convenience, we assume b includes the intercept in its first component: b 1 ( x ) = 1. In this tuning procedure, the analyst must further specify a low-dimensional sub-dictionary b low of b . As in equation (3.7), the output of the tuning procedure is ˆ ρ /lscript , an estimator of the Rr coefficient trained only on observations in I c /lscript .

The tuning procedure is as follows. For observations in I c /lscript

- (1) Initialize ˆ ρ /lscript using b low
- (2) Calculate moments

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (3) While ˆ ρ /lscript has not converged
2. (a) Update normalization

<!-- formula-not-decoded -->

## Appendix A. Computing Auto-DML

(b) Update (

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ˆ ˆ

In step 1, b low is sufficiently low-dimensional that G low /lscript is invertible. In practice, we take dim ( b low ) = dim ( b ) / 40.

where ρ j is the j -th coordinate of ρ and D /lscript,j is the j -th diagonal entry of D /lscript . ˆ

In step 3, ( c 1 , c 2 , c 3 ) are hyper-parameters taken as (1 , 0 . 1 , 0 . 1) in practice. We implement the optimization via generalized coordinate descent with soft-thresholding. See below for a detailed derivation of this soft-thresholding routine. We use the same techniques as Chernozhukov, Newey, and Singh (2018) to improve numerical stability in high dimensional settings. We use ˆ D /lscript + 0 . 2 I instead of ˆ D /lscript , and we cap the maximum number of iterations at 10. We also use warm start: in a given iteration, the optimization to determine ˆ ρ /lscript is initialized as the value of ˆ ρ /lscript in the previous iteration.

A.1.2. Cross-Validation Procedure. In the theoretical tuning procedure, the hyperparameters ( c 1 , c 2 , c 3 ) are chosen by the analyst. The hyperparameter c 1 is of particular importance because it scales r L . We now present a procedure to determine c 1 ∈ { 5 / 4 , 1 , 3 / 4 , 1 / 2 } by cross validation.

In the theoretical tuning procedure, denote by r L ( c 1 ) the value of the regularization parameter and denote by ˆ ρ /lscript ( c 1 ) the estimated Rr coefficient that are obtained using hyperparameter value c 1 and observations in I c /lscript . We define the cross-validated loss for the hyperparameter c 1 by

<!-- formula-not-decoded -->

To determine c 1 by cross-validation, we solve the optimization problem

<!-- formula-not-decoded -->

A.1.3. Justification. The iterative tuning procedure is analogous to Algorithm A.1 and therefore justified by an argument analogous to Theorem 1 of Belloni et al. (2012).

The analogy is as follows. The normalization ˆ D /lscript is the square root of the empirical second moment of the dictionary times the regression residual, just as ˆ Γ /lscript in Belloni et al. (2012). The formula for the regularization parameter is the same, after accounting for the fact that the objective in the present work uses r L whereas the objective in eq. 2.4 of Belloni et al. (2012) uses λ n .

## A.2. Optimization.

A.2.1. Procedure. The tuning procedure, an elaboration of estimating equation (3.7), involves the minimization of a generalized Lasso objective. We generalize the coordinate descent approach for Lasso (Fu 1998, Daubechies et al. 2004, Friedman et al. 2007, Friedman et al. 2010) to the minimum distance Lasso objective used in the present work. Specifically, we use the following coordinate-wise soft-thresholding update.

To lighten notation, we abstract from sample splitting, estimation of the moments and normalization, and special treatment of the intercept. We also scale the objective by 1 / 2:

<!-- formula-not-decoded -->

We denote the j -th element of a generic vector V by V j . We denote the ( j, k )-entry of the matrix G by G jk .

For j = 1 : p

- (1) Calculate loadings that do not depend on ρ j

/negationslash

<!-- formula-not-decoded -->

- (2) Update coordinate ρ j

<!-- formula-not-decoded -->

A.2.2. Justification. In this Section, we derive the coordinate-wise soft-thresholding update and argue that the procedure converges to the minimizer.

Observe that and the loadings ( z j , π j ) do not depend on ρ j .

<!-- formula-not-decoded -->

The subgradient of the penalty term is

<!-- formula-not-decoded -->

In summary, the subgradient of the objective is

<!-- formula-not-decoded -->

Rearranging yields the component-wise update.

In our minimum distance Lasso procedure, the objective is of the form of eq. 21 of Friedman et al. (2007).

<!-- formula-not-decoded -->

where g is differentiable and convex and { h k } are convex. Therefore coordinate descent converges to the minimizer of the objective (Tseng, 2001).

A.3. Minimum Distance Lasso Using Simulated Data. We first validate the minimum distance Lasso estimator for ˆ ρ on a design in which the truth is known. We compare our implementation to the Lasso implementation LassoShooting.fit in the hdm package at each point of departure: minimum distance Lasso formulation, theoretical r L , normalization ˆ D , iteration, and stabilization. Altogether, this exercise confirms the validity of each technique introduced in the tuning procedure.

In this design, the ground truth is ρ 0 = (1 , 1 , 1 , 0 , 0 , ... ) where dim ( ρ 0 ) = 101. The data generating process is

<!-- formula-not-decoded -->

where X = (1 , X 1 , ..., X 100 ) ′ , X j i.i.d. ∼ N (0 , 1), and /epsilon1 ∼ N (0 , 1). Recall that the regression coefficient ρ 0 can be recovered by using the functional m ( w, γ ) = yγ ( x ) in the minimum distance Lasso formulation.

In Table 7, we report MSE defined as | ˆ ρ -ρ 0 | 2 2 of various implementations. Table 7 is cumulative in the sense that each row implements one additional technique relative to the preceding row. Before using theoretical r L , we use r L = 0 . 5. We use the estimator reported in the final row in the empirical examples of Sections 6 and 7; it is precisely the estimator defined in the tuning procedure.

Also define ρ ∗ as

<!-- formula-not-decoded -->

Lemma A1: ‖ G ( ρ ∗ -ρ ) ‖ ∞ ≤ ε n .

/negationslash

Proof: Let e j ∈ R p denote the j -th column of I p . The first-order condition for ρ ∗ imply that for j ∈ J 0 , we have e j ′ G ( ρ ∗ -ρ ) = 0; for j ∈ J c 0 , we have that e j ′ G ( ρ ∗ -ρ ) + ε n z j = 0, where z j = sign( ρ ∗ ,j ) if ρ ∗ ,j = 0 and z j ∈ [ -1 , 1] if ρ ∗ ,j = 0. Therefore, for any j , we have that | e j ′ G ( ρ ∗ -ρ ) | ≤ ε n . Hence, ‖ G ( ρ ∗ -ρ ) ‖ ∞ ≤ ε n . /square .

<!-- formula-not-decoded -->

Table 7. 100 simulations

| algorithm             |    MSE |   R 2 |
|-----------------------|--------|-------|
| Lasso                 | 0.006  |  0.17 |
| generalized Lasso     | 0.006  |  0.17 |
| theoretical r L       | 0.0014 |  0.48 |
| normalization ˆ D     | 0.0016 |  0.56 |
| iteration: cold start | 0.0014 |  0.5  |
| iteration: warm start | 0.0014 |  0.5  |
| max iteration         | 0.0014 |  0.5  |
| ˆ D +0 . 2 I          | 0.0014 |  0.46 |

## Appendix B. Proofs of Results

In this Appendix, we give the proofs of the results of the paper, partly based on useful Lemmas that are stated and proved in this Appendix. We first give a series of Lemmas like those in Bradic et al. (2021) except that ε n is allowed to be larger than √ ln( p ) /n in order to allow m ( w, γ ) to be nonlinear in γ. These Lemmas are used to prove Theorem 1. Let ε n be as given in Assumptions 2 and 6 and s 0 ≥ Cε -2 / (2 ξ +1) n . By Assumption 2 we can define J 0 as indices of a sparse approximation with | J 0 | = s 0 and coefficients ˜ ρ j for j ∈ J 0 such that for ˜ α ( x ) = ∑ j ∈ J 0 ˜ ρ j b j ( X ) , E[ { ¯ α ( X ) -˜ α ( X ) } 2 ] ≤ Cs -2 ξ 0 .

Define ρ to be the coefficients of a linear projection of α 0 ( X ) on b ( X ) so that ˘ α ( X ) = b ( X ) ′ ρ satisfies

<!-- formula-not-decoded -->

Proof: Define ˜ ρ = (˜ ρ 1 , . . . , ˜ ρ p ) ′ as

By the definition of ρ ∗ , we have that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that ˜ α ( x ) = b ( x ) ′ ˜ ρ , so by the defintion of ˘ α ( x ) = b ( x ) ′ ρ we have

<!-- formula-not-decoded -->

where the last inequality follows by by s 0 ≥ Cε -2 / (2 ξ +1) n . The result then follows eq. (B.2) by and ε n ∑ j ∈ J c 0 | ρ ∗ ,j | ≥ 0 . /square .

Define J to be the vector of indices of nonzero elements of ρ ∗ and | A | be be the number non zero elements of any finite set A.

Lemma A3: | J | ≤ Cε -2 / (2 ξ +1) n .

Proof: For all j ∈ J \ J 0 the first order conditions to equation (B.1) imply | e ′ j G ( ρ ∗ -ρ ) | = ε n . Therefore, It follows that

In addition,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last inequality follows by Lemma A2 and λ max ( G ) ≤ C. Combining the above two displays, we obtain

<!-- formula-not-decoded -->

Lemma A4: ‖ ˆ Gρ ∗ -Gρ ∗ ‖ ∞ = O p ( √ ln( p ) /n ) .

Proof: By ( ρ -ρ ∗ ) ′ G ( ρ -ρ ∗ ) -→ 0 and ρ ′ Gρ ≤ E[¯ α ( X ) 2 ] it follows that E[ { b ( X ) ′ ρ ∗ } 2 ] = ρ ′ ∗ Gρ ∗ ≤ C. The conclusion then follows by Assumption 4 and Lemma B2 of Bradic et al. (2021). /square .

Lemma A5: For ∆ = ˆ ρ -ρ ∗ and any ˆ J such that ( ρ ∗ ) ˆ J c = 0 , with probability one then with probability approaching one,

<!-- formula-not-decoded -->

Proof: By the definition of the estimator ˆ ρ , we have

<!-- formula-not-decoded -->

Plugging ˆ ρ = ρ ∗ +∆ into the above equation and rearranging the terms gives

<!-- formula-not-decoded -->

By the definition of ρ and M = E[ b ( X )¯ α ( X )] we have Gρ -M = 0 . Then by Assumption 6, Lemma A1, Lemma A4, and the triangle inequality

<!-- formula-not-decoded -->

Therefore, by the Holder inequality we have ∣ ∣ ∣ ( ˆ M -ˆ Gρ ∗ ) ′ ∆ ∣ ∣ ∣ ≤ ‖ ˆ M -ˆ Gρ ∗ ‖ ∞ ‖ ∆ ‖ 1 = O p ( ε n ) ‖ ∆ ‖ 1 , so that by ε n = o ( r ) ,

<!-- formula-not-decoded -->

with probability approaching one. Then the triangle inequality ‖ ρ ∗ ‖ 1 = ‖ ρ ∗ + ∆ -∆ ‖ 1 ≤ ‖ ρ ∗ +∆ ‖ 1 + ‖ ∆ ‖ 1 and subtracting 2 r ‖ ρ ∗ +∆ ‖ 1 from both sides gives the first conclusion.

Next, since ∆ ′ ˆ G ∆ ≥ 0 it also follows from equation (B.3) that 2 r ‖ ρ ∗ +∆ ‖ 1 ≤ 2 r ‖ ρ ∗ ‖ 1 + r ‖ ∆ ‖ 1 , so dividing through by r gives

<!-- formula-not-decoded -->

It follows by ( ρ ∗ ) ˆ J c = 0 that ‖ ρ ∗ + ∆ ‖ 1 = ‖ ( ρ ∗ ) ˆ J + ∆ ˆ J ‖ 1 + ‖ ∆ ˆ J c ‖ 1 and ‖ ρ ∗ ‖ 1 = ‖ ( ρ ∗ ) ˆ J ‖ 1 . Substituting in the previous display then gives

<!-- formula-not-decoded -->

Subtracting 2 ‖ ( ρ ∗ ) ˆ J +∆ ˆ J ‖ 1 + ‖ ∆ ˆ J c ‖ 1 from both sides gives the second conclusion. /square .

<!-- formula-not-decoded -->

Proof: For ˆ J = J it follows from Assumption 3 and Lemma A5 that with probability approaching one,

<!-- formula-not-decoded -->

Dividing through by ‖ ∆ J ‖ 2 then gives with probability approaching one,

<!-- formula-not-decoded -->

Let N denote the indices corresponding to the largest | J | entries in ∆ J c , so that N ⊂ J c , | N | = | J | and | ∆ j | ≥ | ∆ k | for any j ∈ J c ∩ N and k ∈ J c \ N . By Lemma A5 for ˆ J = J ∪ N it follows exactly as in second previous display that

<!-- formula-not-decoded -->

By Lemma 6.9 of van de Geer and Buhlmann (2011) and Lemma A5,

<!-- formula-not-decoded -->

Therefore, by the triangle inequality with probability approaching one,

<!-- formula-not-decoded -->

Proof of Theorem 1: By Lemma A6,

<!-- formula-not-decoded -->

Then by Lemma A2, the triangle inequality, and Assumption 5, for any c &gt; 0 ,

<!-- formula-not-decoded -->

Taking square roots of both sides gives the conclusion. /square .

Next we give a series of Lemmas that are used to prove Theorem 2.

Lemma A7: If Assumption 7 is satisfied then Assumption 2 is satisfied with ξ = 1 / 2 .

Proof: Let J s denote the indices of the s largest coefficients in absolute value and j s ∈ J s be such that | ρ 0 j s | ≤ | ρ 0 j | for all j ∈ J s . Then

<!-- formula-not-decoded -->

By Assumption 7 J s ⊂ { 1 , ..., p } . Define

Let ρ p = ( ρ 01 , ..., ρ 0 p ) and ρ s be the vector with ρ s j = ρ 0 j if j ∈ J s and ρ s j = 0 otherwise. Then by | ρ 0 j | ≤ | ρ 0 j s | for all j / ∈ J s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It then follows by Assumption 7 and the triangle and Cauch-Scwartz inequalities that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define ρ ∗ ∈ arg min ρ { ‖ ¯ α -b ′ ρ ‖ 2 +2 ε n | ρ | 1 } .

Lemma A8: If Assumption 7 is satisfied then

<!-- formula-not-decoded -->

Proof: Note that by ξ = 1 / 2 as in Lemma A7 we have s = ε -2 / (2 ξ +1) n = ε -1 n . By Lemma A7 and the definition of ρ ∗ ,

<!-- formula-not-decoded -->

The conclusion follows from the terms on the left-hand side both being positive. /square .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ Proof: For ∆ = ˆ ρ -ρ ∗ equation (B.3) can be written as

By Lemma A8 ‖ ¯ α -b ′ ρ ∗ ‖ 2 -→ 0 so that E[( b ( X ) ′ ρ ∗ ) 2 ] ≤ C. Then by Assumption 7, Lemma A8, and the Holder inequality it follows that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

∥ ∥ ∥ ∥ Note that the first order conditions for the minimization of

imply that ‖ Gρ ∗ -M ‖ ∞ = O ( ε n ) , similarly to Lemma A1. Then by the triangle inequality,

<!-- formula-not-decoded -->

∥ ∥ ∥ ∥ ∥ ∥ Then by the ∆ ′ ˆ G ∆ ≥ 0, the Holder and triangle inequalities, and dividing equation (B.5) by 2 r we have

<!-- formula-not-decoded -->

∥ ∥ Then noting that o p (1) ‖ ˆ ρ ‖ 1 ≤ (1 / 2) ‖ ˆ ρ ‖ 1 with probability approaching one we have

<!-- formula-not-decoded -->

Proof of Theorem 2: It follows by Lemma A9 that ∥ ∥ ∥ ( G -ˆ G )ˆ ρ ∥ ∥ ∥ ∞ ≤ ∥ ∥ ∥ G -ˆ G ∥ ∥ ∥ ∞ ‖ ˆ ρ ‖ 1 = O p ( ε n ) O p (1) = O p ( ε n ) . Also, the first order conditions for Lasso imply ∥ ∥ ∥ -ˆ G ˆ ρ + ˆ M ∥ ∥ ∥ ∞ ≤ r. Also ∥ ∥ ∥ ˆ M -M ∥ ∥ ∥ ∞ = O p ( ε n ) and ‖-Gρ ∗ + M ‖ ∞ ≤ ε n by the first order conditions for ρ ∗ . Then by the triangle inequality

<!-- formula-not-decoded -->

Then by Lemma A8

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then we have

<!-- formula-not-decoded -->

for any c &gt; 0. Taking square roots of both sides of the inequality gives the conclusion. /square .

Lemma A10: If Assumption 4 is satisfied then ∥ ∥ ∥ ˆ G -G ∥ ∥ ∥ ∞ = O p ( √ ln( p ) /n ) .

<!-- formula-not-decoded -->

Proof: Define ε ∗ n = √ ln( p ) /n and

For any constant C,

<!-- formula-not-decoded -->

Note that E[ T ijk ] = 0 and

<!-- formula-not-decoded -->

Define K = 2 C 2 b / √ ln 2 ≥ ‖ T ijk ‖ Ψ2 . By Hoeffding's inequality (Vershynin, 2018) there is a constant c such that

<!-- formula-not-decoded -->

for any C &gt; K √ 2 /c. Thus for large enough C , Pr( | ˆ G -G | ∞ ≥ C √ ln( p ) /n ) -→ 0, implying the conclusion. /square .

Proof of Theorem 3: The proof proceeds verifying Assumptions 1-3 of Chernozhukov et al. (2020, LR). Assumption 1 i) of LR is implied by Assumption 10. Let φ ( w, γ, α ) = α ( x )[ y -γ ( x )] . Note that by Assumption 9,

<!-- formula-not-decoded -->

giving Assumptions 1 ii) and 1 iii) of LR.

To verify Assumption 2 of LR, note that by Assumption 8 it follows similarly to Lemma A10 that Assumption 6 is satisfied for

<!-- formula-not-decoded -->

Consider first the first case of Assumption 11 where Assumptions 2 and 3 are satisfied. By Theorem 1, for any c &gt; 0 we have

<!-- formula-not-decoded -->

Choose c = [ d γ + ξ/ (2 ξ +1) -1 / 2] / 2 &gt; 0 . Then by Assumption 11,

<!-- formula-not-decoded -->

Consider now the second case of Assumption 11 where Assumption 7 is satisfied. Then for c = (1 / 4 + d γ -1 / 2) / 2 , the conclusion of Theorem 2 gives

<!-- formula-not-decoded -->

Then by the Cauchy-Schwartz and conditional Markov inequalities we have

<!-- formula-not-decoded -->

so that Assumption 2 iii) of LR is satisfied.

To verify Assumption 3 of LR, note that by Assumption 1 ˆ α /lscript ( x ) = b ( x ) ′ ˆ ρ /lscript ∈ Γ , so that

<!-- formula-not-decoded -->

and E[ m ( W,γ ) -¯ θ + ¯ α ( X ) { Y -γ ( X ) } ] is affine in γ, giving Assumption 3 of LR. It then follow by Lemma 15 of LR that

<!-- formula-not-decoded -->

The first conclusion then follows by the central limit theorem.

To show the second conclusion, let ψ i = ψ 0 ( W i ). Then for i ∈ I /lscript ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The first conclusion implies R p -→ 0. Let W -/lscript denote the observations not in I /lscript . By Assumption 10,

<!-- formula-not-decoded -->

By Assumption 4 and Lemma A9, uniformly in x

<!-- formula-not-decoded -->

Then by Assumption 11,

<!-- formula-not-decoded -->

Also by Assumption 9 and iterated expectations

<!-- formula-not-decoded -->

Then by the triangle inequality,

<!-- formula-not-decoded -->

It then follows by the conditional Markov inequality that ∑ i ∈ I /lscript ∑ 3 j =1 R ij /n = o p (1) . The triangle inequality and adding up over /lscript then gives ( ˆ ψ i/lscript -ψ i ) 2

<!-- formula-not-decoded -->

Note also that by Assumptions 9 and 10,

<!-- formula-not-decoded -->

Then

<!-- formula-not-decoded -->

Furthermore by the Cauchy-Schwartz and Markov inequalities we have

<!-- formula-not-decoded -->

∣ ∣ Then ˆ V p -→ V follows by the triangle inequality and the law of large numbers. /square

Proof of Corollary 4: Define ˆ ζ /lscript = [ ∑ i ∈ I /lscript m ( W i , ˆ α /lscript )] / ∑ i ∈ I /lscript ˆ α /lscript ( X i ) 2 . It follows by m ( W,γ ) linear in γ that

<!-- formula-not-decoded -->

It follows from Assumption 11 similarly to the proof of Theorem 3 that there are ν γn -→ 0 and ν αn -→ 0 such that ‖ ˆ γ /lscript -γ 0 ‖ = O p ( ν γn ), ‖ ˆ α /lscript -¯ α ‖ = O p ( ν αn ), and √ nν γn ν αn -→ 0 . We also have

<!-- formula-not-decoded -->

By ˆ α ∈ Γ we have E[ m ( W i , ˆ α /lscript ) -¯ α ( X i )ˆ α /lscript ( X i ) |W -/lscript ] = 0 . Also by ¯ α ( X ) bounded and E[ m ( W,γ ) 2 ] ≤ C ‖ γ ‖ 2 ,

<!-- formula-not-decoded -->

Then by the triangle and conditional Markov inequalities T 1 = O p (1 / √ n ) = O p ( ν αn ) . Also by the Cauchy-Schwartz and conditional Markov inequalities, ‖ ˆ α /lscript ‖ 2 = O p (1) , and ‖ ˆ α /lscript -¯ α ‖ = O p ( ν αn ) we have

<!-- formula-not-decoded -->

/negationslash

Note also that E[¯ α ( X ) 2 ] &gt; 0 by ¯ α ( X ) = 0 and by similar arguments to those previous it follows that i ∈ I /lscript ˆ α /lscript ( X i ) 2 /n 1 p -→ E[¯ α ( X ) 2 ] &gt; 0 . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similar to previous arguments we have | T 1 | = O p (1 / √ n ) = O p ( ν γn ) and | T 2 | = O p ( ν γn ) , so by the triangle inequality ∣ ∣ 1 n ∑ i ∈ I /lscript ˆ α /lscript ( X i )[ Y i -ˆ γ /lscript ( X i )] ∣ ∣ = O p ( ν γn ) . It now follows by equation (B.7) that so that by equation (B.6) and the triangle inequality,

<!-- formula-not-decoded -->

The conclusion then follows by Theorem 3 and the Slutzky Theorem. Q.E.D.

Proof of Corollary 5: Note that by Assumption 4,

<!-- formula-not-decoded -->

so that Assumption 8 is satisfied. Also, | α 0 ( X ) | = | [ f 1 ( x ) -f 0 ( x )] /f ( x ) | ≤ C by hypothesis, so by the Cauchy-Schwartz inequality,

<!-- formula-not-decoded -->

implying Assumption 10. The conclusion then follows by Theorem 3. Q,E.D.

Proof of Corollary 6 : Integration by parts and Assumption 4 give

<!-- formula-not-decoded -->

so Assumption 8 is satisfied. Also

E[ m ( W,γ ) 2 ] = E[ { S ( U ) γ ( U, Z ) } 2 ] ≤ C E[ γ ( U, Z ) 2 ] = E[ f ( D | Z ) -1 ω ( D ) γ ( X ) 2 ] ≤ C E[ γ ( X ) 2 ] , so Assumption 10 is satisfied. The conclusion then follows by Theorem 3. Q,E.D.

Proof of Corollary 7: By Assumption 4 and m ( w, γ ) = γ (1 , z ) -γ (0 , z ) so by the triangle inequality

<!-- formula-not-decoded -->

and Assumption 8 is satisfied. Also

<!-- formula-not-decoded -->

so Assumption 10 is satisfied. The conclusion then follows by Theorem 3. Q,E.D.

Proof of Lemma 8: Define

<!-- formula-not-decoded -->

For notational convenience we henceforth suppress the k superscript. Let -/lscript,/lscript ′ be the event that ‖ ˆ γ /lscript,/lscript ′ -¯ γ ‖ ≤ ε and note that Pr( -/lscript,/lscript ′ ) -→ 1 for each /lscript and /lscript ′ . When -/lscript,/lscript ′ occurs, by Assumption 11. Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that for any constant C ′ and the event A = { max j | U /lscript ′ j (ˆ γ /lscript,/lscript ′ ) | ≥ C ′ ε ∗ n } where ε ∗ n = √ ln( p ) /n

<!-- formula-not-decoded -->

By Lemma B2 of Bradic et al. (2021) there is C ′ large enough that for any δ &gt; 0 with probability approaching one,

<!-- formula-not-decoded -->

Also 1 -Pr(Γ /lscript,/lscript ′ ) -→ 0, so that Pr( A ) &lt; δ for all n large enough. Therefore

<!-- formula-not-decoded -->

/negationslash

∣ ∣ ∣ ∣ Also by Assumption 12 and Pr(Γ /lscript,/lscript ′ ) -→ 1 for each /lscript and /lscript ′ ,

<!-- formula-not-decoded -->

/negationslash

/negationslash

/negationslash

/negationslash

<!-- formula-not-decoded -->

/negationslash

/negationslash

Proof of Theorem 9: The proof proceeds verifying Assumptions 1-3 of Chernozhukov et al. (2020, LR) similarly to the proof of Theorem 3. By Assumption 14, if Assumptions 2 and 3 are satisfied it follows by Lemma 8 that Assumption 6 is satisfied with ε n = n -dγ , so by Theorem 1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then for c = [ d γ (2 ξ/ (2 ξ + 1) + d γ -1 / 2] / 2 = [ d γ (4 ξ + 1) / (2 ξ + 1) -1 / 2] / 2 &gt; 0 we have ‖ ˆ α k -¯ α k ‖ 2 = o p (1) and for each k . Similarly, by Assumption 14 if Assumption 7 is satisfied (rather than Assumptions 2 and 3) then by Theorem 2 for any c &gt; 0 ,

<!-- formula-not-decoded -->

Then for c = [ d γ / 2 + d γ -1 / 2] / 2 &gt; 0 = [3 d γ / 2 -1 / 2] / 2 we have ‖ ˆ α k -¯ α ‖ = o p (1) and

<!-- formula-not-decoded -->

for each k.

Next, Assumption 1 i) of LR is implied by Assumption 10. Let φ k ( w, γ k , α k ) = α k ( x k )[ y k -γ k ( x k )] and

<!-- formula-not-decoded -->

Note that by E[ { Y k -¯ γ ( X k ) } 2 | X k ] and ¯ α k ( X k ) bounded,

<!-- formula-not-decoded -->

so that Assumptions 1 ii) and 1 iii) of LR are satisfied by the triangle inequality.

By the Cauchy-Schwartz and conditional Markov inequalities we have

<!-- formula-not-decoded -->

Then by the triangle inequality Assumption 2 of LR is satisfied.

To verify Assumption 3 of LR, note that by Assumption 1 ˆ α /lscript ( x ) = b ( x ) ′ ˆ ρ /lscript ∈ Γ , so that

<!-- formula-not-decoded -->

Also note that for each k,

<!-- formula-not-decoded -->

Then by Assumption 13 for all γ with ‖ γ -¯ γ ‖ &lt; ε,

<!-- formula-not-decoded -->

It then follows by Lemma 15 of LR that

<!-- formula-not-decoded -->

The first conclusion then follows by the central limit theorem .

The second conclusion follows by the triangle inequality as in the proof of Theorem 3 with R i 2 replaced by ˆ α k/lscript ( X ki ) 2 { ˆ γ k/lscript ( X ki ) -¯ γ k ( X ki ) } 2 and R i 3 by { ˆ α k/lscript ( X ki ) -¯ α k ( X ki ) } 2 { Y ki -¯ γ k ( X ki ) } 2 for each k. /square

Proof of Corollary 10: The proof proceeds by showing that the conditions of Theorem 9 are satisfied. By ¯ γ k bounded for each k and the triangle inequality, E[ m ( W, ¯ γ ) 2 ] &lt; ∞ . Also, by the triangle inequality,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Applying this calculation to γ K = ˆ γ K gives

<!-- formula-not-decoded -->

Also it follows by π ( d, k | Z ) ≥ C for each k that π ( d | Z ) = Pr( D = d | Z ) ≥ C. Then similarly to the previous inequality we have)

<!-- formula-not-decoded -->

Then collecting terms we have

<!-- formula-not-decoded -->

for ‖ γ ‖ = ∑ K k =1 ‖ γ k ‖ . Thus Assumption 10 is satisfied. Next, by the Gateaux derivative formula in the body of the paper for ( k = 1 , ..., K -1) we have

<!-- formula-not-decoded -->

It follows similarly to the verification of Assumption 10 and by Assumption 4 that

<!-- formula-not-decoded -->

Also, we have

<!-- formula-not-decoded -->

which also has the form like that Assumption 12 where the conclusion of Lemma 8 will also be satisfied. The second part of Assumption 12 follows by a similar argument, so that Assumption 12 is satisfied.

Turning now to Assumption 13, note that for ( k = 1 , ..., K -1) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Each of ¯ α k ( X k ) is bounded by π ( d, k | Z ) ≥ C for d ∈ { 0 , 1 } and ( k = 1 , ..., K -1) and ¯ γ k ( X k ) bounded for each k. Similarly these conditions imply that E[ Y ki -¯ γ (

To verify Assumption 13 iii) note that by algebra we have

<!-- formula-not-decoded -->

Therefore by the Cauchy Scwhartz, triangle, arithmetic mean-geometric mean inequalities,

<!-- formula-not-decoded -->

where the last inequality follows similarly to previous results. The conclusion now follows by Theorem 9. /square .

Proof of Corollary 11 : Note first that for any γ ( X ) it follows as in the proof of Corollary 7 that by Pr( D = 1 | Z ) &lt; 1 -C,

Also note that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The remainder of the proof follows analogously to the proof of Corollary 6. /square .

B.1. Panel Average Derivative and Demand Elasticities. Since own-price elasticity θ ∗ 0 is a deterministic mapping of ˜ θ 0 := ( θ 0 , E [ Y it ]) ′ , we obtain the asymptotic variance V ∗ of θ ∗ 0 from the asymptotic variance ˜ V of ˜ θ 0 using delta method. Specifically,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We estimate the asymptotic variance V ∗ using the empirical analogue ˆ V ∗ , where ψ 0 ( W it ) is replaced by

<!-- formula-not-decoded -->

where and

The covariance estimator recognizes that household i 's observations form a cluster T i . For example, the estimator for the first component of ˜ V is

<!-- formula-not-decoded -->

More generally, we may consider estimating not only own price elasticity but also income elasticity and cross price elasticity. The same arguments go through with light modification.

Concatenate the derivatives as

<!-- formula-not-decoded -->

where the first and second components are scalars and the third component is a vector.

The elasticities are a smooth transform thereof. By arguments in Chernozhukov, Hausman, and Newey (2019)

<!-- formula-not-decoded -->

Likewise the delta method argument goes through. Elasticites θ 0 are a deterministic mapping of ˜ θ 0 = (( θ ∗ 0 ) ′ , E [ Y it ]) ′ . We obtain the asymptotic variance V ∗ of θ ∗ 0 from the asymptotic variance ˜ V of ˜ θ 0 using delta method. Specifically,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where and ˜ V is as before, where the influence function ψ 0 is vector-valued, corresponding to the vector θ 0 .

As an aside, when using OLS, the empirical influence function used in estimating offdiagonal terms is

<!-- formula-not-decoded -->

where /epsilon1 it is the OLS residual for observation W it . As before, we use a variance estimator that recognizes clustering.

<!-- formula-not-decoded -->

|   spec. |   treated |   untreated |   Lasso ATET |   Lasso SE |   RF ATET |   RF SE |   NN ATET |   NN SE |
|---------|-----------|-------------|--------------|------------|-----------|---------|-----------|---------|
|       1 |       185 |         172 |      4071.88 |    3390.11 |   4170.99 | 3277.92 |   1807.48 | 2656.05 |
|       2 |       185 |         172 |      1618.74 |     500.49 |   2047.18 |  504.7  |   1754.79 |  531.1  |
|       3 |       185 |         172 |      3379.15 |    1466.45 |   3589.1  | 1385.49 |   1175.21 | 1735.56 |

Table 8. ATET using NSW treatment and NSW control, by cross validation

Table 9. ATET using NSW treatment and PSID comparison, by cross validation

|   spec. |   treated |   untreated |   Lasso ATET |   Lasso SE |   RF ATET |   RF SE |   NN ATET |   NN SE |
|---------|-----------|-------------|--------------|------------|-----------|---------|-----------|---------|
|       1 |       185 |         727 |      2194.07 |    1060.97 |   1986.25 | 1031.62 |    834.13 | 1004.58 |
|       2 |       185 |         727 |      1686.64 |    1092.13 |   1422.68 | 1125.96 |   1909.87 | 1404.22 |
|       3 |       185 |         727 |      2974.55 |    1108.72 |   2579.75 | 1042.94 |   3057.04 | 1454.26 |

Table 10. ATET using NSW treatment and CPS comparison, by cross validation

|   spec. |   treated |   untreated |   Lasso ATET |   Lasso SE |   RF ATET |   RF SE |   NN ATET |   NN SE |
|---------|-----------|-------------|--------------|------------|-----------|---------|-----------|---------|
|       1 |       185 |        5904 |      1413.98 |     636.68 |   1813.82 |  662.06 |   2043.87 |  657.46 |
|       2 |       185 |        5904 |      1405.09 |     644.1  |   1756.57 |  669.68 |   2025.5  |  653.32 |
|       3 |       185 |        5904 |      1756.87 |     654.73 |   2013.84 |  676.72 |   1823.67 |  651.66 |

## Appendix C. Additional Empirical Results

C.1. Regression Decomposition and ATET. We present ATET estimates from AutoDML using cross validation rather than theoretical iteration to tune the regularization. Our results are broadly similar, with larger standard errors.

- C.2. Panel Average Derivative and Demand Elasticities. We present elasticity estimates from OLS with a simpler specification than the specification used in the main text. We take as b 1 ( X it ) the concatenation of the following variables: log expenditure, and log price of each good. For ˜ H i , we use the time averages of b 1 ( X it ). Note that K = dim ( b 1 ( X it )) = 16 and p = dim ( b it ) = 288. We calculate clustered standard errors derived by delta method as explained in Appendix B.1. Tables 11 and 12 summarize results.

Table 11. Milk elasticities, by OLS

| variable     |   elasticity |   SE |
|--------------|--------------|------|
| income       |         0.42 | 0.05 |
| own-price    |        -0.68 | 0.05 |
| bread        |        -0.03 | 0.02 |
| butter       |         0    | 0.02 |
| cereal       |         0    | 0.02 |
| chips        |         0.02 | 0.03 |
| coffee       |         0    | 0.02 |
| cookies      |         0    | 0.02 |
| eggs         |        -0.03 | 0.03 |
| ice cream    |        -0.03 | 0.03 |
| orange juice |        -0.01 | 0.05 |
| salad        |         0.02 | 0.02 |
| soda         |        -0.02 | 0.02 |
| soup         |        -0.03 | 0.02 |
| water        |        -0.01 | 0.02 |
| yogurt       |         0.01 | 0.04 |

## References

Ahn, H. and C.F. Manski (1993): 'Distribution Theory for the Analysis of Binary Choice under Uncertainty with Nonparametric Estimation of Expectations,' Journal of Econometrics 56, 291-321.

Athey, S., G. Imbens, and S. Wager (2018): 'Approximate Residual Balancing: Debiased Inference of Average Treatment Effects in High Dimensions,' Journal of the Royal Statistical Society, Series B 80, 597-623.

Avagyan, V. and S. Vansteelandt (2017): 'Honest data-adaptive inference for the average treatment effect under model misspecification using penalised bias-reduced double-robust estimation,' https://arxiv.org/abs/1708.03787.

Belloni, A., D. Chen, and V. Chernozhukov (2012): 'Sparse Models and Methods for Optimal Instruments with an Application to Eminent Domain,' Econometrica 80, 2369429.

Belloni, A. and V. Chernozhukov (2013): 'Least Squares After Model Selection in Highdimensional Sparse Models,' Bernoulli 19, 521-547.

Belloni, A., V. Chernozhukov, and C. Hansen (2014a): 'Inference on Treatment Effects after Selection among High-Dimensional Controls,' Review of Economic Studies 81, 608-650.

Table 12. Soda elasticities, by OLS

| variable     |   elasticity |   SE |
|--------------|--------------|------|
| income       |         0.64 | 0.02 |
| own-price    |        -0.65 | 0.04 |
| bread        |        -0.01 | 0.03 |
| butter       |        -0.04 | 0.02 |
| cereal       |         0.01 | 0.03 |
| chips        |         0.02 | 0.03 |
| coffee       |         0.01 | 0.02 |
| cookies      |        -0.03 | 0.02 |
| eggs         |        -0.03 | 0.03 |
| ice cream    |         0.01 | 0.03 |
| milk         |         0.02 | 0.04 |
| orange juice |        -0.05 | 0.05 |
| salad        |        -0.03 | 0.02 |
| soup         |         0.01 | 0.03 |
| water        |         0.01 | 0.02 |
| yogurt       |         0.05 | 0.04 |

Belloni, A., V. Chernozhukov, L. Wang (2014b): 'Pivotal Estimation via Square-Root Lasso in Nonparametric Regression,' Annals of Statistics 42, 757-788.

Belloni, A., V. Chernozhukov, K. Kato (2015): 'Uniform Post-selection Inference for Least Absolute Deviation Regression and Other Z-estimation Problems,' Biometrika 102, 77-94.

- Bickel, P.J. (1982): 'On Adaptive Estimation,' Annals of Statistics 10, 647-671.
- Bickel, P.J. and Y. Ritov (1988): 'Estimating Integrated Squared Density Derivatives: Sharp Best Order of Convergence Estimates,' Sankhy¯ a: The Indian Journal of Statistics, Series A 238, 381-393.

Bickel, P.J., C.A.J. Klaassen, Y. Ritov and J.A. Wellner (1993): Efficient and Adaptive Estimation for Semiparametric Models , Baltimore: Johns Hopkins University Press.

Bickel, P.J., Y. Ritov, and A. Tsybakov (2009): 'Simultaneous Analysis of Lasso and Dantzig Selector,' Annals of Statistics 37, 1705-1732.

Blundell, R.W. and J.L. Powell (2004): 'Endogeneity in Binary Response Models,' Review of Economic Studies 71, 655-679.

Bradic, J. and M. Kolar (2017): 'Uniform Inference for High-Dimensional Quantile Regression: Linear Functionals and Regression Rank Scores,' arXiv:1702.06209.

Bradic, J., S. Wager, and Y. Zhu (2019): 'Sparsity Double Robust Inference of Average Treatment Effects,' https://arxiv.org/pdf/1905.00744.pdf.

Bradic, J., V. Chernozhukov, W. Newey, and Y. Zhu (2019): 'Minimax Semiparametric Learning with Approximate Sparsity,' arXiv.

Burda, M., M. Harding, J.A. Hausman (2008): 'A Bayesian Mixed Logit Probit Model for Multinomial Choice,' Journal of Econometrics 147, 232-46.

- Burda, M., M. Harding, J.A. Hausman (2012): 'A Poisson Mixture Model of Discrete Choice,' Journal of Econometrics 166, 184-203.
- Cai, T.T. and Z. Guo (2017): 'Confidence Intervals for High-Dimensional Linear Regression: Minimax Rates and Adaptivity,' Annals of Statistics 45, 615-646.

Candes, E. and T. Tao (2007): 'The Dantzig Selector: Statistical Estimation when p is much Larger than n ,' Annals of Statistics 35, 2313-2351.

- Cattaneo, M.D., M. Jansson, and W.K. Newey (2018): 'Inference in Linear Regression Models with Many Covariates and Heteroscedasticity,' Journal of the American Statistical Association 113, 1350-1361.
- Chamberlain, G. (1982): 'Multivariate Regression Models for Panel Data,' Journal of Econometrics 18, 5-46.

Chamberlain, G. (1982): 'Efficiency Bounds for Semiparametric Regression,' Econometrica 60, 567-96.

Chamberlain, G. (1984): 'Panel Data,' Handbook of Econometrics Vol 2 , Z. Griliches and M. Intriligator, eds., 1247-1318.

Chatterjee, S. and J. Jafarov (2015): 'Prediction Error of Cross-Validated Lasso,' arXiv:1502.06291. Chen, X. and H. White (1999): 'Improved Rates and Asymptotic Normality for Nonpara- metric Neural Network Estimators,' IEEE Transactions on Information Theory 45, 682-691.

Chernozhukov, V., D. Chetverikov, and K. Kato (2013a): 'Gaussian Approximations and Multiplier Bootstrap for Maxima of Sums of High-Dimensional Random Vectors,' Annals of Statistics 41, 2786-2819.

Chernozhukov, V., I. Fernandez-Val, J. Hahn, W. Newey (2013b): 'Average and Quantile Effects in Nonseparable Panel Models,' Econometrica 81, 535-80.

Chernozhkov, V., C. Hansen, and M. Spindler (2015): 'Valid Post-Selection and PostRegularization Inference: An Elementary, General Approach,' Annual Review of Economics 7 , 649-688.

Chernozhukov, V., J. C. Escanciano, H. Ichimura, W.K. Newey, and J. Robins (2016): 'Locally Robust Semiparametric Estimation,' https://arxiv.org/abs/1608.00033v1.

Chernozhukov, V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W.K. Newey (2017):

'Double/Debiased/Neyman Machine Learning of Treatment Effects,' American Economic Review 107, 261-65.

Chernozhukov, V., D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W.K. Newey, J.M. Robins (2018): 'Double/debiased machine learning for treatment and structural parameters,' Econometrics Journal 21, C1-C68.

Chernozhukov, V., W.K. Newey, and J. Robins (2018): 'Double/De-Biased Machine Learning Using Regularized Riesz Representers,' https://arxiv.org/pdf/1802.08667v1.pdf.

Chernozhukov, V., W.K. Newey, and R. Singh (2018): 'Learning L2-Continuous Regres- sion Functionals via Regularized Riesz Representers,' https://arxiv.org/pdf/1809.05224v1.pdf.

Chernozhukov, V., W.K. Newey, and R. Singh (2019): 'Double/De-Biased Machine Learn- ing of Global and Local Parameters Using Regularized Riesz Representers,' https://arxiv.org/abs/1802.086

Chernozhukov, V., J.A. Hausman, W.K. Newey (2021): 'Demand Analysis with Many Prices,' NBER Working Paper 26424.

Chernozhukov, V., J. C. Escanciano, H. Ichimura, W.K. Newey, and J. Robins (2020):

'Locally Robust Semiparametric Estimation,' https://arxiv.org/abs/1608.00033v4.

Chiang, H.D., K. Kato, Y. Ma, Y. Sasaki (2019): 'Multiway Cluster Robust Double/Debiased Machine Learning,' arXiv:1909.03489.

Daubechies, I., M Defrise, and C. De Mol (2004): 'An Iterative Thresholding Algorithm for Linear Inverse Problems with a Sparsity Constraint,' Communications on Pure and Applied Mathematics 57, 1413-57.

Dehejia, R.H. and S. Wahba (1999): 'Causal Effects in Nonexperimental Studies: Reevaluating the Evaluation of Training Programs,' Journal of the American Statistical Association 94 (448): 1053-62.

Farbmacher, M., M. Huber, L. Laff´ ers, H. Langen, M. Spindler (2020): 'Causal Mediation Analysis with Double Machine Learning,' https://arxiv.org/abs/2002.12710.

Farrell, M. (2015): 'Robust Inference on Average Treatment Effects with Possibly More Covariates than Observations,' Journal of Econometrics 189, 1-23.

Farrell, M., T. Liang, S. Misra (2021): 'Deep Neural Networks for Estimation and Inference,' Econometrica 89, 181-213.

Friedman, J., T. Hastie, H. H¨ ofling, and R. Tibshirani (2007): 'Pathwise Coordinate Optimization,' The Annals of Applied Statistics 1, 302-32.

Friedman, J., T. Hastie, and R. Tibshirani (2010): 'Regularization Paths for Generalized Linear Models via Coordinate Descent,' Journal of Statistical Software 33, 1-22.

Fu, W.J. (1998): 'Penalized Regressions: The Bridge versus the Lasso,' Journal of Computational and Graphical Statistics 7, 397-416.

Graham, B. and J.L. Powell (2012): 'Identification and Estimation of Average Partial Effects in 'Irregular' Correlated Random Coefficient Panel Data Models,' Econometrica 80, 2105-52.

Hasminskii, R.Z. and I.A. Ibragimov (1979): 'On the Nonparametric Estimation of Functionals,' in P. Mandl and M. Huskova (eds.), Proceedings of the 2nd Prague Symposium on Asymptotic Statistics, 21-25 August 1978 , Amsterdam: North-Holland, pp. 41-51.

Hausman, J.A. and W.K. Newey (2016): 'Individual Heterogeneity and Average Welfare,' Econometrica 84, 1225-1248.

Hirshberg, D.A. and S. Wager (2017): 'Balancing Out Regression Error: Efficient Treatment Effect Estimation without Smooth Propensities,' arXiv:1712.00038v1.

Hirshberg, D.A. and S. Wager (2020): 'Debiased Inference of Average Partial Effects in Single-Index Models,' Journal of Business and Economic Statistics 38, 19-24.

Hirshberg, D.A. and S. Wager (2018): 'Augmented minimax linear estimation,' arXiv:1712.00038v5.

- Huber, P. J.: 'The Behavior of Maximum Likelihood Estimates Under Nonstandard Conditions,' in Proceedings of the Fifth Berkeley Symposium in Mathematical Statistics and Probability . Berkeley: University of California Press, 1967.
- Imai, K, L. Keele, and D. Tingley (2010): 'A General Approach to Causal Mediation Analysis,' Psychological Methods 15, 309 -334.

Imbens, G.W. and W.K. Newey (2009): 'Identification and Estimation of Triangular Simultaneous Equations Models Without Additivity,' Econometrica 77, 1481-1512.

Jankova, J. and S. Van De Geer (2015): 'Confidence Intervals for High-Dimensional Inverse Covariance Estimation,' Electronic Journal of Statistics 90, 1205-1229.

Jankova, J. and S. Van De Geer (2016a): 'Semi-Parametric Efficiency Bounds and Efficient Estimation for High-Dimensional Models,' arXiv:1601.00815.

Jankova, J. and S. Van De Geer (2016b): 'Confidence Regions for High-Dimensional Generalized Linear Models under Sparsity,' arXiv:1610.01353.

Javanmard, A. and A. Montanari (2014a): 'Hypothesis Testing in High-Dimensional Regression under the Gaussian Random Design Model: Asymptotic Theory,' IEEE Transactions on Information Theory 60, 6522-6554.

- Javanmard, A. and A. Montanari (2014b): 'Confidence Intervals and Hypothesis Testing for High-Dimensional Regression,' Journal of Machine Learning Research 15: 2869-2909.

Javanmard, A. and A. Montanari (2015): 'De-Biasing the Lasso: Optimal Sample Size for Gaussian Designs,' arXiv:1508.02757.

Jing, B.Y., Q.M. Shao, and Q. Wang (2003): 'Self-Normalized Cram´ er-Type Large Deviations for Independent Random Variables,' Annals of Probability 31, 2167-2215.

- Kennedy, E.H. (2020): 'Optimal Doubly Robust Estimation of Heterogeneous Causal Effects,' https://arxiv.org/abs/2004.14497.

Klaassen, C.A.J. (1987): 'Consistent Estimation of the Influence Function of Locally Asymptotically Linear Estimators,' Annals ot Statistics 15, 1548-1562.

- LaLonde, R.J. (1986): 'Evaluating the Econometric Evaluations of Training Programs with Experimental Data,' The American Economic Review 76, 604-20.
- Leeb, H., and B.M. P¨ otscher (2008a): 'Recent Developments in Model Selection and Related Areas,' Econometric Theory 24, 319-22.
- Leeb H., and B.M. P¨ otscher (2008b): 'Sparse Estimators and the Oracle Property, or the Return of Hodges' Estimator,' Journal of Econometrics 142, 201-211.
- Luo, Ye and M. Spindler (2016): 'High-Dimenstional L2 Boosting: Rate of Convergence,' https://arxiv.org/pdf/1602.08927.pdf.
- Luedtke, A. R. and M. J. van der Laan (2016): 'Optimal Individualized Treatments in Resource-limited Settings,' The International Journal of Biostatistics 12, 283-303.
- Newey, W.K. (1994): 'The Asymptotic Variance of Semiparametric Estimators,' Econometrica 62, 1349-1382.
- Newey, W.K., F. Hsieh, and J.M. Robins (1998): 'Undersmoothing and Bias Corrected Functional Estimation,' MIT Dept. of Economics working paper 98-17.
- Newey, W.K., F. Hsieh, and J.M. Robins (2004): 'Twicing Kernels and a Small Bias Property of Semiparametric Estimators,' Econometrica 72, 947-962.
- Newey, W.K. and J.M. Robins (2017): 'Cross Fitting and Fast Remainder Rates for Semiparametric Estimation,' arXiv:1801.09138.
- Neykov, M., Y. Ning, J.S. Liu, and H. Liu (2015): 'A Unified Theory of Confidence Regions and Testing for High Dimensional Estimating Equations,' arXiv:1510.08986.
- Ning, Y. and H. Liu (2017): 'A General Theory of Hypothesis Tests and Confidence Regions for Sparse High Dimensional Models,' Annals of Statistics 45, 158-195.
- Powell, J.L., J.H. Stock, and T.M. Stoker (1989): 'Semiparametric Estimation of Index Coefficients,' Econometrica 57, 1403-1430.
- Ren, Z., T. Sun, C.H. Zhang, and H. Zhou (2015): 'Asymptotic Normality and Optimalities in Estimation of Large Gaussian Graphical Models,' Annals of Statistics 43, 991-1026.
- Robins, J.M. and A. Rotnitzky (1995): 'Semiparametric Efficiency in Multivariate Regression Models with Missing Data,' Journal of the American Statistical Association 90 (429): 122-129.
- Robins, J.M., A. Rotnitzky, and L.P. Zhao (1995): 'Analysis of Semiparametric Regression Models for Repeated Outcomes in the Presence of Missing Data,' Journal of the American Statistical Association 90, 106-121.
- Robins, J., P. Zhang, R. Ayyagari, R. Logan, E. Tchetgen, L. Li, A. Lumley, and A. van der Vaart (2013): 'New Statistical Approaches to Semiparametric Regression with Application to Air Pollution Research,' Research Report Health E Inst.

Rosenbaum, P.R. and D. B. Rubin (1983): 'The Central Role of the Propensity Score in Observational Studies for Causal Effects,' Biometrika 70: 41-55.

Rothenh¨ ausler, D. and B. Yu (2019): 'Incremental Causal Effects,' arXiv:1907.13258.

Rudelson, M. and S. Zhou (2013): 'Reconstruction From Anisotropic Random Measurements,' IEEE Transactions on Informating Theory 59, 3434-3447.

Scharfstein, D.O., A. Rotnitzky, and J.M. Robins (1999): 'Rejoinder to Adjusting for Nonignorable Drop-out Using Semiparametric Nonresponse Models,' Journal of the American Statistical Association 94, 1096-1146.

Schick, A. (1986): 'On Asymptotically Efficient Estimation in Semiparametric Models,' Annals of Statistics 14, 1139-1151.

Schmidt-Hieber, J. (2020): 'Nonparametric Regression Using Deep Neural Networks with RELU Activation Function,' The Annals of Statistics 48, 1875-1897.

Singh, R. and L. Sun (2019): 'De-biased Machine Learning for Compliers,' arXiv:1909.05244. Smucler, E., A. Rotnitzky, and J.R. Robins (2019): 'A Unifying Approach for Doubly-

- robust L1 Regularized Estimation of Causal Contrasts,' https://arxiv.org/abs/1904.03737. Stock, J.H. (1989): 'Nonparametric Policy Analysis,' Journal of the American Statistical Association 84, 567-575.

Syrgkanis, V., and M. Zampetakis (2020): 'Estimation and Inference with Trees and Forests in High Dimensions,' https://arxiv.org/abs/2007.03210.

Tchetgen Tchetgen, E.J. and I. Shipster (2012): 'Semiparametric Theory for Causal Mediation Analysis: Efficiency Bounds, Multiple Robustness and Sensitivity Analysis,' The Annals of Statistics 40, 1816-1845.

- Toth, B. and M. J. van der Laan (2016), 'TMLE for Marginal Structural Models Based On An Instrument,' U.C. Berkeley Division of Biostatistics Working Paper Series, Working Paper 350.

Tseng, P. (2001): 'Convergence of a Block Coordinate Descent Method for Nondifferentiable Minimization,' Journal of Optimization Theory and Applications 109, 475-94.

Van De Geer, S., P. B¨ uhlmann, Y. Ritov, and R. Dezeure (2014): 'On Asymptotically Optimal Confidence Regions and Tests for High-Dimensional Models,' Annals of Statistics , 42: 1166-1202.

Van der Laan, M. and D. Rubin (2006): 'Targeted Maximum Likelihood Learning,' International Journal of Biostatistics 2.

Van der Laan, M. J. and S. Rose (2011): Targeted Learning: Causal Inference for Observational and Experimental Data, Springer.

Van der Vaart, A.W. (1991): 'On Differentiable Functionals,' Annals of Statistics , 19: 178-204.

Van der Vaart, A.W. (1998): Asymptotic Statistics . New York: Cambridge University Press.

Vermeulen, K. and S. Vansteelandt (2015): 'Bias-Reduced Doubly Robust Estimation,' Journal of the American Statistical Association 110, 1024-1036.

Vershynin, R. (2018): High-Dimensional Probability , New York: Cambridge University Press.

White, H. (1982): 'Maximum Likelihood Estimation of Misspecified Models,' Econometrica 50, 1-25.

Wooldridge, J.M. (2002): Econometric Analysis of Cross-Section and Panel Data , Cambridge, MIT Press.

Wooldridge, J.M. (2019): 'Correlated Random Effects Models with Unbalanced Panels,' Journal of Econometrics 211, 137-50.

Wooldridge, J.M. and Y. Zhu (2020): 'Inference in Approximately Sparse Correlated Random Effects Probit Models With Panel Data,' Journal of Business and Economic Statistics 38, 1-18.

Zhang, C. and S. Zhang (2014): 'Confidence Intervals for Low-Dimensional Parameters in High-Dimensional Linear Models,' Journal of the Royal Statistical Society, Series B 76, 217-242.

Zheng, W., Z. Luo, and M. J. van der Laan (2016), 'Marginal Structural Models with Counterfactual Effect Modifiers,' U.C. Berkeley Division of Biostatistics Working Paper Series, Working Paper 348.

Zhu, Y. and J. Bradic (2017a): 'Linear Hypothesis Testing in Dense High-Dimensional Linear Models,' Journal of the American Statistical Association 112.

Zhu, Y. and J. Bradic (2017b): 'Breaking the Curse of Dimensionality in Regression,' arXiv: 1708.00430.

Zubizarreta, J.R. (2015): 'Stable Weights that Balance Covariates for Estimation with Incomplete Outcome Data,' Journal of the American Statistical Association 90 (429): 122129.