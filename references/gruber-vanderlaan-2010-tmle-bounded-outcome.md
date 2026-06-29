## The International Journal of Biostatistics

Volume 6, Issue 1

2010

Article 26

## A Targeted Maximum Likelihood Estimator of a Causal Effect on a Bounded Continuous Outcome

Susan Gruber, University of California, Berkeley

Mark J. van der Laan, University of California, Berkeley

## Recommended Citation:

Gruber, Susan and van der Laan, Mark J. (2010) "A Targeted Maximum Likelihood Estimator of a Causal Effect on a Bounded Continuous Outcome," The International Journal of Biostatistics : Vol. 6: Iss. 1, Article 26.

DOI:

10.2202/1557-4679.1260

Available at:

http://www.bepress.com/ijb/vol6/iss1/26

©2010 Berkeley Electronic Press. All rights reserved.

## A Targeted Maximum Likelihood Estimator of a Causal Effect on a Bounded Continuous Outcome

Susan Gruber and Mark J. van der Laan

## Abstract

Targeted maximum likelihood estimation of a parameter of a data generating distribution, known to be an element of a semi-parametric model, involves constructing a parametric model through an initial density estimator with parameter ε representing an amount of fluctuation of the initial density estimator, where the score of this fluctuation model at ε = 0 equals the efficient influence curve/canonical gradient. The latter constraint can be satisfied by many parametric fluctuation models since it represents only a local constraint of its behavior at zero fluctuation. However, it is very important that the fluctuations stay within the semi-parametric model for the observed data distribution, even if the parameter can be defined on fluctuations that fall outside the assumed observed data model. In particular, in the context of sparse data, by which we mean situations where the Fisher information is low, a violation of this property can heavily affect the performance of the estimator. This paper presents a fluctuation approach that guarantees the fluctuated density estimator remains inside the bounds of the data model. We demonstrate this in the context of estimation of a causal effect of a binary treatment on a continuous outcome that is bounded. It results in a targeted maximum likelihood estimator that inherently respects known bounds, and consequently is more robust in sparse data situations than the targeted MLE using a naive fluctuation model.

When an estimation procedure incorporates weights, observations having large weights relative to the rest heavily influence the point estimate and inflate the variance. Truncating these weights is a common approach to reducing the variance, but it can also introduce bias into the estimate. We present an alternative targeted maximum likelihood estimation (TMLE) approach that dampens the effect of these heavily weighted observations. As a substitution estimator, TMLE respects the global constraints of the observed data model. For example, when outcomes are binary, a fluctuation of an initial density estimate on the logit scale constrains predicted probabilities to be between 0 and 1. This inherent enforcement of bounds has been extended to continuous outcomes. Simulation study results indicate that this approach is on a par with, and many times superior to, fluctuating on the linear scale, and in particular is more robust when there is sparsity in the data.

KEYWORDS:

targeted maximum likelihood estimation, TMLE, causal effect

## Author Notes:

This work was supported by NIH grant R01AI074345-04.

## Erratum

On September 20, 2010 the following acknowledgment was added for the article: "This work was supported by NIH grant R01AI074345-04."

## 1 Introduction

Targeted maximum likelihood estimation (TMLE) yields semi-parametric efficient substitution estimators of parameters in semi-parametric models (van der Laan and Rubin, 2006). In particular, it can be applied to estimating the statistical counterpart of a causal parameter. In this article a new targeted maximum likelihood estimator for estimating a causal effect of a binary treatment on a continuous outcome is introduced. This estimator is more robust than a previously presented TMLE procedure when there is sparsity in the data that decreases the identifiability of the parameter of interest.

Sparsity is defined as low information in the dataset for the purpose of learning the target parameter. Formally, the Fisher information, I , is defined as sample size n divided by the variance of the efficient influence curve: I = n/var ( D ∗ ( O )) , where D ∗ ( O ) is the efficient influence curve of the target parameter at the true data generating distribution. The reciprocal of the variance of the efficient influence curve can be viewed as the information one observation contains for the purpose of learning the target parameter. Since the variance of the efficient influence curve divided by n times the variance of an asymptotically efficient estimator converges to 1 when the sample size converges to infinity, one can also think of the information I as the reciprocal of the variance of an efficient estimator of the target parameter. Thus, sparsity with respect to a particular target parameter corresponds with small sample size relative to the variance of the efficient influence curve for that target parameter.

Section 2 of the paper provides background on the application of TMLE methodology in the context of sparsity, and its power relative to other semi-parametric efficient estimators by being a substitution estimator respecting global constraints of the semi-parametric model. Even though an estimator can be asymptotically efficient without utilizing global constraints, the global constraints are instrumental in the context of sparsity with respect to the target parameter, motivating the need for semi-parametric efficient substitution estimators, and for a careful choice of fluctuation function for the targeted MLE step that fully respects these global constraints. A rigorous demonstration of the proposed targeted MLE of the causal effect of a binary treatment on a bounded continuous outcome follows, and it is contrasted to a targeted MLE that makes use a fluctuation function that does not respect the bounds.

Simulation studies described in Section 3 compare the new TMLE estimator of the causal effect, which relies on a logistic fluctuation of an initial density estimate, with the traditional TMLE estimator, with and without sparsity in the data. Results for other commonly applied estimators, the inverse-probability-of-treatment weighted estimator (IPTW) (Hernan et al., 2000; Robins, 2000b), a double ro-

bust augmented IPTW estimator (aug-IPTW) (Robins and Rotnitzky, 2001; Robins et al., 2000; Robins, 2000a) that is efficient but not a substitution estimator, and the maximum likelihood substitution estimator according to a parametric model (MLE) (Robins, 1986) are also presented.

## 2 TMLEfor causal effect estimation on a continuous outcome

The targeted MLE is a semi-parametric efficient substitution estimator of a target parameter Ψ( P 0 ) of a true distribution P 0 ∈ M , based on sampling n i.i.d. O 1 , . . . , O n from P 0 . Here P 0 is known to be an element of a semi-parametric statistical model M . Wewill start with providing a succinct summary of how it works. For more details we refer to our articles on this topic (van der Laan et al., 2009).

Firstly, one notes that Ψ( P 0 ) = Ψ( Q 0 ) only depends on P 0 through a relevant part Q 0 = Q ( P 0 ) of P 0 . Secondly, one proposes a loss function L ( Q )( O ) so that Q 0 = arg min Q ∈Q E 0 L ( Q )( O ) , where Q = { Q ( P ) : P ∈ M} . Thirdly, one uses minimum loss-based learning, such as super learning (van der Laan et al., 2007), fully utilizing the power and optimality results for loss-based cross-validation to select among candidate estimators, to obtain an initial estimator Q 0 n of Q 0 . Fourthly, one proposes a parametric fluctuation Q 0 ng ( glyph[epsilon1] ) , possibly indexed by nuisance parameter g 0 = g ( P 0 ) , so that

<!-- formula-not-decoded -->

where D ∗ ( Q 0 , g 0 ) is the canonical gradient/efficient influence curve of Ψ : M→ I R at P 0 . Fifthly, one computes the amount of fluctuation

<!-- formula-not-decoded -->

where g n is an estimator of the unknown nuisance parameter g 0 . This yields an update Q 1 n = Q 0 ng n ( glyph[epsilon1] n ) . This updating of an initial estimator Q 0 n into a next Q 1 n is iterated till convergence resulting in a Q ∗ n . Since at the last step the amount of fluctuation glyph[epsilon1] n ≈ 0 , this final Q ∗ n will solve the efficient influence curve estimating equation

<!-- formula-not-decoded -->

representing a fundamental ingredient for establishing asymptotic efficiency of Ψ( Q ∗ n ) : recall that an estimator is efficient if and only if it is asymptotically linear with influence curve equal to the efficient influence curve D ∗ ( Q 0 , g 0 ) . Finally, the targeted MLE of ψ 0 is the substitution estimator Ψ( Q ∗ n ) .

Thus we see that the targeted MLE involves constructing a parametric model Q 0 n ( glyph[epsilon1] ) through the initial estimator Q 0 n with parameter glyph[epsilon1] representing an amount of fluctuation of the initial estimator, where the score of this fluctuation model at glyph[epsilon1] = 0 equals the efficient influence curve. The latter constraint can be satisfied by many parametric models, since it represents only a local constraint of its behavior at zero fluctuation. However, it is very important that the fluctuations stay within the model for the observed data distribution, even if the parameter can be defined on fluctuations that fall outside the assumed observed data model. In particular, in the context of sparse data, a violation of this property can heavily affect the performance of the estimator.

One important strength of the semi-parametric efficient targeted MLE relative to the alternative semi-parametric efficient estimating equation methodology (van der Laan and Robins, 2003) is that it does respect the global constraints of the observed data model since it is a substitution estimator Ψ( Q ∗ n ) with Q ∗ n an estimator of a relevant part Q 0 of the true distribution of the data in the observed data model. The estimating equation methodology does not result in substitution estimators and thereby often ignores important global constraints of the observed data model, though Tan (2008) introduces a non-parametric likelihood based approach to constructing a double robust estimator that is not a substitution estimator, and offers a comparison with other estimators, including TMLE that is not constrained to remain within the bounds of the observed data model. Ignoring constraints comes at a price in the context of sparsity. Indeed, simulations have confirmed this gain of targeted MLE relative to the efficient estimating equation method in the context of sparsity (Stitelman and van der Laan, 2010), and it is again demonstrated in this article. However, if the targeted MLE starts violating this principle of being a substitution estimator by allowing Q ∗ n to fall outside the assumed observed data model, this advantage is compromised. Therefore, it is crucial that a fluctuation model is used that is guaranteed to stay within the wished observed data model.

To demonstrate this important consideration of selecting a valid fluctuation model in the construction of targeted MLE, we consider the problem of estimating a causal effect of a binary treatment A on a continuous outcome Y , based on observing n i.i.d. copies of O = ( W,A,Y ) ∼ P 0 , where W is the set of confounders. Under non-parametric structural equation model (NPSEM) W = f W ( U W ) , A = f A ( W,U A ) , Y = f Y ( W,A,U Y ) with a structure on the exogenous variables U = ( U W , U A , U Y ) satisfying the no unmeasured confounders assumption ( A ⊥ Y ( a ) | W for the counterfactuals Y ( a ) defined by this NPSEM), the additive

causal effect E ( Y (1) -Y (0)) can be identified from the observed data distribution through the following statistical parameter of P 0 :

<!-- formula-not-decoded -->

Suppose that it is known that Y ∈ [ a, b ] for some a &lt; b . Alternatively, one might have truncated the original data to fall in such an interval and focus on the causal effect of treatment on this truncated outcome, motivated by the fact that estimating conditional means of unbounded, or very heavy tailed, outcomes requires very large data sets.

Let Y ∗ = ( Y -a ) / ( b -a ) be the linearly transformed outcome within [0 , 1] , and define

<!-- formula-not-decoded -->

We note that

<!-- formula-not-decoded -->

An estimate, limit distribution, and confidence interval for Ψ ∗ ( P 0 ) is now immediately mapped into an estimate, limit distribution, and confidence interval for Ψ( P 0 ) , by simple multiplication by ( b -a ) . As a consequence, without loss of generality, we can assume a = 0 and b = 1 so that Y ∈ [0 , 1] .

The efficient influence curve of the statistical parameter Ψ : M→ I R , defined on a non-parametric statistical model M for P 0 , at the true distribution P 0 , is given by

<!-- formula-not-decoded -->

where ¯ Q 0 ( W,A ) = E 0 ( Y | A, W ) , and Q 0 = ( Q W , ¯ Q 0 ) denotes both this conditional mean ¯ Q 0 as well as the marginal distribution Q W of W . Note that indeed Ψ( P 0 ) only depends on P 0 through ¯ Q 0 and the marginal distribution of W . We will use the notation Ψ( P 0 ) and Ψ( Q 0 ) interchangeably.

We will now define a targeted MLE of Ψ( Q 0 ) as follows. Let ¯ Q 0 n be an initial estimator of ¯ Q 0 ( W,A ) = E ( Y | A, W ) with predicted values in (0 , 1) . In addition, we estimate P W with the empirical distribution of W . Let Q 0 n denote the resulting initial estimator of Q 0 . The targeted MLE step will also require an estimator g n of g 0 = P A | W . Only the conditional mean ¯ Q 0 n will be modified by the targeted MLE procedure defined below: this makes sense since the empirical distribution of W is already a non-parametric maximum likelihood estimator so that no bias gain with respect to the target parameter will be obtained by modifying it.

Gruber and van der Laan: TMLE for Bounded Continuous Outcomes

We can represent the estimator ¯ Q 0 n as ¯ Q 0 n = 1 1+exp( -f 0 n ) with f 0 n = log( ¯ Q 0 n / (1 -¯ Q 0 n )) . Consider now the fluctuation model

<!-- formula-not-decoded -->

with parameter glyph[epsilon1] , indexed by a function

<!-- formula-not-decoded -->

Equivalently, we can write this as logit ¯ Q 0 n ( glyph[epsilon1] ) = logit ¯ Q 0 n + glyph[epsilon1]h ( g n ) .

Consider now the following loss function for ¯ Q 0 :

<!-- formula-not-decoded -->

Note that this is the log-likelihood of the conditional distribution of a binary outcome Y , but now extended to continuous outcomes in [0 , 1] . (See also Wedderburn (1974), McCullagh (1983) for earlier use of logistic regression for continuous outcomes.) It is thus known that this loss function is a valid loss function for the conditional distribution of a binary Y , but we need that it is a valid loss function for a conditional mean of a continuous Y ∈ [0 , 1] . We have the following lemma establishing this result about this loss function.

Lemma 1 We have that

For any function h we have

<!-- formula-not-decoded -->

Proof: Let ¯ Q 1 be a local minimum and consider the fluctuation ¯ Q 1 ( glyph[epsilon1] ) defined above. Then the derivative of E 0 L ( ¯ Q 1 ( glyph[epsilon1] )) at glyph[epsilon1] = 0 equals zero. However,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the minimum is taken over all functions of ( W,A ) which map into (0 , 1) . In addition, define the fluctuation function

<!-- formula-not-decoded -->

Thus, it follows that

<!-- formula-not-decoded -->

But this needs to hold for any function h ( W,A ) , which proves that ¯ Q 1 = ¯ Q 0 a.e. ✷

This proves that L ( ¯ Q ) is a valid loss function for the conditional mean ¯ Q 0 . Indeed, we can use L ( ¯ Q ) as loss function to construct an initial estimator of ¯ Q 0 , and or use cross-validation to select among candidate targeted maximum likelihood estimators, such as in the collaborative targeted MLE procedure. For the purpose of construction of an initial estimator one could also use a minimum loss-based super learner based on the squared error loss function L 2 ( ¯ Q ) = ( Y -¯ Q ( W,A )) 2 , possibly with weights.

Given an initial estimator ¯ Q 0 n , and our proposed fluctuation function ¯ Q 0 n ( glyph[epsilon1] ) , we have

<!-- formula-not-decoded -->

giving us the wished first component D ∗ 1 of the efficient influence curve D ∗ = D ∗ 1 + D ∗ 2 .

Let's use the log-likelihood loss function, -logQ W , as loss function for the marginal distribution of W , so that our combined loss function is given by L ( Q ) = -logQ W + L ( ¯ Q ) . In addition, we use as fluctuation of the empirical distribution Q Wn , Q Wn ( glyph[epsilon1] 1 ) = (1+ glyph[epsilon1] 1 D ∗ 2 ( Q )) Q Wn , where D ∗ 2 ( Q ) = ¯ Q ( W, 1) -¯ Q ( W, 0) -Ψ( Q ) is the remaining component of the efficient influence curve. With these choices we indeed now have that

<!-- formula-not-decoded -->

This shows that we succeeded in defining a loss function for Q 0 = ( Q W , ¯ Q 0 ) and fluctuation function so that the wished derivative (1) indeed yields the efficient influence curve.

The MLE of glyph[epsilon1] 1 equals zero, so that the update of Q Wn equals Q Wn itself. The empirical mean of the component D ∗ 2 = ¯ Q ( W, 1) -¯ Q ( W, 0) -Ψ( Q ) of the efficient influence curve is always equal to zero, due to the fact that we estimate the marginal distribution of W with the empirical distribution of W .

The amount of fluctuation of glyph[epsilon1] for fluctuating ¯ Q 0 n is given by

<!-- formula-not-decoded -->

This 'maximum likelihood' estimator of glyph[epsilon1] can be computed with generalized linear regression using the binomial link, i.e. the logistic regression MLE procedure, simply ignoring that the outcome is not binary, which also corresponds with iterative re-weighted least squares estimation using weights 1 / ¯ Q (1 -¯ Q ) .

This provides us with the targeted MLE update Q 1 n = Q 0 n ( glyph[epsilon1] 0 n ) , where the empirical distribution of W did not get updated, and ¯ Q 0 n did get updated as ¯ Q 0 n ( glyph[epsilon1] 0 n ) . Iterating this procedure now defines the targeted MLE Q ∗ n , but as in the binary outcome case, we have that Q 2 n = Q 1 n ( glyph[epsilon1] 1 n ) = Q 1 n since the next MLE glyph[epsilon1] 1 n = 0 . Thus convergence occurs in one step, so that Q ∗ n = Q 1 n . The targeted MLE of ψ 0 is thus given by Ψ( Q ∗ n ) = Ψ( Q 1 n ) . As predicted, we have that the targeted MLE Q ∗ n solves the efficient influence curve estimating equation P n D ∗ ( Q ∗ n , g n , Ψ( Q ∗ n )) = 0 .

An inspection of this efficient influence curve,

<!-- formula-not-decoded -->

reveals that there are two potential sources of sparsity. Small values for g 0 ( A | W ) and large outlying values of Y inflate the variance. Enforcing (e.g., known) bounds on Y and g 0 in the estimation procedure provides a means for controlling these sources of variance. We note that, even if there is strong confounding causing some large values of h g 0 n , the resulting targeted MLE ¯ Q ∗ n remains bounded in (0 , 1) , so that the targeted MLE Ψ( Q ∗ n ) fully respects the global constraints of the observed data model. On the other hand, the augmented IPTW estimator obtained by solving P n D ∗ ( Q 0 n , g n , ψ ) = 0 in ψ yields the estimator

<!-- formula-not-decoded -->

which can easily fall outside [0 , 1] if for some observations W i , g n (1 | W i ) is close to 1 or 0. This represents the price of not being a substitution estimator.

Contrasting with targeted MLE using linear fluctuation function. Alternatively, we would employ the targeted MLE using the L 2 ( ¯ Q ) = ( Y -¯ Q ( W,A )) 2 loss function, and fluctuation function ¯ Q 0 ( glyph[epsilon1] ) = ¯ Q 0 + glyph[epsilon1]h ( g ) , so that (1) is still satisfied. In this case, large values of h ( g ) will result in predicted values of ¯ Q 0 ( glyph[epsilon1] n ) that are out of the bounds [ a, b ] . Therefore, this version of targeted MLE is not respecting the global constraints of the model, i.e., the knowledge that Y ∈ [ a, b ] . A comparison based on simulated data of the targeted MLE using the logistic fluctuation function and the targeted MLE using this linear fluctuation function is provided in the next section.

## 3 Simulation studies for the additive effect of a binary point treatment on a continuous outcome

Two simulation studies illustrate the effects of employing a logistic vs. linear fluctuation on TMLE estimator performance with and without sparsity in the data, where a high degree of sparsity corresponds to a target parameter that is borderlineidentifiable. As above, the parameter of interest is defined as the marginal effect of a binary point treatment on the outcome, ψ 0 = E W ( E ( Y | A = 1 , W ) -E ( Y | A = 0 , W )) .

The 'traditional' targeted maximum likelihood approach to estimating an additive treatment effect when the outcome is continuous is to fluctuate the initial density estimate on a linear scale. Given ¯ Q 0 n ( W,A ) , an initial estimate of the conditional mean of Y given ( W,A ) , the fluctuation function is defined as ¯ Q 0 n ( glyph[epsilon1] ) = ¯ Q 0 n + glyph[epsilon1] ( h g n ) and the loss function L ( ¯ Q ) is chosen to be the squared error loss function, so that we still have the required constraint (1). The estimate glyph[epsilon1] n can be obtained by estimating glyph[epsilon1] with a linear regression of Y on h g n , using the initial fit, ¯ Q 0 n ( W,A ) , as offset.

A second TMLE estimate using the logistic fluctuation method described in Section 2 is also obtained. Y is transformed into Y ∗ ∈ [0 , 1] by shifting and scaling the values. In the simulation setting, Y is not bounded, so that we do not have an a priori a and b bound on Y . Instead of truncating Y and redefining the target parameter as the causal effect on the truncated Y , we still aim to estimate the causal effect on the original Y . Therefore, we set a = min ( Y ) , b = max ( Y ) , and

<!-- formula-not-decoded -->

An initial estimate, ¯ Q 0 ,Y ∗ n ( W,A ) = E ( Y ∗ | W,A ) , is obtained, and then represented as a logistic function of its logit-transformation. Note that logit ( x ) is not defined when x = 0 or 1 , therefore in practice ¯ Q 0 ,Y ∗ n ( W,A ) is bounded away from 0 and 1 by truncating it at ( α, (1 -α )) . We used α = 0 . 005 in these simulation studies, which did not yield appreciably different results than setting α = 0 . 001 or α = 0 . 01 . The function ¯ Q 0 ,Y ∗ n is fluctuated on the logit scale with logit ¯ Q 0 ,Y ∗ n ( glyph[epsilon1] ) = logit ¯ Q 0 ,Y ∗ n + glyph[epsilon1]h ( g n ) , using the same clever covariate, h g n ( W,A ) , employed in the linear fluctuation described above. Fitting glyph[epsilon1] is again carried out using standard software, but this time using logistic regression of Y ∗ on h g n ( W,A ) with offset logit ( Q 0 ,Y ∗ n ( W,A )) . This results in the updated ¯ Q 1 ,Y ∗ n . Fitted values for ¯ Q 1 ,Y ∗ n ( W,A ) are mapped back to the original scale: ¯ Q 1 ,Y n = ¯ Q 1 ,Y ∗ n ( W,A ) ∗ ( b -a )+ a . The marginal distribution is es-

http://www.bepress.com/ijb/vol6/iss1/26

DOI: 10.2202/1557-4679.1260

timated with the empirical distribution of W , giving the Q ∗ n = Q 1 n = ( Q W,n , ¯ Q 1 ,Y n ) of ( Q W , ¯ Q 0 ) . The estimate

<!-- formula-not-decoded -->

is the targeted MLE of the wished additive causal effect ψ 0 .

Parameter estimates were also obtained using the augmented inverse probability of treatment weighed estimator (aug-IPTW),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Both the targeted MLE and the augmented IPTW estimator are double robust so that these estimators will be consistent for ψ 0 if either g n or ¯ Q 0 n is consistent for g 0 and ¯ Q 0 , respectively. Both the targeted MLE and the augmented IPTW estimator are asymptotically efficient if both g n and ¯ Q 0 n are consistent.

In this simulation study we will use simple parametric MLE's as initial estimators ¯ Q 0 n and g n , even though we recommend the utilization of super learning in practice. The purpose of this simulation is to investigate the performance of the updating step under misspecified and correctly specified ¯ Q 0 n , and for that purpose we can work with parametric MLE fits.

Results from two estimation methods that are not double robust and semiparametric efficient are included as well. The maximum likelihood estimator according to a parametric model for ¯ Q 0 (MLE), used as initial estimator in the targeted MLEand augmented IPTW, is included for the sake of evaluating the bias reduction step carried out by these two semi-parametric efficient procedures. Inverse probability of treatment weighted (IPTW) estimators are consistent when g n ( A | W ) is a consistent estimator of the treatment mechanism g 0 ( A | W ) = P ( A = 1 | W ) , but are known to be inefficient. These two estimators are defined as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 3.1 Data generation

Covariates W 1 , W 2 , W 3 were generated as independent binary random variables,

<!-- formula-not-decoded -->

Two treatment mechanisms were defined that differ only in the values of the coefficients for each covariate:

<!-- formula-not-decoded -->

We consider two settings:

<!-- formula-not-decoded -->

We refer to these two treatment mechanisms as g 0 , 1 and g 0 , 2 , respectively. The observed outcome Y was generated as

<!-- formula-not-decoded -->

For both simulations the true additive causal effect equals one: ψ 0 = 1 . In both simulations predicted values for g n ( A | W ) are bounded away from 0 and 1 by truncating at ( p, (1 -p )) , with p = 0 . 01 . Treatment assignment probabilities based on mechanism g 0 , 1 range from 0.269 to 0.881, indicating no sparsity in the data for simulation 1. In contrast, simulation 2 poses a challenging estimation problem in the context of sparse data. Treatment assignment probabilities based on mechanism g 0 , 2 range from 0.047 to 0.998. These extreme values are nevertheless not uncommon for data from observational studies (see for example Dehejia and Wahba (2002); Stukel et al. (2007)).

Estimates were obtained for 1000 samples of size n = 1000 from each data generating distribution. Treatment assignment probabilities, g 0 ( A | W ) , were estimated using a correctly specified logistic regression model. A correctly specified main terms regression model was used to obtain ¯ Q 0 cor ( W,A ) . In addition, a misspecified initial estimate, ¯ Q 0 mis ( W,A ) , was obtained by regressing Y on A .

We expect MLE estimates based on ¯ Q 0 cor to be unbiased and efficient, while those based on ¯ Q 0 mis will be biased. IPTW estimates only depend on consistent estimation of g 0 , thus are identical regardless of how ¯ Q 0 is estimated. For both simulations g n is a consistent estimator, thus it is reasonable to expect unbiased IPTW estimates, with more variation in simulation 2 estimates. The targeted MLE and the augmented IPTW are known to be unbiased if g n is consistent, and asymptotically efficient when both ¯ Q 0 and g 0 are consistently estimated. Though correctly estimat-

ing g 0 will asymptotically correct for any bias due to mis-specification of ¯ Q 0 n , this is not guaranteed in finite samples, especially when there is sparsity. For simulation 2 we expect TMLE log , using the logistic fluctuation, to outperform TMLE lin , using the linear fluctuation.

Table 1: Estimator performance for simulations 1 and 2 when the initial estimator of ¯ Q 0 is correct and misspecified. Results are based on 1000 samples of size n = 1000 .

|              | ¯ Q 0 correctly estimated   | ¯ Q 0 correctly estimated   | ¯ Q 0 correctly estimated   | ¯ Q 0 correctly estimated   | ¯ Q 0 incorrectly estimated   | ¯ Q 0 incorrectly estimated   | ¯ Q 0 incorrectly estimated   | ¯ Q 0 incorrectly estimated   |
|--------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-------------------------------|-------------------------------|-------------------------------|-------------------------------|
|              | ave                         | bias                        | var                         | MSE                         | ave                           | bias                          | var                           | MSE                           |
| Simulation 1 |                             |                             |                             |                             |                               |                               |                               |                               |
| MLE          | 1 . 003                     | 0 . 003                     | 0 . 005                     | 0 . 005                     | 3 . 075                       | 2 . 075                       | 0 . 030                       | 4 . 336                       |
| IPTW         | 1 . 006                     | 0 . 006                     | 0 . 009                     | 0 . 009                     | 1 . 006                       | 0 . 006                       | 0 . 009                       | 0 . 009                       |
| aug-IPTW     | 1 . 003                     | 0 . 003                     | 0 . 005                     | 0 . 005                     | 1 . 005                       | 0 . 005                       | 0 . 010                       | 0 . 010                       |
| TMLE log     | 0 . 993                     | - 0 . 007                   | 0 . 005                     | 0 . 005                     | 0 . 993                       | - 0 . 007                     | 0 . 006                       | 0 . 006                       |
| TMLE lin     | 0 . 993                     | - 0 . 007                   | 0 . 005                     | 0 . 005                     | 0 . 993                       | - 0 . 007                     | 0 . 006                       | 0 . 006                       |
| Simulation 2 |                             |                             |                             |                             |                               |                               |                               |                               |
| MLE          | 1 . 001                     | 0 . 001                     | 0 . 009                     | 0 . 009                     | 4 . 653                       | 3 . 653                       | 0 . 025                       | 13 . 370                      |
| IPTW         | 1 . 554                     | 0 . 554                     | 0 . 179                     | 0 . 485                     | 1 . 554                       | 0 . 554                       | 0 . 179                       | 0 . 485                       |
| aug-IPTW     | 0 . 999                     | - 0 . 001                   | 0 . 023                     | 0 . 023                     | 1 . 708                       | 0 . 708                       | 0 . 298                       | 0 . 798                       |
| TMLE log     | 0 . 989                     | - 0 . 011                   | 0 . 037                     | 0 . 037                     | 0 . 722                       | - 0 . 278                     | 0 . 214                       | 0 . 291                       |
| TMLE lin     | 0 . 986                     | - 0 . 014                   | 0 . 042                     | 0 . 042                     | - 0 . 263                     | - 1 . 263                     | 2 . 581                       | 4 . 173                       |

## 3.2 Results

Table 1 reports the average estimate, bias, empirical variance, and mean squared error (MSE) for each estimator, under different specifications of the initial estimator ¯ Q 0 n . In all cases g n is consistent, and bounded at (0.01, 0.99). In simulation 1, when ¯ Q 0 is correctly estimated all estimators perform quite well, though as expected, IPTW is the least efficient. However, when ¯ Q 0 is incorrectly estimated, the MLE estimator is biased and has high variance relative to the other estimators. Because g n ( A | W ) is correctly specified, IPTW and aug-IPTW provide unbiased estimates, as do both TMLEs. TMLE log is on a par with TMLE lin , as there is no sparsity in the data, and both are more efficient than any of the other estimators.

In simulation 2 all estimators except IPTW are unbiased when ¯ Q 0 is correctly estimated. In this case, both TMLE estimators have higher variance than aug-IPTW, and all three are more efficient than IPTW, but less efficient than the parametric MLE estimator. Though asymptotically the IPTW estimator is expected to be unbiased in this simulation, since g n is a consistent estimator of g 0 2 , these results demonstrate that in finite samples, heavily weighting a subset of observations not only increases variance, but can also bias the estimate.

When the model for ¯ Q 0 is misspecified in simulation 2, The MLE estimator is even more biased than it was in simulation 1. The efficiency of all three doublerobust efficient estimators suffers in comparison with simulation 1 as well. Nevertheless, TMLE log , using the logistic fluctuation, has the lowest MSE of all estimators. Its superiority over TMLE lin in terms of bias and variance is clear. TMLE log also outperforms aug-IPTW with respect to both bias and variance, and performs much better than IPTW or MLE.

## 4 Discussion

Whenanestimation procedure incorporates weights, observations with large weights can heavily influence the point estimate and inflate the variance. Truncating these weights is a common approach to reducing the variance, but it generally introduces bias. The presented TMLE of an additive causal effect of a point treatment intervention, incorporating a logistic fluctuation of the initial conditional mean estimate, dampens the effect of these heavily weighted observations, thereby heavily reducing the reliance on truncation. As a substitution estimator, the proposed TMLE of the additive causal effect respects the global constraints of the observed data model. Simulation study results indicate that this approach is on a par with, and in the context of sparsity often superior to, fluctuating on the linear scale. In particular it is more robust when there is sparsity in the data, outperforming MLE, IPTW, and aug-IPTW.

For the sake of demonstration we considered estimation of the additive causal effect. However, the same targeted MLE, using the logistic fluctuation, can be used to estimate other point-treatment causal effects, including parameters of a marginal structural model. The newly proposed loss function also has applications in prediction of a bounded outcome, and for targeted MLE of the causal effect of a multiple time point intervention in which the final outcome is bounded and continuous. We also pointed out that the proposed fluctuation function and loss function, and corresponding targeted MLE, should also be used for continuous outcomes for which no a priori bounds are known, by simply using the minimal and maximal observed outcome values. In this way, these choices naturally robustify the targeted MLE by enforcing that the updated initial estimator will not predict outcomes outside the observed range.

The TMLE approach presented here using a logistic fluctuation of an initial estimate of the conditional mean of the continuous outcome retains all properties of targeted maximum likelihood estimators, including influence curve-based inference. The method presented here extends to collaborative targeted maximum likelihood estimation without modification.

http://www.bepress.com/ijb/vol6/iss1/26

DOI: 10.2202/1557-4679.1260

## References

- R.H. Dehejia and S. Wahba. Propensity score matching methods for nonexperimental causal studies. The Review of Economics and Statistics , 84:151-61, 2002.
- M. A. Hernan, B. Brumback, and J. M. Robins. Marginal structural models to estimate the causal effect of zidovudine on the survival of HIV-positive men. Epidemiology , 11(5):561-570, 2000.
- P. McCullagh. Quasi-likelihood functions. Annals of Statistics , 11:59-67, 1983.
- J. M. Robins and A. Rotnitzky. Comment on the Bickel and Kwon article, 'Inference for semiparametric models: Some questions and an answer'. Statistica Sinica , 11(4):920-936, 2001.
- J. M. Robins, A. Rotnitzky, and M.J. van der Laan. Comment on 'On Profile Likelihood' by S.A. Murphy and A.W. van der Vaart. Journal of the American Statistical Association - Theory and Methods , 450:431-435, 2000.
- J.M. Robins. A new approach to causal inference in mortality studies with sustained exposure periods - application to control of the healthy worker survivor effect. Mathematical Modelling , 7:1393-1512, 1986.
- J.M. Robins. Robust estimation in sequentially ignorable missing data and causal inference models. In Proceedings of the American Statistical Association , 2000a.
- J.M. Robins. Marginal structural models versus structural nested models as tools for causal inference. In Statistical models in epidemiology, the environment, and clinical trials (Minneapolis, MN, 1997) , pages 95-133. Springer, New York, 2000b.
- O.M. Stitelman and M.J. van der Laan. Collaborative targeted maximum likelihood for time to event data. Technical Report 260, Division of Biostatistics, University of California, Berkeley, 2010.
- T.A. Stukel, E.S. Fisher, D.E. Wennberg, D.A.Alter, D.J.Gottlieb, and M.J. Vermeulen. Analysis of observational studies in the presence of treatment selection bias: Effects of invasive cardiac management on AMI survival using propensity score and instrumental variable methods. JAMA , 297:278-85, 2007.
- Z. Tan. Bounded, efficient, and doubly robust estimation with inverse weighting. Biometrika , 94:1-22, 2008.

- M.J. van der Laan and J.M. Robins. Unified methods for censored longitudinal data and causality . Springer, New York, 2003.
- M.J. van der Laan and D. Rubin. Targeted maximum likelihood learning. The International Journal of Biostatistics , 2(1), 2006.
- M.J. van der Laan, E. Polley, and A. Hubbard. Super learner. Statistical Applications in Genetics and Molecular Biology , 6(25), 2007. ISSN 1.
- M.J. van der Laan, S. Rose, and S. Gruber. Readings in targeted maximum likelihood estimation. Technical report 254, Division of Biostatistics, University of California, Berkeley, Sept 2009.
- R.W.M. Wedderburn. Quasi-likelihood functions, generalized linear models, and the Gauss-Newton method. Biometrika , 61, 1974.