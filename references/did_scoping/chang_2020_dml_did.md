## Semiparametric Difference-in-Differences with Potentially Many Control Variables

Neng-Chieh Chang ∗

## Abstract

This paper discusses difference-in-differences (DID) estimation when there exist many control variables, potentially more than the sample size. In this case, traditional estimation methods, which require a limited number of variables, do not work. One may consider using statistical or machine learning (ML) methods. However, by the well-known theory of inference of ML methods proposed in Chernozhukov et al. (2018), directly applying ML methods to the conventional semiparametric DID estimators will cause significant bias and make these DID estimators fail to be √ N -consistent. This article proposes three new DID estimators for three different data structures, which are able to shrink the bias and achieve √ N -consistency and asymptotic normality with mean zero when applying ML methods. This leads to straightforward inferential procedures. In addition, I show that these new estimators have the small bias property (SBP), meaning that their bias will converge to zero faster than the pointwise bias of the nonparametric estimator on which it is based.

Keyword: difference-in-differences, causal inference, high-dimensional data, Neyman orthogonality, √ N -consistency, undersmoothing

JEL Classification: C13, C14

## 1 Introduction

The difference-in-differences (DID) estimator has been widely used in empirical economics to evaluate causal effects when there exists a natural experiment with a treated group and an untreated group. By comparing the variation over time in an outcome variable between the treated group and the untreated group, the DID estimator can be used to calculate the effect of treatment on the outcome variable. Applications of DID include but are not limited to studies of the effects of immigration on labor markets (Card, 1990), the effects of minimum wage law on wages (Card &amp; Krueger, 1994), the effect of tariffs liberalization on corruption (Sequeira, 2016), the effect of household income on children's personalities (Akee, Copeland, Costello, &amp; Simeonova, 2018), and the effect of corporate tax on wages (Fuest, Peichl, &amp; Siegloch, 2018).

∗ Department of Economics, University of California Los Angeles, 315 Portola Plaza, Los Angeles, CA 90095, USA. email: nengchiehchang@g.ucla.edu

The traditional linear DID estimator depends on a parallel trend assumption that in the absence of treatment, the difference of outcomes between treated and untreated groups remains constant over time. In many situations, however, this assumption may not hold because there are other individual characteristics that may be associated with the variations of the outcomes. The treatment may be taken as exogenous only after controlling these characteristics. To address this problem, Abadie (2005) proposed the semiparametric DID estimators. Compared to the traditional linear DID estimators, the advantages of Abadie's estimators are threefold. First, the characteristics are treated nonparametrically so that any estimation error caused by functional specification is avoided. Second, the effect of treatment is allowed to vary among individuals, while the traditional linear DID estimator does not allow this heterogeneity. Third, the estimation framework proposed in Abadie (2005) allows researchers to estimate how the effect of treatment varies with changes in the characteristics.

This paper is an extension of Abadie (2005). Abadie (2005) considered the case where the number of control variables has to be limited. A practical difficulty empirical researchers encounter is choosing what variables to include when there is a rich data set. Although economic intuition can help us narrow down the choice set, it will not completely select all the important variables. This variable selection problem may lead to the chance of omitted variables in practice. In this paper, I consider the DID estimation with many control variables, potentially more than the sample size. The classical estimation methods which require a fixed number of variables do not work in this situation. One has to consider using ML methods such as Lasso, Logit Lasso, random forests, boosted trees, or various hybrids. However, by the well-known theory of inference of ML methods developed in Chernozhukov et al. (2018), if one directly applies ML methods to the conventional semiparametric DID estimators proposed in Abadie (2005), the result will lead to significant bias and invalid inference. In particular, the regularization bias embedded in ML methods will result in the conventional semiparametric DID estimators failing to be √ N -consistent.

I contribute to the literature by proposing three new DID estimators for three different data structures: repeated outcomes, repeated cross-sections, and multilevel treatment. These new estimators can relieve the impact of the regularization bias of ML methods and achieve √ N -consistency. The key is to find the so-called Neyman-orthogonal scores (Chernozhukov et al., 2018) of Abadie (2005)'s estimands. The Neyman-orthogonal score is a function that identifies the parameter of interest, and its derivatives with respect to the nuisance parameters are zero. This property helps us remove the first-order bias caused by ML methods so that only the second-order bias remains, which is much smaller and easier to control than the first-order bias as in the conventional semiparametric DID estimators. Using the cross-fitting algorithm in Chernozhukov et al. (2018), I show that the new DID estimators can be √ N -consistent and asymptotically normal when using ML methods. Figure 1 presents a Monte Carlo simulation that illustrates the negative effect of directly combining ML methods with Abadie's estimator and the benefit of using the newly proposed DID estimator.

Figure 1: The true value is θ 0 = 3 with sample size N = 200 and the number of control variables p = 300 . The left panel is the behavior of the conventional semiparametric DID estimator proposed in Abadie (2005), where I estimate the propensity score using Logit Lasso. The histogram shows that the simulated distribution of the conventional semiparametric DID estimator is biased. The right panel is the behavior of the new DID estimator proposed in this paper, which is constructed by the Neyman-orthogonal score and cross-fitting. The nuisance parameters are estimated by Logit Lasso and random forests. The simulated distribution of the new estimator is centered at the true value and normally distributed. Note that the simulated data are exactly the same for both panels, and the simulation setting is presented in Section 4.

<!-- image -->

The second contribution is concerned with the conventional semiparametric DID estimators with a limited number of control variables considered in Abadie (2005). In this case, the conventional semiparametric DID estimators are able to achieve √ N -consistency using kernel estimators, but they will require undersmoothing. Undersmoothing is a condition that requires the pointwise bias of the kernel estimators to converge to zero faster than the pointwise standard deviation. This condition will be violated if researchers use standard data-driven methods, such as cross-validation (CV), to choose the bandwidths of kernel estimators because those methods do not undersmooth.

√

In this paper, I show that the new estimators do not require undersmoothing to achieve N -consistency. Specifically, I will show that the new estimators have the small bias property (SBP), in terms of Newey, Hsieh, &amp; Robins (2004), meaning that the bias of the new estimators will converge to zero faster than the pointwise bias of the nonparametric estimator on which it is based. The SBP, as shown in Chernozhukov, Escanciano, Ichimura, &amp; Newey (2016), is a sufficient condition to remove the undersmoothing requirement. Figure 2 shows the Monte Carlo simulation results of Abadie's estimator and the new estimator with bandwidths chosen by CV. We can observe that Abadie's estimator is biased since CV does not undersmooth, and the newly proposed estimator can correct this bias.

Figure 2: The true value is θ 0 = 3 . The first-stage kernel estimators are constructed using standard Gaussian kernel with bandwidths chosen by CV. The simulated data are exactly the same for both estimators, and the simulation setting is presented in Section 4.

<!-- image -->

As an empirical example, I study the effect of tariff reduction on corruption behavior using the trade data between South Africa and Mozambique during 2006 and 2014. The treatment is the large tariff reduction on certain commodities occurring in 2008. This natural experiment was previously studied by Sequeira (2016) using the traditional linear DID estimator. I apply my proposed semiparametric DID estimator and Abadie (2005)'s semiparemetric DID estimator on the same data set (Table 9 of Sequeira (2016)). In comparison to Sequeira (2016) that a decrease in tariff rate will decrease corruption behavior, the two semiparametric estimators consistently suggest that the effect is actually substantially larger than previously reported by Sequeira (2016). A potential explanation for this difference is that the true data generating process violates the linear specification assumed in the traditional linear DID estimator. In addition, when compared to Abadie (2005)'s estimator, my proposed estimator shows that the effect is even larger.

The new estimators proposed in this paper heavily rely on the recent high-dimensional and ML literature: Belloni, Chen, Chernozhukov, &amp; Hansen (2012), Belloni, Chernozhukov, &amp; Hansen (2014), Chernozhukov, Hansen, &amp; Spindler (2015), Belloni, Chernozhukov, Fernández-Val, &amp; Hansen (2017), and Chernozhukov et al. (2018); and the literature of the SBP in semiparametric estimation: Newey, Hsieh, &amp; Robins (1998, 2004) and Chernozhukov, Escanciano, Ichimura, &amp; Newey (2016).

Plan of the paper. Section 2 describes the conventional semiparametric DID estimators and discusses their limitations when applying ML methods. Section 3 presents the new DID estimators and discusses their theoretical properties. Section 4 conducts Monte Carlo simulation to shed some light on the finite sample performance of the proposed estimators. Section 5 provides an application, and Section 6 concludes the paper.

## 2 The Conventional Semiparametric DID Estimators

Let Y i ( t ) be the outcome of interest for individual i at time t and D i ( t ) ∈ { 0 , 1 } the treatment status. The population is observed in a pre-treatment period t = 0 , and in a post-treatment period t = 1 . With potential outcome notations (Rubin, 1974), we have Y i ( t ) = Y 0 i ( t ) + ( Y 1 i ( t ) -Y 0 i ( t ) ) D i ( t ) , where Y 0 i ( t ) is the outcome that individual i would attain at time t in the absence of the treatment, and Y 1 i ( t ) represents the outcome that individual i would attain at time t if exposed to the treatment. Since individuals are only exposed to treatment at t = 1 , we have D i (0) = 0 for all i . To reduce notation, I define D i := D i (1) . Also, let X i ∈ R d be a vector of control variables with dimension d potentially larger than the sample size N .

The traditional linear DID estimator is the parameter α in the following linear model

<!-- formula-not-decoded -->

where ε i ( t ) is an exogenous shock that has mean zero and ( µ, π ( t ) , τ, δ ) are the corresponding parameters. Clearly, the linear specification assumed here is a strong assumption since the true data generating process may be nonlinear. In addition, Meyer, Viscusi, &amp; Durbin (1995) noticed that including control variables in this linear form may not be appropriate if the treatment has different effects for different groups in the population. To deal with these problems, Abadie (2005) proposed the semiparametric DID estimators which can identify average treatment effect on the treated (ATT)

<!-- formula-not-decoded -->

According to the data, there are three particular cases.

## Case 1: Random sample with repeated outcomes

Consider the case that researchers can observe both pre-treatment and post-treatment outcomes for each individual of interest. That is, researchers observe { Y i (0) , Y i (1) , D i , X i } N i =1 . In this case, the ATT can be identified under the following assumptions (Abadie, 2005):

<!-- formula-not-decoded -->

Assumption 2.2. P ( D i = 1) &gt; 0 and with probability one P ( D i = 1 | X i ) &lt; 1 .

Assumption (2.1) is the conditional parallel trend assumption. It states that conditional on individual's characteristics, the average outcomes for treated and untreated groups would have followed parallel paths in the absence of treatment. With these two assumptions, the ATT is identified (Abadie, 2005) as

<!-- formula-not-decoded -->

## Case 2: Random sample with repeated cross sections

Often times, researchers may not be able to observe both pre-treatment and post-treatment outcomes of the same individual. Instead, they observe repeated cross-section data sets. Let T i be a time indicator that takes value one if the observation belongs to the post-treatment sample. Researchers observe { Y i , D i , T i , X i } N i =1 , where Y i = Y i (0) + T i ( Y i (1) -Y i (0)) .

- Assumption 2.3. Conditional on T = 0 , the data are i.i.d. from the distribution of ( Y (0) , D, X ) ; conditional on T = 1 , the data are i.i.d. from the distribution of ( Y (1) , D, X ) .

Suppose Assumptions (2.1)-(2.3) hold, the ATT is identified (Abadie, 2005) as

<!-- formula-not-decoded -->

where λ 0 := P ( T i = 1) .

## Case 3: Multilevel treatments

In many cases, individuals can be exposed to different levels of treatment. Let W ∈ { 0 , w 1 , ..., w J } be the level of treatment, where W = 0 denotes the untreated individuals. Researchers observe { Y i (0) , Y i (1) , W i , X i } N i =1 .

For w ∈ { 0 , w 1 , ..., w J } and t ∈ { 0 , 1 } , let Y w ( t ) be the potential outcome for treatment level w at period t . Denote the ATT for each level of treatment w by

<!-- formula-not-decoded -->

Suppose that Assumptions (2.1) and (2.2) hold for each level of treatment:

<!-- formula-not-decoded -->

for w ∈ { w 1 , ..., w J } and P ( W i = w ) &gt; 0 and with probability one P ( W i = w | X i ) &lt; 1 for w ∈ { w 1 , ..., w J } . Then we have (Abadie, 2005)

<!-- formula-not-decoded -->

where I ( · ) is an indicator function.

Let us focus on Case 1 in which researchers confront repeated outcomes data. To use the identification result (2.1), the first step is to estimate the two nuisance parameters: P ( D i = 1) =: p 0 and P ( D i = 1 | X i ) =: g 0 ( X i ) . The estimator of p 0 is just a sample average ˆ p = N -1 ∑ N i =1 D i , while the propensity score g 0 is infinite-dimensional and needs to be estimated nonparametrically. Denote by ˆ g the estimator of g 0 , then the plug-in estimator based on equation (2.1) is

<!-- formula-not-decoded -->

When ˆ g is estimated using classical nonparametric methods such as kernel or series estimators, the estimator ˆ θ can be √ N -consistent and asymptotically normal under certain conditions provided in the semiparametric estimation literature (Newey, 1994; Newey &amp; McFadden, 1994).

When ˆ g is an ML estimator, however, the estimator ˆ θ will fail to be √ N -consistent in general. By the general theory of inference of ML methods developed in Chernozhukov et al. (2018), the reason is twofold : (1) the score function based on (2.1), ϕ ( W,θ 0 , p 0 , g 0 ) := Y (1) -Y (0) P ( D =1) D -g 0 ( X ) 1 -g 0 ( X ) -θ 0 , has a non-zero directional (Gateaux) derivative with respect to the propensity score g 0 :

̸

<!-- formula-not-decoded -->

where the directional (Gateaux) derivative is formally defined in Section 3; (2) ML estimators usually have a convergence rate slower than N -1 / 2 due to regularization bias. Similarly, the estimators obtained by directly plugging ML estimators into (2.2) and (2.3) will not be √ N -consistent in general. The Monte Carlo simulation in Section 4 supports this theoretical insight and reveals significant bias on the estimators based on (2.1)-(2.3) when using ML estimators in the first-stage nonparametric estimation.

The next section proposes three new score functions to relieve the regularization bias of the firststage ML estimators. These three new score functions are derived under the same identification assumptions as those in Abadie (2005), so that no extra assumption is made. Heuristically, a distinctive feature of the new score functions is that their derivatives with respect to their infinitedimensional nuisance parameters are zero. This property can help us remove the first-order bias of the first-stage estimation so that the bias of the estimators based on these new score functions will be much smaller. In addition, I will use the cross-fitting algorithm to improve the over-fitting phenomena that frequently arise when using highly adaptive ML methods (Chernozhukov et al., 2018).

## 3 The New DID Estimators

## 3.1 The Main Algorithm

Supposing Assumptions (2.1)-(2.3) hold, consider the following three new score functions.

## Case 1: Random sample with repeated outcomes

The new score function for repeated outcomes is

<!-- formula-not-decoded -->

with the unknown constant p 0 and the infinite-dimensional nuisance parameter

<!-- formula-not-decoded -->

## Case 2: Random sample with repeated cross sections

The new score function for repeated cross sections is

<!-- formula-not-decoded -->

where the adjustment term is

<!-- formula-not-decoded -->

The nuisance parameters are the unknown constants p 0 and λ 0 , and the infinite-dimensional parameter

<!-- formula-not-decoded -->

## Case 3: Multilevel treatment

For each w ∈ { w 1 , ..., w J } , the new score function for multilevel treatment is

<!-- formula-not-decoded -->

where the adjustment term is

<!-- formula-not-decoded -->

The nuisance parameters are the unknown constant p w 0 := P ( W = w ) and the infinite-dimensional parameter

<!-- formula-not-decoded -->

Notice that the above three new functions are equal to the original score functions (2.1)-(2.3) plus the adjustment terms, ( c 1 , c 2 , c w ) , which have zero expectations. Thus, the new score functions (3.1)-(3.3) still identify the ATT in each case. I will use these new scores to construct new DID estimators.

To avoid repetition, I will focus on the estimation of ATT when data belongs to repeated outcomes and repeated cross sections. The estimation of multilevel treatment is provided in appendix. Now I combine the score functions described above with the cross-fitting estimation algorithm of Chernozhukov et al. (2018).

## Algorithm 1

1. Take a K -fold random partition ( I k ) K k =1 of observation indices [ N ] = { 1 , ..., N } . For simplicity,

assume that each fold I k has the same size n = N/K . For each k ∈ [ K ] = { 1 , ..., K } , define the auxiliary sample I c k := { 1 , ..., N } \ I k .

2. For each k , construct the intermediate ATT estimators

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ p k = 1 n ∑ i ∈ I c k D i , ˆ λ k = 1 n ∑ i ∈ I c k T i , and ( ˆ g k , ˆ ℓ 1 k , ˆ ℓ 2 k ) are the estimators of ( g 0 , ℓ 10 , ℓ 20 ) constructed using the auxiliary sample I c k .

3. Construct the final ATT estimator ˜ θ = 1 K ∑ K k =1 ˜ θ k .

The estimators ( ˆ g k , ˆ ℓ 1 k , ˆ ℓ 2 k ) can be constructed using any ML methods or classical estimators such as kernel or series estimators. For completeness, I present the Logit Lasso and Lasso estimators here.

Consider a class of approximating functions of X i ,

<!-- formula-not-decoded -->

For example, q i can be polynomials or B-splines. Let Λ( u ) := 1 / (1 + exp ( -u )) be the cumulative distribution function of the standard Logistic distribution, construct the estimator of the propensity score g 0 by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the Logit Lasso estimator and M = N -n is the sample size of the auxiliary sample I c k . Next, define I c kz := I c k ∩ { i : D i = 0 } , M k the sample size of I c kz . Construct the estimators of ℓ 10 and ℓ 20 by

<!-- formula-not-decoded -->

where

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

are the modified Lasso estimators proposed in Belloni, Chen, Chernozhukov, &amp; Hansen (2012). The choices of the penalty levels and loadings ( λ 1 k , λ 2 k , ˆ Υ 1 k , ˆ Υ 2 k ) suggested by Belloni, Chen, Chernozhukov, &amp; Hansen (2012) are provided in appendix.

## 3.2 Theoretical Properties

In this section, I discuss the theoretical properties of the new DID estimator ˜ θ . In particular, I will show that the estimator ˜ θ can achieve √ N -consistency and asymptotic normality as long as the first-stage estimators converge at rates faster than N -1 / 4 . This rate of convergence can be achieved by many ML methods, including Lasso and Logit Lasso. Further, I will show that when using kernel estimators in the first-stage estimation, the estimator ˜ θ has the SBP while the conventional semiparametric DID estimators do not.

## 3.2.1 The Neyman Orthogonality

The differences between the new DID estimators and the conventional semiparametric DID estimators in Abadie (2005) are the score functions on which they are based. The key property of the new score functions (3.1)-(3.3) is that their directional (or the Gateaux) derivatives with respect to their infinite-dimensional nuisance parameters are zero, while the scores based on (2.1)-(2.3) do not have this property. This property is the so-called Neyman orthogonality in Chernozhukov et al. (2018). The Neyman orthogonality enables us to remove the first-order bias of the first-stage estimation so that the estimators based on these Neyman-orthogonal scores can achieve √ N -consistency under less restrictive conditions.

The definition of the Neyman-orthogonal score provided here is slightly different from Chernozhukov et al. (2018) that instead of being orthogonal against all nuisance parameters, the Neyman- orthogonal score defined here is orthogonal against only those infinite-dimensional nuisance parameters. Formally, let θ 0 ∈ Θ be the low-dimensional parameter of interest, ρ 0 be the true value of the finite-dimensional nuisance parameter ρ , and η 0 the true value of the infinite-dimensional nuisance parameter η ∈ T . Suppose that W is a random element taking values in a measurable space ( W , A W ) with probability measure P . Define the directional (or the Gateaux) derivative against the infinite-dimensional nuisance parameter D r : ˜ T → R , where ˜ T = { η -η 0 : η ∈ T } ,

<!-- formula-not-decoded -->

for all r ∈ [0 , 1) . For convenience, denote

<!-- formula-not-decoded -->

In addition, let T N ⊂ T be a nuisance realization set such that the estimator of η 0 take values in this set with high probability.

Definition (The Neyman Orthogonality)

The score ψ obeys the Neyman orthogonality condition at ( θ 0 , ρ 0 , η 0 ) with respect to the nuisance parameter realization set T N ⊂ T if the directional derivative map D r [ η -η 0 ] exists for all r ∈ [0 , 1) and η ∈ T N and vanishes at r = 0 :

<!-- formula-not-decoded -->

Lemma 1 The new score functions (3.1)-(3.3) obey the Neyman orthogonality.

This property embedded in (3.1)-(3.3) will play the key role to make less restrictive assumptions in the following proofs of asymptotic distribution and the SBP.

## 3.2.2 Asymptotic Distribution

In the following, I will discuss the theoretical properties of the new estimator ˜ θ when data belongs to repeated outcomes and repeated cross sections. The results of multilevel treatment can be proven using the same arguments. Let κ and C be strictly positive constants, K ≥ 2 be a fixed integer, and ε N be a sequence of positive constants approaching zero. Denote by ‖ · ‖ P,q the L q norm of some probability measure P : ‖ f ‖ P,q := (∫ | f ( w ) | q dP ( w ) ) 1 /q and ‖ f ‖ P, ∞ := sup w | f ( w ) | .

Assumption 3.1 (Regularity Conditions for Repeated Outcomes)

Let P be the probability law for ( Y (0) , Y (1) , D, X ) . Let D = g 0 ( X ) + U and Y (1) -Y (0) = ℓ 10 ( X )+ V 1 with E P [ U | X ] = 0 and E P [ V 1 | X,D = 0] = 0 . Define G 1 p 0 := E P [ ∂ p ψ 1 ( W,θ 0 , p 0 , η 10 )] and Σ 10 := E P [ ( ψ 1 ( W,θ 0 , p 0 , η 10 ) + G 1 p 0 ( D -p 0 )) 2 ] . Suppose the following conditions hold: (a) Pr ( κ ≤ g 0 ( X ) ≤ 1 -κ ) = 1 ; (b) ‖ UV 1 ‖ P, 4 ≤ C ; (c) E [ U 2 | X ] ≤ C ; (d) E [ V 2 1 | X ] ≤ C ; (e) Σ 10 &gt; 0 ; and (f) given the auxiliary sample I c k , the estimator ˆ η 1 k = ( ˆ g k , ˆ ℓ 1 k ) obeys the following conditions. With probability 1 -o (1) , ‖ ˆ η 1 k -η 10 ‖ P, 2 ≤ ε N , ‖ ˆ g k -1 / 2 ‖ P, ∞ ≤ 1 / 2 -κ , and ‖ ˆ g k -g 0 ‖ 2 P, 2 + ‖ ˆ g k -g 0 ‖ P, 2 × ‖ ˆ ℓ 1 k -ℓ 10 ‖ P, 2 ≤ ( ε N ) 2 .

Assumption 3.2 (Regularity Conditions for Repeated Cross Sections)

Let P be the probability law for ( Y, T, D, X ) . Let D = g 0 ( X )+ U and ( T -λ 0 ) Y = ℓ 20 ( X )+ V 2 with E p [ U | X ] = 0 and E p [ V 2 | X,D = 0] = 0 . Define G 2 p 0 := E P [ ∂ p ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 )] , G 2 λ 0 := E P [ ∂ λ ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 )] , and Σ 20 := E P [ ( ψ 1 ( W,θ 0 , p 0 , η 10 ) + G 2 p 0 ( D -p 0 ) + G 2 λ 0 ( T -λ 0 )) 2 ] . Suppose the following conditions hold: (a) Pr ( κ ≤ g 0 ( X ) ≤ 1 -κ ) = 1 ; (b) ‖ UV 2 ‖ P, 4 ≤ C ; (c) E [ U 2 | X ] ≤ C ; (d) E [ V 2 2 | X ] ≤ C ; (e) E P [ Y 2 | X ] ≤ C ; (f) | E P [ Y U ] |≤ C ; (g) Σ 20 &gt; 0 ; and (h) given the auxiliary sample I c k , the estimators ˆ η 2 k = ( ˆ g k , ˆ ℓ 2 k ) obeys the following conditions. With probability 1 -o (1) , ‖ ˆ η 2 k -η 20 ‖ P, 2 ≤ ε N , ‖ ˆ g k -1 / 2 ‖ P, ∞ ≤ 1 / 2 -κ , and ‖ ˆ g k -g 0 ‖ 2 P, 2 + ‖ ˆ g k -g 0 ‖ P, 2 × ‖ ˆ ℓ 2 k -ℓ 20 ‖ P, 2 ≤ ( ε N ) 2 .

## Theorem 1

For repeated outcomes, suppose Assumptions (2.1), (2.2) and (3.1) hold. For repeated cross sections, suppose Assumptions (2.1)-(2.3) and (3.2) hold. If ε N = o ( N -1 / 4 ) , then the new ATT estimator ˜ θ satisfies

<!-- formula-not-decoded -->

with Σ = Σ 10 for repeated outcomes and Σ = Σ 20 for repeated cross sections.

## Theorem 2 (Variance Estimator)

Construct the estimators of the asymptotic variances as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where E n,k [ f ( W )] = n -1 ∑ i ∈ I k f ( W i ) , ˆ G 1 p = ˆ G 2 p = -˜ θ/ ˆ p k , and ˆ G 2 λ is a consistent estimator of G 2 λ 0 . If the assumptions of Theorem 1 hold, then ˆ Σ 1 = Σ 10 + o P (1) and ˆ Σ 2 = Σ 20 + o P (1) .

The interpretation of Theorem 1 and 2 is that the new DID estimator ˜ θ can achieve √ N -consistency and asymptotic normality provided that the first-stage estimators of the infinite dimensional nuisance parameters converge at a rate faster than N -1 / 4 . This rate of convergence can be achieved by many ML methods. In particular, Van de Geer (2008) and Belloni, Chen, Chernozhukov, &amp; Hansen (2012) provided detail conditions for Logit Lasso and the modified Lasso estimators to satisfy this rate of convergence. It is also worth noting that even when the first-stage estimators do not converge as fast as N -1 / 4 , the new estimator ˜ θ still has smaller bias than the original estimator because the Neyman orthogonality removes the first-order bias of the first-stage estimators.

## 3.2.3 The Small Bias Property

Consider the conventional semiparametric DID estimators with a limited number of control variables studied in Abadie (2005). Let ̂ g h be the kernel estimator of g 0 with bandwidth h → 0 in (2.1) and (2.2). Under the standard assumptions of kernel estimation (Assumption (3.3) below), one can show that the pointwise bias of ˆ g h is of order O ( h m ) , where m can be interpreted as the minimum number of derivatives of g 0 ; and the pointwise standard deviation is sd (ˆ g h ( x )) = O ( ( Nh d +2 s ) -1 / 2 ) . By Theorem 8.11 of Newey &amp; McFadden (1994), one can show that the √ N -consistency of the plugin estimators based on (2.1) and (2.2) requires √ Nh m → 0 . That is, the pointwise bias of the kernel estimator has to converge to zero faster than N -1 / 2 . Since the pointwise standard deviation converges to zero slower than N -1 / 2 , undersmoothing is required. In this case, standard data-driven bandwidth selection methods which do not undersmooth, such as cross-validation, are invalid.

To avoid undersmoothing, by the analysis of SBP in Newey, Hsieh, &amp; Robins (1998, 2004), the estimator of the parameter of interest needs to have smaller bias than the pointwise bias of the first-stage nonparametric estimators. That is, the SBP requires that the bias of the estimator of θ 0 converges to zero faster than h m .

In the following, I will show that the new DID estimator ˜ θ has the SBP. Let ( ˆ g kh , ˆ ℓ 1 kh , ˆ ℓ 2 kh ) be the kernel estimators of ( g 0 , ℓ 10 , ℓ 20 ) using auxiliary sample I c k . I assume here that they have the same bandwidth h and kernel K ( u ) for convenience.

Assumption 3.3 (Newey &amp; McFadden, 1994)

1. K ( u ) is differentiable of order s , the derivatives of order s are bounded, K ( u ) is zero outside a bounded set, ∫ K ( u ) du = 1 , there is a positive m such that for all j &lt; m , ∫ K ( u ) [ ⊗ j ℓ =1 u ] du = 0 .
2. Define γ 0 ( x ) = f 0 ( x ) E ( z | x ) , where z ∈ (1 , D, Y (1) -Y (0) | D = 0 , ( T -λ 0 ) Y | D = 0) and f 0 ( x ) is the true density of x . Assume that γ 0 ( x ) is continuously differentiable to order s with bounded derivatives on an open set containing X , where X is the support of x .
3. There is α ≥ 4 such that E [ | z | α ] &lt; ∞ and E [ | z | α | x ] f 0 ( x ) is bounded.

## Theorem 3

For repeated outcomes, suppose Assumptions (2.1), (2.2), (3.1), and (3.3) hold. For repeated cross sections, suppose Assumptions (2.1)-(2.3), (3.2), and (3.3) hold. Suppose that inf x ∈X f 0 ( x ) = 0 , h = h ( N ) with log N/ ( √ Nh d +2 s ) → 0 . If √ Nh 2 m → 0 , then

<!-- formula-not-decoded -->

with Σ = Σ 10 for repeated outcomes and Σ = Σ 20 for repeated cross sections.

The interpretation of Theorem 3 is that the new estimator ˜ θ only requires √ Nh 2 m → 0 to achieve √ N -consistency, while the conventional semiparametric DID estimators require √ Nh m → 0 under the same assumptions. With the Neyman orthogonality, the bias of ˜ θ is only of the second-order of the pointwise bias of the first-stage kernel estimators. The bias of ˜ θ is h 2 m instead of h m . Hence, ˜ θ satisfies the SBP. In particular, the bandwidth h such that log N/ ( √ Nh d +2 s ) → 0 and √ Nh 2 m → 0

̸

exists only if 2 m &gt; d + 2 s . Under this condition, the optimal bandwidth selected by minimizing mean-square errors (CV), h = N -1 / ( d +2 s +2 m ) , satisfies the conditions for √ N -consistency.

## Theorem 4

Construct the estimators of the asymptotic variances as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ G 1 p = ˆ G 2 p = -˜ θ/ ˆ p k and ˆ G 2 λ is a consistent estimator of G 2 λ 0 . If the assumptions of Theorem 3 hold, then ˆ Σ 1 = Σ 10 + o P (1) and ˆ Σ 2 = Σ 20 + o P (1) .

## 4 Simulation

In this section, I present Monte Carlo simulation results of the conventional semiparametric DID estimators and the new DID estimator ˜ θ in three different data structures: repeated outcomes, repeated cross sections, and multilevel treatment. I use both ML methods and kernel estimators in the first-stage estimation. For ML estimation, I generate high-dimensional (HD) data and estimate the propensity score by Logit Lasso (Multi-Logit Lasso for multilevel treatment). To choose the penalty parameter for Logit Lasso (Multi-Logit Lasso), I use K -fold CV (as recommended by Van de Geer (2008)) with K = 10 . Alternatively, one could use a method developed in Belloni, Chernozhukov, Chetverikov, &amp; Wei (2018). The other infinite-dimensional nuisance parameters are estimated by random forests with 500 regression trees. For kernel estimation, all the infinite-dimensional nuisance parameters are estimated using the standard Gaussian kernel.

Figure 3-20 in appendix show the simulation results. I find that the conventional semiparametric DID estimators are biased when using ML methods, while the new DID estimator ˜ θ can correct the bias. For kernel estimation, the conventional DID estimator with bandwidth selected by CV is biased, while the new DID estimator ˜ θ is centered at the true value. The data generating processes are presented in the following.

## 4.1 Repeated Outcomes

## 4.1.1 ML Estimation

Let N ∈ { 200 , 500 } be the sample size and p ∈ { 100 , 300 } the dimension of control variables, X i ∼ N (0 , I p × p ) . Also, let γ 0 = (1 , 1 / 2 , 1 / 3 , 1 / 4 , 1 / 5 , 0 , ..., 0) ∈ R p and D i is generated by the propensity score

<!-- formula-not-decoded -->

At t = 0 , the potential outcome is generated

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where β 0 = γ 0 + 0 . 5 and θ 0 = 3 , and all error terms follow N (0 , 0 . 1) . Researchers observe { Y i (0) , Y i (1) , D i , X i } for i = 1 , ..., N , where Y i (0) = Y 0 i (0) and Y i (1) = Y 0 i (1) (1 -D i ) + Y 1 i (1) D i . Figure 3-6 present the results.

## 4.1.2 Kernel Estimation

Let N ∈ { 200 , 500 } be the sample size, D i ∼ Bernoulli (0 . 5) , and X i | D i ∼ N ( D i , 1) . At t = 0 , the potential outcome is generated

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where θ 0 = 3 and all error terms follow N (0 , 0 . 1) . Researchers observe { Y i (0) , Y i (1) , D i , X i } for i = 1 , ..., N , where Y i (0) = Y 0 i (0) and Y i (1) = Y 0 i (1) (1 -D i ) + Y 1 i (1) D i . Figure 7-8 present the

and at t = 1 ,

and at t = 1 , results.

## 4.2 Repeated Cross Sections

## 4.2.1 ML Estimation

Let N ∈ { 200 , 500 } be the sample size and p ∈ { 100 , 300 } the dimension of control variables, X i ∼ N (0 . 3 , I p × p ) . Also, let γ 0 = (1 , 1 / 2 , 1 / 3 , 1 / 4 , 1 / 5 , 0 , ..., 0) ∈ R p and D is generated by the propensity score

<!-- formula-not-decoded -->

At t = 0 , the potential outcome is generated

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where β 0 = γ 0 +0 . 5 and θ 0 = 3 , and all error terms follow N (0 , 0 . 1) . Define Y i (0) = Y 0 i (0) and Y i (1) = Y 0 i (1) (1 -D i ) + Y 1 i (1) D i . Let T i follow a Bernoulli distribution with parameter 0 . 5 . Researchers observe { Y i , T i , D i , X i } for i = 1 , ..., N , where Y i = Y i (0) + T i ( Y i (1) -Y i (0)) . Figure 9-12 present the results.

## 4.2.2 Kernel Estimation

Let N ∈ { 200 , 500 } be the sample size, D i ∼ Bernoulli (0 . 5) , and X i | D i ∼ N ( D i , 1) . At t = 0 , the potential outcome is generated

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and at t = 1 ,

and at t = 1 , where θ 0 = 3 and all error terms follow N (0 , 0 . 1) . Let Y i (0) = Y 0 i (0) and Y i (1) = Y 0 i (1) (1 -D i )+ Y 1 i (1) D i . Let T i ∼ Bernoulli (0 . 5) . Researchers observe { Y i , T i , D i , X i } for i = 1 , ..., N , where Y i = Y i (0) + T i ( Y i (1) -Y i (0)) . Figure 13-14 present the results.

## 4.3 Multilevel Treatment

## 4.3.1 ML Estimation

Suppose there are two levels of treatment so that W ∈ { 0 , 1 , 2 } . Let N ∈ { 200 , 500 } be the sample size and p ∈ { 100 , 300 } the dimension of control variables, X i ∼ N (0 , I p × p ) . Also, let γ 0 = (1 , 1 / 2 , 1 / 3 , 1 / 4 , 1 / 5 , 0 , ..., 0) ∈ R p and

<!-- formula-not-decoded -->

At t = 0 , the potential outcome is generated

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where β 0 = γ 0 + 0 . 5 and θ 10 = 3 and θ 20 = 6 , and all error terms follow N (0 , 0 . 1) . Researchers observe { Y i (0) , Y i (1) , W i , X i } for i = 1 , ..., N , where Y i (0) = Y 0 i (0) and Y i (1) = Y 0 i (1) I ( W i = 0) + Y 1 i (1) I ( W i = 1) + Y 2 i (1) I ( W i = 2) . I focus on the estimation of the second level ATT θ 20 . Figure 15-18 present the results

## 4.3.2 Kernel Estimation

Suppose there are two levels of treatment so that W ∈ { 0 , 1 , 2 } . Let N be the sample size, X i | W i ∼ N ( W i , 1) , and

and at t = 1 , and at t = 1 ,

<!-- formula-not-decoded -->

At t = 0 , the potential outcome is generated

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where θ 10 = 3 , θ 20 = 6 , and all error terms follow N (0 , 0 . 1) . Let Y i (0) = Y 0 i (0) and Y i (1) = Y 0 i (1) I ( W i = 0)+ Y 1 i (1) I ( W i = 1)+ Y 2 i (1) I ( W i = 2) . Researchers observe { Y i (0) , Y i (1) , W i , X i } for i = 1 , ..., N . I focus on the estimation of the second level ATT θ 20 . Figure 19-20 present the results.

## 5 Empirical Example

In this example, I analyze the effect of tariffs reduction on corruption behaviors using the bribe payment data collected by Sequeira (2016) between South Africa and Mozambique. There have been theoretical and empirical debates on whether higher tariff rates increase incentives for corruption to occur (Clotfelter, 1983; Sequeira &amp; Djankov, 2014) or lower tariffs encourage agents to pay higher bribes through an income effect (Feinstein, 1991; Slemrod &amp; Yitzhaki, 2002). The former argues that an increase in the tariff rate makes it more profitable to evade taxes on the margin. The latter argues that an increased tariff rate makes the tax payer less wealthy and this, under the decreasing risk aversion of being penalized, tend to reduce evasion (Allingham &amp; Sandmo, 1972).

Sequeira (2016) collected primary data on the bribed payments between the ports in Mozambique and South Africa from 2007 to 2013. The treatment is the large reduction in the average nominal tariff rate (of 5 percent) occurring in 2008. Since not all products were on the tariff reduction list, a credible control group of products is available. This allows for a DID estimation.

This natural experiment between South Africa and Mozambique was previously studied by Sequeira (2016) by pooling the cross section data between 2007 and 2013, with sample size N = 1084 , and estimating the effect of treatment using the traditional linear DID. Here I focus on the specification of one of the main results (Table 9 of Sequeira (2016)):

<!-- formula-not-decoded -->

where y it is the natural log of the amount of bribe paid for shipment i in period t , conditional on paying a bribe. TariffChangeCategory ∈ { 0 , 1 } denotes the treatment status of commodities, POST ∈ { 0 , 1 } is an indicator for the years following 2008, and BaselineTariff is the tariff rate before the tariff reduction. The specification also includes a vector of characteristics Γ i , and time and individual fixed effects p i , w t , and δ i . The parameter γ 1 is the parameter of interest in the traditional linear DID estimation. Sequeira (2016) found that the amount of bribes paid dropped after the tariff reduction ( ˆ γ 1 = -2 . 928 ∗∗ ).

I use the same data set but instead of using the traditional linear DID estimation, I estimate the ATT by Abadie (2005)'s DID estimator and my proposed DID estimator ˜ θ . Since the data is repeated cross sections, I construct the estimators based on (2.2) and (3.2), respectively. The estimators with the first-stage kernel estimation contain one individual characteristic (the natural log of shipment value per ton), which is a significant characteristic in Table 9 of Sequeira (2016). The estimators with the first-stage Lasso estimation contain a list of the significant characteristics in Table 9 of Sequeira (2016), which includes product, shipment, firm-level characteristics, and their interaction terms. Table 1 below shows the results. I find that all these estimators consistently suggest that a decrease in tariff rate will lead to less bribes payment, but the effect of treatment may be actually substantially larger than previously reported by Sequeira (2016).

Table 1

|     | Sequeira (2016)   | Abadie (kernel)   | ˜ θ (kernel)     | Abadie (Lasso)   | ˜ θ (Lasso)     |
|-----|-------------------|-------------------|------------------|------------------|-----------------|
| ATT | -2.928** (0.944)  | -7.986** (3.028)  | -8.670** (3.643) | -7.499** (2.746) | -9.191* (4.854) |

## 6 Conclusion

In this article, I have introduced three new DID estimators based on the newly-derived Neymanorthogonal scores. These new scores do not require any additional conditions other than the original conditions made in Abadie (2005). The new DID estimators will be particularly appropriate when researchers would like to use ML methods in the first-stage nonparametric estimation. When using kernel estimators in the first-stage estimation , the new DID estimators do not require undersmoothing to achieve √ N -consistency. Hence, researchers can use standard data-driven methods, such as CV, to select bandwidths.

## References

- Abadie, A. (2005). Semiparametric difference-in-differences estimators. The Review of Economic Studies , 72 (1), 1-19.
- Akee, R., Copeland, W., Costello, E. J., &amp; Simeonova, E. (2018). How does household income affect child personality traits and behaviors? American Economic Review , 108 (3), 775-827.
- Allingham, M. G., &amp; Sandmo, A. (1972). Income tax evasion: A theoretical analysis. Journal of public economics , 1 , 323-338.
- Belloni, A., Chen, D., Chernozhukov, V., &amp; Hansen, C. (2012). Sparse models and methods for optimal instruments with an application to eminent domain. Econometrica , 80 (6), 2369-2429.
- Belloni, A., Chernozhukov, V., Chetverikov, D., &amp; Wei, Y. (2018). Uniformly valid postregularization confidence regions for many functional parameters in z-estimation framework. The Annals of Statistics , 46 (6B), 3643-3675.
- Belloni, A., Chernozhukov, V., Fernández-Val, I., &amp; Hansen, C. (2017). Program evaluation and causal inference with high-dimensional data. Econometrica , 85 (1), 233-298.
- Belloni, A., Chernozhukov, V., &amp; Hansen, C. (2014). Inference on treatment effects after selection among high-dimensional controlsâĂă. The Review of Economic Studies , 81 (2), 608-650.

- Card, D. (1990). The impact of the mariel boatlift on the miami labor market. ILR Review , 43 (2), 245-257.
- Card, D., &amp; Krueger, A. (1994). Minimum wages and employment: a case study of the fast-food industry in new jersey and pennsylvania. American Economic Review , 84 (4), 772-793.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., &amp; Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal , 21 (1), C1-C68.
- Chernozhukov, V., Escanciano, J. C., Ichimura, H., &amp; Newey, W. K. (2016). Locally robust semiparametric estimation. arXiv preprint arXiv:1608.00033 .
- Chernozhukov, V., Hansen, C., &amp; Spindler, M. (2015). Valid post-selection and post-regularization inference: An elementary, general approach. Annu. Rev. Econ. , 7 (1), 649-688.
- Clotfelter, C. T. (1983). Tax evasion and tax rates: An analysis of individual returns. The review of economics and statistics , 363-373.
- Feinstein, J. S. (1991). An econometric analysis of income tax evasion and its detection. The RAND Journal of Economics , 14-35.
- Fuest, C., Peichl, A., &amp; Siegloch, S. (2018). Do higher corporate taxes reduce wages? micro evidence from germany. American Economic Review , 108 (2), 393-418.
- Meyer, B. D., Viscusi, W. K., &amp; Durbin, D. L. (1995). Workers' compensation and injury duration: evidence from a natural experiment. The American economic review , 322-340.
- Newey, W. K. (1994). The asymptotic variance of semiparametric estimators. Econometrica: Journal of the Econometric Society , 1349-1382.
- Newey, W. K., Hsieh, F., &amp; Robins, J. (1998). Undersmoothing and bias corrected functional estimation.
- Newey, W. K., Hsieh, F., &amp; Robins, J. M. (2004). Twicing kernels and a small bias property of semiparametric estimators. Econometrica , 72 (3), 947-962.

- Newey, W. K., &amp; McFadden, D. (1994). Large sample estimation and hypothesis testing. Handbook of econometrics , 4 , 2111-2245.
- Rubin, D. B. (1974). Estimating causal effects of treatments in randomized and nonrandomized studies. Journal of educational Psychology , 66 (5), 688.
- Sequeira, S. (2016). Corruption, trade costs, and gains from tariff liberalization: evidence from southern africa. American Economic Review , 106 (10), 3029-63.
- Sequeira, S., &amp; Djankov, S. (2014). Corruption and firm behavior: Evidence from african ports. Journal of International Economics , 94 (2), 277-294.
- Slemrod, J., &amp; Yitzhaki, S. (2002). Tax avoidance, evasion, and administration. In Handbook of public economics (Vol. 3, pp. 1423-1470). Elsevier.
- Van de Geer, S. (2008). High-dimensional generalized linear models and the lasso. The Annals of Statistics , 36 (2), 614-645.

## APPENDIX

## Multilevel Treatment

Similarly, I use the cross-fitting algorithm (Chernozhukov et al., 2018).

1. Take a K -fold random partition ( I k ) K k =1 of observation indices [ N ] = { 1 , ..., N } such that the size of each fold I k is n = N/K . For each k ∈ [ K ] = { 1 , ..., K } , define the auxiliary sample I c k := { 1 , ..., N } \ I k .
2. For each k ∈ [ K ] , construct the estimator of p 0 and λ 0 by ˆ p w = 1 n ∑ i ∈ I c k D i . Also, construct the estimators of g w , g z , and ℓ 30 using the auxiliary sample I c k : ˆ g wk = ˆ g w ( ( W i ) i ∈ I c k ) , ˆ g zk = ˆ g z ( ( W i ) i ∈ I c k ) , and ˆ ℓ 3 k = ˆ ℓ 3 ( ( W i ) i ∈ I c k ) .
3. For each k , construct the intermediate ATT estimators

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The estimator ˆ g wk and ˆ g zk can be constructed by Multi-Logit Lasso.

## The Lasso Penalty

The following is suggested by Belloni, Chen, Chernozhukov, &amp; Hansen (2012). Let y i denote Y i (1) -Y i (0) or ( T i -ˆ λ k ) , λ k denote λ 1 k or λ 2 k , and ˆ Υ k denote ˆ Υ 1 k or ˆ Υ 2 k . For k ∈ [ K ] , the loading ˆ Υ k is a diagonal matrix with entries ˆ γ kj , j = 1 , ..., p , constructed by the following steps:

<!-- formula-not-decoded -->

where ¯ y k = M -1 ∑ i ∈ I c k y i , c &gt; 1 and γ → 0 . The empirical residual ˆ ε i is calculated by the modified Lasso estimator β ∗ k in the previous step: ˆ ε i = y i -q ′ i β ∗ k . Repeat the second step B &gt; 0 times to obtain the final loading.

## PROOFS

## Proof of Lemma 1

## Repeated outcomes:

The Gateaux derivative of (3.1) in the direction η 1 -η 10 = ( g -g 0 , ℓ 1 -ℓ 10 ) is

<!-- formula-not-decoded -->

where the second inequality follows from the law of iterated expectations, the third from the definition of ℓ 10 ( X ) and E P [ D -g 0 ( X ) | X ] = 0 .

## Repeated cross sections:

Define ∂ η 2 E P [ ψ 20 ] ( η 2 -η 20 ) := ∂ η 2 E P [ ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 )] ( η 2 -η 20 ) . Similar to the proof of repeated outcomes, the Gateaux derivative of (3.2) in the direction η 2 -η 20 = ( g -g 0 , ℓ 2 -ℓ 20 ) is

<!-- formula-not-decoded -->

where p ′ 0 := p 0 λ 0 (1 -λ 0 ) .

## Multilevel treatment:

Let ∆ w = g w -g w 0 , ∆ z = g z -g z 0 , and ∆ ℓ 3 = ℓ 3 -ℓ 30 . The Gateaux derivative of (3.3) in the direction η w -η w 0 = ( g w -g w 0 , g z -g z 0 , ℓ 3 -ℓ 30 ) is

<!-- formula-not-decoded -->

by the law of iterated expectation on each terms.

The proofs of Theorem 1 and 2 follow the general framework proposed in Chernozhukov et al. (2018).

## Proof of Theorem 1

## Repeated Outcomes:

The proof proceeds in five steps. In Step 1, I show the main result using the auxiliary results (A.1)-(A.4). In Step 2-5, I prove the auxiliary results.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where T N is the set of all η 1 = ( g, ℓ 1 ) consisting of P -square-integrable functions g and ℓ 1 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and P N is the set of all p &gt; 0 such that | p -p 0 |≤ N -1 / 2 . By the regularity condition (3.1) and | ˆ p k -p 0 | = O P ( N -1 / 2 ) , ˆ η 1 k ∈ T N and ˆ p k ∈ P N with probability 1 -o (1) .

Step 1. Observe that we have the decomposition

<!-- formula-not-decoded -->

where ¯ p k ∈ (ˆ p k , p 0 ) . For term (a), by the triangle inequality, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

To bound J 2 ,k , we have

<!-- formula-not-decoded -->

where the last inequality follows from the regularity condition (3.1). By Chebyshev's inequality, J 2 ,k = O P ( n -1 / 2 ) = o P (1) . Next, we bound J 1 ,k . Conditional on the auxiliary sample I c k , ˆ η 1 k can be treated as fixed. Under the event that ˆ η 1 k ∈ T N , we have

<!-- formula-not-decoded -->

by (A.3). By Lemma A.1, J 1 ,k = O P ( ε N ) = o P (1) . Together, we have

<!-- formula-not-decoded -->

For term (b), by the triangle inequality, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

To bound J 4 ,k , we have

<!-- formula-not-decoded -->

where the last inequality follows from the regularity conditions. By Chebyshev's inequality, J 4 ,k = O P ( n -1 / 2 ) = o P (1) . Conditional on I c k , both ¯ p k and ˆ η 1 k can be treated as fixed. Under the event that ˆ p k ∈ P N (thus ¯ p k ∈ P N ) and ˆ η 1 k ∈ T N , we have

<!-- formula-not-decoded -->

by (A.4). By Lemma A.1, J 3 ,k = O P ( ε N ) = o P (1) . Hence, E n,k [ ∂ 2 p ψ 1 ( W,θ 0 , ¯ p k , ˆ η 1 k ) ] = O P (1) .

Combine the above results with ˆ p k -p 0 = E n,k [ D -p 0 ] and (ˆ p k -p 0 ) 2 = O P ( N -1 ) , the decomposition of ˜ θ becomes

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

If we can show that √ NR N = o P (1) , then we are done.

This part is essentially identical to Step 3 in the proof of Theorem 3.1 (DML2) in Chernozhukov et al. (2018). I reproduce it here for reader's convenience. Since K is a fixed integer, which is independent of N , it suffices to show that for any k ∈ [ K ] ,

<!-- formula-not-decoded -->

Define the empirical process notation:

<!-- formula-not-decoded -->

where φ is any P -integrable function on W . By the triangle inequality, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

To bound I 1 ,k , note that conditional on ( W i ) i ∈ I c k the estimator ˆ η 1 k is nonstochastic. Under the event that ˆ η 1 k ∈ T N , we have

<!-- formula-not-decoded -->

by (A.1). Hence, I 1 ,k = O P ( ε N ) by Lemma A.1. To bound I 2 ,k , define the following function

<!-- formula-not-decoded -->

By Taylor series expansion, we have

<!-- formula-not-decoded -->

Note that f k (0) = 0 since E [ ψ 1 ( W,θ 0 , p 0 , η 10 ) | ( W i ) i ∈ I c k ] = E [ ψ 1 ( W,θ 0 , p 0 , η 10 )] . Further, on the event ˆ η 1 k ∈ T N ,

<!-- formula-not-decoded -->

by the orthogonality of ψ 1 . Also, on the event ˆ η 1 k ∈ T N ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Together with the result on I 1 ,k , we have

<!-- formula-not-decoded -->

by n = O ( N ) and ε N = o ( N -1 / 4 ) . Hence, √ NR N = o P (1) .

Step 2. In this step, I present the proof of (A.1). We have the following decomposition:

<!-- formula-not-decoded -->

by (A.2). Thus, Thus, we have

<!-- formula-not-decoded -->

Given κ ≤ g 0 ( X ) ≤ 1 -κ and κ ≤ g ( X ) ≤ 1 -κ ,

<!-- formula-not-decoded -->

By κ ≤ g 0 ( X ) ≤ 1 -κ and κ ≤ g ( X ) ≤ 1 -κ again, we can obtain

<!-- formula-not-decoded -->

Thus, by E P [ U 2 | X ] ≤ C and E P [ V 2 1 | X ] ≤ C ,

<!-- formula-not-decoded -->

Step 3. In this step, I present the proof of (A.2). Define

<!-- formula-not-decoded -->

Then its second-order derivative is

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

Step 4. Notice that

<!-- formula-not-decoded -->

then we have

<!-- formula-not-decoded -->

by Step 2.

Step 5 . Notice that

<!-- formula-not-decoded -->

then we have

<!-- formula-not-decoded -->

where ¯ p ∈ ( p, p 0 ) . Thus,

<!-- formula-not-decoded -->

The term in the second line is bounded by

<!-- formula-not-decoded -->

by ‖ UV 1 ‖ P, 2 ≤‖ UV 1 ‖ P, 4 ≤ C , E P [ U 2 | X ] ≤ C , E P [ V 2 1 | X ] ≤ C , and the conditions on the rates of convergence. Together with Step 2, we obtain

<!-- formula-not-decoded -->

where I assume that ε N converges to zero no faster than N -1 / 2 .

## Repeated cross sections:

In step 1, I show the main result with the following auxiliary results:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where T N is the set of all η 2 = ( g, ℓ 2 ) consisting of P -square-integrable functions g and ℓ 2 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

P N and Λ N are the sets consisting all p &gt; 0 and λ &gt; 0 such that | p -p 0 |≤ N -1 / 2 and | λ -λ 0 |≤ N -1 / 2 , respectively. By the regularity condition (3.2), | ˆ p k -p 0 | = O P ( N -1 / 2 ) , and | ˆ λ k -λ 0 | = O P ( N -1 / 2 ) , we have ˆ η 2 k ∈ T N , ˆ p k ∈ P N , and ˆ λ k ∈ Λ N with probability 1 -o (1) . In Step 2-4, I show the above auxiliary results.

Step 1. Notice that

<!-- formula-not-decoded -->

where the term o P (1) , by the same arguments for the term b in repeated outcomes and the auxiliary results (A.9)-(A.11), contains the second-order terms

<!-- formula-not-decoded -->

where ¯ p k ∈ (ˆ p k , p 0 ) and ¯ λ k ∈ ( ˆ λ k , λ 0 ) . On the other hand, by the same arguments for the term a in repeated outcomes and the auxiliary results (A.7)-(A.8), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, since ˆ p k -p 0 = E n,k [ D -p 0 ] and ˆ λ k -λ 0 = E n,k [ T -λ 0 ] , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Using (A.5)-(A.6) and the same arguments as the step 1 in repeated outcomes, one can show that √ NR ′ N = o P (1) . Hence, it remains to prove the auxiliary results (A.5)-(A.11).

Step 2. Recall that p ′ 0 = p 0 λ 0 (1 -λ 0 ) . For (A.5), notice that

<!-- formula-not-decoded -->

The decomposition becomes

<!-- formula-not-decoded -->

Given that κ ≤ g 0 ( X ) ≤ 1 -κ , κ ≤ g ( X ) ≤ 1 -κ , we have

<!-- formula-not-decoded -->

By κ ≤ g 0 ( X ) ≤ 1 -κ , κ ≤ g ( X ) ≤ 1 -κ again, we obtain

<!-- formula-not-decoded -->

Given E P [ U 2 | X ] ≤ C , E P [ V 2 2 | X ] ≤ C , and the conditions on the rates of convergence,

<!-- formula-not-decoded -->

For (A.6), let f ( r ) = E P [ ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 + r ( η 2 -η 20 ))] . Then the second-order derivative is

<!-- formula-not-decoded -->

It follows that

<!-- formula-not-decoded -->

Step 3. For (A.7), notice that

<!-- formula-not-decoded -->

then we have

<!-- formula-not-decoded -->

by the proof of (A.5).

For (A.8), notice that

<!-- formula-not-decoded -->

Define ∂ λ ψ 20 := ∂ λ ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 ) , then

<!-- formula-not-decoded -->

by (A.5) and E P [ Y 2 | X ] ≤ C .

Step 4. For (A.9), notice that we have

<!-- formula-not-decoded -->

Define ∂ 2 p ψ 20 := ∂ 2 p ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 ) , then we have

<!-- formula-not-decoded -->

where ¯ p ∈ ( p, p 0 ) . Hence, we have

<!-- formula-not-decoded -->

By (A.5), we have ‖ ψ 2 ( W,θ 0 , p 0 , λ 0 , η 2 ) -ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 ) ‖ P, 2 = O ( ε N ) . The term in the second line is bounded by

<!-- formula-not-decoded -->

by ‖ UV 2 ‖ P, 2 ≤‖ UV 2 ‖ P, 4 ≤ C , E P [ U 2 | X ] ≤ C , and E P [ V 2 2 | X ] ≤ C . Thus, we obtain

<!-- formula-not-decoded -->

where I assume that ε N converges to zero no faster than N -1 / 2 .

For (A.10), notice that we have

<!-- formula-not-decoded -->

where c 1 is a constant depending on λ . Define ∂ 2 λ ψ 20 := ∂ 2 λ ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 ) , we have

<!-- formula-not-decoded -->

where ¯ p ∈ ( p, p 0 ) and ¯ λ ∈ ( λ, λ 0 ) . By the triangle inequality, we have

<!-- formula-not-decoded -->

The norm term is the second line is bounded by

<!-- formula-not-decoded -->

by E P [ Y 2 | X ] ≤ C and D ∈ { 0 , 1 } . The two high-order terms are bounded by

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where c 2 and c 3 are constants depending on λ . Using the same arguments in (A.9), one can show that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by ‖ UY ‖ P, 2 ≤ C and E P [ Y 2 | X ] ≤ C .

Finally, we obtain

<!-- formula-not-decoded -->

where I assume that ε N converges to zero no faster than N -1 / 2 .

For (A.11), notice that the derivative is

<!-- formula-not-decoded -->

Also, we have Define ∂ λ ∂ p ψ 20 := ∂ λ ∂ p ψ 2 ( W,θ 0 , p 0 , λ 0 , η 20 ) , then we have

<!-- formula-not-decoded -->

where ¯ p ∈ ( p, p 0 ) . By the triangle inequality, we obtain

<!-- formula-not-decoded -->

Using the same arguments in (A.9) and (A.10), one can show that the high-order term is bounded by

<!-- formula-not-decoded -->

Together with (A.8), we obtain

<!-- formula-not-decoded -->

where I assume that ε N converges to zero no faster than N -1 / 2 .

## Proof of Theorem 2

## Repeated outcomes:

In Step 1, I show the main result using the auxiliary results

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where P N and T N are specified in the proof of Theorem 1, C 1 is a constant, and

<!-- formula-not-decoded -->

In fact, we have E P [ ( ¯ ψ 1 ( W,θ 0 , p 0 , η 10 ) ) 2 ] = Σ 10 . In Step 2, I show the auxiliary results (A.12) and (A.13).

Step 1. Notice that

<!-- formula-not-decoded -->

where the second equality follows from ˆ G 1 p = -˜ θ/ ˆ p k .

Since K is fixed, which is independent of N , it suffices to show that for each k ∈ [ k ] ,

<!-- formula-not-decoded -->

By the triangle inequality, we have

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To bound I 4 ,k , we have

<!-- formula-not-decoded -->

where the last inequality follows from (A.13). By Chebyshev's inequality, I 4 ,k = O P ( n 1 / 2 ) .

Next, we bound I 3 ,k . This part is essentially identical to the proof of Theorem 3.2 in Chernozhukov et al. (2018), I reproduce it here for reader's convenience. Observe that for any number a and δa ,

<!-- formula-not-decoded -->

Denote ψ i = ¯ ψ 1 ( W i , θ 0 , p 0 , η 10 ) and ˆ ψ i = ¯ ψ 1 ( W i , ˜ θ, ˆ p k , ˆ η 1 k ) , and a := ψ i , a + δa := ˆ ψ i . Then

<!-- formula-not-decoded -->

Thus,

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since 1 n ∑ i ∈ I k ‖ ¯ ψ 1 ( W i , θ 0 , p 0 , η 0 ) ‖ 2 = O P (1) , it suffices to bound S N . We have the decomposition

<!-- formula-not-decoded -->

where ¯ θ ∈ ( ˜ θ -θ 0 ) . The first term is bounded by

<!-- formula-not-decoded -->

Also, notice that conditional on ( W i ) i ∈ I c k , both ˆ p k and ˆ η 1 k can be treated as fixed. Under the event that ˆ p k ∈ P N and ˆ η 1 k ∈ T N , we have

<!-- formula-not-decoded -->

by (A.12). It follows that S N = O P ( N -1 +( ε N ) 2 ) . Therefore, we obtain

<!-- formula-not-decoded -->

Step 2. It remains to prove (A.12) and (A.13). By Taylor series expansion,

<!-- formula-not-decoded -->

where ¯ p ∈ ( p, p 0 ) . Then we have

<!-- formula-not-decoded -->

By (A.1), we have ‖ ψ 1 ( W,θ 0 , p 0 , η 1 ) -ψ 1 ( W,θ 0 , p 0 , η 10 ) ‖ P, 2 = O ( ε N ) . The term in the second line is bounded by

<!-- formula-not-decoded -->

where I use ‖ UV 1 ‖ P, 2 ≤‖ UV 1 ‖ P, 4 ≤ C , E P [ U 2 | X ] ≤ C , E P [ V 2 1 | X ] ≤ C , and

<!-- formula-not-decoded -->

by | E P [ UV 1 ] |≤‖ UV 1 ‖ P, 4 ≤ C . Thus, we obtain

<!-- formula-not-decoded -->

where I assume that ε N converges to zero no faster than N -1 / 2 .

For (A.13),

<!-- formula-not-decoded -->

since ‖ UV 1 ‖ P, 4 ≤ C .

## Repeated cross sections:

In Step 1, I show the main result with the auxiliary results:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( P N , Λ N , T N ) are specified in the proof of Theorem 1, C 2 is a constant, and

<!-- formula-not-decoded -->

In fact, we have E P [ ( ¯ ψ 2 ( W,θ 0 , p 0 , λ 0 , G 2 λ 0 , η 20 ) ) 2 ] = Σ 20 . In Step 2, I prove (A.14) and (A.15). Step 1. Notice that

<!-- formula-not-decoded -->

where the second inequality follows from ˆ G 2 p = -˜ θ/ ˆ p k .

Since K is fixed, which is independent of N , it suffices to show that

<!-- formula-not-decoded -->

By the triangle inequality, we have

where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By the same arguments for I 4 ,k in the proof of repeated outcomes and (A.15), we can show J 6 ,k =

o P (1) . Also, by the same arguments for I 3 ,k in the proof of repeated outcomes, we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Since 1 n ∑ i ∈ I k ‖ ¯ ψ 2 ( W,θ 0 , p 0 , λ 0 , G 2 λ 0 , η 20 ) ‖ 2 = O P (1) , it remains to bound S ′ N . Define ¯ ψ 20 := ¯ ψ 2 ( W,θ 0 , p 0 , λ 0 , G 2 λ 0 , η 20 ) . By the triangle inequality, we have

<!-- formula-not-decoded -->

where ¯ θ ∈ ( ˜ θ, θ 0 ) and G 2 λ ∈ ( ̂ G 2 λ , G 2 λ 0 ) . Then we have

<!-- formula-not-decoded -->

The two terms in the last line are bounded by

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Conditional on the auxiliary sample I c k , ( ˆ p k , ˆ λ k , ˆ η 2 k ) can be treated as fixed. Also, under the event that ˆ p k ∈ P N , ˆ λ k ∈ Λ N , and ˆ η 2 k ∈ T N , we have

<!-- formula-not-decoded -->

by (A.14). It follows that S ′ N = O P ( N -1 + ε 2 N ) +( o P (1)) 2 so that

<!-- formula-not-decoded -->

Step 2. It remains to show (A.14) and (A.15). Define ¯ ψ 20 := ¯ ψ 2 ( W,θ 0 , p 0 , λ 0 , G 2 λ 0 , η 20 ) . By the triangle inequality and

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

where ¯ p ∈ ( p, p 0 ) and ¯ λ ∈ ( λ, λ 0 ) . The term in the second line is bounded by

<!-- formula-not-decoded -->

by the same arguments in (A.9)-(A.11) and

<!-- formula-not-decoded -->

since | E P [ UV 2 ] |≤‖ UV 2 ‖ P, 4 ≤ C and | E P [ Y U ] |≤ C . Also, we have

<!-- formula-not-decoded -->

by the same arguments in (A.9)-(A.11) and

<!-- formula-not-decoded -->

since | E P [ UV 2 ] |≤‖ UV 2 ‖ P, 4 ≤ C . Together with (A.5), we have

<!-- formula-not-decoded -->

where I assume that ε N converges to zero no faster than N -1 / 2 .

For (A.15), we have

<!-- formula-not-decoded -->

since ‖ UV 2 ‖ P, 4 ≤ C .

## Proof of Theorem 3:

By the same arguments in the proof of Theorem 1, we can have

<!-- formula-not-decoded -->

for repeated outcomes and

<!-- formula-not-decoded -->

for repeated cross sections. The term ε N is the rate of convergence of the kernel estimators ˆ g kh , ˆ ℓ 1 kh , and ˆ ℓ 2 kh . It remains to show ‖ ˆ g kh -g 0 ‖ P, 2 = o P ( N -1 / 4 ) , ‖ ˆ ℓ 1 kh -ℓ 10 ‖ P, 2 = o P ( N -1 / 4 ) , and ‖ ˆ ℓ 2 kh -ℓ 20 ‖ P, 2 = o P ( N -1 / 4 ) .

Here I use the standard result of kernel estimation in Newey &amp; McFadden (1994). Let ˆ γ kh ( x ) denote the kernel estimator of γ 0 ( x ) = f 0 ( x ) E P [ z | x ] using the auxiliary sample I c k , where z ∈ { 1 , D, Y (1) -Y (0) | D = 0 , ( T -λ 0 ) Y | D = 0 } . By Assumption (3.3) and Lemma 8.10 of Newey &amp; McFadden (1994), we have

<!-- formula-not-decoded -->

by the conditions on h . Let ˆ f kh ( x ) denote ˆ γ k ( x ) with z = 1 and ˆ m kh ( x ) denote ˆ γ k ( x ) with z = D . Then

<!-- formula-not-decoded -->

For the denominator, we have

<!-- formula-not-decoded -->

̸

given inf x ∈X f 0 ( x ) = 0 . For the numerator, let m 0 ( x ) denote γ 0 ( x ) with z = D , we have

<!-- formula-not-decoded -->

̸

given inf x ∈X f 0 ( x ) = 0 . The above two inequalities imply that uniformly over x ∈ X ,

<!-- formula-not-decoded -->

That is, sup x ∈X | ˆ g kh ( x ) -g 0 ( x ) | = o P ( N -1 / 4 ) . Using the same arguments, one can also show that sup x ∈X | ˆ ℓ 1 kh ( x ) -ℓ 0 ( x ) | = o P ( N -1 / 4 ) and sup x ∈X | ˆ ℓ 2 kh ( x ) -ℓ 0 ( x ) | = o P ( N -1 / 4 ) . Since uniform convergence implies L 2 -norm convergence, we complete the proof.

## Proof of Theorem 4:

The proof is the same as the proof in Theorem 2 provided the assumptions in Theorem 3 hold.

## Lemma A.1 (CONDITIONAL CONVERGENCE IMPLIES UNCONDITIONAL)

Let { X m } and { Y m } be sequences of random vectors. (i) If for ϵ m → 0 , Pr ( ‖ X m ‖ &gt; ϵ m | Y m ) p → 0 , then Pr ( ‖ X m ‖ &gt; ϵ m ) → 0 . This occurs if E [ ‖ X m ‖ q /ϵ q m | Y m ] p → 0 for some q ≥ 1 , by Markov's inequality. (ii) Let { A m } be a sequence of positive constants. If ‖ X m ‖ = O P ( A m ) conditional on Y m , namely, that for any ℓ m → ∞ , Pr ( ‖ X m ‖ &gt; ℓ m A m | Y m ) p → 0 , then ‖ X m ‖ = O P ( A m ) unconditionally, namely, that for any ℓ m →∞ , Pr ( ‖ X m ‖ &gt; ℓ m A m ) → 0 .

PROOF: This lemma is the Lemma 6.1 in Chernozhukov et al. (2018).

## SIMULATION

<!-- image -->

Figure 3: Repeated outcomes: N = 200 and p = 300 .

Figure 4: Repeated outcomes: N = 200 and p = 100 .

<!-- image -->

Figure 5: Repeated outcomes: N = 500 and p = 300 .

<!-- image -->

<!-- image -->

Figure 6: Repeated outcomes: N = 500 and p = 100 .

Figure 7: Repeated outcomes: N = 200

<!-- image -->

Figure 8: Repeated outcomes: N = 500

<!-- image -->

<!-- image -->

Figure 9: Repeated cross sections: N = 200 and p = 300 .

Figure 10: Repeated cross sections: N = 200 and p = 100 .

<!-- image -->

Figure 11: Repeated cross sections: N = 500 and p = 300 .

<!-- image -->

<!-- image -->

Figure 12: Repeated cross sections: N = 500 and p = 100 .

Figure 13: Repeated cross sections: N = 200

<!-- image -->

Figure 14: Repeated cross sections: N = 500

<!-- image -->

<!-- image -->

Figure 15: Multilevel treatment: N = 200 and p = 300 .

Figure 16: Multilevel treatment: N = 200 and p = 100 .

<!-- image -->

Figure 17: Multilevel treatment: N = 500 and p = 300 .

<!-- image -->

<!-- image -->

Figure 18: Multilevel treatment: N = 500 and p = 100 .

Figure 19: Multilevel treatment: N = 200

<!-- image -->

Figure 20: Multilevel treatment: N = 500

<!-- image -->