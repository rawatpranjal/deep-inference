## A Practitioner's Guide to Cluster-Robust Inference

## A. Colin Cameron and Douglas L. Miller

## Abstract

We consider  statistical  inference  for  regression  when  data  are  grouped  into  clusters,  with regression model errors independent across clusters but correlated within clusters. Examples include  data  on  individuals  with  clustering  on  village  or  region  or  other  category  such  as industry,  and  state-year  differences-in-differences  studies  with  clustering  on  state.  In  such settings default standard errors can greatly overstate estimator precision. Instead, if the number of clusters is large, statistical inference after OLS should be based on cluster-robust standard errors. We outline the basic method as well as many complications that can arise in practice. These include cluster-specific fixed effects, few clusters, multi-way clustering, and estimators other than OLS.

Colin Cameron is a Professor in the Department of Economics at UC- Davis. Doug Miller is an  Associate  Professor  in  the  Department  of  Economics  at  UC-  Davis.  They  thank  four referees and the journal editor for very helpful comments and for guidance, participants at the 2013 California Econometrics Conference, a workshop sponsored by the U.K. Programme Evaluation  for  Policy  Analysis,  seminars  at  University  of  Southern  California  and  at University of Uppsala, and the many people who over time have sent them cluster-related puzzles  (the  solutions  to  some  of  which  appear  in  this  paper).  Doug  Miller  acknowledges financial support from the Center for Health and Wellbeing at the Woodrow Wilson School of Public Policy at Princeton University.

## I. Introduction

In  an  empiricist's  day-to-day  practice,  most  effort  is  spent  on  getting  unbiased  or consistent point estimates. That is, a lot of attention focuses on the parameters ( 𝛽 ̂ ) .  In  this paper we focus on getting accurate statistical inference, a fundamental component of which is obtaining accurate standard errors ( 𝑠𝑒 , the estimated standard deviation of 𝛽 ̂ ) . We begin with the basic reminder that empirical researchers should also really care about getting this part right.  An  asymptotic  95%  confidence  interval  is 𝛽 ̂ ±1.96 × 𝑠𝑒 ,  and  hypothesis  testing  is typically  based  on  the  Wald  't-statistic' 𝑤 = ( 𝛽 ̂ -𝛽0 )/ 𝑠𝑒 .  Both 𝛽 ̂ and 𝑠𝑒 are  critical ingredients for statistical inference, and we should be paying as much attention to getting a good 𝑠𝑒 as we do to obtain 𝛽 ̂ .

In this paper, we consider statistical inference in regression models where observations can  be  grouped  into  clusters,  with  model  errors  uncorrelated  across  clusters  but  correlated within cluster. One leading example of 'clustered errors' is individual-level cross-section data with  clustering  on  geographical  region,  such  as  village  or  state.  Then  model  errors  for individuals  in  the  same  region  may  be  correlated,  while  model  errors  for  individuals  in different regions are assumed to be uncorrelated. A second leading example is panel data. Then model errors in different time periods for a given individual (e.g., person or firm or region) may be correlated, while model errors for different individuals are assumed to be uncorrelated.

Failure  to  control  for  within-cluster  error  correlation  can  lead  to  very  misleadingly small  standard  errors,  and  consequent  misleadingly  narrow  confidence  intervals,  large t-statistics and low p-values. It is not unusual to have applications where standard errors that control for within-cluster correlation are several times larger than default standard errors that ignore such correlation. As shown below, the need for such control increases not only with increase in the size of within-cluster error correlation, but the need also increases with the size of within-cluster correlation of regressors and with the number of observations within a cluster. A leading example, highlighted by Moulton (1986, 1990), is when interest lies in measuring the effect  of  a  policy  variable,  or  other  aggregated  regressor,  that  takes  the  same  value  for  all observations within a cluster.

One way to control for clustered errors in a linear regression model is to additionally specify a model for the within-cluster error correlation, consistently estimate the parameters of this error correlation model, and then estimate the original model by feasible generalized least squares (FGLS) rather than ordinary least squares (OLS). Examples include random effects estimators and, more generally, random coefficient and hierarchical models. If all goes well this provides valid statistical inference, as well as estimates of the parameters of the original regression model that are more efficient than OLS. However, these desirable properties hold only under the very strong assumption that the model for within-cluster error correlation is correctly specified.

A more recent method to control for clustered errors is to estimate the regression model with limited or no control for within-cluster error correlation, and then post-estimation obtain 'cluster-robust'  standard  errors  proposed  by  White  (1984,  p.134-142)  for  OLS  with  a multivariate dependent variable (directly applicable to balanced clusters); by Liang and Zeger (1986) for linear and nonlinear models; and by Arellano (1987) for the fixed effects estimator in linear panel models. These cluster-robust standard errors do not require specification of a model for within-cluster error correlation, but do require the additional assumption that the number of clusters, rather than just the number of observations, goes to infinity.

Cluster-robust  standard  errors  are  now  widely  used,  popularized  in  part  by  Rogers (1993) who incorporated the method in Stata, and by Bertrand, Duflo and Mullainathan (2004)

who pointed out that many differences-in-differences studies failed to control for clustered errors, and those that did often clustered at the wrong level. Cameron and Miller (2011) and Wooldridge (2003, 2006) provide surveys, and lengthy expositions are given in Angrist and Pischke (2009) and Wooldridge (2010).

One goal of this paper is to provide the practitioner with the methods to implement cluster-robust  inference.  To  this  end  we  include  in  the  paper  reference  to  relevant  Stata commands (for version 13), since Stata is the computer package most often used in applied microeconometrics research. And we will post on our websites more expansive Stata code and the datasets used in this paper. A second goal is presenting how to deal with complications such as determining when there is a need to cluster, incorporating fixed effects, and inference when there are few clusters. A third goal is to provide an exposition of the underlying econometric theory  as  this  can  aid  in  understanding  complications.  In  practice  the  most  difficult complication to deal with can be 'few' clusters, see Section VI. There is no clear-cut definition of 'few'; depending on the situation 'few' may range from less than 20 to less than 50 clusters in the balanced case.

We  focus  on  OLS,  for  simplicity  and  because  this  is  the  most  commonly-used estimation method in practice. Section  II  presents  the  basic  results  for  OLS  with  clustered errors.  In  principle,  implementation  is  straightforward  as  econometrics  packages  include cluster-robust as an option for the commonly-used estimators; in Stata it is the vce(cluster) option.  The  remainder  of  the  survey  concentrates  on  complications  that often  arise  in  practice.  Section  III  addresses  how  the  addition  of  fixed  effects  impacts cluster-robust inference. Section IV deals with the obvious complication that it is not always clear what to cluster over. Section V considers clustering when there is more than one way to do so and these ways are not nested in each other. Section VI considers how to adjust inference when  there  are  just  a  few  clusters  as,  without  adjustment,  test  statistics  based  on  the cluster-robust standard errors over-reject and confidence intervals are too narrow. Section VII presents extension to the full range of estimators - instrumental variables, nonlinear models such  as  logit  and  probit,  and  generalized  method  of  moments.  Section  VIII  presents  both empirical examples and real-data based simulations. Concluding thoughts are given in Section IX.

## II. Cluster-Robust Inference

In this section we present the fundamentals of cluster-robust inference. For these basic results we assume that the model does not include cluster-specific fixed effects, that it is clear how  to  form  the  clusters,  and  that  there  are  many  clusters.  We  relax  these  conditions  in subsequent sections.

Clustered errors have two main consequences: they (usually) reduce the precision of 𝛽 ̂ , and the standard estimator for the variance of 𝛽 ̂ , V � [ 𝛽 ̂ ] , is (usually) biased downward from the true variance. Computing cluster-robust standard errors is a fix for the latter issue. We illustrate these issues, initially in the context of a very simple model and then in the following subsection in a more typical model.

## A. A Simple Example

For simplicity, we begin with OLS with a single regressor that is nonstochastic, and assume no intercept in the model. The results extend to multiple regression with stochastic regressors.

Let 𝑦𝑖 = 𝛽𝑥𝑖 + 𝑢𝑖 , 𝑖 = 1, . . . , 𝑁 ,  where 𝑥𝑖 is  nonstochastic and E [ 𝑢𝑖 ] = 0. The OLS estimator 𝛽 ̂ = ∑ 𝑥𝑖 𝑖 𝑦𝑖 / ∑ 𝑥𝑖 2 𝑖 can be re-expressed as 𝛽 ̂ -𝛽 = ∑ 𝑥𝑖𝑢𝑖 𝑖 / ∑ 𝑥𝑖 2 𝑖 , so in general

<!-- formula-not-decoded -->

If  errors  are  uncorrelated  over 𝑖 ,  then  V [ ∑ 𝑥𝑖𝑢𝑖 𝑖 ] = ∑ V[ 𝑥𝑖𝑢𝑖 ] 𝑖 = ∑ 𝑥𝑖 2 𝑖 V [ 𝑢𝑖 ] .  In  the simplest case of homoskedastic errors, V [ 𝑢𝑖 ] = 𝜎 2 and (1) simplifies to V [ 𝛽 ̂ ] = 𝜎 2 / ∑ 𝑥𝑖 2 𝑖 .

If instead errors are heteroskedastic, then (1) becomes

<!-- formula-not-decoded -->

using V[ 𝑢𝑖 ] = E[ 𝑢𝑖 2 ] since E[ 𝑢𝑖 ] = 0. Implementation seemingly requires consistent estimates of each of the 𝑁 error variances E [ 𝑢𝑖 2 ] . In a very influential paper, one that extends naturally  to  the  clustered  setting,  White  (1980)  noted  that  instead  all  that  is  needed  is  an estimate of the scalar ∑ 𝑥𝑖 2 E[ 𝑢𝑖 2 ] 𝑖 , and that one can simply use ∑ 𝑥𝑖 2 𝑢 � 𝑖 2 𝑖 , where 𝑢 � 𝑖 = 𝑦𝑖 -𝛽 ̂ 𝑥 𝑖 is the OLS residual, provided 𝑁 → ∞ . This leads to estimated variance

<!-- formula-not-decoded -->

The resulting standard error for 𝛽 ̂ is often called a robust standard error, though a better, more precise term, is heteroskedastic-robust standard error.

What if  errors  are  correlated  over 𝑖 ?  In  the  most  general  case  where  all  errors  are correlated with each other,

<!-- formula-not-decoded -->

so

<!-- formula-not-decoded -->

The obvious extension of White (1980) is to use V � [ 𝛽 ̂ ] = �∑ 𝑖 ∑ 𝑥𝑖𝑥𝑗𝑢 � 𝑖 𝑢 � 𝑗 𝑗 ] � /( ∑ 𝑥𝑖 2 𝑖 ) 2 , but this equals  zero  since ∑ 𝑥𝑖𝑢 � 𝑖 𝑖 = 0 .  Instead  one  needs  to  first  set  a  large  fraction  of  the  error correlations E [ 𝑢𝑖𝑢𝑗 ] to zero. For time series data with errors assumed to be correlated only up to, say, 𝑚 periods apart as well as heteroskedastic, White's result can be extended to yield a heteroskedastic- and autocorrelation-consistent (HAC) variance estimate; see Newey and West (1987).

In this paper we consider clustered errors, with E [ 𝑢𝑖𝑢𝑗 ] = 0 unless observations 𝑖 and 𝑗 are in the same cluster (such as same region). Then

<!-- formula-not-decoded -->

where the indicator function 𝟏 [ 𝐴 ] equals 1 if event 𝐴 happens and equals 0 if event 𝐴 does not happen. Provided the number of clusters goes to infinity, we can use the variance estimate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This estimate is called a cluster-robust estimate, though more precisely it is heteroskedasticand cluster-robust. This estimate reduces to V � het [ 𝛽 ̂ ] in the special case that there is only one observation in each cluster.

Typically V � clu [ 𝛽 ̂ ] exceeds V � het [ 𝛽 ̂ ] due  to  the  addition  of  terms  when 𝑖 ≠ 𝑗 .  The amount  of  increase  is  larger  (1)  the  more  positively  associated  are  the  regressors  across observations in the same cluster (via 𝑥𝑖 𝑥𝑗 in (3)), (2) the more correlated are the errors (via E [ 𝑢𝑖𝑢𝑗 ] in  (2)),  and  (3)  the  more  observations  are  in  the  same  cluster  (via 𝟏 [ 𝑖 , 𝑗 in  same cluster ] in (3)).

There are several take-away messages. First there can be great loss of efficiency in OLS estimation  if errors are  correlated  within  cluster  rather  than  completely  uncorrelated. Intuitively, if errors are positively correlated within cluster then an additional observation in the cluster no longer provides a completely independent piece of new information. Second, failure to control for this within-cluster error correlation can lead to using standard errors that are  too  small,  with  consequent  overly-narrow  confidence  intervals,  overly-large  t-statistics, and over-rejection of true null hypotheses. Third, it is straightforward to obtain cluster-robust standard errors, though they do rely on the assumption that the number of clusters goes to infinity (see Section VI for the few clusters case).

## B. Clustered Errors and Two Leading Examples

Let 𝑖 denote  the 𝑖 𝑡ℎ of 𝑁 individuals  in  the  sample,  and 𝑔 denote  the 𝑔 𝑡ℎ of 𝐺 clusters. Then for individual 𝑖 in cluster 𝑔 the linear model with (one-way) clustering is

<!-- formula-not-decoded -->

where 𝒙𝑖𝑔 is a 𝐾 ×1 vector. As usual it is assumed that E �𝑢𝑖𝑔 | 𝒙𝑖𝑔� = 0 . The key assumption is that errors are uncorrelated across clusters, while errors for individuals belonging to the same cluster may be correlated. Thus

<!-- formula-not-decoded -->

## 1. Example 1: Individuals in Cluster

Hersch (1998) uses cross-section individual-level data to estimate the impact of job injury  risk  on  wages.  Since  there  is  no  individual-level  data  on  job  injury  rate,  a  more aggregated measure such as job injury risk in the individual's industry is used as a regressor. Then for individual 𝑖 (with 𝑁 = 5960) in industry 𝑔 (with 𝐺 = 211)

<!-- formula-not-decoded -->

The  regressor 𝑥𝑔 is  perfectly  correlated  within  industry.  The  error  term  will  be positively correlated within industry if the model systematically overpredicts (or underpredicts)  wages  in  a  given  industry.  In  this  case  default  OLS  standard  errors  will  be

downwards biased.

To measure the extent of this downwards bias, suppose errors are equicorrelated within cluster, so Cor [ 𝑢𝑖𝑔 , 𝑢𝑗𝑔 ] = 𝜌 for all 𝑖 ≠ 𝑗 . This pattern is suitable when observations can be viewed as exchangeable, with ordering not mattering. Common examples include the current one,  individuals  or  households  within  a  village  or  other  geographic  unit  (such  as  state), individuals within a household, and students within a school. Then a useful approximation is that for the 𝑘 𝑡ℎ regressor the default OLS variance estimate based on 𝑠 2 ( 𝑿 ′ 𝑿 ) -1 , where 𝑠 is the standard error of the regression, should be inflated by

<!-- formula-not-decoded -->

where 𝜌𝑥𝑘 is a measure of the within-cluster correlation of 𝑥𝑖𝑔𝑘 , 𝜌𝑢 is the within-cluster error correlation, and 𝑁 ̄ 𝑔 is the average cluster size. The result (6) is exact if clusters are of equal size ('balanced' clusters) and 𝜌𝑥𝑘 = 1 for  all  regressors  (Kloek, 1981); see Scott and Holt (1982) and Greenwald (1983) for the general result with a single regressor.

This  very  important  and  insightful  result  is  that  the  variance  inflation  factor  is increasing in

1. the within-cluster correlation of the regressor
2. the within-cluster correlation of the error
3. the number of observations in each cluster.

For clusters of unequal size replace ( 𝑁 ̄ 𝑔 1) in (6) by (( V [ 𝑁𝑔 ]/ 𝑁 ̄ 𝑔 ) + 𝑁 ̄ 𝑔 1) ; see Moulton (1986, p.387). Note that there is no cluster problem if any one of the following occur: 𝜌𝑥𝑘 = 0 or 𝜌𝑢 = 0 or 𝑁 ̄ 𝑔 = 1 .

In an influential paper, Moulton (1990) pointed out that in many settings the inflation factor 𝜏𝑘 can  be  large  even  if 𝜌𝑢 is  small.  He  considered  a  log  earnings  regression  using March CPS data ( 𝑁 = 18,946) , regressors aggregated at the state level ( 𝐺 = 49) , and errors correlated  within  state ( 𝜌 �𝑢 = 0.032) .  The  average  group  size  was 18,946/49 = 387 , 𝜌𝑥𝑘 = 1 for  a  state-level  regressor,  so  (6)  yields 𝜏𝑘 �≃ 1 + 1 × 0.032 × 386 = 13.3 .  The weak correlation of errors within state was still enough to lead to cluster-corrected standard errors being √ 13.3 = 3.7 times larger than the (incorrect) default standard errors!

In such examples of cross-section data with an aggregated regressor, the cluster-robust standard errors can be much larger despite low within-cluster error correlation because the regressor of interest is perfectly correlated within cluster and there may be many observations per cluster.

## 2. Example 2: Differences-in-Differences (DiD) in a State-Year Panel

Interest may lie in how wages respond to a binary policy variable 𝑑𝑡𝑠 that varies by state and over time. Then at time 𝑡 in state 𝑠

<!-- formula-not-decoded -->

where we assume independence over states, so the ordering of subscripts ( 𝑡 , 𝑠 ) corresponds to ( 𝑖 , 𝑔 ) in (4), and 𝛼𝑠 and 𝛿𝑡 are state and year effects.

The binary regressor 𝑑𝑡𝑠 equals one if the policy of interest is in effect and equals 0 otherwise. The regressor 𝑑𝑡𝑠 is  often highly serially correlated since, for example, 𝑑𝑡𝑠 will equal a string of zeroes followed by a string of ones for a state that switches from never having the policy in place to forever after having the policy in place. The error 𝑢𝑡𝑠 is correlated over time for a given state if the model systematically overpredicts (or underpredicts) wages in a

given state. Again the default standard errors are likely to be downwards-biased.

In  the  panel  data  case,  the  within-cluster  (i.e.,  within-individual)  error  correlation decreases as the time separation increases, so errors are not equicorrelated. A better model for the errors is a time series model, such as an autoregressive error of order one that implies that Cor [ 𝑢𝑡𝑠 , 𝑢𝑡 ′ 𝑠 ] = 𝜌 | 𝑡-𝑡 ′ | . The true variance of the OLS estimator will again be larger than the OLS default, although the consequences of clustering  are  less  extreme  than  in  the  case  of equicorrelated errors (see Cameron and Miller (2011, Section 2.3) for more detail).

In such DiD examples with panel data, the cluster-robust standard errors can be much larger than the default because both the regressor of interest and the errors are highly correlated within cluster. Note also that this complication can exist even with the inclusion of fixed effects (see Section III).

The same problems arise if we additionally have data on individuals, so that

<!-- formula-not-decoded -->

In an influential paper, Bertrand, Duflo and Mullainathan (2004) demonstrated the importance of using cluster-robust standard errors in DiD settings. Furthermore, the clustering should be on state, assuming error independence across states. The clustering should not be on state-year pairs since, for example, the error for California in 2010 is likely to be correlated with the error for California in 2009.

The issues raised here are relevant for any panel data application, not just DiD studies. The DiD panel example with binary policy regressor is often emphasized in the cluster-robust literature because it is widely used and it has a regressor that is highly serially correlated, even after mean-differencing to control for fixed effects. This serial correlation leads to a potentially large difference between cluster-robust and default standard errors.

## C. The Cluster-Robust Variance Matrix Estimate

Stacking all observations in the 𝑔 𝑡ℎ cluster, the model (4) can be written as

<!-- formula-not-decoded -->

where 𝒚𝑔 and 𝒖𝑔 are 𝑁𝑔 ×1 vectors, 𝑿𝑔 is an 𝑁𝑔 × 𝐾 matrix, and  there are 𝑁𝑔 observations  in  cluster 𝑔 .  Further  stacking 𝒚𝑔 , 𝑿𝑔 and 𝒖𝑔 over  the 𝐺 clusters  then  yields the model

<!-- formula-not-decoded -->

With

The OLS estimator is

<!-- formula-not-decoded -->

In general, the variance matrix conditional on 𝑿 is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Given error independence across clusters, V [ 𝒖 | 𝑿 ] has a block-diagonal structure, and

(8) simplifies to where

<!-- formula-not-decoded -->

The matrix 𝑩clu , the middle part of the 'sandwich matrix' (7), corresponds to the numerator of (2). 𝑩clu can be written as:

<!-- formula-not-decoded -->

where 𝜔𝑖𝑔 , 𝑗𝑔 = E [ 𝑢𝑖𝑔𝑢𝑗𝑔 | 𝑿𝑔 ] is  the  error  covariance  for  the 𝑖𝑔 𝑡ℎ and 𝑗𝑔 𝑡ℎ observations. We can gain a few insights from inspecting this equation. The term 𝑩 (and hence V [ 𝜷 � ] ) will be  bigger  when:  (1)  regressors  within  cluster  are  correlated,  (2)  errors  within  cluster  are correlated so 𝜔𝑖𝑔 , 𝑗𝑔 is non-zero, (3) 𝑁𝑔 is large, and (4) the within-cluster regressor and error correlations are of the same sign (the usual situation). These conditions mirror the more precise Moulton  result  for  the  special  case  of  equicorrelated  errors  given  in  equation  (6).  Both examples in Section II had high within-cluster correlation of the regressor, the DiD example additionally had high within-cluster (serial) correlation of the error and the Moulton (1990) example additionally had 𝑁𝑔 large.

Implementation requires an estimate of 𝑩clu given in (9). The cluster-robust estimate of the variance matrix (CRVE) of the OLS estimator is the sandwich estimate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and 𝒖 �𝑔 = 𝒚𝑔 - 𝑿𝑔𝜷 � is  the  vector  of  OLS residuals for the 𝑔 𝑡ℎ cluster.  Formally (10)-(11) provides a consistent estimate of the variance matrix if 𝐺 -1 ∑ 𝑿𝑔 ′ 𝒖 �𝑔𝒖 �𝑔 ′ 𝑿𝑔 𝐺 𝑔=1 -𝐺 -1 ∑ E �𝑿𝑔 ′ 𝒖𝑔𝒖𝑔 ′ 𝑿𝑔� 𝐺 𝑔=1 ⟶ 𝑝 𝟎 as 𝐺 → ∞ . Initial derivations of this estimator, by White (1984, p.134-142) for balanced clusters, and by Liang and Zeger (1986) for unbalanced, assumed a finite number of observations per cluster. Hansen (2007a) showed that the CRVE can also be used  if 𝑁𝑔 → ∞ ,  the  case  for  long  panels,  in  addition  to 𝐺 → ∞ .  Carter,  Schnepel  and Steigerwald  (2013)  consider  unbalanced  panels  with  either 𝑁𝑔 fixed  or 𝑁𝑔 → ∞ .  The sandwich formula for the CRVE extends to many estimators other than OLS; see Section VII.

Algebraically, the estimator (10)-(11) equals (7) and (9) with E [ 𝒖𝑔𝒖𝑔 ′ ] replaced with 𝒖 �𝑔𝒖 �𝑔 ′ .  What  is  striking  about  this  is  that  for  each  cluster 𝑔 ,  the 𝑁𝑔 × 𝑁𝑔 matrix 𝒖 �𝑔𝒖 �𝑔 ′ is bound to be a very poor estimate of the 𝑁𝑔 × 𝑁𝑔 matrix  E [ 𝒖𝑔𝒖𝑔 ′ ] -  there  is  no  averaging going on to enable use of a Law of Large Numbers. The 'magic' of the CRVE is that despite this,  by  averaging  across  all 𝐺 clusters  in  (11),  we  are  able  to  get  a  consistent  variance estimate. This fact helps us to understand one of the limitations of this method in practice - the averaging that makes V � [ 𝜷 � ] accurate for V [ 𝜷 � ] is an average based on the number of clusters 𝐺 .    In applications with few clusters this can lead to problems that we discuss below in Section VI.

Finite-sample modifications of (11) are typically used, to reduce downwards bias in

V � clu [ 𝜷 � ] due to finite 𝐺 . Stata uses √𝑐𝒖 �𝑔 in (11) rather than 𝒖 �𝑔 , with

<!-- formula-not-decoded -->

In general 𝑐 ≃ 𝐺 /( 𝐺 1) , though see Subsection III.B for an important exception when fixed effects are directly estimated. Some other packages such as SAS use 𝑐 = 𝐺 /( 𝐺 1) , a simpler correction that is also used by Stata for extensions to nonlinear models. Either choice of 𝑐 usually lessens, but does not fully eliminate, the usual downwards bias in the CRVE. Other finite-cluster corrections are discussed in Section VI, but there is no clear best correction.

## D. Feasible GLS

If errors are correlated within cluster, then in general OLS is inefficient and feasible GLS may be more efficient.

Suppose  we  specify  a  model  for Ω𝑔 = E [ 𝒖𝑔𝒖𝑔 ′ | 𝑿𝑔 ] in  (9),  such  as  within-cluster equicorrelation.  Then  the  GLS  estimator  is ( 𝑿 ′ Ω -1 𝑿 ) -1 𝑿 ′ Ω -1 𝒚 ,  where Ω = Diag [ Ω𝑔 ] . Given a consistent estimate Ω � of Ω , the feasible GLS estimator of 𝜷 is

<!-- formula-not-decoded -->

The FGLS estimator is second-moment efficient, with variance matrix

<!-- formula-not-decoded -->

under the strong assumption that the error variance Ω is correctly specified.

Remarkably,  the  cluster-robust  method  of  the  previous  section  can  be  extended  to FGLS. Essentially OLS is the special case where Ω𝑔 = 𝜎 2 𝑰𝑁𝑔 . The cluster-robust estimate of the asymptotic variance matrix of the FGLS estimator is

<!-- formula-not-decoded -->

where 𝒖 �𝑔 = 𝒚𝑔 - 𝑿𝑔𝜷 � FGLS .  This  estimator  requires  that 𝒖𝑔 and 𝒖ℎ are  uncorrelated  when 𝑔 ≠ ℎ , and that 𝐺 → ∞ , but permits E [ 𝒖𝑔𝒖𝑔 ′ | 𝑿𝑔 ] ≠ Ω𝑔 . The approach of specifying a model for the error variances and then doing inference that guards against misspecification of this model is especially popular in the biostatistics literature that calls Ω𝑔 a  'working' variance matrix (see, for example, Liang and Zeger, 1986).

There are many possible candidate models for Ω𝑔 , depending on the type of data being analyzed.

For individual-level data clustered by region, example 1 in Subsection II.B, a common starting point model is the random effects (RE) model. The error in model (4) is specified to have two components:

<!-- formula-not-decoded -->

where 𝛼𝑔 is a cluster-specific error or common shock that is assumed to be independent and identically distributed (i.i.d.) (0, 𝜎𝛼 2 ) , and 𝜀𝑖𝑔 is an idiosyncratic error that is assumed to be i.i.d. (0, 𝜎𝜀 2 ) .  Then V [ 𝑢𝑖𝑔 ] = 𝜎𝛼 2 + 𝜎𝜀 2 and Cov [ 𝑢𝑖𝑔 , 𝑢𝑗𝑔 ] = 𝜎𝛼 2 for 𝑖 ≠ 𝑗 .  It  follows  that  the intraclass correlation of the error 𝜌𝑢 = Cor [ 𝑢𝑖𝑔 , 𝑢𝑗𝑔 ] = 𝜎𝛼 2 /( 𝜎𝛼 2 + 𝜎𝜀 2 ) , so this model implies equicorrelated errors within cluster. Richer models that introduce heteroskedasticity include random coefficients models and hierarchical linear models.

For panel data, example 2 in Subsection II.B, a range of time series models for 𝑢𝑖𝑡 may be used, including autoregressive and moving average error models. Analysis of within-cluster residual correlation patterns after OLS estimation can be helpful in selecting a model for Ω𝑔 .

Note that in all cases if cluster-specific fixed effects are included as regressors and 𝑁g is small then bias-corrected FGLS should be used; see Subsection III.C.

The  efficiency  gains  of  FGLS  need  not  be  great.  As  an  extreme  example,  with equicorrelated errors, balanced clusters, and all regressors invariant within cluster ( 𝒙𝑖𝑔 = 𝒙𝑔 ) the FGLS estimator equals the OLS estimator - and so there is no efficiency gain to FGLS. With equicorrelated errors and general 𝑿 , Scott and Holt (1982) provide an upper bound to the maximum  proportionate  efficiency  loss  of  OLS,  compared  to  the  variance  of  the  FGLS estimator, of 1/ � 1 + 4 ( 1-𝜌𝑢 )[ 1+ ( 𝑁𝑚𝑎𝑥 -1 ) 𝜌𝑢 ( 𝑁𝑚𝑎𝑥 × 𝜌𝑢 ) 2 � , 𝑁max = max { 𝑁1 , . . . , 𝑁𝐺 } . This upper bound is increasing in the error correlation 𝜌𝑢 and the maximum cluster size 𝑁max . For low 𝜌𝑢 the maximal efficiency gain can be low. For example, Scott and Holt (1982) note that for 𝜌𝑢 = .05 and 𝑁max = 20 there  is  at  most  a  12%  efficiency  loss  of  OLS  compared  to  FGLS.  With 𝜌𝑢 = 0.2 and 𝑁max = 100 the efficiency loss could be as much as 86%, though this depends on the nature of 𝑿 .

There  is  no  clear  guide  to  when  FGLS  may  lead  to  considerable  improvement  in efficiency, and the efficiency gains can be modest. However, especially in models without cluster-specific  fixed  effects,  implementation  of  FGLS  and  use  of  (15)  to  guard  against misspecification of Ω𝑔 is straightforward. And even modest efficiency gains can be beneficial. It is remarkable that current econometric practice with clustered errors ignores the potential efficiency gains of FGLS.

## E. Implementation for OLS and FGLS

For regression software that provides a cluster-robust option, implementation of the CRVE for OLS simply requires defining for each observation a cluster identifier variable that takes one of 𝐺 distinct  values  according  to  the  observation's  cluster,  and  then  passing  this cluster identifier to the estimation command's cluster-robust option. For example, if the cluster identifier is id\_clu , then Stata OLS command regress y x becomes regress y x, vce(cluster id\_clu ).

Wald hypothesis tests and confidence intervals are then implemented in the usual way. In some cases, however, joint tests of several hypotheses and of overall statistical significance may not be possible. The CRVE V � clu [ 𝜷 � ] is  guaranteed  to  be  positive  semi-definite,  so  the estimated  variance  of  the  individual  components  of 𝜷 � are  necessarily  nonnegative.  But V � clu [ 𝜷 � ] is not necessarily positive definite, so it is possible that the variance matrix of linear combinations of the components of 𝜷 � is singular. The rank of V � clu [ 𝜷 � ] equals the rank of 𝑩 � defined in (11). Since 𝑩 � = 𝑪 ′ 𝑪 ,  where 𝑪 ′ = [ 𝑿1 ′ 𝒖 �1⋯𝑿𝐺 ′ 𝒖 �𝐺 ] is  a 𝐾 × 𝐺 matrix,  it  follows that the rank of 𝑩 � ,  and hence that of V � clu [ 𝜷 � ] ,  is  at  most the rank of 𝑪 .  Since 𝑿1 ′ 𝒖 �1 + ⋯ + 𝑿𝐺 ′ 𝒖 �𝐺 = 𝟎 , the rank of 𝑪 is at most the minimum of 𝐾 and 𝐺 1 . Effectively, the rank of V � clu [ 𝜷 � ] equals min ( 𝐾 , 𝐺 1) ,  though  it  can  be  less  than  this  in  some  examples  such  as perfect collinearity of regressors and cluster-specific dummy regressors (see Subsection III.B

for the latter).

A common setting is to have a richly specified model with thousands of observations in far fewer clusters, leading to more regressors than clusters. Then V � clu [ 𝜷 � ] is rank-deficient, so it  will  not be possible to perform an overall F test of the joint statistical significance of all regressors. And in a log-wage regression with occupation dummies and clustering on state, we cannot  test  the  joint  statistical  significance  of  the  occupation  dummies  if  there  are  more occupations  than  states.  But  it  is  still  okay  to  perform  statistical  inference  on  individual regression coefficients, and to do joint tests on a limited number of restrictions (potentially as many as min ( 𝐾 , 𝐺 1) ).

Regression software usually also includes a panel data component. Panel commands may enable not only OLS with cluster-robust standard errors, but also FGLS for some models of within-cluster error correlation with default (and possibly cluster-robust) standard errors. It is  important  to  note  that  those  panel  data  commands  that  do  not  explicitly  use  time  series methods, an example is FGLS with equicorrelation of errors within-cluster, can be applied more generally to other forms of clustered data, such as individual-level data with clustering on geographic region.

For example, in Stata first give the command xtset id\_clu to let Stata know that the  cluster-identifier  is  variable id\_clu .  Then  the  Stata  command xtreg y x, pa corr(ind)  vce(robust) yields OLS estimates with cluster-robust standard errors. Note that  for  Stata xt commands,  option vce(robust) is  generally  interpreted  as  meaning cluster-robust; this is always the case from version 12.1 on. The xt commands use standard normal critical values whereas command regress uses Student's 𝑇 ( 𝐺 1) critical values; see Sections VI and VIIA for further discussion.

For FGLS estimation the commands vary with the model for Ω𝑔 .  For equicorrelated errors, a starting point for example 1 in Subsection II.B, the FGLS estimator can be obtained using command xtreg y x, pa corr(exch) or command xtreg y x, re ; slightly different estimates are obtained due to slightly different estimates of the equicorrelation. For FGLS estimation of hierarchical models that are richer than a random effects model, use Stata command mixed (version 13) or xtmixed (earlier versions). For FGLS with panel data and time variable time , first give the command xtset id\_clu  time to let Stata know both the cluster-identifier and time variable. A starting point for example 2 in Subsection II.B is an autoregressive error of order one, estimated using command xtreg  y  x,  pa  corr(ar  1) . Stata permits a wide range of possible models for serially correlated errors.

In all of these FGLS examples the reported standard errors are the default ones that assume  correct  specification  of Ω𝑔 .  Better  practice  is  to  add  option vce(robust) for xtreg commands, or option vce(cluster  id\_clu) for mixed commands, as this yields standard errors that are based on the cluster-robust variance defined in (15).

## F. Cluster-Bootstrap Variance Matrix Estimate

Not all  econometrics  packages  compute  cluster-robust  variance  estimates,  and  even those that do may not do so for all estimators. In that case one can use a pairs cluster bootstrap that, like the CRVE, gives a consistent estimate of V [ 𝜷 � ] when errors are clustered.

To  implement  this  bootstrap,  do  the  following  steps 𝐵 times:  (1)  form 𝐺 clusters {( 𝒚1 ∗ , 𝑿1 ∗ ), . . . , ( 𝒚𝐺 ∗ , 𝑿𝐺 ∗ )} by resampling with replacement 𝐺 times from the original sample of clusters, and (2) compute the estimate of 𝜷 , denoted 𝜷 � 𝑏 in the 𝑏 𝑡ℎ bootstrap sample. Then, given the 𝐵 estimates 𝜷 � 1 , . . . , 𝜷 � 𝐵 , compute the variance of these

<!-- formula-not-decoded -->

where 𝜷 � = 𝐵 -1 ∑ 𝜷 � 𝑏 𝐵 𝑏=1 and 𝐵 = 400 should be more than adequate in most settings. It is important  that  the  resampling  be  done  over  entire  clusters,  rather  than  over  individual observations. Each bootstrap resample will have exactly 𝐺 clusters, with some of the original clusters  not  appearing  at  all  while  others  of  the  original  clusters  may  be  repeated  in  the resample two or more times. The terms 'pairs' is used as ( 𝒚𝑔 , 𝑿𝑔 ) are resampled as a pair. The term 'nonparametric' is also used for this bootstrap. Some alternative bootstraps hold 𝑿𝑔 fixed while resampling. For finite clusters, if V � clu [ 𝜷 � ] uses √𝑐𝑢 �𝑔 in (11) then for comparability V � clu ; boot [ 𝜷 � ] should be multiplied by the constant 𝑐 defined in (12). The pairs cluster bootstrap leads to a cluster-robust variance matrix for OLS with rank 𝐾 even if 𝐾 &gt; 𝐺 .

An  alternative  resampling  method  that  can  be  used  is  the  leave-one-cluster-out jackknife. Then, letting 𝜷 � 𝑔 denote the estimator of 𝜷 when the 𝑔 𝑡ℎ cluster is deleted,

<!-- formula-not-decoded -->

where 𝜷 � = 𝐺 -1 ∑ 𝜷 � 𝑔 𝐺 𝑔=1 .  This  older  method  can  be  viewed  as  an  approximation  to  the bootstrap that does not work as well for nonlinear estimators. It is used less often than the bootstrap, and has the same rank as the CRVE.

Unlike a percentile-t cluster bootstrap, presented later, the pairs cluster bootstrap and cluster jackknife variance matrix estimates are no better asymptotically than the CRVE, so it is best and quickest to use the CRVE if it is available. But the CRVE is not always available, especially for estimators more complicated than OLS. In that case one can instead use the pairs cluster bootstrap, though see the end of Subsection VI.C for potential pitfalls if there are few clusters, or even the cluster jackknife.

In Stata the pairs cluster bootstrap for OLS without fixed effects can be implemented in several  equivalent  ways  including: regress y x, vce(boot, cluster(id\_clu) reps(400)  seed(10101) ); xtreg  y  x,  pa  corr(ind)  vce(boot,  reps(400) seed(10101)) ; and bootstrap,  cluster(id\_clu)  reps(400)  seed(10101) : regress y x .  The last variant can be used for estimation commands and user-written programs that do not have a vce(boot) option. We recommend 400 bootstrap iterations for published results and for replicability one should always set the seed.

For the jackknife the commands  are  instead, respectively, regress  y  x, vce(jack, cluster(id\_clu) ); xtreg y x, pa corr(ind) vce(jack) ; and jackknife, cluster(id\_clu): regress y x .  For  Stata xt commands, options vce(boot) and vce(jack) are  generally  interpreted  as  meaning  cluster  bootstrap  and cluster jackknife; always so from Stata 12.1 on.

## III. Cluster-Specific Fixed Effects

The  cluster-specific  fixed  effects  (FE)  model  includes  a  separate  intercept  for  each cluster, so

<!-- formula-not-decoded -->

where 𝑑ℎ𝑖𝑔 , the ℎ 𝑡ℎ of 𝐺 dummy variables, equals one if the 𝑖𝑔 𝑡ℎ observation is in cluster ℎ and equals zero otherwise.

There  are  several  different  ways  to  obtain  the  same  cluster-specific  fixed  effects estimator. The two most commonly-used are the following. The least squares dummy variable (LSDV) estimator directly estimates the second line of (17), with OLS regression of 𝑦𝑖𝑔 on 𝒙𝑖𝑔 and the 𝐺 dummy  variables 𝑑 1 𝑖𝑔 , . . . , 𝑑𝐺𝑖𝑔 ,  in  which  case  the  dummy  variable coefficients 𝛼 �𝑔 = 𝑦̄𝑔 -𝒙̄𝑔 ′ 𝜷 � where 𝑦̄𝑔 = 𝑁𝑔 -1 ∑ 𝑦𝑖𝑔 𝐺 𝑖=1 and 𝒙̄𝑔 = 𝑁𝑔 -1 ∑ 𝒙𝑖𝑔 𝐺 𝑖=1 .  The  within estimator, also called the fixed effects estimator, estimates 𝜷 just by OLS regression in the within or mean-differenced model

<!-- formula-not-decoded -->

The main reason that empirical economists use the cluster-specific FE estimator is that it  controls  for  a  limited  form  of  endogeneity  of  regressors.  Suppose  in  (17)  that  we  view 𝛼𝑔 + 𝑢𝑖𝑔 as the error, and the regressors 𝒙𝑖𝑔 are correlated with this error, but only with the cluster-invariant component, i.e., Cov [ 𝒙𝑖𝑔 , 𝛼𝑔 ] ≠ 𝟎 while Cov [ 𝒙𝑖𝑔 , 𝑢𝑖𝑔 ] = 𝟎 . Then OLS and FGLS  regression  of 𝑦𝑖𝑔 on 𝒙𝑖𝑔 ,  as  in  Section  II,  leads  to  inconsistent  estimation  of 𝜷 . Mean-differencing (17) leads to  the  within  model  (18)  that  has  eliminated  the  problematic cluster-invariant error component 𝛼𝑔 . The resulting FE estimator is consistent for 𝜷 if either 𝐺 ⟶ ∞ or 𝑁𝑔 ⟶ ∞ .

The cluster-robust variance matrix formula given in Section II carries over immediately to OLS estimation in the FE model, again assuming 𝐺 ⟶ ∞ .

In the remainder of this section we consider some practicalities. First, including fixed effects generally does not control for all the within-cluster correlation of the error and one should  still  use  the  CRVE.  Second,  when  cluster  sizes  are  small  and  degrees-of-freedom corrections are used the CRVE should be computed by within rather than LSDV estimation. Third, FGLS estimators need to be bias-corrected when cluster sizes are small. Fourth, tests of fixed versus random effects models should use a modified version of the Hausman test.

## A. Do Fixed Effects Fully Control for Within-Cluster Correlation?

While cluster-specific effects will control for part of the within-cluster correlation of the error, in general they will not completely control for within-cluster error correlation (not to mention heteroskedasticity). So the CRVE should still be used. There are several ways to make this important point.

Suppose we have data on students in classrooms in schools. A natural model, a special case of a hierarchical model, is to suppose that there is both an unobserved school effect and, on top of that, an unobserved classroom effect. Letting 𝑖 denote individual, 𝑠 school, and 𝑐 classroom,  we  have 𝑦𝑖𝑠𝑐 = 𝒙𝑖𝑠𝑐 ′ 𝜷 + 𝛼𝑠 + 𝛿𝑐 + 𝜀𝑖𝑠𝑐 .  A  regression  with  school-level  fixed effects (or random effects) will control for within-school correlation, but not the additional within-classroom correlation.

Suppose  we  have  a  short  panel ( 𝑇 fixed, 𝑁 → ∞ ) of  uncorrelated  individuals  and estimate 𝑦𝑖𝑡 = 𝒙𝑖𝑡 ′ 𝜷 + 𝛼𝑖 + 𝑢𝑖𝑡 .  Then  the  error 𝑢𝑖𝑡 may  be  correlated  over  time  (i.e., within-cluster) due to omitted factors that evolve progressively over time. Inoue and Solon (2006) provide a test for this serial correlation. Cameron and Trivedi (2005, p.710) present an FE individual-level panel data log-earnings regressed on log-hours example with cluster-robust standard errors four times the default. Serial correlation in the error may be due to omitting lagged 𝑦 as a regressor. When 𝑦𝑖 , 𝑡-1 is included as an additional regressor in the FE model, the  Arellano-Bond  estimator  is  used  and  even  with 𝑦𝑖 , 𝑡-1 included  the  Arellano-Bond

methodology requires that we first test whether the remaining error 𝑢𝑖𝑡 is serially correlated.

Finally, suppose we have a single cross-section (or a single time series). This can be viewed as regression on a single cluster. Then in the model 𝑦𝑖 = 𝛼 + 𝒙𝑖 ′ 𝜷 + 𝑢𝑖 (or 𝑦𝑡 = 𝛼 + 𝒙𝑡 ′ 𝜷 + 𝑢𝑡 ) , the intercept is the cluster-specific fixed effect. There are many reasons for why the error 𝑢𝑖 (or 𝑢𝑡 ) may be correlated in this regression.

## B. Cluster-Robust Variance Matrix with Fixed Effects

Since  inclusion  of  cluster-specific  fixed  effects  may  not  fully  control  for  cluster correlation (and/or heteroskedasticity), default standard errors that assume 𝑢𝑖𝑔 to be i.i.d. may be invalid. So one should use cluster-robust standard errors.

Arellano (1987) showed that V � clu [ 𝜷 � ] defined in (10)-(11) remains valid for the within estimator that controls for inclusion of 𝐺 cluster-specific fixed effects, provided 𝐺 → ∞ and 𝑁𝑔 is  small.  If  instead one obtains the LSDV estimator, the CRVE formula gives the same CRVE  for 𝜷 � as  that  for  the  within  estimator,  with  the  important  proviso  that  the  same degrees-of-freedom  adjustment  must  be  used  -  see  below.  The  fixed  effects  estimates 𝛼 �𝑔 obtained for the LSDV estimator are essentially based only on 𝑁𝑔 observations, so V � [ 𝛼 �𝑔 ] is inconsistent for V [ 𝛼 �𝑔 ] , just as 𝛼 �𝑔 is inconsistent for 𝛼𝑔 .

Hansen (2007a, p.600) shows that this CRVE can also be used if additionally 𝑁𝑔 → ∞ , for both the case where within-cluster correlation is always present (e.g. for many individuals in each village) and for the case where within-cluster correlation eventually disappears (e.g. for panel data where time series correlation disappears for observations far apart in time). The rates of  convergence  are √𝐺 in  the  first  case  and �𝐺𝑁𝑔 in  the  second  case,  but  the  same asymptotic variance matrix is obtained in either case. Kézdi (2004) analyzed the CRVE in FE models for a range of values of 𝐺 and 𝑁𝑔 .

It is important to note that while LSDV and within estimation lead to identical estimates of 𝜷 , they can yield different standard errors due to different finite sample degrees-of-freedom correction.

It is well known that if default standard errors are used, i.e. it is assumed that 𝑢𝑖𝑔 in (17) is i.i.d., then one can safely use standard errors after LSDV estimation as this correctly views the number of parameters as 𝐺 + 𝐾 rather  than 𝐾 .  If  instead  the  within  estimator  is used, however, manual OLS estimation of (18) will mistakenly view the number of parameters to equal 𝐾 rather than 𝐺 + 𝐾 . (Built-in panel estimation commands for the within estimator, i.e. a fixed effects command, should remain okay to use, since they should be programmed to use 𝐺 + 𝐾 in calculating the standard errors.)

It is not well known that if cluster-robust standard errors are used, and cluster sizes are small,  then  inference  should  be  based  on  the  within  estimator  standard  errors.  We  thank Arindrajit Dube and Jason Lindo for bringing this issue to our attention. Within and LSDV estimation  lead  to  the  same  cluster-robust  standard  errors  if  we  apply  formula  (11)  to  the respective regressions, or if we multiply this estimate by 𝑐 = 𝐺 /( 𝐺 1) .  Differences arise, however, if we multiply by the small-sample correction 𝑐 given in (12). Within estimation sets 𝑐 = 𝐺 𝐺-1 × 𝑁-1 𝑁-( 𝐾-1 ) since there  are only ( 𝐾 1) regressors - the within model is estimated without an intercept. LSDV estimation uses 𝑐 = 𝐺 𝐺-1 𝑁-1 𝑁-𝐺-( 𝐾-1 ) since the 𝐺 cluster dummies are also included as regressors. For balanced clusters with 𝑁𝑔 = 𝑁∗ and 𝐺 large relative to 𝐾 , 𝑐 ≃ 1 for  within  estimation  and 𝑐 ≃ 𝑁∗ /( 𝑁∗ 1) for  LSDV estimation. Suppose there are only  two  observations  per  cluster,  due  to  only  two  individuals  per  household  or  two  time periods in a panel setting, so 𝑁𝑔 = 𝑁∗ = 2 . Then 𝑐 ≃ 2/(2 -1) = 2 for LSDV estimation,

leading  to  CRVE  that  is  twice  that  from  within  estimation.  Within  estimation  leads  to  the correct finite-sample correction.

In Stata the within command xtreg y x, fe vce(robust) gives the desired CRVE. The  alternative  LSDV  commands regress y x i.id\_clu, vce(cluster id\_clu) and, equivalently, regress  y  x,  absorb(id\_clu)  vce(cluster id\_clu) use  the  wrong  degrees-of-freedom  correction.  If  a  CRVE  is  needed,  then  use xtreg .  If  there  is  reason to instead use regress i.id then the cluster-robust standard errors  should  be  multiplied  by  the  square  root  of [ 𝑁 -( 𝐾 1)]/[ 𝑁-𝐺-( 𝐾 1)] , especially if 𝑁𝑔 is small.

The  inclusion  of  cluster-specific  dummy  variables  increases  the  dimension  of  the CRVE, but does not lead to a corresponding increase in its rank. To see this, stack the dummy variable 𝑑ℎ𝑖𝑔 for  cluster 𝑔 into  the 𝑁𝑔 ×1 vector 𝒅𝒉𝑔 .  Then 𝒅𝒉𝑔 ′ 𝒖 �𝑔 = 𝟎 ,  by  the  OLS normal equations, leading to the rank of V � clu [ 𝜷 � ] falling by one for each cluster-specific effect. If there are 𝑘 regressors varying within cluster and 𝐺 1 dummies then, even though there are 𝐾 + 𝐺 1 parameters 𝜷 , the rank of V � clu [ 𝜷 � ] is only the minimum of 𝐾 and 𝐺 1 . And a test that 𝛼1 , . . . , 𝛼𝐺 are jointly statistically significant is a test of 𝐺 1 restrictions (since the intercept or one of the fixed effects needs to be dropped).    So even if the cluster-specific fixed effects are consistently estimated (i.e., if 𝑁𝑔 → ∞ ) ,  it  is  not  possible  to  perform  this  test  if 𝐾 &lt; 𝐺 1 , which is often the case.

If cluster-specific effects are present then the pairs cluster bootstrap must be adapted to account  for  the  following  complication.  Suppose  cluster  3  appears  twice  in  a  bootstrap resample.  Then  if  clusters  in  the  bootstrap  resample  are  identified  from  the  original cluster-identifier,  the  two  occurrences  of  cluster  3  will  be  incorrectly  treated  as  one  large cluster rather than two distinct clusters.

In Stata, the bootstrap option idcluster ensures that distinct identifiers are used in each bootstrap resample.  Examples  are regress  y  x  i.id\_clu,  vce(boot, cluster(id\_clu) idcluster(newid) reps(400) seed(10101) )  and,  more simply, xtreg  y  x,  fe  vce(boot,  reps(400)  seed(10101) ) , as in this latter case Stata automatically accounts for this complication.

## C. Feasible GLS with Fixed Effects

When cluster-specific fixed effects are present, more efficient FGLS estimation can become more complicated. In particular, if asymptotic theory relies on 𝐺 → ∞ with 𝑁𝑔 fixed, the 𝛼𝑔 cannot be consistently estimated. The within estimator of 𝜷 is nonetheless consistent, as 𝛼𝑔 disappears  in  the  mean-differenced  model.  But  the  resulting  residuals 𝑢 � 𝑖𝑔 are contaminated, since they depend on both 𝜷 � and 𝛼 �𝑔 , and these residuals will be used to form a FGLS estimator. This leads to bias in the FGLS estimator, so one needs to use bias-corrected FGLS unless 𝑁𝑔 → ∞ .  The  correction  method  varies  with  the  model  for Ω𝑔 = V [ 𝒖𝑔 ] ,  and currently there are no Stata user-written commands to implement these methods.

For panel data a commonly-used model specifies an AR(p) model for the errors 𝑢𝑖𝑔 in (17). If fixed effects are present, then there is a bias (of order 𝑁𝑔 -1 ) in estimation of the AR(p) coefficients. Hansen (2007b) obtains bias-corrected estimates of the AR(p) coefficients and uses these in FGLS estimation. Hansen (2007b) in simulations shows considerable efficiency gains in bias-corrected FGLS compared to OLS.

Brewer, Crossley, and Joyce (2013) consider a DiD model with individual-level U.S. panel data with 𝑁 = 750,127, 𝑇 = 30 , and a placebo state-level law so clustering is on state with 𝐺 = 50 . They find that bias-corrected FGLS for AR(2) errors, using the Hansen (2007b)

correction,  leads  to  higher  power  than  FE  estimation.  In  their  example  ignoring  the  bias correction does not change results much, perhaps because 𝑇 = 30 is reasonably large.

For  balanced  clusters  with Ω𝑔 the  same  for  all 𝑔 ,  say Ω𝑔 = Ω∗ ,  and  for 𝑁𝑔 small, then the FGLS estimator in (13) can be used without need to specify a model for Ω∗ . Instead we  can  let Ω � ∗ have 𝑖𝑗 𝑡ℎ entry 𝐺 -1 ∑ 𝑢 � 𝑖𝑔 𝐺 𝑔=1 𝑢 �𝑗𝑔 , where 𝑢 � 𝑖𝑔 are  the  residuals  from  initial OLS  estimation. These assumptions may  be  reasonable for a balanced panel.  Two complications  can  arise.  First,  even  without  fixed  effects  there  may  be  many  off-diagonal elements  to  estimate  and  this  number  can  be  large  relative  to  the  number  of  observations. Second, the fixed effects lead to bias in estimating the off-diagonal covariances. Hausman and Kuersteiner (2008) present fixes for both complications.

## D. Testing the Need for Fixed Effects

FE  estimation  is  often  accompanied  by  a  loss  of  precision  in  estimation,  as  only within-cluster variation is used (recall we regress ( 𝑦𝑖𝑔 -𝑦̄𝑔 ) on ( 𝒙𝑖𝑔 -𝒙̄𝑔 )) .  Furthermore, the coefficient of a cluster-invariant regressor is not identified, since then 𝑥𝑖𝑔 -𝑥̄𝑔 = 0 . Thus it  is  standard  to  test  whether  it  is  sufficient  to  estimate  by  OLS  or  FGLS,  without cluster-specific fixed effects.

The usual test is a Hausman test based on the difference between the FE estimator, 𝜷 � FE ,  and  the  RE  estimator, 𝜷 � RE .  Let 𝜷1 denote  a  subcomponent  of 𝜷 ,  possibly  just  the coefficient  of  a  single  regressor  of  key  interest;  at  most 𝜷1 contains  the  coefficients  of  all regressors  that  are  not  invariant  within  cluster  or,  in  the  case  of  panel  data,  that  are  not aggregate time effects that take the same value for each individual. The chi-squared distributed test statistic is

<!-- formula-not-decoded -->

where V � is a consistent estimate of V [ 𝜷 � 1 ; FE -𝜷 � 1 ; RE ] .

Many studies use the standard form of the Hausman test. This obtains V � under  the strong  assumption  that 𝜷 � RE is  fully  efficient  under  the  null  hypothesis.  This  requires  the unreasonably strong assumptions that 𝛼𝑔 and 𝜀𝑖𝑔 in (16) are i.i.d., requiring that neither 𝛼𝑔 nor 𝜀𝑖𝑔 is  heteroskedastic  and  that 𝜀𝑖𝑔 has  no  within-cluster  correlation.  As  already  noted, these assumptions are likely to fail and one should not use default standard errors. Instead a CRVE should be used. For similar reasons the standard form of the Hausman test should not be used.

Wooldridge (2010, p.332) instead proposes implementing a cluster-robust version of the Hausman test by the following OLS regression

<!-- formula-not-decoded -->

where 𝒘𝑔 denotes the subcomponent of 𝒙𝑖𝑔 that varies within cluster and 𝒘̄ 𝑔 = 𝑁𝑔 -1 ∑ 𝒘𝑖𝑔 𝐺 𝑖=1 . If 𝐻0 : 𝜸 = 𝟎 is rejected using a Wald test based on a cluster-robust estimate of the variance matrix, then the fixed effects model is necessary. The Stata user-written command xtoverid , due to Schaffer and Stillman (2010), implements this test.

An  alternative  is  to  use  the  pairs  cluster  bootstrap  to  obtain V � ,  in  each  resample obtaining 𝜷 � 1 ; FE and 𝜷 � 1 ; RE ,  leading  to 𝐵 resample  estimates  of ( 𝜷 � 1 ; FE -𝜷 � 1 ; RE ) .  We  are unaware of studies comparing these two cluster-robust versions of the Hausman test.

## IV. What to Cluster Over?

It is not always clear what to cluster over - that is, how to define the clusters - and there may even be more than one way to cluster.

Before providing some guidance, we note that it is possible for cluster-robust errors to actually  be  smaller  than  default  standard  errors.  First,  in  some  rare  cases  errors  may  be negatively correlated, most likely when 𝑁𝑔 = 2 , in which case (6) predicts a reduction in the standard error. Second, cluster-robust is also heteroskedastic-robust and White heteroskedastic-robust standard errors in practice are sometimes larger and sometimes smaller than the default. Third, if clustering has a modest effect, so cluster-robust and default standard errors  are  similar  in  expectation,  then  cluster-robust  may  be  smaller  due  to  noise.  In  cases where the cluster-robust standard errors are smaller they are usually not much smaller than the default, whereas in other applications they can be much, much larger.

## A. Factors Determining What to Cluster Over

There are two guiding principles that determine what to cluster over.

First, given V [ 𝜷 � ] defined in (7) and (9) whenever there is reason to believe that both the  regressors  and  the  errors  might  be  correlated  within  cluster,  we  should  think  about clustering defined in a broad enough way to account for that clustering. Going the other way, if we think that either the regressors or the errors are likely to be uncorrelated within a potential group, then there is no need to cluster within that group.

Second, V � clu [ 𝜷 � ] is  an  average  of 𝐺 terms  that  gets  closer  to  V [ 𝜷 � ] only  as 𝐺 gets large. If we define very large clusters, so that there are very few clusters to average over in equation  (11),  then  the  resulting V � clu [ 𝜷 � ] can  be  a  very  poor  estimate  of  V [ 𝜷 � ] .  This complication, and discussion of how few is 'few', is the subject of Section VI.

These two principles mirror the bias-variance trade-off that is common in many estimation problems - larger and fewer clusters have less bias but more variability. There is no general solution  to  this  trade-off,  and  there  is  no  formal  test  of  the  level  at  which  to  cluster.  The consensus is to be conservative and avoid bias and use bigger and more aggregate clusters when possible, up to and including the point at which there is concern about having too few clusters.

For example, suppose your dataset included individuals within counties within states, and you were considering whether to cluster at the county level or the state level. We have been inclined  to  recommend  clustering  at  the  state  level.  If  there  was  within-state  cross-county correlation  of  the  regressors  and  errors,  then  ignoring  this  correlation  (for  example,  by clustering at the county level) would lead to incorrect inference. In practice researchers often cluster at progressively higher (i.e., broader) levels and stop clustering when there is relatively little change in the standard errors. This seems to be a reasonable approach.

There are settings where one may not need to use cluster-robust standard errors. We outline  several,  though  note  that  in  all  these  cases  it  is  always  possible  to  still  obtain cluster-robust  standard  errors  and  contrast  them  to  default  standard  errors.  If  there  is  an appreciable difference, then use cluster-robust standard errors.

If  a  key  regressor  is  randomly  assigned  within  clusters,  or  is  as  good  as  randomly assigned, then the within-cluster correlation of the regressor is likely to be zero. Thus there is no need to cluster standard errors, even if the model's errors are clustered. In this setting, if there  are  additionally  control  variables  of  interest,  and  if  these  are  not  randomly  assigned within cluster, then we may wish to cluster our standard errors for the sake of correct inference on the control variables.

If the model includes cluster-specific fixed effects, and we believe that within-cluster

correlation of errors is solely driven by a common shock process, then we may not be worried about clustering. The fixed effects will absorb away the common shock, and the remaining errors will have zero within-cluster correlation. More generally, control variables may absorb systematic  within-cluster  correlation.  For  example,  in  a  state-year  panel  setting,  control variables may capture the state-specific business cycle.

However, as already noted in Subsection III.A, the within-cluster correlation is usually not fully eliminated. And even if it is eliminated, the errors may still be heteroskedastic. Stock and  Watson  (2008)  show  that  applying  the  usual  White  (1980)  heteroskedastic-consistent variance matrix estimate to the FE estimator leads, surprisingly, to inconsistent estimation of V [ 𝜷 � ] if 𝑁𝑔 is small (though it is correct if 𝑁𝑔 = 2 ). They derive a bias-corrected formula for heteroskedastic-robust standard errors. Alternatively, and more simply, the CRVE is consistent for V [ 𝜷 � ] , even if the errors are only heteroskedastic, though this estimator of V [ 𝜷 � ] is more variable.

Finally, as already noted in Subsection II.D we can always build a parametric model of the correlation structure of the errors and estimate by FGLS. If we believe that this parametric model captures the salient features of the error correlations, then default FGLS standard errors can be used.

## B. Clustering Due to Survey Design

Clustering  routinely  arises  due  to  the  sampling  methods  used  in  complex  surveys. Rather  than  randomly  draw  individuals  from  the  entire  population,  costs  are  reduced  by sampling only a randomly-selected subset of primary sampling units (such as a geographic area),  followed  by  random  selection,  or  stratified  selection,  of  people  within  the  chosen primary sampling units.

The survey methods literature uses methods to control for clustering that predate the cluster-robust approach of this paper. The loss of estimator precision due to clustered sampling is called the design effect: 'The design effect or Deff is the ratio of the actual variance of a sample to the variance of a simple random sample of the same number of elements' (Kish (1965),  p.258)).  Kish  and  Frankel  (1974)  give  the  variance  inflation  formula  (6)  in  the non-regression case of estimation of the mean. Pfeffermann and Nathan (1981) consider the more general regression case. The CRVE is called the linearization formula, and the common use  of 𝐺 1 as  the  degrees  of  freedom  used  in  hypothesis  testing  comes  from  the  survey methods  literature;  see  Shah,  Holt  and  Folsom  (1977)  which  predates  the  econometrics literature.

Applied economists routinely use data from complex surveys, controlling for clustering by using a cluster-robust variance matrix estimate. At the minimum one should cluster at the level of the primary sampling unit, though often there is reason to cluster at a broader level, such as clustering on state if regressors and errors are correlated within state.

The survey methods literature additionally controls for two other features of survey data weighting  and  stratification.  These  methods  are  well-established  and  are  incorporated  in specialized software, as well as in some broad-based packages such as Stata. Bhattacharya (2005) provides a comprehensive treatment in a GMM framework.

If sample weights are provided then it is common to perform weighted least squares. Then  the  CRVE  for 𝜷 � WLS = ( 𝑿 ′ 𝑾𝑿 ) -1 𝑿 ′ 𝑾𝒚 is  that  given  in  (15)  with Ω � 𝑔 -1 replaced  by 𝑾𝑔 . The need to weight can be ignored if stratification is on only the exogenous regressors and we assume correct specification of the model, so that in our sample E [ 𝒚 | 𝑿 ] = 𝑿𝜷 .  In  that special case both weighted and unweighted estimators are consistent, and weighted OLS may actually  be  less  efficient  if,  for  example,  model  errors  are  i.i.d.;  see,  for  example,  Solon, Haider,  and  Wooldridge  (2013).  Another  situation  in  which  to  use  weighted  least  squares,

unrelated to complex surveys, is when data for the 𝑖𝑔 𝑡ℎ observation is obtained by in turn averaging 𝑁𝑖𝑔 observations and 𝑁𝑖𝑔 varies.

Stratification of the sample can enable more precise statistical inference. These gains can  be  beneficial  in  the  nonregression  case,  such  as  estimating  the  monthly  national unemployment rate. The gains can become much smaller once regressors are included, since these can partially control for stratification; see, for example, the application in Bhattacharya (2005).  Econometrics  applications  therefore  usually  do  not  adjust  standard  errors  for stratification, leading to conservative inference due to some relatively small over-estimation of the standard errors.

## V. Multi-way Clustering

The discussion to date has presumed that if there is more than one potential way to cluster, these ways are nested in each other, such as households within states. But when clusters are non-nested, traditional cluster-robust inference can only deal with one of the dimensions.

In some applications it is possible to include sufficient regressors to eliminate concern about error correlation in all but one dimension, and then do cluster-robust inference for that remaining dimension. A leading example is that in a state-year panel there may be clustering both within years and within states. If the within-year clustering is due to shocks that are the same across all observations in a given year, then including year fixed effects as regressors will absorb within-year clustering, and inference then need only control for clustering on state.

When  this  approach  is  not  applicable,  the  one-way  cluster  robust  variance  can  be extended to multi-way clustering. Before discussing this topic, we highlight one error that we find some practitioners make, which is to cluster at the intersection of the two groupings. In the preceding example, some might be tempted to cluster at the state-year level. A Stata example is to use the command regress y x, vce(cluster id\_styr) where id\_styr is  a state-year  identifier.  This  will  be  very  inadequate,  since  it  imposes  the  restriction  that observations are independent if they are in the same state but in different years. Indeed if the data is aggregated at the state-year level, there is only one observation at the state-year level, so this is identical to using heteroskedastic-robust standard errors, i.e. not clustering at all. This point was highlighted by Bertrand, Duflo, and Mullainathan (2004) who advocated clustering on the state.

## A. Multi-way Cluster-Robust Variance Matrix Estimate

The  cluster-robust  estimate  of  V [ 𝜷 � ] defined  in  (10)-(11)  can  be  generalized  to clustering in multiple dimensions. In a change of notation, suppress the subscript for cluster and more simply denote the model for an individual observation as

<!-- formula-not-decoded -->

Regular  one-way  clustering  is  based  on  the  assumption  that  E [ 𝑢𝑖𝑢𝑗 | 𝒙𝑖 , 𝒙𝑗 ] = 0 ,  unless observations 𝑖 and 𝑗 are in the same cluster. Then (11) sets 𝑩 � = ∑ ∑ 𝒙𝑖 𝑁 𝑗=1 𝑁 𝑖=1 𝒙𝑗 ′ 𝑢 � 𝑖 𝑢 �𝑗𝟏 [ 𝑖 , 𝑗 in same  cluster ] ,  where 𝑢 � 𝑖 = 𝑦𝑖 -𝒙𝑖 ′ 𝜷 � .  In  multi-way  clustering,  the  key  assumption  is  that E [ 𝑢𝑖𝑢𝑗 | 𝒙𝑖 , 𝒙𝑗 ] = 0 ,  unless  observations 𝑖 and 𝑗 share  any  cluster  dimension.  Then  the multi-way cluster robust estimate of V [ 𝜷 � ] replaces (11) with

<!-- formula-not-decoded -->

This method relies on asymptotics that are in the number of clusters of the dimension with the fewest number of clusters. This method is thus most appropriate when each dimension has many clusters.

Theory  for  two-way  cluster  robust  estimates  of  the  variance  matrix  is  presented  in Cameron, Gelbach, and Miller (2006, 2011), Miglioretti and Heagerty (2006), and Thompson (2006, 2011). See also empirical panel data applications by Acemoglu and Pischke (2003), who  clustered  at  individual  level  and  at  region × time  level,  and  by  Petersen  (2009),  who clustered  at  firm  level  and  at  year  level.  Cameron,  Gelbach  and  Miller  (2011)  present  an extension to multi-way clustering. Like one-way cluster-robust, the method can be applied to estimators other than OLS.

For  two-way  clustering  this  robust  variance  estimator  is  easy  to  implement  given software that computes the usual one-way cluster-robust estimate. First obtain three different cluster-robust 'variance' matrices for the estimator by one-way clustering in, respectively, the first  dimension,  the  second  dimension,  and  by  the  intersection  of  the  first  and  second dimensions.  Then  add  the  first  two  variance  matrices  and,  to  account  for  double-counting, subtract the third. Thus

<!-- formula-not-decoded -->

where  the  three  component  variance  estimates  are  computed  using  (10)-(11)  for  the  three different ways of clustering.

We spell this out in a step-by-step fashion.

1. Identify  your  two  ways  of  clustering.  Make  sure  you  have  a  variable  that identifies each way of clustering. Also create a variable that identifies unique 'group 1 by  group  2'  combinations.  For  example,  suppose  you  have  individual-level  data spanning many U.S. states and many years, and you want to cluster on state and on year. You will need a variable for state (e.g. California), a variable for year (e.g. 1990), and a variable for state-by-year (California and 1990).
2. Estimate your model, clustering on 'group 1'. For example, regress 𝑦 on 𝒙 , clustering on state. Save the variance matrix as V � 1 .
4. Estimate your model, clustering on 'group 1 by group 2'. For example, regress 𝑦 on 𝒙 , clustering on state-by-year. Save the variance matrix as V � 1∩2 .
3. Estimate your model, clustering on 'group 2'. For example, regress 𝑦 on 𝒙 , clustering on year. Save the variance matrix as V � 2 .
5. Create  a  new  variance  matrix V � 2way = V � 1 + V � 2 -V � 1∩2 .  This  is  your  new two-way cluster robust variance matrix for 𝜷 � .
6. Standard errors are the square root of the diagonal elements of this matrix.

If  you  are  interested  in  only  one  coefficient,  you  can  also  just  focus  on  saving  the standard error for this coefficient in steps 2-4 above, and then create se 2way =

�

se

2

2

.

In taking these steps, you should watch out for some potential pitfalls. With perfectly multicollinear regressors, such as inclusion of dummy variables some of which are redundant, a statistical package may automatically drop one or more variables to ensure a nonsingular set of regressors. If the package happens to drop different sets of variables in steps 2, 3, and 4, then the  resulting V � ′ s  will  not  be  comparable,  and  adding  them  together  in  step  5  will  give  a nonsense result. To prevent this issue, manually inspect the estimation results in steps 2, 3, and 4 to ensure that each step has the same set of regressors, the same number of observations, etc. The  only  things  that  should  be  different  are  the  reported  standard  errors  and  the  reported

1

+se

2 -

2

se

1∩2

number of clusters.

## B. Implementation

Unlike the standard one-way cluster case, V � 2way [ 𝜷 � ] is  not guaranteed to be positive semi-definite, so it is possible that it may compute negative variances. In some applications with  fixed  effects, V � [ 𝜷 � ] may  be  non  positive-definite,  but  the  subcomponent  of V � [ 𝜷 � ] associated with the regressors of interest may be positive-definite. This may lead to an error message, even though inference is appropriate  for the parameters of interest. Our informal observation is that  this  issue  is  most  likely  to  arise  when  clustering  is  done  over  the  same groups as the fixed effects. Few clusters in one or more dimensions can also lead to V � 2way [ 𝜷 � ] being  a  non-positive-semidefinite  matrix.  Cameron,  Gelbach  and  Miller  (2011)  present  an eigendecomposition technique used in the time series HAC literature that zeroes out negative eigenvalues in V � 2way [ 𝜷 � ] to produce a positive semi-definite variance matrix estimate.

The  Stata  user-written  command cmgreg , available at the authors' websites, implements  multi-way  clustering  for  the  OLS  estimator  with,  if  needed,  the  negative eigenvalue  adjustment.  The  Stata  add-on  command ivreg2 ,  due  to  Baum,  Schaffer,  and Stillman (2007), implements two-way clustering for OLS, IV and linear GMM estimation. Other researchers have also posted code, available from searching the web.

Cameron, Gelbach, and Miller (2011) apply the two-way method to data from Hersch (1998)  that  examines  the  relationship  between  individual  wages  and  injury  risk  measured separately at the industry level and the occupation level. The log-wage for 5960 individuals is regressed  on  these  two  injury  risk  measures,  with  standard  errors  obtained  by  two-way clustering on 211 industries and 387 occupations. In that case two-way clustering leads to only a  modest  change  in  the  standard  error  of  the  industry  job  risk  coefficient  compared  to  the standard  error  with  one-way  clustering  on  industry.  Since  industry  job  risk  is  perfectly correlated  within  industry,  by  result  (6)  we  need  to  cluster  on  industry  if  there  is  any within-industry error correlation. By similar logic, the additional need to cluster on occupation depends on the within-occupation correlation of job industry risk, and this correlation need not be  high.  For  the  occupation  job  risk  coefficient,  the  two-way  and  one-way  cluster  (on occupation)  standard  errors  are  similar.  Despite  the  modest  difference  in  this  example, two-way clustering avoids the need to report standard errors for one coefficient clustering in one way and for the second coefficient clustering in the second way.

Cameron, Gelbach, and Miller (2011) also apply the two-way cluster-robust method to data on volume of trade between 98 countries with 3262 unique country pairs. In that case, two-way clustering on each of the countries in the country pair leads to standard errors that are 36% larger than one-way clustering an 230% more than heteroskedastic-robust standard errors. Cameron and Miller (2012) study such dyadic data in further detail. They note that two-way clustering  does  not  pick  up  all  the  potential  correlations  in  the  data.  Instead,  more  general cluster-robust methods, including one proposed by Fafchamps and Gubert (2007), should be used.

## C. Feasible GLS

Similar  to  one-way  clustering,  FGLS  is  more  efficient  than  OLS,  provided  an appropriate model for Ω = E [ 𝒖𝒖 ′ | 𝑿 ] is specified and is consistently estimated.

The random effects model can be extended to multi-way clustering. For individual 𝑖 in clusters 𝑔 and ℎ , the two-way random effects model specifies

<!-- formula-not-decoded -->

where the errors 𝛼𝑔 , 𝛿ℎ , and 𝜀𝑖𝑔ℎ are each assumed to be i.i.d. distributed with mean 0. For example, Moulton (1986) considered clustering due to grouping of regressors (schooling, age and weeks worked) in a log earnings regression, and estimated a model with common random shock for each year of schooling, for each year of age, and for each number of weeks worked.

The  two-way  random  effects  model  can  be  estimated  using  standard  software  for (nested) hierarchical linear models; see, for example, Cameron and Trivedi (2009, ch. 9.5.7) for Stata command xtmixed (command mixed from version 13 on). For estimation of a many-way  random  effects  model,  see  Davis  (2002)  who  modeled  film  attendance  data clustered by film, theater and time.

The default standard errors after FGLS estimation require that Ω is correctly specified. For two-way and multi-way random effects models, FGLS estimation entails transforming the data in such a way that there is no obvious method for computing a variance matrix estimate that is robust to misspecification of Ω . Instead if there is concern about misspecification of Ω then one needs to consider FGLS with richer models for Ω and assume that these are correctly specified - see Rabe-Hesketh and Skrondal (2012) for richer hierarchical models in Stata - or do less efficient OLS with two-way cluster-robust standard errors.

## D. Spatial Correlation

Cluster-robust variance matrix estimates are closely related to spatial-robust variance matrix estimates.

In general, for model (19), 𝑩 � in (20) has the form

<!-- formula-not-decoded -->

where 𝑤 ( 𝑖 , 𝑗 ) are weights. For cluster-robust inference these weights are either 1 (cluster in common) or 0 (no cluster in common). But the weights can also decay from one to zero, as in the case of the HAC variance matrix estimate for time series where 𝑤 ( 𝑖 , 𝑗 ) decays to zero as | 𝑖 - 𝑗 | increases.

For spatial data it is assumed that model errors become less correlated as the spatial distance between observations grows. For example, with state-level data the assumption that model errors are uncorrelated across states may be relaxed to allow correlation that decays to zero as the distance between states gets large. Conley (1999) provides conditions under which (10)  and  (22)  provide  a  robust  variance  matrix  estimate  for  the  OLS  estimator,  where  the weights 𝑤 ( 𝑖 , 𝑗 ) decay with the spatial distance. The estimate (which Conley also generalizes to GMM models) is often called a spatial-HAC estimate, rather than spatial-robust, as proofs use mixing conditions (to ensure decay of dependence) as observations grow apart in distance. These conditions are not applicable to clustering due to common shocks which leads to the cluster-robust estimator with independence of observations across clusters.

Driscoll and Kraay (1998) consider panel data with 𝑇 time periods and 𝑁 individuals, with errors potentially correlated across individuals (and no spatial dampening), though this correlation across individuals disappears for observations that are more than 𝑚 time periods apart. Let 𝑖𝑡 denote the typical observation. The Driscoll-Kraay spatial correlation consistent (SCC) variance matrix estimate can be shown to use weight 𝑤 ( 𝑖𝑡 , 𝑗𝑠 ) = 1 -𝑑 ( 𝑖𝑡 , 𝑗𝑠 )/( 𝑚 + 1) in  (22),  where  the  summation  is  now  over 𝑖 , 𝑗 , 𝑠 and 𝑡 ,  and 𝑑 ( 𝑖𝑡 , 𝑗𝑠 ) = | 𝑡 - 𝑠 | if | 𝑡 - 𝑠 | ≤ 𝑚 and 𝑑 ( 𝑖𝑡 , 𝑗𝑠 ) = 0 otherwise.  This  method  requires  that  the  number  of  time periods 𝑇 → ∞ , so is not suitable for short panels, while 𝑁 may be fixed or 𝑁 → ∞ . The Stata add-on command xtscc , due to Hoechle (2007), implements this variance estimator.

An estimator proposed by Thompson (2006) allows for across-cluster (in his example firm) correlation for observations close in time in addition to within-cluster correlation at any time separation. The Thompson estimator can be thought of as using 𝑤 ( 𝑖𝑡 , 𝑗𝑠 ) = 𝟏 [ 𝑖 , 𝑗 share a firm,  or 𝑑 ( 𝑖𝑡 , 𝑗𝑠 ) ≤ 𝑚 ] .  Foote  (2007)  contrasts  the  two-way  cluster-robust  and  these  other variance  matrix  estimators  in  the  context  of  a  macroeconomics  example.  Petersen  (2009) contrasts various methods for panel data on financial firms, where there is concern about both within firm correlation (over time) and across firm correlation due to common shocks.

Barrios,  Diamond,  Imbens,  and  Kolesár  (2012)  consider  state-year  panel  data  on individuals  in  states  over  years  with  state-level  treatment  and  outcome  (earnings)  that  is correlated  spatially  across  states.  This  spatial  correlation  can  be  ignored  if  the  state-level treatment is randomly assigned. But if the treatment is correlated over states (e.g. adjacent states may be more likely to have similar minimum wage laws) then one can no longer use standard errors clustered at the state level. Instead one should additionally allow for spatial correlation of errors across states. The authors additionally contrast traditional model-based inference with randomization inference.

In practice data can have cluster, spatial and time series aspects, leading to hybrids of cluster-robust, spatial-HAC and time-series HAC estimators. Furthermore, it may be possible to parameterize some of the error correlation. For example for a time series AR(1) error it may be  preferable  to  use E � [ 𝑢𝑡𝑢𝑠 ] based  on  an  AR(1)  model  rather  than 𝑤 ( 𝑡 , 𝑠 ) 𝑢 � 𝑡 𝑢 � 𝑠 .  To  date empirical practice has not commonly modeled these combined types of error correlations. This may become more common in the future.

## VI. Few Clusters

We return to one-way clustering, and focus on the Wald 't-statistic'

<!-- formula-not-decoded -->

where 𝛽 is one element in the parameter vector 𝜷 , and the standard error 𝑠 𝛽 � is the square root of the appropriate diagonal entry in V � clu [ 𝜷 � ] .  If 𝐺 → ∞ then 𝑤 ∼ 𝑁 [0,1] under 𝐻0 : 𝛽 = 𝛽0 . For finite 𝐺 the distribution of 𝑤 is unknown, even with normal errors. It is common to use the 𝑇 distribution with 𝐺 1 degrees of freedom.

One  should  at  a  minimum  use 𝑇 ( 𝐺 1) critical  values  and V � clu [ 𝜷 � ] defined  in (10)-(11)  with  residuals  scaled  by √𝑐 where 𝑐 is  defined  in  (12)  or 𝑐 = 𝐺 /( 𝐺 1) .  Most packages rescale the residuals - Stata uses the first choice of 𝑐 and SAS the second. The use of 𝑇 ( 𝐺 1) critical values is less common. Stata uses the 𝑇 ( 𝐺 1) distribution after command regress  y  x,  vce(cluster) . But the alternative  command xtreg  y  x, vce(robust) instead uses standard normal critical values.

It is not unusual for the number of clusters 𝐺 to be quite small. Despite few clusters, 𝛽 ̂ may still be a reasonably precise estimate of 𝛽 if there are many observations per cluster. But with small 𝐺 the asymptotics have not kicked in. Then V � clu [ 𝜷 � ] can be downwards-biased.

Even with both of these adjustments, Wald tests generally over-reject. The extent of over-rejection depends on both how few clusters there are and the particular data and model used. In some cases the over-rejection is mild, in others cases a test with nominal size 0.05 may have true test size of 0.10 .

The  next  subsection  outlines  the  basic  problem  and  discusses  how  few  is  'few' clusters. The subsequent three subsections present three approaches to finite-cluster correction - bias-corrected variance, bootstrap with asymptotic refinement, and use of the 𝑇 distribution

with adjusted degrees-of-freedom. The final subsection considers special cases.

## A. The Basic Problems with Few Clusters

There are two main problems with few clusters.    First, OLS leads to 'overfitting', with estimated residuals systematically too close to zero compared to the true error terms. This leads to a downwards-biased cluster-robust variance matrix estimate. The second problem is that even with bias-correction,  the  use  of  fitted  residuals  to  form  the  estimate 𝑩 � of 𝑩 leads  to over-rejection (and confidence intervals that are too narrow) if the critical values are from the standard normal or even the 𝑇 ( 𝐺 1) distribution.

For the linear regression model with independent homoskedastic normally distributed errors  similar  problems  are  easily  controlled.  An  unbiased  variance  matrix  is  obtained  by estimating  the  error  variance 𝜎 2 by 𝑠 2 = 𝒖 � ′ 𝒖 � /( 𝑁-𝐾 ) rather  than 𝒖 � ′ 𝒖 � / 𝑁 .    This  is  the 'fix' in the OLS setting for the first problem. The analogue to the second problem is that the 𝑁 [0,1] distribution is a poor approximation to the true distribution of the Wald statistic. In the i.i.d.  case,  the  Wald  statistic 𝑤 can  be  shown  to  be  exactly 𝑇 ( 𝑁-𝐾 ) distributed.  For nonnormal homoskedastic errors the 𝑇 ( 𝑁-𝐾 ) is still felt to provide a good approximation, provided 𝑁 is not too small.    Both of these problems arise in the clustered setting, albeit with more complicated manifestations and fixes.

For independent heteroskedastic normally distributed errors there are no exact results. MacKinnon and White (1985) consider several adjustments to the heteroskedastic-consistent variance estimate of White (1980), including one called HC2 that is unbiased in the special case  that  errors  are  homoskedastic.  Unfortunately  if  errors  are  actually  heteroskedastic,  as expected, the HC2 estimate is no longer unbiased for V [ 𝜷 � ] - an unbiased estimator depends on the unknown pattern of heteroskedasticity and on the design matrix 𝑿 . And there is no way to obtain  an  exact 𝑇 distribution  result  for 𝑤 ,  even  if  errors  are  normally  distributed.  Other proposed solutions for testing and forming confidence intervals include using a 𝑇 distribution with data-determined degrees of freedom, and using a bootstrap with asymptotic refinement.

In the following subsections we consider extensions of these various adjustments to the clustered case, where the problems can become even more pronounced.

Before proceeding we note that there is no specific point at which we need to worry about few clusters. Instead, 'more is better'. Current consensus appears to be that 𝐺 = 50 is enough for state-year panel data. In particular, Bertrand, Duflo, and Mullainathan (2004, Table 8)  find  in  their  simulations  that  for  a  policy  dummy  variable  with  high  within-cluster correlation, a Wald test based on the standard CRVE with critical value of 1.96 had rejection rates of .063, .058, .080, and .115 for number of states ( 𝐺 ) equal to, respectively, 50, 20, 10 and 6. The simulations of Cameron, Gelbach and Miller (2008, Table 3), based on a quite different data generating process but again with standard CRVE and critical value of 1.96, had rejection rates of .068, .081, .118, and .208 for 𝐺 equal to, respectively, 30, 20, 10 and 5. In both cases the rejection rates would also exceed .05 if the critical value was from the 𝑇 ( 𝐺 1) distribution.

The preceding results are for balanced clusters. Cameron, Gelbach and Miller (2008, Table  4,  column  8)  consider  unbalanced  clusters  when 𝐺 = 10 .  The  rejection  rate  with unbalanced clusters,  half  of  size 𝑁𝑔 = 10 and  half  of  size 50 ,  is . 183 ,  appreciably  worse than  rejection  rates  of . 126 and . 115 for  balanced  clusters  of  sizes,  respectively, 10 and 100 .  Recent  papers  by  Carter,  Schnepel,  and  Steigerwald  (2013)  and  Imbens  and  Kolesar (2012) provide theory that also indicates that the effective number of clusters is reduced when 𝑁𝑔 varies across clusters; see also the simulations in MacKinnon and Webb (2013). Similar issues may also arise if the clusters are balanced, but estimation is by weighted LS that places different weights on different clusters.    Cheng and Hoekstra (2013) document that weighting

can  result  in  over-rejection  in  the  panel  DiD  setting  of  Bertrand,  Duflo,  and  Mullainathan (2004).

To repeat a key message, there is no clear-cut definition of 'few'. Depending on the situation 'few' may range from less than 20 clusters to less than 50 clusters in the balanced case, and even more clusters in the unbalanced case.

## B. Solution 1: Bias-Corrected Cluster-Robust Variance Matrix

A weakness of the standard CRVE with residual 𝒖 �𝑔 is  that  it  is  biased  for V clu [ 𝜷 � ] , since  E [ 𝒖 �𝑔𝒖 �𝑔 ′ ] ≠ E [ 𝒖𝑔𝒖𝑔 ′ ] .  The  bias  depends  on  the  form  of Ω𝑔 but  will  usually  be downwards. Several corrected residuals 𝒖 �𝑔 to use in place of 𝒖 �𝑔 in (11) have been proposed. The simplest, already mentioned, is to use 𝒖 �𝑔 = �𝐺 /( 𝐺 1) 𝒖 �𝑔 or 𝒖 �𝑔 = √𝑐𝒖 �𝑔 where 𝑐 is defined in (12). One should at least use either of these corrections.

Bell and McCaffrey (2002) use

<!-- formula-not-decoded -->

where 𝑯𝑔𝑔 = 𝑿𝑔 ( 𝑿 ′ 𝑿 ) -1 𝑿𝑔 ′ . This transformed residual leads to unbiased CRVE in the special case  that  E [ 𝒖𝑔𝒖𝑔 ′ ] = 𝜎 2 𝑰 .  This  is  a  cluster  generalization  of  the  HC2  variance  estimate  of MacKinnon and White (1985), so we refer to it as the CR2VE.

Bell and McCaffrey (2002) also use

<!-- formula-not-decoded -->

This transformed residual leads to CRVE that can be shown to equal the delete-one-cluster jackknife estimate of the variance of the OLS estimator. This jackknife correction leads to upwards-biased CRVE if in fact E [ 𝒖𝑔𝒖𝑔 ′ ] = 𝜎 2 𝑰 . This is a cluster generalization of the HC3 variance estimate of MacKinnon and White (1985), so we refer to it as the CR3VE.

Angrist  and  Pischke  (2009,  Chapter  8)  and  Cameron,  Gelbach  and  Miller  (2008) provide a more extensive discussion and cite more of the relevant literature. This literature includes papers that propose corrections for the more general case that E [ 𝒖𝑔𝒖𝑔 ′ ] ≠ 𝜎 2 𝑰 but has a known parameterization, such as an RE model, and extension to generalized linear models.

Angrist  and  Lavy  (2002)  apply  the  CR2VE  correction  (24)  in  an  application  with 𝐺 = 30 to 40 and find that the correction increases cluster-robust standard errors by between 10  and  50  percent.  Cameron,  Gelbach  and  Miller  (2008,  Table  3)  find  that  the  CR3VE correction (24) has rejection rates of .062, .070, .092, and .138 for 𝐺 equal to, respectively, 30, 20, 10 and 5. These rejection rates are a considerable improvement on .068, .081, .118, and .208 for the standard CRVE, but there is still considerable over-rejection for very small 𝐺 .

The literature has gravitated to using the CR2VE adjustment rather than the CR3VE adjustment. This reduces but does not eliminate over-rejection when there are few clusters.

## C. Solution 2: Cluster Bootstrap with Asymptotic Refinement

In  Subsection  II.F  we  introduced  the  bootstrap  as  it  is  usually  used,  to  calculate standard errors that rely on regular asymptotic theory. Here we consider a different use of the bootstrap, one with asymptotic refinement that may lead to improved finite-sample inference.

We consider  inference  based  on 𝐺 → ∞ (formally, √𝐺 ( 𝛽 ̂ -𝛽 ) has  a  limit  normal

distribution). Then a two-sided Wald test of nominal size 0.05 , for example, can be shown to have true size 0.05 + 𝑂 ( 𝐺 -1 ) when the usual asymptotic normal approximation is used. For 𝐺 → ∞ this equals the desired 0.05, but for finite 𝐺 this differs from 0.05 . If an appropriate bootstrap with asymptotic refinement is instead used, the true size is 0.05 + 𝑂 ( 𝐺 -3 / 2 ) . This is closer to the desired 0.05 for  large 𝐺 ,  as 𝐺 -3 / 2 &lt; 𝐺 -1 .  Hopefully  this  is  also  the  case  for small 𝐺 ,  something  that  is  established  using  appropriate  Monte  Carlo  experiments.  For  a one-sided  test  or  a  nonsymmetric  two-sided  test  the  rates  are  instead,  respectively, 0.05 + 𝑂 ( 𝐺 -1 / 2 ) and 0.05 + 𝑂 ( 𝐺 -1 ) .

Asymptotic refinement can be achieved by bootstrapping a statistic that is asymptotically pivotal, meaning the asymptotic distribution does not depend on any unknown parameters. The estimator 𝛽 ̂ is not asymptotically pivotal as its distribution depends on V [ 𝛽 ̂ ] which  in  turn  depends  on  unknown  variance  parameters  in  V [ 𝒖 | 𝑿 ] .  The  Wald  t-statistic defined in (23) is asymptotically pivotal as its asymptotic distribution is 𝑁 [0,1] which does not depend on unknown parameters.

## 1. Percentile-t Bootstrap

A  percentile-t  bootstrap  obtains 𝐵 draws, 𝑤1 ∗ , . . . , 𝑤𝐵 ∗ ,  from  the  distribution  of  the Wald t-statistic as follows. B times do the following:

1. Obtain 𝐺 clusters {( 𝒚1 ∗ , 𝑿1 ∗ ), . . . , ( 𝒚𝐺 ∗ , 𝑿𝐺 ∗ )} by  one  of  the  cluster  bootstrap methods detailed below.
2. Compute the OLS estimate 𝛽 ̂ 𝑏 ∗ in this resample.
3. Calculate the Wald  test statistic 𝑤𝑏 ∗ = ( 𝛽 ̂ 𝑏 ∗ -𝛽 ̂ )/ 𝑠 𝛽 � 𝑏 ∗ where 𝑠 𝛽 � 𝑏 ∗ is the cluster-robust standard error of 𝛽 ̂ 𝑏 ∗ , and 𝛽 ̂ is the OLS estimate of 𝛽 from the original sample.

Note  that  we  center  on 𝛽 ̂ and  not 𝛽0 ,  as  the  bootstrap  views  the  sample  as  the population, i.e., 𝛽 = 𝛽 ̂ , and the 𝐵 resamples are based on this 'population.' Note also that the standard error in step 3 needs to be cluster-robust. A good choice of 𝐵 is 𝐵 = 999 ;  this  is more than 𝐵 for standard error estimation as tests are in the tails of the distribution, and is such that ( 𝐵 +1) 𝛼 is an integer for common choices of test size 𝛼 .

The simplest cluster resampling method is the pairs cluster resampling introduced in Subsection  II.F.  Then  in  step  1.  above  we  form 𝐺 clusters {( 𝒚1 ∗ , 𝑿1 ∗ ), . . . , ( 𝒚𝐺 ∗ , 𝑿𝐺 ∗ )} by resampling with replacement 𝐺 times from the original sample of clusters. This method has the advantage of being applicable to a wide range of estimators, not just OLS. However, for some types of data the pairs cluster bootstrap may not be applicable - see 'Bootstrap with Caution' below.

Let 𝑤 ( 1 ) ∗ , . . . , 𝑤 ( 𝐵 ) ∗ denote the ordered values of 𝑤1 ∗ , . . . , 𝑤𝐵 ∗ . These ordered values trace out the density of the Wald t-statistic, taking the place of a standard normal or 𝑇 distribution. For  example,  the  critical  values  for  a  95%  nonsymmetric  confidence  interval  or  a  5% nonsymmetric Wald test are the lower 2.5 percentile and upper 97.5 percentile of 𝑤1 ∗ , . . . , 𝑤𝐵 ∗ , rather than, for example, the standard normal values of -1.96 and 1.96 .  The p-value for a symmetric test based on the original sample Wald statistic 𝑤 equals the proportion of times that | 𝑤 | &gt; | 𝑤𝑏 ∗ | , 𝑏 = 1, . . . , 𝐵 .

Cameron, Gelbach, and Miller (2008) found that in Monte Carlos with few clusters the pairs cluster bootstrap did not eliminate test over-rejection. The authors proposed using an alternative  percentile-t  bootstrap,  the  wild  cluster  bootstrap,  that  holds  the  regressors  fixed across bootstrap replications.

## 2. Wild Cluster Bootstrap

The wild cluster bootstrap resampling method is as follows. First, estimate the main model, imposing (forcing) the null hypothesis 𝐻0 that you wish to test, to give estimate 𝜷 � H0 . For  example,  for  test  of  statistical  significance  of  a  single  variable  regress 𝑦𝑖𝑔 on  all components of 𝒙𝑖𝑔 except the variable that has regressor with coefficient zero under the null hypothesis. Form the residual 𝑢 � 𝑖𝑔 = 𝑦𝑖 -𝒙𝑖𝑔 ′ 𝜷 � H0 .  Then obtain the 𝑏 𝑡ℎ resample  for  step  1 above as follows:

1a.      Randomly  assign  cluster 𝑔 the  weight 𝑑𝑔 = -1 with  probability 0.5 and  the weight 𝑑𝑔 = 1 with probability 0.5 . All observations in cluster 𝑔 get the same value of the weight.

1b.    Generate  new  pseudo-residuals 𝑢𝑖𝑔 ∗ = 𝑑𝑔 × 𝑢 � 𝑖𝑔 ,  and  hence  new  outcome variables 𝑦𝑖𝑔 ∗ = 𝒙𝑖𝑔 ′ 𝜷 � H0 + 𝑢𝑖𝑔 ∗ .

For the wild bootstrap, the values 𝑤1 ∗ , . . . , 𝑤𝐵 ∗ cannot be used directly to form critical values for a confidence interval. Instead they can be used to provide a p-value for testing a hypothesis.    To form a confidence interval, one needs to invert a sequence of tests, profiling over  a  range  of  candidate  null  hypotheses 𝐻0 : 𝛽 = 𝛽0 .  For  each  of  these  null  hypotheses, obtain the p-value. The 95% confidence interval is the set of values of 𝛽0 for which 𝑝 ≥ 0.05 . This method is computationally intensive, but conceptually straightforward.    As a practical matter, you may want to ensure that you have the same set of bootstrap draws across candidate hypotheses, so as to not introduce  additional bootstrapping noise into the determination of where the cutoff is.

Then proceed with step 2 as before, regressing 𝑦𝑖𝑔 ∗ on 𝒙𝑖𝑔 , and calculate 𝑤𝑏 ∗ as in step 3.  The  p-value  for  the  the  test  based  on  the  original  sample  Wald  statistic 𝑤 equals  the proportion of times that | 𝑤 | &gt; | 𝑤𝑏 ∗ | , 𝑏 = 1, . . . , 𝐵 .

In  principle  it  is  possible  to  directly  use  a  bootstrap  for  bias-reduction,  such  as  to remove bias in standard errors. In practice this is not done, however, as in practice any bias reduction comes at the expense of considerably greater variability. A conservative estimate of the standard error equals the width of a 95% confidence interval, obtained using asymptotic refinement, divided by 2 × 1.96 .

Note that for the wild cluster bootstrap the resamples {( 𝒚1 ∗ , 𝑿1 ), . . . , ( 𝒚𝐺 ∗ , 𝑿𝐺 )} have the same 𝑿 in  each  resample,  whereas  for  pairs  cluster  both 𝒚 ∗ and 𝑿 ∗ vary  across  the 𝐵 resamples.  The  wild  cluster  bootstrap  is  an  extension  of  the  wild  bootstrap  proposed  for heteroskedastic data. It works essentially because the two-point distribution for forming 𝒖𝑔 ∗ ensures that E [ 𝒖𝑔 ∗ ] = 𝟎 and V [ 𝒖𝑔 ∗ ] = 𝒖 �𝑔𝒖 �𝑔 ′ . There are other two-point distributions that also do so, but Davidson and Flachaire (2008) show that in the heteroskedastic case it is best to use the weights 𝑑𝑔 = { -1,1} , called Rademacher weights.

The wild cluster bootstrap essentially replaces 𝒚𝑔 in each resample with one of two values 𝒚𝑔 ∗ = 𝑿𝜷 � H0 + 𝒖 �𝑔 or 𝒚𝑔 ∗ = 𝑿𝜷 � H0 -𝒖 �𝑔 . Because this is done across 𝐺 clusters, there are at most 2 𝐺 possible combinations of the data, so there are at most 2 𝐺 unique values of 𝑤1 ∗ , . . . , 𝑤𝐵 ∗ . If there are very few clusters then there is no need to actually bootstrap as we can simply enumerate, with separate estimation for each of the 2 𝐺 possible datasets.

Webb (2013) expands on these limitations. He shows that there are actually only 2 𝐺-1 possible 𝑡 -statistics in absolute value. Thus with 𝐺 = 5 there are at most 2 4 = 16 possible values of 𝑤1 ∗ , . . . , 𝑤𝐵 ∗ . So if the main test statistic is more extreme than any of these 16 values, for  example,  then  all  we  know  is  that  the  p-value  is  smaller  than 1/16 = 0.0625 .  Full enumeration makes this discreteness clear.    Bootstrapping without consideration of this issue can lead to inadvertently picking just one point from the interval of equally plausible p-values.

As 𝐺 gets to be as large as 11 this concern is less of an issue since 2 10 = 1024 .

MacKinnon and Webb (2013) address the issue of unbalanced clusters and find that, even  with 𝐺 = 50 ,  tests  based  on  the  standard  CRVE  with 𝑇 ( 𝐺 1) critical  values  can over-reject  considerably  if  the  clusters  are  unbalanced.  By  contrast,  the  two-point  wild bootstrap with Rademacher weights is generally reliable.

Webb  (2013)  proposes  greatly  reducing  the  discreteness  of  p-values  with  very  low 𝐺 by instead  using  a  six-point  distribution  for  the  weights 𝑑𝑔 in  step  1b.  In  this  proposed distribution the weights 𝑑𝑔 have a 1/6 chance of taking each value in { -√ 1.5, -√ 1, -√ . 5, √ . 5, √ 1, √ 1.5} . In his simulations this method outperforms the two-point wild bootstrap for 𝐺 &lt; 10 and is the preferred method to use if 𝐺 &lt; 10 .

## 3. Bootstrap with Caution

Regardless  of  the  bootstrap  method  used,  pairs  cluster  with  or  without  asymptotic refinement or wild cluster bootstrap, an important step when bootstrapping with few clusters is to  examine  the  distribution  of  bootstrapped  values.  This  is  something  that  should  be  done whether you are bootstrapping 𝛽 to obtain a standard error, or bootstrapping t-statistics with refinement to obtain a more accurate p-value. This examination can take the form of looking at four things: (1) basic summary statistics like mean and variance; (2) the sample size to confirm that it is the same as the number of bootstrap replications (no missing values); (3) the largest and smallest handful of values of the distribution; and (4) a histogram of the bootstrapped values.

We detail a few potential problems that this examination can diagnose.

First, if  you are using a pairs cluster bootstrap and one cluster is an outlier in some sense, then the resulting histogram may have a big 'mass' that sits separately from the rest of the bootstrap distribution - that is, there may be two distinct distributions, one for cases where that cluster is sampled and one for cases where it is not. If this is the case then you know that your results are sensitive to the inclusion of that cluster.

Second, if you are using a pairs cluster bootstrap with dummy right-hand side variables, then in some samples you can get no or very limited variation in treatment. This can lead to zero or near-zero standard errors. For a percentile-t pairs cluster bootstrap, a zero or missing standard error will lead to missing values for 𝑤 ∗ , since the standard error is zero or missing. If you naively use the remaining distribution, then there is no reason to expect that you will have valid inference. And if the bootstrapped standard errors are zero plus machine precision noise, rather than exactly zero, very large t-values may result. Then your bootstrap distribution of t-statistics will have really fat tails, and you will not reject anything, even false null hypotheses. No variation or very limited variation in treatment can also result in many of your 𝛽 ̂ ∗ 's being 'perfect fit' 𝛽 ̂ ∗ 's with limited variability. Then the bootstrap standard deviation of the 𝛽 ̂ ∗ 's will be too small, and if you use it as your estimated standard error you will over-reject. In this case we suggest using the wild cluster bootstrap.

Third, if your pairs cluster bootstrap samples draw nearly multicollinear samples, you can get huge 𝛽 ̂ ∗ 's. This can make a bootstrapped standard error seem very large. You need to determine what in the bootstrap samples 'causes' the huge 𝛽 ̂ ∗ 's . If this is some pathological but common draw, then you may need to think about a different type of bootstrap, such as the wild cluster bootstrap, or give up on bootstrapping methods. For an extreme example, consider a DiD model, with first-order 'control' fixed effects and an interaction term. Suppose that a bootstrap sample happens to have among its 'treatment group' only observations where 'post = 1'. Then the variables 'treated' and 'treated*post' are collinear, and an OLS package will drop  one  of  these  variables.  If  it  drops  the  'post'  variable,  it  will  report  a  coefficient  on 'treated*post', but this coefficient will not be a proper interaction term, it will instead also

include the level effect for the treated group.

Fourth, with less than ten clusters the wild cluster bootstrap should use the six-point version of Webb (2013).

Fifth, in general if you see missing values in your bootstrapped t-statistics, then you need to figure out why. Don't take your bootstrap results at face value until you know what's going on.

## D. Solution 3: Improved Critical Values using a T-distribution

The simplest small-sample correction for the Wald statistic is to base inference on a 𝑇 distribution, rather than the standard normal, with degrees of freedom at most the number of clusters 𝐺 . Recent research has proposed methods that lead to using degrees of freedom much less than 𝐺 , especially if clusters are unbalanced.

## 1. G-L Degrees of Freedom

Some  packages,  including  Stata  after  command regress ,  use 𝐺 1 degrees  of freedom for 𝑡 -tests and 𝐹 tests based on cluster-robust standard errors. This choice emanates from the complex survey literature; see Bell and McCaffrey (2002) who note, however, that with  normal  errors  this  choice  still  tends  to  result  in  test  over-rejection  so  the  degrees  of freedom should be even less than this.

Even  the 𝑇 ( 𝐺 1) can  make  quite  a  difference.  For  example  with 𝐺 = 10 for  a two-sided  test  at  level 0.05 the  critical  value  for 𝑇 (9) is 2.262 rather  than 1.960 ,  and  if 𝑤 = 1.960 the p-value based on 𝑇 (9) is 0.082 rather than 0.05 . In Monte Carlo simulations by Cameron, Gelbach, and Miller (2008) this choice works reasonably well, and at a minimum one should use the 𝑇 ( 𝐺 1) distribution rather than the standard normal.

Donald and Lang extend this approach to inference on 𝜷 in a model that additionally includes regressors 𝒛𝑖𝑔 that vary within clusters, and allow for unbalanced clusters, leading to 𝐺 - 𝐿 for  the  RE  estimator.  Wooldridge  (2006)  presents  an  expansive  exposition  of  the Donald  and  Lang  approach.  He  also  proposes  an  alternative  approach  based  on  minimum distance estimation. See Wooldridge (2006) and, for a summary, Cameron and Miller (2011).

For models that include 𝐿 regressors that are invariant within cluster, Donald and Lang (2007) provide a rationale for using the 𝑇 ( 𝐺 - 𝐿 ) distribution. If clusters are balanced and all regressors are invariant within cluster then the OLS estimator in the model 𝑦𝑖𝑔 = 𝒙𝑔 ′ 𝜷 + 𝑢𝑖𝑔 is equivalent to OLS estimation in the grouped model 𝑦̄𝑔 = 𝒙𝑔 ′ 𝜷 + 𝑢̄𝑔 . If 𝑢̄𝑔 is i.i.d. normally distributed  then  the  Wald  statistic  is 𝑇 ( 𝐺 - 𝐿 ) distributed,  where V � [ 𝜷 � ] = 𝑠 2 ( 𝑿 ′ 𝑿 ) -1 and 𝑠 2 = ( 𝐺 - 𝐿 ) -1 ∑ 𝑢̄𝑔 � 2 𝑔 . Note that 𝑢̄𝑔 is i.i.d. normal in the random effects model if the error components are i.i.d. normal. Usually if there is a time-invariant regressor there is only one, in addition to the intercept, in which case 𝐿 = 2 .

## 2. Data-determined Degrees of Freedom

For  testing  the  difference  between  two  means  of  normally  and  independently distributed populations with different variances the 𝑡 test is not exactly 𝑇 distributed - this is known as the Behrens-Fisher problem. Satterthwaite (1946) proposed an approximation that was extended to regression with clustered errors by Bell and McCaffrey (2002) and developed further by Imbens and Kolesar (2012).

The 𝑇 ( 𝑁 - 𝑘 ) distribution is the ratio of 𝑁 [0,1] to independent � [ 𝜒 2 ( 𝑁-𝐾 )]/( 𝑁-𝑘 ) . For linear regression under i.i.d. normal errors, the derivation of the 𝑇 ( 𝑁-𝑘 ) distribution for the Wald 𝑡 -statistic uses the result that ( 𝑁-𝐾 )( 𝑠 𝛽 � 2 / 𝜎 𝛽 � 2 ) ∼ 𝜒 2 ( 𝑁-

𝐾 ) ,  where 𝑠 𝛽 � 2 is  the  usual unbiased estimate of 𝜎 𝛽 � 2 = V [ 𝛽 ̂ ] .  This result no longer holds for non-i.i.d.  errors,  even  if  they  are  normally  distributed.  Instead,  an  approximation  uses  the 𝑇 ( 𝑣 ∗ ) distribution where 𝑣 ∗ is such that the first two moments of 𝑣 ∗ ( 𝑠 𝛽 � 2 / 𝜎 𝛽 � 2 ) equal the first two  moments ( 𝑣 ∗ and  2 𝑣 ∗ ) of  the 𝜒 2 ( 𝑣 ∗ ) distribution.  Assuming 𝑠 𝛽 � 2 is  unbiased  for 𝜎 𝛽 � 2 , E [ 𝑣 ∗ ( 𝑠 𝛽 � 2 / 𝜎 𝛽 � 2 )] = 𝑣 ∗ . And V [ 𝑣 ∗ ( 𝑠 𝛽 � 2 / 𝜎 𝛽 � 2 )] = 2 𝑣 ∗ if 𝑣 ∗ = 2[( 𝜎 𝛽 � 2 ) 2 / V [ 𝑠 𝛽 � 2 ]] .

Thus the Wald  t-statistic is treated as being 𝑇 ( 𝑣 ∗ ) distributed where 𝑣 ∗ = 2( 𝜎 𝛽 � 2 ) 2 / 𝑉 [ 𝑠 𝛽 � 2 ] . Assumptions are needed to obtain an expression for V [ 𝑠 𝛽 � 2 ] . For clustered errors with 𝒖 ∼ 𝑁 [ 𝟎 , Ω ] and  using  the  CRVE  defined  in  Subsection  II.C,  or  using  CR2VE  or CR3VE defined in Subsection VI.B, Bell and McCaffrey (2002) show that the distribution of the Wald t-statistic defined in (23) can be approximated by the 𝑇 ( 𝑣 ∗ ) distribution where

<!-- formula-not-decoded -->

and 𝜆𝑗 are  the  eigenvalues  of  the 𝐺 × 𝐺 matrix 𝑮 ′ Ω � 𝑮 ,  where Ω � is  consistent  for Ω , the 𝑁 × 𝐺 matrix 𝑮 has 𝑔 𝑡ℎ column ( 𝑰𝑁 - 𝑯 ) 𝑔 ′ 𝑨𝑔𝑿𝑔 ( 𝑿 ′ 𝑿 ) -1 𝒆𝑘 , ( 𝑰𝑁 - 𝑯 ) 𝑔 is  the 𝑁𝑔 × 𝑁 submatrix for cluster 𝑔 of the 𝑁 × 𝑁 matrix 𝐼𝑁 - 𝑿 ( 𝑿 ′ 𝑿 ) -1 𝑿 ′ , 𝑨𝑔 = ( 𝑰𝑁𝑔 -𝑯𝑔𝑔 ) -1 / 2 for CR2VE, and 𝒆𝑘 is a 𝐾 ×1 vector of zeroes aside from 1 in the 𝑘 𝑡ℎ position if 𝛽 ̂ = 𝛽 ̂ 𝑘 . Note that 𝑣 ∗ needs  to  be  calculated  separately,  and  differs,  for  each  regression  coefficient.  The method extends to Wald tests based on scalar linear combinations 𝒄 ′ 𝜷 � .

The justification relies on normal errors and knowledge of Ω = E [ 𝒖𝒖 ′ | 𝑿 ] .  Bell and McCaffrey (2002)  perform  simulations  with  balanced  clusters ( 𝐺 = 20 and 𝑁𝑔 = 10) and equicorrelated errors within cluster. They calculate 𝑣 ∗ assuming Ω = 𝜎 2 𝑰 , even though errors are in fact clustered, and find that their method leads to Wald tests with true size closer to the nominal size than tests based on the conventional CRVE, CRV2E, and CRV3E.

Imbens and Kolesar (2012) additionally consider calculating 𝑣 ∗ where Ω � is based on equicorrelated  errors  within  cluster.  They  follow  the  Monte  Carlo  designs  of  Cameron, Gelbach and Miller (2008), with 𝐺 = 5 and 10 and equicorrelated errors. They find that all finite-sample  adjustments  perform  better  than  using  the  standard  CRVE  with 𝑇 ( 𝐺 1) critical values. The best methods use the CR2VE and 𝑇 ( 𝑣 ∗ ) , with slight over-rejection with 𝑣 ∗ based  on Ω � = 𝑠 2 𝑰 (Bell  and  McCaffrey)  and  slight  under-rejection  with 𝑣 ∗ based  on Ω � assuming equicorrelated errors (Imbens and Kolesar). For 𝐺 = 5 these methods outperform the two-point wild cluster bootstrap, as expected given the very low 𝐺 problem discussed in Subsection  VI.C.  More  surprisingly  these  methods  also  outperform  wild  cluster  bootstrap when 𝐺 = 10 ,  perhaps because Imbens and Kolesar (2012) may not have imposed the null hypothesis in forming the residuals for this bootstrap.

## 3. Effective Number of Clusters

Carter, Schnepel and Steigerwald (2013) propose a measure of the effective number of clusters. This measure is

<!-- formula-not-decoded -->

where 𝛿 = 1 𝐺 ∑ { 𝐺 𝑔=1 ( 𝛾𝑔 -𝛾̄ ) 2 / 𝛾̄ 2 } , 𝛾𝑔 = 𝒆𝑘 ′ ( 𝑿 ′ 𝑿 ) -1 𝑿𝑔 ′ Ω𝑔𝑿𝑔 ( 𝑿 ′ 𝑿 ) -1 𝒆𝑘 , 𝒆𝑘 is  a 𝐾 ×1

vector of zeroes aside from 1 in the 𝑘 𝑡ℎ position if 𝛽 ̂ = 𝛽 ̂ 𝑘 , and 𝛾̄ = 1 𝐺 ∑ 𝛾𝑔 𝐺 𝑔=1 . Note that 𝐺 ∗ varies with the regression coefficient considered, and the method extends to Wald tests based on scalar linear combinations 𝒄 ′ 𝜷 � .

The quantity 𝛿 measures cluster heterogeneity, which disappears if 𝛾𝑔 = 𝛾 for all 𝑔 . Given the formula for 𝛾𝑔 , cluster heterogeneity ( 𝛿 ≠ 0) can arise for many reasons, including variation in 𝑁𝑔 , variation in 𝑿𝑔 and variation in Ω𝑔 across clusters.

In  simulations  using  standard  normal  critical  values,  Carter  et  al.  (2013)  find  that  test  size distortion occurs for 𝐺 ∗ &lt; 20 . In application they assume errors are perfectly correlated within cluster, so Ω𝑔 = 𝒍𝒍 ′ where 𝒍 is an 𝑁𝑔 ×1 vector of ones. For data from the Tennessee STAR experiment they find 𝐺 ∗ = 192 when 𝐺 = 318 .  For  the  Hersch  (1998)  data  of  Subsection II.B, with very unbalanced clusters, they find for the industry job risk coefficient and with clustering on industry that 𝐺 ∗ = 19 when 𝐺 = 211.

Carter et al. (2013) do not actually propose using critical values based on the 𝑇 ( 𝐺 ∗ ) distribution. The key component in obtaining the formula for 𝑣 ∗ in the Bell and McCaffrey (2002)  approach  is  determining  V [ 𝑠 𝛽 � 2 / 𝜎 𝛽 � 2 ] ,  which  equals  E [( 𝑠 𝛽 � 2 -𝜎𝛽 � 2 )/ 𝜎 𝛽 � 2 ] given 𝑠 𝛽 � 2 is unbiased for 𝜎 𝛽 � 2 . Carter et al. (2013) instead work with E [( 𝑠̃ 𝛽 � 2 -𝜎𝛽 � 2 )/ 𝜎 𝛽 � 2 ] where 𝑠̃ 𝛽 � 2 , defined in their paper, is an approximation to 𝑠 𝛽 � 2 that is good for large 𝐺 (formally 𝑠̃ 𝛽 � 2 / 𝜎 𝛽 � 2 → 𝑠 𝛽 � 2 / 𝜎 𝛽 � 2 as 𝐺 → ∞ ) . Now E [( 𝑠̃ 𝛽 � 2 -𝜎𝛽 � 2 )/ 𝜎 𝛽 � 2 ] = 2(1 + 𝛿 )/ 𝐺 , see Carter et al. (2013), where 𝛿 is defined in (27). This  suggests  using  the 𝑇 ( 𝐺 ∗ ) distribution as an approximation,  and  that  this approximation will improve as 𝐺 increases.

## E. Special Cases

With few clusters, additional results can be obtained if there are many observations in each group. In DiD studies the few clusters problem arises if few groups are treated, even if 𝐺 is large. And the few clusters problem is more likely to arise if there is multi-way clustering.

## 1. Fixed Number of Clusters with Cluster Size Growing

The preceding adjustments to the degrees of freedom of the 𝑇 distribution are based on the assumption of normal errors. In some settings asymptotic results can be obtained when 𝐺 is small, provided 𝑁𝑔 → ∞ .

Bester, Conley and Hansen (2011), building on Hansen (2007a), give conditions under which the 𝑡 -test  statistic  based  on  (11)  is �𝐺 /( 𝐺 1) times 𝑇𝐺-1 distributed.  Then  using 𝒖 �𝑔 = �𝐺 /( 𝐺 1) 𝒖 �𝑔 in (11) yields a 𝑇 ( 𝐺 1) distributed statistic. In addition to assuming 𝐺 is  fixed while 𝑁𝑔 → ∞ ,  it  is  assumed that the within group correlation satisfies a mixing condition (this does not happen in all data settings, although it does for time series and spatial correlation), and that homogeneity assumptions are satisfied, including equality of plim 1 𝑁𝑔 𝑿𝑔 ′ 𝑿𝑔 for all 𝑔 .

Let 𝛽 ̂ 𝑔 denote the estimate of parameter 𝛽 in cluster 𝑔 , 𝛽 ̂ = 𝐺 -1 ∑ 𝛽 ̂ 𝑔 𝐺 𝑔=1 denote the average of the 𝐺 estimates, and 𝑠 𝛽 � 2 = ( 𝐺 1) ∑ ( 𝐺 𝑔=1 𝛽 ̂ 𝑔 -𝛽 ̂ ) 2 denote their variance. Suppose  that  the 𝛽 ̂ 𝑔 are  asymptotically  normal  as 𝑁𝑔 → ∞ with  common  mean 𝛽 ,  and consider  test  of 𝐻0 : 𝛽 = 𝛽0 based  on 𝑡 = √𝐺 ( 𝛽 ̂ 𝑔 -𝛽0 )/ 𝑠 𝛽 � .  Then  Ibragimov  and  Müller (2010)  show  that  tests  based  on  the 𝑇 ( 𝐺 1) distribution  will  be  conservative  tests  (i.e., under-reject)  for  level 𝛼 ≤ 0.083 .  This  approach  permits  correct  inference  even  with

extremely few clusters, assuming 𝑁𝑔 is large. However, the requirement that cluster estimates are asymptotically independent must be met. Thus the method is not directly applicable to a state-year DiD application when there are year fixed effects (or other regressor that varies over time but not states). In that case Ibragimov and Müller propose applying their method after aggregating subsets of states into groups in which some states are treated and some are not.

## 2. Few Treated Groups

Problems arise if most of the variation in the regressor is concentrated in just a few clusters, even when 𝐺 is sufficiently large. This occurs if the key regressor is a cluster-specific binary treatment dummy and there are few treated groups.

Conley and Taber (2011) examine a differences-in-differences (DiD) model in which there  are  few  treated  groups  and  an  increasing  number  of  control  groups.  If  there  are group-time  random  effects,  then  the  DiD  model  is  inconsistent  because  the  treated  groups random effects are not averaged away. If the random effects are normally distributed, then the model of Donald and Lang (2007) applies and inference can use a 𝑇 distribution based on the number of treated groups. If the group-time shocks are not random, then the 𝑇 distribution may be a poor approximation. Conley and Taber (2011) then propose a novel method that uses the distribution of the untreated groups to perform inference on the treatment parameter.

Abadie,  Diamond  and  Hainmueller  (2010)  propose  synthetic  control  methods  that provide a data-driven method to select the  control group in a DiD study, and that provide inference  under  random  permutations  of  assignment  to  treated  and  untreated  groups.  The methods are suitable for treatment that effects few observational units.

## 3. What if you have multi-way clustering and few clusters?

Sometimes we are worried about multi-way clustering, but one or both of the ways has few clusters.    Currently we are not aware of an ideal approach to deal with this problem.    One potential solution is to try to add sufficient control variables so as to minimize concerns about clustering in one of the ways, and then use a one-way few-clusters cluster robust approach on the  other  way.    Another  potential  solution  is  to  model  one  of  the  ways  of  clustering  in  a parametric way, such as with a common shock or an autoregressive error model.    Then you can construct a variance estimator that is a hybrid of the parametric model, and cluster robust in the remaining dimension.

## VII. Extensions

The preceding material has focused on the OLS (and FGLS) estimator and tests on a single  coefficient.  The  basic  results  generalize  to  multiple  hypothesis  tests,  instrumental variables (IV) estimation, nonlinear estimators and generalized method of moments (GMM).

These  extensions  are  incorporated  in  Stata,  though  Stata  generally  computes  test p-values and confidence intervals using standard normal and chisquared distributions, rather than 𝑇 and 𝐹 distributions.  And  for  nonlinear  models  stronger  assumptions  are  needed  to ensure that the estimator of 𝜷 retains its consistency in the presence of clustering. We provide a brief overview.

## A. Cluster-Robust F-tests

Consider Wald joint tests of several restrictions on the regression parameters. Except in the  special  case  of  linear  restrictions  and  OLS  with  i.i.d.  normal  errors,  asymptotic  theory yields only a chi-squared distributed statistic, say 𝑊 , that is 𝜒 2 ( ℎ ) distributed, where ℎ is the number of (linearly independent) restrictions.

Alternatively  we  can  use  the  related 𝐹 statistic, 𝐹 = 𝑊 / ℎ .  This  yields  the  same p-value as the chi-squared test if we treat 𝐹 as being 𝐹 ( ℎ , ∞ ) distributed. In the cluster case, a finite-sample adjustment instead treats 𝐹 as being 𝐹 ( ℎ , 𝐺 1) distributed. This is analogous to using the 𝑇 ( 𝐺 1) distribution, rather than 𝑁 [0,1] , for a test on a single coefficient.

Thus Stata does no finite-cluster correction for tests and confidence intervals following instrumental variables estimation commands, nonlinear model estimation commands, or even after  command regress in  the  case  of  tests  and  confidence  intervals  using  commands testnl and nlcom .  The  discussion  in  Section  VI  was  limited  to  inference  after  OLS regression, but it seems reasonable to believe that for other estimators one should also base inference on the 𝑇 ( 𝐺 1) and 𝐹 ( ℎ , 𝐺 1) distributions, and even then tests may over-reject when there are few clusters.

In Stata, the finite-sample adjustment of using the 𝑇 ( 𝐺 1) for  a  t-test  on  a  single coefficient, and using the 𝐹 ( ℎ , 𝐺 1) for an F-test, is only done after OLS regression with command regress . Otherwise Stata reports critical values and p-values based on the 𝑁 [0,1] and 𝜒 2 ( ℎ ) distributions.

Some of the few-cluster methods of Section VI can be extended to tests of more than one restriction following OLS regression. The Wald test can be based on the bias-adjusted variance  matrices  CR2VE  or  CR3VE,  rather  than  CRVE.  For  a  bootstrap  with  asymptotic refinement  of  a  Wald  test  of 𝐻0 : 𝑹𝜷 = 𝒓 , in  the 𝑏 𝑡ℎ resample  we  compute 𝑊𝑏 ∗ = ( 𝑹𝜷 � 𝑏 ∗ -𝑹𝜷 � ) ′ [ 𝑹 V � clu [ 𝜷 � 𝑏 ∗ ] 𝑹 ′ ] -1 ( 𝑹𝜷 � 𝑏 ∗ -𝑹𝜷 � ) .  Extension  of  the  data-determined  degrees  of  freedom method  of  Subsection  VI.D  to  tests  of  more  than  one  restriction  requires,  at  a  minimum, extension of Theorem 4 of Bell and McCaffrey (2002) from the case that covers 𝛽 , where 𝛽 is a single component of 𝜷 , to 𝑹𝜷 . An alternative ad hoc approach would be to use the 𝐹�ℎ , 𝑣 ∗ � distribution where 𝑣 ∗ is an average (possibly weighted by estimator precision) of 𝑣 ∗ defined in (26) computed separately for each exclusion restriction.

For the estimators discussed in the remainder of Section VII, the rank of V � clu [ 𝜷 � ] is again  the  minimum  of 𝐺 1 and  the  number  of  parameters ( 𝐾 ) .  This  means  that  at  most 𝐺 1 restrictions can be tested using a Wald test, in addition to the usual requirement that ℎ ≤ 𝐾 .

## B. Instrumental Variables Estimators

The cluster-robust variance matrix estimate for the OLS estimator extends naturally to the IV estimator, the two-stage least squares (2SLS) estimator and the linear GMM estimator.

The following additional adjustments must be made when errors are clustered. First, a modified version  of  the  Hausman  test  of  endogeneity  needs  to  be  used.  Second,  the  usual inference  methods  when  instruments  are weak  need  to  be adjusted. Third, tests of over-identifying restrictions after GMM need to be based on an optimal weighting matrix that controls for cluster correlation of the errors.

## 1. IV and 2SLS

In  matrix  notation,  the  OLS  estimator  in  the  model 𝒚 = 𝑿𝜷 + 𝒖 is  inconsistent  if E [ 𝒖 | 𝑿 ] ≠ 𝟎 .  We  assume  existence  of  a  set  of  instruments 𝒁 that  satisfy  E [ 𝒖 | 𝒁 ] = 𝟎 and satisfy other conditions, notably 𝒁 is of full rank with dim [ 𝒁 ] ≥ dim [ 𝑿 ] and Cor [ 𝒁 , 𝑿 ] ≠ 𝟎 .

For the clustered case the assumption that errors in different clusters are uncorrelated is now one  of  uncorrelated  errors  conditional  on  the  instruments 𝒁 ,  rather  than  uncorrelated errors conditional on the regressors 𝑿 . In the 𝑔 𝑡ℎ cluster the matrix of instruments 𝒁𝑔 is an 𝑁𝑔 × 𝑀 matrix, where 𝑀 ≥ 𝐾 , and we assume that E [ 𝒖𝑔 | 𝒁𝑔 ] = 𝟎 and Cov [ 𝒖𝑔𝒖ℎ ′ | 𝒁𝑔 , 𝒁ℎ ] =

𝟎 for 𝑔 ≠ ℎ

.

In the just-identified case, with 𝒁 and 𝑿 having the same dimension, the IV estimator is 𝜷 � IV = ( 𝒁 ′ 𝑿 ) -1 𝒁 ′ 𝒚 , and the cluster-robust variance matrix estimate is

<!-- formula-not-decoded -->

where 𝒖 �𝑔 = 𝒚𝑔 - 𝑿𝑔𝜷 � IV are residuals calculated using the consistent IV estimator. We again assume 𝐺 → ∞ . As for OLS, the CRVE may be rank-deficient with rank the minimum of 𝐾 and 𝐺 1 .

In  the  over-identified  case  with 𝒁 having  dimension  greater  than 𝑿 ,  the  2SLS estimator is the special case of the linear GMM estimator in (29) below with 𝑾 = ( 𝒁 ′ 𝒁 ) -1 , and the CRVE is that in (30) below with 𝑾 = ( 𝒁 ′ 𝒁 ) -1 and 𝒖 �𝑔 the  2SLS residuals. In the just-identified case 2SLS is equivalent to IV.

A  test  for  endogeneity  of  a  regressor(s)  can  be  conducted  by  comparing  the  OLS estimator to the 2SLS (or IV) estimator that controls for this endogeneity. The two estimators have  the  same  probability  limit  given  exogeneity  and  different  probability  limits  given endogeneity. This is a classic setting for the Hausman test but, as in the Hausman test for fixed effects discussed in Subsection III.D, the standard version of the Hausman test cannot be used when errors are clustered. Instead partition = [ 𝑿1 𝑿2 ] , where 𝑿1 is potentially endogenous and 𝑿2 is exogenous, and let 𝒗 � 𝑖𝑔 denote the residuals from first-stage OLS regression of the endogenous regressors on instruments and exogenous regressors. Then estimate by OLS the model

<!-- formula-not-decoded -->

The regressors 𝒙1 are considered endogenous if we reject 𝐻0 : 𝜸 = 𝟎 using a Wald test based on a  CRVE.  In  Stata  this  is  implemented  using  command estat  endogenous . (Alternatively a pairs cluster bootstrap can be used to estimate the variance of 𝜷 � 2SLS -𝜷 � OLS ).

## 2. Weak Instruments

When  endogenous  regressor(s)  are weakly  correlated with instrument(s), after partialling out the exogenous regressors in the model, there is great loss of precision. Then the standard error for the coefficient of the endogenous regressor is much higher after IV or 2SLS estimation than after OLS estimation.

Additionally, asymptotic theory takes an unusually long-time to kick in so that even with large samples the IV estimator can still have considerable bias and the Wald statistic is still  not  close  to  normally  distributed.  See,  for  example,  Bound,  Jaeger,  and  Baker  (1995), Andrews and Stock (2007), and textbook discussions in Cameron and Trivedi (2005, 2009).

For this second consequence, called the 'weak instrument' problem, the econometrics literature has focused on providing theory and guidance in the case of homoskedastic errors. Not all of the proposed methods extend to errors that are correlated within cluster. And the problem may even be greater in the  clustered  case,  as  the  asymptotics  are  then  in 𝐺 → ∞ rather than 𝑁 → ∞ , though we are unaware of evidence on this.

We  begin  with  case  of  a  single  endogenous  regressor.  A  standard  diagnostic  for detecting weak instruments is to estimate by OLS the first-stage regression of the endogenous regressor  on  the  exogenous  regressors  and  the  additional  instrument(s).  Then  calculate  the F-statistic for the joint significance of the instruments; in the case of a just-identified model

there  is  only  one  instrument  to  test  so  the  F-statistic  is  the  square  of  the  t-statistic.  With clustered errors, this F-statistic needs to be based on a cluster-robust variance matrix estimate. It is common practice to interpret the cluster-robust F-statistic in the same way as when errors are  i.i.d.,  using  the  tables  of  Stock  and  Yogo  (2005)  or  the  popular  rule-of-thumb,  due  to Staiger  and  Stock  (1997),  that  there  may  be  a  weak  instrument  problem  if 𝐹 &lt; 10 .  But  it should be noted that these diagnostics for weak instruments were developed for the simpler case of i.i.d. errors. Note also that the first stage cluster-robust F-statistic can only be computed if the number of instruments is less than the number of clusters.

With more than one endogenous variable and i.i.d. errors the F-statistic generalizes to the Cragg-Donald minimum eigenvalue statistic, and one can again use the tables of Stock and Yogo (2005). For clustered errors generalizations of the Cragg-Donald minimum eigenvalue statistic have been proposed, see Kleibergen and Paap (2008), but it is again not clear whether these statistics can be compared to the Stock and Yogo tables when errors are clustered.

Now consider statistical  inference  that  is  valid  even  if  instruments  are  weak,  again beginning with the case of a single endogenous regressor. Among the several testing methods that have been proposed given i.i.d. errors, the Anderson-Rubin method can be generalized to the setting of clustered errors. Consider the model 𝑦𝑖𝑔 = 𝛽𝑥𝑖𝑔 + 𝑢𝑖𝑔 , where the regressor x is endogenous  and  the  first-stage  equation  is 𝑥𝑖𝑔 = 𝒛𝑖𝑔 ′ 𝝅 + 𝑣𝑖𝑔 . (If there  are  additional exogenous regressors 𝒙2 , as is usually the case, the method still works if the variables 𝑦 , 𝑥 and 𝒛 are  defined  after  partialling  out 𝒙2 .)  The  two  equations  imply  that 𝑦𝑖𝑔 -𝛽 ∗ 𝑥𝑖𝑔 = 𝒛𝑖𝑔 ′ 𝝅 ( 𝛽 - 𝛽 ∗ ) + 𝑤𝑖𝑔 , where 𝑤𝑖𝑔 = 𝑢𝑖𝑔 + 𝑣𝑖𝑔 ( 𝛽 - 𝛽 ∗ ). So a test of 𝛽 = 𝛽 ∗ is equivalent to a Wald test of 𝜸 = 𝟎 in the model 𝑦𝑖𝑔 -𝛽 ∗ 𝑥𝑖𝑔 = 𝒛𝑖𝑔 ′ 𝜸 + 𝑤𝑖𝑔 . With clustered errors the test is based on cluster-robust standard errors.

Additionally, a weak instrument 95% confidence interval for 𝛽 can be constructed by regressing 𝑦𝑖𝑔 -𝛽 ∗ 𝑥𝑖𝑔 on 𝒛𝑖𝑔 for  a  range  of  values  of 𝛽 ∗ and  including in the confidence interval for 𝛽 only those values of 𝛽 ∗ for which we did not reject 𝐻0 : 𝜸 = 𝟎 when testing at 5%. As in the i.i.d. case, this can yield confidence intervals that are unbounded or empty, and the method loses power when the model is overidentified.

When there is more than one endogenous regressor this method can also be used, but it can only perform a joint F-test on the coefficients of all endogenous regressors rather than separate tests for each of the endogenous regressors.

Chernozhukov and Hansen (2008) provide a simple presentation of the method and an empirical  example.  Finlay  and  Magnusson  (2009)  provide  this  and  other  extensions,  and provide a command ivtest for Stata. We speculate that if additionally there are few clusters, then some of the adjustments discussed in Section VI would help.

Baum, Schaffer and Stillman (2007) provide a comprehensive discussion of various methods for IV, 2SLS, limited information maximum likelihood (LIML), k-class, continuous updating and GMM estimation in linear models, and present methods using their ivreg2 Stata command.  They  include weak instruments methods for errors that are i.i.d., heteroskedastic or within-cluster correlated errors.

## 3. Linear GMM

For over-identified models the linear GMM estimator is more efficient than the 2SLS estimator if E [ 𝒖𝒖 ′ | 𝒁 ] ≠ 𝜎 2 𝑰 . Then

<!-- formula-not-decoded -->

where 𝑾 is a full rank 𝐾 × 𝐾 weighting matrix. The CRVE for GMM is

<!-- formula-not-decoded -->

where 𝒖 �𝑔 are residuals calculated using the GMM estimator.

For clustered errors, the efficient two-step GMM estimator uses 𝑾 = ( ∑ 𝒁𝑔 ′ 𝒖 �𝑔𝒖 �𝑔 ′ 𝒁𝑔 ) -1 𝐺 𝑔=1 ,  where 𝒖 �𝑔 are  2SLS residuals.  Implementation of  this estimator requires  that  the  number  of  clusters  exceeds  the  number  of  instruments,  since  otherwise ∑ 𝒁𝑔 ′ 𝒖 �𝑔𝒖 �𝑔 ′ 𝒁𝑔 𝐺 𝑔=1 is  not  invertible.  Here 𝒁 contains  both  the  exogenous  regressors  in  the structural  equation  and  the  additional  instruments  required  to  enable  identification  of  the endogenous regressors. When this condition is not met, Baum, Schaffer and Stillman (2007) propose doing two-step GMM after first partialling out the instruments 𝒛 from the dependent variable 𝑦 , the endogenous variables in the initial model 𝑦𝑖𝑔 = 𝒙𝑖𝑔 ′ 𝜷 + 𝑢𝑖𝑔 , and any additional instruments that are not also exogenous regressors in this model.

The over-identifying restrictions (OIR) test, also called a Hansen test or a Sargan test, is a limited test of instrument validity that can be used when there are more instruments than necessary. When errors are clustered the OIR tests must be computed following the cluster version of two-step GMM estimation; see Hoxby and Paserman (1998).

Just as GLS is more efficient than OLS, specifying a model for Ω = E [ 𝒖𝒖 ′ | 𝒁 ] can lead to more efficient estimation than GMM. Given a model for Ω , and conditional moment condition  E [ 𝒖 | 𝒁 ] = 𝟎 ,  a  more  efficient  estimator  is  based  on  the  unconditional  moment condition E [ 𝒁 ′ Ω -1 𝒖 ] = 𝟎 . Then we minimize ( 𝒁 ′ Ω � -1 𝒖 ) ′ ( 𝒁 ′ Ω � -1 𝒁 ) -1 ( 𝒁 ′ Ω � -1 𝒖 ) , where Ω � is consistent  for Ω .  Furthermore  the  CRVE  can  be  robustified  against  misspecification  of Ω , similar to the case of FGLS. In practice such FGLS-type improvements to GMM are seldom used, even in simpler settings that the clustered setting. An exception is Shore-Sheppard (1996) who considers the impact of equicorrelated instruments and group-specific shocks in a model similar  to  that  of  Moulton.  One  reason  may  be  that  this  option  is  not  provided  in  Stata command ivregress .  In  the  special  case  of  a  random  effects  model  for Ω ,  command xtivreg can be used along with a pairs cluster bootstrap used to guard against misspecification of Ω .

## C. Nonlinear Models

For nonlinear models there are several ways to handle clustering. We provide a brief summary; see Cameron and Miller (2011) for further details.

For concreteness we focus on logit regression. Recall that in the cross-section case 𝑦𝑖 takes  value 0 or 1 and  the  logit  model  specifies  that  E [ 𝑦𝑖 | 𝒙𝑖 ] = Pr [ 𝑦𝑖 = 1| 𝒙𝑖 ] = Λ ( 𝒙𝑖 ′ 𝜷 ) , where Λ ( 𝑧 ) = 𝑒 𝑧 /(1 + 𝑒 𝑧 ).

## 1. Different Models for Clustering

The  simplest  approach  is  a  pooled  approach  that  assumes  that  clustering  does  not change the functional form for the conditional probability of a single observation. Thus, for the logit model, whatever the nature of clustering, it is assumed that

<!-- formula-not-decoded -->

This is called a population-averaged approach, as Λ ( 𝒙𝑖𝑔 ′ 𝜷 ) is obtained after averaging out any

within-cluster correlation. Inference needs to control for within-cluster correlation, however, and more efficient estimation may be possible.

The generalized estimating equations (GEE) approach, due to Liang and Zeger (1986) and  widely  used  in  biostatistics,  introduces  within-cluster  correlation  into  the  class  of generalized linear models (GLM), a class that includes the logit model. One possible model for within-cluster  correlation  is  equicorrelation,  with  Cor [ 𝑦𝑖𝑔 , 𝑦𝑗𝑔 | 𝒙𝑖𝑔 , 𝒙𝑗𝑔 ] = 𝜌 .  The  Stata command xtgee  y  x,  family(binomial)  link(logit)  corr(exchangeable) estimates  the  population-averaged  logit  model  and  provides  the  CRVE  assuming  the equicorrelation model  for  within-cluster correlation is correctly specified.  The  option vce(robust) provides  a  CRVE  that  is  robust  to  misspecification  of  the  model  for within-cluster correlation. Command xtgee includes a range of models for the within-error correlation.  The  method  is  a  nonlinear  analog  of  FGLS  given  in  Subsection  II.D,  and asymptotic theory requires 𝐺 → ∞ .

A further extension is nonlinear GMM. For example, with endogenous regressors and instruments 𝒛 that satisfy  E [ 𝑦𝑖𝑔 exp ( 𝒙𝑖𝑔 ′ 𝜷 )| 𝒛𝑖𝑔 ] = 0 ,  a  nonlinear  GMM  estimator minimizes 𝒉 ( 𝜷 ) ′ 𝑾𝒉 ( 𝜷 ) where 𝒉 ( 𝜷 ) = ∑ ∑ 𝒛𝑖𝑔 𝑖 𝑔 ( 𝑦𝑖𝑔 exp ( 𝒙𝑖𝑔 ′ 𝜷 )) .  Other  choices  of 𝒉 ( 𝜷 ) that allow for intracluster correlation may lead to more efficient estimation, analogous to the linear GMM example discussed at the end of Subsection VII.B. Given a choice of 𝒉 ( 𝜷 ) , the two-step nonlinear GMM estimator at the second step uses weighting matrix 𝑾 that is the inverse of a consistent estimator of V [ 𝒉 ( 𝜷 )] , and one can then use the minimized objection function for an overidentifying restrictions test.

Now suppose  we  consider  a  random  effects  logit  model  with  normally  distributed random effect, so

<!-- formula-not-decoded -->

where 𝛼𝑔 ∼ 𝑁 [0, 𝜎𝛼 2 ] . If 𝛼𝑔 is known, the 𝑁𝑔 observations in cluster 𝑔 are independent with joint density

<!-- formula-not-decoded -->

Since 𝛼𝑔 is unknown it is integrated out, leading to joint density

<!-- formula-not-decoded -->

where ℎ ( 𝛼𝑔 | 𝜎𝛼 2 ) is the 𝑁 [0, 𝜎𝛼 2 ] density. There is no closed form solution for this integral, but it is only one-dimensional so numerical approximation (such as Gaussian quadrature) can be used. The consequent MLE can be obtained in Stata using the command xtlogit  y  x,  re . Note  that  in  this  RE  logit  model  (31)  no  longer  holds,  so 𝜷 in  the  model  (32)  is  scaled differently from 𝜷 in (31). Furthermore 𝜷 in (32) is inconsistent if the distribution for 𝛼𝑔 is misspecified,  so  there is little  point  in  using  option vce(robust) after  command xtlogit, re .

It is important to realize that in nonlinear models such as logit, the population-averaged and random effects approaches lead to quite different estimates of 𝜷 that are not comparable since 𝜷 means  different  things  in  the  different  models.  The  resulting  estimated  average marginal effects may be similar, however, just as they are in standard cross-section logit and

probit models.

With few clusters, Wald statistics are likely to over-reject as in the linear case, even if we scale the CRVE's given in this section by 𝐺 /( 𝐺 1) as is typically done; see (12) for the linear  case.  McCaffrey,  Bell,  and  Botts  (2001)  consider  bias-correction  of  the  CRVE  in generalized  linear  models.  Asymptotic  refinement  using  a  pairs  cluster  bootstrap  as  in Subsection VI.C is possible. The wild bootstrap given in Subsection VI.D is no longer possible in a nonlinear model, aside from nonlinear least squares, since it requires additively separable errors. Instead one can use the score wild bootstrap proposed by Klein and Santos (2012) for nonlinear models, including maximum likelihood and GMM models. The idea in their paper is to estimate the model once, generate scores for all observations, and then perform a bootstrap in  the  wild-cluster  style,  perturbing  the  scores  by  bootstrap  weights  at  each  step.  For  each bootstrap replication the perturbed scores are used to build a test statistic, and the resulting distribution of this test statistic can be used for inference. They find that this method performs well in small samples, and can greatly ease computational burden because the nonlinear model need only be estimated once. The conservative test of Ibragimov and Müller (2010) can be used if 𝑁𝑔 → ∞ .

## 2. Fixed Effects

A  cluster-specific  fixed  effects  version  of  the  logit  model  treats  the  unobserved parameter 𝛼𝑔 in  (32)  as  being  correlated  with  the  regressors 𝒙𝑖𝑔 .  In  that  case  both  the population-averaged and random effects logit estimators are inconsistent for 𝜷 .

Instead  we  need  a  fixed  effects  logit  estimator.  In  general  there  is  an  incidental parameters  problem  if  asymptotics  are  that 𝑁𝑔 is  fixed  while 𝐺 → ∞ ,  as  there  only 𝑁𝑔 observations  for  each 𝛼𝑔 ,  and  inconsistent  estimation  of 𝛼𝑔 spills  over  to  inconsistent estimation  of 𝜷 . Remarkably  for  the  logit  model  it  is  nonetheless  possible  to  consistently estimate 𝜷 . The logit fixed effects estimator is obtained in Stata using the command xtlogit y  x,  fe . Note, however, that the marginal effect in model  (32) is 𝜕 Pr [ 𝑦𝑖𝑔 = 1| 𝛼𝑔 , 𝒙𝑖𝑔 ]/ 𝜕𝑥𝑖𝑗𝑘 = Λ ( 𝛼𝑔 + 𝒙𝑖𝑔 ′ 𝜷 )(1 -Λ ( 𝛼𝑔 + 𝒙𝑖𝑔 ′ 𝜷 )) 𝛽𝑘 .  Unlike  the  linear  FE  model  this depends on the unknown 𝛼𝑔 . So the marginal effects cannot be computed, though the ratio of the  marginal  effects  of  the 𝑘 𝑡ℎ and 𝑙 𝑡ℎ regressor  equals 𝛽𝑘 / 𝛽𝑙 which  can  be  consistently estimated.

The logit model is one of few nonlinear models for which fixed effects estimation is possible when 𝑁𝑔 is  small.  The  other  models  are  Poisson  with  E [ 𝑦𝑖𝑔 | 𝑿𝑔 , 𝛼𝑔 ] = exp ( 𝛼𝑔 + 𝒙𝑖𝑔 ′ 𝜷 ) , and nonlinear models with E [ 𝑦𝑖𝑔 | 𝑿𝑔 , 𝛼𝑔 ] = 𝛼𝑔 + 𝑚 ( 𝒙𝑖𝑔 ′ 𝜷 ) , where 𝑚 ( ⋅ ) is a specified function.

The natural approach to introduce cluster-specific effects in a nonlinear model is to include  a  full  set  of  cluster  dummies  as  additional  regressors.  This  leads  to  inconsistent estimation of 𝜷 in  all  models except the linear model (estimated by OLS) and the Poisson regression model, unless 𝑁𝑔 → ∞ . There is a growing literature on bias-corrected estimation in such cases; see, for example, Fernández-Val (2009). This paper also cites several simulation studies that gauge the extent of bias of dummy variable estimators for moderate 𝑁𝑔 , such as 𝑁𝑔 = 20 .

Yoon and Galvao (2013) consider fixed effects in panel quantile regression models with correlation within cluster and provide methods under the assumption that both the number of individuals and the number of time periods go to infinity.

## D. Cluster-randomized Experiments

Increasingly  researchers  are  gathering  their  own  data,  often  in  the  form  of  field  or laboratory experiments. When analyzing data from these experiments they will want to account for the clustered nature of the data. And so when designing these experiments, they should also account for clustering. Fitzsimons, Malde, Mesnard, and Vera-Hernández (2012) use a wild cluster bootstrap in an experiment with 12 treated and 12 control clusters.

Traditional guidance for computing power analyses and minimum detectable effects (see  e.g.  Duflo,  Glennerster  and  Kremer,  2007,  pp.  3918-3922,  and  Hemming  and  Marsh (2013))  are  based  on  assumptions  of  either  independent  errors  or,  in  a  clustered  setting,  a random effects common-shock model. Ideally one would account for more general forms of clustering in these calculations (the types of clustering that motivate cluster-robust variance estimation), but this can be difficult to do ex ante. If you have a data set that is similar to the one you will be analyzing later, then you can assign a placebo treatment, and compute the ratio of cluster-robust standard errors to default standard errors. This can provide a sense of how to adjust the traditional measures used in design of experiments.

## VIII. Empirical Example

In this section we illustrate the most common applications of cluster-robust inference. There are two examples. The first is a Moulton-type setting that uses individual-level cross section data with clustering on state. The second is the Bertrand et al. (2004) example of DiD in a state-year panel with clustering on state and potentially with state fixed effects.

The micro data are from the March CPS, downloaded from IPUMS-CPS (King et al., 2010). We use data covering individuals who worked 40 or more weeks during the prior year, and whose usual hours per week in that year was 30 or more. The hourly wage is constructed as annual  earnings  divided  by  annual  hours  (usual  hours  per  week  times  number  of  weeks worked), deflated to real 1999 dollars, and observations with real wage in the range ($2, $100) are kept.

The cross-section example uses individual-level data for 2012. The panel example uses data aggregated to the state-year level for 1977 to 2012. In both cases we estimate log-wage regressions  and  perform  inference  on  a  generated  regressor  that  has  zero  coefficient. Specifically, we test 𝐻0 : 𝛽 = 0 using 𝑤 = 𝛽 ̂ / 𝑠 𝛽 � . For each example we present results for a single data set, before presenting a Monte Carlo experiment that focuses on inference when there are few clusters.

We contrast various ways to compute standard errors and perform Wald tests. Even when using a single statistical package, different ways to estimate the same model may lead to different  empirical  results  due  to  calculation  of  different  degrees  of  freedom,  especially  in models with fixed effects, and due to uses of different distributions in computing p-values and critical values. To make this dependence clear we provide the particular Stata command used to obtain  the  results  given  below;  similar  issues  arise  if  alternative  statistical  packages  are employed.

The data and accompanying Stata code (version 13) are available at our websites.

## A. Individual-level Cross-section Data: One Sample

In  our  first  application  we  use  data  on  65,685  individuals  from  the  year  2012.  The model is

<!-- formula-not-decoded -->

where 𝑦𝑖𝑔 is log-wage, 𝑑𝑔 is a randomly generated dummy 'policy' variable, equal to one for one-half of the states and zero for the other half, and 𝒛𝑖𝑔 is a set of individual-level controls (age, age squared, and education in years). Estimation is by OLS and by FGLS controlling for state-level random effects.

The policy variable 𝑑𝑔 is often referred to as a 'placebo' treatment, and should be statistically significant in only 5% of tests performed at significance level 0.05.

Table 1 reports the estimated coefficient of the policy variable, along with its standard error computed in several different ways, when there are 51 clusters (states).

OLS results given in the first column of Table 1 are obtained using Stata command regress .  The  default  standard  error  is  misleadingly  small ( 𝑠𝑒 = 0.0042) ,  leading  to  the dummy variable being very highly statistically significant ( 𝑡 = -0.0226/0.0042 = -5.42) even though it was randomly generated independently of log-wage. The White heteroskedastic-robust standard error, from regress option vce(robust) , is similar in magnitude. From Subsection IV.A White standard errors should not be used if 𝑁𝑔 is small, but here 𝑁𝑔 is large. The cluster-robust standard error ( 𝑠𝑒 = 0.0229) using option vce(cluster state) is  5.5  times  larger  and  leads  to  the  more  sensible  result  that  the regressor  is  statistically  insignificant ( 𝑡 = -0.99) .  In  results  not  presented  in  Table  1,  the cluster-robust standard errors of the other regressors - age, age squared and education - were, respectively, 1.2, 1.2 and 2.3 times the default. So ignoring clustering again understates the standard errors. As expected, a pairs cluster bootstrap (without asymptotic refinement) using option vce(boot, cluster(state)) , yields very similar cluster-robust standard error.

Note that formula (6) suggests that the cluster-robust standard errors are 4.9 times the default ( � 1 + (1 × 0.018 × (65685/51 -1) = 4.9) , close to the observed multiple of 5.5. Formula (6) may work especially well in this example as taking the natural logarithm of wage leads to model error that is close to homoskedastic and equicorrelation is a good error model for individuals clustered in regions.

FGLS estimates for a random effects model with error process defined in (16) are given in  the  second  column  of  Table  1.  These  were  obtained  using  command xtreg, re after xtset  state. The cluster-robust standard error defined in (15), and computed using option vce(robust) ,  is  0.0214/0.0199  =  1.08  times  larger  than  the  default.  The  pairs  cluster bootstrap,  implemented  using  option vce(boot) yields  a  similar  cluster-robust  standard error.

In principle FGLS can be more efficient than OLS. In this example, there is a modest gain in efficiency with the cluster-robust standard error equal to 0.0214 for FGLS compared to 0.0229 for OLS.

Finally,  to  illustrate  the  potential  pitfalls  of  pairs  cluster  bootstrapping  for  standard errors when there are few clusters, discussed in Subsection VI.C, we examine a modification with only six states broken into treated (AZ, LA, MD) and control (DE, PA, UT). For these six states,  we  estimate  a  model  similar  to  that  in  Table  1.  Then 𝛽 ̂ = 0.0373 with  default 𝑠𝑒 = 0.0128 . We then perform a pairs cluster bootstrap with 999 replications. The bootstrap 𝑠𝑒 = 0.0622 is similar to the cluster-robust 𝑠𝑒 = 0.0577. However, several problems arise. First, 28 replications cannot be estimated, presumably due to no variation in treatment in the bootstrap samples. Second, a kernel density estimate of the bootstrapped 𝛽 ̂ 𝑠 reveals that their distribution is very multi-modal and has limited density near the middle of the distribution. Considering these results, we would not feel comfortable using the pairs cluster bootstrap in this dataset with these few clusters. Better is to base inference on a wild cluster bootstrap.

This example highlights the need to use cluster-robust standard errors even when model errors are only weakly correlated within cluster (the intraclass correlation of the residuals in

this  application  is  0.018),  if  the  regressor  is  substantially  correlated  within  cluster  (here perfectly  correlated within cluster) and/or cluster sizes are large (ranging here from 519 to 5866).

## B. Individual-level Cross-section Data: Monte Carlo

We next perform a Monte Carlo exercise to investigate the performance of various cluster-robust methods as the number of clusters becomes small. The analysis is based on the same cross-section regression as in the previous subsection.

In each replication, we generate a dataset by sampling (with replacement) states and all their associated observations. For quicker computation of the Monte Carlo simulation, within each state we use only a 20% subsample of individuals, so there are on average approximately 260 observations per cluster.

We explore the effect of the number of clusters 𝐺 by performing varying simulations with 𝐺 equal to 6, 10, 20 or 50. Given a sample of states, we assign a dummy 'policy' variable to one-half of the states. We run OLS regressions of log-wage on the policy variable and the same controls as used for the Table 1 regressions.

In these simulations we perform tests of the null hypothesis that the slope coefficient of the policy variable is zero. Table 2 presents rejection rates that with millions of replications should  equal 0.05 ,  since  we  are  testing  a  true  hypothesis  at  a  nominal  5%  level.  For 𝐺 = 6 and 10 we perform 4,000 simulations, so we expect that 95% of these simulations will yield estimated test size in the range (0.043 , 0.057) if the true test size is 0.05 . For larger 𝐺 there are 1,000 simulations and the 95% simulation interval is instead (0.036 , 0.064).

We begin with lengthy discussion of the many clusters case. These results are given in the final column ( 𝐺 = 50) of Table 2. Rows 1-9 report sizes for Wald tests based on 𝑡 = 𝛽 ̂ / 𝑠𝑒 where 𝑠𝑒 is computed in various ways, while rows 10-15 report sizes for tests using various bootstraps with an asymptotic refinement. Basic Stata commands yield the standard errors in rows 1-4 and 9, while the remaining rows require additional coding.

Row 1 presents the size of tests using heteroskedastic-robust standard errors, obtained using Stata command using regress, vce(robust) .  Ignoring clustering leads to great over-rejection due to considerable under-estimation of the standard error. Using formula (6) for this 20% subsample yields a standard error inflation factor of � 1 + (1 × 0.018 × (0.20 × 65685/51 -1) = 2.38. So 𝑡 = 1.96 using the heteroskedastic-robust  standard  error  is  really 𝑡 = 1.96/2.38 = 0.82. And,  using  standard normal critical values, an apparent 𝑝 = 0.05 is  really 𝑝 = 0.41 since Pr[|z| &gt; 0.82] = 0.41. This crude approximation is fairly close to 𝑝 = 0.498 obtained in this simulation.

Results using cluster-robust standard errors, presented in rows 2-4 and obtained from regress, vce(cluster state) ,  show  that  even  with  50  clusters  the  choice  of distribution to use in obtaining critical value makes a difference. The rejection rate is closer to 0.05 when 𝑇 ( 𝐺 1) critical  values  are  used  than  when 𝑁 [0,1] critical  values  are  used. Using 𝑇 ( 𝐺 2) in row 4, suggested by the study of Donald and Lang (2007), leads to slight further improvement, but there is still over-rejection.

Results using the bias adjustments CR2 and CR3 discussed in Subsection VI.B, along with 𝑇 ( 𝐺 1) critical  values,  are  presented  in  rows  5-6.  Bias  adjustment  leads  to  further decrease in the rejection rates, towards the desired 0.05 .

Rows 7 and  8  use  critical  values  from  the T distribution  with  the  data-determined degrees-of-freedom of Subsection VI.D, equal to 17 on average when 𝐺 = 50 (see rows 14 and 17). This leads to further improvement in the Monte Carlo rejection rate.

Bootstrap standard errors obtained from a standard pairs cluster bootstrap, implemented using command regress, vce(boot, cluster(state)) are used in

row 9. For 𝐺 = 50 the rejection rate is essentially the same as that in row 3, as expected since this bootstrap has no asymptotic refinement.

Rows 10-15 implement the various percentile-t bootstraps with asymptotic refinement presented in Subsection VI.C. Only 399 bootstraps are used here as any consequent bootstrap simulation error averages out over the many Monte Carlo replications. But if these bootstraps were used just once, as in an empirical study, a percentile-t bootstrap should use at least 999 replications.  Row  10  can  be  computed  using  the bootstrap: command, see our posted code,  while  rows  11-15  require  additional  coding.  For 𝐺 = 50 the  various  bootstraps  give similar results, with rejection rates of a bit more than 0.06.

Rows 16-19 give the effective number of clusters.  The  Imbens  and  Kolesar  (2013) measure 𝑣 ∗ in (26), denoted IK, and the Carter, Schnepel and Steigerwald (2013) measure 𝐺 ∗ in  (27),  denoted  CSS, are both equal to 17 on average when 𝐺 = 50. For  the  IK  measure, across the 1,000 simulations, the 5 th percentile is 9.6 and the 95 th percentile is 29.5.

We next examine settings with fewer clusters than 𝐺 = 50. Then most methods lead to rejection rates even further away from the nominal test size of 0.05 .

Consider the case 𝐺 = 6 . Rows 2-4 and 8 compute the same Wald test statistic but use different degrees of freedom in computing p-values. This makes an enormous difference when G is small, as the critical value for a Wald test at level 0.05 rises from 2.571 to 2.776 and 3.182 for,  respectively,  the 𝑇 (5) , 𝑇 (4) and 𝑇 (3) distributions,  and  from  row  16  the  IK degrees of freedom averages 3.3 across the simulations. The CSS degrees of freedom is larger than the IK as, from Subsection VI.D, it involves an approximation that only disappears as 𝐺 becomes large.

Using a bias-corrected CRVE also makes a big difference. It appears from rows 6 and 7 that  it  is  best  to  use  the  CR3  bias-correction  with 𝑇 ( 𝐺 1) critical  values,  and  the  CR2 bias-correction  with 𝑇 ( 𝑣 ∗ ) critical  values  where 𝑣 ∗ is  the  Imbens  and  Kolesar  (2013) calculated degrees of freedom.

A downside to using cluster-robust standard errors is that they provide an estimate of the standard deviation of 𝛽 ̂ that  is  more  variable  than  the  default  or  heteroskedastic-robust standard  errors.  This  introduces  a  potential  bias  -  variance  tradeoff.  To  see  whether  this increased variability is an issue we performed 1000 Monte Carlo replications using the full cross-section micro dataset, resampling the 50 states with replacement. The standard deviation of the cluster-robust standard error across the 1,000 replications equaled 12.3% of the mean cluster-robust  standard  error,  while  the  standard  deviation  of  the  heteroskedastic-robust standard error  equaled  4.5%  of  its  mean.  So  while  the  CRVE  is  less  biased  than heteroskedastic-robust (or default),  it  is  also  more  variable.  But  the  increased  variability  is relatively small, especially compared to the very large bias that can arise if clustering is not controlled for.

Rows 10-15 present various bootstraps with asymptotic refinement. From row 10, the pairs cluster bootstrap performs extremely poorly for 𝐺 ≤ 10 .

The wild cluster bootstrap using the Webb 6 point distribution, see row 14, does not have this complication when 𝐺 = 6 . And it yields essentially the same results as those using the Rademacher 2 point distribution when 𝐺 ≥ 10.

Results for the wild cluster bootstrap using a Rademacher 2 point distribution are presented in rows 11-13. From Subsection VI.B this bootstrap yields only 2 6 = 64 possible datasets when 𝐺 = 6, and hence at most 64 unique values for 𝑤 ∗ . This leads to indeterminacy for the test p-value. Suppose the p-value is in the range [a,b]. Then 𝐻0 is rejected in row 11 if a &lt; 0.05, in row 12 if (a+b)/2 &lt; 0.05, and in row 13 if b &lt; 0.05. The indeterminacy leads to substantially different results for G as low as six , though not for 𝐺 ≥ 10.

Comparing row 15 to row 12, imposing the null hypothesis in performing the wild

bootstrap does not change the rejection rate very much in this set of simulations when 𝐺 ≥ 10 , although it appears to matter when 𝐺 = 6 . By comparison we have found a more substantial difference when simulating from the d.g.p. of Cameron et al. (2008).

In summary, the various wild cluster bootstraps lead to test size that is closer to 0.05 than  using  a  standard  Wald  test  with  cluster-robust  standard  errors  and 𝑇 ( 𝐺 1) critical values. But the test size still exceeds 0.05 and the bias adjustments in rows 6 and 7 appear to do better.

This example illustrates that at a minimum one should use a cluster-robust Wald test with 𝑇 ( 𝐺 1) critical values. Especially when there are few clusters it is better still to use bias-adjusted cluster-robust standard errors or to use a wild cluster bootstrap . In this Monte Carlo experiment, with few clusters the test size was closest to the nominal size using a Wald test with cluster-robust standard errors computed using the CR2 correction and 𝑇 ( 𝑣 ∗ ) critical values, or using the larger CR2 correction with 𝑇 ( 𝐺 1) critical values.

## C. State-Year Panel Data: One Sample

We next turn to a panel difference-in-difference application motivated by Bertrand et al. (2004). The underlying data are again individual-level data from the CPS, but now obtained for each of the years 1977 to 2012.

In applications where the policy regressor of interest is only observed at the state-year level, it is common to first aggregate the individual-level data to the state-year level before OLS regression. Several methods are used; we use the following method.

The model estimated for 51 states from 1977 to 2012 is

<!-- formula-not-decoded -->

where 𝑦 � 𝑡𝑠 is the average log-wage in year 𝑡 and state 𝑠 (after partialling out individual level covariates), 𝛼𝑠 and 𝛿𝑡 are state and year dummies, and 𝑑𝑡𝑠 is a random 'policy' variable that turns on and stays on for the last 18 years for one half of the states. Here 𝐺 = 51 , 𝑇 = 36 and 𝑁 = 1836 .

To speed up bootstraps, and to facilitate computation of the CR2 residual adjustment, we additionally partial out the state fixed effects and year fixed effects in (34) by the standard Frisch-Waugh method. We separately regress 𝑦 � 𝑡𝑠 and 𝑑𝑡𝑠 on the state dummies and the year dummies. Then 𝛽 is estimated by regressing the residuals of 𝑦 � 𝑡𝑠 on the residuals of 𝑑𝑡𝑠 , with no constant. As noted below, regression using residuals leads to slightly different standard errors due to different degrees of freedom used in calculating the CRVE.

The individual level covariates (age, age squared, and years of education) are partialled out  using  a  two-step  estimation  procedure  presented  in  Hansen  (2007b).  Define 𝐷𝑡𝑠 to  be state-by-year dummies. First we OLS regress log wage ( 𝑦𝑖𝑡𝑠 ) on state-by-year dummies 𝐷𝑡𝑠 and on the individual level covariates. And second 𝑦 � 𝑡𝑠 in equation (34) equals the estimated coefficients of the 𝐷𝑡𝑠 dummies.

Table 3 presents results for the policy dummy regressor which should have coefficient zero since it is randomly assigned.

We begin with model 1, OLS controlling for state and year fixed effects. Using default or White-robust standard errors (rows 1-2) leads to a standard error of 0.0037 that is much smaller than the cluster-robust standard error of 0.0119 (row 3), where clustering is on state. Similar standard errors are obtained using the CR2 correction (rows 4 and 5) and bootstrap without asymptotic refinement (row 6). Note that from rows 10 and 11 the IK and CSS degrees of  freedom  are  calculated  to  be,  respectively, 𝐺 1 and G ,  an  artifact  of  having  balanced cluster sizes and a single regressor that is symmetric across clusters.

The inclusion of state and year fixed effects complicates computation of the degrees of freedom ( df ) adjustment used in computing the CRVE. The row 3 column 1 results are obtained from  regression  of  residual  on  residual  without  intercept,  using  command regress, noconstant vce(cluster(state)) .  Then  from  formula  (12) 𝑑𝑓 = 𝐺 𝐺-1 × 𝐺𝑇 𝐺𝑇-1 . If instead we directly regressed log-wage on the state and year fixed effects and the K regressors, again  using regress, vce(cluster(state)) ,  then 𝑑𝑓 = 𝐺 𝐺-1 × 𝐺𝑇 𝐺𝑇-𝑇-𝐺-1 and  the cluster-robust  standard  error  equals  0.0122  rather  than  0.0119  .  And  if  instead  we  directly estimate the log-wage equation using xtreg, vce(robust) after xtset state then 𝑑𝑓 = 𝐺 𝐺-1 and the cluster-robust standard error equals 0.0120. In this example with large T and G these adjustments make little difference, but for small G or T one should use 𝑑𝑓 = 𝐺 𝐺-1 as explained in Subsection III.B.

The corresponding p-values for tests of the null hypothesis that 𝛽 = 0, following OLS regression,  are  given  in  column  4  of  Table  3.  Default  and  heteroskedastic-robust  standard errors lead to erroneously large t-statistics (of 0.0156 / 0.0037 = 4.22), so 𝑝 = 0.000 and the null hypothesis is incorrectly rejected. Using various standard errors that control for clustering (rows 3-6) leads to 𝑝 ≃ 0.20 so that the null is not rejected. Rows 7-9 report p-values from several percentile-t bootstraps that again lead to rejection of 𝐻0 : 𝛽 = 0 .

For  illustrative  purposes  we  also  compute  standard  errors  allowing  for  two-way clustering, see Section V, with clustering on both state and year. These are computed using the user-generated Stata add-on program cgmreg.ado. Clustering on year is necessary if both the regressor and the model errors are correlated across states in a given year. For this application, the result (s.e. = 0.01167) is very similar to that from clustering on state alone (s.e. = 0.01185). In some other panel applications the two-way cluster robust standard errors can be substantially larger than those from clustering on state alone.

The main lesson from the model 1 OLS results is that even after inclusion of state fixed effects one needs to use cluster-robust standard errors that cluster on state. The inclusion of state fixed effects did not eliminate the within-state correlation of the error. In this example the correct cluster-robust standard errors are 3.6 times larger than the default.

Model 2 again uses OLS estimation, but drops the state fixed effects from the model (34). Dropping these fixed effects leads to much less precise estimation as the cluster-robust standard error (row 3) increases from 0.0119 to 0.0226 and this cluster-robust standard error is now 0.0226 / 0.0062 = 3.6 times the default, comparable to a ratio of 3.6 when state fixed effects were included. Note that inclusion of state fixed effects (model 1) did soak up some of the within-state error correlation, as expected, but there still remained substantial within-cluster correlation of the error so that cluster-robust standard errors need to be used.

For  model  2  the  comparisons  of  the  various  standard  errors  and  p-values  are qualitatively similar to those for model 1, so are not discussed further.

Model 3 estimates the same model as model 1, except that the state and year fixed effects are directly estimated, and estimation is now by FGLS allowing for an AR(1) process for  the  errors.  Since  there  are  36  years  of  data  the  bias  correction  of  Hansen  (2007b),  see Subsection  III.C,  will  make  little  difference  and  is  not  used  here.  Estimation  uses  Stata command xtreg, pa corr(ar 1) after xtset state .

Comparing rows 1 and 3, again even with inclusion of state fixed effects one should obtain standard errors that cluster on state, using xtreg, pa option vce(robust)) . The difference is not as pronounced as for OLS, with FGLS cluster-robust standard error that is 0.0084/0.0062 = 1.4 times the default.

FGLS estimation has led to substantial gain in efficiency, with cluster-robust standard error (row 3) for FGLS of 0.0084 compared to 0.0119 for OLS.

This example illustrates that even with state fixed effects included in a state year panel inference  should  be  based  on  cluster-robust  standard  errors.  Furthermore  there  can  be substantial efficiency gains to estimating by FGLS rather than OLS.

## D. State-Year Panel Data: Monte Carlo

We next perform a Monte Carlo exercise to investigate the performance of various cluster-robust methods as the number of clusters becomes small. The analysis is based on the same state-year panel regression as in the previous subsection, with each state-year observation based on log-wages after partialling out the individual level covariates.

In  each  simulation, we  draw a random set of 𝐺 states  (with  replacement),  where G takes values 6, 10, 20, 20 and 50. When a state is drawn, we take all years of data for that state. We then assign our DiD 'policy' variable to half the states, with the policy turned on mid-way through the time period. In these simulations we perform tests of the null hypothesis that the slope coefficient of the policy variable is zero. As in Table 2, for 𝐺 = 6 and 10 we perform 4,000 simulations, so we expect that 95% of these simulations will yield estimated test size in the range (0.043 , 0.057) . For larger 𝐺 there are 1,000 simulations and the 95% simulation interval is instead (0.036 , 0.064).

We begin with the last column of Table 4, with 𝐺 = 50 states. All tests aside from that based on default standard errors (row 1) have rejection rates that are not appreciably different from 0.05, once we allow for simulation error.

As the number of clusters decreases it becomes clear that one should use the 𝑇 ( 𝐺 1) or 𝑇 ( 𝐺 2) distribution for critical values, and even this leads to mild over-rejection with low 𝐺 . The pairs cluster percentile-t bootstrap fails with few clusters, with rejection rate of only 0.005 when 𝐺 = 6 . For low 𝐺 , the wild cluster percentile-t bootstrap has similar results with either 2-point or 6-point weights, with very slight over-rejection.

## XI. Concluding Thoughts

It  is  important  to  aim  for  correct  statistical  inference,  many  empirical  applications feature the potential for errors to be correlated within clusters, and we need to make sure our inference accounts for this. Often this is straightforward to do using traditional cluster-robust variance  estimators  -  but  sometimes  things  can  be  tricky.  The  leading  difficulties  are  (1) determining  how  to  define  the  clusters,  and  (2)  dealing  with  few  clusters;  but  other complications can arise as well. When faced with these difficulties, there is no simple hard and fast  rule  regarding  how  to  proceed.  You  need  to  think  carefully  about  the  potential  for correlations in your model errors, and how that interacts with correlations in your covariates. In this essay we have aimed to present the current leading set of tools available to practitioners to deal with clustering issues.

## References

- Abadie, Alberto, Alexis Diamond, and Jens Hainmueller. 2010. 'Synthetic Control Methods for  Comparative  Case  Studies:  Estimating  the  Effect  of  California's  Tobacco  Control Program,' Journal of the American Statistical Association 105(490): 493-505.
- Acemoglu,  Daron,  and  Jörn-Steffen  Pischke.  2003.  'Minimum  Wages  and  On-the-job Training.' Research in Labor Economics 22: 159-202.
- Andrews, Donald W. K. and James H. Stock. 2007. 'Inference with Weak Instruments.' In Advances  in  Economics  and  Econometrics,  Theory  and  Applications:  Ninth  World Congress of the Econometric Society , Vol. III, ed. Richard Blundell, Whitney K. Newey, and T. Persson, 122-173, Cambridge: Cambridge University Press.
- Angrist, Joshua D., and Victor Lavy. 2002. 'The Effect of High School Matriculation Awards: Evidence from Randomized Trials.' American Economic Review 99 : 1384-1414.
- Angrist,  Joshua  D.,  and  Jörn-Steffen  Pischke.  2009. Mostly  Harmless  Econometrics:  An Empiricist's Companion . Princeton: Princeton University Press.
- Arellano, Manuel 1987. 'Computing Robust Standard Errors for Within-Group Estimators.' Oxford Bulletin of Economics and Statistic 49(4): 431-434.
- Barrios,  Thomas,  Rebecca  Diamond,  Guido  W.  Imbens,  and  Michal  Kolesár.  2012. 'Clustering, Spatial Correlations and Randomization Inference.' Journal of the American Statistical Association 107(498): 578-591.
- Baum,  Christopher  F.,  Mark  E.  Schaffer,  Steven  Stillman.  2007.  'Enhanced  Routines  for Instrumental Variables/GMM Estimation and Testing.' The Stata Journal 7(4): 465-506.
- Bell, Robert M., and Daniel F. McCaffrey. 2002. 'Bias Reduction in Standard Errors for Linear Regression with Multi-Stage Samples.' Survey Methodology 28(2): 169-179.
- Bertrand, Marianne, Esther Duflo, and Sendhil Mullainathan. 2004. 'How Much Should We Trust  Differences-in-Differences  Estimates?.' Quarterly  Journal  of  Economics 119(1): 249-275.
- Bester,  C.  Alan,  Timothy  G.  Conley,  and  Christian  B.  Hansen.  2011.  'Inference  with Dependent Data Using Cluster Covariance Estimators.' Journal of Econometrics 165(2): 137-151.
- Bhattacharya, Debopam. 2005. 'Asymptotic Inference from Multi-stage Samples.' Journal of Econometrics 126: 145-171.
- Bound,  John,  David  A.  Jaeger,  and  Regina  M.  Baker.  1995.  'Problems  with  Instrumental Variables Estimation when the Correlation Between the Instruments and the Endogenous Explanatory Variable is Weak.' Journal of the American Statistical Association 90(430): 443-450.
- Brewer, Mike, Thomas F. Crossley, and Robert Joyce. 2013. 'Inference with Differences-in-Differences Revisited.' Unpublished.
- Cameron, A. Colin, Jonah G. Gelbach, and Douglas L. Miller. 2006. 'Robust Inference with Multi-Way Clustering.' NBER Technical Working Paper 0327.
- ---. 2008. 'Bootstrap-Based Improvements for Inference with Clustered Errors.' Review of Economics and Statistics. 90(3): 414-427.
- ---.  2011.  'Robust  Inference  with  Multi-Way  Clustering.' Journal  Business  and Economic Statistics 29(2): 238-249.
- Cameron, A. Colin, and Douglas L. Miller. 2011. 'Robust Inference with Clustered Data.' In Handbook of Empirical Economics and Finance. ed. Aman Ullah and David E. Giles, 1-28. Boca Raton: CRC Press.
- ---.  2012.  'Robust  Inference  with  Dyadic  Data:  with  Applications  to  Country-pair International Trade.' University of California - Davis. Unpublished.

- Cameron,  A.  Colin,  and  Pravin  K.  Trivedi. 2005. Microeconometrics:  Methods  and Applications. Cambridge University Press.
- ---. 2009. Microeconometrics using Stata. College Station, TX: Stata Press.
- Carter,  Andrew  V.,  Kevin  T.  Schnepel,  and  Douglas  G.  Steigerwald.  2013.  'Asymptotic Behavior of a t Test Robust to Cluster Heterogeneity.' University of California - Santa Barbara. Unpublished.
- Cheng, Cheng, and Mark Hoekstra.    2013. 'Pitfalls in Weighted Least Squares Estimation: A Practitioner's Guide.' Texas A&amp;M University. Unpublished.
- Chernozhukov, Victor, and Christian Hansen. 2008. 'The Reduced Form: A Simple Approach to Inference with Weak Instruments.' Economics Letters 100(1): 68-71.
- Conley, Timothy G. 1999. 'GMM Estimation with Cross Sectional Dependence.' Journal of Econometrics 92(1): 1-45.
- Conley,  Timothy  G.,  and  Christopher  R.  Taber.  2011.  'Inference  with  'Difference  in Differences' with a Small Number of Policy Changes.' Review of Economics and Statistics 93(1): 113-125.
- Davidson, Russell,  and  Emmanuel Flachaire.  2008.  'The  Wild  Bootstrap,  Tamed  at  Last.' Journal of Econometrics 146(1): 162-169.
- Davis, Peter. 2002. 'Estimating Multi-Way Error Components Models with Unbalanced Data Structures.' Journal of Econometrics 106(1): 67-95.
- Donald, Stephen G., and Kevin Lang. 2007. 'Inference with Difference-in-Differences and Other Panel Data.' Review of Economics and Statistics 89(2): 221-233.
- Driscoll, John C., and Aart C. Kraay. 1998. 'Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data.' Review of Economics and Statistics 80(4): 549-560.
- Duflo,  Esther,  Rachel  Glennerster  and  Michael  Kremer.  2007.  'Using  Randomization  in Development Economics Research: A Toolkit.' In Handbook of Development Economics , Vol. 4, ed. Dani Rodrik and Mark Rosenzweig, 3895-3962. Amsterdam: North-Holland.
- Fafchamps,  Marcel,  and  Flore  Gubert.  2007.  'The  Formation  of  Risk  Sharing  Networks.' Journal of Development Economics 83(2): 326-350.
- Fernández-Val, Iván. 2009. 'Fixed Effects Estimation of Structural Parameters and Marginal Effects in Panel Probit Models.' Journal of Econometrics 150(1): 70-85.
- Finlay,  Keith,  and  Leandro  M.  Magnusson. 2009. 'Implementing Weak  Instrument Robust Tests for a General Class of Instrumental-Variables Models.' Stata Journal 9(3): 398-421.
- Fitzsimons,  Emla,  Bansi  Malde,  Alice  Mesnard,  and  Marcos  Vera-Hernández.  2012. 'Household Responses to Information on Child Nutrition: Experimental Evidence from Malawi.' IFS Working Paper W12/07.
- Foote, Christopher L. 2007. 'Space and Time in Macroeconomic Panel Data: Young Workers and State-Level Unemployment Revisited.' Working Paper 07-10, Federal Reserve Bank of Boston.
- Greenwald, Bruce C. 1983. 'A General Analysis of Bias in the Estimated Standard Errors of Least Squares Coefficients.' Journal of Econometrics 22(3): 323-338.
- Hansen, Christian. 2007a. 'Asymptotic Properties of a Robust Variance Matrix Estimator for Panel Data when T is Large.' Journal of Econometrics 141(2): 597-620.
- Hansen,  Christian.  2007b.  'Generalized  Least  Squares  Inference  in  Panel  and  Multi-level Models  with  Serial  Correlation  and  Fixed  Effects.' Journal  of  Econometrics 141(2): 597-620.
- Hausman, Jerry, and Guido Kuersteiner. 2008. 'Difference in Difference Meets Generalized Least Squares: Higher Order Properties of Hypotheses Tests.' Journal of Econometrics 144(2): 371-391.
- Hemming, Karla, and Jen Marsh. 2013. 'A Menu-driven Facility for Sample-size Calculations in Cluster Randomized Controlled Trails.' Stata Journal 13(1): 114-135.

- Hersch, Joni. 1998. 'Compensating Wage Differentials for Gender-Specific Job Injury Rates.' American Economic Review 88(3): 598-607.
- Hoechle, Daniel. 2007. 'Robust Standard Errors for Panel Regressions with Cross-sectional Dependence.' Stata Journal 7(3): 281-312.
- Hoxby,  Caroline,  and  M.  Daniele  Paserman.  1998.  'Overidentification  Tests  with  Group Data.' NBER Technical Working Paper 0223.
- Ibragimov,  Rustam,  and  Ulrich  K.  Müller.  2010.  'T-Statistic Based  Correlation  and Heterogeneity  Robust  Inference.' Journal  of  Business  and  Economic  Statistics 28(4): 453-468.
- Imbens, Guido W., and Michal Kolesar. 2012. 'Robust Standard Errors in Small Samples: Some Practical Advice.' NBER Working Paper 18478.
- Inoue, Atsushi, and Gary Solon. 2006. 'A Portmanteau Test for Serially Correlated Errors in Fixed Effects Models.' Econometric Theory 22(5): 835-851.
- Kézdi,  Gábor.  2004.  'Robust  Standard  Error  Estimation  in  Fixed-Effects  Panel  Models.' Hungarian Statistical Review Special Number 9: 95-116.
- King, Miriam, Steven Ruggles, J. Trent Alexander, Sarah Flood, Katie Genadek, Matthew B. Schroeder, Brandon Trampe, and Rebecca Vick. 2010. Integrated Public Use Microdata Series, Current Population Survey: Version 3.0. [Machine-readable database]. Minneapolis: University of Minnesota.
- Kish, Leslie. 1965. Survey Sampling. New York: John Wiley.
- Kish, Leslie,  and  Martin  R.  Frankel.  1974.  'Inference  from  Complex  Surveys  with Discussion.' Journal Royal Statistical Society B 36(1): 1-37.
- Kleibergen, F. and R. Paap. 2006. 'Generalized Reduced Rank Tests iusing the Singular Value Decomposition.' Journal of Econometrics 133(1): 97-128.
- Klein,  Patrick,  and  Andres  Santos.  2012.  'A  Score  Based  Approach  to  Wild  Bootstrap Inference.' Journal of Econometric Methods :1(1): 23-41.
- Kloek,  T.  1981.  'OLS  Estimation  in  a  Model  where  a  Microvariable  is  Explained  by Aggregates and Contemporaneous Disturbances are Equicorrelated.' Econometrica 49(1): 205-07.
- Liang, Kung-Yee, and Scott L. Zeger. 1986. 'Longitudinal Data Analysis Using Generalized Linear Models.' Biometrika 73(1): 13-22.
- MacKinnon,  James.  G.,  and  Halbert  White.  1985.  'Some  Heteroskedasticity-Consistent Covariance  Matrix  Estimators  with  Improved  Finite  Sample  Properties.' Journal  of Econometrics 29(3): 305-325.
- MacKinnon,  James,  and  Matthew  D.  Webb.  2013.  'Wild  Bootstrap  Inference  for  Wildly Different Cluster Sizes.' Queens Economics Department Working Paper No. 1314.
- McCaffrey, Daniel F., Bell, Robert M., and Carsten H. Botts. 2001. 'Generalizations of Bias Reduced Linearization.' Proceedings of the Survey Research Methods Section , American Statistical Association.
- Miglioretti, D. L., and P. J. Heagerty. 2006. 'Marginal Modeling of Nonnested Multilevel Data using Standard Software.' American Journal of Epidemiology 165(4): 453-463.
- Moulton, Brent R. 1986. 'Random Group Effects and the Precision of Regression Estimates.' Journal of Econometrics 32: 385-397.
- Moulton, Brent R. 1990. 'An Illustration of a Pitfall in Estimating the Effects of Aggregate Variables on Micro Units.' Review of Economics and Statistics 72(3): 334-38.
- Newey,  Whitney  K.,  and  Kenneth  D.  West.  1987.  'A  Simple,  Positive  Semi-Definite, Heteroscedasticity  and  Autocorrelation  Consistent  Covariance  Matrix.' Econometrica 55(3): 703-708.
- Petersen,  Mitchell  A.  2009.  'Estimating  Standard  Errors  in  Finance  Panel  Data  Sets: Comparing Approaches.' Review of Financial Studies 22(1): 435-480.

- Pfeffermann,  Daniel,  and  Gaf  Nathan.  1981.  'Regression  Analysis  of  Data  from  a  Cluster Sample.' Journal American Statistical Association 76(375): 681-689.
- Rabe-Hesketh,  Sophia,  and  Anders  Skrondal.  2012. Multilevel  and  Longitudinal  Modeling Using Stata, Volumes I and II , Third Edition. College Station, TX: Stata Press.
- Rogers, William H. 1993. 'Regression Standard Errors in Clustered Samples.' Stata Technical Bulletin 13: 19-23.
- Satterthwaite, F. E. 1946. 'An  Approximate Distribution of Estimates of Variance Components.' Biometrics Bulletin 2(6): 110-114.
- Schaffer, Mark E., and Stillman, Steven. 2010. 'xtoverid: Stata Module to Calculate Tests of Overidentifying Restrictions after xtreg, xtivreg, xtivreg2 and xthtaylor.' http://ideas.repec.org/c/boc/bocode/s456779.html
- Scott, A. J., and D. Holt. 1982. 'The Effect of Two-Stage Sampling on Ordinary Least Squares Methods.' Journal American Statistical Association 77(380): 848-854.
- Shah, Bbabubhai V., M. M. Holt and Ralph E. Folsom. 1977. 'Inference About Regression Models  from  Sample  Survey  Data.' Bulletin  of  the  International  Statistical  Institute Proceedings of the 41st Session 47(3): 43-57.
- Shore-Sheppard, L. 1996. 'The Precision of Instrumental Variables Estimates with Grouped Data.' Princeton University Industrial Relations Section Working Paper 374.
- Solon, Gary, Steven J. Haider, and Jeffrey Wooldridge. 2013. 'What Are We Weighting For?' NBER Working Paper 18859.
- Staiger, Douglas, and James H. Stock. 1997. 'Instrumental Variables Regression with Weak Instruments.' Econometrica 65: 557-586.
- Stock, James H., and Mark W. Watson. 2008. 'Heteroskedasticity-robust Standard Errors for Fixed Effects Panel Data Regression.' Econometrica 76(1): 155-174.
- Stock,  James  H.  and  M.  Yogo.  2005.  'Testing  for  Weak  Instruments  in  Linear  IV Regressions.' In I dentification and Inference for Econometric Models . Ed. Donald W. K. Andrews and James H. Stock, 80-108. Cambridge: Cambridge University Press.
- Thompson, Samuel. 2006. 'Simple Formulas for Standard Errors that Cluster by Both Firm and Time.' SSRN paper. http://ssrn.com/abstract=914002.
- Thompson, Samuel. 2011. 'Simple Formulas for Standard Errors that Cluster by Both Firm and Time.' Journal of Financial Economics 99(1): 1-10.
- Webb, Matthew D. 2013. 'Reworking Wild Bootstrap Based Inference for Clustered Errors.' Queens Economics Department Working Paper 1315.
- White, Halbert. 1980. 'A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity.' Econometrica 48(4): 817-838.
- White, Halbert. 1984. Asymptotic Theory for Econometricians. San Diego: Academic Press.
- Wooldridge, Jeffrey M. 2003. 'Cluster-Sample Methods in Applied Econometrics.' American Economic Review 93(2): 133-138.
- Wooldridge,  Jeffrey  M.  2006.  'Cluster-Sample  Methods  in  Applied  Econometrics:  An Extended Analysis.' Michigan State University. Unpublished.
- Wooldridge,  Jeffrey  M.  2010. Econometric  Analysis  of  Cross  Section  and  Panel  Data . Cambridge, MA: MIT Press.
- Yoon, Jungmo, and Antonio Galvao. 2013. 'Robust Inference for Panel Quantile Regression Models with Individual Effects and Serial Correlation.' Unpublished.

Table 1 - Cross-section individual level data Impacts of clustering and estimator choices on estimated coefficients and standard errors

|                                   | Estimation Method   | Estimation Method   |
|-----------------------------------|---------------------|---------------------|
|                                   | OLS                 | FGLS (RE)           |
| Slope coefficient                 | 0.0108              | 0.0314              |
| Standard Errors                   |                     |                     |
| Default                           | 0.0042              | 0.0199              |
| Heteroscedastic Robust            | 0.0042              | -                   |
| Cluster Robust (cluster on State) | 0.0229              | 0.0214              |
| Pairs cluster bootstrap           | 0.0224              | 0.0216              |
| Number observations               | 65685               | 65685               |
| Number clusters (states)          | 51                  | 51                  |
| Cluster size range                | 519 to 5866         | 519 to 5866         |
| Intraclass correlation            | 0.018               | -                   |

Notes:  March 2012 CPS data, from IPUMS download.  Default standard errors for OLS assume errors are iid; default standard errors for FGLS assume the Random Effects model is correctly specified.  The Bootstrap uses 399 replications.  A fixed effect model is not possible, since the regressor is invariant within states.

Table 2 - Cross-section individual level data

Monte Carlo rejection rates of true null hypothesis (slope = 0) with different number of clusters and different rejection methods

Nominal 5% rejection rates

| Wald test method                                                          | Numbers of Clusters   | Numbers of Clusters   | Numbers of Clusters   | Numbers of Clusters   | Numbers of Clusters   |
|---------------------------------------------------------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|                                                                           | 6                     | 10                    | 20                    | 30                    | 50                    |
| Different standard errors and critical values                             |                       |                       |                       |                       |                       |
| 1 White Robust, T(N-k) for critical value                                 | 0.439                 | 0.457                 | 0.471                 | 0.462                 | 0.498                 |
| 2 Cluster on state, T(N-k) for critical value                             | 0.215                 | 0.147                 | 0.104                 | 0.083                 | 0.078                 |
| 3 Cluster on state, T(G-1) for critical value                             | 0.125                 | 0.103                 | 0.082                 | 0.069                 | 0.075                 |
| 4 Cluster on state, T(G-2) for critical value                             | 0.105                 | 0.099                 | 0.076                 | 0.069                 | 0.075                 |
| 5 Cluster on state, CR2 bias correction, T(G-1) for critical value        | 0.082                 | 0.070                 | 0.062                 | 0.060                 | 0.065                 |
| 6 Cluster on state, CR3 bias correction, T(G-1) for critical value        | 0.048                 | 0.050                 | 0.050                 | 0.052                 | 0.061                 |
| 7 Cluster on state, CR2 bias correction, T(IK DOF) for critical value     | 0.052                 | 0.050                 | 0.047                 | 0.047                 | 0.054                 |
| 8 Cluster on state, T(CSS effective # clusters)                           | 0.114                 | 0.079                 | 0.057                 | 0.056                 | 0.061                 |
| 9 Pairs cluster bootstrap for standard error, T(G-1) for critical value   | 0.082                 | 0.072                 | 0.069                 | 0.067                 | 0.074                 |
| Bootstrap Percentile-T methods                                            |                       |                       |                       |                       |                       |
| 10 Pairs cluster bootstrap                                                | 0.009                 | 0.031                 | 0.046                 | 0.051                 | 0.061                 |
| 11 Wild cluster bootstrap, Rademacher 2 point distribution, low-p-value   | 0.097                 | 0.065                 | 0.062                 | 0.051                 | 0.060                 |
| 12 Wild cluster bootstrap, Rademacher 2 point distribution, mid-p-value   | 0.068                 | 0.065                 | 0.062                 | 0.051                 | 0.060                 |
| 13 Wild cluster bootstrap, Rademacher 2 point distribution, high-p-value  | 0.041                 | 0.064                 | 0.062                 | 0.051                 | 0.060                 |
| 14 Wild cluster bootstrap, Webb 6 point distribution                      | 0.079                 | 0.067                 | 0.061                 | 0.051                 | 0.061                 |
| 15 Wild cluster bootstrap, Rademacher 2 pt, do not impose null hypothesis | 0.086                 | 0.063                 | 0.050                 | 0.053                 | 0.056                 |
| 16 IK effective DOF (mean)                                                | 3.3                   | 5.6                   | 9.4                   | 12.3                  | 16.9                  |
| 17 IK effective DOF (5th percentile)                                      | 2.7                   | 3.7                   | 4.9                   | 6.3                   | 9.6                   |
| 18 IK effective DOF (95th percentile)                                     | 3.8                   | 7.2                   | 14.5                  | 20.8                  | 29.5                  |
| 19 CSS effective # clusters (mean)                                        | 4.7                   | 6.6                   | 9.9                   | 12.7                  | 17                    |
| 20 Average number of observations                                         | 1554                  | 2618                  | 5210                  | 7803                  | 13055                 |

Notes:  March 2012 CPS data, 20% sample from IPUMS download.  For 6 and 10 clusters, 4000 Monte Carlo replications.  For 20-50 clusters, 1000 Monte Carlo replications.  The Bootstraps use 399 replications.  "IK effective DOF" from Imbens and Kolesar (2013), and "CSS effective # clusters" from Carter, Schnepel and Steigerwald (2013), see Subsection VI.D.  Row 11 uses lowest p-value from interval, when Wild percentileT bootstrapped p-values are not point identified due to few clusters.  Row 12  uses mid-range of interval, and row 13 uses largest p-value of interval.

Table 3 - State-year panel data with differences-in-differences estimation

Impacts of clustering and estimation choices on estimated coefficients, standard errors, and p-values

|                                                                         | Standard Errors   | Standard Errors   | Standard Errors   | p-values   | p-values   | p-values   |
|-------------------------------------------------------------------------|-------------------|-------------------|-------------------|------------|------------|------------|
| Model:                                                                  | 1                 | 2                 | 3                 | 1          | 2          | 3          |
|                                                                         |                   | OLS-no            | FGLS              |            | OLS-no     | FGLS       |
| Estimation Method:                                                      | OLS-FE            | FE                | AR(1)             | OLS-FE     | FE         | AR(1)      |
| Slope coefficient                                                       | 0.0156            | 0.0040            | -0.0042           |            |            |            |
| Standard Errors                                                         |                   |                   |                   |            |            |            |
| 1 Default standard errors, T(N-k) for critical value                    | 0.0037            | 0.0062            | 0.0062            | 0.000      | 0.521      | 0.494      |
| 2 White Robust, T(N-k) for critical value                               | 0.0037            | 0.0055            | na                | 0.000      | 0.470      | na         |
| 3 Cluster on state, T(G-1) for critical value                           | 0.0119            | 0.0226            | 0.0084            | 0.195      | 0.861      | 0.617      |
| 4 Cluster on state, CR2 bias correction, T(G-1) for critical value      | 0.0118            | 0.0226            | na                | 0.195      | 0.861      | na         |
| 5 Cluster on state, CR2 bias correction, T(IK DOF) for critical value   | 0.0118            | 0.0226            | na                | 0.195      | 0.861      | na         |
| 6 Pairs cluster bootstrap for standard error, T(G-1) for critical value | 0.0118            | 0.0221            | 0.0086            | 0.191      | 0.857      | 0.624      |
| Bootstrap Percentile-T methods                                          |                   |                   |                   |            |            |            |
| 7 Pairs cluster bootstrap                                               | na                | na                |                   | 0.162      | 0.878      |            |
| 8 Wild cluster bootstrap, Rademacher 2 point distribution               | na                | na                |                   | 0.742      | 0.968      |            |
| 9 Wild cluster bootstrap, Webb 6 point distribution                     | na                | na                |                   | 0.722      | 0.942      |            |
| 10 Imbens-Kolesar effective DOF                                         | 50                | 50                |                   |            |            |            |
| 11 C-S-S effective # clusters                                           | 51                | 51                |                   |            |            |            |
| Number observations                                                     | 1836              | 1836              | 1836              |            |            |            |
| Number clusters (states)                                                | 51                | 51                | 51                |            |            |            |

Notes:  March 1997-2012 CPS data, from IPUMS download.  Models 1 and 3 include state and year fixed effects, and a "fake policy" dummy variable that turns on in 1995 for a random subset of half of the states.  Model 2 includes year fixed effects but not state fixed effects.  The Bootstraps use 999 replications.    Model 3 uses FGLS, assuming an AR(1) error within each state.   "IK effective DOF" from Imbens and Kolesar (2013), and  "CSS effective # clusters" from Carter, Schnepel and Steigerwald (2013), see Subsection VI.D.

Table 4 - State-year panel data with differences-in-differences estimation Monte Carlo rejection rates of true null hypothesis (slope = 0) with different # clusters and different rejection methods Nominal 5% rejection rates

| Estimation Method                                                       | Numbers of Clusters   | Numbers of Clusters   | Numbers of Clusters   | Numbers of Clusters   | Numbers of Clusters   |
|-------------------------------------------------------------------------|-----------------------|-----------------------|-----------------------|-----------------------|-----------------------|
|                                                                         | 6                     | 10                    | 20                    | 30                    | 50                    |
| Wald Tests                                                              |                       |                       |                       |                       |                       |
| 1 Default standard errors, T(N-k) for critical value                    | 0.589                 | 0.570                 | 0.545                 | 0.526                 | 0.556                 |
| 2 Cluster on state, T(N-k) for critical value                           | 0.149                 | 0.098                 | 0.065                 | 0.044                 | 0.061                 |
| 3 Cluster on state, T(G-1) for critical value                           | 0.075                 | 0.066                 | 0.052                 | 0.039                 | 0.058                 |
| 4 Cluster on state, T(G-2) for critical value                           | 0.059                 | 0.063                 | 0.052                 | 0.038                 | 0.058                 |
| 5 Pairs cluster bootstrap for standard error, T(G-1) for critical value | 0.056                 | 0.060                 | 0.050                 | 0.036                 | 0.057                 |
| Bootstrap Percentile-T methods                                          |                       |                       |                       |                       |                       |
| 6 Pairs cluster bootstrap                                               | 0.005                 | 0.019                 | 0.051                 | 0.044                 | 0.069                 |
| 7 Wild cluster bootstrap, Rademacher 2 point distribution               | 0.050                 | 0.059                 | 0.050                 | 0.036                 | 0.055                 |
| 8 Wild cluster bootstrap, Webb 6 point distribution                     | 0.056                 | 0.059                 | 0.048                 | 0.037                 | 0.058                 |

Notes:  March 1997-2012 CPS data, from IPUMS download.  Models include state and year fixed effects, and a "fake policy" dummy variable that turns on in 1995 for a random subset of half of the states.  For 6 and 10 clusters, 4000 Monte Carlo replications.  For 20-50 clusters, 1000 Monte Carlo replications.  The Bootstraps use 399 replications.