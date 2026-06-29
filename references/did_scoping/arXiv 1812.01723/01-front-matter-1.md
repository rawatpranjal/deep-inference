<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1812.01723.pdf
backend: pdftotext
part: 1/3
-->

# Front Matter 1

<!-- pages: 1-35 -->

Doubly Robust Diﬀerence-in-Diﬀerences Estimators∗

                                                                         Pedro H. C. Sant’Anna†                          Jun B. Zhao‡

arXiv:1812.01723v3 [econ.EM] 5 May 2020
                                                                          Vanderbilt University                   Vanderbilt University

                                                                                                   May 5, 2020

                                                                                                       Abstract

                                                     This article proposes doubly robust estimators for the average treatment eﬀect on the treated (ATT) in
                                                 diﬀerence-in-diﬀerences (DID) research designs. In contrast to alternative DID estimators, the proposed
                                                 estimators are consistent if either (but not necessarily both) a propensity score or outcome regression working
                                                 models are correctly speciﬁed. We also derive the semiparametric eﬃciency bound for the ATT in DID
                                                 designs when either panel or repeated cross-section data are available, and show that our proposed estimators
                                                 attain the semiparametric eﬃciency bound when the working models are correctly speciﬁed. Furthermore,
                                                 we quantify the potential eﬃciency gains of having access to panel data instead of repeated cross-section data.
                                                 Finally, by paying particular attention to the estimation method used to estimate the nuisance parameters,
                                                 we show that one can sometimes construct doubly robust DID estimators for the ATT that are also doubly
                                                 robust for inference. Simulation studies and an empirical application illustrate the desirable ﬁnite-sample
                                                 performance of the proposed estimators. Open-source software for implementing the proposed policy
                                                 evaluation tools is available.

                                             ∗ First complete version: November 29, 2018. We thank the editor, Serena Ng, the associate editor, two anonymous referees, Brantly

                                          Callaway, Alex Poirier, Vitor Possebom, Yuya Sasaki, Tymon Słoczyński, Qi Xu, and the audiences of the 2018 SEA conference, 2019
                                          New York Econometrics Camp, and the 2019 IAAE conference for valuable comments.
                                             † Department of Economics, Vanderbilt University. E-mail: pedro.h.santanna@vanderbilt.edu. Part of this article was written when

                                          I was visiting the Cowles Foundation at Yale University, whose hospitality is gratefully acknowledged.
                                             ‡ Department of Economics, Vanderbilt University. E-mail: jun.zhao@vanderbilt.edu.

                                                                                                           1

1 Introduction
   Diﬀerence-in-diﬀerences (DID) methods are among the most popular procedures practitioners adopted to
conduct policy evaluation with observational data. In its canonical form, DID identiﬁes the average treatment
eﬀect on the treated (ATT) by comparing the diﬀerence in pre and post-treatment outcomes of two groups:
one that receives and one that does not receive the treatment (the treated and comparison group, respectively).
In order to attach a causal interpretation to DID estimators, researchers routinely invoke the (unconditional)
parallel trends assumption (PTA): in the absence of the treatment, the average outcome for the treatment and
comparison groups would have followed parallel paths over time. Although the PTA is fundamentally untestable,
its plausibility is usually questioned if the observed characteristics that are thought to be associated with the
evolution of the outcome are not balanced between the treated and comparison group. In such cases, researchers
usually deviate from the canonical DID setup and incorporate pre-treatment covariates into the DID analysis
and assume that the PTA is satisﬁed only after conditioning on these covariates.
   In this paper, we study the robustness and eﬃciency properties of DID estimators for the ATT when the PTA
holds after conditioning on covariates. We consider both settings where panel data are available and settings
where only repeated cross-section data are available. We contribute to the DID literature in diﬀerent fronts.
First, we derive doubly robust (DR) estimands for the ATT under DID settings and propose DR DID estimators
for the ATT that are consistent when either a working (parametric) model for the propensity score or a working
(parametric) model for the outcome evolution for the comparison group is correctly speciﬁed. The setting
where only repeated cross-section data are available is particularly interesting. We propose two diﬀerent DR
DID estimators for the ATT that diﬀer from each other depending on whether or not one models the outcome
regression for the treated group in both pre and post-treatment periods. Nonetheless, we show DR property
does not depend on such a choice.
   Second, we derive the semiparametric eﬃciency bounds for the ATT under DID designs. The semiparametric
eﬃciency bounds we derive are nonparametric in the sense that we do not assume researchers have additional
knowledge about outcome regressions or the propensity score functional forms. As so, these bounds provide a
standard against which one can compare the eﬃciency of any (regular) semiparametric DID estimator for the
ATT. Here, it is also worth stressing that these semiparametric eﬃciency bounds explicitly incorporate all the
restrictions implied by the invoked identiﬁcation assumptions. Importantly, these restrictions diﬀer depending
on whether panel or repeated cross-section data are available. In both cases they involve the moment restrictions
implied by the conditional PTA, though, when repeated cross-section data are available, they also include the
restrictions implied by the identifying assumption that the joint distribution of covariates and treatment status
is invariant to the sampling period (pre and post-treatment). We emphasize that failing to account for all these

                                                       2

implied restrictions can lead to discrepancies on the derived eﬃciency bound, which, in turn, may suggest that
some estimator is semiparametrically eﬃcient when, in fact, it is not.
   With the semiparametric eﬃciency bounds at hand, we can answer several questions that one may have. For
instance, one may wonder whether there are eﬃciency gains associated with having access to panel instead of
repeated cross-section data. By directly comparing the eﬃciency bounds under these two setups, we not only
show that the answer to the aforementioned question is yes, but also show that such gains tend to be larger when
the sample sizes of the pre and post-treatment repeated cross-section data are more imbalanced.
   Another natural question that arises is whether our proposed DR DID estimators can attain the semipara-
metric eﬃciency bound. We show that when the working models for the propensity score and for the outcome
evolution for the comparison group are correctly speciﬁed, our proposed DR DID estimator for the panel data
setup is locally eﬃcient, though the DR DID estimators for the cross-section setup are not. In fact, when
only repeated cross-section data are available, we show that our proposed DR DID estimator that relies on
modelling the propensity score and the outcome evaluation of both the treated and comparison groups attains
the semiparametric eﬃciency bound when all working models are correctly speciﬁed. We quantify the loss
of eﬃciency associated with using the ineﬃcient DR DID estimator instead of the locally eﬃcient one, and
illustrate via Monte Carlo simulations that such a loss can indeed be large.
   Our proposed methodology accommodates linear and nonlinear working models for the nuisance functions.
            √
We establish n-consistency and asymptotic normality of the proposed DR DID estimators when generic
parametric working models are used for the nuisance functions. In doing so we emphasize that, in general, the
DR property of our estimators is with respect to consistency and not to inference. In other words, the exact
form of the asymptotic variance of our proposed estimators depends on whether the propensity score and/or the
outcome regression models are correctly speciﬁed. Given that, in practice, one does not know a priori which
models are correctly speciﬁed, one should consider the estimation eﬀects from all ﬁrst-step estimators when
estimating the asymptotic variance. Failing to do so may lead to invalid inference procedures.
   Motivated by this observation, a third contribution of this paper is to show that, by paying particular
attention to the estimation method used for estimating the nuisance parameters, it is sometimes possible to
construct computationally simple DID estimators for the ATT that are not only DR consistent and locally
semiparametric eﬃcient, but are also doubly robust for inference. These further improved DR DID estimators
are particularly attractive and easy to implement when researchers are comfortable with a logistic working
model for the propensity score and with linear regression working models for the outcome of interest.
   Related literature: Our proposal builds on two branches of the causal inference literature. First, our
methodological results are intrinsically related to other DID papers; for an overview, see e.g., Section 6.5 of
Imbens and Wooldridge (2009) and references therein. Two leading contributions in this branch of literature

                                                       3

that are particularly relevant to this paper are Heckman et al. (1997), who propose kernel-based DID regression
estimators, and Abadie (2005), who proposes (parametric and nonparametric) DID inverse probability weighted
(IPW) estimators. We note that when the dimension of available covariates is high or even moderate, fully
nonparametric procedures usually do not lead to informative inference because of the “curse of dimensionality”.
In these cases, researchers often adopt parametric methods. Our DR DID estimators fall in this latter category.
    Second, our results are also directly related to the literature on doubly robust estimators, see
Robins et al. (1994), Scharfstein et al. (1999), Bang and Robins (2005), Wooldridge (2007), Chen et al.
(2008), Cattaneo (2010), Graham et al. (2012, 2016), Vermeulen and Vansteelandt (2015), Lee et al. (2017),
Słoczyński and Wooldridge (2018), Rothe and Firpo (2018), Muris (2019), among many others; for an
overview, see section 2 of Słoczyński and Wooldridge (2018), and Seaman and Vansteelandt (2018). Recently,
DR estimators have also been playing an important role when one uses data-adaptive, “machine learning”
estimators for the nuisance functions, see e.g., Belloni et al. (2014), Farrell (2015), Chernozhukov et al. (2017),
Belloni et al. (2017), and Tan (2019). As so, these papers are also broadly related to our proposal, even though
we use parametric ﬁrst-step estimators. On the other hand, we note that the aforementioned papers focus on
either the “selection on observables” or “IV/LATE” type assumptions, whereas we pay particular attention to
the conditional DID design. Thus, our results complement theirs.
    To derive the semiparametric eﬃciency bounds for the ATT under the DID framework, we build on Hahn
(1998) and Chen et al. (2008). Although we follow the structure of semiparametric eﬃciency bound derivation
of the aforementioned papers (which, in turn, follow Newey (1990)), our derived semiparametric eﬃciency
bounds complement theirs as we focus on DID designs while Hahn (1998) and Chen et al. (2008) results rely
on “selection on observables” type assumptions in cross-section setups.
    Our results for the further improved DR DID estimators build on Vermeulen and Vansteelandt (2015),
who propose estimators that are DR for inference in cross-section setups under selection on observables type
assumptions. We extend Vermeulen and Vansteelandt (2015) proposal to DID settings with both panel and
repeated cross-section data. Our further improved DR DID estimators also build on Graham et al. (2012), as
their proposed propensity score estimator is one important component of our proposal.
    Finally, in work related but independent from ours, Zimmert (2019) provides high-level conditions under
which one can use “machine-learning” ﬁrst-step estimators when estimating the ATT in DID setups. His
results complement ours, though we note that his proposed estimators for the repeated cross-section case do not
attain the semiparametric eﬃciency bound derived in this paper, and the loss of eﬃciency can be of ﬁrst-order
importance. We also note that Zimmert (2019) does not provide a detailed comparison between the panel and
repeated cross-section data setups like we do, nor discusses DR inference procedures, which are particularly
relevant under model misspeciﬁcations.

                                                        4

    Organization of the paper: In the next section, we describe this paper’s framework, brieﬂy give an overview
of the existing DID estimators and describe how we combine the strengths of each method to form our DR
DID estimands. We also derive semiparametric eﬃciency bounds for the ATT in Section 2. In Section 3, we
propose diﬀerent DR DID estimators, derive their large sample properties, and show that we can get improved
DR DID estimators by paying particular attention to the estimation method used for estimating the nuisance
parameters. We examine the ﬁnite sample properties of our proposed methodology by means of a Monte Carlo
study in Section 4, and provide an empirical illustration in Section 5. Section 6 concludes. Mathematical
proofs are gathered in the Supplemental Appendix.1 Finally, all proposed policy evaluation tools discussed in
this article can be implemented via the open-source R package DRDID, which is freely available from GitHub
(https://github.com/pedrohcgs/DRDID).

2 Difference-in-differences
2.1 Background
    We ﬁrst introduce the notation we use throughout the article. We focus on the case where there are two
treatment periods and two treatment groups. Let Yit be the outcome of interest for unit i at time t. We assume
that researchers have access to outcome data in a pre-treatment period t = 0 and in a post-treatment period t = 1.
Let Dit = 1 if unit i is treated before time t and Dit = 0 otherwise. Note that Di0 = 0 for every i, allowing us to
write Di = Di1 . Using the potential outcome notation, denote Yit (0) the outcome of unit i at time t if it does not
receive treatment by time t and Yit (1) the outcome for the same unit if it receives treatment. Thus, the realized
outcome for unit i at time t is Yit = DiYit (1) + (1 − Di )Yit (0). A vector of pre-treatment covariates Xi is also
available. Henceforth, we assume that the ﬁrst element of Xi is a constant.
    In the rest of the article, we assume that either panel or repeated cross-section data on (Yit , Di , Xi ), t = 0, 1 are
available. When repeated cross-section data are available, we follow Abadie (2005) and assume that covariates
and treatment status are stationary. We formalize these conditions in the following assumption. Let Ti be a
dummy variable that takes value one if the observation i is only observed in the post-treatment period, and
zero if observation i is only observed in the pre-treatment period. Deﬁne Yi = TiYi1 + (1 − Ti )Yi0 , and let n1
and n0 be the sample sizes of the post-treatment and pre-treatment periods such that n = n1 + n0 . Finally, let
λ = P (T = 1) ∈ (0, 1).

Assumption 1 Assume that either (a) the data {Yi0 ,Yi1 , Di , Xi }ni=1 are independent and identically distributed
(iid); or (b) the pooled repeated cross-section data {Yi , Di , Xi , Ti }ni=1 consist of iid draws from the mixture

1 The Supplemental Appendix is available at https://pedrohcgs.github.io/files/DR-DIDAppendix.pdf

                                                             5

distribution

               P (Y ≤ y, D = d, X ≤ x, T = t) = t · λ · P (Y1 ≤ y, D = d, X ≤ x|T = 1)

                                                              + (1 − t) · (1 − λ ) P (Y0 ≤ y, D = d, X ≤ x|T = 0) ,

where (y, d, x,t) ∈ R× {0, 1} × Rk × {0, 1}, with the joint distribution of (D, X ) being invariant to T .

    Assumption 1(a) covers the case where panel data are available, whereas Assumption 1(b) covers the case
where repeated cross-section data are available, and allows for diﬀerent sampling schemes. For instance, it
accommodates the binomial sampling scheme where an observation i is randomly drawn from either (Y1 , D, X )
or (Y0 , D, X ) with ﬁxed probability λ (here, T is a non-degenerated random variable). It also accommodates the
“conditional” sampling scheme where n1 observations are sampled from (Y1 , D, X ), n0 observations are sampled
from (Y0 , D, X ) and λ = n1 /n (here, T is treated as ﬁxed). On the other hand, Assumption 1(b) rules out settings
with compositional changes in (D, X ), see e.g. Hong (2013) for a discussion.
    The parameter of interest is the average treatment eﬀect on the treated,

                                                   τ = E[Yi1 (1) −Yi1 (0)|Di = 1].

As expectations are linear operators and Yi1 (1) = Yi1 if Di = 1, we can rewrite the ATT as2

                         τ = E[Y1 (1)|D = 1] − E[Y1 (0)|D = 1] = E[Y1 |D = 1] − E[Y1 (0)|D = 1],                                      (2.1)

where we drop subscript i to ease notation; we follow this convention throughout the paper. From the above
representation, it is clear that the main challenge in identifying the ATT is to compute E[Yi1 (0)|Di = 1] from
the observed data. To overcome this challenge, we invoke the following assumptions.

Assumption 2 E[Y1 (0) −Y0 (0)|D = 1, X ] = E[Y1 (0) −Y0 (0)|D = 0, X ] almost surely (a.s.).

Assumption 3 For some ε > 0, P (D = 1) > ε and P (D = 1|X ) ≤ 1 − ε a.s..

    Assumption 2, which we refer to as the conditional PTA throughout the paper, states that in the absence
of treatment, the average conditional outcome of the treated and the comparison groups would have evolved in
parallel. Note that Assumption 2 allows for covariate-speciﬁc time trends, though it rules out unit speciﬁc trends.
Assumption 3 is an overlap condition and states that at least a small fraction of the population is treated and that
for every value of the covariates X , there is at least a small probability that the unit is not treated. These two
assumptions are standard in conditional DID methods, see e.g. Heckman et al. (1997), Heckman et al. (1998),
Blundell et al. (2004), Abadie (2005) and Bonhomme and Sauder (2011).

2 Throughout the rest of the paper, to ease the notation burden we denote E [·] as generic expectations. In the case of panel data, such
  expectations are with respect to the distribution of (Y0 ,Y1 , D, X). In the case of repeated cross-section data, the expectations are with
                                        1 P (T = t) · P (Y ≤ y, D = d, X ≤ x|T = t).
  respect to the mixture distribution ∑t=0                  t

                                                                     6

    Under Assumptions 1-3, there are two main ﬂexible estimation procedures to estimate the ATT: the outcome
regression (OR) approach, see e.g. Heckman et al. (1997), and the IPW approach, see e.g. Abadie (2005).
The OR approach relies on researchers ability to model the outcome evolution. In such cases, under the
aforementioned assumptions one can estimate the ATT using
                                      "                                                                     #
                                    reg                        −1
                                                                                                          
                                  b
                                  τ       = Ȳ1,1 −   Ȳ1,0 + ntreat    ∑                      b 0,0 (Xi ) ,
                                                                                 b 0,1 (Xi ) − µ
                                                                                 µ                                                   (2.2)
                                                                       i|Di =1
where Ȳd,t = ∑i|Di =d,Ti =t Yit /nd,t is the sample average outcome among units in treatment group d and time t,
    b d,t (x) is an estimator of the true, unknown md,t (x) ≡ E[Yt |D = d, X = x],3 see e.g. Heckman et al. (1997).
and µ
    The IPW approach proposed by Abadie (2005) avoids directly modelling the outcome evolution and exploits
that, under Assumptions 1-3, the ATT can be expressed as
                                                                        
                                           1        D − p (X )
                                      τ=        E              (Y1 −Y0 )
                                         E [D]      1 − p (X )
when panel data are available, and as
                                                                          
                                          1       D − p (X ) T − λ
                                     τ=       E                          Y                            (2.3)
                                        E [D]      1 − p (X ) λ (1 − λ )
when repeated cross-section data are available, where p (X ) ≡ P (D = 1|X ) is the true, unknown propensity
score. Abadie’s identiﬁcation results suggest simple two-step estimators for the ATT that do not involve
outcome regressions. For instance, when panel data are available, Abadie (2005) proposes the following
Horvitz and Thompson (1952) type IPW estimator,
                                                                            
                                      ipw,p     1         D−πb (X )
                                    τb      =        En             (Y −Y0 ) ,                               (2.4)
                                              En [D]      1−πb (X ) 1
      b (x) is an estimator of the true, unknown p (x), and for a generic random variable Z, En [Z] = n−1 ∑ni=1 Zi ;
where π
the estimator for the repeated cross-section case is formed using the analogous procedure.
    It is important to emphasize that the reliability of ATT estimators based on the OR and the IPW approaches
depends on diﬀerent, non-nested conditions. For the OR approach, the consistency of the ATT estimator (2.2)
                                      b d,t (·), being correctly speciﬁed, whereas the IPW estimator (2.4) relies on
relies on the estimators of md,t (·), µ
                               b (·) of p (·) being correctly speciﬁed. As so, in practice, it may be hard to “rank”
the propensity score estimator π
these two approaches in terms of their robustness to model misspeciﬁcation.

Remark 1 It is common to see practitioners adopting the two-way ﬁxed eﬀects linear regression model

                                      Yit = α 1 + α 2 Ti + α 3 Di + τ f e (Ti · Di ) + θ ′ Xi + ε it ,                               (2.5)

and interpreting estimates of τ f e as estimates of the ATT, see e.g. chapter 5.2 in Angrist and Pischke (2009).
Although (2.5) may be perceived as a “natural” speciﬁcation, it implicitly imposes additional restrictions
on the data generating process beyond Assumptions 1-3. More speciﬁcally, (2.5) implicitly imposes that

3 In the repeated cross-section case, md,t (x) = E [Y |D = d, T = t, X = x]. In the next section, we diﬀerentiate the notation for the panel
  data and repeated cross-section case to avoid potential confusions.

                                                                        7

(i) E [Y1 (1) −Y1 (0) |X , D = 1] = τ f e a.s., i.e., it assumes homogeneous (in X ) treatment eﬀects, and (ii) for
d = 0, 1, E [Y1 −Y0 |X , D = d] = E [Y1 −Y0 |D = d] a.s., i.e., it rules out X -speciﬁc trends in both treated and
comparison groups.4 When these additional restrictions are not satisﬁed, the estimand τ f e is, in general, diﬀerent
from the ATT, and policy evaluation based on it may be misleading. We further illustrate this point using Monte
Carlo simulations in Section 4; see also Słoczyński (2018) for related results.

2.2 Doubly robust difference-in-differences estimands
      In this section, we argue that instead of choosing between the OR and the IPW approaches, one can combine
them to form doubly robust (DR) moments/estimands for the ATT. Here, double robustness means that the
resulting estimand identiﬁes the ATT even if either (but not both) the propensity score model or the outcome
regression models are misspeciﬁed. As so, the DR DID estimand for the ATT shares the strengths of each
individual DID method and, at the same time, avoids some of their weaknesses.
      Before describing how we exactly combine the OR and the IPW approaches to form our DR DID estimand,
we need to introduce some additional notation. Let π (X ) be an arbitrary model for the true, unknown propensity
                                                                  p            p            p           p
score. When panel data are available, let ∆Y = Y1 −Y0 and deﬁne µ d,∆ (X ) ≡ µ d,1 (X ) − µ d,0 (X ), µ d,t (x) being
                                                  p
a model for the true, unknown outcome regression md,t (x) ≡ E[Yt |D = d, X = x], d,t = 0, 1. When only
repeated cross-section data are available, let µ rc
                                                 d,t (x) be an arbitrary model for the true, unknown regression

mrc                                                                    rc                  rc                      rc
 d,t (x) ≡ E[Y |D = d, T = t, X = x], d,t = 0, 1, and for, d = 0, 1, µ d,Y (T, X ) ≡ T · µ d,1 (X ) + (1 − T ) · µ d,0 (X ),

and µ rc           rc           rc
      d,∆ (X ) ≡ µ d,1 (X ) − µ d,0 (X ).

      For the case in which panel data are available, we consider the estimand
                                        h                                p
                                                                                  i
                              τ dr,p = E w1p (D) − w0p (D, X ; π ) ∆Y − µ 0,∆ (X ) ,                                               (2.6)

where, for a generic g,

                    
                               D              p               g(X ) (1 − D)          g(X ) (1 − D)
                      w1p (D) =    , and w0 (D, X ; g) =                         E                   .                             (2.7)
                             E [D]                               1 − g(X )              1 − g(X )
For the repeated cross-section case, we consider two diﬀerent estimands,
                                      rc                                                  
                         τ dr,rc
                           1     = E  (w 1 (D, T ) − w rc
                                                       0  (D, T, X ; π )) Y − µ rc
                                                                                0,Y (T, X )   ,                                    (2.8)

and
                                                             rc                               
  τ dr,rc
    2     = τ dr,rc
              1     + E µ rc           rc                                rc
                          1,1 (X ) − µ 0,1 (X ) D = 1 − E µ 1,1 (X ) − µ 0,1 (X ) D = 1, T = 1
                                                                      rc                                 
                               − E µ rc              rc                                rc
                                        1,0 (X ) − µ 0,0 (X ) D = 1 − E µ 1,0 (X ) − µ 0,0 (X ) D = 1, T = 0 , (2.9)

4 Note that under Assumptions 1-3, (2.5) suggests that, with probability one, E [Y1 (1) |X, D = 1] = α 1 + α 2 + α 3 + τ + θ ′ X , and
  E [Y1 (0) |X, D = 1] = E[Y0 |D = 1, X] + (E[Y1 |D = 0, X] − E[Y0 |D = 0, X]) = α 1 + α 2 + α 3 + θ ′ X . Point (i) now follows directly.
  Point (ii) follows from analogous arguments.

                                                                    8

where, for a generic g,

  wrc           rc             rc
   1 (D, T ) = w1,1 (D, T ) − w1,0 (D, T ) ,     and wrc                 rc                   rc
                                                      0 (D, T, X ; g) = w0,1 (D, T, X ; g) − w0,0 (D, T, X ; g) , (2.10)

and, for t = 0, 1,
                                           D · 1 {T = t}
                       wrc
                        1,t (D, T ) =                       ,
                                          E [D · 1 {T = t}]

                            
                                          g(X ) (1 − D) · 1 {T = t}     g(X ) (1 − D) · 1 {T = t}
                 wrc
                  0,t (D, T, X ; g) =                                E                              .
                                                   1 − g(X )                    1 − g(X )

Theorem 1 Let Assumptions 1-3 hold. Then:
    (a) When panel data are available, τ dr,p = τ if either (but not necessarily both) π (X ) = p (X ) a.s. or
µ ∆p (X ) = m0,1
             p           p
                 (X ) − m0,0 (X ) a.s.;
    (b) When repeated cross-section data are available, τ dr,rc
                                                          1     = τ dr,rc
                                                                    2     = τ if either (but not necessarily both)
π (X ) = p (X ) a.s. or µ rc          rc          rc
                          0,∆ (X ) = m0,1 (X ) − m0,0 (X ) a.s..

    Theorem 1 states that provided that at least one of the working nuisance models is correctly speciﬁed, we
can recover the ATT with either panel or repeated cross-section data. Thus, our proposed DR DID estimands
are “less demanding” in terms of the researchers’ ability to correctly specify models for the nuisance functions
than either the OR or the IPW approach.
    Given that we consider two diﬀerent estimands for the case of repeated cross-section, it is interesting to
use Theorem 1 to compare them. Given that τ dr,rc
                                            1     does not rely on OR models for the treated group but τ dr,rc
                                                                                                         2

does, one could a priori expect that τ dr,rc
                                       1     would be more robust against model misspeciﬁcation than τ dr,rc
                                                                                                       2     .
Nonetheless, Theorem 1 states that this is not the case as they identify the ATT under the same conditions.
At this stage, one may wonder how this is possible. To answer such a query, it suﬃces to remember that,
under the stationarity condition in Assumption 1(b), for any generic integrable and measurable function g,
E [g (X )| D = 1] = E [ g (X )| D = 1, T = t], t = 0, 1. Given that this holds for any generic function g, it must
also hold for µ rc          rc                              rc                                 rc
                1,t (·) − µ 0,t (·) , t = 0, 1, even when µ d,t (·) are misspeciﬁed models of md,t (·). Such a result

reveals that modeling the OR for the treat group can be “harmless” in terms of identiﬁcation, provided that these
additional models are incorporated into τ dr,rc
                                          1     in an appropriate manner.

2.3 Semiparametric efficiency bound
    In the previous subsection, we derived DR moment equations for the ATT under the DID framework and
showed that the resulting estimands are more robust against model misspeciﬁcations than DID estimands based
on either the OR or the IPW approach. In this subsection, we shift our attention from “robustness” to eﬃciency.
More precisely, we calculate the semiparametric eﬃciency bound for the ATT under Assumptions 1-3 when
either panel or repeated cross-section data are available. These results provide the semiparametric analog of the
Cramér–Rao lower bound commonly used in fully parametric procedures. As so, they provide a benchmark that

                                                             9

researchers can use to assess whether any given (regular) semiparametric DID estimator for the ATT is fully
exploiting the empirical content of Assumptions 1-3.
         p          p        p                                        rc        rc
    Let m0,∆ (x) ≡ m0,1 (x)−m0,0 (x), and, for d = 0, 1, mrc
                                                          d,∆ (X ) ≡ md,1 (X )−md,0 (X ). Recall that λ ≡ P (T = 1).

Next proposition displays the semiparametric eﬃciency bound for the ATT when one has access to panel data
and when one has access to repeated cross-section data. To simplify exposition, we abstract from additional
technical discussions related to the conditions to guarantee quadratic mean diﬀerentiability and their implications
for the precise deﬁnition of eﬃcient inﬂuence function ; see, e.g., Chapter 3 of Bickel et al. (1998) for additional
details.

Proposition 1 Let Assumptions 1-3 hold. Then:
   (a) When panel data are available, the efficient influence function for the AT T is
                                                       
   e,p                  p       p           p
  η (Y1 ,Y0 , D, X ) = w1 (D) m1,∆ (X ) − m0,∆ (X ) − τ
                                                                                                  
                                          + w1p (D) ∆Y − m1,∆ p
                                                                 (X ) − w0p (D, X ; p) ∆Y − m0,∆
                                                                                             p
                                                                                                 (X ) , (2.11)

and the semiparametric efficiency bound for all regular estimators for the ATT is
    h                   i                                         2
       e,p            2        1          p            p
  E η (Y1 ,Y0 , D, X ) =           E  D m 1,∆ (X ) − m 0,∆ (X ) − τ
                            E [D]2
                                                                                                      #
                                                               2 (1 − D) p (X )2                2
                                                       p                                   p
                                         +D ∆Y − m1,∆      (X ) +                    ∆Y − m0,∆ (X )     . (2.12)
                                                                       (1 − p (X ))2
    (b) When only repeated cross-section data are available, the efficient influence function for the AT T is
                            D                              
  η e,rc (Y, D, T, X ) =         mrc          rc
                                  1,∆ (X ) − m0,∆ (X ) − τ
                           E [D]
                                                                                           
                                + wrc                  rc          rc               rc
                                   1,1 (D, T ) Y − m1,1 (X ) − w1,0 (D, T ) Y − m1,0 (X )
                                                                                                            
                                     − wrc                        rc          rc                     rc
                                           0,1 (D, T, X ; p) Y − m0,1 (X ) − w0,0 (D, T, X ; p) Y − m0,0 (X ) , (2.13)

   and the semiparametric efficiency bound for all regular estimators for the ATT is
   h                     i     1     h                            2
  E η e,rc (Y, D, T, X )2 =      2
                                   E  D mrc           rc
                                         1,∆ (X ) − m0,∆ (X ) − τ
                            E [D]
                                  DT                2 D (1 − T )                 2
                               + 2 Y − mrc  1,1 (X ) +           2
                                                                     Y − mrc
                                                                          1,0 (X )
                                  λ                      (1 − λ )
                                                                                                                 #
                                                                                     2
                                (1 − D) p (X )2 T                  2 (1 − D)  p (X )  (1 − T )               2
                            +                        Y − mrc
                                                          0,1 (X ) +                            Y − mrc
                                                                                                     0,0 (X )      . (2.14)
                                (1 − p (X ))2 λ 2                      (1 − p (X ))2 (1 − λ )2

    It is interesting to compare η e,p (D, X ) with η e,rc (D, T, X ). First, note that the ﬁrst component of their
eﬃcient inﬂuence functions are analogous to each other, and depends on the true, unknown conditional ATT,
m1,∆ (X ) − m0,∆ (X ).5 The second and third terms in (2.11) and (2.13) are more diﬀerent from each other. For

5 To avoid excessive notational burden, we supress the “p” and “rc” superscripts unless their omission leads to confusion.

                                                                 10

η e,p , the availability of panel data implies that Y1 and Y0 are observed for all units, and, therefore, we can
directly reweight ∆Y − m1,∆ (X ) and ∆Y − m0,∆ (X ). In contrast, when only repeated cross-section data are
available, one observes Yt only if T = t, t = 0, 1, and, therefore, the eﬃcient inﬂuence function (2.13) depends
on diﬀerent weights for each pair (D, T ) ∈ {0, 1}2 . In this latter case, we also stress the importance of imposing
the stationarity condition in Assumption 1(b) when deriving the eﬃcient inﬂuence function (2.13) – failing to
do so will suggest an “eﬃciency bound” that is wider than (2.14).
    It is also worth mentioning that the eﬃcient inﬂuence functions (2.11) and (2.13) depend on the true,
unknown, outcome regression functions for the treated group, m1,1 (·) and m1,0 (·), in an asymmetric manner.
On one hand, when panel data are available, by simple manipulation, we can rewrite η e,p as
                                                                 
                η e,p (Y1 ,Y0 , D, X ) = w1p (D) − w0p (D, X ; p) (∆Y − m0,∆ (X )) − w1p (D) · τ ,

emphasizing that the eﬃcient inﬂuence function for the ATT when panel data are available does not depend on
m1,1 (·) and m1,0 (·). This is in sharp contrast to the case where only repeated cross-section data are available.
    Another interesting question raised by Proposition 1 is whether the semiparametric eﬃciency bound for the
case of repeated cross-section data is larger than the one for the case of panel data. In order to answer this
question, we consider the case where T is independent of (Y1 ,Y0 , D, X ), so that Assumptions 1(a) and 1(b) are
compatible with each other.6

Corollary 1 Let Assumptions 1-3 hold, and assume that T is independent of (Y1 ,Y0 , D, X ). Then,
   h                    i    h                       i
 E η e,rc (Y, D, T, X )2 − E η e,p (Y1 ,Y0 , D, X )2
                                     r                              r                      !2
                           1              1  − λ                        λ
                     =         E D               (Y1 − m1,1 (X )) +       (Y0 − m1,0 (X ))
                        E [D]2               λ                         1−λ
                                                             r                             r                          !2 
                                                         2
                                      (1 − D) p (X )             1−λ                            λ
                                  +                  2
                                                                     (Y1 − m0,1 (X )) +               (Y0 − m0,0 (X ))  ≥ 0.
                                       (1 − p (X ))               λ                            1−λ

    In other words, under the DID framework it is possible to form more eﬃcient estimators for the ATT when
panel data are available than when only repeated cross-section data are available. In addition, from Corollary
1, we can also see that the eﬃciency loss is convex in λ , implying that the loss of eﬃciency is bigger when the
pre and post-treatment sample sizes are more imbalanced. In fact, when
    "                                                       #
                       2  (1 − D) p (X )2                 2
  E D (Y0 − m1,0 (X )) +                  (Y0 − m0,0 (X )) =
                           (1 − p (X ))2
                                               "                                                      #
                                                                 2  (1 − D) p (X )2                 2
                                             E D (Y1 − m1,1 (X )) +                 (Y1 − m0,1 (X )) , (2.15)
                                                                     (1 − p (X ))2

6 This “restriction” does not aﬀect the semiparametric eﬃciency bound for the case where only repeated cross-section data are available,
  as it does not impose additional restrictions on the observed data.

                                                                   11

we can show that λ = 0.5 is optimal . However, when (2.15) does not hold, the optimal λ depends on the data
                                                       
in a more complicated manner, and is given by λ = σ̃ 1 (σ̃ 0 + σ̃ 1 ), where, for t = 0, 1
                                "                                                        #
                                                                     2
                                                   2  (1 − D) p (X  )                  2
                       σ̃ t2 = E D (Yt − m1,t (X )) +                  (Yt − m0,t (X )) .
                                                       (1 − p (X ))2
These results suggest that, in principle, one may beneﬁt from “oversampling” from either the pre or post-
treatment period. However, it is, in general, not feasible to know the optimal λ during the design stage, i.e.,
at the pre-treatment period, since σ̃ 21 depends on the outcome data from the post-treatment period. Thus, if
one were to design the DID study with repeated cross-section units, it seems that setting λ = 0.5 would be a
“reasonable” choice.

3 Estimation and inference
    In this section, we build on the DR DID estimands in Theorem 1 and the semiparametric eﬃciency bounds
in Proposition 1, and discuss estimation and inference procedures for the ATT in DID designs. Indeed, the
moment equations (2.6), (2.8), and (2.9) suggest a simple two-step strategy to estimate the ATT. In the ﬁrst step,
                                                                          p                          p
one estimates the true, unknown p (·) using π (·), and the true, unknown md,t (·) (mrc                         rc
                                                                                    d,t (·)) using µ d,t (·)(µ d,t (·)),

d,t = 0, 1, when panel data (repeated cross-section data) are available. In the second step, one plugs the ﬁtted
values of the estimated propensity score and regression models into the sample analogue of τ dr,p , τ dr,rc
                                                                                                      1     , or τ dr,rc
                                                                                                                   2     .
    Although, in principle, one can use semi/non-parametric estimators for both the outcome regressions and the
propensity score, see e.g. Heckman et al. (1997), Abadie (2005), Chen et al. (2008) and Rothe and Firpo (2018),
in what follows ,we focus our attention on generic parametric ﬁrst-step estimators. More precisely, we assume
that π (x; γ ∗ ) is a parametric model for p (x) , such that π is known up to the ﬁnite dimensional pseudo-true
                                                                                
                                               p
parameter γ ∗ . Analogously, for d,t = 0, 1, µ d,t   x; β ∗,p
                                                          d,t   (and µ rc x; β ∗,rc ) is a parametric model for m p (x)
                                                                       d,t     d,t                               d,t
                        p                                                                         ∗,p    ∗,rc
(mrc                           rc
  d,t (x)), such that µ d,t (µ d,t ) is known up to the ﬁnite dimensional pseudo-true parameter β d,t (β d,t ). This

is perhaps the most popular approach adopted by practitioners, particularly when the available sample size is
moderate and/or the dimension of available covariates is high or even moderate, as the “curse of dimensionality”
usually prevents one to adopt fully nonparametric procedures.7
    In the case when panel data are available, our proposed DR DID estimator for the ATT is based on (2.6) and
is given by
                                          h                                             p        p i
                            τbdr,p = En       b1p (D) − w
                                              w         b0p (D, X ; γb) ∆Y − µ 0,∆
                                                                               p
                                                                                    X ; βb 0,0 , βb 0,1   ,                           (3.1)

7 Let g (x) be a generic notation for p (x), mld,t (X) , mld,t (X), d,t = 0, 1, l = p, rc. From Newey (1994), Chen et al. (2003), Ai and Chen
  (2003, 2007, 2012), and Chen et al. (2008),
                                                   can see that the use of nonparametric ﬁrst-step estimators gb(x) of g (x) is warranted
                                                   one
                 g (x) − g (x)kH = o p n−1/4 for a pseudo-metric k·kH , H being a vector space of functions. However, when the
  provided that kb
  dimension
          of X is moderate or large, as is usually the case in many empirical applications, conditons ensuring that kb
                                                                                                                       g (x) − g (x)kH =
  op n−1/4   can be rather stringent because of the “curse of dimensionality”.

                                                                    12

where
                              
                                  D                                      π (X ; γ ) (1 − D)                π (X ; γ ) (1 − D)
                  b1p (D) =
                  w                    ,     and       b0p (D, X ; γ ) =
                                                       w                                          En                            ,   (3.2)
                                En [D]                                      1 − π (X ; γ )                   1 − π (X ; γ ))
                                                        p
γb is an estimator for the pseudo-true γ ∗ , βb 0,t is an estimator for pseudo-true β ∗,p
                                                                                      0,t , t = 0, 1, and for a generic β 0
            p                       p                 p
and β 1 , µ 0,∆ (·; β 0 , β 1 ) = µ 0,1 (·; β 1 ) − µ 0,0 (·; β 0 ).
     When only repeated cross-section data are available, we propose two diﬀerent DR DID estimators for the
ATT. The ﬁrst one, which is based on (2.8) and can be interpreted as the analogue of b                    τ dr,p , is given by
                                     h                                                         rc      rc i
                    b
                    τ1 dr,rc
                              =  E n   ( b
                                         w rc
                                              (D, T ) −  b
                                                         w rc
                                                              (D, T, X ; b          rc         b       b
                                                                         γ )) Y − µ 0,Y T, X ; β 0,0 , β 0,1        ,          (3.3)
                                           1               0
                    rc       rc                      rc                         rc  b rc
where µ rc                                    rc                            rc
        0,Y T, ·; β 0,0 , β 0,1 = T · µ 0,1 ·; β 0,1 + (1 − T ) · µ 0,0 ·; β 0,0 , β d,t is an estimator for the pseudo-true

β ∗,rc                              brc
  d,t , d,t = 0, 1, and the weights w               brc
                                      1 (D, T ) and w              b) are, respectively, deﬁned as the sample analogues
                                                      0 (D, T, X ; γ

of wrc             rc                                               b) playing the role of g.
    1 (D, T ) and w0 (D, T, X ; g) deﬁned in (2.10), but with π (x; γ

   The second DR DID estimator for the case of repeated cross-section builds on (2.9) and is given by
                                                                                    
                                 D                                  rc             rc 
   dr,rc    dr,rc
 τb2 = τb1 + En                       −w  rc              rc      b        rc     b
                                         b1,1 (D, T ) µ 1,1 X ; β 1,1 − µ 0,1 X ; β 0,1
                               En [D]
                                                                                                         
                                                    D                              rc                rc 
                                      − En              −w   rc            rc     b            rc    b
                                                           b1,0 (D, T ) µ 1,0 X ; β 1,0 − µ 0,0 X ; β 0,0         , (3.4)
                                                 En [D]
                  rc      rc              rc              rc  
where µ rc                         rc               rc                            brc
         d,∆ ·; β d,1 , β d,0 = µ d,1 ·; β d,1 − µ d,0 ·; β d,0 , and the weights w                 brc
                                                                                    1,t (D, T ) and w 0,t (D, T, X ; b
                                                                                                                     γ ) are,
respectively, deﬁned as the sample analogues of wrc               rc
                                                 1,t (D, T ) and w0,t (D, T, X ; g), t = 0, 1, deﬁned below (2.10),

but with π (x; b
               γ ) playing the role of g.
                                                                                                                                     dr,p
     As we show in the Appendix A, it is relatively straightforward to derive the asymptotic properties of τb                               ,
τbdr,rc
  1
              dr,rc
        and τb2 using generic ﬁrst-step estimators that satisfy some relatively weak, high-level conditions; see
                                                                                 dr,p
Theorems A.1 and A.2 in Appendix A. Indeed, Theorem A.1 indicates that τb                                      is doubly robust, and also
locally semiparametrically eﬃcient, i.e., its asymptotic variance achieves the semiparametric eﬃciency bound
when the working models for the nuisance functions are correctly speciﬁed. Theorem A.2 also indicates that
        dr,rc         dr,rc
both τb1        and τb2       are doubly robust when repeated cross-section data are available. However, Theorem A.2
                              dr,rc                                                               dr,rc
also highlights that τb2              is locally semiparametrically eﬃcient, whereas τb1                    is not. In other words, when
repeated cross-section data are available, τbdr,rc
                                             2     tends to have more attractive properties than τbdr,rc
                                                                                                   1     , regardless of
the ﬁrst-step estimators used.
     Although the results in Theorems A.1 and A.2 accommodate a variety of diﬀerent ﬁrst-step estimators, in
practice, one still needs to choose a particular estimation procedure to be implemented. In what follows, we
attempt to provide some guidance on the choice of ﬁrst-step estimators with the goal of further improving the
(generic) DR DID estimators. We are particularly interested in forming DR DID estimators that are not only
doubly robust in terms of consistency—like described above—but also doubly robust for inference, i.e., their
asymptotic linear representation is also doubly robust. The attractiveness of forming estimators that are DR

                                                                       13

for inference is that there is no estimation eﬀect from ﬁrst-step estimators, which, in turn, implies that the
asymptotic variance of the results DR DID estimator for the ATT is invariant to which working models for
the nuisance functions are correctly speciﬁed. In practice, this usually translates to simpler and more stable
inference procedures.
    To derive these improved DR DID estimators, we focus on the case where a researcher is comfortable with
linear regression working models for the outcome of interest, a logistic working model for the propensity score,
and with covariates X entering all the nuisance models in a symmetric manner. Although these modelling
conditions are more stringent than those allowed by our generic DR DID estimators discussed in Appendix A,
they are much weaker than those implicitly imposed in the TWFE speciﬁcation (2.5), and can be seen as the
default choice in many applications. Hence, these extra assumptions can be seen as a reasonable compromise
to get further improved DR DID estimators that are also computationally tractable and easy to implement in
practice.

3.1 Improved DR DID estimators when panel data are available
   As discussed above, we consider the following working models for the nuisance functions:
                                 exp (X ′ γ )                                                          
                                                          p           p       p           lin,p         p      ′ p
         π (X , γ ) = Λ X ′γ ≡                   ,  and µ 0,∆   X ; β 0,1 , β 0,1    =  µ 0,∆     X ; β 0,∆ ≡ X β 0,∆ .    (3.5)
                                1 + exp (X ′ γ )
Our proposed improved DR DID estimator is given by the three-step estimator
                                                                                               
                                                                                                wls,p
                       dr,p
                     τbimp = En w  p
                                  b1 (D) − w    p
                                               b0 D, X ; γb ipt                   lin,p
                                                                      ∆Y − µ 0,∆ X ; β 0,∆    b             ,

where the ﬁrst two-steps consist of computing
                                                                              
                               γ ipt
                               b       = arg max En DX ′ γ − (1 − D)exp X ′ γ ,
                                             γ ∈Γ
                                                                                      
                               wls,p                    Λ X ′ γbipt                 2
                            βb 0,∆     = arg min En                   ∆Y − X ′ b D = 0 ,
                                             b∈Θ                ′  ipt
                                                      1 − Λ X γb
while in the third and last step, one plugs the ﬁtted values of the working models (3.5) into the sample analogue
                            γ ipt is the inverse probability tilting estimator proposed by Graham et al. (2012) in a
of τ dr,p . Here, note that b
                           wls,p
diﬀerent context, while βb 0,∆ is simply the weighted least squares estimator for β ∗,p
                                                                                    0,∆ .
                                                                                    wls,p
                                                                              and βb 0,∆ instead of other available alternatives.
                                                                        ipt
    At this point, one may wonder why we use the estimators γb
To answer such a query, recall that the main goal here is to propose DID estimators for the ATT that are not
only DR consistent but also DR for inference, i.e., the exact form of their asymptotic variance does not depend
on which working models for the nuisance functions are correctly speciﬁed. As it turns out, the key to obtain
DID estimators for the ATT that are also DR for inference is to choose ﬁrst-step estimators for the nuisance
parameters, say γb and βb , such that the limiting distribution of the resulting DR DID estimator τb
                                                                                                    dr,p
                                                                                                         is equivalent
to that of the infeasible DR DID estimator that uses the pseudo-true values of γb and βb , say γ ∗ and β ∗ . In a more

                                                               14

precise manner, in order to get DID estimators that are DR for inference, we need to guarantee that there will be
no estimation eﬀect from the ﬁrst stage.
      In Appendix A, we show that the estimation eﬀect associated with using generic ﬁrst-step estimators γb and
βb is given by η est
                 p
                     (W ; γ ∗ , β ∗ ) as deﬁned in (A.2). By paying closer attention to the exact form of η est
                                                                                                            p
                                                                                                                (W ; γ ∗ , β ∗ ),
one can see that if
                                                         h                 p             i
                                                    E     w1p − w0p (γ ∗ ) · µ̇ 0,∆ (β ∗ ) = 0,
                                   "                                                      #
                                         (1 − D)                            
                                                                   p       ∗
                                 E                        ∆Y − µ 0,∆ (β ) · π̇ (γ ∗ ) = 0,                                    (3.6)
                                     (1 − π (X ; γ ∗ ))2
                                                      h                                i
                                                   E w0p (γ ∗ ) · ∆Y − µ 0,∆ p
                                                                                  (β ∗ )    = 0,

then there will be no estimation eﬀect from the ﬁrst stage. As the ﬁrst component of X is assumed to be constant
and we adopt the working models (3.5), it follows that (3.6) reduces to
                                                                        
                                         D       exp (X ′ γ ∗ ) (1 − D)
                                E             −                           X   = 0,
                                      E [D] E [exp (X ′ γ ∗ ) (1 − D)]
                          h                                              i
                                                              ∗ 
                         E exp X ′ γ ∗ ∆Y − µ lin,p
                                                  0,∆   X ; β 0,∆    X D = 0  = 0.

However, as n → ∞, these two vectors of moment conditions follow from the ﬁrst-order conditions of the
                                                   wls,p
                                      γ ipt and βb 0,∆ , respectively, even when these working models are
optimization problems associated with b
                                                 wls,p
                             γ ipt and βb 0,∆ , we guarantee that b
misspeciﬁed. Hence, by using b                                    τ dr,p
                                                                    imp is doubly robust for inference as there is
                                                                            ∗,wls,p                                         ipt
no estimation eﬀect from replacing the pseudo-true parameters γ ∗,ipt and β 0,∆     with their estimators γb                      and
   wls,p
βb 0,∆ , respectively.
      The next theorem formalizes this discussion. Deﬁne
                                                                                        
                                                                                         wls,p
                       dr,p            p         p
                    τbimp = En w1 (D) − w0 D, X ; γb      ipt              lin,p
                                                                   ∆Y − µ 0,∆ X ; β 0,∆b             ,                        (3.7)
                                h                                                          i
                                                                                       ∗,wls,p
                     τ dr,p          p         p
                       imp = E w1 (D) − w0 D, X ; γ
                                                       ∗,ipt
                                                                  ∆Y − µ lin,p
                                                                         0,∆     X ; β 0,∆         ,

and let
          
                           ∗,wls,p dr,p
                                                                                        
                                                                                                   ∗,wls,p
                                                                                                           
   η dr,p
     imp   W ; γ ∗,ipt
                       , β 0,∆    , τ imp   = w p
                                                1 (D) − w p
                                                          0 D, X ; γ ∗,ipt
                                                                               ∆Y − µ lin,p
                                                                                      0,∆    X ; β 0,∆        − w1p (D) · τ dr,p
                                                                                                                            imp .

Theorem 2 Suppose Assumptions 1-3 and Assumptions A.1-A.2 stated in Appendix A hold, and that the working
nuisance models (3.5) are adopted. Then,
                                                     ∗,wls,p    p
    (a) If either Λ X ′ γ ∗,ipt = p (X ) a.s or X ′ β 0,∆     = m0,∆ (X ) a.s., then, as n → ∞,
                                                                   p
                                                             τbdr,p
                                                               imp → τ ,

and
                         √                           1 n                                       
                                 dr,p                                             ∗,wls,p dr,p
                             n(τbimp − τ dr,p
                                         imp ) =    √ ∑ η dr,p   W  ; γ ∗,ipt
                                                                              , β 0,∆    , τ imp + o p (1)
                                                      n i=1 imp
                                                               
                                                    d         p
                                                    → N 0,Vimp    ,

                                                                 15

                                                   2 
                                        ∗,wls,p dr,p
        p
where Vimp =E         η dr,p
                          W;γ
                        imp
                              ∗,ipt , β 0,∆ , τ imp       .
                                                                                                                             
                                                        ∗,wls,p    p                                 ∗,ipt , β ∗,wls,p , τ dr,p =
    (b)If both Λ X ′ γ ∗,ipt = p (X ) a.s and X ′ β 0,∆         = m0,∆ (X ) a.s., then η dr,p
                                                                                         imp   W ; γ           0,∆         imp
                                  p
η e,p (Y1 ,Y0 , D, X ) a.s. and Vimp is equal to the semiparametrically efficiency bound (2.12).

    Part (a) of Theorem 2 generalizes the cross-section results of Vermeulen and Vansteelandt (2015) to the DID
                                                                                                    √
framework. It states that the proposed DR DID estimators for the ATT, b   τ dr,p
                                                                            imp , is doubly robust,  n-consistent
                                                                   p
and asymptotically normal. It also states that the exact form of Vimp does not depend on which working models
                                              dr,p
are correctly speciﬁed, implying that τbimp is doubly robust not only in terms of consistency but also terms
of inference. An important consequence of this DR-for-inference property is that it allows one to treat the
summands of τbdr,p                                                                                    p
              imp as if they were independent and identically distributed, and, therefore, estimate Vimp by
                                          "                             2 #
                                 b p          dr,p      ipt b wls,p dr,p
                                 Vimp = En η imp W ; γb , β 0,∆ , τbimp       .

                                                 dr,p
    Part (b) of Theorem 2 indicates that τbimp is semiparametrically eﬃcient when the working model for the
propensity score, and the working models for the outcome regression for the comparison units are correctly
speciﬁed.

Remark 2 From the discussion above, it may be natural to directly use the moment conditions (3.6) to form
(generic) nonlinear generalized method of moment (GMM) estimators for γ and β . However, it is important to
emphasize that to justify the use of such estimation procedure, one must at least establish the local identiﬁcation
of the pseudo-true parameters, which, in turn, requires the matrix of derivatives of (3.6) having full column
rank. Importantly, such a condition may not hold for some working models. This is particularly the case when
one adopts the working models (3.5) and both speciﬁcations are correctly speciﬁed. Thus, care must be taken
when one attempts to use alternative, more general estimation techniques to generalize the DR inference results
discussed above.

Remark 3 As discussed in Appendix A of Graham et al. (2012), it is possible to use alternative speciﬁcations
for the propensity score, e.g., a probit working model. However, when one deviates from the logit speciﬁcation,
the optimization algorithm involved to estimate the nuisance parameters γ tends to be more computationally
                                                                                  ipt
demanding, as it involves numerical integration. As discussed above, γb                 clearly avoids such complications.

3.2 Improved DR DID estimators when repeated cross-section data are available
    In this section, we turn our attention to our proposed improved DR DID estimators for the ATT when only
repeated cross-section data are available. Similar to the panel data case, we consider the case where a researcher
is comfortable with the following speciﬁcations,
                                     exp (X ′ γ )                     rc      lin,rc       rc    ′ rc
              π (X , γ ) = Λ X ′γ ≡                  , and µ rc
                                                             d,t X ; β d,t  = µ d,t    X ; β d,t ≡ X β d,t .               (3.8)
                                    1 + exp (X ′ γ )

                                                              16

      We consider two improved DR DID estimators based on (2.8) and (2.9), namely
                                                                                               
                                                                                      wls,rc   wls,rc
                dr,rc
              τb1,imp = En w  rc            rc
                                          b0 D, T, X ; γb
                             b1 (D, T ) − w
                                                         ipt           lin,rc
                                                                 Y − µ 0,Y          b        b
                                                                                X ; β 0,1 , β 0,0         ,                                          (3.9)

and
                                                                                              
                                   D                                b ols,rc
                                                                                             b wls,rc
   τbdr,rc   bdr,rc
     2,imp = τ 1,imp +       En         −w  rc
                                          b1,1 (D, T )      rc
                                                          µ 1,1 X ; β 1,1            rc
                                                                               − µ 0,1 X ; β 0,1
                                 En [D]
                                                                                                        
                                           D        rc              rc         b ols,rc       rc       b wls,rc
                                − En            −w b1,0 (D, T )   µ 1,0 X ; β 1,0         − µ 0,0 X ; β 0,0          , (3.10)
                                         En [D]
where
                                                                                    
                                  γbipt   = arg max En DX ′ γ − (1 − D)exp X ′ γ ,
                                                γ ∈Γ
                                                                                                  
                                                                   ipt
                                 wls,rc                    Λ X ′ γb                  
                              βb 0,t      = arg min En                   Y − X ′ b 2 D = 0, T = t  ,
                                                b∈Θ                   ipt
                                                         1 − Λ X ′ γb
                                 ols,rc                h          2                 i
                              βb 1,t      = arg min En Y − X ′ b D = 1, T = t .
                                          b∈Θ
                  dr,rc                                                               dr,rc
Here, note that τb1,imp does not rely on OR models for the treated group while τb2,imp does. In addition, when one
            dr,rc        dr,rc       dr,p
compares τb1,imp and τb2,imp with τbimp , it is evident that the latter relies on a single OR model since we observe
Y1 and Y0 for all units; when only repeated cross-section data are available, one needs to model the OR in each
time period (and each treatment group). Another interesting feature worth mentioning is that we estimate the
OR parameters for the treated group via ordinary least squares, whereas we estimate the OR parameters for the
control group with weighted least squares. This follows from the fact that estimating the pseudo-true parameters
β ∗,rc
  1,t , t = 0, 1, does not lead to any estimation eﬀect, and therefore one can choose her favorite estimation

method. Given this observation and the linear speciﬁcation in (3.8), we ﬁnd it natural to estimate β ∗,rc
                                                                                                     1,t , t = 0, 1,

via OLS as this is the most widespread estimation procedure adopted by practitioners.
      Let
                             h                                                                                                     i
                                                                                                                 ∗,wls,rc     ∗,wls,rc
               τ dr,rc              rc
                 imp = E w1 (D, T ) − w0 D, T, X ; γ
                                                      rc                 ∗,ipt
                                                                                     Y − µ lin,rc
                                                                                              0,Y      T, X ; β 0,1       , β 0,0
                                                                      
and for β ∗,rc
          imp  =     β ∗,wls,rc
                       0,1      , β ∗,wls,rc
                                    0,0      , β ∗,ols,rc
                                                 1,1      , β ∗,ols,rc
                                                              1,0        , deﬁne
                                         
          η dr,rc
            1,imp W ; γ
                           ∗,ipt
                                 , β ∗,rc
                                     imp       = η rc,1 1 (W ; β 0,1
                                                                     ∗,wls,rc      ∗,wls,rc
                                                                               , β 0,0      ) − η rc,1
                                                                                                    0 (W ; γ
                                                                                                                ∗,ipt     ∗,wls,rc
                                                                                                                      , β 0,1          ∗,wls,rc
                                                                                                                                   , β 0,0      ),
                                         
          η dr,rc
            2,imp W ; γ
                           ∗,ipt
                                 , β ∗,rc
                                     imp       = η rc,2              ∗,rc         rc,2
                                                        1 (W ; β imp ) − η 0 (W ; γ
                                                                                               ∗,ipt     ∗,wls,rc
                                                                                                     , β 0,1          ∗,wls,rc
                                                                                                                  , β 0,0      ),

where η rc,1  rc,1  rc,2      rc,2
        1 , η 0 , η 1 , and η 0    are deﬁned as in (B.1)-(B.4) in the Appendix B.
                                           dr,rc           dr,rc
      Next theorem states that τb1,imp and τb2,imp are not only doubly robust consistent but also doubly robust for
                                                       dr,rc                                                                         dr,rc
inference. Furthermore, it states that τb2,imp is locally semiparametrically eﬃcient, whereas τb1,imp is not.

Theorem 3 Let n = n1 + n0 , where n1 and n0 are the sample sizes of the post-treatment and pre-treatment
periods, respectively. Suppose Assumptions 1-3 and Assumptions A.1-A.2 stated in Appendix A hold, that
        p
n1 /n → λ ∈ (0, 1) as n0 , n1 → ∞, and that the working nuisance models (3.8) are adopted. Then,

                                                                              17

                                                       ∗,wls,rc         ∗,wls,rc
      (a) If either Λ X ′ γ ∗,ipt = p (X ) a.s or X ′ β 0,1      − X ′ β 0,0      = mrc
                                                                                     0,∆ (X ) a.s., then, for j = 1, 2, as n → ∞,
                                                                           p
                                                               τbdr,rc
                                                                 j,imp → τ ,

and
                                   √                         1 n                                
                                           dr,rc                                            ∗,rc
                                       n(τb j,imp − τ ) =   √ ∑ η dr,rc    W  ; γ ∗,ipt
                                                                                        , β imp + o p (1)
                                                              n i=1 j,imp
                                                            d         rc
                                                                          
                                                            → N 0,V j,imp   ,
                                                  2 
                                               ∗,rc
        rc
where V j,imp =E         η dr,rc
                               W;γ
                           j,imp
                                     ∗,ipt , β imp       .
                                                                                          ∗,wls,rc
     (b) Suppose that Λ X ′ γ ∗,ipt = p (X ) a.s and, for all (d,t) ∈ {0, 1}2 , X ′ β d,t           = mrc
                                                                                                       d,t (X ) a.s.. Then,
                           
η dr,rc       ∗,ipt , β ∗,rc = η e,rc (Y, D, T, X ) a.s., and V rc
  2,imp W ; γ           imp                                    2,imp is equal to the semiparametrically efficiency bound
                             rc
(2.14). On the other hand, V1,imp does not attain the semiparametric efficiency bound.

                                                              dr,rc            dr,rc                            √
      In other words, Theorem 3 states that both τb1,imp and τb2,imp are doubly robust for the ATT,              n-consistent
                                                                                   rc , j = 1, 2, does not depend
and asymptotically normal. Similar to the panel data case, the exact form of the V j,imp
on which working models are correctly speciﬁed, implying that both τbdr,rc     bdr,rc
                                                                     1,imp and τ 2,imp are also doubly robust in

terms of inference.
                                           τ dr,rc
      Part (b) of Theorem 3 indicates that b 2,imp is semiparametrically eﬃcient when the working model for

the propensity score, and all working models for the outcome regressions, for both treated and comparison
units, are correctly speciﬁed. When compared to Theorem 2(b), it is evident that such a requirement is much
                                                                                                                        dr,rc
stronger than when panel data are available. Part (b) of Theorem 3 also indicates that, in general, τb1,imp is
                                                                                                            dr,rc
not locally semiparametrically eﬃcient. As so, we argue that, in practice, one should favor τb2,imp with respect
      dr,rc
to τb1,imp , as both estimators are doubly robust in terms of consistency and inference, but the former is locally
semiparametrically eﬃciency whereas the latter is not.
      We conclude this section by providing a precise characterization of the eﬃciency loss associated with using
τbdr,rc            bdr,rc
  1,imp instead of τ 2,imp when all working models are correctly speciﬁed. Here, our main goal is to illustrate

that by using an estimator that attempts to mimic the panel data setup and that does not explicitly exploit the
stationarity condition in Assumption 1(b), one may incur in substantial eﬃciency loss. As so, we argue that, in
                                                                           τ dr,rc
practice, one should favor estimators based on the DR moment (2.9)—such as b 2,imp —with respect to estimators

                                     τ dr,rc
based on the DR moment (2.8)—such as b 1,imp .

                                                                                             
Corollary 2 Suppose the assumptions in Theorem 3 hold. Furthermore, assume that Λ X ′ γ ∗,ipt = p (X ) a.s
                                     ∗,wls,rc
and, for all (d,t) ∈ {0, 1}2 , X ′ β d,t      = mrc
                                                  d,t (X ) a.s.. Then,
                                    "r                                       r                                #
                                          1  −  λ                              λ                       
   rc
 V1,imp      rc
        −V2,imp  = E [D]−1 ·Var                   mrc             rc
                                                      1,1 (X ) − m0,1 (X ) +       mrc          rc
                                                                                    1,0 (X ) − m0,0 (X ) D = 1 ≥ 0.
                                              λ                                1−λ

                                                                      18

Remark 4 We stress that the result in Corollary 2 does not depend on the fact that one is using the speciﬁcations
in (3.8). As we show in its proof, such a result remains true provided that the (generic) ﬁrst-step estimators
for the nuisance functions are correctly speciﬁed. Thus, Corollary 2 quantiﬁes the loss of eﬃciency associated
with using estimators based on τ dr,rc
                                 1     as deﬁned in (2.8)—which includes the estimator proposed by Zimmert
(2019)— instead of using estimators based on τ dr,rc
                                               2     as deﬁned in (2.9). Given that this loss of eﬃciency is
usually strictly positive, estimators based on τ dr,rc
                                                 1     are not, in general, semiparametrically eﬃcient. As we show
in the next section via Monte Carlo simulations, this loss of eﬃciency can be large.

4 Monte Carlo simulation study
    In this section, we conduct a series of Monte Carlo experiments in order to study the ﬁnite sample properties of
our proposed DR DID estimators. When panel data are available, we compare our proposed DR DID estimators
τbdr,p and τbdr,p
             imp given in (3.1) and (3.7), respectively, to the OR DID estimator (2.2), the Horvitz and Thompson

(1952) type IPW estimator (2.4), and the TWFE regression model (2.5). Given that the weights of the IPW
                                                                  ipw,p
estimator (2.4) are not normalized to sum up to one, τb                   can be unstable particularly when propensity score
estimates are relatively close to one. To assess the role played by the weights, we also consider the Hájek (1971)
type IPW estimator for the ATT
                                       ipw,p                                             
                                     τbstd   = En       b1p (D) − w
                                                        w         b0p (D, X ; γb) (Y1 −Y0 ) ,                          (4.1)

                              b0p (D, X ; γb) are given by (3.2) and are normalized to sum up to one.
                  b1p (D) and w
where the weights w
                                                                                                                        dr,rc
    When only repeated cross-section data are available, we compare our proposed DR DID estimators τb1
     dr,rc                                                                          dr,rc       dr,rc
and τb2      given in (3.3) and (3.4), and their further improved versions τb1,imp and τb2,imp given in (3.9) and (3.10),
to the OR DID estimator (2.2), the plug-in IPW estimator based on (2.3), and the TWFE regression model (2.5).
As in the case of panel data, we also consider the Hájek (1971) type IPW estimator for the ATT
                                       ipw,rc
                                     τbstd           brc
                                              = En [(w             brc
                                                       1 (D, T ) − w              b))Y ] ,
                                                                     0 (D, T, X ; γ                                    (4.2)

where the weights are the same as those in τbdr,rc
                                             1     .
    In all simulation exercises, we consider a logistic propensity score working model and a linear regression
working model for the outcome evolution. All observed covariates enter the working models linearly. With
                      dr,p   dr,rc
the exception of τbimp , τb j,imp , j = 1, 2, where we use the estimation methods proposed in Section 3.1 and in
Section 3.2, the OR models are estimated using ordinary least squares, and the propensity score working model
is estimated using maximum likelihood estimation. When panel data are available, we consider OR models for
∆Y instead of OR models for Y0 and Y1 separately.
    We consider sample size n equal to 1000. For each design, we conduct 10, 000 Monte Carlo simulations.
We compare the various DID estimators for the ATT in terms of average bias, median bias, root mean square
error (RMSE), empirical 95% coverage probability, the average length of a 95% conﬁdence interval, and the

                                                                 19

average of their plug-in estimator for the asymptotic variance. The conﬁdence intervals are based on the
normal approximation, with the asymptotic variances being estimated by their sample analogues. We also
compute the semiparametric eﬃciency bound under each design to allow one to assess the potential loss of
eﬃciency/accuracy associated with using ineﬃcient DID estimators for the ATT.

4.1 Simulation 1: panel data are available
    We ﬁrst discuss the case where panel data are available. For a generic W = (W1 ,W2 ,W3 ,W4 )′ , let

                             freg (W ) = 210 + 27.4 ·W1 + 13.7 · (W2 +W3 +W4 ) ,

                               f ps (W ) = 0.75 · (−W1 + 0.5 ·W2 − 0.25 ·W3 − 0.1 ·W4 ).

Let X = (X1 , X2 , X3 , X4 )′ be distributed as N (0, I4 ), and I4 be the 4 × 4 identity matrix. For j = 1, 2, 3, 4, let
               q                                                                                            3
Z j = Z̃ − E Z̃         Var Z̃ , where Z̃1 = exp (0.5X1 ), Z̃2 = 10 + X2 / (1 + exp (X1 )), Z̃3 = 0.6 + X1 X3 25
and Z̃4 = (20 + X2 + X4 )2 .
    Building on Kang and Schafer (2007), we consider the following data generating processes (DGPs):

     DGP1. Y0 (0) = freg (Z) + v (Z, D) + ε 0 ,              Y1 (d) = 2 · freg (Z) + v (Z, D) + ε 1 (d) , d = 0, 1,
                             exp ( f ps (Z))
                p (Z) =                        ,               D = 1 {p (Z) ≥ U } ;
                           1 + exp ( f ps (Z))
     DGP2. Y0 (0) = freg (Z) + v (Z, D) + ε 0 ,              Y1 (d) = 2 · freg (Z) + v (Z, D) + ε 1 (d) , d = 0, 1,

                             exp ( f ps (X ))
                p (X ) =                        ,              D = 1 {p (X ) ≥ U } ;
                           1 + exp ( f ps (X ))
     DGP3. Y0 (0) = freg (X ) + v (X , D) + ε 0 ,              Y1 (d) = 2 · freg (X ) + v (X , D) + ε 1 (d) , d = 0, 1,
                             exp ( f ps (Z))
                p (Z) =                        ,               D = 1 {p (Z) ≥ U } ;
                           1 + exp ( f ps (Z))
     DGP4. Y0 (0) = freg (X ) + v (X , D) + ε 0 ,              Y1 (d) = 2 · freg (X ) + v (X , D) + ε 1 (d) , d = 0, 1,
                            exp ( f ps (X ))
                p (X ) =                       ,          D = 1 {p (X ) ≥ U } ,
                          1 + exp ( f ps (X ))
where ε 0 , ε 1 (d), d = 0, 1 are independent standard normal random variables, U is an independent standard
uniform random variable, and for a generic W , v (W, D) is an independent normal random variable with
mean D · freg (W ) and variance one. The available data are {Y0,i ,Y1,i , Di , Zi }ni=1 , where Y0 = Y0 (0), and Y1 =
DY1 (1) + (1 − D)Y1 (0). In the aforementioned DGPs, the true ATT is zero, and v plays the role of time-invariant
unobserved heterogeneity.
    Given that we focus on the empirically relevant setting where the observed covariates Z enter all working
models linearly, it is clear that in DPG1, both propensity score (PS) and OR working models are correctly
speciﬁed. In DGP2, only the OR working model is correctly speciﬁed, whereas in DGP3 only the PS working
model is correctly speciﬁed. In DGP4, all working models are misspeciﬁed. The simulation results are presented

                                                          20

in Table 1.

           Table 1: Monte Carlo results under designs DGP1 − DGP4 with panel data. Sample size n = 1, 000.
                            DGP1: OR correct, PS correct                                        DGP2: OR correct, PS incorrect
                         Semiparametric Eﬃciency Bound: 11.1                                 Semiparametric Eﬃciency Bound: 11.6
             Av. Bias     Med. Bias     RMSE       Asy. V   Cover     CIL         Av. Bias     Med. Bias      RMSE      Asy. V      Cover      CIL
bτ fe        -20.952       -20.965      21.123     6392.2   0.000    9.906        -19.286       -19.287       19.468    6640.3      0.000     10.095
bτ reg        -0.001        -0.001      0.100       10.2    0.950    0.396         -0.001        -0.001        0.100     10.1       0.949      0.394
τbipw,p        0.026         0.195      2.774      8078.0   0.952   10.441          2.010         2.054        3.298    7048.3      0.838      9.819
   ipw,p
τbstd          0.008        -0.013      1.132      1286.4   0.948    4.309         -0.794        -0.798        1.225    891.7       0.856      3.623
τbdr,p        -0.001         0.000      0.106       11.1    0.947    0.412         -0.001        -0.002        0.104     10.7       0.947      0.404
τbdr,p
   imp        -0.001         0.000      0.106       10.9    0.945    0.409         -0.001        -0.001        0.104     10.6       0.945      0.404

                            DGP3: OR incorrect, PS correct                                     DGP4: OR incorrect, PS incorrect
                         Semiparametric Eﬃciency Bound: 11.1                                 Semiparametric Eﬃciency Bound: 11.6
             Av. Bias     Med. Bias     RMSE      Asy. V    Cover     CIL         Av. Bias     Med. Bias      RMSE      Asy. V      Cover      CIL
τb f e       -13.170       -13.194      13.364    12687.9   0.004   13.960        -16.385       -16.393       16.538    13160.7     0.000     14.217
τbreg         -1.384        -1.365      1.868      1514.4   0.800    4.816         -5.204        -5.171        5.364     1666.6     0.015      5.053
τbipw,p        0.011         0.158      3.198     10062.5   0.947   11.777         -1.085        -1.017        2.656     6151.4     0.949      9.308
   ipw,p
τbstd         -0.030        -0.032      1.427      1988.0   0.945    5.484         -3.954        -3.949        4.215     2156.5     0.228      5.717
bτ dr,p       -0.051        -0.046      1.214      1400.9   0.942    4.613         -3.188        -3.183        3.454     1704.9     0.308      5.075
τbdr,p
   imp        -0.071        -0.064      1.015       971.2   0.942    3.858         -2.529        -2.514        2.720     970.1      0.274      3.856
Notes: Simulations based on 10,000 Monte Carlo experiments. τb f e is the TWFE outcome regression estimator of τ f e in (2.5), τbreg is the OR-DID
estimator (2.2), τbdr,p is the IPW DID estimator (2.4), τbstd
                                                          ipw,p
                                                                is the standardized IPW DID estimator (4.1), τbdr,p is our proposed DR DID estimator
             dr,p
(3.1), and τbimp is our proposed DR DID estimator (3.7). We use a linear OR working model and a logistic PS working model, where the unknown
parameters are estimated via OLS and maximum likelihood, respectively, except for b   τ dr,p
                                                                                        imp , where we use the estimation methods described in Section
3.1. Finally, “Av. Bias”, “Med. Bias”, “RMSE”, “Asy. V”, “Cover” and “CIL’, stand for the average simulated bias, median simulated bias, simulated
root mean-squared errors, average of the plug-in estimator for the asymptotic variance, 95% coverage probability, and 95% conﬁdence interval length,
respectively. See the main text for further details.

      First, note that the TWFE estimator τb f e is severely biased and its conﬁdence interval for the ATT has almost
zero coverage in all analyzed DGPs. These results should not be unexpected, because, as discussed in Remark
        fe
1, b
   τ         implicitly rules out covariate-speciﬁc trends, and when these are relevant, like in the considered DGPs,
                             τ f e is not the ATT. As so, policy evaluations based on b
the estimand associated with b                                                        τ f e can be misleading.
      The results in Table 1 also suggest that, when both the OR and PS working models are correctly speciﬁed,
                                                                                                                reg    dr,p        dr,p
all semiparametric estimators for the ATT show little to no Monte Carlo bias, but τb , τb                                     and τbimp dominate
                                       ipw,p        ipw,p
the IPW DID estimators τb                      and τbstd on the basis of bias, root mean square error, asymptotic variance, and
length of the conﬁdence interval. Indeed, both IPW DID estimator seem to be substantially less eﬃcient than
τbreg , τbdr,p and τbdr,p                                                                        breg tends to be more
                     imp . The performance of these last three estimators are very close, though τ
                                                                                 reg
eﬃcient than the other two DR DID estimators. Given that τb                            exploits additional assumptions when compared
       dr,p         dr,p                                                                                                                       ipw,rc
to τb         and τbimp , such a result is not unexpected. Also note that the Hájek (1971) type IPW estimator τbstd
is more stable than the Horvitz and Thompson (1952) type IPW estimator τbipw,rc : the RMSE (the asymptotic
                        ipw,rc                                                                     ipw,rc
variance) of τb                  are more than two (four) times bigger than that of τbstd                   . Such a ﬁnding highlights the
practical importance of using weights that are normalized to sum up to one.

                                                                         21

                                                                                                                                      dr,p        dr,p
     When only the OR working model is correctly speciﬁed, our proposed DR DID estimators τb                                                 and τbimp are
competitive with the OR DID estimator τbreg , while the IPW DID estimators are biased, as one should expect. On
the other hand, when only the PS working model is correctly speciﬁed, the IPW and DR estimators show little
                          reg                                                                                           dr,p            dr,p
to no bias, while τb            displays non-negligible bias. Here, it is worth emphasizing that τb                            and τbimp drastically
                  ipw,p           ipw,p        dr,p                                                                                              dr,p
outperform τb             and τbstd , with τbimp also showing substantial improvements with respect to both τb                                           and
  ipw,p
τbstd   . When one compares the two IPW estimators, the role played by the normalized weights is again clear,
     ipw,p                                                   ipw,p
as τbstd     is again much more “stable” than τb                     .
     When both OR and PS working models are misspeciﬁed, not unexpectedly all estimators have non-negligible
biases and inference procedures are, in general, misleading. In this scenario, our DR DID estimators have smaller
biases and RMSE than the OR and the normalized IPW DID estimators, with τbdr,p                    bdr,p .
                                                                          imp strictly dominating τ
                                                                                           ipw,p
However, the Horvitz and Thompson (1952) IPW DID estimator τb                                      seems to perform best in this DGP.
                                                                                                                                      dr,p        dr,p
     In terms of eﬃciency, the results in Table 1 show that the estimated asymptotic variance of τb                                          and τbimp are
very close to the semiparametric eﬃciency bound when both the PS and OR regression are correctly speciﬁed,
which is in agreement with our locally eﬃciency results in Theorems 2 and A.1 (in the Appendix). When the
                                                                                                         dr,p           dr,p
PS is misspeciﬁed but the OR is not, the estimated asymptotic variances of τb                                   and τbimp are still close to the
semiparametric eﬃciency bound in this particular DGP, though we emphasize that this is not predicted by our
results and can be a feature of this particular DGP. Finally, we note that when the OR is misspeciﬁed but the PS
                                                                                                                dr,p           dr,p
is not, the estimated asymptotic variances of our proposed DR DID estimators τb                                        and τbimp are far from the
                                                      dr,p                         dr,p
semiparametric eﬃciency bound, with τbimp outperforming τb                                in terms of eﬃciency in this particular DGP.

4.2 Simulation 2: repeated cross-section data are available
     We now analyze the performance of the DID estimators for the ATT when one only observes repeated cross-
section data. To do so, we consider the same DGPs as in the panel data framework, but instead of observing
data on (Y0 ,Y1 , D, Z), one observes data on (Y0 , D, Z) if T = 0, or on (Y1 , D, Z) if T = 1, where T = 1 {UT ≤ λ },
and UT is a standard uniform random variable, and λ ∈ (0, 1) a ﬁxed constant.
     Table 2 presents the simulation results with λ = 0.5 and with n ≡ n1 + n0 = 1, 000.8 Overall, the simulation
exercise reveals that the eﬃciency bound, RMSE, asymptotic variance, and conﬁdence interval length of the
considered DID estimators are much larger when only repeated cross-section data are available than when panel
data are available. In light of Corollary 1, such a result should be expected, though the magnitude of such
                                                                                                          τ f e is
loss of eﬃciency can be striking. In addition, the results in Table 2 reveal that: (i) the TWFE estimator b
severely biased for the ATT in all DGPs, just like in the panel data case; (ii) the IPW estimator with standardized
             ipw,rc                                                      ipw,rc
weights τbstd         is much more stable and eﬃcient than τb                     in all DGPs, and, as one should expect, when the PS

8 Simulation results with λ = 0.25 and λ = 0.75 reached analogous conclusions to those discussed below and are available upon
  request.

                                                                         22

            Table 2: Monte Carlo results under designs DGP1 − DGP4 with repeated cross section data. Sample
            size n = 1, 000, and λ = 0.5.
                           DGP1: OR correct, PS correct                                                  DGP2: OR correct, PS incorrect
                        Semiparametric Eﬃciency Bound: 44.4                                           Semiparametric Eﬃciency Bound: 46.4
             Av. Bias   Med. Bias   RMSE        Asy. V          Cover          CIL         Av. Bias   Med. Bias   RMSE      Asy. V        Cover       CIL
τb f e       -20.792     -20.741    21.099      12773.9         0.000         13.996       -19.178     -19.125    19.529    13240.7       0.001      14.247
τbreg          0.026      -0.030     7.588      57417.8         0.951         29.675        -0.024      -0.057     8.191    66577.5       0.948      31.945
τbipw,rc      -0.662      -0.932    55.971     3090077.6        0.949        217.762         1.820       1.506    55.050   3023548.1      0.949     215.449
bτ ipw,rc
   std        -0.050      -0.125     9.648      92235.7         0.949         37.560        -0.812      -0.698     9.814    94343.0       0.946      38.031
bτ dr,rc
   1           0.013      -0.007     3.041       9222.0         0.950         11.893        -0.010      -0.022     3.281    10686.4       0.949      12.799
bτ dr,rc
   2           0.004       0.003     0.216        44.4          0.944         0.824         0.000       0.001      0.211      42.3        0.945      0.805
bτ dr,rc
   1,imp       0.014      -0.008     3.041       9220.1         0.951         11.892        -0.009      -0.022     3.282    10686.2       0.949      12.799
bτ dr,rc
   2,imp       0.005       0.002     0.216        42.1          0.937         0.803          0.000       0.001     0.213      41.3        0.940      0.796

                           DGP3: OR incorrect, PS correct                                               DGP4: OR incorrect, PS incorrect
                        Semiparametric Eﬃciency Bound: 44.4                                           Semiparametric Eﬃciency Bound: 46.4
             Av. Bias   Med. Bias   RMSE        Asy. V          Cover          CIL         Av. Bias   Med. Bias   RMSE      Asy. V        Cover       CIL
τb f e       -13.131     -13.092    14.058      25446.9         0.260         19.766       -16.330     -16.354    17.126    26347.3       0.114      20.112
τbreg         -1.376      -1.397     8.137      64143.7         0.942         31.378        -5.338      -5.437     9.977    72665.8       0.908      33.397
τbipw,rc      -0.973      -1.452    57.262     3241967.3        0.947        223.050        -1.391      -0.980    55.178   3101777.5      0.952     218.233
τbipw,rc
   std         0.051      -0.011     9.428      86806.4         0.943         36.483        -4.149      -4.387    10.520    94034.1       0.930      37.971
bτ dr,rc
   1          -0.086      -0.083     5.692      31830.9         0.945         22.060        -3.342      -3.375     7.071    38663.1       0.916      24.290
bτ dr,rc
   2          -0.029      -0.022     4.742      21869.3         0.942         18.261        -3.275      -3.249     6.016    24194.2       0.886      19.159
bτ dr,rc
   1,imp      -0.119      -0.102     4.837      23038.9         0.945         18.804        -2.689      -2.708     5.564    23473.3       0.913      18.979
bτ dr,rc
   2,imp      -0.076      -0.081     4.062      15765.2         0.944         15.550        -2.614      -2.610     4.845    15769.1       0.892      15.552
                                                                     fe                                                          reg
Notes: Simulations based on 10,000 Monte Carlo experiments. τb is the TWFE outcome regression estimator of τ f e in (2.5), τb is the OR-DID estimator
         dr,rc                                                                   ipw,rc                                                 dr,rc dr,rc dr,rc
(2.2), τb      is the IPW DID estimator based on the sample analogue of (2.3), τbstd is the standardized IPW DID estimator (4.2), and τb1 , τb2 , τb1,imp and
τbdr,rc
  2,imp are our proposed DR DID estimators given in (3.3), (3.4), and in (3.9) and (3.10) in Section 3.2. We use a linear OR working model and a logistic PS
                                                                                                                             dr,rc     dr,rc
working model, where the unknown parameters are estimated via OLS and maximum likelihood, respectively, except for τb1,imp and τb2,imp where we use the
 estimation methods described in Section 3.2. Finally, “Av. Bias”,“Med. Bias”, “RMSE”, “Asy. V”, “Cover” and “CIL’, stand for the average simulated bias,
 median simulated bias, simulated root mean-squared errors, , average of the plug-in estimator for the asymptotic variance, 95% coverage probability, and 95%
 conﬁdence interval length, respectively. See the main text for further details.

working model is misspeciﬁed, these IPW estimators display non-negligible biases; (iii) as one should expect,
the OR DID estimator displays non-negligible bias when the OR working models are misspeciﬁed; (iv) all four
DR DID estimators display little to no bias when one of the working models is correctly speciﬁed, but the
                                                        dr,rc             dr,rc
locally eﬃcient DR DID estimators τb2                           and τb2,imp present important eﬃciency gains when compared to all
                                                dr,rc             dr,rc
other DID estimators, including τb1                     and τb1,imp . These gains in eﬃciency are more pronounced when the OR
models are correctly speciﬁed. The simulation results also show that (v) when one compares the performance
                                                                     dr,rc             dr,rc                                                           dr,rc
of the further improved DR DID estimators τb1,imp and τb2,imp with the “traditional” DR DID estimators τb1
    τ dr,rc
and b 2     , it is clear that appropriately choosing the estimation methods for the nuisance parameters can have
practical consequences, especially when the outcome regression working models are misspeciﬁed.
       In terms of eﬃciency, the results in Table 2 highlight that, when all working models are correctly speciﬁed,
the estimated asymptotic variances of τbdr,rc
                                        2     and τbdr,rc
                                                    2,imp are indeed close to the semiparametric eﬃciency bound,
                                           dr,rc          dr,rc
but the asymptotic variances τb1                   and τb1,imp are substantially higher than the semiparametric eﬃciency bound;
these ﬁndings are in agreement with our locally eﬃciency results in Theorems 3 and A.2 (in the Appendix).

                                                                                  23

                                                                                                             dr,rc
Similarly to the panel data case, we ﬁnd that, in this speciﬁc DGP, the estimated asymptotic variances of τb2
and τbdr,rc
      2,imp are still close to the semiparametric eﬃciency bound when the outcome regressions are correctly

speciﬁed but the PS is not, but not when the PS is correctly speciﬁed but the outcome regressions are not.

5 Empirical illustration: the effect of job training on earnings
   In a very inﬂuential study, LaLonde (1986) analyzes whether diﬀerent treatment eﬀect estimators based
on observational data are able to replicate the experimental ﬁndings of the NSW job training program on post
treatment earnings. His negative results led to an increased awareness of the potential pitfalls of observational
data and helped spur the use of randomized controlled trials among economists. In addition, alternative policy
evaluation tools arose to overcome “LaLonde’s critique” of observational estimators. Two prominent examples
are the propensity score matching (PSM), see e.g. Dehejia and Wahba (1999, 2002) (henceforth DW) and
the diﬀerence-in-diﬀerences matching, see e.g. Heckman et al. (1997) and Smith and Todd (2005) (henceforth
ST). For instance, DW show that PSM can replicate the experimental benchmark of the NSW for a particular
subsample of the original data. ST, on the other hand, cast doubt on the “generalizability” of DW PSM results
to a larger population and argue that the conclusions may be sensitive to the propensity score speciﬁcation.
ST also argue that for the NSW data, diﬀerence-in-diﬀerences matching estimators may be more suitable than
cross-section PSM, as they can account for time-invariant unobserved confounding factors.
   Motivated by ST ﬁndings, in what follows, we focus on DID estimators and evaluate whether our proposed
DR DID estimators can better reduce the selection bias when compared to other DID estimation procedures. We
analyze three diﬀerent experimental samples — the original LaLonde experimental sample, the DW sample, and
the “early random assignment” (early RA) subsample of the DW sample considered by ST — and consider data
from the Current Population Survey (CPS) to form a non-experimental comparison group. The pre-treatment
covariates in the data include age, years of education, real earnings in 1974, and dummy variables for high
school dropout, married, black, and Hispanic. The outcome of interest is real earnings in 1978. We also observe
real earnings in 1975, which we use as the pre-treatment outcome Y0 . The experimental benchmark for the ATT
is equal to $886 (s.e. $488), $1794 (s.e. $671), and $2748 (s.e. $1005) for the LaLonde, DW, and early RA
sample, respectively. For additional description and summary statistics for each sample, see Smith and Todd
(2005).
   Following ST, we focus on estimating the average “evaluation bias” of diﬀerent DID estimators. This is only
made possible given the availability of experimental data. First, randomization ensures that both “treatment”
groups are comparable in terms of self selection. Second, given that randomized-out individuals did not receive
training via NSW, the impact of NSW is known to be zero in this group. Thus, applying diﬀerent DID estimators
to data from randomized-out individuals (our pseudo treated group in this exercise) and nonexperimental CPS

                                                       24

comparison observations (our comparison group in this exercise) should produce an estimated ATT equal to
zero, if these DID estimators are consistent. Deviations from zero are what we call evaluation bias.9
                                                                                                                            dr,p
    Like in the Monte Carlo simulation exercises, we compare our proposed DR DID estimators τb                                     and
τbdr,p                        b f e based on (2.5), the OR DID estimator τbreg as deﬁned in (2.2), and the
  imp with the TWFE estimator τ

Horvitz and Thompson (1952) type IPW DID estimator proposed by Abadie (2005), τbipw,p , as deﬁned in (2.4).
                                                                      ipw,p
We also consider the Hájek (1971) type IPW estimator τbstd                    as deﬁned in (4.1). We assume that the outcome
models are linear in parameters and that the propensity score follows a logistic speciﬁcation. The unknown
parameters are estimated using ordinary least squares (OLS) and maximum likelihood, respectively, except in
τ dr,p
b imp , where we use the estimation methods described in Section 3.1.

    In order to assess the sensitivity of the ﬁndings with respect to the model speciﬁcations, we consider three
diﬀerent speciﬁcations for how covariates enter into each model: (i) a linear speciﬁcation where all covariates
enter the models linearly; (ii) a speciﬁcation in the spirit of DW, which adds to the linear speciﬁcation a dummy
for zero earnings in 1974, age squared, age cubed divided by 1000, years of schooling squared, and an interaction
term between years of schooling and real earnings in 1974; and (iii) an “augmented DW” speciﬁcation, which
adds to the “DW” speciﬁcation the interactions between married and real earnings in 1974, and between married
and zero earnings in 1974 — these two interaction terms were used in Firpo (2007).
    Table 3 summarizes the results. Standard errors are reported in parentheses and the estimated evaluation
biases relative to the experimental ATT benchmark are reported in brackets. As argued by ST, these “relative
biases” are useful for comparing DID estimators within each sample, but as the experimental benchmark
estimates for the ATT vary substantially among the three experimental samples, they should not be used for
comparing DID estimators across samples.
    Table 3 highlights some interesting patterns. First, estimators based on two-way ﬁxed eﬀect regression
models tend to be very stable across speciﬁcations, but usually display large positive and statistically signiﬁcant
evaluation biases. Second, DID estimators based on the regression approach tend to lead to the most precise
estimates. However, for the LaLonde sample, point estimates are severely biased downward, leading to statisti-
                                                                              ipw,p
cally signiﬁcant evaluation biases. Abadie’s IPW estimators τb                        for the ATT tend to have the largest standard
errors across all considered estimators, but their evaluation biases are relatively small. Like in our Monte Carlo
                                                                       ipw,p
simulation results, considering normalized weights as in τbstd                  can improve the stability of the IPW estimators
τbipw,p . Finally, note that our proposed DR DID estimators share the favorable bias properties of Abadie’s IPW

9 An alternative way to estimate “evaluation bias” is to compare the ATT using the experimental data with ATT using data from
  randomized-in and nonexperimental comparison units. This is the approach taken by LaLonde (1986) and Dehejia and Wahba (1999,
  2002). A disadvantage of this approach compared to the one we and Smith and Todd (2005) use is that experimental ATT estimates
  are also random and may diﬀer from the “true” ATT. Thus, the computation of “true” evaluation biases is much more challenging if
  not impossible. In any case, results treating the experimental ATT as true eﬀects lead to similar conclusions and are available upon
  request.

                                                                 25

                                             Table 3: Evaluation bias of diﬀerent diﬀerence-in-diﬀerences estimators for the eﬀect of of training
                                             on real earnings in 1978. NSW data with CPS comparison group.
                                      Results for Lalonde sample                                                   Results for DW sample                                            Results for Early RA sample
                                      Evaluation Bias: ATT= 0                                                  Evaluation Bias: ATT= 0                                                Evaluation Bias: ATT= 0

      Spec.      τbdr,p        dr,p
                             τbimp          τbreg      τbipw,p     τbipw,p
                                                                     std        τb f e          τbdr,p      dr,p
                                                                                                          τbimp        τbreg   τbipw,p   τbipw,p
                                                                                                                                           std        τb f e           τbdr,p       dr,p
                                                                                                                                                                                  τbimp       τbreg     τbipw,p    τbipw,p
                                                                                                                                                                                                                     std      τb f e

      Lin.       -871        -901          -1301      -1108         -1022      868               253      253          -230      188      155        2092             -434        -441       -831       -516       -515       1136
                 (396)       (394)         (350)       (409)        (398)     (353)             (451)    (452)        (408)     (459)    (452)       (459)            (605)       (607)      (583)      (611)      (607)     (730)
                [-98%]     [-102%]       [-147% ]    [-125%]      [-115%]     [98%]            [14%]     [14%]       [-13%]    [10%]     [9%]       [117%]           [-16%]      [-16%]     [-30%]     [-19%]     [-19%]     [41%]
