<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1812.01723.pdf
backend: pdftotext
part: 2/3
-->

# Front Matter 2

26    DW          -626        -591          -830        -732         -564      868              408       520          402      -34       481        2092              -246       -176       -264       -495        -223      1136
                 (496)       (467)         (360)       (534)        (487)     (359)            (691)     (588)        (426)    (845)     (672)       (471)            (724)      (683)       (596)      (781)      (718)     (751)
                [-71%]      [-67%]        [-94%]      [-83%]       [-64%]     [98%]           [ 23%]     [29%]        [22%]    [-2%]     [27%]      [117%]            [-9%]      [-6%]      [-10%]     [-18%]      [-8%]     [41%]

      ADW         -597        -599         -1041        -685        -558       868               514      524           27       97       502        2092              -148       -144       -498       -337        -165      1136
                 (491)       (470)          (358)      (523)        (485)     (352)             (663)    (582)        (428)    (793)     (653)      (458)             (701)      (677)       (591)      (740)      (700)     (728)
                [-67%]      [-68%]        [-118%]     [-77%]       [-63%]     [98%]            [29%]     [29%]        [2%]     [5%]      [28%]     [117% ]            [-5%]      [-5%]      [-18%]     [-12%]      [-6%]     [41%]
      Notes: The results (standard errors are in parentheses) represent the estimated average eﬀect of being in the experimental sample (i.e. the estimated evaluation bias) on the 1978 earnings where the experimental control
      group is compared with untreated non-experimental CPS sample. The estimated evaluation biases relative to the experimental ATT benchmark, in percentage terms, are reported in brackets. τb f e is the TWFE outcome
      regression estimator of τ f e in (2.5), τbreg is the OR-DID estimator (2.2), τbdr,p is the IPW DID estimator (2.4), τbdr,p                                             bdr,p is our proposed DR DID estimator (3.1), and τbdr,p
                                                                                                                            std is the standardized IPW DID estimator (4.1), τ                                                   imp
      is our proposed DR DID estimator (3.7). We use a linear OR working model and a logistic PS working model, where the unknown parameters are estimated via OLS and maximum likelihood, respectively, except for
        dr,p
     τbimp , where we use the estimation methods described in Section 3.1. For each DID estimator, we report three diﬀerent speciﬁcations depending on how covariates are included: “lin.” speciﬁcation, where all covariates
      enter the model linearly; “DW” speciﬁcation, which adds to the linear speciﬁcation a dummy for zero earnings in 1974, age squared, age cubed divided by 1000, years of schooling squared, and an interaction term
      between years of schooling and real earnings in 1974; and the “ADW” speciﬁcation, which adds to the “DW” speciﬁcation the interactions between married with real earnings in 1974, and between married and zero
      earnings in 1974.

                                                                                                              dr,p
estimator, but at the same time, have smaller standard errors than IPW estimators. When we compare τb
with τbdr,p                                                     bdr,p
       imp , we note that the further improved DR DID estimator τ imp tends to have smaller standard errors,

particularly when one adopts the “DW” or the “augmented DW” speciﬁcations. Taken together, the results
using the NSW job training data suggest that our proposed DR DID estimators are an attractive alternative to
existing DID procedures.

6 Concluding remarks
    In this article, we proposed doubly robust estimators for the ATT in diﬀerence-in-diﬀerences settings where
the parallel trends assumption holds only after conditioning on a vector of pre-treatment covariates. Our
proposed estimators remain consistent for the ATT when either (but not necessarily both) a propensity score
model or outcome regression models are correctly speciﬁed, and achieve the semiparametric eﬃciency bound
when the working models for the nuisance functions are correctly speciﬁed. We derived the large sample
properties of the proposed estimators in situations where either panel data or repeated cross-section data are
available, and showed that by paying particular attention to the estimation methods used to estimate the nuisance
parameters, one can form DID estimators for the ATT that are not only DR consistent and locally semiparametric
eﬃcient, but also DR for inference. We illustrated the attractiveness of our proposed causal inference tools via
a simulation exercise and with an empirical application.
    Our results can be extended to other situations of practical interest. A leading case is when researchers are
interested in understanding treatment eﬀect heterogeneity with respect to continuous covariates X1 , where X1 is
a (strict) subset of available covariates X . Here, the parameter of interest is the conditional average treatment
eﬀect on the treated CATT(X1 ) ≡ E [Y (1) −Y (0) |X1 , D = 1] and because of its inﬁnite dimensional nature, the
estimation and inference tools proposed in this paper are not directly applicable. However, by combining the
DR DID formulation proposed in this paper with the methodology put forward by Chen and Christensen (2018),
one can propose uniformly valid inference procedures not only for the CATT but also for possibly nonlinear
functionals of the CATT such as (higher order) partial derivatives, conditional average (higher order) partial
derivatives, and partial derivatives of its log.
    Another interesting extension is when researchers want to adopt data-adaptive, “machine learning” ﬁrst-step
estimators instead of the parametric models discussed in this paper. Here, the main challenge is to derive the
inﬂuence function of the DR DID estimator for the ATT, as “machine learning” estimators are, in general, in a
non-Donsker classes of functions. We envision that one can bypass such technical complications by combining
the results derived in this paper with those in Chernozhukov et al. (2017), Belloni et al. (2017), and Tan (2019),
for example; see e.g. Zimmert (2019) for some recent results in this direction. We leave the detailed analysis of
these extensions to future work.

                                                       27

 A Appendix A: Asymptotic Properties of the DR DID estimators based on
        generic first-step estimators
                                                     p
    Let g (x) be a generic notation for π (x), µ d,t    (x) and µ rcd,t (x), d,t = 0, 1. Analogously and with some
                                                                                      
                                                                         p         p                    rc 
 abuse of notation, let g (x; θ ) be a generic notation for π (x; γ ), µ d,t  x, β d,t   and µ rc
                                                                                               d,t x, β d,t , d,t = 0, 1. Let

 W = (Y0 ,Y1 , D, X ) in the panel data case and W = (Y, T, D, X ) in the repeated cross-section data case. Denote
                                                        p
 the support of X by X and for a generic Z, let kZk = trace (Z ′ Z) denote the Euclidean norm of Z.
      Let
                                                                                                     
           h p (W ; κ p ) = w1p (D) − w0p (D, X ; γ ) ∆Y − µ 0,∆                p          p
                                                                                   (X ; β 0,0     p
                                                                                              , β 0,1 ) ,
                                                                                                   rc      rc 
      hrc 1 W ; κ rc 1 = (wrc                          rc                            rc
                                       1 (D, T ) − w0 (D, T, X ; γ )) Y − µ 0,Y T, X ; β 0,0 , β 0,1 ,
                                                                 rc     rc                   rc      rc 
      hrc 2 W ; κ rc 2 = D E [D] · µ rc                                              rc
                                                       1,∆ X ; β 1,1 , β 1,0 − µ 0,∆ X ; β 0,1 , β 0,0
                                                                          rc                                            rc 
                                  +wrc                        rc
                                        1,1 (D, T ) Y − µ 1,1 X ; β 1,1          − wrc                        rc
                                                                                      1,0 (D, T ) Y − µ 1,0 X ; β 1,0

                                                                       rc         rc                                       rc      rc 
                                  − wrc   0,1 (D, T, X ; γ ) Y − µ 0,1 X ; β 0,1         − wrc 0,0 (D, T, X ; γ ) Y − µ 0,0 X ; β 0,0
                                 ′
                       p′     p′                                rc′ ′                                                   
                                                                              rc 2 = γ ′ , β rc′ , β rc′ , β rc′ , β rc′ ′ . In obvious notation,
 where κ p = γ ′ , β 0,0  , β 0,1    , κ rc 1 = γ ′ , β rc′
                                                        0,0 , β 0,1 and κ                     0,0     0,1    1,1     1,0
                                                                                                                                          
 the vector of pseudo-true parameter10 is given by κ ∗,p , κ 0∗,rc 1 , and κ ∗,rc 2 . Let ḣ p (W ; κ p ) = ∂ h p (W ; κ p ) ∂ κ p
                                
 and deﬁne ḣrc j W ; κ rc j , j = 0, 1, analogously.

 Assumption A.1 (i) g (x) = g (x; θ ) is a parametric model, where θ ∈ Θ ⊂ Rk , Θ being compact; (ii) g (X ; θ )
 is a.s. continuous at each θ ∈ Θ; (iii) there exists a unique pseudo-true parameter θ ∗ ∈ int (Θ); (iv) g (X ; θ ) is
 a.s. twice continuously differentiable in a neighborhood of θ ∗ , Θ∗ ⊂ Θ; (v) the estimator θb is strongly consistent
 for the θ ∗ and satisfies the following linear expansion:
                                      √                    1 n
                                         n θb − θ ∗ = √ ∑ lg (Wi ; θ ∗ ) + o p (1) ,
                                                               n i=1
                                                                                 
 where lg (·; θ ) is such that E [lg (W ; θ )] = 0, E lg (W ; θ ∗ ) lg (W ; θ ∗ )′ exists and is positive definite and
                                              ∗
            h                                                   i
 limδ →0 E supθ ∈Θ∗ :kθ −θ ∗ k≤δ klg (W ; θ ) − lg (W ; θ ∗ )k2 = 0. In addition, (vi) for some ε > 0, 0 < π (X ; γ ) ≤
 1 − ε a.s., for all γ ∈ int (Θ ps ), where Θ ps denotes the parameter space of γ .
                                                                        h           i                                 
 Assumption A.2 (i) When panel data are available, assume that E kh p (W ; κ ∗,p )k2 < ∞ and E supκ ∈Γ∗,p ḣ p (W ; κ ) <
 ∞, where Γ∗,p is a small neighborhood of κ ∗,p . (ii) When cross-section data are available, assume that, for
            h                      2i                                        
 j = 1, 2, E hrc, j W ; κ ∗,rc, j      < ∞ and E supκ ∈Γ∗,rc j ḣrc, j (W ; κ ) < ∞, where Γ∗,rc j is a small neighbor-
 hood of κ ∗,rc j .

      Assumptions A.1-A.2 are standard in the literature, see e.g. Abadie (2005), Wooldridge (2007),
 Bonhomme and Sauder (2011), Graham et al. (2012) and Callaway and Sant’Anna (2018).                                            Assumption

10 Note that we allow for possible misspeciﬁcation when we deﬁne pseudo-true parameters.

                                                                       28

A.1 requires that the ﬁrst-step estimators are based on smooth parametric models and that the estimated parame-
          √
ters admit n-asymptotically linear representations, whereas Assumption A.2 imposes some weak integrability
conditions. Under mild moment conditions, these requirements are fulﬁlled when one adopts linear/nonlinear
outcome regressions or logit/probit models, for example, and estimates the unknown parameters by (nonlinear)
least squares, quasi-maximum likelihood, or other alternative estimation methods, see e.g. Chapter 5 in
van der Vaart (1998), Wooldridge (2007), Graham et al. (2012) and Sant’Anna et al. (2018).
                                                                   dr,p      dr,rc              dr,rc
      Next, we derive the asymptotic properties of τb                     , τb1      and τb2            using generic ﬁrst-step estimators that
satisfy Assumptions A.1 and A.2.

A.1 Panel data case
                                                               dr,p                            
   In this section, we discuss the asymptotic properties of τb . Deﬁne π̇ (x; γ ) ≡ ∂ π (x; γ ) ∂ γ and, for t = 0, 1,
                      
         p         p
deﬁne µ̇ 0,t  x; β 0,t   analogously. In what follows, we drop the dependence of the functionals on W to ease the
notational burden. For example, we write w1p = w1p (D), w0p (γ ) = w0p (D, X ; γ ), and so on and so forth.
                            ′        ′
                                         ′
    For generic γ and β 0 = β 0,1 , β 0,0 , let

                                η p (W ; γ , β ) = η 1p (W ; β 0 ) − η 0p (W ; γ , β 0 ) − η est
                                                                                             p
                                                                                                 (W ; γ , β 0 ) ,                        (A.1)

where
                                             h                        h                     ii
                     η 1p (W ; β 0 ) = w1p ·            p
                                                 ∆Y − µ 0,∆ (β 0 ) − E w1p · ∆Y − µ 0,∆p
                                                                                         (β 0 )    ,
                                                  h                     h                           ii
                  η 0p (W ; γ , β 0 ) = w0p (γ ) · ∆Y − µ 0,∆p
                                                                (β 0 ) − E w0p (γ ) · ∆Y − µ 0,∆
                                                                                              p
                                                                                                 (β 0 )    ,

and
                                            h          p              i
    p
  η est (W ; γ , β 0 ) = lreg (β 0 )′ · E
                                        w1p − w0p (γ ) · µ̇ 0,∆ (β 0 )
                                      h                                   h                         i          i
                     + l ps (γ )′ · E α pps (γ ) ∆Y − µ 0,∆   p
                                                                 (β 0 ) − E w0p (γ ) · ∆Y − µ 0,∆
                                                                                              p
                                                                                                  (β 0 )    · π̇ (γ ) , (A.2)
                                  ′               ′ ′
with lreg (β 0 ) = lreg,0,1 β 0,1 , lreg,0,0 β 0,0        , where lreg,d,t (·) is the asymptotic linear representation of
the estimators for the outcome regression as described in Assumption A.1(iv), l ps (·) is deﬁned analogously,
                           ′               ′
   p               p               p
µ̇ 0,∆ (β 0 ) = µ̇ 0,1 β 0,1 , −µ̇ 0,0 β 0,0     and
                                                                ,                     
                                    p             (1 − D)           π (X ; γ ) (1 − D)
                                 α ps (γ ) =                     E                       .
                                              (1 − π (X ; γ ))2       1 − π (X ; γ )
For d,t = 0, 1, let Θreg                                                                  ps
                     d,t be the parameter space for the regression coeﬃcient β d,t , and Θ be the parameter space

for the propensity score coeﬃcient γ . Consider the following claims:

                    ∃γ ∗ ∈ Θ ps : P (π (X ; γ ∗ ) = p (X )) = 1,                                           (A.3)
                                                                                              
        ∃ β ∗,p     ∗,p
            0,1 , β 0,0   ∈ Θreg     reg          p         ∗,p   p       ∗,p    p           p
                             0,1 × Θ0,0 : P µ 0,1 X ; β 0,1 − µ 0,0 X ; β 0,0 = m0,1 (X ) − m0,0 (X ) = 1. (A.4)
                                                                                     dr,p
      Now we are ready to state the large sample properties of τb                           .

                                                                      29

 Theorem A.1 Suppose Assumptions 1-3 and Assumptions A.1-A.2 stated in Appendix A hold.
     (a) Provided that either (A.3) or (A.4) is true, as n → ∞,
                                                                    p
                                                             τbdr,p → τ .

 Furthermore,
                                 √ dr,p                       1 n                      
                                    τ − τ dr,p ) =
                                  n(b                         √ ∑ η p Wi ; γ ∗ , β ∗,p
                                                                                   0     + o p (1)
                                                               n i=1
                                                               d
                                                              → N (0,V p ) ,
                                   2
 where V p = E[η p W ; γ ∗ , β ∗,p
                               0      ].
                                                                 
     (b) When both (A.3) and (A.4) are true, η p W ; γ ∗ , β ∗,p
                                                             0     = η e,p (Y1 ,Y0 , D, X ) a.s. and V p is equal to the
 semiparametrically efficiency bound (2.12).

     Theorem A.1 indicates that, provided that either the propensity score model or the model for the evolution
                                                                τ dr,p is consistent for the ATT, implying that
 of the outcome for the comparison group is correctly speciﬁed, b
 our proposed estimator is indeed doubly robust. In addition, Theorem A.1 indicates that our proposed estimator
                                                                            √
 admits an asymptotically linear representation and as a consequence, it is n-consistent and asymptotically
 normal. When the models for the nuisance functions are correctly speciﬁed, our proposed DR DID estimator is
 semiparametrically eﬃcient.
     Theorem A.1 also suggests that one can use the analogy principle to estimate V p and conduct asymptotically
 valid inference.11 However, it is worth mentioning the fact that the exact form of V p depends on which nuisance
 models are correctly speciﬁed, implying that our (generic) estimator τbdr,p is doubly robust in terms of consistency
 but, in general, not doubly robust for inference. Given that in practice it is hard to know a priori which nuisance
                                                                               p
 models are correctly speciﬁed, one should include all “correction” terms in η est when estimating V p . Failing to
 do so may lead to asymptotically invalid inference procedures.

 A.2 Repeated cross-section data case
     In this section, we turn our attention to our proposed DR DID estimators for the ATT when only repeated
                                                           ′ ′ ′                              ′        ′
                                                                                                             ′
 cross-section data are available. For generic γ and β = β 1 , β 0 , where, for d = 0, 1, β d = β d,1 , β d,0 , let

                             η rcj (W ; γ , β ) = η rc, j          rc, j              rc, j
                                                    1 (W ; β ) − η 0 (W ; γ , β ) − η est (W ; γ , β ) ,                   (A.5)

 such that, for j = 1, 2,

                                   η rc, j          rc, j            rc, j
                                     1 (W ; β ) = η 1,1 (W ; β ) − η 1,0 (W ; β ) ,

                                 η rc, j              rc, j                rc, j
                                   0 (W ; γ , β ) = η 0,1 (W ; γ , β ) − η 0,0 (W ; γ , β ) ,

                                 η rc, j                rc, j                    rc, j
                                   est (W ; γ , β ) = η est,reg (W ; γ , β ) + η est,ps (W ; γ , β ) ,

11 It is easy to show that the plug-in estimator of V p is consistent, see e.g. Lemma 4.3 in Newey and McFadden (1994) and Theorem
   4.4 in Abadie (2005). We omit the detailed derivation of this result for the sake of brevity.

                                                                   30

and the precise deﬁnitions of all these η rc functions are deferred to Appendix B to avoid excess notational
complexity. An aspect of the diﬀerence between η rc      rc
                                                 1 and η 2 that is worth mentioning but is perhaps buried in

the notation is that η rc                                          rc
                       1 depends on β only through β 0 , whereas η 2 depends on both β 1 and β 0 . This is simply
                                             dr,rc                                                                             dr,rc
a consequence from the fact that τb1                 does not rely on outcome regressions for the treated units, but τb2               does.
    Consider the following claims:
                                                                                                                
     ∃ β ∗,rc
          0,1 , β ∗,rc
                  0,0    ∈ Θ reg
                             0,1 × Θreg
                                    0,0 : P     µ rc
                                                  0,1   X ; β ∗,rc
                                                              0,1    − µ rc
                                                                         0,0  X  ; β ∗,rc
                                                                                     0,0    =   m rc
                                                                                                  0,1 (X ) − m rc
                                                                                                               0,0 (X )  = 1, (A.6)
                                                                                                
               ∀ (d,t) ∈ {0, 1}2 ∃ β ∗,rc d,t    ∈ Θreg             rc
                                                       d,t : P µ d,t X ; β d,t
                                                                              ∗,rc
                                                                                     = mrc  d,t (X ) = 1.                     (A.7)

Theorem A.2 Let n = n1 + n0 , where n1 and n0 are the sample sizes of the post-treatment and pre-treatment
periods, respectively. Suppose Assumptions 1-3 and Assumptions A.1-A.2 stated in Appendix A hold, and that
       p
n1 /n → λ ∈ (0, 1) as n0 , n1 → ∞.
    (a) Provided that either (A.3) or (A.6) is true, as n → ∞, for j = 1, 2,
                                                                          p
                                                                   τbdr,rc
                                                                     j     → τ.

Furthermore,
                                  √ dr,rc                            1 n
                                   n(τb j − τ dr,rc
                                              j     ) =             √ ∑ η rcj (Wi ; γ ∗ , β ∗,rc ) + o p (1)
                                                                      n i=1
                                                                    d          
                                                                    → N 0,V jrc ,
                                         2
where V jrc = E[η rcj (W ; γ ∗ , β ∗,rc ) ].
                                                                    ∗    ∗,rc
    (b) Suppose that both (A.3) and (A.7) are true. Then, η rc
                                                            2 (W ; γ , β      ) = η e,rc (Y, D, T, X ) a.s., and
V2rc is equal to the semiparametrically efficiency bound (2.14). On the other hand, V1rc does not attain the
semiparametric efficiency bound when (A.3) and (A.7) are true.

                                                                                                               dr,rc   dr,rc
   In other words, Theorem A.2 states that both proposed estimators for the ATT, τb1 and τb2 , are doubly
       √
robust, n-consistent and asymptotically normal. Similar to the panel data case, the exact form of the V jrc ,
                                                                                                       τ dr,rc
j = 1, 2, depends on which working models are correctly speciﬁed, implying that the generic estimators b 1
       dr,rc
and b
    τ2         are doubly robust in terms of consistency but in terms of inference.
                                                          dr,rc
    Part (b) of Theorem A.2 indicates that τb2                    is semiparametrically eﬃcient when the working model for the
propensity score, and all working models for the outcome regressions, for both treated and comparison units,
are correctly speciﬁed. When compared to Theorem A.1(b), it is evident that such a requirement is stronger
than when panel data are available.

                                                                       31

B Appendix B: Influence function of the DR DID estimators with repeated
       cross-section
                                                                                          dr,rc         dr,rc
      As it is evident from Theorem A.2, the inﬂuence functions of τb1                            and τb2       play a major role in study of
the large sample properties of our proposed DR DID estimators. In this section, we state the precise deﬁnition
of η rcj (W ; γ , β ), j = 1, 2, introduced in (A.5).
                                                            ′ ′ ′                                ′        ′
                                                                                                                ′
                            dr,rc
      We ﬁrst focus on τb1          . For generic γ and β = β 1 , β 0 , where, for d = 0, 1, β d = β d,1 , β d,0 , let
                                                  rc,1             rc,1                 rc,1
                             η rc
                               1 (W ; γ , β ) = η 1 (W ; β 0 ) − η 0 (W ; γ , β 0 ) − η est (W ; γ , β 0 ),

where

                                        η rc,1             rc,1                 rc,1
                                          1 (W ; β 0 ) = η 1,1 (W ; β 0,1 ) − η 1,0 (W ; β 0,0 ),                                      (B.1)

                                     η rc,1                 rc,1                     rc,1
                                       0 (W ; γ , β 0 ) = η 0,1 (W ; γ , β 0,1 ) − η 0,0 (Wi ; γ , β 0,0 ),                            (B.2)

                                     η rc,1                   rc,1                       rc,1
                                       est (W ; γ , β 0 ) = η est,reg (W ; γ , β 0 ) + η est,ps (W ; γ , β 0 ),

and, for t = 0, 1,
                                                                                                              
          η rc,1                rc                  rc                    rc                  rc
            1,t (W ; γ , β ) = w1,t (D, T ) · Y − µ 0,t (X ; β 0,t ) − E[w1,t (D, T ) · Y − µ 0,t (X ; β 0,t ) ] ,
                                                                                                                            
          η rc,1                rc                         rc                    rc                         rc
            0,t (W ; γ , β ) = w0,t (D, T, X ; γ ) · Y − µ 0,t (X ; β 0,t ) − E[w0,t (D, T, X ; γ ) · Y − µ 0,t (X ; β 0,t ) ] ,

and the inﬂuence functions associated with the estimation eﬀects of the nuisance parameters are

                 η rc,1                                ′      rc     rc        rc          rc             rc
                   est,reg (W ; γ , β ) = lreg (W ; β ) · E[(w1,1 − w1,0 ) − (w0,1 (γ ) − w0,0 (γ )) · µ̇ 0,Y (T, X ; β )],

and

  η rc,1
    est,ps (W ; γ , β )
                                                                                                                        
              = l ps (D, X ; γ )′ · E α rc                  rc                    rc               rc
                                        ps,1 (γ ) · Y − µ 0,1 (X ; β 0,1 ) − E[w0,1 (γ ) · Y − µ 0,1 (β 0,1 ) ] π̇ (X ; γ )
                                                                                                                                  
                           − l ps (D, X ; γ )′ · E α rc                  rc                rc               rc
                                                      ps,0 (γ ) · Y − µ 0,0 (β 0,0 ) − E[w0,0 (γ ) · Y − µ 0,0 (β 0,0 ) ] π̇ (X ; γ ) ,

where, for t = 0, 1,

                             
                                                        (1 − D)1 {T = t}               π (X ; γ )(1 − D)1 {T = t}
                  α rc            rc
                    ps,t (γ ) ≡ α ps,t (D, T, X ; γ ) =              2
                                                                                    E                               ,
                                                             (1 − π (X ; γ ))                   1 − π (X ; γ )
and wrc     rc            rc            rc
     1,t ≡ w1,t (D, T ), w0,t (γ ) ≡ w0,t (D, T, X ; γ ).
                                   dr,rc
   The inﬂuence function of τb2 is given by
                                                   rc,2           rc,2                 rc,2
                              η rc
                                2 (W ; γ , β ) = η 1 (W ; β ) − η 0 (W ; γ , β 0 ) − η est (W ; γ , β 0 ),

where

                                                η rc,2           rc,2             rc,2
                                                  1 (W ; β ) = η 1,1 (W ; β ) − η 1,0 (W ; β ),                                        (B.3)

                                           η rc,2                 rc,1
                                             0 (W ; γ , β 0 ) = η 0 (W ; γ , β 0 ),                                                    (B.4)

                                           η rc,1                   rc,1
                                             est (W ; γ , β 0 ) = η est (W ; γ , β 0 ),

                                                                        32

                                                                              
and, for d = 0, 1, µ rc                          rc                   rc
                     d,∆ X ; β d,1 , β d,0 ≡ µ d,1 X ; β d,1 − µ d,0 X ; β d,0 , and
                                                                                                         
              rc,2       ∗            D        rc
                                                                           D rc                          
            η 1,1 (W ; β ) =                 µ 1,∆ X ; β 1,1 , β 1,0 − E        µ       X ; β 1,1 , β 1,0
                                    E [D]                                  E [D] 1,∆
                                                                rc
                                                                                                   rc
                                                                                                                 
                                   +wrc 1,1 (D, T ) · Y − µ 1,1 X ; β 1,1     − E[wrc1,1 · Y − µ 1,1 X ; β 1,1 ] ,
                                                                                                         
               rc,2                   D        rc        rc      rc        D rc                          
             η 1,0 (W ; β ) =                µ 0,∆ X ; β 0,1 , β 0,0 − E        µ       X ; β 0,1 , β 0,0
                                    E [D]                                  E [D] 0,∆
                                                              rc
                                                                                 rc             rc
                                                                                                               
                                   +wrc 1,0 (D, T ) · Y − µ 1,0 X ; β 1,0 − E[w1,0 · Y − µ 1,0 X ; β 1,0 ] .

Note that estimating the OR coeﬃcients associated with the treated group does not lead to any estimation eﬀect.
