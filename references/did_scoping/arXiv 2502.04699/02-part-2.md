<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2502.04699.pdf
backend: pdftotext
part: 2/2
-->

# Part 2

E[Yb (η0 ) − Yb (η̂)|X]
                                                                           
          D − π0 (W )                        D − π̂(W )
    = E                   (∆Y − g0 (W )) −               (∆Y − ĝ(W )) X
          1 − π0 (W )                         1 − π̂(W )
                                                                                                      
                      (1 − D)π0 (W )                               (1 − D)π̂(W )
    = E E D−                             (∆Y − g0 (W )) − D −                         (∆Y − ĝ(W )) W X
                         1 − π0 (W )                                 1 − π̂(W )
                                                                                                              
                                       (1 − D)π0 (W )                     (1 − D)π̂(W )
    = E E D(ĝ(W ) − g0 (W )) −                         (∆Y − g0 (W )) +                  (∆Y − ĝ(W )) W X
                                         1 − π0 (W )                         1 − π̂(W )
                                                                                                                   
                                                                                     (1 − D)π̂(W )
    = E E [π0 (W )(ĝ(W ) − g0 (W )) | W ] − E [∆Y − g0 (W ) | D = 0, W ] + E                       (∆Y − ĝ(W )) W X
                                                                                       1 − π̂(W )
                                                                                                   
                                               1 − D (1 − π0 (W ))π̂(W )
    = E π0 (W )(ĝ(W ) − g0 (W )) + E                                        (∆Y − ĝ(W )) W X
                                            1 − π0 (W )     1 − π̂(W )
                                                                                               
                                            (1 − π0 (W ))π̂(W )
    = E π0 (W )(ĝ(W ) − g0 (W )) + E                           (∆Y − ĝ(W )) D = 0, W X
                                                 1 − π̂(W )
                                                                                                 
                                            (1 − π0 (W ))π̂(W )
    = E π0 (W )(ĝ(W ) − g0 (W )) + E                           (g0 (W ) − ĝ(W )) D = 0, W X
                                                 1 − π̂(W )
                                                   
                              π0 (W ) − π̂(W )
    = E (ĝ(W ) − g0 (W ))                       X
                                 1 − π̂(W )

                                                                23

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

The rates in Theorem 3.6 is an application of Theorem 1 in Foster & Syrgkanis, 2023. We reproduce the theorem in our
notation for completeness. Let d(η̂, η0 ) denote a distance metric for the function space of the nuisance functions F, and
∥(·)∥Θ denote a norm for Θ. We denote Star(Θ, θ) to be the star hull, i.e.Star(Θ, θ) = {tθ + (1 − t)θ′ | ∀θ′ ∈ Θ, t ∈ [0, 1]}.
Moreover, let θ′ be an arbitrary element in Θ.
Assumption D.3 (First Order Optimality). θ′ satisfies the first-order optimality condition for L(θ; η0 ):

                                           ∂θ L(θ; η0 )[θ − θ′ ] ≥ 0    ∀ θ ∈ Star(Θ, θ′ )

Assumption D.4 (Higher Order Smoothness). There exist constant β1 such that:

                                            ∂θ2 L(θ̄, η0 )[θ − θ′ , θ − θ′ ] ≤ β1 ∥θ − θ′ ∥2Θ

for all θ ∈ Θ and all θ̄ ∈ Star(Θ, θ′ ).
Assumption D.5 (Strong Convexity). The population loss is strongly convex with respect to θ, i.e. there exist constants λ,
κ >0 and r ≥ 0, such that for all θ ∈ Θ, θ′ ∈ Star(Θ, θ′ ), and η ∈ F:
                                                                                                4
                                   ∂θ2 L(θ̄, η)[θ − θ′ , θ − θ′ ] ≥ λ∥θ − θ′ ∥2 − κd(η, η0 ) 1+r

Assumption D.6. There exist r ∈ [0, 1) and constant β2 such that for all θ, θ′ ∈ Star(Θ, θ′ ) and all η1 , η2 in F:

                         ∥L(θ; η1 ) − L(θ′ ; η1 ) − L(θ; η2 ) + L(θ′ ; η2 )| ≤ β2 ∥θ − θ′ ∥1−r
                                                                                           Θ d(η1 , η2 )
                                                                                                        2

Theorem D.7 (Theorem 1 from (Foster & Syrgkanis, 2023)). Suppose Assumptions D.3, D.4, D.5, and D.6 are satisfied for
some θ′ ∈ Θ. Then for any θ ∈ Θ, the following holds:
                                                                     2
                                                                  1+r     !
                                      4                          β 2      κ               4
                        ∥θ − θ′ ∥2Θ ≤ (L(θ, η̂) − L(θ′ , η̂)) +         +     d(η0 , η̂) 1+r
                                      λ                          λ        λ

We are finally ready to prove Theorem 3.6.

Proof of Theorem 3.6. Since results follow from Theorem D.7, we first show that the minimizer of the loss in the function
class Θ, i.e. θ∗ , satisfies Assumptions D.3, D.4, D.5, and D.6 for the proposed loss L(θ; η) with ∥(·)∥Θ = ∥(·)∥D=1 .
Assumption D.3 is satisfied when Θ is convex or when θ0 ∈ Θ. Assumptions D.4 and D.5 requires us to bound:

                      ∂θ2 L(θ̄, η̂)[θ − θ∗ , θ − θ∗ ] = E[D(θ(X) − θ∗ (X))2 ] = ρ∥(θ(X) − θ∗ (X))∥2D=1

Thus Assumptions D.4 and D.5 are satisfied with β1 = λ = ρ and κ = 0. To show Assumption D.6, we need to convert the
∥(·)∥2 in D.2 into ∥(·)∥D=1 :
                                           Z
                     ∥(θ(X) − θ∗ (X))∥2 =     (θ(X) − θ∗ (X))2 Pp(X)dX
                                           Z
                                                                                1
                                        =     (θ(X) − θ∗ (X))2 P(D = 1|X)              p(X)dX
                                                                           P(D = 1|X)
                                             Z
                                           1
                                        ≤       (θ(X) − θ∗ (X))2 P(D = 1|X)p(X)dX
                                           c
                                           1
                                        = ∥θ(X) − θ∗ (X)∥2Θ
                                           c

Thus, Lemmas D.1 and D.2 imply Assumption D.6 with r = 0, β2 = 3c , and
                                         "                                         2 #1/2
                                     2                           π0 (W ) − π̂(W )
                            d(η, η0 ) = E E (ĝ(W ) − g0 (W ))                      X
                                                                    1 − π̂(W )

                                                                   24

                           A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Thus invoking Theorem D.7, we get that:

                                                    "                                         2 #
                               4              2                             π0 (W ) − π̂(W )
                  ∥θ − θ′ ∥2Θ ≤ R(n, δ) +          E E (ĝ(W ) − g0 (W ))                      X
                                 ρ           ρ2 c2                             1 − π̂(W )

Analogously, we can prove the rates in the case with instrument. Consider LIV (θ; η) from Proposition A.11, where we let
η denote the set of nuisances π(W ), gD (W ) and gY (W ). We first present an auxiliary lemma to bound |LIV (θ1 ; η1 ) −
LIV (θ2 ; η1 ) − (LIV (θ2 ; η1 ) − LIV (θ2 ; η2 ))|.

Lemma D.8. Let η = (π, gY , gD ) denote the set of nuisance functions, and let η0 = (π0 , g0,Y , g0,D ) be the true nuisance
functions. Consider the loss defined in Proposition A.11. Assume there exist finite constant B such that |θ(X)| ≤ B for all
X with positive measure, and all θ ∈ Θ. Then, we have that for all θ1 , θ2 , η,

                     |LIV (θ1 ; η) − LIV (θ2 ; η) − (LIV (θ2 ; η0 ) − LIV (θ2 ; η0 ))|
                           "                                                     2 # 21
                        2          π0 (W ) − π(W )
                   ≤ 4B E E                            (g0,D (W ) − gD (W )) X              ∥θ(X) − θ(X)∥
                                       1 − π(W )
                           "                                                     2 # 12
                                   π0 (W ) − π(W )
                      + 2E E                           (g0,Y (W ) − gY (W )) X              ∥θ(X) − θ(X)∥
                                      1 − π(W )

Proof of Lemma D.8.

                                  h                                                                          i
                                   Z(η) (∆D − gD (W ))(θ12 (X) − θ22 (X)) − 2(∆Y − gY (W ))(θ1 (X) − θ2 (X))
                                       
   LIV (θ1 ; η) − LIV (θ2 ; η) = E b

                                   h                                                                                i
                                     b 0 ) (∆D − g0,D (W ))(θ12 (X) − θ22 (X)) − 2(∆Y − g0,Y (W ))(θ1 (X) − θ2 (X))
                                          
LIV (θ1 ; η0 ) − LIV (θ2 ; η0 ) = E Z(η

                          h                                      i
                            b 0 )(D − g0,D (W ))(θ2 (X) − θ2 (X)) term:
Let’s first consider the E Z(η                    1        2

                             h                                          i
                               b 0 )(∆D − g0,D (W ))(θ12 (X) − θ22 (X))
                            E Z(η
                                                                                    
                                     (1 − Z)π0 (W )                       2      2
                          = E Z−                      (∆D − g0,D (W ))(θ1 (X) − θ2 (X))
                                       1 − π0 (W )
                          = E[Z∆D(θ12 (X) − θ22 (X))] − E[Zg0,D (W )(θ12 (X) − θ22 (X))]
                            − E E [(∆D − g0,D (W ))|Z = 0, W ] π0 (W )(θ12 (X) − θ22 (X))
                                                                                         

                          = E[∆DZ(θ12 (X) − θ22 (X))] − E[Zg0,D (W )(θ12 (X) − θ22 (X))]

                                                            25

                            A Meta-learner for Heterogeneous Effects in Difference-in-Differences
              h                                    i
Now, for the E Z(η)(D
                b     − gD (W ))(θ12 (X) − θ22 (X)) term:
                    h                                      i
                  E bZ(η)(∆D − gD (W ))(θ12 (X) − θ22 (X))
                                                                           
                           (1 − Z)π(W )
                = E Z−                      (∆D − gD (W ))(θ12 (X) − θ22 (X))
                             1 − π(W )
                 = E[∆DZ(θ12 (X) − θ22 (X))] − E[ZgD (W )(θ12 (X) − θ22 (X))]
                                                                                               
                              1−Z                          (1 − π0 (W ))π0 (W ) 2
                   −E E                (∆D − gD (W ))|W                        (θ1 (X) − θ22 (X))
                          1 − π0 (W )                            1 − π(W )
                 = E[∆DZ(θ12 (X) − θ22 (X))] − E[ZgD (W )(θ12 (X) − θ22 (X))]
                                                                                             
                                                      (1 − π0 (W ))π0 (W ) 2
                   − E E [(∆D − gD (W ))|Z = 0, W ]                        (θ1 (X) − θ22 (X))
                                                           1 − π(W )
                 = E[∆DZ(θ12 (X) − θ22 (X))] − E[ZgD (W )(θ12 (X) − θ22 (X))]
                                                                                    
                        (1 − π0 (W ))π0 (W )
                   −E                        (g0,D (W ) − gD (W ))(θ12 (X) − θ22 (X))
                             1 − π(W )

Putting them together, we get:
        h                                                                                       i
      E Z(η)(∆D
          b         − gD (W ))(θ12 (X) − θ22 (X)) − Z(η
                                                     b 0 )(∆D − g0,D (W ))(θ12 (X) − θ22 (X))
                                                                                                                       
                                                          (1 − π0 (W ))π0 (W )
  = E[Z(g0,D (W ) − gD (W ))(θ12 (X) − θ22 (X))] − E                           (g0,D (W ) − gD (W ))(θ12 (X) − θ22 (X))
                                                               1 − π(W )
                                                                              
                 (1 − π0 (W ))π0 (W )
  = E Z−                                (g0,D (W ) − gD (W ))(θ12 (X) − θ22 (X))
                      1 − π(W )
                                                                  
          π0 (W ) − π(W )
  = E                     (g0,D (W ) − gD (W ))(θ12 (X) − θ22 (X))
             1 − π(W )

Similarly,
                   h                                                                                  i
                 E Z(η)(∆Y
                     b         − gY (W ))(θ1 (X) − θ2 (X)) − Z(η
                                                               b 0 )(∆Y − g0,Y (W ))(θ1 (X) − θ2 (X))
                                                                          
                     π0 (W ) − π(W )
               = E                   (g0,Y (W ) − gY (W ))(θ1 (X) − θ2 (X))
                        1 − π(W )
Thus, we have shown that:
  |LIV (θ1 ; η) − LIV (θ2 ; η) − (LIV (θ2 ; η0 ) − LIV (θ2 ; η0 ))|
                                                                                                                                  
       π0 (W ) − π(W )                             2          2              π0 (W ) − π(W )
= E                      (g0,D (W ) − gD (W ))(θ1 (X) − θ2 (X)) − 2                          (g0,Y (W ) − gY (W ))(θ1 (X) − θ2 (X))
          1 − π(W )                                                             1 − π(W )
    "                                                    2 # 12
           π0 (W ) − π(W )                                                                 1
                                                                    E (θ12 (X) − θ22 (X))2 2
                                                                      
≤ E E                           (g0,D (W ) − gD (W )) X
               1 − π(W )
        "                                                     2 # 21
                π0 (W ) − π(W )                                                                 1
                                                                          E (θ1 (X) − θ2 (X))2 2
                                                                            
   + 2E E                            (g0,Y (W ) − gY (W )) X
                    1 − π(W )
        "                                                      2 # 12
                π 0 (W ) − π(W   )
≤ 4B 2 E E                           (g0,D (W ) − gD (W )) X              ∥θ(X) − θ(X)∥
                    1 − π(W )
        "                                                     2 # 21
                π0 (W ) − π(W )
   + 2E E                            (g0,Y (W ) − gY (W )) X              ∥θ(X) − θ(X)∥
                    1 − π(W )

                                                             26

                           A Meta-learner for Heterogeneous Effects in Difference-in-Differences

We can now prove Theorem A.12.

Proof of Theorem A.12. Since results follows from Theorem D.7, we first show that the minimizer of the loss in the function
class Θ, i.e. θ∗ , satisfies Assumptions D.3, D.4, D.5, and D.6 for the proposed loss LIV (θ; η) with ∥(·)∥Θ = ∥(·)∥Z=1,CM
First, Assumption D.3 is satisfied when Θ is convex or when θ0 ∈ Θ. Now, we look at the second order directional derivative
with respect to θ. Following the same steps as in the proof of Porposition A.11, we get:

                     ∂θ2 LIV (θ̄, η0 )[θ − θ∗ , θ − θ∗ ]
                        Z(η0 )(∆D − g0,D (W ))(θ(X) − θ∗ (X))2 ]
                   = E[ b
                   = E (θ(X) − θ∗ (X))2 |Z = 1, D(1) > D(0) P(Z = 1)P(D(1) > D(0)|Z = 1)
                                                           

                   = hk∥θ(X) − θ∗ (W )∥Z=1,CM

Thus Assumption D.4 is statisfied with β1 = hk
However, for Assumption D.5, we need to bound the second directional derivative for any η. Therefore, we consider the
distance between ∂θ2 LIV (θ̄, η)[θ − θ∗ , θ − θ∗ ] − ∂θ2 LIV (θ̄, η0 )[θ − θ∗ , θ − θ∗ ]:

   ∂θ2 LIV (θ̄, η)[θ − θ∗ , θ − θ∗ ] − ∂θ2 LIV (θ̄, η0 )[θ − θ∗ , θ − θ∗ ]
       Z(η)(∆D − gD (W ))(θ(X) − θ∗ (X))2 ] − E[ b
 = 2E[ b                                                    Z(η0 )(∆D − g0,D (W ))(θ(X) − θ∗ (X))2 ]
                                                                          
         π0 (W ) − π(W )                                                 2
 = 2E                       (g0,D (W ) − gD (W ))(θ(X) − θ∗ (X))              (By the same reasoning in the proof of Lemma D.8)
            1 − π(W )
       "                                                        2 #1/2
                                        π0 (W ) − π̂(W )
 ≤ 2E E (ĝD (W ) − g0,D (W ))                                  X           ∥(θ(X) − θ∗ (X))∥24             (By Cauchy-Schwarz)
                                            1 − π̂(W )
          "                                                        2 #1/2
       2                                    π0 (W ) − π̂(W )
 ≤ 8B E E (ĝD (W ) − g0,D (W ))                                    X          ∥(θ(X) − θ∗ (X))∥22
                                               1 − π̂(W )
           "                                                          2 #1/2
   8B 2
                                                               
                                             π0 (W ) − π̂(W )
 ≤       E E (ĝD (W ) − g0,D (W ))                                 X          ∥(θ(X) − θ∗ (X))∥2Θ
     c                                          1 − π̂(W )

Thus, for sufficiently small nuisance error, Assumption D.5 is satisfied with κ = 0, and

                                                           8B 2
                                                λ = hk −        Error(π, gD )
                                                            c

Lemma D.8 implies Assumption D.6 with r = 0, β2 = 1c max{4B 2 , 2}, and d(η, η0 )2 = Error(π, ĝD ) + Error(π, ĝY ).
Thus invoking Theorem D.7, we get that:
                                                                              !
                   2               4           2        max(4B 2 , 2)
   ∥θ(X) − θ∗ (X)∥Θ ≤           2             Rn +           2              (Error(π, ĝY ) + Error(π, ĝD ))
                       hk − 8Bc Error(π, gD )      c hk − 8Bc Error(π, gD )

E. Additional Experiment Details and Results
E.1. Experiment Setup
Here we describe the data generating processes (DGP) for the fully synthetic experimens. We consider soome observed
covariates W with dimension dW , and some unobserved confounding U , of dimension dU . Let µW , µU be the mean of W

                                                             27

                             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

and U , where each entry is sampled from a uniform ditsribution ranging from 0 to 1. Let Id denote the identity matrix with
dimension d.

                      W ∼ N (µW , IdX )
               Wmasked ∼ Half of the dimensions of W are randomly set to 0
                       U ∼ N (µU , IdU )
                                                      1
                        p=                                                                    (p is clipped s.t. p ∈ [0.9, 0.1])
                              1 + exp(− 12 βD
                                            T (W − µ ) ∗ (αT (U − µ ))2 )
                                                    W      U       U

                       D ∼ Binomial(p)
                            1
                       θ0 = W1 ∗ 1(W2 > 0)
                            2
For experiments with DGP that satisfies the conditional parallel trends assumptions:
                  T
          Y0 = 5(αU (U − µU ))2 W6 + W2 + ϵ0 ,            ϵ0 ∼ N (0, 0.5)
                  T
          Y1 = 5(αU (U − µU ))2 W6 + 1(W1 > 0)W1 + βYT Wmasked + W3 + D ∗ θ0 + ϵ1 ,                  ϵ1 ∼ N (0, 0.5)

The results in Table 1 and 4 are generated using this process with dW = 20 and dU = 5. We also ran experiments with
higher dimensional covariates (dW = 100), and the results are presented in Table 6. The results in Table 2 is generated
using the same setup, but with 0.1 ∗ p as the treatment probabilities. These results all showcase that our proposed doubly
robust CATT learner out performs the baseline methods. In addition to this DGP, we also experimented with a DGP that
does not satisfy the conditional parallel trends assumptions.

                γ ∼ U nif orm([−1, 1])
                     T
              Y0 = (αU (U − µU ))2 X6 + X2 + ϵ0 ,           ϵ0 ∼ N (0, 0.5)
               m = |Y0 |
                     T
              Y1 = (αU (U − µU ))2 X6 + mγ T X ⊙ X + 1(X1 > 0)X2 + D ∗ θ0 + ϵ1 ,               ϵ1 ∼ N (0, 0.5)

Experiment results for this DGP is presented in Table 7. We see that in this case, the conditional parallel trends is violated
so the learner that assumes conditional parallel trends has a higher MSE than the those that assume lagged dependent
outcome (as this DGP has a lagged outcome component). Moreover, we see that even when the assumptions are violated,
the proposed learner is still more robust than the baseline outcome regression learner.
For the semi-sythetic experiments on the minimum wage dataset, each dataset is constructed by first sampling 10000 units
with replacement from the original dataset. We keep the covariate and pre-treatment outcome information, and generate the
treatment assignment and the outcome in the post-treatment time period. The probability of receiving treatment is generated
from the logitistic transformation of a linear transformation of a linear function of 2 ”region” variables that are binary, and
the log average payment information for year 2001 (i.e. 2 ∗ (region 3) − 2 ∗ (region 4) + ((log average pay) − 10)). The time
trends, i.e.Ypost (0) − Ypre (0), is generated by 0.1 ∗ (log average pay) + 0.1 ∗ (region 3) + 0.1 ∗ (years after treatment) +
                                                            1
(region 4) ∗ (years after treatment)2 + (log average pay) 2 ∗ (log average population). The treatment effect is defined as
                                                                  1
0.1 ∗ (log average population) + 0.1 ∗ (log average population) 2 .

E.2. Additional Results

                                                                28

                              A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Table 4. MSE (mean ± standard deviation) over 100 simulations following the conditional parallel trends condition. Each row represent a
different meta-learner, and columns represent the different nuisance function classes.

                                          Basic           Lasso (CV)      Ridge (CV)      Random Forest        Best
           Neural Net (CPTA OR)           0.12 ± 0.02     0.12 ± 0.02     0.12 ± 0.02     0.38 ± 0.18          0.12 ± 0.02
           Neural Net (CPTA DR)           0.1 ± 0.02      0.1 ± 0.03      0.1 ± 0.02      0.14 ± 0.04          0.1 ± 0.02
           Neural Net (Lagged OR)         0.12 ± 0.02     0.14 ± 0.04     0.12 ± 0.02     1.27 ± 0.65          0.12 ± 0.02
           Neural Net (Lagged DR)         0.1 ± 0.02      0.1 ± 0.03      0.1 ± 0.02      0.63 ± 0.4           0.1 ± 0.02
           XGBoost (OR)                   0.09 ± 0.02     0.09 ± 0.02     0.09 ± 0.02     0.31 ± 0.16          0.09 ± 0.02
           XGBoost (DR)                   0.04 ± 0.01     0.04 ± 0.01     0.04 ± 0.02     0.06 ± 0.03          0.04 ± 0.01
           XGBoost (Lagged OR)            0.09 ± 0.02     0.11 ± 0.04     0.09 ± 0.02     1.15 ± 0.69          0.09 ± 0.02
           XGBoost (Lagged DR)            0.04 ± 0.01     0.05 ± 0.03     0.04 ± 0.02     0.54 ± 0.45          0.04 ± 0.01
           Linear (OR)                    0.26 ± 0.07     0.26 ± 0.07     0.26 ± 0.07     0.51 ± 0.18          0.26 ± 0.07
           Linear (DR)                    0.26 ± 0.07     0.26 ± 0.07     0.26 ± 0.07     0.26 ± 0.07          0.26 ± 0.07
           Linear (Lagged OR)             0.26 ± 0.07     0.28 ± 0.08     0.26 ± 0.07     1.18 ± 0.56          0.26 ± 0.07
           Linear (Lagged DR)             0.26 ± 0.07     0.26 ± 0.07     0.26 ± 0.07     0.42 ± 0.19          0.26 ± 0.07

Table 5. MSE (mean ± standard deviation) Over 100 Simulations of Imbalanced Dataset. Each row represent a different meta-learner, and
columns represent the different nuisance function classes.

                                                    Linear
                                 No Controls                      Lasso (CV)      Ridge (CV)      Random Forest       Best
                                                  Regression
   Neural Net (OR)               1.53 ± 0.74      0.22 ± 0.06     0.21 ± 0.06     0.21 ± 0.06     0.4 ± 0.15          0.21 ± 0.05
   Neural Net (DR)               0.52 ± 0.31      0.18 ± 0.07     0.18 ± 0.05     0.18 ± 0.05     0.24 ± 0.07         0.18 ± 0.05
   Neural Net (CATE OR)          0.66 ± 0.27      0.27 ± 0.08     0.27 ± 0.08     0.27 ± 0.08     0.51 ± 0.16         0.27 ± 0.08
   Neural Net (CATE DR)          0.53 ± 0.22      0.22 ± 0.07     0.22 ± 0.07     0.21 ± 0.07     0.33 ± 0.11         0.21 ± 0.07
   XGBoost (OR)                  1.22 ± 0.58      0.21 ± 0.06     0.21 ± 0.06     0.21 ± 0.06     0.34 ± 0.11         0.21 ± 0.06
   XGBoost (DR)                  0.4 ± 0.14       0.12 ± 0.03     0.12 ± 0.03     0.12 ± 0.03     0.18 ± 0.06         0.12 ± 0.03
   XGBoost (CATE OR)             0.66 ± 0.27      0.27 ± 0.08     0.27 ± 0.08     0.27 ± 0.08     0.51 ± 0.16         0.27 ± 0.08
   XGBoost (CATE DR)             0.49 ± 0.22      0.15 ± 0.05     0.15 ± 0.04     0.15 ± 0.05     0.34 ± 0.13         0.15 ± 0.04

Table 6. MSE (mean ± standard deviation) over 100 simulations following the conditional parallel trends condition, with 100 covariates.

                                          Linear
                                                        Lasso (CV)      Ridge (CV)       Random Forest       Best
                                        Regression
             Neural Net OR              0.21 ± 0.05     0.21 ± 0.05     0.2 ± 0.06       1.27 ± 0.69         0.21 ± 0.06
             Neural Net DR              0.18 ± 0.05     0.18 ± 0.06     0.18 ± 0.06      0.64 ± 0.38         0.18 ± 0.06
             Neural Net CATE OR         0.28 ± 0.08     0.28 ± 0.08     0.28 ± 0.08      0.65 ± 0.25         0.28 ± 0.08
             Neural Net CATE DR         0.3 ± 0.1       0.29 ± 0.09     0.29 ± 0.1       1.08 ± 0.71         0.2 ± 0.06
             Linear OR                  0.27 ± 0.08     0.27 ± 0.08     0.27 ± 0.08      1.3 ± 0.63          0.27 ± 0.08
             Linear DR                  0.27 ± 0.08     0.27 ± 0.08     0.27 ± 0.08      0.44 ± 0.13         0.27 ± 0.08
             Linear CATE OR             0.28 ± 0.08     0.27 ± 0.08     0.28 ± 0.08      0.65 ± 0.25         0.27 ± 0.08
             Linear CATE DR             0.29 ± 0.08     0.29 ± 0.08     0.29 ± 0.08      0.82 ± 0.33         0.28 ± 0.08
             XGBoost OR                 0.21 ± 0.06     0.2 ± 0.05      0.21 ± 0.05      0.96 ± 0.45         0.21 ± 0.05
             XGBoost DR                 0.13 ± 0.04     0.12 ± 0.03     0.12 ± 0.03      0.61 ± 0.25         0.12 ± 0.03
             XGBoost CATE OR            0.28 ± 0.08     0.27 ± 0.08     0.28 ± 0.08      0.65 ± 0.25         0.27 ± 0.08
             XGBoost CATE DR            0.26 ± 0.09     0.24 ± 0.08     0.25 ± 0.09      1.18 ± 0.85         0.15 ± 0.05

                                                                  29

                              A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Table 7. MSE (mean ± standard deviation) over 100 simulations that does not satisfy the conditional parallel trends assumption. Each row
represent a different meta-learner, and columns represent the different nuisance function classes.

                              Linear
                                                Lasso (CV)          Ridge (CV)          Random Forest        Best
                            Regression
           Neural Net
                            76.93 ± 135.25      76.34 ± 129.71      74.87 ± 127.94      35.49 ± 91.98        36.85 ± 87.85
          (CPTA OR)
           Neural Net
                            17.07 ± 74.16       15.54 ± 54.25       20.41 ± 86.35       17.24 ± 63.37        18.38 ± 65.13
          (CPTA DR)
           Neural Net
                            70.31 ± 98.19       70.07 ± 93.93       69.98 ± 100.44      26.21 ± 39.93        24.81 ± 32.88
          (Lagged OR)
           Neural Net
                            4.37 ± 4.66         4.93 ± 5.65         4.94 ± 5.89         4.99 ± 14.46         5.09 ± 10.25
          (Lagged DR)
           XGBoost
                            65.67 ± 122.65      63.5 ± 127.89       63.82 ± 113.59      29.39 ± 69.84        30.15 ± 85.27
          (CPTA OR)
           XGBoost
                            20.49 ± 58.55       22.99 ± 81.52       23.31 ± 74.74       26.9 ± 128.52        31.11 ± 149.05
          (CPTA DR)
            XGBoost
                            55.62 ± 82.27       56.87 ± 83.95       53.95 ± 77.19       21.29 ± 32.85        22.39 ± 38.38
          (Lagged OR)
            XGBoost
                            9.88 ± 14.34        9.34 ± 13.13        10.33 ± 21.81       10.76 ± 40.39        8.14 ± 13.38
          (Lagged DR)
            Linear
                            18.41 ± 62.54       18.0 ± 61.84        18.41 ± 62.68       17.61 ± 65.51        17.61 ± 65.51
          (CPTA OR)
            Linear
                            14.56 ± 57.69       14.7 ± 58.43        14.56 ± 57.7        15.84 ± 64.12        15.84 ± 64.12
          (CPTA DR)
             Linear
                            12.08 ± 28.93       11.64 ± 27.55       12.07 ± 28.9        9.78 ± 21.18         9.78 ± 21.18
          (Lagged OR)
             Linear
                            4.78 ± 5.81         4.85 ± 5.97         4.78 ± 5.81         3.99 ± 7.18          3.99 ± 7.18
          (Lagged DR)

                                                                  30

             A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Figure 3. Calibration plot for CATT w.r.t log county population for the XGBoost doubly robust learner.

                                                 31

            A Meta-learner for Heterogeneous Effects in Difference-in-Differences

Figure 4. Calibration plot for CATT w.r.t log county population for the linear doubly robust learner.

                                                 32
