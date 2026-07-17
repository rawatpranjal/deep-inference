<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2408.10509.pdf
backend: pdftotext
part: 6/6
-->

# Appendix C: Proofs 2

Similar to the panel case, by boundedness and assumptions on the kernel function, J2k satisfies

                                                 1  4
                                       2
                                                   E Kh (D − d) ≲ (N h3 )−1 .
                                                               
                                    E[J2k ]≲                                                                     (C.156)
                                                 n

Therefore, by Markov’s inequality, we have J2k = op (1) if nh3 → ∞.

   Next, we bond J1k . For notation simplicity, we define

                                                                    0
                                      ψi :=ψ̃h (Zi , θ0h (d), λ0 , fD (d), η0 (d))                               (C.157)
                                      ψ̂i :=ψ̃h (Zi , θ̂h (d), λ̂k , fˆk (d), η̂k (d)).                          (C.158)

Then, using the same argument as in the panel case, we have
                                                                  
                                            2                 1X 2
                                           J1k ≲ SN      SN +   ψi                                               (C.159)
                                                              n
                                                                      i∈Ik

where SN := n1                       2
                 P
                     i∈Ik |ψ̂i − ψi | .

   By triangle inequality, we have

            1X                                                                                            2
      SN =           ψ̃h (Zi , θ̂h (d), λ̂k , fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), λ0 , fD
                                                                                           0
                                                                                             (d), η0 (d))
            n
              i∈Ik
                    
            1X
          =           ψ̃h (Zi , θ0h (d), λ̂k , fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), λ0 , fD0
                                                                                               (d), η0 (d))
            n
              i∈Ik
                                                        2
                ∂                      ˆ
            +      ψ̃h (Zi , θ̄, λ̂k , fk (d), η̂k (d))
                ∂θ
            1X                                                                                             2
          ≲           ψ̃h (Zi , θ0h (d), λ̂k , fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), λ0 , fD0
                                                                                               (d), η0 (d))      (C.160)
            n
               i∈Ik
            |                                                 {z                                             }
                                                          :=S1N

                                                             62

                                                              2
                  1X           Kh (Di − d)
              +                             (θ̂h (d) − θ0h (d))                                                  (C.161)
                  n
                    i∈Ik
                                  fˆk (d)
                  |                       {z                    }
                                         :=S2N

where the second line holds by Taylor’s theorem with θ̄ between θ0h (d) and θ̂h (d), and the last line
holds by the fact that ∂ ψ̃h (Zi , θ̄, λ̂k , fˆk (d), η̂k (d)) = Kh (Di − d)/fˆk (d).
                               ∂θ

   Note that, using the identical argument as in the panel case,

                                           S2N = Op (h−1 ) × Op ((N h)−1 ).                                      (C.162)

Moreover, by Taylor’s theorem, for f¯ between fD
                                               0 (d) and fˆ (d), and for λ̄ between λ and λ̂ , we
                                                           k                         0      k
have

                  ∥ψ̃h (Zi , θ0h (d), λ̂k , fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), λ0 , fD
                                                                                         0
                                                                                           (d), η0 (d))∥2P,2
                                              0                                    0
                   ≲ ∥ψh (Zi , θ0h (d), λ0 , fD (d), η̂k (d)) − ψh (Zi , θ0h (d), fD (d), η0 (d))∥2P,2           (C.163)
                   + ∥∂λ ψ̃h (Zi , θ0h (d), λ̄, f¯, η̂k (d))∥2P,2 ∥λ̂k − λ0 ∥2P,2                                (C.164)
                   + ∥∂f ψ̃h (Zi , θ0h (d), λ̄, f¯, η̂k (d))∥2P,2 ∥fˆk (d) − fD
                                                                              0
                                                                                (d)∥2P,2                         (C.165)

By boundedness assumption, we have

                                         ∂
                                            ψ̃h (Z, θ, λ, f, η(d)) ≲ Kh (D − d)                                  (C.166)
                                         ∂f
                                         ∂
                                            ψ̃h (Z, θ, λ, f, η(d)) ≲ Kh (D − d)                                  (C.167)
                                         ∂λ

and by the same argument as in the panel case, we have

                                    ∥∂λ ψ̃h (Zi , θ0h (d), λ̄, f¯, η̂k (d))∥2P,2 = O h−1
                                                                                           

                                    ∥∂f ψ̃h (Zi , θ0h (d), λ̄, f¯, η̂k (d))∥2P,2 = O h−1 .
                                                                                        
                                                                                                                 (C.168)

By assumptions, ∥λ̂k − λ0 ∥2P,2 = O(N −1 ) and ∥fˆk (d) − fD
                                                           0 (d)∥2            −1
                                                                 P,2 = O((N h) ).                              Moreover,
                         0 (d), η̂ (d)) − ψ (Z , θ (d), λ , f 0 (d), η (d))∥2
∥ψh (Zi , θ0h (d), λ0 , fD                                                         −1 2
                                  k        h  i 0h       0 D          0     P,2 ≲ h εN                 by the same argu-
ments as in the proof of Theorem 4.2. Therefore, by Markov’s inequality, we have

                                            S1N = Op h−1 ε2N + h−2 N −1 .
                                                                       
                                                                                                                 (C.169)

Combining the results, we have

                                            SN = Op h−1 ε2N + h−2 N −1 .
                                                                      
                                                                                                                 (C.170)

                                                             63

    Since ψi ≲ Kh (D − d), n1                2        −1
                                   P
                                       i∈Ik ψi = Op (h ) by Markov’s inequality. This implies that

                                                
                                            1X 2
                               2
                                              ψi = Op h−2 ε2N + h−3 N −1 .
                                                                        
                              J1k ≲ SN SN +                                                           (C.171)
                                            n
                                                   i∈Ik

Then J1k = op (1) if h−2 ε2N + h−3 N −1 → 0.
                                  2 = σ 2 + o (1).
    Therefore, we conclude that σ̂N                                                                         □
                                       N     p

Proof of Theorem 4.4: We focus on the panel case as the repeated cross-sectional case only
requires minor modifications. To simplify notation, let θ0 (d) denote the true AT T (d), θ0h (d)
denote the true AT Th (d), and θ̂h (d) and θ̂h∗ (d) denote our cross-fitted estimator and bootstrap
estimators respectively. We only need to show the uniform asymptotic linear expansion of the
bootstrap estimator since the cross-fitted estimator can be treated as a special case with ξi = 1.

    Let TN be the set of η(d) = (g(X), fh (d|X), E∆Y (X)) for d ∈ D s.t. supd∈D ∥fh (d|X) −
fh (d|X)∥P,2 ≤ h−1/2 εN , ∥g(X) − g0 (X)∥P,2 ≤ εN , ∥E∆Y (X) − E∆Y
 0                                                              0 (X)∥
                                                                       P,2 ≤ εN , κ < ∥g(X)∥P,∞ <
1 − κ, κ < fh (d|X) < C a.s. ∀d ∈ D, supd∈D ∥∂d fh (d|X)∥P,∞ < C, and ∥E∆Y (X)∥P,∞ < C. Let FN
                                               0 (d)| = O((log(N )/N h)1/2 ), sup               0     2
be the set of f > c such that supd∈D |f (d) − fD                                  d∈D |f (d) − fD (d)| =
O(log(N )/N h), and supd∈D |f (1) (d)| < C. Then by Assumption 4.10, with probability tending to
1, η̂k (d) ∈ TN and fˆk (d) ∈ FN for all d ∈ D and k = 1, · · · , K.

    Recall that the bootstrap estimator has the following form

                   K
                 1 X 1 X Kh (Di − d)ĝk (Xi ) − 1{Di = 0}fˆh,k (d|Xi )
   θ̂h∗ (d) :=
                                                                                         
                        ξi                                             ∆Yi − Ê∆Y,k (Xi ) .           (C.172)
                 K   n
                   k=1   i∈Ik
                                         fˆk (d)ĝk (Xi )

Then we have

                                θ̂h∗ (d) − θ0 (d) = θ̂h∗ (d) − θ0h (d) + θ0h (d) − θ0 (d)             (C.173)

and we can focus on the stochastic part θ̂h∗ (d) − θ0h (d).

      As before, denote the full random sample by IN , and each equal size subsample by Ik for k =
1, · · · , K. Let the score ψh be defined as in (3.8), let En,k (f ) = n1 i∈Ik f (Zi ) denote the empirical
                                                                         P
                                                                                  ∗ (f ) = 1
                                                                                             P
average of a generic function f over the subsample Ik , and similarly let En,k             n    i∈Ik ξi f (Zi )
denote the multiplier version of the empirical average. Then we have the following decomposition,
using Taylor’s theorem:

         θ̂h∗ (d) − θ0h (d)

                                                           64

                 K
             1 X ∗ h                  0               θ0h (d)                             i
         =      En,k ψh (Z, θ0h (d), fD (d), η0 (d)) − 0        Kh (D − d) − E[Kh (D − d)]                       (M)
             K                                        fD (d)
                 k=1
                  K
             1   X
                        ∗                     0                   ∗                     0
         +             En,k [ψh (Z, θ0h (d), fD (d), η̂k (d))] − En,k [ψh (Z, θ0h (d), fD (d), η0 (d))]         (R1)
             K
                 k=1
                  K 
             1   X
                         ∗                        0                                        0
                                                                                                          
         +              En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]
             K
                 k=1

                  × (fˆk (d) − fD
                                0
                                  (d))
                                     0                                0
             + E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))](E[Kh (D − d)] − fD (d))                                    (R2)
                 K
             1 X ∗
         +      En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))](fˆk (d) − fD
                                                                     0
                                                                       (d))2                                    (R3)
             K
                 k=1

where f¯k ∈ (fD
              0 (d), fˆ (d)) and E[∂ ψ (Z, θ (d), f 0 (d), η (d))] = −θ (d)/f 0 (d) in expression (M).
                       k            f h     0h     D        0          0     D
Therefore, based on this decomposition, our goal is to show that the remainder terms satisfy

                                            sup |Rj(d)| = Op ((N h)−1/2 ).                                    (C.174)
                                            d∈D

for each j = 1, 2, 3.

Step 1: Second Order Term R3

   By triangle inequality,

                    ∗
                  |En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − E[∂f2 ψh (Z, θ0h (d), fD
                                                                                      0
                                                                                        (d), η0 (d))]|
                         ∗
                     ≤ |En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − E[∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))]|     (R3.1)
                     + |E[∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − E[∂f2 ψh (Z, θ0h (d), fD
                                                                                       0
                                                                                         (d), η0 (d))]|        (R3.2)

First, to bound (R3.1), note that conditional on the auxiliary sample, we can treat f¯k , η̂k (d) as
fixed. Then, by definition,

                                                           2
                     ∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d)) = ¯2 (ψh (Z, θ0h (d), f¯k , η̂k (d)) + θ0h (d)).      (C.175)
                                                          fk

Therefore, by boundedness of f¯k and θ0h (d), we have

                        ∗
                  sup |En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − E[∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))]|
                  d∈D
                           ∗
                   ≲ sup |En,k [ψh (Z, θ0h (d), f¯k , η̂k (d))] − E[ψh (Z, θ0h (d), f¯k , η̂k (d))]|
                        d∈D

                                                              65

                           p             
                    = Op     log(N )/(N h)                                                                        (a)

We will argue that (a) holds at the end of the proof.

   Second, to bound (R3.2),

              sup |E[∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − E[∂f2 ψh (Z, θ0h (d), fD
                                                                                  0
                                                                                    (d), η0 (d))]|
              d∈D

          ≤ sup E[|∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d)) − ∂f2 ψh (Z, θ0h (d), fD
                                                                             0
                                                                               (d), η0 (d))|]
              d∈D

          ≤ sup E[E[|∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d)) − ∂f2 ψh (Z, θ0h (d), fD
                                                                               0
                                                                                 (d), η0 (d))||Ikc ]]
              d∈D

          ≤            sup          E[|∂f2 ψh (Z, θ0h (d), f, η(d)) − ∂f2 ψh (Z, θ0h (d), fD
                                                                                           0
                                                                                             (d), η0 (d))|]
              d∈D,f ∈FN ,η(d)∈TN

          ≤            sup          ∥∂f2 ψh (Z, θ0h (d), f, η(d)) − ∂f2 ψh (Z, θ0h (d), fD
                                                                                         0
                                                                                           (d), η0 (d))∥P,2
              d∈D,f ∈FN ,η(d)∈TN

          ≲h−1/2 ϵN                                                                                               (b)

where the first inequality holds by Jensen’s inequality, second inequality holds by the law of iterated
expectation, third inequality holds by the fact that conditional on the auxiliary sample, we can
treat the estimated nuisance parameters as fixed, the fourth inequality holds by Cauchy-Schwarz,
and the we will argue that (b) holds at the end of this proof.

   Note that by the standard results, see for example, Li and Racine (2007) chapter 1.10,

                                   sup(fˆk (d) − fD
                                                  0
                                                    (d))2 = Op (log(N )/(N h)).                               (C.176)
                                   d∈D

Therefore,

                         ∗
                    sup En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))](fˆk (d) − fD
                                                                             0
                                                                               (d))2
                    d∈D

               ≲ sup E[∂f2 ψh (Z, θ0h (d), fD
                                            0
                                              (d), η0 (d))](fˆk (d) − fD
                                                                       0
                                                                         (d))2 ] + op ((N h)−1/2 )
                    d∈D

               ≲Op (h log(N )/(N h)) + op ((N h)−1/2 )
               ≲op ((N h)−1/2 )                                                                               (C.177)

which establishes that supd∈D |R3(d)| = op ((N h)−1/2 .

Step 2: First Order Term R2

   To bound the first order term R2, we follow the similar steps as above. By triangle inequality,

                                                            66

we have

                 ∗                        0                                        0
               |En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]|
                     ∗                        0                                        0
                 ≤ |En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η̂k (d))]|    (R2.1)
                                          0                                        0
                 + |E[∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]|         (R2.2)

   Follow the same reasoning as before, we can show that, for the R2.1 term,

                     ∗                        0                                        0
               sup |En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η̂k (d))]|
               d∈D
                       ∗                     0                                     0
               ≲ sup |En,k [ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[ψh (Z, θ0h (d), fD (d), η̂k (d))]|
                 d∈D
                    p                  
               = Op    log(N )/(N h)                                                                           (c)

where we have used the fact that

                             0                        1                     0
         ∂f ψh (Z, θ0h (d), fD (d), η̂k (d)) = −    0     (ψh (Z, θ0h (d), fD (d), η̂k (d)) + θ0h (d)).    (C.178)
                                                   fD (d)

          0 (d) is uniformly bounded below from zero. We argue that (c) holds at the end of this
and that fD
proof.

   Second, to bound (R2.2), following the same reasoning as how we bound the second-order term,
we have

                                            0                                        0
                 sup |E[∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]|
                 d∈D
                                          0                                     0
             ≤ sup E[|∂f ψh (Z, θ0h (d), fD (d), η̂k (d)) − ∂f ψh (Z, θ0h (d), fD (d), η0 (d))|]
                 d∈D
                                            0                                     0
             ≤ sup E[E[|∂f ψh (Z, θ0h (d), fD (d), η̂k (d)) − ∂f ψh (Z, θ0h (d), fD (d), η0 (d))||Ikc ]]
                 d∈D
                                                                                    0
             ≤         sup     E[|∂f ψh (Z, θ0h (d), f, η(d)) − ∂f ψh (Z, θ0h (d), fD (d), η0 (d))|]
                 d∈D,η(d)∈TN
                                                                                  0
             ≤         sup     ∥∂f ψh (Z, θ0h (d), f, η(d)) − ∂f ψh (Z, θ0h (d), fD (d), η0 (d))∥P,2
                 d∈D,η(d)∈TN

             ≲h−1/2 ϵN .                                                                                       (d)

We will establish (d) at the end of this proof.

   Note that by the standard kernel estimation results, see Li and Racine (2007) chapter 1.10 for

                                                            67

example,

                                 sup |fˆk (d) − fD
                                                 0
                                                   (d)| = Op ((log(N )/(N h))1/2 )                      (C.179)
                                 d∈D

which implies that
                                                                                             
             ∗                        0                                        0
        sup En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]
        d∈D

                                                                                ×(fˆk (d) − fD
                                                                                             0
                                                                                               (d))
                                                                                 = op ((N h)−1/2 ).     (C.180)

Moreover, note that by the assumptions on the kernel function and the density, we have

                                                           0
                                     sup |E[Kh (D − d)] − fD (d)| = O(h2 ).                             (C.181)
                                     d∈D

Therefore,

                                  0
        sup E[∂f ψh (Z, θ0h (d), fD                                0
                                    (d), η0 (d))](E[Kh (D − d)] − fD (d)) = op ((N h)−1/2 )             (C.182)
        d∈D

                                                       0 (d), η (d))] = −θ (d)/f 0 (d), the bounded-
where we have used the fact that E[∂f ψh (Z, θ0h (d), fD       0          0h    D
ness assumption, and that the kernel bandwidth is assumed to be undersmoothing. This establishes
supd∈D |R2(d)| = op ((N h)−1/2 .

Step 3: “Neyman” Term R1

   To bound (R1), note that

       ∗                     0                   ∗                     0
     |En,k [ψh (Z, θ0h (d), fD (d), η̂k (d))] − En,k [ψh (Z, θ0h (d), fD (d), η0 (d))]| ≤ R11 + R12 .   (C.183)

Specifically,

                  ∗                     0                                     0
           R11 = En,k [ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[ψh (Z, θ0h (d), fD (d), η̂k (d))|Ikc ]
                      ∗                     0                                    0
                   − En,k [ψh (Z, θ0h (d), fD (d), η0 (d))] + E[ψh (Z, θ0h (d), fD (d), η0 (d))]
                     ∗                     0                   ∗                     0
                :=|Ṗn,k (ψh (Z, θ0h (d), fD (d), η̂k (d)) − Ṗn,k (ψh (Z, θ0h (d), fD (d), η0 (d))|    (C.184)

       ∗ (f ) = 1
                      Pn
with Ṗn,k      n  i=1 ξi f (Zi ) − E[f (Zi )] denoting the centered sample average with multiplier.
                                                  ∗ to denote the centered process conditional on the
With some abuse of notation, we will also use Ṗn,k

                                                          68

auxiliary sample. Note that since ξi ⊥ Zi and ξi ∼ N (1, 1), then E[f (Zi )] = E[ξi f (Zi )]. Moreover,

                                     0
          R12 := |E[ψh (Z, θ0h (d), fD (d), η̂k (d)|Ikc ] − E[ψh (Z, θ0h (d), fD
                                                                               0
                                                                                 (d), η0 (d))]|.         (C.185)

    First, we bound R11 . To simplify notation, denote

                                                 0                                    0
           ∆i (η̂k (d), d) := ψh (Zi , θ0h (d), fD (d), η̂k (d)) − ψh (Zi , θ0h (d), fD (d), η0 (d))     (C.186)

and we can write

                                                       ∗
                                          R11 (d) = |Ṗn,k [∆i (η̂k (d), d)]|.                           (C.187)

Conditional on the auxiliary sample, we can take η̂k (d) as fixed. Then, we can define the following
function class

                                          FN := {∆(η̂k (d), d) : d ∈ D}                                  (C.188)

and establish that

                                            ∗
                           sup R11 (d) = ∥Ṗn,k [∆(η̂k (d), d)]∥FN = op ((N h)−1/2 ).                         (e)
                           d∈D

We will show (e) at the end of the proof.
                                                                       0 (d), η (d))] = 0 for all d, then
    To bound R12 , first, note that by definition E[ψh (Zi , θ0h (d), fD       0
                                                  0 (d), η̂ (d))|I c ]. For notation simplicity, suppressing
it suffices to consider only E[ψh (Zi , θ0h (d), fD        k      k
other arguments in the score and define

                              hk (r, d) := E[ψh (η0 (d) + r(η̂k (d) − η0 (d)), d)|Ikc ]                  (C.189)

where by definition hk (0, d) = E[ψh (η0 (d), d)|Ikc ] = 0 for all d ∈ D, and hk (1, d) = E[ψh (η̂k (d), d)|Ikc ].
Using Taylor’s theorem to expand hk (1, d) around 0 in its first argument, we have

                                                              1 ′′
                          hk (1, d) = hk (0, d) + h′k (0, d) + hk (r̄, d),       r̄ ∈ (0, 1).            (C.190)
                                                              2

Note that, by Neyman orthogonality, for all d ∈ D

                             h′k (0, d) = ∂η (d)E[ψh (η0 (d), d)][η̂k (d) − η0 (d)] = 0                  (C.191)

                                                          69

and use that fact that hk (0, d) = 0, we have

                                                                 ′′
                    sup R12 (d) = sup |hk (1, d)| = sup |hk (r̄, d)|
                    d∈D              d∈D                d∈D

                                 ≤           sup           |∂r2 E[ψh (η0 (d) + r(η(d) − η0 (d)), d)]|
                                     r∈(0,1),d∈D,η(d)∈TN

                                 ≲ h−1/2 ε2N .                                                                    (f)

We will show (f) at the end of this proof. Note that since ϵN = o(N −1/4 ), we have supd∈D R12 (d) =
o((N h)−1/2 ).

   Therefore, we conclude that supd∈D |R1(d)| = op ((N h)−1/2 .

Step 4: Auxiliary Results

   We focus on showing that (e) holds. We adapt and credit this part of the proof to Fan et al.
(2022) (see Lemma 8.2 and its proof in the supplementary material). Recall (e) states the following

                                         ∗
                                      ∥Ṗn,k [∆(η̂k (d), d)]∥FN = op ((N h)−1/2 )

where FN := {∆(η̂k (d), d) : d ∈ D} and recall that

                                                       0                                    0
                 ∆i (η̂k (d), d) := ψh (Zi , θ0h (d), fD (d), η̂k (d)) − ψh (Zi , θ0h (d), fD (d), η0 (d)).

Let ϵ > 0 be given and let AN (ϵ) denote the event that the nuisance parameters η̂k (d) estimated
using the auxiliary sample Ikc satisfy Assumption 4.10 (c) with probability 1 − ϵ.

   Let η(d) satisfy Assumption 4.10 (c), and let F := {∆(η(d), d) : d ∈ D}. Then F has an
envelope F such that,

                                                       F ≲ ξi h−1 .                                           (C.192)

Since ξi ’s are assumed to be i.i.d. N (1, 1), then M := supi F (Zi ) satisfies

                                                             log(N )h−1 .
                                                            p
                                               ∥M ∥P,2 ≲                                                      (C.193)

See van der Vaart and Wellner (1996) chapter 2.2 for example. Additionally, by the assumptions
on the kernel function and the smoothness of nuisance parameters, we have
                                                                                   1
                                   sup log(Nc (ϵ∥F ∥Q,2 , F, ∥ · ∥Q,2 )) ≲ log                                (C.194)
                                     Q                                               ϵ

                                                            70

where Nc denotes the covering number and F is an envelope of F . Note that on AN (ϵ), conditions
(C.192)-(C.194) hold for FN .

   Moreover, on AN (ϵ),

               sup E[∆2i (η̂k (d), d)|Ikc ]
               d∈D

           ≤         sup     E[∆2i (η(d), d)|Ikc ]
               d∈D,η(d)∈TN

           ≤         sup     E[∆2i (η(d), d)]
               d∈D,η(d)∈TN
                                                   0                                 0
           =         sup     E[(ψh (Zi , θ0h (d), fD (d), η(d)) − ψh (Zi , θ0h (d), fD (d), η0 (d)))2 ]
               d∈D,η(d)∈TN

           ≲h−1 ϵ2N .                                                                                     (C.195)

   Therefore, by Chernozhukov et al. (2014a) Corollary 5.1, we have
                                                                                     3
                                                          r
          ∗                                                  h−1 ϵ2N log(N ) log 2 (N )h−1
                 [∆(η̂k (d), d)]∥FN |Ikc 1{AN (ϵ)} ≲
                                        
        E ∥Ṗn,k                                                             +              .             (C.196)
                                                                     N               N

   Then, let ζ > 0 be given, as N → ∞,
                                                                     
                            ∗
                      P ∥Ṗn,k [∆(η̂k (d), d)]∥FN ≥ ζ(N h)−1/2                                            (C.197)
                        h                                                 i
                     =E P ∥Ṗn,k∗
                                    [∆(η̂k (d), d)]∥FN ≥ ζ(N h)−1/2 |Ikc                                  (C.198)
                        h                                                         i
                     =E P ∥Ṗn,k∗
                                    [∆(η̂k (d), d)]∥FN ≥ ζ(N h)−1/2 |Ikc 1{AcN (ϵ)}                       (C.199)
                           h                                                         i
                       + E P ∥Ṗn,k  ∗
                                         [∆(η̂k (d), d)]∥FN ≥ ζ(N h)−1/2 |Ikc 1{AN (ϵ)}                   (C.200)
                            h                                                         i
                                       ∗
                     ≤ϵ + E P ∥Ṗn,k      [∆(η̂k (d), d)]∥FN ≥ ζ(N h)−1/2 |Ikc 1{AN (ϵ)}                  (C.201)
                            "                                                #
                                      ∗ [∆(η̂ (d), d)]∥        c
                              E[∥Ṗn,k         k          FN |Ik ]
                     ≤ϵ + E                                        1{AN (ϵ)}                              (C.202)
                                           ζ(N h)−1/2
                                                    3
                          q
                             h−1 ϵ2N log(N )            )h−1
                                             + log (N
                                                    2
                                    N                 N
                     ≲ϵ +                                                                                 (C.203)
                                     ζ(N h)−1/2
                     ≲2ϵ.                                                                                 (C.204)

Therefore, we conclude that (e) holds. Since (a) and (c) follow from the same argument with only
notation changes, their derivations are omitted here. Moreover, given Assumption 4.10, the auxil-
iary results in the point-wise case (Theorem 2.1) now hold uniformly over D, and hence (b), (d),
and (f) can be shown using the same arguments.

                                                        71

   Therefore, we conclude that

           θ̂h∗ (d) − θ0h (d)
                      K
                 1 X ∗ h                  0               θ0h (d)                             i
           =        En,k ψh (Z, θ0h (d), fD (d), η0 (d)) − 0        Kh (D − d) − E[Kh (D − d)]
                 K                                        fD (d)
                      k=1
                  ∗
           + R (d)                                                                                   (C.205)

where supd∈D |R∗ (d)| = op ((N h)−1/2 ).

   Note that we can establish a similar result for θ̂h (d) − θ0h (d) by replacing multiplier ξi with
constant 1:

           θ̂h (d) − θ0h (d)
                   K
                 1 X     h
                                           0               θ0h (d)                             i
           =         En,k ψh (Z, θ0h (d), fD (d), η0 (d)) − 0        Kh (D − d) − E[Kh (D − d)]
                 K                                         fD (d)
                      k=1
                  ′
           + R (d)                                                                                   (C.206)

where supd∈D |R′ (d)| = op ((N h)−1/2 ).

   Then, taking the difference

     θ̂h∗ (d) − θ̂h (d)                                                                              (C.207)
   =θ̂h∗ (d) − θ0h (d) − [θ̂h (d) − θ0h (d)]                                                         (C.208)
           N
                      "                                                                        #
     1 X                                  0              θ0h (d)                             
   =         (ξi − 1) ψh (Zi , θ0h (d), fD (d), η0 (d)) − 0      Kh (Di − d) − E[Kh (D − d)]         (C.209)
     N                                                   fD (d)
           i=1
         N
       1 X
   +             (ξi − 1)θ0h (d) + R∗ (d) + R′ (d).                                                  (C.210)
       N
           i=1

Since ξ˜i := ξi − 1 are i.i.d N (0, 1), we conclude that

                                    N
                                1 X
                            sup     (ξi − 1)θ0h (d) + R∗ (d) + R′ (d) = op ((N h)−1/2 ).             (C.211)
                            d∈D N   i=1

                                                                                                          □

                                                        72
