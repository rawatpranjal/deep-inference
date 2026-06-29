<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2408.10509.pdf
backend: pdftotext
part: 4/6
-->

# Appendix B: Asymptotic Theories for Repeated Cross-Sections

Appendix B: Asymptotic Theories for Repeated Cross-Sections

In this section, we present the asymptotic theory for the repeated cross-sectional setting, following a
structure analogous to the panel case discussed in detail in the main text. For notational simplicity,
let D denote a closed sub-interval of (dL , dH ) whose boundary points can be chosen arbitrarily
close to dL and dH , and let X and Y λ denote the supports of X and Y λ , respectively. We begin
by introducing regularity conditions that allow us to establish the asymptotic normality of our
estimator and the consistency of the corresponding variance estimator.

Assumption 7.11 (Bounds and Smoothness, Repeated Cross-Sections). (a) Th-ere exist con-
                                             0 (d) > c, |Y | < C, c < f 0 (d|X) < C
stants c > 0 and 0 < C < ∞ such that supd∈D fD                         h
              0 (X)| < C almost surely; (b) f 0 (d) ∈ C 2 (D) and sup
∀d ∈ D, and |EλY                                                           2 0
                                             D                       d∈D |∂d fD (d)| < ∞; (c)
 0
fD|X                                             0
     (d|x) ∈ C 2 (D) ∀x ∈ X and supd,x∈D,X |∂d2 fD|X (d|x)| < ∞; (d) fY λ ,D (t, d) ∈ C 2 (Y λ ) and
supt,d∈Y λ ,D |∂t2 fY λ ,D (t, d)| < ∞.

Assumption 7.12 (Rates, Repeated Cross-Sections). (a) The kernel bandwidth h = hN → 0
                        √
satisfies N h → ∞ and N h5 = o(1); (b) there exists a sequence εN → 0 such that h−1 ε2N = o(1);
(c) with probability tending to 1, ∥fˆh (d|X) − f 0 (d|X)∥P,2 ≤ h−1/2 εN , ∥ĝ(X) − g0 (X)∥P,2 ≤ εN ,
                                                      h
             0 (X)∥
∥ÊλY (X) − EλY                                                                          ˆ
                    P,2 ≤ εN ; (d) with probability tending to 1, κ < ĝ(X) < 1 − κ, c < fh (d|X) < C
almost surely, and ∥ÊλY (X)∥P,∞ < C.

    Next, we state the lemma that characterizes the bias introduced by the kernel smoothing of
AT T (d). As in the panel case, this bias is asymptotically negligible with undersmoothing kernel
bandwidth.

Lemma 7.3 (Bias of AT Th (d), Repeated Cross-Sections). Suppose Assumptions 4.6, 7.11, and
7.12 hold. Then Bh (d) := AT T (d) − AT Th (d) satisfies Bh (d) = O(h2 ) for any d ∈ D.

    We now state the asymptotic normality result for AT
                                                     [  T (d) in the repeated cross-sections.

Theorem 7.5 (Asymptotic Normality, Repeated Cross-Sections). Suppose Assumptions 2.2, 2.3,
2.4, 2.5, 4.6, 7.11, and 7.12 hold. Then, for d ∈ D, if εN = o(N −1/4 ),

                                      AT
                                      [  T (d) − AT T (d)
                                                 √          →d   N (0, 1)
                                         σN (d)/ N

where

       2
      σN (d)
                                                                                     2
                                                                                    
              (2)                0               θ0h (d)
      := E ψh (Z, θ0h (d), λ0 , fD (d), η0 (d)) − 0      Kh (D − d) − E[Kh (D − d)]              (B.1)
                                                 fD (d)

                                                      26

                                                        (2)
for θ0h (d) := AT Th (d) defined as in (3.7) and ψh defined as in (3.9).

   Next, we construct a cross-fitted variance estimator based on the theorem above, which is
followed by a theorem that establishes its consistency.

                  K
                                                                                                     2
                                                                                                    
                1 X           (2)                                      θ̂ h (d)
     2
   σ̂N (d) :=       En,k     ψh (Z, θ̂h (d), λ̂k , fˆk (d), η̂k (d)) −          Kh (D − d) − fˆk (d)     .   (B.2)
                K
                  k=1
                                                                       fˆk (d)

Theorem 7.6 (Variance Estimator Consistency, Repeated Cross-Sections). If the conditions in
Theorem 7.5 hold and assume h−2 ε2N + h−3 N −1 = o(1), then, for d ∈ D,

                                             2        2
                                           σ̂N (d) = σN (d) + op (1)

        2 (d) is defined as in (B.2) and σ 2 (d) is defined as in (B.1).
where σ̂N                                 N


   Let {ξi }N
            i=1 be an i.i.d. sequence of random variables that satisfies Assumption 4.9. Then for
each b = 1, · · · , B, we independently draw such a sequence {ξi }N
                                                                  i=1 and construct estimates based
on the following expression. For the repeated cross-sections, define

                                    K
                                1   XX         Kh (Di − d)ĝk (Xi ) − 1{Di = 0}fˆh,k (d|Xi )
                  AT
                  [  T (d)∗b :=              ξi
                                N
                                    k=1 i∈Ik
                                                                 fˆk (d)ĝk (Xi )
                                                                                 
                                                  Ti − λ̂k
                                             ×                 Yi − ÊλY,k (Xi ) .                           (B.3)
                                                λ̂k (1 − λ̂k )

Let ĉα denote the α-th quantile of {AT
                                      [  T (d)∗b − AT
                                                   [   T (d)}B
                                                             b=1 . Then a 1 − α confidence interval can
be constructed as [AT T (d) − ĉ1−α/2 , AT T (d) − ĉα/2 ].
                    [                   [

   The following assumption strengthens Assumption 7.12 and is used to establish the next theo-
rem, which forms the foundation for constructing valid uniform confidence bands in the repeated
cross-sectional setting using the proposed multiplier bootstrap. See Section 4 for the construction
of the the uniform confidence bands.

Assumption 7.13 (Uniform Inference Rates, Repeated Cross-Sections). (a) The kernel bandwidth
                                       √
h = hN → 0 satisfies N h → ∞ and N h5 = o(1); (b) there exists a sequence εN → 0 such that
h−1 ε2 = o(1); (c) with probability tending to 1, sup
     N                                                ∥fˆh (d|X)−f 0 (d|X)∥P,2 ≤ h−1/2 εN , ∥ĝ(X)−
                                                    d∈D               h
                               0
g0 (X)∥P,2 ≤ εN , ∥ÊλY (X) − EλY (X)∥P,2 ≤ εN ; (d) with probability tending to 1, κ < ĝ(X) < 1 − κ
                                                          (1)
and c < fˆh (d|X) < C almost surely ∀d ∈ D, supd∈D |fˆD (d)| < C, supd∈D ∥∂d fˆh (d|X)∥P,∞ < C,
and ∥ÊλY (X)∥P,∞ < C.

Theorem 7.7 (Uniform Linear Expansion, Repeated Cross-Sections). Suppose Assumptions 2.2,

                                                       27

2.3, 2.4, 2.5, 4.6, 7.11, and 7.13 hold. Then, for d ∈ D, if εN = o(N −1/4 ),

       AT
       [  T (d) − AT
                   [ T (d)∗
              N
                   "                                                                              #
          1 X˙       (2)                    0               θ0h (d)                             
       =         ξi ψh (Zi , θ0h (d), λ0 , fD (d), η0 (d)) − 0      Kh (Di − d) − E[Kh (D − d)]
          N                                                 fD (d)
              i=1
            (2)
       +R         (d)                                                                                 (B.4)

where ξ˙i := ξi − 1 and supd∈D |R(2) (d)| = op ((N h)−1/2 ).
