<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2408.10509.pdf
backend: pdftotext
part: 5/6
-->

# Appendix C: Proofs 1

Appendix C: Proofs

Proof of Theorem 2.1: By definition, AT T (d) = E[Yt (d) − Yt (0)|D = d]. First,

                           E[Yt − Yt−1 |D = d] = E[Yt (d) − Yt−1 (0)|D = d]                           (C.1)

by the fact that Yt = Yt (D) and Yt−1 = Yt−1 (0).

   Second,

                                          fD|X (d)
                                                      
             E (Yt − Yt−1 )1{D = 0}
                                     fD (d)P (D = 0|X)
                                   fD|X (d)
                                                      
             = E (Yt − Yt−1 )                    |D = 0 P (D = 0)
                              fD (d)P (D = 0|X)
                                                      fD|X (d|x)P (D = 0)
                Z
             = E[(Yt (0) − Yt−1 (0))|X = x, D = 0]                        f      (x)dx
                                                     fD (d)P (D = 0|X = x) X|D=0
                Z
             = E[(Yt (0) − Yt−1 (0))|X = x, D = d]
                fD|X=x (d)P (D = 0) P (D = 0|X = x)fX (x)
             ×                                             dx
              fD (d)P (D = 0|X = x)        P (D = 0)
              Z
             = E[(Yt (0) − Yt−1 (0))|X = x, D = d]fX|D=d (x)dx

             = E[(Yt (0) − Yt−1 (0))|D = d]                                                           (C.2)

where the first equality holds by the law of total probability, the second equality holds by the law
of iterated expectation, the third equality holds by that Yt = Yt (D) and Yt−1 = Yt−1 (0), the fourth
equality holds by Bayes’ rule and conditional parallel trend, and the fifth equality holds by the law
of iterated expectation.

                                                   28

   Then combining the above results, we have

                                                                        fD|X (d)
                                                                                      
                 E[Yt − Yt−1 |D = d] − E (Yt − Yt−1 )1{D = 1}
                                                                   fD (d)P (D = 0|X)
                     = E[Yt (d) − Yt−1 (0)|D = d] − E[Yt (0) − Yt−1 (0)|D = d]
                     = E[Yt (d) − Yt (0)|D = d]
                     = AT T (d)                                                                  (C.3)

   Next, for repeated cross-sections, we have
                                 
                 T −λ
             E          Y |D = d
               λ(1 − λ)
                                               
                       T −λ
              =E E             Y |D = d, T |D = d
                      λ(1 − λ)
                                          
                    T −λ
              =E            Y |D = d, T = 1 P (T = 1|D = d)
                   λ(1 − λ)
                                          
                    T −λ
              +E            Y |D = d, T = 0 P (T = 0|D = d)
                   λ(1 − λ)
                                                                         
                     1−λ                             0−λ
              =E            Y |D = d, T = 1 λ + E            Y |D = d, T = 0 (1 − λ)
                   λ(1 − λ)                         λ(1 − λ)
              = E[Yt |D = d] − E[Yt−1 |D = d]
              = E[Yt − Yt−1 |D = d]                                                              (C.4)

where the first equality holds by law of iterated expectation, the second equality holds by definition,
and the last two equalities hold by Assumption 2.2.

   Similarly, by law of iterated expectation and Assumption 2.2

                                                  fD|X (d)
                                                                
                         T −λ
                      E          Y 1{D = 0}
                        λ(1 − λ)            fD (d)P (D = 0|X)
                                                     fD|X (d)
                                                                         
                             1−λ
                      =E            Y 1{D = 0}                     |T = 1 P (T = 1)
                           λ(1 − λ)             fD (d)P (D = 0|X)
                                                     fD|X (d)
                                                                         
                            0−λ
                      +E            Y 1{D = 0}                     |T = 0 P (T = 0)
                           λ(1 − λ)            fD (d)P (D = 0|X)
                                                      fD|X (d)
                                                                         
                             1−λ
                      =E            Yt 1{D = 0}                     |T = 1 λ
                           λ(1 − λ)             fD (d)P (D = 0|X)
                                                        fD|X (d)
                                                                           
                            0−λ
                      +E            Yt−1 1{D = 0}                     |T = 0 (1 − λ)
                           λ(1 − λ)               fD (d)P (D = 0|X)
                                                      fD|X (d)
                                                                   
                      = E (Yt − Yt−1 )1{D = 0}                                                   (C.5)
                                                fD (d)P (D = 0|X)

                                                    29

and the claim follows from the panel case.

                                                                                                 □

Proof of Lemma 3.1: First, consider the panel case. Define the unadjusted score φh as

                               Kh (D − d)g0 (X) − 1{D = 0}fh0 (d|X)
                    φh := ∆Y                 0 (d)g (X)             − AT Th (d)               (C.6)
                                            fD     0

                                                      0 (d) := f (d), f 0 (d|X) := E[K (D −d)|X],
where we use the following notation: ∆Y = Yt −Yt−1 , fD         D      h              h
g0 (X) := P (D = 0|X). We will add an adjustment term to the original score so that the new score
satisfies the Neyman orthogonality w.r.t. the infinite-dimensional parameters.

   The two infinite-dimensional nuisance parameters are fh0 (d|X) and g0 (X), and in particular,
they satisfy fh0 (d|X) = E[Kh (D − d)|X] and g0 (X) = E[1{D = 0}|X]. Then the adjustment term
ch takes the form

             ch := (Kh (D − d) − fh0 (d|X))E[∂1 φh |X] + (1{D = 0} − g0 (X))E[∂2 φh |X]       (C.7)

where ∂1 and ∂2 denote the partial derivatives with respect to fh0 (d|X) and g0 (X) respectively.
Then, we have

                                   fh0 (d|X)
      ch = (1{D = 0} − g0 (X))   0 (d) · g 2 (X) E[∆Y 1{D = 0}|X]
                                fD        0
                          0                1
         − (Kh (D − d) − fh (d|X)) 0               E[∆Y 1{D = 0}|X]
                                   fD (d) · g0 (X)
             [1{D = 0} − g0 (X)]fh0 (d|X) − [Kh (D − d) − fh0 (d|X)]g0 (X) E[∆Y 1{D = 0}|X]
         =                            0 (d) · g (X)
                                     fD        0                                g0 (X)
                                                                           |      {z      }
                                                                                 0 (X)
                                                                              :=E∆Y

             1{D = 0}fh0 (d|X) − Kh (D − d)g0 (X) 0
         =                0 (d) · g (X)          E∆Y (X)                                      (C.8)
                        fD         0

       0 (X) = E[∆Y 1{D = 0}|X]/g (X) = E[∆Y |D = 0, X]. In particular, note that ψ in
where E∆Y                        0                                                 h
the lemma satisfies ψh = φh + ch .

   Now it remains to show the new score ψh satisfies Neyman orthogonality w.r.t. the nuisance
parameters, fh0 (d|X), g0 (X), and E∆Y
                                    0 (X). First, we need to check the moment condition E[ψ ] = 0.
                                                                                           h
Since E[φh ] = 0, we only need to check E[ch ] = 0:

                          1{D = 0}fh0 (d|X) − Kh (D − d)g0 (X) 0
                                                                       
               E[ch ] = E              0 (d) · g (X)          E∆Y (X)
                                     fD         0

                                                30

                           E[1{D = 0}|X]fh0 (d|X) − E[Kh (D − d)|X]g0 (X) 0
                                                                                   
                      =E                          0 (d) · g (X)          E∆Y (X)
                                                 fD        0
                           g0 (X)fh0 (d|X) − fh0 (d|X)g0 (X) 0
                                                                      
                      =E               0 (d) · g (X)           E∆Y (X)
                                     fD         0

                      =0                                                                     (C.9)

where the second equality holds by the law of iterated expectation and the third equality holds by
the fact that E[Kh (D − d)|X] = fh0 (d|X) and E[1{D = 0}|X] = g0 (X).

   Second, we need to show the Gateaux derivative of the score w.r.t. the nuisance parameters
η0 (d) := (fh0 (d|X), g0 (X), E∆Y
                               0 (X)) vanishes at zero, that is, we need to show

                               ∂r E[ψh (η0 (d) + r(η(d) − η0 (d)))]|r=0 = 0.                (C.10)

We use the notation η(d) without the subscript 0 to denote generic nuisance parameters in the set
TN (d). By the definition of Gateaux derivative, it suffices to show the partial derivative is zero
w.r.t. each nuisance parameter separately. In particular, in the following derivations, by assump-
tion in the lemma, we can use the dominated convergence theorem to interchange the derivatives
and the expectations.

w.r.t fh (d|X):

              ∂r E[ψh (fh0 (d|X) + r(fh (d|X) − fh0 (d|X)))]|r=0
                                                            
                     1{D = 0}∆fh (d|X)               0
               =E          0 (d) · g (X)   (∆Y − E∆Y (X))
                         fD         0
                                                                                      
                     E[1{D = 0}∆Y |X] ∆fh (d|X) E[1{D = 0}|X] ∆fh (d|X) 0
               =E                              0 (d)   −                 0 (d) E∆Y (X)
                              g0 (X)         fD                g0 (X)   fD
                                                                     
                      0        ∆fh (d|X) g0 (X) ∆fh (d|X) 0
               = E E∆Y (X)          0 (d) −            0 (d) E∆Y (X)
                                  fD        g0 (X) fD
              =0                                                                            (C.11)

where the first equality holds by definition with ∆fh (d|X) := fh (d|X) − fh0 (d|X), the second
equality holds by the law of iterated expectation, and the third equality holds by the fact that
                           0 (X) and E[1{D = 0}|X] = g (X).
E[∆Y 1{D = 0}|X]/g0 (X) = E∆Y                         0

w.r.t g(X):

               ∂r E[ψh (g0 (X) + r(g(X) − g0 (X)))]|r=0

                                                    31

                    1{D = 0}fh0 (d|X)
                                                                   
                                                    0
              =E −      0 (d) · g 2 (X) (∆Y − E∆Y (X))∆g(X)
                       fD        0
                            E[1{D = 0}∆Y |X] fh0 (d|X)
                 
              = E − ∆g(X)                             0 (d)g (X)
                                     g0 (X)         fD      0
                                           0                
                      E[1{D = 0}|X] fh (d|X) 0
              + ∆g(X)
                          g0 (X)2          fD0 (d) E∆Y (X)

                                         fh0 (d|X)             g0 (X) fh0 (d|X) 0
                                                                                       
                              0
              = E − ∆g(X)E∆Y (X) 0
                                      fD (d)g0 (X)
                                                     + ∆g(X)
                                                               g0 (X)2 fD 0 (d) E∆Y (X)

                 =0                                                                          (C.12)

where the first equality holds by chain rule and the definition ∆g(X) := g(X) − g0 (X), second
equality holds by law of iterated expectation, and the third equality holds by that E[∆Y 1{D =
                0 (X) and E[1{D = 0}|X] = g (X).
0}|X]/g0 (X) = E∆Y                         0

w.r.t E∆Y (X):

                            0                     0
                  ∂r E[ψh (E∆Y (X) + r(E∆Y (X) − E∆Y (X)))]|r=0
                             Kh (D − d)g0 (X) − 1{D = 0}fh0 (d|X)
                      = E[                 0 (d) · g (X)          ∆E(X)]
                                        fD          0
                             E[Kh (D − d)|X]g0 (X) − E[1{D = 0}|X]fh0 (d|X)
                      = E[                      0 (d) · g (X)               ∆E(X)]
                                             fD          0

                      =0                                                                     (C.13)

                                                                 0 (X), the second equality holds
where the first line holds by definition with ∆E(X) = E∆Y (X) − E∆Y
by law of iterated expectation, and the last equality holds by the definition that E[Kh (D − d)|X] =
fh0 (d|X) and E[1{D = 0}|X] = g0 (X).

   This shows that the score ψh is Neyman orthogonal w.r.t. the infinite-dimensional nuisance pa-
rameters. The proof for the repeated cross-sectional case follows the same argument by replacing
         T −λ
∆Y with λ(1−λ) Y.                                                                                 □

Proof of Lemma 3.1 (Efficient Influence Function Approach): We introduce the method
discussed in Hines et al. (2022) and adapt it to the panel case. The result for the repeated cross-
sectional case only requires minor modifications.

   Let P denote the true distribution of the observed data and Ψ(P) denote the target parameter.

                                                    32

Let P̃ denote another fixed distribution, and define a mixture

                                            Pt = tP̃ + (1 − t)P.                               (C.14)

Note that for a distribution function f at õ, we have

                                                           dft (o)
                     ft (o) = tδõ (o) + (1 − t)f (o) =⇒            = δõ (o) − f (o),         (C.15)
                                                             dt t=0

where δõ denote the dirac delta at õ.

   The sensitivity of Ψ to changes in P in the direction of P̃ can be characterized by the Gateaux
derivative (if exists)

                        Ψ(Pt ) − Ψ(P)
                                                           Z
                                        dΨ(Pt )                                     
                   lim                =             =          ψ(o, P) dP̃(o) − dP(o)          (C.16)
                    t↓0        t          dt    t=0

where the second equality holds by Riesz representation theorem. The canonical gradient ψ is
referred to as the efficient influence function. The efficient influence function has mean zero, which
implies that
                                             Z
                            dΨ(Pt )
                                        =        ψ(o, P)dP̃(o) = EP̃ [ψ(O, P)].
                              dt    t=0

Hines et al. (2022) considers perturbing Ψ(Pt ) at a single point, and the above expression gives the
efficient influence function directly as

                                           dΨ(Pt )
                                                       = ψ(o, P).                              (C.17)
                                             dt    t=0

   In our case, while AT T (d) is not path-wise differentiable, the kernel smoothed AT Th (d) is
path-wise differentiable. Therefore, we will work with AT Th (d) instead. Specifically,
                                                                              
                              Kh (D − d)P (D = 0|X) − 1{D = 0}E[Kh (D − d)|X]
               Ψ(P) = E ∆Y                                                                     (C.18)
                                                 fD (d)P (D = 0|X)
                          Kh (s − d)
                      Z
                    = ι              f∆Y,D (ι, s)dιds
                            fD (d)
                             Kh (s − d)
                      Z
                    − ι                     fD,X (s, x)fD,∆Y,X (0, ι, x)dιdsdx                 (C.19)
                          fD (d)fD,X (0, x)

Perturbing at a point õ = (∆Y = ι̃, D = 0, D = s̃, X = x̃), we have (suppressing random variable

                                                      33

subscripts)

      dΨ(Pt )
      Z dt
           Kh (s − d) d
    = ι                      ft (ι, s)      dιds
                 fd       dt            t=0
               Kh (s − d)
         Z
                                                    d
      + ι                    f (s, x)f (0, ι, x) ft (0, x)
               fd f 2 (0, x)                       dt             t=0
           Kh (s − d)                  d
      −ι                  f (0, ι, x) ft (s, x)
           fd f (0, x)                dt             t=0
           Kh (s − d)               d
      −ι                  f (s, x) ft (0, ι, x)          dιdsdx                                      (C.20)
           fd f (0, x)             dt                t=0
        Kh (s̃ − d)            Kh (s − d)
                          Z
    =ι̃                + ι                    f (s, x̃)f (0, ι, x̃)δ0 dιds
            fd                 fd f 2 (0, x̃)
               Kh (s̃ − d)                            Kh (s − d)
         Z                                       Z
      − ι                    f (0, ι, x̃)dι − ι̃                    f (s, x̃)δ0 ds
               f f (0, x̃)                             fd f (0, x̃)
         Z d                                                                                    
                  Kh (s − d)                             Kh (s − d)
                                                   Z
      −         ι               f (ι, s)dιds − ι                       f (s, x)f (0, ι, x)dιdsdx     (C.21)
                       fd                                fd f (0, x)
         |                                              {z                                       }
                                                  =Ψ(P)
        Kh (s̃ − d) E[∆Y |D = 0, X = x̃]
    =ι̃            −                       E[Kh (D − d)|X = x̃]1{D = 0}
            fd         fd P (D = 0|X = x̃)
         Kh (s̃ − d)
      −               E[∆Y |D = 0, X = x̃]
               fd
                   ι̃
      −                       E[Kh (D − d)|X = x̃]1{D = 0}
         fd P (D = 0|X = x̃)
       − Ψ(P)                                                                                        (C.22)
                                         Kh (s̃ − d)P (D = 0|X = x̃) − 1{D = 0}E[Kh (D − d)|X = x̃]
    =(ι̃ − E[∆Y |D = 0, X = x̃])
                                                              fd P (D = 0|X = x̃)
       − Ψ(P).                                                                                       (C.23)

Note that this is the same expression as the score we presented in Lemma 3.1.                            □

Proof of Lemma 4.2: We focus on the panel case. The bias Bh (d) is defined as

                Bh (d) :=AT T (d) − AT Th (d)
                                                             fD|X (d|X)
                                                                          
                        =E[∆Y |D = d] − E ∆Y 1{D = 0}
                                                         fD (d)P (D = 0|X)
                                                                               
                                Kh (D − d)P (D = 0|X) − 1{D = 0}E[Kh (D − d)|X]
                        −E ∆Y
                                                fD (d)P (D = 0|X)

                                                             34

                                                      
                                             Kh (D − d)
                     = E[∆Y |D = d] − E ∆Y
                                                fD (d)
                                     fD|X (d|X) − E[Kh (D − d)|X]
                                                                 
                     −E ∆Y 1{D = 0}                                 .                       (C.24)
                                           fD (d)P (D = 0|X)

First, note that
                                                         
                                              Kh (D − d)
                    E[∆Y |D = d] − E ∆Y
                                                fD (d)
                                                       1 s − d
                    Z                   Z            Z
                       f∆Y,D (t, d)             1
                   = t              dt − t                K      f∆Y,D (t, s)dsdt
                          fD (d)              fD (d)   h    h
                    Z
                           t        (2)
                   = C1         h2 f∆Y,D (t, d)dt + o(h2 )
                        fD (d)
                   =O(h2 )                                                                  (C.25)

where the first equality holds by definition, the second equality holds by change of variables and
Taylor expansion (see Lemma 5.1 in Fan and Yao (2003)), and the last equality holds by assumption.

   Second, by the same argument using the change of variables and Taylor expansion, we have

                                                   1 d − s
                                               Z
                   E[Kh (D − d)|X = x] =             K       fD|X (s|x)ds
                                                   h    h
                                               Z
                                           =       K(u)fD|X (d + hu|x)du
                                                                 (2)
                                           = fD|X (d|x) + C2 h2 fD|X (d|x) + o(h2 ).        (C.26)

                                        (2)
Then by the uniform boundedness of fD|X (d|x) and assumptions on ∆Y, fD (d), P (D = 0|X), ap-
plying the dominated convergence theorem, we have

                                       fD|X (d|X) − E[Kh (D − d)|X]
                                                                    
                         E ∆Y 1{D = 0}                                                      (C.27)
                                             fD (d)P (D = 0|X)
                                                  (2)
                                               fD|X (d|X)    
                             2
                        =C3 h E ∆Y 1{D = 0}                     + o(h2 )                    (C.28)
                                            fD (d)P (D = 0|X)
                        =O(h2 ).                                                            (C.29)

Combining the two results, we have Bh (d) = O(h2 ), which completes the proof. The proof for the
                                                                              T −λ
repeated cross-sectional case follows the same argument by replacing ∆Y with λ(1−λ) Y.           □

Proof of Theorem 4.2, Panel: Let TN (d) be the set of η(d) := (fh (d|X), g(X), E∆Y (X)) such
that ∥fh (d|X) − fh0 (d|X)∥P,2 ≤ h−1/2 εN , ∥g(X) − g0 (X)∥P,2 ≤ εN , ∥E∆Y (X) − E∆Y
                                                                                  0 (X)∥
                                                                                         P,2 ≤ εN ,
κ < ∥g(X)∥P,∞ < 1 − κ, c < fh (d|X) < C, and ∥E∆Y (X)∥P,∞ < C. Let FN (d) be the set

                                                    35

of functions f > c such that |f − fD  0 (d)| ≤ (N h)−1/2 . Then Assumption 4.8 implies that, with

probability tending to 1, η̂k (d) ∈ TN (d) and fˆk (d) ∈ FN (d). Throughout the proof, we use N to
denote the sample size and n := N/K to denote the size of the subsamples. In particular, since K
is fixed, n ≍ N .

   To simplify notation, let θ0 (d) denote the true AT T (d), θ0h (d) denote the true AT Th (d), and
θ̂h (d) denote our cross-fitted estimator. In particular, recall that our estimator is

                    K
                  1 X 1 X Kh (Di − d)ĝk (Xi ) − 1{Di = 0}fˆh,k (d|Xi )                   
     θ̂h (d) :=                                                         ∆Yi − Ê∆Y,k (Xi ) .                         (C.30)
                  K   n
                    k=1     i∈Ik
                                          fˆk (d)ĝk (Xi )

Then we have the following decomposition

                                   θ̂h (d) − θ0 (d) = θ̂h (d) − θ0h (d) + θ0h (d) − θ0 (d)                           (C.31)
                                                      |       {z      } |        {z      }
                                                                    (1)                  (2)

where (1) will be our main focus while the bias term (2) is shown in Lemma 4.2 to be O(h2 ) and
asymptotically negligible by the assumption of the under-smoothing bandwidth.

   By definition,

                     √                                   √           K
                                                                 1 X
                         N (θ̂h (d) − θ0h (d)) =             N       En,k [ψh (Zi , θ0,h , fˆk (d), η̂k (d))]        (C.32)
                                                                 K
                                                                    k=1

where ψh is defined as in (3.8), and En,k (f ) = n1
                                                                          P
                                                                            i∈Ik f (Zi ) denotes the empirical average of
a generic function f over the set Ik . Then we have the following decomposition, using Taylor’s
theorem:

    √                       √ 1 XK
                                                          0
     N (θ̂h (d) − θ0h (d)) = N     En,k [ψh (Z, θ0h (d), fD (d), η̂k (d))]                                           (C.33)
                               K
                                                k=1
                                    √            K
                                            1   X
                              +         N                                       0
                                                      En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))](fˆk (d) − fD
                                                                                                            0
                                                                                                              (d))   (C.34)
                                            K
                                                k=1
                                    √            K
                                            1   X
                              +         N             En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))](fˆk (d) − fD
                                                                                                           0
                                                                                                             (d))2   (C.35)
                                            K
                                                k=1

where f¯k ∈ (fD
              0 (d), fˆ (d)). This decomposition provides a roadmap for the remainder of the proof.
                       k
There are roughly four steps. In the first step, we show the second-order term (C.35) vanishes
rapidly and does not contribute to the asymptotic variance. In the second step, we bound the first-
order term (C.34), which potentially contributes to the asymptotic variance. In step 3, we expand
(C.33) around the nuisance parameter η̂k (d), in which the first-order bias disappears by Neyman

                                                                    36

orthogonality, and we show the second-order terms have no impact on the asymptotics under our
assumptions. In the final step, we verify the results used in the first two steps and conclude.

   Before we start the main proof, we state two well-known results that will be used in the proof.
For an i.i.d. sample {Di }ni=1 , the kernel estimator for the density fD (d) := fD
                                                                                 0 (d) in our setting is

defined as
                                                          n
                                                      1   X
                                               fˆd :=           Kh (Di − d).                              (C.36)
                                                      n
                                                          i=1

Then,

                        fˆd − fD
                               0
                                 (d) = fˆd − E[Kh (D − d)] − (fD
                                                               0
                                                                 (d) − E[Kh (D − d)]).                    (C.37)

One can show that (see for example, Härdle (1990) and Fan and Yao (2003))

                                                                           1
                                           fˆd − E[Kh (D − d)] = Op ((nh)− 2 )                            (C.38)
                                        0
                                       fD (d) − E[Kh (D − d)] = O(h2 ).                                   (C.39)

Therefore, for an under-smoothing h = o(n−1/5 ), we have fˆd − fD
                                                                0 (d) = O ((nh)−1/2 ) and (fˆ −
                                                                         p                   d
 0 (d))2 = O ((nh)−1 ).
fD          p

Step 1: Second Order Terms

   First, we consider (C.35). By triangle inequality, we have

          |En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − E[∂f2 ψh (Z, θ0h (d), fD
                                                                              0
                                                                                (d), η0 (d))]|
             ≤ |En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − En,k [∂f2 ψh (Z, θ0h (d), fD
                                                                                       0
                                                                                         (d), η0 (d))]|   (C.40)
               |                                           {z                                        }
                                                          J1k

          + |En,k [∂f2 ψh (Z, θ0h (d), fD
                                        0
                                          (d), η0 (d))] − E[∂f2 ψh (Z, θ0h (d), fD
                                                                                 0
                                                                                   (d), η0 (d))]| .       (C.41)
               |                                          {z                                          }
                                                          J2k

                                0 (d) is bounded away from zero,
To bound J2k , note that since fD

                                                     2
        ∂f2 ψh (Z, θ0h (d), fD
                             0
                               (d), η0 (d)) =      0
                                                                             0
                                                           (ψh (Z, θ0h (d), fD (d), η0 (d)) + θ0h (d))    (C.42)
                                                 (fD (d))2

which implies that
                                                                                                    2 
         2                1X 2                  0                   2                  0
      E[J2k ]=E             ∂f ψh (Z, θ0h (d), fD (d), η0 (d)) − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]
                          n
                            i∈Ik

                                                                37

                                                                   2 
                       1X 2                        0
                =E            ∂f ψh (Z, θ0h (d), fD  (d), η0 (d))
                      n
                         i∈Ik
                      2                  0
                                                         2
                − E[∂f ψh (Z, θ0h (d), fD  (d), η0 (d))]
                  1
                ≤ E[(∂f2 ψh (Z, θ0h (d), fD 0
                                              (d), η0 (d)))2 ]
                  n
                ≲ E[Kh2 (D − d)]/N
                ≲ (hN )−1 ,                                                                                           (C.43)

where the third line holds by Cauchy-Schwarz inequality and Jensen’s inequality, the fourth line
holds by the boundedness assumption on the components of the score, and the last line holds
by the assumption on the kernel function K. Then by the Markov’s inequality, we have J2k ≤
Op (N h)−1/2 .
            

   Next, for J1k , we have

           2
        E[J1k |Ikc ]
         = E[|En,k [∂f2 ψh (Z, θ0h (d), f¯k , η̂k (d))] − En,k [∂f2 ψh (Z, θ0h (d), fD
                                                                                     0
                                                                                       (d), η0 (d))]|2 |Ikc ]
                                                                                            0
         ≤             sup           E[|∂f2 ψh (Z, θ0h (d), f, η(d)) − ∂f2 ψh (Z, θ0h (d), fD (d), η0 (d))|2 |Ikc ]
             f ∈FN (d),η(d)∈TN (d)
                                                                                            0
         ≤             sup           E[|∂f2 ψh (Z, θ0h (d), f, η(d)) − ∂f2 ψh (Z, θ0h (d), fD (d), η0 (d))|2 ]
             f ∈FN (d),η(d)∈TN (d)

         ≲ h−1 ε2N ,          (a)

where the second line holds by Cauchy-Schwarz inequality and the definition of supremum over the
sets FN (d) and TN (d), and the third line holds since the supremum does not depend on the sample
Ikc . Then by conditional Markov’s inequality, J1k ≤ Op (h−1/2 εN ). Using the previous result that
(fˆk (d) − f 0 (d))2 = Op ((N h)−1 ), we conclude that (C.35) = op (1). We will verify (a) at the end of
          D
this section.

Step 2: First-Order Terms

   To bound (C.34), we first use the triangle inequality to obtain the decomposition

                                        0                                        0
             |En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]|
                                      0                                            0
         ≤ |En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − En,k [∂f ψh (Z, θ0h (d), fD (d), η0 (d))]|                   (C.44)
           |                                           {z                                        }
                                                          J3k
                                      0                                       0
         + |En,k [∂f ψh (Z, θ0h (d), fD (d), η0 (d))] − E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))]| .                      (C.45)
           |                                        {z                                      }
                                                         J4k

                                                               38

We first bound J4k . By definition, we have

                           0                               1                      0
       ∂f ψh (Z, θ0h (d), fD (d), η0 (d)) = −             0 (d) (ψh (Z, θ0h (d), fD (d), η0 (d)) + θ0h (d)).    (C.46)
                                                         fD

By the boundedness assumption,

                                           1
                          2
                       E[J4k ]≤                                     0
                                             E[(∂f ψh (Z, θ0h (d), fD (d), η0 (d)))2 ] ≲ (N h)−1 .              (C.47)
                                           N

Then by Markov’s inequality, we have J4k ≤ Op ((N h)−1/2 ). With the assumption that N h → ∞,
we have J4k = op (1).

   Second, to bound J3k , note that

           2
        E[J3k |Ikc ]
                                        0                                            0
         = E[|En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] − En,k [∂f ψh (Z, θ0h (d), fD (d), η0 (d))]|2 |Ikc ]
                                                  0                                  0
         ≤      sup        E[|∂f ψh (Z, θ0h (d), fD (d), η(d)) − ∂f ψh (Z, θ0h (d), fD (d), η0 (d))|2 |Ikc ]
             η(d)∈TN (d)
                                                  0                                  0
         ≤      sup        E[|∂f ψh (Z, θ0h (d), fD (d), η(d)) − ∂f ψh (Z, θ0h (d), fD (d), η0 (d))|2 ]
             η(d)∈TN (d)

         ≲ h−1 ε2N             (b)

where the first equation holds by definition, the second line holds by Cauchy-Schwarz, and the
third line holds by the construction that all the parameters are estimated using auxiliary sample
Ikc . Then we conclude with the conditional Markov’s inequality that J3k = op (1) provided that
h−1 ε2N = o(1), which holds by assumption. We will show (b) at the end of this section. Therefore,

                                 0                                        0
       En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))] = E[∂f ψh (Z, θ0h (d), fD (d), η0 (d))] +op (1).              (C.48)
                                                    |                {z                 }
                                                                              :=Sf0

   Note that the kernel density estimator satisfies (fˆk (d) − fD
                                                                0 (d)) = O ((N h)−1/2 ), so we can
                                                                          p
rewrite (C.34) as

                               √            K
                                       1 X
                (C.34) =           N                                 0
                                           En,k [∂f ψh (Z, θ0h (d), fD (d), η̂k (d))](fˆk (d) − fD
                                                                                                 0
                                                                                                   (d))
                                       K
                                           k=1
                               √            K
                                       1   X
                           =       N              Sf0 (fˆk (d) − fD
                                                                  0
                                                                    (d)) + op (h−1/2 )
                                       K
                                           k=1
                               √            N
                                       1   X
                           =       N              Sf0 (Kh (Di − d) − E[Kh (D − d)]) + op (h−1/2 )               (C.49)
                                       N
                                            i=1

                                                                  39

where the last equality holds since
                                                      X
                    fˆk (d) − fD
                               0
                                 (d) = (N − n)−1              Kh (Di − d) − E[Kh (D − d)] + O(h2 )
                                                      i∈Ikc

with N − n the sample size of each auxiliary subsample used to estimate the nuisance parameters,
h being an under-smoothing bandwidth, and the fact that K −1 K          ˆ
                                                                P
                                                                   k=1 (fk (d) − E[Kh (D − d)]) =
N −1 N
     P
       i=1 (Kh (Di − d) − E[Kh (D − d)]). In particular, the kernel expression in the last line is
mean-zero and it will contribute to the asymptotic variance.

Step 3: “Neyman Term”

   Now we consider (C.33), which we can rewrite as

    √           K
            1 X                        0
        N       En,k [ψh (Z, θ0h (d), fD (d), η̂k (d))]
            K
               k=1
                 N
       1        X
                                 0
     =√       ψh (Zi , θ0h (d), fD (d), η0 (d))                                                             (C.50)
        N i=1
         √          K
               1 X                         0                                           0
     +       N     (En,k [ψh (Z, θ0h (d), fD (d), η̂k (d))] − En,k [ψh (Zi , θ0h (d), fD (d), η0 (d))]) .   (C.51)
               K   |                                        {z                                        }
                    k=1
                                                                    Rnk

Since K is fixed, n = O(N ), it suffices to show that Rnk = op (N −1/2 h−1 ), so it vanishes when
scaled by the (square root of) asymptotic variance. Note that by triangle inequality, we have the
following decomposition

                                                               R1k + R2k
                                                  |Rn,k | ≤       √                                         (C.52)
                                                                    n

where

                                          0                                        0
            R1k := |Gnk [ψh (Z, θ0h (d), fD (d), η̂k (d))] − Gnk [ψh (Z, θ0h (d), fD (d), η0 (d))]|         (C.53)

                     √           Pn
with Gnk (f ) =           n n1    i=1 f (Zi ) − E[f (Z)] denoting the empirical process, and, with some abuse
of notation, it will also be used to denote conditional version of the empirical process conditioning
on the auxiliary sample Ikc . Moreover,
                    √                        0
         R2k :=         n|E[ψh (Z, θ0h (d), fD (d), η̂k (d))|Ikc ] − E[ψh (Z, θ0h (d), fD
                                                                                        0
                                                                                          (d), η0 (d))]|.   (C.54)

                                                                                            i
   First, we consider R1k . For simplicity, let’s suppress other arguments in ψ and denote ψη(d) :=

                                                               40

                   0 (d), η(d)). Then, by the definition of the empirical process, we have
ψh (Zi , θ0h (d), fD

                                          n
                                     √ 1X
         Gnk ψη̂k (d) − Gnk ψη0 (d) = n     ψη̂i (d) − ψηi 0 (d) − E[ψη̂i k (d) |Ikc ] + E[ψηi 0 (d) ]   (C.55)
                                        n   | k     i=1             {z                              }
                                                                                  :=∆ik

In particular, it can be shown that E[∆ik ∆jk |Ikc ] = 0 for all i ̸= j using the law of iterated
expectation, the i.i.d. assumption of the data, and the fact that the nuisance parameter η̂k (d) is
estimated using the auxiliary sample Ikc . Then, we have

                                   2
                                E[R1k |Ikc ] ≤ E[∆2ik |Ikc ]
                                                ≤ E[(ψη̂i k (d) − ψηi 0 (d) )2 |Ikc ]
                                                                      i
                                                ≤         sup     E[(ψη(d) − ψηi 0 (d) )2 |Ikc ]
                                                    η(d)∈TN (d)
                                                                      i
                                                ≤         sup     E[(ψη(d) − ψηi 0 (d) )2 ]
                                                    η(d)∈TN (d)

                                                ≲ h−1 ε2N              (c)

and using the conditional Markov’s inequality, we conclude that R1k = Op (h−1/2 εN ).
                                                                  0 (d), η (d))] = 0, so it suffices to
   Now we bound R2k . Note that by definition, E[ψh (Z, θ0h (d), fD       0
                         0 (d), η̂ (d))|I c ]. Suppressing other arguments in the score, define
bound E[ψh (Z, θ0h (d), fD        k      k

                                 hk (r) := E[ψh (η0 (d) + r(η̂k (d) − η0 (d)))|Ikc ]                     (C.56)

where by definition hk (0) = E[ψh (η0 (d))|Ikc ] = 0 and hk (1) = E[ψh (η̂k (d))|Ikc ]. Use Taylor’s theo-
rem, expand hk (1) around 0, we have

                                                           1 ′′
                                hk (1) = hk (0) + h′k (0) + hk (r̄),               r̄ ∈ (0, 1).          (C.57)
                                                           2

Note that, by Neyman orthogonality,

                               h′k (0) = ∂η (d)E[ψh (η0 (d))][η̂k (d) − η0 (d)] = 0                      (C.58)

and use that fact that hk (0) = 0, we have
                               √                  √        ′′
                      R2k =        n|hk (1)| =        n|hk (r̄)|
                                                       √ 2
                           ≤           sup               n|∂r E[ψh (η0 (d) + r(η(d) − η0 (d)))]|
                               r∈(0,1),η(d)∈TN (d)
                               √
                           ≲       nh−1/2 ε2N              (d)

                                                                 41

   Combining the above results, we conclude that
                                       √                            √
                                           N Rn,k ≲ h−1/2 εN +          N h−1/2 ε2N ,                (C.59)
                                       √
and for εN = o(N −1/4 ), we have            N Rn,k = op (h−1/2 ).

Step 4: Auxiliary Results

   In this section, we show the auxiliary results (a)-(d) used in the previous steps. We first show
(c) as it will also be used to bound other results.

   Recall that
                               (c) :          sup        E[(ψη(d) − ψη0 (d) )2 ] ≲ h−1 ε2N .
                                           η(d)∈TN (d)

By definition,

                           Kh (D − d)g(X) − 1{D = 0}fh (d|X)              
       ψη(d) − ψη0 (d) =                0                    ∆Y − E∆Y (X)
                                       fD (d)g(X)
                           Kh (D − d)g0 (X) − 1{D = 0}fh0 (d|X)               0
                                                                                      
                           −                0                        ∆Y − E∆Y     (X)
                                          fD (d)g0 (X)
                         Kh (D − d) 0
                       =     0 (d)    (E∆Y (X) − E∆Y (X))
                           fD
                                                                      fh0 (d|X)
                                                                                                
                           1{D = 0} fh (d|X)                                              0
                         −      0 (d)             (∆Y  −  E∆Y (X))  −           (∆Y  −  E∆Y (X))
                              fD           g(X)                        g0 (X)
                         Kh (D − d) 0
                       =     0 (d)    (E∆Y (X) − E∆Y (X))
                           fD
                           1{D = 0} fh (d|X) fh0 (d|X)
                                                             
                         −      0 (d)              −            ∆Y
                              fD           g(X)       g0 (X)
                                                             fh0 (d|X) 0
                                                                               
                           1{D = 0} fh (d|X)
                         +                        E∆Y (X) −           E (X)
                              fD0 (d)      g(X)                g0 (X) ∆Y
                       ≲C1 (fh (X) − fh0 (X)) + C2 (g(X) − g0 (X))
                                                      0
                           + C3 Kh (D − d)(E∆Y (X) − E∆Y (X))                                        (C.60)

where the last line can be shown using the “plus-minus” trick with C1 , C2 , C3 being some con-
stants. Then by the definition of TN (d), the assumptions on the rate of convergence of the nuisance
parameters, and E[Kh2 (D − d)] = O(h−1 ), we have

                 sup       E[(ψη(d) − ψη0 (d) )2 ]
             η(d)∈TN (d)

             ≲ ∥fh − fh0 ∥2P,2 + ∥g − g0 ∥2P,2 + ∥Kh (D − d)∥2P,2 ∥E∆Y − E∆Y
                                                                          0
                                                                             ∥2P,2

                                                             42

                + ∥fh − fh0 ∥P,2 ∥g − g0 ∥P,2 + ∥Kh (D − d)∥P,2 ∥fh − fh0 ∥P,2 ∥E∆Y − E∆Y
                                                                                       0
                                                                                          ∥P,2
                                                       0
                + ∥Kh (D − d)∥P,2 ∥g − g0 ∥P,2 ∥E∆Y − E∆Y ∥P,2
                ≲ h−1 ε2N .                                                                                           (C.61)

This shows (c).

   Next, we consider (a). We want to show

       (a) :               sup            E[|∂f2 ψh (Z, θ0h (d), f, η(d)) − ∂f2 ψh (Z, θ0h (d), fD
                                                                                                 0
                                                                                                   (d), η0 (d))|2 ]
                  f ∈FN (d),η(d)∈TN (d)

                                                                                                        ≲ h−1 ε2N

By definition,

                                                     2
                      ∂f2 ψh (Z, θ0h (d), f, η(d)) =    (ψh (Z, θ0h (d), f, η(d)) + θ0h (d))
                                                     f2
                                                        6
                      ∂f3 ψh (Z, θ0h (d), f, η(d)) = − 3 (ψh (Z, θ0h (d), f, η(d)) + θ0h (d)).                        (C.62)
                                                       f

Then using Taylor’s theorem expand ∂f2 ψh (Z, θ0h (d), f, η(d)) around fD
                                                                        0 (d), we have

                                                                    0
                ∂f2 ψh (Z, θ0h (d), f, η(d)) − ∂f2 ψh (Z, θ0h (d), fD (d), η0 (d))
                                       0                                   0
                = ∂f2 ψh (Z, θ0h (d), fD (d), η(d)) − ∂f2 ψh (Z, θ0h (d), fD (d), η0 (d))
                + ∂f3 ψh (Z, θ0h (d), f¯, η(d))(f − fD 0
                                                         (d))
                       2                         0                               0
                = 0           (ψh (Z, θ0h (d), fD  (d), η(d)) − ψh (Z, θ0h (d), fD (d), η0 (d)))        (i)
                  (fD (d))2
                   6
                − ¯3 (ψh (Z, θ0h (d), f¯, η(d)) + θ0h (d))(f − fD  0
                                                                     (d)) (ii)
                  f

By the assumption, on FN (d), f¯ and fD
                                      0 (d) are bounded away from zero, so that (i) is the leading

term that can be bounded with (c). Moreover, by assumption, (ii) = O((N h)−1/2 ), which is
dominated by (i). Therefore we conclude that

          sup             E[|∂f2 ψh (Z, θ0h (d), f, η(d)) − ∂f2 ψh (Z, θ0h (d), fD
                                                                                 0
                                                                                   (d), η0 (d))|2 ] ≲ h−1 ε2N .       (C.63)
  f ∈FN (d),η(d)∈TN (d)

Similarly, by definition,

                                       0                                  0
                   ∂f ψh (Z, θ0h (d), fD (d), η(d)) − ∂f ψh (Z, θ0h (d), fD (d), η0 (d))
                           1                      0                                0
                    =− 0        (ψh (Z, θ0h (d), fD (d), η(d)) − ψh (Z, θ0h (d), fD  (d), η0 (d)))                    (C.64)
                         fD (d)

                                                             43

and using the same arguments as before, (b) follows from (a) and (c).

   Last, we show (d). It suffices to show

                          sup            |∂r2 E[ψh (η0 (d) + r(η(d) − η0 (d)))]| ≲ h−1/2 ε2N .   (C.65)
                   r∈(0,1),η(d)∈TN (d)

By definition,

   ψh (η0 (d) + r(η(d) − η0 (d)))
       Kh (D − d)        0                     0
                                                       
   =       0      ∆Y − (E∆Y (X) + r(E∆Y (X) − E∆Y (X))) −
         fD (d)
   1{D = 0}(fh0 (d|X) + r(fh (d|X) − fh0 (d|X)))       0                     0
                                                                                     
        0                                        ∆Y − E∆Y (X) − r(E∆Y (X) − E∆Y (X))             (C.66)
       fD (d)(g0 (X) + r(g(X) − g0 (X)))

and we take the second-order partial derivatives w.r.t. r term by term. For simplicity, we omit the
derivations, and we have

        ∂r2 ψh (η0 (d) + r(η(d) − η0 (d))) ≍ C̃1 ∆f ∆g + C̃2 ∆E ∆g + C̃3 ∆f ∆E + C̃4 (∆g )2      (C.67)

                                                       0
where ∆f := fh − fh0 , ∆g := g − g0 , and ∆E := E∆Y − E∆Y and C̃1 , C̃2 , C̃3 , C̃4 are some con-
stants. Then by triangle inequality, Cauchy-Schwarz, and the assumption on the space of nuisance
parameters TN (d), we have

                     E[|∂r2 ψh (η0 (d) + r(η(d) − η0 (d)))|]
                                                                             0
                      ≲ ∥fh − fh0 ∥P,2 ∥g − g0 ∥P,2 + ∥fh − fh0 ∥P,2 ∥E∆Y − E∆Y ∥P,2
                                             0
                      + ∥g − g0 ∥P,2 ∥E∆Y − E∆Y ∥P,2 + ∥g − g0 ∥2P,2
                      ≲ h−1/2 ε2N .                                                              (C.68)

Then (d) follows by Jensen’s inequality.

   Combining previous results, we have

                                AT
                                [  T (d) − AT T (d)
                                         N
                                  1 X                    0
                                =     ψh (Zi , θ0h (d), fD (d), η0 (d))                          (C.69)
                                  N
                                        i=1
                                         N
                                    1   X
                                +              Sf0 (Kh (Di − d) − E[Kh (Di − d)])                (C.70)
                                    N
                                         i=1

                                + op ((N h)−1/2 )                                                (C.71)

                                                         44

                             + θ0 (d) − θ0h (d)                                               (C.72)

where (C.69) and (C.70) are averages of i.i.d. zero-mean terms with the variance growing with
kernel bandwidth h, and recall that Sf0 = E[∂f ψh (Z, θ0h (d), fD
                                                                0 (d), η (d))]; (C.71) are the terms
                                                                        0
that vanish when scaled by the (square root of) asymptotic variance; (C.72) is the bias term which
is shown to be of order O(h2 ) in Lemma 4.2.

   Since h grows with sample size N , we use the Lyapunov Central Limit Theorem for triangular
arrays to establish the asymptotic results. Note that the only term in ψh that grows with N is the
kernel term, therefore, it suffices to show that the Lyapunov conditions are satisfied for the kernel
term. Then, we have
                                                                      2
                 E[|Kh (Di − d) − E[Kh (Di − d)]|2 ] ≤ E[ Kh (Di − d) ]
                                                           1 h  t − d i2
                                                       Z
                                                     =        K            fD (t)dt
                                                          h2       h
                                                              Z
                                                       fD (d)
                                                     =          K 2 (u)du + o(h−1 )           (C.73)
                                                         h

where fD (d) denotes the density of D at d, and the last line follows from change of variables.
Moreover, by the same change of variables argument, we have

                E[|Kh (Di − d) − E[Kh (Di − d)]|3 ] ≤ 8E[|Kh (Di − d)|3 ]
                                                       Z
                                                           1    t − d 3
                                                    =8        K           fD (t)dt
                                                          h3       h
                                                             Z
                                                      fD (d)
                                                    =          |K(u)|3 du + o(h−2 ).          (C.74)
                                                       h2

Therefore, we have

                         2
                        σi,N                            0
                             := V ar(ψh (Zi , θ0h (d), fD (d), η0 (d))) = O(h−1 )
                                                      0
                        ri,N := E[|ψh (Zi , θ0h (d), fD (d), η0 (d))|3 ] = O(h−2 )            (C.75)

Then, the Lyapunov condition is satisfied provided that N h → ∞ (which is assumed):

                              ( N         1/3
                               P
                                i=1 ri,N )
                                              = O (N h)−1/6 = o(1).
                                                           
                               PN                                                             (C.76)
                                     2 )1/2
                              ( i=1 σi,N

The same argument holds for (C.70). Therefore, by Lyapunov Central Limit Theorem, together

                                                   45

with assumptions 4.7 and 4.8, we have

                                   AT
                                   [  T (d) − AT T (d)
                                             √             →d      N (0, 1)                    (C.77)
                                        σN / N

with σN defined by
                                                                         2 
                       2                θ0h
                      σN := E      ψh − 0     (Kh (D − d) − E[Kh (D − d)])                     (C.78)
                                       fD (d)

where we have used the fact that Sf0 = −θ0h /fD
                                              0 (d).                                                □

Proof of Theorem 7.5 (Repeated Cross-Sections) The proof for the repeated cross-sectional
case follows very closely to that of the panel case, with only minor modifications due to the presence
of a new parameter λ = P (T = 1), which can be estimated at the parametric rate.

    Let TN (d) be the set of functions η(d) := (fh (d|X), g(X), EλY (X)) such that ∥fh (d|X) −
fh (d|X)∥P,2 ≤ h−1/2 εN , ∥g(X) − g0 (X)∥P,2 ≤ εN , ∥EλY (X) − EλY
 0                                                              0 (X)∥
                                                                       P,2 ≤ εN , κ < ∥g(X)∥P,∞ <
1 − κ, c < fh (d|X) < C, and ∥EλY (X)∥P,∞ < C. Let PN be the set of λ > 0 such that
|λ − λ0 | ≤ N −1/2 . Let FN (d) be the set of f > c such that |f − fD   0 (d)| ≤ (N h)−1/2 . Then

assumption 7.12 implies that, with probability tending to 1, η̂k (d) ∈ TN (d), fˆk (d) ∈ FN (d), and
λ̂k ∈ PN for all k = 1, · · · , K. Throughout the proof, we use N to denote the sample size and
n := N/K to denote the size of the subsamples. In particular, since K is fixed, n ≍ N .

   To simplify notation, let θ0 (d) denote the true AT T (d), θ0h (d) denote the true AT Th (d), and
θ̂h (d) denote our cross-fitted estimator. In particular, recall that our estimator is

                                  K
                               1 X 1 X Kh (Di − d)ĝk (Xi ) − 1{Di = 0}fˆh,k (d|Xi )
                  θ̂h (d) :=                                                                   (C.79)
                               K
                                 k=1
                                     n
                                       i∈Ik
                                                              fˆk (d)ĝk (Xi )
                                                                                
                                                  Ti − λ̂k
                                            ×                  Yi − Ê∆Y,k (Xi )               (C.80)
                                                λ̂k (1 − λ̂k )

Then we have the following

                           θ̂h (d) − θ0 (d) = θ̂h (d) − θ0h (d) + θ0h (d) − θ0 (d)             (C.81)
                                              |       {z      } |        {z      }
                                                    (1)                 (2)

where (1) will be our main focus while the bias term (2) is shown in Lemma 4.2 to be O(h2 ) and
asymptotically negligible by the assumption of the under-smoothing bandwidth h.

                                                     46

   By definition,

                          √                             √            K
                                                              1 X
                              N (θ̂h (d) − θ0h (d)) =       N     En,k [ψh (Zi , θ0,h , fˆk (d), η̂k (d))]        (C.82)
                                                              K
                                                                  k=1

where ψh is defined as in (3.8), and En,k (f ) = n1
                                                                          P
                                                                          i∈Ik f (Zi ) denotes the empirical average of a
generic function f over the set Ik . Then we have the following decomposition, using a multivariate
Taylor’s theorem,
         √
             N (θ̂h (d) − θ0h (d))
              √           K
                      1 X                             0
          =       N       En,k [ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))]                                            (C.83)
                      K
                          k=1
              √            K
                      1   X
                                                               0
          +       N             En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))](λ̂k − λ0 )                        (C.84)
                      K
                          k=1
              √            K
                      1   X
          +       N                                            0
                                En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))](fˆk (d) − fD
                                                                                           0
                                                                                             (d))                 (C.85)
                      K
                          k=1
              √            K
                      1   X
          +       N             En,k [∂λ2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))](λ̂k − λ0 )2                       (C.86)
                      K
                          k=1
              √            K
                      1   X
          +       N             En,k [∂f2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))](fˆk (d) − fD
                                                                                           0
                                                                                             (d))2                (C.87)
                      K
                          k=1
              √            K
                      1   X
          +       N             En,k [∂λ ∂f ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))](fˆk (d) − fD
                                                                                             0
                                                                                               (d))(λ̂k − λ0 )    (C.88)
                      K
                          k=1

where λ̄k ∈ (λ0 , λ̂k ) and f¯k ∈ (fD
                                    0 (d), fˆ (d)). All the second order terms (C.86)-(C.88) can be shown
                                             k
to be op (1). The first-order term (C.85) can be analyzed in the same way as the repeat outcomes
case. Moreover, since λ̂k = En,k Ti converges at the parametric rate while the kernel estimator fˆk (d)
converges at a slower rate, the influence of (C.84) on the asymptotic variance is negligible. The
main term (C.83) can be analyzed in the same way as in the panel case.

Step 1: Second Order Terms

   First, we consider (C.86). By triangle inequality, we have

         |En,k [∂λ2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))] − E[∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                                                                        0
                                                                                          (d), η0 (d))]|
       ≤ |En,k [∂λ2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))] − En,k [∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                                                                            0
                                                                                              (d), η0 (d))]|      (C.89)
         |                                                 {z                                             }
                                                               J1k

                                                                     47

      + |En,k [∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                         0
                                           (d), η0 (d))] − E[∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                                                                       0
                                                                                         (d), η0 (d))]|                  (C.90)
        |                                              {z                                            }
                                                          J2k

For J2k , since 0 < c < λ0 < 1 − c, by the boundedness assumption, the score ψh satisfies

                                    ∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                                              0
                                                                (d), η0 (d)) ≲ Kh (D − d).                               (C.91)

Therefore, by the assumption of the kernel function, we have

                 1
        2
     E[J2k ]≤      E[(∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                                0
                                                  (d), η0 (d)))2 ] ≲ E[Kh2 (D − d)]/N ≲ (hN )−1 .                        (C.92)
                 N

Then by Markov’s inequality, we have J2k ≤ Op ((hN )−1/2 ).

   For J1k , note that

         2
      E[J1k |Ikc ]
       = E[|En,k [∂λ2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))] − En,k [∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                                                                              0
                                                                                                (d), η0 (d))]|2 |Ikc ]
                                                                                             0
       ≤        sup           E[|∂λ2 ψh (Z, θ0h (d), λ, f, η(d)) − ∂λ2 ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 |Ikc ]
           λ∈PN , f ∈FN (d)
             η(d)∈TN (d)
                                                                                             0
       ≤        sup           E[|∂λ2 ψh (Z, θ0h (d), λ, f, η(d)) − ∂λ2 ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
           λ∈PN , f ∈FN (d)
             η(d)∈TN (d)

       ≲ h−1 ε2N              (a)

where the first equation holds by definition, the second line holds by Cauchy-Schwarz, and the
third line holds by the construction that all the parameters are estimated using auxiliary sample
Ikc and hence can be treated as fixed in the conditional expectation. Then by conditional Markov’s
inequality, the assumption that (λ̂k − λ)2 ≤ Op (N −1 ), and assumption 7.11, we conclude that
(C.86) = op (1). We will show (a) at the end of this section.

   Term (C.87) is bounded in the same way as the panel case. By triangle inequality, we have

        |En,k [∂f2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))] − E[∂f2 ψh (Z, θ0h (d), λ0 , fD
                                                                                       0
                                                                                         (d), η0 (d))]|
      ≤ |En,k [∂f2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))] − En,k [∂f2 ψh (Z, θ0h (d), λ0 , fD
                                                                                           0
                                                                                             (d), η0 (d))]|              (C.93)
        |                                                 {z                                             }
                                                          J3k

      + |En,k [∂f2 ψh (Z, θ0h (d), λ0 , fD
                                         0
                                           (d), η0 (d))] − E[∂f2 ψh (Z, θ0h (d), λ0 fD
                                                                                     0
                                                                                       (d), η0 (d))]| .                  (C.94)
        |                                             {z                                           }
                                                          J4k

                                                                48

                                0 (d) is bounded away from zero,
To bound J4k , note that since fD

                                                          2
       ∂f2 ψh (Z, θ0h (d), λ0 , fD
                                 0
                                   (d), η0 (d)) =       0
                                                                                       0
                                                                (ψh (Z, θ0h (d), λ0 , fD (d), η0 (d)) + θ0h (d))
                                                      (fD (d))2
                                                   ≲ Kh (D − d)                                                          (C.95)

which implies that

                  1
        2
     E[J4k ]≤       E[(∂f2 ψh (Z, θ0h (d), λ0 , fD
                                                 0
                                                   (d), η0 (d)))2 ] ≲ E[Kh2 (D − d)]/N ≲ (hN )−1 .                       (C.96)
                  N

and by Markov’s inequality, we have J4k ≤ Op ((hN )−1/2 ). For J3k , we have

         2
      E[J3k |Ikc ]
       = E[|En,k [∂f2 ψh (Z, θ0h (d), λ̄k , f¯k , η̂k (d))] − En,k [∂f2 ψh (Z, θ0h (d), λ0 , fD
                                                                                              0
                                                                                                (d), η0 (d))]|2 |Ikc ]
                                                                                             0
       ≤          sup         E[|∂f2 ψh (Z, θ0h (d), λ, f, η(d)) − ∂f2 ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 |Ikc ]
           λ∈PN , f ∈FN (d)
             η(d)∈TN (d)
                                                                                             0
       ≤          sup         E[|∂f2 ψh (Z, θ0h (d), λ, f, η(d)) − ∂f2 ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
           λ∈PN , f ∈FN (d)
             η(d)∈TN (d)

       ≲ h−1 ε2N              (b)

Then by conditional Markov’s inequality, (fˆk (d) − fD
                                                     0 (d))2 ≤ O ((N h)−1 ), and assumption 7.11, we
                                                                p
conclude that (C.87) = op (1). We verify (b) at the end of this section.

   Finally, we can bound (C.88) using similar arguments as those for (C.86) and (C.87). To avoid
repetitiveness, we only highlight the difference. In particular, we need

            sup          E[|∂λ ∂f ψJ (Z, θ0h (d), λ̄k , f¯k , η̂k (d)) − ∂λ ∂f ψJ (Z, θ0h (d), λ0 , fD
                                                                                                     0
                                                                                                       (d), η0 (d))|2 ]
      λ∈PN , f ∈FN (d)
        η(d)∈TN (d)

                                                                                                  ≲ h−1 ε2N        (c)

and using conditional Markov’s inequality, (fˆk (d) − fd )(λ̂k − λ0 ) ≤ Op (N −1 h−1/2 ), and assumption
7.11, we conclude that (C.88) = op (1). Claim (c) will be shown later. Therefore, we have shown
that all the second-order terms are asymptotically negligible.

Step 2: First-Order Terms

   We first consider (C.84). By triangle inequality, we have

                                       0                                             0
       |En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] − E[∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|

                                                             49

                                       0                                                 0
     ≤ |En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] − En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|          (C.97)
       |                                                {z                                             }
                                                          J5k
                                       0                                            0
     + |En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))] − E[∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]| .             (C.98)
       |                                             {z                                           }
                                                         J6k

To bound J6k , since λ0 is bounded away from zero, the score ψ satisfies,

                                                            0
                                   ∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d)) ≲ Kh (D − d).                           (C.99)

This implies that

                       1
              2
           E[J6k ]≤                                  0
                         E[(∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d)))2 ] ≲ E[Kh2 (D − d)]/N ≲ (N h)−1 .
                       N

and by Markov’s inequality, we have J6k ≤ Op ((N h)−1/2 ). With the assumption that N h → ∞,
we have J6k = op (1).

   On the other hand, for J5k , note that

         2
      E[J5k |Ikc ]
                                           0
       = E[|En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))]
                                            0
           − En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|2 |Ikc ]
                                                     0                                       0
       ≤       sup       E[|∂λ ψh (Z, θ0h (d), λ0 , fD (d), η(d)) − ∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 |Ikc ]
           η(d)∈TN (d)
                                                     0                                       0
       ≤       sup       E[|∂λ ψh (Z, θ0h (d), λ0 , fD (d), η(d)) − ∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
           η(d)∈TN (d)

       ≲ h−1 ε2N             (d)

where the first equation holds by definition, the second line holds by the Cauchy-Schwarz inequality,
and the third line holds by the construction that all the parameters are estimated using auxiliary
sample Ikc and hence can be treated as fixed. Then we conclude with conditional Markov’s inequality
that J5k = op (1). As before, we will show (d) at the end of this section.

   Therefore,

                                   0
    En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] →p E[∂λ ψh (Z, θ0h (d), λ0 , fD
                                                                                  0
                                                                                    (d), η0 (d))] := Sλ0          (C.100)

Note that (λ̂k − λ0 ) = Op (N −1/2 ), we can rewrite (C.84) as

                                √           K
                                        1 X                                0
                     (C.84) =       N       En,k [∂λ ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))](λ̂k − λ0 )
                                        K
                                           k=1

                                                               50

<!-- pages: 51-72 -->

√           K
                                         1 X 0
                             =       N      Sλ (λ̂k − λ0 ) + op (1)
                                         K
                                             k=1
                                 √            N
                                         1   X
                             =       N             Sλ0 (Ti − λ0 ) + op (1)
                                         N
                                             i=1

where the last equality holds by the definition that λ̂k − λ0 = (N − n)−1 i∈I c Ti − λ0 and the fact
                                                                               P
                                                                                  k
that K −1 K                     1 PN                                         0 = E[∂ ψ 0 ] is bounded by
         P
            k=1 (λ̂ k − λ 0 ) = N  i=1 (T i − λ 0 ). We remark that, since S λ      λ h
a constant and λ̂ converges at parametric rate, (C.84) vanishes when scaled by the square-root of
the asymptotic variance that grows with sample size.

   Term (C.85) will be bounded using the same argument as in the panel setting. First, by the
triangle inequality

                                       0                                             0
       |En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] − E[∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|
                                       0                                                 0
     ≤ |En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] − En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|         (C.101)
       |                                                {z                                             }
                                                              J7k
                                       0                                            0
     + |En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))] − E[∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]| .            (C.102)
       |                                                    {z                                         }
                                                            J8k

                                      0 (d) is bounded away from zero and the score ψ satisfies
We first bound J8k . Note that since fD

                                              0
                     ∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))                                                      (C.103)
                             1                            0
                      =− 0        (ψh (Z, θ0h (d), λ0 , fD  (d), η0 (d)) + θ0h (d)) ≲ Kh (D − d),                 (C.104)
                           fD (d)

which implies that

                  1
        2
     E[J8k ]≤                                   0
                    E[(∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d)))2 ] ≲ E[Kh2 (D − d)]/N ≲ (N h)−1 .                 (C.105)
                  N

Then by Markov’s inequality, we have J8k ≤ Op ((N h)−1/2 ). With the assumption that N h → ∞,
we have J8k = op (1).

   Second, to bound J7k , note that

         2
      E[J7k |Ikc ]
                                           0
       = E[|En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))]
                                            0
           − En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|2 |Ikc ]
                                                     0                                       0
       ≤       sup       E[|∂f ψh (Z, θ0h (d), λ0 , fD (d), η(d)) − ∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 |Ikc ]
           η(d)∈TN (d)

                                                                    51

                                                       0                                       0
         ≤       sup       E[|∂f ψh (Z, θ0h (d), λ0 , fD (d), η(d)) − ∂f ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
             η(d)∈TN (d)

         ≲ h−1 ε2N              (e)

where the first equation holds by definition, the second line holds by Cauchy-Schwarz, and the
third line holds by the construction that all the parameters are estimated using auxiliary sample
Ikc . Then we conclude with the conditional Markov’s inequality that J7k = op (1). Therefore,

                                   0
    En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] →p E[∂f ψh (Z, θ0h (d), λ0 , fD
                                                                                  0
                                                                                    (d), η0 (d))] := Sf0            (C.106)

Since (fˆk (d) − fD
                  0 (d)) = O ((N h)−1/2 ), we can rewrite (C.85) as
                            p

                                √           K
                                        1 X
                  (C.85) =          N                                      0
                                            En,k [∂f ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))](fˆk (d) − fD
                                                                                                       0
                                                                                                         (d))
                                        K
                                            k=1
                                √            K
                                        1   X
                            =       N             Sf0 (fˆk (d) − fD
                                                                  0
                                                                    (d)) + op (h−1/2 )
                                        K
                                            k=1
                                √            N
                                        1   X
                            =       N             Sf0 (Kh (Di − d) − E[Kh (D − d)]) + op (h−1/2 )
                                        N
                                            i=1

where the last equality holds by fˆk (d) − fD
                                            0 (d) = (N − n)−1
                                                              P
                                                               i∈Ikc Kh (Di − d) − E[Kh (D − d)] +
                                                √ 2
O(h ), the under-smoothing assumption that N h ≤ O(1), and the fact that K −1 K
   2                                                                                        ˆ
                                                                                     P
                                                                                       k=1 (fk (d) −
                 1 PN
E[Kh (D − d)]) = N i=1 (Kh (Di − d) − E[Kh (D − d)]). This term will contribute to the asymptotic
variance.

Step 3: “Neyman Term”

   Now we consider (C.83), which can be shown using the same argument as the panel case.

    √            K
          1 X                             0
        N     En,k [ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))]
          K
                 k=1
                   N
       1          X
                                      0
     =√       ψh (Zi , θ0h (d), λ0 , fD (d), η0 (d))
        N i=1
         √           K
                 1 X                              0                                                0
     +       N       (En,k [ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] − En,k [ψh (Zi , θ0h (d), λ0 , fD (d), η0 (d))]) .
                 K   |                                             {z                                             }
                     k=1
                                                                          Rnk
                                                                                                                    (C.107)

Since K is fixed, n = O(N ), it suffices to show that Rnk = op (N −1/2 h−1 ), so it vanishes when

                                                                    52

scaled by the (square root of) asymptotic variance. Note that by triangle inequality, we have the
following decomposition

                                                               R1k + R2k
                                                   |Rn,k | ≤      √                                              (C.108)
                                                                    n

where

                                        0                                             0
     R1k := |Gnk [ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))] − Gnk [ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|           (C.109)

                    √
with Gnk (f ) =         n(Pn − P )(f ) denote the empirical process, and with some abuse of notation,
it will also be used to denote the conditional version of the empirical process conditioning on the
auxiliary sample Ikc . Moreover,
             √                             0                                               0
    R2k :=       n|E[ψh (Z, θ0h (d), λ0 , fD (d), η̂k (d))|Ikc ] − E[ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))]|.     (C.110)

                        i
For simplicity, denote ψη(d)                            0 (d), η(d)).
                             := ψh (Zi , θ0h (d), λ0 , fD

    First, we consider R1k , in which
                                              n
                                         √ 1X
        Gnk ψη̂k (d) − Gnk ψη0 (d) =      n     ψη̂i k (d) − ψηi 0 (d) − E[ψη̂i k (d) |Ikc ] − E[ψηi 0 (d) ] .   (C.111)
                                            n
                                                i=1
                                                |                         {z                              }
                                                                               :=∆ik

In particular, it can be shown that E[∆ik ∆jk ] = 0 for all i ̸= j using the i.i.d. assumption of the
data and that the nuisance parameter η̂k (d) is estimated using the auxiliary sample. Then, we have

                                     2
                                  E[R1k |Ikc ] ≤ E[∆2ik |Ikc ]
                                               ≤ E[(ψη̂i k (d) − ψηi 0 (d) )2 |Ikc ]
                                                                     i
                                               ≤      sup        E[(ψη(d) − ψηi 0 (d) )2 |Ikc ]
                                                   η(d)∈TN (d)
                                                                     i
                                               ≤      sup        E[(ψη(d) − ψηi 0 (d) )2 ]
                                                   η(d)∈TN (d)

                                               ≲ h−1 ε2N            (f )

and using the conditional Markov’s inequality, we conclude that R1k = Op (h−1/2 εN ).
                                                                        0 (d), η (d))] = 0, so it suffices
    Now we bound R2k . Note that by definition, E[ψh (Z, θ0h (d), λ0 , fD       0
                                 0 (d), η̂ (d))|I c ]. Suppressing other arguments in the score, define
to bound E[ψh (Z, θ0h (d), λ0 , fD        k      k

                                  hk (r) := E[ψh (η0 (d) + r(η̂k (d) − η0 (d)))|Ikc ]                            (C.112)

                                                               53

where by definition hk (0) = E[ψh (η0 (d))|Ikc ] = 0 and hk (1) = E[ψh (η̂k (d))|Ikc ]. Use Taylor’s theo-
rem, expand hk (1) around 0, we have

                                                              1 ′′
                                   hk (1) = hk (0) + h′k (0) + hk (r̄),           r̄ ∈ (0, 1).                    (C.113)
                                                              2

Note that, by Neyman orthogonality,

                                   h′k (0) = ∂η (d)E[ψh (η0 (d))][η̂k (d) − η0 (d)] = 0                           (C.114)

and use that fact that hk (0) = 0, we have
                                  √                 √      ′′
                      R2k =           n|hk (1)| =       n|hk (r̄)|
                                                         √ 2
                            ≤             sup              n|∂r E[ψh (η0 (d) + r(η̂k (d) − η0 (d)))]|
                                  r∈(0,1),η(d)∈TN (d)
                                  √
                            ≲         nh−1 ε2N      (g)

   Combining the above results, we conclude that
                                          √                            √
                                              N Rn,k ≲ h−1/2 εN +          N h−1/2 ε2N ,                          (C.115)
                                           √
and for εN = o(N −1/4 ), we have               N Rn,k = op (h−1/2 ).

Step 4: Auxiliary Results

   In this section, we show the auxiliary results (a)-(g) used in the previous steps. Note that replac-
             T −λ
ing ∆Y with λ(1−λ) Y , we can show claims (b),(e),(f),(g) using the same arguments as (a),(b),(c),(d)
respectively in the panel case. Therefore, we focus on (a), (c), and (d) in the repeated cross-sectional
setting.

   First, recall that

       (a) :        sup           E[|∂λ2 ψh (Z, θ0h (d), λ, f, η(d)) − ∂λ2 ψh (Z, θ0h (d), λ0 , fD
                                                                                                 0
                                                                                                   (d), η0 (d))|2 ]
               λ∈PN , f ∈FN (d)
                 η(d)∈TN (d)

                                                                                                        ≲ h−1 ε2N .

In particular,

                                        ∂ 2 Kh (D − d)g(X) − 1{D = 0}fh (d|X) T − λ
           ∂λ2 ψh (λ, fd , η(d)) =                                                    Y                           (C.116)
                                        ∂λ2             fd · g(X)            λ(1 − λ)

                                                                54

where we suppressed the common terms (Z, θ0h (d)) in ψh for simplicity. Then by Taylor’s theorem,

                             ∂λ2 ψh (λ, fd , η(d)) − ∂λ2 ψh (λ0 , fD
                                                                   0
                                                                     (d), η0 (d))
                              = ∂λ2 ψh (λ0 , fD
                                              0
                                                (d), η(d)) − ∂λ2 ψh (λ0 , fD
                                                                           0
                                                                             (d), η0 (d))    (i)
                              + ∂λ2 ∂f ψh (λ̄, f¯d , η(d))(fd − fD
                                                                 0
                                                                   (d))    (ii)
                              + ∂λ3 ψh (λ̄, f¯d , η(d))(λ − λ0 )   (iii)

where λ̄ ∈ (λ, λ0 ) and f¯ ∈ (fd , fD
                                    0 (d)).

    For the first term (i),

     ∂λ2 ψh (λ0 , fD
                   0
                     (d), η(d)) − ∂λ2 ψh (λ0 , fD
                                                0
                                                  (d), η0 (d))
          ∂2                    Y 1{D = 0} fh (d|X) fh0 (d|X)
                                                                   
                   T − λ0
      =                              0 (d)                  −
          ∂λ2 λ0 (1 − λ0 )         fD              g(X)        g0 (X)
            2                   Y 1{D = 0} fh (d|X)(g0 (X) − g(X)) − (fh0 (d|X) − fh (d|X))g(X)
                                                                                             
          ∂        T − λ0
      =                              0 (d)
          ∂λ2 λ0 (1 − λ0 )         fD                                  g(X)g0 (X)
                                                                                              (C.117)

Moreover, by assumption 7.11, for ϵN = o(N −1/4 ), (ii) and (iii) are of smaller order. Therefore,
by the definition of (PN , FN (d), TN (d)), boundedness of the nuisance parameters, and triangle
inequality, we have

                                                                                           0
               sup          E[|∂λ2 ψh (Z, θ0h (d), λ, f, η(d)) − ∂λ2 ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
        λ∈PN , f ∈FN (d)
          η(d)∈TN (d)
                                                                                                 0
         ≲      sup                                     0
                           E[|∂λ2 ψh (Z, θ0h (d), λ0 , fD (d), η(d)) − ∂λ2 ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
             η(d)∈TN (d)

         ≲      sup        ∥fh (d|X) − fh0 (d|X)∥2P,2 + ∥g(X) − g0 (X)∥2P,2
             η(d)∈TN (d)

         ≲ h−1 ϵ2N                                                                                                    (C.118)

which shows (a). Similarly, by Taylor’s theorem,

                                                                       0
                             ∂λ ∂f ψh (λ, fd , η(d)) − ∂λ ∂f ψh (λ0 , fD (d), η0 (d))                                 (C.119)
                                                0                              0
                              = ∂λ ∂f ψh (λ0 , fD (d), η(d)) − ∂λ ∂f ψh (λ0 , fD (d), η0 (d))
                              + ∂λ ∂f2 ψh (λ̄, f¯d , η(d))(fd − fD
                                                                 0
                                                                   (d))
                              + ∂λ2 ∂f ψh (λ̄, f¯d , η(d))(λ − λ0 )                                                   (C.120)

and (c) holds by similar arguments as (a).

                                                             55

   Finally, we show (d):

                                                    0                                       0
             sup        E[|∂λ ψh (Z, θ0h (d), λ0 , fD (d), η(d)) − ∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
          η(d)∈TN (d)

                                                                                                  ≲ h−1 ε2N .

By the same argument as (a),

                                           Kh (D − d)g(X) − 1{D = 0}fh (d|X) T − λ
                ∂λ ψh (λ, fd , η(d)) =                                               Y,                          (C.121)
                                                       fd · g(X)            λ(1 − λ)

which implies

                 0                           0
    ∂λ ψh (λ0 , fD (d), η(d)) − ∂λ ψh (λ0 , fD (d), η0 (d))
                             Y 1{D = 0} fh (d|X) fh0 (d|X)
                                                                
        ∂        T − λ0
     =                           0 (d)                  −
        ∂λ λ0 (1 − λ0 )         fD             g(X)         g0 (X)
                             Y 1{D = 0} fh (d|X)(g0 (X) − g(X)) − (fh0 (d|X) − fh (d|X))g(X)
                                                                                          
        ∂        T − λ0
     =                           0 (d)                                                        .
        ∂λ λ0 (1 − λ0 )         fD                                  g(X)g0 (X)
                                                                                            (C.122)

Therefore, by the definition of TN (d), boundedness of the nuisance parameters, and triangle in-
equality, we have

                                                                                       0
                sup       E[|∂λ ψh (Z, θ0h (d), λ, f, η(d)) − ∂λ ψh (Z, θ0h (d), λ0 , fD (d), η0 (d))|2 ]
            η(d)∈TN (d)

            ≲       sup       ∥fh (d|X) − fh0 (d|X)∥2P,2 + ∥g(X) − g0 (X)∥2P,2
                η(d)∈TN (d)

            + ∥fh (d|X) − fh0 (d|X)∥P,2 ∥g(X) − g0 (X)∥P,2
            ≲ h−1 ϵ2N .                                                                                          (C.123)

This completes the proofs for the auxiliary results.

   Combining previous results, we have

                                 AT
                                 [  T (d) − AT T (d)
                                          N
                                      1 X                         0
                                  =       ψh (Zi , θ0h (d), λ0 , fD (d), η0 (d))                                 (C.124)
                                      N
                                          i=1
                                           N
                                      1   X
                                  +             Sf0 (Kh (Di − d) − E[Kh (Di − d)])                               (C.125)
                                      N
                                          i=1

                                  + op ((N h)−1/2 )                                                              (C.126)

                                                          56

                                  + θ0 (d) − θ0h (d)                                                   (C.127)

where (C.124) and (C.125) are averages of i.i.d. zero-mean terms with the variance growing with
kernel bandwidth h, and recall that Sf0 = E[∂f ψh (Z, θ0h (d), λ0 , fD
                                                                     0 (d), η (d))]; (C.126) are the terms
                                                                             0
that vanish when scaled by the (square root of) asymptotic variance; (C.127) is the bias term which
is shown to be of order O(h2 ) in Lemma 4.2.

   Note that we have arrived at the identical decomposition as in the panel case, and by the same
argument, we have
                                      AT
                                      [  T (d) − AT T (d)
                                                √               →d     N (0, 1)
                                           σN / N
with σN defined by

                                                                             2
                                                                            
                           2               θ0h
                          σN := E     ψh − 0     Kh (D − d) − E[Kh (D − d)]                            (C.128)
                                          fD (d)

                                              0 (d).
where we have used the fact that Sf0 = −θ0h /fD                                                             □

Proof of Theorem 4.3 (Panel) The proof uses the same idea as in Chernozhukov et al. (2018)
and Chang (2020). However, we need to adapt the proof to accommodate the presence of the kernel
term. First, recall that the variance estimator is defined as

                        K
                                                                                               2
                                                                                              
             2    1     X
                                                ˆ                  θ̂h (d)              ˆ
           σ̂N :=          En,k ψh (Z, θ̂h (d), fk (d), η̂k (d)) −         Kh (D − d) − fk (d)
                     K
                       k=1
                                                                   fˆk (d)
                       K
                     1 X                                        2 
                         En,k ψ̃h (Z, θ̂h (d), fˆk (d), η̂k (d))
                             
                :=
                     K
                        k=1

where we define

         ψ̃h (Z, θ, fd , η(d))
              Kh (D − d)g(X) − 1{D = 0}fh (d|X)              θh
         :=                                     ∆Y − E∆Y (X) − Kh (D − d).                             (C.129)
                           fd g(X)                            fd

                          2 = E ψ̃ 2 (Z, θ (d), f 0 (d), η (d)) . Therefore, we need to show that
                                                              
In particular, note that σN       h       0h     D        0

      Jk := En,k ψ̃h2 (Z, θ̂h (d), fˆk (d), η̂k (d)) − E ψ̃h2 (Z, θ0h (d), fD
                                                                            0
                                                                                       
                                                                              (d), η0 (d)) = op (1).   (C.130)

                                                         57

By the triangle inequality, we have

         Jk ≤ En,k ψ̃h2 (Z, θ̂h (d), fˆk (d), η̂k (d)) − En,k ψ̃h2 (Z, θ0h (d), fD
                                                                                 0
                                                                                            
                                                                                   (d), η0 (d))        (C.131)
             |                                          {z                                      }
                                                      :=J1k

            + En,k ψ̃h2 (Z, θ0h (d), fD
                                      0
                                        (d), η0 (d)) − E ψ̃h2 (Z, θ0h (d), fD
                                                                            0
                                                                                       
                                                                              (d), η0 (d)) .           (C.132)
             |                                      {z                                     }
                                                     :=J2k

We bound each term separately.

   First, we consider J2k .

           2
                                                                                                   2 
              ] = E En,k ψ̃h2 (Z, θ0h (d), fD  0
                                                 (d), η0 (d)) − E ψ̃h2 (Z, θ0h (d), fD
                                                                                     0
                                                                  
        E[J2k                                                                          (d), η0 (d))
                     X   n                                     2 
                       1        2                 0
                ≤E            ψ̃h (Zi , θ0h (d), fD (d), η0 (d))
                       n
                          i=1
                  1                      0
                ≤ E ψ̃h4 (Z, θ0h (d), fD
                                                        
                                            (d), η0 (d))
                  n
                  1 
                ≲ E Kh4 (D − d)
                                    
                  n
                ≲ (N h3 )−1 ,                                                                            (C.133)

where the third line holds by Cauchy-Schwarz inequality, the fourth line holds by boundedness
assumption, and the last line holds by change of variables using the assumptions on the kernel.
Therefore, by Chebyshev’s inequality, we have J2k = op (1) if N h3 → ∞.

   Next, we consider J1k . We first state the following convenient fact that will be used in the
proof, see Chernozhukov et al. (2018) and Chang (2020) for example. For any constants a and δ,

                                    |(a + δa)2 − a2 | ≤ 2|δa|(|a| + |δa|).                             (C.134)

In our context, we define (for notation simplicity)

                                                             0
                                     a = ψ̃h (Zi , θ0h (d), fD (d), η0 (d)) :=ψi                       (C.135)
                                a + δa = ψ̃h (Zi , θ̂h (d), fˆk (d), η̂k (d)) :=ψ̂i .                  (C.136)

Then, we have

                1X 2                   1X 2
         J1k =         ψ̂i − ψi2 ≤             ψ̂i − ψi2
                n                      n
                  i∈Ik                   i∈Ik
               2X
             ≤        |ψ̂i − ψi |(|ψi | + |ψ̂i − ψi |)
               n
                  i∈Ik

                                                         58

                 X                  1/2  X                               1/2
                 1                 2        1                             2
              ≤2        |ψ̂i − ψi |                 (|ψi | + |ψ̂i − ψi |)
                 n                         n
                   i∈Ik                        i∈Ik
                 X                  1/2  X               1/2  X                       1/2 
                 1                 2          1            2            1                 2
              ≤2        |ψ̂i − ψi |                   |ψi |       +            |ψ̂i − ψi |                   (C.137)
                 n                           n                         n
                        i∈Ik                            i∈Ik                    i∈Ik

where the third line holds by Cauchy-Schwarz inequality, and the last line holds by the triangle
inequality. Then, we have
                                                                     
                                                 2               1X 2
                                                J1k ≲ SN    SN +   ψi                                        (C.138)
                                                                 n
                                                                     i∈Ik

where SN := n1                       2
                 P
                     i∈Ik |ψ̂i − ψi | .

   We now bound SN . By the definition of ψ̃h , we have

                 1X                                                                                 2
          SN =            ψ̃h (Zi , θ̂h (d), fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), fD
                                                                                     0
                                                                                       (d), η0 (d))
                 n
                   i∈Ik
                         
                 1X
               =           ψ̃h (Zi , θ0h (d), fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), fD0
                                                                                         (d), η0 (d))
                 n
                   i∈Ik
                                                       2
                     ∂                ˆ
                 +      ψ̃h (Zi , θ̄, fk (d), η̂k (d))
                     ∂θ
                 1X                                                                                  2
               ≲           ψ̃h (Zi , θ0h (d), fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), fD0
                                                                                         (d), η0 (d))        (C.139)
                 n
                    i∈Ik
                 |                                           {z                                        }
                                                           :=S1N
                                                                     2
                      1 X Kh (Di − d)
                  +                                (θ̂h (d) − θ0h (d))                                       (C.140)
                      n
                        i∈Ik
                                      fˆk (d)
                      |                          {z                   }
                                                :=S2N

where the second line holds by Taylor’s theorem with θ̄ between θ0h (d) and θ̂h (d), and the last
line holds by the fact that ∂ ψ̃h (Zi , θ̄, fˆk (d), η̂k (d)) = Kh (Di − d)/fˆk (d). We bound S1N and S2N
                                 ∂θ
separately.

   To bound S2N , note that

                                                (θ̂h (d) − θ0h (d))2 1 X 2
                                 S2N =                                  Kh (Di − d).                         (C.141)
                                                       fˆk (d)2      n
                                                                     i∈Ik

Since E[Kh2 (D − d)] = O(h−1 ), by Markov’s inequality, we have n1                              2                 −1
                                                                                       P
                                                                                          i∈Ik Kh (Di − d) = Op (h ).

                                                               59

Moreover, by Theorem 4.2, we have (θ̂h (d) − θ0h (d))2 = Op ((N h)−1 ). Therefore, we conclude

                                                S2N ≤ Op ((N h2 )−1 ).

   Next, we bound S1N . By Taylor’s theorem, for f¯ between fD
                                                             0 (d) and fˆ (d), we have
                                                                         k

                       ψ̃h (Zi , θ0h (d), fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), fD
                                                                                  0
                                                                                    (d), η0 (d))
                                              0                                     0
                        = ψ̃h (Zi , θ0h (d), fD (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), fD (d), η0 (d))       (C.142)
                             ∂
                        +      ψ̃h (Zi , θ0h (d), f¯, η̂k (d))(fˆk (d) − fD
                                                                          0
                                                                            (d)).                        (C.143)
                            ∂f

Note that

         ∂
            ψ̃h (Z, θ, f, η(d))
         ∂f
                                                                           
              ∂ Kh (D − d)g(X) − 1{D = 0}fh (d|X)              θh
         =                                        ∆Y − E∆Y (X) − Kh (D − d)
             ∂f                 f g(X)                          f
            ≲ Kh (D − d)                                                                                 (C.144)

where the last line holds by the boundedness assumption. Therefore, by the assumption on the
kernel, we have

                      ∥∂f ψ̃h (Z, θ0h (d), f¯, η̂k (d))∥P,2 ≲ ∥Kh (D − d)∥P,2 = O(h−1/2 ).

Moreover, by definition

                                           0                                     0
                       ψ̃h (Zi , θ0h (d), fD (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), fD (d), η0 (d))
                                            0                                    0
                       = ψh (Zi , θ0h (d), fD (d), η̂k (d)) − ψh (Zi , θ0h (d), fD (d), η0 (d)).         (C.145)

Then, by triangle inequality and Cauchy-Schwarz inequality, we have

                       ∥ψ̃h (Zi , θ0h (d), fˆk (d), η̂k (d)) − ψ̃h (Zi , θ0h (d), fD
                                                                                   0
                                                                                     (d), η0 (d))∥2P,2
                                          0                                    0
                     ≲∥ψh (Zi , θ0h (d), fD (d), η̂k (d)) − ψh (Zi , θ0h (d), fD (d), η0 (d))∥2P,2
                        + ∥∂f ψ̃h (Zi , θ0h (d), f¯, η̂k (d))∥2P,2 ∥fˆk (d) − fD
                                                                               0
                                                                                 (d)∥2P,2
                     ≲h−1 ε2N + h−2 N −1 .                                                               (C.146)

where the last line holds by the assumptions on the rate of convergence of the fˆk (d) and

                                0
            ∥ψh (Zi , θ0h (d), fD                                    0
                                  (d), η̂k (d)) − ψh (Zi , θ0h (d), fD (d), η0 (d))∥2P,2 ≲ h−1 ε2N       (C.147)

                                                            60

by the same arguments as in the proof of Theorem 4.2. Then by Markov’s inequality, we have

                                          S1N = Op h−1 ε2N + h−2 N −1 .
                                                                     
                                                                                                          (C.148)

   Combining the results, we have

                                          SN = Op h−1 ε2N + h−2 N −1 .
                                                                    

   Note that since ψi ≲ Kh (D − d), n1                   2        −1
                                                P
                                                   i∈Ik ψi = Op (h ) by Markov’s inequality.        This implies
that
                                                
                                            1X 2
                            2
                                              ψi = Op h−2 ε2N + h−3 N −1 .
                                                                        
                           J1k ≲ SN    SN +                                                               (C.149)
                                            n
                                                  i∈Ik

Then J1k = op (1) if h−2 ε2N + h−3 N −1 → 0.
                                 2 = σ 2 + o (1).
   Therefore, we conclude that σ̂N                                                                              □
                                      N     p

Proof of Theorem 4.3 (Repeated Cross-Sections) The proof is nearly identical to the panel
case, and we only highlight the key differences. Again, the main idea follows Chernozhukov et al.
(2018) and Chang (2020), and our proof requires modifications to account for the kernel function
present in the score function.

   First, recall that the variance estimator is defined as

                     K
                                                                                                  2
                                                                                                 
           2    1    X
                                                   ˆ                  θ̂h (d)              ˆ
         σ̂N :=         En,k ψh (Z, θ̂h (d), λ̂k , fk (d), η̂k (d)) −         Kh (D − d) − fk (d)
                  K
                    k=1
                                                                      fˆk (d)
                    K
                  1 X                                              2 
                      En,k ψ̃h (Z, θ̂h (d), λ̂k , fˆk (d), η̂k (d))
                          
             :=                                                                                           (C.150)
                  K
                     k=1

where we define
                                                                                       
                               Kh (D − d)g(X) − 1{D = 0}fh (d|X)    T −λ
   ψ̃h (Z, θ, λ, fd , η(d)) :=                                              Y − EλY (X)                   (C.151)
                                            fd g(X)                λ(1 − λ)
                                 θh
                               − Kh (D − d).                                                              (C.152)
                                 fd
                                                                          
                          2 = E ψ̃ 2 (Z, θ (d), λ , f 0 (d), η (d))
                                      
In particular, note that σN       h       0h     0 D          0            . Therefore, we need to show that

  Jk := En,k ψ̃h2 (Z, θ̂h (d), λ̂k , fˆk (d), η̂k (d)) − E ψ̃h2 (Z, θ0h (d), λ0 , fD
                                                                                   0
                                                                                              
                                                                                     (d), η0 (d)) = op (1). (C.153)

                                                          61

By the triangle inequality, we have

     Jk ≤ En,k ψ̃h2 (Z, θ̂h (d), λ̂k , fˆk (d), η̂k (d)) − En,k ψ̃h2 (Z, θ0h (d), λ0 , fD
                                                                                        0
                                                                                                   
                                                                                          (d), η0 (d))           (C.154)
         |                                                {z                                           }
                                                          :=J1k

        + En,k ψ̃h2 (Z, θ0h (d), λ0 , fD
                                       0
                                         (d), η0 (d)) − E ψ̃h2 (Z, θ0h (d), λ0 , fD
                                                                                  0
                                                                                             
                                                                                    (d), η0 (d)) .               (C.155)
         |                                           {z                                          }
                                                        :=J2k

We bound each term separately.
