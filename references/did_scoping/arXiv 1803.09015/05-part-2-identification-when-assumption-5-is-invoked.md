<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1803.09015.pdf
backend: pdftotext
part: 5/7
-->

# Part 2: Identification when Assumption 5 is invoked.

Part 2: Identification when Assumption 5 is invoked.
    In this case, given the result in Lemma A.2,

                AT T (g, t) = E[AT TX (g, t)|Gg = 1]
                           = E [ (E[Yt − Yg−δ−1 |X, Gg = 1] − E[Yt − Yg−δ−1 |X, Dt+δ = 0, Gg = 0])| Gg = 1]
                                                                                                     
                                                                                                     
                                                        |E[Yt − Yg−δ−1 |X,{zDt+δ = 0, Gg = 0]}|Gg = 1
                           = E[Yt − Yg−δ−1 |Gg = 1] − E                                              
                                                                                      =mny
                                                                                        g,t,δ (X)
                                                                         
                                     Gg       
                           =E                     Yt − Yg−1−a − mny
                                                                 g,t,δ (X)   .
                                    E [Gg ]
                                      ny
Hence, we have that AT T (g, t) = AT Tor (g, t; δ) .
                                         ny
   Next, to show that AT T (g, t) = AT Tipw (g, t; δ), it suffices to show that
                                                            
              pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )
      E                                        (Yt − Yg−δ−1 )
                       1 − pg,t+δ (X)                            E [Gg · E[Yt − Yg−δ−1 |X, Dt+δ = 0, Gg = 0]]
                                                             =                                              .   (A.4)
                      pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )                              E [Gg ]
                  E
                               1 − pg,t+δ (X)

Towards this end, recall that pg,t+δ (X) = P (Gg = 1|X, Gg + (1 − Dt+δ ) (1 − Gg ) = 1) and also notice that

                                    E [Gg |X]                                E [(1 − Dt+δ ) (1 − Gg ) |X]
        pg,t+δ (X) =                                       , 1 − pg (X) =                                   ,     (A.5)
                         E [Gg + (1 − Dt+δ ) (1 − Gg ) |X]                E [Gg + (1 − Dt+δ ) (1 − Gg ) |X]

                                                                      35

it follows that by the law of iterated expectations,
                                                                                                
                           pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )        E [Gg |X] (1 − Dt+δ ) (1 − Gg )
                      E                                       =E
                                    1 − pg,t+δ (X)                  E [(1 − Dt+δ ) (1 − Gg ) |X]
                                                              = E [Gg ] .                                               (A.6)

Next, by exploiting (A.5) and applying the law of iterated expectations, we have that
                                                                        
                       pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )
                   E                                     (Yt − Yg−δ−1 )
                                 1 − pg,t+δ (X)
                                                                          
                           E [Gg |X] (1 − Dt+δ ) (1 − Gg )
                    =E                                      (Yt − Yg−δ−1 )
                            E [(1 − Dt+δ ) (1 − Gg ) |X]
                                                                                                      
                                    E [Gg |X]
                    =E                                  E [ (1 − Dt+δ ) (1 − Gg ) · (Yt − Yg−δ−1 )| X]
                           E [(1 − Dt+δ ) (1 − Gg ) |X]
                     = E [E [Gg |X] · E [ (Yt − Yg−δ−1 )| X, Dt+δ = 0, Gg = 0]]
                     = E [Gg · E [ (Yt − Yg−δ−1 )| X, Dt+δ = 0, Gg = 0]] .

                                                                                               ny
Thus, combined this with (A.6), we establish (A.4), implying that AT T (g, t) = AT Tipw           (g, t; δ).
   Finally, notice that

                                               pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )
                                                                                                                 

            ny
                              Gg                       1 − pg,t+δ (X)                                    ny
                                                                                                                   
       AT Tdr                 E [Gg ] −  pg,t+δ (X) (1 − Dt+δ ) (1 − Gg )   Yt − Yg−δ−1 − mg,t,δ (X) 
               (g, t; δ) = E                                                                                     
                                            E
                                                          1 − pg,t+δ (X)
                                                                                                              
                               ny                  1               E [Gg |X] (1 − Dt+δ ) (1 − Gg )       ny
                         = AT Tipw (g, t; δ) −          E Gg −                                        mg,t,δ (X)
                                                E [Gg ]              E [(1 − Dt+δ ) (1 − Gg ) |X]
                                             1
                                                  E (E [Gg |X] − E [Gg |X]) · mnev
                                                                                          
                         = AT T (g, t) −                                         g,t,δ (X)
                                          E [Gg ]
                        = AT T (g, t) .

                                                                                        nev
where the second equality follows from (A.5) and (A.6), and the third equality from AT Tipw (g, t; δ) = AT T (g, t)
and the law of iterated expectations.

    Proof of Theorem 2: From Theorem 1 it follows that AT T (g, t)’s are point-identified for all groups g and
time periods t such that g ∈ Gδ , t ∈ {2, . . . T − δ} and t ≥ g − δ. For each (g, t) pair, the asymptotic linear
                 √ [ nev
representation of n(AT  T dr (g, t; δ) − AT T (g, t)) follows from Theorem A.1(a) of Sant’Anna and Zhao (2020),
whereas
                     √        dr,nev                    d
                       n(AT
                         [  T t≥(g−δ) − AT Tt≥(g−δ) ) − → N (0, E[Ψdr,nev       dr,nev      0
                                                                   t≥(g−δ) (W )Ψt≥(g−δ) (W ) ])

follows from the Lindeberg–Lévy central limit theorem.

Proof of Theorem 3: Note that, by the conditional multiplier central limit theorem, see Lemma 2.9.5 in van der
Vaart and Wellner (1996), as n → ∞,
                                                n
                                            1 X                        d
                                           √       Vi · Ψdr,nev
                                                         t≥(g−δ) (Wi ) → N (0, Σ),                                      (A.7)
                                             n i=1                     ∗


where Σ = E[Ψdr,nev       dr,nev      0
             t≥(g−δ) (W )Ψt≥(g−δ) (W ) ]. Thus, to conclude the proof that

                                     √
                                                                     
                                                ∗,dr,nev       dr,nev   d
                                         n AT
                                            [ T t≥(g−δ) − AT
                                                          [  T t≥(g−δ) → N (0, Σ),
                                                                            ∗


                                                             36

it suffices to show that, for all g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ} and t ≥ g − δ,
                                      n
                                  1 X        h
                                                dr,nev                   dr,nev         ∗,nev
                                                                                               i
                                 √       Vi · ψbg,t,δ        κnev
                                                       (Wi ; b g,t ) − ψg,t,δ   (W i ; κg,t   )  = op∗ (1) ,                                  (A.8)
                                   n i=1
                                 0
where κ∗,nev
       g,t   =   π ∗0
                   g  , β ∗,nev 0
                         g,t,δ       is the vector of pseudo-true finite-dimensional parameters.
   Towards this end, recall that
               dr,nev                btreat,nev (Wi ; βbnev ) − ψbcomp,nev (Wi ; π                  est,nev
             ψbg,t,δ        bnev
                      (Wi ; κg,t ) = ψg,t,δ             g,t,δ     g,t,δ
                                                                                        nev
                                                                                 bg , βbg,t,δ ) − ψbg,t,δ   (Wi ; π      nev
                                                                                                                  bg , βbg,t,δ )

where
                                                                                                 
                   treat,nev         nev
                ψbg,t,δ      (W ; βbg,t,δ  )=w  bgtreat · Yt − Yg−δ−1 − mnev       g,t,δ
                                                                                           bnev
                                                                                           β g,t,δ
                                                                      h                                                i
                                                    −w  bgtreat · En w   bgtreat · Yt − Yg−δ−1 − mnev               bnev
                                                                                                           g,t,δ βg,t,δ      ,
                                                                                                           
             comp,nev                nev
           ψbg,t,δ      (W ; πbg , βbg,t,δ )=w  bgcomp,nev (b πg ) · Yt − Yg−δ−1 − mnev                bnev
                                                                                               g,t,δ βg,t,δ
                                                                              h                                                   i
                                                    −w  bgcomp (b πg ) · En wgcomp (b    πg ) · Yt − Yg−δ−1 − mnev            bnev
                                                                                                                       g,t,δ βg,t,δ     ,
                                                                0
                est,nev              nev        or,nev bnev            cdr,nev,1 + lgps,nev (b        0 cdr,nev,2
             ψbg,t      (W ; πbg , βbg,t,δ ) = lg,t       βg,t,δ · M      g,t,δ                   πg ) · M  g,t,δ    ,

with
                                                                                      pbg (X; π bg ) C
                                               Gg                                    1  −  p g (X;  π
                                                                                                    bg )
                                   bgtreat =                   bgcomp,nev (b
                                                                                            b
                                   w                  ,        w           πg ) =                         ,
                                             En [Gg ]                                    pbg (X; π
                                                                                                 bg ) C
                                                                                  En
                                                                                        1 − pbg (X; π bg )
and
                                   h                                                  i
             cdr,nev,1 = En        bgtreat − w bgcomp,nev (b  πg ) · ṁnev      bnev
                                                                  
             M g,t,δ               w                                   g,t,δ   β g,t,δ    ,
                                h                                                                    i
              cdr,nev,2 = En α
              M                   bgps,nev (bπg ) · Yt − Yg−δ−1 − mnev             bnev      · ṗg (b
                g,t,δ                                                      g,t,δ βg,t,δ             πg )
                                        h                                                                                           i
                                − En α    bgps,nev (b       bgcomp,nev (b
                                                     πg ) · w           πg ) · Yt − Yg−δ−1 − mnev        g,t,δ
                                                                                                                 bnev
                                                                                                                 β g,t,δ    · ṗg (b
                                                                                                                                   π g )  ,
                                                    ,                           
                                       C                        pg (X; π
                                                                       bg ) C
           bgps,nev (b
           α          πg ) =                      2     E n                        .
                             (1 − pg (X; π  bg ))              1 − pg (X; πbg )


      We first show that
                                  n
                              1 X        
                                            treat,nev         nev        treat,nev        ∗,nev
                                                                                                 
                             √       Vi · ψbg,t,δ     (Wi ; βbg,t,δ ) − ψg,t,δ     (Wi ; βg,t,δ ) = op∗ (1) .                                 (A.9)
                               n i=1

Using the mean-value theorem, we write
                                       n
                                1 X           treat,nev         nev
                               √       Vi · ψbg,t,δ     (Wi ; βbg,t,δ )
                                 n i=1
                                           n
                                  1 X           treat
                                                                               
                                                                                       ∗,nev
                                                                                             
                                =√       Vi · w
                                              bg,i    · Yi,t − Yi,g−δ−1 − mnev
                                                                           g,t,δ Wi ; βg,t,δ
                                   n i=1
                                                             n
                                   √  nev       ∗,nev
                                                       0 1 X
                                                                       treat
                                                                             · ṁnev          nev
                                                                                                    
                               −    n βbg,t,δ − βg,t,δ          Vi · w
                                                                     bg,i        g,t,δ Wi ; β̄g,t,δ
                                                          n i=1


                                                                         37

                                   h                                        i 1 X n
                               − En wbgtreat · Yt − Yg−δ−1 − mnev
                                                              g,t,δ
                                                                      bnev
                                                                      β g,t,δ    √       Vi · w treat
                                                                                              bg,i
                                                                                   n i=1
                                   1,treat     2,treat     3,treat
                               = Ibg,t,δ   − Ibg,t,δ   − Ibg,t,δ   ,

        nev                                             nev      ∗,nev     nev      ∗,nev
where β̄g,t,δ is an intermediate point that satisfies β̄g,t,δ − βg,t,δ ≤ βbg,t,δ − βg,t,δ a.s.. From the strong law of
large numbers and the fact that V is mean zero, independent of W , it follows that
                                       n
                   1,treat   1 X          treat
                                                                     
                                                                             ∗,nev
                                                                                   
                 Ibg,t,δ   =√       Vi · wg,i   · Yt − Yg−δ−1 − mnev
                                                                 g,t,δ Wi ; βg,t,δ    + op∗ (1)
                              n i=1
                                 n
                   3,treat   1 X          treat
                                                   h                             
                                                                                    ∗,nev
                                                                                          i
                 Ibg,t,δ   =√       Vi · wg,i   · E wgtreat · Yt − Yg−δ−1 − mnev
                                                                             g,t,δ βg,t,δ     + op∗ (1)
                              n i=1

                                                                                             2,treat
Similarly, from Assumptions 7 and 8, and the strong law of large numbers, we conclude that Ibg,t,δ   = op∗ (1). Now
(A.9) follows from combining these results.
    Next, we show
                              n
                        1 X        
                                      comp,nev                         comp,nev              ∗,nev
                                                                                                    
                       √       Vi · ψbg,t,δ    (Wi ; π      nev
                                                     bg , βbg,t,δ ) − ψg,t,δ    (Wi ; πg∗ , βg,t,δ ) = op∗ (1) .                                      (A.10)
                         n i=1

Again, by the mean value theorem, we write
             n
        1 X           comp,nev              nev
       √       Vi · ψbg,t,δ    (Wi ; π
                                     bg , βbg,t,δ )
         n i=1
                  n
          1 X                                                        
                                                                             ∗,nev
                                                                                   
        =√            bgcomp,nev (Wi ; π
                 Vi · w                bg ) · Yi,t − Yi,g−δ−1 − mnev
                                                                 g,t,δ Wi ; βg,t,δ
           n i=1
           √  nev             0 1 n
                         ∗,nev
                                    X
                                             bgcomp,nev (Wi ; π
                                                              bg ) · ṁnev          nev
                                                                                          
       −    n βbg,t,δ − βg,t,δ          Vi · w                         g,t,δ Wi ; β̄g,t,δ
                                  n i=1
            h                                                               i 1 X       n
       − En w bgcomp,nev (b πg ) · Yt − Yg−δ−1 − mnev         g,t,δ   βbnev
                                                                        g,t,δ       √          Vi · w bgcomp,nev (Wi ; π    bg )
                                                                                       n i=1
                            pg Xi ; πg∗ Ci
                                           
                n                              
          1 X               1 − pg X; πg∗                                                  
                                                                                                       ∗,nev
                                                                                                              
       =√          Vi ·                            · Yi,t − Yi,g−δ−1 − mnev         g,t,δ   W  i ; β g,t,δ
           n i=1               pbg (X; π bg ) C
                        En
                              1 − pbg (X; π   bg )
                                                      Ci
                                n                                   2
         √               0 1 X             (1 − pg (Xi ; π̄g ))                                              
                                                                                                                         ∗,nev
                                                                                                                               
       + n π   bg − πg∗            Vi ·                               · Yi,t − Yi,g−δ−1 − mnev         g,t,δ   W i ; β g,t,δ    · ṗg (Xi ; π̄g )
                            n i=1                pbg (X; π bg ) C
                                          En
                                                1 − pbg (X; π   bg )
                                          n
         √   
                 nev       ∗,nev
                                  0 1  X
                                                   bgcomp,nev (Wi ; π    bg ) · ṁnev              nev
                                                                                                         
       − n βbg,t,δ    − βg,t,δ               Vi · w                                g,t,δ Wi ; β̄g,t,δ
                                     n i=1
                                      pg Xi ; πg∗ Ci
                                                    
                        n                                
           comp 1
                       X             1 − pg Xi ; πg∗
       − Mg,t,δ √
         c                  Vi ·                            
                    n i=1                pbg (X; π bg ) C
                                 En
                                        1 − pbg (X; π  bg )
                                                                Ci
                                          n                                   2
                  √               0 1  X            (1 − pg (Xi ; π̄g ))
       −Mccomp n π      bg − πg∗              Vi ·                               · ṗg (Xi ; π̄g )
           g,t,δ
                                                        
                                     n i=1                 pbg (X; π  bg ) C
                                                   En
                                                          1 − pbg (X; π   bg )

                                                                          38

            1,comp     2,comp     3,comp     4,comp     5,comp
        = Ibg,t,δ  + Ibg,t,δ  − Ibg,t,δ  − Ibg,t,δ  − Ibg,t,δ  ,

                                                                        ∗,nev              ∗,nev
        nev
where β̄g,t,δ                                                  nev
              and π̄g are intermediate points that satisfies β̄g,t,δ − βg,t,δ     nev
                                                                              ≤ βbg,t,δ − βg,t,δ a.s. and π̄g − πg∗ ≤
bg − πg∗ a.s., respectively, and
π
                                        h                                                    i
                              ccomp = En w
                              M             comp,nev
                                                     (b
                                                      π g ) ·   Yt − Yg−δ−1 − mnev     bnev
                                                                                       β           .
                                g,t,δ     bg                                   g,t,δ     g,t,δ


From the strong law of large numbers and the fact that V is mean zero, has variance one, and is independent of
W , it follows that
                              n
           1,comp   1 X                                                      
                                                                                         ∗,nev
                                                                                               
         Ibg,t,δ  =√       Vi · wgcomp,nev Wi ; πg∗ · Yi,t − Yi,g−δ−1 − mnev
                                                                         g,t,δ   W i ; β g,t,δ    + op∗ (1) ,
                     n i=1
                         n
           3,treat   1 X                                h                             
                                                                                          ∗,nev
                                                                                                i
                            Vi · wgcomp,nev Wi ; πg∗ · E wgcomp · Yt − Yg−δ−1 − mnev
                                                    
         Ibg,t,δ   =√                                                            g,t,δ  β g,t,δ     + op∗ (1) .
                      n i=1

Similarly, from Assumptions 7 and 8, and the strong law of large numbers, we conclude that
                                                 2,comp     4,comp     5,comp
                                               Ibg,t,δ  = Ibg,t,δ  = Ibg,t,δ  = op∗ (1) .

Now (A.10) follows from combining these results.
   Next, we show that
                              n
                          1 X        
                                        est,nev                         est,nev              ∗,nev
                                                                                                    
                         √       Vi · ψbg,t     (Wi ; π      nev
                                                      bg , βbg,t,δ ) − ψg,t     (Wi ; πg∗ , βg,t,δ ) = op∗ (1) .     (A.11)
                           n i=1

From the strong law of large numbers and Assumptions 7 and 8,
                     n
                1 X           est,nev              nev
               √       Vi · ψbg,t     (Wi ; π
                                            bg , βbg,t,δ )
                 n i=1
                       n                                 0                                           
                   1 X             or,nev
                                          
                                                    nev      dr,nev,1    ps,nev           0   dr,nev,2
               =√          Vi · lg,t        Wi ; βg,t,δ · Mg,t,δ
                                                   b                  + lg            bg ) · Mg,t,δ
                                                                                (Wi ; π                  + op∗ (1)
                    n i=1
                          n
                 1 X          est
                                       κnev
                                             
               =√       Vi · lg,t Wi ; b g,t   + op∗ (1) ,
                  n i=1
                                       0
                             0    nev 0
where, for a generic κnev
                      g,t = πg , βg,t,δ    ,

                      est               or,nev      nev 0      dr,nev,1                      0   dr,nev,2
                          W ; κnev                                      + lgps,nev (Wi ; πg ) · Mg,t,δ
                                                         
                     lg,t      g,t   = lg,t    W ; βg,t,δ   · Mg,t,δ

Thus, from Lemma 4.3 in Newey and McFadden (1994) and Assumption 7, it follows that
                                     n
                                                                                    !
                              ∗  1 X        
                                              est       nev
                                                                est
                                                                     
                                                                           ∗,nev
                                                                                 
                         V ar   √       Vi · lg,t Wi ; κ
                                                       bg,t,δ − lg,t Wi ; κg,t,δ
                                  n i=1
                                         n                                      2
                                      1 X  est                     
                                                                          ∗,nev
                                                      bnev      est
                                                            
                                  =         lg,t Wi ; κg,t,δ − lg,t Wi ; κg,t,δ
                                      n i=1

                                  = op (1) ,

which, in turn, implies (A.11).


                                                                  39

    Taking (A.9), (A.10), and (A.11) together, we then establish (A.8). Thus, by (A.7), we have

                                 √
                                                                 
                                            ∗,dr,nev       dr,nev    d
                                   n AT T t≥(g−δ) − AT T t≥(g−δ) → N (0, Σ).
                                       [              [
                                                                                              ∗


    Finally, by the continuous mapping theorem, see e.g. Theorem 10.8 in Kosorok (2008), for any continuous
functional Γ(·)
                                 √
                                                               
                                          ∗,dr,nev       dr,nev     d
                             Γ     n AT T t≥(g−δ) − AT T t≥(g−δ)
                                      [             [               → Γ (N (0, Σ)) ,
                                                                                              ∗

concluding our proof. 
