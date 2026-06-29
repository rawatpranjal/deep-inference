<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1803.09015.pdf
backend: pdftotext
part: 6/7
-->

# Appendix B: Additional Results for Repeated Cross Sections

Appendix B: Additional Results for Repeated Cross Sections
In this section we extend our identification results to the case where one has access to repeated cross sections data
instead of panel data. Here we assume that for each unit in the pooled sample, we observe (Y, G2 , . . . , GT , C, T, X)
where T ∈ {1, . . . , T } denotes the time period when that unit is observed. Let Tt = 1 if an observation is observed
at time t, and zero otherwise.
    We assume that random samples are available for each time period.

Assumption B.1. Conditional of T = t, the data are independent and identically distributed from the distribution
of (Yt , G2 , . . . , GT , C, X) , for all t = 1, . . . , T ., with (G2 , . . . , GT , C, X) being invariant to T .

    Assumption B.1 implies that our sample consists of random draws from the mixture distribution
                                                                T
                                                                X
                          FM (y, g2 , . . . , gT , c, t, x) =         λt · FY,G2 ,...,GT ,C,X|T (y, g2 , . . . , gT , c, x|t),
                                                                t=1

where λt = P (Tt = 1). It also rules-out compositional changes across time. This assumption is related to the
sampling assumption imposed by Abadie (2005) and Sant’Anna and Zhao (2020) in the two periods, two groups
DiD setup. Notice that, once one conditions on the time period, then expectations under the mixture distribution
correspond to population expectations. Also, because X, Gg , and C are observed for all units, by the stationarity
condition one can use draws from the mixture distribution to estimate the generalized propensity score. With some
abuse of notation, we then use pg,s (X) as a short notation for EM [Gg |X, Gg + (1 − Ds ) (1 − Gg ) = 1], where EM [·]
denotes expectations with respect to FM (·). Also, we use pg (X) = pg,T (X) = EM [Gg |X, Gg + C = 1].
    Before formalizing all the results, we need to introduce some additional notation. Let mrc,nev
                                                                                            c,t    (x) ≡ EM [Y |X =
                   rc,treat                                          rc,ny
x, C = 1, T = t], mg,t      (x) ≡ EM [Y |X = x, Gg = 1, T = t] and ms,t (x) ≡ EM [Y |X = x, Ds = 0, Gg = 0, T =
t]. Consider the weights

                  wtreat (a, b) = Tb · Ga / EM [Tb · Ga ] ,

                     
                    comp           Tb · pa (X) C           Tb · pa (X) C
                 wnev    (a, b) =                    EM                    ,
                                    1 − pa (X)              1 − pa (X)

                                          
                 comp              Tb · pa,s (X) (1 − Db ) (1 − Ga )           Tb · pa,s (X) (1 − Db ) (1 − Ga )
                wny    (a, b, s) =                                      EM                                         .
                                              1 − pa,s (X)                                1 − pa,s (X)

Finally, consider the outcome regression (OR) estimands,
                                                                                                       
                                  Gg                                    
           nev
    AT Tor,rc  (g, t; δ) = EM               mrc,treat
                                             g,t      (X) − mrc,treat
                                                             g,g−δ−1  (X) −  mrc,nev
                                                                              c,t    (X) − mrc,nev
                                                                                            c,g−δ−1 (X)     ,
                                EM [Gg ]

                                                                           40

                                
                                      Gg                                                              
         ny                                      rc,treat        rc,treat       rc,ny        rc,ny
     AT Tor,rc (g, t; δ) = EM                   mg,t      (X) − mg,g−δ−1 (X) − mt+δ,t (X) − mt+δ,g−δ−1 (X)    ,
                                    EM [Gg ]

the inverse probability weighted (IPW) estimands

                        nev
                                                    wtreat (g, t) − wtreat (g, g − δ − 1) · Y
                                                                                            
                    AT Tipw,rc (g, t; δ) = EM
                                                 comp              comp
                                         − EM [(wnev   (g, t) − wnev      (g, g − δ − 1)) · Y ] ,
                        ny                       treat             treat
                                                                                              
                    AT Tipw,rc (g, t; δ) = EM w         (g, t) − w        (g, g − δ − 1) · Y
                                               comp                       comp
                                                                                                         
                                         − EM wny       (g, t, t + δ) − wny      (g, g − δ − 1, t + δ) · Y ,

and the doubly-robust (DR) estimands
                          
                               Gg                                                                             
     nev                                    rc,treat           rc,treat                rc,nev          rc,nev
 AT Tdr,rc (g, t; δ) = EM                 mg,t       (X) − mg,g−δ−1 (X) − mc,t                (X) − mc,g−δ−1 (X)
                            EM [Gg ]
                          h                                                                                     i
                     + EM wtreat (g, t) Y − mrc,treat                 treat                          rc,treat
                                                                
                                                  g,t      (X)    − w       (g, g −  δ −  1)   Y − m g,g−δ−1  (X)
                          h                                                                                    i
                             comp                  rc,nev           comp                           rc,nev
                     − EM wnev    (g, t) Y − mc,t         (X) − wnev      (g, g − δ − 1) Y − mc,g−δ−1 (X) ,
                                                                                                                   
     ny                        Gg     
                                            rc,treat           rc,treat
                                                                              
                                                                                       rc,ny          rc,ny
 AT Tdr,rc (g, t; δ) = EM                 mg,t       (X) − mg,g−δ−1 (X) − mt+δ,t (X) − mt+δ,g−δ−1 (X)
                            EM [Gg ]
                          h                                                                                     i
                     + EM wtreat (g, t) Y − mrc,treat                 treat                          rc,treat
                                                                
                                                  g,t      (X)    − w       (g, g −  δ −  1)   Y − m g,g−δ−1  (X)
                          h                                                                                               i
                                                          rc,ny
                     − EM wnycomp
                                  (g, t, t + δ) Y − mt+δ,t (X) − wny        comp
                                                                                  (g, g − δ − 1, t + δ) Y − mrc,ny
                                                                                                                t+δ,g−δ−1 (X)   .


    The OR, IPW and DR estimands respectively generalize Heckman et al. (1997), Abadie (2005) and Sant’Anna
and Zhao (2020) estimands for the two groups, two periods DiD setup to the staggered DiD setup with multiple
periods and multiple groups.

Theorem B.1. Let Assumptions 1, 3, 6, and B.1 hold.
    (i) If Assumption 4 in the main text holds, then, for all g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ} and t ≥ g −δ,

                                           nev                    nev                   nev
                         AT T (g, t) = AT Tipw,rc (g, t; δ) = AT Tor,rc (g, t; δ) = AT Tdr,rc (g, t; δ) .


    (ii) If Assumption 5 in the main text holds, then, for all g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ} and
t ≥ g − δ,
                                        ny                     ny                    ny
                      AT T (g, t) = AT Tipw,rc (g, t; δ) = AT Tor,rc (g, t; δ) = AT Tdr,rc (g, t; δ) .

    We defer the proof of Theorem B.1 to the Supplementary Appendix. The identification results in Theorem B.1
suggest a simple two-step estimation procedure for the AT T (g, t) with repeated cross-section data that is analogous
to the panel data case discussed in Section 4. The asymptotic properties of such two-step estimators follow from
analogous arguments; the details are omitted for brevity. Likewise, we can aggregate these estimators to provide
summary measures of the causal effects like those discussed in Section 3.
