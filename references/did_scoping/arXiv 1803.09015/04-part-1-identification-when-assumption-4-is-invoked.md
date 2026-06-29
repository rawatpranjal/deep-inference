<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1803.09015.pdf
backend: pdftotext
part: 4/7
-->

# Part 1: Identification when Assumption 4 is invoked.

Part 1: Identification when Assumption 4 is invoked.
    In this case, given the result in Lemma A.1,

                  AT T (g, t) = E[AT TX (g, t)|Gg = 1]
                              = E [ (E[Yt − Yg−δ−1 |X, Gg = 1] − E[Yt − Yg−δ−1 |X, C = 1])| Gg = 1]
                                                                                             
                                                                                           
                              = E[Yt − Yg−δ−1 |Gg = 1] − E 
                                                           |E[Yt − Yg−δ−1 |X, C = 1]|Gg = 1
                                                                                            
                                                                       {z           }
                                                                       =mnev
                                                                         g,t,δ (X)
                                                                  
                                  Gg                    nev
                                                                 
                              =E         Yt − Yg−δ−1 − mg,t,δ (X) .
                                 E [Gg ]
                                      nev
Hence, we have that AT T (g, t) = AT Tor  (g, t; δ) .
                                         nev
   Next, to show that AT T (g, t) = AT Tipw  (g, t; δ), it suffices to show that
                                                     
                            pg (X) C
                        E               (Yt − Yg−δ−1 )
                          (1 − pg (X))                    E [Gg · E[Yt − Yg−δ−1 |X, C = 1]]
                                                      =                                   .               (A.1)
                                    pg (X) C                            E [Gg ]
                              E
                                  (1 − pg (X))

Towards this end, by noticing that

                                              E [Gg |X]                   E [C|X]
                                pg (X) =                , 1 − pg (X) =              ,                       (A.2)
                                           E [Gg + C|X]                E [Gg + C|X]

it follows that
                                                                    
                                          pg (X) C         E [Gg |X] C
                                       E              =E
                                         1 − pg (X)         E [C|X]
                                                                            
                                                           E [Gg |X] E [C|X]
                                                      =E
                                                               E [C|X]
                                                         = E [E [Gg |X]]
                                                         = E [Gg ] .                                        (A.3)

Next, by exploiting (A.2) and applying the law of iterated expectations, we have that
                                                                                  
                          pg (X) C                        E [Gg |X] C
                    E                (Yt − Yg−δ−1 ) = E               (Yt − Yg−δ−1 )
                        (1 − pg (X))                       E [C|X]
                                                                                             
                                                          E [Gg |X]
                                                     =E             E [ C · (Yt − Yg−δ−1 )| X]
                                                          E [C|X]
                                                     = E [E [Gg |X] · E [ (Yt − Yg−δ−1 )| X, C = 1]]


                                                            34

                                                                = E [Gg · E [ (Yt − Yg−δ−1 )| X, C = 1]] .

                                                                                    nev
Thus, combined this with (A.3), we establish (A.1), implying that AT T (g, t) = AT Tipw (g, t; δ).
   Finally, notice that

                                                       pg (X) C
                                                                                              

                      nev
                                         Gg         1 − pg (X)                     nev
                                                                                                
                 AT Tdr                  E [Gg ] −  pg (X) C   Yt − Yg−δ−1 − mg,t,δ (X) 
                          (g, t; δ) = E                                                       
                                                    E
                                                       1 − pg (X)
                                                       pg (X) C
                                                                               
                                         Gg         1 − pg (X)                 
                                    = E E [Gg ] −  pg (X) C   (Yt − Yg−δ−1 )
                                                                                
                                                    E
                                                       1 − pg (X)
                                      |                     {z                    }
                                                                        nev (g,t;δ)
                                                                   ≡AT Tipw

                                                                           pg (X) C
                                                                                                     
                                                          Gg            1 − pg (X)    nev
                                                                                                        
                                                    − E  E [Gg ] −  pg (X) C   mg,t,δ (X)
                                                                                                       
                                                                       E
                                                                           1 − pg (X)
                                                                                                    
                                                            1               E [Gg |X] C
                                         = AT T (g, t) −         E Gg −                   mnev
                                                                                            g,t,δ (X)
                                                         E [Gg ]              E [C|X]
                                                            1
                                                                 E (E [Gg |X] − E [Gg |X]) · mnev
                                                                                                         
                                         = AT T (g, t) −                                        g,t,δ (X)
                                                         E [Gg ]
                                         = AT T (g, t) .

                                          nev
where the third equality follows from AT Tipw (g, t; δ) = AT T (g, t), (A.2) and (A.3), and the fourth equality from
the law of iterated expectations.
