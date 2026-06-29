<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2408.10509.pdf
backend: pdftotext
part: 3/6
-->

# Appendix A: Algorithms for Constructing DML Estimators

Appendix A: Algorithms for Constructing DML Estimators

Algorithm 1 (CDID Estimator, Panel). Let {Ik }K
                                              k=1 denote a partition of the random sample
{(Yi,t−1 , Yi,t , Di , Xi }N                                                                         c
                           i=1 , each with equal size n = N/K, and for each k ∈ {1, · · · , K}, let Ik :=
IN \ Ik denote the complement.

    STEP 1. For each k, construct

                        1   X Kh (Di − d)ĝk (Xi ) − 1{Di = 0}fˆh,k (d|Xi )                                   
          AT
          [  T k (d) :=                                                                     ∆Yi − Ê∆Y,k (Xi )    (A.1)
                        n
                            i∈Ik
                                                      fˆk (d)ĝk (Xi )

      where fˆk (d), fˆh,k , ĝk , Ê∆Y,k are the estimators of fd , fh (d|X), g(X) and E∆Y (X) respectively
      using the rest of the sample I c . In particular, fˆk (d) is a kernel density estimator, and fˆh,k ,
                                       k
      ĝk and Ê∆Y,k are estimated using ML methods (e.g. random forests or deep neural networks).


    STEP 2. Average through the K estimators to obtain the final estimator

                                                                   K
                                                        1          X
                                            AT
                                            [  T (d) :=                    AT
                                                                           [  T k (d).                            (A.2)
                                                        K
                                                                   k=1


Algorithm 2 (CDID Estimator, Repeated Cross-Sections). Let {Ik }K
                                                                k=1 denote a partition of
the random sample {(Yi,t−1 , Yi,t , Di , Xi }N
                                             i=1 , each with equal size n = N/K, and for each k ∈
{1, · · · , K}, let Ikc := IN \ Ik denote the complement.

    STEP 1. For each k, construct

                                     1     X Kh (Di − d)ĝk (Xi ) − 1{Di = 0}fˆh,k (d|Xi )
                       AT
                       [  T k (d) :=
                                     n
                                           i∈Ik
                                                                         fˆk (d)ĝk (Xi )
                                                                                           
                                                            Ti − λ̂k
                                                  ×                        Yi − ÊλY,k (Xi )                      (A.3)
                                                          λ̂k (1 − λ̂k )

      where fˆk (d), fˆh,k , ĝk , ÊλY,k are the estimators of fd , fh (d|X), g(X) and EλY (X) respectively
      using the rest of the sample I c . In particular, fˆk (d) is a kernel density estimator, and fˆh,k ,
                                       k
      ĝk and Ê∆Y,k are estimated using ML methods (e.g. random forests or deep neural networks).

    STEP 2. Average through the K estimators to obtain the final estimator

                                                                   K
                                                        1          X
                                            AT
                                            [  T (d) :=                    AT
                                                                           [  T k (d).                            (A.4)
                                                        K
                                                                   k=1


                                                             25
