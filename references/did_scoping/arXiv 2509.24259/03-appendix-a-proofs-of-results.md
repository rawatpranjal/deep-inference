<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2509.24259.pdf
backend: pdftotext
part: 3/7
-->

# Appendix A: Proofs of Results

Appendix A: Proofs of Results

Proof of Proposition 1:

Proof.

                1 ÿ ” `                         ˘    `                      ˘ı
         τ obs “        E Yi2 ´ Yi1 | Di “ 1, xi ´ E Yi2 ´ Yi1 | Di “ 0, xi
               mn iPM
                      n

                1 ÿ ÿ `                                   ˘
             “            E Yi2 ´ Yi1 | Di “ 1, Gi “ g, xi ¨ PpGi “ g | Di “ 1, xi q
               mn iPM gPG
                      n

                 1 ÿ ÿ `                                    ˘
               ´            E Yi2 ´ Yi1 | Di “ 0, Gi “ g, xi ¨ PpGi “ g | Di “ 0, xi q.
                 mn iPM gPG
                          n


               (by iterated expectations law)


             1 ÿ ÿ `                                                    ˘
 τ DATT “                E Yi2 p1, gq ´ Yi2 p0, gq | Di “ 1, Gi “ g, xi ¨ PpGi “ g | Di “ 1, xi q
            mn iPM gPG
                   n

             1   ÿ   ÿ” `                                                 ˘
          “                E Yi2 p1, gq ´ Yi1 p0, 0q | Di “ 1, Gi “ g, xi
            mn iPM gPG
                   n
               `                                             ˘ı
            ´ E Yi2 p0, gq ´ Yi1 p0, 0q | Di “ 0, Gi “ g, xi ¨ PpGi “ g | Di “ 1, xi q

           (by network conditional parallel trends)
             1 ÿ ÿ `                                   ˘
          “            E Yi2 ´ Yi1 | Di “ 1, Gi “ g, xi ¨ PpGi “ g | Di “ 1, xi q
            mn iPM gPG
                   n

               1 ÿ ÿ `                                   ˘
            ´            E Yi2 ´ Yi1 | Di “ 0, Gi “ g, xi ¨ PpGi “ g | Di “ 1, xi q.
              mn iPM gPG
                      n


            (by consistency and the no anticipation assumption)


                                                36

Then, we have

                1 ÿ ÿ `                                     ˘
τ obs ´ τ DATT “           E Yi2 ´ Yi1 | Di “ 0, Gi “ g, xi
                mn iPM gPG
                      n
                   ”                                                 ı
                  ¨ PpGi “ g | Di “ 1, xi q ´ PpGi “ g | Di “ 0, xi q
                1 ÿ ÿ” `                                      ˘   `                        1
                                                                                                 ˘ı
              “             E Yi2 ´ Yi1 | Di “ 0, Gi “ g, xi ´ E Yi2 ´ Yi1 | Di “ 0, Gi “ g , xi
                mn iPM gPG
                      n
                   ”                                                 ı
                  ¨ PpGi “ g | Di “ 1, xi q ´ PpGi “ g | Di “ 0, xi q .


     (since the subtraction of a constant baseline term EpYi2 ´ Yi1 | Di “ 0, Gi “ g 1 , xi q

         leaves the expression unchangedq


Proof of Proposition 2:

Proof.

                     1 ÿ ” `                          ˘    `                      ˘ı
           τ obs “            E Yi2 ´ Yi1 | Di “ 1, xi ´ E Yi2 ´ Yi1 | Di “ 0, xi
                     mn iPM
                            n

                     1 ÿ ÿÿ `                                                ˘
                   “                E Yi2 ´ Yi1 | Di “ 1, Gi “ g, Ui “ u, xi
                     mn iPM gPG uPU
                             n


                        ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q
                       1 ÿ ÿÿ `                                                ˘
                     ´                E Yi2 ´ Yi1 | Di “ 0, Gi “ g, Ui “ u, xi
                       mn iPM gPG uPU
                                 n


                         ¨ PpUi “ u | Di “ 0, Gi “ g, xi q ¨ PpGi “ g | Di “ 0, xi q.

                    (by iterated expectations law)


                                                 37

                 1 ÿ ÿÿ `                                                              ˘
      τ DATT “                  E Yi2 p1, gq ´ Yi2 p0, gq | Di “ 1, Gi “ g, Ui “ u, xi
                 mn iPM gPG uPU
                          n


                   ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q
                1 ÿ ÿÿ” `                                                               ˘
             “                   E Yi2 p1, gq ´ Yi1 p0, 0q | Di “ 1, Gi “ g, Ui “ u, xi
               mn iPM gPG uPU
                       n
                  `                                                     ˘ı
               ´ E Yi2 p0, gq ´ Yi1 p0, 0q | Di “ 0, Gi “ g, Ui “ u, xi

                     ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q

              (by network conditional parallel trends)
                1 ÿ ÿÿ `                                               ˘
             “                E Yi2 ´ Yi1 | Di “ 1, Gi “ g, Ui “ u, xi
               mn iPM gPG uPU
                          n


                    ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q
                   1 ÿ ÿÿ `                                                ˘
                 ´                E Yi2 ´ Yi1 | Di “ 0, Gi “ g, Ui “ u, xi
                   mn iPM gPG uPU
                              n


                     ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q.

                (by consistency and the no anticipation assumption)

Then, we have

                        1 ÿ ÿÿ `                                                ˘
     τ obs ´ τ DATT “                  E Yi2 ´ Yi1 | Di “ 0, Gi “ g, Ui “ u, xi
                        mn iPM gPG uPU
                              n
                           ”
                          ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q
                                                                                         ı
                              ´ PpUi “ u | Di “ 0, Gi “ g, xi q ¨ PpGi “ g | Di “ 0, xi q
                        1     ÿ ÿÿ” `                                            ˘
                    “                   E Yi2 ´ Yi1 | Di “ 0, Gi “ g, Ui “ u, xi
                        mn iPM gPG uPU
                              n
                            `                         1         1
                                                                     ˘ı
                        ´ E Yi2 ´ Yi1 | Di “ 0, Gi “ g , Ui “ u , xi
                           ”
                          ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | Di “ 1, xi q
                                                                                         ı
                              ´ PpUi “ u | Di “ 0, Gi “ g, xi q ¨ PpGi “ g | Di “ 0, xi q .


                                                38

  (since the subtraction of a constant baseline term EpYi2 ´ Yi1 | Di “ 0, Gi “ g 1 , Ui “ u1 , xi q

     which do not depend on g and u.q

Then, under conditional independence between Zi and Gi , the bias become:

                              1 ÿ ÿÿ `                                                  ˘
           τ obs ´ τ DATT “                  E Yi2 ´ Yi1 | Di “ 0, Gi “ g, Ui “ u, xi
                              mn iPM gPG uPU
                                     n
                                 ”
                                ¨ PpUi “ u | Di “ 1, Gi “ g, xi q ¨ PpGi “ g | xi q
                                                                                      ı
                                   ´ PpUi “ u | Di “ 0, Gi “ g, xi q ¨ PpGi “ g | xi q .

After marginalizing over Gi , the expression simplifies to

                               1 ÿ ÿ” `                                     ˘
            τ obs ´ τ DATT “               E Yi2 ´ Yi1 | Di “ 0, Ui “ u, xi
                               mn iPM uPU
                                      n
                                     `                          1
                                                                     ˘ı
                                 ´ E Yi2 ´ Yi1 | Di “ 0, Ui “ u , xi
                                  ”                                                 ı
                                 ¨ PpUi “ u | Di “ 1, xi q ´ PpUi “ u | Di “ 0, xi q .


Proof of Proposition 3:

Proof. Recall that:
               ˆ                                                      ˙
                                    p1 ´ Di q1tGi “ gup1g pi, X, Aq
  τ pgq “ ED       pDi 1tGi “ guq ´                                       p∆Yi ´ ∆u0g pi, X, Aqq .
                                           1 ´ p1g pi, X, Aq

For notational simplicity, ED denote the finite population expectation conditional on X
and A.
   Case 1: When outcome regression models are correctly specified. In this case,
we have that ∆µ0t pi, X, Aq “ ∆m0t pi, X, Aq a.s., i.e. the outcome regression models are
correctly specified.


                                                 39

           „ˆ                                               ˙                        ȷ
  dr                        p1 ´ Di q1tGi “ gup1g pi, X, Aq
τ pgq “ ED   Di 1tGi “ gu ´                                   p∆Yi ´ ∆m0g pi, X, Aqq
                                   1 ´ p1g pi, X, Aq

       “ ED rDi 1tGi “ gu p∆Yi ´ ∆m0g pi, X, Aqqs
              „                                                        ȷ
                 p1 ´ Di q1tGi “ gup1g pi, X, Aq
         ´ ED                                    p∆Yi ´ ∆m0g pi, X, Aqq
                        1 ´ p1g pi, X, Aq

       “ ED rDi 1tG “ gu p∆Y ´ ∆m0g pi, X, Aqqs
              „                                                         ȷ
                   p1t pi, X, Aq
         ´ ED                      p∆Yi ´ ∆m0t pi, X, Aqq | D “ 0, G “ g p0g
                 1 ´ p1g pi, X, Aq

       “ ED rp∆m1g pX, Aq ´ ∆m0g pX, Aqq | D “ 1, G “ gs
               „                                                                    ȷ
                   p1g pi, X, Aq
         ´ ED                       p∆m0g pi, X, Aq ´ ∆m0g pi, X, Aqq | D “ 0, G “ g p0g
                 1 ´ p1g pi, rX, Aq

       “ τ DAT T pgq.

where the third step applies the law of iterated expectations, and the final step is justified
by the conditional parallel trends assumption.
   Case 2: When propensity score model is correctly specified. In this case, we
have that
              „ˆ                                               ˙                        ȷ
  dr                          p1 ´ Di q1tGi “ guπ1g pi, X, Aq
 τ pgq “ ED    Di 1tGi “ gu ´                                    p∆Yi ´ ∆u0g pi, X, Aqq
                                      1 ´ π1g pi, X, Aq
            ˆˆ                                                 ˙      ˙
                               p1 ´ Di q1tGi “ guπ1g pi, X, Aq
       “ ED    Di 1tGi “ gu ´                                     ∆Yi
                                      1 ´ π1g pi, X, Aq
              „ˆ                                                 ˙                ȷ
                                 p1 ´ Di q1tGi “ guπ1g pi, X, Aq
         ´ ED    Di 1tGi “ gu ´                                    ∆u0g pi, X, Aq
                                        1 ´ π1g pi, X, Aq
        “ τ DAT T pgq ´ ED E rpπ1g ´ π1g q∆u0g pi, X, Aqs

        “ τ DAT T pgq.

   The third equality follows from Lemma 3.1 in Abadie (2005) [1] and the law of iterated
expectations, reducing exactly to the formulation in their paper when the indicator G is

                                              40

omitted.


Proof of Theorem 1:

Before proving Theorem 1, we first introduce a definition and a lemma.

Definition 1. A triangular array tZi uni“1 is conditionally ψ-dependent given Fn if there
exists a constant C ą 0 and an Fn -measurable sequence tψn psqus,nPN with ψn p0q “ 1 for all
n such that for every n, h, h1 P N, every s ą 0, every function f P Lh and f 1 P Lh1 , and
every pair pH, H 1 q P Pn ph, h1 ; sq, we have

         ˇ    ´                  ¯ˇ          ´             ¯´              ¯
                          1                1                  1          1
         ˇ Cov f pZ H q, f pZ H q ˇ ď C h h }f }8 ` Lippf q }f }8 ` Lippf q ψn psq
         ˇ                        ˇ
                               1


almost surely; here, ψn psq is called the dependence coefficient of the array.

Lemma 1. Under Assumptions 4, 5, 6(a), 6(b) hold, then for any g P G, the sequence
tϕi pgquni“1 is conditionally ψ-dependent given pX, Aq as per Definition 1, with the depen-
dence coefficient ψn psq defined by (32).

Proof. Let Fn be the σ-algebra generated by pX, Aq, ph, h1 q P N2 , pf, f 1 q P Lh ˆ Lh1 , s ą 0,
and pH, H 1 q P Pn ph, h1 ; sq. Fix g P G and write

           ´                       p1 ´ Di q 1tGi “ gu p1g pi, X, Aq ¯ ´                              ¯
ϕi pgq “    Di 1tGi “ gu ´                                                       ¨ ∆Yi ´∆u0g pi, X, Aq ´ τ DAT T pgq.
                                             1 ´ p1g pi, X, Aq
           looooooooooooooooooooooooooooooooomooooooooooooooooooooooooooooooooon
                                         “:Wi pgq


Define Zi “ ϕi pgq, Z H “ pZi qiPH , ξ “ f pZ H q, and similarly ζ “ f 1 pZ H 1 q.
                            ` N pj,s{2q N pj,s{2q N pj,s{2q ˘                      ´       ¯
                    ps{2q                                            ps{2q           ps{2q
   For fix s, take Djt “ ljt X           ,A         ,ν         and D N pi,s1 {2q “ Dj                                         ,
                                                                                                              jPN pi,s1 {2q
                                                              ps{2q                         ps{2q
define the s{2-local exposure indicator 1i                            pgq “ 1tGpi, D N pi,s{2q , AN pi,s{2q q “ gu, the
                                     “ ∆hit pD N pi,s{2q , X N pi,s{2q , AN pi,s{2q , ϵN pi,s{2q , and the s{2-local
                             ps{2q                    ps{2q
s{2-local difference ∆Yi
weight
                                                                            ps{2q   ps{2q
                  ps{2q              ps{2q    ps{2q              p1 ´ Di       q 1i pgq p1g pi, X, Aq
               Wi         pgq “ Di           1i       pgq ´                                           .
                                                                             1 ´ p1g pi, X, Aq


                                                                41

Set
                                                    ´                         ¯
                         ps{2q           ps{2q          ps{2q
                        Zi       “ Wi            pgq ∆Yi      ´ ∆u0g pi, X, Aq ´ τ DAT T pgq.
            ` ps{2q ˘       ` ps{2q ˘
      Hence, Zi       iPH
                          KK Zj       jPH 1
                                            | Fn , then we have

        ˇ               ˇ ˇ                             ˇ ˇ                                 ˇ
        ˇCovpξ, ζ | Fn qˇ ď ˇCovpξ ´ ξ ps{2q , ζ | Fn qˇ ` ˇCovpξ ps{2q , ζ ´ ζ ps{2q | Fn qˇ
                                      “               ˇ   ‰               “              ˇ    ‰
                          ď 2}f 1 }8 E |ξ ´ ξ ps{2q | ˇ Fn ` 2}f }8 E |ζ ´ ζ ps{2q | ˇ Fn
                                                                             ”                ˇ ı
                                   1                1          1                        ps{2q ˇ
                          ď 2ph}f }8 Lippf q ` h }f }8 Lippf qq max E |Zi ´ Zi | Fn .
                                                                                 iPNn


Thus it remains to bound maxi Er|Zi ´ Zi1 | | Fn s. Write

                ps{2q      `              ps{2q    ˘ `   ˘    ps{2q  `              ps{2q ˘
       Zi ´ Zi                                                         ∆Yi ´ ∆Yi
                             Wi pgq ´ Wi pgq ¨ ∆Yi ´ ∆u0g ` Wi pgq ¨ looooooooomooooooooon
                         “ looooooooooomooooooooooon                                        .
                                      weight gap                                                     outcome gap


Hence, for some constant C0 ą 0,

  ”             ˇ ı    ´ ”                  ˇ   ı   ”                ˇ  ı¯
          ps{2q                       ps{2q                    ps{2q ˇ
 E |Zi ´ Zi | ˇ Fn ď C0 E |Wi pgq ´ Wi pgq| ˇ Fn ` E |∆Yi ´ ∆Yi     | Fn .


Under Lemma 2,

                    ”                  ˇ   ı    ´                          ¯
                                 ps{2q
                   E |Wi pgq ´ Wi pgq| ˇ Fn ď C1 ηn ps{2q ` npi, Kqηn ps{2q .


Under Lemma 3,

                 ˇ ”                ˇ  ıˇ
                              ps{2q ˇ
                 ˇE |∆Yi ´ ∆Yi     | Fn ˇ ď 2γn ps{2q ` Λn pi, s{2q npi, s{2q ηn ps{2q.
                 ˇ                      ˇ


Then there exists C2 ą 0 such that

         “                   ˇ   ‰                ´                                                                         ¯
max E        |Zi ´ Zi1 | ˇ Fn        ď C2 ¨ max γn ps{2q ` ηn ps{2q r1 ` npi, Kq ` Λn pi, s{2q npi, s{2qs .
iPNn                                        iPNn
                                            looooooooooooooooooooooooooooooooooooooomooooooooooooooooooooooooooooooooooooooon
                                                                                “: ψn psq


Assumption 9 (Local Lipschitz Continuity). For each t P t1, 2u there exists Λn pi, sq ą

                                                               42

0 such that for all d, d1 P t0, 1un ,

                                                                                      ÿ
     ˇhit pdN pi,sq , X, A, εN                                   N pi,sq ˇ
     ˇ                         pi,sq
                                                                         ˇ
                             t       q ´ hit pd1N pi,sq , X, A, εt      q ď Λn pi, sq   |dj ´ d1j |.
                                                                                      jPN pi,sq

                                        ´                                ¯
Lemma 2. Fix s, and abbreviate Dj1 “ ljt X N pj,sq , AN pj,sq , ν N pj,sq ,            D 1B “ pDj1 qjPB , B Ď
Nn , and define, for any exposure value g P G,

                   ! `                   )                                ! `                    )
                           N pi,Kq
                                     ˘                                            1N pi,Kq
                                                                                             ˘
         1i pgq “ 1 G i, D         ,A “ g ,                 11i pgq “ 1    G i, D          ,A “ g .


Under Assumption 1 and 4 hold,

                          ”ˇ                  ˇ ˇˇ    ı
                         E ˇ 1i pgq ´ 11i pgq ˇ ˇ X, A ď npi, Kq ηn psq.


Proof. Let J :“ N pi, Kq and enumerate J “ tj1 , . . . , jm u with m “ |J|. Define a sequence
of treatment vectors by changing one coordinate in J at a time:


       D p0q “ D,        D prq “ same as D pr´1q except pD prq qjr “ Dj1 r , r “ 1, . . . , m.


Since G is K-local, hence

                                 m ˇ
            ˇ                ˇ ÿ   ˇ␣       prq
                                                     (   ␣       pr´1q
                                                                               (ˇˇ
            ˇ1i pgq ´ 11i pgqˇ ď   ˇ1 Gpi, D , Aq “ g ´ 1 Gpi, D       , Aq “ g ˇ
                                 r“1
                                 ÿm                       ÿ
                             ď         |Djr ´ Dj1 r | “         |Dj ´ Dj1 |.
                                 r“1                      jPJ


Taking conditional expectations and using Assumption 4,

          ”ˇ
                     1
                          ˇ ˇˇ   ı  ÿ “            ˇ     ‰
         E 1i pgq ´ 1i pgq ˇ X, A ď
           ˇ              ˇ          E |Dj ´ Dj1 | ˇ X, A ď npi, Kq ηn psq.
                                                 jPJ


                                                       43

Lemma 3. Let Bi “ N pi, sq, npi, sq “ |Bi |. Define:

                                                       N pj,sq ˘
                         “ ljt XN pj,sq , AN pj,sq , ν t
                              `
                      1
                     Djt                                         ,    D1t Bi “ pDjt
                                                                                 1
                                                                                    qjPBi ,


and
                                              `                          ˘
                                    Yit1 “ hit D1t Bi , XBi , ABi , εB
                                                                     t
                                                                       i
                                                                           .

Let ∆Yi “ Yi2 ´ Yi1 and ∆Yi1 “ Yi21 ´ Yi11 . Under Assumptions 4, and 9,

             ˇ                               ˇ
                                    1
             ˇ Er∆Yi | X, As ´ Er∆Yi | X, As ˇ ď 2γn psq ` Λn pi, sq npi, sq ηn psq.
             ˇ                               ˇ


Proof. By Assumption 4 and the tower property,

                  ˇ                 “      Bi   Bi  Bi  Bi
                                                                  ‰ˇˇ
                  ˇErYit | X, As ´ E hit pDt , X , A , εt q | X, A ˇ ď γn psq.
                  ˇ


Subtracting t “ 1 from t “ 2 and applying the triangle inequality,

                       ˇ                  “       Bi
                                                              ‰ ˇˇ
                                 X, As      ∆h           X, A    ˇ ď 2γn psq,
                       ˇ
                       ˇ Er∆Yi |       ´ E    i pDt  q |


Then using Assumption 2, 4 and 9,
       ˇ                                      ˇ ˇ                                                  ˇ
                 Bi            1 Bi                        Bi                   1 Bi
                        ∆h              X, As                  ,        h            ,        X, As
       ˇ                                      ˇ ˇ                                                  ˇ
       ˇ Er∆hi pDt  q ´    i pDt    q |        “
                                              ˇ ˇ Erh i2 pD2      ¨q ´    i2 pD 2      ¨q  |       ˇ
                                                   ”ˇ                                       ˇˇˇ      ı
                                                                                  1 Bi
                                               ď E ˇhi2 pDB  2
                                                                i
                                                                  , ¨q ´ h  i2 pD 2    , ¨q ˇ ˇ X, A
                                                           ÿ ”                           ˇ       ı
                                                                                    1 ˇ
                                               ď Λn pi, sq        E |Dj2 ´ Dj2 | ˇ X, A
                                                                     jPBi

                                                     ď Λn pi, sq npi, sq ηn psq,

Therefore,

              ˇ                                ˇ
              ˇ Er∆Yi | X, As ´ Er∆Yi1 | X, As ˇ ď 2γn psq ` Λn pi, sq npi, sq ηn psq.
              ˇ                                ˇ


                                                       44

   Now we are ready to prove Theorem 1.
                                        ? ` dr                  ˘
Proof. We start with the difference      mn τ̂ pgq ´ τ DAT T pgq . The first step is to write this
difference as a main sum plus a few remainder terms. Specifically,

                ? ` dr                  ˘    1 ÿ
                 mn τ̂ pgq ´ τ DAT T pgq “ ?            ϕi pgq ` R1 ` R2 ,
                                             mn iPM
                                                      n
                                           loooooooomoooooooon
                                                        main term

where

       1 ÿ                                                  p̂1g pi, X, Aq ´ p1g pi, X, Aq
R1 “ ?       p1 ´ Di q 1pGi “ gq p∆Yi ´ ∆u0g pi, X, Aqq                                         ,
      mn iPM                                            p1 ´ p̂1g pi, X, Aqqp1 ´ p1g pi, X, Aqq
                 n


       1 ÿ p̂1g pi, X, Aq ´ Di 1pGi “ gq
R2 “ ?                                   p∆u0g pi, X, Aq ´ ∆û0g pi, X, Aqq.
      mn iPM        1 ´ p̂1g pi, X, Aq
                 n


The function ϕi pgq captures the leading contribution of unit i to the difference between the
estimand and true targets. To establish the asymptotic properties of the main term, we
introduce the concept of ψ-dependence as defined in Kojevnikov (2021) [20] to characterize
weak dependence. Let Ld,t represent the set of all real-valued functions f p¨q defined on Rνˆh
that are bounded and Lipschitz continuous. Lippf q be the Lipschitz constant of f P Ld,t
Additionally, define the collection of subset pairs as:

                         !          ˇ                                                      )
         PM ph, h1 ; sq “ pH, H 1 q ˇ H, H 1 Ď DM , |H| “ h, |H 1 | “ h1 , ℓA pH, H 1 q ě s .
                                    ˇ


This set PM ph, h1 ; sq consists of all pairs pH, H 1 q of subsets drawn from DM , where - H and
H 1 have sizes h and h1 , respectively. The minimum separation distance ρpH, H 1 q between
the two subsets is at least s, ensuring a certain level of weak dependence between them.
   Given sigma-algebra Fn generate by pX, Aq , the collection tϕi pgquni“1 is ψ-dependent.
By assumptions on boundedness and dependence (Assumptions 7(a), 7(b)), one can apply
central limit theorem for ψ-dependent sequences (Kojevnikov et al., 2021 [20] , Theorem 3.2).
We can obtain
                                      1 ÿ          d
                               σn´1 ?       ϕi pgq Ý
                                                   Ñ Np0, 1q,
                                     mn iPM
                                               n


                                                   45

i.e., normalized by σn , the main term converges in distribution to standard normal.
   Thus, it remains to show that the remainder terms R1 and R2 are negligible. We begin
by writing µ0g piq “ µ0g pi, X, Aq, p1g piq “ p1g pi, X, Aq and p̂1g piq “ p̂1g pi, X, Aq. Squaring
and taking expectations yields

                     1 ÿ           ” ”                                                         ı
        ErpR1 q2 s “            E E p∆Yi ´ ∆µ0g piqqp∆Yj ´ ∆µ0g pjqq | D, X, A
                    mn i,jPM
                              n
                                                                                ȷ
                        1i pgq1j pgqpp̂1g piq ´ p1g piqqpp̂1g pjq ´ p1g pjqq
                    ˆ
                      p1 ´ p̂1g piqqp1 ´ p1g piqqp1 ´ p̂1g pjqqp1 ´ p1g pjqq
                    CC 1 ÿ           ´ ℓ pi, jq ¯1´2{p
                                        A
                  ď               γn                   E r|p̂1g piq ´ p1g piq||p̂1g pjq ´ p1g pjq|s
                     mn i,jPM             2
                                   n


                    (using Assumption 6 and Lemma C.5 in [23] )
                         8
                    CC 1 ÿ                  ÿ
                  “        γn ps{2q1´2{p       1tℓA pi, jq “ su
                    mn s“0               i,jPM         n


                       ˆ E r|p̂1t piq ´ p1t piq||p̂1t pjq ´ p1t pjq|s

                   (grouping pairs by network distance)
                                          ˜                      ¸1{2
                         8
                    CC 1 ÿ                  ÿ
                  ď         γn ps{2q1´2{p       1tℓA pi, jq “ su
                     mn s“0                 i,j
                      ˜                                             ¸1{2
                        ÿ                     “                   ‰
                    ˆ     1tℓA pi, jq “ suE pp̂1t piq ´ p1t piqq2
                            i,j
                                                ˜                       ¸1{2
                             8                         n
                             ÿ              n      1  ÿ
                  ď CC 1     γn ps{2q1´2{p                |N B pi, sq|2
                         s“0
                                            m n    n i“1
                      ˜                                  ¸1{2
                            n
                         1ÿ “                          ‰
                    ˆ          E pp̂1t piq ´ p1t piqq2        .
                         n i“1

   Under Assumptions 6 and 7(b), the terms on the right-hand side converge to zero,
implying that ErR1 s “ op p1q and thus R1 is negligible. Then, following the proof of Theorem
3.1 in Farrell (2021) [13] , we obtain that R2 “ op p1q. Because each of the remainder terms
R1 , R2 is shown to be negligible relative to the main term, they do not affect the limiting
distribution. This establishes that

                             ? ` dr                  ˘ d  `      ˘
                              mn τ̂ pgq ´ τ DAT T pgq ÝÑ N 0, σ 2 ,

                                                      46

for some limit variance σ 2 .


Proof of Theorem 2:

Define
                                          ϕ̂i pgq “ τ̂idr pgq ´ τ̂ dr pgq,

and let
                                    1 ÿ ÿ
                           σ̂ 2 “              ϕ̂i pgqϕ̂j pgq1tℓA pi, jq ď bn u,
                                    mn iPM jPM
                                            n      n


                                    1 ÿ ÿ
                           σ2 “                ϕi pgqϕj pgq1tℓA pi, jq ď bn u.
                                    mn iPM jPM
                                            n      n


   We first aim to show the convergence result:

                                                                 p
                                                |σ̂ 2 ´ σ 2 | Ý
                                                              Ñ 0.


   Note that:
                  ˇ                                                                  ˇ
                  ˇ 1 ÿ ÿ ´                                       ¯                  ˇ
  |σ̂ 2 ´ σ 2 | “ ˇ                 ϕ̂i pgqϕ̂j pgq ´ ϕi pgqϕj pgq 1tlA pi, jq ď bn uˇ
                  ˇ                                                                  ˇ
                  ˇ mn iPM jPM                                                       ˇ
                           n     n
                  ˇ                                                                         ˇ
                  ˇ 1 ÿ ´                       ¯ ÿ ´                    ¯                  ˇ
                             ϕ̂i pgq ´ ϕi pgq            ϕ̂j pgq ` ϕj pgq 1tlA pi, jq ď bn uˇ
                  ˇ                                                                         ˇ
                “ˇ
                  ˇ mn iPM                       jPM
                                                                                            ˇ
                           n                            n

                     ˜                                 ¸1{2 ˜                                            ¸1{2
                            n                                       n
               n         1 ÿ´                  ¯2                1ÿ         ´              ¯2
             ď                 ϕ̂i pgq ´ ϕi pgq                        max ϕ̂j pgq ` ϕj pgq npi, bn q2          .
               mn        n i“1                                   n i“1 jPNn

   Next, using Theorem 1, we can show that

                                       n
                                    1 ÿ´                  ¯2   `     ˘
                                          ϕ̂i pgq ´ ϕi pgq “ op n´1{2 .
                                    n i“1

   And by Assumptions 5 and 8(c), we have, for some universal constant C ą 0, that


                                                            47

                  n                                         n
               1ÿ         ´              ¯2              1ÿ                     ?
                     max ϕ̂j pgq ` ϕj pgq npi, bn q2 ď C       npi, bn q2 “ Op p nq.
               n i“1 jPNn                                n i“1

   Then, we have |σ̂ 2 ´ σ 2 | is op p1q.
   Next, following the proof strategy in Theorem 4 of Leung (2022) [22] , we can establish
that


                                           σ 2 “ σ̂˚2 ` Rn ` op p1q.

   Specifically, this result follows by adapting Leung’s (2022) [22] arguments, where we
replace his term Zi ´ τi ptq with our ϕ˚i pgq, and utilize our Assumptions 8(b)–(d) in place
of his Assumptions 7(b)–(d). Finally, by applying Proposition 4.1 from Kojevnikov et al.
                                                    p
(2021) [20] , we can establish that |σ̂˚2 ´ σn2 | Ý
                                                  Ñ 0.
