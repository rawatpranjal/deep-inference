<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1803.09015.pdf
backend: pdftotext
part: 3/7
-->

# Appendix A: Proofs of Main Results

Appendix A: Proofs of Main Results
We provide the proofs of our results in this appendix. Before proceeding, we first state and prove several auxiliary
lemmas that help us to prove our main theorems.
   Let
                                  AT TX (g, t) = E[Yt (g) − Yt (0)|X, Gg = 1].


Lemma A.1. Let Assumptions 1, 2, 3, 4, and 6 hold. Then, for all g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ}
and t ≥ g − δ,
                      AT TX (g, t) = E[Yt − Yg−δ−1 |X, Gg = 1] − E[Yt − Yg−δ−1 |X, C = 1] a.s..

Proof of Lemma A.1: In what follows, take all equalities to hold almost surely (a.s.). Then, we have that

                 AT TX (g, t) = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] − E[Yt (0) − Yg−δ−1 (0) |X, Gg = 1]
                                                                       t−g−δ
                                                                        X
                             = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] −             E[∆Yt−` (0) |X, Gg = 1]
                                                                        `=0
                                                                       t−g−δ
                                                                        X
                             = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] −             E[∆Yt−` (0) |X, C = 1]
                                                                        `=0

                             = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] − E[Yt (0) − Yg−δ−1 (0) |X, C = 1]
                             = E[Yt − Yg−δ−1 |X, Gg = 1] − E[Yt − Yg−δ−1 |X, C = 1

where the first equality follows from adding and subtracting E[Yg−δ−1 (0) |X, Gg = 1], the second equality from
simple algebra, the third equality by Assumption 4, the fourth equality by simple algebra, and the last equality
from (2.1) and Assumption 3. 


Lemma A.2. Let Assumptions 1, 2, 3, 5 and 6 hold. Then, for all g and t such that g ∈ Gδ , t ∈ {2, . . . T − δ}
with g − δ ≤ t < ḡ

                 AT TX (g, t) = E[Yt − Yg−δ−1 |X, Gg = 1] − E[Yt − Yg−δ−1 |X, Dt+δ = 0, Gg = 0] a.s..

Proof of Lemma A.2: The proof follows similar steps as the proof of Lemma A.1. Taking all equalities to hold
almost surely (a.s.), we have that

           AT TX (g, t) = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] − E[Yt (0) − Yg−δ−1 (0) |X, Gg = 1]
                                                                t−g−δ
                                                                 X
                        = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] −           E[∆Yt−` (0) |X, Gg = 1]
                                                                 `=0
                                                                t−g−δ
                                                                 X
                        = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] −           E[∆Yt−` (0) |X, Dt+δ = 0, Gg = 0]
                                                                 `=0


                                                         33

                        = E[Yt (g) − Yg−δ−1 (0) |X, Gg = 1] − E[Yt (0) − Yg−δ−1 (0) |X, Dt+δ = 0, Gg = 0]
                        = E[Yt − Yg−δ−1 |X, Gg = 1] − E[Yt − Yg−δ−1 |X, Dt+δ = 0, Gg = 0]

where the first equality follows from adding and subtracting E[Yg−δ−1 (0) |X, Gg = 1], the second equality from
simple algebra, the third equality by Assumption 5 with s = t + δ, the fourth equality by simple algebra, and the
last equality from (2.1) and Assumption 3. 

    Now, we are ready to proceed with the proofs of our main theorems.

Proof of Theorem 1:
