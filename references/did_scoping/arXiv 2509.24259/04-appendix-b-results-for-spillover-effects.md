<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2509.24259.pdf
backend: pdftotext
part: 4/7
-->

# Appendix B: Results for Spillover Effects

Appendix B: Results for Spillover Effects
Beyond estimating the direct average treatment effect on the treated, empirical researchers
may also seek to evaluate the spillover average treatment effect on the treated, which is
defined as

                                1 ÿ
             τ SAT T pg; dq “          E rYi2 pd, gq ´ Yi2 pd, 0q | Di “ 1, Gi “ g, X, As .   (36)
                                mn iPM
                                       n


   Identification is relatively straightforward for τ SAT T pg; 1q, since the potential outcomes
Yi2 p1, gq for units who receive treatment under exposure level g are directly observed in
the data. However, the corresponding counterfactual outcomes Yi2 p0, gq for these same
individuals—i.e., what their outcomes would have been under control, given the same
exposure—are not observed. To identify the direct effect of treatment assignment at each
exposure level g, we impose a parallel trends assumption on Yi2 p0, gq, as follows:


                                                        48

Assumption 10 (Network Conditional Parallel Trends for τ SAT T pg; 1q).


          E rYi2 p0, gq | Di “ 1, Gi “ g, X, As ´ E rYi1 p0, gq | Di “ 1, Gi “ g, X, As

        “ E rYi2 p0, gq | Di “ 0, Gi “ g, X, As ´ E rYi1 p0, gq | Di “ 0, Gi “ g, X, As .   (37)


      Consistent with the direct effects framework, the following expressions represent the
doubly robust estimands for the spillover effects:


                  „ˆ                                          ˙                       ȷ
 dr        1 ÿ                      Di 1tGi “ 0up1g pi, X, Aq
δ pg, 1q “           Di 1tGi “ gu ´                             p∆Yi ´ ∆u10 pi, X, Aqq .
           mn iPM                         p10 pi, X, Aq
                     n


      Following the same logic, a doubly robust estimator for δ dr pg, 0q can be constructed. The
asymptotic distribution of the spillover ATT effect estimators can be established similarly,
following almost the same approach as that for the direct ATT effect.
