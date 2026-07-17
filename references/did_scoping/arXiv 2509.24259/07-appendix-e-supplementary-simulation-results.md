<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2509.24259.pdf
backend: pdftotext
part: 7/7
-->

# Appendix E: Supplementary Simulation Results

Appendix E: Supplementary Simulation Results
The simulation in the main text primarily focuses on estimating τ DAT T . Since our main
objective is to demonstrate how applying GNNs to estimate the nuisance function can
effectively mitigate bias introduced by confounder network interference, in this part we
simulate different model configurations, specifically emphasizing the handling of treatment
network interference. We intentionally exclude network confounding via covariates in order


                                                         52

to isolate and better understand the effects of treatment interference alone.
   Specifically, our simulation model is defined as follows. We first generate the network
adjacency matrix A using the same procedure as described in the main paper. The covari-
ates X1 and X2 are independently drawn from a standard normal distribution. A nonlinear
                                        X2
combination is then applied: X “ 1 ` 1`exppX1q
                                               . All random error terms used throughout
the simulation are independently drawn from Np0, 1q as well.
   We simulate a binary treatment indicator Di P t0, 1u through a logistic model. Specifi-
cally, we define:
                                      Vi “ θD,1 ` θD,2 ¨ Xi ` νi ,

                                                    1
and define the probability of treatment as πi “ 1`expp´Viq
                                                           . The treatment variable is then
drawn as Di „ Bernoullipπi q. The parameter vector is specified as: θD “ p0.4, 1.5q.
   Then we define outcomes for the pre-treatment and post-treatment periods as follows
                                                           #                   +
                                                               n
                                                               ÿ
         Ypre,i “ θpre,1 ` θpre,2 ¨ Di ` θpre,3 ¨ Di ¨ 1           Aij Dj ą 0 ` θpre,4 ¨ Xi ` ϵi ,
                                                            j“1


with parameters: θy,pre “ p1, 0, 0, 0.6q, and
                                                            #                  +
                                                                n
                                                                ÿ
       Ypost,i “ θpost,1 ` θpre,2 ¨ Di ` θpost,3 ¨ Di ¨ 1            Aij Dj ą 0 ` θpost,4 ¨ Xi ` µi ,
                                                               j“1

                                                                                     !ř                 )
                                                                                        n
with parameters θy,post “ p0.5, 0.2, 0.2, 0.8q. In this setup, we use 1                 j“1 Aij Dj ą 0      as
the exposure mapping. Under the above model specifications, the true exposure-specific
DATTs are τ DATT p0q “ 0.2 and τ DATT p1q “ 0.4.
   We estimate τ DATT p1q and τ DATT p0q with the proposed DATT estimators which explic-
itly account for treatment spillover effects. We compare with the DR-DID estimator of
Sant’Anna and Zhao (2020) [26] which ignores heterogeneity in spillover exposure. Table
3 shows the estimation results, with standard errors in parenthesis. As can be seen from
Table 3, the Network DR-DID method delivers accurate DATT estimates under various
sample sizes. The traditional DR-DID estimates lie between τ DATT p1q and τ DATT p0q. The
traditional DR-DID estimator conflates treatment effects across different exposure groups.


                                                   53

                     Table 3: Simulation Results for treatment spillover

       n                   500                       1000                      2000
   # treated               196                        391                       792
   τ̂ DAT T p1q          0.39974                    0.39978                  0.39981
                        (0.07094)                  (0.04971)                (0.03490)
   τ̂ DAT T p0q          0.20314                    0.20155                  0.20084
                        (0.06466)                  (0.04582)                (0.03246)
    DR-DID               0.26358                    0.26321                  0.26423
                        (0.05113)                  (0.03609)                (0.02551)


As a result, its estimate essentially averages across treated units in both exposure groups,
making it difficult to interpret the true causal effects when spillovers are present.


                                              54
