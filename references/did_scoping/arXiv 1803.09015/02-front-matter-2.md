<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 1803.09015.pdf
backend: pdftotext
part: 2/7
-->

# Front Matter 2

0.1                                                                           0.1

                  ●            ●                                                                 ●
         0.0                                                                           0.0                 ●
                                         ●                                                                            ●          ●
                                                                                                                                             ●
                                                    ●                                                                                               ●
        −0.1                                                                          −0.1
                                                                 ●         ●

        −0.2                                                                          −0.2

                2002         2003       2004       2005         2006   2007                     2002      2003       2004       2005        2006   2007

               Group 2006                                                                    Group 2006

         0.2                                                                           0.2

         0.1                                                                           0.1
                               ●                                                                           ●
                                                                                                                      ●          ●
                                         ●          ●
         0.0                                                                           0.0       ●                                           ●
                  ●                                              ●
                                                                                                                                                    ●
                                                                           ●
        −0.1                                                                          −0.1

        −0.2                                                                          −0.2

                2002         2003       2004       2005         2006   2007                     2002      2003       2004       2005        2006   2007

               Group 2007                                                                    Group 2007

         0.2                                                                           0.2

         0.1                                                                           0.1

                               ●         ●                                                                 ●          ●
         0.0                                        ●                                  0.0                                       ●
                  ●                                                        ●                     ●                                                  ●
                                                                 ●                                                                           ●

        −0.1                                                                          −0.1

        −0.2                                                                          −0.2

                2002         2003       2004       2005         2006   2007                     2002      2003       2004       2005        2006   2007

    Notes: The effect of the minimum wage on teen employment estimated under the unconditional parallel trends assumption
    (Panel (a)) and the conditional parallel trends assumption (Panel (b)). Red lines give point estimates and uniform 95%
    confidence bands for pre-treatment periods allowing for clustering at the county level. Under the null hypothesis of the
    parallel trends assumption holding in all periods, these should be equal to 0. Blue lines provide point estimates and uniform
    95% confidence bands for the treatment effect of increasing the minimum wage allowing for clustering at the county level.
    The top row includes states that increased their minimum wage in 2004, the middle row includes states that increased their
    minimum wage in 2006, and the bottom row includes states that increased their minimum wage in 2007. The estimates in
    Panel (b) use the the doubly robust estimator discussed in the text.

increasing the minimum wage on teen employment appears to be negative and increasing in magnitude
the longer states are exposed to the higher minimum wage. In particular, in the first year that a state
increases its minimum wage, teen employment is estimated to decrease by 2.7%, in the second year it
is estimated to decrease by 7.1%, in the third year by 12.5%, and in the fourth year by 13.6%. Notice
that the last two dynamic treatment effect estimates are exactly the same as the estimates coming from
Illinois alone because Illinois is the only state that is treated for at least two years. These results are
robust to keeping the composition of groups constant by “balancing” the groups across different lengths
of exposure to the treatment (see the row in Table 3 labeled ‘Event Study w/ Balanced Groups’). When
we restrict the sample to only include groups that had a minimum wage increase for at least one full
year (i.e., we keep groups 2004 and 2006 but not 2007), we estimate that the effect of increasing the
minimum wage on impact is 2.7% lower teen employment and 7.1% lower teen employment one year after

                                                                                 29

the increase.17
                        Table 3: Minimum Wage Aggregated Treatment Effect Estimates

               (a) Unconditional Parallel Trends
                                                           Partially Aggregated                      Single Parameters

               TWFE                                                                                         -0.037
                                                                                                           (0.006)
               Simple Weighted Average                                                                      -0.052
                                                                                                           (0.006)
               Group-Specific Effects           g=2004      g=2006      g=2007
                                                 -0.091      -0.047      -0.028                             -0.039
                                                (0.019)     (0.008)     (0.007)                            (0.007)
               Event Study                        e=0         e=1         e=2          e=3
                                                 -0.027      -0.071      -0.125       -0.136                -0.090
                                                (0.006)     (0.009)     (0.021)      (0.023)               (0.013)
               Calendar Time Effects            t=2004      t=2005      t=2006       t=2007
                                                 -0.034      -0.071      -0.055       -0.050                -0.052
                                                (0.019)      (0.02)     (0.009)      (0.006)               (0.013)
               Event Study                        e=0         e=1
               w/ Balanced Groups                -0.027      -0.071                                         -0.049
                                                (0.009)     (0.009)                                        (0.008)

               (b) Conditional Parallel Trends
                                                           Partially Aggregated                      Single Parameters

               TWFE                                                                                         -0.008
                                                                                                           (0.006)
               Simple Weighted Average                                                                      -0.033
                                                                                                           (0.007)
               Group-Specific Effects           g=2004      g=2006      g=2007
                                                 -0.044      -0.029      -0.029                             -0.031
                                                (0.020)     (0.008)     (0.008)                            (0.007)
               Event Study                        e=0         e=1         e=2          e=3
                                                 -0.024      -0.041      -0.050       -0.071                -0.046
                                                (0.006)     (0.009)     (0.022)      (0.026)               (0.013)
               Calendar Time Effects            t=2004      t=2005      t=2006       t=2007
                                                 -0.030      -0.025      -0.030       -0.049                -0.033
                                                (0.022)     (0.021)     (0.009)      (0.007)               (0.012)
               Event Study                        e=0         e=1
               w/ Balanced Groups                -0.016      -0.041                                         -0.028
                                                (0.010)     (0.009)                                        (0.008)
    Notes: The table reports aggregated treatment effect parameters under the unconditional and conditional parallel trends
   assumptions and with clustering at the county level. The row ‘TWFE’ reports the coefficient on a post-treatment dummy
   variable from a two-way fixed effects regression. The row ‘Simple Weighted Average’ reports the weighted average (by group
   size) of all available group-time average treatment effects as in Equation (3.10). The row ‘Group-Specific Effects’ summarizes
   average treatment effects by the timing of the minimum wage increase; here, g indexes the year that a county is first treated. The
   row ‘Event Study’ reports average treatment effects by the length of exposure to the minimum wage increase; here, e indexes the
   length of exposure to the treatment. The row ‘Calendar Time Effects’ reports average treatment effects by year; here, t indexes
   the year. The row ‘Event Study w/ Balanced Groups’ reports average treatment effects by length of exposure using a fixed set
   of groups at all lengths of exposure; here, e indexes the length of exposure and the sample consists of counties that have at least
   two years of exposure to minimum wage increases. The column ‘Single Parameters’ represents a further aggregation of each type
   of parameter, as discussed in the text. The estimates in Panel (b) use the the doubly robust estimator discussed in the text.

  17
      Notice that these estimates are exactly the same as in the first two periods for the dynamic treatment effect estimates
that do not hold the composition of groups constant across different lengths of exposure. The reason that they are the same
for initial exposure is coincidental as the results holding group composition constant do not include the group first treated
in 2007 (the estimated effect of the minimum wage in 2007 for the group of states first treated in 2007 is 2.76% lower teen
employment which just happens to correspond to the estimated effect for the balanced groups). On the other hand, for the
second period, they correspond by construction because both estimates only include the groups first treated in 2004 and
2006.

                                                                  30

    Our summary parameters aggregated by group and by calendar time are also consistent with the idea
that increasing the minimum wage had a negative effect on county level teen employment relative to what
would have happened in the absence of the minimum wage increase.
    The second set of results comes from using the conditional parallel trends assumption; that is, we
assume only that counties with the same characteristics would follow the same trend in teen employment
in the absence of treatment. The county characteristics that we use are region of the country, county
population, county median income, the fraction of the population that is white, the fraction of the
population with a high school education, and the county’s poverty rate. We use the doubly robust
estimation procedure discussed above. Thus, estimation requires a first step estimation of the generalized
propensity score and outcome regression discussed above. For each generalized propensity score, we
estimate a logit model that includes each county characteristic along with quadratic terms for population
and median income.18 For the outcome regressions, we use the same specification for the covariates.
    Before presenting these results, we note that our doubly robust estimation procedure is not compu-
tationally demanding. Our estimates of group-time average treatment effects in this section (across all
groups and time periods and including our multiplier bootstrap with 1000 iterations) run in 3.0 seconds on
a laptop with a 2.80-GHz Intel i5 processor with 8GB of RAM and without using any parallel processing.
    For comparison’s sake, we first estimate the coefficient on a post-treatment dummy variable in a model
with unit fixed effects and region-year fixed effects. This is very similar to one of the sorts of models
that Dube et al. (2010) finds to eliminate the correlation between the minimum wage and employment.
Like Dube et al. (2010), using this specification, we find that the estimated coefficient is small and not
statistically different from 0. However, one must have in mind that the approach we proposed in this
article is different from the two-way fixed effects regression. In particular, we explicitly identify group-
time average treatment effects for different groups and different times, allowing for arbitrary treatment
effect heterogeneity as long as the conditional parallel trends assumption is satisfied. Thus, our causal
parameters have a clear interpretation. As pointed out by Wooldridge (2005a), Chernozhukov et al.
(2013), de Chaisemartin and D’Haultfœuille (2020), Borusyak and Jaravel (2017), Goodman-Bacon (2019)
and Sloczyński (2018), the same may not be true for two-way fixed effects regressions in the presence of
treatment effect heterogeneity.19
    The results using our approach are available in Panel (b) in Figure 1 and Panel (b) in Table 3.
Interestingly, we find quite different results using our approach than are suggested by the two-way fixed
effects regression approach. In particular, we continue to find evidence that increasing the minimum wage
tended to reduce teen employment. The estimated group-time average treatment effects range from 0.9%
lower teen employment (not statistically different from 0) in 2006 for the group of states first treated
in 2006 to 7.1% lower teen employment in 2007 for states first treated in 2004. Now, 3 of 7 group-time
average treatment effects are statistically significant. The average effect of increasing the minimum wage

   18
      Using the propensity score specification tests proposed by Sant’Anna and Song (2019), we fail to reject the null hy-
pothesis that these models are correctly specified at the usual significance levels.
   19
      Our approach is also different from that of Dube et al. (2010) in several other ways that are worth mentioning. We focus
on teen employment; Dube et al. (2010) considers employment in the restaurant industry. Their most similar specification to
the one mentioned above includes census division-time fixed effects rather than region-time fixed effects though the results
are similar. Finally, our period of analysis is different from theirs; in particular, there are no federal minimum wage changes
over the periods we analyze.

                                                              31

on teen employment across all groups that increased their minimum wage is a 3.1% reduction in teen
employment. This estimate is much different from the TWFE estimate. In addition, the pattern of
dynamic treatment effects where the magnitude of the effect of increasing the minimum wage tends to
increase with length of exposure is the same as in the unconditional case.
    Overall, our results suggest that increasing the minimum wage decreased teen employment relative to
what it would have been without the policy change. However, there are some important limitations of our
application. First, some of the estimates of pseudo group-time average treatment effects in pre-treatment
periods in Figure 1 are significantly different from zero which provides some suggestive evidence against
the parallel trends assumption. Second, as discussed in the Supplementary Appendix, there is some
heterogeneity in the size of the minimum wage increase itself across states which could complicate the
interpretation of our results. Together, these suggest that our results should be interpreted with some
caution. That being said, we think that the key takeaway from the application is that, (implicitly) holding
the main identifying assumptions constant, in a prominent application in economics that has many very
common features (treatment effect heterogeneity, dynamic effects, and staggered treatment adoption) the
choice of estimation method can potentially lead to qualitatively different conclusions.

6    Conclusion
This paper has considered Difference-in-Differences methods in the case where there are more than two
time periods and units can become treated at different points in time – a commonly encountered setup
in empirical work in economics. In this setup, we have proposed group-time average treatment effects,
AT T (g, t), that are the average treatment effect in period t for the group of units first treated in period
g. Unlike the more common approach of including a post-treatment dummy variable in a two-way fixed
effects regression, AT T (g, t) corresponds to a well defined treatment effect parameter. We also showed
that once AT T (g, t) has been obtained for different values of g and t, they can be aggregated into other
parameters to more concisely summarize heterogeneity with respect to some particular dimension of
interest (such as length of exposure to the treatment) or, alternatively, into a single overall treatment
effect parameter. In addition, our approach is suitable (i) for cases where the parallel trends assumption
holds only after conditioning on covariates, (ii) using different comparison groups such as the never-
treated or not-yet-treated, and (iii) when units can anticipate participating in the treatment and may
adjust their behavior before the treatment is implemented. We view such flexibility as an important
component of our proposed methodology.
    We also provided nonparametric identification results leading to outcome regression, inverse proba-
bility weighting, and doubly robust estimands. Given that our nonparametric identification results are
constructive, we proposed to estimate AT T (g, t) using its sample analogue. We established consistency
and asymptotic normality of the proposed estimators, and proved the validity of a powerful, but easy
to implement, multiplier bootstrap procedure to construct simultaneous confidence bands for AT T (g, t).
The computational costs of our approach are generally low, and code for implementing our approach is
available in the R did package.
    Finally, we applied our approach to study the effect of minimum wage increases on teen employment.
We found some evidence that increasing the minimum wage led to reductions in teen employment. More

                                                     32

interestingly though, in some cases we found notable differences between the results coming from our
approach relative to the more common two-way fixed effects approach. These differences suggest that
using an approach that is robust to treatment effect heterogeneity and dynamics should be strongly
considered by applied researchers.
