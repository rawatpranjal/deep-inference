<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2509.24259.pdf
backend: pdftotext
part: 5/7
-->

# Appendix C: Results for Multiple Time Periods with

Appendix C: Results for Multiple Time Periods with
Staggered Treatment
When treatment timing is common across units, extending the framework to multiple time
periods is straightforward. We simply aggregate all pre-treatment periods into one and all
post-treatment periods into another, denoting them as t “ 1 and t “ 2, respectively. In
contrast, when treatment is staggered across units, the situation becomes more complex.
If we are interested in the ATT effect at a specific time t, the conventional approach is to
compare units that receive treatment at time t with those that never receive treatment,
as in Callaway and Sant’Anna (2021) [5] . The main limitation of this method is that units
already treated before time t may affect the potential outcomes of those treated at time t,
thereby compromising the identification of the causal effect.
      We consider a standard staggered DID setting with four groups and four periods. In each
period, one additional group begins treatment, and once treated, a group remains treated.
Only three groups receive treatment, so one group never receives treatment throughout.

                                                 49


<!-- pages: 51-55 -->

Figure 2: Common Staggered DID design

                                                        Period
                                                    1       2        3            4

                                         1


                            Individual
                                         2

                                         3

                                         4

This structure is illustrated in Figure 2.
   When consider a staggered DID design in a networked setting, where treatment propa-
gates not only through direct assignment but also via neighboring exposure. Let there be
four groups of individuals, each indexed by i “ 1, . . . , 4. These groups may be connected
to one another through a known undirected network structure, represented by the adja-
cency matrix A, as illustrated in Figure 3. In this setting, the black nodes indicate the
group that receives treatment in the current period, gray nodes represent groups that have
already been treated in previous periods, and white nodes correspond to groups that have
not yet received any treatment. For illustration, We consider the exposure mapping Ti as
                                                    ˜                     ¸
                                                           n
                                                           ÿ
                                             Ti “   Di ,         Aij Dj       .
                                                           j“1

   Under this design, the parallel trends assumption boils down to comparing treated and
untreated groups that share the same number of treated neighbors. At t “ 2, the newly
treated Group 1 has zero treated neighbors; among the three still-untreated groups, only
Group 4 likewise has no treated neighbors, making Groups 1 and 4 the valid comparison
pair. The same logic carries over to t “ 3 and t “ 4: when Groups 2 and 3 receive
treatment, Group 4 remains the only group with an identical count of treated neighbors,
so it continues to serve as the appropriate control for the treated groups in those periods.
   In the simple example above, we merely wanted to show that when spillover effects

                                                           50

                              Figure 3: Network Staggered DID


                         1         2                       1         2


                         3         4                       3         4

                             t“1                               t“2
                         1         2                       1         2


                         3         4                       3         4

                             t“3                               t“4

are present, a staggered DID design must isolate units that satisfy our conditional parallel
trends assumption to achieve causal identification. Refining the search for units that meet
this assumption is the price one pays for pinning down more specific causal effects. For
instance, identifying the direct average treatment effect on the treated.
