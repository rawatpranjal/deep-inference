<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2203.13887.pdf
backend: pdftotext
part: 1/2
-->

# Front Matter

<!-- pages: 1-33 -->

Automatic Debiased Machine Learning for Dynamic
                                                 Treatment Effects and General Nested Functionals

                                                       Victor Chernozhukov                Whitney Newey               Rahul Singh


arXiv:2203.13887v5 [econ.EM] 20 Jun 2023
                                                               MIT                            MIT                        MIT
                                                                                   Vasilis Syrgkanis
                                                                                  Microsoft Research

                                                                                      June 22, 2023


                                                                                         Abstract
                                                    We extend the idea of automated debiased machine learning to the dynamic treatment
                                                regime and more generally to nested functionals. We show that the multiply robust formula for
                                                the dynamic treatment regime with discrete treatments can be re-stated in terms of a recursive
                                                Riesz representer characterization of nested mean regressions. We then apply a recursive Riesz
                                                representer estimation learning algorithm that estimates de-biasing corrections without the
                                                need to characterize how the correction terms look like, such as for instance, products of
                                                inverse probability weighting terms, as is done in prior work on doubly robust estimation in
                                                the dynamic regime. Our approach defines a sequence of loss minimization problems, whose
                                                minimizers are the mulitpliers of the de-biasing correction, hence circumventing the need for
                                                solving auxiliary propensity models and directly optimizing for the mean squared error of the
                                                target de-biasing correction. We provide further applications of our approach to estimation of
                                                dynamic discrete choice models and estimation of long-term effects with surrogates.


                                           1    Introduction
                                           Recent progress in the area of causal machine learning has shown how one can automatically de-bias
                                           causal estimands that take the form of a solution to a moment equation which involves nuisance
                                           regression functions [9, 22, 7, 8]. Prominent examples include estimands of the form:

                                                                          θ = E[m(Z; g)], for g(X) := E[Y | X]

                                           encompassing quantities such as the average treatment effect the average policy effect and the
                                           average marginal effect, under suitable conditional exogeneity conditions.
                                           However, all prior work analyzes problems that fall into the static treatment regime setting, i.e.
                                           treatments are given at single period and not over time in a dynamic and adaptive manner. In this
                                           work we present the first automatic debiasing approach for the dynamic treatment regime.


                                                                                              1

The dynamic treatment regime has been well studied in the causal inference and biostatistics liter-
ature with many approaches for doubly robust [17, 18, 11, 12, 25, 23, 2, 26, 16] and multiply robust
[19, 15, 2, 24] estimation. Recent work has also extended this literature to the high-dimensional
regime and to the incorporation of machine learning based regression and propensity estimators
[14, 4, 5, 21]. However, all prior work use explicit de-biasing approaches that analytically charac-
terize the form of the de-biasing term in order to achieve double robustness, such as for instance
products of inverse propensity scores over time.
The key idea behind automatic de-biasing in the static regime, is that the de-biasing term can be
equivalently phrased in terms of the Riesz representer of the linear functional implied by the esti-
mand θ. Hence, de-biasing boils down to estimation of the Riesz representer of a linear functional,
given in an oracle manner and does not require analytic derivation.
We extend the idea of automated debiased machine learning to the dynamic treatment regime
and we show that the multiply robust formula for the dynamic treatment regime with discrete
treatments can be re-stated in terms of a recursive Riesz representer characterization of nested
mean regressions. We then apply a recursive Riesz representer estimation learning algorithm that
estimates de-biasing corrections without the need to characterize how the correction terms look
like, such as for instance, products of inverse probability weighting terms, as is done in prior work
on doubly robust estimation in the dynamic regime.
Our approach defines a sequence of loss minimization problems, whose minimizers are the mulitpliers
of the de-biasing correction, hence circumventing the need for solving auxiliary propensity models
and directly optimizing for the mean squared error of the target de-biasing correction. We also
extend prior work on estimation rates of Riesz representers to account for the estimation error that
stems from the prior steps in the recursive Riesz estimation process, which was not required in prior
work in the static regime.


2     Dynamic Treatment Regime
We consider estimation of treatment effects in the dynamic treatment regime. We assume we have
access to n samples of trajectories

                                   Z := (S1 , T1 , S2 , T2 , . . . , SM , TM , Y ),

with St ∈ St are time-varying confounders and Tt ∈ Tt are treatments over time and Y a final
outcome. For any time t, let S̄t = {S1 , . . . , St } and T̄t = {T1 , . . . , Tt } denote the sequence of the
variables up until time t and similarly, let S t = {St , . . . , SM } and T t = {Tt , . . . , TM }. We will also
denote with s̄t , τ̄t , st , τ t , corresponding realizations of the latter random sequences. Moreover, we
will be denoting with (τ̄t′ , τ t+1 ), the sequences of potential treatment states that follows τ ′ up until
time t and then continues with τ . We let 0 ∈ Tt denote a baseline policy value, which could be
appropriately instantiated based on the context.
For any sequence of treatments τ = (τ1 , . . . , τM ), let Y (τ ) denote the counterfactual outcome under
such a sequence of interventions (sequence of treatment states), equivalently in do-calculus notation
Y | do(T̄M = τ̄M ). Note that with this notation Y ≡ Y (T̄M ) . Under this counterfactual notation,


                                                          2

our target quantity of interest is:
                                                     h       i
                                           θ(τ ) := E Y (τ )


We assume that the data generating process satisfies the following sequential conditional random-
ization assumption:

ASSUMPTION 1 (Sequential Conditional Exogeneity). The data generating process satisfies the
following conditional independence conditions:

                        ∀1 ≤ t ≤ M and ∀τ t ∈ ×M
                                               k=t Tk : Y
                                                          (T̄t−1 ,τ t )
                                                                        ⊥⊥ Tt | St           (dynExog)


Figure 1: Causal diagram describing the causal relationships of the random variables in the time
series.

This condition is for instance satisfied if the data generating process adheres to the causal graph
presented in Figure 1, as can be easily verified from the single-world-intervention graph (SWIG)
in Figure 2. Note that even though we used a Markovian notation and the observational policy
only depend on current state St and the outcome Y only depends on last state SM , one should
really interpret St as the current sufficient statistic of the history up until time t. For instance, St
can contain all prior treatments and prior base states as part of it. For instance, suppose that we
had an observed time series of (X1 , T1 , . . . , XM , Tm , Y ) and we wanted to allow all forward arrows
in the causal graph. Then we could re-define St = (X̄t , T̄t−1 ) and apply our current formulation.
This would lead to identical derivations, modulo this renaming. Thereby our setting is much more
permissive than what one might believe at a first glance and encompasses the general dynamic
treatment regime setting as a special case.
Moreover, we will assume a surrogacy assumption, that under an interventional future treatment
policy, the effect of Tt−1 on future outcomes only goes through St . This is again satisfied if the data
generating process adheres to the causal graph presented in Figure 1, as can be easily verified from
the single-world-intervention graph (SWIG) in Figure 2. In fact, we will only require a conditional
mean-independency assumption.


                                                    3

ASSUMPTION 2 (Sequential Surrogacy). The data generating process satisfies the following
conditional mean-independence conditions:

           ∀1 ≤ t ≤ M and ∀τ t ∈ ×M
                                  k=t Tk : Y
                                             (T̄t−1 ,τ t )
                                                           ⊥⊥mean (Tt−1 , St−1 ) | St         (dynSurr)

We note that if St contains all past treatments and states as a subset, then this assumption is
trivial since Tt−1 , St−1 are deterministic random variables conditional on St and hence are mean
independent with any other random variable.
Finally, we also require a regularity condition of sequential positivity (aka overlap), which states
that the conditional density of treatment is bounded away from zero a.s. To define sequential
positivity, we will denote with π(τt , st ) the marginal densities of the random variables (Tt , St ) and
period t ∈ [1, M ]. Then sequential positivity is defined as:
ASSUMPTION 3 (Sequential Positivity). The density π of the data generating processes satisfy
that: Pr(Tt = τt | St ) > 0, whenever Pr(St ) > 0, for all 1 ≤ t ≤ M .

[VC: The notation [Pr(Tt = τt | St ) > 0] and Pr(St ) > 0 implies treatment and states
are discrete; but indetification argument does not rely on that. Can change to
π(taut , st ) > 0 when pi(st ) > 0?]


Figure 2: Single world intervention diagram from intervening and setting the treatments to τ t from
period t and on-wards.


3    Identification as Nested Regressions
We re-state the identification argument from classical work in the dynamic treatment regime [17,
18, 11, 12, 25, 23, 2, 26, 16], in a manner that will be convenient for our main theorem in the next
section.
Theorem 1 (Non-Parametric Identification). If the data generating processes satisfy Assumption 1
- Assumption 3, then the target quantity θ(τ ) is non-parametrically identified via the following


                                                    4

recursively defined estimands:

                                     θ(τ ) = E[f1 (S1 , τ1 )]
              ∀1 ≤ t < M : ft (St , Tt ) = E [ft+1 (St+1 , τt+1 ) | St , Tt ]             (recursive estimand)
                           fM (SM , TM ) = E [Y | SM , TM ]                                    (base estimand)

Proof. For any st ∈ St and τt ∈ Tt , define:
                                             h                                   i
                           ft (st , τt ) := E Y (T̄t−1 ,τ t ) | St = st , Tt = τt .

Then for any t ≥ 1, we have the recursion:
                      h                                    i
     ft (st , τt ) = E Y (T̄t−1 ,τ t ) | St = st , Tt = τt
                      h                                    i
                   = E Y (T̄t ,τ t+1 ) | St = st , Tt = τt                                        (consistency)
                      h h                                i                    i
                   = E E Y (T̄t ,τ t+1 ) | St+1 , St , Tt | St = st , Tt = τt                       (tower law)
                      h h                         i                   i
                   = E E Y (T̄t ,τ t+1 ) | St+1 | St = st , Tt = τt                                  (dynSurr)
                      h h                                      i                  i
                   = E E Y (T̄t ,τ t+1 ) | St+1 , Tt+1 = τt+1 | St = st , Tt = τt        (dynExog + overlap)
                 = E [ft+1 (St+1 , τt+1 ) | St = st , Tt = τt ]

Moreover, note that:
                                   h                                    i
                  fM (sM , τM ) = E Y (T̄M −1 ,τM ) | SM = sM , TM = τM
                                   h                               i
                                = E Y (T̄M ) | SM = sM , TM = τM                                  (consistency)
                                 = E [Y | SM = sM , TM = τM ]                         (base case identification)

Thus we have that fM (sM , τM ) is identified via the above equation and that by induction, if ft+1
has been identified, then ft is identified in terms of ft+1 , via the recursive equation:

               ft (st , τt ) = E [ft+1 (St+1 , τt+1 ) | St = st , Tt = τt ]           (recursive identification)

Thus ft are identified for any M ≥ t ≥ 1.
Finally, note that:
                                                h       i
                                       θ(τ ) = E Y (τ )
                                                h h            ii
                                             = E E Y (τ ) | S1                                      (tower law)
                                                h h                      ii
                                             = E E Y (τ ) | S1 , T1 = τ1                            (dynExog)
                                             = E [f1 (S1 , τ1 )]

which concludes the proof.

                                                           5

4       Automated Debiasing via Recursive Riesz Representers
We recursively construct a Neyman orthogonal moment for our estimand θ(τ ). In particular, we
recursively apply the Riesz representer theorem to introduce de-biasing terms.
First observe that our estimand is phrased as:

                                         θ(τ ) = E [m0 (Z; f1 )] := E [f1 (S1 , τ1 )]

which is a linear functional of the regression function f1 . Thus we can de-bias the target estimand
with respect to errors in the estimation of f1 by adding a de-biasing term, which will contain
residuals of f1 with the target of the regression. In particular, if we let:

                                                   L1 (g) = E [g(S1 , τ1 )]

be a linear functional, and a1 : S1 × T1 → R, be its Riesz representer function, i.e. the function
that satisfies:

                                             L1 (g) = E [a1 (S1 , T1 )g(S1 , T1 )]

Such a Riesz representer is guaranteed to exist when the functional L1 is Lipschitz continuous in
the L2 (P ) space. This for instance, holds when treatments are discrete and sequential positivity
holds. In any case, throughout this section we will make the abstract assumption that L1 (g) has a
Reisz representer. Then the following is a de-biased moment with respect to f1 :

             θ(τ ) = E [m1 (Z; f1 , a1 , f2 )] := E [f1 (S1 , τ1 ) + a1 (S1 , T1 )(f2 (S2 , τ2 ) − f1 (S1 , T1 ))]

This moment now contains f2 and is not orthogonal with respect to f2 . However, if we look at the
linear functional:

              L2 (g) := E [m1 (Z; f1 , a1 , g)] = E [f1 (S1 , τ1 ) + a1 (S1 , T1 )(g(S2 , τ2 ) − f1 (S1 , T1 ))]

Note that this functional is linear, and at g = 0, it takes value

                     L2 (0) = E[f1 (S1 , τ1 ) − a1 (S1 , T1 )f1 (S1 , T1 )] = 0,                      (by definition of a1 )

Thus under similar conditions as for L1 (g), it also has an inner product representation and hence
a corresponding Riesz representer a2 : S2 × T2 → R. Moreover, note that the Riesz representer of
L2 is the same as the Riesz representer of the simpler functional

                                             L2 (g) = E[a1 (S1 , T1 )g(S2 , τ2 )],

since this is the only part that depends on g.1 Thus we can de-bias m1 with respect to f2 by adding
a similar Riesz based correction term:

    θ(τ ) = E [m2 (Z; f1 , a1 , f2 , a2 , f3 )] := E [m1 (Z; f1 , a1 , f2 ) + a2 (S2 , T2 )(f3 (S3 , τ3 ) − f2 (S2 , T2 )]
    1 Note    here, that since this functional depends on (S1 , S2 , T1 , T2 ), the global RR aglobal
                                                                                                    2      has arguments
(S1 , S2 , T1 , T2 ).     Since g only has arguments (S2 , T2 ), there always exists a minimal RR a2 (S2 , T2 ) =
E[aglobal
   2        (S1 , S2 , T1 , T2 )|S2 , T2 ]. In the remainder of the paper, when we talk about a Riesz representer, we will
be referring to such a minimal one.


                                                                6

Iteratively, we can then define the de-biased moment for every t < M :

   θ(τ ) = E mt (Z; f¯t+1 , āt ) = E mt−1 (Z; f¯t , āt−1 ) + at (St , Tt )(ft+1 (St+1 , τt+1 ) − ft (St , Tt ))
                                                                                                               

where at : St × Tt is the Riesz representer of the linear functional:

                                     Lt (g) := E [at−1 (St−1 , Tt−1 )g(St , τt )] .

Moreover, for t = M , we have that:

                θ(τ ) = E mM −1 (Z; f¯M , āM −1 ) + aM (SM , TM )(Y − fM (SM , TM ))
                                                                                    

which concludes our iterative construction, since no further nuisance components are introduced in
this final step.
Thus in the end, if we follow the notational convention of fM +1 (SM +1 , τM +1 ) := Y , we have that
an overall de-biased moment is of the form:
                         "                M
                                                                                                #
                                          X
                θ(τ ) = E f1 (S1 , τ1 ) +   at (St , Tt ) (ft+1 (St+1 , τt+1 ) − ft (St , Tt ))
                                               t=1

where each at : St × Tt → R is recursively defined as the Riesz representer of the linear functional:

                                      Lt (g) := E [at−1 (St−1 , Tt−1 )g(St , τt )]

where we set a0 (S0 , T0 ) := 1. This leads to the following theorem.
Theorem 2 (Main Theorem). Suppose Assumptions 1-3 hold. Let fM +1 (SM +1 , τM +1 ) := Y and
a0 (S0 , T0 ) := 1. Then the estimand θ has the debiased representation:
                                     "               M
                                                                                                          #
                                                     X
   θ(τ ) = E mM (Z; f¯M , āM ) := E f1 (S1 , τ1 ) +
                              
                                                       at (St , Tt ) (ft+1 (St+1 , τt+1 ) − ft (St , Tt )) ,
                                                           t=1

where ft are recursively defined in Theorem 1 and at are recursively defined as follows: for all t ≥ 1,
at : St × Tt → R is the Riesz representer of the linear functional:

                                     Lt (g) := E [at−1 (St−1 , Tt−1 )g(St , τt )] .

Then (i) moment mM is Neyman orthogonal with respect to all nuisance functions f¯M and āM ;
(ii) For any alternative values of the nuisance functions f¯M    ∗
                                                                   , ā∗M , we have the following mixed bias
property:

                 θ∗ (τ ) − θ(τ ) := E mM (Z; f¯M∗
                                                   , ā∗M ) − mM (Z; f¯M , āM )
                                                                                

                                     M
                                    X h                                                     i
                                  =    E ãt (St , Tt ) f˜t+1 (St+1 , τt+1 ) − f˜t (St , Tt ) ,
                                         t=1


where ãt := a∗t − at and f˜t := ft∗ − ft . (iii) The latter property implies the double robustness: if for
each t, either ãt = 0 or f˜t+1 = f˜t = 0, then θ∗ (τ ) = θ(τ ).


                                                            7

Proof. Consider the directional derivative with respect to ft , in the direction f˜t evaluated at the
true f¯M , āM :

             ∂ft E mM (Z; f¯M , āM ) [f˜t ] = E[at−1 (St−1 , Tt−1 )f˜t (St , τt ) − at (St , Tt )f˜t (St , Tt )]
                                    


By the definition of the Riesz representer at of functional Lt (g), for g = f˜t , we have:
                         h                                i   h                            i
                       E at−1 (St−1 , Tt−1 )f˜t (St , τt ) = E at (St , Tt )f˜t (St , Tt )

Thus we have that:

                                        ∂ft E mM (Z; f¯M , āM ) [f˜t ] = 0
                                                               


Moreover, the directional derivative with respect to at , in the direction ãt evaluated at the true
f¯M , āM :

   ∂at E mM (Z; f¯M , āM ) [ãt ] = E [ãt (St , Tt ) (ft+1 (St+1 , τt+1 ) − ft (St , Tt ))]
                          

                                  = E [ãt (St , Tt ) (E [ft+1 (St+1 , τt+1 ) | St , Tt ] − ft (St , Tt ))] (tower law)
                                  = 0                                                   (recursive definition of ft )

Hence, we conclude that the moment is Neyman orthogonal.
Moreover, note that the second order directional derivative is zero for any pair (at , ft′ ) such that
t′ ∈
   / {t, t + 1} and also it is zero for any pair (at , at ) and (ft , ft ). Moreover, for any pair (at , ft ) the
second order directional derivative is of the form:
                                                                    h                            i
                      ∂at ,ft E mM (Z; f¯M , āM ) [ãt , f˜t ] = −E ãt (St , Tt )f˜t (St , Tt )
                                                 


and for any pair (at , ft+1 ) it is of the form:
                                                                   h                                i
                  ∂at ,ft+1 E mM (Z; f¯M , āM ) [ãt , f˜t+1 ] = E ãt (St , Tt )f˜t+1 (St , τt+1 )
                                               


Thus by an exact second order functional Taylor expansion we can write for any alternative param-
eter values f¯M
              ∗      ¯∗ , and ãt := a∗ − at and f˜t := f ∗ − ft :
                and âM               t                   t

                                               ¯
                                      h                                        i
                 θ∗ (τ ) − θ(τ ) := E mM (Z; fˆM , ¯âM ) − mM (Z; f¯M , āM )
                                        M
                                        X  h                                                   i
                                   =      E ãt (St , Tt ) f˜t+1 (St+1 , τt+1 ) − f˜t (St , Tt )
                                        t=1


Remark 1 (Clever co-variate adjustment). Note that instead of adding de-biasing corrections to the
moment function, one could also implement the same de-biasing with clever covariate adjustment.
In particular, if one has access to the Riesz representers at , then when running the regression of
ft+1 (St+1 , τt ) on St , Tt to estimate the function ft , we add a partially linear regression component
of the form: g(St , Tt ) + ϵt · at (St , Tt ) for some non-linear function g, then we note that a square


                                                          8

loss minimizer over this function space with an   h un-penalized ϵt , will result in a function      i     estimate
fˆt , which satisfies the first order condition: E at (St , Tt ) (ft+1 (St+1 , τt ) − fˆt (St , Tt )) = 0, which is
exactly the de-biasing correction term associated with ft . Thus if we add such a clever co-variate in
each of these regressions, then the resulting de-biasing terms will be identically zero and we can just
perform plug-in estimation without further de-biasing. Thus an alternative approach to de-biasing is
to first estimate the Riesz representer functions, then run a sequence of nested regressions where at
each regression step we also add the Riesz representer as a co-variate in a partially linear manner.
This is a Riesz representer based analogue of the clever co-variate adjustment introduced in [2].


5     Riesz Loss Based Estimation
To estimate the Riesz representers at of Lt , we will use the Riesz loss based approach introduced in
[8], which we provide here for concreteness. Consider the problem of estimating the Reisz representer
function a0 of a bounded linear operator:

                                    L(g) = E[m(Z; g)] = E[a0 (Z) · g(Z)]

                                                                                                   Pn
Let En [·] denote the empirical expectation over a sample of size n, i.e. En [Z] = n1                 i=1 Zi .   We
consider a loss function based approach:

                                     â = arg min En [a(Z)2 − 2m(Z; a)]
                                            a∈A

for some function space A and Z a random p variable with support Z. Let ∥ · ∥2 denote the ℓ2 norm
of a function of a random input, i.e. ∥a∥2 = E[a(Z)2 ]. We also let ∥ · ∥∞ denote the ℓ∞ norm,
i.e. ∥a∥∞ = maxz∈Z a(z).
Theorem 3 ([8]). Let δn be an upper bound on the critical radius of the function spaces:

                          star(A − a0 ) = {z → γ (a(z) − a0 (z)) : a ∈ A, γ ∈ [0, 1]}
                star(m ◦ A − m ◦ a0 ) = {z → γ (m(z; a) − m(z; a0 )) : a ∈ A, γ ∈ [0, 1]}

Suppose that m satisfies the mean-squared continuity property:
                             p
                               E[(m(Z; a) − m(Z; a′ ))2 ] ≤ κ ∥a − a′ ∥2

and that for all f ∈ star(A − a0 ) and f ∈ star(m ◦ A − m ◦ a0 ), ∥f ∥∞ ≤ 1. Then for some universal
constant C, we have that w.p. 1 − ζ:
                                                                               
                                 2       2      2               2    κ log(1/ζ)
                       ∥â − a0 ∥2 ≤ C δn (1 + κ ) + ∥a∗ − a0 ∥2 +
                                                                          n

where a∗ = arg mina∈A ∥a − a0 ∥2 .


                                                        9

6     Automated Riesz Estimation for Dynamic Effects
We can thus apply Theorem 3 to the dynamic treatment effect setting for automated de-biasing:
for each t = 1, . . . , M

    1. Consider the loss function:

                           Lt,n (at ) = En at (St , Tt )2 − 2ât−1 (St−1 , Tt−1 )at (St , τt )
                                                                                              


    2. Construct ât by minimizing Lt,n over a class At :

                                                 ât = arg min Lt,n (at )
                                                        at ∈At


Note that this approach has the caveat that the loss function Lt,n is not simply the empirical
analogue of the loss function Lt (at ) := E[at (St , Tt )2 − 2at−1 (St−1 , Tt−1 )at (St , τt )], which is the
Riesz loss associated with the linear functional Lt , since we also replace at−1 with ât−1 in the
above equation. Thus we need to be augment Theorem 3 to account for plug-in nuisance errors, of
nuisance quantities that appear in our functional.


6.1     Riesz Loss Based Estimation with Nuisances
Consider the problem of estimating the Reisz representer function a0 of a bounded linear operator:

                               L(g, h0 ) = E[m(Z; g, h0 )] = E[a0 (Z) · g(Z)]
                                                                                                   Pn
Let En [·] to denote the empirical expectation over a sample of size n, i.e. En [Z] = n1            i=1 Zi . We
consider a loss function based approach:

                                   â = arg min En [a(Z)2 − 2m(Z; a, ĥ)]
                                           a∈A

where ĥ is some estimate of h0 , A is a function space, and Z is a random variable with psupport Z.
As before, let ∥ · ∥2 denote the ℓ2 norm of a function of a random input, i.e. ∥a∥2 = E[a(Z)2 ].
We also let ∥ · ∥∞ denote the ℓ∞ norm, i.e. ∥a∥∞ = maxz∈Z a(z).
                                                         q
Theorem 4. Let a∗ = arg mina∈A ∥a − a0 ∥2 . Let δn ≥ log log(n)
                                                              n      be an upper bound on the critical
radius of the function spaces:

                    star(A − a∗ ) = {z → γ (a(z) − a∗ (z)) : a ∈ A, γ ∈ [0, 1]}
 star(m ◦ A ◦ H − m ◦ a∗ ◦ h0 ) = {z → γ (m(z; a, h) − m(z; a∗ , h0 )) : a ∈ A, h ∈ H, γ ∈ [0, 1]}

Suppose that m satisfies the following continuity properties:
                                  p
             ∀h ∈ H, a, a′ ∈ A : E[(m(Z; a, h) − m(Z; a′ , h))2 ] ≤ κ∥a − a′ ∥2
                                  p
               ∀a ∈ A, h ∈ H : E[(m(Z; a, h) − m(Z; a, h0 ))2 ] ≤ κ∥h − h0 ∥2
          ∀a ∈ A, h ∈ H : |E [m(Z; a − a∗ , h) − m(Z; a − a∗ , h0 )] | ≤ κ∥a − a∗ ∥2 ∥h − h0 ∥2


                                                        10

for some κ and that for all f ∈ star(A − a∗ ) and f ∈ star(mq
                                                            ◦ A ◦ H − m ◦ a∗ ◦ h0 ), ∥f ∥∞ ≤ 1. Then
for some universal constants C, c0 , c1 , if we let δ = δn + c0 log(cn1 /ζ) , we have that w.p. 1 − ζ:
                                                                                      
                 ∥â − a0 ∥22 ≤ O δ 2 (1 + κ2 ) + ∥a∗ − a0 ∥22 + (1 + κ2 )∥ĥ − h0 ∥22

Proof. Consider the following notation:

                                     L(a, h) = E[a(Z)2 − 2m(Z; a, h)]
                                   Ln (a, h) = En [a(Z)2 − 2m(Z; a, h)]

Note that:

                                    L(a, h0 ) = E[a(Z)2 − 2a0 (Z)a(Z)]

By the definition of the Reisz representer we have for any a ∈ A:

 L(a, h0 ) − L(a0 , h0 ) = E[a(Z)2 − 2a0 (Z)a(Z)] + E[a0 (Z)2 ] = E[(a(Z) − a0 (Z))2 ] = ∥a − a0 ∥22

Let a∗ = arg mina∈A ∥a − a0 ∥2 and let:

                        ℓ(z; a, h) = a(z)2 − 2m(z, a, h) − (a∗ (z)2 − 2m(z; a∗ , h))

Note that ℓ(Z; a∗ , h0 ) = 0 and that ℓ is 6-Lipschitz with respect to the vector (m(z; a, h), m(z; a∗ , h), a(z)),
since the gradient of the function ℓ with respect to these components is (−2, 2, 2a(z)), which has
an ℓ2 norm bounded by 6 (since |a(z)| ≤ 1).
                                                             q
By Lemma 11 of [10], and by our choice of δ := δn + c0 log(cn1 /ζ) , where δn is an upper bound on
the critical radius of star(A − a0 ) and star(m ◦ A ◦ H − m ◦ a∗ ◦ h0 ) and star(m ◦ a∗ ◦ H − m ◦ a∗ ◦ h0 ),
w.p. 1 − ζ: ∀a ∈ A, h ∈ H

|Ln (a, h) − Ln (a∗ , h) − (L(a, h) − L(a∗ , h))| = |En [ℓ(Z; a, h) − ℓ(Z; a∗ , h0 )] − E [ℓ(Z; a, h) − ℓ(Z; a∗ , h0 )]|
                                                                      p                                      
                                                  ≤ O δ ∥a − a∗ ∥2 + E[(m(Z; a, h) − m(Z; a∗ , h0 ))2 ]
                                                           p                                              
                                                    +O δ         E[(m(Z; a∗ , h) − m(Z; a∗ , h0 ))2 ] + δ 2
                                                                      p                                     
                                                  ≤ O δ ∥a − a∗ ∥2 + E[(m(Z; a, h) − m(Z; a∗ , h))2 ]
                                                           p                                              
                                                    +O δ         E[(m(Z; a∗ , h) − m(Z; a∗ , h0 ))2 ] + δ 2
                                                  = O δ κ (∥a − a∗ ∥2 + ∥h − h0 ∥2 ) + δ 2 =: ϵ1 (a, h)
                                                                                             


Moreover, since â is the minimizer of Ln (a, ĥ) over A and since a∗ ∈ A, we have that:

                                          Ln (â, ĥ) − Ln (a∗ , ĥ) ≤ 0

Combining all the above we have:

                          L(â, ĥ) − L(a∗ , ĥ) ≤ Ln (â, ĥ) − Ln (a∗ , ĥ) + ϵ1 ≤ ϵ1


                                                       11

By Lipschitzness of m with respect to h, we also have that:
                           
      L(â, ĥ) − L(a∗ , ĥ) − (L(â, h0 ) − L(a∗ , h0 )) = 2 E[m(Z; â − a∗ , ĥ) − m(Z; â − a∗ , h0 )]

                                                            ≤ 2 κ∥â − a∗ ∥2 ∥ĥ − h0 ∥2

Finally, by the definition of the Riesz representer:

                 L(â, h0 ) − L(a∗ , h0 ) = L(â, h0 ) − L(a0 , h0 ) + L(a0 , h0 ) − L(a∗ , h0 )
                                         = ∥â − a0 ∥22 − ∥a∗ − a0 ∥22

Hence, combining the above inequalities we have:

   ∥â − a0 ∥22 − ∥a∗ − a0 ∥22 = L(â, h0 ) − L(a∗ , h0 )
                              ≤ L(â, ĥ) − L(a∗ , ĥ) + O (κ∥â − a∗ ∥2 ∥h − h0 ∥2 )
                                                                        
                              ≤ ϵ1 (â, ĥ) + O κ∥â − a∗ ∥2 ∥ĥ − h0 ∥2
                                                                                              
                              = O δ κ ∥â − a∗ ∥2 + ∥ĥ − h0 ∥2 + δ 2 + κ∥â − a∗ ∥2 ∥ĥ − h0 ∥2


We can thus conclude that for some universal constant C:
                                                                                                
∥â − a0 ∥22 ≤ ∥a∗ − a0 ∥22 + C δ κ ∥â − a∗ ∥2 + ∥ĥ − h∗ ∥2 + δ 2 + κ∥â − a∗ ∥2 ∥ĥ − h0 ∥2
                                                                             2                                 
                          2           2 2      1                                     2
             ≤ ∥a∗ − a0 ∥2 + C 4Cδ κ +               ∥â − a∗ ∥2 + ∥ĥ − h∗ ∥2 + δ + κ∥â − a∗ ∥2 ∥ĥ − h0 ∥2
                                              16C
                                                                                                              
                          2           2 2      1           2      1           2    2
             ≤ ∥a∗ − a0 ∥2 + C 4Cδ κ +           ∥â − a∗ ∥2 +      ∥ĥ − h∗ ∥2 + δ + κ∥â − a∗ ∥2 ∥ĥ − h0 ∥2
                                              8C                8C
                                                                                                         
                          2           2 2      1           2      1           2    2           2        2
             ≤ ∥a∗ − a0 ∥2 + C 4Cδ κ +           ∥â − a∗ ∥2 +      ∥ĥ − h0 ∥2 + δ + 4Cκ ∥ĥ − h0 ∥2
                                              4C                8C
               1                                                                    
             ≤ ∥â − a0 ∥22 + O δ 2 (1 + κ2 ) + ∥a∗ − a0 ∥22 + (1 + κ2 )∥ĥ − h0 ∥22
               2
                                                                  p       √         a2       2
where we invoked repeatedly the AM-GM inequality (a·b = a2 /σ· σb2 ≤ 2σ                 + σb2 ). Re-arranging,
yields the desired statement:
                                                                                         
                   ∥â − a0 ∥22 ≤ O δ 2 (1 + κ2 ) + ∥a∗ − a0 ∥22 + (1 + κ2 )∥ĥ − h0 ∥22


6.2    Application to Recursive Riesz Estimation

For every t we can apply Theorem 4 with H = At−1 , A = At and L(g, h) = E [h(St−1 , Tt−1 ) g(St , τt )].
Note that the continuity properties are satisfied if (i) the function classes At that we use for the
Riesz representers are bounded in some finite range [−H, H]; and (ii) Pr(Tt = τt | St ) ≥ λ > 0:


                                                       12

E h(St−1 , Tt−1 )2 (a(St , τt ) − a′ (St , τt ))2 ≤ H 2 E (a(St , τt ) − a′ (St , τt ))2
                                                                                      

                                                     H2 
                                                        E Pr(Tt = τt | St ) (a(St , τt ) − a′ (St , τt ))2
                                                                                                           
                                                   ≤
                                                     λ
                                                     H2 
                                                        E Pr(Tt = τt | St ) E (a(St , τt ) − a′ (St , τt ))2 | St
                                                                                                                 
                                                   ≤
                                                     λ
                                                     H2                                    H2
                                                   ≤    E (a(St , Tt ) − a′ (St , Tt ))2 =      ∥a − a′ ∥22
                                                     λ                                       λ
Similarly:

                     E (h(St−1 , Tt−1 ) − h0 (St−1 , Tt−1 ))2 a(St , τt )2 ≤ H 2 ∥h − h0 ∥22
                                                                         

and

|E [m(Z; a − a∗ , h) − m(Z; a − a∗ , h0 )] | = |E [(h(St−1 , Tt−1 ) − h0 (St−1 , Tt−1 )) (a(St , τt ) − a∗ (St , τt ))] |
                                                         p
                                             ≤ ∥h − h′ ∥2 E [(a(St , τt ) − a∗ (St , τt ))2 ]
                                                         p
                                             ≤ ∥h − h′ ∥2 κE [(a(St , Tt ) − a∗ (St , Tt ))2 ]
                                                1
                                             = √ ∥h − h′ ∥2 ∥a − a∗ ∥2
                                                 λ
where the third line uses the reasoning above.
                                     q
Thus we have that as long as δn ≥ log log(n)n    upper bounds the critical radius of: star(At − at,∗ )
and star(m ◦ At ◦ At−1 − m ◦ at,∗ ◦ at−1,0 ) (with at,∗ = arg mina∈At ∥at − at,0 ∥2 and at,0 is the true
Riesz representer for functional Lt ), then we get the corresponding fast rate. The latter critical
radius can also be upper bounded as a function of the entropy integral of the function class At and
At−1 separately.
Thus we can derive a bound on the rate of convergence of â outlined by the iterative Riesz estimation
process. For simplicity of stating the corollary we will also assume that at,0 ∈ A, though a more
general statement can be made, accounting also for the bias terms ∥at,∗ − at,0 ∥2 = minat ∈A ∥at −
at,0 ∥2 .
Corollary 5. Suppose: (i) correct specification at,0 ∈ At ; (ii) the function spaces At contain
                                       setting satisfies strict positivity, i.e. Pr(Tt = τt | St ) ≥ λ > 0
uniformly bounded functions; (iii) the q
                                               log log(n)
a.s. for some constant λ; (iv) δt,n ≥               n upper bounds the critical radius of star(At − at,0 )
                                                                               q
and star(m ◦ At ◦ At−1 − m ◦ at,0 ◦ at−1,0 ). Then, if we let δt = δt,n + c0 log(cn1 /ζ) for some universal
constants (c0 , c1 ), we have that w.p. 1 − ζ:

                                ∥ât − at,0 ∥22 ≤ O δ 2 + ∥ât−1 − at−1,0 ∥22
                                                                              

Hence, for any constant time horizon M , we have that w.p. 1 − ζ:
                                                                     
                                                        2          2
                            ∀t ∈ [1, M ] : ∥ât − at,0 ∥2 ≤ O max
                                                               ′
                                                                  δt′
                                                                        t ≤t


                                                            13

7    Extension: Nested Linear Moment Functionals
We note that even though throughout we consider a counterfactual static treatment sequence τ ,
all our results naturally extend to any deterministic or randomized policy counterfactual dynamic
policy πt : St−1 → Tt .
Moreover, note that even though we used a Markovian notation where observational policies and
outcomes only depend on current state St , one may interpret St as the current sufficient statistic
of the history up until time t. For instance, St can contain all prior treatments and all prior states.
Therefore our analysis applies to many settings.
Finally, our results also naturally extend to contrasts of counterfactual outcomes or counterfactual
policies, or any other weighted linear combination of counterfactual outcomes. In general, if we can
show that our estimand takes the form of a nested moment equation of the form:

                                       θ = E[m1 (Z; f1 )]
              ∀1 ≤ t < M : ft (St , Tt ) = E [mt+1 (Z; ft+1 ) | St , Tt ]                     (recursive estimand)
                         fM (SM , TM ) = E [Y | SM , TM ]                                          (base estimand)

where mt are linear moments in ft , which is a generalization of the estimand presented in Theorem 1,
then our approach easily extends. The linear functionals in this case take the form: Lt (g) =
E [at−1 (St−1 , Tt−1 ) mt (Z; g)]. Our main setting was a special case where mt (Z; g) = g(St , τt ).
Linear combinations of counterfactual quantities fall into the latter more general category for more
complex moment functions mt that involve evaluation of the function g at multiple treatment
points.


8    Extension: Nested Non-Linear Functionals
We now consider a more general type of estimand, defined as the solution to nested non-linear
moment equations. In particular, we have that θ is defined via the set of equations:

                             E[m1 (Z; θ, f1 )] = 0                                                               (1)
         ∀1 ≤ t < M, ∀j ∈ [dt ] : ft,j (Xt,j ) = E [mt+1,j (Z; ft+1 ) | Xt,j ]                (recursive estimand)
                                fM,j (XM,j ) = E [Yj | XM,j ]                                      (base estimand)

where mt+1 is a sequence of potentially non-linear vector-valued functionals, ft is a dt -dimensional
vector-valued function and Y ∈ Y, Xt ∈ Xt are sub-vectors of the random vector Z.
Theorem 6 (Main Theorem for Non-Linear Functionals). For notational convenience define:
mM +1,j (Z, fM +1 ) := Yj and a0,j (X0 ) := 1 and let:
                                                      dt
                                                    M X
                                                    X
       m∗M (Z; θ, f¯M , āM ) := m1 (Z; θ, f1 ) +             at,j (Xt,j )′ (mt+1,j (Z; ft+1 ) − ft,j (Xt,j ))
                                                    t=1 j=1


                                                       14

where f¯M = {f1 , . . . , fM }, āM = {a1 , . . . , aM }, ft are recursively defined in Equation ?? and at are
recursively defined as follows: for all t ≥ 1 and j ∈ [dt ], at,j : Xt → R is the Riesz representer of
the linear functional, with respect to g:
                                                                                              
                                      dt−1
                                       X                       ∂
                Lt,j (g; ft,j ) := E       at−1,j (Xt−1,j ) mt,j (Z; ft,j + τ g, ft,−j ) τ =0 
                                       j=1
                                                               ∂τ

Then the estimand θ0 can be identified by the moment equation:
                                    E m∗M (Z; θ, f¯M , āM ) = 0
                                                           

Moreover, the moment m∗M is Neyman orthogonal with respect to all nuisance functions f¯M , āM .

Proof. Consider the directional derivative with respect to ft,j , in the direction f˜t,j evaluated at the
true f¯M , āM :
                 ∂ft,j E m∗M (Z; f¯M , āM ) [f˜t,j ] = Lt,j (f˜t,j ; ft,j ) − E[at,j (Xt,j )f˜t,j (Xt,j )]
                                           

By the definition of the Riesz representer at,j of functional Lt,j (g; ft,j ), for g = f˜t,j , we have:
                                                        h                       i
                                Lt,j (f˜t,j ; ft,j ) = E at (Xt,j )′ f˜t (Xt,j )

Thus we have that:
                                    ∂ft,j E m∗M (Z; f¯M , āM ) [f˜t,j ] = 0
                                                              


Moreover, the directional derivative with respect to at,j , in the direction ãt,j evaluated at the true
f¯M , āM :
         ∂at,j E m∗M (Z; f¯M , āM ) [ãt,j ] = E [ãt,j (Xt,j ) (mt+1,j (Z; ft+1 ) − ft,j (Xt,j ))]
                                   

                                         = E [ãt,j (Xt,j )′ (E [mt+1,j (Z; ft+1 ) | Xt,j ] − ft,j (Xt,j ))]
                                         = 0                                        (recursive definition of ft )
Hence, we conclude that the moment is Neyman orthogonal.

Estimation of the recursive Riesz representers can be done in an identical manner as for linear
nested moments. Albeit now we need to do it in a sequential manner where we first estimate the
nested regression estimates fˆt , then calculate the functional derivatives at fˆt , i.e. if we let
                                                     ∂
                                 ∂mt (Z; ft , g) =      mt (Z; ft + τ g) τ =0
                                                     ∂τ
and we let:
                                 Lt (g; ft ) = E [at−1 (Xt−1 )′ ∂mt (Z; ft , g)]
then we will run the Riesz loss approach on the linear functional Lt (g; fˆt ). Assuming that the
moments mt is sufficiently smooth with respect to f , then we can also approximate ∂mt by a finite
difference:
                            ˆ t (Z; ft , g) ≈ mt (Z; ft + ϵg) − mt (Z; ft )
                            ∂m
                                                           ϵ
                 −1                                                          ˆ t (Z; ft , g)| = o(ϵ).
which for ϵ = o(n ) will add negligible extra error, i.e. |∂mt (Z; ft , g) − ∂m


                                                       15

8.1    Application: Dynamic Discrete Choice

Consider a dynamic discrete choice problem where at each period a decision maker chooses either
to renew Yt = 1 or not renew Yt = 0. Conditional on renewal, the next period state is independent
of first period state. We have that:

                                           Rjt := Dj (Xt )′ θ + ϵjt

with D1 (Xt ) = (1, 0, . . . , 0) and D00 (Xt ) = 0 and with ϵjt exogenous random shocks that are mean
zero and identically distributed across time and actions and independently of Xt and with a known
distribution. Let σ(x) = Pr(Yt = 1 | Xt = x) be the probability of renewal and let V (x) denote the
value function. Moreover, let:

                          v1 (x) := θ0 + δE[V (Xt+1 ) | Yt = 1] =: V1
                          v0 (x) := D0 (Xt )′ θ0 + δE[V (Xt+1 ) | Xt = x, Yt = 0]

We assume that the decision maker chooses an action that maximizes the δ-discounted reward. By
the Bellman equation we can write:

                                         Yt = arg max vj (Xt ) + ϵjt
                                                j∈{0,1}

Note that:

  σ(x) := Pr(Yt = 1 | Xt = x) = Pr(ϵ0t − ϵ1t ≤ v1 (Xt ) − v0 (Xt ) | Xt = x) = Λ(v1 (x) − v0 (x))

where Λ is the CDF of shock differences ϵ0t − ϵ1t . We further assume that Λ is invertible and hence
we can write v1 (x) − v0 (x) = Λ−1 (σ(x)). Then note that we can write:
                   ˆ                                    ˆ
         V (x) =       max {vj (x) + ϵj }f (ϵ)dϵ = V1 +    max {vj (x) − v1 (x) + ϵj }f (ϵ)dϵ
                     j∈{0,1}                              j∈{0,1}
                        ˆ
               = V1 + max{ϵ1 , −Λ−1 (σ(x)) + ϵ0 }f (ϵ)dϵ =: V1 + Q(σ(x))

Since f is assumed to be known, the function Q is also known.2 Thus we get that:

E[V (Xt+1 ) | Yt = 1] − E[V (Xt+1 ) | Xt , Yt = 0] = E [Q(σ(Xt+1 )) | Yt = 1] − E[Q(σ(Xt+1 )) | Xt , Yt = 0]

for some known function Q. If we let: q := Pr[Yt = 1]
                                                                                          
                                                             Q(σ(Xt+1 ))(1 − Yt )
             σ0 (x) := E[Q(σ(Xt+1 )) | Xt = x, Yt = 0] = E                        | Xt = x
                                                                     σ(Xt )
                                                                    
                                                      Q(σ(Xt+1 )) Yt
                 σ1 := E [Q(σ(Xt+1 )) | Yt = 1] = E
                                                           q
             D(x) := D1 (x) − D0 (x)
  2 For the case of type I extreme value distributed shocks we have that Λ is the logistic function, i.e.   Λ(v) =
    1
1+exp{−v}
          and that Λ−1 (σ) = log(σ) − log(1 − σ) and Q(σ) = γE − log(1 − σ), where γE is the Euler constant.


                                                       16

Then we can also write:

     Pr(Yt = 1 | Xt = x) = Λ(v1 (x) − v0 (x))
                            = Λ (D(Xt )′ θ + δ (E[V (Xt+1 ) | Yt = 1] − E[V (Xt+1 ) | Xt = x, Yt = 0]))
                            = Λ (D(Xt )′ θ + δ (σ1 − σ0 (x)))

which yields the nested non-linear identifying moment vector for θ:

                 m1 (Z; θ, σ0 , σ1 ) := D(Xt ) Λ′ (u(Xt ; θ, σ0 , σ1 )) (Yt − Λ(u(Xt ; θ, σ0 , σ1 )))
                  u(Xt ; θ, σ0 , σ1 ) := D(Xt )′ θ + δ (σ1 − σ0 (Xt ))


Casting as nested non-linear functionals. Note that this is of the form in the previous section,
where m1 is as defined above, Z := (Xt , Yt , Xt+1 ) and:

              f1 := (σ0 , σ1 )                                 f2 := (σ, q)
    (X1,1 , X1,2 ) := (Xt , ∅)                     (X2,1 , X2,2 ) := (Xt+1 , ∅)              (Y1 , Y2 ) = (Yt , Yt )
                     Q(f2,1 (Xt+1 )) (1 − Yt )                      Q(f2,1 (Xt+1 ))Yt
     m2,1 (Z; f2 ) =                                m2,2 (Z; f2 ) =
                             f2,1 (Xt )                                    f2,2


We note that to apply our general framework to this setting it suffices to have samples of “transition
                         (i)  (i)  (i)
triplets” of the form (Xt , Yt , Xt+1 ) from each unit i. However, having trajectories from each
unit i can always improve efficiency and we can perform averaging across the transition triplets
within unit, not treating them as independent. So we can use the estimator:
                                    n     T
                                 1 X 1 X ∗ (i) (i) (i)
                                             m (Xt , Yt , Xt+1 ; θ, f¯2 , ā2 ) = 0
                                 n i=1 T t=1
            PT
But treat T1 t=1 m∗ (Xi,t , Yi,t Xi,t+1 ) as the moment stemming from a single observation and per-
form asymptotics in n.


9      Extension: Nested Non-Linear IV Functionals
We now consider a more general type of estimand, defined as the solution to nested non-linear
instrumental variable moment equations. In particular, we have that θ is defined via the set of
equations:

                                                              E[m1 (Z; θ, f1 )] = 0                               (2)
         ∀1 ≤ t < M, ∀j ∈ [dt ] : E [ft,j (Xt,j ) − mt+1,j (Z; ft+1 ) | Vt,j ] = 0           (recursive estimand)
                                               E [fM,j (XM,j ) − Yj | VM,j ] = 0                  (base estimand)

where mt+1 is a sequence of potentially non-linear vector-valued functionals, ft is a dt -dimensional
vector-valued function and Yj , Vt,j , Xt,j are sub-vectors of the random vector Z.


                                                         17

Theorem 7 (Main Theorem for Non-Linear Functionals). For notational convenience define:
mM +1,j (Z, fM +1 ) := Yj and µ0,j (X0 ) := 1 and let:
                                                        dt
                                                      M X
                                                      X
        m∗M (Z; θ, f¯M , āM ) := m1 (Z; θ, f1 ) +              µt,j (Vt,j )′ (mt+1,j (Z; ft+1 ) − ft,j (Xt,j ))
                                                      t=1 j=1

where f¯M = {f1 , . . . , fM }, µ̄M = {µ1 , . . . , µM }, ft are recursively defined in Equation ?? and at
are recursively defined as follows: for all t ≥ 1 and j ∈ [dt ], at,j : Xt → R is the Riesz representer
of the linear functional, with respect to g:
                                                                                             
                                      dt−1
                                       X                     ∂
                Lt,j (g; ft,j ) := E      µt−1,j (Vt−1,j ) mt,j (Z; ft,j + τ g, ft,−j ) τ =0 
                                       j=1
                                                            ∂τ

Then µt,j is defined based on at,j as the solution to the following conditional moment equation:

                                      E [µt,j (Vt,j ) − at,j (Xt,j ) | Xt,j ] = 0

Then the estimand θ0 can be identified by the moment equation:

                                    E m∗M (Z; θ, f¯M , āM ) = 0
                                                           

Moreover, the moment m∗M is Neyman orthogonal with respect to all nuisance functions f¯M , āM .

Proof. Consider the directional derivative with respect to ft,j , in the direction f˜t,j evaluated at the
true f¯M , āM :

           ∂ft,j E m∗M (Z; f¯M , µ̄M ) [f˜t,j ] = Lt,j (f˜t,j ; ft,j ) − E[µt,j (Vt,j )f˜t,j (Xt,j )]
                                     

                                                = E[at,j (Xt,j )f˜t,j (Xt,j )] − E[µt,j (Vt,j )f˜t,j (Xt,j )]
                                                = E[E [at,j (Xt,j ) − µt,j (Vt,j ) | Xt,j ] f˜t,j (Xt,j )] = 0

Moreover, the directional derivative with respect to µt,j , in the direction µ̃t,j evaluated at the true
f¯M , µ̄M :

            ∂at,j E m∗M (Z; f¯M , µ̄M ) [ãt,j ] = E [µ̃t,j (Vt,j ) (mt+1,j (Z; ft+1 ) − ft,j (Xt,j ))]
                                      

                                               = E [µ̃t,j (Vt,j )′ E [mt+1,j (Z; ft+1 ) − ft,j (Xt,j ) | Vt,j ]]
                                               = 0                                       (recursive definition of ft )

Hence, we conclude that the moment is Neyman orthogonal.

Note that we can relax the requirement on µt,j , instead of being the solution to the non-parametric
IV problem to simply satisfy the set of un-conditional moment restrictions:

                                sup         E [(µt,j (Vt,j ) − at,j (Xt,j )) f (Xt,j )] = 0
                            f ∈Ft,j −Ft,j

where Ft,j is the function class that we assume ft,j lies in. Thus when the function class Ft,j is simple
enough, then estimating the function µt,j (Vt,j ) is a much simpler problem than non-parametric IV.

                                                           18

10      Extension: Change of Measure
We now consider yet another generalization of the last section, where we can even allow the measure
over which we take expectations in the moments the define the nested nuisances, to change for each
function. This allows us to apply our methodology to cases where there is a co-variate shift between
the distribution that is used to train a regression or IV regression function and the distribution
over which we evaluate the moment that we plug it in, so as to get a target effect estimate. We
will see that a prototypical application of this extension is the case of estimating long-term effects
with surrogates from a combination of historical and short-term data.
In particular, we have that θ is defined via the set of equations:

                                                                 EZ∼D [m1 (Z; θ, f1 )] = 0
     ∀1 ≤ t < M, ∀j ∈ [dt ] : EZ∼Dt,j [ft,j (Xt,j ) − mt+1,j (Z; ft+1 ) | Vt,j ] = 0              (recursive estimand)
                                               EZ∼DM,j [fM,j (XM,j ) − Yj | VM,j ] = 0                 (base estimand)

where mt+1 is a sequence of potentially non-linear vector-valued functionals, ft is a dt -dimensional
vector-valued function and Yj , Vt,j , Xt,j are sub-vectors of the random vector Z.
Theorem 8 (Main Theorem for Non-Linear Functionals). For notational convenience define:
mM +1,j (Z, fM +1 ) := Yj and µ0,j (X0 ) := 1 and let f¯M = {f1 , . . . , fM }, µ̄M = {µ1 , . . . , µM }, ft
are recursively defined in Equation ?? and at are recursively defined as follows: for all t ≥ 1 and
j ∈ [dt ], at,j : Xt → R is the Riesz representer of the linear functional, with respect to g:
                                dt−1                                                               
                                X                                ∂
             Lt,j (g; ft ) :=          EZ∼Dt−1,k µt−1,k (Vt−1,k ) mt,k (Z; ft,j + τ g, ft,−j ) τ =0
                                                                 ∂τ
                                k=1

and with respect to the L2 (Dt,j ) inner-product space. Then µt,j is defined based on at,j as the
solution to the following conditional moment equation:

                                       EZ∼Dt,j [µt,j (Vt,j ) − at,j (Xt,j ) | Xt,j ] = 0

Define the moment:

                                                      X d
                                                      M Xt

M ∗ (θ, f¯M , µ̄M ) := EZ∼D [m1 (Z; θ, f1 )] +                  EZ∼Dt,j [µt,j (Vt,j )′ (mt+1,j (Z; ft+1 ) − ft,j (Xt,j ))]
                                                      t=1 j=1

Then the estimand θ0 can be identified by the moment equation:

                                                    M ∗ (θ, f¯M , āM ) = 0

Moreover, the moment M ∗ is Neyman orthogonal with respect to all nuisance functions f¯M , āM .

Finally, we note that the above theorem requires estimating the Riesz representer of a linear func-
tional L(g) = EZ∼D [m(Z; g)], in the L2 (D′ ) space for some other distribution D′ . Under the
assumption that L(g) is continuous in this inner product space, then such a Riesz representer is
guaranteed to exist. Moreover, the statistical learning approach easily extends to this case.


                                                              19

Note that under by the Riesz definition:

                                           L(g) = EZ∼D′ [a(Z)g(Z)]

Thus we can consider the risk:

                                 R(a) = EZ∼D′ a(Z)2 − 2EZ∼D [m(Z; g)]
                                                  

Note that:

    R(a) − R(a0 ) = EZ∼D′ a(Z)2 − 2EZ∼D [m(Z; a)] − EZ∼D′ a0 (Z)2 + 2EZ∼D [m(Z; a0 )]
                                                                

                  = EZ∼D′ a(Z)2 − 2EZ∼D [m(Z; a)] − EZ∼D′ a0 (Z)2 + 2EZ∼D′ a0 (Z)2
                                                                              

                  = EZ∼D′ a(Z)2 − 2EZ∼D [m(Z; a)] + EZ∼D′ a0 (Z)2
                                                                

                  = EZ∼D′ a(Z)2 − 2EZ∼D′ [a0 (Z) a(Z)] + EZ∼D′ a0 (Z)2
                                                                    

                  = EZ∼D′ (a(Z) − a0 (Z))2
                                          

Thus it is equivalent to minimizing the mean-squared-error of a with respect to a0 , over the distri-
bution D′ .
Moreover, consider the case where the moment is of the form θ − E[m0 (Z; f1 )] for some linear
functional m of f1 and we are in the nested regression case (i.e. Vt,j = Xt,j ); thus at,j = µt,j .
Let θ∗ be the solution to M ∗ (θ; f¯M
                                    ∗
                                      , ā∗M ) and θ̂ be the solution to M ∗ (θ; fˆ¯M , ˆ
                                                                                        āM ), and we define as
      ∗      ˜    ∗   ˆ
ã = a − â, f = f − f , then we can derive an extension to the mixed bias property that:
                dt
              M X
              X                 h                                             i
    ∗
   θ − θ̃ =              EZ∼Dt,j ãt,j (Xt,j ) mt+1,j (Z; f˜t+1 ) − f˜t (Xt,j ) ,
               t=1 j=1

              X d
              M Xt q                                     q                                 q                             
          ≤                EZ∼Dt,j [ãt,j (Xt,j   )2 ]        EZ∼Dt,j [m(Z; f˜t+1 )2 ] +        EZ∼Dt,j [f˜t (Xt,j )2 ]
               t=1 j=1

If we further assume that the moment satisfies the following mean-squared-continuity property:
                                                              dt+1
                                                              X
                         EZ∼Dt,j [m(Z; f˜t+1 )2 ] ≤ L                EZ∼Dt+1,j [f˜t+1,j (Xt+1,j )2 ]
                                                              j=1

Then it suffices to control the mean-squared-errors of each regression function ft,j , with respect
to the distribution of data Dt,j on which it is trained. Similarly, it suffices to control the mean-
squared-errors of the Riesz representers at,j with respect to the distribution Dt,j , which is exactly
what the modified risk R(a) is equivalent to.


10.1     Example: Estimation of Long-Term Effects with Surrogates

Consider the case when we have a short term data set that contains observations of X, T, S, where
X are controls, T is a binary treatment and S is a short-term surrogate of a long-term outcome.
Moreover, we have a long-term data set that contains X, S, Y , where Y is a long-term outcome. We

                                                               20

assume that even in the short-term data setting there is a latent long-term Y , coupled with each
observation, which is not observed. Our goal is to estimate the effect of T on Y .
Assume that Y ⊥ T | S, X (surrogacy) and that for each t ∈ {0, 1} Y (t) ⊥⊥ T | X (conditional
exogeneity). Let Ds denote the distribution of the short-term dataset and Dℓ the distribution
of the long term data set. We will use the short-hand Es , Eℓ for EZ∼Ds , EZ∼Dℓ . Assume that
Es [Y | S, X] = Eℓ [Y | S, X] (invariance). We can then write:

             θ = Es [Y (1) − Y (0) ] = Es [Es [Y (1) − Y (0) | X]]
               = Es [Es [Y (1) | T = 1, X] − Es [Y (0) | T = 0, X]]                 (conditional exogeneity)
               = Es [Es [Y | T = 1, X] − Es [Y | T = 0, X]]
               = Es [Es [Es [Y | S, T = 1, X] | T = 1, X] − Es [Es [Y | S, T = 1, X] | T = 0, X]]
               = Es [Es [Es [Y | S, X] | T = 1, X] − Es [Es [Y | S, X] | T = 0, X]]             (surrogacy)
               = Es [Es [Eℓ [Y | S, X] | T = 1, X] − Es [Eℓ [Y | S, X] | T = 0, X]]             (invariance)

Define as:

                   h(S, X) = Eℓ [Y | S, X]                     g(T, X) = Es [h(S, X) | T, X]

Then we can write:

                                           θ0 = Es [g(1, X) − g(0, X)]

We see that this estimand falls into the extended framework presented in this section with f1 = g,
V1 = X1 = (T, X) and f2 = h, V2 = X2 = (S, X) and linear moments m1 (Z; θ, f1 ) = f1 (1, X) −
f1 (0, X) − θ and m2 (Z; f2 ) = f2 (S, X).
Thus we can apply our automated debiasing framework to arrive at a moment equation of the form:

        Es [g(1, X) − g(0, X) + a1 (T, X) (h(S, X) − g(T, X))] + Eℓ [a2 (S, X)(Y − h(S, X))]

Note that this can be taken to data since the long-term outcome Y appears only in the expectation
in the historical data, where it is observed, and also the regression h is viable in the historical
data. Moreover, the treatment T appears only in the expectation in the short-term data, where it
is observed. In the work of [1] the quantity a2 (S, X), that comes out of our automatic de-biasing
framework, is referred to as the surrogate score and the term Eℓ [a2 (S, X)Y ] is referred to as the
surrogate score representation of the treatment effect.
Moreover, note that this setting falls under the linear moment functional setting that we expanded
at the end of the last section, for which the estimand will also have the multiply robust mixed bias
property. In this setting the formula simplifies to:
                                                                    h                    i
                 θ∗ − θ̃ = Es [ã1 (T, X)(h̃(S, X) − g̃(S, X))] − Eℓ ã2 (S, X) h̃(S, X)

Thus if a1 or a2 are correct, then the bias is 0, or if h, g are correct then the bias is 0. Moreover, if
we let ∥ · ∥s denote the RMSE with respect to Ds and similarly ∥ · ∥ℓ , then it suffices to control the
mean-squared-errors quantities:

                       ∥a1 − a1,0 ∥s (∥h − h0 ∥s + ∥g − g0 ∥s ) + ∥a2 − a2,0 ∥ℓ ∥h − h0 ∥ℓ


                                                        21

                                                  U


                                        D         M         Y

                        Figure 3: Causal graph for front-door identification.


The mean-squared-continuity property referred to in the last section, here boils down to assuming
that ∥h − h0 ∥s ≤ L∥h − h0 ∥ℓ , which is satisfied for instance if the density ratio of S, X in the two
settings is bounded. In that case, it suffices to bound the RMSE of h over the data on which it is
trained, which are drawn from Dℓ . In practice, it would be a beneficial to train h in a manner that
controls the RMSE under both Ds and Dℓ using co-variate shift techniques, since we have samples
of co-variates S, X from both domains.


10.2    Application: Front-Door Criterion

Consider identification under the front door criterion in settings where the data generating process
is governed by the causal graph in Figure 4.
In this setting, we have by the front door identification approach that we can estimate the mean
counterfactual response when we intervene on D, and set it to d, as:
                                          ˆ
                                  (d)
               E[Y (d) ] = E[Y (M ) ] = E[Y (m) | M (d) = m] p(M (d) = m)dm
                           ˆ
                         =    E[Y (m) ] p(M (d) = m | D = d)dm
                           ˆ
                         =    E[E[Y (m) | D]] p(M (d) = m | D = d)dm
                           ˆ
                         =    E[E[Y (m) | M = m, D]] p(M (d) = m | D = d)dm
                           ˆ
                         =    E[E[Y | M = m, D]] p(M = m | D = d)dm
                           ˆ ˆ
                         =       E[Y | M = m, D = d′ ]p(D = d′ ) p(M = m | D = d)dm

Let D′ denote a random variable that is drawn from the marginal distribution of D but is indepen-
dent of M and let M denote a random variable that is drawn from the conditional distribution of
M given D. Moreover, let Y be the random variable drawn from the conditional distribution of Y
given M, D′ , X. Then we can write:

                              θ = E[Y (d) ] = E[E[E[Y | M, D′ ] | D = d]]


                                                  22

                                                    U


                                         D         M           Y


                                                    X

               Figure 4: Causal graph for front-door identification with observables.


which is a nested regression estimand. In particular, we can write:

                                              θ = f (d)
                                         f (D) = E[h(D′ , M ) | D]
                                     h(D, M ) = E[Y | D, M ]

Hence we can construct an automated debiased moment via nested Riesz estimation:

       θ = f (d) + E(X,D′ ,D) [af (D)(h(D′ , M ) − f (D))] + E(D,M,Y ) [ah (D, M )(Y − h(D, M ))]

Note that here we have a change of measure setting, since the regression h is trained on data where
D, M are correlated conditional on X, but then it is applied on data were D, M are independent
conditional on X.
The dataset for training f , can be produced by using a correlated sample pair (D, M ) and then
couple it with an independent sample of the treatment D from the marginal distribution of treat-
ments. For instance, we can achieve that by pairing two empirical samples i, j and then using
the vector (Di , Dj , Mj ) as a sample of (D′ , D, M ) and running a regression of h(Di , Mj ) on Dj , X.
Similarly, the sample for training h is produced by taking a sample from the data generating process
(Di , Mi , Yi ) and regressing Yi on Di , Mi .
In this setting, one could also mathematically characterize the two Riesz representers, which will
be of the form:
                                              1{D = d}
                                     af (D) =
                                              Pr(D = d)
                                                  p(M | D′ )
                                                                   
                                 ah (D, M ) = E              | D, M
                                                  p(M | D)
The implicit Riesz estimation avoids the explicit estimation of the density ratio.
If the causal graph contains observable characteristics X that create correlations among the variables
D, M, Y , then a generalization of the above identification strategy yields that:

                                             θ = E[f (d, X)]
                                    f (D, X) = E[h(D′ , M, X) | D, X]
                                h(D, M, X) = E[Y | D, M, X]


                                                    23

where D′ is a random variable the is drawn independently of M conditional on X from the condi-
tional distribution of D | X. For binary treatment, we can write this as:

                            θ = E[f (d, X)]
                     f (D, X) = E[h(1, M, X)p(X) + h(0, M, X)(1 − p(X)) | D, X]
                       p(X) = E[D | X]
                 h(D, M, X) = E[Y | D, M, X]

This falls again under the class of nested regression functionals. Albeit in this formulation the
moment that defines f , is a non-linear moment with respect to the regressions (h, p). Thus we
would need to apply the non-linear automatic debiasing approach.


11     Inference
So far, we have characterized Neyman orthogonal moment functions for a broad class of recursive
functionals. We have also defined sequential estimators for the recursive nuisance functions. In this
section, we combine these results to prove consistency, asymptotic normality, and semiparametric
efficiency for our estimator of the causal parameter.


11.1     Estimator and confidence interval

The final aspect of our inferential procedure is cross fitting, which is a classic idea in semiparametric
statistics [3, 20, 13]. Our overall procedure is a variant of debiased machine learning [6].
For the inference proof, it improves clarity to adorn "true" values with the superscript ∗, e.g. we
write θ∗ = θ(τ ).
Consider the abstract notation
                                           E[ψ(Z; θ∗ , η ∗ )] = 0
where Z concatenates random variables in the model, θ∗ is the true causal parameter value, and η ∗
is the true nuisance value.
Algorithm 1 (Debiased machine learning). Given a sample (Zi ) (i = 1, ..., n), partition the sample
                                               c
into folds (I(q) ) (q = 1, ..., Q). Denote by I(q) the complement of I(q) .

                                                           c
  1. For each fold q, estimate η̂(q) from observations in I(q) .
                                           PQ P
  2. Estimate θ∗ as the solution to n−1          q=1    i∈I(q) ψ(Wi ; θ̂, η̂(q) ) = 0.


  3. Estimate its 95% confidence interval as CI = θ̂ ± 1.96σ̂n−1/2 , where 1.96 is the 1 − 0.95/2
     quantile of the standard Gaussian and σ̂ 2 is defined below after introducing additional notation.

We will show        √
                      n            d                   p
                        (θ̂ − θ∗ ) → N (0, 1),    σ̂ 2 → σ 2 =⇒ P(θ∗ ∈ CI) → 0.95.
                     σ

                                                       24

11.2     Generic asymptotic result

Recall the recursive definition of ft∗ :

                                      θ∗ = E[m1 (Z; f1∗ )]
              ∀1 ≤ t < M : ft∗ (St , Tt ) = E mt+1 (Z; ft+1∗
                                                                        
                                                             ) | St , Tt                   (recursive estimand)
                           ∗
                          fM (SM , TM ) =      E [Y | SM , TM ]                                 (base estimand)

Recall the recursive definition of a∗t : for all t ≥ 1, a∗t : St × Tt → R is the Riesz representer of the
linear functional:

                                Lt (g) := E a∗t−1 (St−1 , Tt−1 )mt (Z; g)
                                                                         

Theorem 9 (Inference for nested linear functionals). Suppose identification holds, i.e. θ∗ =
E[m1 (Z; f1∗ )]. Consider the recursive definitions of ft∗ and a∗t given above. Assume that

  1. ∥mt (Z; ft )∥2 ≤ κ∥ft ∥2
                                                      2
  2. |θ∗ | < C and E[ mt+1 (Z; ft+1
                                ∗
                                    ) − ft∗ (St , Tt ) |St , Tt ] ≤ σ̄ 2

  3. ∥mt (Z; fˆt )∥p ≤ C, ∥fˆt (St , Tt )∥p ≤ C, and ∥ât (St , Tt )∥∞ ≤ C for some p > 2

  4. ∥f˜t (St , Tt )∥2 = op (1) and ∥ãt (St , Tt )∥2 = op (1)
     √                                                  √
  5. n∥ãt (St , Tt )∥2 ∥f˜t (St , Tt )∥2 = op (1) and n∥ãt (St , Tt )∥2 ∥f˜t+1 (St+1 , Tt+1 )∥2 = op (1)
  6. σ 2 > c where
                                       σ 2 = E {mM (Z; f¯M
                                                         ∗
                                                           , ā∗M ) − θ∗ }2 .
                                                                          


Then for the class of nested linear functionals,
                                 √
                         p         n            d
                      θ̂ → θ∗ ,      (θ̂ − θ0 ) → N (0, 1),        P(θ∗ ∈ CI) → 0.95.
                                  σ

11.3     Proofs

11.3.1    Abstract conditions from previous work

To begin, we quote an abstract theorem whose conditions we will verify. Suppose p > 2 is some
constant, as are 0 < c0 ≤ c1 .
ASSUMPTION 4. Assume that for all n ≥ 3 and P

  1. The moment function is valid
  2. The moment function is affine in θ, i.e.

                                           ψ(z; θ, η) = ψ a (z; η)θ + ψ b (z; η)


                                                        25

  3. The moment function, viewed as a functional of the nuisances, is twice Gateaux differentiable
     with respect to the nuisances, i.e.
                                                            η 7→ E[ψ(Z; θ, η)]

  4. The moment function is Neyman orthogonal
  5. In the affine representation, the mean of the component that multiplies the causal parameter,
     i.e.
                                           J ∗ = E[ψ a (Z; η ∗ )]
        has singular values that are between c0 and c1 .
ASSUMPTION 5. Assume that for all n ≥ 3 and P

  1. Within each fold, w.p. 1 − ∆n , η̂(q) ∈ Tn where η ∗ ∈ Tn and Tn satisfies the following
     conditions
  2. Bounded moments:
                                                          sup ∥ψ(Z; θ∗ , η)∥p ≤ c1
                                                          η∈Tn

                                                          sup ∥ψ a (Z; η)∥p ≤ c1
                                                          η∈Tn


  3. The following rate conditions hold
                              rn = sup | E[ψ a (Z; η)] − E[ψ a (Z; η ∗ )]| ≤ δn
                                        η∈Tn

                              rn′ = sup ∥ψ(Z; θ∗ , η) − ψ(Z; θ∗ , η ∗ )∥2 ≤ δn
                                    η∈Tn
                                        √
                              λ′n =         n       sup        |∂r2 E[ψ(Z; θ∗ , η ∗ + r(η − η ∗ ))]| ≤ δn
                                                r∈(0,1),η∈Tn


  4. The variance of the score is non-degenerate: the eigenvalues of
                                                      E[ψ(Z; θ∗ , η ∗ )ψ(Z; θ∗ , η ∗ )′ ]
        are bounded below by c0
Theorem 10 (Gaussian approximation [6]). Suppose Assumptions 4 and 5 hold, and n−1/2 ≤ δn .
Then      √                     n
            n               1 X (J ∗ )−1
              (θ̂ − θ∗ ) = √       −     ψ(Zi ; θ∗ , η ∗ ) + Op (n−1/2 + rn + rn′ + λ′n )
          σ                  n i=1   σ
where
                               σ 2 = (J ∗ )−1 E[ψ(Z; θ∗ , η ∗ )ψ(Z; θ∗ , η ∗ )′ ]((J ∗ )−1 )′
Theorem 11 (Variance estimation [6] ). Suppose Assumptions 4 and 5 hold, and n−1/2∧(1−2/p) ≤
δn . Consider the variance estimator
                      Q X                                                                     Q
                      X                                                                    1X X a
           Jˆ−1 n−1                ψ(Wi ; θ̂, η̂(q) )ψ(Wi ; θ̂, η̂(q) )′ (Jˆ−1 )′ ,   Jˆ =       ψ (Zi ; η̂(q) )
                      q=1 i∈I(q)
                                                                                           n q=1
                                                                                                i∈I(q)


                                                                   26

Corollary 12 (Confidence intervals [6]). Suppose the conditions of Theorem 11 hold. Then the
confidence interval
                                                     σ̂
                                      CI = θ̂ ± 1.96 √
                                                       n
is uniformly valid, i.e.
                                        sup |P(θ∗ ∈ CI) − 1.96| → 0.
                                       P∈Pn


11.3.2     Matching symbols

So what remains is a verification of the conditions in Assumptions 4 and 5 for the various causal
parameters of interest. We verify these conditions for

   1. dynamic treatment effect
   2. nested linear functionals

In particular, the former are a special case of the latter, so we focus on the latter.
Recall the recursive definition of ft∗ :

                                       θ∗ = E[m1 (Z; f1∗ )]
               ∀1 ≤ t < M : ft∗ (St , Tt ) = E mt+1 (Z; ft+1∗
                                                                         
                                                              ) | St , Tt                         (recursive estimand)
                            ∗
                           fM (SM , TM ) =    E [Y | SM , TM ]                                          (base estimand)

Recall the recursive definition of a∗t : for all t ≥ 1, a∗t : St × Tt → R is the Riesz representer of the
linear functional:

                                Lt (g) := E a∗t−1 (St−1 , Tt−1 )mt (Z; g)
                                                                         


Recall the orthogonal moment function:
                                "              M
                                                                                                 #
                                               X
                   ¯∗   ∗                ∗        ∗                      ∗        ∗
                                                                                             
    θ = E mM (Z; fM , āM ) := E m1 (Z; f1 ) +   at (St , Tt ) mt+1 (Z; ft+1 ) − ft (St , Tt )
                                                           t=1

Clearly, the moment function is
                            (                       M
                                                                                                            )
                                                    X
               ψ(Z; θ, η) = θ −     m1 (Z; f1 ) +         at (St , Tt ) (mt+1 (Z; ft+1 ) − ft (St , Tt ))               (3)
                                                    t=1

where
                                                      (                    M
                                                                                                                                   )
                                                                           X
η = (f¯M , āM ),    a
                    ψ (Z; η) = 1,      b
                                      ψ (Z; η) = − m1 (Z; f1 ) +                 at (St , Tt ) (mt+1 (Z; ft+1 ) − ft (St , Tt ))
                                                                           t=1
                                                                                                                        (4)


                                                           27

11.3.3   New results

Lemma 13 (Verification of Assumption 4). Suppose identification holds, i.e. θ∗ = E[m1 (Z; f1∗ )].
Consider the recursive definitions of ft∗ and a∗t given above. Then the conditions of Assumption 4
hold.

Proof. We verify each condition.

  1. The moment function is valid.
     In this proposition, we take identification as given. Therefore the first and second term of ??
     cancel in expectation when η = η ∗ and θ = θ∗ . The final terms are mean zero by the law of
     iterated expectations when η = η ∗ due to the recursive definition of ft∗ .
  2. The moment function is affine in θ, i.e.

                                        ψ(z; θ, η) = ψ a (z; η)θ + ψ b (z; η)

     We verify this property in (??)
  3. The moment function, viewed as a functional of the nuisances, is twice Gateaux differentiable
     with respect to the nuisances, i.e.

                                                 η 7→ E[ψ(Z; θ, η)]

     The proof is identical to the proof of Theorem 2, replacing ft+1 (St+1 , τt+1 ) with mt+1 (Z; ft+1 ).
  4. The moment function is Neyman orthogonal
     The proof is identical to the proof of Theorem 2, replacing ft+1 (St+1 , τt+1 ) with mt+1 (Z; ft+1 ).

  5. In the affine representation, the mean of the component that multiplies the causal parameter,
     i.e.
                                           J ∗ = E[ψ a (Z; η ∗ )]
     has singular values that are between c0 and c1 .
     By (??), J ∗ = 1, which is bounded above and below.


Lemma 14 (Verification of Assumption 5). Assume that

  1. η̂, η ∗ ∈ Tn
  2. ∥mt (Z; ft )∥2 ≤ κ∥ft ∥2
                                                      2
  3. |θ∗ | < C and E[ mt+1 (Z; ft+1
                                ∗
                                    ) − ft∗ (St , Tt ) |St , Tt ] ≤ σ̄ 2

  4. ∥mt (Z; fˆt )∥p ≤ C, ∥fˆt (St , Tt )∥p ≤ C, and ∥ât (St , Tt )∥∞ ≤ C for some p > 2


                                                      28

  5. ∥f˜t (St , Tt )∥2 = op (1) and ∥ãt (St , Tt )∥2 = op (1)
     √                                                  √
  6. n∥ãt (St , Tt )∥2 ∥f˜t (St , Tt )∥2 = op (1) and n∥ãt (St , Tt )∥2 ∥f˜t+1 (St+1 , Tt+1 )∥2 = op (1)
  7. σ 2 = E[ψ(Z; θ∗ , η ∗ )2 ] > c0

Then the conditions of Assumption 5 hold.

Proof. We verify each condition.

  1. Within each fold, w.p. 1 − ∆n , η̂(q) ∈ Tn where η ∗ ∈ Tn and Tn satisfies the following
     conditions.
     Correct specification of η ∗ is a sufficient condition.

  2. Bounded moments:

                                                      sup ∥ψ(Z; θ∗ , η)∥p ≤ c1
                                                      η∈Tn

                                                      sup ∥ψ a (Z; η)∥p ≤ c1
                                                      η∈Tn

      (a) For the former, write
                                            (                   M
                                                                                                                        )
                                                                X
                     ∗              ∗
            ∥ψ(Z; θ , η)∥p = θ −                m1 (Z; f1 ) +         at (St , Tt ) (mt+1 (Z; ft+1 ) − ft (St , Tt ))
                                                                t=1                                                         p
                                                                   M
                                                                   X
                             ≤ |θ∗ | + ∥m1 (Z; f1 )∥p +                  ∥at (St , Tt )∥∞ ∥mt+1 (Z; ft+1 ) − ft (St , Tt )∥p .
                                                                   t=1

           Finally, note that

                         ∥mt+1 (Z; ft+1 ) − ft (St , Tt )∥p ≤ ∥mt+1 (Z; ft+1 )∥p + ∥ft (St , Tt )∥p .

      (b) The latter is trivial since ψ a (Z; η) = 1.

  3. The following rate conditions hold

                           rn = sup | E[ψ a (Z; η)] − E[ψ a (Z; η ∗ )]| ≤ δn
                                   η∈Tn

                           rn′ = sup ∥ψ(Z; θ∗ , η) − ψ(Z; θ∗ , η ∗ )∥2 ≤ δn
                                 η∈Tn
                                   √
                           λ′n =       n        sup       |∂r2 E[ψ(Z; θ∗ , η ∗ + r(η − η ∗ ))]| ≤ δn
                                           r∈(0,1),η∈Tn


      (a) The first inequality is trivial since ψ a (Z; η) = 1.


                                                              29

(b) For the second inequality, write

      ψ(Z; θ∗ , η) − ψ(Z; θ∗ , η ∗ )
               (                 M
                                                                                    )
                                X
      = θ∗ − m1 (Z; f1 ) +           at (St , Tt ) (mt+1 (Z; ft+1 ) − ft (St , Tt ))
                                     t=1
                    (                M
                                                                                                  )
                                     X
            ∗
                    m1 (Z; f1∗ ) +         a∗t (St , Tt )             ∗
                                                                          ) − ft∗ (St , Tt )
                                                                                              
       −θ +                                                 mt+1 (Z; ft+1
                                     t=1
       = − mM (Z; f¯M , āM ) − mM (Z; f¯M
                                         ∗
                                           , āM ) + mM (Z; f¯M
                                                              ∗
                                                                , āM ) − mM (Z; f¯M
                                                                                   ∗
                                                                                     , ā∗M )
                


    Grouping the initial two terms
                                                                          M
                                                                          X                                                   
    mM (Z; f¯M , āM ) − mM (Z; f¯M
                                  ∗
                                    , āM ) = m1 (Z; f˜1 ) +                    at (St , Tt ) mt+1 (Z; f˜t+1 ) − f˜t (St , Tt )
                                                                          t=1

    Grouping the final two terms
                                                             M
                                                             X
       mM (Z; f¯M
                ∗
                  , āM ) − mM (Z; f¯M
                                     ∗
                                       , ā∗M ) =                                           ∗
                                                                                                ) − ft∗ (St , Tt )
                                                                                                                  
                                                                   ãt (St , Tt ) mt+1 (Z; ft+1
                                                             t=1

    Hence

                           ∥ψ(Z; θ∗ , η) − ψ(Z; θ∗ , η ∗ )∥2
                            ≤ ∥m1 (Z; f˜1 )∥2
                                  M
                                  X                                                      
                              +            at (St , Tt ) mt+1 (Z; f˜t+1 ) − f˜t (St , Tt )
                                                                                                  2
                                  t=1
                                  M
                                  X
                                                                    ∗
                                                                        ) − ft∗ (St , Tt ) 2
                                                                                          
                              +            ãt (St , Tt ) mt+1 (Z; ft+1
                                  t=1

    We focus on each term separately.
      i. In the first term

                                              ∥m1 (Z; f˜1 )∥2 ≤ κ∥f˜1 (S1 , T1 )∥2

     ii. In the second term
                                                                           
                             at (St , Tt ) mt+1 (Z; f˜t+1 ) − f˜t (St , Tt )
                                                                                      2
                           ≤ ∥at (St , Tt )∥∞ ∥mt+1 (Z; f˜t+1 ) − f˜t (St , Tt )∥2
                                                                                     
                           ≤ ∥at (St , Tt )∥∞ ∥mt+1 (Z; f˜t+1 )∥2 + ∥f˜t (St , Tt )∥2
                                                                                          
                           ≤ ∥at (St , Tt )∥∞ κ∥f˜t+1 (St+1 , Tt+1 )∥2 + ∥f˜t (St , Tt )∥2


                                                        30

          iii. In the third term,
                                                           ∗
                                                                                  2
                                  ãt (St , Tt ) mt+1 (Z; ft+1 ) − ft∗ (St , Tt ) 2
                                                              ∗
                                                                                    2
                                = E[ãt (St , Tt )2 mt+1 (Z; ft+1 ) − ft∗ (St , Tt ) ]
                                                                 ∗
                                                                                       2
                                = E[ãt (St , Tt )2 E[ mt+1 (Z; ft+1 ) − ft∗ (St , Tt ) |St , Tt ]]
                                ≤ σ̄ 2 E[ãt (St , Tt )2 ]
                                = σ̄ 2 ∥ãt (St , Tt )∥22

               Hence
                                                        ∗
                                                            ) − ft∗ (St , Tt ) 2 ≤ σ̄∥ãt (St , Tt )∥2
                                                                              
                               ãt (St , Tt ) mt+1 (Z; ft+1
      (c) For the third inequality, recall the second order derivatives from Theorem 2, replacing
          ft+1 (St+1 , τt+1 ) with mt+1 (Z; ft+1 ). The second order directional derivative is zero for
          any pair (at , ft′ ) such that t′ ∈ / {t, t + 1} and also it is zero for any pair (at , at ) and
          (ft , ft ). Moreover, for any pair (at , ft ) the second order directional derivative is of the
          form:
                                                                         h                            i
                           ∂at ,ft E mM (Z; f¯M , āM ) [ãt , f˜t ] = −E ãt (St , Tt )f˜t (St , Tt )
                                                      


          and for any pair (at , ft+1 ) it is of the form:
                                                                        h                              i
                      ∂at ,ft+1 E mM (Z; f¯M
                                           ∗
                                             , ā∗M ) [ãt , f˜t+1 ] = E ãt (St , Tt )mt+1 (Z; f˜t+1 ) .
                                                    


          Therefore it is sufficient to bound, using Cauchy-Schwartz
                       √       h                         i  √
                         n E ãt (St , Tt )f˜t (St , Tt ) ≤ n∥ãt (St , Tt )∥2 ∥f˜t (St , Tt )∥2
                 √       h                               i  √
                   n E ãt (St , Tt )mt+1 (Z; f˜t+1 ) ≤ n∥ãt (St , Tt )∥2 ∥mt+1 (Z; f˜t+1 )∥2
                                                            √
                                                           ≤ nκ∥ãt (St , Tt )∥2 ∥f˜t+1 (St+1 , Tt+1 )∥2 .

  4. The variance of the score is non-degenerate: the eigenvalues of

                                            E[ψ(Z; θ∗ , η ∗ )ψ(Z; θ∗ , η ∗ )′ ].

     are bounded below by c0 .
     In our case, we simply assume

                                            σ 2 = E[ψ(Z; θ∗ , η ∗ )2 ] > c0 .


Proof of Theorem 9. The result immediately follows from Lemmas 13 and 14, which verify the
conditions of Corollary 12.


                                                            31
