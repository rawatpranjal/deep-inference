<!--
source: /Users/pranjal/Code/deep-inference/references/did_scoping/arXiv 2505.20536.pdf
backend: pdftotext
part: 2/3
-->

# Appendix A. A graphical illustration of CoDEAL in four-block design is provided in Figure

Appendix A. A graphical illustration of CoDEAL in four-block design is provided in Figure
1. The complete CoDEAL method in general staggered adoption design is summarized in
Algorithm 2.

                                                 7

2.2.1   CoDEAL in Four-Block Design

   We begin with the four-block design, the simplest form of the staggered adoption and
often regarded as foundational building blocks for more general staggered adoption designs.
The four-block design is a 2 × 2 staggered adoption design, where a subset of N1 units never
receive the treatment while the remaining N2 = N − N1 units are exposed to the irreversible
treatment starting at time t = T1 + 1. Without loss of generality, we rearrange the panel
such that the first N1 units (i = 1, . . . , N1 ) are untreated. We call the time periods t ∈ [T1 ]
as pre-treatment periods and t ∈ [T ]\[T1 ] as post-treatment periods. Let T2 = T − T1 denote
the time duration of the post-treatment periods. The four-block design naturally decomposes
the panel into four blocks:
                                                                               
             YA YB     Y (0) YB (0)                                    YA (0) YB (0)
          Y=       = A                             and    Y(0) =                          (2)
              YC YD     YC (0) YD (1)                                  YC (0)  ?

where YA ∈ RN1 ×T1 , YB ∈ RN1 ×T2 , YC ∈ RN2 ×T1 , and YD ∈ RN2 ×T2 . By construction, YA ,
YB , YC are observed outcomes of the untreated with Yit = Yit (0) for (i, t) ∈ A ∪ B ∪ C,
while YD are observed outcomes of the treated with Yit = Yit (1) for (i, t) ∈ D, leaving a
missing block in Y(0). To estimate τi⋆ , it suffices to impute the missing values of YD (0).
Denote the imputed block as Y            b D (0). The unit-specific ATT can be then estimated by

τbi = T12 t:(i,t)∈D [Yit (1) − Ybit (0)] for i ∈ [N ]\[N1 ].
         P


Modeling Covariate Effects. To account for the potentially nonlinear, heterogeneous,
and time-varying effects brought by covariates, we consider a fully connected DNN for gt⋆ (·).
We construct our model using a fully connected DNN with ReLU activation, due to its strong
empirical performance. We refer to this architecture as a deep ReLU network. Let L ∈ N
denote the network depth and let d = (d1 , . . . , dL+1 ) ∈ NL+1 specify the layer widths. A deep
ReLU network is a function mapping Rd0 to RdL+1 and takes the form


                     h(x) = LL+1 ◦ σ̄L ◦ LL ◦ σ̄L−1 ◦ · · · ◦ L2 ◦ σ̄1 ◦ L1 (x),                (3)


where each Lℓ (z) = Mℓ z + bℓ is an affine transformation with weight matrix Mℓ ∈ Rdℓ ×dℓ−1


                                                 8

Figure 1: A graphical illustration of the proposed CoDEAL in a four-block design. Gray and
orange refer to the untreated and treated blocks, respectively. Inputs are observed outcomes
Y ∈ RN ×T , covariates X ∈ RN ×P , indicator matrix Ω ∈ RN ×T . Output is the estimated
unit-specific ATT τbi .


and bias vector bℓ ∈ Rdℓ , and σ̄ℓ : Rdℓ → Rdℓ applies the ReLU activation function entrywise.
For simplicity, we refer to both Mℓ and bℓ as the weights of the deep ReLU network.

Definition 1 (Deep ReLU network class). For any L ∈ N, d ∈ NL+1 , and constants B, C > 0,
we define the class of deep ReLU networks with depth L, width parameter d, and weights
M = {M1 , ..., ML+1 , b1 , ..., bL+1 } bounded by C as

                      n                                                                     o
Gnn0L+1 (L, d, B, C) = h̃(x) = sgn(h(x))(|h(x)| ∧ B) : h of form (3), ∥Mℓ ∥∞ ≤ C, ∥bℓ ∥∞ ≤ C .


   The covariate effects can be estimated via

                                                           X
                           gbt (·) =      arg min              (Yit − gt (Xi ))2 ,         (4)
                                            1 (L,d,B,C)
                                       gt ∈GP           i:Wit =0


for certain functional class GP1 (L, d, B, C) specified in Appendix B. We define the covariate-
adjusted observed outcomes for unit i ∈ [N ] at time t ∈ [T ] as Yeit = Yit − gbt (Xi ).


                                                       9

Estimating Counterfactuals by Matrix Completion via a Multi-Output AE. With
covariate-adjusted outcomes, it remains to model the nonlinear factor structure ϕ⋆i (F⋆t ) in (1).
Inspired by Xiu and Shen (2024), we consider tackling ϕ⋆i (F⋆t ) using a multi-output AE.
   The architecture of the multi-output autoencoder (AE) consists of a single DNN encoder
and multiple independent DNN decoders. The encoder ρ(·) ∈ GN
                                                           K1
                                                              (L1 , d1 , B, C) maps the
input to a K1 -dimensional bottleneck representation. For each of the N output components,
a distinct DNN decoder is used; specifically, the decoder for the i-th output is given by
          1
ϕi (·) ∈ GK 1
              (L2 , d2 , B, C) for i ∈ [N ]. Consequently, the i-th output of the multi-output AE
for an input x is given by ϕi ◦ ρ(x). We formally define the function class of the multi-output
AE as:

      K1                                  K1                         1
                                                                         (L2 , d2 , B, C), i ∈ [N ] , (5)
           
     GAE := (ϕ1 , . . . , ϕN ) ◦ ρ : ρ ∈ GN  (L1 , d1 , B, C), ϕi ∈ GK 1


where K1 ≥ K. Further details on this function class are provided in Appendix B.
   With the multi-output AE, the encoder ρ⋆ (·) effectively approximates the latent factors,
facilitating the factor-model-based matrix completion. The unit-specific decoders ϕ⋆i (·),
i ∈ [N ], are better equipped to capture the heterogeneity across units, in contrast to a
traditional single-output AE which limits ϕ⋆1 = . . . = ϕ⋆N . DNNs involved in the multi-output
AE are estimated by

                                                                          X                                 2
              (ϕb1 , . . . , ϕbN ) ◦ ρb(·) =         arg min                                e ·t )) − Yeit
                                                                                      ϕi (ρ(Y                     .   (6)
                                                                   K
                                               (ϕ1 ,...,ϕN )◦ρ(·)∈GAE1 (i,t):Wit =0


The counterfactuals YD (0) can then be imputed by Ybit (0) = ϕbi (b e ·t )) + gbt (Xi ) for any
                                                                  ρ(Y
(i, t) ∈ D. The complete steps of imputing counterfactuals in the four-block design are
summarized as a standalone algorithm in Algorithm 1 to facilitate further generalizations to
staggered design.


2.2.2    CoDEAL in Staggered Adoption Design

   The extension to staggered adoption design is a natural generalization of the four-block
design. Under the staggered adoption design, we rearrange the panels by sorting the N


                                                              10

Algorithm 1 Estimating Counterfactuals by Matrix Completion in the Four-Block Design
 1: Input: Observed outcomes Y, covariates X, the estimated covariate effects g
                                                                              bt (·) from
    (4), treatment indicators Ω with a four-block design in (2), latent factor dimension K
 2: Output: Imputed outcome matrix Y     b D (0).
 3: Obtain the covariate-adjusted observed outcomes by Yeit = Yit − gbt (Xi ) for i ∈ [N ], t ∈ [T ].

 4: Fit a multi-output autoencoder using the entries in the untreated part {Y
                                                                            eit (0) : Wit = 0}
    and obtain (ϕb1 , . . . , ϕbN , ρb) following (6)
 5: Estimate the missing elements by Y          bit (0) = ϕbi (b e ·t )) + gbt (Xi ) for any (i, t) ∈ D.
                                                               ρ(Y
 6: return YD (0).
            b

Algorithm 2 CoDEAL: Staggered Adoption Design
 1: Input: Observed outcomes Y, covariates X, treatment indicators Ω, latent factor
    dimension K.
 2: Output: Estimated unit-specific ATT τbi for each treated unit i.
 3: Obtain the estimated covariate effect g          bt (·) by (4)
 4: Initialize Y(0) ∈ R
               b           N ×T
                                    by setting Yit (0) = Yit for the untreated {(i, t) : Wit = 0}.
                                                 b
 5: Extract block partitions {Nξ }1≤ξ≤r and {Tη }1≤η≤r from Ω.
 6: for ξ0 = 2, . . . , r do
 7:   for η0 = r + 2 − ξ0 , . . . , r do
 8:      Construct the four-block submatrix Y(ξ0 ,η0 ) and Ω(ξ0 ,η0 ) according to Equation (7).
 9:      Call Algorithm 1 with inputs Y(ξ0 ,η0 ) , X(ξ0 ) , gbt (·), Ω(ξ0 ,η0 ) , and K to obtain the im-
         puted block Y    bD
                             (ξ0 ,η0 )
                                       (0), and extract the estimated block Yb (ξ ,η ) (0) from Y
                                                                                   0 0
                                                                                                bD
                                                                                                  (ξ0 ,η0 )
                                                                                                            (0).

10:     Update the matrix Y(0)
                            b    with the imputed block Y  b (ξ ,η )(0) .
                                                               0 0
11:   end for
12: end for
13: Compute unit-specific estimated ATT as τbi =                                  t Wit .
                                                 P                               P
                                                   t:Wit =1 [Yit (1) − Yit (0)]/
                                                                          b
14: return Estimated unit-specific ATT τbi .


units based on the time at which they began receiving the treatment in a descending order
(with never-treated units at the top, followed by the latest adopters, and then progressively
earlier adopters towards the bottom). This sorting procedure yields a block structure in the
treatment indicator matrix Ω. By grouping the units based on their adoption timing and
segmenting the overall time periods into relevant intervals, we partition the indicator matrix
Ω (as well as the outcome matrix Y) into submatrices Ω(ξ,η) ∈ RNξ ×Tη and Y(ξ,η) ∈ RNξ ×Tη
for ξ = 1, . . . , r and η = 1, . . . , r, where rξ=1 Nξ = N and rη=1 Tη = T . Examples of block
                                                P               P

partitions in the staggered design are included in Appendix A. Let Itr = {(ξ, η) : Ω(ξ,η) = 1}
denote the index set of the treated blocks. By the construction of these partitions, the indices

                                                      11

in Itr satisfy ξ + η > r + 1.
   To estimate the unit-specific ATT, it suffices to estimate the counterfactuals of Y(ξ,η) for
all (ξ, η) ∈ Itr . The imputation of Y(ξ,η) (0) can be reduced to a four-block design problem
(Yan and Wainwright, 2024), therefore the method proposed in the previous Section 2.2.1 can
be directly applied. Specifically, to estimate Y(ξ0 ,η0 ) (0) for some (ξ0 , η0 ) ∈ Itr , we construct
a four-block data matrix Y(ξ0 ,η0 ) as follows
                                                                                          
                               YA           YB(ξ0 ,η0 )      Y           (0) YB(ξ0 ,η0 ) (0)
                 Y(ξ0 ,η0 ) =  (ξ0 ,η0 )                =  A(ξ0 ,η0 )                           (7)
                                YC(ξ0 ,η0 ) YD(ξ0 ,η0 )      YC(ξ0 ,η0 ) (0) YD(ξ0 ,η0 ) (1)

where A(ξ0 ,η0 ) = {(ξ, η) : 1 ≤ ξ ≤ k1 , 1 ≤ η ≤ k2 }, B(ξ0 ,η0 ) = {(ξ, η) : 1 ≤ ξ ≤ k1 , k2 + 1 ≤
η ≤ η0 }, C(ξ0 ,η0 ) = {(ξ, η) : k1 + 1 ≤ ξ ≤ ξ0 , 1 ≤ j ≤ k2 }, and D(ξ0 ,η0 ) = {(ξ, η) : k1 + 1 ≤ ξ ≤
ξ0 , k2 + 1 ≤ j ≤ η0 } with k1 = r + 1 − η0 and k2 = r + 1 − ξ0 . The associated treatment
indicator matrix Ω(ξ0 ,η0 ) is defined analogously and the associated covariate matrix X(ξ0 )
can be obtained by retaining the units included in Y(ξ0 ,η0 ) . The missing potential outcomes
YD(ξ0 ,η0 ) (0) can be directly estimated using Algorithm 1. By construction, YD(ξ0 ,η0 ) ⊇ Y(ξ0 ,η0 ) .
In the case when YD(ξ0 ,η0 ) ⊃ Y(ξ0 ,η0 ) , we treat all the blocks of YD(ξ0 ,η0 ) as missing and
impute their values, but only extract Y
                                      b (ξ ,η ) (0) from the imputed Y
                                          0 0
                                                                     bD
                                                                        (ξ0 ,η0 )
                                                                                  (0); see Figure 2
for examples.
   With the four-block construction in (7) and the CoDEAL method in four-block design
introduced in Section 2.2.1, we obtain the imputed values of Y(ξ,η) (0) for all (ξ, η) ∈ Itr .
The unit-specific ATT can then be estimated by τbi = t:Wit =1 [Yit (1) − Ybit (0)]/ t Wit for
                                                     P                             P

each treated unit i. The complete algorithm of CoDEAL in the staggered adoption design is
summarized as Algorithm 2.


2.3     Theoretical Properties

Assumption 1. Suppose Xi is supported on [0, 1]P and has a density function f satisfying
supx∈[0,1]P f (x) =: fmax < ∞, where fmax is independent of dimension P . We assume there
exists a constant B > 0 such that EFt⋆ = 0 and ∥Ft⋆ ∥∞ ≤ B holds almost surely. We further
assume that εit is sub-Gaussian with sub-Gaussian norm σϵ2 , and that εit , Ft⋆ , Xi are mutually
independent.

                                                      12

         T1       T2       T3        T4         T5                 T1       T2       T3       T4       T5
   N1   Y(1,1)   Y(1,2)   Y(1,3)    Y(1,4)     Y(1,5)        N1   Y(1,1)   Y(1,2)   Y(1,3)   Y(1,4)   Y(1,5)
   N2   Y(2,1)   Y(2,2)   Y(2,3)    Y(2,4)                   N2   Y(2,1)   Y(1,2)   Y(2,3)   Y(2,4)
   N3   Y(3,1)   Y(3,2)   Y(3,3)     ?                       N3   Y(3,1)   Y(3,2)   Y(3,3)
   N4   Y(4,1)   Y(4,2)                                      N4   Y(4,1)   Y(4,2)
   N5   Y(5,1)                                               N5   Y(5,1)              ?

Figure 2: Examples of constructing a four-block submatrix Y(ξ0 ,η0 ) to estimate Y(ξ0 ,η0 ) (0).
Here, r = 5. The question mark denotes the block of interest, with (ξ0 , η0 ) = (3,4) (Left) and
(5,3) (Right). The associated four blocks are designed by green (YA(ξ0 ,η0 ) ), red (YB(ξ0 ,η0 ) ),
blue (YC(ξ0 ,η0 ) ), and orange (YD(ξ0 ,η0 ) ) as in (7).


Assumption 2. The functions {gt⋆ } are assumed to be (β, C)-Hölder smooth, that is, each
function is ⌊β⌋ times continuously differentiable, and its ⌊β⌋th derivative is Hölder continuous
with exponent β − ⌊β⌋ and constant C. Here, ⌊β⌋ denotes the greatest integer less than β.

Assumption 3 (Pervasiveness). There exist a matrix M ⋆ ∈ RK×N and a function ρ⋆ , defined
on the image of the mapping M ⋆ ϕ⋆ , denoted by M ⋆ ϕ⋆ ([−B, B]K ). Assume that ρ⋆ is also
(β, C)-Hölder smooth, and ∥M ⋆ ∥∞ ≲ n−1 , ∥M ⋆ ∥0 ≍ n for some diverging integer n > 0. We
further assume that
                                     sup       ∥ρ⋆ (M ⋆ ϕ⋆ (x)) − x∥2 ≲ n−1 .                                  (8)
                                   x∈[−B,B]K

   Assumption 1 is a standard assumption in the literature of nonparametric regression. Note
that assuming Xi has compact support is equivalent to assuming that the support is [0, 1]P via
centering and scaling. The smoothness condition is standard in the nonparametric regression
literature (cf. Tsybakov (2003), Chapter 1), as it governs the complexity of the underlying
function class. A formal definition is provided in Appendix B. Assumption 3 generalizes
the classical pervasiveness condition (with n = N ) from linear factor models (Bai, 2003).
It effectively enables approximate recovery of the latent factors from the observed input:
when x denotes the latent factors, condition (8) ensures the existence of a reconstruction
map ρ⋆ ◦ M ⋆ that approximately inverts the encoding ϕ⋆ (x) and recovers x from the noiseless
data. To illustrate, consider the linear factor model setup in Bai (2003), where ϕ⋆ (x) = Ax
with bounded loadings satisfying ∥A∥∞ ≲ 1 and ∥A∥0 ≍ N indicating that each factor
affects nearly all observed variables. Assumption 3 is satisfied by setting M ⋆ = N −1 A⊤ and
ρ⋆ (x) = N (A⊤ A)−1 x.


                                                        13

    The following theorem establishes the convergence of the estimated counterfactuals under
a four-block design. The result can be readily extended to general staggered adoption designs.

Theorem 2.1. Suppose that Assumptions 1-3 hold, and let K1 ≥ K, with K1 defined in (5).
Assume that log(N + T ) = o(n), N1 ≲ N2 and T1 ≍ T2 . Then, under a suitable choice of
parameters for the DNN function class (specified in Appendix B), we have

                        T      N
                  1    X       X                                       2
                                    E Ybit (0) − ϕ⋆i (F⋆t ) − gt⋆ (Xi )
                T2 N2 t=T +1 i=N +1
                         1      1
                         2β
                                                                          2β
                                                                                  
               =OP T − 2β+K + T −1 n + N −1 K1 + n−1 log4 (N T ) + N − 2β+k log5 N .


    Additional theoretical properties, technical conditions and proofs are detailed in Ap-
pendix B.


3        Simulation Studies
    In this section, we use extensive simulation studies to demonstrate the finite-sample
performance of the proposed CoDEAL3 .

3.1        Benchmark Methods

    We consider five benchmark methods, including a single-output AE (Single AE) (detailed
below), neuron-enhanced AE (AEMC-NE) (Fan et al., 2024), matrix completion with
nuclear norm minimization (MC-NNM) (Athey et al., 2021), vertical regression (Vert-Reg)
(Doudchenko and Imbens, 2016), and difference-in-differences (DiD) (Imai and Kim, 2021).
    Additionally, we evaluate the algorithm’s performance in terms of the covariate effect
removal. To this end, we compare the performance of CoDEAL using different methods to
remove the covariate effects, specifically, DNN-based removal (as illustrated in Section 2.2),
and linear-regression(LR)-based removal (detailed below).
   We evaluate the algorithm’s performance using two evaluation metrics: the mean absolute
error M AE = ( (i,t):Wit =1 |Yit (1) − Ybit (0) − τi⋆ |)/( (i,t) Wit ) and the mean squared error
                P                                            P

M SE = ( (i,t):Wit =1 (Yit (1) − Ybit (0) − τi⋆ )2 )/( (i,t) Wit ).
         P                                            P

    3
        Our code for implementing CoDEAL is available in the supplementary material.


                                                    14

Matrix Completion by Single-Output AE (as a benchmark method). Our proposed
CoDEAL utilizes a multi-output autoencoder (as illustrated in Figure 1) to recover nonlinear
latent factors while explicitly modeling the unit-specific heterogeneity through multiple
decoders. The multi-output autoencoder uses a shared encoder network ρ(·) to map each
column Y
       e·t into a K-dimensional latent space, followed by N separate decoder networks

{ϕ1 , . . . , ϕN } to reconstruct each unit’s value. To demonstrate CoDEAL’s ability to account
for the unit-specific heterogeneity, we include an additional benchmark that replaces the
multi-output autoencoder with a single-output autoencoder in the CoDEAL algorithm. This
benchmark is referred to as the Single AE in later discussions. A single-output autoencoder
uses the same encoder ρ(·) but only one decoder ϕ(·) (i.e. ϕ = ϕ1 = · · · = ϕN ) for all units.
By design, the common decoder makes the single-output AE inherently unable to model the
unit-specific heterogeneity effectively. For implementation, DNNs in the single-output AE
are estimated by
                                                   X                               2
                         ϕb ◦ ρb(·) = arg min                      e ·t )) − Yeit
                                                               ϕ(ρ(Y                     .
                                             K
                                    ϕ◦ρ(·)∈GAE1 (i,t):Wit =0


Covariate Effect Removal by Linear Regression (as a benchmark method). Our
proposed CoDEAL uses a DNN to remove the potentially nonlinear, heterogeneous, and
time-varying effects brought by covariates. To evaluate the algorithm’s performance in
terms of the covariate effect removal, we include a linear-regression(LR)-based removal as
a benchmark. The LR-based removal assumes that covariates have a linear impact on the
outcome. With covariates Xi ∈ RP , i ∈ [N ], we denote the effect of covariates on the outcome
by a coefficient matrix H = [H1 , . . . , HT ] ∈ RP ×T , and we estimate Ht ∈ RP separately
for each period. For each pre-treatment time t ∈ {1, . . . , T1 }, we estimate the covariate
effect vector H                                b t = arg minHt PN Yit − X⊤ Ht 2 . For each
              b t ∈ RP using OLS on all units: H
                                                                                
                                                                 i=1        i

post-treatment period t ∈ {T1 + 1, . . . , T }, we use only control units to estimate the covariate
                                                  2
effect: H
        b t = arg minHt P
                          i:Wit =0 Yit − Xi Ht
                                              ⊤
                                                     . With the estimated H  b t , t ∈ [T ], we obtain

the LR-based covariate-adjusted observed outcomes by Yeit = Yit − X⊤
                                                                   i Ht for each i ∈ [N ]
                                                                     b

and t ∈ [T ].


                                                    15

3.2    Data Generating Process and Simulation Configurations

   We perform extensive simulation studies under various linear and nonlinear generating
mechanisms, with different setups of unit numbers N , time periods T , matrix ranks K, and
treatment assignments. Specifically, we consider the following configurations 1-4:

 1. Four-Block Design: (N, T, N1 , T1 , P, K) = (100, 200, 50, 100, 3, 4)

 2. Four-Block Design: (N, T, N1 , T1 , P, K) = (200, 120, 100, 60, 5, 3)

 3. Staggered Adoption Design: (N, T, K) = (100, 120, 4), r = {5, 10}

 4. Staggered Adoption Design: (N, T, K) = (200, 120, 3), r = {5, 10}.

   For the generation of factor effects and covariate effects, we first generate covariates
X = (Xit ) ∈ RN ×P , factors F = (Fit ) ∈ RT ×K , factor loadings Λ = (Λit ) ∈ RN ×K with
            i.i.d.
Xit , Fit , Λit ∼ N (0, 1). With these building blocks, we consider various linear and nonlin-
ear generating mechanisms to generate factor effects ϕ⋆i (F⋆t ) and covariate effects gt⋆ (Xi ).
Specifically, we experiment with 4 variants for generating factor effects and 6 variants for
                                                                                      i.i.d
generating covariate effects, as listed in Table 1. The unit-specific causal effect τi⋆ ∼ N (12, 5)
is designed to be added to treated units in treated periods. The observed outcome matrix
Y = (Yit ) ∈ RN ×T is then obtained by Yit = Yit⋆ + τi⋆ · I{Wit = 1} + εit where Wit are
determined based on the treatment assignments according to the simulation configuration,
                                            i.i.d
and the idiosyncratic error εit follows εit ∼ N (0, 0.52 ).


3.3    Evaluation on the Performance of CoDEAL in the Removal of
       Covariate Effects

   To evaluate the algorithm’s performance in terms of the covariate effect removal, we
compare the performance of CoDEAL using different methods, specifically, DNN (as illustrated
in Section 2.2) and LR (detailed in Section 3.1), to remove the covariate effects. Six scenarios
of covariate effects are considered as detailed in Table 1. We also include an oracle benchmark,
when no covariate effects are present and no covariate-adjusted operations are needed.


                                                    16

    Table 1: Data Generation Processes for Latent Factor Effects and Covariate Effects
         Factor Effect Generation (ϕ⋆i (F⋆t ))                          Covariate Effect Generation (gt⋆ (Xi ))
                                i.i.d.                                                                          i.i.d.
 Linear        0.5C1 Λ⊤    i Ft , C1 ∼ N (0, 1)                Matrix Linear   X⊤i Ut , U = (U1 , . . . , UT ) ∼ NP ×T (1, 1)
                                                                                             i.i.d.
 Sine          2C1 sin(Λ⊤     i Ft )                           Vector Linear   X⊤i U, U ∼ NP (1, 1)
                                                    i.i.d.
 Polynomial                ⊤
               0.2C1 Λi Ft + 0.2C2 F⊤   t Ft , C2 ∼ N (0, 1)   tanh            [tanh(|X⊤    i w t |)]
                                                                                                     1/2
                                                                                                          + bt , wt ∼ NP (0, 1)
                  (2)               (1)  (1)       (2)
 ReLU MLP      Ri ReLU(Ri Ft + bi ) + bi                       poly               ⊤       1/2
                                                                               |Xi wt | + bt
                (2) i.i.d.              (1) i.i.d.                                                           i.i.d.
               bi ∼ N (0, 0.52 ), Ri ∼ Nh1 ×r (0, 0.52 )       log             log(|X⊤   i w t + bt |), bt ∼ N (0, 1)
                 (1)     (2) i.i.d.                                                              (t)     (t)     (t)       (t) (t) i.i.d.
               bi , Ri ∼ Nh1 (0, 0.52 ), h1 = 10               ReLU            ReLU(Xi R1 + b1 )R2 + b2 , b2 ∼ N (0, 1)
                                                                                 (t) i.i.d.               2       (t) i.i.d.
                                                                               R1 ∼ NP ×h2 (0, P ), b1 ∼ Nh2 (0, 1)
                                                                                 (t) i.i.d.
                                                                               R2 ∼ Nh2 (0, h22 ), h2 = 32

   Note: The vector linear covariate effect generation setting mimics a situation in which
 covariate effects are heterogeneous across panel units but remain time-invariant over time.
The matrix linear covariate effect generation setting, however, allows for heterogeneous and
                                time-varying covariate effects.

   Table 2 and Table 3 report the comparisons of MAE and MSE using the two covariate
effect removal methods in Config.1 and Config.2, respectively. Under each configuration,
we consider both linear factor effects and nonlinear Sine factor effects, with all six types of
covariate effects.
   As shown in Tables 2 and 3, LR-based removal and DNN-based removal behave very
differently depending on the underlying true types of covariate effects. In cases where the
covariate effect is strictly linear, LR often outperforms NN by a small margin, indicating
that a simple linear adjustment suffices for purely linear effects. In contrast, whenever the
covariate effects enter nonlinearly (e.g. tanh, log, polynomial, or ReLU), the proposed DNN
covariate removal achieves substantially lower errors, with MAE reduced by 10–30 % and
MSE by similar amounts relative to LR. These results confirm that LR is adequate for
linear covariate effects but that the DNN approach is essential to capture complex, nonlinear
covariate–outcome relationships.

3.4    Evaluation on the Performance of CoDEAL in Causal Matrix
       Completion
   Table 4 reports the comparisons of prediction accuracy between the proposed CoDEAL
and five benchmark methods (detailed in Section 3.1) in Config.1-4. Under each configuration,
we consider all four types of factor effects (listed in Table 1), with linear covariate effects and
nonlinear tanh covariate effects.

                                                               17

Table 2: Comparisons of imputation accuracy (measured by MAE and MSE) with covariate
effects removed by LR and DNN under various covariate generations in Config.1. Results
are averaged over 50 replications. Numbers in parentheses are standard errors.
                                  LR, Linear Factor               NN, Linear Factor               LR, Sine Factor                NN, Sine Factor
 Covariate       Method
                                 MAE            MSE              MAE           MSE               MAE           MSE             MAE           MSE
                 CoDEAL      0.560(0.009)     0.578(0.028)   0.560(0.009)     0.578(0.028)   0.933(0.015)    1.722(0.066)    0.933(0.015)   1.722(0.066)
                 Single AE    0.724(0.009)    1.081(0.034)    0.724(0.009)    1.081(0.034)    1.071(0.013)    2.131(0.056)   1.071(0.013)   2.131(0.056)
                 AEMC-NE      0.734(0.009)    1.142(0.039)    0.734(0.009)    1.142(0.039)    1.090(0.013)    2.172(0.056)   1.090(0.013)   2.172(0.056)
 No covariate
                 DiD          0.804(0.010)    1.364(0.043)    0.804(0.010)    1.364(0.043)    1.141(0.013)    2.343(0.059)   1.141(0.013)   2.343(0.059)
                 Vert-Reg     0.601(0.002)   0.574(0.004)     0.601(0.002)   0.574(0.004)     1.024(0.012)    2.228(0.069)   1.024(0.012)   2.228(0.069)
                 MC-NNM       0.580(0.010)    0.593(0.030)    0.580(0.010)    0.593(0.030)    1.093(0.013)    2.147(0.054)   1.093(0.013)   2.147(0.054)
                 CoDEAL      0.585(0.013)    0.615(0.024)    0.646(0.009)    0.722(0.023)    0.983(0.011)    1.889(0.046)    1.079(0.012)   2.135(0.054)
                 Single AE    0.751(0.008)    1.131(0.032)    0.821(0.010)    1.320(0.039)    1.123(0.014)    2.283(0.060)   1.226(0.013)   2.626(0.062)
                 AEMC-NE      0.779(0.010)    1.244(0.041)    0.838(0.009)    1.394(0.041)    1.145(0.013)    2.336(0.057)   1.209(0.013)   2.564(0.060)
 Matrix Linear
                 DiD          0.837(0.010)    1.444(0.045)    0.892(0.010)    1.587(0.045)    1.187(0.013)    2.490(0.060)   1.244(0.013)   2.696(0.063)
                 Vert-Reg     0.686(0.006)    0.766(0.015)    0.757(0.006)    0.957(0.019)    1.103(0.012)    2.498(0.066)   1.169(0.012)   2.647(0.064)
                 MC-NNM       0.604(0.009)    0.664(0.025)    0.699(0.006)    0.895(0.020)    1.140(0.013)    2.293(0.055)   1.201(0.013)   2.510(0.059)
                 CoDEAL      0.577(0.010)    0.592(0.026)    0.652(0.006)    0.715(0.017)    0.964(0.016)    1.775(0.060)    1.053(0.011)   2.035(0.045)
                 Single AE    0.746(0.007)    1.106(0.026)    0.841(0.008)    1.373(0.030)    1.106(0.012)    2.226(0.054)   1.205(0.012)   2.549(0.053)
                 AEMC-NE      0.778(0.010)    1.205(0.032)    0.823(0.008)    1.317(0.029)    1.135(0.013)    2.298(0.053)   1.191(0.012)   2.497(0.053)
 Vector Linear
                 DiD          0.825(0.008)    1.373(0.031)    0.872(0.009)    1.489(0.033)    1.173(0.012)    2.437(0.053)   1.224(0.012)   2.620(0.056)
                 Vert-Reg     0.669(0.005)    0.727(0.012)    0.733(0.006)    0.887(0.017)    1.085(0.011)    2.413(0.057)   1.145(0.011)   2.520(0.053)
                 MC-NNM       0.591(0.008)    0.617(0.020)    0.678(0.007)    0.826(0.020)    1.106(0.017)    2.219(0.099)   1.175(0.012)   2.412(0.053)
                 CoDEAL       0.778(0.008)    1.168(0.029)   0.701(0.008)    0.871(0.031)    1.142(0.013)     2.366(0.059)   1.080(0.013)   2.175(0.059)
                 Single AE    0.889(0.008)    1.554(0.035)    0.851(0.008)    1.383(0.033)    1.299(0.014)    2.910(0.067)   1.257(0.014)   2.748(0.066)
                 AEMC-NE      0.988(0.008)    1.822(0.036)    0.862(0.009)    1.445(0.036)    1.305(0.013)    2.925(0.064)   1.239(0.014)   2.677(0.065)
 tanh
                 DiD          0.855(0.008)    1.473(0.035)    0.916(0.009)    1.642(0.037)    1.199(0.013)    2.536(0.061)   1.274(0.014)   2.817(0.067)
                 Vert-Reg    0.721(0.005)    0.844(0.015)     0.798(0.007)    1.075(0.024)    1.146(0.012)    2.660(0.071)   1.212(0.013)   2.787(0.072)
                 MC-NNM       0.837(0.007)    1.131(0.021)    0.735(0.005)    1.007(0.017)    1.153(0.013)   2.339(0.056)    1.234(0.013)   2.636(0.063)
                 CoDEAL      1.020(0.006)    1.873(0.026)    0.905(0.010)    1.517(0.030     1.345(0.012)    3.233(0.065)    1.253(0.013)   2.849(0.064)
                 Single AE    1.119(0.006)    2.266(0.031)    1.013(0.008)   1.943(0.033)     1.427(0.012)    3.502(0.065)   1.386(0.013)   3.332(0.068)
                 AEMC-NE      1.137(0.007)    2.386(0.036)    1.024(0.008)   2.007(0.036)     1.421(0.012)    3.483(0.064)   1.372(0.014)   3.281(0.069)
 log
                 DiD          1.127(0.007)    2.371(0.037)    1.069(0.008)   2.182(0.038)     1.408(0.013)    3.437(0.066)   1.401(0.014)   3.402(0.071)
                 Vert-Reg     1.100(0.006)    2.087(0.024)    1.055(0.007)   1.940(0.028)     1.485(0.012)    4.054(0.077)   1.439(0.012)   3.737(0.074)
                 MC-NNM       1.044(0.006)    1.998(0.028)    0.939(0.005)   1.649(0.020)     1.373(0.012)    3.266(0.061)   1.366(0.013)   3.233(0.067)
                 CoDEAL       0.833(0.007)    1.290(0.028)   0.715(0.007)    0.913(0.035)    1.182(0.013)    2.507(0.059)    1.024(0.013)   2.086(0.062)
                 Single AE    0.946(0.008)    1.704(0.036)    0.854(0.008)    1.395(0.033)    1.345(0.014)    3.088(0.068)   1.261(0.014)   2.760(0.066)
                 AEMC-NE      1.060(0.008)    2.038(0.037)    0.868(0.008)    1.461(0.036)    1.348(0.012)    3.088(0.063)   1.245(0.014)   2.701(0.065)
 poly
                 DiD          0.906(0.008)    1.596(0.036)    0.921(0.009)    1.653(0.037)    1.235(0.013)    2.659(0.062)   1.279(0.014)   2.832(0.067)
                 Vert-Reg    0.756(0.006)    0.934(0.017)     0.800(0.007)    1.078(0.023)    1.281(0.012)    2.983(0.072)   1.216(0.013    2.790(0.070)
                 MC-NNM       0.847(0.006)    1.305(0.032)    0.747(0.005)    1.014(0.017)    1.290(0.013)    2.763(0.057)   1.238(0.013)   2.652(0.063)
                 CoDEAL       0.946(0.009)   1.702(0.042)    0.717(0.008)    0.916(0.025)     1.269(0.013)    2.817(0.064)   1.121(0.012)   2.269(0.054)
                 Single AE    1.020(0.010)   1.971(0.046)     0.858(0.010)    1.407(0.039)    1.329(0.013)    3.044(0.064)   1.222(0.014)   2.617(0.062)
                 AEMC-NE      1.288(0.011)   2.982(0.053)     0.868(0.010)    1.471(0.043)    1.539(0.014)    4.010(0.075)   1.235(0.013)   2.660(0.062)
 ReLU
                 DiD          0.960(0.009)   1.777(0.046)     0.922(0.010)    1.670(0.047)    1.275(0.013)    2.819(0.062)   1.269(0.014)   2.791(0.064)
                 Vert-Reg    0.779(0.006)    1.013(0.020      0.782(0.006)    1.027(0.021)   1.204(0.012)     2.846(0.069)   1.199(0.012)   2.753(0.067)
                 MC-NNM       0.967(0.009)   1.893(0.035)     0.764(0.009)    1.073(0.024)    1.235(0.012)   2.639(0.057)    1.227(0.013)   2.608(0.060)


                                                                         18

Table 3: Comparisons of imputation accuracy (measured by MAE and MSE) with covariate
effects removed by LR and DNN under various covariate generations in Config.2. Results
are averaged over 50 replications. Numbers in parentheses are standard errors.
                                  LR, Linear Factor               NN, Linear Factor               LR, Sine Factor                NN, Sine Factor
 Covariate       Method
                                 MAE            MSE              MAE           MSE               MAE           MSE             MAE           MSE
                 CoDEAL      0.507(0.004)    0.425(0.010)    0.507(0.004)    0.425(0.010)    0.779(0.007)    1.173(0.027)    0.779(0.007)   1.173(0.027)
                 Single AE    0.622(0.006)    0.761(0.021)    0.622(0.006)    0.761(0.021)    0.963(0.010)    1.772(0.038)   0.963(0.010)   1.772(0.038)
                 AEMC-NE      0.647(0.006)    0.850(0.023)    0.647(0.006)    0.850(0.023)    1.004(0.009)    1.884(0.036)   1.004(0.009)   1.884(0.036)
 No covariate
                 DiD          0.713(0.007)    1.043(0.028)    0.713(0.007)    1.043(0.028)    1.085(0.010)    2.163(0.041)   1.085(0.010)   2.163(0.041)
                 Vert-Reg     0.651(0.002)    0.671(0.004)    0.651(0.002)    0.671(0.004)    0.865(0.007)    1.422(0.034)   0.865(0.007)   1.422(0.034)
                 MC-NNM       0.535(0.005)    0.495(0.013)    0.535(0.005)    0.495(0.013)    1.020(0.009)    1.856(0.034)   1.020(0.009)   1.856(0.034)
                 CoDEAL      0.510(0.005)    0.436(0.013)    0.609(0.004)    0.622(0.010)    0.781(0.007)    1.177(0.025)    0.896(0.008)   1.456(0.029)
                 Single AE    0.645(0.007)    0.807(0.022)    0.777(0.007)    1.147(0.024)    1.007(0.009)    1.885(0.037)   1.146(0.010)   2.330(0.042)
                 AEMC-NE      0.682(0.007)    0.919(0.024)    0.759(0.007)    1.093(0.025)    1.051(0.009)    2.018(0.037)   1.127(0.009)   2.259(0.040)
 Matrix Linear
                 DiD          0.736(0.008)    1.093(0.029)    0.810(0.007)    1.262(0.029)    1.122(0.010)    2.275(0.041)   1.191(0.009)   2.503(0.042)
                 Vert-Reg     0.680(0.003)    0.733(0.006)    0.761(0.004)    0.936(0.011)    0.917(0.007)    1.550(0.031)   1.110(0.007)   2.024(0.027)
                 MC-NNM       0.537(0.005)    0.498(0.013)    0.650(0.005)    0.736(0.015)    1.025(0.009)    1.889(0.033)   1.106(0.009)   2.147(0.037)
                 CoDEAL      0.508(0.005)    0.428(0.010)    0.589(0.004)    0.574(0.010)    0.784(0.007)    1.184(0.026)    0.883(0.007)   1.426(0.025)
                 Single AE    0.659(0.006)    0.843(0.018)    0.772(0.007)    1.135(0.024)    1.010(0.008)    1.890(0.031)   1.147(0.009)   2.333(0.036)
                 AEMC-NE      0.700(0.009)    0.952(0.024)    0.761(0.008)    1.099(0.026)    1.071(0.010)    2.073(0.036)   1.130(0.009)   2.265(0.034)
 Vector Linear
                 DiD          0.737(0.006)    1.093(0.023)    0.800(0.007)    1.234(0.025)    1.125(0.009)    2.279(0.036)   1.186(0.009)   2.478(0.036)
                 Vert-Reg     0.675(0.003)    0.724(0.007)    0.742(0.004)    0.884(0.010)    0.915(0.007)    1.555(0.031)   0.995(0.007)   1.768(0.030)
                 MC-NNM       0.536(0.005)    0.499(0.012)    0.640(0.006)    0.714(0.017)    1.019(0.010)    1.870(0.036)   1.099(0.008)   2.124(0.030)
                 CoDEAL       0.700(0.007)    0.920(0.024)   0.644(0.005)    0.704(0.015)     1.012(0.008)    1.913(0.034)   0.904(0.008)   1.492(0.030)
                 Single AE    0.880(0.009)    1.442(0.032)    0.795(0.007)    1.190(0.027)    1.227(0.011)    2.638(0.047)   1.180(0.010)   2.446(0.043)
                 AEMC-NE      0.902(0.008)    1.489(0.028)    0.771(0.007)    1.120(0.026)    1.243(0.009)    2.681(0.041)   1.160(0.010)   2.365(0.042)
 tanh
                 DiD          0.758(0.007)    1.139(0.028)    0.828(0.008)    1.309(0.030)    1.135(0.010)    2.313(0.041)   1.224(0.010)   2.613(0.044)
                 Vert-Reg     0.731(0.003)    0.849(0.008)    0.770(0.005)    0.968(0.015)   0.968(0.007)    1.698(0.032)    1.048(0.007)   1.921(0.029)
                 MC-NNM      0.570(0.005)    0.555(0.013)     0.697(0.006)    0.859(0.019)    1.136(0.009)    2.317(0.036)   1.145(0.009)   2.281(0.040)
                 CoDEAL       1.030(0.005)    1.867(0.017)   0.878(0.004)    1.385(0.013)    1.236(0.008)    2.683(0.038)    1.107(0.007)   2.264(0.030)
                 Single AE    1.085(0.006)    2.125(0.026)    0.985(0.006)    1.808(0.026)    1.367(0.008)    3.241(0.040)   1.338(0.009)   3.116(0.042)
                 AEMC-NE      1.078(0.006)    2.115(0.028)    0.968(0.007)    1.751(0.027)    1.360(0.008)    3.210(0.040)   1.319(0.009)   3.034(0.042)
 log
                 DiD          1.079(0.007)    2.135(0.030)    1.013(0.007)    1.919(0.030)    1.382(0.009)    3.312(0.043)   1.374(0.009)   3.268(0.045)
                 Vert-Reg     1.275(0.005)    2.722(0.020)    1.108(0.005)    2.079(0.017)    1.516(0.008)    3.956(0.049)   1.388(0.008)   3.301(0.039)
                 MC-NNM      0.999(0.005)    1.795(0.021)     0.907(0.005)    1.503(0.019)    1.306(0.008     2.957(0.038)   1.305(0.009)   2.951(0.040)
                 CoDEAL       0.798(0.006)    1.128(0.022)   0.656(0.005)    0.725(0.013)    1.082(0.009)    2.105(0.038)    0.962(0.008)   1.640(0.030)
                 Single AE    0.965(0.007)    1.672(0.029)    0.808(0.007)    1.223(0.027)    1.298(0.009)    2.895(0.044)   1.187(0.009)   2.463(0.041)
                 AEMC-NE      0.999(0.007)    1.756(0.027)    0.789(0.007)    1.162(0.025)    1.313(0.010     2.934(0.044)   1.173(0.009)   2.412(0.041)
 poly
                 DiD          0.848(0.007)    1.337(0.029)    0.841(0.008)    1.339(0.030)    1.197(0.009)    2.510(0.041)   1.234(0.010)   2.650(0.044)
                 Vert-Reg     0.809(0.004)    1.045(0.010)    0.791(0.005)    1.017(0.015)    1.159(0.007)    2.288(0.033)   1.072(0.008)   2.001(0.032)
                 MC-NNM      0.710(0.005)    0.853(0.014)     0.714(0.006)    0.896(0.018)    1.204(0.009)    2.321(0.036)   1.158(0.009)   2.324(0.039)
                 CoDEAL       0.771(0.007)    1.205(0.027)   0.612(0.005)    0.629(0.012)     1.269(0.013)    2.817(0.064)   0.902(0.008)   1.478(0.029)
                 Single AE    1.078(0.011)    2.110(0.043)    0.764(0.006)    1.069(0.020)    1.329(0.013)    3.044(0.064)   1.121(0.010)   2.232(0.040)
                 AEMC-NE      1.234(0.011)    2.692(0.048)    0.784(0.007)    1.148(0.024)    1.539(0.014)    4.010(0.075)   1.156(0.009)   2.355(0.040)
 ReLU
                 DiD          0.868(0.007)    1.407(0.029)    0.840(0.007)    1.334(0.029)    1.275(0.013)    2.819(0.062)   1.220(0.010)   2.603(0.042)
                 Vert-Reg     0.808(0.004)    1.051(0.011)    0.783(0.004)    0.992(0.011)   1.204(0.012)     2.846(0.069)   1.049(0.007)   1.938(0.030)
                 MC-NNM      0.766(0.006)    1.034(0.019)     0.651(0.005)    0.737(0.016)    1.235(0.012)   2.639(0.057)    1.107(0.009)   2.149(0.036)


                                                                         19

Table 4: Comparisons of imputation accuracy (measured by MAE and MSE) under various factor
and covariate generations in Config.1-4. Results are averaged over 50 replications. Numbers in
parentheses are standard errors.

                            Config. 1, linear covariate   Config. 1, nonlinear covariate     Config. 2, linear covariate   Config. 2, nonlinear covariate
 Factor       Method
                             MAE                MSE          MAE               MSE            MAE                MSE          MAE             MSE
              CoDEAL      0.646(0.009)    0.722(0.023)    0.701(0.008)    0.871(0.031)     0.609(0.004)    0.622(0.010) 0.644(0.005)       0.704(0.015)
              Single AE    0.821(0.010)    1.320(0.039)    0.851(0.008)    1.383(0.033)     0.777(0.007)    1.147(0.024) 0.795(0.007)      1.190(0.027)
              AEMC-NE      0.838(0.009)    1.394(0.041)    0.862(0.009)    1.445(0.036)     0.759(0.007)    1.093(0.025) 0.771(0.007)      1.120(0.026)
 Linear
              DiD          0.892(0.010)    1.587(0.045)    0.916(0.009)    1.642(0.037)     0.810(0.007)    1.262(0.029) 0.828(0.008)      1.309(0.030)
              Vert-Reg     0.757(0.006)    0.957(0.019)    0.798(0.007)    1.075(0.024)     0.761(0.004)    0.936(0.011) 0.770(0.005)      0.968(0.015)
              MC-NNM       0.699(0.006)    0.895(0.020)    0.735(0.005)    1.007(0.017)     0.650(0.005)    0.736(0.015) 0.697(0.006)      0.859(0.019)
              CoDEAL      1.078(0.012)    2.136(0.055)    1.117(0.011)    2.238(0.050)     0.894(0.007)    1.454(0.028) 0.945(0.008)       1.593(0.030)
              Single AE    1.223(0.013)    2.618(0.061)    1.250(0.012)   2.705(0.053)      1.142(0.009)    2.312(0.039) 1.185(0.009)      2.458(0.039)
              AEMC-NE      1.209(0.013)    2.566(0.061)    1.235(0.012)   2.645(0.053)      1.128(0.009)    2.256(0.039) 1.167(0.010)      2.389(0.041)
 Sine
              DiD          1.244(0.013)    2.697(0.063)    1.272(0.012)   2.785(0.055)      1.191(0.009)    2.498(0.042) 1.234(0.009)      2.650(0.041)
              Vert-Reg     1.169(0.012)    2.633(0.063)    1.208(0.011)   2.736(0.059)      1.012(0.007)    1.829(0.031) 1.054(0.008)      1.935(0.034)
              MC-NNM       1.201(0.013)    2.511(0.059)    1.231(0.012)   2.610(0.053)      1.105(0.009)    2.142(0.036) 1.153(0.010)      2.305(0.040)
              CoDEAL      0.908(0.013)    2.250(0.100)    0.952(0.015)    2.477(0.126)     0.772(0.010)    1.399(0.076) 0.807(0.011)       1.539(0.070)
              Single AE    0.962(0.013)    2.727(0.116)    0.997(0.016)    3.014(0.164)     0.820(0.010)    1.759(0.087) 0.854(0.011)      1.964(0.084)
              AEMC-NE      0.997(0.014)    3.198(0.142)    1.038(0.017)    3.500(0.192)     0.856(0.011)    2.132(0.110) 0.887(0.012)      2.339(0.111)
 Polynomial
              DiD          0.995(0.013)    2.708(0.111)    1.025(0.016)    2.939(0.144)     0.854(0.010)    1.860(0.083) 0.886(0.011)      2.065(0.092)
              Vert-Reg     0.953(0.013)    1.906(0.076)    1.019(0.016)    2.327(0.113)     0.855(0.007)    1.289(0.032) 0.884(0.010)      1.456(0.042)
              MC-NNM       0.951(0.013)    2.419(0.101)    0.989(0.015)    2.627(0.130)     0.819(0.009)     1.654(0.07) 0.851(0.011)      1.839(0.078)
          CoDEAL          0.797(0.009)    1.097(0.027)    0.838(0.010)    1.217(0.032)     0.698(0.005)    0.820(0.014) 0.743(0.006)       0.929(0.015)
          Single AE        0.989(0.009)    1.712(0.034)    1.026(0.011)    1.845(0.044)     0.891(0.007)    1.375(0.022) 0.930(0.008)      1.502(0.028)
          AEMC-NE          1.045(0.010)    1.919(0.038)    1.088(0.011)    2.084(0.048)     0.959(0.007)    1.605(0.026) 0.982(0.008)      1.679(0.029)
 ReLU MLP
          DiD              0.930(0.008)    1.499(0.028)    0.963(0.010)    1.616(0.036)     0.842(0.005)    1.218(0.017) 0.870(0.007)      1.303(0.021)
          Vert-Reg         0.888(0.010)    1.315(0.035)    0.958(0.013)    1.544(0.046)     0.847(0.005)    1.166(0.014) 0.894(0.006)      1.302(0.020)
          MC-NNM           0.843(0.007)    1.227(0.025)    0.892(0.009)    1.453(0.031)     0.767(0.005)    1.001(0.013) 0.806(0.006)      1.110(0.019)
                               Config. 3, r = 5                Config. 3, r = 10                Config. 4, r = 5                 Config. 4, r = 10
 Factor       Method
                              MAE            MSE              MAE            MSE               MAE            MSE          MAE             MSE
            CoDEAL         0.614(0.007)    0.721(0.023)    0.625(0.005)    0.754(0.018)    0.532(0.003) 0.506(0.009) 0.560(0.004) 0.591(0.012)
            Single AE      0.789(0.011)    1.263(0.044)    0.795(0.008)    1.270(0.033)     0.693(0.005) 0.948(0.017) 0.714(0.006) 1.009(0.023)
            AEMC-NE        0.731(0.010)    1.125(0.042)    0.730(0.007)    1.110(0.029)     0.647(0.005) 0.846(0.018) 0.660(0.005) 0.886(0.021)
 Linear
            DiD            0.801(0.011)    1.341(0.047)    0.802(0.008)    1.326(0.033)     0.711(0.005) 1.031(0.021) 0.726(0.006) 1.081(0.025)
            Vert-Reg       0.940(0.005)    6.880(2.378)    0.865(0.023)    8.238(2.625)     0.691(0.003) 0.888(0.012) 0.750(0.016) 3.071(0.914)
            MC-NNM        0.599(0.007)    0.685(0.023)    0.616(0.006)    0.733(0.019)      0.537(0.003) 0.520(0.009) 0.565(0.004) 0.596(0.014)
            CoDEAL        0.973(0.010)    1.798(0.041)    0.996(0.010)    1.907(0.039)     0.850(0.007) 1.389(0.024) 0.869(0.008) 1.463(0.028)
            Single AE      1.087(0.011)    2.138(0.047)    1.122(0.011)    2.288(0.047)     1.057(0.008) 2.050(0.030) 1.061(0.008) 2.082(0.035)
            AEMC-NE        1.055(0.012)    2.029(0.047)    1.078(0.011)    2.128(0.044)     1.011(0.008) 1.890(0.029) 1.008(0.009) 1.897(0.034)
 Sine
            DiD            1.110(0.012)    2.209(0.049)    1.146(0.012)    2.361(0.048)     1.091(0.008) 2.172(0.031) 1.088(0.009) 2.189(0.037)
            Vert-Reg       1.549(0.090)    8.954(2.462)    1.524(0.123)   15.689(4.057)     0.974(0.009) 2.153(0.059) 1.047(0.013) 4.384(0.544)
            MC-NNM         1.066(0.011)    2.033(0.046)    1.097(0.011)    2.164(0.043)     0.998(0.008) 1.819(0.031) 1.003(0.008) 1.858(0.032)
            CoDEAL        0.785(0.011)    2.146(0.112)    0.811(0.010)    2.379(0.161)     0.662(0.007) 1.311(0.073) 0.699(0.007) 1.530(0.064)
            Single AE      0.895(0.015)    2.423(0.117)    0.926(0.016)    2.710(0.172)     0.739(0.011) 1.582(0.086) 0.785(0.010) 1.798(0.071)
            AEMC-NE        0.860(0.014)    2.921(0.160)    0.897(0.013)    3.273(0.220)     0.720(0.009) 1.889(0.108) 0.764(0.009) 2.166(0.093)
 Polynomial
            DiD            0.887(0.013)    2.422(0.116)    0.917(0.014)    2.693(0.165)     0.739(0.009) 1.613(0.082) 0.773(0.008) 1.801(0.068)
            Vert-Reg       1.155(0.035)    6.771(1.168)    1.131(0.032)    5.530(0.666)     0.788(0.005) 1.246(0.028) 0.881(0.014) 3.254(0.526)
            MC-NNM         0.839(0.012)    2.117(0.104)    0.865(0.012)    2.308(0.129)     0.703(0.008) 1.397(0.071) 0.735(0.007) 1.569(0.058)
            CoDEAL        0.632(0.004)    0.886(0.012)    0.733(0.004)    0.953(0.013)     0.633(0.004) 0.698(0.011) 0.646(0.004) 0.730(0.011)
            Single AE      0.876(0.005)    1.338(0.017)    0.913(0.006)    1.448(0.019)     0.797(0.005) 1.108(0.015) 0.809(0.005) 1.142(0.015)
            AEMC-NE        0.934(0.006)    1.574(0.021)    0.968(0.007)    1.677(0.024)     0.860(0.006) 1.330(0.023) 0.878(0.006) 1.381(0.022)
 ReLU MLP
            DiD            0.817(0.004)    1.163(0.013)    0.842(0.004)    1.238(0.014)     0.742(0.004) 0.962(0.013) 0.748(0.005) 0.976(0.014)
            Vert-Reg       1.045(0.031)    5.157(0.846)    0.948(0.026)   10.387(6.292)     0.755(0.003) 1.060(0.015) 1.070(0.253) 8.399(2.377)
            MC-NNM         0.721(0.003)    0.897(0.009)    0.750(0.003)    0.976(0.010)     0.656(0.004) 0.741(0.011) 0.662(0.004) 0.756(0.011)


                                                                          20

CoDEAL in Four-Block Design. As shown in Config.1 and Config.2 in Table 4, CoDEAL
consistently achieves the lowest MAE and MSE across nearly all settings of factor
effects and covariate effects, and achieves substantial improvements in MAE and MSE relative
to all competing methods. Under the simplest linear factor model, CoDEAL reduces MAE
by approximately 5–10% and MSE by about 10–20% compared to MC-NNM, the second-best
performing method, and significantly outperforms other baseline methods by even larger
margins. For scenarios involving nonlinear factor models, CoDEAL continues to show superior
performance, typically surpassing MC-NNM by approximately 5–20% reduction in MAE
and 10–20% reduction in MSE. Overall, these findings clearly demonstrate that CoDEAL’s
integration of DNN-based covariate adjustment and deep autoencoder-based latent factor
extraction is robust and highly effective across diverse linear and nonlinear data-generating
settings.

CoDEAL in Staggered Adoption Design. As Config.3 and Config.4 of Table 4 demon-
strate, CoDEAL achieves substantial percentage gains in MAE and MSE relative to all
competing methods across staggered adoption designs. Under Linear factors, CoDEAL
CoDEAL ties MC-NNM within 3% while reduces MAE by 16–35% over other
benchmarks. In staggered design with nonlinear factors, CoDEAL delivers evident advan-
tages. Especially when under ReLU -based factors, CoDEAL delivers its largest margins, with
MAE reductions of 12–40% and MSE reductions of 15–41% versus all competing
approaches. Finally, as r increases from 5 to 10, the missingness pattern grows more complex
and extensive, so all methods show some drop in accuracy. However, CoDEAL’s errors only
suffer a slight increase, whereas simpler methods like Vert-Reg show a dramatic rise in both
MAE and MSE when missingness increases. Above results reemphasize CoDEAL’s consistent
superiority and robustness in staggered-treatment settings.

In summary, CoDEAL achieves performance comparable to MC-NNM in linear settings,
while exhibiting substantial improvements over all considered methods in the presence of
nonlinear effects. By integrating DNN-based covariate adjustments with deep multi-output-
AE–based nonlinear factor analysis, CoDEAL is robustly effective across a wide range of
linear and nonlinear scenarios.

                                             21

4     Real Data: Oxford COVID-19 Government Response
     Tracker (OxCGRT)
    In this section, we illustrate our methodology using data from the OxCGRT (Hale
et al., 2021). The data is publicly available at https://github.com/OxCGRT. OxCGRT
systematically collects panel data on the timing and intensity of 24 different government
interventions, such as school closures, travel restrictions, and vaccination mandates, across
over 185 countries from January 2020 through 2022.
    We examine the impact of two specific policy interventions: (1) the implementation
of mandatory vaccination, and (2) the termination of internal travel restrictions
(results in Appendix D). Our analysis covers their impacts on COVID-19 confirmed cases and
deaths across 64 states and territories in the United States and Canada from April 16, 2021,
to December 11, 2021. This time frame aligns with the period following the detection of the
Delta variant and precedes the rapid emergence of the Omicron variant in North America.
In our analysis, we use four OxCGRT indices (overall government response, containment
and health, stringency, and economic support), averaged over the time frame, as covariates,
resulting in data dimensions of N = 64, T = 240, and P = 4. The policy is adopted in
different states at various times, forming a staggered adoption design (visualized in Figure 3).
    Figure 4 plots the comparisons of the total confirmed cases and deaths across policy-in-
effect states between observed results (with policy executed) and estimated counterfactuals (if
no policy). As shown in Figure 4, if the mandatory vaccination policy was not implemented,
there would be an increase in both confirmed cases and deaths, with a notably larger impact
on mortality. This observation aligns with recent research highlighting a greater reduction
in mortality than in infection rates following vaccination (Hernández Bautista et al., 2023;
Xu et al., 2023; Meslé et al., 2024; Wu et al., 2023). Although the visual differences in the
log-transformed graphs appear modest, converting these back to their original scales reveals
substantial impacts: on December 11, 2021, the actual cumulative counts of confirmed cases
and deaths for states with mandatory vaccination policies were 26.57 million, 0.39 million,
respectively. In contrast, our estimates suggest that without these policies, the counts would
have risen to 29.81 million(↑ 12.2%), 0.52 million(↑ 34.6%).

                                              22

Figure 3: Indicator matrix of the implementation of mandatory vaccination policy.
                Blocks with darker color refer to policy executed.
                                       23

Figure 4: Comparison of the total confirmed cases (Left) and deaths (Right) across policy-
in-effect states between observed results with the mandatory vaccination policy executed
and estimated counterfactuals if no policy. Values are reported in log10 scale, and plots are
visualized with 14-day moving averages.


5    Discussion of Limitations and Future Extensions
    In this paper, we propose CoDEAL, a deep-learning-based causal matrix completion
method and a unifying framework for estimating heterogeneous causal effects in panel data
models. One limitation of CoDEAL is that it is only designed for binary treatment settings.
Causal panel data with multiple treatments is an open research area with limited existing
work in the literature (Abadie, 2021), yet this topic is gradually gaining attention due to its
growing relevance in practical applications (Agarwal et al., 2020; Squires et al., 2022).
    To extend CoDEAL to multiple treatment settings, one possible direction is to adopt
a similar strategy to that mentioned by (Agarwal et al., 2020), in which the treatment
type is incorporated as an additional data dimension in the analysis. Together with the
original panel data structures, the inclusion of treatment dimension forms a three-dimensional
tensor. Methodologically, with different choices of factor models and advanced
neural network architectures, the proposed CoDEAL framework can be further
generalized to accommodate various data types with more complex data structures,
such as tensors, images, and networks. For example, CoDEAL can be naturally extended
to three-dimensional or higher-order tensors by involving tensor factorization methodologies
(Han and Zhang, 2022; Chen et al., 2024; Zhou et al., 2025; Chen et al., 2025), providing a


                                              24

more general approach compared to matrix factor analysis (Bai and Ng, 2021; Luo et al., 2022;
Yu et al., 2022, 2024a). That being said, to extend to multiple treatments, how to handle the
increased level of missingness brought by the treatment assignments in a higher-dimensional
tensor remains challenging and requires careful modeling strategies. We leave the further
investigation on multiple treatment settings as future work.
