<!--
source: /Users/pranjal/Code/deep-inference/literature/causal-deep-learning-2016-2026/downloads/arXiv 2403.04236.pdf
backend: docling
part: 1/1
-->

## Regularized DeepIV with Model Selection

Zihao Li *

ZL9045@PRINCETON.EDU

Princeton University

Hui Lan † Stanford University

HUILAN@STANFORD.EDU

Vasilis Syrgkanis Stanford University

VASILIS SYRGKANIS@STANFORD.EDU

Mengdi Wang Princeton University

MENGDIW@PRINCETON.EDU

Masatoshi Uehara ‡ Genentech

UEHARAM1@GENE.COM

## Abstract

In this paper, we study nonparametric estimation of instrumental variable (IV) regressions. While recent advancements in machine learning have introduced flexible methods for IV estimation, they often encounter one or more of the following limitations: (1) restricting the IV regression to be uniquely identified; (2) requiring minimax computation oracle, which is highly unstable in practice; (3) absence of model selection procedure. In this paper, we present the first method and analysis that can avoid all three limitations, while still enabling general function approximation. Specifically, we propose a minimax-oracle-free method called Regularized DeepIV (RDIV) regression that can converge to the least-norm IV solution. Our method consists of two stages: first, we learn the conditional distribution of covariates, and by utilizing the learned distribution, we learn the estimator by minimizing a Tikhonov-regularized loss function. We further show that our method allows model selection procedures that can achieve the oracle rates in the misspecified regime. When extended to an iterative estimator, our method matches the current state-of-the-art convergence rate. Our method is a Tikhonov regularized variant of the popular DeepIV method with a non-parametric MLE first-stage estimator, and our results provide the first rigorous guarantees for this empirically used method, showcasing the importance of regularization which was absent from the original work.

Keywords: Nonparametric regression, causal inference, instrumental variable, general function approximation

## 1. Introduction

Instrumental variable (IV) estimation is an important problem in various fields, such as causal inference (Angrist and Imbens, 1995; Newey and Powell, 2003; Deaner, 2018; Cui et al., 2020; Kallus et al., 2021, 2022), missing data problems (Miao et al., 2018; Wang et al., 2014), dynamic discrete choice models Kalouptsidi et al. (2021) and reinforcement learning (Liao et al., 2021; Uehara et al., 2022a,b; Shi et al., 2022; Wang et al., 2021; Yu et al., 2022).

In this paper, we focus on nonparametric IV (NPIV) regression (Newey and Powell, 2003). NPIV concerns three random variables X ∈ R d (covariate), Y ∈ R (outcome variable), and Z ∈ R d

* Equal contribution

† Equal contribution

‡ This work was done at Cornell University

(instrumental variables). We are interested in finding a solution h 0 of the following conditional moment equation (Dikkala et al., 2020b; Chernozhukov et al., 2019):

<!-- formula-not-decoded -->

This is equivalently written as T f = r 0 where T : L 2 ( X ) /owner f ( X ) ↦→ E [ f ( X ) | Z ] ∈ L 2 ( Z ) and r 0 ( Z ) = E [ Y | Z ] by denoting L 2 ( X ) , L 2 ( Z ) to be the L 2 space defined on X and Z with respect to the underlying distribution. Both the operator T and E [ Y | Z ] remain unknown. Hence, we aim to solve T f = r 0 by harnessing an identically independent distributed (i.i.d.) dataset { X i , Y i , Z i } i ∈ [ n ] .

There has been a surge in interest in NPIV regressions that try to integrate general function approximation such as deep neural networks beyond classical nonparametric models (Hartford et al., 2017; Singh et al., 2019; Xu et al., 2021; Zhang et al., 2023; Dikkala et al., 2020b; Bennett and Kallus, 2020; Bennett et al., 2023a,b; Kallus et al., 2022; Singh, 2020). Despite these extensive efforts, existing approaches encounter several challenges. The first challenge is the ill-posedness of the inverse problem. Many existing works (Liao et al., 2020a; Newey and Powell, 2003; Florens et al., 2011; Kato et al., 2021) require that the NPIV solution h 0 is unique, and further impose quantitative bounds on measures of ill-posedness. However, it is known that the uniqueness assumption is easily violated in practical scenarios, such as weak IV (Andrews and Stock, 2005; Andrews et al., 2019) or proximal causal inference (Kallus et al., 2021). The second challenge involves the reliance on minimax optimization oracles in many methods (Bennett et al., 2023a; Dikkala et al., 2020b; Liao et al., 2020a; Bennett et al., 2023b; Zhang et al., 2023), which results in minimax non-convex non-concave optimization when invoking deep neural networks. However, currently, such an optimization can be notoriously unstable and may fail to converge (Lin et al., 2020b; Jin et al., 2020; Lin et al., 2020a; Diakonikolas et al., 2021; Razaviyayn et al., 2020). Instead, our approach seeks to address this challenge by proposing a computationally efficient estimator that relies on standard supervised learning oracles rather than minimax oracles. The third challenge is the absence of clear procedures for model selection in existing works (Xu et al., 2021; Zhang et al., 2023; Cui et al., 2020; Hartford et al., 2017). This issue is problematic, because model selection, including techniques like cross-validation, has played a pivotal role in the practical success of machine learning algorithms (Bartlett et al., 2002a; Gold and Sollich, 2003; Guyon et al., 2010; Cawley and Talbot, 2010; Raschka, 2018; Emmert-Streib and Dehmer, 2019; McAllester, 2003). Model selection becomes essential particularly in scenarios where the true NPIV solution h 0 lies outside the chosen function classes optimized by the algorithm, which has been seldom explored in prior works.

To address aforementioned challenges, we propose a two-stage method, which we refer to as the Regularized DeepIV (RDIV) . This approach consists of two steps. First, we learn the operator T by maximum likelihood estimation (MLE). Secondly, we obtain an estimator for h 0 by solving a loss incorporating the learned T and Tikhonov regularization (Ito and Jin, 2014) to handle scenarios where solutions of the conditional moment constraint are nonunique. While our method can be viewed as a regularized variant of the DeepIV method of Hartford et al. (2017) with a non-parametric MLE firststage, no prior theoretical convergence guarantees exist for the DeepIV method. We show that our estimators can converge to the least norm IV solution (even if solutions are nonunique) and derive its L 2 error rate guarantee based on critical radius. Subsequently, we introduce model selection procedures for our estimators. Particularly, we provide theoretical guarantees for model selection via out-of-sample validation approaches, and show an oracle result in our context. Finally, we further illustrate that our method can be easily generalized to an iterative estimator that more effectively leverages the well-posedness of h 0 .

Table 1: Summary of IV regression literature with general function approximation such as neural networks. 'Model Selection' means allowing model selection methods. 'No Minimax' means no need of minimax oracle. 'No Uniquness' means unique solution is not assumed.

|                                               | Model Selection   | No Minimax   | No Uniqueness   | RMSE rates   |
|-----------------------------------------------|-------------------|--------------|-----------------|--------------|
| Hartford et al. (2017) Dikkala et al. (2020a) |                   | /check       |                 |              |
| Liao et al. (2020a)                           |                   |              |                 |              |
|                                               |                   |              |                 | /check       |
| Xu et al. (2021)                              |                   | /check       |                 |              |
| Bennett et al. (2023a)                        |                   |              | /check          | /check       |
| Bennett et al. (2023b)                        |                   |              | /check          | /check       |
| Ours                                          | /check            | /check       | /check          | /check       |

Our contribution is to propose the first estimator for NPIV that (a) operates in the absence of the uniqueness assumption, (b) does not rely on the minimax computational oracle, and (c) allows for model selection. Subsequently, we demonstrate that our estimator can be extended to an iterative estimator, which achieves a state-of-the-art convergence rate in terms of L 2 error analogous to Bennett et al. (2023b), while Bennett et al. (2023b) requires a minimax computational oracle and does not permit us to perform model selection. Therefore, our estimator can be seen as an estimator with a strong theoretical guarantee due to the property (a) while it is practical due to properties (b) and (c). Notably, none of the existing works can enjoy such a guarantee, as shown in Table 1.

## 2. Related Works

Nonparametric IV problem. Nonparametric IV estimation has been extensively explored in past decades. Such estimation is tough to solve even when both the linear operator T and the response r 0 are known, known as ill-posedness. The ill-posedness often refers to the presence of one or more of the following characteristics: (1) the absence of solutions, (2) the existence of multiple solutions, and (3) the discontinuity of the inverse of operator T . Many traditional nonparametric estimators have been proposed to address these challenges, such as series-based estimators (Florens et al., 2011; Ai and Chen, 2003; Chen, 2021; Chen and Pouzo, 2012; Darolles et al., 2011) and kernelbased estimators (Hall and Horowitz, 2005; Horowitz, 2007; Singh et al., 2019). However, these methods cannot directly accommodate modern machine-learning techniques like neural networks.

Recently, there has been growing interest in the application of general function approximation techniques, such as deep neural networks and random forests, to IV problems in a unified manner. Amongthose methods, Bennett and Kallus (2020); Dikkala et al. (2020b); Lewis and Syrgkanis (2018); Liao et al. (2020a); Zhang et al. (2023) reformulate the conditional moment constraint into a minimax optimization and use its solution as the estimator. Notably, Liao et al. (2020a); Bennett et al. (2023b,a) establish L 2 convergence by linking minimax optimization with Tikhonov regularization under the assumption of the source condition. Moreover, Liao et al. (2020b) assumes uniqueness of solution h 0 . Dikkala et al. (2020b); Lewis and Syrgkanis (2018) provide a guarantee for the projected MSE without further assumptions. However, they could not guarantee the convergence rate in strong L 2 metric when multiple solutions to conditional moment constraint exist. Furthermore, these methods require a computation oracle for minimax optimization, which further makes model

## LI LAN SYRGKANIS WANG UEHARA

selection challenging. In contrast, our method does not require computational oracles and enables model selection with statistical guarantees.

Several existing works eschew the need for minimax optimization oracles (Hartford et al., 2017; Xu et al., 2021). However, all these works do not provide finite sample guarantee or model selection. For example, as the most related work, DeepIV (Hartford et al., 2017) introduces a similar loss function to us. However, it lacks an explicit regularization term, which results in the lack of theoretical guarantee and the lack of guarantee for model selection. As another work, Xu et al. (2021) extends the two-stage kernel algorithm in Singh et al. (2019) to deep neural networks, but their algorithm is essentially a bilevel optimization problem, which is hard to solve in general (Hong et al., 2023; Khanduri et al., 2021; Guo et al., 2021).

Model selection. Model selection has been well studied in the regression and supervised machine learning literature (Bartlett et al., 2002a; Gold and Sollich, 2003; McAllester, 2003). The objective can be described more concretely as follows: given M candidate models, { f 1 , . . . , f M } , each having some statistical complexity δ j and some approximation error /epsilon1 j (with respect to some un-known true model f 0 ) we wish to find an aggregated model ˆ f whose mean squared error is closed to the optimal trade-off between statistical complexity and approximation error among all models, i.e.: ‖ ˆ f -f 0 ‖ /lessorsimilar min j M =1 δ j + /epsilon1 j . The statistical complexity of a function space can be accurately characterized, albeit the approximation error is un-attainable as it relates to the unknown true model. Aguarantee of the form above implies that using the observed data we can compete (up to constants) with an oracle that knows the approximation errors and chooses the best model space. We leave the detailed summary of existing works in Appendix A. Despite the abundance of methodologies for IV regression problems, few studies have investigated model misspecification and provided model selection procedures to select the best model class. As a few exceptional works, while Xu et al. (2021) and Ai and Chen (2007) considered the misspecified regime, but they did not discuss model selection approaches. A typical approach to model selection is out-of-sample validation: estimate different models on half the data and select the estimated model that achieves the smallest empirical risk on the second half (or the best convex ensemble of models that achieves the smallest out-ofsample risk). One problem that arises for model selection in this IV regression setup is to transform the excess risk guarantees, which will be in terms of the weak metric, i.e. ‖T ( · ) ‖ 2 , into the desired bound in the L 2 error. In this work, we show that by leveraging the Tikhonov regularization, we can achieve an MSE bound that achieves the same order as the oracle function class.

## 3. Notations

For a function f : X × Y × Z → R , we denote its population expectation by E [ f ( X,Y,Z )] . We denote the empirical mean of f by E n [ f ( X,Y,Z )] := 1 n ∑ n i =1 f ( X i , Y i , Z i ) . We denote the set of all probability distributions defined on set Ω by ∆(Ω) . We denote the L p norm of f by ‖ f ‖ p := E [ | f | p ] 1 /p . Throughout the paper, whenever we use a generic norm of a function ‖ f ‖ , we will be referring to the L 2 -norm. For two density function p ( x ) and q ( x ) , we denote their Hellinger distance by H ( p ( · ) | q ( · )) = ∫ X ( √ p ( x ) -√ q ( x )) 2 dµ ( x ) . For a functional operator T : L 2 ( X ) → L 2 ( Z ) , we denote the range space of T by R ( T ) , i.e., R ( T ) = {T h : h ∈ L 2 ( X ) } . Moreover, we use T ∗ : L 2 ( Z ) → L 2 ( X ) to denote the adjoint operator of T , i.e., 〈 g, T h 〉 L 2 ( Z ) = 〈T ∗ g, h 〉 L 2 ( X ) for any h ∈ L 2 ( X ) , g ∈ L 2 ( Z ) , where 〈· , ·〉 L 2 ( X ) and 〈· , ·〉 L 2 ( Z ) are inner products over L 2 ( X ) and L 2 ( Z ) , respectively. For θ ∈ Θ = { θ | ∑ j θ j = 1 , θ j ≥ 0 , ∀ j } , we denote h θ = ∑ j θ j h j . We use e j to denote the one-hot vector where that is zero except for the j th component, which equals to 1 . For a function class F , we define the localized Rademacher complexity by ¯ R n ( δ ; F ) :=

E [ E /epsilon1 [ sup f ∈F , ‖ f ‖ 2 ≤ δ ∣ ∣ 1 n ∑ n i =1 /epsilon1 i f ( x i , z i ) ∣ ∣ ]] , where /epsilon1 i are i.i.d. Rademacher random variables. For a function class F over X and Z , we define its star hull by star( F ) = { γf, γ ∈ [0 , 1] , f ∈ F} . For a function class F , we denote ¯ F := star( F - F ) to define its symmetrized star hull. We define the critical radius δ n, F of a function class F as any solution to the inequality δ 2 ≥ ¯ R n (star( F -F ) , δ ) . We use µ to denote the Lebesgue measure.

## 4. Problem Statement and Preliminaries

As mentioned in Section 1, we aim to solve the following inverse problem with respect to h , known as the nonparametric IV regression:

<!-- formula-not-decoded -->

While T and r 0 are unknown a priori, using i.i.d. observations { X i , Y i , Z i } i ∈ [ n ] , we aim to solve this equation. We denote its associated distributions by g 0 , e.g., denote the conditional density of X ∈ X given Z ∈ Z by g 0 ( x | z ) ∈ {X × Z → R } . Throughout this work, we assume a solution to Equation (1) exists.

Assumption 1 (Existence of Solutions) We have r 0 ∈ R ( T ) , i.e. N r 0 ( T ) := { h ∈ H : T h = r 0 } /negationslash = ∅ .

Crucially, even though a solution to (1) exists, it might not be unique. Hence, we propose to target a specific solution that achieves the least norm, defined as:

<!-- formula-not-decoded -->

Note this least norm solution is well-defined, as it is defined by the projection of the origin onto a closed affine space N r 0 ( T ) ⊂ L 2 ( X ) . Indeed, with Assumption 1, it is easy to prove that h 0 in (2) always exists (Bennett et al., 2023a, Lemma 1).

As we emphasize the challenges in Section 1, although there have been a lot of method that use minimax optimization for estimating h 0 , when using general function approximation such as neural networks, the minimax optimization tends to be computationally hard (Lin et al., 2020b; Jin et al., 2020; Lin et al., 2020a; Diakonikolas et al., 2021; Razaviyayn et al., 2020). Moreover, it remains unclear how to perform model selection for those methods. Hence, in this paper, we aim to propose a new method that can incorporate any function approximation for estimating the least square norm solution h 0 in (2) with a strong convergence guarantee in L 2 ( X ) under mild assumptions (i.e., such as without the uniqueness of h 0 ) while allowing for model selection.

## 5. Regularized Deep IV

In this section, we introduce a two-stage algorithm, Regularized DeepIV (RDIV), aimed at obtaining the least square solution h 0 as defined in Equation (2). Even though we borrow the DeepIV terminology from the prior work (Hartford et al., 2017), our method can be used with arbitrary function approximators and not necessarily neural network function spaces. Being inspired by the original constrained optimization (2), we aim to solve a regularized version of the problem:

<!-- formula-not-decoded -->

where H ⊂ L 2 ( X ) represents a hypothesis class that consists of possible candidates for h 0 , and α ∈ R + denotes a parameter controlling the strength of regularization. While this formulation itself has

## Algorithm 1 Regularized Deep IV (RDIV)

Require: Validation dataset { X i , Y i , Z i } i ∈ [ n ′ ] that is independent from the training dataset, function class G ⊂ {Z → ∆( X ) } , function class H ⊂ {X → R } , a regularization hyperparameter α ∈ R &gt; 0

1: Learn ˆ g ( x | z ) with MLE:

<!-- formula-not-decoded -->

- 2: Learn ˆ h by the following estimator:

<!-- formula-not-decoded -->

where ˆ T : L 2 ( X ) → L 2 ( Z ) is defined by ˆ T f ( Z ) = E x ∼ ˆ g ( X | Z ) [ f ( X )] using ˆ g in the first step. output ˆ h .

been known in the literature on general inverse problems (Cavalier, 2011; Mendelson and Neeman, 2010), we consider common scenarios in IV where both the conditional expectation operator T and the population expectation in Equation (3) are unknown, and need to leverage dataset { X i , Y i , Z i } .

Importantly, our method does not necessitate a demanding computational oracle such as nonconvex non-concave minimax or bilevel optimization, unlike many existing works for nonparametric IV with general function approximation (Lewis and Syrgkanis, 2018; Xu et al., 2021; Bennett et al., 2023a). Even when using neural networks for G and H , we just need standard ERM oracles for density estimation or regression whose optimization is empirically known to be successful and theoretically more supported (Du, 2019; Chen et al., 2018; Zaheer et al., 2018; Barakat and Bianchi, 2021; Wu et al., 2019; Zhou et al., 2018; Ward et al., 2020). We leave the numerical comparison between our method and existing NPIV methods (Hartford et al., 2017; Dikkala et al., 2020b; Xu et al., 2021; Singh et al., 2019) in Appendix 10.

To address this challenge, by integrating general function approximation such as neural networks, we propose a two-stage method, the Regularized Deep Instrumental Variable (RDIV) , which is summarized in Algorithm 1. In the first stage, given a function class G comprising functions of the form { g : X × Z → R , ∫ X g ( x | z ) µ ( dx ) = 1 for all z } , we aim to learn the conditional expectation operator T by estimating the ground-truth conditional density g 0 ( x | z ) from the dataset { X i , Z i } i ∈ [ n ] with MLE in Equation (4). In the second stage, with the learned conditional density ˆ g in the first step, we learn h 0 by replacing expectation and T in Equation (3) with empirical approximation and ˆ T , respectively, as shown in Equation (5).

Remark 2 (Comparison with Deep IV) Our algorithm shares similarities with DeepIV in (Hartford et al., 2017), and indeed, it draws inspiration from it. However, a key distinction lies in our introduction of an explicit regularization term in Equation (5) . Such a term endows the loss function with strong convexity, which plays a pivotal role in obtaining guarantees without the requirement for solution uniqueness. Furthermore, the original DeepIV work lacks a rigorous discussion on convergence guarantees or model selection. Hence, despite the algorithmic resemblances, our contributions primarily focus on the theoretical aspect, showcasing rapid convergence guarantees under mild as-

sumptions, linking them to a formal model selection procedure, and exploring the iterative version to achieve a refined rate in Section 9.

Remark 3 (Computaion for ˆ T ) Some astute readers might notice it could be hard to evaluate ˆ T h exactly in Equation (5) . However, in practical application when h is parametrized as a neural network, we can sample a batch of { X ′ j } j ∈ [ B ] by ˆ g ( X | Z i ) for every Z i in the dataset, and calculate a stochastic gradient that is an unbiased estimator of the real gradient of the loss function in Equation (5) . Existing theory and empirical results for stochastic first-order methods can then guarantee the performance in many scenarios (Jin et al., 2019; Barakat and Bianchi, 2021; Chen et al., 2018; Hartford et al., 2017).

## 6. Finite Sample Guarantees

In this section, we demonstrate a convergence result of our estimator ˆ h in RDIV to h 0 and derive its L 2 error rate after introducing several assumptions.

We commence by introducing the β -source condition, a concept commonly used in the literature on inverse problems (Carrasco et al., 2007; Ito and Jin, 2014; Engl et al., 1996; Bennett et al., 2023b; Liao et al., 2021), which mathematically captures the well-posedness of the function h 0 .

Assumption 4 ( β -Source Conditon) The least norm solution h 0 satisfies h 0 = ( T ∗ T ) β/ 2 w 0 for some w 0 ∈ H and β ∈ R ≥ 0 , i.e., h 0 ∈ R ( T ∗ T ) β/ 2 . Recall T ∗ is an adjoint operator of T defined in Section 3.

In the following, we present its interpretation. First, as special cases, when X , Z are finite (e.g., discrete random variables), it holds when β = ∞ . However, in our cases of interests where X , Z are not finite, this assumption restricts the smoothness of h 0 . Intuitively, when the parameter β is large, the function h 0 exhibits greater smoothness, and the assumption gets stronger, in the sense that eigenfunctions of h 0 relative to an operator T have smaller eigenvalues as explained in Bennett et al. (2023a, Section 6.4).

Next, we introduce another standard assumption as follows. This requires that the function classes H and G are well-specified. We will later consider misspecified cases as in Section 7.

The final assumption is as follows. This is standard in analyzing the convergence of nonparametric MLE (Wainwright, 2019, Chap 14, p.g. 476). We will later discuss how to relax such an assumption in Remark 11 and Appendix B.

Assumption 5 (Realizability of function classes) We assume h 0 ∈ H , g 0 ∈ G .

Assumption 6 (Lower-bounded density) We assume a constant C 0 &gt; 0 such that g 0 ( x | z ) &gt; C 0 holds for all x ∈ X and z ∈ Z .

Finally, we present our guarantee for Algorithm 1.

Theorem 7 ( L 2 convergence rate for RDIV with MLE) Suppose Assumption 4,5,6 hold. Let ‖ Y ‖ ∞ ≤ C Y , ‖ h ‖ ∞ ≤ C H holds for all h ∈ H , ‖ g ‖ ∞ ≤ C G holds for all g ∈ G . There exists absolute constant c 1 , c 2 , such that with probability at least 1 -c 1 exp( c 2 nδ 2 n ) :

<!-- formula-not-decoded -->

In particular, by setting α = δ 2 2+min { β, 2 } n we have

<!-- formula-not-decoded -->

Here δ n = max { δ n, G , δ n, H } , where δ n, F is the critical radius of star ( F-F ) = { λ ( f -f ′ ) , f, f ′ ∈ F , λ ∈ [0 , 1] } . O ( · ) hides constants of polynomial order of C Y , C G , C H , and 1 /C 0 .

Sketch of the proof. Wenow sketch the proof of Theorem 7. Recall that h ∗ is the optimizer of the Tikhonov-regularized loss function (3). We first introduce the following lemma, which characterizes the bias caused by the regularization.

Lemma 8 (Regularization Bias) Under Assumption 4, we have

<!-- formula-not-decoded -->

Therefore, recalling ‖ ˆ h -h 0 ‖ 2 2 ≤ 2( ‖ h ∗ -h 0 ‖ 2 2 + ‖ ˆ h -h ∗ ‖ 2 2 ) , we only need to bound ‖ ˆ h -h ∗ ‖ 2 2 . Utilizing the strong convexity of (3), we have the following lemma:

Lemma 9 (Empirical Deviation &amp; First-stage Bias) With probability at least 1 -c 1 exp( c 2 nδ 2 n, H ) , we have the following inequality:

<!-- formula-not-decoded -->

Lemma 9 shows we can bound ‖ ˆ h -h ∗ ‖ 2 2 by two terms (a1) and (a2). Here (a1) is a centered empirical process, and (a2) is the error when estimating T by ˆ T . Utilizing localized concentration inequality and the boundedness of function class H , we can bound (a1) by O ( δ 2 n, H + δ n, H ‖ ˆ h -h ∗ ‖ 2 ) . To control (a2), we prove the following lemma:

where L ( f ) := ( Y -f ( Z )) 2 .

Lemma 10 (MLE error) With probability at least 1 -exp( nδ 2 n, G ) , we have

<!-- formula-not-decoded -->

for every h -h ′ ∈ H - H .

Note that such a bound is nontrivial since the standard analysis for MLE only results in a convergence rate in terms of Hellinger distance between the ˆ g and g , and does not directly provide a bound on L 1 norm of T h -ˆ T h . With Lemma 10, we can now bound (a2) from above with the critical radius of G , and the L 2 distance between ˆ h and h ∗ . Finally, combining current arguments, Lemma 9 and 10, we have

<!-- formula-not-decoded -->

for certain constants c ′ . By organizing the above equation, we have

<!-- formula-not-decoded -->

Combine Lemma 8 and Equation (7), we further have

<!-- formula-not-decoded -->

2

select α = δ 2+min { β, 2 } n and we conclude the proof.

/squaresolid

The critical radius δ n measures the statistical complexity of function class H and G . For example, for parametric class or Gaussian Kernel, δ n = ˜ O ( n -1 / 2 ) , while for first order Sobolev class, δ n = ˜ O ( n -1 / 3 ) (Wainwright, 2019; Bartlett et al., 2002b). In those cases, when β ≥ 2 , the final rate in L 2 metric will be ˜ O ( n -1 / 2 ) in the former case and ˜ O ( n -1 / 3 ) in the latter case, respectively. Wenow give the interpretation of our result. The bound of ‖ ˆ h -h 0 ‖ 2 2 consists of two terms. Term (i) comes from a statistical error to estimate h ∗ from H and G (i.e., ‖ ˆ h -h ∗ ‖ 2 ). Here, we use the strong convexity owing to Tikhonov regularization as it enables us to convert the population risk error to an error in L 2 metric as in Lemma 9. Then, we properly bounded the population risk from above by the empirical process term properly as in Lemma 10. While this δ 2 n rate is known as the standard fast rate in nonparametric regression (Wainwright, 2019), our result is still non-trivial because we need to handle a statistical error term properly when approximating T with ˆ T , which comes from the MLE error in the form of Hellinger distance.

The term (ii) comes from the bias ‖ h 0 -h ∗ ‖ 2 incurred by adding a Tikhonov regularization. This analysis has been used in existing works (e.g., (Cavalier, 2011)). Due to min( β, 2) , while we cannot leverage a high smoothness β especially when β ≥ 2 , we will see how to leverage β in such a case by introducing an iterative estimator in Section 9.

We also compare our work to existing state-of-the-art convergence rate O ( δ 2 min { β, 1 } 1+min { β, 1 } n ) in Bennett et al. (2023b), in which they employ a minimax-type algorithm. When β ≥ 2 , we achieve the same rate. We also remark that although our rate is slightly slower than theirs when β ≤ 2 , our method does not require a minimax-optimization oracle and can be incorporated with method selection methods. Besides, we will show that our method can achieve a state-of-the-art rate in our extension to iterative estimator in Section 9.

Remark 11 (Removing the Boundedness Assumption) While the lower-boundedness of density function in Assumption 6 is widely used in existing literature, we can easily remove it in several ways. We show that we can relax it by using a χ 2 -MLE instead of MLE:

<!-- formula-not-decoded -->

We delay detailed results for our methods under χ 2 -MLE in Appendix B.

## 7. Misspecified Setting

Next, we establish the finite sample result when Assumption 5 does not hold, i.e., function classes H and G are misspecified. This result serves as an important role in formalizing the model selection procedure in Section 8.

Theorem 12 ( L 2 convergence rate for RDIV with MLE under misspecification) Suppose Assumption 4 and 6 hold, and there exists h † ∈ H and g † ∈ G such that ‖ h 0 -h † ‖ 2 ≤ /epsilon1 H and

E z ∼ g 0 [ D KL ( g 0 ( ·| z ) | g † ( ·| z ))] ≤ /epsilon1 G . For any 0 &lt; α ≤ 1 , we have holds with probability at least 1 -c 1 exp( c 2 nδ 2 n ) . Here δ n has the same definition in Theorem 7.

<!-- formula-not-decoded -->

The bound for ‖ ˆ h -h 0 ‖ 2 2 consists of three terms: term (b1) measures the statistical deviation of a normalized empirical process, term (b2) measures the regularization error caused by Tikhonov regularization and term (b3) measures the effect of model misspecification. Here term (b3) has a poly ( 1 α ) dependency. This is because model misspecification causes a higher population risk in both stage 1 and 2 of Algorithm 1. Hence, the more convex the loss function, the lesser the shift in the optimizer. The readers may notice that term (b2) is slightly slower than the original bias term in Theorem 14. This is because the difference of the optimal value in (3) due to misspecification of H is of order O ( α min { β +1 , 2 } + /epsilon1 2 H ) , as we will show in Lemma 22 in the Appendix. By the α -strong convexity endowed by Tikhonov regularization, this results in a shift of h ∗ of magnitude O α min { β +1 , 2 }-1 + /epsilon1 2 H /α .

( ) Theorem 12 is particularly useful when we apply estimators based on sample-dependent function classes H and G (a.k.a. sieve estimators) that approximate certain function spaces. For example, H can be linear models with polynomial basis functions that take the form 〈 φ ( X ) , θ 〉 , which can gradually approach H¨ older or Sobolev balls, and G can be a set of neural networks with a growing dimension (Chen, 2007; Chen et al., 2022; Schmidt-Hieber, 2020). More specifically, when X and Z are bounded, and h 0 and g 0 are s -H¨ older smooth, it is well known that a deep ReLU neural network with depth O (log(1 //epsilon1 )) , width O ( d/epsilon1 -d/s ) and weights bounded by ˜ O (1) could satisfy the approximation error in Theorem 12 (Schmidt-Hieber, 2019), recall that d is the dimension of X and Z . In that case, δ 2 n = ˜ O ( /epsilon1 -d/s /n ) (Bartlett et al., 2002b; Chen et al., 2022). Choosing the architecture of the neural network according to /epsilon1 = ˜ O ( n -1 / (1+ d/s ) ) , then Theorem 12 shows that by setting α = O ( n 1 (1+ d/α )(min { β +1 , 2 } +1) ) , we have ‖ ˆ h -h 0 ‖ 2 2 = ˜ O ( n min { β +1 , 2 }-1 (1+ d/s )(min { β +1 , 2 } +1) ) .

## 8. Model Selection

One advantage of employing the proposed two-staged algorithm is that it enables model selection, which is not attainable when a minimax approach is used. In this section, we explain how we perform model selection. We focus on the model selection for the second stage, as the conditional density ˆ g from the first stage can be selected via existing methods for model selection for maximum likelihood estimators (e.g. Birg´ e (2006); Cohen and Pennec (2011); Vijaykumar (2021)).

With an MLE-based estimator ˆ g obtained from the first stage in Algorithm 2, we consider model selection using the regularized loss in the second stage, with theoretical guarantees in the ‖ · ‖ 2 metric. More concretely, given a choice of M candidate models { h 1 , . . . , h M } and a validation dataset { X ′ i , Y ′ i , Z ′ i } n i =1 (distinct from the one used for training models { h i } and ˆ g ), the goal is for the final output of the model selection algorithm to achieve oracle rates with respect to the minimal misspecification error.

We present our algorithm in Algorithm 2. We provide two options for model selection: BestERM and Convex-ERM. Best-ERM selects the model that minimizes the regularized loss on a validation set, while Convex-ERM constructs a convex aggregate of the candidate models that minimizes the regularized loss on a validation set.

## Algorithm 2 Model Selection for Regularized Deep IV

Require: Validation dataset { X ′ i , Y ′ i , Z ′ i } i ∈ [ n ] , M candidate models { h i } i M =1 , a regularization hyperparameter α ∈ R &gt; 0 , an estimator ˆ g , which can obtained by MLE with standard model selection procedure in Birg´ e (2006); Cohen and Pennec (2011).

- 1: Learn ˆ θ with each of the followings:

<!-- formula-not-decoded -->

where h θ = ∑ M j =1 θ i h i , ∑ M j =1 θ j = 1 , θ j ≥ 0 , ˆ T f ( Z ) = E x ∼ ˆ g ( X | Z ) [ f ( X )] and E n [ · ] is defined for { X ′ i , Y ′ i , Z ′ i } i ∈ [ n ] . output h ˆ θ .

<!-- formula-not-decoded -->

Theorem 13 (Model Selection Rates) Consider the model selection problem given M candidate models with any choice of α , over M function classes {H 1 , . . . , H M } . Suppose Assumption 4 and 6 hold, and there exists g † ∈ G and h † j ∈ H j for all j such that ‖ h 0 -h † j ‖ 2 ≤ /epsilon1 H j and E [ ∫ X ( g † ( x | Z ) -g 0 ( x | Z )) 2 dµ ( x ) ] ≤ /epsilon1 G . Assume that Y is almost surely bounded by C Y , each candidate model h j is uniformly bounded in [ -C H , C H ] almost surely. Let δ n,j = max { δ n, G , δ n, H j , δ n,M } , where δ n,M denotes the critical radius of the convex hull over M variables for Best-ERM (i.e. δ n,M = log( M ) n ), and the critical radius of the set of M candidate functions for Convex-ERM (i.e. δ n,M = M n ).

<!-- formula-not-decoded -->

With probability 1 -c 1 exp( c 2 n ∑ M j δ 2 n,j ) , the output of Convex-ERM or Best-ERM ˆ θ , satisfies:

We explain its implications. Most importantly, our obtained rate is the best (i.e., oracle rate) among rates when invoking a result of (convergence result for RDIV in Theorem 12 with misspecified model) for each function class H i . Some astute readers might wonder whether we can just invoke Theorem 12 by making new function classes H best := { h θ : θ = e 1 , . . . , e M } or H conv := { h θ : ∑ j θ j = 1 , θ j ≥ 0 } , and bound the misspecification error /epsilon1 H conv or /epsilon1 H best by ‖ h j -h 0 ‖ will lead to a slower rate with an extra factor of 1 α . The key is only to handle the misspecification error once to avoid the 1 α factor by deferring the invocation of strong convexity and working with the excess risk (difference in the expected loss) instead of the L 2 difference.

## 9. Extension to Iterative Version

One drawback of the result so far is its lack of adaptability to the degree of ill-posedness in the inverse problem, especially for larger values of β corresponding to milder problems, when β ≥ 2 . To address this issue, in this section, we further generalize our results in Section 5 and 6, and propose an iterated Regularized Deep method, which is summarized in Algorithm 3. In this algorithm, instead of targeting (3), we target h m, ∗ , which is given by the following recursive least square

## Algorithm 3 Iterative Regularized Deep IV

Require: Dataset { X i , Y i , Z i } i ∈ [ n ] , function class G , function class H , ˆ h -1 = 0

- 2: for m = 1 , 2 , · · · , M do
- 1: Learn ˆ g ( x | z ) by MLE (4)
- 3: Learn ˆ h m by iterative Tikhonov estimator as the following:

4: end for output ˆ h M

<!-- formula-not-decoded -->

regression with Tikhonov regularization:

<!-- formula-not-decoded -->

and we set h -1 , ∗ = 0 . This is the recursive version of the previous regularized objective in Equation (3), by using Tikhonov regularization around a prior target h m -1 , ∗ instead of 0 . Then, with the learned conditional density ˆ g by MLE in Equation (4), we construct an estimator in (12) by replacing expectation and an operator T with empirical approximation and the learned operator ˆ T , respectively, in Equation (11).

Now, we delve into estimating the finite sample convergence rate of Algorithm 3. Our findings are summarized in the following theorem.

Theorem 14 ( L 2 convergence rate for iterative MLE estimator) Suppose Assumption 4, 5, 6 hold. Let ‖ Y ‖ ∞ ≤ C Y , ‖ h ‖ ∞ ≤ C H holds for all h ∈ H , ‖ g ‖ ∞ ≤ C G holds for all g ∈ G . By setting α = δ 2 2+min { β, 2 m } n , with probability at least 1 -c 1 m exp( c 2 nδ 2 n ) , we have

<!-- formula-not-decoded -->

here δ n has the same definition in Theorem 7.

Importantly, we can have a rate O ( δ 2 β 2+ β n ) in relatively mild conditions while the previous Theorem 7 (non-iteratie version) can only allow for O ( δ 2 min( β, 2) 2+2min( β, 2) n ) , and cannot fully leverage the wellposedness of h 0 , illustrated by the source condtion β . Indeed, if we choose the iteration number m = /ceilingleft min { β/ 2 , log log(1 /δ n ) }/ceilingright , then we get a rate of

<!-- formula-not-decoded -->

Hence for any constant β , as n grows, eventually log log 1 /δ n ≥ β , and we get the rate of O ( δ 2 β 2+ β n ) . This rate can be achieved even if β grows with n , as long as it grows slower than O (log log 1 /δ n ) . If δ n = O ( n -ι ) for some ι &gt; 0 , e.g. RKHS or first order Sobolev space (Wainwright, 2019, Chapt 14.1.2), then we note that we can set m = /ceilingleft min { β/ 2 , √ log(1 /δ n ) }/ceilingright , and 16 √ log(1 /δ n ) = O ( n /epsilon1 )

Figure 1: A typical causal diagram for negative controls. The dashed edges may be absent, and the dashed circle around S ′ indicates that U is unobserved.

![Image](data:image/png;base64,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)

for any /epsilon1 &gt; 0 , thus we still obtain a rate of O ( δ 2 β 2+ β n ) when √ log(1 /δ n ) ≥ β/ 2 . In such a case, we can obtain a O ( δ 2 β 2+ β n ) rate even β grows with n , as long as it grows slower than log(1 /δ n ) .

√ Our results for the iterative estimator match the state-of-the-art convergence rate with respect to L 2 norm for an iterative estimator in Bennett et al. (2023b). However, their method requires a minimax computation oracle, while our method does not.

## 10. Numerical Experiments

In this section, we evaluate our proposal by numerical simulation. In particular, we present the performance of RDIV when we use neural networks as the function approximator and the validity of the proposed model selection procedure. We show that with model selection, our method can achieve state-of-the-art performance in a wide range of data-generating processes.

## 10.1. Experimental Settings

Experiment Design. In our experiment, we test our method on a synthetic dataset. We adjust the data generating process (DGP) for proximal causal inference used in Cui et al. (2020); Miao et al. (2018); Deaner (2021). Concretely, we generate multi-dimensional variables U ′ , S ′ , W ′ , Q ′ , A , where U is an unobserved confounder, S ′ ∈ d S is the observed covariate, W ′ ∈ d W is the negative control outcomes, Q ′ ∈ d Q is the negative control actions, and A is the selected treatment, as described in Figure 1. We left the detailed generation process in Appendix I. For a detailed understanding of this setup, we refer the reader to Section 2 of Kallus et al. (2021). It is well known that there exists a bridge function h ′ 0 such that the following moment condition holds (Cui et al., 2020; Kallus et al., 2021):

<!-- formula-not-decoded -->

which allows the concrete form of (1). To introduce nonlinearity, we transform ( S ′ , W ′ , Q ′ ) into ( S, W, Q ) via S = g ( S ′ ) , W = g ( W ′ ) , Q = g ( Q ′ ) , where g ( · ) is a nonlinear invertible function applied elementwise to S ′ , W ′ , Q ′ respectively. We consider several forms of g ( · ) , including identity, polynomial, sigmoid design, and exponential function. In the final data, we only observe ( S, W, Q ) but not ( S ′ , W ′ , Q ′ ) . Here we use 6 different g ( · ) : Id ( t ) = t , Poly ( t ) = t 3 ,

LogSigmoid ( t ) = log(1 + | 16 ∗ x -8 | ) · sign( x ) , Piecewise ( t ) = 3( x -2)1 x ≤ 1 +log(8 x -8)1 x ≥ 1 , Sigmoid ( t ) = 5 1+exp( -0 . 1 ∗ x ) and CubicRoot = x 1 / 3 .

Methods to compare. In this experiment, our goal is to estimate the counterfactual mean parameter E [ Y (1)] , which is unique as long as (1) holds. We learn h 0 in (1) by RDIV, which corresponds to the procedure in Algorithm 1 with MLE for conditional density estimation. We show results for different values for α ∈ { 0 . 01 , 0 . 1 } , and compare the performance of our approach to that of several different methods, including KernelIV (Singh et al., 2019), DeepIV (Hartford et al., 2017), DeepFeatureIV (Xu et al., 2021), and AGMM (Dikkala et al., 2020a). Note that DeepIV can be viewed as a special case of our methods, with α fixed to be 0. In the first stage of our algorithm, we use a three-layer mixture density network (Hartford et al., 2017; Rothfuss et al., 2019) as the approximator of the conditional density. In the second stage, we use a three-layer fully-connected neural network as the approximators for RDIV, DeepIV, AGMM, and DFIV. We present the results of our method and its comparison with previous benchmarks in terms of MSE normalized by the true estimand value in Table 2-5. Every estimate is calculated by 100 random replications. The confidence interval is calculated by 2 times the standard deviation.

Hyperparameter settings. For RDIV, we use Adam as the optimizer for both density estimation and Tikhonov regression, with a default learning rate of 10 -4 , a batch size of 50 , and a training epoch of 300 . We will show how to choose these hyperparameters with our model selection procedure (Algorithm 2) in Section 10.3. For all baselines except for AGMM, we adapt the hyperparameters in their original codebase. For AGMM, we tune the learning rate for the learner and adversary for every g ( · ) independently. We follow Singh et al. (2019) to use Gaussian RKHS for function approximation and their method for tuning the regularization parameter. When n = 500 , the learning rate of the learner and adversary in AGMM are manually set to 10 -4 for LogSigmoid, Piecewise, and Sigmoid, and 10 -3 for Id, Poly, and CubicRoot. When n = 1000 , the learning rate of the learner and adversary in AGMM are manually set to 10 -4 for Piecewise and Sigmoid, and 10 -3 for LogSigmoid, Piecewise, and CubicRoot. The training parameter of DFIV is adopted from Xu et al. (2021). Note that tuning DFIV is highly intractable in practice, as their method is essentially a bilevel optimization, which is known to be hard to solve (Hong et al., 2023).

## 10.2. Results

First, we can observe that although our estimator resembles DeepIV, the later fix α = 0 in (12), RDIV outperforms DeepIV for all g ( · ) . This is due to the nonzero regularization term, which improves the performance of our estimator by a better tradeoff between bias and variance. Second, in most cases, AGMM and DFIV are outperformed by algorithms that only need single-level optimization (RDIV, KernelIV, DeepIV). This would be because, in these methods, optimization of the loss function is much harder, which results in the inaccuracy of estimators. Thirdly, while it is seen that Kernel IV is comparable to RDIV in some scenarios such as in Table 4 and 5, in the next section, we will show that our RDIV equipped with a model selection procedure can generally outperform KernelIV.

## 10.3. Model selection

We also report our results in model selection for the second stage by implementing Best-ERM in Algorithm 2 and demonstrate how it improves our results. Specifically, our models h 1 , . . . , h M are

## REGULARIZED DEEPIV WITH MODEL SELECTION

Table 2: E [ Y (1)] : d S = d Q = 15 , d W = 1 , n 1 = 500 .

| g ( t )          | RDIV ( α = 0 . 01 )   | RDIV ( α = 0 . 1 )   | KernelIV        | DeepIV          | DFIV            | AGMM            |
|------------------|-----------------------|----------------------|-----------------|-----------------|-----------------|-----------------|
| Id ( t )         | 0.0077 ± 0.0012       | 0.0021 ± 0.0007      | 0.0193 ± 0.0018 | 0.0089 ± 0.0015 | 0.1069 ± 0.0218 | 0.0198 ± 0.0011 |
| Poly ( t )       | 0.0150 ± 0.0057       | 0.0904 ± 0.0202      | 0.0439 ± 0.0062 | 0.0887 ± 0.0276 | 0.0920 ± 0.0046 | 0.0453 ± 0.0023 |
| LogSigmoid ( t ) | 0.0094 ± 0.0013       | 0.0022 ± 0.0009      | 0.0031 ± 0.0008 | 0.0152 ± 0.0026 | 0.1444 ± 0.0080 | 0.0042 ± 0.0010 |
| Piecewise ( t )  | 0.0070 ± 0.0017       | 0.0024 ± 0.0009      | 0.0041 ± 0.0012 | 0.0076 ± 0.0012 | 0.0150 ± 0.0026 | 0.0128 ± 0.0024 |
| Sigmoid ( t )    | 0.0206 ± 0.0026       | 0.0021 ± 0.0006      | 0.0380 ± 0.0025 | 0.0278 ± 0.0025 | 0.1846 ± 0.0092 | 0.0070 ± 0.0014 |
| CubicRoot ( t )  | 0.0095 ± 0.0014       | 0.0024 ± 0.0007      | 0.0511 ± 0.0039 | 0.0161 ± 0.0018 | 0.1357 ± 0.0200 | 0.0536 ± 0.0021 |

Table 3: d S = d Q = 15 , d W = 1 , n 1 = 1000 .

| g ( t )          | RDIV ( α = 0 . 01 )   | RDIV ( α = 0 . 1 )   | KernelIV        | DeepIV          | DFIV            | AGMM            |
|------------------|-----------------------|----------------------|-----------------|-----------------|-----------------|-----------------|
| Id ( t )         | 0.0106 ± 0.0013       | 0.0014 ± 0.0003      | 0.0145 ± 0.0013 | 0.0128 ± 0.0015 | 0.1162 ± 0.0052 | 0.0217 ± 0.0135 |
| Poly ( t )       | 0.0164 ± 0.0020       | 0.0037 ± 0.0027      | 0.0396 ± 0.0038 | 0.0182 ± 0.0023 | 0.1256 ± 0.0044 | 0.0054 ± 0.0031 |
| LogSigmoid ( t ) | 0.0078 ± 0.0009       | 0.0009 ± 0.0003      | 0.0259 ± 0.0023 | 0.0262 ± 0.0023 | 0.1618 ± 0.0482 | 0.0053 ± 0.0010 |
| Piecewise ( t )  | 0.0017 ± 0.0004       | 0.0059 ± 0.0008      | 0.0080 ± 0.0008 | 0.0019 ± 0.0005 | 0.1623 ± 0.0674 | 0.0014 ± 0.0011 |
| Sigmoid ( t )    | 0.0077 ± 0.0016       | 0.0082 ± 0.0023      | 0.0311 ± 0.0014 | 0.0110 ± 0.0019 | 0.2085 ± 0.0443 | 0.0296 ± 0.0023 |
| CubicRoot ( t )  | 0.0254 ± 0.0021       | 0.0048 ± 0.0008      | 0.0459 ± 0.0024 | 0.0248 ± 0.0022 | 0.1401 ± 0.0047 | 0.0650 ± 0.0035 |

Table 4: d S = d Q = 20 , d W = 10 , n 1 = 500 .

| g ( t )          | RDIV ( α = 0 . 01 )   | RDIV ( α = 0 . 1 )   | KernelIV        | DeepIV          | DFIV            | AGMM            |
|------------------|-----------------------|----------------------|-----------------|-----------------|-----------------|-----------------|
| Id ( t )         | 0.0272 ± 0.0022       | 0.0055 ± 0.0009      | 0.0088 ± 0.0016 | 0.0364 ± 0.0025 | 0.0291 ± 0.0060 | 0.3291 ± 0.0115 |
| Poly ( t )       | 0.0067 ± 0.0016       | 0.0230 ± 0.0051      | 0.0697 ± 0.0041 | 0.0263 ± 0.0050 | 0.0997 ± 0.0046 | 0.0409 ± 0.0225 |
| LogSigmoid ( t ) | 0.0905 ± 0.0058       | 0.0525 ± 0.0054      | 0.0335 ± 0.0014 | 0.0960 ± 0.0066 | 0.2059 ± 0.0826 | 0.0218 ± 0.0027 |
| Piecewise ( t )  | 0.0305 ± 0.0043       | 0.0104 ± 0.0021      | 0.0359 ± 0.0010 | 0.0225 ± 0.0031 | 0.7626 ± 0.9996 | 0.0136 ± 0.0010 |
| Sigmoid ( t )    | 0.1481 ± 0.0083       | 0.0106 ± 0.0028      | 0.0018 ± 0.0004 | 0.1983 ± 0.0117 | 0.3545 ± 0.0494 | 0.0307 ± 0.0195 |
| CubicRoot ( t )  | 0.0810 ± 0.0039       | 0.0288 ± 0.0025      | 0.0021 ± 0.0004 | 0.0949 ± 0.0050 | 0.0956 ± 0.0453 | 0.3461 ± 0.0121 |

Table 5: d S = d Q = 20 , d W = 10 , n 1 = 1000 .

| g ( t )          | RDIV ( α = 0 . 01 )   | RDIV ( α = 0 . 1 )   | KernelIV        | DeepIV          | DFIV            | AGMM            |
|------------------|-----------------------|----------------------|-----------------|-----------------|-----------------|-----------------|
| Id ( t )         | 0.0652 ± 0.0035       | 0.0269 ± 0.0020      | 0.0009 ± 0.0002 | 0.0639 ± 0.0033 | 0.1442 ± 0.2461 | 0.1321 ± 0.0029 |
| Poly ( t )       | 0.0861 ± 0.0076       | 0.0224 ± 0.0034      | 0.0465 ± 0.0021 | 0.1148 ± 0.0082 | 0.0951 ± 0.0031 | 0.1796 ± 0.0023 |
| LogSigmoid ( t ) | 0.0649 ± 0.0046       | 0.0280 ± 0.0025      | 0.0197 ± 0.0014 | 0.0759 ± 0.0045 | 0.2949 ± 0.2917 | 0.0247 ± 0.0013 |
| Piecewise ( t )  | 0.0039 ± 0.0008       | 0.0037 ± 0.0006      | 0.0215 ± 0.0006 | 0.0065 ± 0.0012 | 0.5442 ± 0.4784 | 0.0133 ± 0.0009 |
| Sigmoid ( t )    | 0.1112 ± 0.0053       | 0.0091 ± 0.0028      | 0.0037 ± 0.0005 | 0.1493 ± 0.0058 | 0.3332 ± 0.0652 | 0.0650 ± 0.0029 |
| CubicRoot ( t )  | 0.0990 ± 0.0042       | 0.0802 ± 0.0046      | 0.0021 ± 0.0004 | 0.1070 ± 0.0043 | 0.0956 ± 0.0453 | 0.3461 ± 0.0121 |

Table 6: Model selection results based on Best ERM. The left tabular is generated from a data size of n 1 = 500 , while the right tabular is generated from a dataset with n 1 = 1000 . Both datasets satisfies d S = d Q = 20 , d W = 10 .

| g ( t )          | RDIV ( α = 0 . 01 )   | RDIV ( α = 0 . 1 )   | KernelIV        | RDIV ( α = 0 . 01 )   | RDIV ( α = 0 . 1 )   | KernelIV        |
|------------------|-----------------------|----------------------|-----------------|-----------------------|----------------------|-----------------|
| Id ( t )         | 0.0017 ± 0.0017       | 0.0047 ± 0.0021      | 0.0088 ± 0.0016 | 0.0102 ± 0.0028       | 0.0014 ± 0.0009      | 0.0009 ± 0.0002 |
| Poly ( t )       | 0.0032 ± 0.0024       | 0.0272 ± 0.0097      | 0.0697 ± 0.0041 | 0.0313 ± 0.0137       | 0.0049 ± 0.0026      | 0.0465 ± 0.0021 |
| LogSigmoid ( t ) | 0.0121 ± 0.0055       | 0.0019 ± 0.0007      | 0.0335 ± 0.0014 | 0.0078 ± 0.0020       | 0.0008 ± 0.0004      | 0.0197 ± 0.0014 |
| Piecewise ( t )  | 0.0159 ± 0.0121       | 0.0020 ± 0.0019      | 0.0359 ± 0.0010 | 0.0024 ± 0.0013       | 0.0034 ± 0.0027      | 0.0215 ± 0.0006 |
| Sigmoid ( t )    | 0.1655 ± 0.0144       | 0.0937 ± 0.0174      | 0.0018 ± 0.0004 | 0.1538 ± 0.0078       | 0.0863 ± 0.0187      | 0.0037 ± 0.0005 |
| CubicRoot ( t )  | 0.0034 ± 0.0017       | 0.0019 ± 0.0021      | 0.0021 ± 0.0004 | 0.0148 ± 0.0048       | 0.0036 ± 0.0035      | 0.0021 ± 0.0004 |

trained by different hyperparameters. First, we employ model selection for the density function by Best ERM. Then with the trained density function in the first stage, we further apply Best ERM to the models in the second stage. In the model selection experiments, we fix the dimension of our dataset to be d S = d Q = 20 , d W = 10 . We compute the mean and confidence interval with 10 independent trials. We set the candidate training parameters as follows: the number of epochs ∈ { 300 , 400 } , the batch size for the 1st stage ∈ { 30 , 50 } and the batch size for the 2nd stage ∈ { 50 , 60 , 100 } , the learning rate ∈ { 10 -4 , 10 -3 } , the number of mixture components ∈ { 40 , 50 , 60 } . As shown in Table 6, when RDIV is equipped with model selection techniques, our method outperforms KernelIV in all but one case when the dataset size is 500, and outperforms KernelIV in 3 out of 6 settings when the dataset size is 1000. Our approach demonstrates its effectiveness by outperforming previous benchmarks across a diverse set of Data Generating Processes (DGP). This achievement is attributed to both the ease of optimization of RDIV and its theoretically sound integration with model selection procedures.

## 11. Conclusion

In this paper, we study NPIV regression with general function approximation. We propose a new estimator defined by the loss that organically combines Tikhonov regularization and an MLE estimator, namely the Regularized DeepIV (RDIV). We show that our estimator converges to the least norm solution, and derive its convergence rate. Notably, our method does not rely on uniqueness or minimax computation oracle. We further illustrate that our method can be incorporated into model selection and show that our procedure can achieve the oracle rate with respect to the minimal model misspecification error. When extended to an iterative estimator, our method achieves the state-ofthe-art convergence rate. Moreover, we justify our method through numerical simulations. Our experiments show that RDIV outperforms existing benchmarks in a wide range of circumstances.

## References

- Chunrong Ai and Xiaohong Chen. Efficient estimation of models with conditional moment restrictions containing unknown functions. Econometrica , 71(6):1795-1843, 2003.
- Chunrong Ai and Xiaohong Chen. Estimation of possibly misspecified semiparametric conditional moment restriction models with different conditioning variables. Journal of Econometrics , 141 (1):5-43, 2007. ISSN 0304-4076. doi: https://doi.org/10.1016/j.jeconom.2007.01.013. URL https://www.sciencedirect.com/science/article/pii/S0304407607000061 . Semiparametric methods in econometrics.
- Donald Andrews and James H Stock. Inference with weak instruments. 2005.
- Isaiah Andrews, James H Stock, and Liyang Sun. Weak instruments in instrumental variables regression: Theory and practice. Annual Review of Economics , 11:727-753, 2019.
- Joshua Angrist and Guido Imbens. Identification and estimation of local average treatment effects, 1995.
- Anas Barakat and Pascal Bianchi. Convergence and dynamical behavior of the adam algorithm for nonconvex stochastic optimization. SIAM Journal on Optimization , 31(1):244-274, 2021.
- Peter L Bartlett, St´ ephane Boucheron, and G´ abor Lugosi. Model selection and error estimation. Machine Learning , 48:85-113, 2002a.
- Peter L Bartlett, Olivier Bousquet, and Shahar Mendelson. Localized rademacher complexities. In International Conference on Computational Learning Theory , pages 44-58. Springer, 2002b.
- Andrew Bennett and Nathan Kallus. The variational method of moments. arXiv preprint arXiv:2012.09422 , 2020.
- Andrew Bennett, Nathan Kallus, Xiaojie Mao, Whitney Newey, Vasilis Syrgkanis, and Masatoshi Uehara. Minimax instrumental variable regression and l 2 convergence guarantees without identification or closedness, 2023a.
- Andrew Bennett, Nathan Kallus, Xiaojie Mao, Whitney Newey, Vasilis Syrgkanis, and Masatoshi Uehara. Source condition double robust inference on functionals of inverse problems, 2023b.
- Lucien Birg´ e. Model selection via testing: an alternative to (penalized) maximum likelihood estimators. In Annales de l'IHP Probabilit´ es et statistiques , volume 42, pages 273-325, 2006.
- Marine Carrasco, Jean-Pierre Florens, and Eric Renault. Linear inverse problems in structural econometrics estimation based on spectral decomposition and regularization. Handbook of econometrics , 6:5633-5751, 2007.
- Laurent Cavalier. Inverse problems in statistics. In Inverse Problems and High-Dimensional Estimation: Stats in the Chˆ ateau Summer School, August 31-September 4, 2009 , pages 3-96. Springer, 2011.

- Gavin C Cawley and Nicola LC Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. The Journal of Machine Learning Research , 11:2079-2107, 2010.
- Minshuo Chen, Haoming Jiang, Wenjing Liao, and Tuo Zhao. Nonparametric regression on lowdimensional manifolds using deep relu networks : Function approximation and statistical recovery, 2022.
- Qihui Chen. Robust and optimal estimation for partially linear instrumental variables models with partial identification. Journal of Econometrics , 221(2):368-380, 2021.
- Xiangyi Chen, Sijia Liu, Ruoyu Sun, and Mingyi Hong. On the convergence of a class of adam-type algorithms for non-convex optimization. arXiv preprint arXiv:1808.02941 , 2018.
- Xiaohong Chen. Large sample sieve estimation of semi-nonparametric models. Handbook of econometrics , 6:5549-5632, 2007.
- Xiaohong Chen and Demian Pouzo. Estimation of nonparametric conditional moment models with possibly nonsmooth generalized residuals. Econometrica , 80(1):277-321, 2012.
- Victor Chernozhukov, Whitney Newey, and Vira Semenova. Inference on average welfare with high-dimensional state space. 2019.
- Serge Cohen and Erwan Le Pennec. Conditional density estimation by penalized likelihood model selection and applications. arXiv preprint arXiv:1103.2021 , 2011.
- Yifan Cui, Hongming Pu, Xu Shi, Wang Miao, and Eric Tchetgen Tchetgen. Semiparametric proximal causal inference. arXiv preprint arXiv:2011.08411 , 2020.
- Serge Darolles, Yanqin Fan, Jean-Pierre Florens, and Eric Renault. Nonparametric instrumental regression. Econometrica , 79(5):1541-1565, 2011.
- Ben Deaner. Proxy controls and panel data. arXiv preprint arXiv:1810.00283 , 2018.
- Ben Deaner. Many proxy controls. arXiv preprint arXiv:2110.03973 , 2021.
- Jelena Diakonikolas, Constantinos Daskalakis, and Michael I Jordan. Efficient methods for structured nonconvex-nonconcave min-max optimization. In International Conference on Artificial Intelligence and Statistics , pages 2746-2754. PMLR, 2021.
- Nishanth Dikkala, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. Minimax estimation of conditional moment models. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 12248-12262, 2020a.
- Nishanth Dikkala, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. Minimax estimation of conditional moment models. Advances in Neural Information Processing Systems , 33:12248-12262, 2020b.
- Simon Shaolei Du. Gradient descent for non-convex problems in modern machine learning . PhD thesis, Department of Energy award DEAR0000596, Department of the Interior award . . . , 2019.

- Frank Emmert-Streib and Matthias Dehmer. Evaluation of regression models: Model assessment, model selection and generalization error. Machine learning and knowledge extraction , 1(1):521551, 2019.
- Heinz Werner Engl, Martin Hanke, and Andreas Neubauer. Regularization of inverse problems , volume 375. Springer Science &amp; Business Media, 1996.
- Jean-Pierre Florens, Jan Johannes, and S´ ebastien Van Bellegem. Identification and estimation by penalization in nonparametric instrumental regression. Econometric Theory , 27(3):472-496, 2011.
- Dylan J Foster and Vasilis Syrgkanis. Orthogonal statistical learning. arXiv preprint arXiv:1901.09036 , 2019.
- Carl Gold and Peter Sollich. Model selection for support vector machine classification. Neurocomputing , 55(1-2):221-249, 2003.
- Zhishuai Guo, Quanqi Hu, Lijun Zhang, and Tianbao Yang. Randomized stochastic variancereduced methods for multi-task stochastic bilevel optimization. arXiv preprint arXiv:2105.02266 , 2021.
- Isabelle Guyon, Amir Saffari, Gideon Dror, and Gavin Cawley. Model selection: beyond the bayesian/frequentist divide. Journal of Machine Learning Research , 11(1), 2010.
- Peter Hall and Joel L Horowitz. Nonparametric methods for inference in the presence of instrumental variables. 2005.
- Jason Hartford, Greg Lewis, Kevin Leyton-Brown, and Matt Taddy. Deep iv: A flexible approach for counterfactual prediction. In International Conference on Machine Learning , pages 1414-1423. PMLR, 2017.
- Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A two-timescale stochastic algorithm framework for bilevel optimization: Complexity analysis and application to actor-critic. SIAM Journal on Optimization , 33(1):147-180, 2023.
- Joel L Horowitz. Asymptotic normality of a nonparametric instrumental variables estimator. International Economic Review , 48(4):1329-1349, 2007.
- Kazufumi Ito and Bangti Jin. Inverse problems: Tikhonov theory and algorithms , volume 22. World Scientific, 2014.
- Chi Jin, Praneeth Netrapalli, Rong Ge, Sham M. Kakade, and Michael I. Jordan. On nonconvex optimization for machine learning: Gradients, stochasticity, and saddle points, 2019.
- Chi Jin, Praneeth Netrapalli, and Michael Jordan. What is local optimality in nonconvexnonconcave minimax optimization? In International conference on machine learning , pages 4880-4889. PMLR, 2020.
- Nathan Kallus, Xiaojie Mao, and Masatoshi Uehara. Causal inference under unmeasured confounding with negative controls: A minimax learning approach. arXiv preprint arXiv:2103.14029 , 2021.

- Nathan Kallus, Xiaojie Mao, and Masatoshi Uehara. Causal inference under unmeasured confounding with negative controls: A minimax learning approach, 2022.
- Myrto Kalouptsidi, Paul T Scott, and Eduardo Souza-Rodrigues. Linear iv regression estimators for structural dynamic discrete choice models. Journal of Econometrics , 222(1):778-804, 2021.
- Masahiro Kato, Masaaki Imaizumi, Kenichiro McAlinn, Haruo Kakehi, and Shota Yasui. Learning causal models from conditional moment restrictions by importance weighting. arXiv preprint arXiv:2108.01312 , 2021.
- Prashant Khanduri, Siliang Zeng, Mingyi Hong, Hoi-To Wai, Zhaoran Wang, and Zhuoran Yang. A near-optimal algorithm for stochastic bilevel optimization via double-momentum. Advances in neural information processing systems , 34:30271-30283, 2021.
- Guillaume Lecu´ e. Empirical risk minimization is optimal for the convex aggregation problem. Bernoulli , 19(5B):2153 -2166, 2013. doi: 10.3150/12-BEJ447. URL https://doi.org/10.3150/12-BEJ447 .
- Guillaume Lecu´ e and Shahar Mendelson. Aggregation via empirical risk minimization. Probability theory and related fields , 145(3-4):591-613, 2009.
- Guillaume Lecu´ e and Shahar Mendelson. Performance of empirical risk minimization in linear aggregation. arXiv e-prints , art. arXiv:1402.5763, February 2014. doi: 10.48550/arXiv.1402. 5763.
- Guillaume Lecu´ e and Philippe Rigollet. Optimal learning with q-aggregation. 2014.
- Greg Lewis and Vasilis Syrgkanis. Adversarial generalized method of moments, 2018.
- Luofeng Liao, You-Lin Chen, Zhuoran Yang, Bo Dai, Mladen Kolar, and Zhaoran Wang. Provably efficient neural estimation of structural equation models: An adversarial approach. Advances in Neural Information Processing Systems , 33:8947-8958, 2020a.
- Luofeng Liao, Zuyue Fu, Zhuoran Yang, Yixin Wang, Mladen Kolar, and Zhaoran Wang. Instrumental variable value iteration for causal offline reinforcement learning. arXiv preprint arXiv:2102.09907 , 2021.
- Peng Liao, Zhengling Qi, and Susan Murphy. Batch policy learning in average reward markov decision processes. arXiv preprint arXiv:2007.11771 , 2020b.
- Tianyi Lin, Chi Jin, and Michael Jordan. On gradient descent ascent for nonconvex-concave minimax problems. In International Conference on Machine Learning , pages 6083-6093. PMLR, 2020a.
- Tianyi Lin, Chi Jin, and Michael I Jordan. Near-optimal algorithms for minimax optimization. In Conference on Learning Theory , pages 2738-2779. PMLR, 2020b.
- David A McAllester. Pac-bayesian stochastic model selection. Machine Learning , 51:5-21, 2003.
- Shahar Mendelson and Joseph Neeman. Regularization in kernel learning. 2010.

- Wang Miao, Xu Shi, and Eric Tchetgen Tchetgen. A confounding bridge approach for double negative control inference on causal effects. arXiv preprint arXiv:1808.04945 , 2018.
- Charles Mitchell and Sara van de Geer. General oracle inequalities for model selection. Electronic Journal of Statistics , 3(none):176 -204, 2009. doi: 10.1214/08-EJS254. URL https://doi.org/10.1214/08-EJS254 .
- Whitney K Newey and James L Powell. Instrumental variable estimation of nonparametric models. Econometrica , 71(5):1565-1578, 2003.
- Sebastian Raschka. Model evaluation, model selection, and algorithm selection in machine learning. arXiv preprint arXiv:1811.12808 , 2018.
- Meisam Razaviyayn, Tianjian Huang, Songtao Lu, Maher Nouiehed, Maziar Sanjabi, and Mingyi Hong. Nonconvex min-max optimization: Applications, challenges, and recent theoretical advances. IEEE Signal Processing Magazine , 37(5):55-66, 2020.
- Jonas Rothfuss, Fabio Ferreira, Simon Walther, and Maxim Ulrich. Conditional density estimation with neural networks: Best practices and benchmarks, 2019.
- Johannes Schmidt-Hieber. Deep relu network approximation of functions on a manifold. arXiv preprint arXiv:1908.00695 , 2019.
- Johannes Schmidt-Hieber. Nonparametric regression using deep neural networks with relu activation function. 2020.
- Chengchun Shi, Masatoshi Uehara, Jiawei Huang, and Nan Jiang. A minimax learning approach to off-policy evaluation in confounded partially observable markov decision processes. In International Conference on Machine Learning , pages 20057-20094. PMLR, 2022.
- Rahul Singh. Kernel methods for unobserved confounding: Negative controls, proxies, and instruments. arXiv preprint arXiv:2012.10315 , 2020.
- Rahul Singh, Maneesh Sahani, and Arthur Gretton. Kernel instrumental variable regression. Advances in Neural Information Processing Systems , 32, 2019.
- Masatoshi Uehara, Haruka Kiyohara, Andrew Bennett, Victor Chernozhukov, Nan Jiang, Nathan Kallus, Chengchun Shi, and Wen Sun. Future-dependent value-based off-policy evaluation in pomdps. arXiv preprint arXiv:2207.13081 , 2022a.
- Masatoshi Uehara, Ayush Sekhari, Jason D Lee, Nathan Kallus, and Wen Sun. Provably efficient reinforcement learning in partially observable dynamical systems. Advances in Neural Information Processing Systems , 35:578-592, 2022b.
- Aad W van der Vaart, Sandrine Dudoit, and Mark J van der Laan. Oracle inequalities for multi-fold cross validation. Statistics &amp; Decisions , 24(3):351-371, 2006.
- Sara Van de Geer. Hellinger-consistency of certain nonparametric maximum likelihood estimators. The Annals of Statistics , 21(1):14-44, 1993.

- Suhas Vijaykumar. Localization, convexity, and star aggregation. Advances in Neural Information Processing Systems , 34:4570-4581, 2021.
- Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint , volume 48. Cambridge university press, 2019.
- Lingxiao Wang, Zhuoran Yang, and Zhaoran Wang. Provably efficient causal reinforcement learning with confounded observational data. Advances in Neural Information Processing Systems , 34: 21164-21175, 2021.
- Sheng Wang, Jun Shao, and Jae Kwang Kim. An instrumental variable approach for identification and estimation with nonignorable nonresponse. Statistica Sinica , pages 1097-1116, 2014.
- Rachel Ward, Xiaoxia Wu, and Leon Bottou. Adagrad stepsizes: Sharp convergence over nonconvex landscapes. The Journal of Machine Learning Research , 21(1):9047-9076, 2020.
- Xiaoxia Wu, Simon S Du, and Rachel Ward. Global convergence of adaptive gradient methods for an over-parameterized neural network. arXiv preprint arXiv:1902.07111 , 2019.
- Liyuan Xu, Heishiro Kanagawa, and Arthur Gretton. Deep proxy causal learning and its application to confounded bandit policy evaluation. arXiv preprint arXiv:2106.03907 , 2021.
- Mengxin Yu, Zhuoran Yang, and Jianqing Fan. Strategic decision-making in the presence of information asymmetry: Provably efficient rl with algorithmic instruments. arXiv preprint arXiv:2208.11040 , 2022.
- Manzil Zaheer, Sashank Reddi, Devendra Sachan, Satyen Kale, and Sanjiv Kumar. Adaptive methods for nonconvex optimization. Advances in neural information processing systems , 31, 2018.
- Rui Zhang, Masaaki Imaizumi, Bernhard Sch¨ olkopf, and Krikamol Muandet. Instrumental variable regression via kernel maximum moment loss. Journal of Causal Inference , 11(1):20220073, 2023.
- Dongruo Zhou, Jinghui Chen, Yuan Cao, Yiqi Tang, Ziyan Yang, and Quanquan Gu. On the convergence of adaptive gradient methods for nonconvex optimization. arXiv preprint arXiv:1808.05671 , 2018.

## Appendix A. Additional Related Works for Model Selection

Model Selection. Under the classical supervised learning setting, a common approach is to perform empirical risk minimization (ERM) on a separate validation set, and choose the candidate model that achieves the smallest risk (Mitchell and van de Geer, 2009), or similarly, through M-fold cross-validation which splits the data into M folds, and evaluates the risk on the different held out set for each model (Vaart et al., 2006). As an alternative to selecting a single model, convex aggregation or linear aggregation is employed to find the best convex/linear combination of models (Lecu´ e, 2013; Lecu´ e and Mendelson, 2014). However, it can be shown that the aforementioned approaches are sub-optimal in the sense that they cannot achieve the optimal log( M ) n rate for the model selection residual. To tackle this challenge, Lecu´ e and Mendelson (2009) proposed a different approach for convex aggregation by first finding a subset of 'almost minimizers' - a subset of the candidate functions that is sufficiently close to the minimizer within the candidates on the validation set, and then finding a best aggregate in the convex hull of this subset. This approach achieves the optimal model selection rates as it performs ERM on a subset that is much smaller than the convex hull of all candidate models, thereby reducing the statistical error. Furthermore, other optimal model selection approaches include the Q-aggregation approach which performs ERM with a modified loss that adds an additional penalty based on individual model performance (Lecu´ e and Rigollet, 2014).

## Appendix B. Results when Using χ 2 -MLE

In this section, we consider another density estimation for the density estimation, the χ 2 -MLE:

<!-- formula-not-decoded -->

## B.1. Finite Sample Results

Although Assumption 6 is widely accepted in previous works, in practice, it often fails to hold when g 0 does not have full support on X . To address this drawback of MLE, in this subsection, we further discuss the finite sample convergence rate of Algorithm 3 when the conditional density estimation is performed by χ 2 -MLE. In this case, the first step estimation procedure is given by Equation (13). Notably, our guarantee does not relate to the lower bound of g 0 ( x | z ) . Our results rely on the following assumption, which characterizes the smoothness of function class H .

Assumption 15 ( γ -Smoothness) For all h -h ′ ∈ H-H , we assume that ‖ h -h ′ ‖ ∞ ≤ ‖ h -h ′ ‖ γ 2 .

Such a relationship is known for instance to hold for Sobolev spaces and more generally for reproducing kernel Hilbert spaces (RKHS) with a polynomial eigendecay. A notable instance is RKHS with eigendevay at a rate of O (1 /j 1 /p ) for some p ∈ (0 , 1) . In that case, Lemma 5.1 of Mendelson and Neeman (2010) shows that γ = 1 -p . For the Gaussian kernel, which has an exponential eigendecay, we can take p arbitrarily close to 0 . We now summarize our result for χ 2 -MLE in the following theorem.

Suppose Assumption 4,5,15 hold.

n -1 2 n

Theorem 16 ( L 2 convergence rate for RMIV with χ 2 -MLE) By setting α = δ 2 2+(2 -γ ) min { β, 2 } , with probability at least 1 c exp( c nδ 2 ) , we have

<!-- formula-not-decoded -->

Here δ n has the same definition in Theorem 7.

The convergence rate of RMIV with χ 2 -MLE depends on the smoothness parameter γ . As γ → 1 , we have ‖ ˆ h -h 0 ‖ 2 2 ≤ O ( δ 2 min { β, 2 } 2+min { β, 2 } n ) , which recovers the rate in Theorem 14. We further discuss the results for χ 2 -MLE based IV regression under misspecification.

Theorem 17 ( L 2 convergence rate for RMIV with χ 2 -MLE under misspecification) Suppose Assumption 4,15 hold, and there exists h † ∈ H and g † ∈ G such that ‖ h 0 -h † ‖ 2 ≤ /epsilon1 H and E [ ∫ X ( g † ( x | Z ) -g 0 ( x | Z )) 2 dµ ( x ) ] ≤ /epsilon1 G . For any 0 &lt; α ≤ 1 ,with probability at least 1 -c 1 exp( c 2 nδ 2 n ) , we have

<!-- formula-not-decoded -->

Here δ n has the same definition in Theorem 7.

Remark 18 We define /epsilon1 := { /epsilon1 G , /epsilon1 2 H } . If /epsilon1 &lt; 1 , then by setting α = ( δ 2 n + /epsilon1 ) 2 2+(2 -γ ) min { β, 1 } , we have

<!-- formula-not-decoded -->

If /epsilon1 ≥ 1 , then by setting α = 1 , we have ‖ ˆ h -h 0 ‖ 2 2 ≤ O ( /epsilon1 1 / (2 -γ ) ) .

## B.2. Results for Model Selection

Theorem 13 is extended when using χ 2 -MLE. Indeed, if Assumption 15 holds and the candidate function are trained with ˆ g estimated using the χ 2 -MLE approach, the output of Convex-ERM or Best-ERM ˆ θ , satisfies

<!-- formula-not-decoded -->

## B.3. Convergence Results for Iterative Version

We further discuss the finite sample convergence rate of Algorithm 3 when the conditional density estimation is performed by χ 2 -MLE. In this case, the first step estimation procedure is given by Equation (13). Notably, in this case, we do not require the ground truth density g 0 to be uniformly lower bounded, which is assumed in Assumption 6 and serves as a prerequisite for MLE convergence. Our results are summarized by the following theorem.

2

n -1 2 n

Theorem 19 ( L 2 convergence rate for iterative χ -MLE estimator) Under Assumption 1,4,5,15, by setting α = δ 2 2+(2 -γ ) min { β, 2 m } , with probability at least 1 c m exp( c nδ 2 ) , we have

<!-- formula-not-decoded -->

Here δ n has the same definition in Theorem 7.

Remark 20 Similar to Section 7, by setting the iteration number m = /ceilingleft min { β/ 2 , log log(1 /δ n ) }/ceilingright , we have

Therefore, for log log δ n ≥ β , eventually we have the rate of O ( δ 2 β 2+(2 -γ ) β n ) . If δ n = O ( n -ι ) , then we can set m = /ceilingleft min { β/ 2 , √ log(1 /δ n ) }/ceilingright to obtain the same rate. Moreover, if γ → 1 , e.g. RKHS with exponential eigenvalue decay (Mendelson and Neeman, 2010, Lemma 5.1), then we recover the rate of O ( δ 2 β 2+ β n ) even without Assumption 6.

<!-- formula-not-decoded -->

## Appendix C. Proof of Theorem 7 and 16

In this section, we prove the convergence rate of non-iterative RMIV. We prove the results of Theorem 7 and 16 respectively. Recall that we define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

by Lemma 31, we have

Therefore, we only need to provide an upper bound for ‖ ˆ h -h ∗ ‖ 2 2 . Westart by proving the following lemma, and with the convergence rate of MLE and χ 2 -MLE, we conclude the proof of Theorem 7 and Theorem 16 respectively.

Lemma 21 With probability at least 1 -c 1 exp( c 2 nδ 2 n, H ) , we have the following inequality:

<!-- formula-not-decoded -->

Proof By the optimality of h ∗ in Eq. (3), we have

<!-- formula-not-decoded -->

where define L ( T h ) := ( Y -T h ) 2 . Recall that

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

here the inequality comes from the uniform boundedness of ˆ h, h ∗ , T h, T ˆ h, ˆ T h, ˆ T ˆ h , and the O (1) -Lipschitz of L ( · ) .

<!-- formula-not-decoded -->

Here, using Lemma 29, the term Emp is upper-bounded as follows with probability at least 1 -c 1 exp( c 2 nδ 2 n, H ) :

<!-- formula-not-decoded -->

Furthermore, recall that by our iteration in (5), we have

<!-- formula-not-decoded -->

Hence, we have

Combining everything, we have

<!-- formula-not-decoded -->

Here the constant c 1 and c 2 hide constants related to C,C 0 . The first inequality comes from (16). We implicitly use α ≤ 1 in the last inequality.

Proof of Theorem 7. By Assumption 5, we have /epsilon1 G = 0 . By Corollary 26 and Lemma 21, since α ≤ 1 we have

<!-- formula-not-decoded -->

holds with probability at least 1 -c exp( nδ 2 n ) , where c ′ 2 ≤ 1 . By Lemma 32, we have

<!-- formula-not-decoded -->

therefore by Lemma 31, we have

<!-- formula-not-decoded -->

set α = δ 2 2+min { β, 2 } n , and we conclude the proof of Theorem 7.

<!-- formula-not-decoded -->

Proof of Theorem 16. By Assumption 5, we have /epsilon1 G = 0 . By Corollary 26 and Lemma 21, we have

<!-- formula-not-decoded -->

By Assumption 15, we have

<!-- formula-not-decoded -->

By Lemma 32, we have

<!-- formula-not-decoded -->

since γ ∈ (0 , 1) . Therefore, by Lemma 31, we have

<!-- formula-not-decoded -->

By selecting α = O ( δ 2 2+(2 -γ ) min { β, 2 } n ) , we have

<!-- formula-not-decoded -->

and we conclude the proof of Theorem 16.

## Appendix D. Proof of Theorem 12 and 17

In this section, we consider the case when /epsilon1 G and /epsilon1 H doht equal zero, i.e. Assumption 5 does not hold. We aim to establish a convergence rate for ‖ ˆ h -h 0 ‖ 2 for both MLE-based RDIV and χ 2 -MLE based RDIV in terms of δ n , /epsilon1 H and /epsilon1 G .

Lemma 22 Under Assumption 4, for α ∈ (0 , 1) we have

<!-- formula-not-decoded -->

Proof Note that in the misspecified case, we no longer have h 0 ∈ H . We further a augmented function class H ′ = Span( H∪{ h 0 } ) , and the corresponding optimizer of L 0 on H and H ′ :

<!-- formula-not-decoded -->

We define a function

<!-- formula-not-decoded -->

then L 0 is α -strongly convex, and attains its minimum at L 0 (0) . Note that we have the following inequality holds for all h ∈ H , set h = h † , by strong convexity and ∂ L 0 (0) = 0 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore we have

<!-- formula-not-decoded -->

and we conclude our proof for the lemma.

Proof for Theorem 12. By Lemma 21, we have

<!-- formula-not-decoded -->

By Corollary 26, we have ‖ ( T ˆ T )( ˆ h -h ∗ ) ‖ 1 ≤ ( δ 2 n, G + /epsilon1 G ) 1 / 2 ‖ ˆ h -h ∗ ‖ , and we have

<!-- formula-not-decoded -->

therefore by Lemma 32, we have

<!-- formula-not-decoded -->

By Lemma 22, combine everything together:

<!-- formula-not-decoded -->

note that δ n := { δ n, G , δ n, H } , we conclude the proof of Theorem 12.

## Proof of Theorem 17. By Lemma 21, we have

<!-- formula-not-decoded -->

by Lemma 28, we have ‖ ( ˆ T - T )( ˆ h -h ∗ ) ‖ 2 ≤ ( δ 2 n + /epsilon1 G ) 1 / 2 ‖ ˆ h -h ∗ ‖ ∞ , therefore we have

<!-- formula-not-decoded -->

where the second inequality comes from Assumption 15. By Lemma 32, we have

<!-- formula-not-decoded -->

by Lemma 22, combine everything together, we have

<!-- formula-not-decoded -->

and thus we conclude the proof of Theorem 17.

## Appendix E. Proof of Theorem 13

In this section, we will provide the details for the model selection results in the paper. Let /lscript h,g ( Y, Z, X ) denote the loss evaluated for a function h using the likelihood function ˆ g :

<!-- formula-not-decoded -->

Also, to simplify the notation, we use { X i , Y i , Z i } instead of { X ′ i , Y ′ i , Z ′ i } .

For θ ∈ Θ = { θ | ∑ j θ j = 1 , θ j ≥ 0 ∀ j } , denote h θ = ∑ j θ j f j . For any convex combination θ over a set of candidate functions { h 1 , . . . , h M } , we define the notation:

<!-- formula-not-decoded -->

Here we define some optimal aggregates in the following sense:

<!-- formula-not-decoded -->

Proof [Proof of Theorem 13]

<!-- formula-not-decoded -->

When ˆ g is estimated using the standard MLE appraoch, we have that by Corollary 26 and Lemma 21, we have that:

<!-- formula-not-decoded -->

Thus, we have R ( h j , g 0 ) -R ( h ∗ α, H j , g 0 ) ≤ O ( δ 2 n,j + /epsilon1 G α ) . Instantiating this result for the function class H M , which denotes the convex hull when convex-ERM is used, or the set of candidate functions when best-ERM is used, we get that:

<!-- formula-not-decoded -->

where δ n,M = max { δ n, G , δ n, H M } . Since the function classes used to train the candidate functions are typically more complex than the convex hull over M variables, it is safe to assume that δ n, H M ≤ δ n, H . Combining, we get:

<!-- formula-not-decoded -->

For any function class H , we have:

<!-- formula-not-decoded -->

Hence, for any function class H j , we can choose h that attains min H j ‖ h -h 0 ‖ = /epsilon1 H j . Combining, we get that:

<!-- formula-not-decoded -->

Analogously, if ˆ g is estimated using χ 2 -MLE, we have that by Corollary 28, Lemma 21 and Assumption 15:

<!-- formula-not-decoded -->

By the same argument for the standard MLE case, we get:

<!-- formula-not-decoded -->

## Appendix F. Proof of Theorem 14 and 19

In this section, we prove the convergence rate of iterative RMIV in Section 9 under a unified framework. We prove the results of Theorem 19 and 19 respectively. Recall that we define

<!-- formula-not-decoded -->

by Lemma 31 and Assumption 4, we have

<!-- formula-not-decoded -->

Therefore, we only need to provide a upper bound for ‖ ˆ h m -h m, ∗ ‖ 2 2 , and then choose the proper α deliberately. We start by proving the following lemma, and with the different convergence rate of MLE and χ 2 -MLE, we conclude the proof of Theorem 14 and Theorem 19 respectively.

Lemma 23 We have the following inequality holds with probability at least 1 -m exp( nδ 2 n, H ) :

<!-- formula-not-decoded -->

Proof Recall that our solution ˆ h m satisfies

<!-- formula-not-decoded -->

We define

<!-- formula-not-decoded -->

By definition, L m ( τ ) is minimized by τ = 0 . Note that by strong convexity and property of quadratic function, we have

<!-- formula-not-decoded -->

Therefore

<!-- formula-not-decoded -->

and thus we have

<!-- formula-not-decoded -->

holds for all m simultaneously with probability at least 1 -m exp( nδ 2 n, H ) , recall that δ 2 n, H is the critical radius. Here the second inequality comes from triangular inequality and L ( · ) being O (1) -Lipschitz, the third inequality comes from Lemma 29. By Eq. (12),

<!-- formula-not-decoded -->

therefore we have

<!-- formula-not-decoded -->

We divide it into two terms:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For I 2 , we divide it into two terms I 3 and I 4 , defined by

<!-- formula-not-decoded -->

Since each of these is the difference of two centered empirical processes, that are also Lipschitz losses (since h m, ∗ , ˆ h m , h m -1 , ∗ , ˆ h m -1 are uniformly bounded) and since h m, ∗ is a population quantity and not dependent on the empirical sample that is used for the m -th iterate, we can also upper bound these,

<!-- formula-not-decoded -->

combine everything together, we can prove that

<!-- formula-not-decoded -->

Therefore, we have

<!-- formula-not-decoded -->

By applying AM-GM inequality and utilizing α ≤ 1 , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof for Theorem 14. By Corollary 26, we have

<!-- formula-not-decoded -->

therefore by Lemma 23, we have

<!-- formula-not-decoded -->

By Lemma 32, we have

<!-- formula-not-decoded -->

where the second inequality comes from induction. Therefore, by Lemma 31, we have

<!-- formula-not-decoded -->

Set α = δ 2 2+min { β, 2 m } n , and we conclude the proof.

Proof for Theorem 19 By Assumption 15, we have ‖ ˆ h m -h m, ∗ ‖ ∞ ≤ ‖ ˆ h m -h m, ∗ ‖ γ 2 , which implies

<!-- formula-not-decoded -->

by Lemma 32, we have

<!-- formula-not-decoded -->

where the second inequality comes from induction. Therefore, by Lemma 31, we have

<!-- formula-not-decoded -->

Set α = δ 2 2+(2 -γ ) min { β, 2 m } n , Then δ n /α = O ( δ (2 -γ ) min { β, 2 m } 2+(2 -γ ) min { β, 2 m } n ) /lessorsimilar 1 , and since γ ∈ (0 , 1) , we have max { δ 2 n /α 2 , ( δ n /α ) 2 / (2 -γ ) } = ( δ n /α ) 2 / (2 -γ ) , and

<!-- formula-not-decoded -->

and we conclude the proof of Theorem 19.

## Appendix G. Convergence rate of MLE and χ 2 -MLE

## G.1. Convergence rate of MLE

In this section, we aim to characterize the convergence rate of conditional MLE (4) in terms of the critical radius δ n, G of function class G and model misspecification. Specifically, we prove the following Theorem:

Theorem 24 (Convergence rate for misspecified MLE) Suppose Assumption 6 and condition in Theorem 7 holds, and there exists g † ∈ G such that E z ∼ g 0 [ D KL ( g 0 ( ·| z ) , g † ( ·| z ))] ≤ /epsilon1 G . Then we have

<!-- formula-not-decoded -->

holds with probability at least 1 -c 1 exp( c 2 c 0 C + c 0 nδ 2 n ) .

Proof We work with the transformed function class F = { √ g + g 0 2 g 0 ∣ ∣ ∣ ∣ g ∈ G } , and define L f = -log f ( x ) for f ∈ F . Note that F is a function class whose element maps X × Z to R . We define the population version of localized Rademacher complexity for function class F ∗ := star(( F f ∗ ) ∪ { 0 } ) . By Assumption 6 and 1-boundedness of G , F and F ∗ are bounded by a constant b := C 0 + C 2 C 0 in ‖ · ‖ ∞ . The critical radius δ n, F of function class F ∗ is any solution such that

<!-- formula-not-decoded -->

Such critical radius can be easily calculated for a large number of function classes. For example, we can use to calculate δ n, F , where B ( δ, F ∗ ) := { f ∈ F ∗ | ‖ f ‖ 2 ≤ δ } , N n is the empirical covering number conditioned on { ( x i , z i ) } i ∈ [ n ] . For a cost function L : R → R , we define L f ( x, z ) := L ( f ( x, z )) . We make the following definition.

<!-- formula-not-decoded -->

Definition 25 We say L f is γ -strongly convexity at f ∗ if

<!-- formula-not-decoded -->

Note that for any f ∈ F we have and | log f ( x ) -log f ′ ( x ) | ≤ √ 2 | f ( x ) -f ′ ( x ) | since ‖ f ‖ ∞ ≥ 1 / √ 2 . By the definition of Hellinger distance, we have

<!-- formula-not-decoded -->

and since H 2 ( g 1 | g 2 ) ≤ 2 D KL ( f 1 | f 2 ) , we have ‖ f -f ∗ ‖ 2 2 ≤ P ( L f -L f ∗ ) , thus L is 2 -strongly convex at f ∗ . Utilizing strong convexity and Lemma 29, we have the following inequality holds for all f ∈ F .

with probability 1 -exp( nδ 2 n, F ) :

<!-- formula-not-decoded -->

here the first inequality comes from strong convexity, the third inequality comes from log( 2 x x + y ) ≤ 1 2 log( x y ) and the definition of MLE. The forth inequality comes from Lemma 29. Solve this inequality, and recall that ‖ f -h 0 ‖ 2 2 = E z ∼ g 0 ( z ) [ H 2 (( g + g 0 )( ·| z ) / 2 | g 0 ( ·| z ))] , we have

<!-- formula-not-decoded -->

here the first inequality comes from Lemma 30, the second inequality comes from Lemma 35. Thus we conclude the proof of Theorem 24.

We provide the following corollary, which would help characterize the L 1 and L 2 error of T h introduced by MLE.

Corollary 26 Under Assumption 6, for all h ′ ∈ H-H , we have ‖ ( ˆ T -T ) h ′ ‖ 1 ≤ { 1 /c 0 +1 }‖ h ′ ‖ 2 · ( δ 2 n, H + /epsilon1 G ) 1 / 2 and ‖ ( ˆ T -T ) h ′ ‖ 2 ≤ ( C 2 , 4 C ) 1 / 2 · ( C/c 0 +1) ‖ h ′ ‖ 2 · ( δ 2 n, G + /epsilon1 G ) 1 / 4 with probability at least 1 -c 2 exp( c 3 nδ 2 n, G ) .

Proof We first prove the bound for L 1 error ‖ ( ˆ T - T ) h ′ ‖ 1 . We have the following inequality:

<!-- formula-not-decoded -->

where the second inequality comes from Assumption 6. Next, we prove the upper bound for L 2 error ‖ ( ˆ T - T ) h ′ ‖ 2 . We have

<!-- formula-not-decoded -->

and we conclude the proof.

## G.2. Convergence rate of χ 2 -MLE

For the convergence rate of χ 2 -MLE, we present the following theorem:

Theorem 27 (Convergence rate for χ 2 -MLE, Corollary 14.24 of Wainwright (2019) ) For ˆ g generated by 13, we have

<!-- formula-not-decoded -->

with probability at least 1 -c 1 exp( c 2 nδ 2 n, G ) .

Proof By Theorem 13.13 of Wainwright (2019), we have

<!-- formula-not-decoded -->

holds with probability at least 1 -exp( c 1 nδ 2 n, G ) . By Theorem 34, we have

<!-- formula-not-decoded -->

holds for all g ∈ G with probability at least 1 -c 2 exp( c 3 nδ 2 n, G ) , and the proof is done. and the proof is done.

Weprovide the following corollary, which would help characterize the error introduced by χ 2 -MLE.

Corollary 28 With χ 2 -MLE, we have the following inequality holds for all h ∈ H with probability at least 1 -c 2 exp( c 3 nδ 2 n, G ) :

<!-- formula-not-decoded -->

Proof By

<!-- formula-not-decoded -->

We conclude the proof.

## Appendix H. Auxiliary Lemma

We introduce the following lemma, which gives a uniform convergence rate of loss error.

Lemma 29 (Localized Concentration, Foster and Syrgkanis (2019)) For any f ∈ F := × d i =1 F i be a multivalued outcome function, that is almost surely absolutely bounded by a constant. Let /lscript ( Z ; f ( X )) ∈ R be a loss function that is O (1) -Lipschitz in f ( X ) , with respect to the /lscript 2 norm. Let δ n = Ω (√ d log log( n )+log(1 /ζ ) n ) be an upper bound on the critical radius of star ( F i ) for i ∈ [ d ] . Then for any fixed h 0 ∈ F , w.p. 1 -ζ :

<!-- formula-not-decoded -->

If the loss is linear in f ( X ) , i.e. /lscript ( Z ; f ( X ) + f ′ ( X )) = /lscript ( Z ; f ( X ))+ /lscript ( Z ; f ′ ( X )) and /lscript ( Z ; αf ( X )) = α/lscript ( Z ; f ( X )) for any scalar α , then it suffices that we take δ n = Ω (√ log(1 /ζ ) n ) that upper bounds the critical radius of star ( F i ) for i ∈ [ d ] .

Proof For a detailed proof, please refer to Foster and Syrgkanis (2019).

The following lemma is useful when proving the convergence rate of Hellinger distance.

Lemma 30 (Lemma 4.1 in Van de Geer (1993)) For two density functions g 1 and g 2 , define g u = ug 1 +(1 -u ) g 2 , then we have

<!-- formula-not-decoded -->

holds for all u ∈ (0 , 1)

Proof

For a detailed proof, see Lemma 4.1 in Van de Geer (1993).

Lemma 31 (Lemma 5 in Bennett et al. (2023b)) If h 0 is the minimum L 2 -norm solution to the linear inverse problem and satisfies the β -source condition, then the solution to the t -th iterate of Tikhonov regularization h m, ∗ , defined in Equation (11) , with h 0 , ∗ = 0 , satisfies that

<!-- formula-not-decoded -->

Proof For a detailed proof, see Lemma 5 in Bennett et al. (2023b).

The following lemma upper-bounds the bias introduced by Tikhonov regularization.

Lemma 32 For where c 1 , c 2 &gt; 0 , 0 ≤ γ ≤ 1 , we have x ≤ 3max { √ c 1 , c 1 / (2 -γ 1 ) 2 , c 1 / (2 -γ 2 ) 3 } .

<!-- formula-not-decoded -->

Proof Since x 2 -c 2 x γ 1 -c 3 x γ 2 -c 1 is a convex function with negative intercept, we only need to prove that for x 0 = 3max { √ c 1 , c 1 / (2 -γ 1 ) 2 , c 1 / (2 -γ 2 ) 3 } , we have x 2 0 -c 2 x γ 1 0 -c 3 x γ 2 0 -c 1 ≥ 0 . For simplicity, we consider √ c 1 ≥ max { c 1 / (2 -γ 1 ) 2 , c 1 / (2 -γ 2 ) 3 } , and we have

<!-- formula-not-decoded -->

similarly we have the same result when c 1 / (2 -γ 1 ) 2 ≥ max { √ c 1 , c 1 / (2 -γ 2 ) 3 } or c 1 / (2 -γ 2 ) 3 ≥ max { √ c 1 , c 1 / (2 -γ 1 ) 2 } , and we conclude the proof.

Next, we introduce the following lemma that gives a uniform convergence rate for function class F , which is adapted from Wainwright (2019).

Lemma 33 (Theorem 14.20 in Wainwright (2019).) Suppose we have a 1 -uniformly bounded function class F that is star-shaped around a population minimizer f ∗ . Let δ n ≥ c n be the solution to the inequality

<!-- formula-not-decoded -->

Suppose the loss function L f is L -Lipschitz, then with probability at least 1 -c 1 exp( -c 2 nδ 2 n, F /b ) , either of the following events holds for all f ∈ F :

- (1) ‖ f -f ∗ ‖ 2 ≤ δ n ;

<!-- formula-not-decoded -->

The following lemma is a classical result for localization and uniform laws.

Theorem 34 (Theorem 14.1 of Wainwright (2019).) Given a star-shaped and b -uniformly bounded function class F , let δ n be any positive solution of the inequality

<!-- formula-not-decoded -->

Then for any t ≥ δ n , we have

<!-- formula-not-decoded -->

with probability at least 1 -c 1 e -c 2 nδ 2 n b 2 . If in addition nδ 2 n ≥ 2 c 2 log (4 log (1 /δ n )) , then

<!-- formula-not-decoded -->

with probability at least 1 -c ′ 1 e -c ′ 2 n 2 0 b 2 .

The next lemma enables us to upper-bound KL divergence by Hellinger distance.

Lemma 35 (Example 14.10 in Wainwright (2019). ) For any two density function g 1 and g 2 , we have

<!-- formula-not-decoded -->

## Appendix I. Additional Experiment Details

Wefollow the data-generating process in Kallus et al. (2021) and Cui et al. (2020) to generate multidimensional variables U, S, W, Q, A with A ∈ { 0 , 1 } as follows:

1. S ′ ∼ N (0 , 0 . 5 I d S ) , where I d is a d -dimension identity matrix.
2. A | S ′ ∼ Ber( p ( S ′ )) where

where 1 d is all-one vector.

3. Draw W ′ , Q ′ , U from

<!-- formula-not-decoded -->

Here we set the parameters above as µ 0 = α 0 = κ 0 = 0 . 2 1 d , α a = κ a = µ s = α s = κ s = I d , σ 2 q = σ 2 u = σ 2 w = 0 . 1 ( I d + 1 d 1 /latticetop d ) , σ 2 wu = σ 2 zu = 0 . 1 1 d 1 /latticetop d . Finally, we choose σ 2 wq and

<!-- formula-not-decoded -->

µ a to ensure that W ′ ⊥ ( A ′ , Q ′ ) | U, S ′ , which is a prerequisite of proximal causal inference (Kallus et al., 2021, Condition 4 in Assumption 1). To achieve this, note that

<!-- formula-not-decoded -->

where

We simply select σ 2 wq and µ a so that Equation (21) does not depend on A and Q ′ .

4. Draw Y from
5. Set W ′ = W ′ [0: d W ] . Observe S = g ( S ′ ) , Q = g ( Q ′ ) , W = g ( W ′ ) , where g ( · ) is a reversible function that operates component-wise on each variable.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->
