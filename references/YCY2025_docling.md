## Semiparametric M-estimation with overparameterized neural networks

Shunxing Yan ∗ , Ziyuan Chen † and Fang Yao ‡ School of Mathematical Sciences, Center for Statistical Science, Peking University, Beijing, China

## Abstract

We focus on semiparametric regression that has played a central role in statistics, and exploit the powerful learning ability of deep neural networks (DNNs) while enabling statistical inference on parameters of interest that offers interpretability. Despite the success of classical semiparametric method/theory, establishing the √ n -consistency and asymptotic normality of the finite-dimensional parameter estimator in this context remains challenging, mainly due to nonlinearity and potential tangent space degeneration in DNNs. In this work, we introduce a foundational framework for semiparametric M -estimation, leveraging the approximation ability of overparameterized neural networks that circumvent tangent degeneration and align better with training practice nowadays. The optimization properties of general loss functions are analyzed, and the global convergence is guaranteed. Instead of studying the 'ideal' solution to minimization of an objective function in most literature, we analyze the statistical properties of algorithmic estimators, and establish nonparametric convergence and parametric asymptotic normality for a broad class of loss functions. These results hold without assuming the boundedness of the network output and even when the true function lies outside the specified function space. To illustrate the applicability of the framework, we also provide examples from regression and classification, and the numerical experiments provide empirical support to the theoretical findings.

## 1 Introduction

## 1.1 Literature review

Semiparametric models constitute an essential category of statistical methods, involving both finite and infinite-dimensional parameters. Typically, more consideration is given to the finitedimensional components and the latter is regarded as a nuisance (Van der Vaart, 2000; Kosorok, 2008). Compared to parametric models with only finite-dimensional parameters, semiparametric

∗ E-mail: sxyan@stu.pku.edu.cn

† E-mail: chenziyuan@pku.edu.cn

‡ Fang Yao is the corresponding author, E-mail: fyao@math.pku.edu.cn.

models provide stronger representation capabilities. Moreover, in contrast to nonparametric models, they alleviate the curse of dimensionality and allow easier inference on the finite-dimensional parameters of primary interest. Benefiting from these advantages, semiparametric models have garnered increasing attention in statistics over an extended period, including regression analysis (Ahmad et al., 2005; Liang et al., 2010), survival analysis (Cox, 1972, 1975; Huang, 1999; Zeng and Lin, 2007), causal inference (Rosenbaum and Rubin, 1983; Chernozhukov et al., 2018) and many others; see Tsiatis (2006); Kosorok (2008); Horowitz (2012) for a comprehensive introduction. Among these works, one of the most common estimation methods is the semiparametric M -estimation (Van de Geer, 2000; Ma and Kosorok, 2005b; Cheng and Huang, 2010), where the estimators for both parametric and nonparametric parameters are obtained simultaneously by minimizing (or maximizing) certain objective functions. There has been much research on this method, employing classical nonparametric regression techniques, including linear sieves (Ding and Nan, 2011; Ma et al., 2021; Tang et al., 2023), local polynomial method with closed-form representations (Wang et al., 2010; Liang et al., 2010) and penalized methods (Mammen and Van de Geer, 1997; Ma and Kosorok, 2005a). In general, efficient asymptotic √ n -normality of finite-dimensional parameters and minimax optimal convergence of the nonparametric parts can be established. However, most existing works based on traditional techniques often face challenges when applied to complex structured data increasingly encountered in modern applications. Consequently, there is growing interest in developing estimators based on advanced methodologies such as neural networks, with rigorous theoretical guarantees, motivated by the significant achievements of deep learning in recent years.

Deep neural networks (DNNs), especially those with ReLU activation function, have received significant attention in recent studies due to their strong learning abilities. Approximation bounds for ReLU DNNs have been established in various function settings (Yarotsky, 2017, 2018; Schmidt-Hieber, 2020; Suzuki, 2018; Yarotsky and Zhevnerchuk, 2020; Kohler and Langer, 2021), which are shown to be (nearly) optimal in terms of both width and depth (Shen, 2020; Lu et al., 2021; Shen et al., 2022). A key advantage of DNN estimators, distinguishing them from classical nonparametric regression estimators, is their adaptivity to various low intrinsic-dimensional structures, such as the hierarchical composition model where the target function follows a specific compositional structure (Bauer and Kohler, 2019; Schmidt-Hieber, 2020; Kohler and Langer, 2021), andcases with low-dimensional inputs (Schmidt-Hieber, 2019; Chen et al., 2019; Cloninger and Klock, 2021; Nakada and Imaizumi, 2020; Jiao et al., 2021). Therefore, DNNs are increasingly recognized as an important nonparametric approach in a wide range of statistical problems, includ-

ing nonparametric regression (Bauer and Kohler, 2019; Schmidt-Hieber, 2020; Suzuki, 2018; Kohler and Langer, 2021; Wang et al., 2024), survival analysis (Zhong et al., 2021, 2022), causal inference (Farrell et al., 2021; Chen et al., 2024), factor augmented and interaction models (Fan and Gu, 2022; Bhattacharya et al., 2023), repeated measurements model (Yan et al., 2025b), among others.

As a common ground, most of the previously mentioned statistical works study the generalization ability of the (nearly) empirical risk minimizers and bound the estimation error by network size relative to the training sample. However, the optimization landscape in deep learning presents unprecedented difficulty due to nonlinearity and nonconvexity, leading to the estimation more likely being local minimizers. Moreover, in practice, neural networks are often trained with overparameterization, i.e., the network parameters may vastly exceed the training sample size, while still avoiding the traditional pitfall of overfitting (Zhang et al., 2021). This does not align with the statistical analysis in the previously mentioned works. Fortunately, there have been meaningful results to address these gaps. For instance, Jacot et al. (2018) introduced a framework that analyzes the training of wide neural networks, drawing an analogy to kernel regression. Specifically, they compared the gradient flow of the least squares loss with that of kernel regression and demonstrated that, as the network width approaches infinity, the Neural Tangent Kernel (NTK) converges to a time-invariant limit (Arora et al., 2019; Lee et al., 2019; Bietti and Mairal, 2019; Geifman et al., 2020; Chen and Xu, 2020; Bietti and Bach, 2021; Lai et al., 2023a). From the perspective of optimization, wide neural networks with random initialization have been shown, with high probability, can achieve global minimization through gradient-based methods (Du et al., 2018; Li and Liang, 2018; Allen-Zhu et al., 2019). In terms of statistical generalization performance, Hu et al. (2021); Suh et al. (2021) established the convergence rate of penalized least square regression problem, and more recent studies have examined least squares regression with early stopping, providing uniform convergence guarantees for neural network kernels (Lai et al., 2023a; Li et al., 2024; Lai et al., 2023b).

## 1.2 Challenges and main contributions

Despite the impressive empirical performance of deep learning models, they are commonly regarded as black boxes, lacking interpretability and theoretical support. Semiparametric modeling provides a useful way for making interpretable inferences while leveraging the learning ability of neural networks. Training DNNs involves a large-scale network parameter learning via loss minimization, and aligns naturally with the framework of semiparametric M -estimation which simultaneously estimates nonparametric (via network weights) and parametric parameters. Several

studies have explored this problem (e.g., Zhong et al., 2022), but defaulted to some 'good' assumptions on the tangent space-a critical issue we shall discuss below. Another commonly used method is Z -estimation, where the finite-dimensional parameter is typically taken as a functional of the nonparametric component. It usually takes two steps: first estimating the nonparametric component and then substituting it into an equation to solve for the parametric component. For such two-step procedures, the doubly debiased/robust methods (Chernozhukov et al., 2018; Farrell et al., 2021) are employed to achieve efficiency. By comparison, semiparametric M -estimation is computationally straightforward via network training, and requires little statistical manifest with broader applicability. While a simpler two-step plug-in method can also attain efficiency (Chen et al., 2024), one has to impose assumptions on the tangent approximation ability in a similar manner to Zhong et al. (2022).

The above discussion inspires our investigation: what foundation of semiparametric M -estimation theory is undermined in neural network based models,and how to rebuild it? Denote the semiparametric model as P β,f , where β ∈ R p is the finite-dimensional parameter of interest, and f : R d → R is a nuisance function belonging to a infinite-dimensional space. The semiparametric M -estimation procedure aims to optimize an objective function,expressed as P n l β,f with empirical measure integration P n , either in a suitable sieve or with an additional penalty term. Notably, the information of the parameters, β and f , is coupled. Since the nuisance parameter f resides in an infinite-dimensional space, it significantly increases the complexity of the estimation problem. Consequently, it is reasonable that the full set of parameters achieves a nonparametric convergence rate that is slower than n -1 .

A central challenge of semiparametric M -estimations is to establish √ n -consistency and the asymptotic normality of the estimator for the finite-dimensional component β . To decouple the two parts of parameters, taking likelihood estimation as an example, the efficient score is commonly utilized, which removes the effect of nuisance parameters. Specifically, we define Λ β,f as the tangent set for the nuisance parameter, i.e. it contains score functions ∂ f l β,f [ h ] with all appropriate directions h , see Section 3 for a rigorous definition. Additionally, denote Π θ,f as the orthogonal projection operator onto the closure of the linear span of Λ β,f . Then the efficient score can be constructed by ∂ β l β,f subtracting its projection on nuisance tangent space Λ β,f , i.e. ˜ S = S 1 ( β 0 , f 0 ) -Π β 0 ,f 0 S 1 ( β 0 , f 0 ) , where S 1 ( β, f ) = ∂l β,f /∂β . Accordingly, ˜ S is orthogonal to the tangent space Λ β,f , thus mitigating the nuisance information. Once the empirical efficient score P n ˜ S ( ˆ β, ˆ f ) is proven to be small enough as o p ( n -1 / 2 ) , asymptotic normality of √ n ( ˆ β -β 0 ) can then be established via standard Taylor expansion and entropy analysis (Van der Vaart, 2000;

Kosorok, 2008).

However, establishing that

<!-- formula-not-decoded -->

is o p ( n -1 / 2 ) for neural network estimators is highly nontrivial. The first term on the right-hand side is typically easy to bound, as ˆ β is a finite-dimensional (nearly) minimizer. The main challenge arises from bounding the second term. Because Π β 0 ,f 0 S 1 ( β 0 , f 0 ) is in the the closure of the linear span of Λ β,f , there could be some direction ˜ h such that Π β 0 ,f 0 S 1 ( β 0 , f 0 ) = S 2 ( β 0 , f 0 ) [ ˜ h ] . Denote the neural network function as f θ , where θ represents the network parameters. When ˆ f θ is an exact local or global minimizer, we can only obtain that ∂ θ P n l ˆ β, ˆ f θ = 0 , but not necessarily P n S 2 ( ˆ β, ˆ f )[ ˜ h ] = 0 . Denote the tangent space of the network at θ by

<!-- formula-not-decoded -->

Then, for any h ∈ T ˆ θ , we have P n ( S 2 ( ˆ β, ˆ f )[ h ]) = 0 . If the components of ˜ h lie in, or can be well approximated by, the tangent space T ˆ θ , we could conclude that P n S 2 ( ˆ β, ˆ f )[ ˜ h ] is sufficiently small. Therefore, it is preferable for the tangent space to be large to contain or approximate as many possible directions as possible. For traditional linear estimators using a sieve space S = { f α ( x ) : f α ( · ) = ∑ m i =1 α i b i ( · ) } with basis functions { b i ( · ) , i = 1 , · · · , m } (e.g., regression spline estimators), a large tangent space is straightforward to guarantee, since the linearity ensures that the tangent space of S at any f α ( · ) ∈ S is T α S = Span { ∂ α i f α ( · ) } ∼ = S . This provides a strong approximation ability. However, this does not necessarily hold for the tangent space of neural networks.

Therefore, the crucial question arises on the approximation ability of the neural network tangent space, and should not be simply suppressed by presupposed conditions as in previous works (Zhong et al., 2022; Chen et al., 2024):

Whether the tangent space T θ F NN of a neural network always has good approximation ability at every possible θ ? If not, can the semiparametric DNN M -estimators still achieve a √ n -consistency for the finite-dimensional parameters?

Unfortunately, the answer to the former question is negative. As a toy example, we consider a fully connected neural network in which the weights and biases within each layer take the same values. Due to the lack of identifiability among network parameters, the derivatives with respect to parameters in equivalent positions are the same. Consequently, the dimension of the tangent

space T θ F NN is only linearly related to the depth of the neural network, which is much smaller than the number of parameters (i.e., the square of the width multiplied by the depth). In such cases, the tangent space T θ F NN will not provide a sufficiently rich approximation ability for possible ˜ h . As discussed above, whether the efficient score is sufficiently small to establish the √ n -consistency of the parametric component of the DNN estimator remains an open problem.

Nonetheless, the counterexample presented above is so extreme that it is reasonable to presume that it has only a small probability of occurring in learning. Hence, a workable theoretical treatment requires a more careful analysis of the estimation procedure and randomness introduced by the neural network, like the random initialization. Additionally, we hope to address the limitations in most current statistical analyses of deep neural networks: they ignore the nonlinearity and nonconvexity of the optimization landscape and directly consider the 'ideal' global minimizer. Motivated by these considerations, we study overparameterized neural networks that provide a meaningful way for analyzing the optimization and statistical performance of the algorithmic solutions. Further, for generality, we consider a broader class of loss functions beyond the least squares criterion, which introduces additional difficulties for both optimization convergence analysis and statistical inference of the algorithmic solution. In summary, our main contributions to tackling the challenges are elaborated below.

- Methodologically, for the semiparametric M -estimation problem, we employ a new overparameterized neural network estimator, which better aligns with practical settings and facilitates the optimization analysis. The l 2 penalization of the neural network parameters is applied for regularization, allowing subsequent analyses of general loss rather than only least square regression. This also ensures that the scores at the estimation remain sufficiently small, contributing to the asymptotic normality of the finite-dimensional parameter estimation. Then we consider a continuous gradient flow framework to investigate the training dynamics of the neural network. By incorporating random initialization into the analysis, with high probability, we bound the difference between the overparameterized neural network training flow and the ideal RKHS optimization process. It also suggests that the counterexamples mentioned above with degenerate tangent space would be only taken with a small probability. Furthermore, we establish the global algorithmic convergence of the proposed overparameterized neural semiparametric M -estimator, providing a theoretical guarantee of the training procedure.
- Theoretically, we analyze the statistical properties of the algorithmic solution, including the

generalization error of the nonparametric component and the √ n -consistency/asymptotic normality of the parametric component. Specifically, unlike the common convergence analysis of neural network estimation, which often assumes that the network output is bounded, the optimization solution is analyzed directly. Especially when the true function does not lie in the desired space and the loss function is of a general form instead of least squares, estimators that do not impose bounded assumptions on the nonparametric part present significant challenges to theoretical analysis. To establish the convergence rate, we introduce a new condition, referred to as the 'Huberized margin condition', which relaxes the standard assumptions and is easier to satisfy by unbounded nonparametric candidate function classes. Building on the above results and some regularity conditions, we show that the parametric component achieves √ n -consistency and asymptotic normality. The latter result demonstrates the efficiency for the least favorable submodels and enables interpretable inference for parameters of interest. Lastly, we discuss two commonly used models: partially linear regression and classification. In these examples, the aforementioned properties of the proposed estimator are examined to be valid. Moreover, the numerical results corroborate our theoretical analysis by the finite sample behavior.

## 1.3 Notations and organization

In this paper, we use the notation a n /lessorsimilar b n to indicate that a n ≤ Cb n for some constant C &gt; 0 independent of n , with /greaterorsimilar defined analogously. Then a n /equivasymptotic b n means that both a n /lessorsimilar b n and b n /lessorsimilar a n . Denote a n = o ( b n ) if a n /b n → 0 , a n = O ( b n ) if a n /lessorsimilar b n , and a n = Θ( b n ) if a n /equivasymptotic b n . We also use a n = o p ( b n ) to indicate that a n /b n → p 0 , and a n = O p ( b n ) if, for any /epsilon1 &gt; 0 , there exists a constant C /epsilon1 &gt; 0 such that sup n P ( ‖ X n ‖ ≥ C /epsilon1 | Y n | ) &lt; /epsilon1 . Additionally, poly( a, b, ... ) indicates some polynomial about ( a, b, ... ) . The notation I denotes the indicator function. For probabilistic integrals, P represents the theoretical expectation with respect to the population distribution, while P n denotes the empirical expectation derived from the finite sample.

The rest of the article is organized as follows. In Section 2, following an overview of overparameterized neural networks and neural tangent kernels, we introduce the semiparametric model framework for neural M -estimation trained by the gradient flow algorithm. Section 3 develops the statistical theory for the algorithmic estimators, addressing the nonparametric convergence of the nonparametric component and the asymptotic normality of the parametric component. In Section 4, we examine two illustrative examples, the partially linear regression and classification, demonstrating the validity of the theoretical results. Finally, the numerical

experiments are presented in Section 5, providing empirical evidence of the proposed method and theory.

## 2 Semiparametric M-estimation with neural networks

In this section, we first introduce the overparameterized neural network used and some important related concepts. Then we present the considered semiparametric model and neural M -estimation framework. Furthermore, a detailed analysis of the optimization convergence would also be provided.

## 2.1 Overparameterized neural network and neural tangent kernel

In this paper, we primarily consider feedforward fully connected neural networks. Given positive integers m 0 , m 1 , ..., m L +1 with m 0 = d, m L +1 = 1 , the network is defined as following

<!-- formula-not-decoded -->

where σ : R m i → R m i is the activation function. For i ∈ { 0 , 1 , ..., L -1 } , L i ( y ) = √ 2( W i y + b i ) / √ m i +1 , and for i = L , L i ( y ) = W i y + b i , where W i ∈ R m i +1 × m i , b i ∈ R m i for y ∈ R m i . In this work, we default σ to the rectified linear unit (ReLU) function a ↦→ max { a, 0 } , and most results presented can be generalized to other activation functions. Additionally, we use θ to denote all parameters in the neural network. To conveniently analyze the effect of the width m i , we assume that m ≤ min 1 ≤ i ≤ L m i ≤ max 1 ≤ i ≤ L m i ≤ Cm always holds for some constant C .

In practice, the neural networks are overparameterized, which means the width m will be much larger than the sample size n . To better align with practical settings and facilitate the analysis of optimization properties, we also consider overparameterized networks in the following. Before training, random initialization of weights in networks is usually employed to break symmetry and prevent all neurons from learning identical features, thereby improving learning efficiency and promoting better convergence. Therefore, we randomly initialize the neural network weight matrices W i 's and the bias b 0 from an i.i.d. standard normal distribution, while the other biases b i 's for i ≥ 1 are initialized to zero. Denoting the initial parameters by θ 0 , we define

<!-- formula-not-decoded -->

to ensure that f θ is initially zero in the training, i.e. f θ 0 ( x ) ≡ 0 , for theoretical convenience.

In the analysis of overparameterized neural networks, the neural tangent kernel (NTK) (Jacot et al., 2018; Arora et al., 2019) is commonly employed. We now introduce the NTK for finite m as

<!-- formula-not-decoded -->

Fixing L and letting m →∞ , the kernel function converges to

<!-- formula-not-decoded -->

Recent works have established that this convergence holds uniformly (Lai et al., 2023a). In the sequel, we will generally refer to the limiting kernel as NTK for brevity. Moreover, an explicit formula for NTK with the aforementioned random initialization can be derived.

Proposition 2.1 . Under the random initialization mechanism proposed for W i 's and b i 's, the neural tangent kernel has the explicit expression as

<!-- formula-not-decoded -->

where ˜ u := ( x T x ′ + 1) / √ ( ‖ x ‖ 2 2 +1)( ‖ x ′ ‖ 2 2 +1) , κ 0 ( t ) and κ 1 ( t ) are arc-cosine kernels of degree 0 and 1 , i.e.

<!-- formula-not-decoded -->

and h ( r ) , r ≥ 1 denote the r -times composition of a function h while h (0) is the identity map.

These expressions presented here differ slightly from previous results (Bietti and Mairal, 2019; Geifman et al., 2020), due to the special initialization of the bias, which simplifies subsequent derivations related to the RKHS of K NT over a general domain. Specifically, we introduce the following equivalence property that facilitates subsequent statistical analysis.

Proposition 2.2 . For any Ω ⊂ R d with Lipschitz boundary, the RKHS H NT associated to K NT is norm-equivalent to the Sobolev space W ( d +1) / 2 , 2 (Ω) .

Remark 2.1. Since the properties of Sobolev spaces are extensively studied, given the above proposition, many results can be derived more easily using existing results of Sobolev spaces. We emphasize that the smoothness index ( d + 1) / 2 is determined by the ReLU activation function. Generally, smoother activation functions lead to higher smoothness in the corresponding NTK Sobolev space. For example, the rectified power unit activation a ↦→ max { a, 0 } r with positive integer r (Bach, 2014; Vakili et al., 2021), which is weakly differentiable of order r , can be proven to lead to Sobolev space W ( d +2 r -1) / 2 , 2 via similar analysis as Proposition 2.2.

We close this subsection by recalling the motivations behind using overparameterized DNNs in this paper. In practical applications, neural networks are often overparameterized, where the number of parameters exceeds the sample size. Furthermore, rather than studying the ideal global minimizers as in common statistical works, we hope to investigate the statistical properties of the algorithmic estimators, particularly under the overparameterized optimization convergence guarantee. By introducing stochastic initialization and analyzing the optimization process of the overparameterized DNNs, we can avoid, with high probability, the counterexample mentioned in Section 1.2 on the degeneration of the DNN tangent space, and hence helps to establish √ n -consistency/asymptotic normality of the estimation of the parametric component.

## 2.2 Semiparametric neural M-estimation

Consider a general semiparametric statistical model P β,f , where β ∈ R p is a Euclidean parameter of interest and f : R d ↦→ R is the nuisance function in an infinite-dimensional space. Let ( Y, Z, X ) follows the distribution P β,f , where Z ∈ R p and X ∈ R d are finite-dimensional covariates. For simplicity, the domains of Z and X are assumed to be bounded, compact sets with regular boundaries, and their densities are bounded away from zero and infinity. Moreover, we assume that E [ Y 2 | Z, X ] is finite. Consider a nonnegative loss function l β,f = l ( Z T β, f ( X ) , Y ) such that the true parameters ( β 0 , f 0 ) minimizes the risk Pl β,f = ∫ l ( Z T β, f ( X ) , Y ) dP β,f . Common choices for l include the negative log-likelihood function, squared loss function, or other robust loss functions.

Given i.i.d. observations { ( Y i , Z i , X i ) , i = 1 , 2 , . . . , n } , we aim to estimate the unknown parameters ( β, f ) by minimizing the criterion

<!-- formula-not-decoded -->

where λ n ≥ 0 is a tuning parameter and J is a penalty term, within a suitably chosen parameter set ( R p , F n ) . Letting the nonparametric function set F n be the overparmeterized neural network class, for each f θ ∈ F n with network parameter θ and initial parameter θ 0 , we define the penalty term as

<!-- formula-not-decoded -->

which regularizes the complexity of the neural network to prevent overfitting. An alternative regularization technique is early stopping, while existing statistical analyses for early stopping in kernel gradient algorithms mainly focus on least square losses (Yao et al., 2007; Raskutti et al., 2014). For generality, this work considers a broader class of loss functions, hence, we adopt

the penalization strategy. Even under the penalization framework, there are still considerable challenges in the statistical theory. Unlike typical convergence analyses on neural network estimation, which often assume that the nonparametric function is bounded or that the loss function is Lipschitz continuous, we will study the optimization solution without boundedness assumptions. This brings difficulties when the true function does not belong to the desired space and the loss function takes a more general form than the least squares.

## 2.3 Gradient flow optimization and convergence

Given the above considerations, we aim for our estimator to satisfy

<!-- formula-not-decoded -->

To optimize the objective in (3), gradient-based algorithms are widely employed, with various modifications such as stochastic sampling and momentum methods being particularly common in practice. A substantial body of literature addresses optimization algorithms for neural networks, highlighting diverse methodologies and their relative effectiveness in improving training outcomes. In this study, for theoretical convenience, we focus on gradient flow, which serves as the continuous counterpart of gradient descent. Let ˆ θ t denote the neural network parameters at time t ≥ 0 . Correspondingly, let ˆ f t = f ˆ θ t represent the neural network output, and K NT t = K NT ˆ θ t denote the neural tangent kernel at t ≥ 0 . With the initial values set as ˆ θ 0 = θ 0 and ˆ f 0 ( x ) = 0 , for all x ∈ Ω , the gradient flow training process of the parameters ( ˆ β t , ˆ θ t ) is governed by the following equations:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Then the flow of f is defined by a dynamical system as

<!-- formula-not-decoded -->

For theoretical analysis, we take an ideal estimator in the RKHS associated with reproducing kernel K NT as a surrogate, which is shown to be sufficiently close to the neural algorithmic

estimator ( ˆ β t , ˆ θ t ) . In this context, the penalty term is defined as ˜ J ( f ) = ‖ f ‖ 2 H NT , which is standard and encourages smoother solutions by penalizing the complexity of the function f in the RKHS norm. This penalty term leads to the following regularized optimization criterion:

<!-- formula-not-decoded -->

where P n l β,f still represents the empirical risk term and λ n ˜ J ( f ) is the tuning parameter. Consequently, within the RKHS, a gradient flow training process { ˜ β t , ˜ f t } with initial value ˜ θ 0 = 0 and ˜ f 0 ( x ) = 0 , ∀ x ∈ Ω is adopted as and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Define the subspace H 1 = Span { K NT ( · , X i ) , i = 1 , 2 , ..., n } within the considered NTK RKHS. It is straightforward to verify that ˜ f t ∈ H 1 , ∀ t ≥ 0 , meaning that the evolution of ˜ f t is restricted to this finite-dimensional subspace. To analyze the optimization convergence, the following assumption on convexity and smoothness is standard.

Assumption 1 (Conditions for optimization) . The loss function l = l ( · , · , · ) is convex and nonnegative, the gradient ∇ l is B 1 -Lipschitz continuous with a constant B 1 .

This assumption on the loss function ensures that the gradient flow converges. The convexity guarantees convergence, while the Lipschitz continuity of the gradient ensures stability during optimization, preventing abrupt changes that could hinder convergence. Denote the initial loss L 0 = ˜ L n ( β 0 , f 0 ) and assume that the RKHS global minimizer of (7) satisfies ‖ ˜ β ∞ ‖ 2 2 + ‖ ˜ f ∞ ‖ 2 H NT = ˜ B 2 . The following conclusion characterizes the convergence of the ideal gradient flow for general loss functions.

Proposition 2.3 . Given the sample { ( Y i , Z i , X i ) , i = 1 , 2 , . . . , n } and we consider the optimization problem (7) in the Hilbert space H 1 with the gradient flow method described above, the following results on the optimization convergence rate hold.

(1). When Assumption 1 holds, we have

<!-- formula-not-decoded -->

- (2). Additionally, if (7) is µ -strongly convex for a positive number µ , we have

<!-- formula-not-decoded -->

Remark2.2. The above proposition demonstrates that the convergence rate of optimization is sublinear for convex objects and linear for strongly convex objects; which is standard in optimization theory. Therefore, to obtain the optimizer ( ˜ β t , ˜ f t ) that satisfies

<!-- formula-not-decoded -->

we need to train t s that t s /greaterorsimilar n 2 ˜ B 2 when the objective is just convex and t s /greaterorsimilar µ -1 log n when it is µ strongly convex. Furthermore, the penalization procedure guarantees ‖ ˜ f ∞ ‖ 2 H NT /lessorsimilar L 0 /λ n and the consistency (or at least boundedness) of ˜ β ∞ is usually easy to establish in statistical problems. Therefore, without involving neural networks, the upper bounds of ˜ B 2 for specific problems are typically available.

The following theorem demonstrates that, for any given training time, the discrepancy between the neural network and RKHS optimization results can be sufficiently small when the neural network is wide enough.

Theorem 2.1. Given any positive real number ξ ∈ (0 , 1) and the training time t . Let the network m ( t, ξ ) ≥ poly( n, λ -1 , L 0 , log(1 /ξ ) , exp( t )) large enough. Then, with probability at least 1 -ξ over neural network random initialization, the differences between neural network estimation and RKHS estimation can be bounded by

<!-- formula-not-decoded -->

This theorem demonstrates that the gradient flow of DNNs and the NTK RKHS can exhibit significant similarity when the neural network is sufficiently wide. Although the optimization landscape of DNNs is generally complex, in the NTK overparameterized regime, it closely resembles that of the RKHS. This similarity also helps avoid the degeneration of the tangent space as seen in the counterexample in Section 1.2, with high probability. Additionally, the requirement on the width of DNNs includes an exponent term related to the training time, which is used to bound the accumulated dynamic differences between the two training processes. This factor accounts for the cost of accommodating general, unstructured loss functions, but can be omitted when considering the least squares loss as a special case.

## 3 Statistical theory of algorithmic neural M-estimation

In this section, we discuss the statistical properties of the neural semiparametric M -estimator, focusing on the convergence rate of the nonparametric part and the asymptotic normality of the

parametric component. Notably, we do not impose common assumptions such as the Lipschitz continuity (e.g. Huang et al., 2024) or strongly convexity of the loss function, as these conditions may limit the applicability of our results. Furthermore, because our analysis pertains to the algorithmic optimizer, we do not assume that the nonparametric component is bounded; this introduces additional challenges for the theoretical analysis. Before discussing details, we introduce some basic assumptions. Firstly, some basic conditions of the model regularity are summarized below.

Assumption 2. (1) The covariate ( Z, X ) takes value in a bounded domain with a joint probability density function bounded away from 0 and ∞ .

- (2) Conditional on ( Z, X ) , the second order moment E [ Y 2 | Z = z , X = x ] is bounded.
- (3) Derivatives and expectations are exchangeable in the sense that

<!-- formula-not-decoded -->

for u = u 1 and u 2 .

When establishing the nonparametric convergence, the margin condition (Tsybakov, 2004) and the Bernstein condition (Bartlett and Mendelson, 2006) are often employed to achieve fast rates in statistical and learning analysis. Consider a simple nonparametric model P f for example, which depends on a nonparametric parameter f and a corresponding loss function l f . These conditions quantify the identifiability and the curvature of the objective function f ↦→ Pl f at some minimum f ∗ . In the margin condition, f ∗ = f 0 is the minimizer of the risk over all measurable functions, whereas in the Bernstein condition, f ∗ typically minimizes the risk over the candidate function class F , see Lecu´ e (2011) for more discussions. As one of specific forms, these conditions may establish relationships between the excess risk P ( l f -l f ∗ ) and the L 2 norm ‖ f -f ∗ ‖ 2 L 2 /equivasymptotic P ( f ( X ) -f ∗ ( X )) 2 through inequalities of the form P ( l f -l f ∗ ) /greaterorsimilar ‖ f -f ∗ ‖ 2 κ L 2 with typically κ = 1 . This implies a better concentration and smaller localized sets, hence helps for the fast convergence. However, to verify the Margin/Bernstein conditions usually requires that f is near f 0 , for example, ‖ f -f 0 ‖ L ∞ is bounded. This has limited the application of related results when the boundedness of nonparametric functions does not hold naturally. Especially in this work, assuming boundedness for the optimizer under gradient flow is particularly unsuitable. In semiparametric problems, unlike the finite-dimensional parameters for which consistency is typically easy to establish, L ∞ boundedness of the nonparametric estimation is often non-trivial, especially when the true function does not lie within the considered RKHS. The following Huberized margin condition holds more easily for unbounded function class.

Assumption 3 (Huberized margin condition for semiparametric estimation) . There is a constant B 2 &gt; 0 such that for every β ∈ R p , f ∈ F ,

<!-- formula-not-decoded -->

Typically, when the loss is bounded or globally Lipschitz continuous (e.g. Huang et al., 2024), the class complexity is easy to bound, but such assumptions may not be general enough. In the proposed Huberized condition, ignoring finite-dimensional parameter β for convenience, when ‖ f -f 0 ‖ L ∞ /lessorsimilar 1 , the right hand is nearly ‖ f -f 0 ‖ 2 L 2 ( X ) ; while when ‖ f -f 0 ‖ L ∞ /greaterorsimilar 1 , the right hand is even smaller than ‖ f -f 0 ‖ L 1 ( X ) . If only the local curvature is of concern, then this condition is equivalent to the commonly used margin condition P ( l f -l f 0 ) /greaterorsimilar ‖ f -f 0 ‖ 2 L 2 . As pointed out earlier, it is not proper to assume that the algorithmic neural network optimizer is bounded, and the L ∞ consistency or boundedness of nonparametric estimation is usually non-trivial when the true function does not lie within the considered RKHS. Moreover, in (8) for semiparametric models, ‖ β -β 0 ‖ in the denominator is for symmetry, which is usually not necessary because the consistency of the estimation of β is verifiable in common cases.

Whenthe risk is strongly convex, the condition is obviously satisfied. Intuitively, the proposed assumption only requires that the Hessian matrix is non-singular near the minimum and that the gradient is lower bounded at the far end. Strong convexity of the loss function across the entire domain is unnecessary; local strong convexity near the true minimizer is often sufficient. This is not much stricter than Assumption 1, more examples for illustration will be given in Section 4.

The next assumption assumes that the true parameters lie within the ideal space.

## Assumption 4. The true parameter f 0 belongs to the RKHS H NT .

This condition is common in traditional statistical works and yields some boundedness to simplify the analysis. However, it does not always hold in a broad context of learning problems and is in fact not necessary to achieve the desired property. Thus, we relax this condition to allow that the true function does not lie within the reproducing space of NTK.

Assumption 4 ′ . The true parameter f 0 does not reside within the RKHS H NT , but instead belongs to a Sobolev space W s, 2 with s &gt; d/ 2 .

Here, we assume that the smoothness of f 0 satisfies s &gt; d/ 2 , a condition crucial for establishing the statistical properties under consideration, such as the convergence rate and the attainment of parametric asymptotic normality. This assumption is particularly significant in the context of semiparametric statistics, where achieving a √ n -consistent estimator for finite-dimensional

parameters generally necessitates that the non-parametric convergence rate faster than n -1 / 4 (Van der Vaart, 2000; Kosorok, 2008).

Now, we introduce some basic concepts and assumptions for the semiparametric model (Van der Vaart, 2000; Kosorok, 2008). Given a fixed f , let G f denote the collection of all smooth functional curves g b that run through f at b = 0 , i.e. { g b : g b ∈ L 2 (Ω) for b ∈ ( -1 , 1) and lim b → 0 ‖ b -1 ( g b -g 0 ) -h ‖ L 2 → 0 for some function h } . Then let

<!-- formula-not-decoded -->

For any h in T f , denote be all the potential tangent directions for the nuisance parameter. Now we set T f G f be the closed linear span of T f G f under linear combinations.

<!-- formula-not-decoded -->

∣ where g b satisfies ( ∂/ ( ∂b )) g b | b =0 = h . For convenience, the tangent set for f at ( β, f ) is defined as Λ β,f := { S 2 ( β, f )[ h ] : h ∈ T f } . Further, we also define

<!-- formula-not-decoded -->

where h 1 is another function in T f .

As is well known, in the special case when the loss is a negative log-likelihood, the efficient score is essentially the projection of the score function S 1 onto the orthonormal complement of the tangent set Λ β,f . Denote Π θ,f as the orthogonal projection onto the closure of the linear span of Λ β,f . The efficient score function for β at ( β 0 , f 0 ) is then given by

<!-- formula-not-decoded -->

Denote

<!-- formula-not-decoded -->

where h = ( h 1 , h 2 , . . . , h p ) ∈ T p f . Then define S 12 [ h ] , S 21 [ h ] and S 22 [ h 1 ] [ h 2 ] accordingly. If there is ˜ h = ( ˜ h 1 , ˜ h 2 , . . . , ˜ h ∗ p ) ∈ T p f G f , such that for any h = ( h 1 , . . . , h k ) ∈ T p f G f ,

<!-- formula-not-decoded -->

Then we can also determine the efficient score for likelihood estimation as

<!-- formula-not-decoded -->

Similar to the negative log-likelihood, the following assumptions are made for the general loss function.

Assumption 5. (1). (Positive information) There exists ˜ h = ( ˜ h 1 , . . . , ˜ h p ) ∈ T p f G f such that (9) holds for any h = ( h 1 , . . . , h k ) ∈ T p f G f . Further assume that

<!-- formula-not-decoded -->

where s &gt; d/ 2 . Given ˜ h , the matrix

<!-- formula-not-decoded -->

is nonsingular.

(2). (Smoothness of the model) For all possible parameters ( β, f ) satisfying { ( β, f ) : ‖ β -β 0 ‖ + ‖ f -f 0 ‖ L 2 ( X ) /lessorsimilar δ n } with δ n = o ( n -1 / 4 ) ,

<!-- formula-not-decoded -->

In the above assumptions, the first one guarantees the existence and smoothness of directions ˜ h , which corresponds to the least favorable direction when the loss is the negative log-likelihood. This direction can be determined by solving the equations in (9). The second condition can usually be obtained by Taylor expansion. Under the above assumptions, we have the following general conclusion.

Theorem 3.1. Consider the proposed semiparametric neural M -estimation, assume that the estimation ( ˆ β t , ˆ f t ) is trained from (4) , (5) , (6) with training time t s in Remark 2.2 and network width m ( t s , ξ ) in Theorem 2.1 with ξ ∈ (0 , 1) . Then, with probability at least 1 -ξ over neural network random initialization, the following results holds: If ˆ β t s and Pl ˆ β t s , ˆ f t s are bounded, and Assumption 1, 3, 4, 5 hold, setting λ /equivasymptotic n -( d +1) / (2 d +1) , we have

<!-- formula-not-decoded -->

If Assumption 4 is replaced by Assumption 4 ′ , setting λ /equivasymptotic n -( d +1) / (2 s + d ) , the nonparametric rate becomes

<!-- formula-not-decoded -->

In both cases, we have the asymptotic normality

<!-- formula-not-decoded -->

where Σ = A -1 ( P { ˜ S ( β 0 , f 0 ) ˜ S ( β 0 , f 0 ) T } )( A -1 ) T .

The boundedness condition of ˆ β t s and Pl ˆ β t s , ˆ f t s is essentially a boundedness requirement for the parameter part, not for the nonparametric f . For nonparametric problems without the finitedimensional parameter β , this condition can be removed under the same proof. For semiparametric models, this condition is often verifiable in specific examples, as in the applications in the next section. Let s 0 = ( d +1) / 2 denote the smoothness of the RKHS corresponding to the NTK. The convergence rate in (10) can then be expressed as nearly n -2 s 0 / (2 s 0 + d ) with the tuning parameter λ /equivasymptotic n -2 s 0 / (2 s + d ) . This smoothness is essentially determined by the ReLU activation function, and the rate is minimax optimal for many statistics models with Assumption 4. When the true function does not lie within the considered ReLU NTK RKHS, as stated in Assumption 4 ′ , the rate in (11) remains minimax optimal. Furthermore, Assumption 4 ′ ensures that the nonparametric rate is faster than n -1 / 4 . Therefore, regardless of whether the true function resides in the considered RKHS, the estimator is shown to be asymptotically normal.

An estimator is semiparametric efficient if its information (i.e., the inverse of its variance) equals the supremum of the information across all parametric submodels. The submodel that attains this supremum is typically called the least favorable or hardest submodel. By the above result, when the loss function is the negative log-likelihood and the model class contains the hardest submodel, the proposed semiparametric estimator is efficient. Conversely, if the loss function differs, suggesting model misspecification, the parametric estimator remains √ n -consistent and asymptotically normal.

Weconcludethis section with the following remarks. To better align with practice, we consider overparameterized neural networks and general loss functions, allowing the true function to lie outside the desired RKHS. Distinct from existing literature, we refrain from assuming that the output of the optimized network is bounded and study the algorithmic solution, which presents significant challenges for statistical analysis and inspires a new yet weaker margin condition. Then, by combining the peeling argument with the interpolation bound, we precisely characterize the entropy and establish the nonparametric convergence rate. Based on this, we obtain the asymptotic

normality of the parametric estimators, with high probability over the random initialization, avoiding the degeneration of the tangent space and addressing the key question posed in Section 1.2. Lastly, we acknowledge the potential for broader applications of the proposed framework. This essentially provides a fundamental approach to addressing statistical problems that involve characterizing the tangent spaces of the nonparametric component, which has been a challenge that frequently arises in the intersection of deep learning and statistical inference.

## 4 Examples

In this section, we introduce two common examples to illustrate the proposed semiparametric neural M -estimator and the established theoretical framework.

## 4.1 Regression

Now we consider the regression model

<!-- formula-not-decoded -->

where Y i represents the response variable, X i and Z i are bounded covariates with densities bounded away from 0 and ∞ , and /epsilon1 i denotes the independent measurement noise with mean zero, finite variance and a symmetric distribution. For estimation, we define the loss function as l = l ( Y i -f ( X i ) -Z T i β ) , where l is a univariate function.

The following condition is for the covariates distribution and loss function.

## Assumption 6.

- (1) The matrix E [ ( Z -E [ Z | X ])( Z -E [ Z | X ]) T ] is nonsingular.
- (2) The univariate function l is a nonnegative, even and convex function with Lipschitz continuous derivative l ′ and l (0) = l ′ (0) = 0 . Additionally, the pointwise risk R ( s ) = E /epsilon1 [ l ( s + /epsilon1 )] is convex and R ′′ ( s ) ≥ B 3 &gt; 0 for s : | s | ≤ b with some b large enough and a positive number B 3 .

In this example, symmetry of the noise and loss functions is assumed to ensure that ( β 0 , f 0 ) is the minimizer of the risk E P l β,f , thereby simplifying the analysis. Condition (1) in Assumption 6 is standard (Kosorok, 2008) to ensure

<!-- formula-not-decoded -->

in partially linear models. Furthermore, the second condition assumes that the risk function is strongly convex within a local neighborhood. Many commonly used loss functions, such as least squares loss and Huber loss, satisfy this assumption. Assumption 6 guarantee that Assumption 3 holds, as shown in Lemma ?? in the Supplementary Material (Yan et al., 2025a).

Theorem 4.1. Consider the proposed semiparametric neural M -estimation for model (13) and set the hyperparameters as in Theorem 3.1. Under Assumption 4, 5(1), 6, setting λ /equivasymptotic n -( d +1) / (2 d +1) , with probability at least 1 -ξ over neural network random initialization, we have

<!-- formula-not-decoded -->

If Assumption 4 is replaced by Assumption 4 ′ , setting λ /equivasymptotic n -( d +1) / (2 s + d ) , the nonparametric rate becomes

<!-- formula-not-decoded -->

In both cases, we have the asymptotic normality

<!-- formula-not-decoded -->

where Σ = A -1 ( P { ˜ S ( β 0 , f 0 ) ˜ S ( β 0 , f 0 ) T } )( A -1 ) T .

Some previous works studied related challenges in the least squares nonparametric regression problem (Mendelson and Neeman, 2010; Steinwart et al., 2009). In the above theorem, whether the true function belongs to the considered RKHS, we establish the nonparametric optimal convergence rate and parameter asymptotic normality, without assuming that candidate functions are bounded and allowing unbounded responses.

## 4.2 Classification

In this subsection, we consider the following binary classification problem. Suppose that we can observe independently and identically distributed random sample { ( Y i , Z i , X i ) , i = 1 , 2 , · · · , n } , where Y i ∈ { 0 , 1 } denotes binary response, and X ′ i s and Z ′ i s are bounded covariates with densities bounded away from 0 and ∞ . We assume that Y follows a Bernoulli distribution determined by:

<!-- formula-not-decoded -->

where φ : R ↦→ [0 , 1] is a continuously differentiable monotone link function. Hence the loss function can be taken as the negative log-likelihood as

<!-- formula-not-decoded -->

Commonchoicesfor φ include the sigmoid function φ ( t ) = (1+ e -t ) -1 and the cumulative normal distribution function, corresponding to the logistic and probit models, respectively. In theoretical analysis, we allow for model misspecification, assuming that the data-generating process follows:

<!-- formula-not-decoded -->

where ψ = φ is a different link function, and some β 1 ∈ R p , f 1 ∈ L 2 (Ω) . Despite this potential misspecification, we continue to use the working link function φ and loss function as specified in (16). From a variational perspective, the minimizer ( β 0 , f 0 ) of the Pl β,f can also be interpreted as minimizer of the Kullback-Leibler divergence

/negationslash

<!-- formula-not-decoded -->

Thus, the parameters ( β 0 , f 0 ) retain interpretability and are still parameters in terms of the underlying distribution. Solving the equation (9), some calculation implies

<!-- formula-not-decoded -->

The following conditions are standard for the link function.

Assumption 7. The link function φ is Lipschitz continuous, monotone and continuously differentiable; -log φ ( t ) and -log(1 -φ ( t )) are both convex with positive, continuous and bounded second-order derivative. If the model is misspecified, we make the same assumptions for ψ and assume that the underlying β 1 , f 1 are bounded in infinity norm.

This condition ensures that the loss function and distribution satisfy Assumptions 1 and 3. Commonly used link functions, such as the logistic and probit functions, naturally satisfy this condition, thereby making these assumptions applicable in practice.

Theorem 4.2. Consider the proposed semiparametric neural M -estimation for model (15) or (17) and set the hyperparameters as in Theorem 3.1. Under Assumption 4, 5(1), 6(1) and 7, setting λ /equivasymptotic n -( d +1) / (2 d +1) , with probability at least 1 -ξ over neural network random initialization, we have

<!-- formula-not-decoded -->

If Assumption 4 is replaced by Assumption 4 ′ , setting λ /equivasymptotic n -( d +1) / (2 s + d ) , the nonparametric rate becomes

<!-- formula-not-decoded -->

In both cases, we have the asymptotic normality

<!-- formula-not-decoded -->

where Σ = A -1 ( P { ˜ S ( β 0 , f 0 ) ˜ S ( β 0 , f 0 ) T } )( A -1 ) T .

For the partially linear classification problem, the proposed neural M -estimation achieves the minimax nonparametric rate as well as √ n -consistency, with high probability. This performance is attributed to the representational capacity of overparameterized deep neural networks and their tangent space. When the model is correctly specified, i.e. the loss function corresponds to the negative log-likelihood, the resulting parametric estimator attains efficiency.

## 5 Numerical studies

This section demonstrates the practical advantages of the proposed semiparametric neural M -estimator through simulation studies on both regression and classification models described in Section 4. We use an overparameterized fully connected ReLU neural network with depth L = 5 and width m = 1000 . For training, the stochastic gradient descent optimizer in PyTorch is employed, with a learning rate of 0 . 001 and a total of 1000 epochs. For a comprehensive evaluation, we compare our method with four baseline approaches that also estimate β 0 and f 0 ( x ) via (penalized) M -estimation. The first baseline is a regression spline estimator using B-splines, with uniformly spaced knots over the domain [0 , 1] d . The second is the RKHS method, where the RKHS norm of the nonparametric component serves as the penalty, and the Laplacian kernel K h ( x 1 , x 2 ) = exp {-‖ x 1 -x 2 ‖ /h } is used. The third is a local linear estimator with the Epanechnikov kernel k h ( x 1 , x 2 ) = (1 -‖ x 1 -x 2 ‖ 2 /h ) / (2 d (1 -1 / (3 h ))) . The last one also employs the fully connected ReLU neural network with depth L = 5 , and the width serves as a hyperparameter, referred to as 'underparameterized' to be distinguished from the overparameterized regime.

To select the optimal hyperparameters for all methods, we split the full data into a training set (80%) and a validation set (20%) and choose the hyperparameters that minimize validation loss, including the tuning parameters for our method and the second baseline, the number of basis functions for the first baseline, the bandwidth for the third baseline, and the width of the underparameterized neural network. Specifically, the hyperparameters of the regression spline method and the underparameterized neural network method are chosen from a set that ensures the total number of parameters is less than the sample size.

We generate { Z i = ( Z 1 i , Z 2 i ) T } , i = 1 , 2 , . . . , n from a uniform distribution over the interval [0 , 1] 2 . Then, { X i = ( X 1 i , · · · , X di ) T } is generated using the formula X ji = 0 . 9 W ji +0 . 05( Z 1 i + Z 2 i ) , 1 ≤ j ≤ d , where W ji is sampled from a uniform distribution over the interval [0 , 1] . The true finite-dimensional parameter vector is set as β 0 = (1 , 0 . 75) T . For the nonparametric part f 0 ( x ) , we consider four cases with different dimensions and function forms. Here, Case 1 and Case 3 correspond to five-dimensional examples, while Case 2 and Case 4 represent their respective extensions to ten-dimension; which have been studied in the simulation of Zhong et al. (2022) and Yan et al. (2025b).

<!-- formula-not-decoded -->

In the regression setting, the responses Y i are generated by the model Y i = f 0 ( X i )+ Z T i β 0 + ε i , where ε i are i.i.d. normal noise with zero mean and standard deviation σ = 0 . 5 . For the classification model, the responses Y i are drawn from the Bernoulli distribution with probability P ( Y i = 1) = φ ( f 0 ( X i ) + Z T i β 0 -E X [ f 0 ( X )]) , where φ ( x ) is the logistic function φ ( x ) = 1 / (1+ e -x ) . Simulations are performed for three different sample sizes n = 500 , 1000 , 2000 , with 200 repetitions for each case and method. The mean squared error (MSE) of the parametric and nonparametric components is computed, respectively, to evaluate the performance of each method. Due to computational limitations, the spline method can only handle Case 1 and Case 3 with 5 -dimensional nonparametric functions. Tables 1 and 2 report the average MSEs for regression and classification examples, respectively. These results demonstrate that our overparameterized neural M -estimation approach outperforms the three traditional statistical methods, as well as the underparameterized neural network (i.e., properly tuned with fewer parameters than the sample size). Specifically, due to the high dimensionality of the nonparametric component, the curse of dimensionality becomes significant. Consequently, the underparameterized neural network, spline and the local linear estimator suffer from insufficient learning capacity. In all four cases, our method yields the lowest MSE, demonstrating its superior ability. Lastly, we would like to point out that, in the existing literature (e.g. Zhong et al., 2022; Fan and Gu, 2022; Wang et al.,

2024; Yan et al., 2025b), regardless of the theoretical requirements imposed on the network size, their numerical experiments have in fact employed overparameterized neural networks to achieve favorable performance, which also provide certain support for our proposed method and theory.

Given that the estimation of the parametric component satisfies asymptotic normality, it is possible to perform valid statistical inference. To assess it, we simulate the empirical coverage probabilities of the corresponding estimated confidence intervals. In the regression model, the variance of ˆ β is estimated as

<!-- formula-not-decoded -->

In the classification model, the variance of ˆ β is estimated as

<!-- formula-not-decoded -->

where F is also a neural network function class. The coverage rate is defined as the proportion of repeated experiments for which the true parameter falls within the confidence interval. Tables 3 and 4 report the coverage of the 95% confidence intervals constructed by the proposed method based on 500 repeated experiments, for each case in both regression and classification models. Generally, the coverage rate is near 0 . 95 for the proposed overparameterized neural M -estimation method, especially when the sample size n is large. This supports the potential usefulness of our proposed approach for semiparametric inference.

## References

- Ahmad, I., Leelahanon, S., and Li, Q. (2005). Efficient estimation of a semiparametric partially linear varying coefficient model. Annals of statistics .
- Allen-Zhu, Z., Li, Y., and Song, Z. (2019). A convergence theory for deep learning via overparameterization. In International conference on machine learning , pages 242-252. PMLR.
- Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R. R., and Wang, R. (2019). On exact computation with an infinitely wide neural net. Advances in neural information processing systems , 32.
- Bach, F. R. (2014). Breaking the curse of dimensionality with convex neural networks. CoRR , abs/1412.8690.

- Bartlett, P. L. and Mendelson, S. (2006). Empirical minimization. Probability theory and related fields , 135(3):311-334.
- Bauer, B. and Kohler, M. (2019). On deep learning as a remedy for the curse of dimensionality in nonparametric regression. The Annals of Statistics , 47(4):2261-2285.
- Bhattacharya, S., Fan, J., and Mukherjee, D. (2023). Deep neural networks for nonparametric interaction models with diverging dimension. arXiv preprint arXiv:2302.05851 .
- Bietti, A. and Bach, F. (2021). Deep equals shallow for reLU networks in kernel regimes. In International Conference on Learning Representations .
- Bietti, A. and Mairal, J. (2019). On the inductive bias of neural tangent kernels. Advances in Neural Information Processing Systems , 32.
- Chen, L. and Xu, S. (2020). Deep neural tangent kernel and laplace kernel have the same rkhs. arXiv preprint arXiv:2009.10683 .
- Chen, M., Jiang, H., Liao, W., and Zhao, T. (2019). Nonparametric regression on low-dimensional manifolds using deep relu networks: Function approximation and statistical recovery. ArXiv preprint. arXiv:1908.01842 .
- Chen, X., Liu, Y., Ma, S., and Zhang, Z. (2024). Causal inference of general treatment effects using neural networks with a diverging number of confounders. Journal of Econometrics , 238(1):105555.
- Cheng, G. and Huang, J. Z. (2010). Bootstrap consistency for general semiparametric mestimation. Annals of statistics .
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters.
- Cloninger, A. and Klock, T. (2021). A deep network construction that adapts to intrinsic dimensionality beyond the domain. Neural Networks , 141:404-419.
- Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society: Series B (Methodological) , 34(2):187-202.
- Cox, D. R. (1975). Partial likelihood. Biometrika , 62(2):269-276.

- Ding, Y. and Nan, B. (2011). A sieve m-theorem for bundled parameters in semiparametric models, with application to the efficient estimation in a linear model for censored data. Annals of statistics , 39(6):2795.
- Du, S. S., Zhai, X., P´ oczos, B., and Singh, A. (2018). Gradient descent provably optimizes over-parameterized neural networks. CoRR , abs/1810.02054.
- Fan, J. and Gu, Y. (2022). Factor augmented sparse throughput deep relu neural networks for high dimensional regression. ArXiv preprint. arXiv:2210.02002 .
- Farrell, M. H., Liang, T., and Misra, S. (2021). Deep neural networks for estimation and inference. Econometrica , 89(1):181-213.
- Geifman, A., Yadav, A., Kasten, Y., Galun, M., Jacobs, D., and Ronen, B. (2020). On the similarity between the laplace and neural tangent kernels. Advances in Neural Information Processing Systems , 33:1451-1461.
- Horowitz, J. L. (2012). Semiparametric methods in econometrics , volume 131. Springer Science &amp;Business Media.
- Hu, T., Wang, W., Lin, C., and Cheng, G. (2021). Regularization matters: A nonparametric perspective on overparametrized neural network. In International Conference on Artificial Intelligence and Statistics , pages 829-837. PMLR.
- Huang, J. (1999). Efficient estimation of the partly linear additive cox model. The Annals of Statistics , 27(5):1536-1563.
- Huang, K., Liu, M., and Ma, S. (2024). Nearly optimal learning using sparse deep relu networks in regularized empirical risk minimization with lipschitz loss.
- Jacot, A., Gabriel, F., and Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. Advances in neural information processing systems , 31.
- Jiao, Y., Shen, G., Lin, Y., and Huang, J. (2021). Deep nonparametric regression on approximate manifolds: Non-asymptotic error bounds with polynomial prefactors.
- Kohler, M. and Langer, S. (2021). On the rate of convergence of fully connected deep neural network regression estimates. The Annals of Statistics , 49(4):2231-2249.

- Kosorok, M. R. (2008). Introduction to empirical processes and semiparametric inference , volume 61. Springer.
- Lai, J., Xu, M., Chen, R., and Lin, Q. (2023a). Generalization ability of wide neural networks on r. arXiv preprint arXiv:2302.05933 .
- Lai, J., Yu, Z., Tian, S., and Lin, Q. (2023b). Generalization ability of wide residual networks.
- Lecu´ e, G. (2011). Interplay between concentration, complexity and geometry in learning theory with applications to high dimensional data analysis . PhD thesis, Universit´ e Paris-Est.
- Lee, J., Xiao, L., Schoenholz, S., Bahri, Y., Novak, R., Sohl-Dickstein, J., and Pennington, J. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems , 32.
- Li, Y. and Liang, Y. (2018). Learning overparameterized neural networks via stochastic gradient descent on structured data. Advances in neural information processing systems , 31.
- Li, Y., Yu, Z., Chen, G., and Lin, Q. (2024). On the eigenvalue decay rates of a class of neuralnetwork related kernel functions defined on general domains. Journal of Machine Learning Research , 25(82):1-47.
- Liang, H., Liu, X., Li, R., and Tsai, C.-L. (2010). Estimation and testing for partially linear single-index models. Annals of statistics , 38(6):3811.
- Lu, J., Shen, Z., Yang, H., and Zhang, S. (2021). Deep network approximation for smooth functions. SIAM Journal on Mathematical Analysis , 53(5):5465-5506.
- Ma, S. and Kosorok, M. R. (2005a). Penalized log-likelihood estimation for partly linear transformation models with current status data. The Annals of Statistics , 33(5):2256-2290.
- Ma, S. and Kosorok, M. R. (2005b). Robust semiparametric m-estimation and the weighted bootstrap. Journal of Multivariate Analysis , 96(1):190-217.
- Ma, S., Linton, O., and Gao, J. (2021). Estimation and inference in semiparametric quantile factor models. Journal of Econometrics , 222(1):295-323.
- Mammen, E. and Van de Geer, S. (1997). Penalized quasi-likelihood estimation in partial linear models. The Annals of Statistics , 25(3):1014-1035.

- Mendelson, S. and Neeman, J. (2010). Regularization in kernel learning. The Annals of Statistics .
- Nakada, R. and Imaizumi, M. (2020). Adaptive approximation and generalization of deep neural network with intrinsic dimensionality. J. Mach. Learn. Res. , 21(174):1-38.
- Raskutti, G., Wainwright, M. J., and Yu, B. (2014). Early stopping and non-parametric regression: an optimal data-dependent stopping rule. The Journal of Machine Learning Research , 15(1):335-366.
- Rosenbaum,P. R. and Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. Biometrika , 70(1):41-55.
- Schmidt-Hieber, J. (2019). Deep relu network approximation of functions on a manifold. ArXiv preprint. arXiv:1908.00695 .
- Schmidt-Hieber, J. (2020). Nonparametric regression using deep neural networks with ReLU activation function. The Annals of Statistics , 48(4):1875 - 1897.
- Shen, Z. (2020). Deep network approximation characterized by number of neurons. Communications in Computational Physics , 28(5):1768-1811.
- Shen, Z., Yang, H., and Zhang, S. (2022). Optimal approximation rate of relu networks in terms of width and depth. Journal de Math´ ematiques Pures et Appliqu´ ees , 157:101-135.
- Steinwart, I., Hush, D. R., Scovel, C., et al. (2009). Optimal rates for regularized least squares regression. In COLT , pages 79-93.
- Suh, N., Ko, H., and Huo, X. (2021). A non-parametric regression viewpoint: Generalization of overparametrized deep relu network under noisy observations. In International Conference on Learning Representations .
- Suzuki, T. (2018). Adaptivity of deep relu network for learning in besov and mixed smooth besov spaces: optimal rate and curse of dimensionality. ArXiv preprint. arXiv:1810.08033 .
- Tang, W., He, K., Xu, G., and Zhu, J. (2023). Survival analysis via ordinary differential equations. Journal of the American Statistical Association , 118(544):2406-2421.
- Tsiatis, A. A. (2006). Semiparametric theory and missing data , volume 4. Springer.

- Tsybakov, A. B. (2004). Optimal aggregation of classifiers in statistical learning. The Annals of Statistics , 32(1):135-166.
- Vakili, S., Bromberg, M., Shiu, D., and Bernacchia, A. (2021). Uniform generalization bounds for overparameterized neural networks. CoRR , abs/2109.06099.
- Van de Geer, S. A. (2000). Empirical Processes in M-estimation , volume 6. Cambridge university press.
- Van der Vaart, A. W. (2000). Asymptotic statistics , volume 3. Cambridge university press.
- Wang, J.-L., Xue, L., Zhu, L., and Chong, Y. S. (2010). Estimation for a partial-linear single-index model. Annals of statistics .
- Wang, X., Zhou, L., and Lin, H. (2024). Deep regression learning with optimal loss function. Journal of the American Statistical Association , pages 1-20.
- Yan, S., Chen, Z., and Yao, F. (2025a). Supplement to 'semiparametric estimation with overparameterized neural network'.
- Yan, S., Yao, F., and Zhou, H. (2025b). Deep regression for repeated measurements. Journal of the American Statistical Association , 0(ja):1-23.
- Yao, Y., Rosasco, L., and Caponnetto, A. (2007). On early stopping in gradient descent learning. Constructive approximation , 26(2):289-315.
- Yarotsky, D. (2017). Error bounds for approximations with deep relu networks. Neural Networks , 94:103-114.
- Yarotsky, D. (2018). Optimal approximation of continuous functions by very deep relu networks. In Conference on learning theory , pages 639-649. PMLR.
- Yarotsky, D. and Zhevnerchuk, A. (2020). The phase diagram of approximation rates for deep neural networks. Advances in neural information processing systems , 33:13005-13015.
- Zeng, D. and Lin, D. (2007). Maximum likelihood estimation in semiparametric regression models with censored data. Journal of the Royal Statistical Society Series B: Statistical Methodology , 69(4):507-564.

- Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. (2021). Understanding deep learning (still) requires rethinking generalization. Communications of the ACM , 64(3):107-115.
- Zhong, Q., M¨ uller, J., and Wang, J.-L. (2022). Deep learning for the partially linear Cox model. The Annals of Statistics , 50(3):1348 - 1375.
- Zhong, Q., M¨ uller, J. W., and Wang, J.-L. (2021). Deep extended hazard models for survival analysis. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W., editors, Advances in Neural Information Processing Systems , volume 34, pages 15111-15124. Curran Associates, Inc.

Table 1: The mean square error ( × 10 -1 ) for ˆ β and ˆ f of our method and baselines for the regression model.

| Setting   | Setting   | Proposed    | Spline      | RKHS        | Local Linear   | Underpara   |
|-----------|-----------|-------------|-------------|-------------|----------------|-------------|
| Case      | n         | MSE for ˆ β | MSE for ˆ β | MSE for ˆ β | MSE for ˆ β    | MSE for ˆ β |
|           | 500       | 0.2613      | 0.4987      | 0.6905      | 0.4148         | 2.0760      |
| Case 1    | 1000      | 0.1140      | 0.1411      | 0.2511      | 0.1696         | 1.9988      |
|           | 2000      | 0.0623      | 0.0513      | 0.0793      | 0.1607         | 1.9965      |
|           | 500       | 0.2429      | -           | 3.7362      | 132.79         | 3.1735      |
| Case 2    | 1000      | 0.1132      | -           | 0.7956      | 20.478         | 3.0583      |
|           | 2000      | 0.0494      | -           | 0.1922      | 2.1419         | 3.0356      |
|           | 500       | 0.2316      | 0.6546      | 0.3940      | 116.13         | 7.7303      |
| Case 3    | 1000      | 0.1066      | 0.1801      | 0.1895      | 78.535         | 8.1188      |
|           | 2000      | 0.0496      | 0.0755      | 0.0662      | 119.39         | 8.1298      |
|           | 500       | 0.2598      | -           | 0.4656      | 28.555         | 12.192      |
| Case 4    | 1000      | 0.1098      | -           | 0.1908      | 88.956         | 12.195      |
|           | 2000      | 0.0457      | -           | 0.0836      | 114.98         | 12.424      |
|           | MSE       | MSE         | MSE         | MSE         | MSE            | MSE         |
|           | 500       | 0.8625      | 46.266      | 6.6327      | 2.5206         | 2.8144      |
| Case 1    | 1000      | 0.7015      | 3.4216      | 3.0720      | 2.4176         | 2.6657      |
|           | 2000      | 0.6179      | 0.7906      | 1.5700      | 2.3582         | 2.6089      |
|           | 500       | 0.7824      | -           | 17.558      | 67.927         | 2.8644      |
| Case 2    | 1000      | 0.6084      | -           | 7.4870      | 11.378         | 2.7192      |
|           | 2000      | 0.5093      | -           | 3.5917      | 2.1836         | 2.6467      |
|           | 500       | 0.6153      | 67.462      | 4.6669      | 89.284         | 40.214      |
| Case 3    | 1000      | 0.4135      | 7.1053      | 2.3364      | 84.843         | 50.257      |
|           | 2000      | 0.2956      | 2.6464      | 1.2638      | 89.256         | 68.170      |
|           | 500       | 0.9107      | -           | 10.753      | 75.208         | 51.472      |
| Case 4    | 1000      | 0.4489      | -           | 5.5210      | 84.909         | 51.209      |
|           | 2000      | 0.2606      | -           | 3.0471      | 93.994         | 51.154      |

Table 2: The mean square error ( × 10 -1 ) for ˆ β and ˆ f of our method and baselines for the classification model (15).

| Setting   | Setting   | Proposed    | Spline      | RKHS        | Local Linear   | Underpara   |
|-----------|-----------|-------------|-------------|-------------|----------------|-------------|
| Case      | n         | MSE for ˆ β | MSE for ˆ β | MSE for ˆ β | MSE for ˆ β    | MSE for ˆ β |
|           | 500       | 0.8821      | 13.561      | 8.8990      | 6.5733         | 2.4930      |
| Case 1    | 1000      | 0.4411      | 12.637      | 2.9707      | 2.3129         | 1.7517      |
|           | 2000      | 0.2133      | 12.583      | 1.5815      | 1.8543         | 1.6526      |
|           | 500       | 2.1877      | -           | 8.1133      | 12.787         | 4.4772      |
| Case 2    | 1000      | 0.5395      | -           | 2.6608      | 11.939         | 4.1563      |
|           | 2000      | 0.2177      | -           | 1.5869      | 11.583         | 4.1018      |
|           | 500       | 1.0119      | 21.165      | 22.105      | 13.400         | 4.4189      |
| Case 3    | 1000      | 0.5900      | 19.280      | 4.2309      | 11.161         | 4.7265      |
|           | 2000      | 0.3318      | 17.656      | 1.1527      | 8.0244         | 4.5130      |
|           | 500       | 1.6781      | -           | 5.5813      | 20.256         | 3.8216      |
| Case 4    | 1000      | 0.6652      | -           | 2.3483      | 19.570         | 3.7650      |
|           | 2000      | 0.2812      | -           | 2.5641      | 19.158         | 3.6724      |
|           | MSE       | MSE         | MSE         | MSE         | MSE            | MSE         |
|           | 500       | 4.5624      | 39.087      | 9.0062      | 61.677         | 6.9442      |
| Case 1    | 1000      | 2.5099      | 39.043      | 5.0402      | 59.122         | 5.6270      |
|           | 2000      | 1.7524      | 38.999      | 3.9864      | 58.263         | 5.2535      |
|           | 500       | 22.417      | -           | 7.5404      | 29.051         | 11.793      |
| Case 2    | 1000      | 5.0438      | -           | 4.0622      | 28.991         | 11.529      |
|           | 2000      | 1.8980      | -           | 3.1317      | 29.023         | 10.751      |
|           | 500       | 7.0489      | 96.863      | 29.990      | 92.969         | 85.572      |
| Case 3    | 1000      | 3.5228      | 96.762      | 14.225      | 91.686         | 86.227      |
|           | 2000      | 2.2837      | 96.850      | 11.259      | 90.390         | 86.411      |
|           | 500       | 17.616      | -           | 38.825      | 51.957         | 46.156      |
| Case 4    | 1000      | 5.9230      | -           | 35.755      | 51.703         | 45.684      |
|           | 2000      | 3.6942      | -           | 35.791      | 51.598         | 45.623      |

Table 3: The coverage probability for constructed 95% confidence interval for β = ( β 1 , β 2 ) in the regression model.

| Model   | The coverage rate for β 1   | The coverage rate for β 1   | The coverage rate for β 1   | The coverage rate for β 2   | The coverage rate for β 2   | The coverage rate for β 2   |
|---------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
| Setting | n = 500                     | n = 1000                    | n = 2000                    | n = 500                     | n = 1000                    | n = 2000                    |
| Case 1  | 0.938                       | 0.958                       | 0.946                       | 0.944                       | 0.952                       | 0.940                       |
| Case 2  | 0.966                       | 0.958                       | 0.950                       | 0.960                       | 0.950                       | 0.952                       |
| Case 3  | 0.946                       | 0.970                       | 0.948                       | 0.968                       | 0.952                       | 0.950                       |
| Case 4  | 0.986                       | 0.974                       | 0.938                       | 0.988                       | 0.968                       | 0.948                       |

Table 4: The coverage probability for constructed 95% confidence interval for β = ( β 1 , β 2 ) in the classification model.

| Model   | The coverage rate for β 1   | The coverage rate for β 1   | The coverage rate for β 1   | The coverage rate for β 2   | The coverage rate for β 2   | The coverage rate for β 2   |
|---------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
| Setting | n = 500                     | n = 1000                    | n = 2000                    | n = 500                     | n = 1000                    | n = 2000                    |
| Case 1  | 0.964                       | 0.968                       | 0.950                       | 0.966                       | 0.954                       | 0.938                       |
| Case 2  | 0.936                       | 0.950                       | 0.952                       | 0.950                       | 0.962                       | 0.940                       |
| Case 3  | 0.942                       | 0.926                       | 0.932                       | 0.972                       | 0.942                       | 0.930                       |
| Case 4  | 0.940                       | 0.946                       | 0.942                       | 0.970                       | 0.928                       | 0.962                       |