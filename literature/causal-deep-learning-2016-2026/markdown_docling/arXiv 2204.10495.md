<!--
source: /Users/pranjal/Code/deep-inference/literature/causal-deep-learning-2016-2026/downloads/arXiv 2204.10495.pdf
backend: docling
part: 1/1
-->

## Adversarial Estimators

Jonas Metzger Stanford University

June 19, 2022

## Abstract

We develop an asymptotic theory of adversarial estimators ('A-estimators'). They generalize maximum-likelihood-type estimators ('M-estimators') as their average objective is maximized by some parameters and minimized by others. This class subsumes the continuous-updating Generalized Method of Moments, Generative Adversarial Networks and more recent proposals in machine learning and econometrics. In these examples, researchers state which aspects of the problem may in principle be used for estimation, and an adversary learns how to emphasize them optimally. We derive the convergence rates of A-estimators under pointwise and partial identification, and the normality of functionals of their parameters. Unknown functions may be approximated via sieves such as deep neural networks, for which we provide simplified low-level conditions. As a corollary, we obtain the normality of neural-net M-estimators, overcoming technical issues previously identified by the literature. Our theory yields novel results about a variety of A-estimators, providing intuition and formal justification for their success in recent applications.

## 1 Introduction

Although it is not always obvious, nearly all population parameters that are estimated in econometrics and machine learning can be written as the solution of so-called saddle-point or adversarial objectives of the form:

<!-- formula-not-decoded -->

where l is a known loss function, Y is a random variable and Θ , Λ are parameter spaces, containing the unknown parameter of interest θ ∗ and nuisance λ . We examine the natural estimator ̂ θ n that approximately solves the empirical Nash condition:

<!-- formula-not-decoded -->

which replaces the expectation E of the population objective 1.1 with the average of n iid samples, E n . We search for the estimators over so-called sieve spaces ̂ θ n , ̂ λ n ∈ Θ n , Λ n (Grenander [1981]), which approximate the full parameter spaces Θ n , Λ n ⊂ Θ , Λ and grow with the sample size n . These could be neural networks for example, growing in depth and width. The sequences ˜ η n , η n = o P (1) accommodate numerical procedures which only yield approximate Nash equilibria. This class of A-estimators (A for adversarial ) strictly generalizes so-called M-estimators (M for maximum likelihood-type ), which are obtained by fixing Λ to be singleton.

A-Estimators have become a workhorse of econometrics and causal inference long before the advent of deep learning. Hansen et al. [1996]'s continuous-updating Generalized Methods of Moments (GMM), which looks for θ satisfying E [ m ( θ, Y )] = 0 for some known function m ( θ, Y ), can be written as:

<!-- formula-not-decoded -->

and is therefore an A-estimator, but not an M-estimator. In statistics, an earlier example consists of the Empirical Likelihood (EL) approach pioneered in Cosslett [1981], Owen [1988, 1990], Qin and Lawless [1994]. Subsequently, EL was unified with GMM into the Generalized Empirical Likelihood (GEL) framework (Newey and Smith [2004]), also subsuming the exponential-tilting estimator (Imbens et al. [1998]), for example. All GEL estimators are A-estimators, but their adversarial formulation was rarely salient. However, some of their benefits may be owed directly to their adversarial objective: the adversary λ automatically detects which moment violations are most informative at a given parameter guess, adaptively guiding the estimation towards an efficient solution. This contrasts with earlier estimators which weighted the moments in a way that depended on choices of the researcher: the weights of Pearson's Methodof-Moments were manually set by the researcher (implicitly), resulting in inefficient rootn asymptotics. Two-step GMM (Hansen [1982]) required choosing a first-step estimator to compute the weights, yielding inefficient higher-order asymptotics (see Newey and Smith [2004]). Formally, the optimal weights are nuisance parameters, and as we will see in Section 3.2, estimating them via an adversary ensures that ̂ θ n is robust to estimation errors in these nuisance parameters.

A key invention which put a spotlight on adversarial objectives in recent years were Generative Adversarial Networks , or GANs (Goodfellow et al. [2014]). They search for a generative model Y ∼ P θ for which no adversary λ ( Y ) ∈ (0 , 1) (called 'critic' or 'discriminator') could tell apart the generated data from n real samples:

<!-- formula-not-decoded -->

The objective contains the log-likelihood of a binary classifier λ ( Y ) discriminating between an equal number of real and generated samples. As we show in Section 2.1, this directly measures the Jensen-Shannon divergence between P θ and P n . As of today, versions of this objective are key to state-of-the-art image generation, see e.g. Jabbar et al. [2022] for a recent survey. An analogy to human-generated images makes this unsurprising: it is much easier to tell apart a photo from an image drawn by a human, than it is to draw a realistic image, or to define what makes a drawing realistic. This intuition motivates the objective: train the generator until its critic has nothing more to criticize. The ingenuity is that the researcher need not define a meaningful measure of 'realism' of a piece of data anymore. Instead, this measure is learned by the adversary. It is clear that the utility of this idea extends beyond image generation: in Imitation Learning, a sub-field of Robotics, it has been used to teach human behavior to artificial agents without requiring hand-crafted measures of 'humanness' (Ho and Ermon [2016]). In Econometrics, where new causal inference methods can usually only be benchmarked on simulated data sets, Athey et al. [2019] used the objective to limit the impact of researcher's subjective choices by requiring simulations to be indistinguishable from real data. Kaji et al. [2020] proposed to use the objective to estimate structural economic models which produce realistic data beyond the set of features that would otherwise be manually specified by the researcher.

More generally, other adversarial objectives have proven useful beyond fitting models to data. In Reinforcement Learning, a sub-field of Robotics where agents independently discover strategies to reach predefined goals without copying prior examples, Dai et al. [2018] proposed an A-estimator in which the adversary detects and penalizes any systematic deviation from optimal behavior. Cotter et al. [2019] proposed an estimator which extends a standard ML objective by an adversary imposing fairness

constraints across sub-populations. More recently, research in econometrics established A-estimators as a natural framework for integrating machine learning methods into causal inference, where quantities of interest are frequently identified by a continuum of restrictions. Chernozhukov et al. [2020] propose to estimate Riesz representers of causal parameters directly, via an adversary enforcing the restrictions identifying the Riesz representer. Estimating Riesz representers is key to obtaining well-behaved estimates of causal parameters in the presence of nuisance functions, and can also be useful for estimating asymptotic variances, e.g. Chen et al. [2019]. Another line of research develops novel adversarial objectives to estimate causal parameters from conditional moment restrictions, which naturally arise from causal assumptions (e.g. the instrumental variable setting), and are usually more informative than any finite set of unconditional moment restrictions. In this line of research, the adversary can be viewed as adaptively finding the unconditional moment restriction which is most violated at the current parameter guess, among infinitely many which are implied by the conditional moment restriction. The key works are Lewis and Syrgkanis [2018], Dikkala et al. [2020] and Bennett et al. [2019b], Bennett and Kallus [2020]. Metzger [2022] propose a semi-parametrically efficient generalization of GEL to the conditional case via adversarial networks, containing Bennett and Kallus [2020] as a special case.

In summary, a recurring theme of adversarial objectives is that instead of manually defining which specific features of the data are important for a model to capture, the researcher's role is restricted to stating a general principle which should be satisfied by all features of the correct model, and the adversary adaptively focuses the estimation on the model's features which violate this principle the most. Over the course of the paper, we will encounter further interesting connections between various A-estimators, such as their Neyman orthogonality, their information-theoretic foundation via f-Divergences, and their ties to Lagrangian Duality.

Despite their popularity, we are not aware of a unified statistical theory of A-estimators. For some individual estimators, consistency (Bennett et al. [2019b]) and convergence rate results (Dikkala et al. [2020], Singh et al. [2018], Liang [2021], Belomestny et al. [2021]) were obtained, but normality results are limited to parametric θ , either in Kernel settings (Bennett and Kallus [2020]) or leaving high-level assumptions about neural networks unverified (Kaji et al. [2020]). This can be attributed to two main

obstacles: the theory of M-estimation does not apply to A-estimators, and the arguments from which the former is built up are insufficient to e.g. obtain the required uniform convergence of the adversary. The second issue is that adversarial objectives are most popular in the context of (deep) neural networks, whose statistical analysis (particularly their asymptotic normality) is complicated, e.g. due to their non-convex sieve space. Even in M-estimation settings, it was not clear whether known, high-level conditions for normality could be verified for neural networks (cf. the Conclusion of Shen et al. [2019]). We therefore make three separate contributions:

1. We characterize the general class of A-estimators, and show that a wide range of estimators proposed in econometrics and machine learning fall into this class. We point out desirable characteristics shared between A-estimators, which help explain their recent success in practice.
2. We develop a unified statistical theory of A-estimators, yielding their consistency, convergence rates (both under pointand partial identification), and asymptotic normality of functionals of their parameters. We provide highlevel conditions for arbitrary sieves, as well as low-level conditions for semiparametric settings with neural networks, to simplify verification in practice.
3. We extend the theory of neural network M-estimators (as a special case). Our convergence rates hold uniformly over families of losses, allow more general losses than Farrell et al. [2018] and attain a reduced curse-of-dimensionality which Nakada and Imaizumi [2020], Bauer et al. [2019] observed in regression settings with lower-dimensional structures. To the best of our knowledge, we provide the first normality result for functionals of deep neural networks which does not rely on Neyman-orthogonality or unverified high-level assumptions.

The remainder of the paper is structured as follows. In Section 2, we review five different A-estimators proposed in the econometrics and machine learning literatures. We present our general statistical theory of A-estimators in Section 3 and apply it in Section 4 to derive novel results about the examples of Section 2. We conclude by recapping the similar role adversaries play across all examples, providing intuition which types of problems may generally benefit from adversarial formulations. Appendix C and Online Appendix D contain the proofs omitted in Sections 3 and 4, respectively.

## 2 Examples

## 2.1 Minimum f -Divergence

A powerful class of estimation objectives asymptotically minimize an f -divergence D f ( P θ ‖ P ) between the distribution of the data Y ∼ P = P θ ∗ and the distribution of some model P θ , θ ∈ Θ n with support Y . This class, introduced by Nowozin et al. [2016], subsumes GANs (Goodfellow et al. [2014]), and many follow ups such as Mao et al. [2017], Tao et al. [2018]. For a continuous, proper convex function f : R ↦→ R satisfying f (1) = 0, the f -divergence is defined as D f ( P θ ‖ P ) = EP [ f ( d P θ ( Y ) d P ( Y ) )], where d P θ ( Y ) d P ( Y ) denotes the Radon-Nikodym derivative of P θ with respect to P (=likelihood ratio), which we assume exists for all θ ∈ Θ. Notably, D f ( P θ ‖ P ) admits a useful dual representation:

<!-- formula-not-decoded -->

where f ∗ ( t ) := sup λ ∈ R λt -f ( λ ) denotes the convex conjugate of f . The equality above follows from f = ( f ∗ ) ∗ . Various choices 1 for f are presented in Table 2.1. This duality is useful because the right-hand side suggests a finite-sample analog which does not depend on unknown quantities: we obtain an A-estimator for θ ∗ by letting

<!-- formula-not-decoded -->

and solving for ̂ θ n , ̂ λ n satisfying the Nash condition 1.2,1.3 in E n [ l ( θ, λ, Y )]. Normalizing f ( t ) ← f ( t ) -f ′ (1)( t -1) f ′′ (1) without loss of generality 2 , assuming the second derivative f ′′ exists, the function λ attaining the supremum in 2.1 at some θ is λ θ ∗ = f ′ ( d P θ d P ). The adversary ̂ λ n therefore estimates this transformed likelihood ratio at the current guess for ̂ θ n , and the Nash-equilibrium corresponds to the case where it is approximately constant, i.e. the distribution P ̂ θ n is close to that of the data. Notably, E n [ l ( θ, λ, Y )] can be evaluated using only samples from the two distributions 3 . This is crucial for

1 None of the objectives are unique: f ( t ) ← f ( t ) + c ( t -1) for any c yields the same divergence, but changes the expressions. Note that we may also swap E P θ and E n , which yields valid objectives for the respective 'reverse' f -divergences.

2 This implies f ′ (1) = 0, f ′′ (1) = 1 and f ∗ (0) = 0 , f ′ ∗ (0) = f ′′ ∗ (0) = 1, which merely re-scales the divergence 2.1 by a factor of 1 /f ′′ (1)

3 Note that we neither require explicit knowledge of P θ nor infinitely many samples from P θ at a given n : it suffices to draw m /follows n 2 Monte Carlo samples from P θ and solve for the corresponding

GANs, where P θ is only implicitly defined via a push-forward mapping parametrized by a neural net. As proposed by Kaji et al. [2020], this also makes it a drop-in alternative to the Simulated Method of Moments , which similarly estimates economic models from data they generate, but matches only a finite set of moments instead of the full distribution.

| Name              | f ( t )                    | f ∗ ( t ) , domain            | Generative Adversarial Objective for θ                  |
|-------------------|----------------------------|-------------------------------|---------------------------------------------------------|
| Total Variation   | &#124; t - 1 &#124; / 2    | t , for &#124; t &#124; ≤ 1 2 | sup &#124; λ &#124;≤ 1 2 E P θ λ ( Y ) - E n λ ( Y )    |
| KL Divergence     | t log t                    | e t - 1                       | sup λ ∈ R 1+ E P θ λ ( Y ) - E n e λ ( Y )              |
| Reverse KL        | - log t                    | - log( - te ), for t ≤ 0      | sup λ ≤ 0 1+ E P θ λ ( Y )+ E n log( - λ ( Y ))         |
| χ 2 Divergence    | ( t - 1) 2                 | t + t 2 / 4                   | sup λ ∈ R E P θ λ ( Y ) - E n [ λ ( Y )+ λ ( Y ) 2 / 4] |
| Squared Hellinger | ( √ t - 1) 2               | t 1 - t , for t ≤ 1           | sup λ ≤ 1 E P θ λ ( Y ) - E n [ λ ( Y ) 1 - λ ( Y ) ]   |
| rescaled JS (GAN) | t log t (1+ t ) log(1+ t ) | log(1 e t ) , for t < 0       | sup log λ< 0 E P θ log λ ( Y )+ E n log(1 λ ( Y ))      |

-

-

-

-

Table 1: Various adversarial f -divergence objectives. f ∗ ( t ) = ∞ outside the domain.

## 2.2 Generalized Empirical Likelihood

Our next example is a class of A-estimators that was proposed long before the recent success of adversarial objectives in deep learning. In econometrics, many important parameters θ ∗ are identified by a moment restriction of the form:

<!-- formula-not-decoded -->

for some known, possibly vector-valued function m ( Y, θ ). In the Introduction, we presented the continuous-updating GMM objective (Hansen et al. [1996]) for estimating θ ∗ , a workhorse for causal inference in econometrics. In this section, we review the more general class of Generalized Empirical Likelihood (GEL) estimators (Newey and Smith [2004]), which solve the constrained minimization problem:

<!-- formula-not-decoded -->

That is, they seek for a parameter θ and a corresponding population distribution ¯ P that is as close as possible to the sample P n , subject to satisfying the moment

finite sample saddle point. The resulting Monte Carlo approximation error for the expectation E P θ is then of order √ m -1 = n -1 and can thus be accounted for by letting ˜ η n , η n = O P ( n -1 ) in equations 1.2,1.3, which has no impact on our asymptotic results.

constraint E ¯ P [ m ( Y, θ )] = 0. At this high level, it is worth noting that GEL optimizes the same target as the objective in Section 2.1, which imposes ¯ P = P θ instead of a moment constraint. Glossing over some details, we can obtain a tractable estimator in this setting by concentrating out ¯ P from the corresponding Lagrangian:

<!-- formula-not-decoded -->

Which again uses the convex conjugate f ∗ of f (see example 2.1). For a formal proof of this equivalence, see e.g. Imbens et al. [1998]. It is easy to see that GEL is an A-estimator with l ( θ, λ, Y ) = -f ∗ ( λ ′ m ( Y, θ )) where Λ n = R dim( m ) and Θ n is the parameter space of the economic model. A particularly popular version of this objective corresponds to the case D f = χ 2 , where Table 2.1 tells us that f ( t ) = ( t -1) 2 and f ∗ ( t ) = t + t 2 / 4. In this case, we can analytically solve for the optimal adversary given θ . Substituting it in, we get the continuous-updating GMM objective presented in the introduction:

<!-- formula-not-decoded -->

## 2.3 Off-Policy Reinforcement Learning

Next, we review the Smoothed Bellman Error Embedding (SBEED) algorithm introduced by Dai et al. [2018], a popular off-policy learning method in robotics. Off-policy learning aims to learn the optimal policy for an agent from data that was generated under an entirely different policy regime. This problem is not limited to robotics: since it was identified in the monetary policy context by Lucas [1976], it became a primary concern in econometrics and its recognition played a key role in the credibility revolution (Angrist and Pischke [2010]) of econometrics. While problem definitions otherwise differ between these literatures, off-policy learning methods have received recent interest in econometrics (Zhan et al. [2021], Athey and Wager [2021]).

For an agent receiving reward R ( s, a ) for taking action a ∈ A at state s ∈ S , forming an expectation over the future state s + ∈ S , SBEED's goal is to learn the value function V ∗ ( s ) and policy a ∼ P ∗ ( ·| s ) which satisfy the regularized Bellman equation:

<!-- formula-not-decoded -->

where the entropy H ( P, s ) = -E a ∼ P ( ·| s ) [log P ( a | s )] regularizes the optimal policy P ∗ ( ·| s ) towards exploring all actions a ∈ A . Given the researcher's choice of R, β , the goal is to learn P ∗ , V ∗ from finite samples { ( s i , a i , s + i ) } n i =1 . Importantly, the actions a i may be sampled from a suboptimal policy which does not equal P ∗ . Starting from the first-order condition of the Bellman equation, Dai et al. [2018] develop an adversarial population objective, whose finite-sample analog is the A-estimator 1.2,1.3 with loss:

<!-- formula-not-decoded -->

where λ ( s, a ), log P θ ( a | s ) and V θ ( s ) are implemented as neural networks in practice.

## 2.4 A-Estimators for Conditional Moment Restrictions

Another powerful application for A-estimators recently pursued by the econometric literature are conditional moment estimators. These methods estimate parameters θ ∗ which are identified by restrictions of the form:

<!-- formula-not-decoded -->

for some random variables Y = ( X,Z ) and a known function m ( X,θ ). Conditions of this type occur e.g. when estimating some causal effect θ via instrumental variables, or as the first-order conditions of agents optimizing some expected utility given some information Z . As a result, nonparametric conditional moment estimators received considerable interest in econometrics, see e.g. Ai and Chen [2003, 2007], Chen and Qiu [2016]. These earlier estimators rely on first-step estimates of nuisance parameters capturing the conditional means and variances. Intuitively however, estimating the nuisance parameters via predictive objectives in a separate first step may dedicate scarce model capacity to capturing features which are not useful for the purpose of estimating θ ∗ downstream. This motivates recent work on adversarial objectives which unify the estimation into a single objective, more plausibly targeting the nuisance estimation towards the goal of identifying θ ∗ . Specifically, we will examine the estimator of Dikkala et al. [2020], with l ( θ, λ, Y ) = m ( X,θ ) ′ λ ( Z ) -1 4 ‖ λ ( Z ) ‖ 2 2 , yielding the finite sample objective

<!-- formula-not-decoded -->

where Λ n is a class of neural networks. The methods proposed by Bennett et al. [2019a], Bennett and Kallus [2020] are closely related, but differ in the penalty they impose on λ . Dikkala et al. [2020] consider the case of instrumental variable regression, where X = ( y, x ) and m ( X,θ ) = y -θ ( x ), but we will examine the general case. We note that Example 2.3 (SBEED, Dai et al. [2018]) can be viewed as a special case of re-scaled version of this objective, with X = ( s, a, s + ) and Z = ( s, a ), although both literatures seem to be unaware of their connection. One can analytically solve for the optimal adversary λ θ ∗ ( Z ) = 2 E [ m ( X,θ ) | Z ] to rewrite the population objective as:

<!-- formula-not-decoded -->

which can be understood as a measure of distance between θ and θ ∗ , which clearly attains its minimum at θ = θ ∗ , when E [ m ( X,θ ) | Z ] ≡ 0. In Section 4.4, we will apply our theory to derive the asymptotic distribution of this estimator and show that is in fact inefficient . We further discuss how the adversarial formulation of GMM can directly inform a simple modification similar to Bennett and Kallus [2020] which yields an efficient A-estimator.

## 2.5 Estimating Riesz Representers

Chernozhukov et al. [2020] propose a distinct A-estimator to estimate Riesz representers for structural parameters φ ∗ which can be written as linear functionals φ ∗ = φ ( g ∗ ) = E [ m ( Y, g ∗ )]. Here, g ∗ = E [ y | x ] is an unknown function for which an estimate ̂ g n is available from some first-stage regression of y on x , where Y = ( y, x, w ). Quantities like φ ∗ are common in the average treatment effect or asset pricing literature, for example. Unfortunately, especially if ̂ g n is estimated via machine learning, the 'naive' estimator

̂ φ n = E n [ m ( Y, ̂ g n )] is often not well behaved: √ n ( ̂ φ n -φ ∗ ) may not converge in distribution to a Gaussian limit and thus one cannot provide confidence intervals around the estimate. Under the conditions of the Riesz representation theorem however, there may exist a function θ ∗ ∈ Θ called the Riesz representer of the functional φ ( g ), which satisfies:

<!-- formula-not-decoded -->

If a well-behaved estimate ̂ θ n of θ ∗ is available, it can be combined with ̂ g n to define the so-called orthogonalized estimator:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which attains asymptotic normality under rather weak conditions on ̂ g n (see Lemma 17 of Chernozhukov et al. [2020]). Chernozhukov et al. [2020] propose a generalized procedure to estimate ̂ θ n via an A-estimator, which we will simplify as follows:

where Θ n , Λ n are neural networks. To clarify why this objective works, is it useful to analytically solve the adversarial component of the corresponding population objective:

<!-- formula-not-decoded -->

As we will show in Section 4.5, our theory directly yields the convergence rates for ̂ θ n that Chernozhukov et al. [2020]'s Lemma 17 requires for the asymptotic normality of ˜ φ n . It does so at a reduced curse of dimensionality in x for rather general function classes -i.e. under weaker conditions on smoothness and dimension of the data complementing the original work.

## 3 General Theory

Roadmap. This Section will present our general theory of A-estimators. Subsection 3.1 briefly discusses an alternative definition of A-estimators that may be more natural to some readers. In Subsection 3.2, we establish that A-estimators satisfy the desirable condition of Neyman-orthogonality with respect to the adversary and discuss its implications. Next, we characterize the convergence rates of A-estimators: Section 3.3 provides a high-level result for arbitrary sieves such as splines or wavelets, not just neural nets. Under more easily verifiable low-level conditions, Subsection 3.4 provides convergence rates for semiparametric settings involving neural networks, showing they exhibit a reduced curse-of-dimensionality. Finally, we characterize the asymptotic normality of smooth functionals of A-estimators. We again begin with a

general, high-level result for arbitrary sieves, followed with the low-level conditions for the normality of neural networks. Notably, we show that a combination of undersmoothing and regularizing towards a convex target space suffices to overcome a key issue for normality proofs of neural networks: their non-convex sieve space.

Notation. Throughout, we consider random variable Y with support Y , distribution P and corresponding expectation operator E . We also denote the variance operator by V [ f ( Y )] = E ( f ( Y ) -E [ f ( Y )]) 2 for any function f : Y ↦→ R . We denote the sample average, i.e. the expectation under the empirical distribution P n , by E n . Throughout, E will treat estimated parameters as deterministic sequences indexed by n , as is common in the literature. We also consider subvectors of Y , denoted by x ∈ X , ¯ x ∈ ¯ X , with their respective supports X , ¯ X being subspaces of Y . We require various norms: throughout, ‖ x ‖ q will denote the /lscript q norm of a finite dimensional vector x , with ‖ x ‖ = ‖ x ‖ 2 being the Euclidean norm. For a possibly vector-valued function f ( x ), we denote its L q function norm over some subset ˜ X ⊂ X by ‖ f ‖ L q ( ˜ X ) = E [ ‖ f ( x ) ‖ q q | x ∈ ˜ X ] 1 /q . We denote the supremum norm of a vector x with components x i by ‖ x ‖ ∞ = max i | x i | . The supremum norm of f over ˜ X will be denoted by ‖ f ‖ ˜ X = sup x ∈ ˜ X ‖ f ( x ) ‖ ∞ . For ˜ X = X , we may omit the dependence on X by writing ‖ f ‖ ∞ := ‖ f ‖ X . We will often write a ≺ b to denote a = O ( b ), implying that a sufficiently large global constant ∞ &gt; C &gt; 0 exists such that a ≤ Cb , where C does not depend on any varying aspects of the problem, such as any parameters, sample sizes, et cetera. We write a /equivasymptotic b if a ≺ b ≺ a . We will also write a ∨ b = max( a, b ) and a ∧ b = min( a, b ). Throughout, we will write l θ ( λ, Y ) = l ( θ, λ, Y ) and l ( θ, Y ) = l ( θ, λ θ ∗ , Y ) for short, where λ θ ∗ = arg max λ ∈ Λ E l ( θ, λ, Y ). We denote by π n a (not necessarily linear) projection onto the respective sieves, i.e. π n θ ∈ arg inf θ ′ ∈ Θ n ‖ θ ′ -θ ‖ ∞ for any θ ∈ Θ and π n λ ∈ arg inf λ ′ ∈ Λ n ‖ λ ′ -λ ‖ ∞ for any λ ∈ Λ.

## 3.1 Nash vs Minimax

We presented our preferred definition for A-estimators in the introduction, as satisfying a Nash condition of the empirical loss. All results of this paper will apply to this definition. However, the reader may have noticed that the 'simultaneous' Nash condition of the estimator is symmetric in ̂ θ n and ̂ λ n , unlike the 'sequential' mini-max

population objective, which nests a family of inner maximizations:

<!-- formula-not-decoded -->

where the loss l and as a result the solutions λ θ ∗ are indexed by the parameter θ ∈ Θ. The reader may therefore wonder if we could define an A-estimator for θ ∗ in a similar 'sequential' mini-max fashion. That is, we could consider a family of M-estimators ̂ λ θ n approximately maximizing the empirical loss at any value of θ ∈ Θ:

And then look for ̂ θ n ∈ Θ n satisfying:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ¯ η n = o P (1) again accommodates approximate minimization. Fortunately, it turns out that any ̂ θ n satisfying the more compact Nash condition from the introduction always satisfies the mini-max condition presented above, as summarized by the following Lemma:

Lemma 3.1. Any ̂ θ n , satisfying 1.2 and 1.3 for some ̂ λ n , also satisfies 3.3 with some λ θ n for which 3.2, λ ̂ θ n n = λ n and ¯ η n = η n + η n holds.

/negationslash

<!-- formula-not-decoded -->

̂ ̂ ̂ ˜ Proof. Pick any ̂ θ n satisfying 1.2 and 1.3 for some ̂ λ n . Now pick some arbitrary family ̂ λ θ n satisfying 3.2 for all θ = ̂ θ n , and define ̂ λ ̂ θ n n := ̂ λ n . Note that 1.3 directly implies that this ̂ λ θ n also satisfies 3.2 at θ = ̂ θ n . It remains to show that the resulting ̂ θ n and ̂ λ θ n satisfy 3.3:

where the first inequality used ̂ λ ̂ θ n n and the Nash condition 1.2, and the second used the fact that λ θ n was constructed to satisfy 3.2.

̂ This reassures us that it suffices to find one set of values ̂ θ n , ̂ λ n which satisfy the Nash condition from the introduction, rather than a continuum of solutions ̂ λ θ n indexed by θ . The final ̂ θ n will satisfy the mini-max condition regardless, for some (unknown) ̂ λ θ n . For our theory, it was crucial to derive the uniform convergence of ̂ λ θ n , hence we 13

will state the rate results for the more general mini-max definition. For the normality result, it was more convenient to work with the stronger Nash definition.

## 3.2 Adversaries are Neyman-Orthogonal

For many A-estimators, one could construct non-adversarial estimators which capture the same population objective. Whenever the adversarial nuisance parameter λ is a function, this usually requires a non-parametric first-step estimation of an alternative nuisance parameter. However, such an alternative estimator may not have a desirable property that is guaranteed for A-estimators: Neyman-orthogonality of θ ∗ with respect to the nuisance parameter.

This property has a long history in statistics, dating back at least to Neyman [1959]. It was popularized in econometrics by Chernozhukov et al. [2017] as a key setting in which standard machine learning methods can be applied without invalidating causal inference, which sparked follow-up work such as Chernozhukov et al. [2021] seeking to reformulate non-orthogonal problems as orthogonal ones. The notion applies to parameters which are identified by a moment restriction of the form:

<!-- formula-not-decoded -->

where ϕ is known and ν ∗ is an unknown nuisance parameter which has to be estimated in a first step. A popular estimator ̂ θ n in this setting would be Hansen [1982]'s GMM, for example. The moment condition above is called (Neyman-)orthogonal whenever:

<!-- formula-not-decoded -->

Intuitively, this states that the condition identifying θ ∗ is 'locally robust' against perturbations in ν ∗ . This guarantees that the uncertainty introduced by an appropriate first-step estimation of ν ∗ has no first-order effect on the GMM estimator ̂ θ n . Specifically, the asymptotic distribution of ̂ θ n is the same as in the case in which ν ∗ is known. In contrast, when moment restrictions do not satisfy this orthogonality condition, uncertainty about ν ∗ generally amplifies the asymptotic variance of ̂ θ n , see e.g. Chen and Liao [2015], and normality may break down altogether.

Notably, if (and only if) θ ∗ is parametric, we can examine the first order condition

for θ ∗ that is implied by the A-estimation objective 1.1 in this moment restriction framework 4 : let ϕ ( θ, ν ∗ , Y ) = ∇ θ l ( θ, ν ∗ ( θ ) , Y ), where ν ∗ : Θ ↦→ Λ denotes the functional evaluating to ν ∗ ( θ ) = λ θ ∗ . Orthogonality then follows from the continuum of first-order conditions identifying ν ∗ :

<!-- formula-not-decoded -->

since the derivative operators are exchangeable. This implies that as ̂ θ n approaches θ ∗ , an A-estimator ̂ θ n is robust to estimation errors in the adversary ̂ λ θ n relative to λ θ ∗ , meaning they do not reduce the accuracy of ̂ θ n , to a first-order.

Consider the example of Section 2.1, which estimates θ ∗ minimizing the f-Divergence between the model P θ and the data P . As a non-adversarial alternative, we could re-parametrize the problem and estimate ν ∗ := d P via a first-step Kernel density estimator ̂ ν n ( Y ) = ̂ d P n ( Y ), and subsequently approximate the f-Divergence as the average over f ( d P θ ( Y ) ̂ ν n ( Y ) ) . However, the first-order condition for θ ∗ would not satisfy orthogonality, hence a GMM estimator based on this condition may not attain the variance of the analogous GMM estimator using ν ∗ instead. In contrast, the Aestimator of Section 2.1 does attain this variance - due to its orthogonal adversary which we formally establish in Section 4.1. Moreover, this remains true when generalizing to a setting in which θ ∗ contains unknown functions, where no analogous GMM estimator exists that could capture the continuum of first-order conditions in θ ∗ .

## 3.3 Convergence Rate of A-Estimators

We begin with a general theorem characterizing the convergence rates of sieve Aestimators, for arbitrary loss functions and parameter spaces. It can be viewed as a generalization of Shen and Wong [1994]'s M-estimator result. Its proof is provided in Appendix C.1, with the main challenge being that Shen and Wong [1994]'s chaining arguments need to be carefully modified to hold uniformly over Θ. Our theorem adopts a more compact formulation than Shen and Wong [1994] which does not require any norm over Θ , Λ to state our assumptions, although convergence rates are

4 Note however that even when θ ∗ is parametric, we usually cannot estimate it via GMM as ∇ θ l ( θ, ̂ λ θ n , Y ) will not exist if ̂ λ θ n is a typical sieve, such as a neural network. For the same reason, the theory developed in this paper must not rely on any finite-sample first order conditions. Instead, it will use only the approximate Nash condition 1.3, 1.2.

obtained for any (pseudo-)norm d ( θ, θ ∗ ) which is dominated by the objective.

Theorem 3.1 (Convergence Rate of A-Estimators) . Assume that:

- C1: The criterion variance is bounded by a power γ &gt; 0 of its expectation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all θ ∈ Θ , λ ∈ Λ for which the right hand sides are less than some constant.

- C2: For all small ε &gt; 0 , the covering number (Def. 1) is bounded via

<!-- formula-not-decoded -->

for 0 ≤ s &lt; 1 and r ≥ 0 , where r = 0 represents lim r → 0 n s ( ε -r -1) /r = n s log(1 /ε ) . Then the following conclusions hold.

- i) The criterion converges at rate:

<!-- formula-not-decoded -->

where ¯ /epsilon1 n = E [ l ( π n θ ∗ , Y ) -l ( θ ∗ , Y )] and /epsilon1 n = sup θ ∈ Θ n E [ l θ ( λ θ ∗ , Y ) -l θ ( π n λ θ ∗ , Y )] are the sieve approximation errors. 3.10 also holds without 3.6. τ ( γ, s, r, n ) represents:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- ii) Hence, d ( ̂ θ n , θ ∗ ) = o P (1) for any (pseudo-)norm d ( · , · ) under which E [ l ( θ, Y )] compact and continuous. If also d ( θ, θ ∗ ) 1 /q ≺ E [ l ( θ, Y ) -l ( θ ∗ , Y )] for q &gt; 0 , we get:

Remark 3.1 (Discussion of Assumptions) . The theorem extends Shen and Wong [1994]'s convergence rate result for sieve M-estimators to A-estimators. There is a direct mapping between our assumptions and theirs: our C1 combines their assumptions C1 and C2, and our C2 corresponds to their C3. Our proof in Appendix C.1 is structured

in the same way as that of Shen and Wong [1994], although we need to modify their Lemmas to obtain the uniform convergence of the adversary in 3.10, which is crucial to the main result 3.9. The key modifications to our assumptions, which allow us to do so are: C1) that the constant factor implicit in the ' ≺ ' relation of 3.7 must not depend on θ , as implied by the definition of ' ≺ ' at the beginning of this section and C2) that the complexity of the joint sieve space Θ n × Λ n satisfies the entropy bound. Otherwise, the assumptions are conceptually the same and we refer the reader to Shen and Wong [1994] for a more detailed discussion.

Remark 3.2 . Using similar arguments as ours, one may establish the uniform convergence of A-estimators over a third parameter space, generalizing the setting to arbitrary finite sequences of min's and max's over different parameter spaces: e.g. min θ max λ min γ E [ l ( θ, λ, γ, Y )]. This would yield convergence rates towards more general Stackelberg equilibria in so-called empirical games , for which we are currently only aware of a consistency result by Tuyls et al. [2018].

Remark 3.3 . Beyond convergence rates for ̂ θ n and ̂ λ θ n , it is often useful to control the empirical process of arbitrary functions f ( θ, λ, Y ) of the parameters, e.g. to establish conditions for asymptotic normality required by Theorem 3.3. For this purpose, we provide Lemma B.5 in Appendix B.

## 3.4 Semiparametric Rates with Neural Networks

Next, we will apply the general result of the previous section to derive the convergence rates for neural network A-estimators. For generality, we will consider the semiparametric setting in which θ, λ may contain both Euclidean vectors and functions. These lower-level conditions are easy to verify in practice, but are general enough to apply to all estimators considered in Section 2. We will include the proof as it is short and an instructive application of Theorem 3.1. The theorem allows for two types of function classes, both of which can be viewed as generalizations of traditional H¨ older functions with D -dimensional domain, with their own notion of an intrinsic dimension d ∗ ≤ D , which may be smaller than that of the data D . As we will review in Remark 3.5, we observe that neural networks achieve a reduced curse of dimensionality in these settings.

Theorem 3.2 (Semiparametric Rates with Neural Networks) .

Consider the semiparametric setting in which Θ = ¯ B× ¯ A and Λ = B×A , where ¯ B , B are subsets of some Euclidean spaces and ¯ A , A are some function spaces. Let Λ , Θ be compact under ‖ · ‖ ∞ . For all λ, λ ′ ∈ Λ , θ, θ ′ ∈ Θ , assume the following conditions

hold:

- A0: Assume that θ ∗ ∈ Θ ∗ satisfies either
- a) Θ ∗ ⊂ ¯ B × H (¯ p, ¯ X ) on some ¯ X ⊂ [0 , 1] ¯ D with dim M ¯ X = ¯ d ∗ ≤ ¯ D (see Def. 3 and 4)
- b) Θ ∗ ⊂ ¯ B × G (¯ p, ¯ d ∗ , [0 , 1] ¯ D ) (see Def. 6)

and that { λ θ ∗ : θ ∈ Θ } ⊂ Λ ∗ satisfies either

- a) Λ ∗ ⊂ B × H ( p, X ) on some X ⊂ [0 , 1] D with dim M X = d ∗ ≤ D

<!-- formula-not-decoded -->

- A1: l ( θ, λ, Y ) -l ( θ ′ , λ ′ , Y ) ≺ ‖ θ -θ ′ ‖ ¯ X + ‖ λ -λ ′ ‖ X

· A2: V [ l ( θ, Y ) -l ( θ ∗ , Y )] ≺ E [ l ( θ, Y ) -l ( θ ∗ , Y )] ≺ ‖ θ -θ ∗ ‖ 2 ˜ X + P (¯ x /negationslash∈ ˜ X ) ∀ ˜ X ⊂ ¯ X · A3: V [ l θ ( λ θ ∗ , Y ) -l θ ( λ, Y )] ≺ E [ l θ ( λ θ ∗ , Y ) -l θ ( λ, Y )] ≺ ‖ λ -λ θ ∗ ‖ 2 ˜ X + P ( x /negationslash∈ ˜ X ) ∀ ˜ X ⊂ X Pick any two values ¯ r &gt; r ≥ ( d ∗ p ∨ ¯ d ∗ ¯ p ) . Consider the A-estimator 3.2 with η n , ¯ η n = o P ( n -2 / (2+¯ r ) ) where Λ n = B × F σ ( L, W n , w n , κ n ) and Θ n = ¯ B × F σ ( ¯ L, ¯ W n , ¯ w n , ¯ κ n ) implement neural networks (cf. Definition 2) satisfying W n , ¯ W n , w n , ¯ w n /equivasymptotic n r/ ( r +2) and κ n , ¯ κ n /equivasymptotic n c for any large enough choice of L, ¯ L, c &gt; 0 . For A0a) choose σ ( x ) = ReLU( x ) and for A0b) choose σ ( x ) = tanh( x ) . Then:

<!-- formula-not-decoded -->

Hence, d ( ̂ θ n , θ ∗ ) = o P (1) for any (pseudo-)norm d ( · , · ) under which E [ l ( θ, Y )] is compact and continuous. Further, if d ( θ, θ ∗ ) 1 /q ≺ E [ l ( θ, Y ) -l ( θ ∗ , Y )] for q &gt; 0 , we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. We will verify the conditions of Theorem 3.1. A2 and A3 imply C1 (3.6 and 3.7) with γ = 1. Lipschitzness A1 together with Lemma B.1 imply C2 (B.3) with s = t/ ( t + 2) for any t : ¯ r &gt; t &gt; r and r = 0. Therefore Theorem 3.1 applies with n -τ ( γ,s,r,n ) = n 2 / (2+ t ) log n ≺ n 2 / (2+¯ r ) , which dominates η n and ¯ η n by assumption. We are therefore left with bounding /epsilon1 n and ¯ /epsilon1 n . By A3, we can bound /epsilon1 n ≺ sup θ ∈ Θ n ‖ π n λ θ ∗ -λ θ ∗ ‖ 2 ˜ X + P ( x /negationslash∈ ˜ X ) for any ˜ X ⊂ X . In the case of A0a), we set ˜ X = X and use Lemma B.2 to obtain sup θ ∈ Θ n ‖ π n λ θ ∗ -λ θ ∗ ‖ 2 X ≺ ( W n ∧ w n ) -2 p/d ∗ ≺ n -2 pr/d ∗ / (2+ r ) ≺ n 2 / (2+ r ) which yields /epsilon1 n = o ( n 2 / (2+¯ r ) ). For A0b), Lemma B.3 yields

the same bound as Lemma B.2, but only over a subset ˜ X ⊂ X with P ( x /negationslash∈ ˜ X ) ≺ n -k for some arbitrarily large constant k &gt; 0, which only affects the constant c in the bound on κ n . Hence we conclude that /epsilon1 n ≺ n 2 / (2+ r ) + n -k ≺ n 2 / (2+ r ) . Analogous arguments yield the same bound for ¯ /epsilon1 n .

/negationslash

Remark 3.4 (Discussion of Assumptions) . A0 defines the function classes addressed by the Theorem. Both are generalizations of traditional H¨ older classes which arise for d ∗ = D , see Remark 3.5. Condition A1 requires the loss to be Lipschitz in both parameters, which simplifies (but is not necessary for) the verification of C2. Condition A2 (and analogously A3) consists of two parts. First, it states that for a given parameter, the variance of the criterion difference must be bounded by its expectation, a simplified version of Assumption C1 of Theorem 3.1 which happens to be satisfied in all of our examples, but versions of this Theorem with γ = 1 can be derived via the same steps as the proof above. The second part of the condition bounds the expected loss by a squared sup-norm over any subset ˜ X of the function domain X . For the case of A0a), it would have sufficed to state the condition with ˜ X = X only, but for A0b) we require arbitrary subsets ˜ X to apply the approximation result of Lemma B.3. A2 is implied, for example, by E [ l ( θ, Y ) -l ( θ ∗ )] ≺ ‖ h ( θ ) -h ( θ ∗ ) ‖ 2 L q ( X ) for some q and Lipschitz map h : Θ ↦→ Θ. The assumption is significantly weaker than Shen et al. [2019] or Farrell et al. [2018] who impose E [ l ( θ, Y ) -l ( θ ∗ )] /equivasymptotic ‖ θ -θ ∗ ‖ 2 L 2 ( X ) , which would not hold for Examples 2.2 or 2.4. It could be generalized further to allow for arbitrary powers of the sup-norms (and proved in the same way via Theorem 3.1), but the squares arise rather universally via Taylor expansions.

Remark 3.5 . Theorem 3.2 clarifies that neural networks do not necessarily exhibit the curse of dimensionality, as the lower bound on ¯ r does not depend on the dimension D of the data. Instead, what matters is the intrinsic dimension d ∗ of the target function. In the setting A0a), introduced by Nakada and Imaizumi [2020], d ∗ refers to the Minkowski dimension of the manifold X which supports the data. It has been observed that d ∗ /lessmuch D for many high-dimensional types of data: intuitively, d ∗ is low whenever there is strong statistical dependency between the individual dimensions of the data. Examples include the characteristics of physical products, images and natural language. In the setting A0b), introduced by Bauer et al. [2019], d ∗ refers to the order of a generalized hierarchical interaction model. It is common for structural models in e.g. economics or optimal control to suggest that an unknown function is hierarchically composed of some finite number of individual functions which only

depend on d ∗ /lessmuch D inputs at a time. The result underscores that neural networks can adaptively - that is, without the researcher modifying the estimation procedure exploit structures in the target function which allow them to model the relationships more efficiently than what standard convergence results suggest.

## 3.5 Asymptotic Normality of A-Estimators

In applications, it we a often interested in estimating a quantity of the form F ( θ ∗ ), where F : Θ ↦→ R is some known functional. To derive confidence intervals around the plug-in estimate F ( ̂ θ n ), we need its asymptotic distribution. To this end, we present Theorem 3.3, which can roughly be viewed as a generalization of Shen [1997] to A-Estimators. For this section, we make use of the pathwise derivative presented in Definition 7. We require a particular inner product over the space Θ:

<!-- formula-not-decoded -->

As discussed in Definition 7, the notation ∇ θ ∗ → θ implicitly assumes that the corresponding limit exists and is linear in θ . For short, we write λ ′ θ ∗ [ v ] := ∇ θ → v λ θ ∗ , l ′ ( θ, Y )[ v ] := ∇ θ → v l ( θ, Y ) and l ′ ( θ, λ, Y )[ v, w ] := ∇ θ → v l ( θ, λ, Y ) + ∇ λ → w l ( θ, λ, Y ).

Theorem 3.3 (General Normality of A-Estimators) .

Consider the estimators ̂ θ n , ̂ λ n satisfying the Nash conditions 1.2 and 1.3. Fix a sequence e n = o ( n -1 / 2 ) . Assume F is smooth enough and ̂ θ n , ̂ λ n converge fast enough such that a Riesz representer v ∗ ∈ Θ ∗ exists, satisfying:

<!-- formula-not-decoded -->

Where ̂ Θ n and ̂ Λ n ( θ ) are the shrinking neighborhoods defined in Lemma B.5. For v ∈ { v ∗ , -v ∗ } , define the local perturbations ¯ θ n ( θ ) = θ -e n v and ¯ λ θ n ( λ ) = λ + e n λ ′ θ ∗ [ v ] and assume:

CONDITION N1: Stochastic Equicontinuity

<!-- formula-not-decoded -->

CONDITION N2: Population Criterion Difference

<!-- formula-not-decoded -->

CONDITION N3: Approximation Error

<!-- formula-not-decoded -->

If 1.2 and 1.3 are satisfied with η n , η n = O P ( e 2 n ) , then:

Remark 3.6 (Discussion of Assumptions) . In contrast to our convergence rate result, our proof requires the A-estimator to satisfy the (stronger) Nash condition from the introduction. Our conditions N1-3 are analogues of Shen [1997]'s and play the same roles in our proof. N1 combines their assumptions A and D, N2 corresponds to their B, and N3 to their C. Shen [1997]'s high-level discussion of their assumptions therefore applies to ours as well, and we again refer the reader there for additional context. The main difference is that their conditions are formulated to control the remainder of a second order Taylor expansion, whereas we look at the convergence of the first derivative, which results in O P ( e n ) = o P ( n -1 / 2 ) requirements for N1 and N2, rather than the O P ( e 2 n ) = o P ( n -1 ) found in Shen [1997]'s conditions A and B.

<!-- formula-not-decoded -->

Remark 3.7 . Condition N3 is a version of a known condition on approximation error in M-estimation settings (see Condition C4 in Shen et al. [2019] and Condition C in Shen and Wong [1994]). Its verification usually exploits convexity of Θ n , such that π n ¯ θ n ( θ ) = θ + e n π n v ∗ . This holds for series or kernel based estimators, but not neural networks. Shen et al. [2019] therefore leave it as an explicit assumption, concluding that it is unclear how to verify it for neural networks. In Theorem 3.4, we resolve this issue, showing that N3 can be verified for non-convex sieves such as neural networks by adhering to two simple implementation choices: 1) undersmoothing , i.e. choosing a sieve which grows faster than rate-optimal, achieving an approximation error of o ( n -1 ) and 2) regularizing the sieves towards the convex target classes containing θ ∗ , λ ∗ .

## 3.6 Semiparametric Normality with Neural Networks

Next, we present Theorem 3.4, which strengthens the assumptions of our previous neural network convergence rate result (Theorem 3.2) in a way that allows us to derive the asymptotic normality of functionals F ( ̂ θ n ) via Theorem 3.3. A crucial innovation is that we are able to work around the non-convexity issues of deep neural networks discussed in Remark 3.7, to obtain a normality result from low-level conditions, which

only consist of general properties that the loss function must satisfy (A4-A7), and certain implementation choices for the neural networks that must be followed. To the best of our knowledge, the theorem therefore also provides the first low-level conditions for the normality of smooth functionals of deep neural network M-estimators (as the special case where Λ is singleton).

Theorem 3.4 (Semiparametric Normality with Neural Networks) . Let all assumptions of Theorem 3.2 be satisfied with d ∗ p ∨ ¯ d ∗ ¯ p &lt; 1 / 4 , and choose 2 ≥ ¯ r &gt; r &gt; 2 / 3 . Let Θ ∗ , Λ ∗ be convex and θ ∗ , λ θ ∗ lie in their interior. Replace the neural network sieves Θ n , Λ n with the following regularized versions:

<!-- formula-not-decoded -->

for any /epsilon1 &gt; 0 which is small enough to guarantee that Θ n , , Λ n are nonempty. Further, for all θ, v ∈ Θ , λ, w ∈ Λ , assume:

- A4: Lipschitz Derivative: l ′ ( θ, λ, Y )[ v, w ] -l ′ ( θ ′ , λ ′ , Y )[ v, w ] ≺ ‖ θ -θ ′ ‖ ¯ X + ‖ λ -λ ′ ‖ X
- A5: The perturbations are smooth: v ∗ ∈ Θ ∗ , λ ′ θ ∗ [ v ∗ ] ∈ Λ ∗
- A6: The Taylor remainders vanish with the loss:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- A7: For non-Donsker classes, the variance of the derivatives is bounded by the loss:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If ̂ θ n , ̂ λ n satisfy the Nash condition 1.2,1.3 with η n , ˜ η n = o P ( n -1 ) , then:

Remark 3.8 (Discussion of Assumptions) . The Theorem requires that the neural network sieves Θ n , Λ n are implemented to undersmooth (i.e. grow faster than the rateoptimal sieve would) via the condition on r , while being regularized towards the convex target spaces Θ ∗ , Λ ∗ . Note that this does not affect the sieve's approximation power towards these spaces, and there always exists an /epsilon1 &gt; 0 for which Θ n , Λ n are non-empty due to their o ( n -1 ) approximation rates. While in principle just an implementation choice, the current sup-norm regularization is arguably not practical and

<!-- formula-not-decoded -->

future work may be able to clarify whether e.g. an appropriate L2 penalty on the weights suffices. Conditions A4-A7 are general conditions on the loss function which can be satisfied in all our examples. A4 is a simple Lipschitz condition analogous to A1. The smoothness of the Riesz representer (A5) is most easily verified by computing and examining a given v ∗ , λ ′ θ ∗ [ v ∗ ] directly, although the Riesz representation theorem can provide general conditions under which v ∗ lives in the same space as θ ∗ . A6 is a standard condition controlling the Taylor remainder. For a discussion, see e.g. Assumptions 4.5 in Ai and Chen [2003] and Ai and Chen [2007], or Assumption 3.5ii) in Chen and Pouzo [2015]. Whether it holds depends on how non-linear the objective is: e.g. for the quadratic objective of Dikkala et al. [2020], the left-hand side is zero. A7 serves to control the empirical process (N1). It can be easily satisfied either by bounding the variances of the derivatives, or by relying on the Donsker property of the target space (cf. Remark 3.9).

Remark 3.9 . Note that the Donsker property and thus A7 always holds if p &gt; D/ 2, where standard results using bracketing number bounds imply that the H¨ older spaces Θ ∗ , Λ ∗ satisfy the Donsker property. We conjecture that this analogously holds for our lower-dimensional classes A0a) and A0b) whenever d ∗ /p &lt; 2, which would make the verification of A7 unnecessary in general, since we require d ∗ p ∨ ¯ d ∗ ¯ p &lt; 1 / 4. Verifying this conjecture is beyond the scope of this paper however, hence we provide A7 as an explicit assumption for maximum flexibility.

## 4 Application to Examples

## 4.1 Minimum f -Divergence

Applying our general Theorem 3.2 to the estimator of Section 2.1 yields Proposition 4.1, which provides the convergence rate of semiparametric ̂ θ n if Λ and all unknown functions in Θ are approximated by classes of neural networks.

Proposition 4.1. Let θ ∗ ∈ Θ ∗ ⊂ Θ , λ θ ∗ = f ′ ( d P θ d P ) ∈ Λ ∗ ⊂ Λ , where Θ , Λ are compact under ‖ · ‖ ∞ and path-connected, and the target function classes Θ ∗ , Λ ∗ satisfy A0 in Theorem 3.2. Fix some C &lt; ∞ . For any θ ∈ Θ , let 0 &lt; f ′′ ( d P θ d P ( Y ) ) &lt; C wp1 and for any λ ∈ Λ , let 0 &lt; f ′′ ∗ ( λ ( Y )) &lt; C wp1. Let ∥ ∥ ∥ d P θ d P -d P θ ′ d P ∥ ∥ ∥ ∞ ≺ ‖ θ -θ ′ ‖ ∞ . Let Θ n , Λ n be constructed as in Theorem 3.2, with all neural networks growing in width

at some rate n r/ ( r +2) satisfying r ≥ d ∗ p ∨ ¯ d ∗ ¯ p . Then for any ¯ r &gt; r :

<!-- formula-not-decoded -->

Remark 4.1 . In general, the convergence rate of ̂ θ n is faster the slower the growth rate n r/ ( r +2) of the neural network. However, the growth must be fast enough to control the approximation error of the sieves Θ n , Λ n relative to the target function classes Θ ∗ , Λ ∗ . This lower bound depends on the ratio of the smoothness of the target classes p and ¯ p and their intrinsic dimensions d ∗ and ¯ d ∗ , which may be smaller than that of the data Y , in which case f -GANs attain a reduced curse-of-dimensionality relative to traditional nonparametric density estimators.

Remark 4.2 . This convergence rate result stands in contrast to Arora et al. [2017], who argued that Generative Adversarial Networks do not generalize with respect to the metric given by the population objective, only under a weaker 'neural net distance' which they introduce. The convergence rate result above clarifies that the broad class of f -GANs in fact does converge quickly under population divergence.

While a fast convergence rate of the model distribution P ̂ θ n is a key goal in semiand nonparametric estimation, whenever some function F ( ̂ θ n ) of the estimate informs downstream decision-making, we are often interested in obtaining confidence intervals around F ( ̂ θ n ). To this end, we derive the asymptotic normality of the adversarial f -Divergence objective - an entirely novel result at this level of generality, to the best of our knowledge. First, we compute the inner product defined in Section 3.5, which can be expressed concisely:

<!-- formula-not-decoded -->

Where ∇ θ ∗ → θ log d P θ ∗ ( Y ) = ∇ θ ∗ → θ d P θ ∗ ( Y ) d P ( Y ) is a pathwise derivative of the RadonNikodym derivative. Conditions under which the normality result of Section 3.5 applies are presented in Proposition 4.2.

Proposition 4.2. Consider a functional F ( θ ) for which a Riesz representer v ∗ exists satisfying 3.11 with 〈· , ·〉 defined above. Let all assumptions of Theorem 4.1 be satisfied for d ∗ /p ∨ ¯ d ∗ / ¯ p &lt; 1 / 4 and assume that × ∗ is Donsker. Let Θ ∗ , Λ ∗ be convex, let θ ∗ , λ θ ∗ lie in their interior, and let them contain v ∗ , λ ′ ,θ ∗ [ v ∗ ] . Assume the Lipschitz condition ‖∇ θ → v d P θ d P -∇ θ ′ → v d P θ ′ d P ‖ ∞ ≺ ‖ θ -θ ′ ‖ ∞ and let f ′′ be Lipschitz. Pick 2 ≥ ¯ r &gt; r &gt; 2 / 3 and regularize Θ n , Λ n as in Theorem 3.4. Finally, for any ˜ θ, ˜ θ ′ on a path between θ ∗

and θ , assume that:

Then:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 4.3 . In applications, the key difficulty lies in verifying that the third derivative above is bounded by the loss. This condition serves to control the higher order term of the Taylor expansion. Such assumptions are common in the semiparametric literature, e.g. Ai and Chen [2003]'s Assumptions 4.5 and 4.6 play the same role. It is easiest to verify in the parametric setting, where

<!-- formula-not-decoded -->

Note that 〈· , ·〉 and hence the asymptotics of ̂ θ n are independent of f , so the f -divergences are asymptotically equivalent. An example for a smooth functional F ( θ ) that is of particular interest in the semiparametric setting θ = ( β, α ) is F ( θ ) = β ′ ζ , which 'picks out' a linear combination of the parametric components. This allows us to derive the asymptotic normality of the vector √ n ( ̂ β -β ∗ ) in the following Corollary, which makes use of the orthogonal scores assumption that is standard in the semiparametric literature.

Corollary 4.2.1. In addition to the assumptions of Proposition 4.2, assume the orthogonal scores condition holds:

<!-- formula-not-decoded -->

Then the parametric component ̂ β n attains the Cram´ er-Rao bound: √ n ( ̂ β n -β ∗ ) d →N ( 0 , I -1 ) , where I = E [ ∇ β ∗ log d P β ∗ ,α ∗ ( Y ) · ∇ β ′ ∗ log d P β ∗ ,α ∗ ( Y ) ]

Proof. We simply choose v ∗ = ( I -1 ζ, 0 ), such that 〈 θ -θ ∗ , v ∗ 〉 = ( β -β ∗ ) ′ ζ = F ( θ ) -F ( θ ∗ ). Since 〈 v ∗ , v ∗ 〉 = ζ ′ I -1 ζ , Proposition 4.2 yields √ n ( ̂ β n -β ∗ ) ′ ζ d →N (0 , ζ ′ I -1 ζ ). The result then follows via the Cram´ er-Wold device.

The f -GAN objective therefore attains the efficient asymptotics of maximum likelihood, but does not require explicit knowledge of the model density P θ .

## 4.2 Generalized Empirical Likelihood

For the class of Generalized Empirical Likelihood estimators introduced in Section 2.2, the √ n -normality and asymptotic efficiency of ̂ θ n is long established in the parametric case (Imbens et al. [1998], Imbens [2002], Newey and Smith [2004]). However, our theoretical framework still allows us to extend the known results to the semiparametric case where θ may contain unknown functions, which are approximated by a class of neural networks Θ n which may grow with n . In this case, we can characterize the convergence rate of ̂ θ n to the identified set Θ ∗ = { θ ∈ Θ : E [ m ( Y, θ )] = 0 } , which is unlikely to be singleton given that an infinite-dimensional parameter is hardly pinned down by a finite number of unconditional moment restrictions. We obtain the following result:

Proposition 4.3. Let D f = χ 2 and consider the A-estimator ̂ θ n , ̂ λ n satisfying 1.2,1.3 with l ( θ, λ, Y ) = -f ∗ ( λ ′ m ( Y, θ )) . Let Θ ∗ , Θ n be as in Theorem 3.2 and Λ ∗ = Λ n = R dim( m ) , with ¯ r &gt; d ∗ p . Assume that m ( Y, θ ) -m ( Y, θ ′ ) ≺ ‖ θ -θ ′ ‖ ∞ and | m ( Y, θ ) | &lt; ∞ . Then:

<!-- formula-not-decoded -->

Proof. We verify the conditions of Theorem 3.2. Assumption A0 holds by assumption, and A1 follows from the Lipschitzness of m ( Y, · ) and that of f ∗ ( t ) = t + t 2 / 4. To verify Assumption 2, note that l ( θ ∗ , Y ) = 0 and boundedness of m ( Y, θ ) imply:

<!-- formula-not-decoded -->

For the second part of condition A2, simply verify that E [ l ( θ, Y ) -l ( θ ∗ , Y )] ≺ ‖ λ θ ∗ -λ θ ∗ ∗ ‖ 2 2 ≺ ‖ θ, θ ∗ ‖ 2 ˜ X + P (¯ x /negationslash∈ ˜ X ), which follows by applying the Lipschitzness of m in θ and the tower-property of E to λ θ ∗ = -2 E [ m ( Y, θ ) m ( Y, θ ) ′ ] -1 E [ m ( Y, θ )], akin to the proof of 4.1. Assumption A3 can be verified for the Euclidean λ via a Taylor expansion, yielding: V [ l ( θ, λ, Y ) -l ( θ, λ θ ∗ , Y )] /equivasymptotic ‖ λ -λ θ ∗ ‖ 2 2 /equivasymptotic E [ l ( θ, λ, Y ) -l ( θ, λ θ ∗ , Y )].

## 4.3 Off-Policy Reinforcement Learning

Next, we will use our theory to the extend the known results about SBEED, the off-policy RL algorithm of Dai et al. [2018] introduced in Section 2.3. Theorem 3.2 makes it easy to obtain the convergence rates of the corresponding A-estimator:

Proposition 4.4. Consider the A-estimator ̂ θ n , ̂ λ n satisfying 1.2,1.3 with l ( θ, λ, Y ) as in 2.4. Assume the observations are iid for simplicity, and that P ∗ = P θ ∗ and V ∗ = V θ ∗ , where θ ∗ ∈ Θ ∗ , λ θ ∗ ∈ Λ ∗ satisfy A0 in Theorem 3.2 with X = ¯ X = S × A . Let Θ ∗ ⊂ Θ , Λ ∗ ⊂ Λ , with Θ , Λ compact under ‖ · ‖ ∞ and path-connected. Let R ( · , · ) , V θ ( · ) , P θ ( ·|· ) be continuous. Let the parametrizations P θ , V θ satisfy the Lipschitz conditions ‖ log P θ -log P θ ′ ‖ ∞ ≺ ‖ θ -θ ′ ‖ ∞ and ‖ V θ -V θ ′ ‖ ∞ ≺ ‖ θ -θ ′ ‖ ∞ . Let the neural network classes Θ n , Λ n be constructed as in Theorem 3.2, for any r ≥ d ∗ p ∨ ¯ d ∗ ¯ p . Then for any ¯ r &gt; r :

<!-- formula-not-decoded -->

Remark 4.4 . In contrast to the original work, our result also applies in the case where A and S are continuous, and we characterize the optimal rate of growth for the neural network function approximators, which optimally trade off bias and variance. While following almost trivially from the general Theorem 3.2, our result yields significantly faster convergence rates than the o P ( √ n ) rates obtained by Dai et al. [2018], and our rates further exhibit the reduced curse of dimensionality of neural networks.

Remark 4.5 . We noticed that SBEED can be viewed as a special case of some of the econometric conditional moment estimators treated in Example 2.4, such as Dikkala et al. [2020]. We therefore refer the reader to Section 4.4 for an application of our asymptotic normality result. Interestingly, neither literature seems to be aware of this connection. Dai et al. [2018] cite convex conjugation and the interchangeability principle as the inspiration for their objective, whereas the adversarial conditional moment estimators in econometrics were inspired by Hansen [1982]'s Generalized Method of Moments.

## 4.4 A-Estimators for Conditional Moment Restrictions

We will now apply our theory to examine the asymptotic behavior of the conditional moment estimator of Dikkala et al. [2020], introduced in Section 2.4. We can apply Theorem 3.2 to obtain the rate at which θ n converges:

̂ Proposition 4.5. Let Θ n , Θ ∗ , Λ n , Λ ∗ be as in Theorem 3.2. Let m ( X,θ ) be ‖ · ‖ ∞ -Lipschitz in θ . Let the support of Y be bounded. Then, for any ¯ r &gt; d ∗ p ∨ ¯ d ∗ ¯ p , we get:

For the instrumental variable regression setting studied by Dikkala et al. [2020], where

<!-- formula-not-decoded -->

m ( X,θ ) = y -θ ( x ) , this implies:

Proof. Condition A0 is satisfied by assumption, and A1 follows from Lipschitzness of m ( X, · ) and boundedness. Assumptions A2 and A3 can be verified by using boundedness to establish

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark 4.6 . Note that just like in the previous Example 2.2, this result does not require the parameter θ ∗ to be identified by the restriction 2.5. If that is the case however, the above rates can be translated into similar rates in any norm ‖ ̂ θ n -θ ∗ ‖ which is dominated by the objective, usually by construction. See Ai and Chen [2003] for an example of such a norm in the semi-parameteric setting.

Remark 4.7 . In contrast to Dikkala et al. [2020], our convergence rate result allows for general m and possibly vector-valued, semiparametric Θ in which unknown functions are approximated by neural networks. Our rates are also exhibit the reduced curse of dimensionality of neural networks.

Next, we use Theorem 3.4 to derive the asymptotic variance of the estimator, showing that the estimator is in inefficient in general. For this purpose, it suffices to only consider the simpler parametric setting.

Proposition 4.6. Consider the parametric case where Θ n = Θ ∗ = Θ is Euclidean. In addition to the assumptions of Proposition 4.5, assume that the identification condition 2.5 holds. Let d ( X,θ ) := ∇ θ m ( X,θ ) be bounded and satisfy the Lipschitz condition | d ( X,θ ) -d ( X,θ ′ ) | ≺ ‖ θ -θ ′ ‖ ∞ . Assume that E [ l ( θ, Y )] is three times differentiable in θ . For all θ ∈ Θ , let λ θ ∗ := 2 E [ m ( X,θ ) | Z ] ∈ Λ ∗ for a Λ ∗ satisfying A0 with d ∗ p &lt; 1 4 , let θ ∗ , λ θ ∗ lie in the respective interiors of Θ , Λ ∗ , and let λ ′ θ ∗ [ v ∗ ]( · ) := 2 v ′ ∗ E [ d ( X,θ ) | Z = · ] ∈ Λ ∗ for any v ∗ ∈ Θ . Let Λ n be regularized as in Theorem 3.4. Then:

̂ where V = E [ E [ ∇ θ ∗ m ( X,θ ∗ ) ′ | Z ] E [ m ( X,θ ∗ ) m ( X,θ ∗ ) ′ | Z ] E [ ∇ θ ∗ m ( X,θ ∗ ) | Z ]] -1 .

<!-- formula-not-decoded -->

Chamberlain [1987] derived the efficiency bound for the parametric conditional mo-

ment setting, corresponding to the smallest (in a p.s.d. sense) √ n -asymptotic variance for any unbiased estimator. It is given by the covariance matrix:

<!-- formula-not-decoded -->

/negationslash

Note that V = V ∗ in general, implying that ̂ θ n is an inefficient estimator. By extension, this also applies to the Reinforcement Learning algorithm of Example 2.3. Comparing the GMM objective of Example 2.2 - which is known to be efficient in the unconditional moment setting - to the population objective of the present example, this may be unsurprising: in contrast to GMM, the population objective of Dikkala et al. [2020] corresponds to a regular /lscript 2 norm, without the inverse covariance weighting which is crucial for asymptotic efficiency in the unconditional case. Generalizing GEL to the conditional moment setting by replacing the constant adversary with a neural network Λ n , Metzger [2022] therefore proposes the A-estimator given by:

<!-- formula-not-decoded -->

which nests a simplified variant of Bennett and Kallus [2020] for D f = χ 2 , and for D f = D KL can be viewed as alternative to the Kernel approach of Kitamura et al. [2004]. Metzger [2022] provides a similar information theoretic foundation as the GEL estimator and - building on the theory developed in the present paper - derives the convergence rates and asymptotic efficiency of this estimator, where Θ n may contain unknown functions which are modeled as neural networks.

## 4.5 Estimating Riesz Representers

Finally, we show that Theorem 3.2 can be used to quickly derive the convergence rates of Chernozhukov et al. [2020]'s adversarial estimator for Riesz representers, which we introduced in Section 2.5.

Proposition 4.7. Let Θ n , Θ ∗ , Λ n , Λ ∗ be as in Theorem 3.2. Let m ( Y, λ ) = m ( Y, λ ( x )) be Lipschitz in λ ( x ) . Let the support of Y be bounded. Then, for any ¯ r &gt; d ∗ p ∨ ¯ d ∗ ¯ p :

<!-- formula-not-decoded -->

This result clarifies that the Riesz representer of Chernozhukov et al. [2020] can similarly benefit from the adaptivity properties of neural networks, which yield faster

rates for our target classes if d ∗ &lt; D . In combination with their Lemma 17, this implies that compared to other non-parametric sieves, neural networks guarantee the asymptotic normality of the orthogonalized estimator ˜ φ n under weaker conditions on smoothness and D . Since the normality of ˜ φ n -φ ∗ is of primary interest and already follows from Chernozhukov et al. [2020]'s Lemma 17 given our convergence rates, we refrain from deriving it for arbitrary functionals ̂ φ n ( g ) = E [ ̂ θ n ( x ) g ( x )], although it would be possible to use Theorem 3.4 to derive √ n ( ̂ φ n ( g ) -φ ( g )) d → N (0 , V g ) for some V g for example.

## 5 Conclusion

We characterize the general class of adversarial estimators ('A-estimators') , subsuming many estimators independently proposed in the fields of econometrics and machine learning. Our unified framework suggests interesting commonalities between A-estimators: their adversary is always Neyman-orthogonal with respect to the main model, guaranteeing that its estimation errors have no first-order asymptotic impact on the estimated model. Most objectives have versions which asymptotically minimize an f -divergence criterion and are asymptotically efficient. Typically, Aestimators adaptively learn how to optimally emphasize the restrictions implied researcher's estimation assumptions, performing particularly well when this set is large. This makes them a promising framework for incorporating machine learning methods into causal inference, where even simple target parameters often satisfy a continuum of restrictions. We characterize the convergence rates of A-estimators, as well as the asymptotic normality of smooth functionals of their parameters. We also provide low-level analogues of these results for semi-parametric models, in which unknown functions are approximated by deep neural networks. Our convergence and normality results also extend the theory of neural network M-estimators, as a special case: building on recent results in approximation theory, our neural network converge rates exhibit a reduced curse of dimensionality for more general losses than previously examined, which hold uniformly over a second parameter space. Our normality result overcomes a problem previously posed by the non-convexity of neural network sieves, showing that a particular regularization, combined with under-smoothing, can be used to satisfy a strong, high-level approximation error condition which the literature left hitherto unverified.

## References

- Chunrong Ai and Xiaohong Chen. Efficient estimation of models with conditional moment restrictions containing unknown functions. Econometrica , 71 (6):1795-1843, 2003. doi: https://doi.org/10.1111/1468-0262.00470. URL https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00470 .
- Chunrong Ai and Xiaohong Chen. Estimation of possibly misspecified semiparametric conditional moment restriction models with different conditioning variables. Journal of Econometrics , 141(1):5-43, 2007. URL https://EconPapers.repec.org/RePEc:eee:econom:v:141:y:2007:i:1:p:5-43 .
- Joshua D. Angrist and J¨ orn-Steffen Pischke. The credibility revolution in empirical economics: How better research design is taking the con out of econometrics. Journal of Economic Perspectives , 24(2):3-30, June 2010. doi: 10.1257/jep.24.2.3. URL https://www.aeaweb.org/articles?id=10.1257/jep.24.2.3 .
- Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, and Yi Zhang. Generalization and equilibrium in generative adversarial nets (GANs). In Doina Precup and Yee Whye Teh, editors, Proceedings of the 34th International Conference on Machine Learning , volume 70 of Proceedings of Machine Learning Research , pages 224-232. PMLR, 06-11 Aug 2017. URL https://proceedings.mlr.press/v70/arora17a.html .
- Susan Athey and Stefan Wager. Policy learning with observational data. Econometrica , 89(1):133-161, 2021. doi: https://doi.org/10.3982/ECTA15732. URL https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA15732 .
- Susan Athey, Guido W Imbens, Jonas Metzger, and Evan M Munro. Using wasserstein generative adversarial networks for the design of monte carlo simulations. Technical report, National Bureau of Economic Research, 2019.
- Benedikt Bauer, Michael Kohler, et al. On deep learning as a remedy for the curse of dimensionality in nonparametric regression. Annals of Statistics , 47(4):2261-2285, 2019.
- Denis Belomestny, Eric Moulines, Alexey Naumov, Nikita Puchkin, and Sergey Sam-

sonov. Rates of convergence for density estimation with gans. arXiv preprint arXiv:2102.00199 , 2021.

- Andrew Bennett and Nathan Kallus. The variational method of moments. CoRR , abs/2012.09422, 2020. URL https://arxiv.org/abs/2012.09422 .
- Andrew Bennett, Nathan Kallus, and Tobias Schnabel. Deep generalized method of moments for instrumental variable analysis. In NeurIPS , 2019a.
- Andrew Bennett, Nathan Kallus, and Tobias Schnabel. Deep generalized method of moments for instrumental variable analysis. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d ' Alch´ e-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019b. URL https://proceedings.neurips.cc/paper/2019/file/15d185eaa7c954e77f5343d941e25fbd-Pa
- Gary Chamberlain. Asymptotic efficiency in estimation with conditional moment restrictions. Journal of Econometrics , 34(3):305-334, 1987.
- Minshuo Chen, Wenjing Liao, Hongyuan Zha, and Tuo Zhao. Statistical guarantees of generative adversarial networks for distribution estimation, 2020.
- Xiaohong Chen and Zhipeng Liao. Sieve semiparametric two-step gmm under weak dependence. Journal of Econometrics , 189(1):163-186, 2015. ISSN 0304-4076. doi: https://doi.org/10.1016/j.jeconom.2015.07.001. URL https://www.sciencedirect.com/science/article/pii/S0304407615002031 .
- Xiaohong Chen and Demian Pouzo. Sieve wald and qlr inferences on semi/nonparametric conditional moment models. Econometrica , 2015.
- Xiaohong Chen and Yin Jia Qiu. Methods for nonparametric and semiparametric regressions with endogeneity: a gentle guide. Cowles Foundation Discussion Papers 2032, Cowles Foundation for Research in Economics, Yale University, 2016. URL https://EconPapers.repec.org/RePEc:cwl:cwldpp:2032 .
- Xiaohong Chen, Oliver Linton, and Ingrid Van Keilegom. Estimation of semiparametric models when the criterion function is not smooth. Econometrica , 71(5): 1591-1608, 2003.

- Xiaohong Chen, Demian Pouzo, and James L. Powell. Penalized sieve gel for weighted average derivatives of nonparametric quantile iv regressions. Journal of Econometrics , 2019.
- Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, and Whitney Newey. Double/debiased/neyman machine learning of treatment effects. The American Economic Review , 107:261-265, 2017.
- Victor Chernozhukov, Whitney Newey, Rahul Singh, and Vasilis Syrgkanis. Adversarial estimation of riesz representers, 2020.
- Victor Chernozhukov, Whitney Newey, Victor Quintas-Martinez, and Vasilis Syrgkanis. Automatic debiased machine learning via neural nets for generalized linear regression. 2021.
- Stephen Cosslett. Maximum likelihood estimator for choice-based samples. Econometrica , 49:1289-1316, 02 1981. doi: 10.2307/1912755.
- Andrew Cotter, Heinrich Jiang, and Karthik Sridharan. Two-player games for efficient non-convex constrained optimization. In Aur´ elien Garivier and Satyen Kale, editors, Proceedings of the 30th International Conference on Algorithmic Learning Theory , volume 98 of Proceedings of Machine Learning Research , pages 300-332. PMLR, 22-24 Mar 2019. URL https://proceedings.mlr.press/v98/cotter19a.html .
- Bo Dai, Albert Shaw, Lihong Li, Lin Xiao, Niao He, Jianshu Chen, and Le Song. Sbeed: Convergent reinforcement learning with nonlinear function approximation. CoRR , abs/1712.10285, 2018. URL http://arxiv.org/abs/1712.10285 .
- Nishanth Dikkala, Greg Lewis, Lester Mackey, and Vasilis Syrgkanis. Minimax estimation of conditional moment models. arXiv preprint arXiv:2006.07201 , 2020.
- Max H Farrell, Tengyuan Liang, and Sanjog Misra. Deep neural networks for estimation and inference. arXiv preprint arXiv:1809.09953 , 2018.
- Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems , pages 2672-2680, 2014.

Ulf Grenander. Abstract inference. Technical report, 1981.

- Lars Peter Hansen. Large sample properties of generalized method of moments estimators. Econometrica , 50(4):1029-1054, 1982. ISSN 00129682, 14680262. URL http://www.jstor.org/stable/1912775 .
- Lars Peter Hansen, John Heaton, and Amir Yaron. Finite-sample properties of some alternative gmm estimators. Journal of Business &amp; Economic Statistics , 14(3): 262-280, 1996. ISSN 07350015. URL http://www.jstor.org/stable/1392442 .
- Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. NIPS , abs/1606.03476, 2016. URL http://arxiv.org/abs/1606.03476 .
- Guido W. Imbens. Generalized method of moments and empirical likelihood. Journal of Business &amp; Economic Statistics , 20(4):493-506, 2002. ISSN 07350015. URL http://www.jstor.org/stable/1392419 .
- Guido W. Imbens, Richard H. Spady, and Phillip Johnson. Information theoretic approaches to inference in moment condition models. Econometrica , 66(2):333-357, 1998. ISSN 00129682, 14680262. URL http://www.jstor.org/stable/2998561 .
- Abdul Jabbar, Xi Li, and Bourahla Omar. A survey on generative adversarial networks: Variants, applications, and training. ACM Computing Surveys (CSUR) , 54: 1 - 49, 2022.
- Tetsuya Kaji, Elena Manresa, and Guillaume Pouliot. An adversarial approach to structural estimation, 2020.
- Yuichi Kitamura, Gautam Tripathi, and Hyungtaik Ahn. Empirical likelihood-based inference in conditional moment restriction models. Econometrica , 72(6):16671714, 2004.
- Greg Lewis and Vasilis Syrgkanis. Adversarial generalized method of moments, 2018.
- Tengyuan Liang. How well generative adversarial networks learn distributions. Journal of Machine Learning Research , 22(228):1-41, 2021.
- Robert E. Lucas. Econometric policy evaluation: A critique. CarnegieRochester Conference Series on Public Policy , 1:19-46, 1976. ISSN

0167-2231. doi: https://doi.org/10.1016/S0167-2231(76)80003-6. URL https://www.sciencedirect.com/science/article/pii/S0167223176800036 .

- Xudong Mao, Qing Li, Haoran Xie, Raymond Y. K. Lau, Zhen Wang, and Stephen Paul Smolley. Least squares generative adversarial networks, 2017.
- Jonas Metzger. Adversarial conditional moment estimation, 2022.
- Ryumei Nakada and Masaaki Imaizumi. Adaptive approximation and generalization of deep neural network with intrinsic dimensionality. Journal of Machine Learning Research , 21(174):1-38, 2020. URL http://jmlr.org/papers/v21/20-002.html .
- Whitney K. Newey and Richard J. Smith. Higher order properties of gmm and generalized empirical likelihood estimators. Econometrica , 72(1):219-255, 2004. ISSN 00129682, 14680262. URL http://www.jstor.org/stable/3598854 .
- Jerzy Neyman. Optimal asymptotic tests of composite statistical hypotheses. Probability and Statistics: The Harald Cramer Volume , 1959.
- Sebastian Nowozin, Botond Cseke, and Ryota Tomioka. f-gan: Training generative neural samplers using variational divergence minimization, 2016.
- Art Owen. Empirical Likelihood Ratio Confidence Regions. The Annals of Statistics , 18(1):90 -120, 1990. doi: 10.1214/aos/1176347494. URL https://doi.org/10.1214/aos/1176347494 .
- Art B. Owen. Empirical likelihood ratio confidence intervals for a single functional. Biometrika , 75(2):237-249, 06 1988. ISSN 0006-3444. doi: 10.1093/biomet/75.2. 237. URL https://doi.org/10.1093/biomet/75.2.237 .
- Jin Qin and Jerry Lawless. Empirical likelihood and general estimating equations. The Annals of Statistics , 22(1):300-325, 1994. ISSN 00905364. URL http://www.jstor.org/stable/2242455 .
- Xiaotong Shen. On methods of sieves and penalization. The Annals of Statistics , 25(6):2555-2591, 1997. ISSN 00905364. URL http://www.jstor.org/stable/2959045 .

- Xiaotong Shen and Wing Hung Wong. Convergence rate of sieve estimates. Ann. Statist. , 22(2):580-615, 06 1994. doi: 10.1214/aos/1176325486. URL https://doi.org/10.1214/aos/1176325486 .
- Xiaoxi Shen, Chang Jiang, Lyudmila Sakhanenko, and Qing Lu. Asymptotic properties of neural network sieve estimators. arXiv preprint arXiv:1906.00875 , 2019.
- Shashank Singh, Ananya Uppal, Boyue Li, Chun-Liang Li, Manzil Zaheer, and Barnab´ as P´ oczos. Nonparametric density estimation under adversarial losses. arXiv preprint arXiv:1805.08836 , 2018.
- Chenyang Tao, Liqun Chen, Ricardo Henao, Jianfeng Feng, and Lawrence Carin Duke. Chi-square generative adversarial network. In Jennifer Dy and Andreas Krause, editors, Proceedings of the 35th International Conference on Machine Learning , volume 80 of Proceedings of Machine Learning Research , pages 4887-4896. PMLR, 10-15 Jul 2018. URL https://proceedings.mlr.press/v80/tao18b.html .
- K Tuyls, J Perolat, M Lanctot, JZ Leibo, and T Graepel. A generalised method for empirical game theoretic analysis. In AAMAS'18: Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems , pages 77-85. ACM, 2018.
- Dmitry Yarotsky. Error bounds for approximations with deep relu networks. Neural Networks , 94:103-114, 2017.
- Ruohan Zhan, Vitor Hadad, David A. Hirshberg, and Susan Athey. Off-policy evaluation via adaptive weighting with data from contextual bandits. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining , KDD '21, page 2125-2135, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450383325. doi: 10.1145/3447548.3467456. URL https://doi.org/10.1145/3447548.3467456 .

## A Definitions

Definition 1 (Covering Number) .

For some norm ‖ · ‖ over some metric space Λ, the covering number N ( δ, Λ , ‖ · ‖ ) is

defined as the cardinality of the smallest set C ⊂ Λ such that sup λ ∈ Λ inf c ∈ C ‖ λ -c ‖ ≤ δ . The quantity log N ( δ, Λ , ‖ · ‖ ) is also called metric entropy.

## Definition 2 (Deep Neural Networks) .

We define the class of deep σ networks f ∈ F σ ( L, W, w, κ, B ) as parametrized functions of the form:

<!-- formula-not-decoded -->

where the A ( l ) 's are weight matrices and b ( l ) 's are intercept vectors with real-valued elements, and σ : R ↦→ R is applied element-wise. For example, the choice σ ( x ) = ReLU( x ) = max { 0 , x } (rectified linear unit) gives rise to the class of deep ReLU networks, and σ ( x ) = tanh ( x ) gives rise to the class of tanh networks. We say the network is L layers deep and call the upper bound sup l dim( b ( l ) ) ≤ w its width. Further, we assume that

<!-- formula-not-decoded -->

## Definition 3 (Minkowski Dimension) .

i.e. all elements in the A ( l ) 's and b ( l ) 's are bounded in absolute value by κ , and there are at most W non-zero parameters in total. Finally, we assume ‖ f ‖ ∞ ≤ B &lt; ∞ for all f . If the particular value B is an arbitrary large enough constant, we may suppress the notation and write F σ ( L, W, w, κ, B ) = F σ ( L, W, w, κ ).

The (upper) Minkowski dimension of a set X ⊂ [0 , 1] D is defined as

<!-- formula-not-decoded -->

where N ( ε, X , ‖ · ‖ ∞ ) is given by Definition 1. As shown in Nakada and Imaizumi [2020], this definition generalizes many other notions of intrinsic dimension, such as the manifold dimension.

## Definition 4 (H¨ older Space) .

For a function f : R D → R , ∂ d f ( x ) is a partial derivative with respect to a d -th component, and ∂ α f := ∂ α 1 1 · · · ∂ α D D f using multi-index α = ( α 1 , . . . , α D ) . For z ∈ R /floorleft z /floorright denotes the largest integer that is less than z . Let p &gt; 0 be a degree of smoothness. For f : [0 , 1] D → R , the H¨ oder norm is defined as

/negationslash

<!-- formula-not-decoded -->

Then, the H¨ older space on [0 , 1] D is defined as

<!-- formula-not-decoded -->

Also, H ( p, [0 , 1] D , M ) = { f ∈ H ( p, [0 , 1] D ) | ‖ f ‖ H ( p, [0 , 1] D ) ≤ M } denotes the M -radius closed ball in H ( p, [0 , 1] D ) . Definition 5 ((p, C)-smoothness) .

Let p = q + s for some q ∈ N 0 and 0 &lt; s ≤ 1. A function m : R d → R is called ( p, C ) -smooth, if for every α = ( α 1 , . . . , α d ) ∈ N d 0 with ∑ d j =1 α j = q the partial derivative ∂ q m ∂x α 1 ...∂x α d d exists and satisfies

<!-- formula-not-decoded -->

for all x, z ∈ R d .

Definition 6 (Generalized Hierarchical Interaction Models) .

Let C ∈ R ≥ 0 , D ∈ N , d ∗ ∈ { 1 , . . . , D } , m : R D → R and p = q + s for some q ∈ N 0 and 0 &lt; s ≤ 1.

<!-- formula-not-decoded -->

- a) We say that m satisfies a generalized hierarchical interaction model of order d ∗ and level 0 with bound C , if there exist a 1 , . . . , a d ∗ ∈ R D and some f : R d ∗ → R such that

and where f is Lipschitz continuous with constant C and all of its partial derivatives of order less than or equal to q are bounded in absolute value by by C .

- b) We say that m satisfies a generalized hierarchical interaction model of order d ∗ and level l +1 with bound C if there exist K ∈ N , g k : R d ∗ → R ( k = 1 , . . . , K ) and f 1 ,k , . . . , f d ∗ ,k : R D → R ( k = 1 , . . . , K ) such that f 1 ,k , . . . , f d ∗ ,k ( k = 1 , . . . , K ) satisfy a generalized hierarchical interaction model of order d ∗ and level l and

<!-- formula-not-decoded -->

where g k are Lipschitz continuous with constant C and all of their partial derivatives of order less than or equal to q are bounded by some constant C .

- c) We say that the generalized hierarchical interaction model defined above is ( p, C )-smooth, if all functions occurring in its definition are ( p, C )-smooth, cf. Definition 5.
- d) We define G ( p, d ∗ , C, [0 , 1] D ) as the class of all functions m : [0 , 1] D → R satisfying a ( p, C )-smooth generalized hierarchical interaction model of order d ∗ and level l with bound C , where l ≤ C . Since the particular value of C is not important as long as C &lt; ∞ , we also write G ( p, d ∗ , [0 , 1] D ).

Definition 7 (Pathwise Derivatives) .

For some θ ∈ Θ, λ ∈ Λ and some functional l : Θ × Λ ↦→ R d , we define the first pathwise derivative in the direction θ ′ ∈ Θ as

<!-- formula-not-decoded -->

for some real number τ ∈ R . Throughout this paper, the usage of ∇ θ → θ ′ implicitly assumes that the derivative and limit on the RHS exists and is linear in θ ′ .

## B Supporting Lemmas

Lemma B.1 (Covering Number of Neural Networks) .

Consider the class of deep neural networks f ∈ F σ ( L, W, w, κ ) (Definition 2), with activation σ satisfying σ : | σ ( x ) | ≤ x, | σ ( x ) -σ ( x ′ ) | ≤ | x -x ′ | ∀ x, x ′ ∈ R (e.g. ReLU, tanh) and consider the norm ‖ f ‖ ∞ = sup x ∈X | f ( x ) | for some X ⊂ [0 , 1] D where D ≤ w . Its δ -covering number (Definition 1) can be bounded by:

Proof. This is Lemma 7 in Chen et al. [2020]. While they only state the Lemma for the case of ReLU networks σ ( x ) = max(0 , x ), their proof works for any activation σ satisfying | σ ( x ) | ≤ x and | σ ( x ) -σ ( x ′ ) | ≤ | x -x ′ | for all x, x ′ ∈ R . We substituted the bound B = 1 and renamed some variables.

<!-- formula-not-decoded -->

Lemma B.2 (Approximation by Deep ReLU Networks on Low Dimensional Data) . Consider the H¨ older space H ≡ H ( p, [0 , 1] D ) (Definition 4) and some support X ⊂ [0 , 1] D with Minkowski dimension (Definition 3) bounded by dim M X ≤ d ∗ ≤ D . For any small enough /epsilon1 &gt; 0 , the class of deep ReLU networks F ≡ F ReLU ( L, W ( /epsilon1 ) , w ( /epsilon1 ) , κ ( /epsilon1 )) (Definition 2) satisfies:

<!-- formula-not-decoded -->

as long as W ( /epsilon1 ) ≥ c 1 /epsilon1 -d ∗ /p , w ( /epsilon1 ) ≥ c 2 /epsilon1 -d ∗ /p , κ ( /epsilon1 ) ≥ c 3 /epsilon1 -c 4 for any large enough choice of L , c 1 , c 2 , c 3 , c 4 &gt; 0 .

Proof. The case d ∗ &lt; D is covered by Theorem 5 in Nakada and Imaizumi [2020]. While they do not state a bound on the width w ( /epsilon1 ), it is easy to see that any network described by Definition 2 with at most W ( /epsilon1 ) non-zero parameters can be represented by a network with width bounded by w ( /epsilon1 ) ≤ W ( /epsilon1 ). In the case of d ∗ = D , the Lemma simply states the approximation error for conventional H¨ older spaces as established in Yarotsky [2017].

Lemma B.3 (Approximation of Generalized Interaction Models by Deep ReLU Networks) .

Consider the function class G ≡ G ( p, d ∗ , [0 , 1] D ) (Definition 6d) and consider some arbitrary random variable x ∈ [0 , 1] D with probability measure P x . For any small enough /epsilon1,η &gt; 0 , the class of deep tanh networks F ≡ F tanh ( L, W ( /epsilon1 ) , w ( /epsilon1 ) , κ ( /epsilon1 )) (Definition 2) satisfies:

<!-- formula-not-decoded -->

for some subset X ⊂ [0 , 1] D with P x ( x /negationslash∈ X ) ≤ η as long as W ( /epsilon1 ) = c 1 /epsilon1 -d ∗ /p , w ( /epsilon1 ) = c 2 /epsilon1 -d ∗ /p , κ ( /epsilon1 ) = c 3 /epsilon1 -c 4 /η for any large enough choice of L , c 1 , c 2 , c 3 , c 4 &gt; 0 .

Proof. This directly follows from Theorem 3 in Bauer et al. [2019], however our notation is greatly simplified by the fact that we are not interested in most of their constants, and that we offloaded most of the assumptions into Definition 6. What matters is that the network they construct has a depth that bounded by a constant (their equation (6)), and a number of non-zero parameters that is proportional to what they call ( M n + 1) d ∗ in their Theorem 3 (by their equations (7) and (5) and the definition of M ∗ in their Theorem 3). Since we assumed bounded support (leaving their a n as a constant), their bound yields an approximation error of /epsilon1 =: cM -p n for some c &gt; 0, such that the number of non-zero parameters can be bounded as W ( /epsilon1 n ) = O ( ( M n +1) d ∗ ) = O ( /epsilon1 -d ∗ /p ). They bound κ ( /epsilon1 ) ( α in their notation) in terms of M n and η yielding κ ( /epsilon1 ) = O ( /epsilon1 -c 4 /η ) for some large enough constant c 4 &gt; 0. Finally, their theorem holds only for activation functions σ which satisfy a property they call N-admissible . While this is technically not satisfied by σ ( x ) = tanh ( x ), it is easy to verify that this property is satisfied by the activation function ˜ σ ( x ) = 1 / 2+tanh( x ) / 2. Since for any ˜ f ∈ F ˜ σ ( L, W, w, κ ) there exists some f ∈ F tanh ( L, W, w, 2 κ +1 / 2) such that ˜ f = f , the same approximation bound holds with σ ( x ) = tanh ( x ).

Lemma B.4

If Y is iid and l ( θ, Y ) : θ Θ is P -Donsker for some l : Θ R satisfying

(Empirical Process of Donsker Classes) . ∈ Y { ∈ } ×Y ↦→

then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for any δ n = o P (1) .

Proof. This directly follows from Lemma 1 in Chen et al. [2003].

Lemma B.5 (Empirical Process Rates for A-Estimators) .

Under the assumptions of Theorem 3.1, for any function f ( θ, λ, Y ) satisfying the following conditions:

- For any sequence e n ≥ 0 and all θ ∈ Θ , λ ∈ Λ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

at least if the right hand sides are smaller than some C &gt; 0 .

- For all small ε &gt; 0 , we have:

<!-- formula-not-decoded -->

we obtain the following empirical processes bounds:

<!-- formula-not-decoded -->

where ̂ Λ n ( θ ) := { λ ∈ Λ n : E [ l θ ( λ θ ∗ , Y ) -l θ ( λ, Y )] ≺ E [ l θ ( λ θ ∗ , Y ) -l θ ( ̂ λ θ n , Y )] } and ̂ Θ n := { θ ∈ Θ : E [ l ( θ, Y ) -l ( θ ∗ , Y )] ≺ E [ l ( ̂ θ n , Y ) -l ( θ ∗ , Y )] } are shrinking neighborhoods around λ θ ∗ and θ ∗ containing ̂ λ θ n and ̂ θ n .

## C Proofs

## C.1 Theorem 3.1 and Lemma B.5

Theorem 3.1 and Lemma B.5 are simplified versions of the slightly more general Theorems C.1 and C.2, which modify Shen and Wong [1994]'s M-estimator convergence rate arguments to hold uniformly over another parameter space and accommodate estimators which are finite-sample optimal up to some stochastic remainder. Theorem C.1 is presented in C.1.1 and derives the uniform convergence rates for ̂ λ θ n . Theorem C.2 is presented in C.1.2 and derives the rates for ̂ θ n . In C.1.3, we then discuss how Theorem 3.1 and Lemma B.5 follow from these results.

Theorem C.1 (Uniform Convergence Rates of Sieve M-Estimators) . Let ρ θ ( · , · ) be a pseudo-distance on Λ , possibly indexed by θ ∈ Θ . For the estimator ̂ λ θ n of 3.2, assume: CONDITION C1a. For some constants A 1 &gt; 0 and α &gt; 0 , and all small ε &gt; 0 :

## C.1.1 Uniform convergence rate of ̂ λ θ n

<!-- formula-not-decoded -->

CONDITION C1b. For some constants A 2 &gt; 0 and β &gt; 0 , and all small ε &gt; 0 :

<!-- formula-not-decoded -->

CONDITION C2. Let F n = { l θ ( λ, · ) -l θ ( π n λ θ ∗ , · ) : λ ∈ Λ n , θ ∈ Θ n } . For some r 0 &lt; 1 2 , A 3 &gt; 0 and all small ε &gt; 0 , its entropy (Def. 1) is bounded as:

<!-- formula-not-decoded -->

where either r &gt; 0 or r = 0 + , which is understood to represent ε -0 + = log(1 /ε ) . Let /epsilon1 n := sup θ ∈ Θ n ρ θ ( π n λ θ ∗ , λ θ ∗ ) ∨ ∣ ∣ E [ l θ ( λ θ ∗ , Y ) -l ( π n λ θ ∗ , Y ) ]∣ ∣ 1 / 2 α , then where τ = τ ( α, β, r, r 0 , n ) is given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

And for any f ( θ, λ, Y ) satisfying C1a and C2 when l θ ( λ, Y ) is replaced by f ( θ, λ, Y ) , we can bound the empirical process as follows:

<!-- formula-not-decoded -->

where ̂ Λ n ( θ ) is defined as in Lemma B.5. Proof. The theorem generalizes Theorem 1 of Shen and Wong [1994] such that it holds uniformly over a family of losses indexed by the parameter θ ∈ Θ n , and to allow for the finite-sample optimum to hold approximately up to a possibly stochastic sequence η n . Fortunately, the proof can remain almost identical. Shen and Wong [1994] prove the Theorem by induction, through a chaining argument. They use their Lemma 2 to derive an initial, slow rate which corresponds to the induction start, yielding the assumptions of their Lemma 3 at step k = 2. Next, their Lemma 3 is repeatedly applied as the induction steps until the rates of Theorem C.1 are obtained. We do not reproduce these algebraic steps, as they are the same as in Shen and Wong [1994]. Like Shen and Wong [1994], we also do not provide the proof for the induction start as it is similar, but simpler than the proof of the induction step, which we present in Lemma C.1.

Lemma C.1 (Induction Step for Theorem C.1) .

Suppose Conditions C1a, C1b and C2 hold. If at Step k -1 we have a rate ε ( k -1) n = n -α k -1 &gt; max ( n -(1 -2 r 0 ) / [ α ( r +2)] , /epsilon1 n ) so that where δ ∗ = min ( r +4 r 0 r +2 , βr (1 -2 r 0 ) 4 α + r 0 ) and L = (1 -ε ) min ( M 2 D 2 α , M 3 D 4 α -β (2 -r ) / 2 ) Then at Step k, we can find an improved rate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α k = (1 -2 r 0 ) / (4 α ) + α k -1 β (2 -r ) / (4 α ) , so that

And for any function f ( θ, λ, Y ) satisfying C1b and C2 when l θ ( λ, Y ) is replaced by f ( θ, λ, Y ) , and Λ n ( θ ) is defined as in Lemma B.5, the same bound applies to:

Proof. We assume D &gt; 1 (wlog) and we only prove the case of 4 α ≥ β (2 -r ) / 2 . Let B ( i ) n = { Dε ( i ) n ≤ sup θ ∈ Θ n ρ θ ( ̂ λ θ n , λ θ ∗ ) &lt; Dε ( i -1) n } for i = 2 , . . . , k . Then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To prove the Lemma , we only need to tackle P ( B ( k ) n ) . By Condition C1a,

The last inequality requires A 1 ( Dε ( k ) n ) 2 α -sup θ ∈ Θ n E [ l θ ( λ θ ∗ , Y ) -l θ ( π n λ θ ∗ , Y )] -η n &gt; 0. This is holds for A 1 &gt; 2 (wlog), which follows from ε ( k ) n ≥ /epsilon1 n and ε ( k ) n ≥ η 1 / 2 α n .

Thus

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let v k = sup { Dε ( k ) n ≤ ρ θ ( λ,λ θ ∗ ) &lt;Dε ( k -1) n ,λ ∈ Λ n ,θ ∈ Θ n } Var ( l θ ( π n λ θ ∗ , Y ) -l θ ( λ, Y ) ) . By Condition C1b and ε ( k -1) n ≥ /epsilon1 n we get v k ≤ 4 A 2 ( Dε ( k -1) n ) 2 β . Since ε ( k ) n satisfies we know that n 1 / 2 ( Dε ( k ) n ) 2 α ≥ max ( c 1 n -(2 -r -8 r 0 ) / [2( r +2)] , c 2 ( Dε k -1 n ) 2 β (2 -r ) / 4 n r 0 ) for some constants c 1 &gt; 0 and c 2 &gt; 0. We can therefore apply Shen and Wong [1994]'s Lemma 1 and obtain:

<!-- formula-not-decoded -->

ψ 1 ( A 1 n 1 / 2 ( Dε ( k ) n ) 2 α , v k , F n ) ≥ 3 A 1 4 n ( Dε ( k ) n ) 2 α ≥ M 2 D 2 α nn -2(1 -2 r 0 ) / ( r +2) ≥ M 2 D 2 α n ( r +4 r 0 ) / ( r +2)

The behavior of ψ 1 ( · ) can be analyzed via Shen and Wong [1994]'s Remark 12. (i) If ( Dε ( k ) n ) 2 α A 1 &gt; 12 ( Dε ( k -1) n ) 2 β , then for some constant M 2 &gt; 0. (ii) If Dε ( k ) n 2 α A 1 ≤ 12 Dε ( k -1) n 2 β , then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some M 3 &gt; 0. Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Take δ ∗ , L and ε ( k ) n as defined in the Lemma, and we obtain P ( B ( k ) n ) ≤ 5 exp ( -Ln δ ∗ ) . This yields the convergence rate of ̂ λ θ n . The statement about arbitrary f ( θ, λ, Y ) follows by applying analogous arguments (starting at the definition of v k ) to the expression f ( θ, π n λ θ ∗ , Y ) -f ( θ, λ, Y ) instead of l θ ( π n λ θ ∗ , Y ) -l θ ( λ, Y ).

## C.1.2 Convergence of ̂ θ n

Theorem C.2 (Convergence Rate of A-Estimators) . Consider the family of Mestimators ̂ λ θ n defined in 3.2 and the A-estimator ̂ θ n defined in 3.3. Assume that conditions C1b and C2 are satisfied with ρ θ ( λ, λ ′ ) := ∣ ∣ E [ l θ ( λ, Y ) -l θ ( λ ′ , Y )] ∣ ∣ (hence C1a is automatically satisfied with α = 1 / 2 ). Let ¯ ρ ( · , · ) be some pseudo-distance on Θ . Assume that the following conditions are satisfied:

CONDITION C1a' For some constants ¯ A 1 &gt; 0 and ¯ α &gt; 0 , and all small ε &gt; 0 :

<!-- formula-not-decoded -->

CONDITION C1b'. For some constants ¯ A 2 &gt; 0 and ¯ β &gt; 0 , and all small ε &gt; 0 :

<!-- formula-not-decoded -->

CONDITION C2'. Let ¯ F n = { l ( θ, π n λ θ ∗ , · ) -l ( π n θ ∗ , π n λ π n θ ∗ ∗ , · ) : θ ∈ Θ n } . For some ¯ r 0 &lt; 1 2 , ¯ A 3 &gt; 0 and all small ε &gt; 0 , its entropy (Def. 1) is bounded as:

where either ¯ r &gt; 0 or ¯ r = 0 + , which represents ε -0 + = log(1 /ε ) .

<!-- formula-not-decoded -->

Let ¯ /epsilon1 n := ρ ( π n θ ∗ , θ ∗ ) ∨ | E l ( π n θ ∗ , Y ) -l ( θ ∗ , Y ) | 1 / 2¯ α be the approximation error of Θ n . Then:

Where τ = τ (1 / 2 , β, r, r 0 , n ) and /epsilon1 n are defined as in Thm C.1 and ¯ τ = τ (¯ α, ¯ β, ¯ r, ¯ r 0 , n ) . Also, for every f ( θ, λ, Y ) satisfying C2 and C3 when l ( θ, λ, Y ) is replaced by f ( θ, λ, Y ) (recall l ( θ, Y ) = l ( θ, λ θ ∗ , Y ) ), we can bound the empirical process:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. The proof is similar to that of Theorem C.1. Again, we will only prove the induction step via Lemma D.1 in the Online Appendix, as the remaining arguments are analogous to the proof of the previous Theorem or that of Theorem 1

in Shen and Wong [1994]. The proof of Lemma D.1 largely mirrors that of Lemma C.1, but uses the results of Theorem C.1 to control the convergence of the adversary. The main additional complexity lies in properly switching back and forth between the sieve spaces and the target function spaces, when bounding the empirical process terms and variances respectively.

## C.1.3 Proofs of Theorem 3.1 and Lemma B.5

Proof. To see that Theorem 3.1 and Lemma B.5 follow from the previous results, simply choose ρ θ ( λ, λ ′ ) = | E [ l θ ( λ, Y ) -l θ ( λ ′ , Y )] | and ¯ ρ ( θ, θ ′ ) = | E [ l ( θ, Y ) -l ( θ ′ , Y )] | , such that Conditions C1a and C1a' are automatically satisfied with α = 1 / 2. Further, substitute γ = 2 β ∨ 2 ¯ β such that Conditions C1b and C1b' hold by assumptions 3.7 and 3.6 for all ε &lt; 1 ∧ C . Conditions C2 and C2' directly follow from 3.8, substituting s = 2 r 0 = 2¯ r 0 and fixing r = ¯ r . This yields Theorem 3.1, and Lemma B.5 with e n = 0.

/negationslash

For a proof of Lemma B.5 with e n = 0, note that Condition C1b in Theorem C.1 is only needed to verify v k ≤ 4 A 2 ( Dε ( k -1) n ) 2 β in the proof of Lemma C.1. Hence we can re-define /epsilon1 n ← /epsilon1 n + e n such that the definition of ε ( k -1) n ≥ /epsilon1 n automatically ensures v k ≺ ( Dε ( k -1) n ) 2 β . This change in constants does not affect the result. Analogous arguments can be applied to Lemma D.1 and thus Theorem C.2.

## C.2 Theorem 3.3

Proof. The approximate Nash conditions 1.2 and 1.3 imply

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̂ ̂ The second line uses Taylor's theorem. The third line substitutes the definitions of ¯ θ n , ¯ λ ̂ θ n n and applies Condition N3. Since the signs of v, λ ′ ̂ θ n ∗ [ v ] are arbitrary, we may replace the inequality with an equality, which yields O P ( e n ) = E n l ′ ( ̂ θ n , ̂ λ n , Y )[ v, λ ′ ̂ θ n ∗ [ v ]]. Adding and subtracting a few terms, we get:

<!-- formula-not-decoded -->

with V = V ( l ′ ( θ ∗ , Y )[ v ]) by the standard central limit theorem.

## C.3 Theorem 3.4

Proof of Theorem 3.4. Note that the regularization does not interfere with Theorem 3.2: the approximation power relative to Θ ∗ , Λ ∗ is not reduced as we remove only elements form the sieves Θ n , Λ n that are far away from Θ ∗ , Λ ∗ , and the regularization is sufficiently slow to guarantee that the sieves are nonempty for small enough /epsilon1 &gt; 0. Hence Theorem 3.2 holds with ¯ r = 2, yielding rates o P ( n -2 / (2+¯ r ) ) = o P ( n -1 / 2 ). Also note that the assumption d ∗ p ∨ ¯ d ∗ ¯ p &lt; 1 / 4 along with the lower bound r &gt; 2 / 3 ensures sup θ ∈ Θ ∗ ‖ θ -π n θ ‖ ¯ X = o ( n -1 ) and sup λ ∈ Λ ∗ ‖ λ -π n λ ‖ X = o ( n -1 ). We first verify condition N1, decomposing it into two parts by adding and subtracting l ′ ( θ, λ ′ θ ∗ , Y )[ v ∗ , λ ′ θ ∗ [ v ∗ ]] = l ′ ( θ, Y )[ v ∗ ]. First, we show

<!-- formula-not-decoded -->

If A7ii) holds with V [ l ′ ( θ ∗ , Y )[ v ] -l ′ ( θ, Y )[ v ]] ≺ E [ l ( θ, Y ) -l ( θ ∗ , Y )], then this can be established with Lemma B.5 for γ = 1, using the Lipschitz condition A4 and analogous arguments to those in the proof of Theorem 3.2. Lemma B.5 then yields the same o P ( n -1 / 2 ) rates as Theorem 3.2. If A7ii) instead asserts that Θ ∗ is P -Donsker, the same result follows from Lemma B.4, which can be applied because the Lipschitz continuity A4 implies the L2 continuity required by the Lemma. This implies condition N1, together with:

<!-- formula-not-decoded -->

which can be established via analogous arguments and A7i). We proceed to verify condition N2. Using a similar decomposition, we note that

<!-- formula-not-decoded -->

which follows from A6i) and the o P ( n -1 / 2 ) rates of Theorem 3.2. Similarly, A6ii) implies sup θ ∈ ̂ Θ n ,λ ∈ ̂ Λ n ( θ ) E l ′ ( θ, Y )[ v ∗ ] -l ′ ( θ ∗ , Y )[ v ∗ ] -〈 θ -θ ∗ , v ∗ 〉 = O P ( e n ) hence condition N2 holds. Finally, we verify condition N3. Define π ∗ θ := arg inf θ ′ ∈ Θ ∗ ‖ θ ′ -θ ‖ ∞ as the projection onto Θ ∗ . Similarly, π ∗ λ := arg inf λ ′ ∈ Λ ∗ ‖ λ ′ -λ ‖ ∞ . Due to the reguarlization, we know ‖ θ -π ∗ θ ‖ ∞ = o ( n -1 ). Therefore ‖ ¯ θ n ( θ ) -( π ∗ θ -e n v ∗ ) ‖ ∞ = o ( n -1 ). By convexity of Θ ∗ , we have ( π ∗ θ -e n v ∗ ) ∈ Θ ∗ for n large enough. Given that sup θ ′ ∈ Θ ∗ ‖ θ ′ -π n θ ′ ‖ ∞ = o ( n -1 ) due to d ∗ p ∨ ¯ d ∗ ¯ p &lt; 1 / 4 and our choice of r , we get ‖ ( π ∗ θ -e n v ∗ ) -θ n ( π ∗ θ -e n v ∗ ) ‖ ∞ = o ( n -1 ). Taken together, these statements imply ‖ ¯ θ n ( θ ) -π n ¯ θ n ( θ ) ‖ ∞ = o ( n -1 ). Analogous arguments yield ‖ ¯ λ θ n ( λ ) -π n ¯ λ θ n ( λ ) ‖ ∞ = o ( n -1 ). Hence N3 holds.

## D Online Appendix

## D.1 Proof of Theorem C.2

The induction step for the proof of Theorem C.2 is given by the following Lemma.

Lemma D.1 (Induction Step for Theorem C.2) . Suppose the Conditions of Theorem C.2 hold. If at Step k -1 we have a rate so that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where δ ∗ = min ( r +4¯ r 0 r +2 , ¯ β ¯ r (1 -2¯ r 0 ) 4¯ α + ¯ r 0 ) and L = (1 -ε ) min ( M 2 D 2¯ α , M 3 D 4¯ α -¯ β (2 -¯ r ) / 2 ) , then at Step k , we can find an improved rate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ¯ α k = (1 -2¯ r 0 ) / (4¯ α ) + ¯ α k -1 ¯ β (2 -¯ r ) / (4¯ α ) , so that

Furthermore, for every f ( θ, λ, Y ) satisfying Conditions C1b and C2 when l ( θ, λ, Y ) is replaced by f ( θ, λ, Y ) (recall l ( θ, Y ) = l ( θ, λ θ ∗ , Y ) ), the same bound applies to:

<!-- formula-not-decoded -->

Proof. As in the proof of Lemma C.1 we assume D &gt; 1 (wlog) and we only prove the case of 4¯ α ≥ ¯ β (2 -¯ r ) / 2. Let B ( i ) n = Dε ( i ) n ≤ ¯ ρ θ n , θ ∗ &lt; Dε ( i -1) n for i = 2 , . . . , k .

As before, we only need to bound P B ( k ) n . To this end, it will be useful to define

<!-- formula-not-decoded -->

{ ( ̂ ) } ( )

such that Theorem C.1 implies r n = O P ( n τ + /epsilon1 n + η n ), which also implies, by definition of r n and /epsilon1 n : sup θ ∈ Θ n ∣ ∣ ∣ E n [ l ( θ, ̂ λ θ n , Y ) -l ( θ, π n λ θ ∗ , Y )] ∣ ∣ ∣ ≤ r n + /epsilon1 n ≤ 2 r n . Together with 3.3 this yields:

<!-- formula-not-decoded -->

By Condition C1a', we can therefore bound:

<!-- formula-not-decoded -->

Where the last line used C1a', the definition of the approximation errors, assumed large enough ¯ A 1 (wlog) and used various lower-bounds implied by the definition of ε ( k ) n . Together with D.1, this yields:

<!-- formula-not-decoded -->

Let v k = sup { Dε ( k ) n ≤ ¯ ρ ( θ,θ ∗ ) &lt;Dε ( k -1) n ,θ ∈ Θ n } V [ l ( π n θ ∗ , π n λ π n θ ∗ ∗ , Y ) -l ( θ, π n λ θ ∗ , Y )] . To bound v k , we add and subtract terms and apply the Cauchy-Schwartz inequality:

<!-- formula-not-decoded -->

By Conditions C1b and C1b', and since ε ( k -1) n ≥ ¯ /epsilon1 n , we obtain v k ≤ 4 ¯ A 2 ( Dε ( k -1) n ) 2 ¯ β , assuming ¯ A 2 is large enough (wlog). The remaining arguments are unchanged from the proof of Lemma C.1, which eventually yields: P ( B ( k ) n ) ≤ 5 exp ( -Ln δ ∗ ) . This completes the proof for the convergence rate of ̂ θ n . To prove the statement about arbitrary f ( θ, λ, Y ) satisfying C1b and C2, simply repeat the arguments of the previous proof (starting at the definition of v k ) with l ( θ, λ, Y ) replaced by f ( θ, λ, Y ).

## D.2 Proof of Proposition 4.1

Proof. The first order conditions for λ in 2.1 yield the optimal population adversary λ θ ∗ ( y ):

We verify the conditions of Theorem 3.2. The Lipschitz condition A1 can be verified by writing l ( θ, λ, Y ) = ∫ λ ( y ) d P θ d P ( y )d P ( y ) -f ∗ ( λ ( Y )) and using the Lipschitzness of d P θ d P in θ and that of f ∗ (which follows from boundedness of Λ and differentiability of f ). Towards A2, apply a 2nd order Taylor expansion (with mean value reminder) of D f ( P θ ‖ P ) at d P θ d P = d P θ ∗ d P in direction of d P θ d P , which yields for some ˜ θ ∈ Θ on a path from θ ∗ to θ :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step follows from strict positivity and boundedness of f ′′ ( d P θ d P ) wp1. Further note that due to Lipschitzness of f ∗ . Also note that Lipschitzness of d P θ d P in θ implies

<!-- formula-not-decoded -->

Taken together, this implies A2. A3 can be verified analogously, starting with a Taylor expansion yielding E [ l ( θ, λ, Y ) -l ( θ, λ θ ∗ , Y )] = -∫ f ′′ ∗ ( ˜ λ )( λ -λ θ ∗ ) 2 d P /equivasymptotic -‖ λ -λ θ ∗ ‖ 2 L 2 ( Y ) for some ˜ λ ∈ Λ on a path from λ θ ∗ to λ . D.3 Proof of Proposition 4.2

Proof. First, we verify the conditions of Theorem 3.4. Note that

<!-- formula-not-decoded -->

The Lipschitz condition A4 is therefore satisfied by the Lipschitzness of f ′ ∗ , d P θ d P and that of ∇ θ → v d P θ d P . Towards A6 i), we apply a Taylor expansion with 2nd order mean-

<!-- formula-not-decoded -->

where we used ∇ λ θ ∗ → λ -λ θ ∗ E [ l ( θ, λ θ ∗ , Y )] ≡ 0 to get rid of the first-order term. The last line follows from the boundedness (in absolute value) of f ′′ ∗ ( · ) and w ( · ) on their compact support. Hence A6i) is satisfied. A7i) follows from:

<!-- formula-not-decoded -->

where the last step used the Lipschitzness of f ′ ∗ and again the boundedness of w ( · ). Assumption A7 ii) is satisfied since × ∗ is Donsker by assumption. Finally, we verify A6 ii). Applying the mean value theorem twice, we get that for some ˜ θ, ˜ θ ′ on a path between θ ∗ and θ , E [ l ′ ( θ, Y )[ v ∗ ] -l ′ ( θ ∗ , Y )[ v ∗ ] -〈 θ -θ ∗ , v ∗ 〉 = ∇ ˜ θ → ˜ θ ′ -θ ∗ ∇ ˜ θ → θ -θ ∗ E [ l ′ ( ˜ θ, Y )[ v ∗ ]] which is dominated by E [ l ( θ, Y )] = D f ( P θ ‖ P θ ∗ ) via the last assumption stated in the proposition. Therefore A6ii) is satisfied, and Theorem 2.4 applies.

## D.4 Proof of Proposition 4.6

Proof. Note that V = ∇ θ ∗ ∇ θ ′ ∗ E [ l ( θ ∗ , Y )] = V [ ∇ θ ∗ l ( θ ∗ , Y )]. We verify the conditions of Theorem 3.4, for v ∗ = V -1 ζ , such that its conclusion becomes √ n 〈 θ -θ ∗ , v ∗ 〉 = √ n ( θ -θ ∗ ) ′ ζ → N (0 , ζ ′ V -1 ζ ), which yields the Proposition via the Cram´ er-Wold device. Note that l ′ ( θ, λ, Y )[ v, w ] = v ′ d ( X,θ ) ′ λ ( Z ) + m ( X,θ ) ′ w ( Z ) -1 2 λ ( Z ) ′ w ( Z ) Hence assumption A4 follows from boundedness and Lipschitzness of d ( X, · ) , m ( X, · ). A5 holds by assumption. To verify A6i), notice that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality used the fact that E [ v ′ ∗ d ( X,θ ) | Z ] = 1 2 λ ′ θ ∗ [ v ∗ ]( Z ). Towards Assumption A6ii), note that we can apply a first-order Taylor expansion with meanvalue reminder twice, which yields for some ¯ θ, θ on a path between θ ∗ and θ :

Given the identification assumption 2.5, we can use a second-order Taylor expansion with mean-value reminder to show that ‖ θ -θ ∗ ‖ 2 2 /equivasymptotic E [ l ( θ, Y ) -l ( θ ∗ , Y )], which then yields A6ii). Similarly, we can show A7ii) via a mean-value reminder:

<!-- formula-not-decoded -->

We similarly can establish A7i) via

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## D.5 Proof of Proposition 4.4

Proof. A0 holds by assumption, and condition A1 is implied by Lipschitzness of V θ , P θ in θ . Continuity of V θ , P θ , compactness of Θ and A0 further imply that there is some 0 ≤ M &lt; ∞ such that

<!-- formula-not-decoded -->

Given that λ θ ∗ ∗ ≡ 0 = ⇒ l ( θ ∗ , Y ) = 0, condition A2 then follows from

<!-- formula-not-decoded -->

as well as E [ l ( θ, Y ) -l ( θ ∗ , Y )] ≺ E [( λ θ ∗ ( s, a ) -λ θ ∗ ∗ ( s, a )) 2 | ( s, a ) ∈ ˜ X ] + P (( s, a ) /negationslash∈ ˜ X ) ≺ ‖ θ -θ ∗ ‖ 2 ˜ X + P (( s, a ) /negationslash∈ ˜ X ) , ∀ ˜ X ⊂ ¯ X . A3 can be established analogously, hence the conclusions of Theorem 3.2 hold.

## D.6 Proof of Proposition 4.7

Proof. Note that λ θ ∗ ( x ) = θ ( x ) -θ ∗ ( x ) follows from the first order conditions of the adversary. Condition A0 is satisfied by assumption, and A1 follows from Lipschitzness of m ( Y, · ) and boundedness. Lipschitzness of m ( θ, λ ( x )) in λ ( x ) and boundedness imply that l ( θ, λ, Y ) = m ( Y, λ ) -θ ( x ) λ ( x ) -λ ( x ) 2 / 2 is Lipschitz in λ ( x ). This implies

<!-- formula-not-decoded -->

and together with λ θ ∗ ( x ) = θ ( x ) -θ ∗ ( x ), it yields:

<!-- formula-not-decoded -->

Both bounds imply A3 and A2 respectively. The result follows because the loss E [ l ( θ, Y )] is proportional to the squared L2 norm of θ -θ ∗ .
