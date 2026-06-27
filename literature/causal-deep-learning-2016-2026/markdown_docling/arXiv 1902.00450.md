<!--
source: /Users/pranjal/Code/deep-inference/literature/causal-deep-learning-2016-2026/downloads/arXiv 1902.00450.pdf
backend: docling
part: 1/1
-->

## Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders

Ioana Bica 1 2 Ahmed M. Alaa 3 Mihaela van der Schaar 2 3 4

## Abstract

The estimation of treatment effects is a pervasive problem in medicine. Existing methods for estimating treatment effects from longitudinal observational data assume that there are no hidden confounders, an assumption that is not testable in practice and, if it does not hold, leads to biased estimates. In this paper, we develop the Time Series Deconfounder, a method that leverages the assignment of multiple treatments over time to enable the estimation of treatment effects in the presence of multi-cause hidden confounders. The Time Series Deconfounder uses a novel recurrent neural network architecture with multitask output to build a factor model over time and infer latent variables that render the assigned treatments conditionally independent; then, it performs causal inference using these latent variables that act as substitutes for the multi-cause unobserved confounders. We provide a theoretical analysis for obtaining unbiased causal effects of time-varying exposures using the Time Series Deconfounder. Using both simulated and real data we show the effectiveness of our method in deconfounding the estimation of treatment responses over time.

## 1. Introduction

Forecasting the patient's response to treatments assigned over time represents a crucial problem in the medical domain. The increasing availability of observational data makes it possible to learn individualized treatment responses from longitudinal disease trajectories containing information about patient covariates and treatment assignments (Robins et al., 2000a; Robins &amp; Hern´ an, 2008; Schulam

1 University of Oxford, Oxford, United Kingdom 2 The Alan Turing Institute, London, United Kingdom 3 UCLA, Los Angeles, USA 4 University of Cambridge, Cambridge, United Kingdom. Correspondence to: Ioana Bica &lt; ioana.bica@eng.ox.ac.uk &gt; .

Proceedings of the 37 th International Conference on Machine Learning , Online, PMLR 119, 2020. Copyright 2020 by the author(s).

&amp;Saria, 2017; Lim et al., 2018; Bica et al., 2020a). Existing methods for estimating individualized treatment effects over time assume that all confounders-variables affecting the treatment assignments and the potential outcomes-are observed, an assumption which is not testable in practice 1 and probably not true in many situations.

To understand why the presence of hidden confounders introduces bias, consider the problem of estimating treatment effects for patients with cancer. They are often prescribed multiple treatments at the same time, including chemotherapy, radiotherapy and/or immunotherapy based on their tumor characteristics. These treatments are adjusted if the tumor size changes. The treatment strategy is also changed as the patient starts to develop drug resistance (Vlachostergios &amp;Faltas, 2018) or the toxicity levels of the drugs increase (Kroschinsky et al., 2017). Drug resistance and toxicity levels are multi-cause confounders since they affect not only the multiple causes (treatments) 2 , but also the patient outcome (e.g. mortality, risk factors). However, drug resistance and toxicity may not be observed and, even if observed, may not be recorded in the electronic health records. Estimating, for instance, the effect of chemotherapy on the cancer progression in the patient without accounting for the dependence on drug resistance and toxicity levels (hidden confounders) will produce biased results.

Wang &amp; Blei (2019a) developed theory for deconfoundingadjusting for the bias introduced by the existence of hidden confounders in observational data-in the static causal inference setting and noted that the existence of multiple causes makes this task easier. Wang &amp; Blei (2019a) observed that the dependencies in the assignment of multiple causes can be used to infer latent variables that render the causes independent and act as substitutes for the hidden confounders.

In this paper, we propose the Time Series Deconfounder, a method that enables the unbiased estimation of treatment responses over time in the presence of hidden confounders, by taking advantage of the dependencies in the sequential assignment of multiple treatments. We draw from the main

1 Since counterfactuals are never observed, it is not possible to test for the existence of hidden confounders that could affect them.

2 Causes and treatments are used interchangeably throughout the paper.

idea in Wang &amp; Blei (2019a), but note that the estimation of hidden confounders in the longitudinal setting is significantly more complex than in the static setting, not just because the hidden confounders may vary over time but in particular because the hidden confounders may be affected by previous treatments and covariates. Thus, standard latent variable models are no longer applicable, as they cannot capture these time dependencies.

The Time Series Deconfounder relies on building a factor model over time to obtain latent variables which, together with the observed variables render the assigned causes conditionally independent. Through theoretical analysis we show that these latent variables can act as substitutes for the multicause unobserved confounders and can be used to satisfy the sequential strong ignorability condition in the potential outcomes framework for time-varying exposures (Robins &amp; Hern´ an, 2008) and obtain unbiased estimates of individualized treatment responses, using weaker assumptions than standard methods. Following our theory, we propose a novel deep learning architecture, based on a recurrent neural network with multi-task outputs and variational dropout, to build such a factor model and infer the substitutes for the hidden confounders in practice.

The Time Series Deconfounder shifts the need for observing all multi-cause confounders (untestable condition) to constructing a good factor model over time (testable condition). To assess how well the factor model captures the distribution of assigned treatments, we extend the use of predictive checks (Rubin, 1984; Wang &amp; Blei, 2019a) to the temporal setting and compute p -values at each timestep. We perform experiments on a simulated dataset where we control the amount of hidden confounding applied and on a real dataset with patients in the ICU (Johnson et al., 2016) to show how the Time Series Deconfounder allows us to deconfound the estimation of treatment responses in longitudinal data. To the best of our knowledge, this represents the first method for learning latent variables that can act as substitutes for the unobserved confounders in the time series setting.

## 2. Related Work

Previous methods for causal inference mostly focused on the static setting (Hill, 2011; Wager &amp; Athey, 2017; Alaa &amp; van der Schaar, 2017; Shalit et al., 2017; Yoon et al., 2018; Alaa &amp; Schaar, 2018; Zhang et al., 2020; Bica et al., 2020c), and less attention has been given to the time series setting. We discuss methods for estimating treatment effects over time, as well as methods for inferring substitute hidden confounders in the static setting.

Potential outcomes for time-varying treatment assignments . Standard methods for performing counterfactual inference in longitudinal data are found in the epidemiol- ogy literature and include the g-computation formula, gestimation of structural nested mean models, and inverse probability of treatment weighting of marginal structural models (Robins, 1994; Robins et al., 2000a; Robins &amp; Hern´ an, 2008). Alternatively, (Lim et al., 2018) improves on the standard marginal structural models by using recurrent neural networks to estimate the propensity weights and treatment responses, while (Bica et al., 2020a) propose using balancing representations to handle the time-dependent confounding bias when estimating treatment effects over time. Despite the wide applicability of these methods in forecasting treatment responses, they are all based on the assumption that there are no hidden confounders. Our paper proposes a method for deconfounding such outcome models, by inferring substitutes for the hidden confounders which can lead to unbiased estimates of the potential outcomes.

The potential outcomes framework has been extended to the continuous-time setting by (Lok et al., 2008). Several methods have been proposed for estimating treatment responses in continuous time (Soleimani et al., 2017; Schulam &amp;Saria, 2017), again assuming that there are no hidden confounders. Here, we focus on deconfounding the estimation of treatment responses in the discrete-time setting.

Sensitivity analysis methods that evaluate the potential impact that an unmeasured confounder could have on the estimation of treatment effects have also been developed (Robins et al., 2000b; Roy et al., 2016; Scharfstein et al., 2018). However, these methods assess the suitability of applying existing tools, rather than propose a direct solution for handling the presence of hidden confounders in observational data.

Latent variable models for estimating hidden confounders. The most similar work to ours is the one of Wang &amp; Blei (2019a), who proposed the deconfounder, an algorithm that infers latent variables that act as substitutes for the hidden confounders and then performs causal inference in the static multi-cause setting. The deconfounder involves finding a good factor model of the assigned causes which can be used to estimate substitutes for the hidden confounders. Then, the deconfounder fits an outcome model for estimating the causal effects using the inferred latent variables. Our paper extends the theory for the deconfounder to the time-varying treatments setting and shows how the inferred latent variables can lead to sequential strong ignorability. To estimate the substitute confounders, Wang &amp; Blei (2019a) used standard factor models (Tipping &amp; Bishop, 1999; Ranganath et al., 2015), which are only applicable in the static setting. To build a factor model over time, we propose an RNN architecture with multitask output and variational dropout.

Several other methods have been proposed for taking advantage of the multiplicity of assigned treatments in the

static setting and capture shared latent confounding (Tran &amp; Blei, 2018; Heckerman, 2018; Ranganath &amp; Perotte, 2018). These works are based on Pearl's causal framework (Pearl, 2009) and use structural equation models. Alternative methods for dealing with hidden confounders in the static setting use proxy variables as noisy substitutes for the confounders (Lash et al., 2014; Louizos et al., 2017; Lee et al., 2018).

A different line of research involves performing causal discovery in the presence of hidden confounders (Spirtes et al., 2000). In this context, several methods have been proposed to perform causal graphical model structure learning with latent variables (Leray et al., 2008; Jabbari et al., 2017; Raghu et al., 2018). However, in this paper, we are not aiming to discover causal relationships between patient covariates over time. Instead, we improve existing methods for estimating the individualized effects of time-dependent treatments by accounting for multi-cause unobserved confounders.

## 3. Problem Formulation

Let the random variables X ( i ) t ∈ X t be the time-dependent covariates, A ( i ) t = [ A ( i ) t 1 . . . A ( i ) tk ] ∈ A t be the possible assignment of k treatments (causes) at timestep t and let Y ( i ) t +1 ∈ Y t be the observed outcomes for patient ( i ) . Treatments can be either binary and/or continuous. Static features, such as genetic information, do not change our theory, and, for simplicity, we assume they are part of the observed covariates.

The observational data for patient ( i ) , also known as the patient trajectory, consists of realizations of the previously described random variables ζ ( i ) = { x ( i ) t , a ( i ) t , y ( i ) t +1 } T ( i ) t =1 , with samples collected for T ( i ) discrete and regular timesteps. Electronic health records consist of data for N independent patients D = { τ ( i ) } N i =1 . For simplicity, we omit the patient superscript ( i ) unless it is explicitly needed.

We leverage the potential outcomes framework proposed by Rubin (1978) and Neyman (1923), and extended by Robins &amp; Hern´ an (2008) to take into account time-varying treatments. Let Y (¯ a ) be the potential outcome, either factual or counterfactual, for each possible course of treatment ¯ a .

Let ¯ A t = ( A 1 , . . . , A t ) ∈ ¯ A t be the history of treatments and let ¯ X t = ( X 1 , . . . , X t ) ∈ ¯ X t be the history of covariates until timestep t . For each patient, we want to estimate individualized treatment effects, i.e. potential outcomes conditional on the patient history of covariates and treatments:

<!-- formula-not-decoded -->

for any possible treatment plan ¯ a ≥ t that starts at timestep t and consists of a sequence of treatments that ends just before the patient outcome Y is observed. The observational data can be used to fit a regression model to estimate

E [ Y | ¯ a ≥ t , ¯ A t -1 , ¯ X t ] . Under certain assumptions, these estimates are unbiased so that E [ Y (¯ a ≥ t ) | ¯ X t , ¯ A t -1 ] = E [ Y | ¯ a ≥ t , ¯ A t -1 , ¯ X t ] . These conditions include Assumptions 1 and 2, which are standard among the existing methods for estimating treatment effects over time and can be tested in practice (Robins &amp; Hern´ an, 2008).

Assumption 1. Consistency . If ¯ A ≥ t = ¯ a ≥ t , then the potential outcomes for following the treatment plan ¯ a ≥ t is the same as the observed (factual) outcome Y (¯ a ≥ t ) = Y .

glyph[negationslash]

Assumption 2. Positivity (Overlap) (Imai &amp; Van Dyk, 2004): If P ( ¯ A t -1 = ¯ a t -1 , ¯ X t = ¯ x t ) = 0 then P ( A t = a t | ¯ A t -1 = ¯ a t -1 , ¯ X t = ¯ x t ) &gt; 0 for all a t .

The positivity assumption means that at each timestep t , each treatment has a non-zero probability of being given to the patient. This assumption is testable in practice.

In addition to these two assumptions, existing methods also assume sequential strong ignorability :

<!-- formula-not-decoded -->

for all possible treatment plans ¯ a ≥ t and for all t ∈ { 0 , . . . , T } . This condition holds if there are no hidden confounders and it cannot be tested in practice. To understand why this is the case, note that the sequential strong ignorability assumption requires the conditional independence of the treatments with all of the potential outcomes, both factual and counterfactual, conditional on the patient history. Since the counterfactuals are never observed, it is not possible to test for this conditional independence.

In this paper, we assume that there are hidden confounders. Consequently, using standard methods for computing E [ Y | ¯ a ≥ t , ¯ A t -1 , ¯ X t ] from the dataset will result in biased estimates since the hidden confounders introduce a dependence between the treatments at each timestep and the potential outcomes ( Y (¯ a ≥ t ) glyph[negationslash]⊥ ⊥ A t | ¯ A t -1 , ¯ X t ) and therefore:

<!-- formula-not-decoded -->

glyph[negationslash]

By extending the method proposed by Wang &amp; Blei (2019a), we take advantage of the multiple treatment assignments at each timestep to infer a sequence of latent variables ¯ Z t = ( Z 1 , . . . , Z t ) ∈ ¯ Z t that can be used as substitutes for the unobserved confounders. We will then show how ¯ Z t can be used to estimate the treatment effects over time.

## 4. Time Series Deconfounder

The idea behind the Time Series Deconfounder is that multicause confounders introduce dependencies between the treatments. As treatment assignments change over time we infer substitutes for the hidden confounders that take advantage of the patient history to capture these dependencies.

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

Figure 1. (a) Graphical factor model. Each Z t is built as a function of the history, such that, with X t , it renders the assigned causes conditionally independent: p ( a t 1 , . . . , a tk | z t , x t ) = ∏ k j =1 p ( a tj | z t , x t ) . The variables can be connected to Y (¯ a ≥ t ) in any way. (b) Graphical model explanation for why this factor model construction ensures that Z t captures all of the multi-cause hidden confounders.

![Image](data:image/png;base64,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)

## 4.1. Factor Model

The Time Series Deconfounder builds a factor model to capture the distribution of the causes (treatments) over time. At time t , the factor model constructs the latent variable z t = g ( ¯ h t -1 ) , where ¯ h t -1 = (¯ a t -1 , ¯ x t -1 , ¯ z t -1 ) is the realization of history ¯ H t -1 . The latent variable z t , together with the observed patient covariates x t , render the assigned treatments conditionally independent:

<!-- formula-not-decoded -->

Figure 1(a) illustrates the corresponding graphical model for timestep t . The factor model of the assigned treatments is built as a latent variable model with joint distribution:

<!-- formula-not-decoded -->

where θ 1: k are parameters. The distribution of the treatments p (¯ a T ) is the corresponding marginal. Notice that we do not assume that, in the observational data, the patient covariates x t at timestep t are independent of the patient history. The graphical factor model shows how the latent variables z t that can act as substitutes for the multi-cause unobserved confounders are built. As we will see in Section 4.2 these latent variables will be used as part of an outcome model that estimates the potential outcomes Y (¯ a ≥ t ) .

By taking advantage of the dependencies between the multiple treatment assignments, the factor model allows us to infer the sequence of latent variables ¯ Z t that can be used to render the assigned causes conditionally independent. Through this factor model construction and under correct model specifications, we can rule out the existence of other multi-cause confounders that are not captured by Z t . To understand why this is the case, consider the graphical model in Figure 1(b). By contradiction, assume that there exists another multi-cause confounder V t not captured by Z t . Then, by d -separation the conditional independence between the assigned causes given Z t and X t does not hold anymore. This argument cannot be used for single-cause confounders, such as L t , which are only affecting one of the causes and the potential outcomes. Thus, we assume sequential single strong ignorability (no hidden single cause confounders).

Assumption 3. Sequential single strong ignorability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Causal inference relies on assumptions. Existing methods for estimating treatment effects over time assume that there are no multi-cause and no single-cause hidden confounders. In this paper, we make the weaker assumption that there are no single-cause hidden confounders. While this assumption is also untestable in practice, as the number of treatments increases for each timestep, it becomes increasingly weaker: the more treatments we observe, the more likely it becomes for a hidden confounder to affect multiple of the these treatments rather than a single one of them.

Theorem 1. If the distribution of the assigned causes p (¯ a T ) can be written as the factor model p ( θ 1: k , ¯ x T , ¯ z T , ¯ a T ) , we obtain sequential ignorable treatment assignment:

<!-- formula-not-decoded -->

Theorem 1 is proved by leveraging Assumption 3, the fact that the latent variables Z t are inferred without knowledge of the potential outcomes Y (¯ a ≥ t ) and the fact that the causes ( A t 1 , . . . , A tk ) are jointly independent given Z t and X t . The result means that, at each timestep, the variables

¯ X t , ¯ Z t , ¯ A t -1 contain all of the dependencies between the potential outcomes Y (¯ a ≥ t ) and the assigned causes A t . See Appendix A for the full proof.

glyph[negationslash]

As discussed in Wang &amp; Blei (2019a), the substitute confounders Z t also need to satisfy positivity (Assumption 2), i.e. if P ( ¯ A t -1 = ¯ a t -1 , ¯ Z t = ¯ z t , ¯ X t = ¯ x t ) = 0 then P ( A t = a t | ¯ A t -1 = ¯ a t -1 , ¯ Z t = ¯ z t , ¯ X t = ¯ x t ) &gt; 0 for all a t . After fitting the factor model, this can be tested (Robins &amp;Hern´ an, 2008). When positivity is limited, the outcome model estimates of treatment responses will also have high variance. In practice, positivity can be enforced by setting the dimensionality of Z t to be smaller than the number of treatments (Wang &amp; Blei, 2019a).

Predictive Checks over Time : The theory holds if the fitted factor model captures well the distribution of assigned treatments. This condition can be assessed by extending predictive model checking (Rubin, 1984) to the time-series setting. We compute p -values over time to evaluate how similar the distribution of the treatments learned by the factor model is with the distribution of the treatments in a validation set of patients. At each timestep t , for the patients in the validation set, we obtain M replicas of their treatment assignments { a ( i ) t, rep } i M =1 by sampling from the factor model. The replicated treatment assignments are compared with the actual treatment assignments, a t, val, using the test statistic T ( a t ) :

<!-- formula-not-decoded -->

which is related to the marginal log likelihood (Wang &amp; Blei, 2019a). The predictive p -value for timestep t is computed as follows:

<!-- formula-not-decoded -->

where 1 ( · ) represents the indicator function.

If the model captures well the distribution of the assigned causes, then the test statistics for the treatment replicas are similar to the test statistics for the treatments in the validation set, which makes 0 . 5 the ideal p -value in this case.

## 4.2. Outcome Model

If the factor model passes the predictive checks, the Time Series Deconfounder fits an outcome model (Robins et al., 2000a; Lim et al., 2018) to estimate individualized treatment effects over time. After sampling the sequence of latent variables ˆ ¯ Z t = ( ˆ Z 1 , . . . , ˆ Z t ) from the factor model, the outcome model can be used to estimate E [ Y | ¯ a ≥ t , ¯ A t -1 , ¯ X t , ˆ ¯ Z t ] = E [ Y (¯ a ≥ t ) | ¯ A t -1 , ¯ X t , ˆ ¯ Z t ] .

To compute uncertainty estimates of the potential outcomes, we can sample ˆ ¯ Z t repeatedly and then fit an outcome model for each sample to obtain multiple point estimates of Y (¯ a ≥ t ) . The variance of these point estimates will represent the uncertainty of the Time Series Deconfounder.

DAmour (2019) raised some concerns about identifiability of the mean potential outcomes using the deconfounder framework in Wang &amp; Blei (2019a) in the static setting and illustrated some pathological examples where identifiability might not hold. 3 In practical settings, the outcome estimates from the Time Series Deconfounder are identifiable, as supported by the experimental results in Sections 6 and 7. Nevertheless, when identifiability represents an issue, the uncertainty in the potential outcomes can be used to assess the reliability of the Time Series Deconfounder. In particular, the variance in the potential outcomes indicates how the finite observational data inform the estimation of substitutes for the hidden confounders and subsequently the treatment outcomes of interest. When the treatment effects are nonidentifiable, the estimates of the Time Series Deconfounder will have high variance.

By using this framework to estimate substitutes for the hidden confounders we are trading off confounding bias for estimation variance (Wang &amp; Blei, 2019a). The treatment effects computed without accounting for the hidden confounders will inevitably be biased. Alternatively, using the latent variables from the factor model will result in unbiased, but higher variance estimates of treatment effects.

## 5. Factor Model over Time in Practice

Since we are dealing with time-varying treatments, we cannot use standard factor models, such as PCA (Tipping &amp; Bishop, 1999) or Deep Exponential Families (Ranganath et al., 2015), as they can only be applied in the static setting. Using the theory developed for the factor model over time we introduce a practical implementation based on a recurrent neural network (RNN) with multitask output and variational dropout as illustrated in Figure 2.

The recurrent part of the model infers the latent variables Z t such that they depend on the patient history:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where L consists of randomly initialized parameters that are trained with the rest of the parameters in the RNN.

The size of the RNN output is D Z and this specifies the size of the latent variables that are inferred as substitutes for the hidden confounders. In our experiments, we use an LSTM unit (Hochreiter &amp; Schmidhuber, 1997) as part of the RNN. Moreover, to infer the assigned treatments at

3 See Wang &amp; Blei (2019a) for a longer discussion addressing the concerns in (DAmour, 2019).

-1

t-1

FC Layers

RNN

RNN

RNN

FC Layers

FC Layers

Ok

FC Layers

FC Layers

FC Layers

Figure 2. (a) Proposed factor model implementation. Z t is generated by RNN as a function of the history ¯ H t -1 , given by the hidden state h t , and current input. Multitask output is used to construct the treatments such that they are independent given Z t and X t . (b) Closer look at a single timestep.

![Image](data:image/png;base64,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)

timestep t , A t = [ A t 1 , . . . , A tk ] such that they are conditionally independent given the latent variables Z t and the observed covariates X t , we propose using multitask multilayer perceptrons (MLPs) consisting of fully connected (FC) layers:

<!-- formula-not-decoded -->

for all j = 1 , . . . k and for all t = 1 , . . . T , where θ j are the parameters in the FC layers used to obtain A tj . We use a single FC hidden layer before the output layer. For binary treatments, the sigmoid activation is used in the output layer. For continuous treatments, MC dropout (Gal &amp; Ghahramani, 2016a) can instead be applied in the FC layers to obtain p ( A tj | X j , Z j ) .

To model the probabilistic nature of factor models we incorporate variational dropout (Gal &amp; Ghahramani, 2016b) in the RNN as illustrated in Figure 2. Using dropout enables us to obtain samples from Z t and treatment assignments A tj . These samples allow us to obtain treatment replicas and to compute predictive checks over time, but also to estimate the uncertainty in Z t and potential outcomes.

Using the treatment assignments from the observational dataset, the factor model can be trained using gradient descent based methods. The proposed factor model architecture follows from the theory developed in Section 4 where at each timestep the latent variable Z t is built as a function of the history (parametrized by an RNN). The multitask output is essential for modeling the conditional independence between the assigned treatments given the latent variables generated by the RNN and the observed covariates. The factor model can be extended to allow for irregularly sampled data by using a PhasedLSTM (Neil et al., 2016).

Note that our theory does not put restrictions on the factor model that can be used. Alternative factor models over time are generalized dynamic-factor model (Forni et al., 2000; 2005) or factor-augmented vector autoregressive models (Bernanke et al., 2005). These come from the econometrics literature and explicitly model the dynamics in the data. The use of RNNs in the factor model enables us to learn complex relationships between ¯ X t , ¯ Z t , and ¯ A t from the data, which is needed in medical applications involving complex diseases. Nevertheless, predictive checks should be used to assess any selected factor model.

## 6. Experiments on Synthetic Data

To validate the theory developed in this paper, we perform experiments on synthetic data where we vary the effect of hidden confounding. It is not possible to validate the method on real datasets since the true extent of hidden confounding is never known (Wang &amp; Blei, 2019a; Louizos et al., 2017).

## 6.1. Simulated Dataset

To keep the simulation process general, we propose building a dataset using p -order autoregressive processes. At each timestep t , we simulate k time-varying covariates X t,k representing single cause confounders and a multi-cause hidden confounder Z t as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for j = 1 , . . . , k , α i,k , λ i,j ∼ N (0 , 0 . 5 2 ) , ω i,k , β i ∼ N (1 -( i/p ) , (1 /p ) 2 ) , and η t , glyph[epsilon1] t ∼ N (0 , 0 . 01 2 ) . The value of Z t changes over time and is affected by the treatment

RNN

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

assignments.

Each treatment assignment A t,j depends on the single-cause confounder X t,j and multi-cause hidden confounder Z t :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ˆ X tj and ˆ Z t are the sum of the covariates and confounders respectively over the last p timesteps, λ = 15 , σ ( · ) is the sigmoid function and γ A controls the amount of hidden confounding applied to the treatment assignments. The outcome is also obtained as a function of the covariates and the hidden confounder:

<!-- formula-not-decoded -->

where γ Y controls the amount of hidden confounding applied to the outcome. We simulate datasets consisting of 5000 patients, with trajectories between 20 and 30 timesteps, and k = 3 covariates and treatments. To induce time dependencies we set p = 5 . Each dataset undergoes a 80/10/10 split for training, validation and testing respectively. Hyperparameter optimization is performed for each trained factor model as explained in Appendix B. Using the training observational dataset, we fit the Time Series Deconfounder to perform one-step ahead estimation of treatment responses.

## 6.2. Evaluating Factor Model using Predictive Checks

Our theory for using the inferred latent variables as substitutes for the hidden confounders and obtain unbiased treatment responses relies on the fact that the factor model captures well the distribution of the assigned causes.

Figure 3. Predictive checks over time. We show the mean p -values at each timestep and the std error.

![Image](data:image/png;base64,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)

To assess the suitability of our proposed factor model architecture, we compare it with the following baselines: RNN without multitask output (predicting the k treatment assignments by passing X t and Z t through a FC layer and output layer with k neurons) and multilayer perceptron (MLP) used instead of the RNN at each timestep to generate Z t . The MLP factor model does not use the entire history for generating Z t . See Appendix C for details.

Figure 3 shows the p -values over time computed for the test set in 30 simulated datasets with γ A = γ Y = 0 . 5 . The p -values for the MLP factor model decrease over time, which means that there is a consistent distribution mismatch between the treatment assignments learned by this model and the ones in the test set. Conversely, the predictive checks for our proposed factor model are closer to the ideal p -value of 0.5. This illustrates that having an architecture capable of capturing time-dependencies and accumulating past information for inferring the latent confounders is crucial. Moreover, the performance for the RNN without multitask is similar to our model, which indicates that the factor model constraint does not affect the performance in capturing the distribution of the causes.

## 6.3. Deconfounding the Estimation of Treatment Responses over Time

We evaluate how well the Time Series Deconfounder 4 can remove hidden confounding bias when used in conjunction with the following outcome models:

Standard Marginal Structural Models (MSMs) . MSMs (Robins et al., 2000a; Hern´ an et al., 2001) have been widely used in epidemiology to estimate treatment effects over time. MSMs use inverse probability of treatment weighting (IPTW) to adjust for the time-dependent confounding bias present in observational datasets (Mansournia et al., 2017; Bica et al., 2020b). MSMs compute the propensity weights by using logistic regression; through IPTW, these models construct a pseudo-population from the observational data where the treatment assignment probability no longer depends on the time-varying confounders. The treatment responses over time are computed using linear regression. For full implementation details in Appendix D.1.

Recurrent Marginal Structural Networks (R-MSNs) . RMSNs (Lim et al., 2018) also use IPTW to remove the bias from time-dependent confounders when estimating treatment effects over time. However, R-MSNs estimate the propensity scores using RNNs instead. The use of RNNs is more robust to changes in the treatment assignment policy. To estimate the treatment responses over time R-MSNs also use a model based on RNNs. For implementation details, see Appendix D.2.

4 The implementation of the Time Series Deconfounder can be found at https://bitbucket.org/mvdschaar/ mlforhealthlabpub/src/master/alg/time\_ series\_deconfounder/ and at https://github. com/ioanabica/Time-Series-Deconfounder .

Figure 4. Results for deconfounding the one-step ahead estimation of treatment responses in two outcome models: (a) Marginal Structural Models (MSM) and (b) Recurrent Marginal Structural Networks (R-MSN). The average RMSE and the standard error in the results are computed for 30 dataset simulations for each different degree of confounding, as measured by γ .

![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA5EAAAEYCAIAAABKkOXRAAEAAElEQVR4nOzdB3RUVfoA8Pvem94zSSaT3nvvkNBC7yAIKthQ7GV1/6vuIjYsu7prW10LCipVAQHpvXfSCElIJb1P7zOv/c/MwxhDCyEJAe7veI5hMnnvZZLc+d693/0+hKZpAEEQBEEQBEGDGHqrLwDqFw0NDZWVlVf8lEqlKikpgfcqEATdkcrKypqbm6/4KYvFUlhYaLfbB/yiICf4vgPdJBiz3oGam5s/+eQTk8l0xc+yWKzVq1cfOHDgip8lSXLbtm1r1qwxGo1dH6+pqVm2bFlubm6vr8pgMOh0uhv6kmPHjl3xjFardfPmzX/9619feumlH3/88WrvT7dKTU3Nzp07L39fbGhoWLZs2alTp7o+SJLknj17Vq9erdFoenLw48ePnz179obOC0F3j7Nnz3733XcIgtA0feDAgeLi4q6f5XA4hw4dWr169dWCp5MnT3799ddffvnl//73v/Xr11/tzn9QOXny5PLlyxsbG7s+qNPpVq1atXv3boqiendYq9WqUqlu6MtLS0v37t17+ZdQFHXs2LFFixa98MILn3/+eXl5ORhMrjZyMq/hxo0bHQ4H80hHR8emTZt6OFb3ml6v37Jli1qt7tez3KZgzHqnoShq9erVoaGhSUlJV3yCTCabMmXKhg0brvgngeP48uXL33jjjW6B0fr16xcvXrxv375eX9iaNWu+/vprkiR7/iVbtmzZu3dvtwdJkvzqq6/WrFmTmZk5YsSIky5gMCkpKfnpp5/MZnO3x8vLy5csWfLZZ59ZrdbOB5uamv71r3/95z//aWlp6cnBt27dunv37hs6LwTdJQwGw6pVq8aPH+/t7U0QxC+//HL8+PGuT2CxWLNnzz558mRJSckVj/Dbb7/t3r1bLpfLZLKysrLFixf/9ttvYHDbsWPH66+/3u06jx07tnjx4lWrVvV6avPo0aPvv//+DU00nD59eu3atZefccuWLR9++GFQUNCkSZMaGxu3bds2qCZcrzZydnR0fPrpp2+++Wbn1EljY+PXX3/d2trar9ejUqm+/fbbwTYXM0iwbvUFQH2svLw8Ly/vn//8J4IgV3tOenr6unXr9uzZ88ADD3T7FE3TIpEoODj46NGjI0eOxDAMANDe3l5SUhITE8P8s3eGDh1qs9mucVWXY7t0e7CmpubQoUNvvfVWWloaAGDatGk4jjOzAqdOnUpISHB3d+/1RTY0NNTW1mZnZ6No72/nMAzjcDhX/E5DQkIMBsP58+czMjKYR86ePcvn86/2/B6+Jtc9LwTdDY4ePWo2m0eMGMH8k8PhsFjd3+P8/f3j4uI2bNgQHR19+YCGomhsbOy8efOYf+7cufOrr76Kj48PCQkBt45Op8vPz8/MzBQKhZd/FsOwmJiY/Px8nU4nk8kAAARBnDhxIiIiQiAQ9PqkERERbDb7ho7AYrE4HE63B00m0/r16+fNm8e83UyePJmJDmmaPnnypLe3d3BwcD+9Mj10tZGTpmlfX18+n//bb7+lp6ez2WwURblc7tXeHSorKzUaTWZmJrg5CIJc4yx3Ofii3GnOnj3r7u7u5+fX9UEmquvE4XCSkpJOnz7dueTRFU3T6enplZWVnfd5586dwzAsNDS066IPjuNXvFfufLBzSpX5IDExMTMzs+vf4eVLSFc7Zld2l86ndY6qOp3u888/r6mpATeIIIjOS83Pz//mm2+6vVzXRZIkQRDXfRpFUZ6ensHBwceOHes89alTp5KTk3k8Xtdv/IpHY57Q7V2WoqirvWI4jt/QrDYE3dZomj5x4kRsbCyfz+98kIlZu/2NZGRkVFRUdHR0XPE4XcelqKgoh8PRuRZM03TvRq3OL7/iP6/49971RI2NjZ999plKpbraBcfExFit1s7J49ra2qamptTU1K7jQ9eB7oqX0XlG5quCgoJycnJ4PN4VL+nax+yKJEmLxdJ5FgRBRCIRgiAEQXz//ffdcqW6XdLVvt/OIfrar8zV0DTdk0GepmkURadNm1ZeXl5UVHTF53T92R08eHDlypU9H3V7+MzOHyJFUVd8y76rwHnWOwpN02VlZcHBwZ1TcRcuXNiwYYNKpVIqlfPnzw8ICGAej4iI2LVrl1qt9vb27nYQiqIiIyPVavXZs2f9/f1pmj5y5EhCQkJDQwPzl9PW1rZ27dq6ujqBQDB79uyUlJT29vYNGzbEx8cfO3ZsxIgRaWlpW7duzcvL8/f3F4vFjY2NDz/8cHl5ucFgmDlzZnV19bZt23x9fU+fPs3j8RYsWBASEtLa2vrzzz93PebVvseQkJDo6Oi33nprwYIFcXFxkZGRKIq2tLQsXbq0vb192bJlJ06ceOihhyorK8vKyhQKxdmzZx977LFjx475+PgMHz5cr9f/8ssvY8aMCQ0NNZlMGzZsOHXqFIqiOTk5CoVi8+bNTU1Nb7/9dmpq6r333rt//36dTjd79mySJNevX+/n5zds2LBNmzZhGGY2mzs6OhYsWLB3794TJ04QBDFixIipU6dePs3QFYZhWVlZu3fv7ujo8PT0vHjxYnNz8+zZszuz7lQq1S+//FJZWenh4TFr1qyYmBhmCvnXX3/Nzc2NiIhobGyMjIxkxrudO3cePHiQw+Hcc889nRO3KIpqNJrVq1czSWOTJ0+eNGkSnHyF7nhms7m2tjYrK6vzERRFy8vLv/rqqwsXLmRnZ8+aNYv58wwKCnI4HPX19UqlsttBEASx2Wx6vZ6mabVa/f333wcFBYWGhtI0ffjw4e3bt9M0PW3atBEjRiAIUltbu3LlyoaGBqlUOnfu3MjIyJ9//nn06NFhYWG1tbW7du2aN28en8//6aefQkNDi4uLfXx8kpOT9+zZExsbe+DAgVmzZoWGhv7yyy8FBQVeXl7z588PCgratWtXY2MjiqJ5eXnx8fGPPPJIe3v78uXL1Wr1Z599FhER8cgjj3Sb+6Rp2sPDw9PT8/Dhw9nZ2UyGq4eHh7+/f3t7O4qiZrN5/fr158+fBwCMGzduwoQJOI6vWLGCuSovL69777336NGje/fulUqlAQEB1dXVEydOFAgEp06dmjdvXmNj49atW7uN2N2OOX78+KvNC0ql0jFjxnz11VcmkyktLS0mJobH45nN5pUrV1ZXVzscjpqamvvvvx/DsJ07dzKvzOzZszs6OjQaTbexF8fxvXv37tq1y+FwxMXFZWVlrVmzpvOVYcb5y0fskydPVlZWMu8FTz75ZGVl5fbt281mc0xMzAMPPCCVSq/xS0WSZHx8fFlZ2ebNm1NSUroOpK2tratXr66rq4uKipo3b15+fv7evXv1ev1bb72Vmpqq1+sTExOTk5Obm5vXr19/zz33BAQEVFVV7d+/f968eRcuXPjtt98sFktKSso999wjFArXrFkjl8uZrINhw4Yxv70EQaxZs4bH482aNSs3N3fz5s1Go9HDw+Opp57y8fEBdyU4z3pHIQiivb1doVB0PtLc3BwTEzN//nytVvv555933th5eHjgOH7FXCWKokQiUXp6+vHjx2mabmtrq6ysHDJkSOeQpFar3d3dH3jgAaVS+dFHH2k0GpvNtm7dun/9618CgSAoKGjr1q07d+6cPHlyeXn53r17R48eLZVK8/LyTpw4wYRlq1evLi4unjhxYnV19bJlywAAGo2m6zGvcd/M5/Nff/31rKysTz/99IknnnjjjTfa29uFQmF8fLxAIEhKSkpNTeXxeDU1NV988cXatWtjY2NFItGBAweY4dVkMm3fvp2ZQv76668PHz5833333XPPPRaLxcPDIywsTCKRDBs2jIkL8/Lyjh07xkyuHDx4kJnGOHfu3Pvvv3/ixInk5GQcxy0Wy4wZMyZNmvTjjz92TqBeDUmSycnJFEUxQerJkye9vLwCAwNJkkRR1Gq1fvTRRw0NDbNnz3Zzc3vvvfeYaeO1a9euWLFiyJAhPj4+zPsZs2q5Zs2ayZMnJyUlffXVV21tbczjKIpu2LDh/PnzCxcunDVrFpfL7e1vEwTdTgwGg8Vi6UwNQhCEoqiLFy+GhIQkJSV9/fXXncnxYrGYw+G0t7dffhAWi3XixIkXX3zxmWeeef755wEAixcvdnNzO378+Ndffz1y5MjRo0cvW7asvLxcpVK98847CII88sgjaWlpGo3GaDRu3769qakJANDS0rJ161aTyUTT9P79+995552Ojo6YmBiNRrNq1apPP/3Uy8vL29t7+fLlp0+ffvDBBwUCwRdffGG320tLS5cvX87hcEaOHLlu3bqjR49KJJK4uDg+n5+enp6cnHx5ahBN0ywWKzs7u7i4WKfTMas36enpnas3FouFoqjZs2dnZWV99dVXRUVFCIJ0XlVcXFxBQcG33347bNgwFou1YsWK1NRUPz+/mpqa3bt34ziuVqsvH7HNZnO3Y17jR7PA5ZdffnnmmWdeeOGF0tJSDoeTkJAglUrDw8OHDBkilUpVKtXKlSuZV0apVObm5l4+9m7btu37778fM2bM/PnzeTweiqIJCQmdrwyGYVccsevq6pj3gpiYGIFA0NbWNnLkyHnz5h0/fnzt2rXX/qWiaZrD4cyYMaOgoKC2tpZZ5kJR1GKx/Pvf/zaZTA899FBNTc3KlSsDXTw8PIYPHx4ZGVlSUrJt2zYAQEFBwYoVK5jp5EOHDhUWFubn53/yySfx8fEzZ848evTot99+y/zI3n///ZKSksTERAzDmOB4xYoVv/76a3R0tNFo/Pzzz6OiohYuXJiYmNiTZb07FZxnvaMQBGG1WkUiUecjY8aMYT5g5iBxHGf+6gQCAUVRNpvtisdBUTQ7O3vXrl1NTU3nzp2TSCQxMTHr1q1jPhvjAgAQCoW7d++uq6tTKBQ0Tc+fP5/JA8vNzU1LSxs+fDiO42vXrk1OTuZwOCiKduaWyWSyefPmRURENDU1HT58mCTJy4/p4eFxtW9ToVAsXrz45ZdfPnXq1L///e+vv/76rbfeGj58+Nq1a4cNGxYbG8uMNUKh8I033ggLC2O+a+YbRxCEzWazWKy6urqjR48y98SdR7548WJ1dfWUKVOYf2IY1nnNLBaLOQKzavbPf/6TeZ0ffPBBAIDNZtu8eXNRUdHo0aOv8QOiaVqhUMTHxx89enTo0KFnzpzJyckRiUQURbFYrPz8/Nra2m+//dbNzW348OH19fWbN29+8sknd+/evWDBgvvvvx8AcOLECeYHt3nzZnd3dxzH+Xy+VqvNy8tj5pBomiZJ0mg02u32pKQkuVze418fCLqN2Ww2iqI6EwOYcG3kyJETJ05k9tPs27dvwoQJTM4li8XquhWyE0EQ6enpf//73xsaGt59990JEyYwa02bNm0SCATMCOBwOE6fPs3n80mSfOWVVzpvC5ubm1ksVuetI5vNZiIPkiSHDx++ZMkSZmwEADz11FMTJkxobGzcuXPnyJEjtVqtu7v73r176+vrEQRhEmpRFD148GBdXd348eOzsrJ27NgxZswYLy+vK37vFEUlJyejKFpQUODv769Wq4cMGcIUh2FSkh577DFmumHz5s3FxcXx8fFdr+qHH37w8vKaOHFiXFzc6dOno6KimImPzvj48hFboVB0O+bVdv0yo/rjjz/+4IMPlpaWfvbZZ//85z+/+eabrKwsLy+vxMREZsysqalBUZR5ZZgXsOvYy2KxcBz/9ddf58yZM2PGDADA8OHDAQA8Hm/Lli2dr8wVR2wEQTrfCwAAs2fPZp5w8uTJc+fOMU+47kRDUFDQpk2bxo0bhyAIi8UqKirKz89/9NFHNRqNQqE4evToM888w7z1MNc/bNiw9evXq9XqsrKytLS04uLiadOmFRYWpqWlbd++fdiwYcx47uHh8eabbzJLYYmJie+//z6Xy62traUo6ocffigsLPzggw/i4+NVKpXdbtfpdHw+/5577rmb181gzHqHa2xsXLduXVNTU0dHB0VRPfxdJ0kyPDxcoVAcO3asvLw8LS1NIpF0JjMZDIYNGzaUlpaazWamohZFUW5ubgkJCcwTwsLCzp49e/HixWPHjvn6+l6+ZiQQCJiAj7keBEGMRuP69eu7HvO6hELhmDFjLBYLE4vjON413YeiqNDQ0KCgoG7ZUcjv2tra2Gx2t9QI5iB2u/3y6cnOl46m6aSkJOb6SZI8dOjQ3r177XZ7RUVFVFTUta+ZpmkMw4YNG/btt9+ePXtWr9cPHTrUYDAwn2psbBSJRGKxmHkyM6uq0+kcDkfnNgXUxWw2G41GNze3srIyiqLGjx8fHh5+8eJF5jhz5sxpa2v7+9//LhKJnn/+eWYMhaA73uV5kJ1/tgEBAZWVlXa7vTOmuVo6vkgk8nWZOHHiqlWrMjMzWSyWWq0WCoUVFRUkSWZlZSUkJOzdu9fHx+dq6xhdhwsOh9N5Y0zTtKenZ3x8PLPiRJIkjuOlpaUAgFmzZjF3mDKZDEVRHMdRFGWO43A4mHHpat84SZJyuZyJOBsaGgICAvz9/ZnZOCZzdPv27SdOnLDZbA0NDUwhsK5XFRISsnPnzvPnz587d04sFkskkm6vSbcRmwnuux3zuj8dLpebnJz82muv/d///V9lZWVCQkLXzFQmw4F5ZS5/MZn3CJPJ1G3D1jVemc5LIkmy872AWSj77bffdDpdTU3N5ckhl2MmFKZNm/bdd9+Fh4ezWCyaptvb2/l8vsaFxWLdc889nVsISJLEMCwxMXHFihX5+fmtra2TJ0/evn17fn6+Wq2Ojo4+fPgws/oPAPD09MQwjLnbiYuLY36dUBRVqVSNjY0ymYx5N/Tw8HjxxRe/+eabjRs3pqSkLFq0qOtq6l0Fxqx3FBaLxaQKMf/UarXvvPNOXFzcokWLCgoKvvnmm86402azMVsgr3gciqLYbHZ2dvaWLVswDLvvvvu6Lrd9+umnBoPhL3/5i81me+2115h8g87hlVm+Jwjis88+CwgIePzxxy/futv1DQNBEBzHP/nkk8uPeUUmkwnHcTc3N+afKIryeDzm1ExE2PlMDMO6vi0xoTNFUQRBMPOdOI63tbVdnhjU9V2tM8btHFuZ+2zm4z179ixbtuyVV16JjY1dvHhxT2oZUhSVkJDA4/F++umnkJAQX19frVbLfMrLy4sZBJl567q6Oh8fH6lUyuFwOithkSTJTCHL5fKUlJSulR+qqqqY43t4eLz99ttWq3X16tU//vhjWlrazdRSgKDbAo/HwzDsamtH9fX1UqmUGfEcDgdJklfbEd85aMydO/fIkSPbtm2bM2eOUqn08fF58cUXO59WXFxcUlLS9f626y5JgiA6RwPmPrPzCztXfuVyuVQqnTx5cnJycudnma+6PJ7uNrhdUU5OzqefflpcXPzAAw90jn4Yhq1Zs2bHjh2vvvqqn5/fK6+80jlid14VhmESiWTp0qVSqfTVV1/tHF2v+LIgCHK1Y16Rw+HQ6XSdMRYzA925KHTFV+bysZcZ9EQiUX19/ZAhQ7peVddX5oojdtf3ggsXLixZsuTBBx8cO3bsqlWrmHnWnhg+fPi6deuYOoMIgigUCrFY/Oijj3Z9rZifHXMxfn5+gYGB27dvF4lEI0aMOHTo0ObNm93c3GJiYuRyOTNWMzkkNpvNy8ur63dBURSPx1u0aJFarf7f//4XFRWlVCpHudTW1i5evHjLli0LFy4EdyUYs95RWCyWQqHozNMiCIJJb6qrq2O2E3WOCCqVisViXTH93G63MzfoWVlZX3/9dUJCQlhYGE3TDoeDGQX0ej1BECqV6tSpU01NTcxdu91u7xyjmZIrcXFxzAYsiUSCYRiO48wkKHNbzIwgBEE4HA6apg0GA47jXY/JDHaX7+4sLS398ssvx4wZk5yc3NTUtHz58rFjxzLrfTiO79+/v76+Pj09nbngrqVMylx2795dU1ND03RAQEBmZuZnn302b948h8NRVVX10EMPSaXS5ubmLVu28Pn8CRMmCASC2tra8+fPl5eXFxQUMNW1mDld5shms9lisdjtdmaZiSnXQJJk18oGnZjHCYJwd3ePjY1dvnz5v/71L+YFsdlsBEFkZGT4+vp+8MEHnelTTz31lEgkGjZs2A8//MCsRZ44cWLatGk8Hm/69OlLly5ls9kBAQGnTp0aO3Ys87Njsr6am5uTk5P1ej2fz7+ZCmUQdLuQSCQCgaCz7DSCIHa7/eDBgwkJCRUVFfv373/llVeYu02TyWS32z09PS8/CDNvx3zMbE5as2ZNdnb2Pffc8+GHH7q5uUVHR589ezYpKWns2LF79uz58MMPc3JyqqqquFwuk4hVWFgolUo3bNig0WiYccxut3eGdF1HP19f32HDhn3yySePPPIIm80uKCh48MEHSZLsHLgcDgczFPP5fKvVumvXLg8PjxEjRnSLKZkhFAAQFxfHZEGkp6czoyszIBiNRpvNZrFYDh8+XFZWNnLkyG5XZTAYUBRNTEwUCoVqtdrPz48ZbZgv7zZiX/GYTH0xZjzv9pKqVKo33ngjNjZ2xIgRdrv9xx9/jIiIYJbpuVzuiRMnOBwOs3+g67B5+djL5XKnTZu2du1aFoslkUjOnTs3adIksVjc+cqMHj1aJBKdOXOm24jd9arsdrvRaCQIoqSk5OjRo8z9xtVGbGZkZn4fBALBlClT3nrrLalUShBEYmKir6/vkiVL5s6dq1arGxsbFy5cKJFIqqqqtmzZIhaLc3JykpKSPv300+eee87NzS0kJGTp0qULFy6UyWQzZ878+OOP3dzcfH1916xZk5GRERERYbVaO7NUCYLg8XihoaE5OTknTpz45ptvnnzyyV9++SU6OlooFFIUdTMlzG53MGa9oyAIEhkZmZubSxAEi8Xy9PRcuHDhunXr2tra/P39J06c2Bm+VFVVeXl5XZ4zimHYiBEjmPWXwMDA1157zdfXl1kNSU9PZxatFixY8M033/zwww8RERHTp0+Xy+VCoXDUqFFMaUAcxw0GQ3l5ucVicTgcS5cuHTVqFDMTyfTW8vDwyMnJYf7qQkNDbTYbsxf166+/7npMAEBKSsrlVfeSkpKmTZt28ODBHTt2sFisKVOmzJ8/HwDg7u4+e/bso0ePAgCGDBkSHBw8dOhQ5iYew7A5c+asWLFi2bJlycnJ8+bNc3d3R1H0pZde+uWXXzZv3owgyLBhwyQSSVJS0pAhQ/bt28cM6xMnTqysrPz2229TU1PnzZsXGhrKFO3qrP8yZsyYioqKFStWhIeHT5w4kUnJZd6KLp/D9vHxGTVqFPO1M2fOVCqVQ4cOZZYCx4wZIxaLBQLB22+/vWLFinXr1nl4eLzzzjvMUP7YY4+hKLpx48bY2Nj77ruPWeSaNGkShmG7d+8+ePBgTEyMj48PjuPMeX18fE6dOpWbmysSiV588UXm5wJBdzaBQMAkADD/RBAkKyvLaDQeOHCgqanp2WefHT9+PPOphoYGNpvt7+9/+UFSUlK6xi7Tpk2rq6urqanJzs5+5ZVXNm3adPLkybCwMH9/f29v7yVLlqx0YQIRhUJx33337d69u729PS0tjckcQFF02LBhnedyd3cfNWoUk3SLYdhTTz3l5ub222+/oSg6dOhQqVQaExPDJCyhKJqZmckMxQEBAdOmTTt16lRSUtLlqVbM9iMmXer55583m83MpGZERAQzoTBz5sza2trly5dHRUVNnjw5JCSEGfGYq6Jp2mg0drbRWr16tZ+f30cffcRsumexWJeP2Fc8JvNZHMe75QkolcqHHnpo69atH3/8MUVRSUlJCxcuZOZZ58yZs3bt2tzc3MjISA8Pj85X5mpj75w5cwQCAZOLFRcXp1AopFJp5yuDIMiECRPKy8u7fVVISIjVamVet9jY2Pvvv3/Hjh2BgYHp6elMItbVRmym4kHnHcLo0aNLSkpsNptIJOLz+YsXL16xYsXKlStFItH48eMxDBs1alRubu6+ffuYFOphw4ZVVlYyebcjRoyoq6tj7mpGjRqFYdhvv/2Wm5s7YsSI+++/n8vlpqend6Y9iMXikSNHCgQCHo/HvEkZDAZPT8+dO3daLJasrCwmo/fu5PyFvtXXAPWl4uLi995778MPPwwMDOy6XNL1OTiO//3vf4+Pj3/00Uf7/AJqampefvnld955JzExEQCwb9++Tz/99Lvvvuvz2hxqtZrH4/W8lPTlr0NfoShqENZ/HpxXBUH9Z/PmzTt27Pj888+7lmi93FdffdXc3PzOO+/0yRJEt4Fl0P7dXW0AtFgsTz/99LRp0+bMmcMUxv/rX//66quvMsFW7455RVqtFkGQwXALPWh/RnfwlfcVOM96p4mKikpISNi1a9dTTz3FPHL5mHLu3Dmj0Thp0qT+uAB3d/fg4OBffvlFrVbTNL137974+Pj+GKduNEez//ZaDs5BZHBeFQT1n1GjRu3bt+/kyZPXKN/R0tJSWFj41FNP9VXOTLeBZdD+3V1tAORyuYmJifv372emD8+ePatQKHrYm+qGBtUrpsneEoP2Z3QHX3lfgfOsd6DGxsaPP/544cKFTOmNbkwm00cffZSVlcWsX/QHjUazZ8+e6upqBEEiIiLGjRt37brNEARBfeLkyZObNm169dVXr1gsj6bpb775BkGQp59++lZc3SBls9mY0qEEQfj6+o4bN65bJ0UIuktjVoIgjEZj50ZvqD8gCFJRUYHjOJNe2U17e3tdXV1qamr/3bExxVCZtHcOh9N1Cy0E3TGYxpUSiWQQjmY2m81qtV57ifyOhCBIfn6+u7t7Z8+/rsxmc1FRUWJi4t28i+VyTNVqptIfh8OhafpurloPDTYOhwNF0UvV1gY4Zs3Pz3///fdFIhGc4u5XzLLXFeuPMJVKmJJJ/XoNnfWn+vUsEHRLMNvSeTzeV199NQibje3cufObb765OztKsFgsyuUWjn63IzhiQ4MQgiAmkykhIWHRokXOThMDfHqmVNszzzzTufMagiDotoNhWFFR0datWwfne7xGo+HxeI8//vitvhAIgqDeQ1F027ZtKpWKmegc6JiVzWZ7eHiMHDny8qbJEARBtxEWi3XkyBEwKGEY5u/v39luB4Ig6DZVVVVVXl7OfHwLFuhhrgwEQXeAQb7EDJPIIQi6A3TNcoRJpRAEQYOdTqe7PD62WCxarRZOAUAQdJeAMSsEQdCgduTIke+//75bzJqXl7d169b8/Py1a9dqtdpbd3UQBEEDBMasEARBg5fBYMjLyzMYDF0fNBqNO3bsGDJkyJgxY7hc7q5du27dBUIQBA0QGLNCEAQNUhRFVVZW+vv7dyun1dDQoNFomMZCSqXy/PnzOI7fusuEIAgaCDBmhSAIGqRqamqEQqFSqexWa9lgMJAkyRR/YbPZJpOJaeEBQRB0BxvoWlcQBEGDB2W1mo4eJTUagGGizEz2lZon3SoGg0Gj0aSnp584cQJ16fwURVE0TXf236Jdbt2VQhAEXQuhUpmOHKFxAmGzJBMmoEIhuCUxK0VRWq324MGDoaGhycnJAACtVpubm6vRaCwWy4wZM+7ORiwQBN0WKKOx49NPrYXnEC7X/7//HVQxa1FRUWVlZUdHR2lpaXV19dGjR7OyspgWd0LXiM9MvhIEwePxOBzOrb5eCIKgK3NUVzcvWkQZTZhYLMzM7HXMerO5AXq9vqSk5NSpU7W1tQAAHMcPHDgQHBw8depUq9X6xRdfwCwrCIIGLczDg5+ahvJ4HD9ffnoaGEwyMzPvu+++0aNHR0ZG+vj4ZGZm0jRdXV1tNpv9/f1FIpFerwcAtLW1hYWFwZgVgqBBixsZyY9PQFBUmJXF8vDo9XFuNmZ1c3MbMWKEQqFg1q1wHD9x4oRKpRIKhenp6RUVFTqd7iZPAUEQ1E8QFMVkMuc6O4/HGmSLQmw2WyAQtLa2njt3rrm5OT8/32AwLF++/OLFi3K5fPjw4fn5+XV1dR0dHZMmTbrVFwtBEHRVqESKSSQ0RWHu7sifd5TegnzWzlQqgUCwaNEiZt2qvr5eoVCIRKI+OQUEQVC/YIYvmr70wSDj6+v78ssvM+1YuVzu4sWLmSnVyZMnt7S0qFSqefPmSaXSW32ZEARBV9c5wDIf/J6Lf+v3YLm7uwMAOjo6ioqKHn74YT6f3+engCAIukuwXTr/2XVE9Xa5RdcFQRB0R9S6slgse/funT59OrMrC4IgCIIgCIJufcyKuTAfkyR55syZVJfq6mqj0dgnp4AgCIIgCILuWjebG2Cz2Wpqaqqrq2majo+P9/HxWb169cmTJwMCAqxWq0wme/bZZ/voUiEIgiAIgqC7VN/ks86dO7ez0kpkZKRcLmdqXAcFBQkEgj45BQRBEARBEHTbQTC0Tza53mzMyuPxol06Hxk6dOhNXxUEQRAEQRB027NXVOi377CWlCBdtpP2DuzdCkEQBEEQBPU93fr17R9/gjc3o1wu0qUB9SCqGwBBEARBEATdhWhXW2kG5u5OdHSwPD1RqfTm0wNgzApBEARBEATdLEKtVi9b3vzaa0RrK/OIMCPD+733/L7+mh8TQzscvewl8DuYGwBBEARBEAT1nrWk1Lhnt37TZrypibJY+ElJ8gcfdM6MikRuD9wPANCyWAhNIeCmplphzApB0N3tplOsIAiC7k40SVpyc3W//mo6cpRobgYIggoEoowMTnBw16ddaNL90y3LlBMm4gR8YnZ4iLi9Ox2MWSEIuntRVqujqqpPirBAEATdVWylF9r++YE1L580mwFFsbyVouxh0lmzhEOHIKw/hZc2B5Er8m8P8PdhUyTV+/EWxqwQBN2lSKOxdcm7pkOHUDbbWVAagJvMtYIgCLp7kEaDNT+fslrZfr7S6TMkkyfx4+Ov+Mwod26cG+tAgyVdhLvzLrVN7QUYs0IQdDey19a2vvW2+fBhZ8lABIHxKgRB0LVZi4rMx49LJk/hBAYAAARpaW6PPoIJRbJ772UpFNf4QjaGomy2A2WxMBJFez/YwpgVgqC7Dm23t77xpvHgQUwoxNzdSbUaQWDQCkEQdAU0jlvOnNGuW285ddJRV0/TtOezz7q6W2Fef3vFdc9/HYROH1OZi7dZIkyAto0BIhHoFRizQhB010E4HNl9c23l5e4LHiU0Ws2yZbf6iiAIggYdoqPDfPyE9pdfrIWFlMUCKIoTFEjb7DRFXWoQ0IOAFQCA19fft+e7B20Wii+k9I9iMGaFIAjqKQSRTJnCCQ3lR0e3f/wJTVG3+oIgCIIGEVKr1a5da9y121paStvtCIfDCQqSTp8mmTSJFxNzo0fjRkRsmPFC/sWOkTE+z7u79/qqYMwKQdBdgbLZVN98yw0Jlk6f7opaEX50tHPZCwasEARBf0ao1ZpVqx0XL2ISiSArSzxxomzGDEzuBnoFc5efjx6y09GijAtGeLzeHQTGrBAE3RVIra71o490a9eyPDw5gYH8xMRbfUUQBEGDCIXj5uPHuaGhHH9/58xoSLB4zBg8PFz2wP3CrCxMIrm5o9Ns3MEjHSzcfjOHgTErBEF3OEd9fcviN8xHjiAoyouOwm5iZQqCIOgOQ7S1GQ8f1m/41ZKbK3/yCeVrrzmzVFFM+foiZ1kVrPelqfocjFkhCLqTWQoLW99821Z0DmCYdNYsr3/8nSWX3+qLgiAIuvXwxkbtr7+a9u6zlZbSOI5wONa8fMruQHnOPlU3s4jfT2DMCkHQHcu4f3/LG28QzS2IQOC+cKHnX15EBtOcAQRB0MCjHQ5rcbFu4ybT3r1EezuN46hYLMjIkM6YLp4wAeVy+vyMGIrcTFnWTjBmhSDozqRd+3P7hx+SWi0qk3r9/R9u980FTHEWCIKguxhpNLa9+6755CmUx0MlEmHWUNmcOYLMzF6XoLo2gqT3XmirbDWybjpshTErBEF3JspqIQ0GdkCA8p13xKNzbvXlQBAE3TKO+nqUx2O6VbHkckFmJtHeIRqdI5s7lx8b23/387Uq82sbi05fVFM0YN30WWDMCkHQnUn+4IOApPipqYKU5Ft9LRAEQbeGo6ZGt3Gjftt2YXaWz5IlzvAUQdwff1w2Zy43NKS/z44goFFjteGUgMMiSPImjwZjVgiC7hz2ixdpB86LimSaXbk/sfD6X9PZx6VnDV0gCIJuj4arubn6LVuNu3eTWi3tcFBmM/GXv7A8PZ3Bn6cn80Gfa9Hbdha3pAW6JfjJAACB7sJnc0KNNnxPSfuRyo6bPDiMWSEIukNYi843v/YajeP+337DDQ3t0dfQNGU2IwChCYIymTCuc7csBEHQ7YqmKavVuH+/fstWy+nTpEaDsFioRCIaPVo2cwbaPxmrjA6j/eez9VsKm/PqtHPS/P43P5VJYJ2fGQgAOFGtpmj6Jk8BY1YIgu4Eht27295731FXh3K5lrNnexizEh0dltOnaIcdb2g0nzgpmTa1/68UgiCov9A43vyPRYatW2mCQFCUEx4uHp0ju3eOc/Wp35aSqtpNmwua1uU1NmosDpKSCtg2nLThpIh7KcgkKfqm41UnGLNCEHRHlAj46CNCo8E8PBQvvyybM6eHX4gKhfJHHiEmtSMYi+vKKIAgCLrN0DRTXZXJiXIOZdu28WJiZHPniEaN4ob0V9IqRdMXO8wrT9XtKWmtajcjCBDxWGNCFQ8NDcoOcxdy+j7ChDErBEG3Mcpm6/jvF5rly2mLlRMU5L3kbdGoUT3/clQodLvvPjBY6fX61tZWi8Uik8mCg4O7fqqmpqajo0MoFJrN5qioKMlNdlaEIOg2RNnt1txc7c+/AAzz+/g/TM8qt7lzuYGBwuHDMbG4X89utpOLN5/fU9KGoohCzB0e4fnwkMAhIe5YX5RivSIYs0IQdLsi2jvaPvlYv34DTRD8xATvd9/lJyWCO4XBYPj111+TkpIAAF988cWjjz6akJDQ+dnjx49v3rzZz89v6NChMTExt/RKIQgaaKTJZNq3X7d5kzU3z1mFWiKxLXycFxfnDOzc3SWTJ/fTeQmSdpCkwDWHKuaxxkZ7FdbrJsQpH8wMTA2S91uwegmMWSEIul0Zdu/Wrf0ZQVHx6Bzv995j+/qCOwhBEBqNxsvLy9fX95tvvqmuru6MWSmKCgkJ+eyzz5RKJYsFh3EIuovYL140HTyk27DeXn2RtloBinJjY8WjczA3t/49L0GdrFYtP17LQpGlD6WyMGex1QcyAkZEeEZ49e+Ebic42EEQdLuSTp9mPnkS5fGUbyzu9XhN04AGABl8pa7kcvnf/vY3HMcLCgo8PT0zMjI6P4UgCEEQ1dXVjY2N3t7egYHObbkQBN3xND/8qP7hB7y+nqYohMfjJyXJ7r1XNDqH4+/ffye1E9SBsrbVp+pPXVRrLbiIi+XXazOC3QEAQi5rwAJWGLNCEHS7oWna4UBcRakwqdT3P/9GOBykt3ONeiv+xYHKOrWFjaILsoPSg+VgkCktLT106NCQIUOUSmXngwiC8Hg8oVDo7+//008/DR06NC0t7ZZeJgRBAwHh8ezV1Sx3d8GQIbJp00TjxqI8Xv+dTmtxHCpr/+lEXUGDzoqTCAIS/KRTEnyC3IU3dBwUQVDXrADG/K+3YMwKQdDtRP3TCsuZMz7vvYvJnfElKhDczNEcJHWiSn26RiPkYtOTvMHgk5iYGBER8dZbb5EkOXPmTOZBmqZTU1Mx134LuVy+Y8eOlJQUtN+6L0IQdEuQRqN+6zZHRYXX64sQNhsAIJkyGW9qEmQNFaanM4/0n+1FzV8fqi5s1NtxksvCYrwl8zIDJscr/dxueNSt7jBVtBlRBCmo1+osDpnAWeKgF2DMCkHQ7YGyWFRffa1a+h1lMrJ9fJSLX7/5Y3qIuNlh7vn1Wn83wfDwfukK02ulpaXHjx9/6KGH+Hy+p6fnkSNHOmPWpqamNWvWPP744+7u7iwWy2g03uqLhSCoL9mrqox79+o2bnJcrAEUKRyWLR471jlPKZEo/vZ/A3MNNoI6VaPhsrCsUPd7Uvxmp/iKeb2Mkg02nMvCQj1FFE3bcKrXlwRjVgiCbgOkXt+65F39li00jvNTkvtqVyziWrRi+rb2X32W3jEYDFqtlqIou93e1tYWFRXlcDi2b9+emZnJ4XCUSiWXy8VxvL6+PjU1FU6yQtCdgKKsJSW6detMR446amqcS0l8Pjc6FuHzB+DkjVrrofL2OWn+XJZzPJkQq1yQHZQW5DY+RunW25lRRrRSsuyRNJKmUQTxEHEHXcxK0zQy2HY0QBB0e3LU1ra8+ab5yFEaAMm4cV5vvtGvGw4GidTUVABAbm6u0WiMioq6//77cRxvb2+3Wq2hoaHp6el5eXkWiyUoKGjatGm3+mIhCLpZ1qKiji//Zzl1ijIYaIJgKRSCjHTZvfcKhw3r16RVAMCFFsNvhU2/nG1UmewSPnt6og8AQMRlfXRvYp9EcnwOFqrog7axNxuzUhSl1WoPHjwYGhqanJzsrDFrNp84cYIp1DJ8+HBRfza3hSDojmc5d65l0eu24mKExZLdO1v5j39gMhm4C7DZ7CFDhuh0OoIgPDw8mAefeuop5oPo6GhfX1+CIOSuvF4Igm53eFOTcfdugKJsLy/xhPHSmTMFiYlMm4D+c6HFsPpU3a6S1nqNFQAg4GAVbabOzw62qcebjVn1en1JScmpU6cwDGNi1q1btwqFwrFjxx45cmTjxo0PPfQQnHCFIKh3LAWFjc8+izc1oSKRx9NPezz7TK9LBNymZFcP0GHvKwi6rdkuXDCfOeM2+15U5NyGLxo9WjrrHo5/gNvcOWw/v349tZ0gC+v16/Iathc16yw4QdFKCW90tOeDmUHJAYN3UuBmR383N7cRI0acOnWKyaayWCyFhYXMpoHw8PA9e/ZMnz79GmMuBEHQNbA83NlKL8ps9nrtVbd58wfdXT8EQVBXFEUThLPmM4o4b7CvNGdHk6S1sFC34VfToUN4UxMmEstmz3KmrnK5vh9+iHBuKnO0hy52mBeuyG3WWdkYKhdyZiT53pvqlxrYv10Jbl7fzFjQNM18YDQadTodz5V4wefzzWazXq+HMSsEQb3D8ff3+fe/8ZZW0fBht/paIAiCrsOwf3/HZ5/TOM6LivResqRbIhOp15tPnND9utFy8iRlNtMkyZLL8Zbmzif0a8Bqw0kUQTiu/VXBHqIIhYiNITMSfeZlBoQPYF+Am9HHq2wURZEk2flPkiQJgujbU0AQdGcjVCrDnj1us2YhrrtfblgYNyzsVl8UBEHQ9aFcrqOmhjKZOL6+oEsiE6nX637daNy1y1pYSFmtCJvNUiql06ZKpkzhxcX191XZcWr7+Zb1uQ2zU/zuTXNmHfDY6D9nx+MkFesjBbePPo5ZMReKchbfoigKRVF2P9e8hSDoTuKoq2tZ9Lr5+HG8oVHx0l+YflcQBEG3BUFaunPq1GblJyViXfag22tr2//zH0qvR/h8fkqKZOJE6cwZbO9+72OiMtn3lbatOlVX1KTXW3GjHZ8QpxTznLHfQPZcHVwxKxOqAgDEYrFMJjObzQAAk8kkFAphYgAEQT1kyS9ofestW1ERjaGU+Y+9qxAEQbeHSwmsiDNptaiIExiISZ0Tmfy4OGF2Fqk3yB+cLxyaxfK8VAmk/+gsjvV5jZsLmgsatDhBcdlYWqDbAxmBt/Wu+JuNWW02W01NTXV1NU3T8fHxAQEBw4YNKykpiYiIOHfuXFZWFtzZCkFQTxj37m154y2ipQURiTyefMLzuefuthIBEATd7hCOc20ZYbF069Zrlv/g+fLL7o8/5nwEw3w//hjhcPq70ioAgKTo749dXHGirlZtthOUgMPKDJPfnxEwJd5b5JphvX31zdXPnTuX83vi8Lhx40pLS4uLiwMDAxMTE/vk+BAE3dk0q1Z1/OdjUqtFZTLlokWyOfcC2NgJgqDbCmWzG7ZvI/V6gCBERweN4+ZTp9zmz2PiVGwAp/COVnSUNBtkAvbQUPdHs4JHhHtI+HdCoubNxqw8Hi/apfMRLpebnJyM4zjMZIUg6LpIvV71zbea5ctpm40THKx8523RyJG3+qIgCIJuAGWzmY4cUS9bbi3IRxAU0DQnMNDtgQekM2egA5KUX1ivNTvI7DBnygGGIk+PCmNj6PwhgSMjPNnYnXP/31+zxDBghSCoJyiDwbhvL2kyCVJSfP75wQBsoe0GQ2/n9C4Igm410mRqfuUV06FDlMnsrHaCobTDIZk61f2Jhf19apoG55t0K07W7SpuVYi5q5/I9JbyAQBDQ9yzwzzuvKHt9s5sgCDodsf291e++abu141e//dXtr//AJ+9RmVu1Fpg2ApBUK+hfD7C5pB6gzA9XTJtmvqHH5x5+az+7blKUPTZGvWqU/X7ytp0ZpykaQxF6tQWJma9U8c0GLNCEHQL4K2tbC8vZo+taPhw4ZAhyMAuzuAEtSG/8X8Hq9QmB+sOHd8hCOon1nPnCJVaPGY0s7/K4+mn+MlJ0ilTMIWn5qefgLMRVn9xENSJatWKk/UnqlQqs52NIUopb3qiz5w0/xjv26981Q2BMSsEQQNNv2NH+78+dH/ySfm8B5i9VgMcsJ5r0P1nT8Wh8naHqwTM7438IAiCrsNWXq5dvVq/aTNLoeCGhXICA517e2JieDExzoJ9hYWkwQhIyl5ZRdttCLfvqwQsP17z3vYLDsJZCD9KKZ4Yp3xwSGCQuxDcBWDMCkHQgNKsXt3x74/xjnbNjz+KRo7gDGw+gMGGf3fk4oqTdS16GwtFhkd48tnowfL2gbwGCIJuR3hzs+anFfrt2/H6egRBSA7HUVfHxKydHLW1AAGoQEC0t5MmM6uPYlaKptHfC6sm+knFXJZUxp6XGTAt0SfY466IVhkwZoUgaIBQVmvHf/+rWf4DZbXywsO8Fr/O9vEZsLPbCepQefvn+ysL6nUkRfvKeI9kBT2bE/b53oq9pW0DdhkQBN128KYm3caNup9/cTQ1AZJkKZWSCRPkjz7CCQ7u9kzphAmirCxA0wibjfZFT6VWvW1XScvRStXiKTFMeJoaJP9yfnKcj9RL0u+lXgcbGLNCEDQQSI2m9V8f6jdsoEmSn5zs/e4SfkLCQF5AQb32hTX5WgvB56ATY5X/Nz4iztfZn4agYGYABEFXZTp8pPW99+wVFYAGmEwqGjPG/dFH+YlXHr4QPp/Fd+6CunkdRvuGvMZNBY3FzQaLnUgLdHtmVBgAgIOhY6K8wF0JxqwQBPU7+8Wa1jfeMB8/DjBMPHaM93vvDUCj7W7i/aSpgfI6jfnlcRH3JPvBfVcQBPUESyYjOzpQgUCUM8p9wQJBWtrvDVr7S1W7cVtR69rT9Y06i52gRFzWsFjlXZKxem0wZoUgqH8RWm3rW2+ZDh9GuFzZ3Ller76KufXBkllPFNbrSJpODXQDAAg5rA/vTcBJKtRTNDBnhyDodkTb7cZDh/hxcWxfX+f+qvg4xauvsjw8RKNz+ruhdKvetvToxZ3nWy52mGkAhFxsVKTiwcyAEZGeQg4M2GDMCkFQP8OkUsmUybbSUvlDD3m88DyC9W/ZQkaH0f7D8Zplx2pCFaKfFmR4ip2taALkggE4NQRBtynKbjcdOqz58UfLyZPuzzzj9eorzilVFHWb98DAXIDG4lif29CotSgkvFERigcy/IeHe96pxVZ7AcasEAT1LwRF3ebO5cXG8mNjmcpW/W1PSeuXB6vO1GgoGjRprdUdRiZmhSAIujKaNh07plm5ynz0KGUyIWy2rfg8heMoh9PPp6ULG3RKKd9b6txQFeklnpHkozHjDw4JHBIi76wVAF05ZrVarXxX+rDZbBYKYfIEBEG9QZktqm+/4aekiEeNcv4bRfnx8QNw3uoO09eHqn/NazQ7SAEHGxvt9ddxEZHKQVFn2+FwoCjKYrHsdjuKorDBNQQNEuYzZ7QrVxr37afMFgTD+AkJsnkPSKdNQ/vzjxQnqOPV6rVn63cUtTybE/r3SdGIq3/VW9Ni2dhA3Nvf3jGr2Wz+8ccfz549m5KS8thjjzU1NR05cmTu3LlSqXNrLQRBUA+ROl3LkiX6TZs5fn7spUt50VEDcFIbTm7Ia/j60MWqdhOCgHCF6KWx4dOTfLmsQTH6nzx5csWKFRwO59FHH42MjNywYUNYWFhmZuatvi4IuqtRNlvHZ59p164lNVqEzeYEBcoffkgydSpbqey/kxIkvbe0dV1uw9Eqlc6Cs1DkwIX2F0dHCLnOvCkYsPYoZi0uLvby8lqyZElubu5vv/2WnZ1dW1trNpthzApBUM85amqbFy+2HD8OAOBGRmLiAdrwdK5Rt2TbBbXJ4SHizk3ze3ZUqLesbyrO3DydTldeXv7iiy+azeaTJ0+y2WyCIOrq6mDMCkG3Fsrl0jhBdKi4YWHSWfe4PfCAs6d0v9FYHEfKO1aeqsut1VpxAkWQGG/J9ESfuel+As5AJPrfOTGrQCBISEiQSCSzZs2qqak5d+6cw+FAYC4FBEE9ZiksbFn0uq2kBGGx3ObO9fr7a9hA3fQm+btNS/C50GJ4ZUJkTpQCDCYURUVFRSkUCrFYHBMTU1BQ0NjYGBU1ENPPEAR146ivt1dViXNynPurEMT9scdQiUQ6eRI3PPzmD17UpN9Z1EJQlI9MMDfdT9hls7/Bii/edH5TQRNB0lwWFqWUzM8MmBAHi1jdgD9ezZCQkP3792/fvv3JJ58MDg7m8/lqtZrHu+u6LEAQ1DuG7TvaPvjA0dCAiUTuzz3r8dTTCKsfZw5oALYWNnPZ6IRY5yoel4W+OTWajaEC7qDbWurm5iaRSJYtWzZq1KiMjIzs7GyVSuXu7n6rrwuC7i6Oxkb9lq3alSspuz1ozWqe676R7euj+MuLfXWKix2mLw9WWXEyK9R9VoqzVFYnEY/lLuRSNMgIls9J85+Z5CMT9O8GrzvPH4O7UCicOnWqw+Fg4lR3d/c5c+ZQFGUymVgsFgxeIQi6Bs3Kle3/+ZjU6VgKT8X//Z/bA/1bGqZObf7yYNWGvEYZnx2lFAe6JiqkvXoDoF1tsOjfP+gPCILExMSEhIRwXHuQKYoaP368w+EwGo0YhvF4PHRAyilA0F2L1Ot1v27U/vKLo6KCJklUKLQWFjIxa98aEuLuLuK0GeypAW44Sf17dzkbQ14aG+HMQ0CQJ0aEJAe4jYzwcBfBSia98acJCRRFeTxeU1PTli1bSktL29ra7HY7hmFyuTw5OXn27NnK/sxKhiDo9kVqdaRGzQ0JVb67RDRiRP+dyOwgN+Y3fnGgqk5tBrSz5Gqr3sbErL2gtTjy6rQkDVr11tM16lGR/ZhUwOPxcBzfvXv3iRMnampqLBYLSZJCoTAwMHDy5MnDhg2DuVgQ1OdInc6we4/mxx/tFRW0w4FKJJKRI90eeViQktIfp/OS8LgsjI0hZ2o0e748XtVhEnFYE2K9o72d1UsC5AJYJfpmdF9EMxgM+/fv9/X1HTJkCI/HQxCEoiir1VpbW7t79+65c+cylbAgCIK6cn/yCcDChJlDBKn98k7AKKjXfbK3/FBFhx2nxDzW/MyAhcNDbuo9gAahChmCcjgowsb6Pang9OnTra2tkyZNkkqlmKu3Ao7jGo2muLhYLpfHxsb29wVA0F3FWlLa+vbb1oIC2uFA+DzRsBz5o4+Iho/oj3ZWBEWrTfYjFR0Gq3MvUGGjDidpMY81PMyDA0sBGJpB2TZAOgDKBknzALeX9Qe7/9hUKlWiS7fHU1NTc3NzVSqVv79/by8ZgqA7ir2mBpAEN8y5cQHl8Tyffbb/zqUy2b8/WrPqVF270Y6hYESE5wujw0ZGeN7kYblsIiO4LNRDg6CIl0QGgBz0G4fDQRDEww8/zESrXaWkpOTn5/ffqSHo7sRylzsaGmiKEg0f7jbvAfHEif3Xh++L/ZXfHrlodZCue2FaxmePjFQ8NDQwI0gOG1kBTQ3Y8waw6wFPCqKm9FnMyuVyy8rKQkJCRCJR50IVTdN6vb6pqSkwMPCmLxyCoDuBJS+vZdHrCJ/v99mnnKCg/j6d3kosP3ZRY8H9ZYLHhgU9NDSwT7Yv6Cy6xZv/UX6xArDBqmdWhSlCQb9hsVgOh6O6ujokJITVZZrH4XBUVFRcLaXVbDbjOC4Wiy+PdC0Wi91uF4vFXY8GQXc1ijKfOcNSKLghIc79VUqlctE/SJNJOm0aJu7L3iI2nCqo1yKIM4GVeYTHxrQWh5DLomnaQdAzknw/mDUQjVRuD16xQBkHLh4CQcOAqPdzDd1HOh8fn/Pnzy9ZsgRBEBaLhaIoSZIURdE0PX78eE/Pm53VgCDoDmDYs6f1zTeJllaEx7MWnR+AmDXUU/jSuIjcWs0rE6KivSV9dVgviVd26LDyhgovd6/hYcNBf0JRNDY2dvny5UajkWmIRdM0juMAAKVS+fjjj3d7vsPhOHbsGIIgBoOhoqLioYce6rqjIC8vr6qqysPDo7m5eerUqW5ubv168RA0+Flyc9U//mjas1cydarPP/+JcJ23tdLp0/vwFA0aS1mr8VhVx5EKVXWHKdnfbcMzQ5kuAJPilBacTA9y+79155q0Vi9XL1aIoaWofBuO2yiWFR9G0Tysj2JWBEEmTpwYERFRXl7e0tLicDi4XK6vr29sbKyv75+qNkAQdHdS/7RC9emnpFaLurkpX39dMnlSf5zFTlCrT9XaCeqZUWHMI49nBy8cFsLp075WGIopxApAAz6b7yXpx1riDF9f35deeun8+fP19fV6vR5BELlcHhYWFhsbe3krV61Wu2PHjueffz4oKOjJJ588efLkPffcw3zKaDTu2LHj4YcfDgwMXLdu3a5dux7o50INEDSY2crK1D/8YNq3j2jvAAhir6oiVB3sPgpaKJpu0dsOl3ecqFaVNBuq2k0WB4GiCI+NNeusKpPD2xWeBnkI/29cRJ3GYscpGoAmrYWgKBYsCeJS1HhuUvFpHAeCgn0XZnUEyHuZZXrlFaUQl24PkiR5+eIUBEF3D8ps7vjiS/Xy5bTdzg0JUb7zdj+VCMir0368p/xgebuYx070l2WFerhyT/tl/KFoisk/Yz7ob2KxOMul64M0TV8+wHp6er722mvu7u4qlYrFYgUHB3d+qqGhQaPRMHOrSqVy165d99577+VRLwTd8eyVldrVa/Rbt5IqFU3TnJAQ6fTp8gfns/puWbhFb3v8x9xzjTqCdC46i7jslAC3aG/JiAjPYWEeCsmfqladvqhuN9rsBFXYoNdZcA9Y08ol2M1vsrv3jua6eJm3G7/3C2U3kAV19uzZlJQUpr4gBEF3G6K9o/WD9w3bd9A4zk9M9PnnP3mxMX1+Fp3F8e3hi6tO17UZ7GwUifQSS3h3fihWX19vt9sjIpxFHDuhKOrp6VlUVLR79+7JkycnJSV1fspgMJAkyaTAstlsk8nkcDhgzArdXWhavWy5+ofleEMjgqKYQiGbMV12/wPcsJvKSjfbiXONun0X2u9J9o33dbbx8xLz/Nx455uQRD/p6GivjGB5jLfE5yqtobPDPH56LIOiaJmAczeMXT1StS/g9LcrePYH+Qhf7C7mCPosZjWbzWVlZTabreuDCIJYrdaysrLk5ORenwmCoNua7tdf9b9uBBgmmThR+cbivlp360TR9O7i1s/2VRY16kmaDpALFmQHPZYdzL+D2nBXV1e3trZe/nhpaWlCQsIVvyQqKkogEKxcudLf37+zoguzx6DrNlm6/zoiQNDg5Oq8itfUsry9xRPGuy9YcDPNV2tU5tIWw/Eq1eGKjhadVWPBORjKxKwsDHkuJ/yZUWGxPhLe9VZ7fGV836uEs3cXEgfO6oGuMUpbB4o3Srj8mXxsl3OoopG+ilkJgvjll19UKlXXxlcIgpjNZrlcDpu1QNBdy+2+ueZTp1gKhXLx65jUOZT3oVa97aNdZb8VNhvtBJ+Njo/x/uv4iD7ca3UNFOqsTYMjDqT/69EUFxf//PPPl++Xam9vj4uL6/agSqWqqalJSUkJCwvjcrnr1q3rjFmFQmcPBZJ0XjlBEDweD66AQXcDUqu1nj8vHDIEcf3CS6ZOIY0GYXa2MD29dwdsM9h2FreeqFKVtRqrO0w2nMRQlM9G/d34XVt8JAfI+ux7uOO1lYD8FcCiBjO/AphrXIqaDEJzvm+48JWxJebmUja6x6xSqXT06NFBQUGhoX+aXbfb7bt27SJJEi4/QdBdhbJaUVcnEUwu9//6KwTDEG7fZ2g5SOpMrUZnxRP8pS/mhM9I8kEHpKIhiVNyvX8Ckq1wKCxaB8+zf7f6ZmZmkiQ5derUbi2vcnNzL98tcP78+R07dsTHx2MYhuM4j8ejKKqmpkapVPr7+4tEIr1eL5FI2trawsLCYMwK3dlIvd6wa7fmhx/w5ma/L79gMunZXl6Kl1660UM5CIqNocyf4LkG3eubzjO7pqR8dpyvNMFXOjzcIzvMUyqA0U6v1J8ERz8GCApSHgbBrg0PYm9y3s+rv5hWYGmOdvbJBn2Zz5qQkMDlcrvFpmw2e+jQoTBghaC7imb5D9bSUufEqsw5zYAK+rjrIE071/eYloZ/GRNe3GRYOCLY323gehvazHhow7CnsWTUjKhqLHLP/p3ZdXNzS0xMvDy+jI6OZiZNu0pMTDQajeXl5Waz2WKxLFy40Gw2L1u27IEHHoiPjx8+fHh+fj5FUR0dHVOmTOnXy4agW4gmCMP27dq1P1vO5tK4A8EwS15eL3Z/Gm342VrtvgttZjv5+pRohdh57x3hJY5WSlAEjI31GhriEeEl8pLAGlU3wqwCF7YAdRUY+w7AXCFixATgGQW8EwH/jxYtNMrh0Ww2wsFodh/HrD4+Pld8Kqx1BUF3D8pqbf/sM+0PP5ImEyfA3/PFF/v2+O0G29eHq60O8v174pkmMXPS/OekgQHGF3MQd6u91oZxKM+gviw5fkVcLrfbEhZD5rol6EYul0+cOLG2thZBkMWLF4vFYpqm33jjDSbknTx5cktLi0qlmjdvnrSvUzUgaDCgHQ7T4SOan34ynzlD22wIlyscOlT+yMOinJyeH6SizXihxXi4ov14lbrNYLM4CARBpiV6j4121rbzlwtWLsyUC9lc1p2TNz8QTG2gKR9c2AoqdwNjizNvNXo6CBji/JTUHzx1FPClzqnW33XUGcc0PJnBe9KtjW/R2oQevVwXgt1TIAjqjlCr2/75L93GjYAkhZkZfV7Qasf5lv8drDpbq2VjyOgoxfjYP0rlDzAUQzABADRCoaRINuiW1zkcTtdiAgiC8F15Ggxvl1t0aRDUv/CGhtZ/fWg6cIAymRAOhxcfL1/wqGT8eEzS08WQDbmNO4qbL7SYatVmB0GyUJTPwfzdBBkh7krppb8jDEWY6qpQj5AOZ6ha+huoOQzaioHdCFA24AicE6tEl737gksp+zbCZrAYFBIFizBEIQ0UT4CiKEKaAOjlihaMWSEI+hN7TU3L669bTpxEMEw0bpz3e++yu7RfuklV7cb/HazeVNBkdpAiDjY+Vhnk4dxOdAuZgdFCW0SAD3feQ9DggfD5eGMjZTTyEuLd7r9fNvMeVHKdlRCVyU5SdOfi/rkm3frcJg4LlQs5kYGyeF/ZqEjPjCC5TMgZiGT5OwlNgZYiZ+fVC7+BlnPAYXYGr2wB8MsAgUNB7Gzglw7Y3UP/HUU7Fv+6GGEj217c7haoQH2VphoHojSzZb1fF+r7mNVsNhcWFjocDpvNlpGR4e5+qRUvBEGDnyU3r+Wtt+zFxQibLb1vrtcrrzCZrDfPhpPrzjZ8fbi6usOMoki0UvzS2PCpCT5929fqhhitxtWnV39+4csgLGUqePBWXQYEQQzr+fMoj8eUrGJ5eHg8/ZSjrk46fTr7KimLTE68zuI4U6s5WNaRW6vJCJZ/MCue+dSoSEVhvW5stCItyD3aWywXDrqFlMGOpoCpHZTvBOXbQXM+0NQ6cwDYPCDyAmHjQNRUEJABJJeyRmkAVMaOFn1Lgt+lsn06q66gqgDlo6VNxSPCR2nQCDWOI4iWZtJe+zZmdTgcLBara3ErlUp13XJXNE0fP3480OXixYtMR0FYIQuCbgvms2ebnnueaGtDxGKPp5/yePZZpI/+eNUmx3Nr8o5WqnCSdhdy7s/wf3pk6C3f67Ds2Pcvr/krQEA7qskAOehArTvRNG2327vWEzSbzSRJSnq86AlBdxh7RYV2/Xrdz7/wk5ICvv+OKU4imXTVvtAkRVe0GYub9UcqVMerVBqzw+ogLQ7SQVKtBpvSNbaMjlSMCPdgYzAC6RUSBwfeB7nLgFULCKvz5kDmD7yTnYWroqYCyaW7CBqAkqbiwobCXed37Sre5e/uv+MvO7ylzpyloaFDHxv/WJRXVIxPrIO0IwiFAOd/4CZcdYyuqKjYt2/fggULmOz+rVu35uXlLVq06LoVVS5cuGC32yMjIyUSCUUNRC9ECIL6BEsuR2Uy1GZTvvmmbPasS1v6+wKKAIOVIEh6eLjHS2PDh4f3WVvFG2XDbbzf17Bmpc7+997/AD1/NHU/H0hphL5UAbufEQSxbNmypKSk7OxsAEBjY+OXX345a9asjIyMATg7BA0qeEuLs/nqb7856uoRBNjKy+21tbzIyGt8icrkeHXDufNN+nq1haBoNobw2FiQhzA7zH1stJeIeymwQRAAA9Ybg1tcM6muZF8UA8YmoG8EfBnwSQYxM0HoKOCbyjzRQeJVbZXbirYdqThyruFcY0cjIJxfanFYylrKmJg1yCNo2SPLmOcbrMabK3J1vZg1MDDQYrG8/fbbTz311OnTp8+ePfvUU09dN2BFECQ7O/ujjz4qLCxMTEwcN24cnGSFoMGAslhMx44BgkQ4bEFa2hVX/Lmhob4f/4cyOGt03/wZCZLSWBwKsTNAdBNy3pwWk1+vXZAdzL9eI5l+ojFrVp1a9dOJn757+LuUwBTXluGAfS9sNm1ehVU3k7xGtCMU+F4q2t+v2Gx2TEzM0qVL1Wp1QEDA999/n5aWBrsMQncbvKlJv3WbdtUqvKmJJgiWQiGZMMHtkYe5YWHdntlqsFW1GWN9pW4CZxAi4mLlrcaKNpOPjBfhJUr2cxvhSlQV8eAWnd6iSZC3Epz4LwgdDab82xl+IihIfRTwZCBmhrMgAOp8bVVmTVF94YnqE1vObTnfdN7ZMxV3zrVyxdxE38TU4NTJ8ZMzgi/deyP9MAVw1R+wWCz+xz/+sXTp0ieeeCIlJeXdd9+9vHfLFYWGhk6cOPHQoUPFxcWhoaHKvtu9AUFQrxHt7Y3Pv0CZTJhMFrjiJ0HapbJSREeH6dBh6T0zEZZzNODHX0oFu0kXO0wf762wOshP5ibJXKW5h4S4Dwm5ldntBqvh1Q2v2rX2/x3437IFzrt/BIBoeRBg5wJRHmDxgGbqwMSsAICcnBxvb+9FixZZLJZXXnllzJgxA3NeCBok9Fu2qP73lb2igiZJTCQSTZggnz9fkJbatZmz1owfq+o4WqkqatRVtZvevyf+gYwAAACPjT07KqzDZM8KdY9UiqV8WDm+V5zbTuk/KlLVHgb1BcBhBKMXO+dWnZOX2SAwmwRAb9Edrzi6/fz2vNq8oqYih8nhHD3ZgMfjJYUlTUuclhWWleCXIBf+UZC1GxRF+mST67VuSgoKCoqKiqZPn97R0bF169Z7771XcL2K4jabbfv27aNHj37kkUdWrFjxzTffLFmypIfBLgRB/Yfl6SlITzcfPcqLje2cxrBXV7csfsNy+oyjpcXz+ef6JHvVaCM25DV+eaCyUWulaDA5Xnlvqj+4RZp0TRyM4yl2piIEeQQ9P/r5ExUnJidMdjW8ds0BCDxOYy9daDWLJOTM4LEDNkuj0+l27NgRHBzs5eV14MABf3//rjWtIOjOh6K2C2WoUCAeNUr+8EPCIUOYZCSrg6zqMBU36g+Ut5+t1WjMDhtOkRSFImh5q7Hzq+cPcQavUC/pG0FTHij6BXhEgrFvuqZEMWfbKofZubmK86daLo3quhEfjWhQN9A4DUhn2Ojv458UkDQiYsTU+KkhihAO06D16gwd1nOHmoxNOAvDUITTrQvgDbnqEF1ZWfnll1/OmTNn0qRJHR0dn3zyyRdffPG3v/3t8gaDXel0Or1eHxAQgKLoggULLBaLVquFMSsE3XKoUMhSKGiSZMndmMQAS15+y+uLbGXlCIqSWk2fnOVsjeaTfRVHK1UOgpLy2A8OCcgIvjVzq2qz+vsj368+vnpC4oSP7v2IGSUXT1mMTEWk/C6VVhDEgnroSClFEYAtGphrI0ly6dKlJpPpzTfflEqlGzdu/PDDD1955ZWoqKiBuQAIGni03W4+e1aYkYG4kgzFY8d6Pv+sKDlJMHaMM2D63Z7S1re3lDTpbKQrUVXAwSK8RCMjPLNCPeL9YO+Mm0PYQf0pZ3XVuqOgrRTYLcAjGGT/5dKsavBIEDySQNDy5pJ9pfvmps31ljnTUhViBZ/NBzTw8fQZGzN2QuyERP/EWJ/Y656NpoHDSpQcbSo51qxtsnDYnDa0Lle7/V7rUA+RRx/HrEKh8LXXXot05UF7enq++eabZ8+epSjq2jGrTCaTSCRnzpyJiopqamoKCAiAuQEQNCjQNOiyJ1K/bXv7Bx84GhsxsdjjuWfdn3jiJidZW/W274/VrD5dpzbZWSg6OtLzhTHh2WG9HJhuRr2m/uczP39z+Jua1hpgBybC9HzO84Eegc4BSnCFLF5mK6trN+sAFWilKGrs2LExMTFM6YBZs2aFhYVdd7cABA0qFE0zIwqCOCvzX+OZNI6bjh5TL/vempfv888PpPfc47yL5vEUr77aYiGPlanb9JYHMgJZroMEugvbjXYPETfGW5wUIBsT5ZXgJxX+vq0K6g0SBy2FoPogKN0M2i84J1MpHHDEIDjLGaeivwd1riSBC43nh/5zqNli5rP5T4580pkwxuH/a/a/aIoeHTP6Tzf814PbiF1Lz9cWqZ0JCFyqAs3d4VipsjXa8C7dB27QVX8PunVw5fP5I3rQC4fH402dOrW0tPTcuXPMl1w3nQCCoAHjXBAnKc3Kle3/+ZjU6VgKheKVV9zum3uTh61TW176ueBYlRpFgJ+b4IkRIfMzAiQDnmTWYexYemTpmlNrSutLAQ1QLjoqdtSLY190F19rrpeiWDQgBrLGOJvNTklxbgLrlJBwqaIhBN0uNuQ1rs9txBAkJVD2yoTIKy/40rT5zBnN8h9Nx45SBgOHzXKcPEFOndphpY5VdhypVF1o1pe1GlkoMiLcM9DduSQd6yP5/P7kIHdBtI9EyIGh6k2gaWcOQNlWULELNBcAfZPz9oLFd5asip4GoqfSXvEGjvRQ6Z6DZQf+PunvSqlzhjFUEern5ldjq6lV13YeaWbyzF6cn83DPPxEdcXqwBj3qNGKHbvUE8oeiYrwVYgVvf6e+v4Xws3NLTs7myAIlmtLBwRBgwfC5ZjPnjWdOkUZjdzQUOWSJaJhfVAiQCHmeoi5XBY6PdHn5XHh4V7XaVfT52pUNWtPr116ZGldex0gAFvAzonKeX708xPjJrKvXr/aYSUunGiurUAw52DlGNhLhqDbm9ZK7K9S4ySllF95ZspSUKD54UfTgQOkwYCgKCc+vm3YhMMpo46vLjxbozHYCCtOUBQQcLBgD2GzzsrErGwMvTfVb8C/mzssVG0ADWfAha2ger+zuipudUar7qHOblWxM0HQyGKj6lxz2a7Df99furtV10rjdJxv3MLhC50TlGz+0oeXyoXycK/wGz8z3VCmNWvt0VnOpAIEQZLHBSpDpMEJHigLrT6SgNHtUUI/Dqv3a0r9FVbCgBWCBiMEoS0Wyu4QpKV6v7uEF3v9nKSrMdoIjdnOvM3wOdjfJ0bNSPSZEKcc4IKIrfrWb498u/b02vKGckADjIeNjR/7wpgXxsaM5bKdZcmviKZB4wVN/u66+hI1hbAQVxkXCIJ6LlvoGGup7zDZp4oV3SZZKbNZ9cEH5u3brHojYLE5gYHy+fNE02d8fEy9ZkcVG0W4LJTHweI8pSMjPIeFe0R6iX3dXDVBoZunrgSbngJ1J5ztVTEO4ElBaA6ImurwTatGBdtK9x05/GRhfUFje4OzoioGUA6qcFc4iEs37QiCjIi4/qL6FZUebznyczmggLuPUBHk7JAikHJCU5yzqjRFEzROABynb2p2AI7UEHRHo2lbaSlNknxm9ZkkUQ8PyfDhilf+xvby6vVRz9So/7WzXG22//LkEKXU+WYTqhCFKgZoDxOjuqN61clV3x39rqmjCZCAI+SMjb4UrbKuF4MSDvLY+srWi3oOn+vFa7bZMRS4YZ15XRAEXY/y2O73tn8MUFTIKwPjM0CXhHiUx61q1pbyA06Hx2fE+j/21AzM25ltmOBpPiHnR3lL0oPkoyI9431lPDas4H7TcCtoKwbyUCBwlZoSKQBFOhPBQnNA6Gi1T/o5AjlRX7j10MtF9Xk2q7WzomqCb0JqUOqUhCnZYdkSXh804fP0FyEAYQswvcrKxKx/QC41bLnJTjXdR3baVUHr8sQUiqJIkmSzYRU0CLodkKSjpcV08KD56FFrYSE7MMjZDhFDKRwXxcb6fPRhr3dcacyOrw5V/3ymvt1owxBka1HLE8NDwID76uBXn+7+tKq5CiDOudWJsRNfGP3C6KjRbFaPxig2F0scH+D4rSZ4pIem7Ct5fcXXjsTEFmV8QN+Up70aiqKu2GYFx3EMw2AHFug2ogmPX5o4U49xF44b5YuidEODRaPRBUUdvNB68qK6KHR6rSdhoNEOX48FroAVADAnzX9YuEeEl5h3ixqL3IG0tWDXP0DVPjDpQ5D2mPMRngzkLCZxy36zZePFs/kn3znXUOAw2jorqiaGJTorqoZmJfgnuAtvqq6LzYxXnGkLTfYUypyLWopAybjHYiWePE//7ulhuI3EbSSCIDYzQREUykL7JmbdsWMHj8djClzX1NSgKBoY6Nxve/78+Y0bN77++utwcysEDVo0SToqKy0FhaaDByxnc0mjkbbbaZJE+AK8qYlpZIKwWb0LWEmK3n6+5fN9FaXNRpKmgz2EC7KDmRLfA4+iqaqGKp6ENz52/AujX8iJyrn2LClNg6YKbXuNPmFsEAsjQVNBRHJcWIJCTaseza+ZS58rpkrPt03u15i1sbFxx44d8+fPFwqFJpOpuro6JiaGzWYTBPHvf/971KhRWVlZ/Xd2COpbJv/greEjG9ji6e5eLSvXrNxXckwUUK1oM1gcNgdJAUTMY2V4CiPlPAInWa4g1VPM9RRfNWMH6ildPeC7Aa4rLuRJndmrehVoyiOTHyZoksvigojxFAD//d+M7ce3APRSRdVE/8SRESOnJkwN8Qy5mYxShsNGVBd0FOyu66gz6jusw+4NQ1xlH0JTrtyXu6Pe2HpRDwDdUKI2Gxxi+aUG2jcbs168eFEsvhQg79q1SyAQPPLII84yv1ZrW1sbMwsLQdAgRON486LXLSdOOBoaAEUhHA4qEHCTk8RjxggyMjhBQTSBu553w3/FFE1Xt5u+OFC1tajFbCcEHGxSnPLlcRERA7XXiqTIGlXN8arj96bdK3TVu56VMqteU58TmTMuZhwLu36OU12Jeve35+02SohpInWfgKrDrJlfgehp7qTbfI9nqi+2zBehOaHD+/W7MBqN58+fJwgCANDc3Lx8+fJ33nlHJpPRNN3R0WGxWPr17BDUh2gA7DgptRvH1532PX62yGD7adizzUJ3vtEu4GDR/uLR0YqMYPcYb4lS2svoBOoOt4Kao+DCZnDxiLNVVeL9zgf5bmDIc7bo6T+1N//6yaTZGXOechWoYgMwIiw7r/rMuNhxE+Mm9rCiak/QNF13Xl2wt76pXEvgFMZCLQY7SVAszrVmDaQKwbA5YSRBYyyEK+h9Vmr3r0RRtLMCK5vN7txKhbn0+jQQBPU5mqLsFRWYVMr2dm3SxDDaarVXVrL9/blRUcLMDNGYMbyICKQzpae395was+Oxn3LLWw0ogiT4Sf8yJnxqgs9NpiXdkCMVR2Z/PdtoMcoEshlJM5zF+GQ+H937Uc+P4OYjEcl5qAEnAA/UnwXaRnBhG4ieysE40bxoPRXsjuFKce8TfHsCQZDOERVFUQ7nj34wGIbdTG8YCBoYDoKsbDeVNBuOV6tPVHQ8XrpzQsNZh4PgSWQjxfaOCEV2gHhMtFeEl5jb28VfqDvc6qyuWrkXlG4BmirnP0kHqNjRFjSCzxU581CTHkBosP6/k/ef3qeyqhZkP8pMoz4z6pknhz95xaLUvdZcpSvYU193Xu2wESiG+EW6xef4haUqsOv9uEVu3MQxfbAo1z1mRRCkM2m1a5wK6wBA0CBBkyTe3Gw6eMh09KitqEgybZpy8evOT6Co24PzOeFhouHDedFRKP/PBWhuYpFELuSMiVI0aiyPDQt+fFiwj2ygd/j6uflJ+BKD0VDWUsbErD3RXm/k8lhSOQ0u7JC6h4x6KJbNAYoAMXBfBNTVIOlB5mk4jVPATgCCpEiWcyGtH7FYLC7XuTbKYrFQFGUGWzabDTNZoUGLomkbTp1r0B2tUuXXaavaTfUaC8Vi0Wyulid2UAAbkuX35MK3Y5NknnIOBm+9+ghNOXNVS38DlXtAyzlg7gAsHkDZeqHXYYH/7opzJ4smPz76yedyngUAcBEwPnKUyaiZmToTdbUGcHYaY5IH+oi21Zy/p746r91icCAo4u4jShzjF5Gh5IkGdJvTFfZgtba2VlVVAQDa2tp4PB7zcW1tLUk682cH8uIgCPoDRdkqKqwFBcYDBy25uRSTq2q32y6UUmYzKnSumAuHDHG27b4iZ7Ma1y1oz8KjBq2lTm0Z5mpkhSLIszlhs1L8Egaqd2Jxc/Evp36ZmjQ1MyQTABDuFf7R7I8UEkVWWI8yPs16e/HhpoK9jX4B1CTJe1j9YRA/x/e+HwHiyuJKceY73RIWi6W8vFwoFNbV1en1+vLycpFIRBCEXq+/VZcEQde2ubD5w51lHSa7FScdOIUgQCFg+anqgjUNNhbv9dSHn/rHgrCkgD+1qIduhuYiqDsFyraAmsPAbgAkATBuCdezEMd3ox77Wltb7bk07gAWsJK/4qkRTzLJUX8d/9eXxr3Ewfp+x5Ghw1p8tLn0eJNZa6dpIPMSRA1VJoz2F4hvwe6m7jEri8VatWrVzp07nRnWJhOCIDt37kQQxGw2h4eHw8kACLol8MbG1nffs5WUOOrrO3NVecnJojFjhEOHMP27r43UaOylpQiK2i9edDQ0cPz9r/ZMGoAthU3/3V+lMtl/eiwjyd+5tKQQcxUDsnmisq3yy4Nfbszb2NjYWNFRseaJNczmqrnpPWrWRdOguqA9d3tNe72ZQtitlSq9rFrORYBFDYxqIHEmUVwGoTsLsfQnFotVX1//8ssvM/uuTCbTq6++iqIoRVFms/mhhx7q7wuAoOuyOsjzTXqDDR8bfSlVBsWwaq1NgFAeQk5qoNvwcEWCt9D7iFplDHqsRnKBLV/AhaVV+4LDDGqOOCdWG04BlbPatANlXyTRrZT0iNF0jrY2WPXArgKos6Kql0w5Pmv8lIQpnTOJ1y3w15srshLFR5pKjzWrmkzOYqtiTkSGV8Jof7n3LbtD6f5N3nvvvcOGDWM+7jqrStO0WCyGKa0QNDBoinJUVXECAhBXS3pULLYUFBANDSwfH15UlCAzUzxmDDci/I9c1eshzWbgsHP8/RAASK0WXCVmvdBi+N/B6q1FTRYHxWOj+fVaJmYdAOcazy07umzVqVVanRZQQOohFXAEBEn0vGyqusl0ZltNTYHKQSBsjA5nH0iVbJSFhoPEN0H8vYBztfKxNOIM1Pt9g6m/v/9nn31mtVqZoZWJVjs/9r/6XQQE9SsagOp2U1mr8fRF9ZHKjuoOc5yvNCnE04OLWgvy/X/d+XC1NqSteuyDU8MfGY+5toeD0Id1Z0p88/fwEbWwxQNEX3m3OHQDSreAjY8DilDTWBHFOuEgtuBEkd1uo0zO4v+klSPkJga6KqrGT8kOz3YTuPXr5dQUdpzcXK1qNFEEzROxghM8k8YHeHWrunrLY1Z3lx7WFIQgqH9yVQ+ajhy1V1Yo335bnJPjTC6XSDyeepIyW0TDsrnR0Sj/hic2WF5eyv/9D1A0wDBu4BVy4a0Ocs3p+m+PVNeqLRiKxPlI/jouYmxM/25LYlxoufDlgS835W9q6WgBCBCIBLOSZz2b8+zQ0KE9PILDRhTtrz9/uEmvJhAM9WRVpkm2hkXiWMbfQfhkZzmYq7Fq4xzfBniUsblsVqMQhI4C/YbL5YaGhnZ7kCRJOBcA3RI0DYqb9ccqO07XaCvbjTUdZjtBYRjKE/DUZkflroP00R2aQ0f4bW3PIjQic5PaDZcCVhev/KNvHV9KYZi3tBrkJN9ssfi7DYUDVRVoOw9ipgPMVVrBJwm4haxtq/1UbysiHXacApQzRuNxeQlBCc6KqmFZiX6J7qKbqqjac7oOa1uNgc3FAhLkKRMC/aPlTDWrwRWzNjU1NTY2pqamslgsmqaPHz++fPlyk8mUkZGxYMGCy8NZCIL6AEXZyiushQXG/QcseXmXclUdDvPhw0zMChDE/fHHb+YMHXb6H/l2O0EJOay/ycmILkEvTlJnajSf7q04eVHtICmFiDcv0/+pEaEe/Z8MkF+f//2R79ecWaM36AEJ5G7yqYlTXxjzQmpQKtLDxXqSuJjfnLurobXOSgG2EFPHSE+lJGj4aQ+DyGkAvU7WBE0SJAkwhAAIC7c5+nU3gcPhyM/PDw4O9nJ1IFOpVN9++21+fr5SqZw/fz4szgoNDJKimdDTQZAv/1JY0qzHSWcdSzcBOz5AFi1hJelqY/IOiFad7DCbaIriBAeJRo92mzeP8+c7riZF0J7QLJwE2X7RcI2gkw236a16mqY5LI5MIOvcEdVd+W6w/hEHximzmC1e8UOC0oB7OHhgDVV5+ux3TwIU+Cn8kvySRkSMmJo4NdQz9OYrqvaEpsUs8eCzXM3JooZ6a1stPuHSiDSvXrcA6PeYdc2aNQ6HIzU1lekjsGTJkvHjx6ekpPz2229ffvnl66+/DgsIQFDfMp86pV623H7hwu+5qlxUwOelpIhyckQjR/bVWXCCOlvZ1mqw+0h59hF/mmfVWvAl20rP1GgEHFZOhOKlseFZrq1X/ep84/kvDnzxW8Fv7ap2gAKRWDQndc7To57OCM7o+UG0VRfz9zRWnLPaCQ4GiGBxaWaKVjl0LAgZCbAeTUXbKPFe9bNNrVo2F5viiA8G/aisrOyLL7549913AQB2u/2TTz6prKx8+umnGxoaPv/8c7lcHhUV1Z/nh+5eDoKqU5sPlnfk1mrmpPmNi1ECADhsLDNYXt1uzI5UDA9zT/OXhiklXtrW+kcWmcsqaB6X5e8vmzlDOn06NyLi8mMecov4MP5eAEC9t8+YAcgHv03sLNn9j9/epkkQ5xe1dP6XfzSasumdWaq+KQBxBVHyICCQf9tc868fnpN7h+99ebdSogTKhCyW5K9T/prgn5AamBrnGzdgl23S2k9vuVhfrM55KCoowTn+80XsnAejBtvs+Z8CULvd3tjYeN999zGB6Z49e/z8/P7617+iKBoWFvbWW281NTUxbbEgCOo1miSJlha2ry+zmkaTpHHPXkDgLKXSmas6xJWrGn4Duao94SPlZ4V6rM9rSPSXxvn8KUXVQ8Sdm+qns+CPDwt+NCuQw+rHpWqSJgvqC745+M26vHVGgxFQwMPdY0bSjBfGvJDon9jz49jN+IWTzflbKowmDCCYh1CVkmqLHJGBBacAcAMTEggL84n0ErpLUAzhiv9cHayvVVVVhYeHh4Q4W91evHgxLy/vnXfeGeKq86BSqQ4dOgRjVqhvNWqt5a2GvDrtoYqOqnaT2UEarDiXjY2NUTK7Dp8dHf746Egfu54n5AOhK+Fb4MfKHMJGMemMGbK5c9muNYErygyQvjE2BEVBqOfVMsXvRkJtTXBLISCBN9uG0SRwmEBTPqjYDS78prIaCob8FVfETY4dCzyiwNRPJdW5zWveMbFraztqnTErAMEeQR/f9/HAXzZJUo1lWlWTqeJsW2C8BxOqDraAtXvMirjguLNZDkEQBQUFI0aMYDJZ+Xw+i8VyOBy37lIh6PbPVW1qMh48ZD58mNBoAn9Yjrk5k+gFycmye2ZyIsJFQ4ZynXVV+2UTLoI4S1a5tpc6/1/Waiis181N90cRZ5LS/CGBo6O9gj0GYjfo+9vf33x0M+AAiVRyf9r9T4x8Ii0wradfTOLArrMB+a5vzzdc0FAki8OlYhOpxDFx0vAwV/OXG8MTsIfN6Z5j2k8wDGNGV2YVi8/ndwapEonEbrcPzGVAd4PCBu3SwxfL2kzV7SaTHUcRhMvGRBwswVeaHvTH3h0fxG7ctq1l7c/iiZM8nnNW+gQo6vnii4Cm2T4+1z7F0FD3oaEwXbC78crQGIWYR9g9pAJw6ktQf1rfmH9Er96FI2dwurD85TD/6LRXDikkChA1dZRX0iccabxvfKxv3zSpuiFWo7PSKk/oHDalHvzEMX66NkvcSL9BGKpeOWblcDjJycmrVq3icDj5+fkdHR2dNQSqqqpomvb0hHsDIagXuarll+qqMrmqNhtAENOJE9IpU5zvEQKB7ycf97Bsaq8xoxCKICY7sfxYzX8PVGrNDncRd5xrlxWPjfVfwOrsTWrqUIgVzrgNwR4e8nDexbxJCZNeHPPijY3ULXlgzzvAYeTc/wtXzCcJOijBPW2c0i9WcWm5bXBLSkrasGHD6tWr/f39V61aNWrUKJnMOeFts9nKy8tzmMRlCOqVZr2tXW9N8Jcxt6Y4QW8qbLLhtJiLxftKo70l6UHykZGeAXIh21X231Ffb9y7V7v2Z0dtLWW14mqN2/x5mOsXkumrB/UOLg9eaUOGoZi8vbKo/l+7LPg+ArTQgKZoQDprNNA01aJvccasAAS6+b087uWBv0iHjajOb8/dUacMkYxdEMNUL0ka2wd9qvpb94F+9uzZZrP5o48+wjDs//7v/2JiYpg51z179qSkpDAjLARBPaTfuk2/aZOtvAJv6JKrmpwsGjuGH9clV2kA6nIgzm3CHBZaUK87XaOxOUgxj9Wk7fce9wX1BR/u/FBtVm96bpOI61xDnJo4NUIZcaPNrwmCNp4rcKvcAWgUrT2QOePegGhpeJqCw7sNolVGYGDgCy+88O2337a0tAwdOvTRRx9lHs/PzzcYDBkZN5DI2x8Ikv50X0Wr3oZT1INDAjOC5Lf2eqDrIki61WA7Wtlx6qK6pNmgszi2vTBcKXVuQo/1kUyO9w6UC7LCPCK8xAHyP/Je8OZm3bp1+u07HJWVNEWhfL5oxAi3B+7vSZln6KooHDSeJc/93Fx1WEo5PrUSebitHnfFqQAgHEwp8xwXM258zPjkwORwRfitukyapuvOqwv21jeVa3EHZdbbk8aaPAP6smNWv+o+3ItEomeffXbhwoXdGre++uqrHPgLDUHXQxMEZbZg0ktF7GwXLhh27ERYGMvLq/9yVa9Nb8UPlrefa9Dn1mrYKGLDSYqmx8Z4vTgmvOsqYT8paiz65dgvwNmnYMu8zHnOWRyM3dOA1aIGzQXAP72jnXXk53KbNnRKyAJZiB8IGiUXs+SK66xdDkIZGRnp6ek4jncdTjMyMlJSUniuQrzdaLXa2tpas9ns5+cXFBTU9VM1NTUdHR1CodBsNkdFRUkkN1s3kQZg/4W23FotQEBOpAL86WzQINJhtFe2GwvrdQfLO0qa9SYbYcVJkqIFHFZRo56JWQVc1tKH0rqWpnKiKNXSpeply0mViiYITCoVZmVJ750tHj0agbure01VSVw8VJm3+kxd3q9602EcGBDXnxMFODxeok98SmDKlIQpw8KH9XdF1etqrtIV7K6rK9Y4bASKIv7RbvEjfd19b6cWZt1/TQ0ulIuz+sXvEATh8/lMiRYIgq6cq3rgoOngQU5AgPKdtxHX1Kl43DhrYaFoxHBhRmb/5ap2hZOUxuyobDdlBsvZmPMaDFb8xTUFRjsh5LBQ1FnR5rmcsHmZgeL+maGkaKqkuSTaO5rpyzIzeeaohFFxvnGpgc5qJD1lN4Bza0HhWtCUB+5bbiTGN1doCQLUDH8lefQVtjDfFnAcV6lUOI7Tv+u6kUChUPD//Ouh1+vPnj2rVCpxHP/8888XLFiQkJDQ+dnjx49v3rzZz89v6NChzILYTWJjyLhYr9w6baRSHOtziyuHQ9fw6b6K1afqTHYSQQCXhfLYWJiXaES4Z2aIe1LAH2uh3QNWF7yllWhuxmQy0ZjRsjlzhEOGDOT9850p78fqPR9MM4Bq0hmnAgQI2SCRhUxVhmRO/zAxfNSAVVS9Bm2LOX9vfXVeu8XgzGF19xEljvWLSFfyRLfZT7/7m9YPP/zw3XffhYSE8Hg8giA6I1eSJBMTE5csWQLLX0PQFXJVmbqqJhNttaJSqcezzzA5YYLkpMAVP/X3HEar3tagtdSqzecbDUWNuvJWIw3oTc9mRXs7a+l7SXiZIXK9ldBaHPVqS3Kk21Mj+2XXEU3TRyuPfrH/i13Fu1YuXDkzeaYztZ8v3fnSTh77CpOIV6ZvdHYvzFtKtdXSpANDSVB7Jnji7OQJgTwhKybnNsi4uprq6uqnn36aJElvb2+KogiCYF40mqbZbPbixYuTk5O7Pr+iouLIkSNvvPEGl8s9derUrl27OmNWiqJCQkI+++wzpVLZh/UHZXwORdM8NirgwHF+UNC67j9P16gjvMQTYp37yl1/UywLToYrRFHe4hR/t1FRiggvEY99pR8ZRZlPnMBVatmM6a5tmKj7o4/QBOE2axY/BXYB6BWaAm3F9R01hRQ23Dl1KgNB2X5yX9KmZpFkelDGcHffhIu7I0lzSECkPGHKpX4Bt46hw+rsv3q82ayzO6vwegmihirjR/sLxLflynn3wS4hISE6OhrH8bi4uLFjx/r6+jLBK0mSPB4PdsOCIAZltep/22Lcu9dWVoY3NDhzVblclO+qqzpubNf51P4LWCmaXn26bv+F9la9rV5j6TDZSQqgiDNplcdCS5uNTMzKxtB/z0nksbB3tpZWd5j6432Koqnjlcf/u/+/u4p3mQwmQIPVp1dPTZzKTLX2NGA1toK8H0HxOtBa1EbGnjEtCvVtj5mUBELHIyiadU8o6to7cvvy8PDIzs7Ozc2Vy+VjxoxJTEwUi8WdpVou3+EaFBQ0atSozlmDrskDCIIQBFFdXd3Y2Ojt7d1XJQgp17lo5zaRPjke1BsURWstjlMXNcerVOeb9FXtpiaddWKcckyUF8v1J3Bvqn+MtzRSKQ5TiJgdV5ejSdJ8/IRu3TrT4cMAQfhxsdywMOdO6+Bgn/ffG/Dv6Q6Su7xizzv3NjSeJ5BlC5Y9lr0ABI8QPLrlm456msUfGjpUymJlLgpt0uhflsX/3y0NWB1W4vyRpgvHmlVNJucEipgTkemVmOPv5n07JQN00/3dNCcnJzs7++zZswcPHly6dKlSqRw9enRqairsgAVBNEG49lG5bk9pWr91q3HvXpTDcdVVjRRkDhGPHcMNC+untbYmrbVRa7nQakjyd0vyd64Aoghyokq9Pq+Ry0LZKOor5fvLBf5u/FhfaYKfjHkOU+UqyL2/BimSIg9XHP5y/5fbz293WJy18IJ8guakzXkm5xkM7fFcXUcZKFoHClYC/UUT6XXO9kyJdaIJl3RosKCwLIHr2m/3gJWJWd99992mpqb9+/cfOHBgz549mZmZw4YNu1pZVk9Pz7FjxzprttfXq9XquXPndn4KQRAejycUCv39/X/66aehQ4empfW4Xtj1oAhy5Uk7qJ9pzI7tRc0Hyzvy67Rai8OGUwRFs1AkUimS8dkERbFc65yhnqJrlESlzGbTsWO6tb+YT5+iLM5NltzgIEd9PROzQjeOVjefLzKqUwNTJTwJELh7m1u1OAAEXdx43vl5jgjxTZngm8I8O7c+/5zFaCfAmY56O2Hnsvq9m+DlbGb8YmFH4d56VaOJImieiBWc6Jk8PkAReNvn/FxhBojD4WS7NDY25uXl7dmzZ/369eHh4TNnzoy4UjMMCLpbclX375PNmSudPo0pUCUaNoyyWiUTJggz0rlRfZyrStE0QdJtBtv5Jn1Bg66yzdigccasTTrrS2MjOuPRrDAPtdmR5C+L95UFyJ0xq1x4lRUf+lJ91qtNzNwokiKPVBz57/7/7inZYzFaAAa8PL0ey37s4ayHo5Q9ro2vKgdnvwelW4G2nMZEFY7JBeZZbY4wQDqkMhA3wtvZVfVKw9RtCkVRf3//Rx999L777isuLj58+PAnn3wikUhGjhw5ceJE9pXudoxG4+7dux944IGuwy9N06mpqUymllwu37FjR0pKSp+sg2Eo2qK3vbWlOMhD6CniKsQ8DzHXS8yVCzk8NsZmDYKO43cQiqatDpLLQlmu1PNWvfXDXeWNWiufg/HZaIinMCvUY2ioe6yPJNRTxLle/0zKYjHs3KXfvNly9ixlNiMcDicoSDZ3jmTiRO6f265CPeGw6vJz12w6v+NgbcF5o+a7h797cOiDIGi4eOZX/7XYcbEyO2To5V9V3FhEUziXw63pqNJb9UyNvwGmaTEfXFnmsJFsLhoQI0+ZEOgfI2cKWt3urvVm4OcSFRW1ZcuWDRs2eHl5wZgVuutyVfPzL9VVNZmc6aoEKZ08CbiW+90XLnRf+Hg/zaqeqFIv3ny+RW+zE5QNJ3HSuVjLYSFhChFN0xRNM6Hn/MyABzICehJIGGx4vcaMIWiL3tpusCskvb/7J0jiQPmBL/d/ubN4J2F1JmWG+obOzZj7zKhn/OU9azxOOkBLEchfAUo2AXMbQOl2JCPPMKvamkEQKIdNhKQq0qcGu/vesf11+Hx+enp6ZGTksWPHli5d2tjYOGzYMDdXj4mucBw/efLkiBEjIiMjNRqNm5sb88bT1NS0Zs2axx9/3N3dncViGY3GvrowFAFGG74hrwlDERaKYL//R9P0+Fjlp3OTmMipXmM5WtmhEPNEXJaYxxLz2GIeS8RjcVyxF3RtFgdZ2WYsbtIfqVQVN+k/uz8p3VVZLFQhzolSNGgsGcHyUZGKBF+p6EY2Suo2bmx9+x3abkc4HH5SknTGdOnMmSyPfu/DfIexWdSFNWcOFmzcWLD5nF7lzNxxFQE4dfGUM2YVeoCMJ+65+pfPTJqZFphGA1rAEciFA1owjqZpZnzwCpIExrmbtPbUiYGhyZ7o9e52biNX/XvQ6/WHDx8+cuRIc3NzWFjYxx9/HB8fP7DXBkG3BqnV6n/bYjp2zHbhQtdcVX5KinjC+M6NCwj7Zuf/aFcf8Gad9VyDLq9OI+Cy/zougusaX0iaLm8zoYiz2r+vjB/jK0nyd4tRSgLkfF83fudcKdPFqifUFkdlu1HEYzVqrXUac+9iVpzED5Yd/OLAF/tK99lMNoABH4XPwuELHxzyYLjXjVQcNDSDTU85WxryuDZuQJF95nldjtEuQSmHwp+fPiU4NMUTvXOjHxzHy8rK9u7dW1hYyOfz586dO2TIkMurXzscjk2bNhmNRoFAUFJSIhaLs7Ozd+/enZmZyeFwlEoll8vFcby+vj41NbWvNhvQNBByWFFKvs6KG6wESdEkRdtxymjHjTacqUQBADh9Ub3wp1wpny3msSV8lpjrjFnFPJaAw8qJ8pyfeSm/Vm22W+ykm5DDxlBn7OsskADuTiTlnFLNr9cereoorNdXtRsbtVZnmXmKKmrQMTErl4W+Pjlawne+jD09rNmMCS9l/ohHj1Z9/Q0mk8kffkg8ahQL1vm5EQTpqG4p23Tyx92VxwpqzuodrgGaBSRC3tjw7BkZD2aHD+/JcWQCmUwwsJXsaaBqMp3bX+8fLY/IcG7Uw1joqPmRKIYKJLflRqtr6P6HYTabL1y4cPDgwdLSUnd396ysrDFjxkilzp0cEHRboKxWoq3N+RGKcvz9e743lqYopkAV0dqm+vZbvLYWYNilXNUhQ8VjRvdJripJ0fUaS6PWWtFmPN+oL2zQNeutdoIy2fAgd+HcND8mUy1cIXo+J9RLwksOcItUivls7IqVa3rOW8Jb8VgmRdEsDIlS9iaryYbbHlj6wNZzW0mbs0x2REDEfRn3PT3yaR9Zz+qk4jbnZAXbVdtcFgAC0mirtgZMOK2Z2qGVUAQhktJxI4ISx/rzRXfaOMugKKq+vv706dPHjx+3WCxxcXH/+Mc/oqOjr/b8tra2kpISk8lUWlrqnFOfPx9BkI6ODqvVGhoamp6enpeXZ7FYgoKCpk1z5qv0CYKilDLeNw+mesv4FjvRbrKrjPZ2g71JZw10F3T+MSEIEu0tsROU0YrXmp1v78yuLTtOuYs48zMvPW31qfrP91cKOJhCzPMUczzFPGe+gYTrLuQm+Dl3ETFPsxMUhiDMBqM71S9nGz7eW64xO6wO57IJggAfGT9MIUrwlaYG/jG/zlRX7QlHTY1+23b9tm2+//k33zWjxPb2Dvj+O7avHya5bUrEDwZFdXnHq05sOrPqaP05m83u7AKAAC8BOyUka0rUyGmZD/l5BKGu7aSDE0VSR9aWXzynaqsx+EbKhVLn+Clyu8X1CvpJ9x/D+vXrP/nkk6CgoLS0tOjoaBaLtX//fqZcq0KhGDFiBCwdAA1y1oKC5kWvIxiGCoVB635Br1Sq/U+5qo2NzkpV+Xmezz3HcwUQnNAQflIiy8tLMmnSzeeqUjSNkxQbQ5nJUTtBPr8mv6TZYLQ5V9UxFHBZGJeFhniK0oLcWL8HBT4y/utT+qDoZiceG8sIvqmFKhbGQhCEtJJ+3n5Pjnhyfub8EM+QHn0lDUDFLnDiCxA2Bgz/q/MRBNVG/uXMhfEXm73tZhuL5QhN8UydFKgMuZNvj+vr61988cXW1taMjIwhQ4Z4eXmVlZWVlpZSFIVhWFZWllJ5qZgRw9/ff8mSJd0O8uSTTzIfREdH+/r6EgQhl/fx+iPizEJB+WyMz8bcRVzwp4u6ZHyMV7yv1GgnTDZcb8VVJke70dZhtDdoLMm/J1s7w26j3WwjbDjZZrA7W1k4fxWcZyBIavHU6Fd/z3tecaJ234U2d5Eza9ZLwnXFtc7o1lPM9RBxb3ksqzLZLQ4SAUDMY8sEPbprNduJgnqdFSfGxVx6+TAUqekwi/lspZSXESwfFuYR7ysL9xKJuDccDDnq6rRr1hr37rVXVQEc161fz8SsAEGYEQzqEZranLdh2fEf82tzm9UdTHtrLgcZ6uVzT8a84TETkkKzEHa/V9TuPfpSS26UhcYM8+loMPpFy7Fb/cfS37r/tUgkktTUVKFQ2N7e3tTU1Fn1miTJmJiY4cN7NDcOQbcQKhBQBgPR0cFPTWXmTa+AJG3lFa5c1QOW/HzKZCL1en50NDPiIxyOz7/+hfL4CLeXE342nGzUWhu0lso2U0G99kKL4Z+zEoaGOotvcFwTpnaCilSKfWW8EA9RnJ802V8W7CnisZzrp2DQMNqMu0t24wT+QOYDzsECZS2avCglIGXh8IVK6ZUCmauhSXBuDSjdAdrOg+QH7Zh78eHGwgN6o9YDISyKIFHyuICood53/LIxiqJhYWFKpRJBkLy8PJIkO7PQ2Gx2ZGRkt5j1um6+91U3zE+gJz8GCZ8t4V8heiMouuuv8H1p/rHeYrXZ0W60q00Ogw032QiDjTDa8CD5pRVtigbHq1Xbz7dwWc6/DRZ2KY+WhSIOgv5yXvK0xEsT+RvyGlUmu7eUJ+WznXm0fLbYmU3LFvFY/fqr88bmkh3nW1AUeWRo4NvTr9rCjaToynZTWYvh5EX1sUpVndqSFCBLC5S7ubZFZgTL/zYhMslflhni7iZg924rpPX8ed2GXw07d5IdHTRBsNzdBUOHiidMuLnv7+5ispuEXJHz1T/+Wem2d7c16QAKRBhI9vAaFzHinpR7IuNnsDl/tLodhBw2ouJ0m7rZlDUrjM11lZJIUfiEySSegzjC7qeYdZbLFZ9KkiRsKAANfpzQMG5EBNHWJho27PIO2nhLq2HXTvPx47YLXeqqCvj8lGR2l96YWK/yYarbTadq1CXN+up2c5POWq+xmOyEc1IRAYUNWiZmxVDkH5OjTXYi2F3o6yYYzMXbP97z8Tsb3wlQBGSGZDJTqmlBaWlBPaupZNUBczvwcO3aRDGQ+ihoKwZJ8wGLS+LUuf31hnarUMqJzg5IGhcglt+Zy1jdBAQEfPLJJ1f7bGcIewtpzTiKIFacNDt/b3uj23bABD9pgt8ff0okRRvtzoDVZCO8JH/80EdFKoQcltps7zA6OozOfYckReMkTVBUZ2anFSd/OlF7oKydx8YEHEzCZ0lc0aqYy5YK2K9OiIz2dkbwFE2fa9BxWZhCwhVyWUwS7U3eCoZ6CnGSwgAS7nWFTYE0DZxZqpUdubXaqnZTrdqMkzSGIlwWqrU4Wg02JmYN9hAumtz7SVC8ra3j409Mhw7hLS0IhmEymXB0jtv99wtSUxH4vtwz+8oPfXfwKwFX+N0j37NQDBC2WQ7dFil7ol/cuKGPJoUNF/r+qanHIETTdO15VcHehqYyLUVS3mHSiHTnjS6bi7HvgoD1ynuwamtr16xZg+P4/Pnzw1wV3XAc37ZtW0lJyWuvvXbFaiwQNIhQlPM/ppzqZSxnTre+/Q5wxQeuXNUowdAh4tGjueHhN1r83+IgatUWEYcV4H7ppnzVqdqvD1907atwLlCIeawYb4mfGz/IQ5Tg90e11KzQ22Mnb4JfAsbChBxhm6Gtp2kAzP6q8p3g1FeAIsETB4DAtXIdOppeeAjhO6MKAQ9kTg+pzG3PnBbsHTawmxVuNYfDsXnz5jNnzgwfPnzy5MnMcFpZWblixYqpU6dmZv6eB3or2Alq67lmZ7GeJmNhgz6yV0nP14ahiIzPlv15ghZFwKNZQY8MDSJp534vnKR0Fue8bIfRrjE7or0vpWY6CCrEU6i1yGw4pbfiapOjRW9n3sW5LPSv4y7VtHEQ1HOrC1oNNg4LFXNZComzXJenmOMh4gbIhRPivNwEzgiSIJ0H4bExYQ+W5hP9pUw5qrSAS4mnNO0MjplQ2IqTL/1SUNlmYop7eIg4YQpRnI80K8w9K9TjqrXnbhDK5ZpPnsQbGzlBQcKcHPm8B3gxMbCR1XXhJM7C2M6XqfHMqW1/X1dwGnDBy+NeTvBLADGzoiy6I1FTOAFDwK2oonqjmit1+Xvq6ovVDhuJYohPuBunf/pvD2bdv+GWlpZ//etfXl5eCIJ88MEH7733nk6n++abb1pbWxcuXNiHTQIhqN/RtKOhwbh7N9Gh8njuWcy1kMqLjeVFRKASiXjCeGF6Bjc66toJr12RFG3Dycp2U1GjrqhRX6syV6vMY6O8Pro3gXnviPaRinlsLykv2V8a7yuLUop93fj+boLrFlYcADRNW3Gra+UX4bK5KNL9kuyEfVPBpsNlh9+e8baXxLnjeFzMuA3PbUgOSA5071mbJVM7KFwDitY661iRdsAWgqq9IOE+h5UsOdpoNZNDZ0qYFypqqHfUEG+MfetflgG2evXqQ4cOpaenr1271uFwTJo0ae3atdu2bUtPTw8Pv5HCC/2AhSLPjgpVmR0kRXWdHB0YCAJYiDMfgMtCRVyWn1v3xVkJj/3O9Di91eGcqbUSJjvhmpd17g/T23Bf2aVJJquDNNpwgqJwuzMqrVGbmeZeJEkHeQiTA6RMzNqgtb75W7HZTkgFHDcB28NVidZTzFGIuV4SXrCna+34j2u79C8rQZa1GveVtubX6x4eGjgq0ll6k89GM4LlLTpbRrB8eLhncoAsXCF2v+l9hLTDYTp2jO2p4MXHOcN9mczj2WfsZeWy+++DSavXZXVYz9ScWXtmLQKQL+Z/6ZxVrdw7s/H0Kj4YP2S+TOC691BEgckf3Rb7PTXN5oK99dX57RaDA0ERd19R4hi/iAwlT3jXzSF2j0GPHTsWFBT097//HQDw0UcfvfTSSxRFjRo16t1334XVA6DbgnO6lKYRDke/fbtu/QZSp0VYLNGokcKhzvrPnKCggJUrWHL55WkD19Cst60721BYrytp0WvNDmfNVIKkKGdFqsJGHU5RTFnK8TFeqYFuvjK+c9vIIJsBqVXX5vw7x+qwSviS1U+szgjO6PyU0WbceX7nf/f/90TVCdpGR/tEvzjmRdc8sXhm8sweHV1VAUo2g9wfgK4WEA7AFYHQMSB5HoieDgAoPd50eE05i4spg8UhSc4OpdggCOIHXmtra35+/ptvvhkaGjp06NB///vfv/76q1wuf+utt1JSLjXRuYUwFLk/IwAMVggCZIIrb4Gi6T8mHEU81vIF6e0Ge7vRpjLZ2wx2ncVZpUtrcfjK/ui40WG0H61UaS0ODEVQV6zMclXjQgAI8hDu+stwphNYrdr84/HadqMNIAhJ0c+vKWgz2M0OwmQj3IQcJmZFEOTlsRF/GRPhI+P3ScsFUq83Hz2mWb3acvq0ZNIk308+RrjOWUC3B5xp5dA1mGym3Lrcg2UH1+duqGyrJOw4YIPnRj8X5xsHIqfExR7IV8QLRr0CJL7gNqHvsBYfabpwvNmss9M0cPMSRGV5J+T48cW3RbDd/zGrSqWKjb2UYx4dHV1cXLxo0aKrtRa8mrKystraWgzDlEolrOoKDRyaxlta1D/+aK+qQtkstKXZARBUKORGRjKDPhPRsq+504WkaLOdKGs1tBrso6MUzK7eRo3ls30VOgvOY6M8NsbnYBFe4gQ/aaKfLEIp7nyjkgk4MtcsziDEZ/MBAtrV7TxvnrMDoYsNt/1W8Nv/Dv3vaOVRYAOADSIDIn2kPatdxVBXg7wfQclGZ9hKA8AVgvDxIONxEDEOoJemvgLj3d39xVwBSyi7DRbg+o9arXZzc/P3d7ZdCA0N5fP5o0ePnjNnDq/HM/3QFXW9P2RjKFPrtCsbThpsBElScuGl30BfN/7L4yJa9VZnxQODrd3oDG1J2lmJVsZno7//RZe3mv53sAoBiICL0QBUtps4GCrls9MD3dKC/ihQdfmscO+QRqNx507dhg3WgkLKZkNYLLytjdBo2N7efXL8OxVO4qUtpRtyNxwqO5Rbn2czWwHmXFHyYIEJiZO4zN5/r1gwb51AcNt0obdbieLDjaXHW9SNJgQBfDEnIlOZmOPn5t1fjbhvy5iVpmnO7/NPbDY7IyPjRgPW4uLigwcPPvTQQ6WlpQcPHoyJiYE7t6CBofv11/b//Ido7+CgwMgV1nj5Z82dJBs31llX9ZppLToL7qqZailu0hc16ktaDBqzXS7ghngI43ydywvBHsIJsV4GKxHjI3E2SvWTekv5TPH/24VCokgPSq9rqIv1iY3yjtJb9duLtv93339PXzwNcAAwkBye/EjWI49mPyrl92xFpbkAFK51ZgKY2gGJO9vDhI0DaY+CsDE4iVafbfcKoty8nMOrm5dw8tPxEnc+m3dXDwUURbFdmNFVoVBMmDABBqwDgMfGmHnTTr4y/l/GOJMxKFcSrbPgP06qTM4tUxwMYf8eswo4aE6koklnbdI582omxSnHRXsNj/D0kfKYbqt9xdHQYDp4SLt2rb2qirbZEA5HkJEhnT5dOmN67/aD3g0IiiioLzheeXxd7rr8+ny73Q4o51DmxwMZGJgiYE8NiPec+g9E4epbi7HBbRKw2i1EdUF74b56VYOJImieiBWc6JkyPsAzsO9TzG873d/IcRw/cOCAVqtFEKSwsLCtrW3t2rXMaKtUKnNycq5dn5UkyR07dsTGxgoEgoSEBBiwQgMKY5Ft7RSPf1gZsyk4u8039MgL064dERAk/f720rN12mattVlvdZAUAhAOC+WwULODUJmcmzycbVpFnA/uSZAJ2IOqFtUNQRHUmcOKAAfhWH169VcHvzpRdQLYnXOrMUExz+U8Nztltpe0B41zSBy0FoHc5aBsG9A3AgQDPCmImQaSHwEhwwFAW6oNudsvVhd2RA/1HvdYLOqqF3gHd2HtOQRBampqVq5cyeFwzGZzeXn5unXr3N3daZpGUXTkyJHecDptwDk7yWEIG3PGtW4CTrjiT7+ow8I8UgPl24qaX91QxMbQRZOjIrz6Pm6gKar9P//Rb9rs/CXh8QSZmW4P3C/OGY3Ju/fyhVxLYWSDtmFj/sa9JXvP1p5Va9UAdYaqIgwM44DZAmyIxC0uajyInwUCs4HoxurH3XIWg+PQ6vKqvDaKpFkcNCDGPWVigH+0/I4vBdjLmNXf3//MmTMtLS2dj+zcubOzPuuoUaOufTij0VhRUREYGFhQUKBSqVJTU3t6IRB04+xVVZbCc5JxY5mpCPGYMfQrf2v19FtZghZa2ffKaT5FOgstuahMdmfbUrW5pMkwOloxJMR5z40ioLrDfLi8ncvGXDuLBb4yfpRSnOAvTfCVef3ekwZFkJvfVHFrme1mo80I2OBQxaH9Zfud0SoGUiNTF2QveGToIyJej2PKC9vApieAVe9saiX1B1FTQPrjwMeZjmnU2IsONJw/1Ggz4yiGmHV2h424C3cJXI3EZffu3cydvEQiOXPmDLO6xWKxYmJiYMw62CAIIuBgciGHiReYeiB9hTSZMJHz7w5BUUF6unH3HsHQoW6zZ4nGj0dvJNv+btNoaJ3+v1nnLxY4k5Ew4MYFaWxkHIueLRYF+iVjkRNA/BzgHgYu22Z6W+CL2BgboUjaN0KWNDYgNNkTva0W9AY6Zp0xY8aUKVOu+FQURa87aWq329VqNZ/Pz8zMPHbs2Hffffe3v/2NfxM9hCDoivCmJs3qNYbt2x11dcinn8juucc5zSoRuz37LN2uEVWcQsyEDxfTmOylbZZ8V1X/eo21UWtp0dusDtJBUpdiVhSZkeSjkHAygtwDPYQBcr639A75daVpuvPWvKKt4uWfXz5acRSwAEmQgAYJIQnP5Tx3T8o9nmLPHhyL+uMNwC8VcCUA44KkB0HS/cDHWdGQokDlmda8XbUdDSZA02I3XvJ4/+hsHxiwduXv7//RRx91NmrpBpYRHLRw0lk7rw9jVntFhfqnFYAivd96C3Elh0inTOGGhfETE2+m5d4diaIptUm9/fz2zJDMaKWzYIKvrjaSspShIJmHzOHQw7mcFJmCHT7WGar6ZwDB7VFJsBNN06pGk0XvCIxzviUhKJIxNdg3wi00RcEXwTHhejEr5gJ6i8ViicViZpOBr69vRUVFe3t7YGDPCuVAUA/YL17Ub9yk27gRb24BJIHJ3PCm5q47h2kHziIcQgo52UFv+eSIwe7c4+/4f/bOAz6KOv3/32nbe0uy2fRCCkkghY4gHQELYMHe8Wynp97pnf2vV2w/PfVUrNiwi1QBqYLUEEghpPeyvddp/9d3JywxKCIEBDJv8sprmZ2dndnMfuczz/d5Pg/FMCzAEBCnEBkUwv7VxwtKTAtKTOA8otXW+uH2D1tdrQ/Pfjg7DvpWEhhR2VXp8/oAAeJUcU/Oe/K6sddJhSeQyx/xgYYfoIPVlL+DhBFwiSoZXPYWUKUAfZ8jpqXVs3tVS1uVjYowAhGeWWoovShVM7QLBX4RBEFi1QI85xA/1Fq8YQpDke+rzcMTT8lROFBe7vrmW+/q1ZQT+pnIZ8yUXwhnLzG1WjpmzODt8vlDt7vn0tcvK6/d+9ClD/1n4X8QlsX3vfP3QOPfdFihVC4wlYLsmWD45XBcOjdnz2t39Pz4eT0hwhb8rUSpg8V86nipOp4fP3+ZQfZblcvlRqMxGITp6lwTbX6M5hkcGCbS3e38dJlnzZpIczNAEEyplE2erL3uOnFpMUAQkmbK25zOCDvG3nzb2tduAGijNvXV3LkogWukgky9bESSaniiIk0nM6nF/XvwnNNQDOUP+Ss7K+1+++zhs4UELIuu6a554qsnAAsuLrqY06xJmqR7p9z77o/v1rXXlaaU3jH5jhN9g7APrLwH2Dth+cJl/+vr65k1nXsy5CcrNrTX/Njtc4RQFIlPV5RelJY+Uh8ru+bhOQ8wyIXj0rUoiqh/yWnrhGAY/549zs8+92/bRlmtCIbhGo181kxB2tHeezwcvpBv8+HNuYm5mXrY0ihOrDCJFQcw7HBPHRQVKAZy5o1s3Ahy54HcS0DyGHDEBeUcRaoSRMI0jLa2+znNynPmNKtAIJg6dWp9fX1mZubevXsvuOCCuLgTqOrg4fktwo2NHX+6M9zYCBgG1+lkkyZpb7heNBJOTFf3eH5qsK6u7DnY6TZo5Z+X4gIco0KRPEH4bzOz8lN0w41KpRgnBrXO9w/E7rO32Fpa7a0V7RV7W/ceaD/g9DqT9En5iflZBlgKnROfM7V4KogAmbAvSxVH8QdnPbi3bW9dax3D9k10/ip+K8wE4GpsZXGg8CrQsB4YR8C+VmjfiMHQbNN+y741rdZ2L8OwCq0o/4LEEdOShRK+7QjP+caDM4b9ZXo2Ek0lOomXBw9WWl99NfDTT7TfD1hWkJoqnzFdfdVVwj+6i8RZhd1nr2ivWFu1dvnB5a3drQ9f8vAzlz2LAEBs+fdfe7feNTx/7KIXoWAFAGTPAveUA+kJJDWdlUSCVPshR3KeRiCGo2VynvaCK7Pi0lWGlL6WbzzHYfAvMJMnT66qqqqpqUlKSiouLj6+zwAPzwlCmExEcjLZ1SWfOVN97bXikhJnhPluX+f66p4DHa52RwAAIMRROhxpMWR9cMXDW+ot903Lfmj6+dAwhqRJT9BzoOPA3ta9Fe0VLbaWFluLzWkDXG9aHP44A85edy+nWVO0KR/d8pFOpiOwo2EhFnr6wI61bJRfrkINOsGBT0DFx9C1avrTcK4NQcCkv4LxfwbKvvQJlmHt3f7yta2N+y1kiCaEaEahYdTcNH0yP+DynJ9gKIJxMwwnRbilxbtuHcBxwmRSXb5QedFFwuy+vBqeCBUpbyv/Zv83W+u2VnQcoEIktFbF0N3Ne2iawjEcxA8fJ8QBTgEKDvIQgQT+nIOwDNtaZatY395V7xq/MHPkjBQ4xKJI0dSzt5fH+a9ZCYIoLi6maZp3ueI5FcKNTc7PPxePKFJGiwJRiSTuL/cz5N2SkSMBgryysfG97c2OQCQUgU2+jSpxYaJyWl7c7OHxKIq6UEGAEHvQc9vBvt3Rvq1+W0V7xe7m3TXdNd6QF5ZPkbBeH6BArBCn69JTNCkFpoLSlNLS1FKTpk9W4hieoBxYge62Nlxv2TcnDhc5Kmzte/QpP29t72wF1d/A7gD2BhAJwZSAUbfBFDE4d/WzeIbTHPju5QqvI4QiiCFVUTo7JbM0jk8G4OGJQdnswepqycgRfX4mUyarFy3CjUb1oqsIfuIxSjASrOio2Fy7+ZuKb6o6q8gICU0AcJAhQctQcFnK8KnXvQ4FK2y/MRVc+QlIGQdksOvY2Y+jx99xyMEwrFQlSC/S44I+IdTV4Ny/rr29xk6GYOzA1umjSQYX8EG938fpmsjjBSvPSUN2dDg+/sTz/feRhgZxQYFy2hQgFIcpBuQXSI58wW3eUIcjoBQTxSnqOQUJE7P0BYlKPGoFuqPRVtXuRFl2Q3XPP2YPG+AlftYSoSIWr8XitYxIGgGNVAHY1bzrujeug8FUOEMGUAKVSqS5CbmjUkaVppZmGDLSdGlJGljveCIwIe/FWBCRC12Mjwm5jj7haoc9Vw99C8w1sJRNIAXDJoHRi4F4YDMhDoVObEiWhwPUiGlJhReaZOrzJDOYh+fXYCMRloY6A8Fx5Lj2DrTb7fr2W893K4LV1cYXX1BdDHsXYwpl/NNPoXzziGj+fYO54duKb9fXrK9or/C4PVCDoEApRKcLiYuF6BgcZCEkCHcCbxcwRKPRMgMYPh+cO/S2uLd8UkeTdEKWKiVfiwswR7e/YkN7U7kl4I2gKKJNlI2YmpRZauAF60nAJ5/xnEWEGxtdX3/t/nY53dsLGEZqMgoK8qsbesu92Ff72jMNspeuGMHNaV9RlsQCdmZe/Oh0reDn9nViAX5BNiwDkggw5BRm9E43LMsGI0GJsG+S6/l1zz/73bMJ2oSqp6ok0Zmv4uRio96ol+nTDekFpoLRaaNHJo/USrUC/GTqGjWmESBzGjiwTJl3KRg2Ey7qrQIHl4GDnwFPF+wUINGCtAtA2a1wtX5JBTDVrMsXCdEJGTBohBPohCuzw34yPp1vz8MzJOh5/HHvDxsRDFNdcYXhgb/84jqR9nbP99+7vvgy0tLCRiIIQYSqa8C8eVwxOy9YAQDbG7c/8e0TP7X8FAqFAA3VR7xcXIyzc8WCuQSdiACUiQB5AjCNAnmXAlMpODdJztPI1AK/K2LKUYcD1L61bbU7uv2uMMsCdZwkd1xCwYVJYjlvYnWS8JqV56wg0t7u/OQTz/fryJYWAYEhajU5dsKO4unbRcZ9K1varT6SZlpsfrM3FB8t+S9IVBYkFvzipoqTVR/f+vOJ77OGEBnqdffuad1T3lp+sO1gWUbZ05c8zaWWamXaIBnscfe0O9pz4mHD5DRd2op7VhhVxmMn+k8CBEE5JYpgOAyp7nkbHF4NnM3RPAMVtAIouxWkTgDYQEFc+1P3T183SRSCuXcXybXww1fHnZPJZDw8Jwcql1M2G4Jhv9BGNepn4vr8c8/368L19YBlUYlEMmqU5tprJWPHnKPuSyeIO+h2+p0AAVKB9BdtnhmWCYQDUqGUG+KsPvum2s0ABSIBNk6huxQJThThRWwYYULQ71mTAYYvAMNmQ0+9I41gzkVkKhGKoyiO9Da7Ow45LG1eBAFiuWDY6PjCC01q3gTw1OA1K8/ZAGv73+vOjz5BUJTRx1VmFe8snf0jrehpo4OBboZmDHLhiGTVpGyD5ByZ6I9BM3SLraXN3na45/C+tn17Wva02FuCkSBMSw2Dw5bD/7joHyIBFIIz8mY8f9XzI5NHpmj6/IwxFCtJGdROcgwFcCFo+wm8Pwt4zdAHQJkIsmaAsttAUtmvtY1BEMTnCpNh2t7t4zQrD8+QQjpqtPOjjxGCkI4fN+CpcEND+5/ujDQ1AYbB1GrYdvWqq2QXTDx+CsH5wcs/vPzvFf9GUGRG4Yyv//R1X1H/Efa37X9mxTNd3q7Pb/88VQctvcYnj7wlu9gUsFyBU9lMAGdpQAeBwgRSJoD8S6DNquCc7/Ds6Pbbu3xkmEYQ0NvsoSOMSEakjdAVz0jWJ5/bnlxnCbxm5fnDYILBI01fEM38Bd6NmxQTJoquufbJA/S66l4J5hcT6PAE+ZzChAuy9YUmlfBcaGFHM3SEjrTaWstby/e07jncexj6UtlaKX+0yB8BQAAInFBpVMVJxWMzxsaMNdL16Q/OfHCQ94ZlYQsr7lrCMCDggI9DLpgJIIsD+ZeCkhtB4i/I4pCfJIQYFv3AM0vivPZQXJrSlMt3P+cZ2nDf1n4dTIikJEFyEtXVLZ8+Tb1okWRU2VBQqxxpunQSZehAZHhiISdYI1QklrnUbGv+du+3AAH72vZBzcpQhh//8w6wAGAFEQbgYmDIgz56GVOgj965SSRIua1BgRhX6vu6l1VsaK/c3EEIcQRBUBQkFelKZqaYctW/7NPC8/vhNSvPH0C4ocH9+edkxX7jm282AenWOku2PmXsV1/iySkAQS70NfXavKPTtfMKE0alaySCc+ksfWPLGy+te6nb3R2mwjCYCis3oE5NTEhM1aZmGjJHJo+ETQgTcqVCKX7E7nTwYVlQ+Rk4tBwYhoOpj8ElCAKEchhblepA0ZVg1GKg+wXDnUiQOryrZ//69jEXp+eMhTkJuAAtm5t2uvaTh+fshxMcCMJGIoF9++xLl8ovnKKaD/tFoxKJ4b77wJ9ZcVFRn6IdMhQmDEuSyl0he7oybsOhDV/u+bLJ0fTeDe+laOFM0ficqYvGztdgWJ4xH66NYkCkAo4OEDcMmgAUXAGz588pyyoyTPvdYalSSAihQKdI5psX9lvbvcUzk8cv7LPaVehEhBBHMYSO0HkTEy+8Lof3VBlcziU1wHMeQLW1ej/80Ll+va3XftA0fN+H+6ppyaEu14z8+LG397UuXFSWNK8wIVlztg9nDMt8vvfzddXrRiaP/PO0P3ML/WF/S1sLkAEBLjBoDMUpxWPSxuQb89P0aanaVLno9JiY0hHgs4KufUCdBhIK+xbWrgB7vgA5PWDCfVCtIgiMbVBhkFgGLnrxFzfTVe/cu6ql47CTDNMVP7RnFBu40ZmHZ0jDsgBFWZo2//vf4drDZE9PpKVVcdFsrrJKPOJcDROeInGdu29mPcvEyNOrnu7w2hg/A3Cwr3Ufp1nj61a+h1pFSWNAAmeSjYCR1wGFEWRcCCOsZz8soGkm6CMdXT5bp8/R7ffYggFPZOKV2SnDYb8VFEVEMoKK0B5bMOZ4nTsuwZCq+OH9Q/4wrdCKeME66PCalecMxla/+LJ77fqaiHCfsWzHyBKzWB3sphnabZALCQwJhki5CE6raaQCjfSsa/nrDXqbbc2t9tbsuOzc6CjMAvbT3Z+u2raqvrj+jsl3CHFoBzs1d+qts28dlTaqLLUsOz5bRIg446rBx28DzhZgbwZde0HnPmA5DMJuMPYukPACfBZBQM484O6CJgDRVgIwLQEjo7ZZ7LEb89iCFRvaD//UE/RRKAbSCrUjZ6TEnAV5eIYgbDjs+2ln6NCh4N69MLUGgMCOn1iGEeXmyiZOABTX0mPI0eXq2t+2f1PtplV7P+n1kn4EsDYLIEBu9sgxKSO51FU43ljqRA0/Qiu9kpuAISpbDTnw5ywm4In4nGGfM+TsDdg7vbYuv9cWpCmWphiaZFiWZWi2t9ndp1kxZOT05Jwx8fHpytjUv0wtEopx7j+REAVNZ3nVOqjwmpXntEObe61vvxv4fm2HxfO/siv3JhZ6UQFKkyIE5MTJZg2Pn5prGG5USoVn19lIMVQgHKjuqt7bure8rbzZ2txsa+6x9Dww94EXroC6EEOwcRnjKloqxmSMCZEhTrOWpkJ7/9OyQwwNyAAwV4POvaCrAjiaYdW/twfQdDSGKoK/PV2w0IpLOci/DORdAv1WowTc5PaambQni6hLKu31q+L7lrMMe+inngMb2qwdPji3pRePmJaUP8EoPOnW6jw85woMw1IUEwiQnZ3htrZIWzvZ2am9/jphDpRWTDhseemlwM6dqFSKisUswxDJyeorr1DMmSNI7VNmQwdnwLmlbsvKgyv3te6r6qwCoah8IACgwOzii24rmFbsOJySkAtiZaN5c4GlGuTOO8t7AYSDVFu1zdLqdVsCPlfYDzVrmKIYBMAOVRiBYhgqlOIKrViXJNMapUm5R42rOfE6gNYau88VpiJ0d6M7HCT5gXRwObtUAs/5R7PNL+m0eD77jHI6dSlpoYRkF0LkqoiyVMPM4QkXZOukZ1O6qs1na7W3tlhbDnQc2N2yu7Kj0hlwUhQFIjCsCnWqEPOEPLH1H5jxwL1T7xULxKcrmMqyMJhqrYNStWM36DkA/HaYCUAGo/OVGJDHw3wATTowjgSmMpgYEKveJfrKAjgon8tiFVJMnsAaijgsIB6mqHY3usrXtrZW2WmSEUrwzBJD2Zw0FW9lxXNe0q92yrt5c2DX7khHe6StnerpgV0DKIolSdrjkZSVcZoVUyjE+flsKIQIBJGmJlQgML7wgnRUGRhK0Ay9vGL56srVa2vWmt1mNsLCPnxiNG9YcaJY1tK0LVeA/Adz5h5cAnvpNcXBts9x0QTW5DHgum8AcnbN1YT8pLXdGw5QicNUYhmczQu4I7u+bbZ3wTt2CIKI5YRGLZSpRHKtUGeSaxOhVBVIcK4m9TeRyAXDxsQzNKsySPjSq0HnLJILPOcPDO3pMv9kDq9v9mxvtC2+IP2SyxeEHM64W26+m9VcEWbHpGoyDKfL1qTL2bXp8CYWsCJCNL94/m/WOVE0tbtl99b6rZUdlU3WplZ7q81pg92nuO8HAURCUaYpszS1tCytLDsuOzvuaOmSABecnMP/r8KygAoBDAfokbvzzf+ELVUZGs4xoQRUogIZLPZPLIO229oMoEkb0F71F5GJgrP1r9ABPyaRaVQjgn6mYl1TzY89flcYxZCETOWouempBVqET8DiOfeBApSiaLs90tYW6eiItLaSHZ3yWTOVc+dyK3jWrnW89z4qFAIMQwgBQuAIjqMymTA7+4iZCUR/770AsIH9Fd0PPggQBFMOCbsihoWJENx9OMMyz679d8XhfUAEUFxo0mtmZ46frYkrxlmdu6XdKcohcGA5AAcogQzEF/RN8kCQP1iwcgmpXjLoi+gSZdzIZmnzrnz1AE2x8x8sNg2DQ7dUKZBrRZEQZUiR65LkmgSpXCuWqYUytfAEReoAErPVidm8xcrpgtesPINJMBg52GJdv616faOrE5cFInSYoldX9Vz/yD90BAYQdPrp34fy9vLrl1wPMKCQKOYUzMF/nnLAMIzFa2mwNAyLG2ZQwEkriqEe/+7xTRWboCiMph9JFJJUbWqaLi3PmDc6bXRJSkmCKoGb+j+9hNzgx5dA5edg3D0wM5VLS5VooeF2fBaMp+pzQfJokFgMnaqOMf8/Pogm+UD8/9Xt6kkdrs/q1O9ZssfW4WMZVqETF04xFUw2CcX8aMBzTsKEwqio7+vp37nLveI7sqMj0t5Ou9wsRQGKYiIRxu9HpNKYZpWWloYO1QoSErC4OGFKMpGcLEhJIYxGVCRC8KNfBMIIrTPQ2sNH3gmKufMYiqE+3vXJyv0rZhTMWDxpMfwEMOK6kZdQ5sbxCskc47ALkJDCtRs4SBD2ATqcI4ApEx0CpankBrToyp9r1j8oITU6v+/ogT6ptk6f2xzQJMrm3l0oVcIzRCjBhWKcplgqzFm6AIEYn3FLPiHCcAHGl0yd/fBXKZ7BwRlmPtvesKnWUtVmtwECZSUihs7QSWcVJEzL0WOCM5fTk5eQl5mQ2djROHvUbKkQZm2SNImj0DAPANDubF/84eIdh3Y8d81zd06+EwAgIkTZcdk7ZTvzEvNGp48uTinOjstO0aYka5JP414yFAi6QM9BOOOff1lfjQImAK3bQHcdtP0f86c+k/+yW0H2DKBOhZr1lDIQ0AiQk4i/o5VpbaqLBChCgGWU6ktmpuqTT4+bAQ/PaUg/ZcPhSGcn2d4RbmsjOzrIzg5RUZHhz/dy345wQ71j6Yfwqy4QoAIBwHFEJBLExxPGBFEeV8AOUcyZI5syFdf9Qj7iL8AeKVs8H92sKIZCEZSLqtIs8vwPLx86dMBL+m6ecDOBEcBcc1Pb2ivixImUF3T/CD8KTABbk4g1VoGi3NpaHQgQpVfcPfvfZ37PGYalScZjC9o6fLYun7PXH3BFfM4QTCclowmpCIJiiN8VDrgjnGbVJEhn31FACDGV4WgGlFR1+uMRPIMEr1l5BoFwXV3d19+/1a1pE+sELJZKe0uy4mYXJFxYmqEUn+kM9CRNUrwyvrGjEQHI+zve3928u6a75sUrXhyVNgoAoBKrut3dfpe/urM69pJH5zz66NxH9TL9IE/0DyDsAY4W+NO9H1b691aBiA8EXNC2kNOshBg22lalwAaGsXJTXRb8OSloknH0+gMeODUmUQoRwCIIS4ZIimKNmaqRM5OzSuL4slaesxMmFGT8QUytgubsUeMRy6uvkq2tZHcvGwpy6acsSTLhMGm26O+6CyHgaqKCAvnUqZhKFXX7h9FTQXISptUiOI5gR6eqYVmV9ES7aDKBQLQ9B8v4jmQ9nvuEqXBlR+WOph2rD6xeULrgjsl3wDCkve5WhWKZUjA5fVSEikDNKtGobLWqoBsoEkBcAUydjx8OU+eNJRp7wwUfXz6OaRWWXHaGkwB6WzyN+8z2Lp+jyx8KUAzFUBTDUNEoOIKIZYTGKJQqhQqdSGOU6ZLksRx9QojxE/fnNLxm5TkZgiS9p8Wxrqb36lKT6asPrF99pejoKBx1Q3x85JJM+aQppXk5yUBwpv2qSJrsdHZ+Xf51vbkeCMDn+z7/7MfP4HQ/BarHV/dpVonq7xf9PTwtXJp2tLo/UZ14WnaIZaAfqq0OKtTOcmBvgGUK7g5ARaBSxEVw0l8ZD5VrjHH3nFwwlWVYmmYDnrDHGjRmqVEMStGQn1zzv0q3LTjl2tzhk+AxsiwQivGSC01FF5ok0cADz9lMQ0NDXV2d1WqNi4ubOXMm1k91mc3m8vJyHMcFAsHEiRP7P3VysBRle/110mxmSUp91ZWSkkHtG3y8N2bhuUtRtMtFtrfD+v32Dqqzg+zpxfWGhKeewDSwUpt2u33r1tM+H5y+JwiYfiqV4vHxgpRkcV5+rNhFXFCQtOSt/jmpp45v61ba5UYwzLdx05n7WE4PESpS21O78uDKTYc3VXZW2p12QAGcIG6deCuO4aC3+hbzT9frcW3pJSA6SQVz5ac9Be+r9cOAOgVmJR0Bc7ZKIm6AocBcCV0CThu9LZ7Oww6JQpA7LoH7Q/c2u/esbIlm+KM4jqI4qlAKtIlSTaJMlyiXa4QyjUiqEuLR2xie8wles/L8DiIUU9Xl3t5gXVnZ02jxuYKkVETc7nGHG5qEyUlPZAPDtePkwzIBdkZjq56g52DnwT0te6APS9s+f9AfLVcCLM3q9fpUbWqqNrX/RP+iUYtO7w6xDGjZFq303wO6yoG3F9BhWOnPMDA/VaqDs/zqVJBQBMMViSOBUHn0tScsWIPeiN8dCXgiXnvQaQ44ewKuXr/PHUZR5Iq/l6mjVlZCCQ47XwMk6I1wr6IpJi5DOfbSjNNy4DyDSkdHx8GDB2fMmBEMBp955hmapufNm8c9FYlEvvzyy3HjxuXm5i5btoxl2QsvvPBU349lvRs3BffvBwgimzgRnDZxRjtdtMtFmBK5NqdMINB5/1/C9fW00wmORE+5H8JkIs1mTrMSRqPysssAyxKpqcJo9JQwmVCJBKaf9puyh/HUfgmpg4JwWLby0ksAihKp0C3/XMQddFd2Vm6t27ry4MoDHQcjkTBs0YcBrRAUyohZORdSDAU1a8p4Rdp46E5FH3GfRXF4I/2LOJrhWCdWAUstYOlBqbgKeCJ+V9hjC6riJVpjX51u1eaOys2dCZmqYaPjsaj5aUKGMjlPI5ITmgSpNlGmM8kUOjFGoHxC6nkPr1l5fhuaYc2e0Jqqni111ooOl8VPoQQuRJAMnVQhE8mvuJLBCflFF0lGjjxju8QC1hP07G3Zu6py1d7WvQc7Dvo9fvgEDgQiAcMwVIS6tPTSR+c+mqpN1cq0p3NXWMCQwG+F0lMYHWQZEqx+AHTuj9r447D7FC6G5VOmUihSddlQsyoTf18cimJDftLVG3D0+F3mANeRxe+GeVqRYN/VBcUQFENxAeqyBDnNiuLojFuGAwRo4iWxlDzefuVcwWw2//TTTzNmzIiLi0tLSztw4EBMs3Z1dbW2tl5zzTVisTg5OXnHjh2TJk1CTy3bEiEI+dSpwf37hVmZ/VM/Tx6W5dJPYeOoaPQ00tZKmc2Uzc74fIn//a8oGya9IBgWqqoiOzsRLoAqEuEGPZGcTJiSRFlZuL7PE4NISDD++1/gj0B3++3g3CQUCext2bvywLfbm/cc7KwM+PwAhffzYgSME4G5EnS8RFwo0whHXwEI2NML9qla9Bl00DsRihaBlLHwNluiObnZob6EVGvQ1uWzd/qcloDfEfa5wm5LoHhWyqRFw7jVNEaZVCkQCDGGYTB4AECfJJ9zV6FYfta1nuE53fCalee32dPqWPxRud0fibAIiuNGyptnaZ03f9LsGaVKMY4iiOgf/zjDuxQmw1NfnHqg7QAdobmAQZw+riipaGLWxAuGXfC3r/62q3pXvjG/JGZwffrw9YLProaRhrmvgKIr4RKUAEmjQcgF9DlANwwklYHEUqBKhoULJwILgn6SJhmZSsglm3bVuzYtrQ16IzTNcplbLN2nQAViXJ0glSgEEoVAoRNrEqSqeEmspgpFkeT8Pgds9oha5TXruUJhYWFqaqpcLodmFxZLfn7U9jKKzWYjSZKIximlUqnNZguFQhLJqRrrYioVYFmoHX//3DobidAOB2m1CZJMcDsAsDTd++STnjVrOecpGDqNRGALDFjHg4br6znNCnBcvWgR7XYJ0jMEyUmC1FRcp+Nm/3/2Bvx5e+L4eiub93xc/vWq2h8a3BYqHG3IhIJEAhRhYIZKMSc+LVWfgRtyQdIokDACKI7cQiPIiQpWmGiVBH9+DzTFwBopJ5wdcnT7rR1eR7c/5P95QuqRCSKWYTmDqrwJxpwx8SIpgR2Z60cxhBesQxNes/L8AmGKbrT4UzQSmQieIUaVRCkiXN5AsbdnUtvewp76LOBLuaZIdKY6fDAs02huXH9o/Zj0MVyXKQEuSNOmlTeW6zS6ycMmzx4+e2TyyILEAhzD97TsqemsASj47uB3j859VMTFDwZhJ2iYeGo5BNp3wkap058CcuiDA10JHS3Aawbd5X2aFUHBpL9Csypt5m/qVNgPkGIBAmJegFXbOqs2d6rjpVNvyBVw/lMMC1taMyxGYIQQU+rFaqNUFSfRxEth2pYSCtbfHMEjAcptDSAo4nOG/LCKlh/xz3YEAoFOpwMA7N69WyKRzD3i0wRTt0mSYZjY7QdFUTTd591zSnBeTtF6o99WqG53pL0d/rS1kZ2dlNVKO5xkV1fcE4+rLrmEm6anHE44uS+XIwSBikR4fDyRnBSd2U8WFxVym0JwXH/vr8w+85wIDA3oEAh6KKkezu/DXkzbf3z78ufd0WdRoBKAKVLpRXGpozLGDU8pRQw5QJkCk1NPjUiIJkNwkgcjUJH0N64FLM3u/K7Z3OL2u2GjKZitFD3FUJxLSEXkCmG0XkqmTZDJdSKFVhzzihbL+FZSPH3wmpXnKDTDVne5tzfaVh7sbrb6l1xfOnmYPnT4sOizz6/b16q2dg+zNcvUCvnMKYobbxRk903cnAE8Qc+0/5vW0dNx16y7OM2KIuhfZvzlkpGXzMqfpZFp+reh8oa8cYq4BFUCgRGcOfbJQ4WgHnW2wQZUnftgvX/QBaggoEmQPRMMXwDXEUjAxAdgrVX2jKMvVP9qa8dIkOLyUH3OkMsccPT6bR2+lOHaC67sa1UQCVLtNY6gl/S5wpqoZlXGSUrnpIqkAo0R6lSRXIBhyO81/w/6Is5ePyFAvfag1xbgNeu5QktLy6FDh+688061+mi9syBa4MhEJSbDMARB4IOYxIkg/WOctMdD2x2U3Yap1cKMvkxo25Il9jfe7HPvj0RYkuRkLhuJkE1NsdcqL54nGjZMkJ4uSE0hEhMxmQyaT51yuRjPz9j37p6Nz73tCfQYS76640t4ly43XhxnfBcPpCcWzjUOm5E1IT5jAqpMhEWfg8eBje17V7WgCJJWpJu1uKD/U5EQdejHbmunN3es0ZQTPW8RYGl1N+23ojiCE6hSJ5ZGe02p4sSw15RJptSLMRzlikd5eH4NXrPyQCye8PpDvRsOmQ92ujudAc7Zble7p6ynuutvD5M9vZMBi2o00ssXqK+5WlTQFyA5TXiCnu0N29fVrFs0atGYjDEwyUqsGJU6qtfS6wl6SJqEDiwAjM0YOzZj7LEvH5sxdt0D6+DdP4KJf96/9ISgQsDVATr3gs49wFoPK/1dbSASiCanCmCxv1gFNJnwAQeC/WqNQnQujAzTbkvQ0eNz9gbclqN5qEFf32WephiFRsQwLFdAkDhMM/qStPg0JWcoCACQa0RjLjnVqimZSnjRnwpZlkVRVHOkuIHnLMfhcBw+fHjhwoVCobChoSEjI4P7C8bFxeE4Hg6H5XK51+s1GAwi0SDNJyAIEwr7fvyRDQYjrW1kTzftcFJ2O9ndrZw71/jcf7i1MLmcdrtRuRwRCDCtljCZBKkpgqQkIilJ3C+HQTF7Npg9e3B2jAcOTZ2gax/o2MVoMqnRiwVctSsd2dHR9I4XAMuqqq6qstQyEF9ouv6rFQxiSiwExOlqxQx7k0ajrSiGNuw1O3r9RVOSuIArTbE/fdsUcIVlKiGnWREUyRipF0pwQ5pSHSeRaUQylVCi4O+ceX4fvGYd6tT2eN7a1rzlsMXuj4QoBqBocry6MF4yI8cwoyiR6GQAiuFanWzWTM3114lyB6My41c41HPoYMfB9dXrNxze0OPsYQIMwzKcZkUR9Ml5Tz48++ECE5z9P/52JAJJui79d7990AlqV4K2HbDS39kG46YwnkpBqSpSgcRh0DY1LmpMaCqFtf/H9UZgGPbwT90dh5wuC6yXokiGphiagtmoXGopASMNIrFCIFEKZUpBYraKZVgQ1azxaYr4tMFvEYkLsKTcvtxWnnMCu93+wgsvRCKR/fv3O53OyZMnx8fH//e//50/f/6wYcPy8vKqqqrKysoaGxunTp06aGnKOE719PQ+/gQXQAVUX4UfyzC0xwOTB6KVXpLx4+OfelKYmSVITsLj4mD6KYEfNRXmGRRYBt4wu9qB9TDo2g9vpN2dtaHgfr/vOyB17V694u7lIkII0i6YP+X2zw6X55oK++7ShTIkeazpNO4YsHf73JYAgqEYwTZVWOr3mlkWuj5zg4xQjGWM0Lvtof52/cMnJw2fnMSnJfOcCrxmHYowLIseGTmarf5PdrehCCKTiYsMsmliX0HF2rEX3yjJjZpDZWbFP/kEptNJT4/xTYSKtNhaVlWu2lq/9WDHwXZzO6BgQRUiQHQ6nVqqZlmWuxgPNw0fzDdmaBCwAVsjiM+HkhRWLZjBhieAvRXgGCyhxYQw3yuxBNYo6HOBJjU63f/Lw23QGzm0o7u3yT1iejJnWI0AUL/HXL/XTAhxjEAwOLJjUgWuTpCo4iTqBKlSJ5YoYZhBqhT83ll+niECiqIXXnhhMBjkHpeUlIjF4lmzZsXFxaEoevnllzc0NFRXV0+cOLGg4Gczs6cKy8JTVibDtFpBWiphShKmpuAJCYLko4ZxosxMUWbmYL4pDwcVhu543ftBx15gOww1q7OdDrq6WXRVmN1AMvtJpo0EgPQDz/e1PYdGJo8E8YXJC99a5bXq5X0eC6cJhmG99lBrla3jkMPW4fPYghgswUUYmsVwVCjGwjEPExS9YFE2LsQIwdE8EF6t8pw6vGYdWlR3uX9ssNWbvQ/NGmZUwjvy0enaK8ekyYX4LMKZ9tP31A/f093dHhmQPPssF1BRzJw56Lth99srOyt3Nu5ccXBFZWdlMBQEJLx3J2REYWJhcUrxRQUXTcyaqJKoTiZ65GoDh9dA3UiIwMjrAHpk0KRCUIlyG2zdAZZdBU2prv0KpE2CS9SpIH0yUDVFg6mlsNhflwWwo/OtFMUGXMGAO+L3hJ29gaA3UnRhkkIPP0MyRNf82N3T4NIny/s0K4ok5WpoilXqxap4mISqNkpkKhGfsMVz4qjV6hkz+iVJRyk5cveoUqnKysooihrMTNboiY7r9XGP/kNSWsqnn55RGjaA6m9g6ryjGdARQIdomj5II3sp8F0I7IhEPAwADOyQIpAKRqSNGJM5RibqS/JBADh9gpWhWVunt7fJ3Vpp62nyRMIUFWEQwIpkBMvC3IDi6cl5E41yrRgXHKkrQABf189zOuA165DA4Y9sOGT+odZc3ubscAQYFkzNMRiLoN7SyQRPDxewX33u2bDB39bBogiqMwDp6cp33Fy3+fM9n+9v21/ZVRn2huFYSwChSFiQUXBx4cXjMscVmgpPdfC1HgYr7oapolI9GHYRDFq074T2/ggKZj7T18dFooaNUl0WYGvq06y4CMx6FkjlAIVGUQwLqAjjtfid3X5nb8Bl9vuc4YAnEoDlU2GKYjEMSc7TcppVrBToTHKBCJdpjmrcwilJI6Ynx9wAeHhOB4MsWKNBVlQuExcV4drT6Wo8xGFZ4DfDcUkaB1LG9S1sWA92vQlEUhLBmkh6eRjZEgYVEdJCsdzsEyEQ5CTkXDbi0guyLxiZPFIjPe2pPpEgdeinntZKm6PL57aFAIDxVEKIxacp00fonOZA9bYuhmLICK1N5FPkec4EvGY9n7H5wlVd7nXVvT/Ums2ecIiErp4JClFBklIqwlmaDh8+7PziS/+aNRGrlaVpwmiUTZ0K81aHDRusiRyKofxhv1Lc1+rpu4rv3vr+rWjnUmAymopMRRdkXzC3cG6mIVOAD9J9OWwzmANjFQIpWDIJ+Cxwui3ih60IS27o06yaNKhf5XqQDBu6cgSQ+N5ql9vS7uwJOHuhdX8kSDM0l4rKcM4sCIpEJ/SFQgkeawxICLCZt8Hukf1n+QkhH53iOTeJ9gL4o3fivMPTBYQKIIx6J9MkWH43qPkGFF4Fbfm5pKOsmY6mLR/3tnwRZMt9vhDDQOdpBsgVipFJIyZkTbhs5GUFpgLhCdo8nyywcx4KS/s5aV2/u7e9xo4RmFwj1CXKEoepUwt1GqMUw9G63b0uSxBFQDTg2pfExcNzWjk3NOunu9u21FsRACZm6q8de652zxt0mm2+2m4vwwKDQliWoj62bd3Kg92PLq8OkQyOIRICG5GkmjcicUKWvsAoJzCUdrt7n33Wt3kLKhBgOp181iz1oqvEwwczbXTpT0s//unjfFP+y1e+zA3LM/Nnfr3/60nZk2YPn12UVJSfmI8MSt0GGQS2etC8BVZQWQ7B2CqGAn83oGgYTxVJgTEfJJcB5ZFm2QKZL/0qc7Ob2h/MKpVzH52t3bX61Qo4ZEf3CM7j4yiGIwKxQGkQq+OlmgSJUi+RRstdBzSz5uOpPDznLbvfggMLgoKMKaDwihN9FU3CxiI9lXCSx1wNx6UJ94Pi6+FTuACmy+Mi4OkMh7yEUAbd+tIneRd88NL/FrSZ64AQiETikuTiS0deOi5j3IjkERLB6Sr/jxEOUFVbO5srrMUzUzJLDJy3f1KeBsPRjJH6hEyV1iTrP+hllcVlFMPVEJTvVMJzhjg3NGu92fflvk4EgDjFIPm5nBesqex9bt1hkmYuyDJ8fvtoAEBNt8fpD4/L1HElVpkGmYjAMg3yMWmaeSOMpek6EQoARUI9BwCmUEjHjQ03NMqnT9fecL1w2CD4rXa7uxEWSVBFzfYB2Nm884e9P+xs2fmPOf/gZvxn5M+ofqpaIVYMjlT1W2ESWNsO+OPqgBmrVBAgKIPLvLSJFBh0E8ay+mK/vCiI6LxeIHYJE454XNZsbtm9olmiFKYVagXR1glKg1iiFBBCXKYSSJQClUGqTpCoEyRKnRgXYLwq5Tnv6bNlRVEul52nj/adYN9SOGJJNL+hWf026I7naIFFVPD+uQbO8EATkjCgWNC4oU+zAgDKbrUlT3jj8I/f/nvyU5c9Na9oHsCIFGP+gsI5B5WmafnTLh15aYYhg/P1O32EAiSOo3i0Uopl2ab9ltZKm1guyCjWczJ09Lw0KEp/KQsfRZFjAyU8PKeVc0OzTsjUfbizjWHZiVmwKwwPx9Rcw8s/1DMMyIqTrarqWV3Zs7vFgaPIstvHZOhhdtGIJPUHN5Wl6mRJaph2GWlo6H7nXVShiPv7I3A8QhD1NdfKp0wV5eWdym6wLOsMOLfUbVlXs25Xw65x2eP+d+3/OEl6cdHFh7oOzcifEZv3x1AslidwMsDQhRtItX2zae4u8PUtgAwBHAdCCZApgXYsLdRbaxtaAmVOeqTQNsNfHww4AwF3vccWyCwxzLmzkJvBVxqgEkUAG/SSnGaVa0Rz7iwUSQmJUigQ8TP7PEMMmo7s3CPEBGh7N9XUIjAa/+gdOmvImgGrozAcZB/jNUuTsIS/pxLapnYfhILV2Qq83VGbPAS2wcOEMLE+fjhIHgtSxgcifgEuxFEcGHI9uGzJx/d1NrevzVwLNWuUR+c+KhPKTrdUpUimt9nVetDWWecqm5PKhUtFUiIlXyuSElllR+ajYJdU/u6F5yzi3NCsEiGGIrAsZnlF9742V/+mhQzLaqSCK8uStVKoiize8Jf7OjwhirstjLVXZwGQC/FbJqYR0W+gxRtetqedZlio3I68CwIQmmVNavElIxKF0aBauyOw4kDXgBbtSPRNkzSSeUWJ3E1mo8W3sdbc9yz3ltHHFMNk6OUz8vu+/3W9ni31VgKN2Uz16S6GAXkJirEZfRUPlR2uig4XCnesb+e49RkWDDcqRib3xQkPdrj2tTlRFMEx5LsDXR/vavNHKIYFWqmgwxHgNKtchE/MgtHNUE2N87PPPWvWUFYrIhQq584RFxXBP79ajfdrrvO7oBm6urv6YMfBNZVrNtdttnqtbIQFJOj0dP5rwb9UYmggNXv47Ol50wdt/G37CfzwBHC0gts29fW51g8DqWMAwoTFOW5JkVtUbHareurtDpuTQYRsAETWNzAMG/2bAKGYYGgG5l1FP9aUfO38h4qVeolI0vctQFEkPv0U9DQPz7kMTVN7HbWuZJQOWye4u0+1icX5hEgVnf9GgTz+Z8tZGqy6Hxz6DnZ1hpHUEMwARVEgi4cZ8+pUkFAEkscA44gur31/96GN+79f++6dL175wtwi2IY3XZN0VekVNQk1E7ImxDaplpzkgHwi+N1he6ev/ZCjtdrusQTICEOGaLlWlD7SwF1lRl8Mna159z2es5ZzQ7OyLNRtDMt+Wd5J0Uz/WWWKYtMN0ul5cX2a1RN6fVNjpzuI//xbx7CsUSG+fnwqEQ2f9bqD/15zOET1ZS7GCJPM+CztzLx4TrM2WnxPrKxhmIH1SCTNTB5mmFNo5OTnwQ7X37+tYqI72Z9ghF5QbIpp1j2tjvuWHRAfU5oTJpnFF6THNOvamt5n19QO2H8YJaWZv0zNimnWr/d3vrShQSeDR+3wRwgMHW5UXlSQMHmYPi/hqCN9uK7O8fHH3g0/kF3dCIbier3iootOpSI4TIXreutWHVz1Y+OPBzsO9lh7YKEABlAhaowzzsidMT1/uvBIh0AEQU5SsEIPFT+cX3N3gMIrARo9UVkGtG2DIdXuPX2aFRc2Jj3b0kC5nBqvNeC1+hjKi2AILpDCJwnEYBSroR+qRB0nkaph5xX0yKSnWC7g3Vh4eGKQCPtCNtmopiIk/Zqa5jXrUWCyRHRALn8P+B0g/zIwbBb3BFSr7k4gVgKREhhGA2MxdMrTDwPaTCBUOCPBLbUbV269f1/LvqrOgyAEAA02HNrAaVYAwOPzHpeLolVZp5NIiOqqd7VV2Xqb3fZufyRAwSorASZVCZKGaTJKYZCVg1erPENXs7LRvpSDmJqNAFCQqFSKCTq6ZQ6KZhOUQjEnRQGQCvCyNE2qL4whyNGVoi25tTIhdmRnpAJ8VJomQjNcCJbb36jFPZufoIBB3ShKMVGWomH6byi6JkWzw+JksQPTSImRyep+O9VHiKLTdVA8HVlNMCJZJYruarQzd98LwjSbqD6aX6+XCwuMSm4f+tZg4T+SZg390nkNcuGweLndH6FpdnymbvEF6eMydOJ+Bs7h+nr70g+9339P2+0swxAmo3zadM111wqz+/ra/y56Pb2VHZXbG7avrFx5qOdQJByBjqoAiBXiImNRaVrpnMI5Y9PHKiWnFqcM2GEplbkGtG4H7buAtwcIJCB9ClBEE2SNxZ7kq92kXkJl9oluBO2wxVXubkGRboAAqVKo0stFMtzc5g14IsNGJVx4XS6Gw4L+U9orHp4hAMpil4YeoGmUopkselBbeJyLMBT0HnF1gKRRP/U2dftIDI1M2fmmMuCCSQKcZkUQUHQVUCSC1AnAOBLI9LCTMwCNjs791Ru+r1yztmaN2W1mwyz0VRWCnNSc0tTS6XnTY29yWgWr1x60dvraquztNXafM0xFaIZmcQGmT5Hrk+SpBbqkPLVELuSbl/GcQ5wuzRoKhZYvXz516lS9fnCMjmFjSxR5aGb2+Ew9zcAGmBwsbOyJyIR9B5Kslbx81Ugmph+jD7j/oAgiOJKak6KVvHNDaX9RGJORBI5KjoRChycqP7hpVHSdo4KU27YAR2Pp56PTdR/cFLNMOrpJlgWSfiJyUrYh/3Yl1E8/f1eWBQrR0WDkZSNNk7MNgJPc/d6bZYFacnS1RaOSRyUrb3lnZ5CiL8vXT809moHE4d+z1/nBBwiOYzqdYs5F6quuOum81U21mx5Y9kCVuYr205yjqkQsKR5WPK9o3pj0MYWmQpUk2krq5KAjsDkhrKP6CVgOA3s9bEmFsoAQAkzCAByxNSJRzcpg4q2Rv7ZUOXLFgumFfa9OSJPoTZKkXI0hVa5JkCoNEnuX/7v/28/QTMchx+DeNfHwnMdgOGZK0vU6QpgwJFGd3nzKsxQ6Agefzn2gYzew1AJ7M5znWfTpM9Wb1/b4gRCsNErnJpcAXb/b/uxZ8Cdq9m92966rWbemak1F2/7GnkZ4V48BVIAmJSTNzp89q3DWCNOIVF3qGTiOrjrnoR09lja3oydAhWkEQwgBptCJk4frkvM1hmS5MmoszcNzznG6NOvevXtXrVo1ceLEQdna/nZnkKQBCxosvlnD+2rSfxEMRfoLu18Dx1C9/Ldd7oQ4Gq/8bacCiQCTCH57CJAJ8Zi2Pg5qCXEih6CVCX/aU5HRVUcBpG6zL1JiEohF4cZGYVoaiPatUc6b6127lkhN0d54ozArC5wwgUjgcM9hi8dywbALOIMVISGsNlfTITotKW1E0ojJwybPKZiTrE0++URVmgT2BmBrgMHUtu2wCwAVAmQAIAwgRKQq38UYPUyiWTSpN2AqtGdkwiQrmE5G4HTE6yMDcopkONeV7NHxmSVxR/uvABDwRHLGJkTrH2KBdR4ent8ARYE0EWP3s4wsgsqGjD9rwAEcTVCedu2FplTWuuhYFIRN8lgABGLgaJmRNHztvq9RgDBTngQTbwHYwJyiNZVr3v3x3U11m1wBFwjDMAMmxQpTC0dnjJ5bOPeC7AvOQAJAyE+KpH0Dsr3bV7mpA8EQgQiPz1DGpSpSC3WJ2Wrhkdx9Hp5zlNNyBtvtdrfbbTAczZI5RdxBUiuFEtPhjwzWNs8DesWqBrWJAqhepQ8eqnV8+7X3x+3Gf/1LOm4sZ2Vl+u9/Me3v7pWyrnrd4vcWh5DQxgc2lqWWAZiSUfDvBf9OUCaUpJQMiz9lSyxzNVj3KNSp7mZ4YcAIAK+QYre0xIyN7nXrncEUD2vyulkqHGZISpPmzYR7ASOmI2ckZxTrdSY5dsR7BfqtCH6mTHUm2bSbTskJgYdniBKd+6EBDXtpsMyRQtDz686PZaAktdRChdpdDuxNMAcAVvrTcK6fEMNKf3UKMJbA8qm4PBBfkLZnGbwHxpBUQzp8FpY0kCRNxjxTt9Zv/WbbN0AMcCGelZY1t3DutLxpRaaiOMXA6a/TQVu1vWpLp0hGTLp6GBGd2UvIUCbnaQxpiuR8rT5ZLlOd3jYEPDznsGaNRCINDQ05OTlbt27lUlpPnb/NynlgOpRKMaXCAwC4Yfrw0c/cw4TCotb03g+7aIuFCYfdy7+Vjh0DB18EORHB2uXs2t++3+a1XTnqSm4I1sv1johDCITdrm5uHYVY8cCMB05yL4NO0F0BTf6zZwJJNA0VQYG5HLh6fIJhbiTBEU7oFUw0B0zegIAGQopCWIoEjE8sI/RGiUwjjEs7WlIWl6qISz36Xx4enkEEBVgn0rwq9Pa6b5RqpQqD+QLQvhPHcAyFjwmUgA+iS3AMRxl0QdmC4Yl9ya9rqtZ0WDsyEzKn5E7hlG6ztflg50EUQTEUi/0+9r8oimIIhqFYhj5DJoKeJzRD97h7gmRQJ9PFSul9YV8oEoKvgplZA7d2ollA3l7wxQ2gczdMt6LDsFACw2GNvzodFvsnloCkUcCQNyCYCotCYQtV+BZf7PvitQ2vGbXGj2/5GMfgNfSSkZfsato1Kn3UxSMuLkstExGn10ecZdign5QcKSF1mQO1P/VIVYL8iYkJGbCiQJekmP/XEt5Smuf8Y/A1a1tbm9FoVCqVg9gLm8DQI0VWQw6WhlEPlqKZYADXaDhvAtrptL35FtXRIQz6WJpmqysZhsH0evWcOeqrrvzNbTIsY/Pafqj9YcOhDeWt5VWdVTKRrDCpsCSlBABQaCr8+NaPE1QJI5JGnOROM1RfpT8AoHMv+PQyWC57wzqQOgnuvDS5Rvpwe5vbS2S4aWMwiACaRACc68dwWhUnjE/XxKUruaZTMjUfIeDhOUOgAHED6yF676HDXK79kR+mf0p/n0kfJARS9akxzfrc989t/Wnr7MmzL8y5kFOQyyuWP/DRAwCFm4avijZM4kQnJ1K5B/A3wEQC0ae3fTo+c3x0bs19z6f3VLVU3T3z7vum38dt/43Nb3yx+wuMiErVIy88Kl65JdwDFjOoDH+f+7gJR8GuN7q7Dr7GKm2o8LaJt5aZCqBIJUOfUXgdIhVJtLgsDlen4vosXJWC0wKsrQZvP0SgUZmOYjgusjRte0iGjhKgqT17Qd7M3c27fyz/0ZhqbLQ25sTnAABKU0vX3r/2dLeqYhjW5wi1VNraquxiOXHhtTlcL4CETGX6SH18ulKuiRm28J35eM5PBlmz2my2PXv2DB8+vLW11el01tbWyuVyqfRo7TzP8dy8AGDD4WBlFdnbSzsclNVK2Wx9vy0WwLLpa1ZD2QrvsxnH0qWs14vK5IBFML1ePnu25tprhJmZx3mTCBWp7Krc37Z/TeWaHxt/dPgcIBK9GglAvDweZmJFUYgVV4266ncfQsQHbI1wxq1lK6AC4KIXgBQmh9CSeDdR5rCxektIGS0/QETSFmpyvdeGowyGRVQaoUKvUMdJ4tOV8ekKhV7Svz0gDw/PGYMBjAbE34I+YUd6zKCDFAQEUgyXAFwCgIBlGJqij8DQNMtQISpe2WdZyrJsqi61KbkpXZ8eSyeQCqVqtZoGNMMwXL4BfCHcDtWng2O/o5X1ITLEvTBMhqu7qpuam3rcPbHdO9RzaF/NPhDLou9vdD0gxkoCTZzm7mn3w0tc+VJzb8v/nMAdAOMzxpalloIxfwLFN7639pUNtTuB0AOQdsDuhjkRx+jyaGYEgjPMaAF6vQQRU3D3FpQsqOyonJwzWSPtm8gSYALBMUmugwVDs7YOr7nV03zAam7xhAMUGaYlcqJgsonzkzYkKy77S/Evtqri4TnPGGTNKhQKhw8fjkVrgCiKwnHoMzS4b3HuwobDtN9PWa20zX5Ej1oom50Nh3X33iOKWlDRHk/vU08F9uxBiL6BmYVz6SiC44hEQtlsnGZFpVJx8Ug6EgnUHmJI0nDNooR7//xr7xsiQ9Vd1SsOrNjRtKOys9Jmt8HLAw5wAW4ymi4quOjCnAtHJI1I1f7+glaGBH47aNsJ2neAngrgaIBWhSxGi+PR0a1IVLMGBCmbIn9vteGTrOnRrFSAYogxXUIGpMZsVVyqQqEXqwwS4hjbWh4enjMMA5gEJDUFyWIZmF1EIKgQxhoRDAWEEJHrBDKdUGEQyTQCTIAAlGUQWssl/ETTzZ+97JkHZz6oFqtjw/6lIy8dkTSCYRlOrcLfLNSvcAlz9DEnZLnUee6FCrHi6Uue7pnYMzYbZuf3bW3EpXqpnkFYCrAUCyiapCI+Ouwjvb2Uq4P29sIlLEuxbMQwPD6xUC6QALkWpIwVy0wjbO42v1cvi/rY5F4MzQd3L5dKqxAMYVnYaoST3dBX8IhRI8stAIBBkD0kfZmXWJ4xKx+AUamj1t6/FvayOq1/C5p1mf3NB2zd9S5rh9frCLEMixEoIcINqYr0ETqF9kgGAgIH1dO6Mzw8ZwmD/K2Ty+VFRUUMw/z4449er5dhmJiF+3kPEwgwPh/t8zFeL3zg8VB2h3TsWGEGLHpnKarzvvv927cfne6HM/4UIEmWZZXzL+M0KyqTYXodkZSEa7WoXI7KpJhMjqlVuF6PGwy4tq91LSoSpSxd+sHWd158Zw1NkhPQXa/Tfxpwox+MBDcf3rylbsuqqlWNlkYyTIJoKbBcJS8yFY1OhwWtpamlMiHMHvt9OFuhi2pXOQyp9h4EERegKD+l84BEBz3KTA6zOFImugyJyXBdsUoOpHqCsAe9kagAhwtLLkotmZ3Kj7M8PGchJBqSG0SARP3ukKeHMxCA6aIYgWIYiuIIhmNyjVAVF83eGQ9ER6ptE1WmRJWp/6biFHEnV4ckFUoXjV40YOElIy+5ZOQl8JG7A45CvTUw9chRwwbMAKdYBQoYjGVoIFKy858AeZdg8OqDgIXvDwPIuqhu7j9IfnDT+xQDjw6KZpqmGIqkSRgDZiiKpmK/GZbZcGjDI9880kyzESFMpudyWE8TVIS2dvh6Gt3NByy2Th8ZoikS+ohL1UJ9ksyYpUobYdAkSPipf56hyWn57pEkqVKp7rzzTpVKNVhlWH88DANVJk2z4TBpscCup7o+Ben48EPflq1sJExH1Srj9cIHXh/LMInPP89pVgRFAcvSHg8qFiMYBoRCWMagVuN6PaaCkpTbFCoSxT38MBuJYAoFKpNjMiki/OWEToQgQi3NeJhEAaDaO9hgCMgEDMugSN9YFiJD1713ncPiABic/ZfL5KPSRs0rnFeWWlaYVPi7pSoVgV2pWreBzl3AUQ/srUyEJBGFjc7oIWdbqQwXZfTScX5KARiajsCIcGL0dTiBjr0klZ6TrEmQxubvYta2PDw8Zw8wospgpMBfvDAtUZcQ8IT9rrDLGnRbAm5L0G0NUWFYMB8OkV5XoLPBjmJoQpZcaYBOf2SY3rW8yecMDxubkF7UNzay0Y4sg9BdiaGhKVV3BejYBXuOOJqgZg1AA2aA4gghArgI0WaApDGwiEqfAwy5AD0ydQNjwuDYYfTEK6W6XF0IQHAAEPaoNfigE/KTB35o725w2zq8fjd0yMFxVCDBTMPU6SN0celKXZKcz5viGeKcFs0qFAqLou3szznYUIhlGFTSl0ofrm8I7NtL2R2UzUZbrWQ0u5Rxu2mv1/DQQ7rbb+NWC1bXuFesQGPiEsNQmYxITESlUkAc+YRRVH35QsnoUURCAq7VQqmq0aACIXSUxTBojciBIKJhJ2omtTBuPNb7GRUJl+aMDLPUu1ve3FCz7v7p91+QfQH0Z5Uop+dN3yfcV5peOjV36szhM40q4++bzwp7QDSuALHWgo/n0AHSTeo9jNHCjOklc61kRpiV0gxBMRhgaBRhlGpWoZMq4ySa+KPlCInZp7GDNg8Pz6BAMfTKzs8akB4qGCoEdxcYMzXGI6UIbNT+imK9jqDLHNWvloDXHgYIkOv6lF/AHW4st1haPfoUeUyz7l/X1t3o0iRIVXESmVokUQjEMkKsEJxQmJAMQOcp7k7X1QbemQpFKhWGxf4sAIQItkhVpUB5aioFptGwnzN+Wko2Y5EXLKaDB4lwkCIEWN+ME8tWb+ny2EMojij1Yn2yLDFbnTZCr9SJ+SkpHh6Oc8NhmCVJOI0OAEzrPJLoeSow4TDV2wuTSq22I3mlNtrpoh122cSJurvv5lbzbtjQ8/jjqAgOynBaG0EQHIf97mmKtlljW5OUljIuFx4fD2fwdVpMq8VkMlQux2SyWCwW9hSYOvX3T8P/AtykVZAM1mar7y+0hcnQ6zPysjDkvmX3ki4yVZvKaVYUQf81/18UTWXF/Y6GAn30VoJdr8DrxMKPgAw2cSAVmbu9f7L5tG4m0UPpSRpHGBpFGRxHhBI0Pl5szFQakhXKeIlSJ+adq3l4zjlomt5oXn0YqQVBcL1v3s+ei9ahYzjQGmVa45FhjAXhAEmI+r7sGIGmFemEUlxvOrpC52Fn3R4zEW35gQsxqUIolhNiuUCqFMBE9jiJyiBR6MQCEdRtR8Ox5hqw63/A1QoufQsoo8kGsjiAi6G1qiwOxBdA59SEIqDJAJpUQJzean3OsYtlWAqh6s31Baa+jNtTJOCJlK9rNTd7xl6Wwd3VC6WCjBKD2xrMKjUYUhQ6k2wQ4tM8POcX54a2sLz0kvvLr1gEqC6bH/f3R46/MkuSsTl6mFfq9dEuJ9nbi+v16ssXchNG3u+/73nscUBRMMeJpuEDigIMwwSDsSArrAZNTxNmZ+N6PUwzlcswpRKDqlSHa7WiHGhxwqFeuEC9cMHRWOng4Qq47H67w+ew++1Wr7XV3tph72h1tLbZ27rd3REqEmEiAAUUwqol6qtHX93Y3Zgdd7SvYJou7YTehqFgfmrADlLGQ/NUAEiv175rvytiSMovl5bNhdcsgbgJm2fzhVFAC0QgLg5X6ORakyw+HTZZEcsIlM+v4uE5lxERopfHvr/ly0Nak3R62uTffgEChEcaL0FVqRZNujqHoRn0SItshmZMw9QAAeEAFfREAp6I2xZ0W4PwORTBcM7vCj6QqYUKFWYabiiYbIIh2LCX2fshy7JY1w6gjJr3EWIw69/wt6kUCOVHrfTOCD8c+gEEAIMw62vWLyhZcNLbCXojQgnRFzRFQO32brc1pDXCeCpnUDVpUTZAYM3tYO49D895xLmhWVG5MuLyMAgGFCo4T9NnWUpRTic0gaIZcVEhIoDJ9aFDtZbnn6fdbsbvjypXL+3xwjonihKXlqoum49EDe0wlYolSS4rFBWJopVPehglVakkZaUx5ynphAkpS5diSiUqk8KJ/l/dv1OSayzLxspsLV7L91Xft9ha2p3tve5eh99h99nhb68dtgQ8+o7RP11snio6dfWfhf+RCCQn2iSQZUDIDbr3gZaNoKeatTRTEYq4exuQQ+cav2z4Kt+z3oBopk3OdZTCCDRrpFYbH0jMUWmNMqVerNCJeFMIHp7zBoZmw/XCeCSZsGFhCwqO1yT7lxlgC4riKKy2vCg16I0EfHTQR/u9tMfscXY53F1Oj9VPkhhN4RSFBbrCXY0kSaGFU5LgK+OHdasX7mocoV0vKY0LybUieC+ddzHNRm1ez/io89i8x24afxNAwMnMWQEQCVLdja6WAzZrh3fC5VnGLBUsTpUJskfF+5zhtBF9xQxRTxX+zp+H59zXrK2GiYdH0ImW3Uh5PfXAg9E5fStlhy5RbCgkzMlJWrKEiIfVqUww4N+5k3a7Yc0TgiBicV8BvkQiTE+P1QCJ8vKM//43plHDYnydDpVIoZsLvOX/WboSJpdj8sHsEx2IBGiGjsnKHxt+/HL3l0Kh8JGLHuGs/up76+/85E6/3w/ofibeCMCluE6t00q1GqlGK9UaVcY0bZpSorzns3vC4TCnHU+oPtfbA+z10JeqbRvTXu3xEu6wyhZJ6iavdYG0S9vb5flQs0o1ckyqkqBkEOszX0QAGHtZxuCUU/Dw8Jx9sIDFxWhcppwm+7yfBguxXCCWhoCoE6AtwHsA4D8xxEFKE/bTGicZ747oXUySW5FjyhzbF2IUqHtzH2ssb/X0SkZSfWVPzl7/1k/rMAJVJ0hVBolcLRIrBGI5IZELsNNcmTQmfQyAlbS/D68jZGnzdtTa26rsPmeYitBkmGlI6eU0K4KACxYNi/Yr5EdUHp7zS7OSqMgrMSVYXkPbfR44gkV7P+M4VJkIAOEQCPq5NQljguaG6wGGEQYDptdjcjmXV8r9juXC4nq98pKLT+MO0yRFU96QF87mOzpa7a1tjrYuR5fNZRufM/6f8//Jrba3Ze+r37yqSdbcOP5GTrNqZJoEZYIVtabqUlO1qUnapDRtmklt0sq1Win80Ug1UmFfxLfT1U1/9mfAAneo7/B/GToAnVNbtoL2/bSlKWC1m92GXjrPzkzw0HFuOp6khYCmMIw2e/ScmsYFyMxb8lAcUcUdzZTg1SoPz3kMhqFTrs9laKhWBeLBuDQwFKzu79gDOvcByyGYH+9sBdCWH0FxgYAQCcQudXwiMKWDxJFAmUmpj+Yy6ZOVRZP0IikhUfS5U/ld4Y5aJxmiOWcUXIBJlLCiS6IQSBQChUGiijo9K/VigRj/WXbsmSXoi3QedrVV2SxtHnu3nwrTCIoQAkypF5tyNBklRyMLfA4AD8/5qVmNKaIKAHr1pXJhmJBJMLFQoNcS8XpBXBxh0IsSE1htPE0yKIYQCca4R34j4XVwYRimzdHW7ep2Bpw9rp52R3urvbXdDn/bfDYSOsOQMGjK/UQAhcIKKs7hryipqKywLDk+OeYamKnP3PTgJqVYKSSEBEoc3932u5/eT6aCAAU1tWuCU+4SH9s50N0Ofnop1FDptoVcIY0lmNxNXeViUiggpliCYRDAUIQA6OIxhU4KZ/yT+owWEQQxZsNgAA8Pz9BB1C8/9SQJuWG+KacrW7aBZVcCMgiL/RkKThxJNCAuD6gzQFw+rKMyFgOJCqBHQgn9NpMyXJsyXMsybEx6SpXC/AuMAVck6IOpsUFvxGMLebjs2GhOAmzhiiEojshUQqVBMnJGcsyxJOQncQLDo6VgJ8HhXb32Ti9AgDFTnXbEEmEALnPA0u5pq3Z01NgDXpKK0CzDCkQYl/GfUqA15Wj44lQenlPk3PgKYQGHmHYcyr1eKBMKZGJUSKAoQBGA2gHiAGgdhWw6HG07zf3AgSmlQJs/wci9vKvO2VnnFMmIgkkmLv89HKS66p2wvRSBQaNsHMUIaJSNw98oRqDQIgD2A0SiuaIsBq204Qu9Ie+upl2Nlsbp+dMzDbBRaoSKPPTVQ2sOrg6GQ9C0n2urjUSTTTFAYJiYEAkwwiCPS9OlJamTxmaOZWEfKsj4zPHf//V7jaSvASAMb+CCJE00o+sEmIEgo1TxKMuIBDKEJgFLAX8nsDaBhFIggj39PGbf3u/FdvJyD5vgozTQW5ClcZzFcFQhQw2pivg0hT5ZrtCJFXoxf9PPwzPUgcNIdHTCcID8Tl+nlq3gp1dhdf9FzwF5NBlWHg/EWkB1A102MI0CpjIQnw9UqUCdcoKb7B8rVSdIp1yXGw1kksGoZvW5Ix5b0G0OuCwBtzUYCcFOW1SQsXh8lnZf7ri+hFyGYXd81ejo9mWVxY2YlnzEJIFhmehE3QmMe037LYd39qAIMnImc6xmbaqw1O822zp9zl5/NHSC4gJUY5SmDtcm52t1SXKZ+rQ4cPHwDEHOAc3KUlTcgceuSdvxnn2Z14sApzeW6flrOVc0yYhkgvwJff9tr3XuWt6s0IvzJxjRaMaqxxZa/84hmmKj6UR9XaWjnaXh+ghshQeFLEogDEJFqMik+fk5o+AIeNhS+/4H3zEWkXb6sIxrMhEEiAQiWURf4J8iEgooPEKBCAlIHMOStKZUfUq6IT3VkJKqT5GJpQJCIBAQBE4gNM6VT4kIkQCIqBCFEkcs+qKHBaD4ZY/+HOkj2PfDHTnCUr6SOs//hWiZ0SfPwcThhr29Hz3lpzSpt+olGYXwc5Aa2wRznG4KRRmZAkS7pIoNyfKETJUmUUoIMH66n4eHpw86Ar68Adib4YPp/w/kzj3emo4WYG8ADAvyjrhi+SzgwNdAJAKlt/RpVnUamPMSVK6GYQCXDFbxFHR4lREAHC2KhU1gKcbnCkW9Y0Musz/gieiS+soGaJJprrC6zP641CNu0wBUrGur3tatS4LlpDA7VhP1jpVD71icGCjWM0sNrZU2FEWyR8OZfZZlyTAtOGLy1VXnPLSjG0URkZSIS1ckpKtSC7QJmSq+HzUPz1DUrAiGhfSl5sPO4ezXirxcYXYpRcJ2dgwJfUoZaAkAza5pkqXgyMXSFIiEaJ0myHmqwuAl65eKAlIB7GfNQQW8CBNCGKhTWRZho6HRqBjkHnMm0tEWLixCRoiwO8C9UClRFoumClET3eBkKBKLJsheZ1xQKxAQmAiuze0zAIiTBW6ANLE9KNOL9CAYi8LuhwjCkFNvzE3MhbPwlM+z/YsGcw8zLI8ecWkZF9ho2FzVXOXGcQZnwygSwUEYAxEMhHGExNgwhkTgf9kwjkYQhydbGjrknSTUJmACQXfQtLL7HppkL+sKJmfAY1cYFBkj9RRJG7NU6jipQi+OJYfx8PDw/BwUBF2gaz+06z9aAXoEOgI8PbBXauduYD4MnC3AVhe1Sh0NZNGcIkMeKFoA1KlA1RfLhNZUORedif2G02uYOk6qjvsFdxcERYpnJTt7/El5fTNaLMvauvz2Lp+zB1YCwAYFBCpRCMUKqFlhdqwO3t4ro9mxIgnBqVgERQQirH6v+dD2bm2ibOIVfR4CaUX6rgZXSp42OU+jNcnEcn6M5eEZwpoVIEij4urycP5NCbfjfgpUC/uGUxh05B70+81FJNEwgtwBwAh4Cw5AvuTrEdr/Y1RJOLIXANhmsKF7+VzF6wQiogFBsfCHZgka4LClExDAx2z0N4DLw2FEj88CAFbNp2pSRyUcsJo3GQUCFOmzMMyQC8x4PYMIGIAzAGNYjGEwhkFpNvqY+wEYgOIYIBEPGYzmYAFAhSM9jc7WFjRO4oZTclHN2tvsrNodwAR4NOwrAIBAEE4KswiAihr+hrKYRTBMRjhZBOjUcOSVqGUyrZRiMFKawol1jEAvuCqbD6by8PD8NhgOsmeDuu+jU/mlcInfBhzNwN4EOveAjr3QcoQKw/ZUXHKqUAb75AWdfZo1Lg9c/dkZdk49EXACLZmV+vNlSPoIvUhC+JwhLtMg4Il4HSGPvW9khpNsOAIzwqLZsQIJwcWIMQLtbXI37jWbm93Fs1Kk0RCAKUd9+cNlfFdVHp4zwFk3vhwLzdDlzU1WwHoBIiUpAcVAP1Qk2u+0L+OKgYKPoVmWYRiYs8lSYYwNc0qNBKDeWpfF+ljag7MMt9BHuVGmRkJgGIrjKI7D1H2sr0kgF2ztDxUAsr4OAgJUeEHidsb2OUidAe20oiSagnMzP2EBTgMBi+BQvEL9SjAIHv1NMEDAIHhUB+M0zeiUMBEWjoAiUW62xyB2JGUkxLLHVAZJaoqTRVmaQSkGo2mMphGKwWDk92hqQDSEzDIUK0JAAAch+MIE5czFI4ViXKGDurzvUHjBysPDc4Ig0CIQqtI9S4CrA9gaoGb19gCGhssJMWyOqskAiSUgaRSIGw40aUAZy79HzkLB+osgCMgui8sugxP9IT/JVXQF3BGPPejsDbij2bHhIM3STCTEWNt9co2IuzqwDJs+Qm9p9aQWaPEjTrSwPSLBD7M8PGeCc2CIQRGUVHzZIPx0V4hMy5qcM/pmZ9Bt8zucgejvoNsRdDkCTpvPYfM7XEE/zbJhkr4L010RfbkQgA9DYEU7U5wl+wjFuQpVY8bk97uu1csNCSpTgsqYoDYZFHE4TkTLXaPlXQCDAxvC/UaAoN+U09wX0Nn/gqPzkQGayJtGDBt/dIW+4euI8P3ZaMYlz/Z16CakshHXzwUMAzfFVdoCMHxOSf6sEVxaQt9WogFahmYpkmZgxy6YEUHToHPL5s7d9WLcwfbEM6HxmEiakAFLr3h4eM4brFar2WzOzc3Ffu4e3dLSYrVapVKp3+/PyclRKI4ma54SKA483WDrc7Den2Vh40CFEWjS4Q8nVQ15ULweGa/OdURSArolJAzMjvW7w9HsWKhiQ4FIc4UNBlAo1pSjnv9QSb/yAx4enjPHOaBZAYLcqSKCqsjLQbCsq9n64zJnwGHz2Z1+h8NrAWHmaG4A1x0KRRBGMC4Y5DQrAGD6uJvTMy7MjM/EsD5TlRHpY0akjznJ/cGE8Odne4gB/EiL7d8LKoC73X9j0AHhl68Hwp//vboU8R2MlGIF/gCeQzPnyTWEh4cnCsuylZWVu3fvrq+vf/bZZwdo1h07dixfvtxkMo0dOzYvj2tXN3gIpFChmsrgjzYDxlb72Zuc33DZsSqDRGXocw9srbI1lVv7ihWiK/yR+8fDM4Q5BzQrAkCVcfzfPEvWBingbARNjbEnBFKBVq/VyrQaiUYr0yaqElN1qUmapFRNaoYBpp9yzMybDgZ7SD8byJlWlFwWYVkgEGGY5Gg+AA8Pz3kAgiA5OTl+v7+hoWHAUwzDpKenv/zyy/Hx8Tg+qMM4Q8HA6sL3QWIxOOIbzcPBq1Uenj+Wc0CzAgBsCtNaEg4WeUl5w+KHQVWqTTVpTDqpDvYylcHuUJJjHfXPd2BHRL5GlYfn/EUoFAoEgmPbeyIIQlFUU1NTZ2dnQkJCSsqJOp7+BigO8wEEcqBK4gVrjPYaBxmiEAxprbJrjSc7pcbDwzNENGuHvRWlaAaA+6bfd8uEW9DzJZWKh4eH5/hECy4HgiCISCSSSqVJSUlLly4dO3ZsaWnpKb8TAxxNMFE16ADeXqBIPNUNni8wDKsxyhAU0ORRw0QeHp4zz7mhWbc3bGegZAW7mnfdNvG2P3p3eHh4eP5IWJYtKSnhMlw1Gs2aNWuKi4uP3+r5t6FJ0LQJCMTA2QrMtTCflSfK6HlpxTOg6axAfG5cMXl4zlfOjW/grRNvnZg1EQCQHZf9R+8LDw8Pz5mDSwwYkB7Q1dX16aef3nLLLVqtFsdxr9c7CO+EEWDmP6HfKsuAlHGDsMHzKgvrj94JHh6ec0WzjkobNSpt1B+9Fzw8PDxnlEgk4vF4/FEwDAuHw+vWrRs9erRAIIiPjxcKhSRJtre3l5SUnGqQlTNnzZ45OPvNw8PDcxrgE0N5eHh4zlLq6+ubm5sNBsP27dtdLhfLshaLJRgMGgyGsrKy8vLyH374ITU19dJLL/2j95SHh4fntHNuxFl5eHh4hiDDo/RfsnjxYu5Bbm5uYmIiRVEazVBxTuXh4Rni8JqVh4eH55xk0Hpf8fDw8JwL8LkBPDw8PDw8PDw8Zzu8ZuXh4eHh4eHh4TnbOdOaFUEQFEVFItEZfl8eHh6eweUXO1SdPXDurTw8PDznNARBxEbaM53PGolErFbrqlWrBAK+MSAPD885TF1dncfj+cU+VX84DMO0tLSsW7fuj94RHh4enlPi4MGDFEWxLIsgyJnWrAKBQCKRfPPNN4PgJsjDw8PzB4EgSCAQUKvVZ+dQplAoAoHAF1988UfvCA8PD8/JgyCI0+kcMWJE33/PcJCAZVmSJPlJKx4ennMdNgpBEODsg47Cj7Q8PDznOgzDwAgrjv8BmpWHh4eHh4eHh4fn93I2zmrx8PDw8PDw8PDw9IfXrDw8PDw8PDw8PGc7Z69mpSjK7/f/2rNerxcMVX7t2FmWdbvdYKji9Xr5RJdj8fl8NE3/2rPBKGDoEQgESJI8iZHn/CMYDIbD4V98imEYn88HhiTHOXaSJD0eDxiqDOWL70l/LF6v99cGnCF7XQ6FQr828px7vVurq6s/+OCDYDA4efLkBQsW9K/MdTqd7777blNTU0pKyuLFi9VqNRgyHOfYe3t733777c7OTp1Od+utt6alpYEhQygUevfdd6urqxMSEm655ZbExMQBK7S3t3/88cfz58/PyckBQwaSJJctW7Zz506lUnnrrbdmZmb2f9bn83366acNDQ1KpXLx4sV6vR4MGb7++utNmzZJJJKrr7565MiR/Z/asWPHypUrGYbJy8u78sorxWIxOK/54Ycfli9fjuP45ZdfPn78+P5PNTc3v/vuuy6Xq6Sk5Prrr+cKIIYIxzn2qqqqjz76yOl05uTk3HLLLSqVCgwZzGbzm2++2dvbm5+ff+ONN8pksgEr7Nu374cffuCvy/2f7ezs/OSTT6xWa1pa2q233ioUCgEY6tdlmqa/++67Xbt2IQgyadKk2bNnn7jR9dkYZ3U6ne+999706dMfeeSRHTt27Ny5M/YUy7JffPEFwzDPPPOMRCL5+OOPGYYBQ4PjHDtFUZ988smIESP+85//qNXql19+ORQKgSHDihUrent7n3766dTU1A8++ICiqP7P0jT97bffrly50mKxgKHE5s2bDx48+Pjjj48fP/6dd97pHzukKGrJkiV+v/+xxx677777htQFZt++fZs2bXrwwQfnzJnzwQcfOJ3O2FNtbW2fffbZ1Vdf/fDDD7e0tKxevRqc19TV1X3zzTd33nnnjTfe+Omnn7a1tcWeCofD7733XkFBweOPP15fX79+/XowZDjOsdvt9i+//PKqq656+umn29rali5dOnTmdmiafv/99xMSEp555hmXy7V8+fIBK/h8vk8++WTz5s1DKjZ/fE3i8XheeeWVzMzMp5566oYbbhhStvQrfv26vHPnzu3bt99333133nnn999/X15efuKbPRs1a0NDA8uykydPNplMI0aM2LNnT+ypQCBw6NCh6dOna7XaGTNm1NbWulyuP3RnzxzHOXaWZUeNGjV9+nSVSjV//nyr1Tp09FkkEikvL586daper7/wwgs7OjoGHPv+/fu7u7sHhNPOe1iWLS8vHzt2bEJCwuTJkz0eT2tra+zZpqamhoaGadOm1dfX0zQ9pEJoe/bsKSoqSktLGzt2LIZhtbW1safcbjeKosOGDdNoNElJSVarFZzXVFRUpKam5uXljRgxQqfTHTx4MPZUV1eX0+mcOnVqXFzchAkTdu3aNXSiA8c5dgRBZs+ePXLkyISEhHnz5tXU1Ay4Qz6PsVgs7e3ts2bN0mq1kydPrqioGDDZvWbNGrFYnJaWNnR0/G9qku3bt2MYVlRUVFdXd5a3zTuT12WbzabT6YxGY0pKilKptNvt57ZmtVqtYrGYu45qNBqHwxH7Dni93lAoJJfLOdNsiqKGTl7RcY6dIIiJEydyHXHNZrNcLlcoFGBoEAgEfD4fN0MnlUoRBHE4HLFn/X7/V199NW3aNIVCMXQuulwk1eFwcAFUoVAoEon6p1vV19d3dHSsWLHiiy++eOihh3p7e8GQwWq1ch+LQCCQSqX9hWlOTk5xcfErr7yyZMkSs9k8Z84ccF5jsVhiA4VSqRxwUcEwjEuN0Gg0Ho9n6EzdHOfYNRrN2LFjOeVhNpvj4uKGjgOuy+ViGEYqlQIAVCpVIBDor9dbW1t37tw5d+5cFEWHlGY9viaprq5ubW39+uuv33777WeeeWbofIkCx70uX3DBBUKh8LXXXvvvf/9rMBjGjh17bmvW/iUjCIJwPbu4/zJRuCEDQaC57NARIidy7A6HY9WqVZdddtnQybJiGIam6djHwsm12LPr1q2TyWQXXHABTdNnZ7+i09q8o//H0v9C4nK5JBLJ4sWLn3vuObVavXHjRjA0YFm2/5mAomj/WBGKogiC2KOEQqHzPvxMUVQs8MONtP2fio02KIpy/wVDgxM59oaGhv3791911VVDZ1ShaXrABaj/U998883IkSOHDRvGMMzQ0fG/eV12uVxGo/Ghhx565plnOjo6qqqqwNCAOe51GcdxmqbtdrvNZqMo6nedMGfj900oFEYiEe4rEQ6HCYKIDazcVYQ7eO5ic3Y2oTkd/Oaxc3nAEydOnDVrFhgy4DiOomgkEomNqrGcIZ/Pt3z5co/H88EHHzQ0NKxevfrQoUNgaICiqEAg4KoyGYYZ0HxOIpGkp6drtVoAQGpqak9PDxgaIAhCEAQX7WBZNhwO9y+J2Lt3744dO/7xj3888sgjCQkJH3/88fkdMRIIBNwXhxtS+ifbCQQChmG40SYSiWAYdt4r+BM/9oaGhk8//fSGG27Iz88HQwaBQMDdCcfudmLX5ZaWls2bN3d3dy9durS1tfWLL74YOkPK8a/Lcrk8NzcXRVGNRqPX6zs6OgAY6tdlEE119fv9TzzxxJNPPtnT07Nu3bpzW7MajUa3282VjLS3t8fHx8e+GwqFQiqVdnd3AwC6u7sFAsHQCSge/9hdLtcnn3xSVlY2pAQrN++g0Wi4scBqtdI0bTAYuKeEQuG99947a9Yso9GI43h8fPzQSZnAcdxgMHBVNW63OxAIcAqVIzU1lZvS4s4cbmJriGAymdrb27m8EafTaTKZYk95PB4URblqaIVCcd7nyptMJm48YVm2p6en/0cRHx8fiUS4ArWWlha9Xj906p2Pf+wtLS0rVqxYtGhRUVERGEpotVoURbkEko6ODoVCEVMhcXFxDz/8cGlpqV6vx3E8KSlp6Jwtx78up6WlcZlXNE3H5sqH+HUZRONrEomEi62IRKL+hbC/ydl465yVlZWUlLRkyZKMjIza2toHH3wQAPDRRx8lJydPmjTpwgsv/Pbbb0mSXL169aRJk4bO5VYkEvU/9smTJ8vlcs6n5qKLLnr55Zdra2tJkty9e7dMJrv66quHyNcDw7CZM2d+/PHHYrF427ZtZWVlOp2utrZ28+bNN910U2lpKXcTvGLFitLS0v5X5fOeadOm/e9//1u5cuXBgweHDRuWmpra3t6+fPnyP/3pT3l5eTqd7v3334+Li+vs7LzyyivBkOHCCy/817/+9dVXX3V0dOh0utzcXIfD8eGHH1511VUlJSVr1qxZsmRJSkrKli1brrnmmvO7ZmLs2LEbN2786KOPuLuXsrKyQCDw7rvvzp49OyMjY/jw4UuXLi0pKfnxxx8XL158fn8U/TEajf2P/Y477mBZdsmSJRdeeGFCQsLTTz8tEonWrl37zTffpKSkXHnllUMkPUCr1Y4bN+7DDz+cNm3a6tWr58+fj6Lohg0bvF7v/PnzOaO09vb2tWvXTpgwQaPRgCF5XeY0ye7du6uqqm655ZYLLrhg586dX3zxhdvtVigUJSUlYGiA/fp1+dZbb50yZcpLL7301VdfkSTZ09Nz7bXXnviWf5aVcvZgt9tXrFgRCoUmTpw4fPhwAMCBAwdUKlVqamokEtm0adPhw4fz8vImT548pMwjjj32Q4cOYRiWlZVVXl7e09PDRhGLxRMmTOCS5YcCDMNs27atoqIiMzNzypQpXGFNc3NzcXExN01DUVRVVZXJZBpSLqSxyW6TyTRt2jSVSuVyuWpra0eNGoVhmNVqXb16tcfjmTFjxpCyreUsNjdv3qzT6aZNm2YwGAKBwIEDB4YPH65QKDo7O9evX+/3+8vKykaPHn3eC7Xm5ua1a9dKJJJp06YlJSWFw+GKioqsrCytVuvxeFavXm2320dFAUOJAcfOMEx5eXlaWppMJtu9ezcXgGcYRq/Xjx8//rw/SWIEg8G1a9d2dHQUFxePGzcOw7DGxsZwOBzLkfD5fLW1tfn5+VwUbchel7ky+eLiYgRBGhoaNmzYIBAI5s2bFxcXB4YMzHGvy9XV1du2bUNRdMqUKdnZ2ee8ZuXh4eHh4eHh4eGJMSQmNXh4eHh4eHh4eM5peM3Kw8PDw8PDw8NztsNrVh4eHh4eHh4enrOds9E3gOeshSTJ8vLypqYmFEXT09MLCgp+V659c3Pzvn37EhISuPz9U9wZt9vd0tKSn59PkmR3d/epG6wwDFNdXW00GnU63SnuGw8PD89J4/F49u7d29XVJZVKs7Oz8/LyfteAeeDAgcOHDw+Pcuo7093d7Xa7c3Nz7Xa71+tNTk4+RauEYDBYV1eXnZ09pEq1eAYFPs7Kc6LYbLYnn3zynXfesdvtbW1tS5Ys+V0OyQ0NDc8//3xzczPn1nbq+9PQ0PDiiy8GAoHGxsZ///vfp96ANBKJvPLKK/27rvPw8PCcYQ4fPvzXv/51xYoVPp+voqLi3Xff9fl8J/7yTZs2vfnmmz09Pf2b8Z4K27Zte+eddwAAW7Zsee2114LB4Clu0Gw2P/fcc0OqZTTPYMHHWXlOCIqi/u///s/j8bz44otKpZJrKPy7jMa2bduWlJT08MMPD9YucZ02GYYZNmzY008/feo+VgiCcNscpB3k4eHh+X04nc5//vOfxcXF99xzDxdb9fl8Jx6PZBhmw4YN06dPX7BgwWDtEtfyimGYGTNmTJgw4dSDo/17aPHw/C74OCvPCVFTU3PgwIHbbruNE6xc/89YS0OLxdLe3h5rKEySJE3TgUDAarVyC8PhsNPpZFnW5/Nx9mper7elpSUWP6BpOtb5nXv5L26Hw+fzud1ukiRRFGVZViAQGAwGgiBYlo1EIlwjY6/XG1ufoiiz2RwIBPx+/y9GeUmStNlswWCw/0jqcrmampq49qccdrvd5XKFw2Gr1RoMBmma5nqRezwerse03W5vamqKHQjLst3d3ceJRodCof7HxRvP8fAMcX744QeSJK+//vpYMoBMJuPm4mma7ujo4Hy4uacikQg3/jgcDm6hz+fzeDwURQUCAW4dm83W2trav0kvN+ZwoyXDMLFh0+l0ut3u/jvjdrs9Hg/LsggCbTGlUqlOp0MQhBuuI5EINxLG1g8GgxaLJRwOu91uu91+7IAWDAadTmckEumfXWA2m1tbW7khlJPdFovF5/NxW+PeiHst1x2ToqiWlpb+UeRjlxz7vrHt8yPtOQ0fZ+U5IQ4ePGgwGDIyMgYsD4VCS5YsOXjwIIZhcrn8/vvvN5lMn3766a5duxITE7n00CeeeOLAgQMbN26kKIqm6fvuu2/fvn2ffPKJQCAgSXLRokVTp07dsGHDjh07Hn/8cRRFn3nmmZKSkosvvnjAdh577DG1Wr169ep33303KSkJx/FwOEwQRENDw+uvv/7444+HQqHHHnssKyurp6envb393nvvnTJlisViee2112w2G0VRYrE4Pj7+9ttv79/ItKGh4T//+Q9JkmlpaT09PdylYs2aNV9++SVBEGq1+v7779fpdJ988snOnTu5dtICgeDmm2/u7u7+7rvvlEolRVF/+9vfNmzY8P333+M4bjQaH3zwQRzHX3vttcOHD7MsO3r06Jtvvrl/H2qGYVavXr1+/XqxWHz77bdnZma2tLRIJJIhZTrNw8PTH5ZlKyoqcnNzj+0jZbVaX375ZYvFwjBMVlbW3XffLZVK//nPfwYCAZFIVFlZOWnSpLvvvvvbb789ePAgl4F64403fvbZZ5s3bxYKhQKB4M4778zJyXnrrbcEAsHtt99utVqfeeaZxYsXp6SkPPDAA0aj0e/319bW3nzzzZdddlk4HH7vvfc2bNiQlZXlcDjUajWGYevXr9+xY8dTTz21a9euJUuW5Ofn19XVhUKhJ554Ijs7u6am5u233+YCDSKRaPjw4TfccEP/QW/r1q1vvPGGWq1WqVSBQIDrR//BBx/s2rULAJCfn3/XXXdFIpE33niDk7BisVihUNx0001vv/22TCZramqaO3fuhAkTnn/+eZfLRVHUggUL5syZ09vb+9JLL/Vf0v9zCwQCy5Yt2717d1pa2h133KFWqw8dOpSSkjJ0eu6cZ/BxVp4TgpufOjYZYO3atYcPH/7nP//53//+Nzk5+a233qKitLe3X3/99f/6179aWlqqqqrGjx8/ZcqUcePG3X///RaL5cMPP7zllltef/31a6+99sMPP+zp6QmFQrFQgcPh4O6nSZLsv526urqOjo7333//xhtvfOaZZ8rKyrhbZ4qi7HY7Fz3t6elJSEj417/+NWrUqHXr1gEAVq1ahWHY//3f/40cOVKv1z/88MP9BStJkkuWLDEajS+99NKiRYtwHMcwrKOj49NPP73lllteeukljUbDHeO2bduefvrpxYsX+3y+Bx98MD8/PxwO796922Qy3X///W1tbStWrLj33ntfeOEFiqI2bdr0ww8/NDU1PfPMM0899dSBWQDX7gAANRlJREFUAweqq6v7f24VFRUOh4PrYrdr1y6v19ve3j50GhHz8PAcC8uywWBQJpMNWM4wzPvvvy8Wi19++eUXXnjBYrF89dVXCIJ4PB632/2Xv/zlvvvu2759u9lsXrhwYVFR0WWXXXbdddft2LFj69atTzzxxGuvvVZcXPzmm28Gg0Gv1+vxeLht2my2SCSCIIjZbJbJZE888cS8efPWrVtHUdTOnTs3bNjw2GOP/eMf/0hJSeFG10AgwI3SDMO0tbWNHz/+pZdekkgkO3bsYBjmk08+GTVq1IsvvqhQKEaNGnXrrbf2F6x2u/2tt96aNWvWc889N2XKFJZlMQzbuXPnjh07Hn300eeee667u3v37t1bt261Wq0vvfTStGnTMAx77LHHEhISrFbrvn37br311pkzZ7777rtSqfT5559fvHjx6tWrrVbrhx9+GFuyatUqm83W//PcunWrSqW65557UlNTy8vLubkysVh8Bv+qPIMJH2flOSHi4uK2b9/u8/nUanVsIU3TO3bsGD9+PBcdvOiiix5//HGz2YwgSEFBQXJyciQSUavVgUAAx3GJREJRlFwuX79+vUKhGDt2LIIgEyZM+Oqrr/bv34/jeGy2KJZUyjBM/+0Eg8Ha2lqlUjl79myCIDIzM4VCISdzudeyLKtSqcaOHSuVStPS0nbv3s1NnykUCqFQqFQqj836t9vtHR0djzzyiFarlclknJytqalxOBybN2/eunWrxWKRSCTBYJAgCJVKxU2ExbRyfn7+9ddfL5VK165d6/V616xZw02o2e329vZ2v9//wQcfMAwTiUQcDkfsTRmGwXH8qquuEgqFRUVFO3bsqK6u1mq1fBUtD89QBkVRrVZrNpu56fjYcofDUVNTc++990qjTJs2beXKlVdccQVBEKNHj+a6mhMEEQ6HpVKpUCiUyWRisXjnzp2FhYWpqakAgJkzZ27YsKG1tRXDsNiWYyUBMplszJgxUqk0PT19z549kUikoqJiZBQAQHp6+v79+2MZ/9xIm5SUNGLECKlUajKZuDwEiqJUKpVIJJJKpbFUhBgNDQ0oil588cVyuTwzM1MsFtM0feDAAb/f/9lnn7Es6/f7XS4XSZJSqVQsFqvVai6HgXvT+fPnjx071uVyVVdXq9Xq1157jcvLam1trampUSgU3BKKotxud8z4JRAImEymgoICAEBubu7WrVvr6+tNJtMp+h7w/IHwmpXnhCgpKfnggw/2798/derU2EIUReko3H8ZhqFpmhORXKor919uiGSjcKvFkjhZluW2wI2eMWJvMWA7DMNgGBYbN4/dz/56l1uSkZHxv//9LxAImM3ma6+9dkDuP7cR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qP8eLl68WK1Wf/XVV+vXr8/IyGhpaTk24djj8ZjN5gGnVnt7eygUysrKii1BUXSADj5FaJpubm4Oh8M0TWdkZHBzOziO6/X62tra/g1Lz1H6D0HJycnZ2dk7d+6cMGFCZWXlJ5984vF4srKyFi9eLBaLv/rqq82bN7MsazQa//znP6Mo+v777zc2NqIoOmXKlHnz5jU2Nn744YfJycnV1dUikeiRRx5pb2/fuHGjz+d74403brrppu7u7o8++sjj8Uil0htuuCEnJ2flypWBQGDRokUOh+Ptt9++5pprDAbDK6+8EjsPb7311jFjxjidztdee623tzctLc1ms3Hx2nXr1q1Zs4YkyQkTJlx11VVut7v/OgNue+rr61etWnXHHXcIhcJXX301Nzd39uzZDQ0NK1euvP3227mpiZqamtTUVO4q1t7e/t577/U/23/tA5w9e/batWuXLVt2+eWXc8PjH8ygjjEdHR3PP//8ggULLr74YhRFKyoqnnvuOZVKNX78+N+7qUC0IflQK6MfIpqV/dU8lAGyifvzH3sS/B51tXLlSoZh7rjjDm441keJPevz+QiC4FIFuBbMLMuGQiGxWIwgCMMwZrOZcxXlbi5JkgyHwxKJJNZkj+utN6A3NIZhwWCQ60F15FDgZrnG0DF1GAwGueqc2P5wUvXY2+jT/LlDGK7i+9dfeIrBV7VazTCMx+PZunVrY2Pjgw8+WF9f/9lnnxkMhtdee62oqGjOnDkul4sgiLq6uo8++uimm25KT0//9NNP33rrrYceeqi+vp6iqAceeGDDhg1ffvnluHHjpkyZUl5efvHFF+fk5HzzzTc1NTX33XdfJBJ55ZVXNBpNZmZmZWUl93m2trZy+2C1Wnfs2PH0008XFhY+//zzs2bNwjDs3Xffvf766/Pz89966y1uxHn77belUukjjzyybt26L774YubMme+99951113HrXPsnTRFUQ6HY4DTYWNjIxcgjC3p7e1lGIazjHW73fX19dwfmiTJvLy82GngdDo3bNjgcrn6X5ZYlsUwbOLEif0VzK+h0WgaGhpOT/n5cU4hBADm11fjlpzS/uj1+vr6epfL9cYbbwwbNuyOO+74/PPPV61aNXv27FdfffWiiy6aOHGi2WzWaDQDzgeVSjVp0qSqqqrk5OSbb775888///LLL5999lku8j1//nyd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)

Notice that these outcome models were also chosen because they are capable of handling multiple treatments (that may be assigned simultaneously) at the same timstep. In the simulated dataset, parameters γ A and γ Y control the amount of hidden confounding applied to the treatments and outcomes respectively. We vary this amount through γ A = γ Y = γ .

The Time Series Deconfounder is used to obtain unbiased estimates of one-step ahead treatment responses. For a comparative evaluation, the outcome models are trained to estimate these treatment responses in the following scenarios: without information about ¯ Z t in which case they estimate E [ Y t +1 ( a t ) | ¯ A t -1 , ¯ X t ] (Confounded), with information about the simulated (oracle) ¯ Z t which leads to the following regression model E [ Y t +1 ( a t ) | ¯ A t -1 , ¯ X t , ¯ Z t ] (Oracle), as well as after applying the Time Series Deconfounder with different model specifications and in this case E [ Y t +1 ( a t ) | ¯ A t -1 , ¯ X t , ˆ ¯ Z t ] is estimated (Deconfounded). To highlight the importance of Assumption 3, we also apply the Time Series Deconfounder after removing the singlecause confounder X 1 , thus violating the assumption.

Figure 4 shows the root mean squared error (RMSE) for the one-step ahead estimation of treatment responses for patients in the test set. We notice that the Time Series Deconfounder gives unbiased estimates of treatment responses, i.e. close to the estimates obtained using the simulated (oracle) confounders. The method is robust to model misspecification, performing similarly when D Z = 1 (simulated size of hidden confounders) and when D Z = 5 (misspecified size of inferred confounders). When there are no hidden confounders ( γ = 0 ), the extra information from ˆ ¯ Z t does not harm the estimations (although they have higher variance).

When the sequential single strong ignorability assumption (Assumption 3) is invalidated, namely when the single cause confounder X 1 is removed from the observational dataset, we obtain biased estimates of the treatment responses. The performance in this case, however, is comparable to the performance when there is no control for the unobserved confounders.

In Appendix E, we consider an experimental set-up with a different simulated size for the hidden confounders (true D Z = 3 ) and show results when the size of the hidden confounders is underestimated in the Time Series Deconfounder. We also include additional results on a simulated setting with static hidden confounders.

Source of gain: To understand the source of gain in the Time Series Deconfounder, consider why the outcome models fail in the scenarios when there are hidden confounders. MSMs and R-MSNs make the implicit assumption that the treatment assignments depend only on the observed history. The existence of any multi-cause confounders not captured by the history results in biased estimates of both the propensity weights and of the outcomes. On the other hand, the construction in our factor model rules out the existence of any multi-cause confounders which are not captured by Z t . By augmenting the data available to the outcome models with the substitute confounders, we eliminate these biases.

## 7. Experiments on MIMIC III

Using the Medical Information Mart for Intensive Care (MIMIC III) (Johnson et al., 2016) database consisting of electronic health records from patients in the ICU, we show how the Time Series Deconfounder can be applied on a real dataset. From MIMIC III we extracted a dataset with 6256 patients for which there are three treatment options at each timestep: antibiotics, vasopressors, and mechanical ventilator (all of which can be applied simultaneously). These treatments are common in the ICU and are often used to treat patients with sepsis (Schmidt et al., 2016; Scheeren et al.,

Table 1. Average RMSE × 10 2 and standard error in the results for predicting the effect of antibiotics, vassopressors and mechanical ventilator on three patient covariates. The results are for 10 runs.

|                           | White blood cell count   | White blood cell count   | Blood pressure   | Blood pressure   | Oxygen saturation   | Oxygen saturation   |
|---------------------------|--------------------------|--------------------------|------------------|------------------|---------------------|---------------------|
| Outcome model             | MSM                      | R-MSN                    | MSM              | R-MSN            | MSM                 | R-MSN               |
| Confounded                | 3 . 90 ± 0 . 00          | 2 . 91 ± 0 . 05          | 12 . 04 ± 0 . 00 | 10 . 29 ± 0 . 05 | 2 . 92 ± 0 . 00     | 1 . 74 ± 0 . 03     |
| Deconfounded ( D Z = 1 )  | 3 . 55 ± 0 . 05          | 2 . 62 ± 0 . 07          | 11 . 69 ± 0 . 14 | 9 . 35 ± 0 . 11  | 2 . 42 ± 0 . 02     | 1 . 24 ± 0 . 05     |
| Deconfounded ( D Z = 5 )  | 3 . 56 ± 0 . 04          | 2 . 41 ± 0 . 04          | 11 . 63 ± 0 . 10 | 9 . 45 ± 0 . 10  | 2 . 43 ± 0 . 02     | 1 . 21 ± 0 . 07     |
| Deconfounded ( D Z = 10 ) | 3 . 58 ± 0 . 03          | 2 . 48 ± 0 . 06          | 11 . 66 ± 0 . 14 | 9 . 20 ± 0 . 12  | 2 . 42 ± 0 . 01     | 1 . 17 ± 0 . 06     |
| Deconfounded ( D Z = 20 ) | 3 . 54 ± 0 . 04          | 2 . 55 ± 0 . 05          | 11 . 57 ± 0 . 12 | 9 . 63 ± 0 . 14  | 2 . 40 ± 0 . 01     | 1 . 28 ± 0 . 08     |

2019). For each patient, we extracted 25 patient covariates consisting of lab tests and vital signs measured over time that affect the assignment of treatments. We used daily aggregates of the patient covariates and treatments and patient trajectories of up to 50 timesteps. We estimate the effects of antibiotics, vasopressors, and mechanical ventilator on the following patient covariates: white blood cell count, blood pressure, and oxygen saturation.

Hidden confounding is present in the dataset as patient comorbidities and several lab tests were not included. However, since this is a real dataset, it is not possible to evaluate the extent of hidden confounding or to estimate the true (Oracle) treatment responses.

Table 1 illustrates the RMSE when estimating one-step ahead treatment responses by using the MSM and R-MSN outcome models directly on the extracted dataset (Confounded) and after applying the Time Series Deconfounder and augmenting the dataset with the substitutes for the hidden confounders of different dimensionality D Z (Deconfounded). We notice that in all cases, the Time Series Deconfounder enables us to obtain a lower error when estimating the effect of antibiotics, vasopressors, and mechanical ventilator on the patients' white blood cell count, blood pressure, and oxygen saturation. By modeling the dependencies in the assigned treatments for each patient, the factor model part of the Time Series Deconfounder was able to infer latent variables that account for the unobserved information about the patient states. Using these substitutes for the hidden confounders in the outcome models resulted in better estimates of the treatment responses. While these results on real data require further validation from doctors (which is outside the scope of this paper), they indicate the potential of the method to be applied in real medical scenarios.

In Appendix E, we include results for an additional experimental set-up where we remove several patient covariates from the dataset and we show how the Time Series Deconfounder can be used to account for this bias. Moreover, in Appendix F, we provide further discussion and directions for future work.

## 8. Conclusion

The availability of observational data consisting of longitudinal information about patients prompted the development of methods for modeling the effects of treatments on the disease progression in patients. All existing methods for estimating the individualized effects of time-dependent treatment from observational data make the untestable assumption that there are no hidden confounders. In the longitudinal setting, this assumption is even more problematic than in the static setting. As the state of the patient changes over time and the complexity of the treatment assignments and responses increases, it becomes much easier to miss important confounding information.

In this paper, we proposed the Time Series Deconfounder, a method that takes advantage of the patterns in the multiple treatment assignments over time to infer latent variables that can be used as substitutes for the hidden confounders. Moreover, we developed a deep learning architecture based on an RNN with multitask output and variational dropout for building a factor model over time and computing the latent variables in practice. Through experimental results on both synthetic and real datasets, we show the effectiveness of the Time Series Deconfounder in removing the bias from the estimation of treatment responses over time in the presence of multi-cause hidden confounders.

## Acknowledgements

The authors would like to thank the reviewers for their helpful feedback. The research presented in this paper was supported by The Alan Turing Institute, under the EPSRC grant EP/N510129/1 and by the US Office of Naval Research (ONR).

## References

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Man´ e, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever,

- I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Vi´ egas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., and Zheng, X. TensorFlow: Largescale machine learning on heterogeneous systems, 2015. URL https://www.tensorflow.org/ . Software available from tensorflow.org.
- Alaa, A. and Schaar, M. Limits of estimating heterogeneous treatment effects: Guidelines for practical algorithm design. In International Conference on Machine Learning , pp. 129-138, 2018.
- Alaa, A. M. and van der Schaar, M. Bayesian inference of individualized treatment effects using multi-task gaussian processes. In Advances in Neural Information Processing Systems , pp. 3424-3432, 2017.
- Bartsch, H., Dally, H., Popanda, O., Risch, A., and Schmezer, P. Genetic risk profiles for cancer susceptibility and therapy response. In Cancer Prevention , pp. 19-36. Springer, 2007.
- Bernanke, B. S., Boivin, J., and Eliasz, P. Measuring the effects of monetary policy: a factor-augmented vector autoregressive (favar) approach. The Quarterly journal of economics , 120(1):387-422, 2005.
- Bica, I., Alaa, A. M., Jordon, J., and van der Schaar, M. Estimating counterfactual treatment outcomes over time through adversarially balanced representations. International Conference on Learning Representations , 2020a.
- Bica, I., Alaa, A. M., Lambert, C., and van der Schaar, M. From real-world patient data to individualized treatment effects using machine learning: Current and future methods to address underlying challenges. Clinical Pharmacology &amp; Therapeutics , 2020b.
- Bica, I., Jordon, J., and van der Schaar, M. Estimating the effects of continuous-valued interventions using generative adversarial networks. arXiv preprint arXiv:2002.12326 , 2020c.
- DAmour, A. On multi-cause approaches to causal inference with unobserved counfounding: Two cautionary failure cases and a promising alternative. In The 22nd International Conference on Artificial Intelligence and Statistics , pp. 3478-3486, 2019.
- Forni, M., Hallin, M., Lippi, M., and Reichlin, L. The generalized dynamic-factor model: Identification and estimation. Review of Economics and statistics , 82(4):540-554, 2000.
- Forni, M., Hallin, M., Lippi, M., and Reichlin, L. The generalized dynamic factor model: one-sided estimation and forecasting. Journal of the American Statistical Association , 100(471):830-840, 2005.
- Gal, Y. and Ghahramani, Z. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning , pp. 1050-1059, 2016a.
- Gal, Y. and Ghahramani, Z. A theoretically grounded application of dropout in recurrent neural networks. In Advances in neural information processing systems , pp. 1019-1027, 2016b.
- Geng, C., Paganetti, H., and Grassberger, C. Prediction of treatment response for combined chemo-and radiation therapy for non-small cell lung cancer patients using a bio-mathematical model. Scientific reports , 7(1):13542, 2017.
- Heckerman, D. Accounting for hidden common causes when infering cause and effect from observational data. arXiv preprint arXiv:1801.00727 , 2018.
- Hern´ an, M. A., Brumback, B., and Robins, J. M. Marginal structural models to estimate the joint causal effect of nonrandomized treatments. Journal of the American Statistical Association , 96(454):440-448, 2001.
- Hill, J. L. Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics , 20(1):217-240, 2011.
- Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural computation , 9(8):1735-1780, 1997.
- Howe, C. J., Cole, S. R., Mehta, S. H., and Kirk, G. D. Estimating the effects of multiple time-varying exposures using joint marginal structural models: alcohol consumption, injection drug use, and hiv acquisition. Epidemiology (Cambridge, Mass.) , 23(4):574, 2012.
- Imai, K. and Van Dyk, D. A. Causal inference with general treatment regimes: Generalizing the propensity score. Journal of the American Statistical Association , 99(467): 854-866, 2004.
- Jabbari, F., Ramsey, J., Spirtes, P., and Cooper, G. Discovery of causal models that contain latent variables through bayesian scoring of independence constraints. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pp. 142-157. Springer, 2017.
- Johnson, A. E., Pollard, T. J., Shen, L., Li-wei, H. L., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., and Mark, R. G. Mimic-iii, a freely accessible critical care database. Scientific data , 3:160035, 2016.
- Kallenberg, O. Foundations of modern probability . Springer Science &amp; Business Media, 2006.
- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.

- Kong, D., Yang, S., and Wang, L. Multi-cause causal inference with unmeasured confounding and binary outcome. arXiv preprint arXiv:1907.13323 , 2019.
- Kroschinsky, F., St¨ olzel, F., von Bonin, S., Beutel, G., Kochanek, M., Kiehl, M., and Schellongowski, P. New drugs, new toxicities: severe side effects of modern targeted and immunotherapy of cancer and their management. Critical Care , 21(1):89, 2017.
- Lash, T. L., Fox, M. P., MacLehose, R. F., Maldonado, G., McCandless, L. C., and Greenland, S. Good practices for quantitative bias analysis. International journal of epidemiology , 43(6):1969-1985, 2014.
- Lee, C., Mastronarde, N., and van der Schaar, M. Estimation of individual treatment effect in latent confounder models via adversarial learning. arXiv preprint arXiv:1811.08943 , 2018.
- Leray, P., Meganek, S., Maes, S., and Manderick, B. Causal graphical models with latent variables: learning and inference. In Innovations in Bayesian Networks , pp. 219-249. Springer, 2008.
- Lim, B., Alaa, A., and van der Schaar, M. Forecasting treatment responses over time using recurrent marginal structural networks. In Advances in Neural Information Processing Systems , pp. 7493-7503, 2018.
- Lok, J. J. et al. Statistical modeling of causal effects in continuous time. The Annals of Statistics , 36(3):14641507, 2008.
- Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., and Welling, M. Causal effect inference with deep latentvariable models. In Advances in Neural Information Processing Systems , pp. 6446-6456, 2017.
- Mansournia, M. A., Etminan, M., Danaei, G., Kaufman, J. S., and Collins, G. Handling time varying confounding in observational research. bmj , 359:j4587, 2017.
- Miao, W., Geng, Z., and Tchetgen Tchetgen, E. J. Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika , 105(4):987-993, 2018.
- Neil, D., Pfeiffer, M., and Liu, S.-C. Phased lstm: Accelerating recurrent network training for long or event-based sequences. In Advances in neural information processing systems , pp. 3882-3890, 2016.
- Neyman, J. Sur les applications de la th´ eorie des probabilit´ es aux experiences agricoles: Essai des principes. Roczniki Nauk Rolniczych , 10:1-51, 1923.
- Pearl, J. Causality . Cambridge university press, 2009.
- Platt, R. W., Schisterman, E. F., and Cole, S. R. Timemodified confounding. American journal of epidemiology , 170(6):687-694, 2009.
- Raghu, V. K., Ramsey, J. D., Morris, A., Manatakis, D. V., Sprites, P., Chrysanthis, P. K., Glymour, C., and Benos, P. V. Comparison of strategies for scalable causal discovery of latent variable models from mixed data. International journal of data science and analytics , 6(1):33-45, 2018.
- Ranganath, R. and Perotte, A. Multiple causal inference with latent confounding. arXiv preprint arXiv:1805.08273 , 2018.
- Ranganath, R., Tang, L., Charlin, L., and Blei, D. Deep exponential families. In Artificial Intelligence and Statistics , pp. 762-771, 2015.
- Robins, J. M. Correcting for non-compliance in randomized trials using structural nested mean models. Communications in Statistics-Theory and methods , 23(8):2379-2412, 1994.
- Robins, J. M. and Hern´ an, M. A. Estimation of the causal effects of time-varying exposures. In Longitudinal data analysis , pp. 547-593. Chapman and Hall/CRC, 2008.
- Robins, J. M., Hernan, M. A., and Brumback, B. Marginal structural models and causal inference in epidemiology, 2000a.
- Robins, J. M., Rotnitzky, A., and Scharfstein, D. O. Sensitivity analysis for selection bias and unmeasured confounding in missing data and causal inference models. In Statistical models in epidemiology, the environment, and clinical trials , pp. 1-94. Springer, 2000b.
- Roy, J., Lum, K. J., and Daniels, M. J. A bayesian nonparametric approach to marginal structural models for point treatments and a continuous or survival outcome. Biostatistics , 18(1):32-47, 2016.
- Rubin, D. B. Bayesian inference for causal effects: The role of randomization. The Annals of statistics , pp. 34-58, 1978.
- Rubin, D. B. Bayesianly justifiable and relevant frequency calculations for the applies statistician. The Annals of Statistics , pp. 1151-1172, 1984.
- Scharfstein, D., McDermott, A., D´ ıaz, I., Carone, M., Lunardon, N., and Turkoz, I. Global sensitivity analysis for repeated measures studies with informative drop-out: A semi-parametric approach. Biometrics , 74(1):207-219, 2018.

- Scheeren, T. W., Bakker, J., De Backer, D., Annane, D., Asfar, P., Boerma, E. C., Cecconi, M., Dubin, A., D¨ unser, M. W., Duranteau, J., et al. Current use of vasopressors in septic shock. Annals of intensive care , 9(1):20, 2019.
- Schmidt, G. A., Mandel, J., Sexton, D. J., and Hockberger, R. S. Evaluation and management of suspected sepsis and septic shock in adults. UpToDate. Available online: https://www. uptodate. com/contents/evaluation-andmanagement-of-suspected-sepsisand-septic-shock-inadults (accessed on 29 September 2017) , 2016.
- Schulam, P. and Saria, S. Reliable decision support using counterfactual models. In Advances in Neural Information Processing Systems , pp. 1697-1708, 2017.
- Shalit, U., Johansson, F. D., and Sontag, D. Estimating individual treatment effect: generalization bounds and algorithms. In International Conference on Machine Learning , pp. 3076-3085, 2017.
- Soleimani, H., Subbaswamy, A., and Saria, S. Treatmentresponse models for counterfactual reasoning with continuous-time, continuous-valued interventions. arXiv preprint arXiv:1704.02038 , 2017.
- Spirtes, P., Glymour, C. N., Scheines, R., and Heckerman, D. Causation, prediction, and search . MIT press, 2000.
- Tipping, M. E. and Bishop, C. M. Probabilistic principal component analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 61(3):611622, 1999.
- Tran, D. and Blei, D. M. Implicit causal models for genomewide association studies. International Conference on Learning Representations , 2018.
- Vlachostergios, P. J. and Faltas, B. M. Treatment resistance in urothelial carcinoma: an evolutionary perspective. Nature Reviews Clinical Oncology , pp. 1, 2018.
- Wager, S. and Athey, S. Estimation and inference of heterogeneous treatment effects using random forests. Journal of the American Statistical Association , 2017.
- Wang, Y. and Blei, D. M. The blessings of multiple causes. Journal of the American Statistical Association , (justaccepted):1-71, 2019a.
- Wang, Y. and Blei, D. M. Multiple causes: A causal graphical view. arXiv preprint arXiv:1905.12793 , 2019b.
- Yoon, J., Jordon, J., and van der Schaar, M. Ganite: Estimation of individualized treatment effects using generative adversarial nets. International Conference on Learning Representations (ICLR) , 2018.
- Zhang, Y., Bellot, A., and van der Schaar, M. Learning overlapping representations for the estimation of individualized treatment effects. International Conference on Artificial Intelligence and Statistics , 2020.

## A. Proof for Theorem 1

Before proving Theorem 1, we introduce several definitions and lemmas that will aid with the proof. Note that the these are extended from the static setting in Wang &amp; Blei (2019a). Remember that at each timestep t , the random variable Z t ∈ Z t is constructed as a function of the history until timestep t : Z t = g ( ¯ H t -1 ) , where ¯ H t -1 = ( ¯ Z t -1 , ¯ X t -1 , ¯ A t -1 ) takes values in ¯ H t -1 = ¯ Z t -1 × ¯ X t -1 × ¯ A t -1 and g : ¯ H t -1 →Z . In order to obtain sequential ignorable treatment assignment using the substitutes for the hidden confounders Z t , the following property needs to hold:

<!-- formula-not-decoded -->

## Definition 1. Sequential Kallenberg construction

∀ ¯ a ≥ t and ∀ t ∈ { 0 , . . . , T } .

At timestep t , we say that the distribution of assigned causes ( A t 1 , . . . A tk ) admits a sequential Kallenberg construction from the random variables Z t = g ( ¯ H t -1 ) and X t if there exist measurable functions f tj : Z t ×X t × [0 , 1] → A j and random variables U jt ∈ [0 , 1] , with j = 1 , . . . , k such that:

<!-- formula-not-decoded -->

where U tj marginally follow Uniform [0 , 1] and jointly satisfy:

<!-- formula-not-decoded -->

Lemma 1. Sequential Kallenberg construction at each timestep t ⇒ Sequential strong ignorability . If at every timestep t , the distribution of assigned causes ( A t 1 , . . . A tk ) admits a Kallenberg construction from Z t = g ( ¯ H t -1 ) and X t then we obtain sequential strong ignorability.

for all ¯ a ≥ t .

Proof. Assume that A j for j = 1 , . . . , k are Borel spaces. For any t ∈ { 1 , . . . , T } assume Z t and X t are measurable spaces and assume that A tj = f tj ( Z t , X t , U tj ) , where f tj are measurable and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for all ¯ a ≥ t . This implies that:

Since the A tj 's are measurable functions of ( Z t , X t , U t 1 , . . . U tk ) and ¯ H t -1 = ( ¯ Z t -1 , ¯ X t -1 , ¯ A t -1 ) , we have that sequential strong ignorability holds:

<!-- formula-not-decoded -->

Lemma 2. Factor models for the assigned causes ⇒ Sequential Kallenberg construction at each timestep t . Under weak regularity conditions, if the distribution of assigned causes p (¯ a T ) can be written as the factor model p ( θ 1: k , ¯ x T , ¯ z T , ¯ a T ) then we obtain a sequential Kallenberg construction for each timestep.

Regularity condition: The domains of the causes A j for j = 1 , . . . , k are Borel subsets of compact intervals. Without loss of generality, assume A j = [0 , 1] for j = 1 , . . . , k .

The proof for Lemma 2 uses Lemma 2.22 in Kallenberg (2006) (kernels and randomization): Let µ be a probability kernel from a measurable space S to a Borel space T . Then there exists some measurable function f : S × [0 , 1] → T such that if ϑ is U (0 , 1) , then f ( s, ϑ ) has distribution µ ( s, ˙ ) for every s ∈ S .

Proof. For timestep t , consider the random variables A t 1 ∈ A 1 , . . . A tk ∈ A k , X t ∈ X t , Z t = g ( ¯ H t -1 ) ∈ Z t and θ j ∈ Θ . Assume sequential single strong ignorability holds. Without loss of generality, assume A j = [0 , 1] for j = 1 , . . . , k .

From Lemma 2.22 in Kallenberg (1997), there exists some measurable function f tj : Z t ×X t × [0 , 1] → [0 , 1] such that U tj ∼ Uniform [0 , 1] and:

<!-- formula-not-decoded -->

and there exists some measurable function h tj : Θ × [0 , 1] → [0 , 1] such that:

<!-- formula-not-decoded -->

where ω tj ∼ Uniform [0 , 1] and j = 1 , . . . , k .

From our definition of the factor model we have that ω tj for j = 1 , . . . , k are jointly independent. Otherwise, A tj = f tj ( Z t , X t , h tj ( θ j , ω tj )) would not have been conditionally independent given Z t , X t .

Since sequential single strong ignorability holds at each timestep t , we have that A tj ⊥ ⊥ Y (¯ a ≥ t ) | X t , ¯ H t -1 ∀ ¯ a ∈ ¯ A , ∀ t ∈ { 0 , . . . , T } and for j = 1 , . . . , k which implies:

<!-- formula-not-decoded -->

∀ ¯ a ≥ t and ∀ j ∈ { 1 , . . . , k } . Using this, we can write:

<!-- formula-not-decoded -->

where the second and third steps follow form equation (26) and the fact that ω t 1 , . . . , ω tk are jointly independent. This gives us:

<!-- formula-not-decoded -->

Moreover, since the latent random variable Z t is constructed without knowledge of Y (¯ a ≥ t ) , but rather as a function of the history ¯ H t -1 we have:

<!-- formula-not-decoded -->

θ 1: k are parameters in the factor model and can be considered point masses, so we also have that:

<!-- formula-not-decoded -->

Since U tj = ( h ij ( θ j , ω tj )) are measurable functions of θ j and ω tj we have that:

<!-- formula-not-decoded -->

We have thus obtained a sequential Kallenberg construction at timestep t .

Theorem 2. If the distribution of the assigned causes p (¯ a T ) can be written as the factor model p ( θ 1: k , ¯ x T , ¯ z T , ¯ a T ) then we obtain sequential ignorable treatment assignment:

<!-- formula-not-decoded -->

for all ¯ a ≥ t and for all t ∈ { 0 , . . . , T } .

Proof. Theorem 1 follows from Lemmas 1 and 2. In particular, using the proposed factor graph, we can obtain a sequential Kallenberg construction at each timestep and then obtain sequential strong ignorability.

## B. Implementation Details for the Factor Model

The factor model described in Section 5 was implemented in Tensorflow (Abadi et al., 2015) and trained on an NVIDIA Tesla K80 GPU. For each synthetic dataset (simulated as described in Section 6.1), we obtained 5000 patients, out of which 4000 were used for training, 500 for validation, and 500 for testing. Using the validation set, we perform hyperparameter optimization using 30 iterations of random search to find the optimal values for the learning rate, minibatch size (M), RNN hidden units, multitask FC hidden units and RNN dropout probability. LSTM (Hochreiter &amp; Schmidhuber, 1997) units are used for the RNN implementation. The search range for each hyperparameter is described in Table 2.

The trajectories for the patients do not necessarily have to be equal. However, to be able to train the factor model, we zero-padded them such that they all had the same length. The patient trajectories were then grouped into minibatches of size Mand the factor model was trained using the Adam optimizer (Kingma &amp; Ba, 2014) for 100 epochs.

Table 2. Hyperparameter search range for the proposed factor model implemented using a recurrent neural network with multitask output and variational dropout.

| Hyperparameter            | Search range            |
|---------------------------|-------------------------|
| Learning rate             | 0.01, 0.001, 0.0001     |
| Minibatch size            | 64, 128, 256            |
| RNN hidden units          | 32, 64, 128, 256        |
| Multitask FC hidden units | 32, 64, 128             |
| RNN dropout probability   | 0.1, 0.2, 0.3, 0.4, 0.5 |

Table 3 illustrates the optimal hyperparameters obtained for the factor model under the different amounts of hidden confounding applied (as described by the experiments in Section 6.1). Since the results for assessing the Time Series Deconfounder are averaged across 30 different simulated datasets, we report here the optimal hyperparameters identified through majority voting. We note that when the effect of the hidden confounders on the treatment assignments and the outcome is large, more capacity is needed in the factor model to be able to infer them.

Table 3. Optimal hyperparameters for the factor model when different amounts of hidden confounding are applied in the synthetic dataset. The parameter γ measures the amount of hidden confounding applied.

| Hyperparameter            |   γ = 0 |   γ = 0 . 2 |   γ = 0 . 4 |   γ = 0 . 6 |   γ = 0 . 8 |
|---------------------------|---------|-------------|-------------|-------------|-------------|
| Learning rate             |    0.01 |        0.01 |        0.01 |        0.01 |       0.001 |
| Minibatch size            |   64    |       64    |       64    |       64    |     128     |
| RNN hidden units          |   32    |       64    |       64    |      128    |     128     |
| Multitask FC hidden units |   64    |      128    |       64    |      128    |     128     |
| RNN dropout probability   |    0.2  |        0.2  |        0.1  |        0.3  |       0.3   |

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

&lt;latexi sh

1\_b

64="(nu

)

&gt;

/

Figure 5. (a) Proposed factor model using a recurrent neural network with multitask output and variational dropout. (b) Alternative design without multitask output. (c) Factor model using an MLP (shared across timestep) and multitask output. This baseline does not capture time-dependencies. MC dropout (Gal &amp; Ghahramani, 2016a) is applied in the MLP to be able to sample from the substitutes for the hidden confounders.

![Image](data:image/png;base64,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)

## C. Baselines for Evaluating Factor Model

Figure 5 illustrates the architecture at each timestep for our proposed factor model and the baselines used for comparison. Figure 5(a) represents our proposed architecture for the factor model consisting of a recurrent neural network with multitask output and variational dropout. We want to ensure that the multitask constraint does not cause a decrease in the capability of the network to capture the distribution of the assigned causes. To do so, we compare our proposed factor model with the network in Figure 5(b) where we predict the k treatment assignments by passing X t and Z t through a hidden layer and having an output layer with k neurons. Moreover, to highlight the importance of learning time-dependencies to estimate the substitutes for the hidden confounders, we also use as a baseline the factor model in Figure 5(c). In this case, a multilayer perceptron (MLP) is shared across the timesteps and it infers the latent variable Z t using only the previous covariates and treatments. Note that in this case there is no dependency on the entire history.

The baselines were optimised under the same set-up described for our proposed factor model in Appendix B. Tables 4 and 5 describe the search ranges used for the hyperparameters in each of the baselines.

Table 4. Hyperparameter search range for factor model without multitask (Figure 5(b)).

| Hyperparameter            | Search range            |
|---------------------------|-------------------------|
| Learning rate             | 0.01, 0.001, 0.0001     |
| Minibatch size            | 64, 128, 256            |
| Max gradient norm         | 1.0, 2.0, 4.0           |
| RNN hidden units          | 32, 64, 128, 256        |
| Multitask FC hidden units | 32, 64, 128             |
| RNN dropout probability   | 0.1, 0.2, 0.3, 0.4, 0.5 |

Table 5. Hyperparameter search range for MLP factor model. Figure 5(c))

| Hyperparameter            | Search range            |
|---------------------------|-------------------------|
| Learning rate             | 0.01, 0.001, 0.0001     |
| Minibatch size            | 64, 128, 256            |
| MLP hidden layer size     | 32, 64, 128, 256        |
| Multitask FC hidden units | 32, 64, 128             |
| MLP dropout probability   | 0.1, 0.2, 0.3, 0.4, 0.5 |

## D. Outcome Models

After inferring the substitutes for the hidden confounders using the factor model, we implement outcome models to estimate the individualised treatment responses:

<!-- formula-not-decoded -->

We train the outcome models and evaluate them on predicting the treatment responses for each timestep, i.e. one-step-ahead predictions, for the patients in the test set. For training and tuning the outcome models, we use the same train/validation/test splits that we have used for the factor model. This means that the substitutes for the hidden confounders estimated using the fitted factor model on the test set are also used for testing purposes in the outcome models.

## D.1. Marginal Structural Models

MSMs (Robins et al., 2000a; Hern´ an et al., 2001) have been widely used in epidemiology to perform causal inference in longitudinal data. MSMs use inverse probability of treatment weighting during training to construct a pseudo-population from the observational data that resembles the one in a clinical trial and thus remove the bias introduced by time-dependent confounders (Platt et al., 2009). The propensity scores for each timestep are computed as follows:

<!-- formula-not-decoded -->

where f ( · ) is the conditional probability mass function for discrete treatments and the conditional probability density function for continuous treatments. We adopt the implementation in Hern´ an et al. (2001); Howe et al. (2012) for MSMs and estimate the propensity weights using logistic regression as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ω glyph[star] , φ glyph[star] and w glyph[star] are regression coefficients and σ ( · ) is the sigmoid function.

For predicting the outcome, the following regression model is used, where each individual patient is weighted by its propensity score:

<!-- formula-not-decoded -->

where β glyph[star] and l glyph[star] are regression coefficients. Since MSMs do not require hyperparameter tuning, we train them on the patients from both the train and validation sets.

## D.2. Recurrent Marginal Structural Networks

R-MSNs, implemented as descried in Lim et al. (2018) 5 , use recurrent neural networks to estimate the propensity scores and to build the outcome model. The use of RNNs is more robust to changes in the treatment assignment policy. Moreover, R-MSNs represent the first application of deep learning in predicting time-dependent treatment effects. The propensity weights are estimated using recurrent neural networks as follows:

<!-- formula-not-decoded -->

For predicting the outcome, the following prediction network is used:

<!-- formula-not-decoded -->

5 We used the publicly available immlementation from https://github.com/sjblim/rmsn\_nips\_2018.

where in the loss function, each patient is weighted by its propensity score. Since the purpose of our method is not to improve predictions, but rather to assess how well the R-MSNs can be deconfounded using our method, we use the optimal hyperparameters for this model, as identified by Lim et al. (2018). R-MSNs are then trained on the combined set of patients from the training and validation sets.

Table 6. Hyperparameters used for R-MSN.

| Hyperparameter   | Propensity networks   | Propensity networks   | Propensity networks   | Prediction network   |
|------------------|-----------------------|-----------------------|-----------------------|----------------------|
| Hyperparameter   | t &#124; ¯ A t - 1    | ) f ( A t             | &#124; ¯ H t )        | Prediction network   |
| Dropout rate     | 0.1                   | 0.1                   |                       | 0.1                  |
| State size       | 6                     | 16                    |                       | 16                   |
| Minibatch size   | 128                   |                       | 64                    | 128                  |
| Learning rate    | 0.01                  |                       | 0.01                  | 0.01                 |
| Max norm         | 2.0                   |                       | 1.0                   | 0.5                  |

R-MSNs (Lim et al., 2018), can also be used to forecast treatment responses for an arbitrary number of steps in the future. In our paper we focus on one-step ahead predictions of the treatment responses. However, the Time Series Deconfounder can also be applied to estimate the effects of a sequence of future treatments.

Figure 6. Results for deconfounding one-step ahead estimation of treatment responses in two outcome models: (a) Marginal Structural Models (MSM) and (b) Recurrent Marginal Structural Networks (R-MSN). The simulated (true) size of the hidden confounders is D Z = 3 . The average RMSE and the standard error in the results are computed for 30 dataset simulations for each different degree of confounding, as measured by γ .

![Image](data:image/png;base64,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VcFg8OOPP961axdBEBEREU6nc/z48dOnT+/6iZqamhYsWFBdXd2nT5/q6urMzEzZu/3uu+9WrlzJ83z//v3vuecerVa7bt26b7/9luO4yMhIjuPCw8NvvfXWhQsXxsfHV1VVabXaf//73xs3bvz4449pmo6Jibn//vsTEhLKy8vffvttm82m1WrvvPPO46dJ29vbP/jgg8LCQovF8s9//jM1NdVmszU0NOTl5SkB7AsX5ZtTOC2CwSDLsifMDG5ubl6wYEF2dvYLL7xw6623fv755xs2bIBhuKmpaffu3ZMmTXr00UcPHDiwfv16s9k8e/bsiIiI6667buLEiWvXrv3yyy9vvfXWF198MSsr6+WXX25ra+N5/tChQ/K1fjAYPHTokBzeOGZrmzZtkksudu3aNW/evDlz5shX6hAEsSx76NAhOTP4hK/66aefKioq/v3vfw8YMKC4uPjWW28dOHBg109EUdSrr77KMMxTTz01ZcqUziTjn376afPmzf/617+eeOKJ3377bdu2bYcOHfriiy9uvPHGO+64Y/fu3UOHDp09e7YoilVVVZ999llubu6MGTMKCws//vjjG2+8ccGCBX6//9tvv6Uo6vXXX1epVK+88srMmTM//vjjpqamrjvgdrtXrFgxfPjwF198ccCAARs2bOB5vry8vNc2IlZQUDgdaJpmGEar1R7/lMPhWLBgQXh4+Pz58++9995ffvll8eLF8uObN28eMWLEU0891dDQsHLlytjY2BkzZoSHh8+dO3fkyJG7du16//33Z86c+eKLL44ZM+Y///mPnChcVVXldrvlN62qqpKNtt1u77q1n3/+GQCwaNGiX3/99Z577rn99ttJkmRZFoIgjuMOHz7s9Xr/6FVr1qzZsmXLvHnzLrnkksLCwuuvv37UqFFdPxHHce+++25TU9O///3vK664gud5OVi7du3aRYsWPfDAAy+88MKhQ4d+/vnnpqam//73v1OnTn344YcPHDiQlpZ20003kSRZX1//0UcfxcbGXnvtteXl5W+99daMGTNeeeUVrVb75Zdf2my2hQsXynGZ4cOHf/DBB/Ledj3aK1asyMjImD9//pQpU9asWcOy7OHDhzUajeIWX9AoMWOF00Kr1ZIk2d7efvxT69atQ1F0zpw5arU6MTGxuLh46dKlo0ePhmF4yJAhgwcP5jguMTHR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NmzYe++9J29k48aNsss7c+bMZ5999o033hg0aFBJSYnVapUHrKysLJ1O53a7a2pqEhMTZUMtD0wAgPHjx//666+PP/74lClTGhsb29vbb7rppj59+jzxxBPXXnsty7LFxcVz5849PnTdCcuyaWlpKpXq7HyxCucOxTNWOF1gGL799ttzcnKWLFlSWlqK43jfvn3z8/O1Wu3zzz//xRdf/PTTTyaT6dlnn+3fv78gCAUFBXJCAoIgQ4cOTU1NBQDo9fqxY8fKV/9Dhgx59tlnv/vuu/Ly8szMzOuuu06r1Wo0muuuu279+vXNzc2DBg2yWCxy2d8xW5PDHtdddx3P86tXr05NTb388svlBtEajWbcuHFysdrx+8CyrM/nO3z4sCwz1NjYGBsbu2DBAnk1GRRFH3roof/9738//vhjfn7+DTfckJSUBACQE5qXLl26a9eupKSknJycuro6u92+detWHMedTudbb7319NNP5+bmjh49ujPMkJSU9Pzzz3/99deffPKJHHLGMOzee+/97rvvVqxYIR+HrknGx2O32/9s/yoFBYVey8yZMxMTExctWvTtt98iCCLnwqrV6ieeeOLzzz9ftmyZVqv95z//Kdd49O3btzPiO3DgQLl6z2KxjBkzRl7OyMh44YUXvvnmmx9//DEuLu6+++6Tpwevu+66xYsXf/XVV/369bvxxhvltLQTbm3q1Kler3fDhg0NDQ0TJ070er2ysObo0aNl03f8q2TluLq6ui1btoiiKOfsvfDCC7JHK4MgyJ133gnD8LJly/r06SOXd8tp00899dS333776aefxsTEjBkzJhAIuN3ubdu2yflvb7zxxj/+8Y9JkyaNGDEiOTlZ3lpERMRzzz332Wefff755waDYcqUKXq9/tFHH/3iiy8WLVqE4/iIESNOXgPt8Xj+bEdYhZ6JFGA73/ugcOHB8/xZbU8vz9ydjS3X1dXdc889jz/++JAhQwAA5eXlf//73+fNm3fC9qdyhvEfbUoQhL///e/x8fEPP/ywXKI3b968sLCwp5566oTrn3xrfwRFUZ9//vmsWbMUYQoFBYU/hSAIZ7U44ewZ6vb29rvvvvvmm2+WawRbWlruvPPO66+//qqrrjr9jXSa3GeffTYYDL744otyTPqFF16w2+2vv/76CUexv2aoeZ7/9ttvBw8eLEdtFC5olJix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N954Y3Jy8jHnw1133eXxeD788MPy8vLHH3/8hhtuyM7O/vLLL3fu3EkQxIwZM8aNG2ez2b755puZM2fGxMTs3LnzwIEDc+fOra+vX7RokewvqlSqv/3tb7GxsT6f79NPPy0rK0tPT3e5XPJXdvz5f+jQof/9739ycwee548/taqrqxEEOaY3b0VFhVqt7to1F0VRv9/v8XhO2cZcPobHn1rHn1eyP1deXs5xXHp6emdi9/k+rzroxgm2rkfDYDAMGzZsy5YtV199Nc/zX3zxxbZt2/R6/c0335yTk+P3+z/66KO9e/fCMDxmzJjrr7/++POqrq6u6/lwzz33IAhy8vOKYZgvvviib9++AwYMqKmpWb58+Zw5c1wu148//njMeSXHcdevX5+YmFhTUyML+xx//judzvfff7+pqSklJcXv9x/vHDscDpvNdkxrjKamJq/X2zWdDIZhjuPsdvtpHsbTOa/k3OjjT7bY2Fi73e5wOOSuJeea7p6vrampWbBgwbRp06666ioEQQ4cOPDcc8+pVKorrrjiL2yNCXU0BL2DXh8z7lZqa2u/++67O+64Y9CgQQiCEATRp0+frs3VTnm55vV61Wp1VFSUHPjsrnBR10dOOBH/R+GicwPU7Sbh9+h0Oq1Wa7PZXC7Xf/7znz59+jz//PONjY0///xzdXX1woULhw8f/vLLL1922WUoipaVlb355pvjxo17/PHHHQ7HG2+8wXFcaWnp999/P3jw4EsvvfSLL744dOhQcnJy//79ExISpk6diuP4K6+8AgB46qmnBgwYsHDhQlnbeOPGjXIgp6ioaN++fbLd//LLLy0Wy5w5c9avX79lyxaGYV566SW3233LLbcYjUaPx4NhWHl5+fvvv3/llVc+8sgj69atKyoqKi8vX7BgQX5+/uzZs30+n1z4cszHbGlp6Xqyyd91RUVFnz59uq5cV1enVqtlJ6OxsbGoqKiiouLgwYOVlZVdBxWO47Zu3frDDz8s+j0//vjjunXrjrnWgmHY5XKtXr16//79t956a2fqrcVi8fv952suIhTXO4vAMBwREdHc3Myy7DvvvCMIwvz5861W61dffeVyuV5++WWDwfDiiy/OmTNHo9HYbLaXX345Ojr66aefltsuejye5ubmrufDxo0bw8LChg8fHh4ePmXKlISEhA8++GDfvn2PPPLI1Vdf/cknn2zfvj0YDK5fv97lcsk+69atW2W78e2339pstltvvbWpqen7778XRfH999/fvn277AbZ7XYURd1ud+f539TUtGrVKqfTOX/+fLVafeONN2IY5vP5jj+vmpqadDrdMT5oSUlJUlJS1wlueU5GzuRxOp1FRUVlZWWHDh0qKyuTuwF3sn///hOeV8uWLXM6nV3XRFG0tbVVpVKJovjWW291TqOd7/PqrJ9f0dHRdrtdEIRvv/22qKjoscceGz9+/CeffOJ2u997772amprHH3/84YcfDg8Pb21tPea8crlcfr+/6/nwzTffWCyWk5xXW7duFUVxy5YttbW1ctPNdevWBYPBY7bz3XffAQBWrFjx2WefTZs2bdy4cW1tbRAEsSz79ttvy+d/WFjYF198QVHUq6++evjw4Tlz5sTExNjt9uPPq9bWVgRBjklFKy8v1+v1XcNJTqdTnp2TL5OKi4v3799/6NChAwcOOByOrq+tq6v76aefFp2IzubVJ0ej0cAw3NbWBi4Kfvjhh+jo6GuvvVb+8WZnZ1933XU//PDDMcetk2N+pyCUoef3+094gXH8PMbFxEUbM+ZstvY332QbGyHkmM8ofZfG667TjRnT8QCCMIcO1d/1NwiGulo6UeCJ1LTwfzwMhZJ9ufZ2WK+HTxpm27VrlzzNevxTZWVlX375ZXt7e2Ji4i233BIREbF///61a9dardYdO3ZERETcfffddrv9iy++qKio+Oc//3n77bdHRkZ++OGHBw4c0Ol0V1xxxZAhQwKBwCeffDJu3LjMzMwDBw5s3rx57ty5VVVVv/76a+d27rjjDqvVesLr9cbGxo8//ri6ujorK+uOO+7Q6/V79+794osvSJLsrkzZY7D77U8ufrK6rRpC/tD3hSG4sKYQ4OBQ66Hpb05/duazgxIHyU/NXzF/64GtydHJr13zGoZIoVC7347BmF7159LveJ5nWVatVpeUlNTW1ubn5//666//396VhzV15e17s0FCICQGUAIEJEQWSQIaIKCMtQWsgvrhMtW6zaig9sOn1bZPYaZat3FmXL5Ony7T1rq1iq2M41IVnAGEKijKokjYCWuQQCCE7Mm993vC0WsMCtqi9rG8f/HcXE7uvXnvOe/5nff3OxaLpbKysru729PTE8yhgW3xk08+CQgIABsvrV69Oj09XSaTUSgU8SD8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)

## E. Additional Results

## E.1. Experiments on Synthetic Data

We considered an additional experimental set-up where we have simulated hidden confounders of dimension D Z = 3 . In Figure 6 we illustrate the root mean squared error (RMSE) for one-step-ahead estimation of treatment responses for patients in the test set without adjusting for the bias from the hidden confounders (Confounded), when using the simulated hidden confounders (Oracle) and after applying the Time Series Deconfounder with different model specifications (Deconfounded). We notice that the Time Series Deconfounder can still account for the bias from hidden confounders when the true size for the hidden confounders is underestimated in the factor model and set to ( D Z = 1 ). The performance is improved when setting D Z to the true number of hidden confounders or when overestimating the number of hidden confounders.

## E.2. Model of Tumour Growth

To show the applicability of our method in a more realistic simulation, we use the pharmacokinetic-pharmacodynamic (PK-PD) model of tumor growth under the effects of chemotherapy and radiotherapy proposed by Geng et al. (2017). The tumor volume after t days since diagnosis is modeled as follows:

<!-- formula-not-decoded -->

where K,ρ,β c , α r , β r , e t are sampled as described in Geng et al. (2017). C ( t ) is the chemotherapy drug concentration and d ( t ) is the dose of radiation. Chemotherapy and radiotherapy prescriptions are modeled as Bernoulli random variables that depend on the tumor size. Full details about treatments are in Lim et al. (2018).

Table 7. Average RMSE × 10 2 (normalised by the maximum tumour volume) and the standard error in the results for predicting the effect of chemotherapy and radiotherapy on the tumour volume.

| Outcome model             | MSM           | R-MSN       |
|---------------------------|---------------|-------------|
| Confounded                | 7.29 ± 0.14   | 5.31 ± 0.16 |
| Deconfounded ( D Z = 1 )  | 6.47 ± 0.16   | 4.76 ± 0.17 |
| Deconfounded ( D Z = 5 )  | 6.25 ± 0.14   | 4.79 ± 0.19 |
| Deconfounded ( D Z = 10 ) | 6.31 ± 0.11   | 4.54 ± 0.17 |
| Oracle                    | 6.92 ± 0 . 19 | 5.00 ± 0.15 |

To account for patient heterogeneity due to genetic features (Bartsch et al., 2007), the prior means for β c and α r are adjusted according to three patient subgroups as described in Lim et al. (2018). The patient subgroup S ( i ) ∈ { 1 , 2 , 3 }

represents a confounder because it affects the tumor growth and subsequently the treatment assignments. We reproduced the experimental set-up in Lim et al. (2018) and simulated datasets with 10000 patients for training, 1000 for validation, and 1000 for testing. We simulated 30 datasets and averaged the results for testing the MSM and R-MSN outcome models without the information about patient types (confounded), with the true simulated patient types, as well as after applying the Time Series Deconfounder with D Z ∈ { 1 , 5 , 10 } .

The results in Table 7 indicate that our method can infer substitutes for static hidden confounders such as patient subgroups which affect the treatment responses over time. By construction, ¯ Z t also captures time dependencies which help with the prediction of outcomes. This is why the performance of the deconfounded models is slightly better than of the oracle model which uses static patient groups.

## E.3. MIMIC III

We performed an additional experiment using the dataset extracted from the MIMIC III database where we have removed 3 patient covariates from the dataset (temperature, glucose, hemoglobin). In Table 8 we report the results for estimating the effects of antibiotics, vasopressors, and mechanical ventilator on the patient's white blood cell count when including all variables, after removing these 3 patient covariates (which we notice that further confound the results) and after applying the Time Series Deconfounder with different settings for D Z .

Table 8. Average RMSE × 10 2 and the standard error in the results for predicting the effect of antibiotics, vasopressors, and mechanical ventilator on white blood cell count. The results are for 10 runs.

|                          | White blood cell   | White blood cell   |
|--------------------------|--------------------|--------------------|
| Outcome model            | MSM                | R-MSN              |
| All patient covariates   | 3 . 90 ± 0 . 00    | 2 . 91 ± 0 . 05    |
| Removed 3 covariates     | 4 . 12 ± 0 . 00    | 3 . 11 ± 0 . 03    |
| Deconfounded ( D Z = 1 ) | 3 . 98 ± 0 . 02    | 3 . 05 ± 0 . 05    |
| Deconfounded ( D Z = 3 ) | 3 . 91 ± 0 . 03    | 2 . 87 ± 0 . 08    |
| Deconfounded ( D Z = 5 ) | 3 . 85 ± 0 . 04    | 2 . 81 ± 0 . 03    |

## F. Discussion

The Time Series Deconfounder firstly builds a factor model to infer substitutes for the multi-cause hidden confounders. If Assumption 3 holds and the fitted factor model captures well the distribution of the assigned causes, which can be assessed through predictive checks, the substitutes for the hidden confounders help us obtain sequential strong ignorability (Theorem 1). Then, the Time Series Deconfounder uses the inferred substitutes for the hidden confounders in an outcome model that estimates individualized treatment responses. The experimental results show the applicability of the Time Series Deconfounder both in a controlled simulated setting and in a real dataset consisting of electronic health records from patients in the ICU. In these settings, the Time Series Deconfounder was able to remove the bias from hidden confounders when estimating treatment responses conditional on patient history.

In the static causal inference setting, several methods have been proposed to extend the deconfounder algorithm in Wang &amp; Blei (2019a). For instance, Wang &amp; Blei (2019b) augment the theory in the deconfounder algorithm in Wang &amp; Blei (2019a) by extending it to causal graphs and show that by using some of the causes as proxies of the shared confounder in the outcome model one can identify the effects of the other causes. DAmour (2019) also suggests using proxy variables to obtain non-parametric identification of the mean potential outcomes (Miao et al., 2018). Additionally, Kong et al. (2019) proves that identification of causal effects is possible in the multi-cause setting when the treatments are normally distributed and the outcome is binary and follows a logistic structural equation model.

For the Time Series Deconfounder, similarly to Wang &amp; Blei (2019a), identifiability can be assessed by computing the uncertainty in the outcome model estimates, as described in Section 4.2. When the treatment effects are non-identifiable, the Time Series Deconfounder estimates will have high variance. Thus, future work could explore building upon the results in Wang &amp; Blei (2019b) and DAmour (2019) and using proxy variables in the outcome model to prove identifiability of causal estimates in the multi-cause time-series setting.

## References

- Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jia, Y., Jozefowicz, R., Kaiser, L., Kudlur, M., Levenberg, J., Man´ e, D., Monga, R., Moore, S., Murray, D., Olah, C., Schuster, M., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Vi´ egas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., and Zheng, X. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. URL https: //www.tensorflow.org/ . Software available from tensorflow.org.
- Alaa, A. and Schaar, M. Limits of estimating heterogeneous treatment effects: Guidelines for practical algorithm design. In International Conference on Machine Learning , pp. 129-138, 2018.
- Alaa, A. M. and van der Schaar, M. Bayesian inference of individualized treatment effects using multi-task gaussian processes. In Advances in Neural Information Processing Systems , pp. 3424-3432, 2017.
- Bartsch, H., Dally, H., Popanda, O., Risch, A., and Schmezer, P. Genetic risk profiles for cancer susceptibility and therapy response. In Cancer Prevention , pp. 19-36. Springer, 2007.
- Bernanke, B. S., Boivin, J., and Eliasz, P. Measuring the effects of monetary policy: a factor-augmented vector autoregressive (favar) approach. The Quarterly journal of economics , 120(1):387-422, 2005.
- Bica, I., Alaa, A. M., Jordon, J., and van der Schaar, M. Estimating counterfactual treatment outcomes over time through adversarially balanced representations. International Conference on Learning Representations , 2020a.
- Bica, I., Alaa, A. M., Lambert, C., and van der Schaar, M. From real-world patient data to individualized treatment effects using machine learning: Current and future methods to address underlying challenges. Clinical Pharmacology &amp; Therapeutics , 2020b.
- Bica, I., Jordon, J., and van der Schaar, M. Estimating the effects of continuous-valued interventions using generative adversarial networks. arXiv preprint arXiv:2002.12326 , 2020c.
- DAmour, A. On multi-cause approaches to causal inference with unobserved counfounding: Two cautionary failure cases and a promising alternative. In The 22nd International Conference on Artificial Intelligence and Statistics , pp. 3478-3486, 2019.
- Forni, M., Hallin, M., Lippi, M., and Reichlin, L. The generalized dynamic-factor model: Identification and estimation. Review of Economics and statistics , 82(4):540-554, 2000.
- Forni, M., Hallin, M., Lippi, M., and Reichlin, L. The generalized dynamic factor model: one-sided estimation and forecasting. Journal of the American Statistical Association , 100(471):830-840, 2005.
- Gal, Y. and Ghahramani, Z. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning , pp. 1050-1059, 2016a.
- Gal, Y. and Ghahramani, Z. A theoretically grounded application of dropout in recurrent neural networks. In Advances in neural information processing systems , pp. 1019-1027, 2016b.
- Geng, C., Paganetti, H., and Grassberger, C. Prediction of treatment response for combined chemo-and radiation therapy for non-small cell lung cancer patients using a bio-mathematical model. Scientific reports , 7(1):13542, 2017.
- Heckerman, D. Accounting for hidden common causes when infering cause and effect from observational data. arXiv preprint arXiv:1801.00727 , 2018.
- Hern´ an, M. A., Brumback, B., and Robins, J. M. Marginal structural models to estimate the joint causal effect of nonrandomized treatments. Journal of the American Statistical Association , 96(454):440-448, 2001.
- Hill, J. L. Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics , 20(1): 217-240, 2011.
- Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural computation , 9(8):1735-1780, 1997.

- Howe, C. J., Cole, S. R., Mehta, S. H., and Kirk, G. D. Estimating the effects of multiple time-varying exposures using joint marginal structural models: alcohol consumption, injection drug use, and hiv acquisition. Epidemiology (Cambridge, Mass.) , 23(4):574, 2012.
- Imai, K. and Van Dyk, D. A. Causal inference with general treatment regimes: Generalizing the propensity score. Journal of the American Statistical Association , 99(467):854-866, 2004.
- Jabbari, F., Ramsey, J., Spirtes, P., and Cooper, G. Discovery of causal models that contain latent variables through bayesian scoring of independence constraints. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , pp. 142-157. Springer, 2017.
- Johnson, A. E., Pollard, T. J., Shen, L., Li-wei, H. L., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., and Mark, R. G. Mimic-iii, a freely accessible critical care database. Scientific data , 3:160035, 2016.
- Kallenberg, O. Foundations of modern probability . Springer Science &amp; Business Media, 2006.
- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014.
- Kong, D., Yang, S., and Wang, L. Multi-cause causal inference with unmeasured confounding and binary outcome. arXiv preprint arXiv:1907.13323 , 2019.
- Kroschinsky, F., St¨ olzel, F., von Bonin, S., Beutel, G., Kochanek, M., Kiehl, M., and Schellongowski, P. New drugs, new toxicities: severe side effects of modern targeted and immunotherapy of cancer and their management. Critical Care , 21 (1):89, 2017.
- Lash, T. L., Fox, M. P., MacLehose, R. F., Maldonado, G., McCandless, L. C., and Greenland, S. Good practices for quantitative bias analysis. International journal of epidemiology , 43(6):1969-1985, 2014.
- Lee, C., Mastronarde, N., and van der Schaar, M. Estimation of individual treatment effect in latent confounder models via adversarial learning. arXiv preprint arXiv:1811.08943 , 2018.
- Leray, P., Meganek, S., Maes, S., and Manderick, B. Causal graphical models with latent variables: learning and inference. In Innovations in Bayesian Networks , pp. 219-249. Springer, 2008.
- Lim, B., Alaa, A., and van der Schaar, M. Forecasting treatment responses over time using recurrent marginal structural networks. In Advances in Neural Information Processing Systems , pp. 7493-7503, 2018.
- Lok, J. J. et al. Statistical modeling of causal effects in continuous time. The Annals of Statistics , 36(3):1464-1507, 2008.
- Louizos, C., Shalit, U., Mooij, J. M., Sontag, D., Zemel, R., and Welling, M. Causal effect inference with deep latent-variable models. In Advances in Neural Information Processing Systems , pp. 6446-6456, 2017.
- Mansournia, M. A., Etminan, M., Danaei, G., Kaufman, J. S., and Collins, G. Handling time varying confounding in observational research. bmj , 359:j4587, 2017.
- Miao, W., Geng, Z., and Tchetgen Tchetgen, E. J. Identifying causal effects with proxy variables of an unmeasured confounder. Biometrika , 105(4):987-993, 2018.
- Neil, D., Pfeiffer, M., and Liu, S.-C. Phased lstm: Accelerating recurrent network training for long or event-based sequences. In Advances in neural information processing systems , pp. 3882-3890, 2016.
- Neyman, J. Sur les applications de la th´ eorie des probabilit´ es aux experiences agricoles: Essai des principes. Roczniki Nauk Rolniczych , 10:1-51, 1923.
- Pearl, J. Causality . Cambridge university press, 2009.
- Platt, R. W., Schisterman, E. F., and Cole, S. R. Time-modified confounding. American journal of epidemiology , 170(6): 687-694, 2009.
- Raghu, V. K., Ramsey, J. D., Morris, A., Manatakis, D. V., Sprites, P., Chrysanthis, P. K., Glymour, C., and Benos, P. V. Comparison of strategies for scalable causal discovery of latent variable models from mixed data. International journal of data science and analytics , 6(1):33-45, 2018.

- Ranganath, R. and Perotte, A. Multiple causal inference with latent confounding. arXiv preprint arXiv:1805.08273 , 2018.
- Ranganath, R., Tang, L., Charlin, L., and Blei, D. Deep exponential families. In Artificial Intelligence and Statistics , pp. 762-771, 2015.
- Robins, J. M. Correcting for non-compliance in randomized trials using structural nested mean models. Communications in Statistics-Theory and methods , 23(8):2379-2412, 1994.
- Robins, J. M. and Hern´ an, M. A. Estimation of the causal effects of time-varying exposures. In Longitudinal data analysis , pp. 547-593. Chapman and Hall/CRC, 2008.
- Robins, J. M., Hernan, M. A., and Brumback, B. Marginal structural models and causal inference in epidemiology, 2000a.
- Robins, J. M., Rotnitzky, A., and Scharfstein, D. O. Sensitivity analysis for selection bias and unmeasured confounding in missing data and causal inference models. In Statistical models in epidemiology, the environment, and clinical trials , pp. 1-94. Springer, 2000b.
- Roy, J., Lum, K. J., and Daniels, M. J. A bayesian nonparametric approach to marginal structural models for point treatments and a continuous or survival outcome. Biostatistics , 18(1):32-47, 2016.
- Rubin, D. B. Bayesian inference for causal effects: The role of randomization. The Annals of statistics , pp. 34-58, 1978.
- Rubin, D. B. Bayesianly justifiable and relevant frequency calculations for the applies statistician. The Annals of Statistics , pp. 1151-1172, 1984.
- Scharfstein, D., McDermott, A., D´ ıaz, I., Carone, M., Lunardon, N., and Turkoz, I. Global sensitivity analysis for repeated measures studies with informative drop-out: A semi-parametric approach. Biometrics , 74(1):207-219, 2018.
- Scheeren, T. W., Bakker, J., De Backer, D., Annane, D., Asfar, P., Boerma, E. C., Cecconi, M., Dubin, A., D¨ unser, M. W., Duranteau, J., et al. Current use of vasopressors in septic shock. Annals of intensive care , 9(1):20, 2019.
- Schmidt, G. A., Mandel, J., Sexton, D. J., and Hockberger, R. S. Evaluation and management of suspected sepsis and septic shock in adults. UpToDate. Available online: https://www. uptodate. com/contents/evaluation-and-management-ofsuspected-sepsisand-septic-shock-in-adults (accessed on 29 September 2017) , 2016.
- Schulam, P. and Saria, S. Reliable decision support using counterfactual models. In Advances in Neural Information Processing Systems , pp. 1697-1708, 2017.
- Shalit, U., Johansson, F. D., and Sontag, D. Estimating individual treatment effect: generalization bounds and algorithms. In International Conference on Machine Learning , pp. 3076-3085, 2017.
- Soleimani, H., Subbaswamy, A., and Saria, S. Treatment-response models for counterfactual reasoning with continuous-time, continuous-valued interventions. arXiv preprint arXiv:1704.02038 , 2017.
- Spirtes, P., Glymour, C. N., Scheines, R., and Heckerman, D. Causation, prediction, and search . MIT press, 2000.
- Tipping, M. E. and Bishop, C. M. Probabilistic principal component analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology) , 61(3):611-622, 1999.
- Tran, D. and Blei, D. M. Implicit causal models for genome-wide association studies. International Conference on Learning Representations , 2018.
- Vlachostergios, P. J. and Faltas, B. M. Treatment resistance in urothelial carcinoma: an evolutionary perspective. Nature Reviews Clinical Oncology , pp. 1, 2018.
- Wager, S. and Athey, S. Estimation and inference of heterogeneous treatment effects using random forests. Journal of the American Statistical Association , 2017.
- Wang, Y. and Blei, D. M. The blessings of multiple causes. Journal of the American Statistical Association , (just-accepted): 1-71, 2019a.

- Wang, Y. and Blei, D. M. Multiple causes: A causal graphical view. arXiv preprint arXiv:1905.12793 , 2019b.
- Yoon, J., Jordon, J., and van der Schaar, M. Ganite: Estimation of individualized treatment effects using generative adversarial nets. International Conference on Learning Representations (ICLR) , 2018.
- Zhang, Y., Bellot, A., and van der Schaar, M. Learning overlapping representations for the estimation of individualized treatment effects. International Conference on Artificial Intelligence and Statistics , 2020.
