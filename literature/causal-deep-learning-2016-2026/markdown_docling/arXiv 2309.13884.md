<!--
source: /Users/pranjal/Code/deep-inference/literature/causal-deep-learning-2016-2026/downloads/arXiv 2309.13884.pdf
backend: docling
part: 1/1
-->

## Estimating Treatment Effects Under Heterogeneous Interference

Xiaofeng Lin /a0 , Guoxi Zhang, Xiaotian Lu, Han Bao, Koh Takeuchi, and Hisashi Kashima

Graduate School of Informatics, Kyoto University, Kyoto, Japan {lxf,guoxi,lu}@ml.ist.i.kyoto-u.ac.jp ; {bao,takeuchi,kashima}@i.kyoto-u.ac.jp

Abstract. Treatment effect estimation can assist in effective decisionmaking in e-commerce, medicine, and education. One popular application of this estimation lies in the prediction of the impact of a treatment (e.g., a promotion) on an outcome (e.g., sales) of a particular unit (e.g., an item), known as the individual treatment effect (ITE). In many online applications, the outcome of a unit can be affected by the treatments of other units, as units are often associated, which is referred to as interference. For example, on an online shopping website, sales of an item will be influenced by an advertisement of its co-purchased item. Prior studies have attempted to model interference to estimate the ITE accurately, but they often assume a homogeneous interference, i.e., relationships between units only have a single view. However, in real-world applications, interference may be heterogeneous, with multi-view relationships. For instance, the sale of an item is usually affected by the treatment of its copurchased and co-viewed items. We hypothesize that ITE estimation will be inaccurate if this heterogeneous interference is not properly modeled. Therefore, we propose a novel approach to model heterogeneous interference by developing a new architecture to aggregate information from diverse neighbors. Our proposed method contains graph neural networks that aggregate same-view information, a mechanism that aggregates information from different views, and attention mechanisms. In our experiments on multiple datasets with heterogeneous interference, the proposed method significantly outperforms existing methods for ITE estimation, confirming the importance of modeling heterogeneous interference.

Keywords: Causal Inference· Treatment Effect Estimation· Heterogeneous Graphs· Interference

## 1 Introduction

In recent years, treatment effect estimation has been performed to enable effective decision-making in many fields, such as medicine [25], education [22], and e-commerce [18,29,37]. For example, estimating treatment effects helps us understand whether an advertisement affects the sales of the advertised products. The

Ϭ

Ϭ

16

p

p

v

v

1

p

1

2

p

2

1

1

1

g

z

W

t

z

W

2

2

v

′

W

1

If

p

W

= 1

t

= 1

1

g

3

g

3

v

′

g

1

3

2

W

2

g

p

z

1

3

v

v

′

3

′

g

2

2

Ϭ

W

2

z

Co-Purchased Graph

3

Co-Viewed Graph

Co-Purchased Graph graphs

3

2

p

1

W

z

W

z

g

Project

f

v

1

2

W

p

2

z

2

W

3

2

′

z

g

2

g

1

ϱ

2

2

W

Covariate Representations

f

t

= 0

2

= 0

t

W

1

f

y

1

W

If

z

3

W

z

y

W

W

However, HAN aggregates information from each view at the end of forward

= 0

t

However, HAN aggregates information from each view at the end of forward

1

Covariate Representations

W

z

If

If

p

W

p

3

3

2

z

3

W

p

z

2

z

2

3

[1,3,9,15,17,32,33,36] or hyper-graphs [16]. A few studies have considered

W

z

graphs

2

W

= 0

3

2

z

z

[1,3,9,15,17,32,33,36] or hyper-graphs [16]. A few studies have considered

t

= 1

z

1

g

g

propagation only once, while the proposed HINITE does aggregation layer-by-

= 1

If

= 0

t

1

Project

2

HIA Layer propagation only once, while the proposed HINITE does aggregation layer-by-

[1,3,9,15,17,32,33,36] or hyper-graphs [16]. A few studies have considered

t

If

= 1

Treated (

)

= 1

t

If propagation only once, while the proposed HINITE does aggregation layer-by-

[1,3,9,15,17,32,33,36] or hyper-graphs [16]. A few studies have considered

Estimating Treatment Effects Under Heterogeneous Interference

u

ϭϬ

f

W

y

Project

p

0

2

W

0

p

y

2

v

Estimating Treatment Effects Under Heterogeneous Interference

′

W

z

2

Co-Viewed Graph

W

2

z

W

v

2

′

Ϭ

Project

p

u

z

3

W

p

ϱ

z

W

3

z

2

3

f

ϭϬ

Project

2

g

W

p

3

1

y

ψ

2

3

2

Project

f

p

v

3

′

Project

W

y

p

1

HIA Layer

3

v

′

HIA Layer

3

W

2

p

3

v

3

HIA Layer

Fig. 1.

′

p

W

p

p

W

ψ

1

u

HIA Layer

′

v

1

p

3

g

1

g

graphs

ϭϱ

propagation only once, while the proposed HINITE does aggregation layer-by-

2

t

= 1

3

W

W

15

t

p

Control (

If

f

Treated (

v

1

2

graphs

z

heterogeneous graphs. For example, Qu et al. [20] assumed a partial interference

W

t

W

u

W

15

= 0

3

heterogeneous graphs. For example, Qu et al. [20] assumed a partial interference

If

t

3

heterogeneous graphs. For example, Qu et al. [20] assumed a partial interference heterogeneous graphs. For example, Qu et al. [20] assumed a partial interference

Estimating Treatment Effects Under Heterogeneous Interference

′

u

W

v

1

3

z

1

p

HIA Layer

g

W

φ

HIA Layer

1

1

2

W

z

t

Interference Representations

z

= 1

W

p

If

)

v

Ϯϱ

′

Ϯϱ

ϮϬ

W

p

′

layer, which is essential for capturing cross-view interference. In addition, we use

p

1

layer, which is essential for capturing cross-view interference. In addition, we use

= 1

If

W

W

z

p

t

layer, which is essential for capturing cross-view interference. In addition, we use

1

y

Interference Representations

f

W

p

layer, which is essential for capturing cross-view interference. In addition, we use

W

Control (

3

y

z

ϮϬ

3

1

2

2

1

1

′

0

1

W

ϯϬ

ϯϬ

and could only estimate ATE. Zhao et al. [46] proposed a method to construct

p

1

0

t

′

v

= 0

y

f

t

f

t

= 0

)

y

1

2

1

2

W

p

0

and could only estimate ATE. Zhao et al. [46] proposed a method to construct

1

W

W

p

1

y

z

p

W

f

y

y

W

f

1

2

p

ϭϱ

and could only estimate ATE. Zhao et al. [46] proposed a method to construct

If

If

g

= 1

1

and could only estimate ATE. Zhao et al. [46] proposed a method to construct

W

p

p

Project

v

3

1

W

If

f

t

p

15

2

W

p

0

W

= 0

W

p

v

′

W

p

1

f

3

LeakyRelu (for view-level attention) instead of the tanh function as an activation

Co-Purchased Graph

LeakyRelu (for view-level attention) instead of the tanh function as an activation

W

LeakyRelu (for view-level attention) instead of the tanh function as an activation

LeakyRelu (for view-level attention) instead of the tanh function as an activation

0

y

f

y

1

2

Estimating Treatment Effects Under Heterogeneous Interferenc

W

W

z

1

z

a heterogeneous graph from a homogeneous graph by learning a set of weights

p

0

g

1

f

3

= 0

t

If

)

1

Co-Purchased Graph

2

u

3

g

1

g

= 1

W

W

3

z

3

t

z

p

W

p

g

1

Fig. 1.

2

3

t

Treated (

= 1

p

Covariate Representations

Treated (

z

1

A figure caption is always placed below the illustration. Please note that short

A figure caption is always placed below the illustration. Please note that short

3

p

p

2

1

z

φ

W

W

g

Covariate Representations

1

1

W

1

g

)

g

)

1

Covariate Representations

g

1

z

W

p

1

u

3

p

2

1

2

Project

3

p

y

2

1

2

z

t

function to address the vanishing gradient issue, and we use single-head instead a heterogeneous graph from a homogeneous graph by learning a set of weights

= 0

t

captions are centered, while long ones are justified by the macro package automatically.

captions are centered, while long ones are justified by the macro package automatically.

W

W

2

2

W

2

g

z

p

Interference Representations

W

′

v

t

Control (

W

Interference Representations

Co-Viewed Graph

Co-Purchased Graph

Control (

Interference Representations

1

)

Covariates

W

= 0

)

Project

2

z

1

W

W

2

v

2

′

1

z

1

2

2

2

2

W

g

p

p

Covariate Representations

′

1

2

z

HIA Layer function to address the vanishing gradient issue, and we use single-head instead

Treated (

= 1

W

= 1

function to address the vanishing gradient issue, and we use single-head instead

Treated (

Co-Viewed Graph

v

′

p

a heterogeneous graph from a homogeneous graph by learning a set of weights

p

p

a heterogeneous graph from a homogeneous graph by learning a set of weights

W

2

3

′

W

function to address the vanishing gradient issue, and we use single-head instead

f

3

v

2

p

g

g

v

3

for each edge using an attention mechanism, but their method cannot capture

2

′

p

1

y

1

f

p

1

for each edge using an attention mechanism, but their method cannot capture

W

1

W

p

1

HIA Layer

z

W

v

2

3

= 1

f

Treated (

t

3

p

2

t

= 1

v

2

p

for each edge using an attention mechanism, but their method cannot capture

W

W

′

p

2

g

Interference Representations

W

2

′

Control (

t

p

′

HIA Layer

Heterogeneous Edges in

W

z

Covariates

1

p

3

g

g

v

p

3

3

W

W

z

p

3

3

2

Co-Purchased Graph

′

t

= 0

p

W

p

W

p

z

1

2

v

If

t

= 0

W

z

3

2

W

v

1

′

v

3

2

v

2

v

z

′

g

If

′

t

Heterogeneous Edges in

W

z

1

3

2

Heterogeneous Edges in

3

3

Project

W

z

p

z

W

v

1

1

3

1

Proof.

z

v

Co-Viewed Graph

W

Co-Viewed Graph

1

3

p

3

t

= 0

W

If

2

v

1

3

′

1

1

z

p

If

t

= 1

W

p

W

z

p

W

1

g

= 0

p

W

If

3

1

v

′

If

t

′

1

p

v

′

v

1

2

2

1

W

p

p

ψ

(View 1)

p

Heterogeneous Edges in

f

If

t

= 1

W

p

p

p

f

v

v

′

p

v

p

′

Project

If

t

= 1

W

p

1

v

′

1

Proof.

)

p

1

for each edge using an attention mechanism, but their method cannot capture

= 1

t

t

of multi-head attention for better efficiency.

3

W

Treated (

v

1

3

p

of multi-head attention for better efficiency.

W

f

1

= 1

t

Treated (

2

3

= 0

Treated (

)

= 1

t

)

p

t

y

Control (

Control (

= 0

v

1

1

v

′

′

t

= 0

)

p

ψ

t

1

v

′

2

1

2

y

W

f

0

p

y

)

of multi-head attention for better efficiency.

y

of multi-head attention for better efficiency.

p

p

W

p

′

′

p

= 1

W

W

u

v

2

1

3

)

′

= 0

)

v

p

1

)

0

3

p

interference between multi-view graph structures. We offer the first approach for

′

′

2

p

′

v

1

p

p

interference between multi-view graph structures. We offer the first approach for

p

W

Co-Purchased Graph

1

interference between multi-view graph structures. We offer the first approach for

)

z

1

2

v

interference between multi-view graph structures. We offer the first approach for

2

Control (

ψ

p

Co-Purchased Graph

1

W

φ

1

v

′

2

1

1

= 1

p

2

2

1

W

y

W

p

1

v

2

)

= 0

1

p

v

t

′

1

2

2

Control (

Co-Purchased Graph

Co-Purchased Graph

p

φ

3

3

p

v

1

g

′

1

1

3

p

y

W

3

f

1

p

v

handling interference on multi-view graphs.

v

′

handling interference on multi-view graphs.

1

p

Covariate Representations

2

z

HIA Layer

1

p

2

v

v

′

2

Conclusion

z

ψ

W

3

p

1

Proofs, examples, and remarks have the initial word in italics, while the

Proofs, examples, and remarks have the initial word in italics, while the

2

following text appears in normal font.

0

3

2

HIA Layer

Project

φ

W

following text appears in normal font.

′

2

f

v

f

v

′

1

(View 1)

0

HIA Layer

y

′

2

v

z

Project

2

W

y

W

p

2

p

p

3

2

W

0

W

W

f

2

W

z

3

z

v

Estimating Treatment Effects Under Heterogeneous Interference

t

Control (

1

v

2

handling interference on multi-view graphs.

2

v

2

v

Covariate Representations

v

′

1

1

Project

z

t

= 0

W

)

v

Co-Viewed Graph

′

′

′

1

t

= 1

Treated (

Control (

v

′

1

′

p

)

1

2

Treated (

2

1

2

t

= 1

Control (

Conclusion

)

)

2

Co-Purchased Graph

Co-Purchased Graph

Co-Viewed Graph

Covariates

Covariates

p

2

Covariate Representations handling interference on multi-view graphs.

p

2

Co-Viewed Graph

Co-Viewed Graph

2

W

2

W

2

p

p

2

p

p

3

g

′

p

p

′

Meanwhile, heterogeneous graphs have been the subject of recent graph anal-

z

v

Interference Representations

3

Covariate Representations

y

0

2

p

2

v

′

2

p

(View 2)

p

1

1

φ

p

2

p

3

′

2

p

z

′

W

2

z

y

1

2

f

W

2

1

z

2

W

W

W

p

3

2

f

y

v

p

1

p

y

HIA Layer

(View 2)

W

W

1

2

p

1

HIA Layer

3

u

′

p

)

Meanwhile, heterogeneous graphs have been the subject of recent graph anal-

1

3

1

′

2

Covariate Representations

v

W

Meanwhile, heterogeneous graphs have been the subject of recent graph anal-

p

2

3

v

v

3

′

p

W

p

1

z

2

6

Conclusion

6

Conclusion

2

ψ

Co-Viewed Graph

v

Co-Viewed Graph

t

= 0

t

= 0

)

′

v

6

v

′

1

2

Meanwhile, heterogeneous graphs have been the subject of recent graph anal-

Interference Representations

ψ

v

′

ψ

′

v

ψ

v

′

′

p

ysis studies, focusing on tasks such as node classification, link prediction, and

g

ysis studies, focusing on tasks such as node classification, link prediction, and

v

2

p

3

W

Interference Representations

Estimating Treatment Effects Under Heterogeneous Interference

2

Heterogeneous Edges in

p

1

2

2

2

3

ysis studies, focusing on tasks such as node classification, link prediction, and

z

W

p

3

1

y

′

f

p

3

v

′

1

W

p

3

v

′

2

v

W

1

p

p

1

)

W

p

1

v

Interference Representations

6

v

2

Heterogeneous Edges in

2

1

v

ysis studies, focusing on tasks such as node classification, link prediction, and

p

2

p

= 0

If

t

p

v

graph

Interference Representations

Covariates

Covariates

W

1

If

= 0

p

p

p

p

1

p

2

′

g

v

′

ψ

3

3

v

′

1

v

The proposed

1

t

W

Project

2

[4,10,14,27,28,38,39,44,45].

15

p

Co-Purchased Graph

2

′

1

3

t

If

1

= 0

p

3

2

φ

φ

Covariates

ψ

3

Covariate Representations

If

2

graph classification

[4,10,14,27,28,38,39,44,45].

HIA Layer

The proposed

HINITE shares

Co-Purchased Graph

Co-Viewed Graph

p

1

Treated (

t

3

v

Covariate Representations

= 1

Estimating Treatment Effects Under Heterogeneous Interference

Estimating Treatment Effects Under Heterogeneous Interference

2

graph

′

For citations of references, we prefer the use of square brackets and consecutive

For citations of references, we prefer the use of square brackets and consecutive

Cross-View Spillover

v

′

v

p

p

′

v

2

p

2

W

z

v

′

3

W

v

p

3

2

W

2

)

z

t

1

Cross-View Spillover

t

= 1

′

W

z

Treated (

= 1

v

Treated (

′

p

p

1

p

p

W

HINITE shares

v

HINITE shares

[4,10,14,27,28,38,39,44,45].

Covariate Representations

In this paper, we described the problem of heterogeneous interference and the

The proposed

Covariate Representations classification

= 1

[4,10,14,27,28,38,39,44,45].

In this paper, we described the problem of heterogeneous interference and the

z

z

′

W

φ

t

p

Covariate Representations

p

1

v

2

2

φ

φ

In this paper, we described the problem of heterogeneous interference and the

Co-Viewed Graph

φ

(View 1)

In this paper, we described the problem of heterogeneous interference and the

1

′

v

)

)

= 1

t

2

2

1

graph

Heterogeneous Edges in

(View 1)

2

Control (

classification

v

′

2

15

2

Heterogeneous Edges in

Interference Representations classification

t

= 1

some similarities with the heterogeneous graph attention network (HAN) [39].

Heterogeneous Edges in

t

= 0

1

Treated (

If numbers. Citations using labels or the author/year convention are also accept-

numbers. Citations using labels or the author/year convention are also accept-

1

3

1

v

p

some similarities with the heterogeneous graph attention network (HAN) [39].

HINITE shares

The proposed difficulty of treatment effect estimations under heterogeneous interference. This

f

f

v

v

some similarities with the heterogeneous graph attention network (HAN) [39].

3

W

2

g

Covariates

W

Covariates

p

2

ψ

Covariate Representations

3

3

Heterogeneous Edges in

y

difficulty of treatment effect estimations under heterogeneous interference. This

v

u

Covariate Representations

= 1

t

W

If

Interference Representations

3

Covariate Representations some similarities with the heterogeneous graph attention network (HAN) [39].

2

p

difficulty of treatment effect estimations under heterogeneous interference. This difficulty of treatment effect estimations under heterogeneous interference. This

2

′

1

z

3

′

2

Heterogeneous Edges in

If

t

W

If

Cross-View Interference

1

W

p

p

z

= 0

= 0

v

1

2

1

Covariates

If

However, HAN aggregates information from each view at the end of forward

′

Interference Representations

Interference Representations

y

0

0

f

′

able. The following bibliography provides a sample reference list with entries

If

v

2

1

t

= 0

t

However, HAN aggregates information from each view at the end of forward

However, HAN aggregates information from each view at the end of forward

If

t

Interference Representations

W

g

p

Covariates

Heterogeneous Edges in

Heterogeneous Edges in

y

Interference Representations

1

p

v

v

Covariates

Heterogeneous Edges in

3

1

0

However, HAN aggregates information from each view at the end of forward

1

W

1

1

Interference Representations

g

2

z

Heterogeneous Edges in

v

p

1

= 0

If

v

f

2

v

′

t

= 1

v

y

t

g

= 0

t

propagation only once, while the proposed HINITE does aggregation layer-by- propagation only once, while the proposed HINITE does aggregation layer-by-

2

W

W

3

p

3

If

)

p

Control (

Heterogeneous Edges in

Covariate Representations

3

Cross-View Interference

u

2

′

= 0

W

t

v

3

v

1

p

p

′

1

Co-Purchased Graph

Covariate Representations

1

Control (

t

p

Interference

z

= 1

W

1

W

z

t

= 0

)

Control (

t

= 0

)

′

If

t

= 1

v

′

p

3

v

z

′

3

1

W

2

v

p

′

t

p

2

v

′

p

2

ψ

u

p

3

v

W

p

t

= 0

1

v

3

= 1

1

Interference Representations

Heterogeneous Edges in

ψ

v

p

(View 2)

1

2

v

′

2

)

2

able. The following bibliography provides a sample reference list with entries

′

node-level

1

paper proposed HINITE to model the propagation of heterogeneous interference using

Heterogeneous Edges in aggregation,

v

paper proposed HINITE to model the propagation of heterogeneous interference

Heterogeneous Edges in

2

paper proposed HINITE to model the propagation of heterogeneous interference

= 0

(View 2)

1

2

2

If

Project

v

aggregation,

Heterogeneous Edges in

)

If

= 1

Cross-View Spillover

Heterogeneous Edges in

Heterogeneous Edges in

Heterogeneous Edges in

2

paper proposed HINITE to model the propagation of heterogeneous interference

v

(View 1)

If

t

p

v

= 0

If

t

propagation only once, while the proposed HINITE does aggregation layer-by- propagation only once, while the proposed HINITE does aggregation layer-by-

v

v

2

g

v

′

2

p

2

2

3

t

= 1

p

2

Computer

ψ

p

2

HIA layers that contain view-level

HIA layers that contain

HIA Layer using

Heterogeneous Edges in

If

Cross-View Spillover

f

y

Cross-View Interference

= 0

p

p

t

)

Cross-View Interference

(View 1)

node-level

= 1

0

W

2

t

p

W

1

v

= 0

1

v

z

2

3

v

W

aggregation,

1

′

p

aggregation,

Covariate Representations

If view-level

Interference

t

= 0

Heterogeneous Edges in

Heterogeneous Edges in

t

= 1

3

v

node-level aggregation,

view-level aggregation,

f

Treated (

t

= 1

t

If

= 1

v

(View 2)

y

layer, which is essential for capturing cross-view interference. In addition, we use layer, which is essential for capturing cross-view interference. In addition, we use

0

y

2

3

2

v

2

)

= 1

node-level

2

and attention mechanisms. We conducted extensive experiments to verify the

2

t

3

If

1

and attention mechanisms. We conducted extensive experiments to verify the and attention mechanisms. We conducted extensive experiments to verify the

(View 1)

(View 1)

layer, which is essential for capturing cross-view interference. In addition, we use

2

p

2

p

3

If

t

If

t

= 0

2

If

y

1

0

If

v

1

′

W

p

3

3

′

p

3

φ

g

φ

ψ

1

3

v

Covariate Representations

p

3

t

= 1

)

3

Node-Level Attention

p

3

Treated (

t

= 1

)

v

v

′

v

f

Cross-View Spillover

Cross-View Spillover

Treated (

1

1

y

1

Covariate Representations

p

′

v

1

0

′

′

2

g

p

Interference Representations

2

2

1

Covariate Representations

p

φ

′

v

t

= 1

If

t

= 1

1

p

2

f

′

Node-Level Attention

Covariate Representations

2

0

p

If

f

y

16

Interference Representations

Covariates

Interference Representations

p

y

f

f

place place

v

Project

1

)

Interference Representations

′

W

v

layer, which is essential for capturing cross-view interference. In addition, we use

y

f

= 0

y

)

0

f

f

aggregation, view-level

aggregation,

= 0

Interference Representations

)

W

p

g

y

and attention mechanisms. We conducted extensive experiments to verify the

If

If

LeakyRelu (for view-level attention) instead of the tanh function as an activation

1

p

0

y

(View 2)

0

y

f

t

= 0

v

Interference

(View 2)

If

t

LeakyRelu (for view-level attention) instead of the tanh function as an activation

0

y

y

f

0

1

t

z

W

1

2

2

v

′

(View 2)

1

v

f

)

t

1

LeakyRelu (for view-level attention) instead of the tanh function as an activation

Cross-View Interference

Cross-View Interference

Covariate Representations

= 0

)

Project

2

Covariates

Treated (

t

= 1

)

Please

Please

1

φ

Control (

t

Node-Level Aggregation

Covariate Representations

Control (

t

= 0

Acknowledgements

Acknowledgements

v

your

v

′

1

Control (

p

1

f

t

1

y

Project

LeakyRelu (for view-level attention) instead of the tanh function as an activation performance of the proposed HINITE, where the results validate the effectiveness

performance of the proposed HINITE, where the results validate the effectiveness performance of the proposed HINITE, where the results validate the effectiveness

performance of the proposed HINITE, where the results validate the effectiveness

p

(View 2)

v

′

W

f

p

1

3

2

Co-Purchased Graph

Co-Purchased Graph

1

f

f

= 1

f

y

y

(View 2)

z

W

Co-Purchased Graph

Cross-View Spillover

Cross-View Spillover

If

t

= 0

p

Heterogeneous Edges in function to address the vanishing gradient issue, and we use single-head instead

p

y

1

1

Cross-View Spillover the

of the HINITE in ITE estimation under heterogeneous interference.

acknowledgments at of the HINITE in ITE estimation under heterogeneous interference.

= 1

function to address the vanishing gradient issue, and we use single-head instead

y

HIA Layer

Treated (

Heterogeneous Edges in

If

f

2

′

your

3

Co-Viewed Graph

Treated (

Treated (

Spillover

y

f

2

Cross-View Spillover the

of the HINITE in ITE estimation under heterogeneous interference.

Covariate Representations function to address the vanishing gradient issue, and we use single-head instead

function to address the vanishing gradient issue, and we use single-head instead

2

t

= 1

Spillover

Computer

2

)

= 1

of the HINITE in ITE estimation under heterogeneous interference.

Cross-View Spillover

Cross-View Spillover

z

HIA Layer

W

= 1

)

Treated (

v

2

t

= 1

Control (

Co-Viewed Graph

Treated (

3

v

′

p

3

1

t

)

t

= 1

)

y

1

Co-Viewed Graph

)

Control (

t

Control (

)

Cross-View Interference

Cross-View Interference

Cross-View Interference end of

paper, preceded by an unnumbered run-in heading (i.e. 3rd-level heading).

paper, preceded by an unnumbered run-in heading (i.e. 3rd-level heading).

Lin et al.

y

0

f

1

y

0

t

Interference Representations

Co-Purchased Graph

′

HIA Layer

Heterogeneous Edges in

v

p

If

Interference Representations

Co-Purchased Graph

y

Node-Level Aggregation

1

View-Level Attention

p

Interference

= 0

Lin et al.

If

Control (

t

= 0

)

v

p

Covariates

Interference

t

= 0

1

v

Treated (

t

p

= 0

Cross-View Interference of multi-head attention for better efficiency.

Interference Representations

p

of multi-head attention for better efficiency.

t

= 1

If

t

)

= 1

2

v

′

1

Cross-View Interference

W

p

Cross-View Interference the

Node-Level Attention

Interference

Interference

Interference

2

Concat

)

)

= 0

W

z

1

of multi-head attention for better efficiency.

= 1

g

)

)

(

u

,

g

)

Control (

t

= 0

)

Treated (

t

Concat

(

u

,

Control (

t

= 0

)

Co-Purchased Graph

Interference Representations

Covariate Representations

Heterogeneous Edges in

1

v

Heterogeneous Edges in

In this paper, we described the problem of heterogeneous interference and the

v

difficulty of treatment effect estimations under heterogeneous interference. This

2

In this paper, we described the problem of heterogeneous interference and the

Interference Representations

Covariate Representations

Heterogeneous Edges in

Cross-View Spillover node-level

difficulty of treatment effect estimations under heterogeneous interference. This

v

Heterogeneous Edges in

Cross-View Interference

1

node-level aggregation,

Interference Representations

v

paper proposed HINITE to model the propagation of heterogeneous interference paper proposed HINITE to model the propagation of heterogeneous interference

2

and attention mechanisms. We conducted extensive experiments to verify the and attention mechanisms. We conducted extensive experiments to verify the

performance of the proposed HINITE, where the results validate the effectiveness acknowledgments at

(View 1)

Covariates

v

′

f

end of the

Node-Level Attention

ψ

of multi-head attention for better efficiency.

t

Interference

= 0

Computer

v

′

![Image](data:image/png;base64,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)

′

y

Node-Level Attention

Heterogeneous Edges in performance of the proposed HINITE, where the results validate the effectiveness

t

2

1

= 1

v

φ

′

2

1

2

Covariate Representations

1

′

φ

Covariate Representations

View-Level Aggregation

Heterogeneous Edges in

2

v

v

View-Level Attention

p

3

f

y

Cross-View Spillover

Cross-View Spillover

ψ

t

Cross-View Spillover

Spillover

Spillover

Spillover

Spillover

v

1

1

view-level aggregation,

aggregation, view-level

aggregation, performance of the proposed HINITE, where the results validate the effectiveness

Cross-View Spillover performance of the proposed HINITE, where the results validate the effectiveness

Interference

Spillover

(View 2)

y

t

1

Advertisement

Interference

Node-Level Attention

If

= 0

0

Co-Purchased Graph

Advertisement

Mouse 1

v

φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Mouse 2 Outcome Predictors Mouse 1 Mouse 2 Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 HSIC x 1 x 2 x 3 u 1 u 2 HSIC x 1 x 2 x 3 u 1 u 2 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ Multiply Treatment Assignment Acknowledgements This work was supported by JST, the establishment of university fellowships towards the creation of science technology innovation, Grant Number JPMJFS2123, and supported by JSPS KAKENHI Grant Number 20H04244. Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Outcome Predictors Mouse 1 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) φ Covariates Heterogeneous Edges in Heterogeneous Edges in (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in Heterogeneous Edges in (View 1) (View 2) Cross-View Spillover Cross-View Interference (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 HSIC x 1 x 2 x 3 u 1 u 2 HSIC x 1 x 2 x 3 u 1 u 2 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ Multiply Treatment Assignment Acknowledgements This work was supported by JST, the establishment of university fellowships towards the creation of science technology innovation, Grant Number JPMJFS2123, and supported by JSPS KAKENHI Grant Number 20H04244. Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Fig. 1: An example of the difference between interference on a homogeneous graph and heterogeneous graphs. An edge in a co-purchased graph represents the relationship that both items are bought together by many customers, while an edge in a co-viewed graph represents the relationship that both items are viewed on an e-commerce platform together by many customers. Edges in different views or graphs constitute multi-view or heterogeneous edges.

Mouse 2

Interference

Mouse 1

Advertisement

Advertisement

v

v

1

φ

Mouse 2

Spillover

Node-Level Attention

Node-Level Attention

View-Level Attention

φ

Mouse 2

Mouse 1

Node-Level Aggregation

View-Level Aggregation

View-Level Attention

Node-Level Attention

Computer

Mouse 1

Mouse 2

View-Level Aggregation

View-Level Attention

Outcome Predictors

Mouse 1

Node-Level Aggregation

Mouse 2

Mouse 2

View-Level Aggregation

View-Level Aggregation

Spillover

Outcome Predictors

Heterogeneous Edges in

1

Spillover

Heterogeneous Edges in

View-Level Aggregation

Node-Level Attention

Interference

Cross-View Spillover

(View 2)

Outcome Predictors

Node-Level Aggregation

v

2

Mouse 2

(View 1)

Mouse 1

View-Level Attention

Computer

Node-Level Aggregation

v

1

Cross-View Interference

Node-Level Attention

Computer

View-Level Attention

View-Level Aggregation

View-Level Aggregation

Covariates

Node-Level Aggregation

Mouse 1

Heterogeneous Edges in

v

1

Mouse 1

Outcome Predictors

View-Level Attention

Heterogeneous Edges in

View-Level Attention

Outcome Predictors

Outcome Predictors

Node-Level Attention

Computer

Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Mouse 2 Mouse 1 Mouse 2 View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Mouse 1 Mouse 2 Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 This study aims to estimate treatment effects from observational graph data, which contain records of covariates of units, relationships between units (i.e., graph structure), and treatment assignments with their outcomes. For example, data from an e-commerce platform typically include the logs of information regarding assignments of advertisements, sales of items, item profiles, and relationships between items, e.g., a co-purchased relationship.

Covariates

Outcome Predictors

View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Mouse 2 Mouse 1 Mouse 2 Mouse 1 Mouse 2 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Mouse 2 Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 2 Mouse 1 Mouse 2 Mouse 1 Mouse 2 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference effect of a treatment (e.g., advertisement) for a particular unit (e.g., product) is known as the individual treatment effect (ITE) [42], while that for a given group is known as the average treatment effect (ATE) [42].

As units are associated in these graphs, the outcome for a unit will be influenced by the treatments assigned to its neighboring units. This phenomenon is referred to as interference [17,21], an example of which is shown in Figure 1a. In a co-purchased graph, many customers buy the Mouse when they buy the Computer. In this case, advertising the Computer may also influence the sales of the Mouse, whose sales can no longer be independent of the advertisement, making it challenging to estimate the ITE accurately. Previous works have attempted to accurately estimate ITE given graph data by modeling interference, such as group-level interference [9,15,32], which is a partial interference and models interference within subgroups of units but ignores inter-group interference; pairwise interference [1,3,21,36], which considers interference from immediate neighbors only; and networked interference [17], which can model interference from distant neighbors. All these methods assume single-view interference, such that a graph is homogeneous and can only represent the same relationship among units, such as a co-purchased graph.

However, real-world graphs are rarely homogeneous, e.g., YouTube dataset [31], and Amazon dataset [8]. Therefore, we consider addressing interference on heterogeneous graphs that have multi-view edges, such as co-viewed and co-purchased item-to-item graphs of the Amazon dataset [8]. In this case, units are influenced by treatments of their heterogeneous neighbors via the multi-view

Spillover

Mouse 2 Co-Purchased Graph Co-Viewed Graph ψ Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph View-Level Aggregation Outcome Predictors Mouse 1 Node-Level Aggregation View-Level Attention View-Level Aggregation Co-Viewed Graph ψ Computer Node-Level Attention Cross-View Interference Interference Cross-View Interference Interference Cross-View Spillover Cross-View Interference φ Covariates Heterogeneous Edges in v 1 φ Covariates Heterogeneous Edges in Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Interference Interference Spillover of the HINITE in ITE estimation under heterogeneous interference. Concat ( u , g ) of the HINITE in ITE estimation under heterogeneous interference. Concat ( u , g ) φ Covariates Heterogeneous Edges in v p 2 p v ′ 2 3 16 Lin et al. (View 1) (View 2) Cross-View Spillover Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover (a) Interference on a homogeneous graph Mouse 2 Heterogeneous Edges in v 2 (View 1) Spillover Computer φ Covariates Heterogeneous Edges in v Cross-View Spillover Cross-View Interference Interference Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph Treated ( ) Control ( t = 0 ) Co-Purchased Graph Node-Level Aggregation View-Level Attention Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Node-Level Attention Node-Level Aggregation View-Level Attention Co-Viewed Graph ψ MJFS2123, and supported by JSPS KAKENHI Grant Number 20H04244. Computer Node-Level Attention Cross-View Interference Interference Cross-View Interference Interference Cross-View Spillover Cross-View Interference If t = 0 If t = 1 f Co-Purchased Graph Co-Viewed Graph ψ Covariate Representations Interference Representations Treated ( t = 1 ) Control ( t = 0 ) φ Covariates Heterogeneous Edges in v 1 φ Covariates Heterogeneous Edges in Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Interference Interference Spillover of the HINITE in ITE estimation under heterogeneous interference. Concat ( u , g ) of the HINITE in ITE estimation under heterogeneous interference. Concat ( u , g ) φ Covariates Heterogeneous Edges in v p 2 p v ′ 2 3 16 Lin et al. (View 1) (View 2) Cross-View Spillover Heterogeneous Edges in v 2 (View 1) (View 2) (b) Interference on heterogeneous graphs

v

1

v

2

v

1

v

2

y

Concat

Concat

(View 1)

0

(

u

,

(

u

g

,

)

g

)

0

f

1

edges, which is referred to as heterogeneous interference and often leads to crossview interference , an example of which is shown in Figure 1b. Although there is no direct edge between the Computer and the Mouse 2, the advertisement of the Computer still affects sales of the Mouse 2 via the edge between the Computer and the Mouse 1 in the co-purchased graph and the edge between the Mouse 1 and the Mouse 2 in the co-viewed graph. Without properly modeling the heterogeneous interference, the cross-view interference cannot be addressed, which will result in inaccurate ITE estimation.

The contributions of this study can be summarized as follows:

To overcome the difficulty caused by heterogeneous interference, we propose a novel method called I ndividual T reatment E ffects Estimator Under H eterogeneous In terference (HINITE; see Figure 2). The core idea of HINITE is to model the propagation of heterogeneous interference across units and views. To this end, inspired by Wang et al. [39], we design a heterogeneous information aggregation (HIA) layer, as shown in Figure 3. In the HIA layer, multiple single-layered graph neural networks (GNNs) [12] are used to capture information within the same views, and a view-level information aggregation mechanism is then used to combine information from different views. To properly model heterogeneous interference, the HIA layer also infers importances of different edges and views of heterogeneous graphs by applying attention mechanisms [34,35,39]. A single HIA layer can help units aggregate information from their 1-hop or direct neighbors across all views of heterogeneous graphs, enabling the HINITE to model the propagation of cross-view interference by stacking multiple HIA layers. Other components of the HINITE are explained in Section 3.

- -This study describes a new issue of interference on heterogeneous (multiview) graphs. Moreover, we formalize the problem of estimating ITE under heterogeneous interference.
- -Results of extensive experiments reveal that the proposed method outperforms existing methods for estimating ITE under heterogeneous interference while confirming the importance of modeling heterogeneous interference.
- -This study proposes a method to address interference on heterogeneous graphs with multi-view edges.

## 2 Problem setting

In this study, we aim to estimate ITE from observational heterogeneous graphs. Herein, we use x i ∈ R d to denote the covariates of a unit i (e.g., brand), t i ∈ { 0 , 1 } to denote the treatment assigned to a unit i (e.g., an advertisement), y i ∈ R to denote the observed outcome of a unit i (e.g., the observed sales of a unit i ), and non-bold, italicized, and capitalized letters (e.g., X i ) to denote random variables. Moreover, a unit with t = 1 is treated, and t = 0 is controlled.

Homogeneous graphs. Homogeneous graphs have only a single view of edges. We use an adjacency matrix A ∈ { 0 , 1 } n × n to represent the structure of a homogeneous graph, where n is the number of nodes (units). If there is an edge between units j and i , A ij = 1; otherwise, A ij = 0 . We let A ii = 0 .

Heterogeneous graphs. This study considers heterogeneous graphs 1 that have multiple views of edges [30], which are called heterogeneous or multi-view edges. We use the H = { A v } m v =1 to denote all the multi-view graph structures, where A v ∈ { 0 , 1 } n × n denotes the adjacency matrix of the v -th view, and m is the number of views. We use N v i to denote the set of neighboring units of the unit i in the v -th view, N i = { N v i } m v =1 to denote the set of neighbors of the unit i across all views. Here, the units in N i are heterogeneous neighbors of the unit i .

ITE estimation without interference. In traditional treatment effect estimation [24,42], non-graph data are given and it is assumed that there is no interference between units [24,42]. In this case, the potential outcomes y 1 i and y 0 i of a unit i are defined as the real value of outcome for a unit i with treatment value t = 1 and t = 0 , 2 respectively [42]. Additionally, the ITE is defined as τ i = E [ Y 1 i | X i = x i ] -E [ Y 0 i | X i = x i ] [42].

ITE estimation under heterogeneous interference. This study aims to estimate the ITE from observational heterogeneous graph data. The data can be denoted by ( X , T , Y , H ) , where X = { x i } n i =1 , T = { t i } n i =1 , and Y = { y i } n i =1 . We assume that there exists interference between units in heterogeneous graphs. In this case, the outcome of a unit is not only influenced by its own treatments and covariates but also influenced by those of its neighbors [17,21]. In heterogeneous graphs, every unit can receive interference from its heterogeneous neighbors through multi-view edges, so the interference in heterogeneous graphs is referred to as heterogeneous interference. Such heterogeneous interference contains two types of interference: same-view interference and cross-view interference. The former is that interference occurs within the same views, and the latter happens when interference propagates across different views through multi-view edges. To formalize the ITE under heterogeneous interference, we use s i to denote a summary vector of X -i and T -i on heterogeneous graphs H , where the subscript -i denotes all other units except i . The potential outcomes of the unit i in heterogeneous graphs, denoted by y 1 i ( s i ) and y 0 i ( s i ) , are real outcomes for the unit i under s i and treatment value t = 1 and t = 0 , respectively. Then, we define the ITE under heterogeneous interference as follows:

<!-- formula-not-decoded -->

Confounder. The existence of confounders is a well-known issue when estimating the ITE from observational data [26]. Confounders are parts of covariates, which can simultaneously affect the treatment assignment and outcome [42], resulting in an imbalance in the distributions of different treatment assignments. For instance, we consider that the treatment is whether a product is advertised. Famous brands have more promotion funds to advertise their products. Meanwhile,

1 Heterogeneous graphs can be classified into two types: those with multiple types of nodes and multiple types (views) of edges [30], and those with a single type of node and multiple types of edges [30]. In this study, we focus on the latter type.

2 Outcomes with 1 -t are called counterfactual outcomes [42].

customers tend to buy a product (e.g., a computer) from a famous brand (e.g., Apple). In this case, the brand is a confounder. Without accurately addressing confounders, ITE estimation will be biased.

Assumption 1. Following the previous studies [16,17], we assume that there exists an aggregation function that can aggregate information of other units on heterogeneous graphs while outputting a vector s , i.e., s i = AGG ( T -i , X -i , H ) . Here, we extend the neighbor interference assumption [3] to heterogeneous interference, for ∀ i , ∀ T -i , T ′ -i , ∀ X -i , X ′ -i , and ∀ H , H ′ : when s i = AGG( T -i , X -i , H ) = AGG( T ′ -i , X ′ -i , H ′ ) = s ′ i , Y t i ( S i = s i ) = Y t i ( S i = s ′ i ) holds.

Assumption 2. We extend consistency assumption [3] to heterogeneous interference setting. We assume Y i = Y t i i ( S i = s i ) on the heterogeneous graphs H for the unit i with t i and s i .

Assumption 3. To address confounders, we extend the unconfoundedness assumption [3,16] to the heterogeneous interference setting. For any unit i , given the covariates, the treatment assignment and output of the aggregation function are independent of potential outcomes, i.e., T i , S i ⊥ ⊥ Y 1 i ( s i ) , Y 0 i ( s i ) | X i .

Theoretical analysis. To model potential outcomes using observed data under heterogeneous interference, we prove the identifiability of the expected potential outcome Y t i ( s i ) ( t = 1 or t = 0 ) based on the above assumptions as follows:

<!-- formula-not-decoded -->

Based on the above proof, once we aggregate X -i and T -i on heterogeneous graphs H into s i , we can estimate the potential outcomes Y 1 i ( s i ) and Y 0 i ( s i ) . This enables us to estimate the ITE using Eq. (1).

## 3 Proposed Method: Individual Treatment Estimator Under Heterogeneous Interference

This study proposes HINITE, a method that can estimate the ITE from observed data ( X , T , Y , H ) under heterogeneous interference. Figure 2 shows the architecture of HINITE. As can be seen, HINITE consists of three components to address confounders, model heterogeneous interference, and predict outcomes, respectively. Specifically, the first component addresses confounders by learning balanced representations of covariates with the Hilbert-Schmidt Independence Criterion (HSIC) regularization [6]. The second component aggregates interference by modeling the propagation of interference across units and views, and generates representations of units, which are referred to as interference representations. The last component consists of two outcome predictors that infer potential outcomes using the covariate and interference representations.

of multi-head attention for better efficiency.

6

In this paper, we described the problem of heterogeneous interference and the

Ϭ

3

W

1

p

ϭϱ

ϭϬ

ϱ

Conclusion

6

Meanwhile, heterogeneous graphs have been the subject of recent graph anal-

Fig. 1.

3

W

z

A figure caption is always placed below the illustration. Please note that short

1

Project

p

W

W

W

W

W

W

z

′

Ϯϱ

ϯϬ

2

2

p

p

W

p

2

1

z

z

z

1

1

W

W

z

2

z

A figure caption is always placed below the illustration. Please note that short of multi-head attention for better efficiency.

Conclusion

Fig. 1.

3

3

3

2

ϮϬ

captions are centered, while long ones are justified by the macro package automatically.

ysis studies, focusing on tasks such as node classification, link prediction, and

Fig. 1.

In this paper, we described the problem of heterogeneous interference and the

In this paper, we described the problem of heterogeneous interference and the

6

Esti

3

p

captions are centered, while long ones are justified by the macro package automatically.

In this paper, we described the problem of heterogeneous interference and the classification

difficulty of treatment effect estimations under heterogeneous interference. This

W

W

some similarities with the heterogeneous graph attention network (HAN) [39].

W

W

However, HAN aggregates information from each view at the end of forward

A figure caption is always placed below the illustration. Please note that short

W

z

HINITE shares

The proposed

2

3

2

z

z

1

′

W

p

v

1

HIA Layer

W

v

1

W

W

p

p

1

v

v

W

W

2

1

3

′

z

′

W

p

p

graph

[4,10,14,27,28,38,39,44,45].

1

2

some similarities with the heterogeneous graph attention network (HAN) [39].

difficulty of treatment effect estimations under heterogeneous interference. This

In this paper, we described the problem of heterogeneous interference and the difficulty of treatment effect estimations under heterogeneous interference. This

difficulty of treatment effect estimations under heterogeneous interference. This paper proposed HINITE to model the propagation of heterogeneous interference

captions are centered, while long ones are justified by the macro package automatically.

Conclusion

Conclusion

v

1

p

W

p

1

z

3

p

p

p

′

2

z

z

1

3

p

2

1

In this paper, we described the problem of heterogeneous interference and the propagation only once, while the proposed HINITE does aggregation layer-by-

Conclusion

p

p

W

W

p

p

6

6

However, HAN aggregates information from each view at the end of forward difficulty of treatment effect estimations under heterogeneous interference. This

paper proposed HINITE to model the propagation of heterogeneous interference

1

p

paper proposed HINITE to model the propagation of heterogeneous interference

2

v

W

p

paper proposed HINITE to model the propagation of heterogeneous interference

1

′

′

layer, which is essential for capturing cross-view interference. In addition, we use

3

Proof.

Conclusion

1

v

2

difficulty of treatment effect estimations under heterogeneous interference. This

v

′

2

W

2

node-level

z

Proofs, examples, and remarks have the initial word in italics, while the

Proof.

2

p

1

p

v

2

v

Proofs, examples, and remarks have the initial word in italics, while the

HIA layers that contain

W

p

In this paper, we described the problem of heterogeneous interference and the

3

2

2

′

W

W

p

W

W

z

2

p

node-level

W

aggregation, node-level

aggregation,

W

W

Estimating Treatment Effects Under Heterogeneous Interference

Estimating Treatment Effects Under Heterogeneous Interference view-level

view-level following text appears in normal font.

1

2

3

propagation only once, while the proposed HINITE does aggregation layer-by-

p

Conclusion

In this paper, we described the problem of heterogeneous interference and the

In this paper, we described the problem of heterogeneous interference and the

15

paper proposed HINITE to model the propagation of heterogeneous interference and attention mechanisms. We conducted extensive experiments to verify the

3

using

v

1

′

following text appears in normal font.

1

u

p

v

3

p

2

′

2

′

v

aggregation, view-level

u

aggregation,

LeakyRelu (for view-level attention) instead of the tanh function as an activation

1

Proofs, examples, and remarks have the initial word in italics, while the aggregation,

1

3

u

p

p

3

3

p

′

3

3

v

′

v

p

1

1

HIA layers that contain

HIA layers that contain

v

′

using using

2

layer, which is essential for capturing cross-view interference. In addition, we use

Proof.

HIA layers that contain

6

6

paper proposed HINITE to model the propagation of heterogeneous interference

In this paper, we described the problem of heterogeneous interference and the difficulty of treatment effect estimations under heterogeneous interference. This

p

node-level

W

HIA layers that contain using

1

1

u

g

In this paper, we described the problem of heterogeneous interference and the

g

′

v

and attention mechanisms. We conducted extensive experiments to verify the

v

2

aggregation, aggregation,

view-level

g

′

1

′

and attention mechanisms. We conducted extensive experiments to verify the

2

In this paper, we described the problem of heterogeneous interference and the

Covariate Representations

For citations of references, we prefer the use of square brackets and consecutive

W

difficulty of treatment effect estimations under heterogeneous interference. This

3

Covariate Representations

p

1

p

2

1

difficulty of treatment effect estimations under heterogeneous interference. This

For citations of references, we prefer the use of square brackets and consecutive and attention mechanisms. We conducted extensive experiments to verify the

performance of the proposed HINITE, where the results validate the effectiveness

LeakyRelu (for view-level attention) instead of the tanh function as an activation

g

p

aggregation, function to address the vanishing gradient issue, and we use single-head instead

v

2

difficulty of treatment effect estimations under heterogeneous interference. This

HIA layers that contain using

following text appears in normal font.

v

3

numbers. Citations using labels or the author/year convention are also accept- aggregation,

v

′

v

1

g

p

2

p

node-level

p

1

3

2

aggregation,

′

2

1

1

Interference Representations

2

view-level

1

using

3

2

′

function to address the vanishing gradient issue, and we use single-head instead

1

′

v

1

′

v

p

W

aggregation, view-level

aggregation, node-level

2

g

Interference Representations

2

2

p

p

paper proposed HINITE to model the propagation of heterogeneous interference

p

If

t

2

g

g

′

2

W

g

2

v

2

′

numbers. Citations using labels or the author/year convention are also accept- performance of the proposed HINITE, where the results validate the effectiveness

difficulty of treatment effect estimations under heterogeneous interference. This using

3

t

If difficulty of treatment effect estimations under heterogeneous interference. This

paper proposed HINITE to model the propagation of heterogeneous interference of multi-head attention for better efficiency.

performance of the proposed HINITE, where the results validate the effectiveness

Project

= 0

= 0

g

able. The following bibliography provides a sample reference list with entries and attention mechanisms. We conducted extensive experiments to verify the

paper proposed HINITE to model the propagation of heterogeneous interference

Covariate Representations paper proposed HINITE to model the propagation of heterogeneous interference

of the HINITE in ITE estimation under heterogeneous interference.

performance of the proposed HINITE, where the results validate the effectiveness and attention mechanisms. We conducted extensive experiments to verify the

able. The following bibliography provides a sample reference list with entries

For citations of references, we prefer the use of square brackets and consecutive

v

3

aggregation,

If

p

v

3

v

2

p

aggregation,

u

,

g

)

paper proposed HINITE to model the propagation of heterogeneous interference

HIA layers that contain numbers. Citations using labels or the author/year convention are also accept-

Interference Representations of multi-head attention for better efficiency.

If

1

2

2

HIA Layer

1

of the HINITE in ITE estimation under heterogeneous interference.

node-level

′

′

If

′

′

3

view-level

p

2

2

3

t

t

p

p

g

g

z

= 1

= 1

p

Covariate Representations node-level

node-level node-level

aggregation,

f

Covariate Representations for journal articles [1], an LNCS chapter [2], a book [3], proceedings without

aggregation, aggregation,

for journal articles [1], an LNCS chapter [2], a book [3], proceedings without

t

= 0

HIA layers that contain

W

HIA layers that contain of the HINITE in ITE estimation under heterogeneous interference.

performance of the proposed HINITE, where the results validate the effectiveness

HIA layers that contain

(

2

3

v

v

y

f

p

p

2

′

view-level

2

′

2

0

view-level of the HINITE in ITE estimation under heterogeneous interference.

(

and attention mechanisms. We conducted extensive experiments to verify the performance of the proposed HINITE, where the results validate the effectiveness

using using

Concat

z

3

(

and attention mechanisms. We conducted extensive experiments to verify the aggregation,

1

view-level

Interference Representations

Advertisement and attention mechanisms. We conducted extensive experiments to verify the

of the HINITE in ITE estimation under heterogeneous interference.

paper proposed HINITE to model the propagation of heterogeneous interference

(

u

W

node-level

(

using

v

1

v

1

able. The following bibliography provides a sample reference list with entries

t

p

aggregation,

)

z

g

g

using

HIA layers that contain

If

p

view-level view-level

Project

Project

y

3

3

0

for journal articles [1], an LNCS chapter [2], a book [3], proceedings without editors [4], and a homepage [5]. Multiple citations are grouped [1-3], [1, 3-5].

2

aggregation,

p

p

f

Interference Representations

′

2

aggregation,

HIA Layer

v

aggregation,

HIA Layer

1

aggregation, of the HINITE in ITE estimation under heterogeneous interference.

and attention mechanisms. We conducted extensive experiments to verify the

6

Conclusion aggregation,

W

v

performance of the proposed HINITE, where the results validate the effectiveness

Covariate Representations

Covariate Representations

1

f

y

1

z

1

editors [4], and a homepage [5]. Multiple citations are grouped [1-3], [1, 3-5].

′

2

z

W

z

If

W

y

If

p

)

Concat

Project

= 1

1

t

W

2

′

′

v

3

2

z

= 0

W

place

1

Interference Representations

Interference Representations

Treated (

t

Treated (

= 1

t

= 1

)

t

3

z

W

Please your

u

,

g

)

Advertisement

,

u

6 and attention mechanisms. We conducted extensive experiments to verify the

Concat

u

,

g

)

Advertisement node-level

HIA layers that contain performance of the proposed HINITE, where the results validate the effectiveness

and attention mechanisms. We conducted extensive experiments to verify the performance of the proposed HINITE, where the results validate the effectiveness

f

= 1

Concat

W

using performance of the proposed HINITE, where the results validate the effectiveness

Advertisement

Advertisement

Concat

HSIC

,

)

Acknowledgements acknowledgments at

If

If

W

z

z

= 0

t

t

= 0

= 0

)

Control (

2

2

2

p

t

W

HIA Layer

f

p

of the HINITE in ITE estimation under heterogeneous interference.

(

of the HINITE in ITE estimation under heterogeneous interference.

HSIC

)

t

= 0

Co-Purchased Graph

W

)

0

performance of the proposed HINITE, where the results validate the effectiveness

W

′

1

y

1

Acknowledgements

g

Control (

y

1

acknowledgments at performance of the proposed HINITE, where the results validate the effectiveness

of the HINITE in ITE estimation under heterogeneous interference.

of the HINITE in ITE estimation under heterogeneous interference.

Concat

= 1

In this paper, we described the problem of heterogeneous interference and the

(

Covariate Representations acknowledgments at

paper, preceded by an unnumbered run-in heading (i.e. 3rd-level heading).

place

′

the end of

the

In this paper, we described the problem of heterogeneous interference and the

Advertisement

1

the end of

the

3

3

z

W

z

W

of the HINITE in ITE estimation under heterogeneous interference.

of the HINITE in ITE estimation under heterogeneous interference.

Please

p

W

If

Co-Purchased Graph

= 1

t

t

= 1

If

f

Interference Representations

2

v

p

2

′

2

)

p

1

f

1

1

p

p

1

W

W

y

,

u

Treated (

Concat

HSIC

HSIC

x

Concat

Advertisement

Concat

1

u

2

HSIC

x

1

f

f

paper, preceded by an unnumbered run-in heading (i.e. 3rd-level heading).

(

p

v

′

p

the

Confounder Addressing

v

= 1

p

end of

1

f

y

y

0

1

place your

the

Estimating Treatment Effects Under Heterogeneous Interference

Co-Viewed Graph

v

Please your

difficulty of treatment effect estimations under heterogeneous interference. This

t

0

t

Co-Viewed Graph

Treated (

0

y

y

v

2

ψ

f

f

2

W

= 1

)

Estimating Treatment Effects Under Heterogeneous Interference

Estimating Treatment Effects Under Heterogeneous Interference

(

x

u

u

,

,

g

g

)

)

paper proposed HINITE to model the propagation of heterogeneous interference

1

3

3

2

)

paper proposed HINITE to model the propagation of heterogeneous interference

Treated (

t

p

)

v

′

2

difficulty of treatment effect estimations under heterogeneous interference. This

t

t

= 0

x

HSIC

1

x

2

Concat

Concat

u

,

1

1

g

)

![Image](data:image/png;base64,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)

1

u

)

Control (

Advertisement

u

2

2

2

2

u

u

g

(View 2)

x

1

x

paper, preceded by an unnumbered run-in heading (i.e. 3rd-level heading).

Confounder Addressing

Interference Modeling

If

= 0

Control (

p

ψ

Covariate Representations

p

W

and attention mechanisms. We conducted extensive experiments to verify the

Covariate Representations view-level

15

aggregation, performance of the proposed HINITE, where the results validate the effectiveness

Estimating Treatment Effects Under Heterogeneous Interference

Editor,

Interference Representations

15

Estimating Treatment Effects Under Heterogeneous I

Editor,

(eds.) CONFERENCE 2016, LNCS, vol. 9999, pp. 1-13. Springer, Heidelberg (2016).

https://doi.org/10.10007/1234567890

1

g

g

(eds.) CONFERENCE 2016, LNCS, vol. 9999, pp. 1-13. Springer, Heidelberg (2016).

2

3

g

Cross-View Spillover https://doi.org/10.10007/1234567890

Node-Level Attention

2

z

W

W

u u x u u Project HIA Layer W z 1 Computer Node-Level Attention Node-Level Aggregation View-Level Attention (View 2) Cross-View Spillover Cross-View Interference Interference (View 2) Cross-View Spillover Cross-View Interference Interference Computer Node-Level Attention Node-Level Aggregation View-Level Attention f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Cross-View Interference Interference Spillover Computer Cross-View Interference Interference Spillover Computer Cross-View Spillover Cross-View Interference Interference Spillover g 3 Project HIA Layer W z 1 W z 2 W z 3 W z 1 W z 2 W z 3 W z 1 W z 2 W z 3 W p ψ φ Covariates Heterogeneous Edges in Heterogeneous Edges in (View 1) (View 2) Cross-View Spillover Cross-View Interference = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates φ Covariates Heterogeneous Edges in v 1 Co-Viewed Graph ψ φ x 3 u 1 u 2 x 1 x 2 x 3 u Fig. 2: An example of the model architecture of HINITE. In this case, there are two views, i.e., v 1 and v 2 .

3

z

W

p

2

2

HIA Layer

(View 2)

HIA Layer

HIA Layer

Treated (

HSIC

W

(View 1)

z

p

W

1

1

1

1

1

W

p

Node-Level Attention

(View 1)

u

W

W

Computer

W

2

z

W

1

1

1

2

2

2

2

p

p

p

Heterogeneous Edges in

Heterogeneous Edges in

2

Co-Viewed Graph

Covariate Representations

Interference Representations

t

= 0

Spillover

Co-Purchased Graph

v

2

F.,

S.

Heterogeneous Edges in

Heterogeneous Edges in

t

Co-Viewed Graph

Control (

ψ

)

Interference

Spillover

Interference

(View 2)

If

f

= 1

t

Cross-View Spillover

(View 1)

(View 1)

t

0

0

If

y

y

f

Spillover

View-Level Aggregation

Spillover

v

(View 1)

Outcome Predictors

Computer

Computer

Outcome Predictors

Co-Viewed Graph

View-Level Aggregation

Heterogeneous Edges in

2

Covariates

ψ

Co-Viewed Graph

Node-Level Attention

ψ

φ

Node-Level Attention

Node-Level Attention

Mouse 1

Node-Level Aggregation

v

φ

Heterogeneous Edges in

1

Node-Level Aggregation

Node-Level Aggregation

Heterogeneous Edges in

v

2

Mouse 1

Covariates

Mouse 2

View-Level Attention

Cross-View Spillover

View-Level Attention

View-Level Attention

Mouse 2

Heterogeneous Edges in

Covariates

Heterogeneous Edges in

Outcome Predictors

View-Level Aggregation

Spillover

v

v

2

2

v

Interference

1

v

1

Node-Level Attention

Computer

Mouse 1

View-Level Attention

Node-Level Aggregation

Mouse 2

f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph Mouse 2 Mouse 2 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates t = 1 Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Mouse 2 Specifically, we learn a balanced covariate representation u i for the x i using a map function ϕ that consists of multiple feed-forward (FF) layers, i.e., u i = ϕ ( x i ) , resulting in covariate representations for all units, denoted as U . We train ϕ by minimizing the HSIC between u and t , which is denoted as HSIC ϕ and designed as follows:

W p 1 W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 1 Mouse 2 Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 W z 3 W p W p 2 W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 W p 3 p v ′ 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 W p 3 p v ′ 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( ) W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) v 1 v 2 (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 2) Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 3.1 Learning Balanced Covariate Representations To address the imbalance in distributions of different treatment groups caused by confounders, HINITE learns balanced covariate representations using an existing approach [17]. The key idea is to find a representation space in which the treatment assignments and covariate representations become approximately independent [17]. This goal can be achieved by applying the HSIC regularization [6], which is an independence test criterion of two random variables. The value of HSIC is 0 when two random variables are independent. Thus, minimizing the HSIC can achieve the abovementioned goal.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover (View 2) Cross-View Spillover Cross-View Interference Interference Spillover (View 2) Cross-View Spillover Cross-View Interference Interference Spillover where N is the number of training units, · ⊤ represents the transposition operation, I N is the identity matrix, and 1 N is the vector of all ones. K and L represent the Gaussian kernel applied to U and T , respectively, i.e.,

Node-Level Attention

View-Level Attention

View-Level Attention

View-Level Attention

View-Level Aggregation

View-Level Aggregation

Computer

Outcome Predictors

Outcome Predictors

Outcome Predictors

Node-Level Attention

Node-Level Aggregation

View-Level Aggregation

## View-Level Attention View-Level Aggregation Node-Level Aggregation View-Level Attention Mouse 1 Mouse 2 Mouse 1 Mouse 2 Mouse 1 Mouse 2 3.2 Learning Heterogeneous Interference Representations

Outcome Predictors Mouse 1 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 To properly model heterogeneous interference, it is necessary to capture both same-view and cross-view interference. To this end, we model the propagation of the same-view and cross-view interference. Inspired by Wang et al. [39], we design an HIA layer, as shown in Figure 3, which contains node-level and view-level aggregation mechanisms. The node-level aggregation mechanism aggregates sameview interference received by units. It utilizes m single-layered GNNs [12,35] to perform aggregations within each view. The view-level aggregation mechanism

View-Level Aggregation

Outcome Predictors

= 1

15

15

15

15

f

y

0

Spillover

Computer

f

y

0

If

If

t

t

= 1

= 0

If

If

Interference

t

t

= 1

= 0

Interference

Cross-View Interference

Interference Representations

Spillover

Computer

Cross-View Spillover

Cross-View Interference

Covariate Representations

Interference Representations

p

3

p

2

v

p

2

v

′

2

(View 2)

(View 1)

Cross-View Spillover

Covariate Representations

p

3

2

p

1

v

v

′

1

p

p

2

v

v

′

′

1

φ

′

W

p

3

p

2

v

′

2

Heterogeneous Edges in

Covariates

1

φ

v

′

1

(View 1)

v

2

Heterogeneous Edges in

v

2

Heterogeneous Edges in

v

1

Heterogeneous Edges in

v

1

′

1

W

p

3

Covariates

2

(View 2)

ψ

W

p

2

ψ

W

p

2

u

3

g

1

g

2

3

u

u

g

1

g

3

g

1

g

2

2

g

g

3

g

3

3

1

= 1

If

t

= 0

= 0

If

= 1

t

)

= 0

y

1

t

f

Treated (

t

If

Treated (

t

If

f

t

= 1

= 0

)

Control (

t

t

= 1

If

t

= 1

f

Control (

If

)

t

= 1

If

t

t

y

f

Treated (

= 1

)

= 0

t

Control (

t

= 1

)

t

= 0

= 0

Control (

)

y

0

y

f

y

Co-Purchased Graph

0

f

f

Co-Viewed Graph

0

Co-Purchased Graph

y

1

y

1

Co-Purchased Graph

y

)

0

Control (

Co-Purchased Graph

y

t

= 1

t

= 0

t

= 0

)

Estimating Treatment Effects Under Heterogeneous Interference

Control (

Co-Viewed Graph

Co-Purchased Graph

)

Co-Purchased Graph

Estimating Treatment Effects Under Heterogeneous Interference

Estimating Treatment Effects Under Heterogeneous Interference

f

Treated (

Co-Viewed Graph

1

f

Estimating Treatment Effects Under Heterogeneous Interference

ψ

Co-Viewed Graph

t

Treated (

)

1

= 1

ψ

Treated (

ψ

Co-Viewed Graph

ψ

Estimating Treatment Effects Under Heterogeneous Interference

Co-Viewed Graph

ψ

15

φ

Estimating Treatment Effects Under Heterogeneous Interference

t

= 1

t

Control (

= 1

)

ψ

t

Control (

)

φ

t

Treated (

)

t

= 1

= 0

t

u

3

g

φ

Control (

φ

φ

= 0

= 0

t

Co-Purchased Graph

)

= 0

u

)

u

1

3

3

Control (

φ

)

Covariates

Covariates

Co-Purchased Graph

)

g

g

Co-Purchased Graph

Co-Purchased Graph

15

15

15

15

Estimating Treatment Effects Under Heterogeneous Interference

Covariates

g

1

2

1

Covariates

Co-Viewed Graph

Co-Viewed Graph

Heterogeneous Edges in

3

g

u

v

Co-Viewed Graph

g

v

3

1

Heterogeneous Edges in

Heterogeneous Edges in

Heterogeneous Edges in

Covariates

g

g

Project

u

Project

3

u

15

Estimating Treatment Effects

v

Covariates

1

15

Heterogeneous Edges in

15

Estimating Treatment Eff

ψ

Estimating Treatment Effects

2

u

Heterogeneous Edges in

Heterogeneous Edges in

v

3

3

u

Estimating Treatment Effects Under Heterogeneous Interference

3

g

16

HIA Layer

Project

W

W

g

g

W

W

W

W

p

1

3

W

W

W

p

W

p

1

Co-Viewed Graph

If

If

f

f

1

Control (

t

t

Treated (

g

g

Cross-View Spillover

1

Cross-View Spillover

15

g

1

Project

2

g

1

2

g

Co-Viewed Graph

g

Heterogeneous Edges in

Estimating Treatment Effects Under Heterogeneous Interference

Heterogeneous Edges in

ψ

Covariates

Heterogeneous Edges in

1

W

W

z

z

2

z

1

v

1

g

![Image](data:image/png;base64,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)

φ

HIA Layer

Covariates

Covariates

φ

HIA Layer

3

3

v

′

v

′

Heterogeneous Edges in

p

1

2

2

v

2

v

′

′

p

2

φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in Heterogeneous Edges in (View 1) (View 2) Cross-View Spillover f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in Heterogeneous Edges in (View 1) (View 2) Cross-View Spillover 15 15 15 15 Consider again the co-purchased and co-viewed graphs in Figure 1b. Suppose that we feed units and their co-purchased and co-viewed graphs to a network stacked by two HIA layers. For the Mouse 1, the first HIA layer performs two node-level aggregations. One aggregation helps the Mouse 1 aggregate interference within the co-purchased graph, while the other helps the Mouse 1 aggregate interference within the co-viewed graph, resulting in two aggregated results. Then, the view-level aggregation mechanism combines these results obtained by node-level aggregations to generate the Mouse 1's new representation, while updating the new representation in all views. This enables the Mouse 1 to aggregate interference from the Computer. Similarly, the first HIA layer also generates new representations for other units. Then, by taking these new representations of all units as inputs of the second HIA layer, the second HIA layer enables the Mouse 2 to capture interference from the Mouse 1, which contains interference from the Computer. Therefore, the cross-view interference from the Computer to the Mouse 2 can be captured by stacking two HIA layers.

View-Level Aggregation

Mouse 2 Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Cross-View Spillover Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention Cross-View Interference Interference Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention Apart from cross-view interference, another challenge is that the importance of edges and views may differ in heterogeneous graphs [39]. For example, in a co-viewed graph, the importance of products in the same category tends to be higher than that of products in different categories. Here, the weights of edges in the same view can be different. Furthermore, a co-purchased graph may have more significant importance than a co-viewed graph in terms of interference, leading to different importance for each view. To overcome these difficulties and properly model the propagation of interference, we infer different weights for every edge via a graph attention mechanism [35] (called node-level attention)

Outcome Predictors

Outcome Predictors

View-Level Aggregation

Outcome Predictors

Outcome Predictors

Mouse 1

Mouse 1

Mouse 2

Mouse 1

Mouse 2

Mouse 2

Mouse 1

Mouse 2

ψ

Heterogeneous Edges in

v

1

v

1

Estimating Treatment Effects Under Heterogeneous Interference

HIA Layer

W

1

Cross-View Spillover

Cross-View Spillover

Heterogeneous Edges in

Cross-View Spillover

v

Co-Viewed Graph

= 0

)

Control (

= 1

)

t

Treated (

t

Co-Purchased Graph

= 0

)

= 1

)

W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover Cross-View Interference W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 (View 1) (View 2) Cross-View Spillover W z 3 W p 1 W p 2 W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 W z 2 W z 3 W p 1 W p 2 W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 p 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ φ Covariates Heterogeneous Edges in v 1 Heterogeneous Edges in v 2 W z 3 W p 1 W p 2 W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ W z 3 W p 1 W p 2 W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph Co-Viewed Graph ψ p 1 p v ′ 1 2 v ′ p 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y W z W p W p W p p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 W z W z W p W p W p p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 p 1 1 p v ′ 1 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f This work was supported by JST, the establishment of university fellowships towards the creation of science technology innovation, Grant Number JPMJFS2123, and supported by JSPS KAKENHI Grant Number 20H04244. View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Spillover Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Computer Node-Level Attention Node-Level Aggregation View-Level Attention View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 Estimating Treatment Effects Under Heterogeneous Interference Estimating Treatment Effects Under Heterogeneous Interference u 3 g 1 g 2 g 3 Project HIA Layer W z 1 W z 2 W z 3 Estimating Treatment Effects Under Heterogeneous Interference t = 1 ) t = 0 ) Estimating Treatment Effects Under Heterogeneous Interference u 3 g 1 g 2 g 3 Project HIA Layer W z 1 W z 2 W z 3 W p 1 W p 2 W p 3 p v ′ 1 1 p v ′ 1 2 p v ′ 2 2 p v ′ 2 3 Covariate Representations Interference Representations If t = 0 If t = 1 f y 0 f y 1 Treated ( t = 1 ) Control ( t = 0 ) Co-Purchased Graph View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 View-Level Aggregation Outcome Predictors Mouse 1 Mouse 2 v 1 v 2 Cross-View Interference Node-Level Aggregation View-Level Aggregation Fig. 3: The architecture of the HIA layer. This layer consists of node-level and view-level aggregation mechanisms with their attention mechanisms. combines (i.e., sums up) the results aggregated by the node-level aggregations to generate new representations of units. Therefore, by employing an HIA layer, units are able to aggregate interference received from their one-hop heterogeneous neighbors. This enables capturing cross-view interference by stacking HIA layers. Similarly, same-view interference from multi-hop neighbors can also be captured by stacking HIA layers.

y

0

t

t

= 1

f

y

0

= 0

If

If

t

= 1

v

1

v

1

v

v

2

2

t

= 0

Interference Representations

Co-Purchased Graph

f

y

1

y

Co-Viewed Graph

g

g

2

v

v

Estimating Treatment Eff

1

2

Co-Viewed Graph

Covariate Representations

Interference Representations

HIA Layer

Lin et al.

(View 2)

2

1

15

1

before node-level aggregations, and learn different importance for every view via an attention mechanism [34,39] (called view-level attention) before view-level aggregations.

Now, we describe the architecture of the HIA layer in detail. First, the HIA layer infers the edge weight α v ij by the node-level attention mechanism as follows:

More specifically, given covariate representations U , treatment assignments T , and structures of heterogeneous graphs H , we aim to obtain interference representations G using a function ψ that consists of multiple HIA layers, i.e., G = ψ ( U , T , H ) . For a unit i , its interference representation g i is supposed to capture the interference from its heterogeneous neighbors. Let p be a representation of a unit, which is the input of the current HIA layer and the output of the previous HIA layer. For the first HIA layer, p is the concatenation of u and t . Let z denote a new representation for the unit i computed by the current HIA layer, α v ij denote the inferred weight of the edge between units j and i at the v -th view, w v i denote the learned importance of the v -th view for the unit i , and β v i denote the normalized value for w v i .

<!-- formula-not-decoded -->

where a and W represent a learnable parameter vector and matrix, respectively, and ∥ represents the concatenation operation. Next, it performs node-level aggregations. The node-level aggregation at the v -th view is computed as follows:

<!-- formula-not-decoded -->

where σ is an activation function, such as ReLU. Next, the view-attention mechanism is applied to learn the importance of different views as follows:

<!-- formula-not-decoded -->

where b is a bias vector, and q is a learnable parameter vector. Finally, the viewlevel aggregation is applied to aggregate the information from different views as follows:

<!-- formula-not-decoded -->

## 3.3 Outcome Predictions and ITE estimation

Given the covariate representations U , interference representations G , and treatment assignments T , we train two predictors that consist of multiple FF layers to infer the outcomes with different t . Specifically, let f y 0 and f y 1 denote the predictor for t = 0 and t = 1 , respectively. We optimize the two predictors by

minimizing the following mean square error (MSE) between prediction outcomes and observed outcomes with the HSIC regularization:

<!-- formula-not-decoded -->

where the γ is a regularization hyperparameter.

Finally, we can estimate the ITE using ˆ τ i = f y 1 ( u i , g i ) -f y 0 ( u i , g i ) .

## 4 Experiments

## 4.1 Datasets

We used three heterogeneous graph datasets: Amazon Software (AMZ S) [8], Youtube [31], and Flicker [40]. Following prior studies on ITE/ATE [17,16,26], we simulated outcomes 3 as the ground-truth values for counterfactual outcomes are not available.

Outcome Simulation: Similar to the outcome simulation in Ma et al. [16], we used available data and heterogeneous graph structures to simulate outcomes under heterogeneous interference of the unit i :

<!-- formula-not-decoded -->

where f 0 ( x i ) = w ⊤ 0 x i simulates the outcome of a unit i under treatment t i = 0 without interference, and every element of w 0 follows a Gaussian distribution or uniform distribution (i.e., N (0 , 1) or U (0 , 1) ). f t ( t i , x i ) = t i × w ⊤ 1 x i simulates the ITE of the unit i , where w 1 ∼ N (0 , I ) or U (0 , I ) . In the literature, the effect caused by interference is known as spillover effect [21]. We simulate it through f s ( T , X , N i ) = o (1) i + o (2) i , where o (1) i = Agg ( Concat ( X , T ) , N i ) represents a spillover effect from 1-hop heterogeneous neighbors for the unit i , o (2) i = Agg ( O ( 1 ) , N i ) represents the spillover effect of 2-hop heterogeneous neighbors, and O ( 1 ) represents the spillover effects from 1-hop heterogeneous neighbors for all units. Here, the aggregation function is defined as Agg ( C , N i ) = ∑ m v =1 e v ( 1 | N v i | ∑ j ∈ N v i w ⊤ ij c j ) , where e v and every element of w ij follow N (0 , 1) or U (0 , 1) . Lastly, ϵ i ∼ N (0 , 1) is a random noise.

Amazon Software dataset [8]: The Amazon dataset [8] is collected from Amazon 4 . In the graphs of the Amazon dataset, each node is a product. To study causal effects, we chose the co-purchased and co-viewed graphs from the software category of the Amazon dataset. After removing nodes with missing values, there are 11,089 items with 11,813 heterogeneous edges. The covariates consist of reviews and the number of customer reviews of items. We put reviews into the SimCSE [5] model to generate 768-dimensional sentence embeddings.

3 The simulated outcomes and the codes of the HINITE are available at https:// github.com/LINXF208/HINITE.

4 https://www.amazon.com/

The review rating of items is considered as a treatment: an item is treated ( t = 1 ) when the average review rating is at least 3, and an item is controlled ( t = 0 ) when the average review rating is less than 3. The causal problem in this dataset is whether review rating has a role in influencing the sales of items. Due to the heterogeneous edges among items, the sales of an item might be influenced by its heterogeneous neighbors' treatments.

YouTube dataset [31]: Tang et al. [31] used YouTube Data API 5 to crawl the information of contacts, subscriptions, and favorites of users from YouTube 6 , while extending them to a contact graph, co-subscription graph, co-subscribed graph, and favorite graph. Every node in the graphs is a user of YouTube. In this case, we consider a causal problem: 'how much recommendation of a video (treatment) to a user will affect the user's experience of this video (outcome)?' Moreover, users might share the recommended video with heterogeneous neighbors, which constitutes heterogeneous interference. We took 5,000 users with their heterogeneous graphs containing 3,190,622 heterogeneous edges to simulate outcomes and study heterogeneous interference. As detailed information about each user is missing, we simulated the covariates via x i ∼ N (0 , I ) (100-dimensional vector), and simulated treatment t i as follows, following most existing works, such as Ma et al. [16]:

<!-- formula-not-decoded -->

where Ber ( · ) represents a Bernoulli distribution, w t is a 100-dimensional vector in which every element follows U ( -1 , 1) , and ϵ t i is random Gaussian noise. Flicker dataset [40]: Flicker 7 is an online social website where users can share their images. Qu et al. [19] constructed a dataset with multi-view graphs, i.e., friendship view and similarity view, from the Flicker dataset [40]. Every node in the graphs is a user of Flicker. Following Qu et al. [19], we also consider friendship-view and similarity-view graphs that have 7,575 users with approximately 1,236,976 heterogeneous edges. Here, the causal question is: 'how much recommending a hot photo (treatment) to a user will affect the user's experience (outcome) of this photo?' In this case, users might share recommended photos with their heterogeneous neighbors, which constitutes heterogeneous interference. We used the 1206-dimensional embeddings that are provided by Guo et al. [7], generated using a list of users' interest tags, and simulated the treatments using Eq. (10).

## 4.2 Baselines

BNN [11]: Balancing Neural Network [11] (BNN) addresses confounders by minimizing the discrepancy of distributions of units belonging to different groups, without considering interference. Following Johansson et al. [11], we considered two structures: BNN-4-0 and BNN-2-2. The former has four representation layers

5 https://developers.google.com/youtube/?csw=1

7 https://www.flickr.com/

6 https://www.youtube.com/

but no prediction layers, and the latter has two representation layers and two prediction layers. Both have one linear output layer.

TARNet [26]: TARNet consists of the same model architecture as the CFR model but removes the balance term (MMD or Wasserstein distance).

CFR [26]: Counterfactual Regression (CFR) [26] minimizes the maximum mean discrepancy (MMD) and Wasserstein distance between different distributions of two groups. Similar to BNN, it also ignores interference. Following Shalit et al. [26], we considered two different schemes: CFR MMD and CFR Wass . The former minimizes the MMD of two different distributions, while the latter minimizes the Wasserstein distance.

GCN-based methods [17]: Ma et al. [17] proposed methods to address interference on a homogeneous graph using graph convolutional networks (GCNs) [41]. The GCN-based method can use only a single view rather than all views of heterogeneous graphs. To overcome it, we consider two schemes. The first scheme is to replace heterogeneous graphs with a projection graph A Proj and apply the GCN-based method to the A Proj , denoted as GCN Proj . If two units have an edge in either of the original heterogeneous graphs, there will be an edge in this projection graph. The second scheme is to augment the GCNbased method with mixing operations, which includes two variants: MGCN C and MGCN M . The MGCN C concatenates interference representations from different views into a single vector, while the MGCN M computes the mean vector of these interference representations.

## 4.3 Experiment Settings

For all datasets, we calculated ϵ PEHE / ϵ ATE to evaluate the error on ITE/ATE estimations as follows:

<!-- formula-not-decoded -->

Following Ma et al. [17], the entire X , T , and heterogeneous graph structures were given during the training, validation, and testing phases. However, only the observed outcomes of the units in the training set were provided during the training phase. We randomly split all datasets into training/validation/test splits with a ratio of 70%/15%/15%. Results on the Youtube and Flicker datasets were averaged over ten realizations, while the results on the AMZ S dataset were averaged over three repeated executions. We trained all models with the NVIDIA RTX A5000 GPU. All methods utilized the Adam optimizer with 2,000 training iterations for all datasets. In addition, dropout and early stopping were applied for all methods to avoid overfitting.

For all datasets, we set the learning rate to 0.001 with a weight decay of 0.001, set the training batch size to 512, and searched γ in the range of { 0 . 01 , 0 . 1 , 0 . 5 , 1 . 0 , 1 . 5 } using the validation sets. We used ReLU as activation function for ϕ , f y t i , and node-level aggregations. The hidden layers of ϕ were set to (128 , 64 , 64) -dimensions, ψ are set to (64 , 64 , 32) -dimensions, f y t i are set

Table 1: Results (mean ± standard errors) of performance of ITE and ATE estimation. Results in bold indicate the lowest mean error. HINITE is our method.

|          | Youtube        | Youtube         | Flicker      | Flicker     | AMZ S          | AMZ S          |
|----------|----------------|-----------------|--------------|-------------|----------------|----------------|
| Method   | ϵ PEHE         | ϵ ATE           | ϵ PEHE       | ϵ ATE       | √ ϵ PEHE       | ϵ ATE          |
| TARNet   | 40.75 ± 7.95   | 0.51 ± 0.23     | 24.20 ± 6.79 | 0.30 ± 0.26 | 112.37 ± 11.54 | 103.91 ± 12.78 |
| BNN-2-2  | 93.03 ± 16.02  | 0 . 26 ± 0 . 23 | 27.91 ± 7.53 | 0.13 ± 0.07 | 199.37 ± 0.20  | 196.36 ± 0.20  |
| BNN-4-0  | 105.38 ± 22.50 | 0.26 ± 0.23     | 29.22 ± 7.53 | 0.13 ± 0.07 | 206.03 ± 0.08  | 203.12 ± 0.08  |
| CFR MMD  | 42.02 ± 9.96   | 0.43 ± 0.36     | 24.44 ± 7.49 | 0.29 ± 0.17 | 103.18 ± 25.02 | 89.76 ± 32.13  |
| CFR WASS | 39.36 ± 8.76   | 0.51 ± 0.41     | 24.02 ± 6.71 | 0.35 ± 0.17 | 109.91 ± 24.49 | 99.40 ± 30.40  |
| GCN Proj | 42.37 ± 7.45   | 0.61 ± 0.39     | 24.59 ± 5.11 | 0.21 ± 0.13 | 139.14 ± 20.63 | 135.57 ± 22.86 |
| MGCN C   | 53.10 ± 11.83  | 0.29 ± 0.27     | 26.87 ± 6.43 | 0.25 ± 0.20 | 95.14 ± 8.25   | 72.08 ± 13.47  |
| MGCN M   | 53.99 ± 13.46  | 0.37 ± 0.33     | 29.48 ± 7.17 | 0.29 ± 0.25 | 87.33 ± 3.40   | 60.81 ± 3.27   |
| HINITE   | 14.43 ± 3.27   | 0.21 ± 0.20     | 18.45 ± 4.42 | 0.15 ± 0.11 | 76.16 ± 3.82   | 15.21 ± 3.89   |

to (128 , 64 , 32) -dimensions, and the dimensions of view-level attention were set to (128 , 128 , 64) -dimensions. Moreover, we searched for hyperparameters for all baseline methods from the search range suggested in the corresponding literature.

## 4.4 Results

Treatment effect estimation performance. Table 1 lists the results of ITE and ATE estimations on test sets of all datasets. It can be seen that the HINITE outperforms all baseline methods in ITE estimation, while there are significant gaps (p-values of the t-test are far less than 0.05) in ITE estimation between the proposed and baseline methods. It can also be seen that HINITE outperforms most baseline methods in ATE estimation, at least, achieving comparative performance of ATE estimation to those of the baseline methods. These results reveal that HINITE has a powerful ability to address heterogeneous interference. Moreover, the GCN Proj and MGCN with some simple mixers cannot always achieve better performance than other baseline methods. This implies that modeling cross-view interference using the HIA layers is important.

Ablation study. To further investigate the importance of each component of HINITE, we conducted ablation experiments. Let us start by introducing some variants of HINITE: (i) HINITE-PG applies the HINITE to the projection graph A Proj , which was described when introduced the GCN-based method. (ii) HINITE-NHG replaces the HIA layers with GCN layers [12] while using the A Proj . (iii) HINITE-NB removes the HSIC regularization by setting γ to 0.

Figure 4 presents the results of the ablation experiments. A clear performance gap can be seen in ITE and ATE estimation between the HINITE-PG/HINITENHG and HINITE. This implies that it is important to model the heterogeneous interference using the information of heterogeneous graphs and the proposed HIA layer. Comparing the results of HINITE and HINITE-NB, we can also observe

30-

22 -

20 г

28

20-a

26

18

1 24

$ 22

₴ 18

₴ 16-

20-

18

ЄРЕНЕ

$ 16-

16

14-

12-

14

12

18.0

18.0

HINITE-PG

0.2

0.2

HINITE-NHG

HINITE-NB

0.4

0.6

0.4

0.6

Y

Y

0.8

0.8

HINITE

1.0

1.0

0.6

0.6

1.0

0.5

0.5 -

0.8

90.6-

0.2 °

0.2

0.0

HINITE-PG

0.6.0

0.6.0

0.2

0.2

HINITE-NHG HINITE-NB

0.6

Y

Y

Fig. 4: Results (mean and standard error) of ablation experiments. We set α to 1.5 for HINITE-PG, HINITE-NHG, and HINITE in the ablation experiments.

![Image](data:image/png;base64,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)

Fig. 5: Performance changes on Filker and Youtube datasets with different γ (in the range of { 0 . 01 , 0 . 1 , 0 . 5 , 1 . 0 } ) . Results are averaged over ten realizations with a fixed value of γ .

![Image](data:image/png;base64,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)

0.4

0.4

0.6

0.8

0.8

HINITE

1.0

1.0

that removing the HSIC regularization results in performance degradation. This reveals that it is also important to balance the different distributions.

Sensitivity analysis. To investigate whether HINITE is sensitive to γ , we conducted experiments with different γ and present the results in Figure 5. No significant change in performance was observed with different values of γ . This reveals that HINITE is not particularly sensitive to the value of γ .

## 5 Related work

In the literature, efforts have been made to estimate treatment effect without interference [2,7,11,13,23,24,26,42,43] and with interference on homogeneous graphs [1,3,9,15,17,32,33,36] or hyper-graphs [16]. A few studies have considered heterogeneous graphs. For example, Qu et al. [20] assumed a partial interference and could only estimate ATE. Zhao et al. [46] proposed a method to construct a heterogeneous graph from a homogeneous graph by learning a set of weights for each edge using an attention mechanism, but their method cannot capture interference between multi-view graph structures. We offer the first approach for handling interference on multi-view graphs.

Meanwhile, heterogeneous graphs have been the subject of recent graph analysis studies, focusing on tasks such as node classification, link prediction, and graph classification [4,10,14,27,28,38,39,44,45]. The proposed HINITE shares some similarities with the heterogeneous graph attention network (HAN) [39]. However, HAN aggregates information from each view at the end of forward propagation only once, while the proposed HINITE does aggregation layer-bylayer, which is essential for capturing cross-view interference. In addition, we use LeakyReLU (for view-level attention) instead of the tanh function as an activation function to address the vanishing gradient issue, and we use single-head instead of multi-head attention for better efficiency.

## 6 Conclusion

In this paper, we described the problem of heterogeneous interference and the difficulty of treatment effect estimations under heterogeneous interference. This paper proposed HINITE to model the propagation of heterogeneous interference using HIA layers that contain node-level aggregation, view-level aggregation, and attention mechanisms. We conducted extensive experiments to verify the performance of the proposed HINITE, where the results validate the effectiveness of the HINITE in ITE and ATE estimation under heterogeneous interference.

## Acknowledgements

This work was supported by JST, the establishment of university fellowships towards the creation of science technology innovation, Grant Number JPMJFS2123, and supported by JSPS KAKENHI Grant Number 20H04244.

## Ethics

This study only involved public datasets that are freely available for academic purposes.

## References

1. Aronow, P.M., Samii, C.: Estimating average causal effects under general interference, with application to a social network experiment. The Annals of Applied Statistics 11 , 1912-1947 (2017)
3. Forastiere, L., Airoldi, E.M., Mealli, F.: Identification and estimation of treatment and interference effects in observational studies on networks. Journal of the American Statistical Association 116 (534), 901-918 (2021)
2. Chu, Z., Rathbun, S.L., Li, S.: Graph infomax adversarial learning for treatment effect estimation with networked observational data. In: Proceedings of the 27th ACMSIGKDD Conference on Knowledge Discovery and Data Mining. pp. 176-184 (2021)
4. Fu, X., Zhang, J., Meng, Z., King, I.: Magnn: Metapath aggregated graph neural network for heterogeneous graph embedding. In: Proceedings of the Web Conference 2020. pp. 2331-2341 (2020)
6. Gretton, A., Bousquet, O., Smola, A., Schölkopf, B.: Measuring statistical dependence with hilbert-schmidt norms. In: Proceedings of the 16th International Conference on Algorithmic Learning Theory. pp. 63-77 (2005)
5. Gao, T., Yao, X., Chen, D.: SimCSE: Simple contrastive learning of sentence embeddings. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (2021)
7. Guo, R., Li, J., Liu, H.: Learning individual causal effects from networked observational data. In: Proceedings of the 13th International Conference on Web Search and Data Mining. pp. 232-240 (2020)
9. Hudgens, M.G., Halloran, M.E.: Toward causal inference with interference. Journal of the American Statistical Association 103 (482), 832-842 (2008)
8. He, R., McAuley, J.: Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In: Proceedings of the 2016 World Wide Web Conference. pp. 507-517 (2016)
10. Jin, D., Huo, C., Liang, C., Yang, L.: Heterogeneous graph neural network via attribute completion. In: Proceedings of the Web Conference 2021. pp. 391-400 (2021)
12. Kipf, T.N., Welling, M.: Semi-supervised classification with graph convolutional networks. In: International Conference on Learning Representations (2017)
11. Johansson, F., Shalit, U., Sontag, D.: Learning representations for counterfactual inference. In: Proceedings of the 33rd International Conference on Machine Learning. vol. 48, pp. 3020-3029 (2016)
13. Li, Q., Wang, Z., Liu, S., Li, G., Xu, G.: Deep treatment-adaptive network for causal inference. The International Journal on Very Large Data Bases pp. 1-16 (2022)
14. Liang, X., Ma, Y., Cheng, G., Fan, C., Yang, Y., Liu, Z.: Meta-path-based heterogeneous graph neural networks in academic network. International Journal of Machine Learning and Cybernetics 13 (6), 1553-1569 (2022)

15. Liu, L., Hudgens, M.G.: Large sample randomization inference of causal effects in the presence of interference. Journal of the American Statistical Association 109 (505), 288-301 (2014)
17. Ma, Y., Tresp, V.: Causal inference under networked interference and intervention policy enhancement. In: Proceedings of the 24th International Conference on Artificial Intelligence and Statistics. vol. 130, pp. 3700-3708 (2021)
16. Ma, J., Wan, M., Yang, L., Li, J., Hecht, B., Teevan, J.: Learning causal effects on hypergraphs. In: Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. pp. 1202-1212 (2022)
18. Nabi, R., Pfeiffer, J., Charles, D., Kıcıman, E.: Causal inference in the presence of interference in sponsored search advertising. Frontiers in Big Data 5 (2022)
20. Qu, Z., Xiong, R., Liu, J., Imbens, G.: Efficient treatment effect estimation in observational studies under heterogeneous partial interference. arXiv preprint arXiv:2107.12420 (2021)
19. Qu, M., Tang, J., Shang, J., Ren, X., Zhang, M., Han, J.: An attention-based collaboration framework for multi-view network representation learning. In: Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. pp. 1767-1776 (2017)
21. Rakesh, V., Guo, R., Moraffah, R., Agarwal, N., Liu, H.: Linked causal variational autoencoder for inferring paired spillover effects. In: Proceedings of the 27th ACM International Conference on Information and Knowledge Management. pp. 16791682 (2018)
23. Rosenbaum, P.R., Rubin, D.B.: The central role of the propensity score in observational studies for causal effects. Biometrika 70 (1), 41-55 (1983)
22. Raudenbush, S.W., Schwartz, D.: Randomized experiments in education, with implications for multilevel causal inference. Annual Review of Statistics and Its Application 7 (1) (2020)
24. Rubin, D.B.: Randomization analysis of experimental data: The fisher randomization test comment. Journal of the American statistical association 75 (371), 591-593 (1980)
26. Shalit, U., Johansson, F.D., Sontag, D.: Estimating individual treatment effect: generalization bounds and algorithms. In: Proceedings of the 34th International Conference on Machine Learning. vol. 70, pp. 3076-3085 (2017)
25. Schnitzer, M.E.: Estimands and estimation of COVID-19 vaccine effectiveness under the test-negative design: Connections to causal inference. Epidemiology 33 (3), 325 (2022)
27. Shi, C., Ding, J., Cao, X., Hu, L., Wu, B., Li, X.: Entity set expansion in knowledge graph: a heterogeneous information network perspective. Frontiers of Computer Science 15 (1), 1-12 (2021)
29. Sun, W., Wang, P., Yin, D., Yang, J., Chang, Y.: Causal inference via sparse additive models with application to online advertising. In: Proceedings of the 29th AAAI Conference on Artificial Intelligence. p. 297-303 (2015)
28. Song, Y., Yang, X., Xu, C.: Self-supervised calorie-aware heterogeneous graph networks for food recommendation. ACM Transactions on Multimedia Computing, Communications, and Applications (2022)
30. Tang, L., Wang, X., Liu, H.: Uncovering groups via heterogeneous interaction analysis. In: Proceedings of IEEE International Conference on Data Mining. pp. 503512 (2009)
31. Tang, L., Wang, X., Liu, H.: Uncoverning groups via heterogeneous interaction analysis. In: 2009 Ninth IEEE International Conference on Data Mining. pp. 503512. IEEE (2009)

32. Tchetgen, E.J.T., VanderWeele, T.J.: On causal inference in the presence of interference. Statistical Methods in Medical Research 21 (1), 55-75 (2012)
34. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in Neural Information Processing Systems 30 (2017)
33. Tchetgen Tchetgen, E.J., Fulcher, I.R., Shpitser, I.: Auto-g-computation of causal effects on a network. Journal of the American Statistical Association 116 (534), 833-844 (2021)
35. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y.: Graph attention networks. In: Proceedings of the 6th International Conference on Learning Representations (2018)
37. Wang, P., Sun, W., Yin, D., Yang, J., Chang, Y.: Robust tree-based causal inference for complex ad effectiveness analysis. In: Proceedings of the 8th ACM International Conference on Web Search and Data Mining. pp. 67-76 (2015)
36. Viviano, D.: Policy targeting under network interference. arXiv preprint arXiv:1906.10258 (2019)
38. Wang, X., Bo, D., Shi, C., Fan, S., Ye, Y., Philip, S.Y.: A survey on heterogeneous graph embedding: methods, techniques, applications and sources. IEEE Transactions on Big Data (2022)
40. Wang, X., Tang, L., Liu, H., Wang, L.: Learning with multi-resolution overlapping communities. Knowledge and information systems 36 , 517-535 (2013)
39. Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., Yu, P.S.: Heterogeneous graph attention network. In: Proceedings of the 2019 World Wide Web Conference. pp. 2022-2032 (2019)
41. Welling, M., Kipf, T.N.: Semi-supervised classification with graph convolutional networks. In: Proceedings of the 4th International Conference on Learning Representations (2016)
43. Yao, L., Li, S., Li, Y., Huai, M., Gao, J., Zhang, A.: Representation learning for treatment effect estimation from observational data. In: Advances in Neural Information Processing Systems. vol. 31 (2018)
42. Yao, L., Chu, Z., Li, S., Li, Y., Gao, J., Zhang, A.: A survey on causal inference. ACM Transactions on Knowledge Discovery from Data 15 (5), 1-46 (2021)
44. Zhang, C., Song, D., Huang, C., Swami, A., Chawla, N.V.: Heterogeneous graph neural network. In: Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery &amp; data mining. pp. 793-803 (2019)
46. Zhao, Z., Kuang, K., Xiong, R., Wu, F.: Learning individual treatment effects under heterogeneous interference in networks. arXiv preprint arXiv:2210.14080 (2022)
45. Zhao, J., Wang, X., Shi, C., Hu, B., Song, G., Ye, Y.: Heterogeneous graph structure learning for graph neural networks. In: Proceedings of the 35th AAAI Conference on Artificial Intelligence. vol. 35, pp. 4697-4705 (2021)
