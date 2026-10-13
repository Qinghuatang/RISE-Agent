# COUPLING EXPERTS AND ROUTERS IN MIXTURE-OF-EXPERTS VIA AN AUXILIARY LOSS

Ang Lv<sup>1,2∗</sup>, Jin Ma<sup>2</sup>, Yiyuan Ma<sup>2†</sup>, Siyuan Qiao<sup>2</sup> <sup>1</sup>Renmin University of China, GSAI <sup>2</sup>Bytedance Seed 

## ABSTRACT

Mixture-of-Experts (MoE) models lack explicit constraints to ensure the router’s decisions align well with the experts’ capabilities, which ultimately limits model performance. To address this, we propose expert-router coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router’s decisions with expert capabilities. Our approach treats each expert’s router embedding as a proxy token for the tokens assigned to that expert, and feeds perturbed router embeddings through the experts to obtain intermediate activations. The ERC loss enforces two constraints on these activations: (1) Each expert must exhibit higher activation for its own proxy token than for the proxy tokens of any other expert. (2) Each proxy token must elicit stronger activation from its corresponding expert than from any other expert. These constraints jointly ensure that each router embedding faithfully represents its corresponding expert’s capability, while each expert specializes in processing the tokens actually routed to it. The ERC loss is computationally efficient, operating only on $n ^ { 2 }$ activations, where n is the number of experts. This represents a fixed cost independent of batch size, unlike prior coupling methods that scale with the number of tokens (often millions per batch). Through pre-training MoE-LLMs ranging from 3B to 15B parameters and extensive analysis on trillions of tokens, we demonstrate the effectiveness of the ERC loss. Moreover, the ERC loss offers flexible control and quantitative tracking of expert specialization levels during training, providing valuable insights into MoEs. 

## 1 INTRODUCTION

Mixture-of-Experts (MoE, Shazeer et al., 2017; Fedus et al., 2022; Lepikhin et al., 2021; Zoph et al., 2022) is a core architecture in modern large language models (LLMs). In MoE models, the feedforward layer is split into multiple small, specialized “experts.” A linear classifier, known as the “router,” selects which experts process each input token. By activating a few experts per token, MoE balances efficiency with scaled parameter counts, enabling the training of trillion-parameter models. 

Ideally, a router should possess an accurate representation of each expert’s capabilities to enable effective token routing. However, traditional MoEs offer no explicit constraints to guarantee this. Without direct access to expert parameters (and therefore their true capabilities), routers resort to trial-and-error learning of routing strategies, often resulting in misrouted tokens whose gradients interfere with expert specialization. While some methods (Lv et al., 2025; Pham et al., 2024) incorporated all experts’ activations for routing guidance, they incur substantial computational and memory costs due to denser activation. A lightweight and effective solution to better couple routing decisions with true expert capabilities remains an open challenge. 

We propose expert-router coupling loss (ERC loss), a novel auxiliary loss for MoE models that tightly couples routers and experts with negligible overhead. The loss is based on interpreting the router parameter matrix $\boldsymbol { R } \in \bar { \mathbb { R } } ^ { n \times d }$ as cluster centers, where each row $R [ i ]$ serves as the center for the token set $\mathcal { X } _ { i }$ routed to expert i. The ERC loss comprises three key steps: 

(1) Each R[i] is augmented with bounded random noise $\delta _ { i }$ to obtain $\tilde { R } [ i ]$ ], serving as a proxy for tokens in $\bar { \mathcal { X } } _ { i }$ . Here, $\delta _ { i }$ is bounded by half the minimum distance between adjacent cluster centers, ensuring that the noise simulates input variations within $\mathcal { X } _ { i }$ while preventing the crossing of cluster boundaries. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/69860293d913642d0685da66eaedf593530c13dfa52234854ef37b157efe9be8.jpg)



Figure 1: Three steps for computing the expert-router coupling loss.


(2) Inspired by prior works (Geva et al., 2021; Liu et al., 2023; Lv et al., 2025), the intermediate activation norm serves as an indicator of how well its capabilities align with the token. We measure the intermediate activation norms of all experts that take $\tilde { R } [ i ]$ as input. This step produces a matrix $M \in \mathbb { R } ^ { n \times n }$ , with $M [ i , j ]$ being the activation norm from expert j given input $\tilde { R } [ i ]$ 

(3) For all $i \neq j ,$ , the ERC loss imposes a penalty wherever the off-diagonal elements $M [ i , j ]$ or $M [ j , i ]$ exceed $\nu M [ i , i ]$ , where α is a scalar hyperparameter: 

$$
\mathcal {L} _ {\mathrm{ERC}} = \frac {1}{n ^ {2}} \sum_ {i = 1} ^ {n} \sum_ {j \neq i} ^ {n} \left(\max \left(\boldsymbol {M} [ i, j ] - \alpha \boldsymbol {M} [ i, i ], 0\right) + \max \left(\boldsymbol {M} [ j, i ] - \alpha \boldsymbol {M} [ i, i ], 0\right)\right).
$$

Minimizing it tightly couples experts and routers through two effects: 

• Expert specialization: The proxy token $\tilde { R } [ i ]$ elicits the strongest activation from expert i versus all other experts. This indicates that expert i is optimized to best match the features of its assigned token cluster $\mathcal { X } _ { i }$ 

• Precise token routing: Expert i is most activated by its designated vector $\tilde { R } [ i ]$ than to any other $\tilde { R } [ j ]$ for $j \neq i .$ . This demonstrates that $R [ i ]$ aligns well with the capabilities of expert $i ,$ ensuring that the router assigns to this expert the tokens that need it most. 

We conducted large-scale pre-training experiments on models from 3B to 15B parameters, using a total of several trillion tokens. The ERC loss not only significantly enhances model performance and narrows the performance gap with a competitive yet more computationally expensive MoE variant (Lv et al., 2025) but also retains the efficiency of vanilla MoEs. 

Furthermore, building on the first effect, we establish that the ERC loss serves as a powerful tool for studying expert specialization. This property arises from two key features of the ERC loss: (1) the specialization level is explicitly controlled by $\alpha ,$ , and (2) the bound of noise $\delta _ { i }$ provides a quantitative measure for this level. Through this lens, we reveal a trade-off between specialization and model performance. Our findings challenge some beliefs about expert specialization that were derived from small-scale experiments. These quantitative and qualitative analysis methods offer new pathways to advance the understanding of MoE models. 

## 2 BACKGROUND

Mixture-of-Experts Our description follows the prevailing SwiGLU structure used by advanced LLMs (Qwen, 2024; DeepSeek-AI, 2025; OpenAI et al., 2025). An MoE layer consists of n experts, where each expert i is parameterized by three matrices: $\boldsymbol { W } _ { g } ^ { i } \in \mathbb { R } ^ { d \times D } , \boldsymbol { \dot { W } } _ { p } ^ { i } \in \mathbb { R } ^ { d \times D }$ , and $\bar { \cal W } _ { o } ^ { i } \in$ $\mathbb { R } ^ { D \times d } .$ The layer also includes a router with the weight matrix $\ b { R } \in \mathbb { R } ^ { n \times d }$ , which takes a token $\pmb { x } \in \mathbb { R } ^ { d }$ as input and outputs an expert weight<sup>1</sup> vector: 

$$
\boldsymbol {w} = \operatorname{softmax} (\boldsymbol {x} \boldsymbol {R} ^ {\top}) \in \mathbb {R} ^ {n}.
$$

Typically, the top-K experts with the highest expert weights are selected to process the token. The processing of x by expert i is given by: 

$$
E _ {i} (\pmb {x}) = \left(\mathrm{SiLU} (\pmb {x} \pmb {W} _ {g} ^ {i}) \odot (\pmb {x} \pmb {W} _ {p} ^ {i})\right) \pmb {W} _ {o} ^ {i},
$$

where ⊙ denotes element-wise multiplication. The final output of the entire MoE layer is the weighted sum of the outputs of the selected experts: 

$$
\sum_ {k} ^ {K} \boldsymbol {w} [ k ] E _ {k} (\boldsymbol {x}), \text {   where   } k \in \operatorname{Top-K} (\boldsymbol {w}).
$$

Expert-router coupling via denser activation Autonomy-of-Experts (AoE; Lv et al., 2025) encodes the routing function into expert parameters. AoE factorizes $W _ { g }$ into two r-rank matrices $W _ { d o w n } ^ { i } \in \mathbb { R } ^ { d \times r }$ and $\boldsymbol { W _ { u p } ^ { i } } \in \mathbb { R } ^ { r \times D }$ . Each expert processes a token up to the point after the $\pmb { W } _ { d o w n } ^ { i }$ projection. The expert weight vector is computed using the activation norm at this stage: 

$$
\boldsymbol {w} = \operatorname{softmax} \left(\left\{\| \boldsymbol {x} \boldsymbol {W} _ {\text { down }} ^ {i} \| \text {   for   } i = 1, \dots , n \right\}\right).
$$

The top-K experts exhibiting the highest activation norms are selected to continue their forward computation, and the others are terminated early. This norm-based selection is justified by the fact that the activation norm of MLPs represents how well their capabilities match their inputs (Geva et al., 2021; Liu et al., 2023). The computational overhead of AoE scales with the number of tokens during both training and inference. Moreover, this inefficiency worsens as the number of experts n increases or the selection count K decreases. Wu et al. (2025) found that using only a small, representative subset of neurons per expert is sufficient for “autonomous expert selection,” reducing but not eliminating AoE’s token-dependent cost. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/dc49f192a84089a862eaebbaf221212f75c513ed3c29f9b31ef76894314c1bb3.jpg)



(a) Mixture-of-Experts


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/af26563580edf45b1ed349d101840737fdd45443da72482af25936b92114b93d.jpg)



(b) Autonomy-of-Experts



Figure 2: The overview of MoE and AoE models.


Pham et al. (2024) use experts’ final output norms to supervise router logits. There is no inference overhead but the model is fully dense-activated during training, contradicting the core sparsity principle of MoE. Therefore, we include it only for background discussion, not as a baseline. 

## 3 METHOD

After analyzing the strengths and limitations of prior work, we distill three design principles to ensure a lightweight, effective, and practically applicable enhancement for expert-router coupling in MoE-LLMs: 

(1) Routers must be retained in MoE architectures to preserve routing efficiency. 

(2) An auxiliary loss that enables interaction between experts and routers can strengthen their coupling. 

(3) The loss must have complexity independent of the number of input tokens and must not introduce activation density beyond that of a vanilla MoE. 

Below, we introduce expert-router coupling loss, which fulfills all these principles. 

## 3.1 EXPERT-ROUTER COUPLING LOSS

The expert-router coupling (ERC) loss is motivated by a clustering-based interpretation of MoE routing: The routing mechanism in traditional MoE models can be interpreted as a clustering process, where router parameters $\ b { R } \in \mathbb { R } ^ { n \times d }$ are viewed as $n$ cluster centers. For any input token $\pmb { x } \in \mathbb { R } ^ { d }$ the router computes an n-dimensional logit vector representing the weight assigned to each expert. Specifically, the weight for expert i is derived from the inner product between x and the cluster center $R [ i ]$ ]. When x belongs to the cluster centered at $R [ i ]$ , this inner product is maximized<sup>2</sup>, making expert i the top choice. 

A key advantage of this clustering view is that it enables probing an expert’s responsiveness to a set of tokens without feeding every token to all experts, unlike prior methods (See $\mathbf { \bar { \ S } } 2 )$ . Instead, we leverage each cluster center $R [ i ]$ as a proxy for tokens routed to expert i (denoted as $\textstyle { \mathcal { X } } _ { i } ) .$ , enabling us to derive intermediate activations and evaluate how well the expert aligns with a proxy token. 

Our ERC loss is computed in three key steps: 

(1) For each cluster center $R [ i ]$ , we create a perturbed proxy token $\tilde { \pmb { R } } [ i ] = \pmb { R } [ i ] \odot \delta _ { i } . \delta _ { i } \in \mathbb { R } ^ { d }$ is bounded multiplicative random noise, which we elaborate in §3.2. This noise ensures the proxy generalizes to tokens in $\mathcal { X } _ { i }$ . Notably, the perturbed $\tilde { R }$ is used only for loss computation; routing still uses the clean R to compute router logits, as in standard MoEs. 

(2) Each proxy token is processed by the $W _ { g }$ parameter of all n experts, yielding a total of $n ^ { 2 }$ intermediate activations. The $L ^ { 2 }$ norm of each activation is computed to form a matrix $M \in \mathbb { R } ^ { n \times n }$ where $M [ i , j ]$ corresponds to the norm from expert $j$ given input $\tilde { R } [ i ]$ 

$$
\boldsymbol {M} [ i, j ] = \left\| \tilde {\boldsymbol {R}} [ i ] \cdot \boldsymbol {W} _ {g} ^ {j} \right\|.
$$

(3) To enforce expert-router coupling, for all i and $j \neq i ,$ the ERC loss imposes two constraints, where a scalar $\alpha \in [ 0 , 1 ]$ determines their strength: 

$$
\boldsymbol {M} [ i, j ] <   \alpha \boldsymbol {M} [ i, i ],\tag{1}
$$

$$
\boldsymbol {M} [ j, i ] <   \alpha \boldsymbol {M} [ i, i ].\tag{2}
$$

Constraint 1 ensures the proxy token $\tilde { R } [ i ]$ activates its corresponding expert i more than any other expert $j .$ . Since tokens similar to $R [ i ]$ are routed to expert $i ,$ and given their similarity to $\tilde { R } [ i ]$ , they also elicit a stronger activation in expert i than in other experts. This strongest activation indicates that expert i is optimized to develop capabilities best suited to $\mathcal { X } _ { i }$ (Lv et al., 2025). 

Constraint 2 requires that expert i responds more strongly to its own proxy token $\tilde { R } [ i ]$ than by any other $\tilde { R } [ j ]$ . This ensures each $R [ i ]$ accurately represents expert i, guaranteeing that tokens most needing expert i are correctly routed to it. 

As α decreases, the two constraints become stricter, thereby enforcing stronger expert-router coupling. Additionally, α enables flexible regulation of specialization: a smaller α increases the gap between $M [ i , i ]$ and $M [ i , j ]$ , reflecting greater expert specialization as experts exhibit more differentiated responses to the same inputs. This feature makes the ERC loss a useful tool for investigating expert specialization and provides deeper insight into MoE behavior, as demonstrated in §4.2. 

We translate these two constraints into expert-router coupling loss, formally defined as: 

$$
\mathcal {L} _ {\mathrm{ERC}} = \frac {1}{n ^ {2}} \sum_ {i = 1} ^ {n} \sum_ {j \neq i} ^ {n} \left(\max \left(\boldsymbol {M} [ i, j ] - \alpha \boldsymbol {M} [ i, i ], 0\right) + \max \left(\boldsymbol {M} [ j, i ] - \alpha \boldsymbol {M} [ i, i ], 0\right)\right).\tag{3}
$$

The three steps for computing expert-router coupling loss are illustrated in Figure 1. For implementation details, we provide PyTorch-style pseudocode in Figure 8. 

## 3.2 BOUNDED RANDOM NOISE FOR GENERATING PROXY TOKENS

The perturbed proxy token $\tilde { \pmb { R } } [ i ] = \pmb { R } [ i ] \odot \pmb { \delta } _ { i }$ makes expert i’s coupling generalize effectively from $R [ i ]$ alone to $\mathcal { X } _ { i } .$ To ensure the perturbed point $\tilde { R } [ i ]$ remains within its original cluster, we require a bounded perturbation. We therefore model the noise $\delta _ { i }$ as a multivariate uniform distribution, $\delta _ { i } \sim \mathcal { U } ( 1 - \dot { \epsilon } _ { i } , 1 + \epsilon _ { i } ) ^ { d }$ . Let $\begin{array} { r } { j = \arg \operatorname* { m i n } _ { j ^ { * } \neq i } \| \pmb { R } [ i ] - \pmb { R } [ j ^ { * } ] \| } \end{array}$ be the nearest cluster center. For the noise level ϵ to be sufficient to avoid perturbing the cluster, it must satisfy: 

$$
\epsilon_ {i} \leq \frac {\| \boldsymbol {R} [ i ] - \boldsymbol {R} [ j ] \|}{2 \| \boldsymbol {R} [ i ] \|}.\tag{4}
$$

The derivation of this bound is provided in Appendix B. We set $\epsilon _ { i }$ to its maximum value, i.e., the right-hand side of this inequality. Notably, the value of $\epsilon _ { i }$ is dynamically computed at each layer and every training step. 

## 3.3 EFFICIENCY ANALYSIS

Training efficiency In a standard MoE layer, T tokens are processed by K experts, resulting in a total computational cost of 6TKDd FLOPs. expert-router coupling loss introduces only $2 n ^ { 2 } D d$ additional $\mathrm { F L O P s } .$ , a cost that is negligible in practical pre-training setups where K is often in the millions. In contrast, AoE introduces an additional overhead of $2 T \bar { ( \boldsymbol { n } - \boldsymbol { K } ) }$ )dr FLOPs (recall that r is AoE’s factorization rank; see §2). Given that typical MoE-LLMs operate at sparsity levels far below 25% $( \mathrm { i . e . , } n > 4 K )$ , this overhead ratio exceeds $r / D ,$ , making it prohibitive. A detailed breakdown of the FLOP calculations supporting the above theoretical analysis is provided in Appendix C.1. 

The efficiency of our method is confirmed in practice. The ERC loss maintains low overhead during LLM pre-training under multiple parallelism strategies, adding only 0.2–0.8% overhead in our experiments. We provide a complete analysis of these real-world distributed conditions and measured throughputs in Appendix C.2. 

Overhead-free inference Expert-router coupling loss introduces no additional inference overhead, since it is not applied at that stage. In contrast, AoE maintains the same forward computation, along with its associated overhead. 

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

We compare the ERC-loss-augmented MoE against both the vanilla MoE and AoE baselines. All models are trained from scratch with 3B parameters. This parameter size is chosen because it represents the largest scale at which we could successfully train the AoE model under our available resources. Our implementation is based on OLMoE (Muennighoff et al., 2025). The models comprise 12 layers with $d \ : = \ : 1 5 3 6$ and $D \ = \ 7 6 8$ Each Transformer (Vaswani et al., 2017) layer has 16 attention heads and n = 64 experts, where $K = 8$ experts are selected per token. For the AoE model, we set $r \ : = \ : 5 1 2$ to ensure consistent total parameter count. The number of activated parameters is 500M. Each model is trained on 500B tokens from the open-source dataset dolmap-v1.5-sample (Soldaini et al., 2024), using a batch size of 3 million tokens. We use the AdamW optimizer (Loshchilov & Hutter, 2019) with $( \bar { \beta } _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ , a weight decay of 0.1, and a learning rate of 4e-4 with a cosine schedule decaying to 4e-5. A load balancing loss (Fedus et al., 2022) with a weight of 0.01 is applied consistently in all experiments. 

For simplicity, the loss weight of the ERC loss is fixed at 1, and we use $\alpha = 1$ by default if not specified. 

We evaluate LLMs on the following tasks: ARC-Challenge (Clark et al., 2018), CommonsenseQA (Talmor et al., 2019), COPA (Roemmele et al., 2011), BoolQ (Clark et al., 2019), HellaSwag (Zellers et al., 2019), OpenbookQA (Mihaylov et al., 2018), SciQ (Welbl et al., 2017), Social IQa (Sap et al., 2019), WinoGrande (Sakaguchi et al., 2021), and MMLU (Hendrycks et al., 2021a). 

## 4.2 PERFORMANCE, EFFICIENCY, AND LOAD BALANCING

Figure 3(a) reports the average accuracy across all tasks and task-specific results are presented in Figure 9. It shows that the ERC-loss-augmented MoE achieves stable performance gains, which significantly outperforms the vanilla MoE and narrows the gap between AoE and MoE. 

In terms of efficiency, MoE models with and without ERC loss have nearly identical throughput and memory costs. By contrast, AoE requires 1.6× more training hours and 1.3× higher memory usage, limiting further scaling due to impractical training times and out-of-memory issues. 

Expert-router coupling loss is compatible with the load balancing loss. As shown in Figure 3(b), the difference in load balancing loss between MoE combined with $\mathcal { L } _ { \mathrm { E R C } }$ and the vanilla MoE is on the order of $1 0 ^ { - 5 }$ . This difference is negligible given that the overall load balancing loss magnitude remains around $1 0 ^ { - 2 }$ . By comparison, the loss difference between AoE and vanilla MoE is approximately $4 \times 1 0 ^ { - 4 }$ . Although this difference is still small, it is notably larger than the difference exhibited by ours. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/1f9515a2e5f5f1b74fc7b4f5dae3617b8a065a20d9c51452c138706c069e4fb2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/8cfc18b88482bc2d4cedd0f7ff12e723a5981369e0cb2aa3279d1dd7e7194949.jpg)



Figure 3: The 3B-scale MoE with ERC loss achieves substantial, stable gains while maintaining effective load balancing. Figure 9 shows task-specific details.


## 4.3 VALIDATING ERC LOSS IN 15B-PARAMETER MOES

We scale models to 15 billion parameters by increasing n to 256 (keeping K=8) and doubling the model depth. This configuration results in a total of 15B parameters with approximately 700M activated. Other training hyper-parameters largely follow the setup in Section 4.1. As a large-scale, high-sparsity model, the AoE method failed to train due to being overly costly and is thus omitted from comparison. 

Table 1 shows that the benefits of the ERC loss persist across various public benchmarks more challenging than those used for 3B models, including MMLU (Hendrycks et al., 2021a), C-Eval (Huang et al., 2023), MMLU-Pro (Wang et al., 2024b), AGI-Eval (Zhong et al., 2024), BBH (Suzgun et al., 2023), MATH (Hendrycks et al., 2021b), GSM8K (Cobbe et al., 2021), and TriviaQA (Joshi et al., 2017). The consistent performance improvements demonstrate that our method effectively addresses the expert-router decoupling problem even at scale. Throughout this large-scale training, we observed no loss spikes or abnormal gradients. 


Table 1: Scaling to 15B parameters: ERC loss improves performance on challenging benchmarks.


<table><tr><td></td><td>MMLU</td><td>C-Eval</td><td>MMLU-Pro</td><td>AGI-Eval</td><td>BBH</td><td>MATH</td><td>GSM8K</td><td>TriviaQA</td></tr><tr><td>MoE</td><td>63.2</td><td>67.5</td><td>31.0</td><td>42.0</td><td>44.3</td><td>25.7</td><td>45.2</td><td>47.2</td></tr><tr><td><eq>MoE + \mathcal{L}_{ERC}</eq></td><td>64.6</td><td>69.0</td><td>31.9</td><td>44.2</td><td>45.6</td><td>26.1</td><td>45.8</td><td>49.1</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/0c2a7e83e91c91cd11eb1c987d15320f009d56f4aed7b8da444e9c5ff01d40cd.jpg)



Figure 4: t-SNE projections of $W _ { g }$ in MoE experts trained without and with the ERC loss. Our ERC loss provides greater expert specialization.


## 4.4 THE ERC LOSS IS AN EFFECTIVE TOOL FOR EXPLORING EXPERT SPECIALIZATION

With the ERC loss, experts are more specialized, as they exhibit greater discrimination between tokens they process and those they do not, compared to vanilla MoE (without the ERC loss). An intuitive demonstration of this specialization comes from visualizing expert parameters. Following (Yang et al., 2025), we use t-SNE (van der Maaten & Hinton, 2008) to project each row of ${ \cal { W } } _ { a } ^ { \bar { i } }$ (where i mod $8 = 0 )$ from layer 6 (the middle depth) onto a 2D point. As shown in Figure 4, experts in vanilla MoE lack specialization, as their parameter features do not form meaningful clusters. By contrast, MoE enhanced with the ERC loss exhibits more distinct clusters, indicating specialized capabilities. 

Beyond merely promoting specialization, the ERC loss can also serve as a powerful tool for exploring it. We show this capability through two features below. 

Feature 1: α enables a controllable investigation into optimal specialization In the ERC loss, α governs the coupling strength between experts and the router. When $\alpha = 0$ , the ERC loss encourages $R [ i ]$ to be orthogonal to the parameters of other experts, thereby maximizing specialization. Conversely, when $\alpha  1$ 1, the loss permits smaller differences in how all experts’ responsiveness to $R [ i ]$ , thus reducing specialization. Notably, $\alpha = 1$ only weakens the ERC loss’s constraints to their maximum extent; it still retains a degree of specialization stronger than the spontaneously emerged specialization in a vanilla MoE model. 

Feature 2: ϵ provides a quantitative measure for specialization The noise level ϵ exhibits a strong correlation with $\alpha ,$ , and it can reflect changes in expert specialization throughout the training process. This correlation exists because as α increases, experts are allowed to be more homogeneous. This growing homogeneity among experts, in turn, reduces the separation between the cluster centers in the router as they are tightly coupled. A smaller separation between cluster centers ultimately derives a smaller ϵ. Thus, ϵ is a quantitative metric tracking expert specialization. 

Experiments The following experiments support these two features. In Figure 5(a), we plot ϵ at each training step across a parameter search over $\alpha \in \{ 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0 \}$ . Consistent with our analysis, increasing α, which reduces expert specialization, indeed leads to a corresponding decrease in ϵ. Note that measuring router cluster distance is uninformative in vanilla MoE training without the ERC loss, because the router and experts are uncoupled and cluster distances do not reflect expert capability dynamics. We further compared downstream task performance across different values of α. Figure 5(b) shows that all tested α values outperform the vanilla MoE model. This not only confirms the robust effectiveness of the ERC loss but also demonstrates that the specialization spontaneously formed by vanilla MoE models is inadequate. 

The optimal specialization degree Figure 5(b) shows that pursuing extreme specialization is not advisable, as model performance degrades with overly strict α. This highlights a trade-off between promoting expert specialization and maintaining effective collaboration, which is under-discussed in previous work. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/c353b998e1ce296df4e46e56be2ddf4cf021b8e5591b5698424df85a852f3bc5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/f5f2f2e5f5c67b9343b32f9c6a76858e274cf11e1e6d804c2024a09a74a349b2.jpg)



Figure 5: (a) Since routers are deeply coupled with experts, the distance between neighboring cluster centers $( \mathrm { i . e . }$ , the maximum noise level ϵ) quantitatively reflects changes in expert specialization during training, which is controlled by α. (b) Downstream performance across different values of α.


The optimal specialization degree is influenced by several factors. The core consideration is whether, among all $\binom { n } { K }$ possible expert combinations, an effective K-expert set can be assembled for any given input. In general, smaller values of n favor more generalist experts, while larger n can support a higher degree of specialization. However, we currently lack quantitative metrics to characterize $\mathrm { \tilde { \Omega } } ^ { \mathrm { { s } q } } \mathrm { { s } } ^ { \mathrm { { } } , \mathrm { { } } }$ or “small” n and K across different models; as a result, determining the optimal trade-off remains largely empirical. For example, in our experiments with a fixed $K = 8 \mathrm { : }$ : When $n = 6 4$ the optimal $\alpha = 1$ , suggesting $n = 6 4$ is not $\mathrm { \tilde { \Omega } } ^ { \mathrm { 6 6 } } | \mathrm { a r g e } ^ { , , }$ for our 3B-parameter models. In contrast, with $n = 2 5 6$ , we searched for an optimal $\alpha = 0 . 5$ , indicating $n ^ { ^ { \prime } } = 2 5 6 \mathrm { i s } ^ { \cdots } \mathrm { l a r g e } ^ { , , }$ for our 15Bparameter models. This trade-off is also shaped by other architectural choices, such as the use of shared experts<sup>3</sup>. A deeper investigation into these interacting factors, reliable quantitative metrics for specialization, and an automated evaluation of the optimal specialization degree for a given model are left as important problems for future works. For practitioners implementing the ERC loss, we recommend starting with $\alpha = 1$ , which eliminates expert decoupling and should provide some gains. Further improvement may be achieved by experimenting with lower α values, depending on the specific configuration of your model. 

Several studies (Guo et al., 2026; Liu et al., 2024; Hendawy et al., 2024) have promoted specialization via expert output orthogonality. We argue, however, that orthogonalizing expert outputs does not equate to achieving extreme specialization, as the magnitude (norm) of an expert’s response to a token remains unconstrained. Moreover, finding a set of orthogonalized high-dimensional vectors is not difficult, making it unclear whether such orthogonality yields sufficiently discriminative representations. Consequently, one should not interpret these fine-tuning experiments as supporting a broad claim that “more specialization is always better.” On a separate note, orthogonality among router embeddings (Baidu-ERNIE-Team, 2025) is only weakly correlated with specialization, since the router and experts are typically decoupled. As demonstrated in ablation studies, enforcing router orthogonality might not be a critical factor for pre-training MoE models. 

## 4.5 ABLATION STUDIES

Choice of activations for computing M We considered five candidates for calculating M: using the norms of (a) $\tilde { R } W _ { g } ,$ (b) $\tilde { R } W _ { p } ,$ (c) $\mathrm { S i L U } ( \tilde { R } W _ { g } )$ , (d) the post-SwiGLU activations $( \mathrm { i . e . }$ $\mathrm { S i L U } ( \tilde { R } W _ { g } ) \odot \tilde { R } W _ { p } )$ , and (e) experts’ final outputs (i.e., $( \mathrm { S i L U } ( \tilde { R } W _ { g } ) \odot \tilde { R } W _ { p } ) W _ { o } )$ . As shown in Figure 6(a), $\tilde { R } W _ { g }$ is the most effective among all alternatives. While using the final output achieves comparable performance, it incurs a higher cost. We therefore adopt $\tilde { R } W _ { g }$ as our default choice. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/46641f9c32146583142d30a538a4d05e3c0706fc51c0b6d505ca7ba2a38ef28f.jpg)



Figure 6: Results of ablation studies. For detailed task-specific results, please refer to Figure 9.


Random noise δ enables the generalization of coupling The random noise δ allows $\tilde { R } [ i ]$ to better capture the samples within $\mathcal { X } _ { i }$ . To validate its importance, we conducted an ablation study where we trained an MoE with the ERC loss but removed δ. Specifically, we computed M directly using the original R instead of the noise-augmented ${ \tilde { R } } .$ As shown in Figure 6(b), removing δ greatly degrades performance. This is because the coupling between routers and experts becomes overfitted to R, failing to generalize to the real inputs that $\bar { \mathbf { \Pi } } \bar { \mathbf { \Pi } } ^ { \top } \bar { \mathbf { \Xi } }$ s represent. 

Comparison with contrastive regularization solely on routers The router orthogonalization loss (Baidu-ERNIE-Team, 2025) requires $\hat { R }$ (the row-wise normalization of $\mathbf { \delta } _ { R } )$ to satisfy: 

$$
\hat {\boldsymbol {R}} \hat {\boldsymbol {R}} ^ {\top} = \boldsymbol {I}.
$$

As shown in Figure $6 ( \mathrm { c } )$ , the orthogonalization loss yields only limited gains. We attribute this to our finding that the router embeddings in our baseline MoE model are already nearly orthogonal, with an average absolute cosine similarity of 0.15. This value corresponds to angles between router embeddings mostly ranging from arccos $( 0 . 1 5 ) = 8 1 ^ { \circ }$ to arccos(- $. 0 . { \overset { \cdot } { 1 5 } } ) = 9 9 ^ { \circ }$ . Notably, we do not imply that all MoEs always have nearly orthogonal router embeddings, as this may depend on the data or specific architecture; we report this only as a characteristic of our models, which explains the limited gains from the orthogonalization loss. 

This result further demonstrates that weak coupling between routers and experts is a more critical issue than imperfect orthogonality in router embeddings. The significant gains from ERC, even when applied to a baseline with already near-orthogonal routers, provide clear evidence. 

Furthermore, it is important to note that even if both routers and experts are orthogonalized, there is no guarantee that each $R [ i ]$ will be aligned with $W _ { g } ^ { i }$ . Therefore, the ERC loss cannot be reduced to contrastive techniques applied individually to routers or experts, such as orthogonalization loss. 

Additional analyses Appendix A provides analyses addressing several frequently asked questions, including the effect of $\alpha > 1$ , and verifying that the model decreases the ERC loss by learning meaningful coupling rather than by manipulating parameter norms. 

## 5 RELATED WORKS

Auxiliary loss for MoEs Auxiliary losses are crucial for training large-scale MoE models. Most existing work in this area focuses primarily on enhancing training stability. For instance, many studies have proposed auxiliary losses to address load balancing challenges (Fedus et al., 2022; Qiu et al., 2025; Wang et al., 2024a); Zoph et al. (2022) introduced the z-loss, which penalizes excessively large logits in the gating network to enable stable training. MoE concepts have also inspired mixtures of attention heads or entire layers (Gong et al., 2024; Lin et al., 2024), where auxiliary losses play a critical role in effective optimization. Our ERC loss is the first tailored to strengthen the expert-router coupling. Other related auxiliary losses enhancing expert specialization or orthogonality are discussed below. 

Expert specialization Dai et al. (2024) introduced a shared expert to handle general capabilities, encouraging the others to be more specialized. Guo et al. (2026) proposes an auxiliary loss to minimize the pairwise projections of the selected top-K experts’ outputs for each token, reducing expert overlap but incurring high cost due to $K ^ { 2 }$ cosine similarity calculations per token. Other methods scale the number of tiny experts to millions, making each expert more atomic and thus more specialized (Yang et al., 2025; Park et al., 2025; He, 2024), but are memory-bounded. Beyond efficiency, these methods face three major limitations: (1) no quantitative control over specialization degree; (2) no exploration of the specialized-generalized ability trade-off; and (3) failure to strengthen expertrouter coupling. Our method addresses all three, both efficiently and effectively. 

Some works (Guo et al., 2026; Liu et al., 2024; Hendawy et al., 2024) maximize specialization by training orthogonal experts, but their evaluations are based on fine-tuning (or reinforcement learning) experiments. We contend that orthogonalizing expert outputs is not equivalent to achieving extreme specialization, and further, that the optimal degree of specialization is a complex problem affected by various factors and requires further exploration. 

Contrastive learning Constraints 1 and 2 bear similarity to contrastive learning (Chen et al., 2020; van den Oord et al., 2019; Khosla et al., 2020). Some MoE research (Luo et al., 2024; Guo et al., 2026) applied contrastive learning to expert outputs, encouraging specialization. However, naively applying contrastive learning to either routers or experts leaves the weak expert-router coupling unaddressed. 

## 6 CONCLUSIONS

The weak coupling between router decisions and expert capabilities limits MoE models in multiple important aspects. We propose expert-router coupling loss that tightly couples router parameters with their corresponding experts. The proposed ERC loss improves MoE-based LLMs on downstream tasks while incurring negligible training overhead. In addition, it exhibits several desirable properties that not only provide deeper insight into the behavior of MoE models but also offer a promising tool for future research on expert specialization. 

## STATEMENTS ON ETHICS, REPRODUCIBILITY, AND LLM USAGE

Our research does not raise ethical issues. For reproducibility, we used public data and code, and provide algorithm code in Figure 8. We used LLMs solely for typo checking. 

## ACKNOWLEDGMENTS

We thank Songhao Wu and Ziteng Wang for their insightful discussions. We are grateful to Ruobing Xie, Yining Qian, and Kaiyi Zhang for proofreading and writing suggestions. Additionally, we sincerely acknowledge the anonymous ICLR reviewers for their constructive comments and questions, which have greatly improved this work. 

## REFERENCES



Baidu-ERNIE-Team. Ernie 4.5 technical report, 2025. 





Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations, 2020. URL https://arxiv.org/abs/ 2002.05709. 





Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. BoolQ: Exploring the surprising difficulty of natural yes/no questions. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2924–2936, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1300. URL https://aclanthology.org/N19-1300/. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018. URL https://arxiv.org/abs/1803.05457. 





Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168. 





Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2401. 06066. 





DeepSeek-AI. Deepseek-v3 technical report, 2025. URL https://arxiv.org/abs/2412. 19437. 





William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23(120):1–39, 2022. URL http://jmlr.org/papers/v23/21-0998.html. 





Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main.446. 





Zhuocheng Gong, Ang Lv, Jian Guan, Wei Wu, Huishuai Zhang, Minlie Huang, Dongyan Zhao, and Rui Yan. Mixture-of-modules: Reinventing transformers as dynamic assemblies of modules. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 20924–20938, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main. 1164. URL https://aclanthology.org/2024.emnlp-main.1164/. 





Hongcan Guo, Haolang Lu, Guoshun Nan, Bolun Chu, Jialin Zhuang, Yuan Yang, Wenhao Che, Xinye Cao, Sicong Leng, Qimei Cui, and Xudong Jiang. Advancing expert specialization for better moe, 2026. URL https://arxiv.org/abs/2505.22323. 





Xu Owen He. Mixture of a million experts, 2024. URL https://arxiv.org/abs/2407. 04153. 





Ahmed Hendawy, Jan Peters, and Carlo D’Eramo. Multi-task reinforcement learning with mixture of orthogonal experts. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=aZH1dM3GOX. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021a. 





Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021b. 





Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-eval: A multilevel multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems, 2023. 





Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan (eds.), Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL https://aclanthology.org/ P17-1147/. 





Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 18661–18673. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/ file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf. 





Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. {GS}hard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=qrwe7XHTmYb. 





Hongzhan Lin, Ang Lv, Yuhan Chen, Chen Zhu, Yang Song, Hengshu Zhu, and Rui Yan. Mixture of in-context experts enhance LLMs’ long context awareness. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/ forum?id=RcPHbofiCN. 





Boan Liu, Liang Ding, Li Shen, Keqin Peng, Yu Cao, Dazhao Cheng, and Dacheng Tao. Diversifying the mixture-of-experts representation for language models with orthogonal optimizer, 2024. URL https://arxiv.org/abs/2310.09762. 





Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, and Beidi Chen. Deja vu: contextual sparsity for´ efficient llms at inference time. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023. 





Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7. 





Tongxu Luo, Jiahe Lei, Fangyu Lei, Weihao Liu, Shizhu He, Jun Zhao, and Kang Liu. Moelora: Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models, 2024. URL https://arxiv.org/abs/2402.12851. 





Ang Lv, Ruobing Xie, Yining Qian, Songhao Wu, Xingwu Sun, Zhanhui Kang, Di Wang, and Rui Yan. Autonomy-of-experts models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=8BIDrYWCeg. 





Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018. URL https://arxiv. org/abs/1809.02789. 





Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. Olmoe: Open mixture-of-experts language models, 2025. URL https://arxiv.org/abs/2409. 02060. 





OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, Che Chang, Kai Chen, Mark Chen, Enoch Cheung, Aidan Clark, Dan Cook, Marat Dukhan, Casey Dvorak, Kevin Fives, Vlad Fomenko, Timur Garipov, Kristian Georgiev, Mia Glaese, Tarun Gogineni, Adam Goucher, Lukas Gross, Katia Gil Guzman, John Hallman, Jackie Hehir, Johannes Heidecke, Alec Helyar, Haitang Hu, Romain Huet, Jacob Huh, Saachi Jain, Zach Johnson, Chris Koch, Irina Kofman, Dominik Kundel, Jason Kwon, Volodymyr Kyrylov, Elaine Ya Le, Guillaume Leclerc, James Park Lennon, Scott Lessans, Mario Lezcano-Casado, Yuanzhi Li, Zhuohan Li, Ji Lin, Jordan Liss, Lily, Liu, Jiancheng Liu, Kevin Lu, Chris Lu, Zoran Martinovic, Lindsay McCallum, Josh McGrath, Scott McKinney, Aidan McLaughlin, Song Mei, Steve Mostovoy, Tong Mu, Gideon Myles, Alexander Neitz, Alex Nichol, Jakub Pachocki, Alex Paino, Dana Palmie, Ashley Pantuliano, Giambattista Parascandolo, Jongsoo Park, Leher Pathak, Carolina Paz, Ludovic Peran, Dmitry Pimenov, Michelle Pokrass, Elizabeth Proehl, Huida Qiu, Gaby Raila, Filippo Raso, Hongyu Ren, Kimmy Richardson, David Robinson, Bob Rotsted, Hadi Salman, Suvansh Sanjeev, Max Schwarzer, D. Sculley, Harshit Sikchi, Kendal Simon, Karan Singhal, Yang Song, Dane Stuckey, Zhiqing Sun, Philippe Tillet, Sam Toizer, Foivos Tsimpourlas, Nikhil Vyas, Eric Wallace, Xin Wang, Miles Wang, Olivia Watkins, Kevin Weil, Amy Wendling, Kevin Whinnery, Cedric Whitney, Hannah Wong, Lin Yang, Yu Yang, Michihiro Yasunaga, Kristen Ying, Wojciech Zaremba, Wenting Zhan, Cyril Zhang, Brian Zhang, Eddie Zhang, and Shengjia Zhao. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925. 





Jungwoo Park, Ahn Young Jin, Kee-Eung Kim, and Jaewoo Kang. Monet: Mixture of monosemantic experts for transformers. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=1Ogw1SHY3p. 





Quang Pham, Giang Do, Huy Nguyen, TrungTin Nguyen, Chenghao Liu, Mina Sartipi, Binh T. Nguyen, Savitha Ramasamy, Xiaoli Li, Steven Hoi, and Nhat Ho. Competesmoe – effective training of sparse mixture of experts via competition, 2024. 





Zihan Qiu, Zeyu Huang, Bo Zheng, Kaiyue Wen, Zekun Wang, Rui Men, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5005–5018, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176- 251-0. doi: 10.18653/v1/2025.acl-long.249. URL https://aclanthology.org/2025. acl-long.249/. 





Qwen. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024. 





Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S. Gordon. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In 2011 AAAI Spring Symposium Series, 2011. 





Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106, 2021. 





Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social IQa: Commonsense reasoning about social interactions. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4463–4473, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1454. URL https://aclanthology.org/ D19-1454. 





Noam Shazeer, *Azalia Mirhoseini, *Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer. In International Conference on Learning Representations, 2017. URL https: //openreview.net/forum?id=B1ckMDqlg. 





Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. arXiv preprint, 2024. 





Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,¨ Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023, pp. 13003–13051, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.824. URL https://aclanthology.org/2023. findings-acl.824/. 





Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. URL https://aclanthology.org/ N19-1421/. 





Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019. URL https://arxiv.org/abs/1807.03748. 





Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of Machine Learning Research, 9(86):2579–2605, 2008. URL http://jmlr.org/papers/v9/ vandermaaten08a.html. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf. 





Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts, 2024a. URL https://arxiv.org/abs/2408. 15664. 





Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-pro: A more robust and challenging multitask language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024b. URL https://openreview. net/forum?id=y10DM6R2r3. 





Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin (eds.), Proceedings of the 3rd Workshop on Noisy User-generated Text, pp. 94–106, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4413. URL https://aclanthology.org/W17-4413/. 





Songhao Wu, Ang Lv, Ruobing Xie, Xingwu Sun, Di Wang, Rui Yan, and Yankai Lin. Union-of-experts: Experts in mixture-of-experts are secretly routers, 2025. URL https: //openreview.net/forum?id=Ksgiup7ZNZ. 





Xingyi Yang, Constantin Venhoff, Ashkan Khakzar, Christian Schroeder de Witt, Puneet K. Dokania, Adel Bibi, and Philip Torr. Mixture of experts made intrinsically interpretable. In Fortysecond International Conference on Machine Learning, 2025. URL https://openreview. net/forum?id=6QERrXMLP2. 





Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Llu´ıs Marquez` (eds.), Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10. 18653/v1/P19-1472. URL https://aclanthology.org/P19-1472/. 





Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. AGIEval: A human-centric benchmark for evaluating foundation models. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Findings of the Association for Computational Linguistics: NAACL 2024, pp. 2299–2314, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-naacl.149. URL https://aclanthology.org/2024.findings-naacl.149/. 



Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models, 2022. URL https://arxiv.org/abs/2202.08906. 

## A FREQUENTLY ASKED QUESTIONS

## A.1 WHAT HAPPENS IF $\alpha > 1 2$

Some readers might be interested in the value of α at which the ERC loss degenerates to no effective constraints, and the trained model consequently degenerates to a vanilla MoE. For our baseline MoE, we seek the minimum α that zeros the ERC loss computed from the M matrices of the last checkpoint. Table 2 shows that achieving zero ERC loss across all layers requires $\alpha = 5$ in our pre-trained vanilla MoE baseline. This provides direct evidence that the router-expert coupling in the vanilla MoE is very weak. 

We further pre-trained 3B MoE models with the ERC loss at α values of 2 and 3. It is important to note that this experiment is to only demonstrate the effects of loosening the ERC constraints. We do not recommend using $\alpha > 1$ in practice, as it contradicts our core motivation: the router and experts will shift from a state of no mismatch toward looser coupling constraints, ultimately causing the model to degenerate into a vanilla MoE. As shown in Figure 7, the model with $\alpha = 2$ yields only limited improvement, while the model with $\alpha = 3$ shows almost no improvement over the vanilla MoE. 

## A.2 DO MODELS REDUCE ERC LOSS THROUGH MANIPULATING PARAMETER NORMS?

Some readers might assume that simply increasing or decreasing the norms of certain router embeddings or experts will increase the diagonal entries of M, thereby reducing the ERC loss. Below, we (1) explain that any attempt to reduce one term of the ERC loss by manipulating norms will simultaneously increase other terms, and (2) present detailed parameter norms as evidence. 

Note that $M [ i , j ]$ can be written as $\| \mathbfcal { R } [ i ] \| \| W _ { g } ^ { j } \| \mathrm { c } \tilde { \mathrm { o s } } _ { i , j }$ , where $\mathrm { c } \tilde { \mathrm { o } } { \mathrm { s } _ { i , j } }$ denotes the averaged cosine similarity between $R [ i ]$ and $\boldsymbol { W _ { g } ^ { j } } .$ 4 

Increasing $\lVert \mathbfcal { R } [ i ] \rVert$ decreases the following loss term (as the second term increases): 

$$
\| \boldsymbol {R} [ j ] \| \| \boldsymbol {W} _ {g} ^ {i} \| \mathrm{cos} _ {j, i} - \| \boldsymbol {R} [ i ] \| \| \boldsymbol {W} _ {g} ^ {i} \| \mathrm{cos} _ {i, i}.
$$

<sup>4</sup>Formally, ˜cos $\begin{array} { r } { { \bf \Phi } _ { i , j } = \sum _ { k \le D } \frac { \| \mathbf { W } _ { g } ^ { j } [ : , k ] \| } { \| \mathbf { W } _ { g } ^ { j } \| } \cos \theta _ { i , j , k } } \end{array}$ , where $\theta _ { i , j , k }$ is the angle between $\pmb { R } [ i ]$ and the k-th column of $\boldsymbol { W } _ { g } ^ { j }$ . For clarity, we use this compact form. 


Table 2: Post-hoc ERC loss evaluation of the vanilla MoE across α values. For clarity, loss values are computed using the original R rather than $\tilde { R } ,$ making the results deterministic.


<table><tr><td rowspan="2">Layer</td><td colspan="5">Value of α</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>0</td><td>0.87</td><td>0.69</td><td>0.26</td><td>0.00</td><td>0.00</td></tr><tr><td>1</td><td>0.42</td><td>0.28</td><td>0.10</td><td>0.00</td><td>0.00</td></tr><tr><td>2</td><td>0.45</td><td>0.19</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>3</td><td>0.25</td><td>0.15</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>4</td><td>0.28</td><td>0.08</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>5</td><td>0.24</td><td>0.22</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>6</td><td>0.22</td><td>0.15</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>7</td><td>0.21</td><td>0.13</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>8</td><td>0.15</td><td>0.05</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>9</td><td>0.16</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>10</td><td>0.21</td><td>0.09</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>11</td><td>0.50</td><td>0.44</td><td>0.20</td><td>0.20</td><td>0.00</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/3db6c94b715a7a7abc921fb91fe5c1865afb291143e101cc5bed11e452c890e9.jpg)



Figure 7: $\alpha \ > \ 1$ leads to degeneration into vanilla MoEs


However, it simultaneously increases the following term (as the first term grows): 

$$
\| \boldsymbol {R} [ i ] \| \| \boldsymbol {W} _ {g} ^ {j} \| \mathrm{cos} _ {i, j} - \| \boldsymbol {R} [ j ] \| \| \boldsymbol {W} _ {g} ^ {j} \| \mathrm{cos} _ {j, j}.
$$

Similarly, any attempt to manipulate the norms of $W _ { g }$ to reduce one part of the loss necessarily increases others, assuming that $\| W _ { g } ^ { j } [ : , k ] \| / \| W _ { g } ^ { j } \|$ remains fixed. 

This property ensures that the overall ERC loss is minimized only when the router embedding norms are kept comparable and a meaningful coupling is established between routers and their experts. 

As shown in the first four columns of Table 3, the average parameter norms for models trained with and without the ERC loss are comparable. Meanwhile, the lower standard deviation under the ERC loss reflects more consistent norms across both router embeddings and experts. In the last two columns of the table, we present the ERC loss for each model. The ERC loss is significantly higher in the baseline model despite its similar average parameter norms. 

## B DETERMINING THE MAXIMUM MULTIPLICATIVE NOISE LEVEL

In what follows, we write $\mathbf { { \mathit { R } } } _ { i }$ for $R [ i ]$ and $R _ { i , k }$ for the k-th element of $\pmb { R } [ i ]$ to avoid excessive brackets. $\delta _ { i }$ is a random vector where each component $\delta _ { i , k }$ follows a uniform distribution $\mathcal { U } ( 1 -$ $\epsilon , 1 + \epsilon )$ , and all components are mutually independent. The perturbed point is given by: 

$$
\tilde {\boldsymbol {R}} _ {i} = \left(\boldsymbol {\delta} _ {i, 1} (\boldsymbol {R} _ {i, 1}), \boldsymbol {\delta} _ {i, 2} (\boldsymbol {R} _ {i, 2}), \dots , \boldsymbol {\delta} _ {i, d} (\boldsymbol {R} _ {i, d})\right).
$$

To ensure that $\tilde { R } _ { i }$ remains in the same cluster as $\scriptstyle R _ { i } .$ , it must satisfy: 

$$
\left\| \tilde {\boldsymbol {R}} _ {i} - \boldsymbol {R} _ {i} \right\| ^ {2} <   \left\| \tilde {\boldsymbol {R}} _ {i} - \boldsymbol {R} _ {j} \right\| ^ {2},
$$

where j = arg mi $\iota _ { j ^ { * } \neq i } \left\| \boldsymbol { R } [ i ] - \boldsymbol { R } [ j ] \right\|$ 

Expanding the squared norms on both sides of the inequality yields: 

$$
\| \tilde {\boldsymbol {R}} _ {i} - \boldsymbol {R} _ {i} \| ^ {2} = \sum_ {k = 1} ^ {d} (\boldsymbol {\delta} _ {i, k} - 1) ^ {2} (\boldsymbol {R} _ {i, k}) ^ {2},
$$


Table 3: The first four columns show parameter norms for models trained with and without ERC loss, while the last two show the corresponding layer-wise ERC loss. These results show that $\mathbf { M o E } + \mathcal { L } _ { \mathrm { E R C } }$ learns a meaningful coupling, rather than trivially minimizing the loss through norm manipulation. All values are evaluated on the last checkpoint.


<table><tr><td rowspan="2">Layer</td><td colspan="2"><eq>\|R[i]\|</eq></td><td colspan="2"><eq>\|W_g^i\|</eq></td><td colspan="2"><eq>\mathcal{L}_{ERC}</eq> Values</td></tr><tr><td>Baseline</td><td>+<eq>\mathcal{L}_{ERC}</eq></td><td>Baseline</td><td>+<eq>\mathcal{L}_{ERC}</eq></td><td>Baseline</td><td>+<eq>\mathcal{L}_{ERC}</eq></td></tr><tr><td>0</td><td>1.85±0.39</td><td>1.67±0.31</td><td>25.46±3.93</td><td>24.14±3.02</td><td>0.87</td><td>0.00</td></tr><tr><td>1</td><td>1.25±0.13</td><td>1.13±0.12</td><td>30.14±0.68</td><td>29.42±0.69</td><td>0.42</td><td>0.00</td></tr><tr><td>2</td><td>1.17±0.12</td><td>1.07±0.09</td><td>30.63±0.77</td><td>29.88±0.76</td><td>0.45</td><td>0.00</td></tr><tr><td>3</td><td>1.10±0.08</td><td>1.01±0.07</td><td>30.18±0.77</td><td>29.42±0.78</td><td>0.25</td><td>0.00</td></tr><tr><td>4</td><td>1.03±0.08</td><td>0.89±0.05</td><td>30.59±1.21</td><td>29.88±1.09</td><td>0.28</td><td>0.00</td></tr><tr><td>5</td><td>0.93±0.08</td><td>0.87±0.06</td><td>30.33±1.13</td><td>29.86±1.06</td><td>0.24</td><td>0.00</td></tr><tr><td>6</td><td>0.86±0.08</td><td>0.83±0.07</td><td>30.65±1.15</td><td>29.82±1.11</td><td>0.22</td><td>0.00</td></tr><tr><td>7</td><td>0.82±0.07</td><td>0.75±0.06</td><td>30.56±1.20</td><td>29.96±1.16</td><td>0.21</td><td>0.00</td></tr><tr><td>8</td><td>0.77±0.06</td><td>0.76±0.06</td><td>30.46±1.02</td><td>29.82±0.88</td><td>0.15</td><td>0.00</td></tr><tr><td>9</td><td>0.80±0.07</td><td>0.74±0.06</td><td>30.58±0.88</td><td>29.86±0.79</td><td>0.16</td><td>0.00</td></tr><tr><td>10</td><td>0.74±0.08</td><td>0.69±0.06</td><td>30.80±1.03</td><td>30.16±0.89</td><td>0.21</td><td>0.00</td></tr><tr><td>11</td><td>0.80±0.14</td><td>0.73±0.10</td><td>32.03±1.46</td><td>31.50±1.26</td><td>0.50</td><td>0.00</td></tr></table>

$$
\| \tilde {\boldsymbol {R}} _ {i} - \boldsymbol {R} _ {j} \| ^ {2} = \sum_ {k = 1} ^ {d} (\delta_ {i, k} \boldsymbol {R} _ {i, k} - \boldsymbol {R} _ {j, k}) ^ {2}.
$$

Substituting into the inequality and simplifying gives: 

$$
\sum_ {k = 1} ^ {d} [ 2 \pmb {\delta} _ {i, k} (\pmb {R} _ {i, k} (\pmb {R} _ {j, k} - \pmb {R} _ {i, k}) + (\pmb {R} _ {i, k} ^ {2} - \pmb {R} _ {j, k} ^ {2}) ] <   0.
$$

To ensure this inequality holds for all realizations of $\delta _ { i } ,$ we consider the worst-case scenario that maximizes the left-hand side. Define: 

$$
A _ {k} = 2 \boldsymbol {R} _ {i, k} (\boldsymbol {R} _ {j, k} - \boldsymbol {R} _ {i, k}), \quad B = \sum_ {k = 1} ^ {d} (\boldsymbol {R} _ {i, k} ^ {2} - \boldsymbol {R} _ {j, k} ^ {2}),
$$

so the inequality becomes: 

$$
\sum_ {k = 1} ^ {d} A _ {k} \pmb {\delta} _ {i, k} + B <   0.\tag{5}
$$

The worst-case $\delta _ { i , k }$ is chosen to maximize $\textstyle \sum A _ { k } \delta _ { i , k } \colon$ 

$$
\boldsymbol {\delta} _ {i, k} = \left\{ \begin{array}{l l} 1 + \epsilon & \text { if } A _ {k} > 0, \\ 1 - \epsilon & \text { if } A _ {k} <   0. \end{array} \right.
$$

Substituting these values gives: 

$$
\sum_ {k = 1} ^ {d} A _ {k} + \epsilon \sum_ {k = 1} ^ {d} | A _ {k} | + B <   0.\tag{6}
$$

Now simplify $\sum A _ { k } + B \colon$ 

$$
\begin{array}{r l} \sum A _ {k} + B & = 2 \sum R _ {i, k} R _ {j, k} - 2 \sum R _ {i, k} ^ {2} + \sum R _ {i, k} ^ {2} - \sum R _ {j, k} ^ {2} \\ & = 2 \sum R _ {i, k} R _ {j, k} - \sum R _ {i, k} ^ {2} - \sum R _ {j, k} ^ {2} \\ & = - \left(\sum R _ {i, k} ^ {2} - 2 \sum R _ {i, k} R _ {j, k} + \sum R _ {j, k} ^ {2}\right) \\ & = - \| R _ {i} - R _ {j} \| ^ {2} \end{array}\tag{7}
$$

Substituting equation 7 into equation 6 yields: 

$$
- \left\| \boldsymbol {R} _ {i} - \boldsymbol {R} _ {j} \right\| ^ {2} + 2 \epsilon \sum_ {k = 1} ^ {d} \left| \boldsymbol {R} _ {i, k} \left(\boldsymbol {R} _ {j, k} - \boldsymbol {R} _ {i, k}\right) \right| <   0.
$$

Solving for ϵ gives: 

$$
\epsilon_ {\max} <   \frac {\left\| \boldsymbol {R} _ {j} - \boldsymbol {R} _ {i} \right\| ^ {2}}{2 \sum_ {k = 1} ^ {d} \left| \boldsymbol {R} _ {i , k} (\boldsymbol {R} _ {j , k} - \boldsymbol {R} _ {i , k}) \right|}.
$$

However, computing the denominator of this expression is relatively complex. To balance the efficiency of loss calculation, we instead adopt a tighter upper bound for ϵ. 

By the Cauchy-Schwarz Inequality, the following relationship holds: 

$$
\sum_ {k = 1} ^ {d} \left| \boldsymbol {R} _ {i, k} \left(\boldsymbol {R} _ {j, k} - \boldsymbol {R} _ {i, k}\right) \right| \leq \| \boldsymbol {R} _ {i} \| \| \boldsymbol {R} _ {j} - \boldsymbol {R} _ {i} \|.
$$

Thus, we have: 

$$
\epsilon_ {\max} = \frac {\| \boldsymbol {R} _ {j} - \boldsymbol {R} _ {i} \| ^ {2}}{2 \sum_ {k = 1} ^ {d} | \boldsymbol {R} _ {i , k} (\boldsymbol {R} _ {j , k} - \boldsymbol {R} _ {i , k}) |} \geq \frac {\| \boldsymbol {R} _ {j} - \boldsymbol {R} _ {i} \| ^ {2}}{2 \| \boldsymbol {R} _ {i} \| \| \boldsymbol {R} _ {j} - \boldsymbol {R} _ {i} \|} = \frac {\| \boldsymbol {R} _ {j} - \boldsymbol {R} _ {i} \|}{2 \| \boldsymbol {R} _ {i} \|}.
$$

The term on the right-hand side of the final inequality is the value of ϵ we used in the main text. This choice ensures that the perturbed $\tilde { R } [ i ]$ always remains closer in Euclidean distance to $R [ i ]$ than to any other $R [ j \neq i ]$ 

## C EFFICIENCY ANALYSIS

Appendix C.1 analyzes the ideal FLOPs cost breakdown of the vanilla MoE, as well as the overhead introduced by AoE and ERC loss. Appendix C.2 discusses efficiency with consideration of the multiple parallelism strategies used in real-world MoE pre-training. Both analyses demonstrate the practicality of our method. 

## C.1 FLOPS COST BREAKDOWN OF THREE METHODS

MoE forward Each expert in a MoE layer involves the following operations, with their respective FLOP counts: 

• Two matrix multiplications of dimension $T \times d$ with $d \times D$ , accounting for $4 T d D$ FLOPs. These correspond to the linear transformations parameterized by $W _ { g }$ and $W _ { p }$ 

• One element-wise multiplication of $T \times D$ tensors and one SiLU activation applied to a $T \times D$ tensor. The computational cost of these operations is negligible compared to the matrix multiplications. 

• One matrix multiplication of dimension $T \times D$ with $D \times d ,$ contributing 2TDd FLOPs. This corresponds to the linear transformation parameterized by $W _ { o }$ 

Summing these components gives a total of 6T dD FLOPs per expert. For K experts processing $T$ tokens, the total computational cost is therefore 6KTdD FLOPs. 

Computational overhead of AoE AoE factorizes the expert matrix $W _ { g } \in \mathbb { R } ^ { D \times d }$ into two lowrank matrices of rank r. To maintain the same number of parameters as the original matrix, we require $d r + D r = D d .$ , which gives $\begin{array} { r } { r = \frac { D d } { d + D } } \end{array}$ 

The change in FLOPs compared to an MoE is: 

$$
T \left(\underbrace {2 n d r} _ {\text { All   experts   use   } \boldsymbol {W} _ {\text { down }}} + \underbrace {2 K D r} _ {\text { Top - K   experts   use   } \boldsymbol {W} _ {\text { up }}} - \underbrace {2 K D d} _ {\text { Top - K   experts   use   original   } \boldsymbol {W} _ {g}}\right),
$$

where $T$ is the number of tokens. Substituting the value of r and simplifying leads to an extra computational cost of: 

$$
2 T (n - K) d r.
$$

Computational overhead of expert-router coupling loss It introduces $n ^ { 2 }$ matrix multiplications, each operating on tensors of shape $1 \times d$ and $\bar { d } \times D$ . In total, this results in $2 n ^ { 2 } D d$ extra FLOPs. 

## C.2 THROUGHPUTS UNDER MULTIPLE PARALLELISM STRATEGIES

We now assess the overhead of the ERC loss in a realistic large-scale pre-training setup that employs both data parallelism (DP) and expert parallelism (EP). As derived in our previous analysis, the computational cost of the ERC loss is equivalent to a forward pass on $n ^ { 2 } / 3$ tokens. When distributed across devices, the costs are: 

• Base MoE forward: $K \cdot T / \mathrm { d p } _ { - } \mathrm { s i z e }$ 

• ERC overhead: $n \cdot ( n / \looparrowright$ 

Consider training our 15B-parameter model with the configuration: $K = 8 , T = 3 \times 1 0 ^ { 6 } , n = 2 5 6$ $\mathrm { d } \mathsf { p } _ { - } \mathsf { s i z e } = 6 4$ , and $\tt e p { \mathrm { - } } s \ i z e = 8$ . In this scenario, the ERC overhead constitutes a mere 0.72% of the base model’s forward pass cost. This theoretical estimate is consistent with our empirical measurements: we observe a throughput of 62.03B tokens/day for the baseline versus 61.52B tokens/day for our model, representing only a 0.82% reduction. With a smaller $n = 6 4$ , as in our 3B models trained with dp size=32 and ep size=1 (i.e., EP disabled), the overhead ratio drops further to 0.18%. This analysis confirms the practical efficiency of our method. 

```python
import torch
import torch.nn as nn
import PseudoExpertClass

class MoE(nn.Module):
    def __init__(self, args):
    super().__init__()

    self.experts = PseudoExpertClass(args)
    # Shape of experts.Wg: (n, D, d)
    self.R = torch.nn.Parameter(torch.empty(
    args.n, args.d))

    self.alpha = args.alpha

    def erc_loss(self, M):
    row_diff = (M - self.alpha * torch.diag(M).unsqueeze(1))
    row_diff_clamped = torch.clamp(row_diff, min=0.0)

    col_diff = (M - self.alpha * torch.diag(M).unsqueeze(0))
    col_diff_clamped = torch.clamp(col_diff, min=0.0)

    mask = torch.ones_like(M) - torch.eye(M.size(0), device=M.device)
    total_diff = (row_diff_clamped + col_diff_clamped) * mask

    return total_diff.mean()

    def get_noisy_router(self, R):
    with torch.no_grad():
    norm_R = torch.norm(R, dim=1)
    distances = torch.cdist(R, R, p=2)
    distances.fill_diagonal_(float('inf'))
    min_dist, _ = torch.min(distances, dim=1)
    eps = min_dist / 2 / norm_R

    low = (1 - eps).unsqueeze(1)
    high = (1 + eps).unsqueeze(1)
    noise = torch.rand_like(R)

    return (low + noise * (high - low)) * R

    def forward(self, x):
    erc_loss = 0.0
    if self.training:
    R = self.get_noisy_router(self.R)
    M = torch.norm(torch.einsum('jDd,id->ijD', self.experts.Wg, R), dim=-1)
    erc_loss = self.erc_loss(M)

    logits = x.view(-1, x.shape[-1]) @ self.R.T
    scores = logits.softmax(dim=-1)
    expert_weights, expert_indices = torch.topk(scores, dim=-1)

    return self.experts(x, expert_weights, expert_indices), erc_loss 
```


Figure 8: Pseudo code for expert-router coupling loss in PyTorch.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/1395705f-c105-45c2-9258-b42da8d48794/b84ee197ce85dddbe3106d558b7a2d1924183a70cf1252e519d3fffc8553d267.jpg)



Figure 9: Task-specific downstream results for previous experiments.
