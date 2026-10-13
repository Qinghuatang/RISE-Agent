# CHAINGPT: DUAL-REASONING MODEL WITH RE-CURRENT DEPTH AND MULTI-RANK STATE UPDATES

Yunao Zheng Xiaojie Wang<sup>∗</sup> 

Beijing University of Posts and Telecommunications (BUPT), Beijing, China 

Lei Ren Wei Chen Li Auto Inc., Beijing, China 

## ABSTRACT

Large language models, constrained by the fixed-depth Transformer architecture, struggle to solve complex reasoning tasks in an end-to-end manner. Existing approaches, such as Chain of Thought, improve reasoning depth to some extent but rely heavily on natural language generation, with computational costs increasing rapidly as the length of the generated sequence grows. To address these limitations, we propose ChainGPT, a dual-reasoning model that shifts reasoning into latent computational space. Within each layer, ChainGPT employs multi-substep state updates combined with state-guided sparse attention, enabling deep local computation and efficient long-range modeling without quadratic costs. Across layers, recurrent depth approach iteratively refine latent states, supported by adaptive training and stopping strategies that balance reasoning depth against computational budget. Theoretically, we show that ChainGPT can, in principle, simulate general computation, and empirically it delivers consistent improvements over comparable models, including on reasoning tasks that remain challenging for existing systems. By unifying efficiency and reasoning ability, ChainGPT provides a principled foundation for next-generation language models. 

## 1 INTRODUCTION

Large Language Models (LLMs) have demonstrated strong capabilities in natural language interaction, but their architectures remain fundamentally constrained in expressive power. From the perspective of computational complexity theory, the fixed depth of standard Transformers (Vaswani et al., 2017) places them within circuit classes such as AC<sup>0</sup> or TC<sup>0</sup> (Arora & Barak, 2009; Hahn, 2020), which limits their ability to reliably perform reasoning tasks that scale with problem size, including multi-step planning, symbolic manipulation, and combinatorial problem-solving requiring deep recursion. In other words, under finite precision, LLMs are not Turing complete and thus cannot solve tasks involving deep planning in a purely end-to-end manner. 

To address these limitations, Chain of Thought (CoT) (Wei et al., 2022) has emerged as a mainstream approach, guiding models toward solutions through intermediate natural language steps. However, since reasoning is represented as discrete token sequences, it is inherently restricted by the expressiveness of language, making it difficult to capture nonlinear or parallel reasoning modes (Yao et al., 2023). Furthermore, as task complexity increases, the length of CoT outputs grows rapidly, introducing significant computational overhead. 

Researchers have explored two major directions to address these challenges. The first focuses on improving reasoning strategies, such as Tree of Thoughts (ToT) (Yao et al., 2023), which employs tree search to explore more complex reasoning paths. The second focuses on improving underlying architectural efficiency. For example, RWKV-7 (Peng et al., 2025) leverages faster Recurrent Neural Network (RNN) designs to reduce computational cost, while Jamba (Lieber et al., 2024) and Samba (Ren et al., 2024) adopt hybrid architectures to balance reasoning quality with efficiency. 

Despite these advances, fundamental limitations remain. On the one hand, improved reasoning strategies can capture richer semantic structures but still rely on natural language, and their computational cost grows exponentially with the number of branches or samples. On the other hand, RNNs are more efficient but their finite-dimensional state matrices cannot faithfully retain detailed context, which hinders performance on tasks requiring precise memory (Jelassi et al., 2024). Current hybrid architectures are mostly sequential combinations that lack effective coordination: preserving global attention faces scalability bottlenecks, while restricting attention to local windows fails to resolve the problem of limited state matrix capacity. 

These observations indicate that an effective solution must address two fundamental challenges at once: enabling deep reasoning beyond simple token generation, and sustaining computational efficiency while preserving long-range dependencies. To meet these challenges, we introduce ChainGPT, a novel architecture that shifts reasoning from explicit token-level processes into a latent computational space. By integrating multi-substep computation within layers and iterative refinement across layers, ChainGPT achieves both reasoning depth and efficiency, maintaining near-linear complexity. 

Intra-layer reasoning is realized through the Chain-Block, which integrates a state layer and an attention layer. In the state layer, we employ RWKV-Product, a LoRA-based multi-substep update mechanism that enables the model to perform multiple state updates within a single forward pass. Since each substep is low-rank, the computation remains highly efficient, while their cumulative effect can approximate high-rank transformations, allowing this layer to significantly enhance the expressiveness of state updates without incurring substantial computational overhead. To ensure the model can retain comprehensive contextual information, the attention layer introduces State-Guided Sparse Attention (SGSA). Unlike other RNNs that compress the entire history into a finite state, SGSA only attends to recent tokens within sliding windows and a set of periodic anchors. As the sequence grows, contextual information is progressively written into the state, enabling the state space to scale naturally with length and distributing memory across block-level fragments. Through this approach, the model maintains near-linear complexity while still guaranteeing effective recall of long-range information. 

Inter-layer reasoning employs recurrent depth approach (Geiping et al., 2025), dividing the model into a bottom-level feature extractor, a recursive reasoning core, and a top-level output layer. The recursive core iteratively refines latent states until convergence, while training stability is ensured through a two-phase “gradient-free warmup + truncated backpropagation” strategy with entropybased early stopping, offering controllable trade-offs between reasoning depth and compute. 

Theoretically, ChainGPT is Turing complete and can model any computable function. Empirically, it solves problems that remain challenging for state-of-the-art LLMs, even without pretraining or CoT supervision. Across benchmarks, ChainGPT consistently outperforms comparably sized baselines on diverse tasks. 

The main contributions of this work are summarized as follows: 

1. We propose the ChainGPT architecture, introducing a dual-reasoning mechanism for language model reasoning. ChainGPT combines intra-layer deep computation with inter-layer iterative refinement, while maintaining near-linear complexity. 

2. We introduce RWKV-Product, a LoRA-based multi-substep state update mechanism that significantly enhances intra-layer reasoning depth and the effective rank of state updates with minimal parameter overhead. And State-Guided Sparse Attention (SGSA), which builds long-range dependency paths with near-linear complexity and effectively resolves the state bottleneck problem of RNN. 

3. By combining recurrent depth approach with intra-layer sub-step reasoning, ChainGPT shifts iterative reasoning from token generation to multi-round latent optimization, enabling dual recursive reasoning within the model. We prove theoretically that ChainGPT is Turing complete under ideal conditions, and experiments show that it surpasses existing methods on some challenging reasoning tasks and consistently outperforms baseline models of comparable scale on benchmark evaluations. 

## 2 RELATED WORK

## 2.1 EXPRESSIVE BOTTLENECKS OF RNN MODELS

Although RNNs offer efficiency advantages, theoretical studies reveal a fundamental trade-off between expressiveness and computational feasibility. Merrill et al. (2024) showed that while linear RNNs with full-rank transition matrices can, in theory, recognize any regular language, their training costs are prohibitively high, limiting practical applicability. Grazzi et al. (2025) further demonstrated that under finite numerical precision, RNNs with diagonal transition matrices (e.g., Mamba (Gu & Dao, 2023)) fail to perform even basic state-tracking tasks. This observation was empirically validated by Jelassi et al. (2024), who found that RNNs exhibit significant performance bottlenecks in tasks such as multi-hop reasoning. 

To balance efficiency and expressiveness, RWKV-7 introduced a “diagonal + rank-1” structural refinement, which achieved modest performance gains without substantial computational overhead. However, since each update is restricted to rank-1, its expressive power remains limited. DeltaProduct (Siems et al., 2025) attempted to address this issue through multi-step Householder updates, but the additional parameter overhead makes it impractical. The RWKV-Product proposed in this work introduces a LoRA-based multi-substep decoupled state update mechanism. Within a single forward pass, it executes multiple sub-steps, enabling “multiple rounds of reasoning within one layer.” With only a small increase in parameters, the effective rank of updates can be dynamically extended to M (a tunable hyperparameter), resulting in faster convergence and stronger representational power. 

## 2.2 DIFFERENT DESIGN PRINCIPLES OF HYBRID ARCHITECTURES

Given the respective limitations of pure RNN and pure Transformer architectures, recent years have seen the emergence of various hybrid models that attempt to combine the strengths of both. 

Jamba adopts a 1:7 alternating stack of Mamba and Transformer layers, retaining full self-attention mechanisms. While this approach excels in long-sequence tasks, the embedded $O ( N ^ { 2 } )$ selfattention modules become the primary scalability bottleneck. For instance, at a 32k context length, the computation of a single attention layer already exceeds that of seven Mamba layers combined. In contrast, ChainGPT employs a sparse attention design, reducing complexity to $\dot { O ( N ( W + N / G ) ) }$ where W denotes window size and G the global anchor interval, thereby enabling near-linear scalability. 

Samba alternates between Mamba and sliding-window attention (SWA) to achieve hybrid modeling. However, pure window-based mechanisms still face state bottlenecks. Our experiments confirm that for long-context tasks (>8k), pure SWA architectures exhibit degraded performance, whereas the global anchor mechanism introduced in ChainGPT completely resolves this issue. 

## 2.3 REASONING METHODS BEYOND FIXED DEPTH

Research aimed at enhancing the reasoning ability of fixed-depth Transformers can be broadly divided into two categories: generation-based reasoning extensions and internal state-based reasoning enhancements. 

The former is exemplified by CoT. CoT and its variants extend effective computational depth by generating structured reasoning steps. ToT generalizes this idea into tree search, enabling the exploration of multiple reasoning paths. Self-Consistency (Wang et al., 2022) improves reliability by ensembling multiple reasoning chains through voting. However, these methods face inherent limitations: computational cost is tightly coupled with generation length, and the reasoning process is constrained by the expressive power of natural language. 

The latter category focuses on iterative computation within the model’s latent state space. Universal Transformer (Dehghani et al., 2019) introduced depth-adaptive mechanisms that allow different positions to perform varying numbers of computational steps. Looped Transformer (Liu et al., 2023) placed the entire model in a loop, effectively increasing depth via multiple forward passes. More recent work explores recursion in Transformers, such as the Hierarchical Reasoning Model (HRM) (Wang et al., 2025), which performs iterative computation across hierarchical levels, and the Depth-Recurrent Model, which refines the Looped Transformer by proposing recurrent depth approach. 

ChainGPT belongs to the category of internal-state reasoning but innovatively introduces a dualreasoning mechanism: RWKV-Product performs multi-substep state updates within a single forward pass, explicitly enhancing intra-layer reasoning depth; recurrent depth approach, combined with truncated backpropagation and adaptive early stopping, achieve a stable trade-off between reasoning depth and computational budget. 

## 3 CHAIN-BLOCK

## 3.1 OVERALL DESIGN

ChainGPT is composed of multiple stacked Chain-Blocks. Each block consists of two core substructures: a state layer and an attention layer, as illustrated in Figure 1. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/1c696a1dbb104063dca929c324bef15600972cd2f5a1ad074aaa83c52b5400e4.jpg)



Figure 1: Chain-Block


State Layer: The core of the state layer is the proposed RWKV-Product mechanism. It employs a “multi-substep reasoning” approach, in which multiple substeps are executed within a single forward pass. Each substep maintains its own key-value representations and gating strategy. By adjusting the number of substeps, the model can explicitly increase its intra-layer reasoning depth without adding additional layers. 

Attention Layer: Based on the output of the state layer, SGSA attends only to tokens within a local window and anchor tokens sampled at fixed intervals. By leveraging context summaries generated through multi-substep reasoning, SGSA constructs sparse yet globally covering dependency connections. 

This design of “multi-step reasoning + sparse aggregation” enables ChainGPT to achieve deep reasoning within a single layer at near-linear computational cost, while efficiently modeling long-range semantic dependencies across layers. 

## 3.2 RWKV-PRODUCT

Let $\mathbf { x } _ { t }$ denote the input at step t, and $\mathbf { s } _ { t } \in \mathbb { R } ^ { D \times D }$ the state matrix (D is the input dimension): 

$$
\mathbf {s} _ {t} = A (\mathbf {x} _ {t}) \mathbf {s} _ {t - 1} + B (\mathbf {x} _ {t}).\tag{1}
$$

Here $A ( \mathbf { x } _ { t } ) \in \mathbb { R } ^ { D \times D }$ is the transition matrix, and $B ( \mathbf { x } _ { t } )$ the injection matrix. Each sub-step j has its own $A _ { j } ( \mathbf { x } _ { t } )$ and $B _ { j } ( \mathbf { x } _ { t } )$ , so 

$$
A (\mathbf {x} _ {t}) = \prod_ {j = 0} ^ {M - 1} \left(\operatorname{diag} \left(\mathbf {a} _ {t, j}\right) - \beta_ {b, j} \mathbf {k} _ {t, j} ^ {(b)} \mathbf {k} _ {t, j} ^ {\top}\right),\tag{2}
$$

where diag $( \mathbf { a } _ { t , j } )$ applies channel-wise decay, and $\beta _ { b , j }$ controls the strength of the subsequent rank-1 correction. With independent update vectors, $A ( \mathbf { x } _ { t } )$ exhibits a “diagonal + rank-M” structure. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/6e2bee3ad015e8f01431a6ad58bffb6b1e2a36f03c3b31e9658821be28662a74.jpg)



Figure 2: State-Guided Sparse Attention mechanism


The injection matrix aggregates sub-step contributions: 

$$
B (\mathbf {x} _ {t}) = \sum_ {j = 0} ^ {M - 1} \Big (\prod_ {k = j + 1} ^ {M - 1} A _ {t, k} \Big) \big (\beta_ {c, j} \mathbf {k} _ {t, j} ^ {(c)} \mathbf {v} _ {t, j} ^ {\top} \big),\tag{3}
$$

where $\mathbf { k } _ { t , j } ^ { ( c ) }$ and $\mathbf { v } _ { t , j }$ denote c-path key and value vectors, and $\beta _ { c , j }$ determines the weight of the sub-step contribution. 

To balance expressiveness and parameter efficiency, all keys/values adopt a shared baseline plus LoRA increments: 

$$
k _ {t, j} ^ {(b)} = k _ {t} ^ {\mathrm{base}} + x _ {t} W _ {j} ^ {(b, k 1)} W _ {j} ^ {(b, k 2)},\tag{4}
$$

$$
k _ {t, j} ^ {(b)} = \frac {k _ {t , j} ^ {(b)}}{\| k _ {t , j} ^ {(b)} \| _ {2} + \epsilon},\tag{5}
$$

$$
k _ {t, j} ^ {(c)} = k _ {t} ^ {\mathrm{base}} + x _ {t} W _ {j} ^ {(c, k 1)} W _ {j} ^ {(c, k 2)},\tag{6}
$$

$$
v _ {t, j} = v _ {t} ^ {\mathrm{base}} + x _ {t} W _ {j} ^ {(c, v 1)} W _ {j} ^ {(c, v 2)}.\tag{7}
$$

Step sizes are specific to each sub-step, with shared bias $\alpha _ { 0 , j }$ between the b- and c-paths. The symbol σ denotes the sigmoid activation function: 

$$
\beta_ {b, j} = \sigma \big (\alpha_ {0, j} + (\mathbf {x} _ {t} W _ {1, j} ^ {(b)}) W _ {2, j} ^ {(b)} \big),\tag{8}
$$

$$
\beta_ {c, j} = \sigma \big (\alpha_ {0, j} + (\mathbf {x} _ {t} W _ {1, j} ^ {(c)}) W _ {2, j} ^ {(c)} \big).\tag{9}
$$

This multi-substep design strictly enhances representational capacity with increasing M. Proofs and details appear in Appendix A; implementation in Appendix D.1. 

## 3.3 STATE-GUIDED SPARSE ATTENTION (SGSA)

To capture long-range dependencies, we introduce SGSA, which operates on state outputs. For a query position, the accessible key set consists of: 

• Local Window: W neighbors around the query 

• Global Anchors: positions sampled every stride G 

Thus, the Chain-Block decomposes retrieval into “write and pointer-read”: RWKV-Product aggregates local fragments and propagates them to anchors, while SGSA retrieves content via sparse addressing (Figure 2). 

Appendix B proves that with proper hyperparameters, Chain-Block solves the Multi-Query Associative Recall (MQAR) task (Arora et al., 2024) for arbitrarily long sequences: for any $( k _ { i } , v _ { i } )$ , if $q _ { j } = k _ { i }$ , the model retrieves v<sub>i</sub>. 

Compared with dense $O ( T ^ { 2 } )$ attention, our method requires only $O ( T ( W + T / G ) )$ sparse links, enabling efficient global modeling. Implementation is provided in Appendix D.2. 

## 4 RECURRENT DEPTH APPROACH

## 4.1 OVERALL DESIGN

The recurrent depth approach decomposes the model into three modules: a bottom-level feature extractor, a recursive reasoning core, and a top-level output module. Each is constructed from stacked 

Chain-Blocks. As shown in Figure 3, the bottom-level module generates stable contextual representations, the recursive core conducts multi-round reasoning over them, and the top-level module maps the final states into the target output distribution. Typically, the bottom-level and top-level modules each contain a single block. A full algorithmic description is provided in Appendix D.3. 

This design enables ChainGPT to achieve computational universality with sufficient memory and time, allowing it to simulate any Turing machine and thus model arbitrary computable functions. A formal proof is provided in Appendix C. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/2e1163b6adc741a9e6d1f07f5d1bba70234145912a97a2df9916f71859908cdb.jpg)



Figure 3: Recurrent depth approach


## 4.2 ADAPTIVE REASONING DEPTH

To reduce redundancy, an entropy-based early stopping mechanism monitors output stability. At iteration t, the core output is decoded: 

$$
\operatorname{logits} _ {t} = \operatorname{Decode} (X _ {t}), \quad \ell_ {t} = \operatorname{logits} _ {t} [:, - 1,: ], \quad p _ {t} = \operatorname{softmax} (\ell_ {t}).\tag{10}
$$

Stopping Criterion: Entropy for sample b is 

$$
H _ {t} (b) = - \sum_ {i = 1} ^ {V} p _ {t, i} (b) \log \left(p _ {t, i} (b) + \varepsilon\right),\tag{11}
$$

with difference 

$$
\Delta H _ {t} (b) = H _ {t - k} (b) - H _ {t} (b).\tag{12}
$$

If $\Delta H _ { t } ( b ) \leq \tau$ , recursion halts at step t. Threshold τ is a fixed constant. 

## 5 EXPERIMENTS

This section validates the effectiveness of the ChainGPT architecture through comprehensive benchmarks, systematic ablations, and scalability analyses. Unless otherwise stated, all experiments are conducted on 8 NVIDIA L20 GPUs. 

## 5.1 COMPREHENSIVE PERFORMANCE EVALUATION

We train ChainGPT models with 0.5B and 1.5B parameters, as well as baselines (Qwen2.5 (Qwen Team, 2024)), on 20B and 40B tokens from the FineWeb dataset (Penedo et al., 2024). Evaluation follows the lm-eval-harness framework (Biderman et al., 2024) under a zero-shot setting, covering six representative tasks on language understanding and commonsense reasoning (Clark et al., 2018; Zellers et al., 2019; Bisk et al., 2020; Welbl et al., 2017; Wang et al., 2018). As shown in Table 1, ChainGPT significantly outperforms baselines with the same parameter counts, with notable gains on ARC-Challenge and HellaSwag—tasks that require more complex reasoning—indicating stronger general-purpose reasoning ability. 


Table 1: Comparison of comprehensive performance results


<table><tr><td>Model</td><td>arc_challenge</td><td>arc_easy</td><td>hellaswag</td><td>piqa</td><td>sciq</td><td>glue</td><td>Avg.</td></tr><tr><td>Qwen2.5-0.5B</td><td>0.2218</td><td>0.4082</td><td>0.3224</td><td>0.6425</td><td>0.5290</td><td>0.4664</td><td>0.4317</td></tr><tr><td>ChainGPT-0.5B</td><td>0.2389</td><td>0.4773</td><td>0.3644</td><td>0.6632</td><td>0.5330</td><td>0.4679</td><td>0.4575</td></tr><tr><td>Qwen2.5-1.5B</td><td>0.2696</td><td>0.5488</td><td>0.4091</td><td>0.6915</td><td>0.6380</td><td>0.4783</td><td>0.5059</td></tr><tr><td>ChainGPT-1.5B</td><td>0.2986</td><td>0.5779</td><td>0.4269</td><td>0.7018</td><td>0.6860</td><td>0.4836</td><td>0.5291</td></tr></table>

## 5.2 VALIDATION OF ARCHITECTURAL EFFECTIVENESS

## Effectiveness of the LoRA Sub-Step Mechanism

Building on BlinkDL/modded-nanogpt-rwkv (BlinkDL, 2024), we extend RWKV-7 with a single LoRA sub-step to construct RWKV-Product. As shown in Table 2, with only a 0.1M increase in parameters, RWKV-Product reaches lower validation loss with fewer training steps, indicating that intra-layer multi-substep updates improve both training efficiency and representational capacity. 


Table 2: Validation of the Effectiveness of the LoRA Sub-step Mechanism


<table><tr><td>Model</td><td>Training Steps</td><td>Validation Loss</td></tr><tr><td>GPT2</td><td>19560</td><td>≈3.28</td></tr><tr><td>RWKV-7</td><td>3200</td><td>3.2715</td></tr><tr><td>RWKV-Product</td><td>2500</td><td>3.2684</td></tr><tr><td>RWKV-Product</td><td>3200</td><td>3.1901</td></tr></table>

## Impact of Sub-Step Count on Performance

To study how the number of sub-steps affects performance, we evaluate models (180M parameters) trained with different sub-step counts M. Training uses 8B tokens from FineWeb and evaluation is performed on 1,000 validation samples. As shown in Figure 4, validation loss consistently decreases as M increases, with M=2 offering a favorable trade-off. These results suggest that sub-steps effectively deepen intra-layer computation, thereby improving representational capacity. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/2ebf0dfe469e689476fdbc906456c4e43b5c8bbae2fd80c25b226be60d3c42e4.jpg)



Figure 4: Effect of sub-step count on model performance


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/1b40297286c44d1c94569407e5d608fe32c802847114705adbc35e03f1b94c0a.jpg)



Figure 5: Comparison of coupled and decoupled state propagation paths


## Necessity of Decoupling State Propagation Paths

We compare decoupled and coupled designs for state propagation. With approximately matched parameter counts, the decoupled variant consistently outperforms the coupled version throughout training, as shown in Figure 5. 

## Hybrid Architecture Resolves Associative Recall Bottlenecks

On the MQAR task, with head dimension = 64 and number of heads = 2, the pure State Space Model architecture (RWKV-7) exhibits noticeable accuracy degradation as sequence length grows, whereas ChainGPT with SGSA maintains accuracy above 99% across settings (Table 3). This indicates that SGSA’s “anchor writing + sparse pointer retrieval” mechanism effectively mitigates the capacity bottleneck caused by finite state dimensionality. 


Table 3: Performance on the MQAR task


<table><tr><td>Model</td><td>(128,8)</td><td>(256,16)</td><td>(512,64)</td><td>(1024,128)</td><td>(2048,256)</td></tr><tr><td>RWKV-7</td><td>&gt;99%</td><td>&gt;99%</td><td>98.43%</td><td>95.01%</td><td>72.93%</td></tr><tr><td>ChainGPT</td><td>&gt;99%</td><td>&gt;99%</td><td>&gt;99%</td><td>&gt;99%</td><td>&gt;99%</td></tr></table>

## Effectiveness of the Global Anchor Strategy

We evaluate different global anchor intervals G on PG-19 (Rae et al., 2019). As shown in Table 4, pure sliding-window attention (Samba approach) degrades beyond 8K context. In contrast, introducing only sparse periodic global anchors yields a perplexity trajectory that closely matches the much more expensive global attention (Jamba approach), with no notable degradation. This suggests that SGSA’s “compression–aggregation” mechanism maintains long-range dependency modeling at near-linear complexity. 


Table 4: Perplexity comparison for global anchor strategies


<table><tr><td>Global Token Interval (G)</td><td>2k</td><td>4k</td><td>6k</td><td>8k</td><td>10k</td><td>12k</td><td>14k</td><td>16k</td></tr><tr><td>Global Attention</td><td>17.78</td><td>16.82</td><td>16.46</td><td>16.20</td><td>16.11</td><td>16.07</td><td>16.07</td><td>16.08</td></tr><tr><td>G=32</td><td>17.78</td><td>16.82</td><td>16.46</td><td>16.20</td><td>16.11</td><td>16.07</td><td>16.07</td><td>16.09</td></tr><tr><td>G=64</td><td>17.78</td><td>16.82</td><td>16.46</td><td>16.20</td><td>16.11</td><td>16.07</td><td>16.07</td><td>16.08</td></tr><tr><td>G=128</td><td>17.77</td><td>16.81</td><td>16.46</td><td>16.20</td><td>16.11</td><td>16.07</td><td>16.07</td><td>16.08</td></tr><tr><td>Pure Window Attention</td><td>17.78</td><td>16.83</td><td>16.47</td><td>16.21</td><td>16.12</td><td>16.09</td><td>16.10</td><td>16.12</td></tr></table>

## Impact of Recurrence Depth on Performance

Keeping other conditions fixed, we increase the number of recursive iterations from ×1 to ×16. As shown in Figure 6, validation perplexity decreases steadily with more iterations and reaches its best at ×12. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/d3756096a9975a821c48e75ed5c123aa22d23f81f9f2849324c5015ede2c48ae.jpg)



Figure 6: Effect of recurrence depth on model performance


## Comparison with Other Recurrent Models and CoT Methods

To directly evaluate the performance of ChainGPT against other recurrent models and CoT methods, we conducted controlled arithmetic reasoning experiments on the public GOAT arithmetic dataset (Liu & Low, 2023) in Table 5. All models were trained from scratch under identical training–testing splits, tokenization schemes, and optimization settings to ensure fair comparison. The evaluated models include the baseline architectures Qwen3 and RWKV-7, their recurrent variants (Qwen3+Loop, RWKV-7+Loop), HRM, our proposed ChainGPT, as well as their CoT-augmented counterparts (Qwen3+CoT, ChainGPT+CoT). 


Table 5: Accuracy on the GOAT dataset


<table><tr><td>Model</td><td>Accuracy</td></tr><tr><td>Qwen3</td><td>33.82%</td></tr><tr><td>RWKV-7</td><td>23.69%</td></tr><tr><td>Qwen3 + Loop</td><td>50.54%</td></tr><tr><td>RWKV-7 + Loop</td><td>24.81%</td></tr><tr><td>HRM</td><td>54.00%</td></tr><tr><td>ChainGPT</td><td>57.53%</td></tr><tr><td>Qwen3 + CoT</td><td>88.43%</td></tr><tr><td>ChainGPT + CoT</td><td>99.98%</td></tr></table>

Without CoT supervision, ChainGPT outperforms existing recurrent reasoning architectures, indicating that its intrinsic latent recursion mechanism enables more effective multi-step arithmetic reasoning. When combined with explicit CoT guidance, ChainGPT+CoT significantly surpasses Qwen3+CoT, achieving nearly perfect accuracy. This result suggests that ChainGPT’s implicit recursive states and the external CoT trajectory are complementary, jointly enhancing the model’s reasoning capacity. 

## Analysis on Challenging Reasoning Tasks

To assess ChainGPT in complex reasoning scenarios, we conduct systematic evaluations on three representative hard tasks: ARC-AGI-1 (Chollet, 2019), Sudoku-Extreme, and Maze-Hard (Wang et al., 2025). Following HRM, we train 30M-parameter ChainGPT from scratch on 1000 samples, and train Direct Pred as a Transformer baseline under identical settings. Results are shown in Figure 7. For ARC-AGI-1, the scores are taken from the official leaderboard, while for Sudoku-Extreme and Maze-Hard, the scores are obtained through the corresponding APIs. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/02b1ba1beb169ca98845663cc8bb48a2a028805485bd252bd78d4cc6284b3e54.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/7e6b65c9eaac7a87c045b00a018dbdcc5c479d1186046346bbf06fd89c552d2d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/445a9762ea90e5fbf314c15072c10b7de649d1b5240a8a3768251b46f8678eb5.jpg)



Figure 7: Benchmark accuracy comparison across different tasks


ARC-AGI-1 evaluates abstract rule induction under extreme few-shot conditions, where each problem provides only a handful of input–output examples. The central challenge lies in learning reusable compositional rules rather than memorizing patterns. Conventional deep learning and CoTbased approaches often fail to achieve stable generalization in such settings. 

Sudoku-Extreme aggregates acknowledged hard Sudoku instances ranging from Kaggle datasets to Magictour-1465 and Forum-Hard/Extreme, with tdoku backtracking steps used as an objective difficulty measure. The difficulty stems from the need for precise constraint propagation and backtracking. Fixed-depth Transformers are not well-suited for long-trajectory backtracking, and parallel CoT sampling cannot reliably ensure deterministic correctness. 

Maze-Hard requires finding shortest paths on 30 × 30 grids with enforced path lengths > 110. The challenge is to construct optimal solutions—mere reachability is insufficient. Large Transformers and CoT reasoning often generate suboptimal or invalid paths on large-scale mazes, showing their limitations in stable graph search. 

ChainGPT excels in these scenarios due to its dual-reasoning mechanism, which enables multiround, revisitable reasoning beyond conventional one-pass prediction or shallow CoT sampling. This unified design supports iterative rule induction and constraint propagation with contradiction checking, allowing the model to emulate algorithmic processes more faithfully across tasks. As shown in Figure 7, ChainGPT achieves 38.6% accuracy on ARC-AGI-1—surpassing strong baselines such as o3-mini-high and Claude-3.7-8K—54.4% accuracy on Sudoku-Extreme, and 77.4% accuracy on Maze-Hard with valid optimal solutions. These results suggest that ChainGPT effectively addresses the limitations of existing approaches and demonstrates robust performance on challenging reasoning tasks. 

## 5.3 SPEED BENCHMARKING

We evaluate the computational efficiency of different modules on sequence lengths from 4K to 16K using a single NVIDIA A800 GPU, with total model dimension = 1024 and head dimension = 64, and attention implemented via F.scaled dot product attention. Results in Figure 8 show that standard attention suffers from a quadratic scalability bottleneck as sequence length N increases, whereas RWKV-Product achieves approximately linear runtime growth with controllable overhead across sub-step configurations, enabling an efficiency–performance trade-off. Furthermore, the hybrid architecture, which integrates state layers with sparse attention, demonstrates near-linear scaling through a “state compression—sparse aggregation” strategy, offering clear efficiency advantages for long-range modeling under constrained computational cost. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/17ca9f83-8a13-4048-8f61-fdcc24b90006/25bb55e74a97dba00ed261c31c3dc04301349e9a66eea161d1e6d7533e096c01.jpg)



Figure 8: Computation efficiency comparison (ms)


## 6 CONCLUSION

In summary, ChainGPT introduces a principled dual-reasoning architecture that directly addresses the expressiveness bottlenecks of existing LLMs by shifting reasoning from token-level generation into latent computational space. Through the combination of RWKV-Product’s multi-substep state updates and SGSA’s near-linear global modeling, ChainGPT achieves intra-layer depth and efficient long-range reasoning simultaneously, while recurrent depth approach further extend its computational universality. Theoretical analysis establishes Turing completeness under proper configurations, and extensive experiments confirm that ChainGPT not only surpasses state-of-the-art baselines of comparable scale but also solves reasoning tasks that remain unsolved by current LLMs. These results highlight ChainGPT’s unique advantage in unifying efficiency, scalability, and reasoning power, opening new directions for the design of next-generation language models. 

## 7 ACKNOWLEDGMENTS

The work is supported by Beijing Natural Science Foundation (L247010) and National Key RD Program of China (No. 2024YFF0907003). We are grateful to the anonymous reviewers for their valuable and encouraging feedback and suggestions on this study. 

## REFERENCES



Sanjeev Arora and Boaz Barak. Computational Complexity: A Modern Approach. Cambridge University Press, 2009. 





Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Re. Zoology: Measuring and improving recall in efficient language mod-´ els. In The Twelfth International Conference on Learning Representations (ICLR), 2024. URL https://openreview.net/forum?id=LY3ukUANko. 





Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, and et al. Lessons from the trenches on reproducible evaluation of language models. 2024. arXiv preprint. 





Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2020. 





BlinkDL. modded-nanogpt-rwkv: Nanogpt speedrun + rwkv integration. https://github. com/BlinkDL/modded-nanogpt-rwkv, 2024. Achieves 3.278 validation loss in 7 min on 8×H100; integrates Rotary, QK-Norm, ReLU², Muon optimizer, RWKV-7 attention-free model. Accessed June 2025. 





Franc¸ois Chollet. On the measure of intelligence: The abstraction and reasoning corpus (arc-agi). 2019. Introduces the ARC-AGI benchmark focused on fluid intelligence. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. 2018. arXiv preprint. 





Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. In International Conference on Learning Representations (ICLR), 2019. Introduces per-position adaptive halting (ACT) for variable computation depth. 





Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025. URL https://arxiv.org/abs/2502.05171. 





Riccardo Grazzi, Julien Siems, Arber Zela, Jorg K.H. Franke, Frank Hutter, and Massimiliano Pon-¨ til. Unlocking state-tracking in linear rnns through negative eigenvalues. In Proceedings of the International Conference on Learning Representations (ICLR), 2025. 





Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2023. arXiv preprint. 





Michael Hahn. Theoretical limitations of self-attention in neural sequence models. Transactions of the Associationfor Computational Linguistics (TACL), 8:156–171, 2020. 





Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach. Repeat after me: Transformers are better than state space models at copying, 2024. arXiv preprint. 





Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, and Yoav Shoham. Jamba: A hybrid transformer-mamba language model, 2024. arXiv preprint. 





Tiedong Liu and Bryan Kian Hsiang Low. Goat: Fine-tuned llama outperforms gpt-4 on arithmetic tasks. arXiv preprint arXiv:2305.14201, 2023. doi: 10.48550/arXiv.2305.14201. 





Yang Liu, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. arXiv preprint arXiv:2311.12424, 2023. 





William Merrill, Jackson Petty, and Ashish Sabharwal. The Illusion of State in State-Space Models, 2024. arXiv preprint. 





Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. 2024. arXiv preprint. 





Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing Liu, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, and Christian Zhou-Zheng. RWKV-7 “goose” with expressive dynamic state evolution, 2025. arXiv preprint. 





Qwen Team. Qwen2.5 technical report. 2024. arXiv preprint. 





Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. arXiv preprint arXiv:1911.05507, 2019. doi: 10.48550/arXiv.1911.05507. 





Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling, 2024. arXiv preprint. 





Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi. Deltaproduct: Improving state-tracking in linear rnns via householder products, 2025. arXiv preprint. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pp. 5998–6008, 2017. 





Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. 2018. arXiv preprint. 





Guan Wang, Jin Li, Yuhao Sun, Xing Chen, Changling Liu, Yue Wu, Meng Lu, Sen Song, and Yasin Abbasi-Yadkori. Hierarchical reasoning model. arXiv preprint arXiv:2506.21734, 2025. Two-level recurrent modules for multi-timescale reasoning in a single forward pass. 





Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Denny Zhou, Sharan Narang, and Aakanksha Chowdhery. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022. 





Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. 2022. arXiv preprint. 





Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. 2017. arXiv preprint. 





Shunyu Yao, Dian Yu, Jeffrey Zhao, and et al. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023. 





Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019. 



## A THEORETICAL ADVANTAGES OF RWKV-PRODUCT’S EXPRESSIVE POWER

This appendix presents a rigorous theoretical analysis of the multi-substep update mechanism used in RWKV-Product. Our goal is to show, mathematically, that as the number of sub-steps M increases, both the family of realizable linear transformations and the capacity of information injection strictly expand. This analysis provides a solid mathematical foundation for the advantages of multi-substep updates over traditional single-step updates and supports the design’s rationale. 

To keep notation consistent, we first define the update operator for each sub-step. Let $D _ { t , j } : =$ $\mathrm { d i a g } ( \mathbf { \bar { a } } _ { t , j } ) \in \mathbb { R } ^ { D \times D }$ be a diagonal matrix. For simplicity, we absorb the step-size factors into the vectors by defining $\mathbf { u } _ { t , j } : = \sqrt { \beta _ { b , j } } \mathbf { k } _ { t , i } ^ { ( b ) }$ and $\mathbf { v } _ { t , j } : = \sqrt { \beta _ { b , j } } \mathbf { k } _ { t , j }$ . Hence, at time $t ,$ the transformation matrix of the j-th sub-step has the following “diagonal minus rank-1” form: 

$$
A _ {t, j} = D _ {t, j} - \mathbf {u} _ {t, j} \mathbf {v} _ {t, j} ^ {\top}\tag{13}
$$

Basic Assumption: To ensure the validity of subsequent derivations, we assume all diagonal matrices $D _ { t , j }$ are invertible; i.e., no diagonal entry is zero. 

## A.1 LOW-RANK STRUCTURE OF PRODUCTS: COMPACT WY REPRESENTATION

Theoretical significance: Equivalently viewing the composite effect of M successive rank-1 updates as a single “diagonal + rank-M” transformation $A ( \mathbf { x } _ { t } ) = \widetilde { D } - \widetilde { U } \widetilde { T } \widetilde { V } ^ { \top }$ clarifies the algebraic structure underlying enhanced expressiveness: the upper bound on attainable rank increases from 1 to M. 

Lemma 1: Given vector pairs $\{ ( \mathbf { u } _ { j } , \mathbf { v } _ { j } ) \} _ { j = 0 } ^ { M - 1 }$ , let $\begin{array} { r } { \hat { A } = \prod _ { j = M - 1 } ^ { 0 } ( I - \mathbf { u } _ { j } \mathbf { v } _ { j } ^ { \top } ) } \end{array}$ . Then there exist matrices $U , V \in \mathbb { R } ^ { D \times M }$ and an upper-triangular matrix $T \in \dot { \mathbb { R } } ^ { M \times M }$ such that 

$$
\hat {A} = I - U T V ^ {\top}, \quad \text { and } \quad \operatorname{rank} (U T V ^ {\top}) \leq M.\tag{14}
$$

Proof: By induction. Let $\begin{array} { r } { \hat { A } _ { m } = \prod _ { j = m - 1 } ^ { 0 } ( I - \mathbf { u } _ { j } \mathbf { v } _ { j } ^ { \top } ) } \end{array}$ ). The base case $m = 1$ is immediate. Assume $\hat { A } _ { m } = I - U _ { m } T _ { m } V _ { m } ^ { \top }$ with $T _ { m } \in \mathbb { R } ^ { m \times m }$ upper triangular. Consider $\hat { A } _ { m + 1 } = \hat { A } _ { m } ( I - \mathbf { u } _ { m } \mathbf { v } _ { m } ^ { \top } )$ : 

$$
\hat {A} _ {m + 1} = (I - U _ {m} T _ {m} V _ {m} ^ {\top}) (I - \mathbf {u} _ {m} \mathbf {v} _ {m} ^ {\top})\tag{15}
$$

$$
= I - U _ {m} T _ {m} V _ {m} ^ {\top} - \mathbf {u} _ {m} \mathbf {v} _ {m} ^ {\top} + (U _ {m} T _ {m} V _ {m} ^ {\top} \mathbf {u} _ {m}) \mathbf {v} _ {m} ^ {\top}\tag{16}
$$

$$
= I - \left[ U _ {m}, \mathbf {u} _ {m} \right] \left[ \begin{array}{c c} T _ {m} & - T _ {m} (V _ {m} ^ {\top} \mathbf {u} _ {m}) \\ \mathbf {0} & 1 \end{array} \right] \left[ \begin{array}{c} V _ {m} ^ {\top} \\ \mathbf {v} _ {m} ^ {\top} \end{array} \right].\tag{17}
$$

Define $U _ { m + 1 } = [ U _ { m } , \mathbf { u } _ { m } ] , V _ { m + 1 } = [ V _ { m } , \mathbf { v } _ { m } ]$ , and the corresponding $( m + 1 ) \times ( m + 1 )$ uppertriangular $T _ { m + 1 }$ . The inductive step holds. 

Lemma 2: Suppose each sub-step transformation is $A _ { j } = D _ { j } - \mathbf { u } _ { j } \mathbf { v } _ { j } ^ { \intercal }$ . Then their product $A : =$ $\textstyle \prod _ { j = M - 1 } ^ { 0 } A _ { j }$ can be written as 

$$
A = \widetilde {D} - \widetilde {U} \widetilde {T} \widetilde {V} ^ {\top},\tag{18}
$$

where $\begin{array} { r } { \widetilde { D } = \prod _ { j = M - 1 } ^ { 0 } D _ { j } } \end{array}$ and rank $\lvert \tilde { U } \tilde { T } \tilde { V } ^ { \top } ) \leq M$ 

Proof: Again by induction. Let $\begin{array} { r } { P _ { m } = \prod _ { j = m - 1 } ^ { 0 } A _ { j } } \end{array}$ . The base case $m = 1$ is trivial. Assume $P _ { m } = \widetilde { D } _ { m } - L _ { m }$ , where $\begin{array} { r } { \widetilde { D } _ { m } = \prod _ { j = m - 1 } ^ { 0 } D _ { j } } \end{array}$ and ran $\mathtt { c } ( L _ { m } ) \le m$ . Consider $P _ { m + 1 } = A _ { m } P _ { m } =$ $( D _ { m } - \mathbf { u } _ { m } \mathbf { v } _ { m } ^ { \top } ) ( \widetilde { D } _ { m } - L _ { m } )$ : 

$$
P _ {m + 1} = (D _ {m} \widetilde {D} _ {m}) - \underbrace {\left[ D _ {m} L _ {m} + (\mathbf {u} _ {m} \mathbf {v} _ {m} ^ {\top}) \widetilde {D} _ {m} - (\mathbf {u} _ {m} \mathbf {v} _ {m} ^ {\top}) L _ {m} \right]} _ {L _ {m + 1}}.\tag{19}
$$

Let $\widetilde { D } _ { m + 1 } = D _ { m } \widetilde { D } _ { m }$ . The column space of the low-rank part $L _ { m + 1 }$ is contained in $\mathrm { c o l } ( D _ { m } L _ { m } ) +$ span $\left( \mathbf { u } _ { m } \right)$ . Since ran $\mathtt { k } ( L _ { m } ) \leq m$ , we have dim $( \mathrm { c o l } ( D _ { m } L _ { m } ) ) \le m$ , hence dim $\begin{array} { r } { ( \mathrm { c o l } ( L _ { m + 1 } ) ) \le } \end{array}$ $m + 1 , \mathrm { i . e . }$ ., ran $\left[ L _ { m + 1 } \right) \leq m + 1$ . The induction closes. 

Conclusion 1: $A ( \mathbf { x } _ { t } ) = \widetilde { D } - \widetilde { U } \widetilde { T } \widetilde { V } ^ { \top }$ with $\operatorname { r a n k } ( \widetilde { U } \widetilde { T } \widetilde { V } ^ { \top } ) \leq M$ 

## A.2 STRICT INCLUSION: $\mathcal { F } _ { 1 } \subsetneq \mathcal { F } _ { M }$ FOR $M \geq 2$

Theoretical significance: We prove, in a constructive manner, that multi-substep updates strictly enlarge the family of realizable linear transformations. Concretely, there exist matrices representable by a product of two “diagonal minus rank-1” factors $\left( M { = } 2 \right)$ , but not representable by any single-step mechanism $( M { = } 1 )$ ). This establishes a set-theoretic notion of strictly stronger expressive power. 

Define the matrix families (diagonal factors are assumed invertible, consistent with the basic assumption in Appendix A): 

$$
\mathcal {F} _ {M} := \left\{\prod_ {j = M - 1} ^ {0} \left(D _ {j} - \mathbf {u} _ {j} \mathbf {v} _ {j} ^ {\top}\right): D _ {j} \text {   invertible   diagonal } \right\},\tag{20}
$$

$$
\mathcal {F} _ {1} := \left\{D - \mathbf {u v} ^ {\top}: D \text {   invertible   diagonal } \right\}.\tag{21}
$$

Theorem 1: For dimension $D \geq 3$ and sub-steps $M \geq 2$ , we have the strict inclusion 

$$
\mathcal {F} _ {1} \subsetneq \mathcal {F} _ {M}.\tag{22}
$$

Proof: We show (i) $\mathcal { F } _ { 1 } \subseteq \mathcal { F } _ { M }$ , and (ii) there exists a matrix in $\mathcal { F } _ { 2 }$ but not in $\mathcal { F } _ { 1 }$ . 

(i) Inclusion $\mathcal { F } _ { 1 } \subseteq \mathcal { F } _ { M }$ . Take any $A \in { \mathcal { F } } _ { 1 }$ , so $\boldsymbol { A } = \boldsymbol { D } - \mathbf { u v } ^ { \top }$ for some invertible diagonal D. Construct an element of ${ \mathcal { F } } _ { M }$ by setting 

$$
D _ {0} = D, \mathbf {u} _ {0} = \mathbf {u}, \mathbf {v} _ {0} = \mathbf {v}, \quad D _ {j} = I, \mathbf {u} _ {j} = \mathbf {0} (j = 1, \dots , M - 1),
$$

where I is the identity matrix (invertible diagonal). Then 

$$
\prod_ {j = M - 1} ^ {0} \left(D _ {j} - \mathbf {u} _ {j} \mathbf {v} _ {j} ^ {\top}\right) = (I) \dots (I) \left(D - \mathbf {u v} ^ {\top}\right) = A,
$$

hence $\mathcal { F } _ { 1 } \subseteq \mathcal { F } _ { M }$ 

(ii) Strictness via a constructive separation. Proposition 1 below constructs an explicit matrix $A ^ { \star } \in$ $\mathcal { F } _ { 2 }$ such that $A ^ { \star } \notin { \mathcal { F } } _ { 1 }$ . Since $\mathcal { F } _ { 2 } \subseteq \mathcal { F } _ { M }$ for all $\bar { M } \geq 2$ , this implies $\mathcal { F } _ { 1 } \subsetneq \mathcal { F } _ { M }$ 

Proposition 1 (Constructive separation): Let $D \geq 3$ and $\{ e _ { i } \} _ { i = 1 } ^ { D }$ be the standard basis of $\mathbb { R } ^ { D }$ Define 

$$
A ^ {\star} := (I - e _ {2} e _ {3} ^ {\top}) (I - e _ {1} e _ {2} ^ {\top}) = I - e _ {1} e _ {2} ^ {\top} - e _ {2} e _ {3} ^ {\top}.\tag{23}
$$

Then $A ^ { \star } \in { \mathcal { F } } _ { 2 }$ but $A ^ { \star } \notin { \mathcal { F } } _ { 1 }$ 

Proof: The membership $A ^ { \star } \in { \mathcal { F } } _ { 2 }$ is immediate: both factors are of the form 

$$
I - e _ {a} e _ {b} ^ {\top} = D - \mathbf {u v} ^ {\top} \quad \text {with} \quad D = I, \mathbf {u} = e _ {a}, \mathbf {v} = e _ {b},
$$

where D is invertible diagonal. 

To show $A ^ { \star } \notin { \mathcal { F } } _ { 1 }$ , assume for contradiction that 

$$
A ^ {\star} = D - \mathbf {u v} ^ {\top}
$$

for some invertible diagonal D and vectors ${ \mathbf { u } } , { \mathbf { v } } \in \mathbb { R } ^ { D }$ . For any $i \neq j$ , the diagonal matrix contributes nothing to off-diagonal entries, hence 

$$
A _ {i j} ^ {\star} = - (\mathbf {u v} ^ {\top}) _ {i j} = - u _ {i} v _ {j}, \qquad i \neq j.\tag{24}
$$

From the explicit form $A ^ { \star } = I - e _ { 1 } e _ { 2 } ^ { \top } - e _ { 2 } e _ { 3 } ^ { \top }$ , we have the off-diagonal pattern 

$$
A _ {1 2} ^ {\star} = - 1, \qquad A _ {2 3} ^ {\star} = - 1, \qquad A _ {1 3} ^ {\star} = 0.
$$

The equalities $A _ { 1 2 } ^ { \star } = - u _ { 1 } v _ { 2 } = - 1$ and $A _ { 2 3 } ^ { \star } = - u _ { 2 } v _ { 3 } = - 1$ imply 

$$
u _ {1} \neq 0, v _ {2} \neq 0, \quad \text { and } \quad u _ {2} \neq 0, v _ {3} \neq 0.
$$

In particular, $u _ { 1 } \neq 0$ and $v _ { 3 } \neq 0$ together imply $u _ { 1 } v _ { 3 } \neq 0 .$ , hence 

$$
A _ {1 3} ^ {\star} = - u _ {1} v _ {3} \neq 0,
$$

which contradicts $A _ { 1 3 } ^ { \star } = 0$ . Therefore $A ^ { \star } \notin { \mathcal { F } } _ { 1 }$ 

Remark (Degrees-of-freedom heuristic). A single-step form $D - \mathbf { u } \mathbf { v } ^ { \top }$ has roughly $D + 2 D - 1 =$ $3 D - 1$ effective degrees of freedom (accounting for the scale redundancy $( \mathbf { u } , \mathbf { v } ) \mapsto ( \alpha \mathbf { u } , \alpha ^ { - 1 } \mathbf { v } ) ) . \mathtt { A }$ 1 two-step product introduces (at least) twice as many parameters, suggesting a strictly larger family. However, since the parameterization is not injective, we rely on the explicit separation in Proposition 1 (rather than on dimensional counting) to establish strict inclusion. 

Conclusion 2: There exists $A \in \mathcal { F } _ { M } \setminus \mathcal { F } _ { 1 }$ for any $M \geq 2$ and $D \geq 3 ( \mathrm { e . g . , } A ^ { \star }$ above), hence $\mathcal { F } _ { 1 } \subsetneq \mathcal { F } _ { M }$ 

## A.3 RANK UPPER BOUND AND EXPANSION FOR THE INJECTION TERM $B ( \mathbf { x } _ { t } )$

Theoretical significance: Not only does the expressiveness of the state transition A increase, but the capacity of the injection channel B expands in tandem, jointly enhancing the affine transformation capability. 

By definition, the injection matrix $\begin{array} { r } { B ( \mathbf { x } _ { t } ) = \sum _ { j = 0 } ^ { M - 1 } P _ { t , j } R _ { t , j } } \end{array}$ , where rank $( R _ { t , j } ) = 1$ 

Proposition 2: For fixed input $\mathbf { x } _ { t } ,$ the rank of $B ( \mathbf { x } _ { t } )$ satisfies 

$$
\operatorname{rank} (B (\mathbf {x} _ {t})) \leq M.\tag{25}
$$

Equality holds when the image spaces of $\{ P _ { t , j } R _ { t , j } \} _ { j }$ are linearly independent. 

Proof: By the rank inequality for matrix products, rank $( P _ { t , j } R _ { t , j } )$ ≤ min $\{ \mathrm { r a n k } ( P _ { t , j } )$ ), rank $\cdot ( R _ { t , j } ) \} = 1$ . Using subadditivity of rank, 

$$
\operatorname{rank} (B (\mathbf {x} _ {t})) = \operatorname{rank} \left(\sum_ {j = 0} ^ {M - 1} P _ {t, j} R _ {t, j}\right) \leq \sum_ {j = 0} ^ {M - 1} \operatorname{rank} (P _ {t, j} R _ {t, j}) \leq M.\tag{26}
$$

Conclusion 3: rank $\mathbf { \Gamma } ( B ( \mathbf { x } _ { t } ) ) \leq M$ , and the upper bound is tight. 

## A.4 SPECTRAL PROPERTIES AND STABILITY

Theoretical significance: Enhanced expressiveness does not come at the expense of numerical stability. The diagonal factors control spectral contraction, while rank-1 terms act as bounded perturbations that contribute expressiveness. 

Lemma 3: Let $\begin{array} { r } { \widetilde { D } = \prod _ { j = M - 1 } ^ { 0 } D _ { t , j } } \end{array}$ . Owing to commutativity of diagonal matrices, the spectral radius $\rho ( \cdot )$ satisfies 

$$
\rho (\widetilde {D}) \leq \prod_ {j = 0} ^ {M - 1} \rho (D _ {t, j}).\tag{27}
$$

Proof: Write $D _ { t , j } = \mathrm { d i a g } ( d _ { t , j } ^ { ( 1 ) } , \dots , d _ { t , j } ^ { ( D ) } )$ . Then $\begin{array} { r } { \widetilde { D } = \mathrm { d i a g } ( \prod _ { j } d _ { t , j } ^ { ( 1 ) } , \dots , \prod _ { j } d _ { t , j } ^ { ( D ) } ) } \end{array}$ . Hence 

$$
\rho (\widetilde {D}) = \max _ {i} \left| \prod_ {j} d _ {t, j} ^ {(i)} \right| = \max _ {i} \prod_ {j} \left| d _ {t, j} ^ {(i)} \right| \leq \prod_ {j} \left(\max _ {k} \left| d _ {t, j} ^ {(k)} \right|\right) = \prod_ {j} \rho (D _ {t, j}).\tag{28}
$$

Lemma 4: The norm of the difference between $A ( \mathbf { x } _ { t } )$ and its diagonal-dominant part $\widetilde { D }$ is bounded. Using the telescoping identity: 

$$
\prod_ {j = 0} ^ {M - 1} X _ {j} - \prod_ {j = 0} ^ {M - 1} Y _ {j} = \sum_ {m = 0} ^ {M - 1} \left(\prod_ {j = 0} ^ {m - 1} X _ {j}\right) (X _ {m} - Y _ {m}) \left(\prod_ {j = m + 1} ^ {M - 1} Y _ {j}\right),\tag{29}
$$

and setting $X _ { j } = A _ { t , j }$ and $Y _ { j } = D _ { t , j }$ , we have $X _ { m } - Y _ { m } = - \mathbf { u } _ { t , m } \mathbf { v } _ { t , m } ^ { \top }$ . Substituting and applying norm inequalities yields: 

$$
\left\| A \left(\mathbf {x} _ {t}\right) - \prod_ {j} D _ {t, j} \right\| _ {2} \leq \sum_ {m = 0} ^ {M - 1} \left(\prod_ {j = 0} ^ {m - 1} \| A _ {t, j} \| _ {2}\right) \| \mathbf {u} _ {t, m} \| _ {2} \| \mathbf {v} _ {t, m} \| _ {2} \left(\prod_ {j = m + 1} ^ {M - 1} \| D _ {t, j} \| _ {2}\right),\tag{30}
$$

so the bound depends solely on the norms of sub-step parameters. 

Conclusion 4: The spectral radius is controlled stepwise by the diagonal factors, while the rank-1 terms serve as bounded perturbations. 

## A.5 STRICT EXPANSION OF THE AFFINE TRANSFORMATION FAMILY

Theoretical significance: Collecting the above results, we provide a unified functional-family statement of the expressiveness gains afforded by multi-substep updates. 

Define the family of affine maps: 

$$
\mathcal {G} _ {M} := \{s \mapsto A (\mathbf {x}) s + B (\mathbf {x}): A (\mathbf {x}) \in \mathcal {F} _ {M}, B (\mathbf {x}) \text {   satisfies   Proposition   2 } \}.\tag{31}
$$

Theorem 2: For $M \geq 2$ and dimension $D \geq 2 .$ , the single-substep affine family $\mathcal { G } _ { 1 }$ is a strict subset of the multi-substep family $\mathcal { G } _ { M } \mathbf { : }$ 

$$
\mathcal {G} _ {1} \subsetneq \mathcal {G} _ {M}.\tag{32}
$$

Proof: This follows directly from the preceding results. By Theorem 1, the linear part expands strictly, $\mathcal { F } _ { 1 } \subsetneq \mathcal { F } _ { M }$ . By Proposition 2, the rank upper bound of the affine injection increases from 1 to M. The simultaneous enhancement of both parts ensures a strict expansion of the overall function family. 

Conclusion 5: $\mathcal { G } _ { 1 } \subsetneq \mathcal { G } _ { M } -$ the intra-layer expressive power of RWKV-Product increases strictly with the number of sub-steps M. 

## B CONSTRUCTIVE PROOF OF CHAIN-BLOCK SOLVING MQAR FOR ARBITRARY SEQUENCE LENGTHS

This appendix proves that a single Chain-Block (RWKV-Product + SGSA) can solve the Multi-Query Associative Recall (MQAR) task (Arora et al., 2024) for arbitrarily long sequences under idealized assumptions. The purpose is to justify the main-text intuition: RWKV-Product performs lossless “write/overwrite” in latent state, while SGSA performs sparse “pointer-read” over local windows and periodic anchors. 

## B.1 PROBLEM SETUP AND OBJECTIVE

MQAR Task (Latest-Write-Wins). We observe a length-T sequence where some positions are writes of key-value pairs $( k _ { t } , v _ { t } )$ and some positions are queries with $q _ { t } \in \mathcal { K }$ . For each query at time t, the correct answer is the value associated with the most recent write of the same key: 

$$
\operatorname{Ans} (t) = v _ {\tau^ {\star}}, \qquad \tau^ {\star} = \max \{\tau \leq t: k _ {\tau} = q _ {t} \}.
$$

This is the Latest-Write-Wins (LWW) semantics. 

Chain-Block. A Chain-Block consists of: (1) an RWKV-Product state layer that updates a latent state and emits per-token representations; and (2) a State-Guided Sparse Attention (SGSA) layer that performs causal sparse attention over (i) a local window of size W and (ii) a set of periodic anchor positions with stride G. 

Goal. We show there exists a parameter setting such that a single Chain-Block outputs the correct Ans(t) at every query position t, for any sequence length T, while respecting LWW. 

## B.2 IDEALIZED ASSUMPTIONS (EXISTENCE PROOF)

We assume: 

1. Finite key universe and sufficient dimension. The key set K is finite, and the model has enough hidden dimension to allocate a dedicated “associative memory” subspace. 

2. Orthogonal key encoding. There exists an encoding map $\phi : \mathcal { K }  \mathbb { R } ^ { d _ { k } }$ such that 

$$
\langle \phi (k), \phi (k ^ {\prime}) \rangle = \delta_ {k, k ^ {\prime}}.
$$

3. Injective value encoding. There exists an encoding $\psi : \mathcal { V } \to \mathbb { R } ^ { d _ { v } }$ that is injective (e.g., one-hot or any separable embedding). 

4. Sufficient numerical precision. We can make attention “hard” by scaling logits and we can treat linear updates as exact up to an arbitrarily small error. 

5. RWKV-Product expressivity for low-rank affine state updates. RWKV-Product can realize (up to small error) the rank-1 projector-and-inject updates described below on a designated memory subspace, by choosing appropriate diagonal decay and rank-1 terms (consistent with its “diagonal + low-rank” structure). 

## B.3 MEMORY ENCODING FOR MQAR

We use an explicit associative-memory construction. 

Outer-product notation. We use ab<sup>⊤</sup> for the outer product, producing a matrix in $\mathbb { R } ^ { d _ { k } \times d _ { v } }$ 

Periodic anchors. Define the anchor set 

$$
\mathcal {A} = \{t \in \{1, \dots , T \}: t \bmod G = G - 1 \}.
$$

SGSA will allow each query t to attend to positions in 

$$
\mathcal {D} (t) = \{t ^ {\prime}: \max (1, t - W) \leq t ^ {\prime} \leq t \} \cup \{a \in \mathcal {A}: a \leq t \}.
$$

Two-level memory: segment buffer + global memory. We maintain two associative memories: 

• Segment (recent) memory $( P _ { t } ^ { \mathrm { s e g } } , M _ { t } ^ { \mathrm { s e g } } )$ that stores writes since the most recent anchor. 

• Global memory $( P _ { t } ^ { \mathrm { g l o b } } , M _ { t } ^ { \mathrm { g l o b } } )$ that stores all writes up to time t. 

Each memory consists of a presence vector $P \in \mathbb { R } ^ { d _ { k } }$ and a value matrix $M \in \mathbb { R } ^ { d _ { k } \times d _ { v } }$ 

$$
P _ {t} (\cdot) \approx \sum_ {k \in \mathcal {K}} \mathbf {1} \{k \text {   has   appeared } \} \phi (k), \quad M _ {t} \approx \sum_ {k \in \mathcal {K}} \phi (k) \psi (v _ {k} (t)) ^ {\top},
$$

where $v _ { k } ( t )$ denotes the latest written value of key k up to time t. 

Correct readout (no vec/Kronecker required). If the above invariant holds, then for any query key $q \in { \mathcal { K } }$ 

$$
M _ {t} ^ {\top} \phi (q) = \psi (v _ {q} (t)),\tag{33}
$$

i.e., multiplying by $\phi ( q )$ selects the corresponding value embedding (because $\phi ( \cdot )$ are orthonormal). 

## B.4 RWKV-PRODUCT LAYER: LWW WRITE/OVERWRITE AND PERSISTENCE

We now specify update rules that enforce LWW exactly, and argue RWKV-Product can implement them on its memory subspace. 

Rank-1 overwrite for LWW. Let $e _ { t } : = \phi ( k _ { t } ) \in \mathbb { R } ^ { d _ { k } }$ and $u _ { t } : = \psi ( v _ { t } ) \in \mathbb { R } ^ { d _ { v } }$ for a write token at time t. Define the project-and-inject update: 

$$
M _ {t} = \left(I - e _ {t} e _ {t} ^ {\top}\right) M _ {t - 1} + e _ {t} u _ {t} ^ {\top}.\tag{34}
$$

This overwrites the row corresponding to $k _ { t }$ with $u _ { t }$ while leaving all other keys unchanged: 

$$
M _ {t} ^ {\top} \phi (k _ {t}) = u _ {t}, \qquad M _ {t} ^ {\top} \phi (k ^ {\prime}) = M _ {t - 1} ^ {\top} \phi (k ^ {\prime}) \text {for} k ^ {\prime} \neq k _ {t}.
$$

Likewise, presence can be maintained by 

$$
P _ {t} = \left(I - e _ {t} e _ {t} ^ {\top}\right) P _ {t - 1} + e _ {t},\tag{35}
$$

which sets the coefficient of key $k _ { t }$ in $P _ { t }$ to 1 (in the orthonormal basis). 

Global and segment updates. For a write at time t: 

$$
M _ {t} ^ {\mathrm{glob}} = (I - e _ {t} e _ {t} ^ {\top}) M _ {t - 1} ^ {\mathrm{glob}} + e _ {t} u _ {t} ^ {\top}, P _ {t} ^ {\mathrm{glob}} = (I - e _ {t} e _ {t} ^ {\top}) P _ {t - 1} ^ {\mathrm{glob}} + e _ {t},
$$

and similarly for the segment memory: 

$$
M _ {t} ^ {\mathrm{seg}} = (I - e _ {t} e _ {t} ^ {\top}) M _ {t - 1} ^ {\mathrm{seg}} + e _ {t} u _ {t} ^ {\top}, P _ {t} ^ {\mathrm{seg}} = (I - e _ {t} e _ {t} ^ {\top}) P _ {t - 1} ^ {\mathrm{seg}} + e _ {t}.
$$

For a non-write token, we keep both memories unchanged (identity transition). 

At an anchor position $t \in { \mathcal { A } }$ , after producing the token representation (see below), we reset the segment buffer: 

$$
M _ {t} ^ {\mathrm{seg}} = 0, \qquad P _ {t} ^ {\mathrm{seg}} = 0.\tag{36}
$$

This ensures the segment buffer always represents writes after the last anchor. 

RWKV-Product updates its state via an affine map with a “diagonal + low-rank” transition and lowrank injection. On a designated memory subspace, one can choose: (i) diagonal factors close to 1 (to preserve memory), (ii) rank-1 terms proportional to $e _ { t } e _ { t } ^ { \top }$ (to realize $( I - \bar { e } _ { t } e _ { t } ^ { \top } ) )$ , and (iii) injection terms proportional to $e _ { t } u _ { t } ^ { \top }$ (to realize the overwrite). Anchor resets in (36) can be implemented by conditioning on a binary anchor-indicator feature $\mathbf { 1 } _ { \left\{ t \in \mathcal { A } \right\} }$ and applying a diagonal contraction to the segment channels. Since RWKV-Product supports multi-substep low-rank updates, these operations can be composed within a single time step on disjoint channel blocks. 

## B.5 WHAT RWKV-PRODUCT EMITS TO SGSA

Let $y _ { t }$ denote the RWKV-Product output at position t (the input to SGSA). We construct $y _ { t }$ to expose: 

• Segment summary (always visible locally): a linearized payload of $( P _ { t } ^ { \mathrm { s e g } } , M _ { t } ^ { \mathrm { s e g } } )$ 

• Global checkpoint (only at anchors): if $t \in A ,$ , additionally expose a payload of $( P _ { t } ^ { \mathrm { g l o b } } , M _ { t } ^ { \mathrm { g l o b } } )$ ; otherwise this part is zeroed. 

• Anchor indicator: a bit $\mathbf { 1 } _ { \left\{ t \in \mathcal { A } \right\} }$ 

Concretely, we may write (schematically) 

$$
y _ {t} = \Big [ \underbrace {P _ {t} ^ {\text {seg}} \oplus \operatorname{vec} (M _ {t} ^ {\text {seg}})} _ {\text {segment payload}} \oplus \mathbf {1} _ {\{t \in \mathcal {A} \}} \cdot \underbrace {\big (P _ {t} ^ {\text {glob}} \oplus \operatorname{vec} (M _ {t} ^ {\text {glob}}) \big)} _ {\text {anchor payload}} \oplus (\text {other channels}) \Big ].
$$

This matches the intended “recent write in local window + far history at sparse anchors” decomposition. 

## B.6 SGSA LAYER: SPARSE POINTER-READ RETRIEVES THE CORRECT VALUE

We now show SGSA can recover the correct value for each query using only the sparse domain $\mathcal { D } ( t )$ . 

Attention domain. For a query position t, SGSA attends over 

$$
\mathcal {D} (t) = \{t - W, \dots , t \} \cup \mathcal {A} _ {\leq t}.
$$

Keys: presence-based addressability. Define the query vector (in the key-embedding space) as 

$$
Q _ {t} = \alpha \phi (q _ {t}), \quad \alpha > 0.
$$

For each visible position $\mathrm { p o s } \in \mathcal { D } ( t )$ , define its addressing key as 

$$
K _ {\mathrm{pos}} = \left\{ \begin{array}{l l} P _ {\mathrm{pos}} ^ {\mathrm{seg}}, & \mathrm{pos} \notin \mathcal {A}, \\ P _ {\mathrm{pos}} ^ {\mathrm{glob}}, & \mathrm{pos} \in \mathcal {A}. \end{array} \right.
$$

Then, by orthogonality, 

$$
\langle Q _ {t}, K _ {\mathrm{pos}} \rangle = \alpha \cdot \langle \phi (q _ {t}), K _ {\mathrm{pos}} \rangle \in \{0, \alpha \},
$$

indicating whether the queried key $q _ { t }$ is present in that memory snapshot. 

Recency bias enforces LWW selection. Let the attention score be 

$$
\operatorname{score} (t, \text { pos }) = \langle Q _ {t}, K _ {\text { pos }} \rangle + \rho (\text { pos } - t),\tag{37}
$$

where $\rho ( \Delta )$ is any strictly increasing function of $\Delta \leq 0 ( \mathrm { i . e }$ ., more recent positions get larger bias). Under the causal mask, pos $\leq t$ always holds. 

In the limit $\alpha \to \infty$ , the softmax concentrates on the most recent visible position where the key is present: 

$$
\operatorname{pos} ^ {\star} (t) = \arg \max _ {\operatorname{pos} \in \mathcal {D} (t)} \operatorname{score} (t, \operatorname{pos}) = \max \{\operatorname{pos} \in \mathcal {D} (t): \langle \phi (q _ {t}), K _ {\operatorname{pos}} \rangle = 1 \}.
$$

Values: return the selected memory payload. Let SGSA return (approximately) the value payload of the selected position: 

$$
\widetilde {m} _ {t} \approx \left\{ \begin{array}{l l} \mathrm{vec} (M _ {t} ^ {\mathrm{seg}}), & \mathrm{pos} ^ {\star} (t) = t (\text {segment case}), \\ \mathrm{vec} (M _ {\mathrm{pos} ^ {\star} (t)} ^ {\mathrm{glob}}), & \mathrm{pos} ^ {\star} (t) \in \mathcal {A} (\text {anchor case}). \end{array} \right.
$$

(Using self-attention, $\mathrm { p o s } ^ { \star } ( t ) = t \mathrm { i s } \mathrm { a l l o w e d . } )$ 

Final readout. From the selected matrix $M _ { \mathrm { s e l } }$ (segment or global), we obtain the value embedding by 

$$
\widehat {\psi} (v) = M _ {\mathrm{sel}} ^ {\top} \phi (q _ {t}),
$$

which equals the correct $\psi ( \mathrm { A n s } ( t ) )$ by the LWW invariants of (34). This multiplication is a standard associative-memory read; it can be implemented by a fixed readout head that consumes both the query representation (which contains $\phi ( q _ { t } ) )$ ) and the retrieved payload (which contains $M _ { \mathrm { s e l } } )$ , e.g., via a bilinear form on the designated subspace. 

## B.7 CORRECTNESS SKETCH

We summarize why the above construction solves MQAR for any $T .$ 

Case 1: the latest write of $q _ { t }$ occurs after the last anchor. Then $q _ { t }$ is present in the segment buffer at time t, i.e., $\left. \phi ( q _ { t } ) , P _ { t } ^ { \mathrm { s e g } } \right. = 1$ , so position t itself is a valid match. By the strictly increasing recency bias $\rho , \mathrm { p o s } ^ { \star } ( t ) = t$ . SGSA retrieves $M _ { t } ^ { \mathrm { s e g } }$ , and by (33) outputs the value of the latest write within the segment, which equals the latest write overall. 

Case 2: the latest write of $q _ { t }$ occurs at or before the last anchor. Then $\langle \phi ( q _ { t } ) , P _ { t } ^ { \mathrm { s e g } } \rangle = 0$ (it is absent from the segment buffer), but $q _ { t }$ is present in the global memory checkpoint at the most recent anchor $a \leq t .$ . Thus $\mathrm { p o s } ^ { \star } ( t ) = a$ , and SGSA retrieves $M _ { a } ^ { \mathrm { g l o b } }$ which encodes the latest value up to a. Since there is no later write after $a$ (by assumption of this case), this equals the correct LWW answer. 

Therefore, each query retrieves $\psi ( \mathrm { A n s } ( t ) )$ exactly (under the idealized assumptions). Since anchors exist periodically, the argument does not depend on T. 

## C TURING COMPLETENESS OF CHAINGPT

This appendix proves that the proposed recurrent depth formulation of ChainGPT is, in principle, computationally universal. 

## C.1 SETUP AND MAIN RESULT

Recall the recurrent depth formulation in the main text. Let the input token sequence be embedded and processed by a bottom module to obtain a static basis (read-only context) 

$$
H _ {0} = \mathrm{Bottom} (\mathrm{Embed} (X)) \in \mathbb {R} ^ {S \times d},\tag{38}
$$

which remains fixed during recursion. Let $X _ { t } \in \mathbb { R } ^ { S \times d }$ denote the writable latent state at recurrent iteration t, updated by a fixed-parameter map 

$$
X _ {t + 1} = F _ {\theta} (X _ {t}, H _ {0}),\tag{39}
$$

where θ is shared across all iterations (a single “interpreter step” repeatedly applied). 

Idealized conditions. We make the following standard assumptions, which are common in universality proofs for neural architectures: 

1. Unbounded workspace and time: the sequence length S (workspace size) and the number of recurrent iterations can be arbitrarily large. 

2. Arbitrary-precision real computation: internal activations support arbitrarily precise real values. Equivalently, we may assume that sufficiently sharp winner-take-all selection (hardmax limit) and saturated gates can be implemented to any desired tolerance. 

3. Structured primitives available in a fixed-depth block: the architecture provides (i) residual connections, (ii) elementwise gating/multiplicative interactions (e.g., gated-MLP / RWKV-style gates) enabling masked overwrite, and (iii) position-adjacent access operators (e.g., time shift or an equivalent fixed Toeplitz/conv operator). 

4. Sparse attention compatibility: for the theoretical construction, we store the program on global anchor positions so that it is always visible under SGSA (or, as a degenerate special case, we set the anchor stride $G = 1$ so that all program slots are anchors). This ensures random access to program instructions within the sparse retrieval domain. 

Under these assumptions, we prove computational universality by simulation of a two-counter Minsky machine (2CM), which is known to be equivalent in power to a Turing machine. 

Theorem C.1 (Turing Completeness). Under the above idealized conditions, ChainGPT with recurrent depth updates (39) can simulate an arbitrary Turing machine; hence it is Turing complete in principle. 

We give a constructive simulation of one step of a 2CM by a single application of $F _ { \theta }$ . Since any Turing machine can be compiled into a 2CM program, repeated application of $F _ { \theta }$ reproduces the computation trace. 

## C.2 STATE LAYOUT AND ENCODING

We interpret the sequence axis as a 1D workspace of length $S \left( \mathrm { a \ ^ { * } t a p e ^ { * } } \right)$ , where each position stores a d-dimensional vector. We reserve contiguous segments of positions for different roles: 

$$
\{1, \dots , S \} = \mathcal {P} \cup \mathcal {R} _ {1} \cup \mathcal {R} _ {2} \cup \mathcal {U},\tag{40}
$$

where: 

• $\mathcal { P }$ is a program segment containing instruction slots (read-only, stored in $H _ { 0 } )$ ; 

$\mathcal { R } _ { 1 } , \mathcal { R } _ { 2 }$ are register segments (writable, stored in $X _ { t } )$ for counters $R _ { 1 } , R _ { 2 } ;$ 

• U is unused padding (all zeros), allowing the workspace to expand. 

Program storage on anchors (SGSA compatibility). Let the SGSA anchor set be 

$$
\mathcal {A} = \{i \in \{1, \dots , S \}: i \bmod G = G - 1 \}.\tag{41}
$$

For the proof, we place each instruction slot at an anchor position: ${ \mathcal { P } } \subseteq A$ (or simply take $G = 1 )$ Thus, any causal SGSA query position occurring after $\mathcal { P }$ can attend to all program slots through the anchor links. 

Instruction representation. A 2CM program is a finite list of instructions indexed by $m \in$ $\{ 1 , \ldots , M _ { \mathrm { p r o g } } \}$ , each of the form 

$$
(\mathrm{op}, r, \text {   target }, \text {   next }), \quad \mathrm{op} \in \{\text { INC }, \text {   DECJZ } \}, r \in \{1, 2 \}.
$$

We store each instruction in $H _ { 0 }$ at one program position $p ( m ) \in \mathcal { P }$ using fixed channels: 

• an op code indicator (INC vs. DECJZ), 

• a register selector indicator $( r = 1 \mathrm { o r } r = 2 )$ 

• two one-hot masks over ${ \mathcal { P } } ;$ : TargetMask(m) ∈ {0, 1}<sup>|P|</sup> and $\mathrm { N e x t M a s k } ( m ) \in \{ 0 , 1 \} ^ { | \mathcal { P } | }$ 2 embedded as channels aligned with program positions. 

This “mask-in-program” representation avoids explicit arithmetic on program indices and makes jumps purely mask-based. 

Program counter (PC) encoding. We represent the current instruction pointer as a one-hot mask over program slots, stored in a dedicated channel of the writable state: 

$$
\mathrm{PC} _ {t} (i) \in \{0, 1 \}, i \in \mathcal {P}, \quad \sum_ {i \in \mathcal {P}} \mathrm{PC} _ {t} (i) = 1.\tag{42}
$$

Intuitively, exactly one program slot is “active” at each step. 

Register encoding (unary with delimiter). For each register $R _ { r } ~ ( r \in \{ 1 , 2 \} )$ , we use a unary encoding within segment $\textstyle { \mathcal { R } } _ { r }$ : 

$$
R _ {r} = n \quad \Longleftrightarrow \quad [ \underbrace {1 , 1 , \ldots , 1} _ {n \text { ones }} | 0, 0, \ldots ],\tag{43}
$$

where the delimiter $\mathbf { \hat { \mu } } ^ { 6 6 } | \mathbf { \hat { \mu } } ^ { 9 }$ is represented by a one-hot delimiter mask channel 

$$
\operatorname{Delim} _ {t} ^ {(r)} (i) \in \{0, 1 \}, i \in \mathcal {R} _ {r}, \quad \sum_ {i \in \mathcal {R} _ {r}} \operatorname{Delim} _ {t} ^ {(r)} (i) = 1.\tag{44}
$$

The tape also contains a data channel holding the cell value (approximately 0 or 1). The delimiter position marks the boundary between ones and zeros. 

## C.3 PRIMITIVE OPERATIONS REALIZABLE BY ONE FIXED-DEPTH STEP

We now outline how one application of $F _ { \theta }$ can implement the required discrete operations using a fixed-depth composition of Chain-Blocks (state update + SGSA + gated-MLP), under the idealized assumptions. 

## C.3.1 POINTER-DRIVEN INSTRUCTION FETCH UNDER SGSA

Because program slots lie on anchors $( { \mathcal { P } } \subseteq A )$ , SGSA can retrieve from all instruction positions via sparse anchor links. 

Let $u ( i ) \in \mathbb { R } ^ { d }$ denote the instruction embedding stored at program slot $i \in \mathcal { P }$ (as part of $H _ { 0 } )$ . We define an attention head (or an equivalent linear aggregation) that computes an instruction summary 

$$
\operatorname{Instr} _ {t} = \sum_ {i \in \mathcal {P}} \mathrm{PC} _ {t} (i) u (i),\tag{45}
$$

which equals the active instruction embedding since $\mathrm { P C } _ { t }$ is one-hot. 

Implementation note. Equation (45) can be realized by SGSA by setting keys to include the $\mathrm { P C } _ { t } ( i )$ channel (or using it to bias logits) and using a sufficiently sharp selection temperature. Under arbitrary precision, the softmax can approximate hard selection to any desired tolerance; we additionally apply saturated gates to keep $\mathrm { P C } _ { t } ^ { \bar { } }$ binary-stable (see §C.5). 

## C.3.2 ADJACENCY ACCESS VIA SHIFT OPERATORS

We assume access to fixed shift operators along the sequence axis for designated mask channels: 

$$
\operatorname{ShiftRight} (z) (i) = z (i - 1), \quad \operatorname{ShiftLeft} (z) (i) = z (i + 1).\tag{46}
$$

Such operators can be implemented by RWKV-style time shift or an equivalent fixed convolution/Toeplitz linear map. 

These shifts allow each cell to locally detect whether it is the delimiter position or a neighbor of the delimiter, enabling local rule-based updates without global write operations. 

## C.3.3 MASKED OVERWRITE (CONSTANT-SUPPORT WRITEBACK)

A key requirement is to modify only a constant number of tape locations per instruction (in the sense of net change), while leaving the remaining cells unchanged. 

Given a (possibly multi-hot) mask $m ( i ) \in \{ 0 , 1 \}$ supported on a small index set $I ,$ a gated-MLP can implement masked overwrite: 

$$
X ^ {\prime} (i) = X (i) + m (i) \odot \bigl (V (i) - X (i) \bigr),\tag{47}
$$

where $V ( i )$ is the desired writeback value at masked positions and ⊙ denotes elementwise multiplication provided by gating. When m is supported on I, only positions in I change. 

In our construction, $m ( i )$ will be formed from delimiter masks and their shifted versions, so $| I |$ is a constant (e.g., {b} or $\{ b , b \pm 1 \} )$ . 

## C.3.4 ZERO TEST FOR UNARY REGISTER

Let $\textstyle { \mathcal { R } } _ { r }$ be the register-r segment. Let b be the delimiter position, i.e., Delim $\mathsf { l } _ { t } ^ { ( r ) } ( b ) = 1$ . The register is zero iff the delimiter is at the left boundary of the segment, equivalently there is no $" 1 "$ immediately before it inside the segment. 

Using shifts, we can form a pre-delimiter mask: 

$$
\operatorname{PreDelim} _ {t} ^ {(r)} = \operatorname{ShiftLeft} \left(\operatorname{Delim} _ {t} ^ {(r)}\right),\tag{48}
$$

and then read the data channel at that location. A piecewise-linear threshold (hard-sigmoid in the idealized setting) yields a binary flag: 

$$
\mathrm{NonZero} _ {t} ^ {(r)} \in \{0, 1 \}, \qquad \mathrm{FLAG} _ {t} = 1 - \mathrm{NonZero} _ {t} ^ {(r)}.\tag{49}
$$

For DECJZ, $\mathrm { F L A G } _ { t } = 1$ indicates the jump-on-zero branch. 

## C.4 SIMULATING ONE STEP OF A 2CM

We now describe how $F _ { \theta }$ implements the two instruction types. 

## C.4.1 INC

Suppose the fetched instruction is (INC, r, target, next). Let $\mathrm { D e l i m } _ { t } ^ { ( r ) }$ denote the delimiter mask for register r. Define 

$$
m _ {\mathrm{inc}} = \operatorname{Delim} _ {t} ^ {(r)} + \operatorname{ShiftRight} \left(\operatorname{Delim} _ {t} ^ {(r)}\right),\tag{50}
$$

which is supported on $\{ b , b + 1 \}$ . Using (47), we overwrite: 

• at b: write data-value 1 and clear delimiter; 

• at $b + 1 \colon$ set delimiter and keep data-value 0 (or leave as 0). 

All other positions remain unchanged. 

The program counter advances deterministically: 

$$
\mathrm{PC} _ {t + 1} = \text { NextMask } (\text { Instr } _ {t}),\tag{51}
$$

where $\operatorname { N e x t M a s k } ( \operatorname { I n s t r } _ { t } )$ is decoded from the fetched instruction embedding (a one-hot mask over P). 

## C.4.2 DECJZ

Suppose the fetched instruction is (DECJZ, r, target, next). 

First compute the zero-test flag $\mathrm { F L A G } _ { t } \in \{ 0 , 1 \}$ as in §C.3.4. 

• If $\mathrm { F L A G } _ { t } = 1$ (zero case), we leave the register unchanged and jump: 

$$
\mathrm{PC} _ {t + 1} = \text { TargetMask } (\text { Instr } _ {t}).\tag{52}
$$

• If $\mathrm { F L A G } _ { t } = 0$ (nonzero case), we decrement by moving the delimiter left by one: define 

$$
m _ {\mathrm{dec}} = \operatorname{Delim} _ {t} ^ {(r)} + \operatorname{ShiftLeft} \left(\operatorname{Delim} _ {t} ^ {(r)}\right),\tag{53}
$$

supported on $\{ b - 1 , b \}$ , and overwrite: 

– at $b - 1 \colon$ set delimiter; 

– at b: clear delimiter and write data-value 0. 

Then advance: 

$$
\mathrm{PC} _ {t + 1} = \mathrm{NextMask} (\mathrm{Instr} _ {t}).\tag{54}
$$

## C.5 BINARY STABILITY AND ERROR CONTROL

Because the construction relies on near-discrete masks (PC and delimiters), we include a stability mechanism to prevent small numerical leakage from accumulating over arbitrarily many recurrent iterations. 

After each step, we apply a bistable contraction (e.g., hard-sigmoid with thresholds) on the dedicated mask channels: 

$$
\Pi (z) = \left\{ \begin{array}{l l} 0, & z \leq \frac {1}{3}, \\ 3 z - 1, & \frac {1}{3} <   z <   \frac {2}{3}, \\ 1, & z \geq \frac {2}{3}, \end{array} \right.\tag{55}
$$

ensuring that PC and delimiter masks remain within an ε-neighborhood of {0, 1}, and the one-hot constraints remain satisfied up to an arbitrarily small tolerance (guaranteed by the arbitrary-precision assumption). 

## C.6 CORRECTNESS OF THE SIMULATION

Let $\mathrm { S t e p } _ { 2 C M }$ denote the true one-step transition function of the 2CM on configurations encoded as (PC, Delim<sup>(1)</sup>, Delim<sup>(2)</sup>, data). 

Lemma C.1 (Single-step correctness). There exist fixed parameters θ such that for any valid encoded 2CM configuration $X _ { t } ,$ , one application of $F _ { \theta }$ satisfies 

$$
F _ {\theta} (X _ {t}, H _ {0}) = X _ {t + 1} = \operatorname{Encode} \bigl (\operatorname{Step} _ {2 C M} (\operatorname{Decode} (X _ {t})) \bigr),\tag{56}
$$

and the net modification of the tape is supported on a constant-size set ofpositions (those adjacent to the active delimiter), while all other positions are preserved by residual identity paths. 

Instruction fetch (45) selects exactly the active instruction. The INC/DECJZ updates are implemented by masked overwrite (47) with masks supported on $\{ b , b \pm 1 \}$ . PC updates (51)–(54) are mask-based, thus remain one-hot after applying Π(·). Residual connections preserve unmodified positions. 

Lemma C.2 (Multi-step correctness). By induction on t, repeated application of $F _ { \theta }$ produces a sequence $\{ X _ { t } \}$ that matches the 2CM execution trace step-by-step. 

## C.7 FROM 2CM TO TURING MACHINE

It is a classical result that a two-counter Minsky machine can simulate an arbitrary Turing machine (and vice versa). Therefore, for any Turing machine M, there exists a 2CM program P that simulates M. We encode $P$ into the read-only basis $H _ { 0 }$ on the program segment $\mathcal { P }$ , and encode the initial tape/state of M into the writable register segments $\mathcal { R } _ { 1 } , \mathcal { R } _ { 2 }$ (and auxiliary padding U). 

If M halts, the corresponding 2CM reaches a halting configuration. We can represent halting by a dedicated HALT instruction that forces PC into an absorbing slot and/or sets a halting flag channel to 1, after which $F _ { \theta }$ becomes the identity map on the writable state. The output module can then read out the final configuration $( \mathrm { e . g . }$ , via the logits at a designated readout position). 

Combining Lemma C.2 with the 2CM–TM equivalence yields Theorem C.1. 

## D ALGORITHMIC IMPLEMENTATION DETAILS

For clarity, this section provides the core algorithmic implementations of ChainGPT. 

## D.1 RWKV-PRODUCT

Given an input sequence $H = \{ h _ { t } \} _ { t = 1 } ^ { T } \in \mathbb { R } ^ { B \times T \times C }$ and the number of sub-steps M. Let the state matrix sequence be ${ \cal S } = \{ s _ { 1 } , s _ { 2 } , \ldots , s _ { T } \}$ 

Initialization: 

$$
s _ {0} = 0
$$

For $t = 1$ to $T \colon$ : 

$$
\tilde {h} _ {t} = \left\{ \begin{array}{l l} 0, & t = 1 \\ h _ {t - 1}, & t > 1 \end{array} \right.
$$

$$
\Delta h _ {t} = \tilde {h} _ {t} - h _ {t}
$$

$$
x _ {t} ^ {r} = h _ {t} + \Delta h _ {t} \odot x ^ {r}
$$

$$
x _ {t} ^ {w} = h _ {t} + \Delta h _ {t} \odot x ^ {w}
$$

$$
x _ {t} ^ {k} = h _ {t} + \Delta h _ {t} \odot x ^ {k}
$$

$$
x _ {t} ^ {v} = h _ {t} + \Delta h _ {t} \odot x ^ {v}
$$

$$
x _ {t} ^ {\alpha} = h _ {t} + \Delta h _ {t} \odot x ^ {\alpha}
$$

$$
x _ {t} ^ {g} = h _ {t} + \Delta h _ {t} \odot x ^ {g}
$$

$$
\begin{array}{c} {R _ {t} = W _ {r} x _ {t} ^ {r}} \\ {W _ {t} = - \mathrm{softplus} (- (w _ {0} + \tanh (x _ {t} ^ {w} w _ {1}) w _ {2})) - 0. 5} \\ {K _ {t} ^ {0} = W _ {k} x _ {t} ^ {k}} \\ {V _ {t} ^ {0} = W _ {v} x _ {t} ^ {v}} \end{array}
$$

For $j = 0$ to $M - 1$ 

$$
\begin{array}{c} r _ {t, j} = \left\{ \begin{array}{l l} R _ {t}, & j = M - 1 \\ 0, & \text {otherwise} \end{array} \right. \\ w _ {t, j} = \left\{ \begin{array}{l l} W _ {t}, & j = 0 \\ - \infty , & \text {otherwise} \end{array} \right. \\ \Delta K _ {t, j} ^ {b} = x _ {t} ^ {k} A _ {j} ^ {b} B _ {j} ^ {b} \\ \Delta K _ {t, j} ^ {c} = x _ {t} ^ {k} A _ {j} ^ {c} B _ {j} ^ {c} \\ K _ {t, j} ^ {b} = K _ {t} ^ {0} + \Delta K _ {t, j} ^ {b} \\ K _ {t, j} ^ {c} = K _ {t} ^ {0} + \Delta K _ {t, j} ^ {c} \\ \beta_ {t, j} ^ {b} = \sigma (\alpha_ {0, j} ^ {b} + (x _ {t} ^ {\alpha} \alpha_ {1, j} ^ {b}) \alpha_ {2, j} ^ {b}) \\ \beta_ {t, j} ^ {c} = \sigma (\alpha_ {0, j} ^ {c} + (x _ {t} ^ {\alpha} \alpha_ {1, j} ^ {c}) \alpha_ {2, j} ^ {c}) \\ \hat {K} _ {t, j} ^ {c} = K _ {t, j} ^ {c} \odot [ 1 + (\beta_ {t, j} ^ {c} - 1) \cdot k ^ {a c} ] \\ b _ {t, j} ^ {-} = - \frac {K _ {t , j} ^ {b}}{\| K _ {t , j} ^ {b} \|} \\ b _ {t, j} ^ {+} = \frac {K _ {t , j} ^ {b}}{\| K _ {t , j} ^ {b} \|} \odot \beta_ {t, j} ^ {b} \\ V _ {t, j} = V _ {t} ^ {0} + x _ {t} ^ {v} A _ {j - 1} ^ {v} B _ {j - 1} ^ {v} \end{array}
$$

Call the CUDA implementation: 

$$
\begin{array}{c} X _ {\text {inter}} = \text {RUN\_CUDA\_RWKV7g} (r, w, \hat {k} ^ {c}, v, b ^ {-}, b ^ {+}, H _ {\text {head}}) \\ X _ {\text {att}} = \text {reshape} (X _ {\text {inter}}, B, T, M, C) [:,:, - 1,: ] \\ X _ {\text {gn}} = \text {GroupNorm} (X _ {\text {att}}) \\ \text {term2} _ {t} = \sum_ {h = 1} ^ {H} (r _ {t, M - 1} ^ {h} \cdot k _ {t, M - 1} ^ {h} \cdot r _ {k} ^ {h}) v _ {t, M - 1} ^ {h} \\ X ^ {\prime} = X _ {\text {gn}} + \text {term2} _ {t} \\ g _ {t} = \sigma (x _ {t} ^ {g} g _ {1}) g _ {2} \\ s _ {t} = W _ {o} (X ^ {\prime} \odot g _ {t}) \end{array}
$$

## D.2 STATE-GUIDED SPARSE ATTENTION

Given a state sequence $X \in \mathbb { R } ^ { B \times S \times C }$ , window size W, and global interval G. 

$$
Q = X W _ {q}, \quad K = X W _ {k}, \quad V = X W _ {v}
$$

$$
Q, K, V \in \mathbb {R} ^ {B \times H \times S \times D}
$$

For $i = 0$ to $d / 2 - 1$ 

$$
\theta_ {t} ^ {(i)} = t / 1 0 0 0 0 ^ {2 i / d}
$$

$$
f = i / (d / 2)
$$

$$
\beta (f) = \beta_ {\mathrm{slow}} + (\beta_ {\mathrm{fast}} - \beta_ {\mathrm{slow}}) f
$$

$$
n _ {s} = 1 + \frac {\alpha - 1}{\beta (f)}
$$

$$
\tilde {\theta} _ {t} ^ {(i)} = \theta_ {t} ^ {(i)} / n _ {s}
$$

Apply $\tilde { \theta } _ { t } ^ { ( i ) }$ to rotate-encode $q _ { 2 i } , q _ { 2 i + 1 }$ and $k _ { 2 i } , k _ { 2 i + 1 }$ 

$$
Q ^ {\prime} = \operatorname{RoPE} (Q), \quad K ^ {\prime} = \operatorname{RoPE} (K)
$$

Construct a sparse mask $M _ { i j }$ for $i , j = 1$ to S: 

$$
M _ {i j} = \left\{ \begin{array}{l l} 0, & i - j <   W \text {   and   } j \leq i \\ 0, & j \bmod G = G - 1 \text {   and   } j \leq i \\ - \infty , & \text { otherwise } \end{array} \right.
$$

If num kv heads $< H$ , expand $K ^ { \prime } , V ^ { \prime }$ to match the head dimension. 

$$
\text { Attention } (Q ^ {\prime}, K ^ {\prime}, V ^ {\prime}; M) = \text { softmax } \left(\frac {Q ^ {\prime} K ^ {\top}}{\sqrt {d}} + M\right) V ^ {\prime}
$$

$$
\text { Out } = \text { concat   heads } \cdot W _ {o}
$$

## D.3 RECURRENT DEPTH APPROACH

Let the input sequence be $\boldsymbol { X } \in \mathbb { R } ^ { B \times S }$ , and the hidden dimension be d. 

Embedding and Bottom-Level: 

$$
H = \operatorname{Embed} (X) \in \mathbb {R} ^ {B \times S \times d}, \qquad H _ {0} = \operatorname{Bottom} (H) \in \mathbb {R} ^ {B \times S \times d}.
$$

Looped Iterations: 

$$
Z _ {t} = \left[ X _ {t}; H _ {0} \right] \in \mathbb {R} ^ {B \times S \times 2 d}, \qquad \tilde {X} _ {t} = Z _ {t} W _ {a} + b _ {a} \in \mathbb {R} ^ {B \times S \times d}, \qquad X _ {t + 1} = \mathcal {R} (\tilde {X} _ {t}).
$$

Two-Phase (no-gradient $n _ { 0 }$ steps + backprop $n _ { k }$ steps) with truncation: 

(No-Grad Phase) 

$$
X _ {t + 1} = \mathcal {R} \big ([ X _ {t}; H _ {0} ] W _ {a} + b _ {a} \big), \qquad t = 0, \ldots , n _ {0} - 1,
$$

(Truncation Boundary) 

$$
\bar {X} _ {0} = \operatorname{stopgrad} \left(X _ {n _ {0}}\right),
$$

(Backprop Phase) 

$$
\bar {X} _ {k + 1} = \mathcal {C} \Big (\mathcal {R} \big ([ \bar {X} _ {k}; H _ {0} ] W _ {a} + b _ {a} \big) \Big), \qquad k = 0, \ldots , n _ {k} - 1,
$$

(Output) 

$$
X ^ {\star} = \bar {X} _ {n _ {k}}.
$$

Top Layers and Normalization: 

$$
X ^ {\mathrm{top}} = \operatorname{Top} (X _ {T}), \qquad Y = \operatorname{Norm} (X ^ {\mathrm{top}}).
$$

Adaptive Inner-Loop Reasoning: After the t-th iteration, obtain the hidden state 

$$
x _ {t} \in \mathbb {R} ^ {B \times S \times d}.
$$

Decode logits by running only the top blocks: 

$$
\operatorname{logits} _ {t} = \operatorname{Decode} (x _ {t}) \in \mathbb {R} ^ {B \times S \times V},
$$

where V is the vocabulary size. 

Extract the distribution at the last position: 

$$
\ell_ {t} = \operatorname{logits} _ {t} [:, - 1,: ] \in \mathbb {R} ^ {B \times V}.
$$

Convert to probabilities: 

$$
p _ {t} = \mathrm{softmax} (\ell_ {t}) \in \mathbb {R} ^ {B \times V}.
$$

Compute the entropy for each batch element b: 

$$
H _ {t} (b) = - \sum_ {i = 1} ^ {V} p _ {t, i} (b) \log \left(p _ {t, i} (b) + \varepsilon\right), \qquad \varepsilon = 1 0 ^ {- 1 0}.
$$

Define the entropy difference: 

$$
\Delta H _ {t} (b) = H _ {t - k} (b) - H _ {t} (b),
$$

where k is the checking interval. 

Early-Stopping Criterion (entropy-diff): 

$$
\Delta H _ {t} (b) \leq \tau \quad \Longrightarrow \quad \text { terminate   early   at   step } t,
$$

where τ is a threshold, either a constant or an adaptive function $( \mathbf { e } . \mathbf { g } . , \tau ( t ) = c / \sqrt { t } )$ 

## E COMPARISON OF RECENT RNN MODULES FOR LANGUAGE MODELING (TABLE 6)

## F EXPERIMENTAL REPRODUCTION GUIDE

The code are provided in the supplementary materials. For convenience, Table 7 lists the main hyperparameters for ChainGPT models at different scales. 

For MQAR task reproduction, a wide-range time convolution needs to be applied at the RWKV-Product layer to ensure stability. This is an optimization for training stability and does not affect related conclusions. The experimental setup follows Zoology (Arora et al., 2024). 

You can calculate the total number of model parameters using the following method. Let V be the vocabulary size, D the hidden dimension, L the number of decoder layers, with $L _ { e } = \lceil L / 2 \rceil$ even-indexed RWKV-v7 layers and $L _ { o } = \lfloor L / 2 \rfloor$ odd-indexed GQA layers. Let $D _ { m }$ be the MLP intermediate size, M the micro-step count, r the LoRA rank, $D _ { \mathrm { D E C } } , D _ { \mathrm { G A T E } } , D _ { \mathrm { A A A } }$ the low-rank bottlenecks, and $H _ { q } , H _ { k v } , d _ { h }$ the query heads, key/value heads, and per-head dimension, with $K =$ $H _ { k v } d _ { h }$ . Non-layer terms are: 

$$
P _ {\text { embed }} = V D, \quad P _ {\text { lm }} = D V, \quad P _ {\text { adapter }} = 2 D ^ {2} + D, \quad P _ {\text { core - norm }} = 2 D,
$$

and per-layer shared components: 

$$
P _ {\mathrm{2LN}} = 4 D, \qquad P _ {\mathrm{MLP}} = 3 D D _ {m}.
$$


Table 6: Comparison of recent RNN modules used for language modeling


<table><tr><td>Architecture</td><td>State Evolution</td><td>FP</td><td>DP</td><td>HRU</td><td>GE</td></tr><tr><td>RWKV-4</td><td><eq>s_{t}=e^{-w}\odot s_{t-1}+e^{k_{t}}\odot v_{t}</eq></td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>RetNet</td><td><eq>S_{t}=wS_{t-1}+v_{t}^{T}k_{t}</eq></td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>RWKV-5</td><td><eq>s_{t}^{\prime}=e^{-w}\odot s_{t-1}^{\prime}+e^{k_{t}}</eq></td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Mamba</td><td><eq>S_{t}=S_{t-1}\text{diag}(w)+v_{t}^{T}k_{t}</eq></td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>RWKV-6 &amp; GLA</td><td><eq>S_{t}=S_{t-1}\odot\exp(-(w_{t}^{T}\mathbf{1})\odot\exp(A))+(w_{t}\odot v_{t})^{T}k_{t}</eq></td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>HGRN-2</td><td><eq>S_{t}=S_{t-1}\text{diag}(w_{t})+v_{t}^{T}k_{t}</eq></td><td>√</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Mamba-2</td><td><eq>S_{t}=S_{t-1}\text{diag}(w_{t})+v_{t}^{T}(1-w_{t})</eq></td><td>×</td><td>×</td><td>×</td><td>√</td></tr><tr><td>TTT<eq>^{a}</eq></td><td><eq>S_{t}=w_{t}S_{t-1}+v_{t}^{T}k_{t}</eq></td><td>×</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Gated DeltaNet</td><td><eq>S_{t}=S_{t-1}\odot(I-a_{t}^{T}k_{t}^{2})+(a_{t}x_{t})^{T}k_{t}</eq></td><td>×</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Titans<eq>^{a}</eq></td><td><eq>S_{t}=w_{t}S_{t-1}(I-a_{t}k_{t}^{T}k_{t})+a_{t}v_{t}^{T}k_{t}</eq><eq>M_{t}=(1-\alpha_{t})M_{t-1}+S_{t}</eq><eq>S_{t}=w_{t}S_{t-1}-a_{t}\nabla l(M_{t-1},k_{t},v_{t})</eq></td><td>×</td><td>×</td><td>×</td><td>√</td></tr><tr><td>RWKV-7</td><td><eq>S_{t}=S_{t-1}(\text{diag}(w_{t})-\hat{\kappa}_{t}^{T}(\alpha_{t}\odot\hat{\kappa}_{t}))+v_{t}^{T}k_{t}</eq></td><td>√</td><td>×</td><td>×</td><td>√</td></tr><tr><td>DeltaProduct</td><td><eq>H_{i}=\left(\prod_{j=1}^{n_{h}}(I-\beta_{i,j}k_{i,j}k_{i,j}^{T})\right)H_{i-1}+\sum_{j=1}^{n_{h}}\left(\prod_{k=j+1}^{n_{h}}(I-\beta_{i,k}k_{i,k}k_{i,k}^{T})\right)\beta_{i,j}k_{i,j}v_{i,j}^{T}</eq></td><td>×</td><td>×</td><td>√</td><td>√</td></tr><tr><td>RWKV-Product (ours)</td><td><eq>S_{t}=A(x_{t})S_{t-1}+B(x_{t})</eq>, where<eq>A(x_{t})=\prod_{j=0}^{M-1}(\text{diag}(a_{t,j})-\beta_{b,j}k_{t,j}^{(b)}k_{t,j}^{(b)T})</eq><eq>B(x_{t})=\sum_{j=0}^{M-1}\left(\prod_{k=j+1}^{M-1}A_{t,k}\right)(\beta_{c,j}k_{t,j}^{(c)}v_{t,j}^{T})</eq></td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

FP (Full-dim Params): all parameters match the model dimensionality, enabling adaptive channel-wise modulation. 

DP (Decoupled Path): parameters for state transition and information injection are linearly independent, improving expressiveness and convergence. 

HRU (High-Rank Update): state updates possess a multi-substep high-rank structure, allowing a single-step update to capture complex, diverse state changes. 

GE (Generalized Eigenvalue): the state update matrix admits a generalized eigenvalue factorization, enabling efficient diagonal-form updates. 


Table 7: Main hyperparameters for different model scales


<table><tr><td>Parameter</td><td>180M</td><td>0.5B</td><td>1.5B</td></tr><tr><td>RWKV head dim</td><td>64</td><td>64</td><td>64</td></tr><tr><td>GQA head dim</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Hidden dim</td><td>1024</td><td>2048</td><td>2560</td></tr><tr><td>KV heads</td><td>2</td><td>2</td><td>8</td></tr><tr><td>Layers</td><td>8</td><td>8</td><td>12</td></tr><tr><td>A_low_rank_dim</td><td>64</td><td>96</td><td>128</td></tr><tr><td>Decay_low_rank_dim</td><td>64</td><td>96</td><td>128</td></tr><tr><td>Gate_low_rank_dim</td><td>128</td><td>256</td><td>320</td></tr><tr><td>M_rank_dim</td><td>96</td><td>128</td><td>256</td></tr><tr><td>Sub-steps</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Recursive iters</td><td>12</td><td>16</td><td>16</td></tr><tr><td>Grad-enabled iters</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Layer division strategies</td><td>C121</td><td>C121</td><td>C141</td></tr><tr><td>Window size</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Anchor interval</td><td>64</td><td>64</td><td>64</td></tr></table>

RWKV-v7 attention parameters per layer are 

$$
P _ {\mathrm{RWKV}} = 4 D ^ {2} + 2 D (D _ {\mathrm{DEC}} + D _ {\mathrm{GATE}}) + 6 M D r + 4 M D D _ {\mathrm{AAA}} + (2 M + 1 1) D,
$$

while GQA attention parameters per layer are 

$$
P _ {\mathrm{GQA}} = 2 D \left(H _ {q} d _ {h}\right) + 2 D K + 2 d _ {h} \quad (\text { if } H _ {q} d _ {h} = D, P _ {\mathrm{GQA}} = 2 D ^ {2} + 2 D K + 2 d _ {h}).
$$

The total trainable parameter count is therefore 

$$
\left| P _ {\text {total}} = 2 V D + \left(2 D ^ {2} + D\right) + 2 D + L \left(3 D D _ {m} + 4 D\right) + L _ {e} P _ {\mathrm{RWKV}} + L _ {o} P _ {\mathrm{GQA}} \right|.
$$

With weight tying, replace 2VD by VD and adjust if biases are added elsewhere. 

## G LLM USAGE STATEMENT

In this work, LLMs were utilized exclusively for improving the clarity of the manuscript and enhancing the code, and were not applied to any other critical components. 