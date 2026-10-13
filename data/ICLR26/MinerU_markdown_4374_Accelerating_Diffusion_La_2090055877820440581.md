# ACCELERATING DIFFUSION LARGE LANGUAGE MOD-ELS WITH SLOWFAST SAMPLING: THE THREE GOLDEN PRINCIPLES

Qingyan Wei<sup>1</sup> Yaojie Zhang<sup>1</sup> Zhiyuan Liu<sup>1</sup> Puyu Zeng<sup>1</sup> Yuxuan Wang<sup>1</sup> Biqing Qi<sup>2</sup> Dongrui Liu<sup>2</sup> Linfeng Zhang<sup>1†</sup> 

<sup>1</sup>EPIC Lab, Shanghai Jiao Tong University <sup>2</sup>Shanghai Artificial Intelligence Laboratory 

Code: https://github.com/LiangrunFlora/Slow-Fast-Sampling 

## ABSTRACT

Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autoregressive LLMs by enabling parallel token generation and significantly reducing inference latency. However, existing sampling strategies for dLLMs, such as confidence-based or semi-autoregressive decoding, often suffer from static behavior, leading to suboptimal efficiency and limited flexibility. In this paper, we propose SlowFast Sampling, a novel dynamic sampling strategy that adaptively alternates between exploratory and accelerated decoding stages. Our method is guided by three golden principles: certainty principle, convergence principle, and positional principle, which govern when and where tokens can be confidently and efficiently decoded. We further integrate our strategy with dLLM-Cache to reduce redundant computation. Extensive experiments across various benchmarks demonstrate the efficiency of our method. Specifically, on the GPQA benchmark, SlowFast Sampling achieves up to 15.63× speedup on LLaDA with minimal accuracy drop, and up to 34.22× when combined with caching. Notably, our approach outperforms strong autoregressive baselines like LLaMA3 8B in throughput, demonstrating that well-designed sampling can unlock the full potential of dLLMs for fast and high-quality generation. 

## 1 INTRODUCTION

Large Language Models (LLMs) (Zhao et al., 2025) have rapidly become cornerstone technologies in artificial intelligence, demonstrating remarkable capabilities across a diverse range of natural language understanding and generation tasks. However, the prevalent autoregressive nature of most LLMs, where tokens are generated sequentially one after another, introduces significant inference latency, particularly for long sequences. To address this inherent bottleneck, diffusion-based LLMs (dLLMs) (Ye et al., 2025; Nie et al., 2025b) have emerged as a promising alternative paradigm. These models are capable of generating multiple tokens in parallel, departing from the strict token-by-token process. This parallel decoding capability offers the distinct advantage of potentially accelerating text generation significantly, positioning dLLMs as a compelling and forward-looking direction for efficient language model inference. 

However, current ways of sampling with dLLMs often don’t perform as well as they could. Common methods include confidence-based selection (Chang et al., 2022) like Fast-dLLM (Wu et al., 2025), where tokens exceeding a confidence threshold are selected for decoding. Another popular method, semi-autoregressive decoding (Arriola et al., 2025), divides the sequence into fixed blocks and decodes within them. Unfortunately, these methods frequently yield unsatisfactory results (i.e., significant accuracy drop when decoding many tokens in parallel), and are characterized by a static, constant sampling speed throughout the generation process. This lack of flexibility highlights the need for a more dynamic sampling approach: one that can smartly decide how many tokens to sample at each step and where these tokens should be located in the sequence. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/e23c785dcda3054922ad1ed2815fdbd0a440d102a8686664f27d6b4a0d42aa2f.jpg)



Decoding Step


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/6268b3149d5aeb9e438a44b781f57eb88e406f94e6cf898ac2fdcd1625d536fd.jpg)



(a) Token Confidence Converges during Decoding



Decoding Step


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/1f746b4dceba767179d2f8e6dda8f36aa60486220497cdcbf8260d501d1b5fa0.jpg)



(b) Token Confidence during Decoding



Figure 1: The Three Golden Principles for sampling in diffusion LLMs. (a) Convergence Principle: As decoding proceeds, the confidence values of tokens largely converge to high values, while a few tokens converge to lower values. (b) The confidence map over 256 diffusion steps: Highconfidence tokens (in deep red) emerge progressively and are preferentially decoded (the Certainty Principle), while selection tends to cluster in contiguous regions (the Positional Principle), enabling cache reuse and acceleration.


Motivated by these limitations, we introduce a novel dynamic sampling approach designed to accelerate dLLMs, aiming to unlock the real potential of dLLMs under high-parallel decoding. As illustrated in Figure 1, our method is guided by three core observations, which we formulate as the Three Golden Principles for effective acceleration: 

• The Certainty Principle: Tokens exhibiting higher confidence are inherently more determined. Consequently, they are more likely to be decoded correctly early in the process and require less adjustment in subsequent diffusion steps. 

• The Convergence Principle: As the diffusion process unfolds and tokens are progressively refined, the semantic meaning of many tokens stabilizes, and their associated confidence scores converge towards a steady value. This convergence indicates that these tokens have largely settled into their final form and require minimal further refinement. 

• The Positional Principle: We observe that even without explicit constraints, the model’s sampling preferences often gravitate towards tokens in specific, frequently neighboring or clustered, positions. This inherent positional bias can be strategically exploited. For instance, parts of the sequence can be effectively cached, leading to significant acceleration gains. 

Integrating these principles, we propose SlowFast Sampling with two phases: an Exploratory Stage and an Accelerated Decoding Stage. In the exploratory stage, the model loosely decodes to locate spans with emerging certainty and convergence. The accelerated stage then parallelly decodes these high-certainty tokens, reducing effort on already determined parts. This division yields significant speedups, reaching 15.63× on LLaDA and up to 34.22× when combined with dLLM-Cache (Liu et al., 2025) on the GPQA benchmark, with minimal accuracy loss. Our contributions are threefold: 

1. We propose three golden principles based on token certainty, convergence, and positional influence, which critically govern effective and efficient sampling in dLLMs. 

2. Building on these principles, we introduce SlowFast Sampling, a novel two-stage dynamic strategy specifically designed to leverage these principles for optimal acceleration of dLLM. 

3. Through experiments on various benchmarks, we demonstrate that SlowFast Sampling achieves significant inference acceleration (e.g., up to 15.63× on LLaDA with SlowFast Sampling alone, and up to 34.22× when combined with dLLM-Cache on the GPQA dataset) without compromising response quality, thereby offering a superior speed-quality trade-off compared to baseline and simpler sampling methods. 

## 2 RELATED WORK

## 2.1 DIFFUSION MODELS FOR LANGUAGE

Diffusion Models (DMs) (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021) have revolutionized generative modeling, particularly in continuous domains like images (Rombach et al., 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/962fcce368c0a0781f0e50ff0e137f141ab9b1ec2e6f72af03d50ee868d56965.jpg)



Figure 2: Throughput and accuracy comparison on GPQA (8-shot, Length=1024) on LLaDA with our method, including (1) vanilla decoding, (2) SlowFast Sampling, and (3) SlowFast Sampling further enhanced by dLLM-Cache. Compared to the vanilla setting, SlowFast Sampling alone achieves a 15.63× speedup while maintaining comparable accuracy. With dLLM-Cache, throughput improves further to 54.75 tokens/sec (up to 34.22× speedup), with only minor drops in accuracy.


2022; Peebles & Xie, 2023; Ma et al., 2025b). However, adapting these models to discrete data such as text presents unique challenges due to its discrete nature. A promising approach in discrete diffusion models involves Masked Diffusion Models (MDMs) (Austin et al., 2021; Lou et al., 2023; Shi et al., 2024; Nie et al., 2025a;b; Hoogeboom et al., 2021; Campbell et al., 2022), which iteratively predict masked tokens based on their context. These advancements have transformed text generation, offering a compelling alternative to autoregressive paradigms in LLMs. Notable examples include LLaDA (Nie et al., 2025b), an 8B MDM trained from scratch with a bidirectional Transformer, and Dream (Ye et al., 2025), which initializes from pre-trained ARM weights. Both models demonstrate performance comparable to similarly-sized ARMs like LLaMA3 8B (Dubey et al., 2024). Their bidirectional architecture may overcome ARM limitations such as the reversal curse (Berglund et al., 2023), making diffusion a competitive alternative for foundational LLMs. 

## 2.2 ACCELERATION METHODS FOR DIFFUSION-BASED LLMS

The high inference latency of dLLMs, primarily due to their iterative denoising process (Nie et al., 2025b; Ye et al., 2025), has spurred research into acceleration techniques. Various strategies have been developed, mainly including caching mechanisms and advanced sampling techniques. 

Caching Mechanisms. Feature caching reduces redundant computations by reusing intermediate features. dLLM-Cache (Liu et al., 2025) combines long-interval prompt and short-interval response caching with a V-verify mechanism for faster inference. Sparse-dLLM (Song et al., 2025) applies dynamic cache eviction with sparse attention, retaining only salient tokens to cut memory and boost speed. dKV-Cache (Ma et al., 2025a) adopts delayed KV caching to reuse decoded tokens’ representations, achieving 2–10× acceleration. 

Advanced Sampling Techniques. Optimizing the sampling process itself is another major direction for accelerating dLLMs. Low-confidence remasking (Chang et al., 2022; Nie et al., 2025b) prioritizes high-confidence tokens to speed up convergence; semi-autoregressive (Nie et al., 2025b; Arriola et al., 2025) remasking divides sequences into blocks, applying random and low-confidence strategies. Additionally, GtR (Yan et al., 2026) accelerates MAR models through a hierarchical structure-to-detail sampling strategy, while exact simulation methods for MDMs like the first-hitting sampler (Zheng et al., 2024) have made progress in reducing sampling steps or enhancing per-step efficiency. 

## 3 METHODOLOGY

## 3.1 PRELIMINARY

Inference Process of Diffusion Large Language Models. Diffusion Large Language Models (dLLMs) generate text $\mathbf { y } = ( y _ { 1 } , \dots , y _ { L } )$ from a prompt $\mathbf { c } = ( c _ { 1 } , \hdots , c _ { M } )$ through an iterative denoising process. The model refines an intermediate state $\mathbf { y } ^ { ( k ) } \in \mathcal { T } ^ { L }$ over N discrete steps, from $k = N { \mathrm { ~ t o ~ } } k = 0$ , where T is the token vocabulary. The process begins with a fully masked sequence: 

$$
\mathbf {y} ^ {(N)} = \big (\underbrace {[ \mathrm{MASK} ] , \ldots , [ \mathrm{MASK} ]} _ {L \text {times}} \big)\tag{1}
$$

where [MASK] $\in \mathcal { T }$ is the special mask token. 

At each step $k \in \{ N , N - 1 , \ldots , 1 \}$ , a mask predictor $p _ { \theta }$ estimates the original clean sequence $\mathbf { r } _ { 0 } = ( r _ { 0 , 1 } , \hdots , r _ { 0 , L } )$ from the noisy state $\mathbf { y } ^ { ( k ) }$ and prompt c: 

$$
P _ {\theta} (\mathbf {r} _ {0} | \mathbf {c}, \mathbf {y} ^ {(k)})\tag{2}
$$

An estimate of the clean sequence at step $k , \hat { \mathbf { r } } _ { 0 } ^ { ( k ) }$ , is obtained through greedy decoding: 

$$
\hat {r} _ {0, i} ^ {(k)} = \underset {v \in \mathcal {T}} {\arg \max} P _ {\theta} (r _ {0, i} = v | \mathbf {c}, \mathbf {y} ^ {(k)}) \quad \forall i \in \{1, \ldots , L \}\tag{3}
$$

Although the predictor $p _ { \theta }$ can decode all masked tokens [MASK] in one step, to ensure high-quality generation, dLLM adopts a multi-step decoding process. At each step, the remask strategy refines the tokens. The transition to the next state $\mathbf { y } ^ { ( k - 1 ) }$ is governed by a sampling strategy S: 

$$
\mathbf {y} ^ {(k - 1)} = S (\hat {\mathbf {r}} _ {0} ^ {(k)}, \mathbf {y} ^ {(k)}, k)\tag{4}
$$

This iterative denoising process is a sampling procedure. Our work aims to improve the sampling efficiency of dLLM inference, which can be computationally expensive due to the N sequential steps. Using LLaDA (Nie et al., 2025b) as an example, we describe its core sampling strategies. In LLaDA, time steps are defined as $t _ { k } = k / N$ and $t _ { k - 1 } = ( k - 1 ) / N$ 

LLaDA explores various strategies for the transition function S in Equation 4, which differ mainly in how tokens from $\hat { \mathbf { r } } _ { 0 } ^ { ( k ) }$ update $\mathbf { y } ^ { ( k ) }$ to form $\mathbf { y } ^ { ( k - 1 ) }$ , especially for positions that were [MASK] in $\mathbf { y } ^ { ( k ) }$ 

Random Remasking. In this strategy, for each position i: 0 

If $y _ { i } ^ { ( k ) } \neq$ [MASK], then $y _ { i } ^ { ( k - 1 ) } = y _ { i } ^ { ( \bar { k } ) }$ (known tokens are preserved). I $\dot { \cdot } y _ { i } ^ { ( k ) } = \left[ \mathrm { M A S K } \right]$ , then $y _ { i } ^ { ( k - 1 ) }$ is set to $\hat { r } _ { 0 , i } ^ { ( k ) }$ with probability $1 - { \frac { k - 1 } { k } }$ , and remains [MASK] with probability $\frac { k - 1 } { k }$ , ensuring the expected number of masked tokens aligns with the noise schedule. 

Low-Confidence Remasking. This deterministic strategy aims to improve sample quality by selectively unmasking tokens. For each position i, if $y _ { i } ^ { ( k ) } = \mathsf { [ M A S K ] }$ , the model predicts $\hat { r } _ { 0 , i } ^ { ( k ) }$ and computes its confidence. Specifically, the confidence $c _ { i }$ is given by: 

$$
c _ {i} = P _ {\theta} (\hat {r} _ {0, i} ^ {(k)} | \mathbf {c}, \mathbf {y} ^ {(k)}).\tag{5}
$$

If $y _ { i } ^ { ( k ) } \neq \mathrm { [ M A S K ] }$ , then $c _ { i } = 1$ 

The target number of unmasked tokens for state $\mathbf { y } ^ { ( k - 1 ) }$ is given by: 

$$
n _ {u n} = \left\lfloor L (1 - t _ {k - 1}) \right\rfloor = \left\lfloor L \left(1 - \frac {k - 1}{N}\right) \right\rfloor .\tag{6}
$$

The tokens corresponding to the $n _ { u n }$ highest confidences are unmasked in $\mathbf { y } ^ { ( k - 1 ) }$ , while the remaining positions are set to [MASK]. 

Semi-Autoregressive Remasking. This strategy is used in LLaDA after Supervised Fine-Tuning. The sequence is divided into blocks, and generation proceeds block by block from left to right. Within each block, the random remasking or the low-confidence remasking strategy is applied iteratively to denoise the block before moving to the next. 

The choice of sampling strategy S, along with the number of steps N, significantly affects both the generation quality and latency. 

## 3.2 THREE GOLDEN PRINCIPLES OF DLLMS

Building upon the iterative denoising process outlined in Section 3.1, our empirical analysis of dLLM behavior, particularly with models like LLaDA, has revealed consistent patterns in token generation. These patterns, which we term the Three Golden Principles, form the bedrock of our proposed acceleration strategy. They describe how token certainty, convergence, and positional effects interact during the diffusion process, offering key insights into optimizing sampling. 

The Certainty Principle: High Confidence Indicates Determination. We observe that tokens predicted with high confidence by the mask predictor $p _ { \theta }$ are significantly more likely to be part of the final, correct sequence and tend to remain unchanged in subsequent denoising steps. The confidence for a predicted token $\hat { r } _ { 0 , i } ^ { ( k ) }$ at position i and step k is given by $P _ { \theta } ( \hat { r } _ { 0 , i } ^ { ( k ) } | \mathbf { c } , \mathbf { y } ^ { ( k ) } )$ . As illustrated in Figure 1 (b), at any given step $k ,$ a subset of tokens typically exhibits substantially higher confidence scores compared to others. These high-confidence tokens are prime candidates for early "acceptance" or less frequent resampling. 

Implicationfor Acceleration: By prioritizing tokens that quickly reach a high confidence threshold, we can reduce redundant computations on already determined parts of the sequence. 

The Convergence Principle: Tokens Stabilize Over Iterations. During the iterative refinement process, individual tokens (both their predicted identity $\hat { r } _ { 0 , i } ^ { ( k ) }$ and their confidence $P _ { \theta } ( \hat { r } _ { 0 , i } ^ { ( k ) } | \mathbf { c } , \mathbf { y } ^ { ( k ) } ) )$ undergo a period of fluctuation before settling. As shown in Figure 1 (a), a representative token might initially change its predicted identity and confidence across several early diffusion steps. However, as k decreases, the token’s confidence converges are often to a stable value. This convergence signals that the model has formed a consistent belief about the token’s identity within its current context. 

Implicationfor Acceleration: Tokens that have demonstrated convergence (i.e., stable identity and confidence over a window of recent steps) are less likely to change. Aggressively decoding can prevent unnecessary re-evaluation, thereby speeding up the process. 

The Positional Principle: Decoding Exhibits Regional Preferences. Beyond individual token behaviors, we find that the generation process often exhibits spatial patterns. High-confidence and early-converging tokens do not appear randomly scattered throughout the sequence. Instead, they frequently emerge in contiguous blocks or localized regions, as suggested by Figure 1 (b). This phenomenon might be due to local semantic dependencies or the influence of strongly contextualized parts of the prompt c. For example, after a few initial steps, a particular span of tokens might collectively achieve high confidence and stability, while other regions remain largely masked or uncertain. The model appears to focus its decoding efforts on specific segments at different stages. 

Implicationfor Acceleration: Recognizing these regions allows for targeted decoding. Instead of uniformly processing all tokens, computational resources can be concentrated on regions that are currently most amenable to decoding. 

These three principles collectively indicate that a one-size-fits-all, static sampling strategy is inherently inefficient. They motivate a dynamic approach where the number of tokens sampled, their selection criteria, and their locations are adapted based on the evolving state of the generated sequence. Our SlowFast Sampling methodology, detailed in Section 3.3, is designed to explicitly leverage these observations to achieve significant inference speedups. 

## 3.3 SLOWFAST SAMPLING

1. Exploratory Stage (Slow Phase): Identifying the Next Stable Region. As illustrated in Figure 3, the primary goal of this stage is to cautiously advance decoding while identifying a promising, stable region for subsequent rapid processing. Starting from $\begin{array} { r } { s _ { c y c l \epsilon } } \end{array}$ and extending to the end of the full sequence $L ,$ this stage operates as follows for a limited number of dLLM steps: 

• Cautious Decoding: At each dLLM step k within this stage, we perform a conservative decoding update. We select the $\mathrm { t o p } { - } k _ { s l o w }$ tokens within the current exploratory window $[ s _ { c y c l e } , L ]$ that exhibit the highest confidence $P _ { \theta } ( \hat { r } _ { 0 . i } ^ { ( k ) } | \mathbf { c } , \mathbf { y } ^ { ( k ) } )$ , and these are unmasked to form part of $\hat { \mathbf { r } } _ { 0 } ^ { ( k ) }$ (as per Equation 3 for these tokens, followed by Equation 4). 

• End Point of Convergence Prediction: Concurrently, the model predicts a end point of candidate convergence, $e _ { c a n d } ^ { ( k ) } .$ This is defined as the furthest token position $i \in [ s _ { c y c l e } , L ]$ for which the predicted confidence $P _ { \theta } ( \hat { r } _ { 0 , i } ^ { ( k ) } | \mathbf { c } , \mathbf { y } ^ { ( k ) } )$ ) exceeds a minimum confidence threshold $\tau _ { m i n \_ c o n f }$ Mathematically: 

$$
e _ {c a n d} ^ {(k)} = \max \{i \mid i \in [ s _ {c y c l e}, L ] \wedge P _ {\theta} (\hat {r} _ {0, i} ^ {(k)} | \mathbf {c}, \mathbf {y} ^ {(k)}) > \tau_ {m i n \_ c o n f} \}\tag{7}
$$

This $e _ { c a n d } ^ { ( k ) }$ represents an estimate of how far into the sequence the model can confidently decode at the current step k. 

• Stability Check: Throughout the Exploratory Stage, candidate convergence horizons $e _ { c a n d } ^ { ( j ) } { \mathrm { p r e } } .$ dicted at each dLLM step j are tracked. Let $H _ { W } ( k _ { s } ) = \{ e _ { c a n d } ^ { ( l ) } \mid l \in [ \operatorname* { m a x } ( 1 , k _ { s } - W _ { h i s t } + 1 ) , k _ { s } ] \}$ be the set of candidate horizons in the sliding window of the latest $W _ { h i s t }$ predictions ending at exploratory step $k _ { s }$ . Stability is achieved, concluding this stage, when $\mathrm { V a r } ( \hat { H _ { W } } ( k _ { s } ) ) < \sigma _ { s t a b l e } ^ { 2 }$ for a window where $k _ { s } \geq W _ { h i s t } .$ . The exploratory stage ends at step $k _ { f i n a l } ,$ , defined as: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/124360ad759820599a393b941e49b3a1ba3568d5e548c6765d0a7d6a1a6843fe.jpg)



Figure 3: Overview of the SlowFast Sampling Pipeline. The method alternates between a Slow (Exploratory) stage and a Fast (Accelerated) stage for efficient token generation. In the Slow phase (left), the model conducts cautious decoding by selecting top-k high-confidence tokens per step while continuously predicting the End Point ofConvergence and calculating confidence variance across a history window. Once variance drops below threshold $( e . g . , 0 . 2 2 < 0 . 2 3 )$ , the corresponding region $[ s _ { c y c l e } , e _ { c y c l e } ]$ is considered stable. In the Fast phase (right), this stable span is decoded in parallel with aggressive unmasking of high-confidence tokens, while tokens beyond the span are temporarily skipped and their results cached for reuse. This alternating structure reduces redundant computation and accelerates decoding while maintaining output quality.


$$
k _ {f i n a l} = \min \left(\{k _ {s} \mid W _ {h i s t} \leq k _ {s} \leq K _ {m a x} \land \operatorname{Var} (H _ {W} (k _ {s})) <   \sigma_ {s t a b l e} ^ {2} \} \cup \{K _ {m a x} \}\right)\tag{8}
$$

If the stability criterion is not met by $K _ { m a x }$ (the maximum allotted exploratory steps), then $k _ { f i n a l } = K _ { m a x }$ . The cycle’s convergence horizon $e _ { c y c l e }$ , is then set to the mean of the candidate horizons in the final window $H _ { W } ( k _ { f i n a l } )$ 

$$
e _ {c y c l e} = \mathrm{Mean} (H _ {W} (k _ {f i n a l})) = \frac {1}{| H _ {W} (k _ {f i n a l}) |} \sum_ {e \in H _ {W} (k _ {f i n a l})} e\tag{9}
$$

Once this stability criterion is met, the Exploratory Stage concludes. The endpoint for the subsequent Accelerated Decoding Stage, $e _ { c y c l e } ,$ is set to the last recorded candidate horizon, $e _ { c a n d } ^ { ( k ) } .$ . If stability is not achieved within a maximum number of exploratory steps, $e _ { c y c l e }$ can be set to a conservatively determined position or the process might default to a full-sequence cautious decode for that cycle. 

2. Accelerated Decoding Stage (Fast Phase): Rapid Parallel Refinement. As shown in the right half of Figure 3, once a stable region $[ s _ { c y c l e } , e _ { c y c l e } ]$ is identified, this stage aims to rapidly denoise tokens within this span, while efficiently handling tokens outside it: 

• Out-of-Span Caching: For tokens outside the identified span $( i . e .$ , positions $i > e _ { c y c l e } )$ , if their current predicted confidence is low $( e . g .$ ., below $\tau _ { m i n \_ c o n f } )$ , their predicted values $\hat { r } _ { 0 , i } ^ { ( k ) }$ from one dLLM step within this stage are computed and then cached. These cached values can be reused in subsequent dLLM steps for these positions, provided they remain outside an active decoding span, thus saving redundant computations. 

• In-Span Parallel Decoding: Within the span $[ s _ { c y c l e } , e _ { c y c l e } ]$ , an aggressive parallel decoding is attempted. All tokens $y _ { i } ^ { ( k ) } = \mathsf { [ M A S K ] }$ for $i \in [ s _ { c y c l e } , e _ { c y c l e } ]$ for which the predicted confidence $P _ { \theta } ( \hat { r } _ { 0 , i } ^ { ( k ) } | \mathbf { c } , \mathbf { y } ^ { ( k ) } )$ exceeds the high certainty threshold $\tau _ { h i g h \_ c o n f } , i . e .$ 

$$
P _ {\theta} (\hat {r} _ {0, i} ^ {(k)} | \mathbf {c}, \mathbf {y} ^ {(k)}) > \tau_ {h i g h \_ c o n f}\tag{10}
$$

are unmasked by setting their value to the corresponding prediction $\hat { r } _ { 0 , i } ^ { ( k ) }$ . This update is performed simultaneously for all such qualifying tokens in a single conceptual step to form $\hat { \mathbf { r } } _ { 0 } ^ { ( k ) }$ 

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="2">Inference Efficiency</td><td>Performance</td><td rowspan="2">Method</td><td colspan="2">Inference Efficiency</td><td>Performance</td></tr><tr><td>TPS↑</td><td>Speed↑</td><td>Score↑</td><td>TPS↑</td><td>Speed↑</td><td>Score↑</td></tr><tr><td colspan="9">Mathematics &amp; Science</td></tr><tr><td>GSM8K</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>4.55<eq>7.45_{+2.90}</eq>14.57<eq>_{+10.02}</eq></td><td>1.00×<eq>1.64 \times_{+0.64}</eq>3.20×<eq>_{+2.20}</eq></td><td>69.8369.60<eq>_{-0.23}</eq>69.59<eq>_{-0.27}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>8.16<eq>12.72_{+4.56}</eq>17.15<eq>_{+8.99}</eq></td><td>1.00×<eq>1.55 \times_{+0.55}</eq>2.10×<eq>_{+1.10}</eq></td><td>77.0273.09<eq>_{-3.93}</eq>76.50<eq>_{-0.52}</eq></td></tr><tr><td>GPQA</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>3.31<eq>11.72_{+8.41}</eq>16.36<eq>_{+13.05}</eq></td><td>1.00×<eq>3.54 \times_{+2.54}</eq>4.94×<eq>_{+3.94}</eq></td><td>31.4732.13<eq>_{+0.66}</eq>31.91<eq>_{+0.44}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>5.43<eq>15.88_{+10.45}</eq>16.56<eq>_{+11.13}</eq></td><td>1.00×<eq>2.92 \times_{+1.92}</eq>3.05×<eq>_{+2.05}</eq></td><td>35.9331.01<eq>_{-4.92}</eq>35.94<eq>_{+0.01}</eq></td></tr><tr><td>Math</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>5.14<eq>8.94_{+3.80}</eq>11.27<eq>_{+6.13}</eq></td><td>1.00×<eq>1.74 \times_{+0.74}</eq>2.19×<eq>_{+1.19}</eq></td><td>30.1630.52<eq>_{+0.36}</eq>29.64<eq>_{-0.52}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>8.48<eq>22.51_{+14.03}</eq>23.00<eq>_{+14.52}</eq></td><td>1.00×<eq>2.65 \times_{+1.65}</eq>2.71×<eq>_{+1.71}</eq></td><td>38.6834.14<eq>_{-4.54}</eq>38.24<eq>_{-0.44}</eq></td></tr><tr><td colspan="9">General Tasks</td></tr><tr><td>MMLU-pro</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>9.16<eq>15.62_{+6.46}</eq>23.14<eq>_{+13.98}</eq></td><td>1.00×<eq>1.71 \times_{+0.71}</eq>2.53×<eq>_{+1.53}</eq></td><td>23.3023.50<eq>_{+0.20}</eq>23.85<eq>_{+0.55}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>14.97<eq>25.50_{+10.53}</eq>22.80<eq>_{+7.83}</eq></td><td>1.00×<eq>1.70 \times_{+0.70}</eq>1.52×<eq>_{+0.52}</eq></td><td>24.1419.53<eq>_{-4.61}</eq>22.91<eq>_{-1.23}</eq></td></tr><tr><td>MMLU</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>5.02<eq>12.94_{+7.92}</eq>16.81<eq>_{+11.79}</eq></td><td>1.00×<eq>2.58 \times_{+1.58}</eq>3.35×<eq>_{+2.35}</eq></td><td>62.1161.67<eq>_{-0.44}</eq>66.56<eq>_{+4.45}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>8.46<eq>17.39_{+8.93}</eq>18.43<eq>_{+9.97}</eq></td><td>1.00×<eq>2.05 \times_{+1.05}</eq>2.18×<eq>_{+1.18}</eq></td><td>72.6164.59<eq>_{-8.02}</eq>75.13<eq>_{+2.52}</eq></td></tr><tr><td>BBH</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>4.04<eq>10.73_{+6.69}</eq>21.19<eq>_{+17.15}</eq></td><td>1.00×<eq>2.66 \times_{+1.66}</eq>5.24×<eq>_{+4.24}</eq></td><td>44.9744.13<eq>_{-0.84}</eq>44.60<eq>_{-0.37}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>6.93<eq>27.84_{+20.91}</eq>28.14<eq>_{+21.21}</eq></td><td>1.00×<eq>4.01 \times_{+3.01}</eq>4.06×<eq>_{+3.06}</eq></td><td>51.8352.28<eq>_{+0.45}</eq>50.55<eq>_{-1.28}</eq></td></tr><tr><td colspan="9">Code</td></tr><tr><td>MBPP</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>4.98<eq>8.15_{+3.17}</eq>13.32<eq>_{+8.34}</eq></td><td>1.00×<eq>1.64 \times_{+0.64}</eq>2.67×<eq>_{+1.67}</eq></td><td>40.8040.80<eq>_{+0.00}</eq>41.00<eq>_{+0.20}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>8.92<eq>22.87_{+13.95}</eq>29.07<eq>_{+20.15}</eq></td><td>1.00×<eq>2.56 \times_{+1.56}</eq>3.26×<eq>_{+2.26}</eq></td><td>54.2049.40<eq>_{-4.80}</eq>54.60<eq>_{+0.40}</eq></td></tr><tr><td>HumanEval</td><td>LLaDA+ Fast-dLLM (Parallel)+ SlowFast</td><td>11.24<eq>23.05_{+11.81}</eq>35.46<eq>_{+24.22}</eq></td><td>1.00×<eq>2.05 \times_{+1.05}</eq>3.15×<eq>_{+2.15}</eq></td><td>31.7132.32<eq>_{+0.61}</eq>33.54<eq>_{+1.83}</eq></td><td>Dream+ Fast-dLLM (Parallel)+ SlowFast</td><td>11.49<eq>24.33_{+12.84}</eq>25.38<eq>_{+13.89}</eq></td><td>1.00×<eq>2.12 \times_{+1.12}</eq>2.21×<eq>_{+1.21}</eq></td><td>34.1532.92<eq>_{-1.23}</eq>35.36<eq>_{+1.21}</eq></td></tr></table>


Table 1: Performance of LLaDA 8B and Dream 7B with SlowFast Sampling on 8 benchmarks.



• Fallback Top-k Refinement: If the number of tokens meeting the $\tau _ { h i g h \_ c o n f }$ criterion within the span is insufficient to make substantial progress (e.g., less than one), we revert to a more conservative update for this dLLM step within the span. Specifically, we select the $ { \mathrm { t o p } }  { - } k _ { f a s t }$ tokens within $[ s _ { c y c l e } , e _ { c y c l e } ]$ based on confidence and unmask them. This ensures steady progress even when widespread high certainty is not yet achieved.


After the Accelerated Decoding Stage completes, the starting position for the next cycle’s Exploratory Stage is updated to $s _ { c y c l e } \gets e _ { c y c l e }$ . This cyclical process of exploration and accelerated decoding continues until the entire sequence is generated. This SlowFast approach dynamically adapts the decoding focus and intensity, leveraging the Certainty and Positional principles to identify promising regions and the Convergence principle to stabilize them efficiently. 

## 4 EXPERIMENT

## 4.1 EXPERIMENT SETTINGS

Implementation Details To evaluate the effectiveness of our proposed dynamic sampling approach SlowFast Sampling, we conducted experiments on representative dLLMs: LLaDA 8B (Nie et al., 2025b) and Dream 7B (Ye et al., 2025), focusing on measuring the inference acceleration across various benchmarks. All experiments were conducted on NVIDIA RTX 4090 GPUs. 

Evaluation Metrics We evaluated sampling acceleration and generation quality of SlowFast Sampling using quantitative metrics. Inference speed is measured in Tokens Per Second (TPS), indicating the average number of tokens generated per second. Generation quality is assessed using task-specific metrics, e.g., accuracy on GSM8K, reflecting the model’s performance under inference acceleration. 

## 4.2 MAIN RESULTS

Performance and efficiency gains across models. Table 1 reports throughput and model performance for both LLaDA 8B and Dream 7B, with and without SlowFast Sampling. These results demonstrate that our method brings significant improvements in inference efficiency and achieves lossless acceleration in most cases. In our experiments, the key hyperparameters of SlowFast Sampling, $\tau _ { m i n \_ c o n f }$ and $\tau _ { h i g h \_ c o n f }$ , were set to 0.1 and 0.85, respectively. The remaining hyperparameters in our method were set as follows: maximum exploratory steps $K _ { m a x } = 8 .$ , sliding window size 


Table 2: Performance of LLaDA 8B and Dream 7B with SlowFast Sampling and dLLM-Cache.


<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="2">Inference Efficiency</td><td>Performance</td><td rowspan="2">Method</td><td colspan="2">Inference Efficiency</td><td>Performance</td></tr><tr><td>TPS↑</td><td>Speed(TPS)↑</td><td>Score↑</td><td>TPS↑</td><td>Speed(TPS)↑</td><td>Score↑</td></tr><tr><td colspan="9">Mathematics &amp; Science</td></tr><tr><td>GSM8K</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>4.55<eq>15.50_{+10.95}</eq><eq>26.99_{+22.44}</eq></td><td>1.00×<eq>3.41 \times_{+2.41}</eq><eq>5.93 \times_{+4.93}</eq></td><td>69.83<eq>68.77_{-1.06}</eq><eq>69.60_{-0.23}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>8.16<eq>31.07_{+22.91}</eq><eq>46.17_{+38.01}</eq></td><td>1.00×<eq>3.81 \times_{+2.81}</eq><eq>5.66 \times_{+4.66}</eq></td><td>77.02<eq>69.45_{-7.57}</eq><eq>72.10_{-4.92}</eq></td></tr><tr><td>GPQA</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>3.31<eq>30.21_{+26.90}</eq><eq>29.06_{+25.75}</eq></td><td>1.00×<eq>9.13 \times_{+8.13}</eq><eq>8.78 \times_{+7.78}</eq></td><td>31.47<eq>33.03_{+1.56}</eq><eq>33.48_{+2.01}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>5.43<eq>40.98_{+35.55}</eq><eq>39.63_{+34.20}</eq></td><td>1.00×<eq>7.54 \times_{+6.54}</eq><eq>7.30 \times_{+6.30}</eq></td><td>35.93<eq>32.37_{-3.56}</eq><eq>34.82_{-1.11}</eq></td></tr><tr><td>Math</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>5.14<eq>21.50_{+16.36}</eq><eq>26.50_{+21.36}</eq></td><td>1.00×<eq>4.18 \times_{+3.18}</eq><eq>5.16 \times_{+4.16}</eq></td><td>30.16<eq>28.34_{-1.82}</eq><eq>29.42_{-0.74}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>8.48<eq>46.80_{+38.32}</eq><eq>56.44_{+47.96}</eq></td><td>1.00×<eq>5.52 \times_{+4.52}</eq><eq>6.66 \times_{+5.66}</eq></td><td>38.68<eq>33.24_{-5.44}</eq><eq>37.10_{-1.58}</eq></td></tr><tr><td colspan="9">General Tasks</td></tr><tr><td>MMLU-pro</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>9.16<eq>27.85_{+18.69}</eq><eq>33.38_{+24.22}</eq></td><td>1.00×<eq>3.04 \times_{+2.04}</eq><eq>3.64 \times_{+2.64}</eq></td><td>23.30<eq>27.66_{+4.36}</eq><eq>25.53_{+2.23}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>14.97<eq>42.73_{+27.76}</eq><eq>43.50_{+28.53}</eq></td><td>1.00×<eq>2.85 \times_{+1.85}</eq><eq>2.91 \times_{+1.91}</eq></td><td>24.14<eq>22.57_{-1.57}</eq><eq>21.81_{-2.33}</eq></td></tr><tr><td>MMLU</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>5.02<eq>32.36_{+27.34}</eq><eq>38.42_{+33.40}</eq></td><td>1.00×<eq>6.45 \times_{+5.45}</eq><eq>7.65 \times_{+6.65}</eq></td><td>62.11<eq>61.45_{-0.66}</eq><eq>61.20_{-0.91}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>8.46<eq>44.73_{+36.27}</eq><eq>44.18_{+35.72}</eq></td><td>1.00×<eq>5.28 \times_{+4.28}</eq><eq>5.22 \times_{+4.22}</eq></td><td>72.61<eq>64.46_{-8.15}</eq><eq>71.57_{-1.04}</eq></td></tr><tr><td>BBH</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>4.04<eq>22.56_{+18.52}</eq><eq>36.04_{+32.00}</eq></td><td>1.00×<eq>5.58 \times_{+4.58}</eq><eq>8.92 \times_{+7.92}</eq></td><td>44.97<eq>45.35_{+0.38}</eq><eq>44.81_{-0.16}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>6.93<eq>57.02_{+50.09}</eq><eq>70.20_{+63.27}</eq></td><td>1.00×<eq>8.22 \times_{+7.22}</eq><eq>10.13 \times_{+9.13}</eq></td><td>51.83<eq>44.95_{-6.88}</eq><eq>48.24_{-3.59}</eq></td></tr><tr><td colspan="9">Code</td></tr><tr><td>MBPP</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>4.98<eq>22.18_{+17.20}</eq><eq>27.26_{+22.28}</eq></td><td>1.00×<eq>4.45 \times_{+3.45}</eq><eq>5.47 \times_{+3.87}</eq></td><td>40.80<eq>38.20_{-2.60}</eq><eq>39.00_{-1.80}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>8.92<eq>50.67_{+41.75}</eq><eq>69.48_{+60.56}</eq></td><td>1.00×<eq>5.68 \times_{+4.68}</eq><eq>7.79 \times_{+6.79}</eq></td><td>54.20<eq>50.40_{-3.80}</eq><eq>51.00_{-3.20}</eq></td></tr><tr><td>HumanEval</td><td>LLaDA+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>11.24<eq>26.25_{+15.01}</eq><eq>41.14_{+29.90}</eq></td><td>1.00×<eq>2.34 \times_{+1.34}</eq><eq>3.66 \times_{+2.66}</eq></td><td>31.71<eq>29.88_{-1.83}</eq><eq>31.10_{-0.61}</eq></td><td>Dream+ Fast-dLLM (Parallel+Cache)+ SlowFast + Cache</td><td>11.49<eq>49.47_{+37.98}</eq><eq>47.86_{+36.37}</eq></td><td>1.00×<eq>4.31 \times_{+3.31}</eq><eq>4.17 \times_{+3.17}</eq></td><td>34.15<eq>29.27_{-4.88}</eq><eq>35.36_{+1.21}</eq></td></tr></table>

<table><tr><td>Sampling Strategy</td><td>TPS↑</td><td>Score↑</td></tr><tr><td>Autoregressive (AR)</td><td>5.25</td><td>60.80</td></tr><tr><td>Diffusion Sampling</td><td>4.55</td><td>69.83</td></tr><tr><td>Semi-Autoregressive</td><td>5.44</td><td>66.41</td></tr><tr><td>SlowFast Sampling</td><td>9.87</td><td>69.59</td></tr></table>

<table><tr><td>Method</td><td>TPS↑</td><td>Speed(TPS)↑</td><td>Accuracy↑</td></tr><tr><td>LLaMA3 8B (Dubey et al., 2024)</td><td>33.79</td><td>21.12×</td><td>31.92</td></tr><tr><td>LLaDA</td><td>1.60-32.19</td><td>1.00×</td><td>31.47-0.45</td></tr><tr><td>+ SlowFast</td><td>25.00-8.79</td><td>15.63×</td><td>31.47-0.45</td></tr><tr><td>+ SlowFast + Cache (<eq>K_{p} = 100, K_{r} = 5</eq>)</td><td>48.80+15.01</td><td>30.50×</td><td>30.13-1.79</td></tr><tr><td>+ SlowFast + Cache (<eq>K_{p} = 500, K_{r} = 30</eq>)</td><td>54.75+20.96</td><td>34.22×</td><td>28.79-3.13</td></tr></table>


Table 3: Comparison of Sampling Strategies on inference efficiency and generation performance.



Table 4: Comparison of LLaDA 8B Base with other representative LLMs. Compared to LLaMA3 8B, LLaDA with SlowFast Sampling and dLLM-Cache achieves significantly higher throughput (up to +20.96 TPS) while maintaining comparable accuracy.


$W _ { h i s t } = 2 ,$ , and stable variance threshold $\sigma _ { s t a b l e } ^ { 2 } = 1 . 0$ , consistent with the default configuration used in Section 4.2 and ablation studies. 

Compatibility with dLLM-Cache. Recently, several studies have explored leveraging feature cache to reduce the computational cost of dLLM inference. Our SlowFast Sampling is highly compatible with existing caching mechanisms and the integration can lead to higher acceleration compared to using either alone. Table 2 compares the performance and inference speed of LLaDA with and without applying our method in combination with dLLM-Cache. The results show that this integration can deliver higher throughput while maintaining comparable model performance. 

Comparison with Other Sampling Strategies. We compared our method SlowFast Sampling with four alternative sampling strategies: diffusion sampling, diffusion semi-autoregressive sampling, autoregressive (AR) sampling and Fast-dLLM (Wu et al., 2025). Diffusion sampling adopts a remasking strategy to iteratively select token to decode in parallel, while AR sampling generates tokens strictly from left to right. The semi-autoregressive approach generates blocks left-to-right and applies the remasking strategy within each block. Fast-dLLM, in contrast, performs parallel decoding only based on token confidence, and we report results for both its parallel and cache-augmented variants. As shown in Table 1, 2 and 3, our method SlowFast Sampling achieves higher inference efficiency while maintaining competitive generation quality. 

## 4.3 ABLATION STUDY

Case Study of SlowFast Sampling in Sentence Generation. As shown in Figure 4, this case study shows how SlowFast Sampling generates text by alternating slow and fast phases. In the slow phases, the model outputs a few reliable tokens such as subjects, verbs, or punctuation to anchor the sentence. The fast phases then produce longer spans of high-confidence tokens in one step, e.g., “she has 9 - 4 = 5 yuan $l e f t ^ { , }$ , ensuring both efficiency and correctness. This balance of cautious slow decoding and confident fast decoding maintains accuracy while accelerating generation, making the approach intuitive and effective. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/379d14e082f3e683b26d53ce9adaa9e657a9bb6c7e369e814843e67e5fbefe7d.jpg)



end point 2 each bought bought ␣ 2 pears at ␣ 2 yuan each, so she spent ␣ 2 * ␣ 2 = 4 yuan. \n ␣ 2 pears at ␣ 2 yuan each, so she spent ␣ 2 * ␣ 2 = 4 yuan. \n Therefore \n she bought ␣ 2 pears at ␣ 2 yuan each, so she spent ␣ 2 * ␣ 2 = 4 yuan. \n Therefore \n she bought ␣ 2 pears at ␣ 2 yuan each, so she end point 3 spent ␣ 2 * ␣ 2 = 4 yuan. \n Therefore, \n she bought ␣ 2 pears at ␣ 2 yuan each, so she spent ␣ 2 * ␣ 2 = 4 yuan. \n Therefore, she has ␣ 9 - ␣ 4 = 5 yuan left \n she bought ␣ 2 pears at ␣ 2 yuan each, so she spent ␣ 2 * ␣ 2 = 4 yuan. \n Therefore, she has ␣ 9 - ␣ 4 = ␣ 5 yuan left.


Figure 4: Case study of sentence generation with SlowFast Sampling. The figure illustrates the dynamic evolution of the confidence map across three phases, where exploratory and accelerated decoding are alternated. Blue tokens denote those generated in the current step, highlighting how the method gradually builds on high-confidence spans to produce a coherent final response. 

Effect of Hyperparameters in the Stability Check. The stability check, which transitions the model from the slow to the fast phase, is governed by three hyperparameters whose effects are shown in Figure 5. The maximum number of exploratory steps, $\bar { K } _ { m a x }$ , is for allowing the convergence horizon to stabilize; we find $K _ { m a x } = 8$ provides sufficient exploration for high-quality generation without incurring excessive overhead. This is complemented by the sliding window size, $W _ { h i s t } .$ Since prediction changes and convergence occur rapidly, we find a smaller window of $W _ { h i s t } = 2$ is effective, striking a strong balance between maintaining quality and maximizing inference speed. Finally, a strict stable variance threshold of $\sigma _ { s t a b l e } ^ { 2 } = 1 . 0$ ensures that the accelerated phase is only triggered for genuinely stable regions, solidifying the reliability of our method. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/fadbf6d202a3dc848f0bffe0ee09619bf20517b6c67f5ab9df6c489438ae4e38.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/64b3718a8ce6d005a787cfd391b4f0123966f72509e4ea1616d137c8d0b96f86.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/9418f8e4290228a0a4212cc49a6f814a713398a55a7724b816c1310bac3bcac5.jpg)



Figure 5: The sensitivity study on hyper-parameters in the stability-Check. Accuracy and TPS vary with $K _ { m a x } , \sigma _ { s t a b l e } ^ { 2 } .$ and $W _ { h i s t }$ . The chosen defaults $( K _ { m a x } = 8 , \sigma _ { s t a b l e } ^ { 2 } = 1 . 0 , \dot { W } _ { h i s t } = 2 )$ offer strong speed-quality trade-offs.


Outperforming Autoregressive LLMs in Inference Speed. As shown in Table 4, when equipped with our SlowFast Sampling and dLLM-Cache, LLaDA Base not only accelerates significantly over its default setting but also outperforms the autoregressive LLaMA3 8B (Dubey et al., 2024) in inference speed (54.75 vs. 33.79 TPS), while maintaining comparable accuracy. This demonstrates that dLLMs, with proper sampling and optimization, can surpass traditional autoregressive models in both efficiency and practicality. 

## 5 CONCLUSION

In this work, we present SlowFast Sampling, a dynamic and principled approach to accelerate diffusion-based large language models (dLLMs). By leveraging three key observations: token certainty, convergence, and positional bias, we design a two-stage decoding pipeline that adaptively balances exploration and efficient parallel decoding. Extensive experiments across benchmarks demonstrate that our method not only significantly improves inference speed (up to 15.63× and up to 34.22× combined with dLLM-Cache on the GPQA benchmark), but also maintains strong generation quality, outperforming even autoregressive LLMs in throughput. We believe this work marks an important step toward making dLLMs practical and competitive in real-world deployment. 

## ACKNOWLEDGEMENT

This project is sponsored by CCF-Tencent Rhino-Bird Funds. 

## REPRODUCIBILITY STATEMENT

We have made every effort to ensure the reproducibility of our results. The main text (Sections 3–4) provides detailed descriptions of the proposed SlowFast Sampling method, its integration with dLLM-Cache, and the corresponding experimental setups. Hyperparameter configurations and additional implementation details are presented in Appendix A.3. Comprehensive performance comparisons across benchmarks are reported in Tables 1–4. Ablation studies and a case study are provided in Section 4.3 (e.g., Figure 4) to illustrate the robustness of our method. To facilitate replication, we include the source code and scripts for running experiments in the supplementary materials. 

## ETHICS STATEMENT

This work focuses on developing efficient sampling strategies for diffusion-based large language models (dLLMs). Our research does not involve human subjects, personal or sensitive data, or applications directly impacting individuals. All experiments were conducted on publicly available benchmarks, ensuring fairness, reproducibility, and transparency. We release code to promote open research and to enable verification of our results. We are not aware of any potential misuse or harmful applications of the proposed methods beyond standard concerns inherent to language models, such as bias or misuse in downstream tasks, and we encourage responsible use in line with the ICLR Code of Ethics. 

## REFERENCES



Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. arXiv preprint arXiv:2503.09573, 2025. 





Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in Neural Information Processing Systems, 34:17981–17993, 2021. 





Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. The reversal curse: Llms trained on" a is b" fail to learn" b is a". arXiv preprint arXiv:2309.12288, 2023. 





Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. Advances in Neural Information Processing Systems, 35:28266–28279, 2022 





Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11315–11325, 2022. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 





Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 





Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. Advances in Neural Information Processing Systems, 34:12454–12465, 2021. 





Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching, 2025. URL https://github.com/maomaocun/dLLM-cache. 





Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023. 





Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models, 2025a. URL https://arxiv.org/abs/2505.15781. 





Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869, 2025b. 





Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text, 2025a. URL https://arxiv.org/abs/ 2410.18514. 





Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models, 2025b. URL https: //arxiv.org/abs/2502.09992. 





William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023. 





Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings ofthe IEEE/CVF confer ence on computer vision and pattern recognition, pp. 10684–10695, 2022. 





Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K Titsias. Simplified and generalized masked diffusion for discrete data. arXiv preprint arXiv:2406.04329, 2024. 





Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015. 





Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 





Yuerong Song, Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, and Xipeng Qiu. Sparse-dllm: Accelerating diffusion llms with dynamic cache eviction, 2025. URL https://arxiv.org/abs/2508.02558. 





Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding, 2025. URL https://arxiv.org/abs/2505.22618. 





Feihong Yan, Peiru Wang, Yao Zhu, Kaiyu Pang, Qingyan Wei, Huiqi Li, and Linfeng Zhang. Generation then reconstruction: Accelerating masked autoregressive models via two-stage sampling, 2026. URL https://arxiv.org/abs/2510.17171. 





Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b, 2025. URL https://hkunlp.github.io/blog/2025/dream. 





Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. A survey of large language models, 2025. URL https://arxiv.org/abs/ 2303.18223. 





Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908, 2024. 



## A APPENDIX

## A.1 LLM USAGE STATEMENT.

We used a Large Language Model (LLM) solely to aid in polishing the writing and improving the clarity of language. The LLM was not involved in research ideation, experimental design, data analysis, or generation of scientific content. All research ideas, methods, analyses, and conclusions presented in this paper are entirely the work of the authors. The authors take full responsibility for the content of this paper. 

## A.2 COMPATIBILITY WITH ADVANCED SAMPLING METHODS.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/08e264bd71be2e1b3adc3da20f65559954565426e14bae0d2a0dd76f7bc9d6f1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/2d06d5e553128c30107dc7626f482aa2a16f27c9eceecde4bd2bb62147b3752b.jpg)


Figure 6: Effect of Confidence Thresholds on GSM8K. $\tau _ { m i n \_ c o n f }$ controls the exploratory range, and $\tau _ { h i g h \_ c o n f }$ balances accuracy and speed during fast decoding. Other hyperparameters follow the default settings in Section 4.2. 

Sensitivity to Confidence Thresholds. The core confidence thresholds, $\tau _ { m i n \_ c o n f }$ and $\tau _ { h i g h \_ c o n f } ,$ dictate the behavior of our SlowFast pipeline. As shown in Figure 6, the minimum confidence threshold $\tau _ { m i n \_ c o n f }$ defines the candidate region in the exploratory stage. A moderate value of $\tau _ { m i n \_ c o n f } = 0 . 1$ is optimal, as lower values lead to unstable regions and higher values create overly conservative, inefficient regions. The high certainty threshold $\tau _ { h i g h \_ c o n f }$ directly manages the speedquality trade-off during the accelerated stage. A higher value leads to more cautious and accurate generation at the cost of speed (TPS). We select $\tau _ { h i g h \_ c o n f } = 0 . 8 5$ , which achieves a near-peak GSM8K score without a drastic reduction in inference speed, striking an effective balance. 

## A.3 EXPERIMENTAL DETAILS

To ensure fair and reproducible comparisons, we standardize the setting of inference generation across benchmarks and run each model with its officially recommended settings. The benchmark-specific parameters considered include the number of inference steps, the block length, the total generation length, and the number of few-shot examples. For clarity, we summarize the settings for LLaDA and Dream in Tables 5 and 6, respectively. 


Table 5: Experimental settings for LLaDA across benchmarks.


<table><tr><td>Task</td><td>Steps</td><td>Block Length</td><td>Generation Length</td><td>Few-shot</td></tr><tr><td>MMLU</td><td>3</td><td>3</td><td>3</td><td>5</td></tr><tr><td>MMLU-pro</td><td>256</td><td>256</td><td>256</td><td>0</td></tr><tr><td>GSM8K</td><td>256</td><td>256</td><td>256</td><td>4</td></tr><tr><td>Math</td><td>256</td><td>256</td><td>256</td><td>0</td></tr><tr><td>GPQA</td><td>256</td><td>256</td><td>256</td><td>5</td></tr><tr><td>HumanEval</td><td>256</td><td>256</td><td>256</td><td>0</td></tr><tr><td>MBPP</td><td>256</td><td>256</td><td>256</td><td>3</td></tr><tr><td>BBH</td><td>256</td><td>256</td><td>256</td><td>3</td></tr></table>


Table 6: Experimental settings for Dream across benchmarks.


<table><tr><td>Task</td><td>Steps</td><td>Generation Length</td><td>Few-shot</td></tr><tr><td>MMLU</td><td>3</td><td>3</td><td>5</td></tr><tr><td>MMLU-pro</td><td>256</td><td>256</td><td>0</td></tr><tr><td>GSM8K</td><td>256</td><td>256</td><td>8</td></tr><tr><td>Math</td><td>256</td><td>256</td><td>4</td></tr><tr><td>GPQA</td><td>256</td><td>256</td><td>5</td></tr><tr><td>HumanEval</td><td>256</td><td>256</td><td>0</td></tr><tr><td>MBPP</td><td>256</td><td>256</td><td>3</td></tr><tr><td>BBH</td><td>256</td><td>256</td><td>3</td></tr></table>

## A.4 ABLATION STUDY ON HYPERPARAMETERS AND ROBUSTNESS.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/69ac05ec848fdcba1a8b7d00cae4c60b532d278430dd5ce05751338c955dc6c4.jpg)



Minimum Confidence Threshold


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/a9f81694101148c344dd5b57821c786ddfd5645571d08fbd3f61176429c1d9b5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/ec4d083f093361fec86f1f5516bd7ccdfa3208902c0029e68f284971f4be3d6a.jpg)



Figure 7: Parameter Robustness and Sensitivity Across Multiple Benchmarks. Figure shows sensitivity across MATH, MBPP, and HumanEval. Our setting is shown to be highly robust across all tasks.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/b5cf213fdaf3767eb19b83670cf8e2331998d43b468e089d1785d623ef6ccf41.jpg)



Minimum Confidence Threshold


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/bbf73ecbd54b4c46cc28e8eea2871dc7eaac20354d5fdfb41ceaace43a15e3d9.jpg)



High Certainty Threshold


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/9448cb8f7fad81698c2877bbe0422470d6d60018dbebd3867e5f33f0542f95f3.jpg)



Figure 8: Parameter Stability Across Model Variants on Dream 7B. Sensitivity analysis for τ<sub>min_con</sub> ${ \it f } , \tau _ { h i g h \_ c o n f } ,$ , and $\sigma _ { s t a b l e } ^ { 2 }$ was conducted on Dream 7B using the GSM8K benchmark. The findings confirm that the unified set of hyperparameters successfully transfers to this demanding model variant, validating overall parameter stability.


Hyperparameter Robustness and Generalization Analysis. The overall hyperparameter analysis confirms the robustness and strong generalization ability of the SlowFast approach. As shown in Figure 7, a single, unified parameter set maintains near-optimal performance across diverse benchmarks (MATH, MBPP, HumanEval). This stability is further validated by Figure 8, where the same unified settings successfully transfer to Dream 7B on the GSM8K benchmark. These findings confirm that the SlowFast approach is highly robust and does not require extensive tuning for new tasks or model variants. 

## A.5 PERFORMANCE-EFFICIENCY TRADE-OFF ACROSS STRATEGIES

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/899a70bb7651a941eefd4ba2e4f0ed303807050e893ab9e495d3f1f57e044a43.jpg)



Figure 9: Performance Efficiency Trade-off on GPQA.


Figure 9 evaluates the Performance-Efficiency for our method against various baselines under two distinct evaluation setups. The results confirm that the integration of our method consistently shifts the trade-off curve decisively towards the top-right corner (higher TPS and higher Accuracy) compared to the baseline. This superiority holds true across both the standard configuration (5- shot, 256 tokens) and the more demanding scenario (8-shot, 1024 tokens), validating the robustness and scalability of our adaptive sampling strategy. 

## A.6 CASE STUDIES ON POSITIONAL PRINCIPLE AND NON-CONTIGUOUS DECODING.

Robustness to Minor Positional Uncertainty. Figure 10 confirms SlowFast’s ability to cover multiple stable blocks even when separated by unstable regions. Crucially, the distance between the end of one stable block and the start of the next is generally small, allowing our adaptive mechanism to efficiently jump over the intermediate low-confidence span. This ensures that the overall acceleration is maintained without sacrificing accuracy, even when faced with minor positional uncertainties. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/68f551785e774aece39000daa913892097af5e9402acd6674296a7817e822b51.jpg)



Figure 10: Case Study I: Dynamic Span Coverage During Positional Jumps. This case study illustrates a scenario where the model exhibits a positional jump between decoded tokens, confirming that the dynamic search window of SlowFast successfully identifies and covers the subsequent stable region.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/7decb90e-b10b-4a88-a405-93fd8625468d/b21fb0bd2375cd40f23f29ba27c7b33fb77a00d79c6fb453d3015d73ac40cdea.jpg)



Figure 11: Case Study II: SlowFast’s Adaptive Response to Induced Non-Contiguous Blocks. This case study demonstrates the adaptive capability of the SlowFast framework. By manually inducing a positional jump, the figure shows how the dynamic adjustment mechanism immediately recognizes and proceeds to the subsequent high-confidence region.


Adaptive Handling of Non-Contiguous Blocks. Figure 11 rigorously tests SlowFast’s adaptive limits by manually constructing an extreme scenario featuring two widely separated high-confidence blocks. The visualization confirms that upon encountering the unstable region between the blocks, the dynamic adjustment mechanism successfully identifies the subsequent high-confidence region (Block 2) and resumes acceleration (Fast Phase 2). This successful navigation demonstrates the robustness of SlowFast’s dynamic scheduling, confirming its ability to automatically identify and accelerate multiple non-contiguous stable blocks even under extreme conditions. 

## A.7 ALGORITHM: SLOWFAST SAMPLING FOR DLLMS

Algorithm 1 outlines the SlowFast Sampling Strategy, a dynamic, two-phase framework for accelerating dLLMs. It starts with the Slow, Exploratory Stage, which proceeds step-by-step (Lines 10-12) to determine a safe decoding span boundary. This boundary is validated by a Variance-based Stability Check (Lines 16-22), confirming stability when the variance of endpoint predictions falls below $\sigma _ { s t a b l e } ^ { 2 }$ . Once stability is confirmed, the algorithm enters the Fast, Accelerated Decoding Stage. Here, acceleration is achieved via In-Span Parallel Decoding (Lines 40-47), where high-confidence tokens within the span are unmasked simultaneously. Additionally, Out-of-Span Caching (Lines 34-39) prepares for the subsequent cycle. This adaptive, two-stage mechanism ensures maximum efficiency is achieved only in reliably stable regions. 

Algorithm 1 SlowFast Sampling Strategy
Require: Masked sequence $y^{(N)}$ , Prompt c, Max steps N, Exploratory limit $K_{max}$ , History window $W_{hist}$ , Stability threshold $\sigma_{stable}^2$ , Confidence thresholds $\tau_{min\_conf}, \tau_{high\_conf}$ .
Ensure: Generated sequence $y^{(0)}$ .
1: $k \leftarrow N$ ▷ Initialize diffusion step
2: $s_{cycle} \leftarrow 1$ ▷ Start of the current decoding cycle
3: $H_W \leftarrow \emptyset$ ▷ History of convergence points
4: while k > 0 and $s_{cycle} < L$ do
5: // Phase 1: Exploratory Stage (Slow)
6: step_count ← 0
7: is_stable ← False
8: while step_count < $K_{max}$ and k > 0 do
9: Predict $\hat{r}_0^{(k)}$ and confidence $P_\theta(\cdot | c, y^{(k)})$ 10: Cautious Decoding:
11: Unmask Top- $k_{slow}$ tokens in range [ $s_{cycle}, L$ ]
12: End Point Prediction:
13: $e_{cand}^{(k)} \leftarrow \max\{i \mid i \in [s_{cycle}, L] \land P_\theta(\hat{r}_{0,i}^{(k)}) > \tau_{min\_conf}\}$ 14: Update history window $H_W$ with $e_{cand}^{(k)}$ 15: Stability Check:
16: Var $_{curr}$ ← Variance( $H_W$ )
17: if $|H_W| \geq W_{hist}$ and Var $_{curr} < \sigma_{stable}^2$ then
18: is_stable ← True
19: $e_{cycle} \leftarrow Mean(H_W)$ 20: k ← k - 1
21: break ▷ Exit slow phase early
22: end if
23: Update state $y^{(k-1)}$ using standard scheduler
24: k ← k - 1
25: step_count ← step_count + 1
26: end while
27: if not is_stable then
28: $e_{cycle} \leftarrow e_{cand}^{(k)}$ ▷ Fallback if no convergence
29: end if
30: // Phase 2: Accelerated Decoding Stage (Fast)
31: if k > 0 then
32: Predict $\hat{r}_0^{(k)}$ and confidence $P_\theta(\cdot | c, y^{(k)})$ 33: Out-of-Span Caching:
34: for i ∈ ( $e_{cycle}, L$ ) do
35: if $P_\theta(\hat{r}_{0,i}^{(k)}) < \tau_{min\_conf}$ then
36: Cache features for position i
37: end if
38: end for
39: In-Span Parallel Decoding:
40: $S_{high} \leftarrow \{i \mid i \in [s_{cycle}, e_{cycle}] \land P_\theta(\hat{r}_{0,i}^{(k)}) > \tau_{high\_conf}\}$ 41: if $S_{high} \neq \emptyset$ then
42: $y_i^{(k-1)} \leftarrow \hat{r}_{0,i}^{(k)}$ for all i ∈ $S_{high}$ ▷ Aggressive unmasking
43: else
44: Unmask Top- $k_{fast}$ tokens in [ $s_{cycle}, e_{cycle}$ ] ▷ Fallback refinement
45: end if
46: $s_{cycle} \leftarrow e_{cycle}$ ▷ Advance the stable region start
47: k ← k - 1
48: end if
49: end while
50: return $y^{(0)}$ 