# Speculative Diffusion Decoding: Accelerating Language Generation through Diffusion

Jacob K. Christopher University of Virginia csk4sr@virginia.edu 

Tal Ben-Nun Lawrence Livermore National Laboratory talbn@llnl.gov 

Bhavya Kailkhura Lawrence Livermore National Laboratory kailkhura1@llnl.gov 

## Abstract

Speculative decoding has emerged as a widely adopted method to accelerate large language model inference without sacrificing the quality of the model outputs. While this technique has facilitated notable speed improvements by enabling parallel sequence verification, its efficiency remains inherently limited by the reliance on incremental token generation in existing draft models. To overcome this limitation, this paper proposes an adaptation of speculative decoding which uses discrete diffusion models to generate draft sequences. This allows parallelization of both the drafting and verification steps, providing significant speedups to the inference process. Our proposed approach, Speculative Diffusion Decoding (SpecDiff), is validated on standard language generation benchmarks and empirically demonstrated to provide up to 7.2x speedups over standard generation processes and up to 1.75x speedups over existing speculative decoding approaches. 

## 1 Introduction

As autoregressive language modeling with transformers (Vaswani et al., 2017) is scaled to larger compute levels, performance improves and new capabilities emerge (Kaplan et al., 2020; Brown et al., 2020). Indeed, scaling has been shown to to improve the performance of large language models (LLMs) for a diverse array of tasks, including code generation, question answering, summarization, and many other use cases (Achiam et al., 2023; Gemini Team, 2023; Llama Team, 2024). For instance, models such as LLaMA 3.2 90B (Llama Team, 2024), ChatGPT (OpenAI et al., 2024), Cohere 52B (Ruis et al., 2023), Google’s Gemini-ULTRA (Team et al., 2024) exemplify the ongoing 

Brian R. Bartoldson Lawrence Livermore National Laboratory bartoldson1@llnl.gov 

Michael Cardei University of Virginia ntr2rm@virginia.edu 

Ferdinando Fioretto University of Virginia fioretto@virginia.edu 

trend of deploying and releasing increasingly large models, enabling broader access and application across various domains. 

However, while these desired capability arise, running LLMs in inference mode for millions of users produces burdensome electricity, time, and monetary demands. Many methods exist to mitigate these costs – including sparsity, quantization, and distillation – but they often introduce new tradeoffs, e.g., their application can degrade the performance of the model (Hong et al., 2024). 

Unlike other methods for accelerating LLM inference, speculative decoding (Xia et al., 2023; Leviathan et al., 2023) can improve LLM efficiency by 2–3 with no degradation in the quality ofthe model outputs. In Leviathan et al. (2023), speculative decoding achieves this by sequentially generating multiple tokens with a small, efficient draft model, then running the target, large, LLM in parallel on all of the drafted tokens, simultaneously evaluating their consistency with the target LLM’s output token probabilities. Provided that the drafting model’s tokens are frequently accepted by the target model and that the drafting model operates substantially faster than the target model, speculative decoding can directly match the sampling output from the target model while significantly reducing runtime (Leviathan et al., 2023). This functioning is shown in Figure 1 (left). 

Notably, since both the drafting model’s speed and its alignment relative to the target model are critical to the success of speculative decoding, simultaneous improvements in each of these areas are necessary to ensure speculative decoding’s relevance to future, more capable target models. For instance, a small GPT-2 (Radford et al., 2019) drafting model could produce drafts that are often rejected by GPT-4 (Achiam et al., 2023), and simply scaling the drafting model to address its weaker generations risks diminishing the speed advantage necessary to speculative decoding’s success. To address this challenge, previous efforts have focused on introducing additional parallelization techniques that incorporate prediction trees and branching to refine the drafting process (Fu et al., 2024; Miao et al., 2023b; Svirschevski et al., 2024). However, the gain in generation efficiency are at the expense of much increased number of operations and/or memory for each generation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/96331fd7-c2e9-46c8-ad96-9ab1675dcbd7/95ac957e04b64a630db277ebdd8fb6569d3191bd27814082aad8dc0451690044.jpg)



Figure 1: Illustration of classical speculative decoding (left) and speculative diffusion decoding (right).


This paper proposes a fundamentally different approach to improve speculative decoding: It proposes to replace the auto-regressive drafter with recently introduced discrete diffusion models (Lou et al., 2024; Sahoo et al., 2024). These models offer several key advantages when used as drafters: Firstly, they provide a smooth trade off between the compute cost of generation and the quality of generation (via the number of reverse diffusion steps). Second, while they have historically struggled relative to traditional language models, recent diffusion models have been shown to require 32 fewer function evaluations than autoregressive models to produce text with comparable perplexity (Lou et al., 2024), with more recent works reporting even further speedups (Sahoo et al., 2024). This is a trend which is anticipated to continue. Finally, future advances in diffusion model generation quality are highly aligned with their ability to perform strongly as speculative drafters: as drafted tokens are accepted by the target model at a higher rate, a larger number of proposed drafted tokens becomes optimal from an efficiency/speed point of view, and (unlike sequential drafters) diffusion models can easily accommodate generation of many more tokens since they are able to generate entire sequences in a single step. 

Contributions. More specifically, this paper makes the following contributions: 

1. It introduces a novel integration of generative diffusion language models with speculative decoding, schematically illustrated in Figure 1 (right). 

2. It empirically demonstrate the hybrid model’s ability to significantly accelerate inference times while maintaining the same high-quality outputs of the original, target large language model. 

3. The proposed method ensures that all generations from the diffusion language model, which are empirically shown to produce outputs with significantly higher perplexity than current stateof-the-art autoregressive models (Sahoo et al., 2024; Lou et al., 2024; Austin et al., 2021; Gloeckle et al., 2024), align with the outputs generated by larger, more computationally demanding models. 

4. Finally, the paper sets a new benchmark for speed in language completion tasks on the CN-N/DM (Nallapati et al., 2016)and OpenWebText datasets (Gokaslan et al., 2019). 

## 2 Related Work

While autoregressive language models provide state-of-the-art performance on language generation tasks, the incremental decoding used by these architectures results in significant overhead at inference time (Miao et al., 2023a). This is largely a result of the inability to parallelize the sequential process of generating tokens in the output sequence as each token generation is dependent upon the preceding tokens in the sequence; consequentially, scaling the compute associated with the inference cannot directly reduce this overhead when using standard decoding schemes. In recent literature studying how to accelerate large language model generation, two primary approaches have been explored: (1) advanced decoding implementations that better parallelize token generation and (2) non-autoregressive language models allowing full sequences to be generated simultaneously. 

Speculative decoding. Speculative decoding accelerates autoregressive generation by leveraging a smaller autoregressive models of the same architecture (the drafter model) to predict candidate sequences, which the original model (the target model) then verifies (Leviathan et al., 2023; Chen et al., 2023). Notably, the earliest literature on speculative diffusion adapted a non-autoregressive model to act as the drafter model (Xia et al., 2023), using a masked language model with a bidirectional decoder (Ghazvininejad et al., 2019). However, the integration of non-autoregressive draft models has not received much attention due to the difficulty introduced by the necessary additional training in existing approaches and the modest speedups that were previously reported using these methods (less than 2x speedup over vanilla decoding schemes). 

Thus, recent advancements in speculative decoding have focused on overcoming memory-related constraints, with improvements achieved through various approaches: drafting directly with the target model (Cai et al., 2024; Zhang et al., 2024), enhancing draft algorithms (Sun et al., 2024), and introducing additional parallelization techniques that incorporate branching to refine the drafting process (Fu et al., 2024; Miao et al., 2023b; Svirschevski et al., 2024). 

Non-autoregressive language models. Models which stray from the autoregressive paradigm have been shown to speedup generation by generating blocks or even entire sequences simultaneously. 

Gloeckle et al. (2024) propose a method of adapting traditional autoregressive models to sample blocks of tokens, improving inference time over similarly scaled models. In a similar vein, diffusion language models have been recognized for their efficiency in generating extended token sequences concurrently, offering even greater speed enhancements. These models recast language generation as a diffusion process either across the embedding space (Austin et al., 2021) or, more recently, through the probability distributions of generated tokens (Sahoo et al., 2024; Lou et al., 2024). Recent models report up to a 32x speedup over similarly sized GPT-2 models (Lou et al., 2024), and current state-of-the-art further improves runtime speed (Sahoo et al., 2024). However, despite they have been shown to dramatically accelerate the inference time for language generation, diffusion models typically perform less effectively than state-of-the-art autoregressive models in terms of standard language metrics, often exhibiting significantly higher perplexity scores (Zheng et al., 2024). In the following section, we will demonstrate,for thefirst time, how the speed of these models can be leveraged without being subject to this critical limitation. 

## 3 Preliminaries and Settings

For open-ended language generation, we focus on the task of token generation, where given a sequence of tokens $x _ { 1 } , x _ { 2 } , \ldots , x _ { i }$ , denoted here with shorthand notation $x _ { 1 : i } .$ , the goal is to generate the next n tokens $x _ { i + 1 } , \ldots , x _ { i + n }$ from the conditional distributions $p ( x _ { i + 1 } | x _ { 1 : i } ) , \dots , p ( x _ { i + n } | x _ { 1 : i + n - 1 } )$ or more succinctly $p _ { i + 1 } , \ldots , p _ { i + n }$ 

Speculative decoding. Speculative decoding leverages two LLMs, $M _ { p }$ and $M _ { q }$ , to parallelize token generation: 

$M _ { p }$ is the original, target, model whose output probability distributions for the tokens are $p _ { i + 1 } , \ldots , p _ { i + n } .$ 

$M _ { q }$ is a smaller and more efficient drafter model, used to generate approximations of the distribution of $M _ { p }$ as $q _ { i + 1 } , \ldots , q _ { i + n }$ 

This process follows a draft-then-verify approach (Stern et al., 2018), where $M _ { q }$ efficiently computes a candidate sequence of tokens, which $M _ { p }$ then verifies in parallel. 

During each speculative decoding iteration, $M _ { q }$ generates a subset of the total n tokens that are required for the generation task. The size of this subset is denoted as $\gamma .$ As shown in Figure 1 (left), the tokens $x _ { i + 1 : i + \gamma }$ sampled from $M _ { q }$ are then used by $M _ { p }$ to generate the corresponding probability distributions $p _ { i + 1 } , . . . , p _ { i + \gamma } .$ . The distributions $q _ { i + 1 } , \ldots , q _ { i + \gamma }$ from $M _ { q }$ are stored for evaluating acceptance in subsequent steps. Critically, the target model’s inference over $p _ { i + 1 } , . . . , p _ { i + \gamma }$ can now be run in parallel as the model has access to tokens $x _ { i + 1 : i + \gamma }$ , alleviating the sequential dependency for generation with $M _ { p }$ 

To ensure high-quality outputs despite potential discrepancies between $M _ { p }$ and $M _ { q }$ , tokens are subjected to an acceptance criterion. For each token $x _ { j }$ with $j \in [ i + 1 , i + \gamma ]$ , if $q ( x _ { j } ) \leq p ( x _ { j } )$ , the token is accepted. If $q ( x _ { j } ) > p ( x _ { k } )$ , the token is rejected with a probability of $1 - { \frac { p ( x _ { j } ) } { q ( x _ { j } ) } }$ . This criterion is applied sequentially from left to right; rejection of any token results in the discard of all subsequent tokens. Hence, the token acceptance is maximized when the output distributions of $M _ { q }$ and $M _ { p }$ are closely aligned. 

Previous literature quantifies the likelihood of token acceptance, denoted $\alpha ,$ , and theoretically demonstrate that $\alpha = 1 - \mathbb { E } ( D _ { L K } ( p , q ) )$ where $D _ { L K }$ represents a divergence measure between distributions (Leviathan et al., 2023). This has led to the prevalent use of drafters taken from the same series as the target models, a paradigm that we challenge in this paper. 

## 4 Speculative Diffusion Models

Speculative decoding has provided state-of-the-art results for improving language generation inference time but requires meticulous tuning of the associated hyperparameters to achieve optimal results. Particularly $\gamma _ { : }$ , the sequence length generated by the drafter model, needs to be appropriately calibrated not only to maximize potential speedup but to even outperform standard autoregressive decoding. This is an important consideration when using current autoregressive draft models, provided that the inference time to generate $M _ { q } ( x )$ , the draft logits, is directly scaled by the size of $\gamma .$ . Increasing this value too high reduces the number of oper ations that are conducted in parallel, potentially leading to speculative decoding increasing inference time, while reducing this value too low results in speculative decoding “missing out” on token generations that could have been handled by the draft model. 

Leviathan et al. (2023) has conducted theoretical analysis on how to best optimize the value of $\gamma _ { : }$ however, it has been contingent upon accurately estimating the percentage of tokens in a the sequence that will be accepted by the target model. By their own acknowledgment, it would be necessary to predict this value for each draft and numerically solve for the optimal value of $\gamma$ to fully realize the potential speedup of speculative decoding. Thus, a significant portion of the residual suboptimality in current implementations can be attributed directly to the sensitivity of this hyperparameter. 

Diffusion language models are juxtaposed to conventional language models in that they do not sample token sequences in a sequential manner, rather generating entire sequences in parallel. This has resulted in significant speedup over similarly sized autoregressive models when generating extended sequences (Lou et al., 2024). This can particularly be observed in longer sequence generations as scaling the draft length $\gamma$ results in minimal overhead due to the ability to directly parallelize token generation. 

## 4.1 SpecDiff: Formulation

Diffusion models generally operate on continuous data spaces by progressively adding Gaussian noise to the data in a forward diffusion process and then learning to reverse this process to generate new samples (Sohl-Dickstein et al., 2015b; Ho et al., 2020; Song and Ermon, 2019). This framework is well-suited for data types such as images, where pixel values can be treated as continuous and thus can naturally accommodate additive Gaussian noise. However, when dealing with discrete data like natural language (tokens), the assumption of continuous noise addition does not hold, as it would result in non-integer values that do not correspond to valid tokens. 

Discrete diffusion models address this limitation by redefining the forward processes to transition between discrete token states, such as replacing tokens according to a transition matrix or introducing randomness through categorical distributions (Austin et al., 2021; Hoogeboom et al., 2021; Sohl-Dickstein et al., 2015a). This process enables the generation of coherent and meaningful sequences in natural language processing tasks. However, unlike continuous diffusion models, traditional scorematching techniques cannot be directly applied to learn discrete diffusion models. Instead, various surrogate losses have been proposed for training. 

In particular, Sahoo et al. introduce the Masked Diffusion Language Model (MDLM), which gradually masks and then reconstructs tokens within a text sequence, enabling efficient text generation. MDLM optimizes a continuous-time Negative ELBO (NELBO) objective to minimize the negative log-likelihood over a continuous-time diffusion process, which can be formulated as follows: 

$$
L _ {\infty} ^ {\mathrm{NELBO}} = \mathbb {E} _ {\tilde {q}} \left[ \int_ {t = 0} ^ {t = 1} \frac {(1 - \beta_ {t}) ^ {\prime}}{\beta_ {t}} \sum_ {i} \log \langle x _ {\theta} (z _ {i} ^ {t}), x _ {i} \rangle d t \right]\tag{1}
$$

where $x _ { \theta } ( z _ { i } ^ { t } )$ represents the model’s estimate of the original token $x _ { i }$ at time t given the current noisy state $z _ { i } ^ { t } .$ and $\beta _ { t }$ denotes the noise schedule controlling the diffusion process. Here, $\tilde { q }$ is the forward noising process of in the masked diffusion, defining the distribution over the noisy latent variable $z _ { i } ^ { t } ,$ and can be related to the NELBO as described in Equation (8) of Sahoo et al., 2024. The expectation $\mathbb { E } _ { \tilde { q } }$ signifies averaging over the possible outcomes of q˜ allowing the model to accound for all possible variations of z . The Noise Schedule Derivative term, $\frac { ( 1 - \beta _ { t } ) ^ { \prime } } { \beta _ { t } }$ , represents the rate of change in the noise schedule $\beta _ { t }$ over time, and t is a continuous timestep between 0 and 1. 

This loss is directly used for pretraining our draft model. As this process learns to denoise over the probability mass vectors, the output of the draft model is a matrix of $\mathbb { R } ^ { \times m }$ where $n = \gamma$ and m is the size of the vocabulary. The candidate sequence can then be generated using standard decoding methods over these probability mass vectors, the logits of which are stored to be used when determining whether to accept each draft token. 

Now, the draft logits produced by the output matrix of the discrete diffusion drafter directly substitute the autoregressive drafter used to generate $M _ { q } ( [ x _ { 0 : i } ] \circ [ x _ { i + 1 } , \dots , x _ { i + \gamma } ] )$ , where is a list concatenation operator. This substitutes the draft step taken by Leviathan et al. and Chen et al. and is the primary difference between SpecDiff and standard speculative decoding approaches. The subsequent steps of verifying this draft with the target model follow the previously proposed decoding algorithm, thus our proposal requires minimal modifications to existing speculative decoding code bases. 

A complete overview of the SpecDiff decoding is provided in Algorithm 1 (adapted from Leviathan et al.). 

Drafter’s evaluations. Next, we highlight an important difference between standard speculative decoding and our approach. While in standard speculative decoding the number of evaluations by the drafter model is dictated by the value of $\gamma$ (used in the first loop for Algorithm 1), in speculative diffusion it is dictated by the number of diffusion steps, T. This allows SpecDiff to scale γ to higher values, as discussed further in Section 6.2. 

Algorithm 1: SpecDiff Decoding

▷ Take T diffusion steps to generate the draft. $q_{i+1,\ldots,i+\gamma}^{T} \sim \mathcal{N}(\mathbf{0}, \sigma_{T}\mathbf{I})$ for t = T to 1 do $\big\lfloor q_{i+1,\ldots,i+\gamma}^{t-1}(x) \leftarrow M_{q}([x_{0:i}] \circ [q_{i+1,\ldots,i+\gamma}^{t}(x)], t)$ $x_{i+1,\ldots,i+\gamma} \sim q^{0}$ ▷ Run $M_{p}$ in parallel. $p_{i}(x), \ldots, p_{i+\gamma+1}(x) \leftarrow M_{p}(x_{0:i}), \ldots, M_{p}(x_{0:i+\gamma})$ ▷ Determine the number of accepted guesses n. $r_{i} \sim U(0,1), \ldots, r_{i+\gamma} \sim U(0,1)$ $n \leftarrow \min(\{j-1 \mid i \leq j \leq i+\gamma, r_{j} > \frac{p_{j}(x)}{q_{j}(x)}\} \cup \{\gamma\})$ ▷ Adjust the distribution from $M_{p}$ if needed. $p'(x) \leftarrow p_{n+1}(x)$ if $n < i + \gamma$ then $\big\lfloor p'(x) \leftarrow norm(max(0, p_{n+1}(x) - q_{n+1}(x)))$ ▷ Return one token from $M_{p}$ , n tokens from $M_{q}$ . $t \sim p'(x)$ return $x_{1}, \ldots, x_{n}, t$ 

Instead, the value of $T$ is selected to optimize the trade-off between draft quality and computational overhead. While analysis by Lou et al. shows that lower values of T lead to higher perplexity in the generated sequence, this only impacts SpecDiff with respect to its effect on the percentage of tokens from the draft which are accepted (as can be noted in the analysis reported in Figure 2 and discussed in details Section 6.2). 

Sequence initialization. An important consideration in implementing SpecDiff is the initial alignment between the diffusion draft model and the target model’s data distribution since these models architectures are fundamentally different one another. A key strength of SpecDiff lies in its ability to leverage the alignment between the prefixes used in discrete diffusion and the target model’s data distribution. Specifically, the better the prefixes align with the target distribution, the more effectively the diffusion drafter can generate longer, coherent sequences matching the target distribution, resulting in progressively higher acceptance rates. 

This however, also means that when the diffusion draft model has not been finetuned, its output distribution may initially differ from that of the target model, potentially leading to lower acceptance rates at the beginning of the generation process. To address this, we employ standard speculative decoding for the initial few tokens, thereby optimizing SpecDiff’s performance from the outset. This strategy ensures that SpecDiff achieves speedups that are empirically at least as significant as those observed with standard speculative decoding and can realize substantial improvements, as we show in Tables 2 and 4. Moreover, once the diffusion draft model is finetuned to better match the target distribution, this initial speculative decoding becomes unnecessary, as the drafter effectively aligns with the target model from the beginning. 

## 5 Comparative Analysis of Speculative Decoders

Table 1 compares the computational aspects of SpecDiff with state-of-the-art baselines in speculative decoding. 

According to the Work-Depth parallel computation model (Blumofe and Leiserson, 1999), an algorithm can be represented by a DAG, in which each node is an operation and each edge is a dependency. The longest shortest path in this computational DAG (i.e., the longest dependency chain) is called the depth. It follows that the average parallelism of a computation is the total number of nodes divided by the depth, and that higher depth means fewer opportunities for parallelism. We represent the operations by the FLOPs of the model. 

The table reasons about the approaches through three parameters for $M _ { p }$ and $M _ { q } \colon C$ (memory consumption), F (arithmetic/floating-point operations, or FLOPs), and D (depth). Due to the quadratic memory and computation requirements of transformers, and for simplicity, $C _ { \{ p , q \} }$ and $F _ { \{ p , q \} }$ are represented as functions of the number of tokens after the initial prompt, $\mathrm { i } . \mathrm { e } . , C _ { p } ( k )$ is the memory cost for generating $P +$ k tokens, where $P$ is the number of prompt tokens. The depth $D _ { \{ p , q \} }$ is generally independent of the number of tokens computed and is thus represented by a scalar. 

In Medusa, the prediction tree is set by a number of fixed hyperparameters (Cai et al., 2024), which we aggregate by using the number of nodes and the depth of the tree $\mathcal { T } _ { n }$ and $\mathcal { T } _ { d } .$ . Medusa-2 exhibits the same parallel properties as Medusa-1, but requires a joint fine-tuning of $M _ { p }$ along with $M _ { q }$ . Hydra is an adaptation of Medusa, in which the heads are applied in sequence. It thus only affects the depth 

of the computations. 

EAGLE uses a drafter model that contains the target embedding layer, one autoregressive layer, and the target LM head. The fixed tree size is 5 levels deep and contains 25 nodes (Li et al., 2024b). EAGLE-2 (Li et al., 2024a) dispenses with the fixed tree defined in EAGLE, and the induced dynamic tree used in generation is represented with $\tau$ 

Notice that in our method, the depth of the algorithm is dependent on $T _ { \cdot }$ , the number of diffusion steps, rather than $\gamma .$ . Empirically, $T$ is smaller than $\gamma ,$ which enables more parallelism in the computation w.r.t. the number of generated tokens. Combined with diffusion models enabling longer prediction horizons, this allows SpecDiff to produce more speculative predictions faster. 

Practically, we observe that this enables SpecDiff to produce near state-of-the-art inference acceleration while requiring significantly less computational overhead and reduced memory requirements. 

## 6 Evaluation

## 6.1 Experimental Setup

Settings. To empirically evaluate the improvements provided by using SpecDiff, the paper provides an empirical analysis on three standard natural language processing tasks: (1) text summarization using the CNN/DM dataset (Nallapati et al., 2016), (2) text generation on the OpenWebText (OWT) dataset (Gokaslan et al., 2019), and (3) text generation using MT Bench (Zheng et al., 2023). Additionally, we assess the performance of our method against the current state-of-the-art using SpecBench, a unified evaluation platform for speculative decoding techniques (Xia et al., 2024). 

In each setting, the model is queried for 1024 tokens using a greedy decoding scheme (temperature $= 0 )$ . For the experiments, we evaluate the pretrained target models $M _ { p }$ GPT-2 XL (1.5B), GPT-NEO (2.7B), and Vicuna (33B). We use Masked Diffusion Language Model (110M) as our drafter model $M _ { q }$ , which is a comparable size to the baseline drafter GPT-2 (86M) employed for our standard speculative decoding baseline (Sahoo et al., 2024). 

All evaluation is conducted on two NVIDIA A100 series GPUs (80GB) using CUDA 12.2. Additionally, FlashAttention (Dao et al., 2022) is used to optimize the performance in all experiments. 

Evaluation metrics. Our method is assessed empirically by walltime speedup, acceptance rate $\alpha ,$ 

<table><tr><td>Method</td><td>Total Memory Consumption</td><td>FLOPs</td><td>Depth</td><td>Max Tokens</td><td>Extra Training Requirements</td></tr><tr><td>Autoregressive</td><td><eq>C_p(1)</eq></td><td><eq>F_p(1)</eq></td><td><eq>D_p</eq></td><td>1</td><td>N/A</td></tr><tr><td>SpS (Leviathan et al., 2023)<eq>^1</eq></td><td><eq>C_p(\gamma) + C_q(\gamma)</eq></td><td><eq>F_p(\gamma) + F_q(\gamma)</eq></td><td><eq>D_p + \gamma D_q</eq></td><td><eq>1 + \gamma</eq></td><td><eq>M_q</eq></td></tr><tr><td>PLD (Saxena, 2023)</td><td><eq>C_p(\gamma)</eq></td><td><eq>F_p(\gamma) + \mathcal{O}(Pn)</eq></td><td><eq>D_p + \mathcal{O}(n)</eq></td><td><eq>1 + \gamma</eq></td><td>None</td></tr><tr><td>Lookahead (Fu et al., 2024)<eq>^1</eq></td><td><eq>C_p(2\gamma(n-1))</eq></td><td><eq>F_p(2\gamma(n-1)) + F_p(1)</eq></td><td><eq>D_p + \mathcal{O}(1)</eq></td><td><eq>n-1</eq></td><td>None</td></tr><tr><td>Medusa-1 (Cai et al., 2024)</td><td><eq>C_p(\mathcal{T}_n) + KC_q(\gamma)</eq></td><td><eq>F_p(\mathcal{T}_n) + K F_q(\gamma)</eq></td><td><eq>D_p + D_q</eq></td><td><eq>1 + \mathcal{T}_d</eq></td><td><eq>M_q</eq></td></tr><tr><td>Medusa-2 (Cai et al., 2024)<eq>^2</eq></td><td>—</td><td>—</td><td>—</td><td>—</td><td><eq>M_p</eq></td></tr><tr><td>Hydra (Ankner et al., 2024)<eq>^3</eq></td><td>—</td><td>—</td><td><eq>D_p + KD_q</eq></td><td>—</td><td><eq>M_q</eq></td></tr><tr><td>EAGLE (Li et al., 2024b)</td><td><eq>C_p(25) + C_q(25)</eq></td><td><eq>F_p(25) + 5F_q(5)</eq></td><td><eq>D_p + 5D_q</eq></td><td>6</td><td><eq>M_q</eq></td></tr><tr><td>EAGLE-2 (Li et al., 2024a)<eq>^4</eq></td><td><eq>C_p(\mathcal{T}_n) + C_q(\mathcal{T}_n)</eq></td><td><eq>F_p(\mathcal{T}_n) + \mathcal{T}_d F_q(\frac{\mathcal{T}_n}{\mathcal{T}_d})</eq></td><td><eq>D_p + \mathcal{T}_d D_q</eq></td><td><eq>1 + \mathcal{T}_d</eq></td><td><eq>M_q</eq></td></tr><tr><td>SpecDiff (Ours)</td><td><eq>C_p(\gamma) + C_q(\gamma)</eq></td><td><eq>F_p(\gamma) + TF_q(\gamma)</eq></td><td><eq>D_p + TD_q</eq></td><td><eq>1 + \gamma</eq></td><td><eq>M_q</eq></td></tr></table>


Table 1: Comparison of different speculative decoding strategies and their parallel properties. The quantities represent the cost of one additional decoding step (i.e., following the initial prompt). Note that $M _ { q }$ (and related quantities such as $F _ { q } )$ are not constant across methods; e.g., EAGLE uses a different draft model architecture than SpecDiff. T is the number of diffusion steps, is a sparse draft tree with $\mathcal { T } _ { n }$ nodes and depth $\mathcal { T } _ { d } .$ , K is the number of heads in a multi-head speculative decoder, P is the number of prompt tokens, n is an n-gram length, and γ is the number of proposed tokens. <sup>1</sup> (1) represents a database lookup. <sup>2</sup> Medusa-1 with target fine-tuning for Medusa-2. <sup>3</sup> Medusa with sequential draft heads for Hydra. <sup>4</sup> EAGLE with dynamic draft trees for EAGLE-2.


<table><tr><td></td><td></td><td><eq>M_p</eq></td><td><eq>M_q</eq></td><td>γ</td><td>α</td><td>Speedup</td></tr><tr><td rowspan="4">CNN/DM</td><td rowspan="2">SpS</td><td>GPT-2 XL</td><td>GPT-2</td><td>8</td><td>0.92</td><td>3.58x</td></tr><tr><td>GPT-NEO</td><td>GPT-2</td><td>9</td><td>0.95</td><td>5.45x</td></tr><tr><td rowspan="2">Ours</td><td>GPT-2 XL</td><td>MDLM</td><td>15</td><td>0.87</td><td>4.80x</td></tr><tr><td>GPT NEO</td><td>MDLM</td><td>15</td><td>0.88</td><td>6.63x</td></tr><tr><td rowspan="4">OpenWebText</td><td rowspan="2">SpS</td><td>GPT-2 XL</td><td>GPT-2</td><td>8</td><td>0.93</td><td>3.66x</td></tr><tr><td>GPT-NEO</td><td>GPT-2</td><td>7</td><td>0.85</td><td>4.12x</td></tr><tr><td rowspan="2">Ours</td><td>GPT-2 XL</td><td>MDLM</td><td>15</td><td>0.89</td><td>5.38x</td></tr><tr><td>GPT NEO</td><td>MDLM</td><td>20</td><td>0.88</td><td>7.23x</td></tr></table>


Table 2: Evaluation of walltime speedup over autoregressive decoding using SpecDiff (Ours) compared to standard speculative decoding (SpS). The best result for each setting and target model is displayed in bold.


## 6.2 Results and Discussion

and total floating point operations (FLOPs). The reported results are compared to recognized baselines of vanilla autoregressive decoding, standard speculative decoding (SpS) implementations as proposed by Leviathan et al.; Chen et al., which our method most closely resembles, and Eagle-2 (Li et al., 2024a), the state-of-the-art for speculative decoding methods. Additionally, we provide comprehensive comparison to these method, as well as the current fastest-to-date speculative decoding approaches, detailing the improvements SpecDiff provides in reduction of FLOPs and memory footprint in Table 1. 

Empirically we highlight the comparison between our approach, other speculative decoding methods, and vanilla autoregressive generation. Across the tested settings and target model architectures, SpecDiff significantly outperforms standard speculative decoding, achieving speedups of up to 7.2x compared to the target models and increasing the efficiency of standard speculative decoding by more than 1.75x. Furthermore, SpecDiff rivals the performance of current state-of-the-art methods while reducing the FLOPs/draft by over 33% and reducing memory consumption (shown in Table 1). 

Table 2 First, we compare SpecDiff to the performance of standard speculative decoding. Despite SpS generally reporting slightly higher acceptance rates, the improved parallelization provided by SpecDiff results in greater speedups ranging from 1.45–1.75 improvement over the baseline. As we have not finetuned the target model or draft model, we utilize sequence initialization with SpS to enhance generation speed by sampling the first 100 tokens of the sequence. We ablate the performance of SpecDiff without this enhancement in the Table 4. 

Table 3 To provide comparison to other state-ofthe-art speculative decoding methods, evaluation is conducted on generation prompts from MT Bench (Zheng et al., 2023) using Vicuna 33B as the target model. Generation lengths are reduced to 512 tokens to better align with the task. For this setting, MDLM (141M) is pretrained with the Llama 1 tokenizer for compatibility with the target model used. As Vicuna is instruction tuned, we utilize a teacher-student instruction tuning approach resembling the knowledge distillation approach proposed by Zhou et al. to align the output distribution of MDLM with the target model. This approach is similar to the additional instruction tuning conducted to align Eagle and Eagle-2 on ShareGPT instruction tuning dataset (Li et al., 2024b,a). Because of this finetuning step, it is not necessary to use sequence initialization as with pretrained models. Notably, SpecDiff provides a speedup 70–85% of the state-of-the-art methods, Eagle and Eagle-2, while requiring 33% fewer floating point operations per draft generation. Thorough comparison to Eagle-2 and other speculative decoding strategies is highlighted in Table 1. 

<table><tr><td rowspan="4">MT Bench</td><td></td><td><eq>M_p</eq></td><td><eq>M_q</eq></td><td>γ</td><td>α</td><td>FLOPs/draft</td><td>Task Speedup</td><td>Overall Speedup</td></tr><tr><td>EAGLE†</td><td>Vicuna 33B</td><td>AR Head (990M)</td><td>≈5</td><td>0.80</td><td><eq>2.01 \times 10^{10}</eq></td><td>2.73x</td><td>2.41x</td></tr><tr><td>EAGLE-2†</td><td>Vicuna 33B</td><td>AR Head (990M)</td><td>≈6</td><td>0.84</td><td><eq>8.35 \times 10^{10}</eq></td><td>3.03x</td><td>2.60x</td></tr><tr><td>Ours</td><td>Vicuna 33B</td><td>MDLM (141M)</td><td>15</td><td>0.76</td><td><eq>5.53 \times 10^{10}</eq></td><td>2.61x</td><td>2.61x</td></tr></table>


Table 3: Evaluation on MT Bench using SpecDiff (Ours), EAGLE, EAGLE-2, and SpS with Vicuna target models. FLOPs/step computes that floating point operations for the drafter to generate each sequence. ‘Overall Speedups’ for the baselines are measured on all Spec-Bench (Xia et al., 2024) tasks. † denotes results collected using Spec-Bench.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/96331fd7-c2e9-46c8-ad96-9ab1675dcbd7/011dfb1200fc23f741b60b42a33d904989085ddb55cb98369ab22e0bcf35aa44.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/96331fd7-c2e9-46c8-ad96-9ab1675dcbd7/08d32a369b2ab92dea8b26f0559872512fbb42c6d6d835882a2f9d3071048faf.jpg)



Figure 2: Evaluation of SpecDiff’s sensitivity to γ and number of diffusion steps when optimizing speed (left) and accepted tokens per draft (right) as reported on the OpenWebText task using GPT-2 NEO as the target model. Average token acceptance increases with T (x-axis), as examined theoretically in Appendix A. Despite this, the additional compute required as T is scaled results in reduced speedup.


Table 4 While we observe a slight decrease in SpecDiff’s speed when standard speculative decoding is not used to initialize the generation, the change in α is most noticeable when comparing to 

<table><tr><td></td><td></td><td><eq>M_p</eq></td><td><eq>M_q</eq></td><td>γ</td><td>α</td><td>Speedup</td></tr><tr><td rowspan="2">CNN</td><td rowspan="2">Ours</td><td>GPT-2 XL</td><td>MDLM</td><td>10</td><td>0.75</td><td>3.82x</td></tr><tr><td>GPT NEO</td><td>MDLM</td><td>15</td><td>0.77</td><td>5.32x</td></tr><tr><td rowspan="2">OWT</td><td rowspan="2">Ours</td><td>GPT-2 XL</td><td>MDLM</td><td>15</td><td>0.78</td><td>4.06x</td></tr><tr><td>GPT NEO</td><td>MDLM</td><td>15</td><td>0.82</td><td>6.19x</td></tr></table>


Table 4: Performance of a “vanilla” version of SpecDiff, that does not use SpS initialization.


Table 2. This reflects the poor acceptance rates at the beginning of the generation. While SpecDiff does outperform SpS in this setting, providing up to 1.5x speedups over SpS, this highlights the benefit of adjointly using SpecDiff and other speculative decoding methods. Such hybrid approaches can be particularly effective for shorter generations. 

While previous implementations of speculative decoding rely on a common architecture between the drafter and target models (Leviathan et al., 2023; Chen et al., 2023), using smaller versions of the same architecture to generate draft sequences, these experiments demonstrate a robustness to using a completely different architecture for sequence drafting. This is particularly significant given the absence of finetuning in the reported results (Tables 2 and 4). Pretrained diffusion models can be directly purposed as draft models requiring no additional training. 

The much larger values of γ used for SpecDiff should particularly be highlighted. This is a key discrepancy between diffusion language models, which generate entire sequences in parallel, and autoregressive models. Hence, there is minimal overhead to increasing the sequence length generated by the diffusion-based drafter, and γ can be significantly increased without incurring significant cost. 

The hyperparameters used in the reported results have been optimized empirically. We highlight that while in standard speculative diffusion the performance is highly sensitive to γ, SpecDiff is robust to a range of values for γ making it unnecessary to precisely tune this hyperparameter (in our experiments we found between 10 and 20 worked well). Rather, SpecDiff’s performance is much more sensitive to the number of diffusion steps selected. Similar to the role of γ in an autoregressive model, the number of diffusion steps T dictates the number of network evaluations during a single drafting step. As reported in the Figure 2, while increasing this hyperparameter arbitrarily results in higher values of α, SpecDiff performs best when this is optimized to balance the objectives of maximizing the number of accepted tokens and minimizing the drafter’s overhead. 

## 7 Conclusion

Motivated by the costly inference time of current large language models, this paper has proposed the novel integration of discrete diffusion models with autoregressive language models. The proposed method, Speculative Diffusion Decoding, alters existing speculative decoding schemes to integrate a non-autoregressive diffusion model as the draft model. As shown by the empirical evaluation on standard language generation benchmarks, the proposed method leverages the dramatic runtime advantages of discrete diffusion models while also maintaining the dramatically higher generation quality of autoregressive target models. The reported results demonstrate the utility of this approach in effectively accelerating runtime, outperforming vanilla decoding by over 7x and speculative decoding methods by over 1.75x. 

## 8 Limitations

While our proposed Speculative Diffusion Decoding method represents a significant step forward in accelerating large language model inference, several limitations warrant discussion. Addressing these limitations highlights promising avenues for future research that could further enhance the efficiency and applicability of SpecDiff. 

Calibration of discrete diffusion models. A primary limitation of our approach lies in the challenge of using different architectures for the drafter and verifier models. Specifically, the adopted discrete diffusion models do not output well-calibrated probability distributions that align with those of the target autoregressive models, particularly when the sampling temperature (T) is greater than zero. The diffusion models tend to produce over-confident predictions, often assigning near-certain probability to the top-1 token while assigning negligible probabilities to all other tokens. This results in deterministic sampling regardless of the temperature setting, challenging the applicability of SpecDiff in scenarios where diversity and stochasticity in generation are desired. 

Thus, the development of techniques to better align the output probabilities of diffusion drafters with the target models is an key area of future work. Achieving proper calibration would enable effective use of SpecDiff at higher temperatures, as well as unlocking massive further speedups beyond what we have reported, as the acceptance rates are predicted to increase dramatically. 

Limited tokenization availability. We note that available discrete diffusion models are based on the GPT-2 tokenizer, which would restrict one immediate compatibility with target models using different tokenization schemes. For our experiments, we indeed trained a new discrete diffusion model from scratch with a different tokenizer, a process that demanded substantial computational resources and time. All our models will be released on Huggin-Face and thus be directly used by the community. 

Performance on shorter generation tasks. We observe that SpecDiff exhibits optimal performance on longer sequence generations. In shorter generation tasks, the benefits of parallelization are less pronounced, and finetuning the diffusion drafter may be necessary to achieve comparable efficiency gains. Without finetuning, the drafter may not effectively capture the target model’s token distributions for shorter sequences. 

An interesting outcome of these observations is that tailoring the drafter to better model shorter sequences, could improve its alignment with the target model, thereby maintaining speedups even in less extensive generation tasks. We believe that this adjustment may broaden SpecDiff’s applicability. 

Stochastic sampling. Note that our experiments are conducted with the sampling temperature set to zero, resulting in deterministic token generation. While this setting simplifies the verification process it limits the exploration of the model’s capabilities in generating diverse outputs. In the future, we plan to exploring SpecDiff’s performance at non-zero temperatures. As highlighted earlier, addressing the calibration issue of the diffusion drafter would enable effective stochastic sampling. 

Despite these challenges, our work lays the foundational framework for integrating discrete diffusion models with autoregressive models in speculative decoding. Each limitation discussed highlights a specific area where further research could yield significant benefits for the performance of the proposed SpecDiff. We are optimistic that overcoming these challenges will not only reinforce the strengths of SpecDiff but also unlock new possibilities for accelerating language model inference. 

## 9 Acknowledgements

This work is partially supported by NSF awards 2143706, 2232054, and 2242931. This work was performed under the auspices of the U.S. Department of Energy by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344 and was supported by the LLNL-LDRD Program under Project No. 24-ERD-010 (LLNL-CONF-2000386). This research used resources of the National Energy Research Scientific Computing Center (NERSC), a Department of Energy Office of Science User Facility using NERSC award ASCR-ERCAP0027427. The views and conclusions provided in this paper reflect those of the authors only. 

## References



Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. 





Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon. 2024. Hydra: Sequentially-dependent draft heads for medusa decoding. Preprint, arXiv:2402.05109. 





John Austin, David D. Johnson, Jonathan Ho, David Tarlow, and Remco van den Berg. 2021. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, volume 34, pages 17981–17993. 





Robert D. Blumofe and Charles E. Leiserson. 1999. Scheduling multithreaded computations by work stealing. J. ACM, 46(5):720–748. 





Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901. 





Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. Medusa: Simple llm inference acceleration framework with multiple decoding heads. Preprint, arXiv:2401.10774. 





Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318. 





Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359. 





Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. 2024. Break the sequential dependency of llm inference using lookahead decoding. arXiv preprint arXiv:2402.02057. 





Gemini Team. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805. 





Marjan Ghazvininejad, Omer Levy, Yinhan Liu, and Luke Zettlemoyer. 2019. Mask-predict: Parallel decoding of conditional masked language models. arXiv preprint arXiv:1904.09324. 





Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. 2024. Better & faster large language models via multi-token prediction. Preprint, arXiv:2404.19737. 





Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. 2019. Openwebtext corpus. 





Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840– 6851. 





Junyuan Hong, Jinhao Duan, Chenhui Zhang, LI Zhangheng, Chulin Xie, Kelsey Lieberman, James Diffenderfer, Brian R Bartoldson, AJAY KUMAR JAISWAL, Kaidi Xu, et al. 2024. Decoding com pressed trust: Scrutinizing the trustworthiness of efficient llms under compression. In Forty-first International Conference on Machine Learning. 





Erik Hoogeboom, Niels Bonnier, Tal Schuster Cohen, and Max Welling. 2021. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, volume 34, pages 12454–12465. 





Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361. 





Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR. 





Gen Li, Yuting Wei, Yuxin Chen, and Yuejie Chi. 2023. Towards faster non-asymptotic convergence for diffusion-based generative models. arXiv preprint arXiv:2306.09251. 





Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024a. EAGLE-2: Faster inference of language models with dynamic draft trees. In Empirical Methods in Natural Language Processing. 





Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024b. EAGLE: Speculative sampling requires rethinking feature uncertainty. In Interna tional Conference on Machine Learning. 





Llama Team. 2024. The llama 3 herd of models. 





Aaron Lou, Chenlin Meng, and Stefano Ermon. 2024. Discrete diffusion modeling by estimating the ratios of the data distribution. In Forty-first International Conference on Machine Learning. 





Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Hongyi Jin, Tianqi Chen, and Zhihao Jia. 2023a. Towards efficient generative large language model serving: A survey from algorithms to systems. arXiv preprint arXiv:2312.15234. 





Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. 2023b. Specinfer: Accelerating generative large language model serving with tree-based speculative inference and verification. arXiv preprint arXiv:2305.09781. 





Ramesh Nallapati, Bowen Zhou, Caglar Gulcehre, Bing Xiang, et al. 2016. Abstractive text summarization using sequence-to-sequence rnns and beyond. arXiv preprint arXiv:1602.06023. 



OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal Lama Ahmad, Ilge Akkaya, Florencia Leoni Ale man, Diogo Almeida, Janko Altenschmidt, Sam Alt man, Shyamal Anadkat, Red Avila, Igor Babuschkin Suchir Balaji, Valerie Balcom, Paul Baltescu, Haim ing Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brock man, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai Cory Decareaux, Thomas Degry, Noah Deutsch Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Ful ford, Leo Gao, Elie Georges, Christian Gibson, Vi Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo Lopes, Jonathan Gordon, Morgan Grafstein, Scot Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Hee woo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Ka mali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirch ner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Kon stantinidis, Kyle Kosic, Gretchen Krueger, Visha Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Maver, Andrew Mayne. Bob McGrew, Scott Mavei McKinney, Christine McLeavey, Paul McMillan Jake McNeil David Medina, Aalok Mehta, Jacok Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Danie Mossing Tong Mu. Mira Murati, Oleg Murk, Davic Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambat tista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perel man, Filipe de Avila Belbute Peres, Michael Petrov Henrique Ponde de Oliveira Pinto, Michael, Poko rny, Michelle Pokrass, Vitchyr H. Pong, Tolly Pow ell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh Cameron Raymond, Francis Real, Kendra Rimbach Carl Ross, Bob Rotsted, Henri Roussez, Nick Ry 



der, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774. 





Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9. 





Laura Ruis, Akbir Khan, Stella Biderman, Sara Hooker, Tim Rocktäschel, and Edward Grefenstette. 2023. The goldilocks of pragmatic understanding: Finetuning strategy matters for implicature resolution by llms. Preprint, arXiv:2210.14986. 





Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. 2024. Simple and effective masked diffusion language models. arXiv preprint arXiv:2406.07524. 





Apoorv Saxena. 2023. Prompt lookup decoding. 





Jascha Sohl-Dickstein, Eric Weiss, Nikhil Maheswaranathan, and Surya Ganguli. 2015a. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. 





Jascha Sohl-Dickstein, Eric Weiss, Niru Mah eswaranathan, and Surya Ganguli. 2015b. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR. 





Yang Song and Stefano Ermon. 2019. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems, 32. 





Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31. 





Ziteng Sun, Ananda Theertha Suresh, Jae Hun Ro, Ahmad Beirami, Himanshu Jain, and Felix Yu. 2024. Spectr: Fast speculative decoding via optimal transport. Preprint, arXiv:2310.15141. 





Ruslan Svirschevski, Avner May, Zhuoming Chen, Beidi Chen, Zhihao Jia, and Max Ryabinin. 2024. Specexec: Massively parallel speculative decoding for interactive llm inference on consumer devices. arXiv preprint arXiv:2406.02532. 



Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Ange liki Lazaridou, Orhan Firat, James Molloy, Michae Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ryan Doherty, Eli Collins, Clemens Meyer, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Jack Krawczyk, Cosmo Du, Ed Chi, Heng Tze Cheng, Eric Ni, Purvi Shah, Patrick Kane, Betty Chan, Manaal Faruqui, Aliaksei Severyn, Hanzhao Lin, YaGuang Li, Yong Cheng, Abe Ittycheriah, Mahdis Mahdieh, Mia Chen, Pei Sun, Dustin Tran, Sumit Bagri, Balaji Lakshminarayanan, Jeremiah Liu, Andras Orban, Fabian Güra, Hao Zhou, Xiny ing Song, Aurelien Boffy, Harish Ganapathy, Steven Zheng, HyunJeong Choe, Ágoston Weisz, Tao Zhu, Yifeng Lu, Siddharth Gopal, Jarrod Kahn, Maciej Kula, Jeff Pitman, Rushin Shah, Emanuel Taropa, Majd Al Merey, Martin Baeuml, Zhifeng Chen, Lau rent El Shafey, Yujing Zhang, Olcan Sercinoglu George Tucker, Enrique Piqueras, Maxim Krikun, Iain Barr, Nikolay Savinov, Ivo Danihelka, Becca Roelofs, Anaïs White, Anders Andreassen, Tamara von Glehn, Lakshman Yagati, Mehran Kazemi, Lu cas Gonzalez, Misha Khalman, Jakub Sygnowski, Alexandre Frechette, Charlotte Smith, Laura Culp, Lev Proleev, Yi Luan, Xi Chen, James Lottes, Nathan Schucher, Federico Lebron, Alban Rrustemi, Na talie Clay, Phil Crone, Tomas Kocisky, Jeffrey Zhao, Bartek Perz, Dian Yu, Heidi Howard, Adam Blo niarz, Jack W. Rae, Han Lu, Laurent Sifre, Mar cello Maggioni, Fred Alcober, Dan Garrette, Megan Barnes, Shantanu Thakoor, Jacob Austin, Gabrie Barth-Maron, William Wong, Rishabh Joshi, Rahma Chaabouni, Deeni Fatiha, Arun Ahuja, Gaurav Singh Tomar, Evan Senter, Martin Chadwick, Ilya Kor nakov, Nithya Attaluri, Iñaki Iturrate, Ruibo Liu, Yunxuan Li, Sarah Cogan, Jeremy Chen, Chao Jia, Chenjie Gu, Qiao Zhang, Jordan Grimstad, Ale Jakse Hartman, Xavier Garcia, Thanumalayan Sankara narayana Pillai, Jacob Devlin, Michael Laskin, Diego de Las Casas, Dasha Valter, Connie Tao, Lorenzo Blanco, Adrià Puigdomènech Badia, David Reitter, Mianna Chen, Jenny Brennan, Clara Rivera, Sergey Brin, Shariq Iqbal, Gabriela Surita, Jane Labanowski, Abhi Rao, Stephanie Winkler, Emilio Parisotto, Yim ing Gu, Kate Olszewska, Ravi Addanki, Antoine Miech, Annie Louis, Denis Teplyashin, Geoff Brown, Elliot Catt, Jan Balaguer, Jackie Xiang, Pidong Wang, 

Zoe Ashwood, Anton Briukhov, Albert Webson, San jay Ganapathy, Smit Sanghavi, Ajay Kannan, Ming Wei Chang, Axel Stjerngren, Josip Djolonga, Yut ing Sun, Ankur Bapna, Matthew Aitchison, Pedram Pejman, Henryk Michalewski, Tianhe Yu, Cindy Wang, Juliette Love, Junwhan Ahn, Dawn Bloxwich, Kehang Han, Peter Humphreys, Thibault Sellam, James Bradbury, Varun Godbole, Sina Samangooei, Bogdan Damoc, Alex Kaskasoli, Sébastien M. R. Arnold, Vijay Vasudevan, Shubham Agrawal, Jason Riesa, Dmitry Lepikhin, Richard Tanburn, Srivat san Srinivasan, Hyeontaek Lim, Sarah Hodkinson, Pranav Shyam, Johan Ferret, Steven Hand, Ankush Garg, Tom Le Paine, Jian Li, Yujia Li, Minh Gi ang, Alexander Neitz, Zaheer Abbas, Sarah York, Machel Reid, Elizabeth Cole, Aakanksha Chowdh ery, Dipanjan Das, Dominika Rogozinska, Vitaliy´ Nikolaev, Pablo Sprechmann, Zachary Nado, Lukas Zilka, Flavien Prost, Luheng He, Marianne Mon teiro, Gaurav Mishra, Chris Welty, Josh Newlan, Dawei Jia, Miltiadis Allamanis, Clara Huiyi Hu Raoul de Liedekerke, Justin Gilmer, Carl Saroufim, Shruti Rijhwani, Shaobo Hou, Disha Shrivastava, Anirudh Baddepudi, Alex Goldin, Adnan Ozturel, Albin Cassirer, Yunhan Xu, Daniel Sohn, Deven dra Sachan, Reinald Kim Amplayo, Craig Swan son, Dessie Petrova, Shashi Narayan, Arthur Guez, Siddhartha Brahma, Jessica Landon, Miteyan Pa tel, Ruizhe Zhao, Kevin Villela, Luyu Wang, Wen hao Jia, Matthew Rahtz, Mai Giménez, Legg Yeung, James Keeling, Petko Georgiev, Diana Mincu, Box Wu, Salem Haykal, Rachel Saputro, Kiran Vodra halli, James Qin, Zeynep Cankara, Abhanshu Sharma, Nick Fernando, Will Hawkins, Behnam Neyshabur, Solomon Kim, Adrian Hutter, Priyanka Agrawal, Alex Castro-Ros, George van den Driessche, Tao Wang, Fan Yang, Shuo yiin Chang, Paul Komarek, Ross McIlroy, Mario Luciˇ c, Guodong Zhang, Wae´ Farhan, Michael Sharman, Paul Natsev, Paul Michel Yamini Bansal, Siyuan Qiao, Kris Cao, Siamak Shak eri, Christina Butterfield, Justin Chung, Paul Kishan Rubenstein, Shivani Agrawal, Arthur Mensch, Kedar Soparkar, Karel Lenc, Timothy Chung, Aedan Pope, Loren Maggiore, Jackie Kay, Priya Jhakra, Shibo Wang, Joshua Maynez, Mary Phuong, Taylor Tobin, Andrea Tacchetti, Maja Trebacz, Kevin Robinson, Yash Katariya, Sebastian Riedel, Paige Bailey, Kefan Xiao, Nimesh Ghelani, Lora Aroyo, Ambrose Slone, Neil Houlsby, Xuehan Xiong, Zhen Yang, Elena Gri bovskaya, Jonas Adler, Mateo Wirth, Lisa Lee, Music Li, Thais Kagohara, Jay Pavagadhi, Sophie Bridgers, Anna Bortsova, Sanjay Ghemawat, Zafarali Ahmed, Tianqi Liu, Richard Powell, Vijay Bolina, Mariko Iinuma, Polina Zablotskaia, James Besley, Da-Woon Chung, Timothy Dozat, Ramona Comanescu, Xi ance Si, Jeremy Greer, Guolong Su, Martin Polacek Raphaël Lopez Kaufman, Simon Tokumine, Hexiang Hu, Elena Buchatskaya, Yingjie Miao, Mohamed Elhawaty, Aditya Siddhant, Nenad Tomasev, Jin wei Xing, Christina Greer, Helen Miller, Shereen Ashraf, Aurko Roy, Zizhao Zhang, Ada Ma, Ange los Filos, Milos Besta, Rory Blevins, Ted Klimenko, Chih-Kuan Yeh, Soravit Changpinyo, Jiaqi Mu, Os car Chang, Mantas Pajarskas, Carrie Muir, Vered 

Cohen, Charline Le Lan, Krishna Haridasan, Amit Marathe, Steven Hansen, Sholto Douglas, Rajku mar Samuel, Mingqiu Wang, Sophia Austin, Chang Lan, Jiepu Jiang, Justin Chiu, Jaime Alonso Lorenzo, Lars Lowe Sjösund, Sébastien Cevey, Zach Gle icher, Thi Avrahami, Anudhyan Boral, Hansa Srini vasan, Vittorio Selo, Rhys May, Konstantinos Aiso pos, Léonard Hussenot, Livio Baldini Soares, Kate Baumli, Michael B. Chang, Adrià Recasens, Ben Caine, Alexander Pritzel, Filip Pavetic, Fabio Pardo, Anita Gergely, Justin Frye, Vinay Ramasesh, Dan Horgan, Kartikeya Badola, Nora Kassner, Subhra jit Roy, Ethan Dyer, Víctor Campos Campos, Alex Tomala, Yunhao Tang, Dalia El Badawy, Elspeth White, Basil Mustafa, Oran Lang, Abhishek Jin dal, Sharad Vikram, Zhitao Gong, Sergi Caelles, Ross Hemsley, Gregory Thornton, Fangxiaoyu Feng, Wojciech Stokowiec, Ce Zheng, Phoebe Thacker, Çaglar Ünlü, Zhishuai Zhang, Mohammad Saleh˘ James Svensson, Max Bileschi, Piyush Patil, Ankesh Anand, Roman Ring, Katerina Tsihlas, Arpi Vezer, Marco Selvi, Toby Shevlane, Mikel Rodriguez, Tom Kwiatkowski, Samira Daruki, Keran Rong, Allan Dafoe, Nicholas FitzGerald, Keren Gu-Lemberg, Mina Khan, Lisa Anne Hendricks, Marie Pellat, Vladimir Feinberg, James Cobon-Kerr, Tara Sainath Maribeth Rauh, Sayed Hadi Hashemi, Richard Ives, Yana Hasson, Eric Noland, Yuan Cao, Nathan Byrd, Le Hou, Qingze Wang, Thibault Sottiaux, Michela Paganini, Jean-Baptiste Lespiau, Alexandre Mou farek, Samer Hassan, Kaushik Shivakumar, Joost van Amersfoort, Amol Mandhane, Pratik Joshi, Anirudh Goyal, Matthew Tung, Andrew Brock, Hannah Shea han, Vedant Misra, Cheng Li, Nemanja Rakicevi´ c,´ Mostafa Dehghani, Fangyu Liu, Sid Mittal, Jun hyuk Oh, Seb Noury, Eren Sezener, Fantine Huot Matthew Lamm, Nicola De Cao, Charlie Chen, Sid harth Mudgal, Romina Stella, Kevin Brooks, Gau tam Vasudevan, Chenxi Liu, Mainak Chain, Nivedita Melinkeri, Aaron Cohen, Venus Wang, Kristie Sey more, Sergey Zubkov, Rahul Goel, Summer Yue, Sai Krishnakumaran, Brian Albert, Nate Hurley, Motoki Sano, Anhad Mohananey, Jonah Joughin, Egor Filonov, Tomasz K˛epa, Yomna Eldawy, Jiaw ern Lim, Rahul Rishi, Shirin Badiezadegan, Taylor Bos, Jerry Chang, Sanil Jain, Sri Gayatri Sundara Padmanabhan, Subha Puttagunta, Kalpesh Krishna Leslie Baker, Norbert Kalb, Vamsi Bedapudi, Adam Kurzrok, Shuntong Lei, Anthony Yu, Oren Litvin Xiang Zhou, Zhichun Wu, Sam Sobell, Andrea Si ciliano, Alan Papir, Robby Neale, Jonas Bragagnolo, Tej Toor, Tina Chen, Valentin Anklin, Feiran Wang, Richie Feng, Milad Gholami, Kevin Ling, Lijuan Liu, Jules Walter, Hamid Moghaddam, Arun Kishore, Jakub Adamek, Tyler Mercado, Jonathan Mallinson, Siddhinita Wandekar, Stephen Cagle, Eran Ofek, Guillermo Garrido, Clemens Lombriser, Maksim Mukha, Botu Sun, Hafeezul Rahman Mohammad Josip Matak, Yadi Qian, Vikas Peswani, Pawel Janus Quan Yuan, Leif Schelin, Oana David, Ankur Garg, Yifan He, Oleksii Duzhyi, Anton Älgmyr, Timo thée Lottaz, Qi Li, Vikas Yadav, Luyao Xu, Alex Chinien, Rakesh Shivanna, Aleksandr Chuklin, Josie Li, Carrie Spadine, Travis Wolfe, Kareem Mohamed, 

Subhabrata Das, Zihang Dai, Kyle He, Daniel von Dincklage, Shyam Upadhyay, Akanksha Maurya Luyan Chi, Sebastian Krause, Khalid Salama, Pam G Rabinovitch, Pavan Kumar Reddy M, Aarush Sel van, Mikhail Dektiarev, Golnaz Ghiasi, Erdem Gu ven, Himanshu Gupta, Boyi Liu, Deepak Sharma, Idan Heimlich Shtacher, Shachi Paul, Oscar Aker lund, François-Xavier Aubet, Terry Huang, Chen Zhu, Eric Zhu, Elico Teixeira, Matthew Fritze, Francesco Bertolini, Liana-Eleonora Marinescu, Mar tin Bölle, Dominik Paulus, Khyatti Gupta, Tejasi Latkar, Max Chang, Jason Sanders, Roopa Wil son, Xuewei Wu, Yi-Xuan Tan, Lam Nguyen Thiet, Tulsee Doshi, Sid Lall, Swaroop Mishra, Wanming Chen, Thang Luong, Seth Benjamin, Jasmine Lee Ewa Andrejczuk, Dominik Rabiej, Vipul Ranjan, Krzysztof Styrc, Pengcheng Yin, Jon Simon, Mal colm Rose Harriott, Mudit Bansal, Alexei Robsky Geoff Bacon, David Greene, Daniil Mirylenka, Chen Zhou, Obaid Sarvana, Abhimanyu Goyal, Samue Andermatt, Patrick Siegler, Ben Horn, Assaf Is rael, Francesco Pongetti, Chih-Wei "Louis" Chen, Marco Selvatici, Pedro Silva, Kathie Wang, Jack son Tolins, Kelvin Guu, Roey Yogev, Xiaochen Cai Alessandro Agostini, Maulik Shah, Hung Nguyen Noah Ó Donnaile, Sébastien Pereira, Linda Friso, Adam Stambler, Adam Kurzrok, Chenkai Kuang Yan Romanikhin, Mark Geller, ZJ Yan, Kane Jang, Cheng-Chun Lee, Wojciech Fica, Eric Malmi, Qi jun Tan, Dan Banica, Daniel Balle, Ryan Pham Yanping Huang, Diana Avram, Hongzhi Shi, Jasjo Singh, Chris Hidey, Niharika Ahuja, Pranab Sax ena, Dan Dooley, Srividya Pranavi Potharaju, Eileen O’Neill, Anand Gokulchandran, Ryan Foley, Ka Zhao, Mike Dusenberry, Yuan Liu, Pulkit Mehta Ragha Kotikalapudi, Chalence Safranek-Shrader, An drew Goodman, Joshua Kessinger, Eran Globen, Pra teek Kolhar, Chris Gorgolewski, Ali Ibrahim, Yang Song, Ali Eichenbaum, Thomas Brovelli, Sahitya Potluri, Preethi Lahoti, Cip Baetu, Ali Ghorbani, Charles Chen, Andy Crawford, Shalini Pal, Mukund Sridhar, Petru Gurita, Asier Mujika, Igor Petrovski Pierre-Louis Cedoz, Chenmei Li, Shiyuan Chen Niccolò Dal Santo, Siddharth Goyal, Jitesh Pun jabi, Karthik Kappaganthu, Chester Kwak, Pallavi LV, Sarmishta Velury, Himadri Choudhury, Jamie Hall, Premal Shah, Ricardo Figueira, Matt Thomas, Minjie Lu, Ting Zhou, Chintu Kumar, Thomas Ju rdi, Sharat Chikkerur, Yenai Ma, Adams Yu, Soo Kwak, Victor Ähdel, Sujeevan Rajayogam, Travis Choma Fei Liu Aditva Barua Colin Ji Ji Hd Park, Vincent Hellendoorn, Alex Bailey, Taylan Bi lal, Huanjie Zhou, Mehrdad Khatir, Charles Sut ton, Wojciech Rzadkowski, Fiona Macintosh, Kon stantin Shagin, Paul Medina, Chen Liang, Jinjing Zhou, Pararth Shah, Yingying Bi, Attila Dankovics, Shipra Banga, Sabine Lehmann, Marissa Bredesen, Zifan Lin, John Eric Hoffmann, Jonathan Lai, Ray nald Chung, Kai Yang, Nihal Balani, Arthur Bražin skas, Andrei Sozanschi, Matthew Hayes, Héctor Fer nández Alcalde, Peter Makarov, Will Chen, Anto nio Stella, Liselotte Snijders, Michael Mandl, Ante Kärrman, Paweł Nowak, Xinyi Wu, Alex Dyck, Kr ishnan Vaidyanathan, Raghavender R, Jessica Mal let, Mitch Rudominer, Eric Johnston, Sushil Mit tal, Akhil Udathu, Janara Christensen, Vishal Verma Zach Irving, Andreas Santucci, Gamaleldin Elsayed, Elnaz Davoodi, Marin Georgiev, Ian Tenney, Nan Hua, Geoffrey Cideron, Edouard Leurent, Mah moud Alnahlawi, Ionut Georgescu, Nan Wei, Ivy Zheng, Dylan Scandinaro, Heinrich Jiang, Jasper Snoek, Mukund Sundararajan, Xuezhi Wang, Zack Ontiveros, Itay Karo, Jeremy Cole, Vinu Rajashekhar, Lara Tumeh, Eyal Ben-David, Rishub Jain, Jonathan Uesato, Romina Datta, Oskar Bunyan, Shimu Wu, John Zhang, Piotr Stanczyk, Ye Zhang, David Steiner, Subhajit Naskar, Michael Azzam, Matthew Johnson, Adam Paszke, Chung-Cheng Chiu, Jaume Sanchez Elias, Afroz Mohiuddin, Faizan Muhammad, Jin Miao, Andrew Lee, Nino Vieillard, Jane Park, Ji ageng Zhang, Jeff Stanway, Drew Garmon, Abhiji Karmarkar, Zhe Dong, Jong Lee, Aviral Kumar, Lu owei Zhou, Jonathan Evens, William Isaac, Geoffrey Irving, Edward Loper, Michael Fink, Isha Arkatkar, Nanxin Chen, Izhak Shafran, Ivan Petrychenko, Zhe Chen, Johnson Jia, Anselm Levskaya, Zhenka Zhu, Peter Grabowski, Yu Mao, Alberto Magni, Kaisheng Yao, Javier Snaider, Norman Casagrande, Evan Palmer, Paul Suganthan, Alfonso Castaño, Irene Giannoumis, Wooyeol Kim, Mikołaj Rybinski,´ Ashwin Sreevatsa, Jennifer Prendki, David Soergel, Adrian Goedeckemeyer, Willi Gierke, Mohsen Jafari Meenu Gaba, Jeremy Wiesner, Diana Gage Wright, Yawen Wei, Harsha Vashisht, Yana Kulizhskaya, Jay Hoover, Maigo Le, Lu Li, Chimezie Iwuanyanwu, Lu Liu, Kevin Ramirez, Andrey Khorlin, Albert Cui, Tian LIN, Marcus Wu, Ricardo Aguilar, Keith Pallo, Abhishek Chakladar, Ginger Perng, Elena Al lica Abellan, Mingyang Zhang, Ishita Dasgupta, Nate Kushman, Ivo Penchev, Alena Repina, Xihui Wu, Tom van der Weide, Priya Ponnapalli, Car oline Kaplan, Jiri Simsa, Shuangfeng Li, Olivier Dousse, Fan Yang, Jeff Piper, Nathan Ie, Rama Pa sumarthi, Nathan Lintz, Anitha Vijayakumar, Danie Andor, Pedro Valenzuela, Minnie Lui, Cosmin Padu raru, Daiyi Peng, Katherine Lee, Shuyuan Zhang, Somer Greene, Duc Dung Nguyen, Paula Kurylow icz, Cassidy Hardin, Lucas Dixon, Lili Janzer, Kiam Choo, Ziqiang Feng, Biao Zhang, Achintya Sing hal, Dayou Du, Dan McKinnon, Natasha Antropova, Tolga Bolukbasi, Orgad Keller, David Reid, Danie Finchelstein, Maria Abi Raad, Remi Crocker, Pe ter Hawkins, Robert Dadashi, Colin Gaffney, Ken Franko, Anna Bulanova, Rémi Leblond, Shirley Chung, Harry Askham, Luis C. Cobo, Kelvin Xu, Felix Fischer, Jun Xu, Christina Sorokin, Chris Al berti, Chu-Cheng Lin, Colin Evans, Alek Dimitriev, Hannah Forbes, Dylan Banarse, Zora Tung, Mark Omernick, Colton Bishop, Rachel Sterneck, Rohan Jain, Jiawei Xia, Ehsan Amid, Francesco Piccinno, Xingyu Wang, Praseem Banzal, Daniel J. Mankowitz, Alex Polozov, Victoria Krakovna, Sasha Brown, Mo hammadHossein Bateni, Dennis Duan, Vlad Firoiu, Meghana Thotakuri, Tom Natan, Matthieu Geist, Ser tan Girgin, Hui Li, Jiayu Ye, Ofir Roval, Reiko Tojo, Michael Kwong, James Lee-Thorp, Christo pher Yew, Danila Sinopalnikov, Sabela Ramos, John 

Mellor, Abhishek Sharma, Kathy Wu, David Miller, Nicolas Sonnerat, Denis Vnukov, Rory Greig, Jen nifer Beattie, Emily Caveness, Libin Bai, Julian Eisenschlos, Alex Korchemniy, Tomy Tsai, Mimi Jasarevic, Weize Kong, Phuong Dao, Zeyu Zheng Frederick Liu, Fan Yang, Rui Zhu, Tian Huey Teh, Jason Sanmiya, Evgeny Gladchenko, Nejc Trdin, Daniel Toyama, Evan Rosen, Sasan Tavakkol, Lint ing Xue, Chen Elkind, Oliver Woodman, John Car penter, George Papamakarios, Rupert Kemp, Sushant Kafle, Tanya Grunina, Rishika Sinha, Alice Tal bert, Diane Wu, Denese Owusu-Afriyie, Cosmo Du, Chloe Thornton, Jordi Pont-Tuset, Pradyumna Narayana, Jing Li, Saaber Fatehi, John Wieting, Omar Ajmeri, Benigno Uria, Yeongil Ko, Laura Knight, Amélie Héliou, Ning Niu, Shane Gu, Chenxi Pang, Yeqing Li, Nir Levine, Ariel Stolovich, Re beca Santamaria-Fernandez, Sonam Goenka, Wenny Yustalim, Robin Strudel, Ali Elqursh, Charlie Deck, Hyo Lee, Zonglin Li, Kyle Levin, Raphael Hoff mann, Dan Holtmann-Rice, Olivier Bachem, Sho Arora, Christy Koh, Soheil Hassas Yeganeh, Siim Põder, Mukarram Tariq, Yanhua Sun, Lucian Ionita, Mojtaba Seyedhosseini, Pouya Tafti, Zhiyu Liu, An mol Gulati, Jasmine Liu, Xinyu Ye, Bart Chrzaszcz, Lily Wang, Nikhil Sethi, Tianrun Li, Ben Brown, Shreya Singh, Wei Fan, Aaron Parisi, Joe Stan ton, Vinod Koverkathu, Christopher A. Choquette Choo, Yunjie Li, TJ Lu, Abe Ittycheriah, Prakash Shroff, Mani Varadarajan, Sanaz Bahargam, Rob Willoughby, David Gaddy, Guillaume Desjardins, Marco Cornero, Brona Robenek, Bhavishya Mit tal, Ben Albrecht, Ashish Shenoy, Fedor Moiseev, Henrik Jacobsson, Alireza Ghaffarkhah, Morgane Rivière, Alanna Walton, Clément Crepy, Alicia Par rish, Zongwei Zhou, Clement Farabet, Carey Rade baugh, Praveen Srinivasan, Claudia van der Salm, Andreas Fidjeland, Salvatore Scellato, Eri Latorre Chimoto, Hanna Klimczak-Plucinska, David Bridson,´ Dario de Cesare, Tom Hudson, Piermaria Mendolic chio, Lexi Walker, Alex Morris, Matthew Mauger, Alexey Guseynov, Alison Reid, Seth Odoom, Lu cia Loher, Victor Cotruta, Madhavi Yenugula, Do minik Grewe, Anastasia Petrushkina, Tom Duerig, Antonio Sanchez, Steve Yadlowsky, Amy Shen, Amir Globerson, Lynette Webb, Sahil Dua, Dong Li, Surya Bhupatiraju, Dan Hurt, Haroon Qureshi, Ananth Agarwal, Tomer Shani, Matan Eyal, Anuj Khare, Shreyas Rammohan Belle, Lei Wang, Chetan Tekur, Mihir Sanjay Kale, Jinliang Wei, Ruoxin Sang, Brennan Saeta, Tyler Liechty, Yi Sun, Yao Zhao, Stephan Lee, Pandu Nayak, Doug Fritz, Man ish Reddy Vuyyuru, John Aslanides, Nidhi Vyas, Martin Wicke, Xiao Ma, Evgenii Eltyshev, Nina Mar tin, Hardie Cate, James Manyika, Keyvan Amiri, Yelin Kim, Xi Xiong, Kai Kang, Florian Luisier, Nilesh Tripuraneni, David Madras, Mandy Guo, Austin Waters, Oliver Wang, Joshua Ainslie, Jason Baldridge, Han Zhang, Garima Pruthi, Jakob Bauer, Feng Yang, Riham Mansour, Jason Gelman, Yang Xu, George Polovets, Ji Liu, Honglong Cai, Warren Chen, XiangHai Sheng, Emily Xue, Sherjil Ozair, Christof Angermueller, Xiaowei Li, Anoop Sinha, Weiren Wang, Julia Wiesinger, Emmanouil Koukoumidis, 

Yuan Tian, Anand Iyer, Madhu Gurumurthy, Mark Goldenson, Parashar Shah, MK Blake, Hongkun Yu, Anthony Urbanowicz, Jennimaria Palomaki, Chrisan tha Fernando, Ken Durden, Harsh Mehta, Nikola Momchev, Elahe Rahimtoroghi, Maria Georgaki, Amit Raul, Sebastian Ruder, Morgan Redshaw, Jin hyuk Lee, Denny Zhou, Komal Jalan, Dinghua Li, Blake Hechtman, Parker Schuh, Milad Nasr, Kieran Milan, Vladimir Mikulik, Juliana Franco, Tim Green Nam Nguyen, Joe Kelley, Aroma Mahendru, Andrea Hu, Joshua Howland, Ben Vargas, Jeffrey Hui, Kshi tij Bansal, Vikram Rao, Rakesh Ghiya, Emma Wang, Ke Ye, Jean Michel Sarr, Melanie Moranski Preston, Madeleine Elish, Steve Li, Aakash Kaku, Jigar Gupta, Ice Pasupat, Da-Cheng Juan, Milan Someswar, Tejv M., Xinyun Chen, Aida Amini, Alex Fabrikant, Eric Chu, Xuanyi Dong, Amruta Muthal, Senaka Buth pitiya, Sarthak Jauhari, Nan Hua, Urvashi Khan delwal, Ayal Hitron, Jie Ren, Larissa Rinaldi, Sha har Drath, Avigail Dabush, Nan-Jiang Jiang, Har shal Godhia, Uli Sachs, Anthony Chen, Yicheng Fan, Hagai Taitelbaum, Hila Noga, Zhuyun Dai, James Wang, Chen Liang, Jenny Hamer, Chun-Sung Ferng, Chenel Elkind, Aviel Atias, Paulina Lee, Vít Listík, Mathias Carlen, Jan van de Kerkhof, Marcin Pikus, Krunoslav Zaher, Paul Müller, Sasha Zykova, Richard Stefanec, Vitaly Gatsko, Christoph Hirn schall, Ashwin Sethi, Xingyu Federico Xu, Chetan Ahuja, Beth Tsai, Anca Stefanoiu, Bo Feng, Ke shav Dhandhania, Manish Katyal, Akshay Gupta, Atharva Parulekar, Divya Pitta, Jing Zhao, Vivaan Bhatia, Yashodha Bhavnani, Omar Alhadlaq, Xiaolin Li, Peter Danenberg, Dennis Tu, Alex Pine, Vera Filippova, Abhipso Ghosh, Ben Limonchik, Bhar gava Urala, Chaitanya Krishna Lanka, Derik Clive, Yi Sun, Edward Li, Hao Wu, Kevin Hongtongsak, Ianna Li, Kalind Thakkar, Kuanysh Omarov, Kusha Majmundar, Michael Alverson, Michael Kucharski, Mohak Patel, Mudit Jain, Maksim Zabelin, Paolo Pelagatti, Rohan Kohli, Saurabh Kumar, Joseph Kim, Swetha Sankar, Vineet Shah, Lakshmi Ramachan druni, Xiangkai Zeng, Ben Bariach, Laura Wei dinger, Tu Vu, Alek Andreev, Antoine He, Kevin Hui, Sheleem Kashem, Amar Subramanya, Sissie Hsiao, Demis Hassabis, Koray Kavukcuoglu, Adam Sadovsky, Quoc Le, Trevor Strohman, Yonghui Wu Slav Petrov, Jeffrey Dean, and Oriol Vinyals. 2024. Gemini: A family of highly capable multimodal mod els. Preprint, arXiv:2312.11805. 



Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30. 





Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. 2023. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3909–3925. 





Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhi-





fang Sui. 2024. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. In Findings of the Associationfor Computational Linguistics ACL 2024, pages 7655–7671, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics. 





Aonan Zhang, Chong Wang, Yi Wang, Xuanyu Zhang, and Yunfei Cheng. 2024. Recurrent drafter for fast speculative decoding in large language models. Preprint, arXiv:2403.09919. 





Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. 2024. Masked diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical sampling. arXiv preprint arXiv:2409.02908. 





Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623. 





Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agar wal. 2024. Distillspec: Improving speculative decoding via knowledge distillation. Preprint, arXiv:2310.08461. 



## A Drafter Convergence Analysis

In previous speculative decoding approaches the computational overhead of the drafting stage has been proportional to $\gamma _ { : }$ , as this parameter dictates the number of network evaluations during the draft phase. When using a discrete diffusion draft model, the number of network evaluations is dictated by the number of diffusion steps $T .$ . The following reports an analysis of the possible speedups that can be achieved by our approach, as a function of the diffusion steps $T$ 

First, note that the expected number of tokens per draft can be derived from α and $\gamma \colon$ 

$$
\mathbb {E} (\# t o k e n s) = \frac {1 - \alpha^ {\gamma + 1}}{1 - \alpha}.\tag{2}
$$

In prior studies, theoretical results have focused on determining the optimal $\gamma$ to maximize the throughput of speculative decoding methods (Leviathan et al., 2023). For SpecDiff, extending $\gamma$ introduces minimal overhead and becomes less important to consider. The number of sequential diffusion operations, T, instead impacts α as increasing this number improves the convergence of $M _ { q } ( x )$ to the learned distribution, aims to closely approximate the distribution of $M _ { p } ( x )$ . Hence, an implicit dependency arises between $T$ and $\alpha ,$ which is reflected in Figure 2 (right). 

First, note that the computation overhead of a single network evaluation of $M _ { q } ( x )$ and $M _ { p } ( x )$ is constant. $M _ { q } ( x )$ is scaled by the number of diffusion steps, whereas all evaluations of $M _ { p } ( x )$ are conducted in parallel. Now, consider that, provided Equation 2: 

$$
\mathbb {E} (\# t o k e n s / s e c o n d) = \frac {\left(\frac {1 - \alpha^ {\gamma + 1}}{1 - \alpha}\right)}{T c _ {1} + c _ {2}}\tag{3}
$$

where $c _ { 1 }$ is the computation overheads of a single network evaluation of $M _ { q } ( x )$ and $c _ { 2 }$ is the computation overheads of a single network evaluation of $M _ { p } ( x )$ 

Next, consider that the convergence of $q ( x )$ to the original data distribution, which we will denote as ${ \hat { q } } ( x )$ , is proportional to the $1 / T$ 

Theorem A.1 ((Li et al., 2023)). Under standard assumptions, the convergence rate of samplers based on the probabilityflow Ordinary Differential Equation (ODE), converge at the rate 

$$
\begin{array}{c} T V (q _ {1}, \hat {q} _ {1}) \leq c _ {3} \frac {d ^ {2} \log^ {4} T}{T} + c _ {3} \frac {d ^ {6} \log^ {6} T}{T ^ {2}} \\ + c _ {3} \sqrt {d \log^ {3} T \epsilon_ {\text {score}}} + c _ {3} d (\log T) \epsilon_ {\text {Jacobi}} \end{array}
$$

where d is the dimensions of the sample, $\epsilon _ { s c o r e }$ is the error in the scorefunction estimation, $\epsilon J a c o b i$ is the error in the Jacobian matrices, and universal constant $c _ { 3 } > 0$ 

Given that $q _ { 1 } \approx q _ { 0 }$ , this result provides a practical upper bound on the distance between $q ( x )$ and ${ \hat { q } } ( x )$ . As Theorem ${ \mathrm { A . 1 } }$ provides an explicit relation to $T$ , this can be used to determine an upper bound on the distance between $q ( x )$ and $p ( x )$ for a given number of diffusion steps. By the triangle inequality: 

$$
T V (p (x), q (x)) \leq T V (p (x), \hat {q} (x)) + T V (\hat {q} (x), q (x))\tag{4}
$$

Now, the remaining step to find the upper bound on the distance from $q ( x )$ to $p ( x )$ is to determine $T V ( p ( x ) , \hat { q } ( x ) )$ . First, consider that this relation can be expressed as follows: 

Definition A.2 ((Leviathan et al., 2023)). $\begin{array} { r l r } { D _ { L K } ( p , q ) } & { { } = } & { \sum _ { x } \| p ( x ) - \ M ( x ) \| \quad = } \end{array}$ $\textstyle \sum _ { x } \left\| q ( x ) - M ( x ) \right\|$ where $\begin{array} { r } { M ( x ) = \frac { p ( x ) + q ( x ) } { 2 } } \end{array}$ Corollary A.3 ((Leviathan et al., 2023)). $\alpha = 1$ $\mathbb { E } ( D _ { L K } ( p , q ) ) = \mathbb { E } ( \operatorname* { m i n } ( p , q ) )$ 

Provided Corollary ${ \mathrm { A } } . 3 , D _ { L K } ( p ( x ) , { \hat { q } } ( x ) )$ can be computed empirically by setting $T$ arbitrarily high and evaluating $\alpha ;$ subsequently, we will refer to this distance as $c _ { 4 }$ . We note that the $D _ { L K } ( p ( x ) , \hat { q } ( x ) )$ captures any error in the draft model’s learned distribution that is introduced in Theorem A.1 as $\epsilon _ { \mathrm { s c o r e } }$ and $\epsilon _ { \mathrm { J a c o b i } }$ , so we will set these to zero when applying the theorem. 

Note that the metric $D _ { L K }$ is equivalent to the discretized total variation: 

Proof. 

$$
\begin{array}{l} D _ {L K} (p, q) = \sum_ {x} \| p (x) - M (x) \| \\ \qquad = \frac {1}{2} \sum_ {x} \| p (x) - q (x) \| \\ \qquad \approx \frac {1}{2} \int \| p (x) - q (x) \| d x \\ \qquad = T V (p, q) \end{array}
$$

For practical applications such as this, the discrete total variation is used, and for the purpose of this analysis we will consider the metrics equivalent. Now, we are ready to compute a lower bound for α that is dependent on $T .$ , applying Equation 4: 

$$
\begin{array}{l} \alpha = 1 - \mathbb {E} (D _ {L K} (p, q)) = 1 - \mathbb {E} (T V (p, q)) \\ \qquad \geq 1 - (c _ {4} + T V (\hat {q} (x), q (x))) \end{array}
$$

$$
\alpha \geq 1 - (c _ {4} + c _ {3} \frac {d ^ {2} \log^ {4} T}{T} + c _ {3} \frac {d ^ {6} \log^ {6} T}{T ^ {2}})\tag{5}
$$

Equation 5 provides a lower bound on α that is dependant on $T .$ . While in practice this remains computationally intractable, given that $c _ { 3 }$ is unknown, this can be approximated using a surrogate network to predict this constant; this is not dissimilar from the suggestion by Leviathan et al. to optimize runtime using such an approach to predict $\alpha .$ 

We are now ready to connect this to Equation $3 { : }$ 

$$
\mathbb {E} \left(\# t o k e n s / s e c o n d\right) \geq \frac {1 - \left(1 - c _ {4} - c _ {3} \frac {d ^ {2} \log^ {4} T}{T} - c _ {3} \frac {d ^ {6} \log^ {6} T}{T ^ {2}}\right) ^ {\gamma + 1}}{c _ {4} + c _ {3} \frac {d ^ {2} \log^ {4} T}{T} + c _ {3} \frac {d ^ {6} \log^ {6} T}{T ^ {2}}} \times \frac {1}{T c _ {1} + c _ {2}}
$$

This equation can now be solved analytically to optimize the lower bound. Practically, in the presence of a surrogate network, this can be simplified further, given convergence of the diffusion model is proportional to $1 / T$ 

$$
\mathbb {E} (\# t o k e n s / s e c o n d) \geq \frac {1 - (1 - c _ {4} - c _ {5} \frac {1}{T}) ^ {\gamma + 1}}{c _ {4} + c _ {5} \frac {1}{T}} \times \frac {1}{T c _ {1} + c _ {2}}\tag{6}
$$

Hence, the dependency between $T$ and α can be exploited to estimate the optimal number of diffusion steps. In our experiments we find that the optimal value of $T \leq 5$ 

## B Accepted Draft Lengths

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/96331fd7-c2e9-46c8-ad96-9ab1675dcbd7/d5792155c4a1e38c6544db6edf3f7a7ac4f21672c6232fdf257e365b70acc781.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/96331fd7-c2e9-46c8-ad96-9ab1675dcbd7/6a893bbd7b5e9f936f20ae82f120ba8b0d6a31d2d30c9af259542b55e8b6abf2.jpg)



Figure 3: Accepted draft lengths for OpenWebText evaluation.


We notice a starch constrast in the lengths of accepted drafts between these two settings. While the distribution of accepted draft lengths in Figure 4 is what would be anticipated given the lower values of $\alpha ,$ , the longer generations on OpenWebText (Figure 3) speak to the parallelism that can be realized when the distribution is effectively aligned to the target model. 

<table><tr><td></td><td></td><td><eq>M_p</eq></td><td><eq>M_q</eq></td><td>Strategy</td><td>γ</td><td>T</td><td>Gen Length</td><td>Precision</td></tr><tr><td rowspan="2">CNN</td><td rowspan="2">Ours</td><td>GPT-2 XL (1.5B)</td><td>MDLM (110M)</td><td>ddpm cache</td><td>15</td><td>2</td><td>1024</td><td>32 bit</td></tr><tr><td>GPT NEO (2.7B)</td><td>MDLM (110M)</td><td>ddpm cache</td><td>15</td><td>2</td><td>1024</td><td>32 bit</td></tr><tr><td rowspan="2">OWT</td><td rowspan="2">Ours</td><td>GPT-2 XL (1.5B)</td><td>MDLM (110M)</td><td>ddpm cache</td><td>15</td><td>2</td><td>1024</td><td>32 bit</td></tr><tr><td>GPT NEO (2.7B)</td><td>MDLM (110M)</td><td>ddpm cache</td><td>20</td><td>2</td><td>1024</td><td>32 bit</td></tr><tr><td>MT</td><td>Ours</td><td>Vicuna (33B)</td><td>MDLM (141M)</td><td>ddpm cache</td><td>15</td><td>2</td><td>512</td><td>16 bit</td></tr></table>


Table 5: Additional details on parametric setups for reported results.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-24/96331fd7-c2e9-46c8-ad96-9ab1675dcbd7/bde5c92acf8666cde4a492ba66b21faf4a3ae27afc007ffda7f3f9991902e53e.jpg)



Figure 4: Accepted draft lengths for MT Bench evaluation.


<table><tr><td></td><td><eq>M_p</eq></td><td><eq>M_q</eq></td><td><eq>\gamma</eq></td><td><eq>\alpha</eq></td><td>Speedup</td></tr><tr><td rowspan="2">OWT</td><td rowspan="2">Ours</td><td>GPT-2 XL</td><td>SEDD Small</td><td>10</td><td>0.70</td></tr><tr><td>GPT NEO</td><td>SEDD Small</td><td>10</td><td>0.77</td></tr></table>


Table 6: Performance of SpecDiff utilizing SEDD as the drafter. Note that unlike experiments in Tables 2 and 4, the draft model has been finetuned on the selected datasets; without finetuning, acceptance rates are below $\alpha = 0 . 2$ , making SEDD impractical as a solely pretrained drafter.


## C Implementation Details

For all evaluation we utilize the following hyperparameter setups. If not explicitly noted here, parameters are consistent with those specified by the authors of MDLM (Sahoo et al., 2024). For reported results on CNN/DM and OWT, 200 iterations are conducted on samples randomly selected from the datasets. For evaluation on MT bench, 160 iterations are conducted. 

## D Choice of Diffusion Model

The selection of a discrete diffusion model as the drafter plays a critical role in optimizing the overall framework’s speedup performance. The models explored, MDLM and SEDD, represent the current state-of-the-art in discrete diffusion, achieving near auto-regressive perplexity results with comparably sized models. We observe significant speedups over SpS when using MDLM as our drafter, as MDLM demonstrates superior generation speeds overall. These gains are not observed in SEDD for two primary reasons: first, SEDD exhibits lower perplexity compared to MDLM, resulting in a lower acceptance rate and second, MDLM’s generation speed surpasses that of SEDD. 

## E Temperature Impact on Acceptance Rate

As discussed in Section 8, overconfidence exhibited by discrete diffusion models, a result of poor model calibration, results in the top-1 token having a probability close to 1 for the vast majority of generations. For these outputs, the temperature is effectively zero. In the speculative decoding framework, this overconfidence is reflected as higher $q ( x )$ values, which has a significant impact on the token acceptance rate when sampling stochastically. Specifically, high $q ( x )$ values lead to a decrease in the number of accepted tokens because it increases the frequency of $q ( x ) > p ( x )$ . Increasing the temperature has a smoothing effect on $p ( x )$ leading to misaligned distributions between $p ( x )$ and $q ( x )$ Consequently, tokens are more frequently rejected with a probability of $1 - { \frac { p ( x ) } { q ( x ) } }$ , where a larger $q ( x )$ further increasing the likelihood of rejections. 

This obstacle is avoided when sampling deterministically (temperature=0) as in the results reported. This challenge motivates future study of discrete diffusion model calibration. 