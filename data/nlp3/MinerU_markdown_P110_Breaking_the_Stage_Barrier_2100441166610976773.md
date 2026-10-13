# Breaking the Stage Barrier: A Novel Single-Stage Approach to Long Context Extension for Large Language Models

Haoran Lian<sup>1</sup>*, Junmin Chen<sup>2</sup>*, Wei Huang<sup>4</sup>*, Yizhe Xiong<sup>3</sup>*, Wenping Hu<sup>2</sup>*, Guiguang Ding<sup>3†</sup>, Hui Chen<sup>3</sup>, Jianwei Niu <sup>1,5,6†</sup>, Zijia Lin<sup>2†</sup>, Fuzheng Zhang<sup>2</sup>, Di Zhang<sup>2</sup> 

<sup>1</sup>Beihang University, <sup>2</sup>Kuaishou Technology, <sup>3</sup>Tsinghua University, <sup>4</sup>BUPT, <sup>5</sup>Zhongguancun Laboratory, <sup>6</sup>Zhengzhou University 

Correspondence<sup>†</sup>: niujianwei@buaa.edu.cn, dinggg@tsinghua.edu.cn, linzijia07@tsinghua.org.cn 

## Abstract

Recently, Large language models (LLMs) have revolutionized Natural Language Processing (NLP). Pretrained LLMs, due to limited training context size, struggle with handling long token sequences, limiting their performance on various downstream tasks. Current solutions toward long context modeling often employ multi-stage continual pertaining, which progressively increases the effective context length through several continual pretraining stages. However, those approaches require extensive manual tuning and human expertise. In this paper, we introduce a novel single-stage continual pretraining method, Head-Adaptive Rotary Position Encoding (HARPE), to equip LLMs with long context modeling capabilities while simplifying the training process. Our HARPE leverages different Rotary Position Encoding (RoPE) base frequency values across different attention heads and directly trains LLMs on the target context length. Extensive experiments on 4 language modeling benchmarks, including the latest RULER benchmark, demonstrate that HARPE excels in understanding and integrating long-context tasks with single-stage training, matching and even outperforming existing multi-stage methods. Our results highlight that HARPE successfully breaks the stage barrier for training LLMs with long context modeling capabilities. 

## 1 Introduction

In recent years, generative Large Language Models (LLMs) (Brown, 2020; Raffel et al., 2020; Touvron et al., 2023a; Fu et al., 2023; Su et al., 2024b; Xiong et al., 2024; Lian et al., 2024b,a) have dominated the field of Natural Language Processing, outperforming traditional task-specific methods on many tasks, like text summarization (Liu and Lapata, 2019; Zaheer et al., 2020; Wu et al., 2021), information extraction (Wei et al., 2021, 2023b,a) 

and question answering (Brown, 2020; Raffel et al., 2020). In the process of utilizing LLMs for downstream tasks, it is often necessary for the LLM to handle long token sequences. For example, when conducting text summarization with an LLM, the input sequence may include an entire book (Zhang et al., 2024a; Karpinska et al., 2024), which contains millions of words. To equip LLMs with the capability to handle long texts, current methods typically continually pretrain LLMs on a larger context window (Xiong et al., 2023; Peng et al., 2023; Fu et al., 2024a) compared to that in LLM pretraining. Given that Rotary Position Encoding (RoPE) (Su et al., 2024a) is the prevailing position encoding in most LLMs, among those methods, the mainstream approach is to increase the RoPE base frequency in the positional encoding during continual pretraining (Xiong et al., 2023), as studies have demonstrated that a larger base frequency is the prerequisite for handling longer text sequences (Liu et al., 2023; Men et al., 2024). 

To achieve a large effective context size, existing works commonly employ a multi-stage approach, progressively increasing the context length through a series of continued pretraining steps. For instance, Large World Model (Liu et al., 2024c), GLM4- Chat-1M (ChatGLM, 2024), MiniCPM-2.4B-128K (Hu et al., 2024) and Llama 3.1 (Dubey et al., 2024) utilize multi-stage pipelines to reach context windows of 1M and 128k, respectively. This approach has become the dominant method in the community for equipping LLMs with long context capabilities. 

Our single-stage experiments show that directly scaling a larger RoPE base in a single stage is less effective than using multi-stage approaches. This likely explains why most publicly available models employ multi-stage ABF (Adjusted Base Frequency) training. We hypothesize that direct scaling to the final training length without intermediate stages struggles to adapt to increased complexity, which is better managed through gradual, multi-

<table><tr><td rowspan="2">Metric</td><td colspan="2">Three-Stage ABF on LLaMA</td></tr><tr><td>Uniform 2B Tokens</td><td>Carefully Selected</td></tr><tr><td>NiaH</td><td>67.83</td><td>81.36</td></tr><tr><td>Benchmark</td><td>62.59</td><td>63.10</td></tr></table>


Table 1: Performance comparison of different continual pertaining pipelines when conducting three-stage ABF (Adjusted Base Frequency) (Xiong et al., 2023). For both pipelines, we report the average score of the Needle-in-a-Haystack task (6 lengths, 8 tasks) and on Short-Context Benchmarks (5 tasks). For more details on the experiment settings, please refer to Sec. 4.


stage adjustments. 

Although multi-stage approaches have shown promising results, our experiments reveal that they require substantial manual tuning and human expertise to achieve good performance on long context modeling benchmarks. For instance, our results in Tab. 1 demonstrate that a carefully scheduled three-stage pipeline outperforms a naive approach by 13.5% on the NiaH benchmark, highlighting the limited generalizability of hyperparameters across different LLM sizes and architectures. Moreover, multi-stage training poses practical challenges due to varying data and resource requirements. This motivates the need for single-stage continual pretraining approaches. However, single-stage training also presents challenges, such as the risk of suboptimal outcomes when training with a much longer context window and larger RoPE base frequency, as shown in our experiments. 

In this paper, we introduce a novel single-stage approach, termed Head-Adaptive Rotary Position Encoding (HARPE), designed to address the longcontext problem. Our goal is to achieve an effective context length comparable to that of multi-stage methods. Drawing inspiration from the finding that different attention heads can acquire distinct knowledge during training (Li et al., 2023), we propose to distribute the training of different stages across multiple attention heads concurrently. Specifically, we leverage RoPE (Su et al., 2024a) with varying base values to represent different effective context lengths, thereby simulating multiple training stages. By assigning different base values to different attention heads, we enable the LLMs to be trained in a single stage. 

To determine the RoPE base values for each attention head, we employ a complementary approach, carefully selecting values that fill in the peaks and valleys of the sine and cosine waves in RoPE, thereby optimizing the experimental results. In contrast to existing methods, our proposed HARPE offers a significant advantage in terms of simplicity and efficiency. By pretraining LLMs in a single stage, we substantially streamline the process of data preparation and pipeline adjustment, eliminating the need for multiple stages and associated complexities. 

We conduct a comprehensive evaluation of HARPE on 4 benchmarks, including the recently introduced RULER benchmark (Hsieh et al., 2024), to assess its effectiveness on both long-context and short-context tasks. The experimental results demonstrate that HARPE consistently matches or surpasses the performance of its multi-stage counterparts across all evaluated benchmarks. Notably, on the challenging Needle-in-a-haystack task, HARPE achieves a significant improvement of 5.46% over the multi-stage Adjused Base Frequency (ABF) (Xiong et al., 2023) approach, underscoring its exceptional capabilities in long-context modeling. 

Unlike inference methods that employ multiple RoPE bases simultaneously to support long contexts (Zhang et al., 2024c; Chen et al., 2024b), our HARPE approach fundamentally alters the learning dynamics of LLMs during continual pretraining, enabling a straightforward and streamlined training pipeline. In summary, our contributions are threefold: 

• We introduce a novel single-stage continual pretraining approach, termed Head-Adaptive Rotary Position Encoding (HARPE), to address the long context problem in LLMs. By doing so, we significantly simplify the process of data preparation and pipeline adjustment. 

• To overcome the limitations of traditional multi-stage approaches, we propose a novel training strategy that distributes the training of different stages across multiple attention heads. We utilize different RoPE base values to represent distinct training stages and carefully select these values to complement the attention scores. 

• We conduct a comprehensive evaluation of HARPE on 4 long context benchmarks, including the recently introduced RULER benchmark. Our experimental results demonstrate that HARPE consistently yields comparable or even better performance than existing multi-stage methods across all benchmarks. 

## 2 Related Works

Large Language Models (LLMs). Language models are a type of statistical model that aims to maximize the likelihood token sequences (Touvron et al., 2023a). The Transformer architecture (Vaswani et al., 2017) marked a turning point in the evolution of language models, accelerating their development. Transformer-based models, like BERT (Devlin, 2018), GPT-2 (Radford et al., 2019), and T5 (Raffel et al., 2020), have achieved groundbreaking results across numerous natural language processing tasks. More recently, the release of GPT-4 (Achiam et al., 2023) has further pushed the boundaries of LLMs performance, showcasing exceptional capabilities. As these models continue to scale and evolve architecturally, they have become the driving force behind cutting-edge research in natural language processing, exhibiting notable adaptability and versatility across a wide range of applications (Liu et al., 2024a; Cai et al., 2024; ?). Consequently, LLMs have profoundly transformed human-computer interaction. 

Long Context Modeling. Trained on relatively short context sequences (i.e., generally <10K tokens), open-source LLMs show dramatic performance drops on long context modeling (Touvron et al., 2023a; Bai et al., 2023; Liu et al., 2024a). Methods to improve the ability of LLMs to handle long context can be mainly divided into the following categories: attention mechanism optimizing, long-term memory caching, contexual processing, and positional encoding optimizing. Attention mechanism optimizing methods (Beltagy et al., 2020; Ma et al., 2021; Dao et al., 2022) reduce the computational and memory bottlenecks of the Transformer, thereby enabling the model to process longer text sequences. Long-term memory caching methods (Wang et al., 2024; Bulatov et al., 2022; Martins et al., 2020; Dai et al., 2019) utilize internal or external memory caches to fetch information in long context. Contextual processing methods (Ding et al., 2020; Izacard and Grave, 2020) process long context inputs by calling the model multiple times to process different parts of the long text sequence. 

Apart from those methods, the most common approach is to improve the RoPE (Su et al., 2024a) while conducting continual pretraining with a longer context window. Specifically, Position Interpolation (PI) (Chen et al., 2023) reduces the input position index to match the original context window size. ABF (Xiong et al., 2023) adjusts the RoPE base (i.e., θ) to scale the low-frequency part more significantly, thereby dispersing the interpolation pressure to multiple dimensions. NTKby-parts interpolation (bloc97, 2023) interpolates RoPE bases according to the wavelength of different dimensions in RoPE relative to the context size: high-frequency dimensions are not interpolated, low-frequency dimensions are fully interpolated, and intermediate frequency dimensions are partially interpolated using a ramp function. YARN (Peng et al., 2023) combines the NTK-by-parts interpolation with the attention scaling technique to achieve an even longer effective context length. Self-Extend (Jin et al., 2024) constructs a two-layer attention mechanism, consisting of group attention and neighbor attention, to successfully expand the context window without additional training. Studies have also explored the construction (Chen et al., 2024a) and training strategies (Bai et al., 2024) of long context data. 

While methods based on improving positional encoding have achieved promising experimental results (Liu et al., 2024c; He et al., 2024; Zhang et al., 2024b), they typically rely on complicated multi-stage training pipelines to gradually increase the effective context length(e.g., $8 k  1 6 k $ $3 2 k \dots \to 1 2 8 k )$ In contrast, our proposed HARPE offers a simplified single-stage continual pretraining approach. Experimental results demonstrate that our HARPE achieves comparable performance to existing multi-stage methods. 

## 3 Head-Adaptive Rotary Position Encoding based Approach

In this section, we first revist the formulation of Rotary Position Encoding (RoPE) in Sec. 3.1. We then present the proposed HARPE in Sec. 3.2, detailing its multi-head RoPE base mechanism and base selection strategies. 

## 3.1 Preliminaries

RoPE (Su et al., 2024a) is a widely adopted technique for position encoding in LLMs built on the Transformer architecture (GLM et al., 2024; Liu et al., 2024a; Yang et al., 2023; Dubey et al., 2024). The primary objective of RoPE is to encode positional information in a way that the inner product of the query and key embeddings inherently captures the relative position information, which can be formally expressed as: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/813c6511-70c0-4fa9-affb-a583cc3e48a6/40df67a436b4da472b451a30ca75b8ff2d3467312dee7007e577fcf1623e4b38.jpg)



Figure 1: Illustration of the multi-stage and our proposed single-stage (HARPE) continual pretraining pipeline.


$$
f (q _ {m}, m) ^ {T} f (k _ {n}, n) = g (q _ {m}, k _ {n}, m - n)\tag{1}
$$

Here, $f$ represents the positional encoding function applied to the query embeddings $q _ { m }$ at position m and key embeddings $k _ { n }$ at position $n .$ . To satisfy this condition, the function $f$ is defined as a $d -$ dimensional rotation matrix, denoted as $R _ { \Theta , m } ^ { d } \colon$ 

$$
f (x _ {\{q, k \}}, m) = \pmb {R} _ {\Theta , m} ^ {d} x _ {\{q, k \}}\tag{2}
$$

where 

$$
\boldsymbol {R} _ {\Theta , m} ^ {d} = \operatorname{diag} \left(\boldsymbol {R} (m \theta_ {1}), \dots , \boldsymbol {R} (m \theta_ {d / 2})\right)\tag{3}
$$

$$
\boldsymbol {R} (m \theta_ {i}) = \left( \begin{array}{c c} \cos m \theta_ {i} & - \sin m \theta_ {i} \\ \sin m \theta_ {i} & \cos m \theta_ {i} \end{array} \right)\tag{4}
$$

$$
\Theta = \{\theta_ {i} = b ^ {- 2 (i - 1) / d}, i \in [ 1, 2, \dots , d / 2 ] \}\tag{5}
$$

These formulas indicate that the rotation angle Θ can be adjusted by modifying the base frequency b. Specifically, increasing b (i.e., decreasing Θ) mitigates the severe decaying effect of RoPE on attention scores for distant tokens, thereby enabling LLMs to process longer input sequences (Xiong et al., 2023). 

## 3.2 HARPE

As illustrated in Fig. 1, HARPE leverages the diverse capabilities of each attention head by predefining a base set B, where the cardinality of B is equal to the number of attention heads. Specifically, HARPE assigns a unique base $b _ { h }$ from B to the RoPE in each attention head h. For head h, Θ in Eq. 5 is defined as: 

$$
\Theta_ {h} = \{\theta_ {h, i} = b _ {h} ^ {- 2 (i - 1) / d}, i \in [ 1, \dots , d / 2 ] \}\tag{6}
$$

then 

$$
\boldsymbol {R} _ {\Theta_ {h}, m} ^ {d} = \mathrm{diag} \left(\boldsymbol {R} (m \theta_ {h, i})\right)\tag{7}
$$

$$
\boldsymbol {R} (m \theta_ {h, i}) = \left( \begin{array}{c c} \cos m \theta_ {h, i} & - \sin m \theta_ {h, i} \\ \sin m \theta_ {h, i} & \cos m \theta_ {h, i} \end{array} \right)\tag{8}
$$

To determine the base set B, we establish two distinct RoPE base selection strategies. 

The first strategy involves uniformly distributing the bases $B _ { u n i f o r m }$ within a predefined range, bounded by a maximum base $b _ { m a x }$ and a minimum base $b _ { m i n }$ 

$$
B _ {u n i f o r m} = \{b _ {h} = b _ {m i n} + h \times \frac {b _ {m a x} - b _ {m i n}}{N - 1} \}\tag{9}
$$

where $h = 0 , 1 , \ldots , N - 1$ 

The second strategy adopts the search method proposed by (Chen et al., 2024b), which seeks to ensure that the attention waveform valleys of any given base overlap with peaks from different bases, and vice versa. To achieve this, a candidate base set $B _ { c }$ is initially generated by discretizing the range between $b _ { m a x }$ and $b _ { m i n }$ with a relatively small stride s. 

$$
B _ {c} = \left\{b _ {\min} + i \times s, i \in [ 1, \frac {b _ {\max} - b _ {\min}}{s} ] \right\}\tag{10}
$$

Algorithm 1 The searching algorithm of $B_{s}$ Require: A candidate base set $B_{c}$ 1: Define a function $f_{p}(b) \mapsto$ peak positions in attention waveforms corresponding to base $b$ 2: Define a function $f_{v}(b) \mapsto$ valley positions in attention waveforms corresponding to base $b$ 3: Initialize the searched base set $B_{s} \leftarrow \{b_{min}\}$ 4: $P_{s} \leftarrow f_{p}(b_{min}); V_{s} \leftarrow f_{v}(b_{min})$ 5: while $|B_{s}| < N$ do

6: for $b_{j}$ in $B_{c}$ do

7: $P_{j} \leftarrow f_{p}(b_{j}); V_{j} \leftarrow f_{v}(b_{j})$ 8: $d_{j}^{+} \leftarrow \sum_{p_{j,i} \in P_{j}} |p_{j,i} - v_{s,i}|$ $v_{s,i} \in V_{s}$ 9: $d_{j}^{-} \leftarrow \sum_{v_{j,i} \in V_{j}} |v_{j,i} - p_{s,i}|$ $p_{s,i} \in P_{s}$ 10: $d_{j} \leftarrow d_{j}^{+} + d_{j}^{-}$ 11: end for

12: $B_{s} \leftarrow B_{s} \cup \{b'_{j}\text{ with the minimum } d_{j}\}$ 13: $P_{s} \leftarrow P_{s} \cup f_{p}(b'_{j}); V_{s} \leftarrow V_{s} \cup f_{v}(b'_{j})$ 14: end while

15: return $B_{s}$ Head Base Head Base Head Base Head Base
1 1.00 9 2.50 17 3.01 25 3.61
2 1.15 10 2.65 18 3.04 26 3.88
3 1.30 11 2.68 19 3.10 27 4.09
4 1.45 12 2.71 20 3.13 28 4.15
5 2.17 13 2.74 21 3.16 29 4.39
6 2.20 14 2.80 22 3.22 30 4.45
7 2.23 15 2.83 23 3.43 31 4.51
8 2.47 16 2.92 24 3.46 32 4.54 


Table 2: RoPE base frequency settings for each head in HARPE, with each base value expressed in millions $( \times 1 0 ^ { 6 } )$ , and stride of 30k.


Subsequently, the final searched base set $B _ { s }$ is determined by iteratively complementing the valleys and peaks of attention waveforms of different bases within $\mathit { B _ { c } } ,$ as shown in Algorithm 1. 

## 4 Experimental Setup

We select LLama2-7B-Base (Touvron et al., 2023b) as our base model, which is configured with a RoPE base frequency of 10k and a context length of 4k. 

## 4.1 Baseline Systems

We compare HARPE with 4 continual pretraining methods and one training-free method. 

PI (Chen et al., 2023) employs a linear downscaling of the input position indices to match the original context window size, thereby avoiding extrapolation beyond the trained context length, which can lead to catastrophically high attention scores that compromise the self-attention mechanism. The interpolation scale is set to 32 = 128k/4k. 

<table><tr><td>Method</td><td>Proof-pile</td><td>GovReport</td></tr><tr><td>Llama2-7B-Base</td><td>4336.96</td><td>7289.38</td></tr><tr><td>PI</td><td>20.73</td><td>11.47</td></tr><tr><td>ABF Single-Stage</td><td>3.06</td><td>3.58</td></tr><tr><td>ABF Multi-Stage</td><td>3.03</td><td>3.57</td></tr><tr><td>YaRN</td><td>4.53</td><td>4.52</td></tr><tr><td>Self-Extend*</td><td>5.45</td><td>3.76</td></tr><tr><td>HARPE(ours)</td><td>3.02</td><td>3.54</td></tr></table>


Table 3: Sliding window perplexity $\left( \mathrm { S } = 2 5 6 \right)$ for Proofpile and GovReport documents. Asterisks (*) denote the training-free method. Lower perplexity values indicate better model performance.


ABF Single-Stage (Xiong et al., 2023) implements a minimal yet necessary modification to the RoPE positional encoding for long-context modeling: increasing the hyperparameter "base frequency" b to 5m (i.e., decreasing the rotation angle), which mitigates the decaying effect of RoPE for distant tokens. Concurrently, the input sequence length is increased to 128k. 

ABF Multi-Stage also increases the base and input sequence length, but with a key difference: it does so in a gradual, multi-stage manner. Specifically, we divide the process into three stages: $( 1 ) b = 1 m ; l = 3 2 k , ( 2 ) b = 2 m ; l = 6 4 k$ , and $( 3 ) b = 5 m ; l = 1 2 8 k .$ 

YaRN (Peng et al., 2023) utilizes the RoPE formula Eq. (5) to distinguish between high-frequency and low-frequency positional components. It adjusts the base within a 64-dimensional space according to these frequency components, applying a scale factor of 32. 

Self-Extend (Jin et al., 2024) is a training-free method. We apply it with window_size = 1024 and group_size = 32. 

## 4.2 HARPE Base Setting

We adopt the second strategy (i.e.,the peak-valley search method) mentioned in Sec. 3.2. We set $b _ { m i n } = 1 m ; b _ { m a x } = 5 m ; s = 3 0 k$ . The final bases are shown in Tab. 2. And we will discuss other base settings in Sec. 5.2. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/813c6511-70c0-4fa9-affb-a583cc3e48a6/e447bc1e7950bceafc2e7e7149d0e073a9e70a812a57ac3220840386f37fdcf1.jpg)



Figure 2: Traditional Single-Key Needle-in-a-Haystack: the x-axis represents the number of tokens in the test sample, ranging up to 128k tokens with finer granularity. The y-axis shows the depth of the needle’s position within the current test sample.


<table><tr><td>Method</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td><td>Avg.</td></tr><tr><td>Llama2-7B-Base</td><td>90.90</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PI</td><td>77.56</td><td>26.59</td><td>16.50</td><td>0.00</td><td>0.00</td><td>0.00</td><td>20.11</td></tr><tr><td>ABF Single-Stage</td><td>92.44</td><td>88.78</td><td>84.16</td><td>78.03</td><td>70.81</td><td>62.72</td><td><eq>79.49_{(3rd)}</eq></td></tr><tr><td>ABF Multi-Stage</td><td>95.19</td><td>91.72</td><td>87.53</td><td>78.84</td><td>72.78</td><td>62.13</td><td><eq>81.36_{(2nd)}</eq></td></tr><tr><td>YaRN</td><td>83.88</td><td>73.66</td><td>64.84</td><td>46.53</td><td>12.69</td><td>0.00</td><td>46.93</td></tr><tr><td>Self-Extend*</td><td>76.47</td><td>66.25</td><td>58.84</td><td>52.16</td><td>1.38</td><td>0.00</td><td>42.52</td></tr><tr><td>HARPE(ours)</td><td>97.03</td><td>96.88</td><td>93.72</td><td>86.66</td><td>79.41</td><td>67.19</td><td><eq>86.82_{(1st)}</eq></td></tr></table>


Table 4: Upgraded Needle-in-a-Haystack Tests: Average scores for 8 NiaH tasks at various lengths. Asterisks (*) denote training-free methods.


## 4.3 Evaluation Metric

Perplexity (PPL) is evaluated on the Proof-pile (Zhangir Azerbayev, 2022) and GovReport (Huang et al., 2021) datasets. Following the setup in Yarn, for the Proof-pile dataset, we selected samples with a minimum of 128k tokens and measured perplexity for token lengths ranging from 2k to 128k in increments of 2k, averaging the scores for each length. For the GovReport dataset, we reported the average PPL scores for samples with a context window of 32k tokens. Evaluations are conducted using the sliding window method proposed by Press (Press et al., 2021), with a window size of 256 tokens. 

Needle-in-a-Haystack is a task that assesses a model’s ability to accurately locate and recite a specific sentence, referred to as the "needle", within a lengthy document, known as the "haystack". To provide a more comprehensive evaluation of a model’s long-context capabilities, we extend this method, inspired by RULER (Hsieh et al., 2024), to include multi-key, multi-value and multi-query scenarios, as well as diverse types of needles and background documents in each scenario. A multikey task involves multiple keys, similar to ’the needle’, in the background, where the model must find the target needle among the distractions. In a multi-value task, multiple needles are inserted in haystack, and the model earns one point for each correct needle found. 

<table><tr><td>Method</td><td>ABF Single-Stage</td><td>ABF Multi-Stage</td><td>HARPE (Ours)</td></tr><tr><td>MMLU</td><td>40.87</td><td>41.10</td><td>40.74</td></tr><tr><td>Hellaswag</td><td>77.33</td><td>77.83</td><td>77.99</td></tr><tr><td>ARC-c</td><td>52.82</td><td>52.65</td><td>52.73</td></tr><tr><td>PIQA</td><td>78.56</td><td>78.56</td><td>78.56</td></tr><tr><td>TriviaQA</td><td>62.39</td><td>63.29</td><td>63.72</td></tr><tr><td>Avg.</td><td>62.39</td><td>62.69</td><td>62.75</td></tr></table>


Table 5: Short-Context Benchmark Results: Evaluation Results of the Top 3 Long-Context-Performance Models on 5 Short-Context Datasets.


Short-Context Benchmarks assess whether short-context capabilities are preserved during long-context training. We include five widely used short-context evaluation datasets: 5-shot MMLU (Hendrycks et al., 2020), 10-shot Hellaswag (Zellers et al., 2019), 25-shot ARC-Challenge (Clark et al., 2018), 0-shot PiQA (Bisk et al., 2019), and 5-shot TriviaQA (Joshi et al., 2017). 

## 4.4 Training Configuration

For continual pretraining, we follow the configurations outlined in (Fu et al., 2024b), utilizing the upsampling dataset from (Yaofu, 2023b). We employ the Llama2-7B-Base model as the pre-trained backbone, with a learning rate of $\mathrm { ^ { 2 e ^ { - 5 } } }$ and AdamW optimizer settings of $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 5$ . All models were continually pre-trained with 6B tokens using these consistent settings. 

## 5 Experimental Results

## 5.1 HARPE vs. Baseline Systems

We utilize HARPE to conduct a comparative evaluation with the five long-context methods outlined in Sec. 4.1, employing three evaluation metrics as detailed in Sec. 4.3. 

First, to evaluate the long context modeling capability of HARPE, we evaluate HARPE and the competing methods with the PPL metric. As shown in Tab. 3, on the tested Proof-pile and GovReport datasets, our HARPE achieves comparable or even better results compared to the state-of-the-art multistage methods and various single-stage methods. This indicates that the proposed HARPE has the capability to handle long text sequences. 

Furthermore, we employ the upgraded Needlein-a-Haystack test, as defined in the RULER benchmark (Hsieh et al., 2024), to evaluate the long-context relationship capturing performance of 

HARPE and its competitors. As shown in Tab. 4, HARPE significantly outperforms all listed methods. Notably, HARPE proves more effective than multi-stage approaches, surpassing the multi-stage ABF by 5.5%. While typical single-stage methods, such as YARN and PI, fail as the context length increases, HARPE successfully extends the effective context length to 128K tokens. More details on the NiaH results, including scores for each of the 8 NiaH tasks (e.g., multi-key and multi-value), are provided in the Appendix Tab. 8. Simultaneously, we evaluate traditional NiaH tasks at a finer granularity, following the code in (Liu et al., 2024b). As shown in Fig. 2, HARPE achieves a 100% accuracy rate across various lengths within 128k tokens. 

We also evaluate HARPE on the short-context benchmarks. Results in Tab. 5 show that HARPE also yields comparable or even slightly better performance than competing methods in terms of average accuracy across 5 short-context tasks. 

## 5.2 Study of Various Base Schemes

In this section, we evaluate the performance of two base selection methods for the head-specific RoPE bases in HARPE: uniform distribution and peakvalley opposition. For the uniform distribution method, we conduct two experiments with uniform ascending and descending intervals to analyze the impact of different base orders on model performance. For the peak-valley opposition method, as described in algorithm 1 and Eq. (10), we test five variations with different base strides (10k, 20k, 30k, 40k, 50k) to further explore their effects. 

The results of various HARPE configurations, along with the original LLaMA2 model, on the upgraded Needle-in-a-Haystack test are presented in Tab. 6. Under different RoPE base settings, our HARPE consistently outperforms the original LLaMA2-7B-Base model. Among the two methods evaluated, the peak-valley opposition approach with stride = 30k demonstrates the best performance, surpassing the next closest competitor by 1.25%. As a result, we adopt the peak-valley approach with a stride of 30k for HARPE. 

## 5.3 Comparative Results on RULER Evaluation

In this section, we evaluate HARPE against various open-source pre-trained models on a range of long-context tasks using the RULER benchmark. RULER is a comprehensive and widely recognized standard for long-context evaluation, comprising 

<table><tr><td>Method</td><td>Model</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td><td>Avg.</td></tr><tr><td>RoPE</td><td>Llama2-7B-Base</td><td>90.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="5">complementarity peak&amp;valley</td><td>stride = 10k</td><td>96.38</td><td>96.38</td><td>88.97</td><td>84.09</td><td>76.00</td><td>65.22</td><td>84.51</td></tr><tr><td>stride = 20k</td><td>96.97</td><td>96.78</td><td>91.69</td><td>85.97</td><td>74.44</td><td>67.59</td><td><eq>85.57_{(2nd)}</eq></td></tr><tr><td>stride = 30k</td><td>97.03</td><td>96.88</td><td>93.72</td><td>86.66</td><td>79.41</td><td>67.19</td><td><eq>86.82_{(1st)}</eq></td></tr><tr><td>stride = 40k</td><td>96.84</td><td>96.09</td><td>91.13</td><td>83.13</td><td>75.94</td><td>65.00</td><td>84.69</td></tr><tr><td>stride = 50k</td><td>92.75</td><td>92.4</td><td>87.31</td><td>83.97</td><td>75.31</td><td>70.34</td><td>83.68</td></tr><tr><td rowspan="2">same stride</td><td>ascending order</td><td>96.53</td><td>95.13</td><td>89.22</td><td>84.72</td><td>74.31</td><td>65.88</td><td>84.30</td></tr><tr><td>descending order</td><td>96.50</td><td>95.22</td><td>91.91</td><td>87.16</td><td>76.31</td><td>64.69</td><td><eq>85.30_{(3rd)}</eq></td></tr></table>


Table 6: Upgraded Needle-in-a-Haystack Results of HARPE: Comparison of different base selection schemes in HARPE models.


<table><tr><td>Model</td><td>Size</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td><td>Avg.</td></tr><tr><td>Jamba (AI21, 2024)</td><td>52B</td><td>81.20</td><td>75.4</td><td>68.8</td><td>65.3</td><td>61.00</td><td>51.4</td><td>67.18</td></tr><tr><td>Mixtral (Mistral.AI, 2023)</td><td>7B</td><td>91.60</td><td>89.80</td><td>86.30</td><td>77.20</td><td>52.30</td><td>8.00</td><td>67.50</td></tr><tr><td>Llama2-7B-Base</td><td>7B</td><td>79.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Together (Together.AI, 2023)</td><td>7B</td><td>84.6</td><td>78.7</td><td>68.3</td><td>57.9</td><td>0.0</td><td>0.0</td><td>48.25</td></tr><tr><td>Yarn (Peng et al., 2023)</td><td>7B</td><td>77.30</td><td>67.50</td><td>59.00</td><td>47.30</td><td>38.60</td><td>13.90</td><td>50.60</td></tr><tr><td>LongLoRA (Chen et al., 2024c)</td><td>7B</td><td>81.90</td><td>80.4</td><td>75.6</td><td>65.1</td><td>60.80</td><td>0.0</td><td>60.63</td></tr><tr><td>LWM (Liu et al., 2024c)</td><td>7B</td><td>77.50</td><td>74.00</td><td>69.60</td><td>64.60</td><td>61.30</td><td>59.00</td><td><eq>67.67_{(3rd)}</eq></td></tr><tr><td>llama-2-7b-80k (Yaofu, 2023a)</td><td>7B</td><td>87.95</td><td>80.68</td><td>72.70</td><td>63.47</td><td>54.62</td><td>47.65</td><td><eq>67.85_{(2nd)}</eq></td></tr><tr><td>HARPE (ours)</td><td>7B</td><td>88.48</td><td>83.44</td><td>74.87</td><td>68.10</td><td>55.64</td><td>51.88</td><td><eq>70.40_{(1st)}</eq></td></tr></table>


Table 7: RULER Benchmark Results: Comparison of HARPE and Open-Source Base Models Across All Lengths for 13 RULER Tasks.


13 tasks that include "needle in a haystack" as well as additional tasks such as Variable Tracing, Aggregation Ability, and Question Answering. 

As shown in Tab. 7, our comparison in 10 base models primarily involves 7B models, along with model using other architecture such as Jamba. HARPE surpasses all LLaMA2-based models and ranks 1st overall, surpassing the 2nd by 2.55%. Notably, HARPE demonstrates a significant advantage in shorter context performance compared to the multi-stage ABF-trained LWM with a 1M fine-tuning length. Furthermore, HARPE consistently outperforms the YaRN model with a 128k fine-tuning length, achieving an average improvement of nearly 20 points across various lengths. Additionally, when compared to the llama-2-7b-80k model, which has the same training parameters and dataset but a shorter fine-tuning length of 80k, HARPE still shows superior performance in shorter context tasks with lengths less than 32k. 

## 6 Conclusion

In this paper, we present a novel single-stage continual pretraining method, HARPE, to enhance the long-context modeling capabilities of LLMs. Specifically, our HARPE distributes the different training stages across different attention heads, and assigns different base values in the RoPE for different attention heads during continual pretraining stage. Experimental results across 4 benchmarks demonstrate that HARPE outperforming or matching existing multi-stage methods in long-context modeling tasks, while maintaining comparable performance on short-context tasks. In practical applications, our HARPE breaks the stage barrier, offering a simplified pipeline with minimal manual tuning and expertise, thereby streamlining the process of equipping LLMs with long-context capabilities. 

## 7 Limitations

Despite that HARPE demonstrates promising results on benchmarks with long context lengths, limitation still remains. Our research is primarily concentrated on the continual pretraining stage, leaving its applicability to other stages, such as supervised fine-tuning, unexplored. We will address those limitations in our future research. 

## Acknowledgment

This work was supported in part by National Key R&D Program of China under Grant No. 2023YFB4503700, National Natural Science Foundation of China under Grant No. 62372027, U23B2025, Beijing Natural Science Foundation (L247026). 

## References



Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774. 





AI21. 2024. Introducing jamba: Ai21’s groundbreaking ssm-transformer model. https://www.ai21.com blog/announcing-jamba. Accessed: 2024-08-31. 





Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609. 





Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. 2024. Longalign: A recipe for long context alignment of large language models. arXiv preprint arXiv:2401.18058. 





Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150. 





Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2019. Piqa: Reasoning about physical commonsense in natural language. 





bloc97. 2023. Add ntk-aware interpolation "by parts" correction. Accessed: 2023. 





Tom B Brown. 2020. Language models are few-shot learners. arXiv preprint arXiv:2005.14165. 





Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. 2022. Recurrent memory transformer. Advances in Neural Information Processing Systems, 35:11079– 11091. 





Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, Xin Chen, Xun Chen, Zehui Chen, Zhi Chen, Pei Chu, et al. 2024. Internlm2 technical report. arXiv preprint arXiv:2403.17297. 





ChatGLM. 2024. Glm: Long scaling pre-trained model contexts to millions. Accessed: 2024-09-14. 





Longze Chen, Ziqiang Liu, Wanwei He, Yunshui Li, Run Luo, and Min Yang. 2024a. Long context is not long at all: A prospector of long-dependency data for large language models. arXiv preprint arXiv:2405.17915. 





Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595. 





Yuhan Chen, Ang Lv, Ting-En Lin, Changyu Chen, Yuchuan Wu, Fei Huang, Yongbin Li, and Rui Yan. 2024b. Fortify the shortest stave in attention: Enhancing context awareness of large language models for effective tool use. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11160– 11174. 





Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. 2024c. Longlora: Efficient fine-tuning of long-context large language models. In Proceedings of the International Conference on Learning Representations (ICLR). 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. 





Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V Le, and Ruslan Salakhutdinov. 2019. Transformer-xl: Attentive language models beyond a fixed-length context. arXiv preprint arXiv:1901.02860. 





Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359. 





Jacob Devlin. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. 





Ming Ding, Chang Zhou, Hongxia Yang, and Jie Tang. 2020. Cogltx: Applying bert to long texts. Advances in Neural Information Processing Systems, 33:12792– 12804. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783. 





Jiayi Fu, Lei Lin, Xiaoyang Gao, Pengli Liu, Zhengzong Chen, Zhirui Yang, Shengnan Zhang, Xue Zheng, Yan Li, Yuliang Liu, et al. 2023. Kwaiyiimath: Technical report. arXiv preprint arXiv:2310.07488. 





Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. 2024a. Data engineering for scaling language models to 128k context. arXiv preprint arXiv:2402.10171. 





Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. 2024b. Data engineering for scaling language models to 128k context. 





Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793. 





Junqing He, Kunhao Pan, Xiaoqun Dong, Zhuoyang Song, LiuYiBo LiuYiBo, Qianguosun Qianguosun, Yuxin Liang, Hao Wang, Enming Zhang, and Jiaxing Zhang. 2024. Never lost in the middle: Mastering long-context question answering with positionagnostic decompositional training. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13628–13642. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. 





Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654. 





Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. Minicpm: Unveiling the potential of small language models with scalable training strategies. 





Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. 2021. Efficient attentions for long document summarization. arXiv preprint arXiv:2104.02112. 





Gautier Izacard and Edouard Grave. 2020. Leveraging passage retrieval with generative models for open domain question answering. arXiv preprint arXiv:2007.01282. 





Hongye Jin, Xiaotian Han, Jingfeng Yang, Zhimeng Jiang, Zirui Liu, Chia-Yuan Chang, Huiyuan Chen, and Xia Hu. 2024. Llm maybe longlm: Self-extend llm context window without tuning. arXiv preprint arXiv:2401.01325. 





Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. 





Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, and Mohit Iyyer. 2024. One thousand and one pairs: A" novel" challenge for long-context language models. arXiv preprint arXiv:2406.16264. 





Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli. 2023. Functional interpolation for relative positions improves long context transformers. arXiv preprint arXiv:2310.04418. 





Haoran Lian, Yizhe Xiong, Zijia Lin, Jianwei Niu, Shasha Mo, Hui Chen, Peng Liu, and Guiguang Ding. 2024a. Lbpe: Long-token-first tokenization to improve large language models. arXiv preprint arXiv:2411.05504. 





Haoran Lian, Yizhe Xiong, Jianwei Niu, Shasha Mo, Zhenpeng Su, Zijia Lin, Peng Liu, Hui Chen, and Guiguang Ding. 2024b. Scaffold-bpe: Enhancing byte pair encoding with simple and effective scaffold token removal. arXiv preprint arXiv:2404.17808. 





Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, et al. 2024a. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434. 





Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. 2024b. Lwm. https://github. com/zhangir-azerbayev/proof-pile. Accessed: 2024-09-13. 





Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. 2024c. World model on million-length video and language with blockwise ringattention. arXiv preprint arXiv:2402.08268. 





Xiaoran Liu, Hang Yan, Shuo Zhang, Chenxin An, Xipeng Qiu, and Dahua Lin. 2023. Scaling laws of rope-based extrapolation. arXiv preprint arXiv:2310.05209. 





Yang Liu and Mirella Lapata. 2019. Text summarization with pretrained encoders. arXiv preprint arXiv:1908.08345. 





Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, and Luke Zettlemoyer. 2021. Luna: Linear unified nested attention. Advances in Neural Information Processing Systems, 34:2441– 2453. 





André Martins, António Farinhas, Marcos Treviso, Vlad Niculae, Pedro Aguiar, and Mario Figueiredo. 2020. Sparse and continuous attention mechanisms. Advances in Neural Information Processing Systems, 33:20989-21001 





Xin Men, Mingyu Xu, Bingning Wang, Qingyu Zhang, Hongyu Lin, Xianpei Han, and Weipeng Chen. 2024. Base of rope bounds context length. arXiv preprint arXiv:2405.14591. 





https://mistral. ai/news/la-plateforme/ d 





Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071. 





Ofir Press, Noah A Smith, and Mike Lewis. 2021. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409. 





Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9. 





Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67. 





Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024a. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063. 





Zhenpeng Su, Zijia Lin, Xue Bai, Xing Wu, Yizhe Xiong, Haoran Lian, Guangyuan Ma, Hui Chen, Guiguang Ding, Wei Zhou, et al. 2024b. Maskmoe: Boosting token-level learning via routing mask in mixture-of-experts. arXiv preprint arXiv:2407.09816. 





Together.AI. 2023. Preparing for the era of 32k context: Early learnings and explorations. https://www. together.ai/blog/llama-2-7b-32k. Accessed: 2024-08-31 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. 





Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. 2024. Augmenting language models with long-term memory. Advances in Neural Information Processing Systems, 36. 





Kaiwen Wei, Xian Sun, Zequn Zhang, Li Jin, Jingyuan Zhang, Jianwei Lv, and Zhi Guo. 2023a. Implicit event argument extraction with argument-argument relational knowledge. IEEE Trans. Knowl. Data Eng., 35(9):8865–8879. 





Kaiwen Wei, Xian Sun, Zequn Zhang, Jingyuan Zhang, Zhi Guo, and Li Jin. 2021. Trigger is not sufficient: Exploiting frame-aware knowledge for implicit event argument extraction. In Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 4672–4682. Association for Computational Linguistics. 





Kaiwen Wei, Yiran Yang, Li Jin, Xian Sun, Zequn Zhang, Jingyuan Zhang, Xiao Li, Linhao Zhang, Jintao Liu, and Zhi Guo. 2023b. Guide the many-toone assignment: Open information extraction via iou-aware optimal transport. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pages 4971– 4984. Association for Computational Linguistics. 





Chuhan Wu, Fangzhao Wu, Tao Qi, Yongfeng Huang, and Xing Xie. 2021. Fastformer: Additive attention can be all you need. arXiv preprint arXiv:2108.09084. 





Wenhan Xiong, Jingyu Liu, Igor Molybog, Hejia Zhang, Prajjwal Bhargava, Rui Hou, Louis Martin, Rashi Rungta, Karthik Abinav Sankararaman, Barlas Oguz, et al. 2023. Effective long-context scaling of foundation models. arXiv preprint arXiv:2309.16039. 





Yizhe Xiong, Xiansheng Chen, Xin Ye, Hui Chen, Zijia Lin, Haoran Lian, Jianwei Niu, and Guiguang Ding. 2024. Temporal scaling law for large language models. arXiv preprint arXiv:2404.17785. 





Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. 2023. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305. 





Yaofu. 2023a. Llama-2 7b 80k model. https: //huggingface.co/yaofu/llama-2-7b-80k/ tree/main. Accessed: 2024-08-31. 





Yaofu. 2023b. Slimpajama-per-source-length-upsample. https://huggingface.co/datasets/yaofu/ slimpajama-per-source-length-upsample. Accessed: 2024-09-03. 





Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. 2020. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297. 





Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a 





machine really finish your sentence? arXiv preprint arXiv:1905.07830. 





Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, et al. 2024a. ∞ bench: Ex tending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15262–15277. 





Yiyuan Zhang, Handong Li, Jing Liu, and Xiangyu Yue. 2024b. Explore the limits of omni-modal pretraining at scale. arXiv preprint arXiv:2406.09412. 





Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, and Zhangyang Wang. 2024c. Found in the middle: How language models use long contexts better via plug-and-play positional encoding. arXiv preprint arXiv:2403.04797. 





Bartosz Piotrowski Zhangir Azerbayev, Edward Ayers. 2022. proof-pile. https://github. com/zhangir-azerbayev/proof-pile. Accessed: 2022. 



## A Detail Scores

<table><tr><td>Method</td><td>Task</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td><td>Avg.</td></tr><tr><td rowspan="8">PI</td><td><eq>niah\_single_1</eq></td><td>94.00</td><td>35.00</td><td>20.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>24.83</td></tr><tr><td><eq>niah\_single_2</eq></td><td>99.00</td><td>46.00</td><td>21.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>27.67</td></tr><tr><td><eq>niah\_single_3</eq></td><td>99.00</td><td>45.00</td><td>29.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>28.83</td></tr><tr><td><eq>niah\_multikey_1</eq></td><td>88.00</td><td>39.00</td><td>26.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>25.50</td></tr><tr><td><eq>niah\_multikey_2</eq></td><td>85.00</td><td>14.00</td><td>5.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>17.33</td></tr><tr><td><eq>niah\_multikey_3</eq></td><td>55.00</td><td>2.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>9.50</td></tr><tr><td><eq>niah\_multivalue</eq></td><td>32.25</td><td>16.25</td><td>17.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>10.92</td></tr><tr><td><eq>niah\_multiquery</eq></td><td>68.25</td><td>15.50</td><td>14.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>16.29</td></tr><tr><td rowspan="8">ABF Single-Stage</td><td><eq>niah\_single_1</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.00</td><td>89.00</td><td>91.00</td><td>96.50</td></tr><tr><td><eq>niah\_single_2</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.00</td><td>93.00</td><td>98.67</td></tr><tr><td><eq>niah\_single_3</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>88.00</td><td>98.00</td></tr><tr><td><eq>niah\_multikey_1</eq></td><td>92.00</td><td>93.00</td><td>89.00</td><td>91.00</td><td>86.00</td><td>86.00</td><td>89.50</td></tr><tr><td><eq>niah\_multikey_2</eq></td><td>95.00</td><td>98.00</td><td>94.00</td><td>82.00</td><td>70.00</td><td>37.00</td><td>79.33</td></tr><tr><td><eq>niah\_multikey_3</eq></td><td>63.00</td><td>54.00</td><td>31.00</td><td>16.00</td><td>6.00</td><td>3.00</td><td>28.83</td></tr><tr><td><eq>niah\_multivalue</eq></td><td>92.00</td><td>72.25</td><td>66.00</td><td>51.50</td><td>34.75</td><td>44.00</td><td>60.08</td></tr><tr><td><eq>niah\_multiquery</eq></td><td>97.50</td><td>93.00</td><td>93.25</td><td>84.75</td><td>81.75</td><td>59.75</td><td>85.00</td></tr><tr><td rowspan="8">ABF Multi-Stage</td><td><eq>niah\_single_1</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>96.00</td><td>99.00</td><td>99.17</td></tr><tr><td><eq>niah\_single_2</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>95.00</td><td>99.17</td></tr><tr><td><eq>niah\_single_3</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>94.00</td><td>99.00</td></tr><tr><td><eq>niah\_multikey_1</eq></td><td>95.00</td><td>95.00</td><td>95.00</td><td>94.00</td><td>83.00</td><td>89.00</td><td>91.83</td></tr><tr><td><eq>niah\_multikey_2</eq></td><td>96.00</td><td>96.00</td><td>91.00</td><td>76.00</td><td>73.00</td><td>14.00</td><td>74.33</td></tr><tr><td><eq>niah\_multikey_3</eq></td><td>79.00</td><td>69.00</td><td>48.00</td><td>15.00</td><td>8.00</td><td>1.00</td><td>36.67</td></tr><tr><td><eq>niah\_multivalue</eq></td><td>95.25</td><td>81.00</td><td>74.75</td><td>55.50</td><td>40.25</td><td>40.75</td><td>64.58</td></tr><tr><td><eq>niah\_multiquery</eq></td><td>96.25</td><td>92.75</td><td>91.50</td><td>90.25</td><td>82.00</td><td>64.25</td><td>86.17</td></tr><tr><td rowspan="8">YaRN</td><td><eq>niah\_single_1</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>95.00</td><td>0.00</td><td>82.50</td></tr><tr><td><eq>niah\_single_2</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>87.00</td><td>0.00</td><td>0.00</td><td>64.50</td></tr><tr><td><eq>niah\_single_3</eq></td><td>100.00</td><td>100.00</td><td>97.00</td><td>65.00</td><td>0.00</td><td>0.00</td><td>60.33</td></tr><tr><td><eq>niah\_multikey_1</eq></td><td>79.00</td><td>69.00</td><td>51.00</td><td>30.00</td><td>1.00</td><td>0.00</td><td>38.33</td></tr><tr><td><eq>niah\_multikey_2</eq></td><td>81.00</td><td>64.00</td><td>42.00</td><td>12.00</td><td>0.00</td><td>0.00</td><td>33.17</td></tr><tr><td><eq>niah\_multikey_3</eq></td><td>79.00</td><td>69.00</td><td>48.00</td><td>15.00</td><td>8.00</td><td>1.00</td><td>36.67</td></tr><tr><td><eq>niah\_multivalue</eq></td><td>86.00</td><td>75.00</td><td>59.50</td><td>34.50</td><td>3.50</td><td>0.00</td><td>43.08</td></tr><tr><td><eq>niah\_multiquery</eq></td><td>84.00</td><td>72.25</td><td>67.25</td><td>42.75</td><td>2.00</td><td>0.00</td><td>44.71</td></tr><tr><td rowspan="8">Self-Extend</td><td><eq>niah\_single_1</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>94.00</td><td>11.00</td><td>0.00</td><td>67.50</td></tr><tr><td><eq>niah\_single_2</eq></td><td>100.00</td><td>100.00</td><td>96.00</td><td>87.00</td><td>0.00</td><td>0.00</td><td>63.83</td></tr><tr><td><eq>niah\_single_3</eq></td><td>100.00</td><td>99.00</td><td>90.00</td><td>92.00</td><td>0.00</td><td>0.00</td><td>63.50</td></tr><tr><td><eq>niah\_multikey_1</eq></td><td>78.00</td><td>71.00</td><td>56.00</td><td>43.00</td><td>0.00</td><td>0.00</td><td>41.33</td></tr><tr><td><eq>niah\_multikey_2</eq></td><td>28.00</td><td>9.00</td><td>3.00</td><td>2.00</td><td>0.00</td><td>0.00</td><td>7.00</td></tr><tr><td><eq>niah\_multikey_3</eq></td><td>26.00</td><td>5.00</td><td>3.00</td><td>1.00</td><td>0.00</td><td>0.00</td><td>5.83</td></tr><tr><td><eq>niah\_multivalue</eq></td><td>89.25</td><td>64.50</td><td>52.75</td><td>46.25</td><td>0.00</td><td>0.00</td><td>42.13</td></tr><tr><td><eq>niah\_multiquery</eq></td><td>90.50</td><td>81.50</td><td>70.00</td><td>52.00</td><td>0.00</td><td>0.00</td><td>49.00</td></tr><tr><td rowspan="8">HARPE</td><td><eq>niah\_single_1</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td><eq>niah\_single_2</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td><eq>niah\_single_3</eq></td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.00</td><td>99.83</td></tr><tr><td><eq>niah\_multikey_1</eq></td><td>96.00</td><td>97.00</td><td>91.00</td><td>92.00</td><td>93.00</td><td>90.00</td><td>93.17</td></tr><tr><td><eq>niah\_multikey_2</eq></td><td>93.00</td><td>96.00</td><td>96.00</td><td>87.00</td><td>78.00</td><td>46.00</td><td>82.67</td></tr><tr><td><eq>niah\_multikey_3</eq></td><td>91.00</td><td>91.00</td><td>81.00</td><td>38.00</td><td>23.00</td><td>6.00</td><td>55.00</td></tr><tr><td><eq>niah\_multivalue</eq></td><td>98.50</td><td>95.00</td><td>92.25</td><td>87.75</td><td>61.50</td><td>43.00</td><td>79.67</td></tr><tr><td><eq>niah\_multiquery</eq></td><td>97.75</td><td>96.00</td><td>89.50</td><td>88.50</td><td>79.75</td><td>53.50</td><td>84.17</td></tr></table>


Table 8: Detail Scores of the 8 upgraded Needle-in-a-Haystack Tasks Across Different Lengths.
