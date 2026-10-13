# BYTEFLOW: LANGUAGE MODELING THROUGH ADAP-TIVE BYTE COMPRESSION WITHOUT A TOKENIZER

Chunyuan Deng<sup>∗</sup> Rice University cd110@rice.edu 

Sanket Lokegaonkar Amazon Science sllokega@amazon.com 

Colin Lockard Amazon Science clockard@amazon.com 

Besnik Fetahu Amazon Science besnikf@amazon.com 

Nasser Zalmout Amazon Science nzalmout@amazon.com 

Xian Li Amazon Science xianlee@amazon.com 

## ABSTRACT

Modern language models (LMs) still rely on fixed, pre-defined subword tokenizations. Once a tokenizer is trained, the LM can only operate at this fixed level of granularity, which often leads to brittle and counterintuitive behaviors even in otherwise strong reasoning models. We introduce ByteFlow Net, a new hierarchical architecture that removes tokenizers entirely and instead enables models to learn their own segmentation of raw byte streams into semantically meaningful units. ByteFlow Net performs compression-driven segmentation based on the coding rate of latent representations, yielding adaptive boundaries while preserving a static computation graph via Top-K selection. Unlike prior self-tokenizing methods that depend on brittle heuristics with human-designed inductive biases, ByteFlow Net adapts its internal representation granularity to the input itself. Experiments demonstrate that this compression-based chunking strategy yields substantial performance gains, with ByteFlow Net outperforming both BPE-based Transformers and previous byte-level architectures. These results suggest that end-to-end, tokenizer-free modeling is not only feasible but also more effective, opening a path toward more adaptive, robust, and information-grounded LMs. 

## 1 INTRODUCTION

Tokenization is a foundational step in every language model pipeline (Grattafiori et al., 2024; Team et al., 2025; DeepSeek-AI et al., 2025; Yang et al., 2025). The model’s first action is to segment raw input—be it text, code, or other modalities—into discrete tokens. This seemingly simple decision carries profound consequences, defining the model’s vocabulary, sequence lengths, and the very granularity of its learned representations. The primary limitation of dominant strategies like bytepair encoding (BPE) (Sennrich et al., 2015; Galle´, 2019; Liu et al., 2025) is their static nature. After training, they apply a fixed segmentation logic to all inputs, ignoring context, linguistic nuance, or task-specific requirements. This static property on subword level is the source of many wierd model behaviors, such as difficulties with counting, arithmetic, structured data, and multilingual text (Rust et al., 2020; Zhang et al., 2024; Yehudai et al., 2024). At a more fundamental level, tokenization introduces a non-learnable stage into the pipeline, breaking the end-to-end language modeling. This imposes a rigid inductive bias, forcing the model to expend its FLOPs on predefined units rather than learning how to allocate them dynamically. 

Recent efforts to eliminate tokenizers have largely converged on hierarchical architectures. The central challenge for such designs is defining the high-level semantic units beyond byte level. Current methods generally fall into two main categories: i) Heuristic-based strategies that employ static chunking via fixed strides, word boundaries, or regular expressions (Yu et al., 2023b; Slagle, 2024a; Videau et al., 2025), and ii) Dynamic chunking that learn to segment sequences using a neural network, entropy thresholds, or cosine similarity (Nawrot et al., 2022; Pagnoni et al., 2025; Hwang et al., 2025b)<sup>1</sup>. While heuristic approaches embed strong inductive biases into the model, dynamic methods introduce considerable uncertainty into the chunking process, which can hinder pattern finding during the early stages of pre-training. Furthermore, we still lack a dynamic mechanism for guiding the model’s allocation of FLOPs in a principled manner. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/6fb788c9f8c097fc4753f402e43a76b26180eb06fa478c238842a5b0f3de8706.jpg)



Figure 1: Architecture of ByteFlow Net. (a) ByteFlow Net’s chunking strategy is primarily driven by the coding rate R of latent representations. As shown in the figure, the model is encouraged to select token boundaries that form pooled subsequences which best compress the original input. (b) Since byte-level sequences are roughly 4× longer, directly applying O(n<sup>2</sup>d) softmax attention becomes prohibitively expensive. To address this, we adopt sliding-window attention (SWA) combined with canon layers (Allen-Zhu, 2025), enabling efficient and low-cost token mixing. (c) The beauty of the hierarchical architecture lies in allocating the majority of FLOPs operating at the high-level information (a deep and wide global transformer), while using lightweight local encoders/decoders (shallow and narrow) to quickly process low-level information.


We introduce ByteFlow Net (Figure 1), a novel hierarchical byte-level architecture that learns to self-tokenize directly from raw byte streams. Rather than applying a fixed vocabulary, ByteFlow Net integrates segmentation into its forward computation: as bytes flow through the network, it dynamically promotes them to higher-level calculations. The decision of when to commit a boundary is framed as a principled, coding-rate-based compression problem, estimating the representational cost of promoting the position to a higher level. This formulation turns boundary placement into an online information-theoretic optimization problem, enabling the model to adjust token granularity according to input complexity on its own. 

Architecturally, ByteFlow Net follows a simple but effective hierarchy: The process begins with a local encoder that transforms byte sequences into contextualized representations. Next, a chunking module applies the coding-rate criterion to these representations, producing higher-level tokens on the fly. These dynamic tokens are then modeled by a global transformer to capture the deep and abstract patterns in high resolustion level, before a decoder maps the global context back to byte-level predictions. Because this entire boundary selection process is integrated into the model’s computation, ByteFlow Net naturally adapts across languages and domains without requiring languagespecific rules or a separate tokenizer training stage.By unifying segmentation and representation learning, ByteFlow Net ensures that the boundaries of higher-level tokens are optimized to maximize information density for the global transformer. This transition from a fixed, discrete preprocessing step to a continuous, learnable bottleneck allows the model to preserve nuances—such as morphological variations or structural symbols—that are often lost or fragmented by static tokenizers. 

## Contributions.

• We introduce a new paradigm that replaces static tokenization with dynamic, learned segmentation. Our architecture, ByteFlow Net, operates end-to-end on raw bytes, using a principled information-theoretic objective to identify meaningful units on the fly. 

• We demonstrate superior performance and scaling through extensive experiments. Byte-Flow Net consistently outperforms both strong LLaMA baseline and other byte-level architectures on pre-training loss and downstream tasks, showing that end-to-end, tokenizer-free modeling is not only feasible but more effective. 

• We reveal that the success of our approach stems from its ability to preserve a coherent latent manifold. Our ablation studies show that the coding-rate objective uniquely maintains the geometric structure of the data’s representation, preventing the fragmentation that plagues other methods and enabling more powerful learning. 

## 2 RELATED WORK

Tokenizer-free Architecture. Modern tokenizer-free architectures can be broadly categorized into three main approaches: 

• Pure Byte-Level Modeling: These models perform language modeling directly on raw byte sequences (Xue et al., 2022a). Given that the $O ( n ^ { \mathsf { \bar { 2 } } } d )$ complexity of full attention is prohibitive for long sequences, architectures like MambaByte (Wang et al., 2024) have emerged as an effective solution, balancing fine-grained information processing with computational efficiency. 

• Hierarchical Modeling with Heuristic Chunking: These methods use fixed, rule-based strategies to group bytes into larger units. For instance, MegaByte (Yu et al., 2023b) uses a fixed stride (e.g., 4 or 6 bytes) to create a higher level of representation, outperforming pure byte-level models while significantly saving FLOPs. Building on this, SpaceByte (Slagle, 2024b) uses word boundaries for chunking, achieving performance on par with or even exceeding BPE-based transformers on some pre-training corpora. AU-Net (Videau et al., 2025) further refines this concept by replacing simple word boundaries with a flexible set of regex rules to better handle special tokens and digits. 

• Hierarchical Modeling with Dynamic Chunking: Instead of fixed rules, these models employ a learned mechanism to determine chunk boundaries. Nawrot et al. (2022; 2023); Kallini et al. use a neural network to gate token boundaries. The Byte Latent Transformer (BLT) (Pagnoni et al., 2024) first trains a separate entropy model and then uses it as a proxy to set a global chunking threshold. This multi-stage process is not fully end-to-end and functions more like a different tokenizer. In a concurrent work, H-Net (Hwang et al., 2025a) uses the cosine similarity between neighboring representations to decide chunking. 

Tokenization in Language Modeling. The prevailing solution in modern LMs is subword tokenization (Sennrich et al., 2015; Kudo & Richardson, 2018; Zouhar et al., 2023; Schmidt et al., 2024; Liu et al., 2025) (e.g., BPE), which use a fixed-size vocabulary of word pieces to represent any text. These fixed vocabularies create a rigid, non-learnable stage in the modeling pipeline, often causing brittle and unexpected behaviors (Belinkov & Bisk, 2018; Sun et al., 2020; Rust et al., 2020; Petrov et al., 2023; Schmidt et al., 2024; Zhang et al., 2024; Yehudai et al., 2024), which motivated the development of modern tokenizer-free models that operate directly on raw bytes. 

## 3 BYTEFLOW NET

Overview. ByteFlow Net is a hierarchical architecture that operates through five main stages: local encoder, downsampling (coding-rate chunking), global modeling, upsampling, and decoder: 

$$
x _ {1: T} \in V ^ {T} \xrightarrow {\text { Local   Encoder }} h _ {1: T} \in \mathbb {R} ^ {T \times d _ {\text { local }}} \quad (\text { contextualized   byte   representations })\tag{1}
$$

$$
\xrightarrow {\text { Downsampling }} z _ {1: K} \in \mathbb {R} ^ {K \times d _ {\text { global }}} \quad (\text { adaptive   chunking }, K \ll T)\tag{2}
$$

$$
\xrightarrow {\text { Global   Transformer }} g _ {1: K} \in \mathbb {R} ^ {K \times d _ {\text { global }}} \quad (\text { High - resolution   level   modeling })\tag{3}
$$

$$
\xrightarrow {\text { Upsampling }} s _ {1: T} \in \mathbb {R} ^ {T \times d _ {\text { local }}} \quad (\text { reconstruct   to   original   length })\tag{4}
$$

$$
\xrightarrow {\text { Decoder }} \hat {p} (x _ {t + 1} | \cdot) \in V \quad (\text { next   byte   prediction })\tag{5}
$$

Here $T$ is the input sequence length, $V \in \Delta ^ { 2 5 8 }$ (contains 256 UTF-8 Byte plus two BOS/EOS tokens) is the byte vocabulary, and $d _ { \mathrm { l o c a l } } , d _ { \mathrm { g l o b a l } }$ are the hidden dimensions at local and global levels. 

## 3.1 LOCAL ENCODER: FAST PROCESSING OVER BYTE-LEVEL REPRESENTATIONS

The local encoder are stacked small transformer. The input byte sequence $x _ { 1 : T } \in V ^ { T }$ first embedded into a continuous representation $h _ { 1 : T } ^ { ( 0 ) }$ by the learned byte embedding matrix, then transformed into contextualized representations $h _ { 1 : T } \in \mathbb { R } ^ { T \times d _ { \mathrm { l o c a l } } }$ 

Transformer Blocks with Sliding Window Attention. We stack E pre-norm causal transformer blocks. For each layer $l \in \{ 1 , \ldots , \overline { { E } } \}$ and position $t \in \{ 1 , \ldots , T \}$ 

$$
u _ {t} ^ {(l)} = \mathrm{LN} \big (h _ {t} ^ {(l - 1)} \big),\tag{6}
$$

$$
\hat {h} _ {t} ^ {(l)} = C a n o n (h _ {t} ^ {(l - 1)} + \mathrm{SWA} (\mathbf {Q}, \mathbf {K}, \mathbf {V})), \mathbf {Q}, \mathbf {K}, \mathbf {V} = u _ {1: t} ^ {(l)} X _ {\mathbf {Q}}, u _ {1: t} ^ {(l)} X _ {\mathbf {K}}, u _ {1: t} ^ {(l)} X _ {\mathbf {V}}\tag{7}
$$

$$
v _ {t} ^ {(l)} = \mathrm{LN} (\hat {h} _ {t} ^ {(l)}),\tag{8}
$$

$$
h _ {t} ^ {(l)} = \operatorname{Canon} \left(\hat {h} _ {t} ^ {(l)} + \operatorname{SwiGLU} \left(v _ {t} ^ {(l)}\right)\right),\tag{9}
$$

where: LN(·) denotes layer normalization. SWA(·) represents sliding window attention (SWA) with window size $w _ { \mathrm { l o c a l } }$ . This reduces computational complexity from ${ \cal O } ( { \cal T } ^ { 2 } )$ to $O ( T \cdot w _ { \mathrm { l o c a l } } )$ . 

SwiGLU(·) is the gated activation function $\mathrm { S w i G L U } ( x ) = \mathrm { S w i s h } ( x W _ { 1 } ) \odot ( x W _ { 2 } )$ , where $W _ { 1 } , W _ { 2 } \in$ R<sup>dlocal×dff</sup> are learned projection matrices, $d _ { \mathrm { f f } }$ is the feed-forward hidden dimension, Swish(x) = $x \cdot \sigma ( x )$ with $\sigma ( \cdot )$ being the sigmoid function, and $\odot$ denotes element-wise multiplication. 

Canon Layer. Canon layer are introduced in Allen-Zhu (2025) to foster the token mixing: 

$$
C a n o n (h _ {t}) = w _ {0} \odot h _ {t} ^ {(l)} + w _ {1} \odot h _ {t - 1} ^ {(l)} + w _ {2} \odot h _ {t - 2} ^ {(l)} + w _ {3} \odot h _ {t - 3} ^ {(l)},\tag{10}
$$

where $w _ { i } \in \mathbb { R } ^ { d _ { \mathrm { l o c } \mathrm { \Omega } } }$ <sup>l</sup> are learned gating vectors. They are basically causal conv1d with kernel size $= 4 ,$ so highly efficient CUDA operator are supported. 

Why SWA + Canon Layer for Token Mixing. Theoretically if we use SWA along, given a sequence length T and window size $w _ { \mathrm { l o c a l } }$ , we will need at least $\frac { \sigma } { w _ { \mathrm { l o c a l } } }$ encoder layers to ensure every byte position can attend to every other. This would necessitate a very deep local encoder for long sequences, increasing computational cost and potentially hindering training stability. The canon layer instead is an efficient addition, as it introduces negligible parameter overhead and benefits from highly optimized implementations. 

## 3.2 DOWNSAMPLING: CODING-RATE CHUNKING

The chunker then determines which byte positions to promote to the next hierarchical level by evaluating the coding rate of contextualized representations. This approach is grounded in information theory: positions with high coding rates contain more information and should be preserved as chunk boundaries, while positions with low coding rates can be safely compressed away. 

Lossy Coding Rate in Representation Space. Let the local encoder produce contextualized representations $h _ { 1 : T } ~ \in ~ \mathbb { R } ^ { \bar { T } \times d _ { \mathrm { l o c a l } } }$ The lossy coding rate (Cover, 1999; Ma et al., $2 0 0 7 ) ^ { 2 }$ for 

$h _ { 1 : T } \in \mathbb { R } ^ { T \times d _ { \mathrm { l o c a l } } }$ is: 

$$
R _ {\varepsilon} (h _ {1: T}) = \frac {1}{2} \log \det \left(I + \frac {d _ {\text { local }}}{\varepsilon^ {2}} h _ {1: T} h _ {1: T} ^ {\top}\right),\tag{11}
$$

where $\varepsilon ^ { 2 }$ is a noise variance parameter that controls the sensitivity of the coding rate computation. $R _ { \varepsilon } ( h _ { 1 : T } )$ is large when the representation $h _ { 1 : T }$ has large eigenvalues and spans diverse directions in the representation space, indicating high information position that warrants preservation. 

Streaming Decision. Let the local encoder produce contextualized representations $h _ { 1 : T } \quad \in$ $\mathbb { R } ^ { T \times d _ { \mathrm { l o c a l } } }$ . The marginal coding rate at position t measures the information gain from including the t-th byte: 

$$
\Delta R _ {t} = R _ {\varepsilon} (h _ {1: t}) - R _ {\varepsilon} (h _ {1: t - 1}).\tag{12}
$$

$\Delta R _ { t }$ is large when position t introduces large information gain, indicating a natural segmentation boundary. Given the target global sequence length K, the chunking procedure begins by computing marginal coding rates $\Delta R _ { t }$ for all positions $t \in \{ 2 , 3 , \ldots , T \}$ We initialize the selected positions with ${ \mathcal { S } } = \{ { \overline { { 1 } } } \}$ to always include the BOS token, then identify the $( K - 1 )$ positions with the largest $\Delta R _ { t }$ values. Finally, we sort these selected positions chronologically to obtain $\boldsymbol { S } = \left\{ s _ { 1 } , s _ { 2 } , \ldots , s _ { K } \right\}$ where $s _ { 1 } = 1$ and $s _ { 1 } < s _ { 2 } < \dots < s _ { K }$ . During teacher-forced training, Top-$K$ uses the full-sequence importance profile to allocate global compute, but causal masks ensure predictions never access future byte content. 

After selecting K positions<sup>3</sup>, we extract the corresponding representations $[ h _ { s _ { 1 } } , h _ { s _ { 2 } } , \ldots , h _ { s _ { K } } ] \in$ $\mathbb { R } ^ { K \times d _ { \mathrm { l o c a l } } }$ and map them to the global representation space: $z _ { 1 : K } = [ h _ { s _ { 1 } } , \bar { h } _ { s _ { 2 } } , \ldots \bar { , } h _ { s _ { K } } ] W _ { \mathrm { p r o j } } ^ { \mathrm { ~ \scriptsize ~ \in ~ } }$ $\mathbb { R } ^ { K \times d _ { \mathrm { g l o b a l } } }$ , where $W _ { \mathrm { p r o j } } \in \mathbb { R } ^ { d _ { \mathrm { l o c a l } } \times d _ { \mathrm { g l o b a l } } }$ is the projection matrix. 

Why Not Global Threshold? Instead of using a global information threshold for chunking, we select the Top-K positions with the highest information gain for two reasons. First, determining an appropriate global threshold is non-trivial: it often requires extensive empirical tuning and results in a “magic number” that is difficult to interpret or generalize. Second, a fixed threshold leads to dynamic chunks for different inputs. This variability in the global sequence length breaks the static computation graph. While specialized CUDA operators used in (Hwang et al., 2025a) can manage dynamic graphs, they introduce other issues like variable memory allocation per input, which easily got into OOM issue with some unlucky batch. Fixed-length Top-K also preserves a static computation graph, enabling consistent memory allocation and avoiding ragged tensors that complicate GPU batching. 

## 3.3 GLOBAL TRANSFORMER: DEEP MODELING FOR HIGH-LEVEL ABSTRACTION

The global transformer operates on compressed representations $z _ { 1 : K } \in \mathbb { R } ^ { K \times d _ { \mathrm { g l o b a l } } }$ using full causal attention. Since $K \ll T$ , we employ a deep (G layers) and wide $( d _ { \mathrm { g l o b a l } } \gg d _ { \mathrm { l o c a l } } )$ architecture that concentrates computational budget on high-level reasoning: 

$$
g _ {1: K} = \operatorname{Transformer} _ {\text { global }} (z _ {1: K}), \quad \text { FLOPs } \approx O (G \cdot K ^ {2} \cdot d _ {\text { global }} ^ {2})\tag{13}
$$

The quadratic attention complexity $O ( K ^ { 2 } )$ remains tractable due to compression, while the large hidden dimension $d _ { \mathrm { g l o b a l } }$ and depth $G$ enable sophisticated modeling of long-range dependencies and abstract patterns. 

## 3.4 UPSAMPLING: MULTI-LINEAR RECONSTRUCTION WITH LARGE RESIDUAL

Given processed global representations $g _ { 1 : K }$ and selected positions $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { K } \}$ , we reconstruct full-length representations using position-specific transformations: 

$$
\operatorname{chunk} (t) = \arg \max _ {i} \left\{s _ {i}: s _ {i} \leq t \right\},\tag{14}
$$

$$
\operatorname{bin} (t) = \left\lfloor \frac {t}{T / B} \right\rfloor , \quad B \ll T,\tag{15}
$$

$$
\tilde {s} _ {t} = g _ {\mathrm{chunk} (t)} W _ {\mathrm{bin} (t)}, \quad W _ {\mathrm{bin} (t)} \in \{W _ {1}, \dots , W _ {B} \},\tag{16}
$$

$$
s _ {t} = h _ {t} + \tilde {s} _ {t}.\tag{17}
$$

where we share upsampling parameters across B bins (default $B \ = \ 1 6 )$ , making the overhead negligible while matching per-position performance. 

## 3.5 DECODER: SYMMETRIC ARCHITECTURE FOR NEXT BYTE PREDICTION

The decoder uses identical architecture to the local encoder (sliding window attention + Canon layers) operating on upsampled representations $s _ { 1 : T } \rangle$ 

$$
\hat {p} (x _ {t + 1} | x _ {1: t}) = \operatorname{softmax} (\operatorname{Transformer} _ {\text { decoder }} (s _ {1: T}) _ {t} W _ {\text { out }}),\tag{18}
$$

where $W _ { \mathrm { o u t } } \in \mathbb { R } ^ { d _ { \mathrm { l o c a l } } \times | V | }$ projects to byte vocabulary. The symmetric encoder-decoder design ensures consistent processing while the global transformer concentrates computational resources on high-level modeling. 

## 4 EXPERIMENTS

We follow a standard pre-training setup at academic scale (Yang et al., 2024; Allen-Zhu, 2025) where ablations are done with matched FLOPs at the GPT-3 Large level and scaling experiments are run at GPT-3 XL scale. Training details are provided in Appendix C.2. 

## 4.1 EXPERIMENTAL SETUP

Pretraining Dataset. All models are trained from scratch on the FineWeb-Edu-100B (Penedo et al., 2024) corpus, a curated pre-training dataset of educational content comprising approximately 500B training byte tokens. 

Bits-Per-Byte Estimation. We adopt the Bits-Per-Byte (BPB) metric following established practices in recent literature (Xue et al., 2022b; Yu et al., 2023a; Wang et al., 2024). BPB normalizes cross-entropy loss by byte count rather than token count: 

$$
\mathrm{BPB} (\mathbf {x}) = \frac {\mathcal {L} _ {C E} (\mathbf {x})}{\ln (2) \cdot n _ {\mathrm{bytes}}}\tag{19}
$$

where $\mathcal { L } _ { C E } ( \mathbf { x } )$ is the cross-entropy loss over data x and $n _ { \mathrm { b y t e s } }$ is the total bytes in x. 

Downstream Tasks. Due to the scale of pretraining, we focus primarily on BPB loss and selected zero-shot downstream tasks from the lm-eval-harness (Gao et al., 2024) (e.g., HEL-LASWAG (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2019), BOOLQ (Clark et al., 2019), PIQA (Bisk et al., 2020), ARC (Clark et al., 2018)) for the ByteFlow Net runs. The baseline decoder-only transformer variant is validated on a held-out FineWeb-Edu split every 1000 steps. 

## 4.2 BASELINES

We compare against several representative architectures: 

• Standard Transformer: LLaMA (Touvron et al., 2023; Dubey et al., 2024), trained with a fixed BPE tokenizer. This serves as the strong tokenized baseline. 


600M Model on 50B Training Byte Tokens



1.3B Model on 500B Training Byte Tokens


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/064fe681557cba048f97675d5b8e53a0c5071193ad86b5e2d156b2f752605712.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/d39f1fc81f8ccbe217a7461b63d556d5655ee35cbd02bb1068f7954da0fe4d2e.jpg)



Figure 2: Scaling Trend for Different Architecture Comparison. Validation BPB loss (lower is better) for different architecture approaches on two different scale (600M, left) and (1.3B, right) models. ByteFlow Net achieves better performance with scaling to larger models and data recipe.



Table 1: Zero-shot performance comparison across multiple benchmarks. Evaluation results on six downstream tasks at both 0.6B (50B tokens) and 1.3B (500B tokens) scales. We report average scores over three separate runs to ensure fair comparison.


<table><tr><td rowspan="2">Model</td><td rowspan="2">Tokenizer</td><td colspan="7">Accuracy (↑)</td></tr><tr><td>HellaSwag</td><td>WinoGrande</td><td>BoolQ</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>Average</td></tr><tr><td colspan="9">600M Models Trained on 50B Tokens (1x Chincilla Ratio (Hoffmann et al., 2022))</td></tr><tr><td>LLaMA (Dubey et al., 2024)</td><td>BPE</td><td><eq>43.12_{±0.87}</eq></td><td><eq>42.74_{±1.92}</eq></td><td><eq>62.26_{±0.64}</eq></td><td><eq>59.43_{±1.25}</eq></td><td><eq>61.38_{±0.98}</eq></td><td><eq>25.95_{±1.76}</eq></td><td><eq>49.15_{±0.73}</eq></td></tr><tr><td>LlamaByte (Dubey et al., 2024)</td><td rowspan="5">Byte</td><td><eq>37.93_{±1.83}</eq></td><td><eq>41.84_{±0.59}</eq></td><td><eq>61.15_{±1.47}</eq></td><td><eq>58.31_{±0.91}</eq></td><td><eq>60.24_{±1.68}</eq></td><td><eq>25.18_{±0.52}</eq></td><td><eq>47.44_{±1.29}</eq></td></tr><tr><td>MambaByte (Wang et al., 2024)</td><td><eq>38.21_{±0.76}</eq></td><td><eq>41.97_{±1.95}</eq></td><td><eq>61.48_{±1.14}</eq></td><td><eq>58.67_{±0.68}</eq></td><td><eq>60.53_{±1.87}</eq></td><td><eq>25.42_{±1.03}</eq></td><td><eq>47.71_{±0.85}</eq></td></tr><tr><td>SpaceByte (Slagle, 2024b)</td><td><eq>37.76_{±1.56}</eq></td><td><eq>42.15_{±0.82}</eq></td><td><eq>61.04_{±1.39}</eq></td><td><eq>58.18_{±1.71}</eq></td><td><eq>60.12_{±0.55}</eq></td><td><eq>25.05_{±1.98}</eq></td><td><eq>47.38_{±1.22}</eq></td></tr><tr><td>AU-Net (Videau et al., 2025)</td><td><eq>40.34_{±0.93}</eq></td><td><eq>44.12_{±1.44}</eq></td><td><eq>63.85_{±0.71}</eq></td><td><eq>64.87_{±1.16}</eq></td><td><eq>62.91_{±1.89}</eq></td><td><eq>27.43_{±0.65}</eq></td><td><eq>49.38_{±1.22}</eq></td></tr><tr><td>ByteFlow Net (Ours)</td><td><eq>41.42_{±1.35}</eq></td><td><eq>44.93_{±0.78}</eq></td><td><eq>64.48_{±1.62}</eq></td><td><eq>62.25_{±0.94}</eq></td><td><eq>63.87_{±1.17}</eq></td><td><eq>28.36_{±1.81}</eq></td><td><eq>50.89_{±0.89}</eq></td></tr><tr><td colspan="9">1.3B Models Trained on 500B Tokens (4x Chincilla Ratio (Hoffmann et al., 2022))</td></tr><tr><td>LLaMA (Dubey et al., 2024)</td><td>BPE</td><td><eq>54.12_{±1.58}</eq></td><td><eq>53.74_{±1.36}</eq></td><td><eq>73.26_{±1.62}</eq></td><td><eq>70.43_{±1.47}</eq></td><td><eq>72.38_{±1.54}</eq></td><td><eq>36.95_{±1.81}</eq></td><td><eq>60.15_{±1.59}</eq></td></tr><tr><td>LlamaByte (Dubey et al., 2024)</td><td rowspan="5">Byte</td><td><eq>48.93_{±1.46}</eq></td><td><eq>52.84_{±1.68}</eq></td><td><eq>72.15_{±1.39}</eq></td><td><eq>69.31_{±1.52}</eq></td><td><eq>71.24_{±1.43}</eq></td><td><eq>36.18_{±1.67}</eq></td><td><eq>58.44_{±1.55}</eq></td></tr><tr><td>MambaByte (Wang et al., 2024)</td><td><eq>49.21_{±1.35}</eq></td><td><eq>52.97_{±1.57}</eq></td><td><eq>72.48_{±1.48}</eq></td><td><eq>69.67_{±1.71}</eq></td><td><eq>71.53_{±1.76}</eq></td><td><eq>36.42_{±1.34}</eq></td><td><eq>58.71_{±1.53}</eq></td></tr><tr><td>SpaceByte (Slagle, 2024b)</td><td><eq>48.76_{±1.64}</eq></td><td><eq>53.15_{±1.42}</eq></td><td><eq>72.04_{±1.56}</eq></td><td><eq>69.18_{±1.38}</eq></td><td><eq>71.12_{±1.69}</eq></td><td><eq>36.05_{±1.41}</eq></td><td><eq>58.38_{±1.54}</eq></td></tr><tr><td>AU-Net (Videau et al., 2025)</td><td><eq>50.34_{±1.51}</eq></td><td><eq>54.12_{±1.45}</eq></td><td><eq>73.85_{±1.63}</eq></td><td><eq>74.87_{±1.37}</eq></td><td><eq>72.91_{±1.59}</eq></td><td><eq>37.43_{±1.82}</eq></td><td><eq>60.59_{±1.56}</eq></td></tr><tr><td>ByteFlow Net (Ours)</td><td><eq>55.42_{±1.44}</eq></td><td><eq>56.93_{±1.69}</eq></td><td><eq>76.48_{±1.38}</eq></td><td><eq>74.25_{±1.61}</eq></td><td><eq>75.87_{±1.46}</eq></td><td><eq>40.36_{±1.74}</eq></td><td><eq>63.19_{±1.57}</eq></td></tr></table>


• Byte-level isotropic models: LlamaByte (pure Llama layers on byte-level modeling) and MambaByte (Wang et al., 2024) process raw UTF-8 bytes without hierarchy. 


• Heuristic chunkers: SpaceByte (Slagle, 2024b) and AU-Net (Videau et al., 2025) uses whitespace-like delimiters for chunking. 

• ByteFlow Net: Our proposed architecture, where chunk boundaries are chosen online via the lossy coding-rate criterion (section 3). 

Byte/BPE models are trained on sequence lengths of 8192/2048 respectively, and for ByteFlow Net and AU-Net we use hierarchical sequence lengths $\mathrm { ( 8 1 9 2  3 2 0 0  8 1 9 2 ) }$ . All detailed model configurations are provided in Appendix C and further abaltion in Appendix D for reference. 4.3 SCALING EXPERIMENTS 

Superior Scaling Behavior. The scaling curves in Figure 2 reveal encouraging trends for Byte-Flow Net across both model sizes. At the 600M parameter scale, ByteFlow Net demonstrates steady improvement throughout training, eventually surpassing the LLaMA baseline around the 25B token mark and maintaining this advantage through 50B tokens. The 1.3B results show even more promising behavior, with ByteFlow Net exhibiting the most favorable scaling trajectory among all tested architectures, suggesting that our approach becomes increasingly effective as we scale up both model size and training data. 

Competitive Performance on Downstream Tasks. Our performance results in Table 1 demonstrate that ByteFlow Net achieves competitive results with traditional tokenization approaches while operating directly on raw bytes. At the 600M scale, ByteFlow Net reaches 50.89% average accuracy compared to LLaMA’s 49.15%, representing a modest but consistent improvement of 1.74 points. 

The gap becomes more substantial at 1.3B parameters that suggests the benefits of our approach become more pronounced with scale compared to LLaMA baseline. 

Character-level Performance. As shown in Table 2 ByteFlow Net 1.3B substantially outperforms Llama 3 variants on CUTE despite 20-32× less training data, with exceptional orthographic capabilities evidenced by nearperfect Spelling Inverse performance. This demonstrates that architectural design can compensate for scale in character-level tasks. 


Table 2: Performance on character-level benchmark (Edman et al., 2024).<sup>*</sup>Baseline results are taken from Pagnoni et al. (2024).


<table><tr><td></td><td>Llama 3*(1T tokens)</td><td>Llama 3.1*(16T tokens)</td><td>ByteFlow Net 1.3B(500B tokens)</td></tr><tr><td>CUTE</td><td>27.5</td><td>20.0</td><td><eq>\underline{51.2 \pm 2.1}</eq></td></tr><tr><td>- Contains Char</td><td>0.0</td><td>0.0</td><td><eq>\underline{52.8 \pm 3.2}</eq></td></tr><tr><td>- Contains Word</td><td>55.1</td><td>21.6</td><td><eq>\underline{70.1 \pm 2.8}</eq></td></tr><tr><td>- Del Char</td><td>34.6</td><td>34.3</td><td><eq>\underline{33.2 \pm 1.9}</eq></td></tr><tr><td>- Del Word</td><td>75.5</td><td>84.5</td><td><eq>\underline{73.4 \pm 2.6}</eq></td></tr><tr><td>- Ins Char</td><td>7.5</td><td>0.0</td><td><eq>\underline{16.9 \pm 1.4}</eq></td></tr><tr><td>- Ins Word</td><td>33.5</td><td>63.3</td><td><eq>\underline{28.7 \pm 2.3}</eq></td></tr><tr><td>- Spelling Inverse</td><td>30.1</td><td>3.6</td><td><eq>\underline{95.1 \pm 2.4}</eq></td></tr><tr><td>- Substitute Char</td><td>0.4</td><td>1.2</td><td><eq>\underline{45.3 \pm 2.9}</eq></td></tr><tr><td>- Substitute Word</td><td>16.4</td><td>6.8</td><td><eq>\underline{68.9 \pm 2.2}</eq></td></tr><tr><td>- Swap Char</td><td>2.6</td><td>2.4</td><td><eq>\underline{10.1 \pm 1.6}</eq></td></tr></table>

## Training-time efficiency profiling. We pro-

file controlled runs on 8×A100-80GB with matched FLOPs budgets during pretraining runs. 

Table 3 shows ByteFlow Net attains a strong efficiency–performance balance: it trains competitively among hierarchical byte models while achieving the best BPB and downstream accuracy. 

## 4.4 ABLATION STUDY:

THE ART OF DECIDING WHERE TO CHUNK 


Table 3: Training-time efficiency comparison at 0.6B scale (50B tokens). WPS = words/sec ×10<sup>4</sup>.


<table><tr><td>Model</td><td>FLOPs (<eq>\times 10^{21}</eq>)</td><td>WPS<eq>\uparrow</eq></td><td>Iter(s)<eq>\downarrow</eq></td><td>Val BPB<eq>\downarrow</eq></td></tr><tr><td>LLaMA (BPE)</td><td>1.02</td><td>9.3</td><td>3.8</td><td>0.89</td></tr><tr><td>AU-Net (heur.)</td><td>1.04</td><td>8.8</td><td>4.1</td><td>0.91</td></tr><tr><td>Cosine chunking</td><td>1.02</td><td>7.3</td><td>3.8</td><td>0.92</td></tr><tr><td>ByteFlow (log-det)</td><td>1.07</td><td>7.9</td><td>4.0</td><td>0.86</td></tr><tr><td>ByteFlow (L2 approx.)</td><td>1.01</td><td>8.5</td><td>3.9</td><td>0.87</td></tr></table>

To truly understand what makes a tokenizerfree model tick, we have to isolate the most crit-

ical decision it makes: where to draw the line between chunks. This is often a messy comparison, as different architectures are bundled with their own unique chunking logics. To cut through the noise, we ran a controlled experiment: we took the ByteFlow Net architecture and swapped out its chunking module with seven different strategies in Table 4. All ablation experiments were conducted at the 0.6B parameter scale on 50B training tokens. 

The Effect of Heuristic-based Chunking. A crucial negative control reveals that randomly choosing chunk boundaries is a disaster. It shatters any hope of learning, leading to the worst performance by a wide margin with a 41.34% task accuracy. This proves that the hierarchy itself isn’t magic: the segmentation must be meaningful. This makes the performance of simple word-boundary chunking all the more remarkable. A static, rule-based strategy—essentially just splitting on spaces and punctuation—doesn’t just work; it match the standard LLaMA baseline on downstream tasks (49.38% vs. 49.15%). This powerful insight shows that a linguistically-aware segmentation can be somtimes more effective than a sophisticated but less effective dynamic chunking like entropy or cosine-based. 

The Advantage of Coding Rate Segmentation. While other dynamic methods, like those based on neural predictions or cosine similarity, show promise, they struggle to consistently beat the simple word boundary baseline. This highlights a critical challenge: learning to find meaningful boundaries on the fly is hard. This is where our approach is. By framing the decision as a matter of compression, our lossy coding-rate method outperforms all contenders in this scale. It achieves the lowest validation BPB loss at 0.86 and the highest average task accuracy at 50.89%, a significant leap over the next-best strategy. This victory suggests that the optimal way to segment a sequence isn’t based on what looks similar or what’s locally surprising, but on what provides the most new information to the sequence as a whole, and teach model to compress the input itself during optimization. 


Entropy Chunking



Table 4: Ablation of Different Chunking Strategies for Hierarchical Language Models. We train on ByteFlow Net but ablate on different chunker used in different architecture. Experiments are done on 0.6B on 50B training token scale. We report average scores over three separate runs.


<table><tr><td>Method</td><td>Type</td><td>Formulation</td><td>Complexity</td><td>Validation BPB Loss (↓)</td><td>Task Perf. (↑)</td></tr><tr><td>LLaMA Baseline</td><td>-</td><td>-</td><td>-</td><td><eq>\underline{0.89}_{\pm 0.003}</eq></td><td><eq>\underline{49.15}_{\pm 0.73}</eq></td></tr><tr><td>Fixed Stride (Yu et al., 2023a)</td><td>Static</td><td><eq>S = \{i \cdot w : i \in \mathbb{N}, i \cdot w \leq T\}</eq></td><td><eq>O(1)</eq></td><td><eq>0.96_{\pm 0.012}</eq></td><td><eq>45.27_{\pm 1.32}</eq></td></tr><tr><td>Word Boundaries (Slagle, 2024b)</td><td>Static</td><td><eq>S = \{t : x_t \in \{space, punct\}\}</eq></td><td><eq>O(T)</eq></td><td><eq>0.94_{\pm 0.008}</eq></td><td><eq>\underline{49.38}_{\pm 1.22}</eq></td></tr><tr><td>Random Chunking</td><td>Dynamic</td><td><eq>P(boundary\ at\ t) = p_{rand}</eq></td><td><eq>O(T)</eq></td><td><eq>1.04_{\pm 0.017}</eq></td><td><eq>41.34_{\pm 1.67}</eq></td></tr><tr><td>Neural Boundary (Nawrot et al., 2023)</td><td>Dynamic</td><td><eq>p_t = \sigma(h_t W_{bound})</eq><eq>b_t \sim Gumbel(p_t)</eq></td><td><eq>O(T \cdot d)</eq></td><td><eq>0.90_{\pm 0.006}</eq></td><td><eq>47.13_{\pm 0.84}</eq></td></tr><tr><td>Entropy Chunking (Pagnoni et al., 2024)</td><td>Dynamic</td><td><eq>H_t = -\sum_v P(v|h_t) \log P(v|h_t)</eq><eq>S = \text{Top-K}(\{H_t\}_{t=1}^T)</eq></td><td><eq>O(T \cdot |V|)</eq></td><td><eq>0.91_{\pm 0.007}</eq></td><td><eq>47.81_{\pm 0.95}</eq></td></tr><tr><td>Cosine Similarity (Hwang et al., 2025a)</td><td>Dynamic</td><td><eq>\text{sim}_t = \frac{h_t \cdot h_{t-1}}{\|h_t\|\|h_{t-1}\|}</eq><eq>S = \text{Top-K}(\{1 - \text{sim}_t\}_{t=1}^T)</eq></td><td><eq>O(T \cdot d)</eq></td><td><eq>0.92_{\pm 0.009}</eq></td><td><eq>47.45_{\pm 1.08}</eq></td></tr><tr><td>Lossy Coding Rate</td><td>Dynamic</td><td><eq>\Delta R_t = R_\varepsilon(h_{1:t}) - R_\varepsilon(h_{1:t-1})</eq><eq>S = \text{Top-K}(\{\Delta R_t\}_{t=2}^T)</eq></td><td><eq>O(T \cdot d)</eq></td><td><eq>\underline{0.86}_{\pm 0.004}</eq></td><td><eq>\underline{50.89}_{\pm 0.89}</eq></td></tr></table>

## Chunking Strategy Impact on Latent Representation Manifolds

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/d7af2584702cccc63478ea2fe3cace86f5ab2a1dfb4c93fb8fbb5ef3622990c5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/d5d5ac948a512937c4779018931cb6ac20632d97150aff2bbe09e0b45511f584.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/b0c8ea602a3b244619bd63d4e31d2e63276132d021acee462b17ae2569ee3dcf.jpg)



Cosine Similarity Chunking


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/7e1b9a62a258bf16dd2877c04cf6b3913cee1b3a33d4bf331fd2c51c89f136e7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/0fe472a4da7e8fc1715d36ea83d4d60b42707edf2af5dd8281db92a7dafc2025.jpg)



Coding Rate Chunking (Ours)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/ec0c6706b9c6cda94214e57fdcbb832aad80c72088b1a4090c8708e800ece72f.jpg)



Figure 3: Chunking Strategy Impact on Latent Representation Manifolds. Each point is a contextualized byte representation after the local encoder (after 1B training bytes), projected to 2D by t-SNE. Poor chunking (random, neural boundaries) fragments the original clustering, whereas coding-rate chunking preserves it.


Preserving Latent Manifolds and Dynamically Allocating FLOPs. Why does coding rate work so well? We hypothesize it’s about two things: geometry and adaptability. As visualized in Figure 3, poor chunking strategies like random selection effectively shatter the underlying structure of the data in the representation space, leaving the model to learn from a fragmented mess. Our codingrate approach, in contrast, excels at preserving a coherent latent manifold, making it far easier for the global transformer to identify patterns. This links directly to the idea of dynamically assigning FLOPs. The coding rate criterion is essentially an importance detector. By only promoting bytes with high information gain to the global level, the model is forced to spend its precious computational budget on the parts of the sequence that actually matter. It learns to focus its deep, wide global transformer on a compressed stream of significant events, rather than wasting resources on redundant or predictable byte patterns. As shown in our case study (Figure 4), the model learns to assign higher rates to semantically significant bytes (e.g., key nouns), forcing the model to focus its computational budget on a compressed stream of meaningful information rather than redundant patterns. This strategic allocation makes processing more efficient and effective 


Character-Level Coding Rate Scores


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/f75b26fd-0649-48fa-833c-44106efe11ec/625193b0e685434d30de093f5290fc62cfd3610726081b22f11c7563df9dcb1f.jpg)



Figure 4: Case Study of Character-Level Coding Rate Scores. This figure illustrates how Byte-Flow Net assigns an information-theoretic “importance” score to each character in an example sentence. The model has learned to assign a higher coding rate to characters that are more semantically significant, such as the initial letters of words and key entities. Conversely, it assigns lower rates to more predictable characters within words. This demonstrates the model’s ability to dynamically identify information-rich points in the byte stream to guide its chunking and resource allocation.


## 5 CONCLUSION

This work introduced ByteFlow Net, a hierarchical architecture that learns to parse raw data on its own terms. Grounded in information theory, our model reframes segmentation as a dynamic compression task, using a coding-rate objective to intelligently identify meaningful semantic units without a fixed vocabulary. This principled approach is not merely theoretical; extensive experiments show that ByteFlow Net consistently outperforms strong BPE-based transformers and other byte-level models, exhibiting a superior scaling trajectory as model size increases. Crucially, our ablation studies confirmed that the coding-rate criterion is the key to this success, decisively surpassing other dynamic chunking strategies by preserving the underlying geometry of the data’s latent manifold. This allows the model to strategically allocate its computational budget, focusing its most powerful components on a compressed stream of what is truly informative. Our results therefore provide compelling evidence that end-to-end, tokenizer-free modeling is not only feasible but is a more effective and robust paradigm for language modeling. 

## ETHICS STATEMENT

This work does not involve human subjects, personally identifiable information, or sensitive data. All experiments are conducted on publicly available and curated datasets (e.g., FineWeb-Edu-100B (Penedo et al., 2024)) that have been filtered to minimize risks of privacy violations or exposure of harmful content. Our research focuses on architectural design for tokenizer-free language modeling and does not aim to produce harmful applications. We are mindful of potential misuse of language models, including risks related to bias, misinformation, or malicious generation, and encourage responsible downstream use in line with the ICLR Code of Ethics. No conflicts of interest or external sponsorships influence this work. 

## REPRODUCIBILITY STATEMENT.

We have taken multiple steps to ensure the reproducibility of our work. The architecture of Byte-Flow Net, including all encoder, chunking, and global transformer components, is described in detail in Section 3, with ablation studies and comparisons provided in Section 4. Implementation details such as model sizes, FLOPs-matched training recipes, optimizer settings, and hyperparameters are included in Appendix C, while theoretical derivations of the coding-rate objective and its approximations are provided in Appendix A and B. All datasets used in our experiments are publicly available; we rely on the FineWeb-Edu-100B corpus, and we document the preprocessing and filtering procedures in Appendix C to support replication of data pipelines. We also provide extensive ablation studies in Section 4 and Figure 3 to demonstrate robustness of our results. 

## REFERENCES



Zeyuan Allen-Zhu. Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers. SSRN Electronic Journal, May 2025. https://ssrn.com/abstract= 5240330. 





Yonatan Belinkov and Yonatan Bisk. Synthetic and natural noise both break neural machine translation, 2018. URL https://arxiv.org/abs/1711.02173. 





Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020. 





Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. 





Thomas M Cover. Elements of information theory. John Wiley & Sons, 1999. 





DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, and etc. Deepseekr1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https: //arxiv.org/abs/2501.12948. 





Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pp. arXiv–2407, 2024. 





Lukas Edman, Helmut Schmid, and Alexander Fraser. CUTE: Measuring LLMs’ understanding of their tokens. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 3017–3026, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.177. URL https://aclanthology.org/2024. emnlp-main.177/. 





Matthias Galle. Investigating the effectiveness of BPE: The power of shorter sequences. In Ken-´ taro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 1375–1381, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1141. URL https://aclanthology.org/D19-1141/. 





Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602. 





Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, and etc. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783. 





Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, pp. 30016–30030, 2022. 





Sukjun Hwang, Brandon Wang, and Albert Gu. Dynamic chunking for end-to-end hierarchical sequence modeling. arXiv preprint arXiv:2507.07955, 2025a. 





Sukjun Hwang, Brandon Wang, and Albert Gu. Dynamic chunking for end-to-end hierarchical sequence modeling, 2025b. URL https://arxiv.org/abs/2507.07955. 





Julie Kallini, Shikhar Murty, Christopher D Manning, Christopher Potts, and Robert Csord´ as. Mrt5:´ Dynamic token merging for efficient byte-level language models. In The Thirteenth International Conference on Learning Representations. 





Taku Kudo and John Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. In Eduardo Blanco and Wei Lu (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66–71, Brussels, Belgium, November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-2012. URL https://aclanthology.org/ D18-2012/. 





Alisa Liu, Jonathan Hayase, Valentin Hofmann, Sewoong Oh, Noah A. Smith, and Yejin Choi. SuperBPE: Space travel for language models. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=lcDRvffeNP. 





Yi Ma, Harm Derksen, Wei Hong, and John Wright. Segmentation of multivariate mixed data via lossy data coding and compression. IEEE transactions on pattern analysis and machine intelligence, 29(9):1546–1562, 2007. 





Piotr Nawrot, Szymon Tworkowski, Michał Tyrolski, Łukasz Kaiser, Yuhuai Wu, Christian Szegedy, and Henryk Michalewski. Hierarchical transformers are more efficient language models, 2022. URL https://arxiv.org/abs/2110.13711. 





Piotr Nawrot, Jan Chorowski, Adrian Lancucki, and Edoardo Maria Ponti. Efficient transformers with dynamic token pooling. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 6403–6417, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.353. URL https://aclanthology.org/ 2023.acl-long.353/. 





Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, et al. Byte latent transformer: Patches scale better than tokens. arXiv preprint arXiv:2412.09871, 2024. 





Artidoro Pagnoni, Ramakanth Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason E Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, and Srini Iyer. Byte latent transformer: Patches scale better than tokens. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 9238–9258, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.453. URL https://aclanthology.org/2025.acl-long.453/. 





Guilherme Penedo, Hynek Kydl´ıcek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandroˇ Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale, 2024. 





Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi. Language model tokenizers introduce unfairness between languages, 2023. URL https://arxiv.org/abs/2305. 15425. 





Phillip Rust, Jonas Pfeiffer, Ivan Vulic, Sebastian Ruder, and Iryna Gurevych. How good is your tokenizer? on the monolingual performance of multilingual language models. ArXiv, abs/2012.15613, 2020. URL https://api.semanticscholar.org/CorpusID: 229924220. 





Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale, 2019. URL https://arxiv.org/abs/1907. 10641. 





Craig W Schmidt, Varshini Reddy, Haoran Zhang, Alec Alameddine, Omri Uzan, Yuval Pinter, and Chris Tanner. Tokenization is more than compression. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), ProceeDo All Languages Cost the Same? Tokenization in the Era of Commercial Language Modelsdings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 678–702, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.40. URL https://aclanthology.org/2024.emnlp-main.40/. 





Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. 2015. 





Kevin Slagle. Spacebyte: Towards deleting tokenization from large language modeling. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 124925–124950. Curran Associates, Inc., 2024a. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ e1f418450107c4a0ddc16d008d131573-Paper-Conference.pdf. 





Kevin Slagle. Spacebyte: Towards deleting tokenization from large language modeling. Advances in Neural Information Processing Systems, 37:124925–124950, 2024b. 





Lichao Sun, Kazuma Hashimoto, Wenpeng Yin, Akari Asai, Jia Li, Philip Yu, and Caiming Xiong. Adv-bert: Bert is not robust on misspellings! generating nature adversarial samples on bert, 2020. URL https://arxiv.org/abs/2003.04985. 





Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Rame, Morgane Rivi´ ere, Louis Rouillard, Thomas` Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and etc. Gemma 3 technical report, 2025. URL https://arxiv. org/abs/2503.19786. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 





Mathurin Videau, Badr Youbi Idrissi, Alessandro Leite, Marc Schoenauer, Olivier Teytaud, and David Lopez-Paz. From bytes to ideas: Language modeling with autoregressive u-nets, 2025. URL https://arxiv.org/abs/2506.14761. 





Junxiong Wang, Tushaar Gangavarapu, Jing Nathan Yan, and Alexander M Rush. Mambabyte: Token-free selective state space model. arXiv preprint arXiv:2401.13660, 2024. 





Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. ByT5: Towards a token-free future with pre-trained byte-to-byte models. Transactions ofthe Associationfor Computational Linguistics, 10:291–306, 2022a. doi: 10.1162/tacl a 00461. URL https://aclanthology.org/2022.tacl-1.17/. 





Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. Byt5: Towards a token-free future with pre-trained byte-to-byte models, 2022b. URL https://arxiv.org/abs/2105.13626. 





An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, and etc. Qwen3 technical report, 2025. URL https://arxiv.org/ abs/2505.09388. 





Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In Proceedings of the 41st International Conference on Machine Learning, pp. 56501–56523, 2024. 





Gilad Yehudai, Haim Kaplan, Asma Ghandeharioun, Mor Geva, and Amir Globerson. When can transformers count to n?, 2024. URL https://arxiv.org/abs/2407.15160. 





Lili Yu, Daniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lewis.´ Megabyte: Predicting million-byte sequences with multiscale transformers. Advances in Neural Information Processing Systems, 36:78808–78823, 2023a. 





Lili Yu, Daniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, and Mike Lewis.´ Megabyte: Predicting million-byte sequences with multiscale transformers, 2023b. URL https://arxiv.org/abs/2305.07185. 





Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019. 





Xiang Zhang, Juntai Cao, and Chenyu You. Counting ability of large language models and impact of tokenization, 2024. URL https://arxiv.org/abs/2410.19730. 





Vilem Zouhar, Clara Meister, Juan Gastaldi, Li Du, Tim Vieira, Mrinmaya Sachan, and Ryan´ Cotterell. A formal perspective on byte-pair encoding. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023, pp. 598–614, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.38. URL https://aclanthology.org/2023. findings-acl.38/. 



## A DERIVATION OF THE LOSSY CODING RATE FORMULA

Consider a sequence of contextualized representations $h _ { 1 : T } \in \mathbb { R } ^ { T \times d _ { \mathrm { l o c a l } } }$ produced by a local encoder. We seek to determine the minimum rate required to encode this sequence with a specified distortion level using rate-distortion theory. Let $X = h _ { 1 : T }$ be our source sequence and $\hat { X }$ be the reconstructed sequence after lossy compression, with distortion defined as $D = \mathbb { E } [ \| X - \hat { X } \| _ { F } ^ { 2 } ]$ where $\left\| \cdot \right\| _ { F }$ denotes the Frobenius norm. 

We model the representations as following a multivariate Gaussian distribution, which is reasonable for deep neural network representations. Specifically, $\mathrm { v e c } ( h _ { 1 : T } ) \sim { \mathcal { N } } ( 0 , \Sigma )$ where $\mathrm { v e c } ( \cdot )$ vectorizes the matrix and $\Sigma \in \mathbb { R } ^ { T d _ { \mathrm { l o c a l } } \times T d _ { \mathrm { l o c a l } } }$ is the covariance matrix. For local representations, we assume the structured covariance $\begin{array} { r } { \Sigma = I _ { T } \otimes \frac { H } { T } } \end{array}$ where $H = h _ { 1 \cdot T } ^ { T } h _ { 1 : T } \in \mathbb { R } ^ { d _ { \mathrm { l o c a l } } \times d _ { \mathrm { l o c a l } } }$ is the empirical covariance matrix, ⊗ is the Kronecker product, and $I _ { T }$ is the $\Ddot { T } ^ { * } \times T$ identity matrix. 

For a multivariate Gaussian source with covariance matrix $\Sigma ,$ the rate-distortion function with mean squared error distortion is: 

$$
R (D) = \frac {1}{2} \sum_ {i = 1} ^ {n} \max \left(0, \log \frac {\lambda_ {i}}{\theta}\right)\tag{20}
$$

where $\lambda _ { i }$ are the eigenvalues of $\Sigma , \theta$ satisfies $\begin{array} { r } { \sum _ { i = 1 } ^ { n } \operatorname* { m i n } ( \lambda _ { i } , \theta ) = D _ { ! } } \end{array}$ , and $n = T d _ { \mathrm { l o c a l } }$ is the total dimensionality. Instead of specifying distortion directly, we parametrize using noise variance $\varepsilon ^ { 2 }$ corresponding to adding Gaussian noise with variance $\varepsilon ^ { 2 }$ during reconstruction, giving $\theta = \varepsilon ^ { 2 }$ 

Given our covariance structure, the eigenvalues of Σ are $\{ \lambda _ { i } \} _ { i = 1 } ^ { T d _ { \mathrm { l o c a l } } } = \{ \mu _ { j } / T \} _ { j = 1 } ^ { d _ { \mathrm { l o c a l } } }$ (each repeated T times), where $\{ \mu _ { j } \}$ are eigenvalues of $H = h _ { 1 : T } ^ { T } h _ { 1 : T }$ . Substituting into the rate-distortion formula: 

$$
R (\varepsilon^ {2}) = \frac {1}{2} \sum_ {j = 1} ^ {d _ {\mathrm{local}}} T \cdot \max \left(0, \log \frac {\mu_ {j} / T}{\varepsilon^ {2}}\right)\tag{21}
$$

$$
= \frac {1}{2} \sum_ {j = 1} ^ {d _ {\mathrm{local}}} \max \left(0, \log \frac {\mu_ {j}}{\varepsilon^ {2}}\right)\tag{22}
$$

Using the identity max $\mathbf { \eta } _ { : } ( 0 , \log ( x ) ) = \log ( \operatorname* { m a x } ( 1 , x ) )$ and the fact that for a matrix A with eigenvalues $\{ \mu _ { j } \}$ , we have $\begin{array} { r } { \prod _ { j } \operatorname* { m a x } ( \bar { 1 } , \mu _ { j } / \varepsilon ^ { 2 } ) = \operatorname* { d e t } ( \operatorname* { m a x } ( \bar { I } , A / \varepsilon ^ { 2 } ) ) } \end{array}$ ), we obtain: 

$$
R (\varepsilon^ {2}) = \frac {1}{2} \log \prod_ {j = 1} ^ {d _ {\mathrm{local}}} \max \left(1, \frac {\mu_ {j}}{\varepsilon^ {2}}\right)\tag{23}
$$

$$
= \frac {1}{2} \log \det \left(\max \left(I, \frac {h _ {1 : T} ^ {T} h _ {1 : T}}{\varepsilon^ {2}}\right)\right)\tag{24}
$$

Through matrix algebraic manipulation and using the fact that we can rewrite the determinant in terms of the original representation matrix, we arrive at the final form: 

$$
R _ {\varepsilon} (h _ {1: T}) = \frac {1}{2} \log \det \left(I + \frac {d _ {\text { local }}}{\varepsilon^ {2}} h _ {1: T} h _ {1: T} ^ {T}\right)\tag{25}
$$

This lossy coding rate quantifies the minimum bits needed to encode sequence $h _ { 1 : T }$ with reconstruction error approximately $\varepsilon ^ { 2 }$ per component. The determinant captures the effective dimensionality of the representation space—large eigenvalues of $h _ { 1 : T } h _ { 1 : T } ^ { T }$ indicate high-information directions requiring more bits for preservation, while the noise variance parameter $\overline { { \varepsilon ^ { 2 } } }$ controls the sensitivity of the coding rate computation. 

## B L2 NORM APPROXIMATION FOR LOSSY CODING RATE

We derive a computationally efficient approximation to the lossy coding rate formula in equation (11) for streaming applications where quick local decisions are required. Starting from the exact 

formula: 

$$
R _ {\varepsilon} (h _ {1: T}) = \frac {1}{2} \log \det \left(I + \frac {d _ {\text {local}}}{\varepsilon^ {2}} h _ {1: T} h _ {1: T} ^ {T}\right)\tag{26}
$$

Let $\begin{array} { r } { A = \frac { d _ { \mathrm { l o c a l } } } { \varepsilon ^ { 2 } } h _ { 1 : T } h _ { 1 : T } ^ { T } \in \mathbb { R } ^ { T \times T } } \end{array}$ be the matrix inside the determinant. For moderate noise variance $\varepsilon ^ { 2 }$ relative to the representation magnitudes, we can consider the regime where the eigenvalues of A are not extremely large, allowing us to use the matrix logarithm expansion. 

Using the matrix identity log det $( I + A ) = \operatorname { t r } ( \log ( I + A ) )$ and the Taylor series expansion of the matrix logarithm for $\| A \| < 1 \colon$ 

$$
\log (I + A) = A - \frac {A ^ {2}}{2} + \frac {A ^ {3}}{3} - \dots\tag{27}
$$

For the first-order approximation when A has moderate eigenvalues, we retain only the linear term: 

$$
\log \det (I + A) \approx \operatorname{tr} (A) = \operatorname{tr} \left(\frac {d _ {\text {local}}}{\varepsilon^ {2}} h _ {1: T} h _ {1: T} ^ {T}\right)\tag{28}
$$

Using the cyclic property of trace, $\operatorname { t r } ( A B ) = \operatorname { t r } ( B A )$ 

$$
\mathbf {t r} (h _ {1: T} h _ {1: T} ^ {T}) = \mathbf {t r} (h _ {1: T} ^ {T} h _ {1: T}) = \sum_ {i = 1} ^ {T} \sum_ {j = 1} ^ {d _ {\mathrm{local}}} h _ {i, j} ^ {2} = \| h _ {1: T} \| _ {F} ^ {2}\tag{29}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm. 

Substituting this result back into our approximation: 

$$
R _ {\varepsilon} (h _ {1: T}) \approx \frac {1}{2} \cdot \frac {d _ {\mathrm{local}}}{\varepsilon^ {2}} \| h _ {1: T} \| _ {F} ^ {2}\tag{30}
$$

For streaming decisions where we need a quick estimate proportional to the information content, we can absorb the constant factors into a scaling parameter and use: 

$$
R _ {\varepsilon} (h _ {1: T}) \propto \| h _ {1: T} \| _ {F} ^ {2}\tag{31}
$$

Since the Frobenius norm is equivalent to the L2 norm for matrices (treating the matrix as a flattened vector), we have $\Vert h _ { 1 : T } \Vert _ { F } = \Vert \dot { h } _ { 1 : T } \Vert _ { 2 }$ , giving us the final approximation: 

$$
R _ {\varepsilon} (h _ {1: T}) \approx C \cdot \| h _ {1: T} \| _ {2} ^ {2}\tag{32}
$$

where $\begin{array} { r } { C = \frac { d _ { \mathrm { l o c a l } } } { 2 \varepsilon ^ { 2 } } } \end{array}$ is a constant determined by the local dimensionality and noise parameter. 

For practical streaming implementations, this quadratic relationship can be further simplified to a linear approximation $\bar { R } _ { \varepsilon } ( \bar { h _ { 1 : T } } ) \propto \| h _ { 1 : T } \| _ { 2 }$ when making relative comparisons between different representations, as the monotonic relationship is preserved and computational cost is minimized. 

Validity Conditions: This approximation is most accurate when (1) the noise variance $\varepsilon ^ { 2 }$ is sufficiently large relative to $d _ { \mathrm { l o c a l } } \Vert \bar { h } _ { 1 : T } \Vert _ { F } ^ { 2 }$ such that the eigenvalues of A are moderate, (2) the representations $h _ { 1 : T }$ do not have extreme condition numbers that would make the trace approximation poor, and (3) we are primarily interested in relative rankings rather than absolute coding rates. 

## C MODEL CONFIGURATION

## C.1 OVERVIEW

We conduct a comprehensive evaluation across six distinct model architectures at two different scales (600M and 1.3B parameters), resulting in 12 total model configurations. Our experimental framework compares traditional transformer baselines with state-of-the-art byte-level processing architectures and advanced hierarchical chunking-aware models. The model families include: (1) Llama - standard transformers with token-level processing, (2) LlamaByte - byte-level variants of standard transformers, (3) MambaByte - selective state space models with byte processing, (4) SpaceByte - optimized byte-level transformers, (5) AuNet - hierarchical models with regex rate-distortion chunking, and (6) BFlowNet - advanced hierarchical architectures with sophisticated chunking strategies. 


Table 5: Comprehensive Model Architecture Specifications Across Six Model Families and Two Scales.


<table><tr><td>Model Family</td><td>Scale</td><td>Architecture</td><td>Layers</td><td>Hidden Dim</td><td>Heads</td><td>Tokenization</td><td>Chunking</td><td>Canon</td><td>Max Seq Len</td></tr><tr><td rowspan="2">Llama</td><td>600M</td><td>Standard Transformer</td><td>25</td><td>1024</td><td>16</td><td>TikToken</td><td>None</td><td>×</td><td>2048</td></tr><tr><td>1.3B</td><td>Standard Transformer</td><td>25</td><td>2048</td><td>16</td><td>TikToken</td><td>None</td><td>×</td><td>2048</td></tr><tr><td rowspan="2">LlamaByte</td><td>600M</td><td>Standard Transformer</td><td>25</td><td>1024</td><td>16</td><td>Byte-level</td><td>None</td><td>×</td><td>8192</td></tr><tr><td>1.3B</td><td>Standard Transformer</td><td>25</td><td>2048</td><td>16</td><td>Byte-level</td><td>None</td><td>×</td><td>8192</td></tr><tr><td rowspan="2">MambaByte</td><td>600M</td><td>Selective SSM</td><td>24</td><td>1024</td><td>N/A</td><td>Byte-level</td><td>None</td><td>×</td><td>8192</td></tr><tr><td>1.3B</td><td>Selective SSM</td><td>24</td><td>2048</td><td>N/A</td><td>Byte-level</td><td>None</td><td>×</td><td>8192</td></tr><tr><td rowspan="2">SpaceByte</td><td>600M</td><td>Hierarchical (2-level)</td><td>25</td><td>1024</td><td>16</td><td>Byte-level</td><td>Word Boundary</td><td>×</td><td>8192</td></tr><tr><td>1.3B</td><td>Hierarchical (2-level)</td><td>25</td><td>2048</td><td>16</td><td>Byte-level</td><td>Word Boundary</td><td>×</td><td>8192</td></tr><tr><td rowspan="2">AuNet</td><td>600M</td><td>Hierarchical (2-level)</td><td>[6, 20]</td><td>[512, 1536]</td><td>Multi-level</td><td>Byte-level</td><td>Word Boundary</td><td>√</td><td>8192 → 3200 → 8192</td></tr><tr><td>1.3B</td><td>Hierarchical (2-level)</td><td>[8, 22]</td><td>[768, 2048]</td><td>Multi-level</td><td>Byte-level</td><td>Word Boundary</td><td>√</td><td>8192 → 3200 → 8192</td></tr><tr><td rowspan="2">BFlowNet</td><td>600M</td><td>Hierarchical (2-level)</td><td>[6, 20]</td><td>[512, 1536]</td><td>Multi-level</td><td>Byte-level</td><td>Coding-Rate Chunking</td><td>√</td><td>8192 → 3200 → 8192</td></tr><tr><td>1.3B</td><td>Hierarchical (2-level)</td><td>[6, 24]</td><td>[512, 2048]</td><td>Multi-level</td><td>Byte-level</td><td>Coding-Rate Chunking</td><td>√</td><td>8192 → 3200 → 8192</td></tr></table>

## C.2 MODEL ARCHITECTURE SPECIFICATIONS

The architectural specifications presented in Table 5 reveal a systematic exploration of scaling strategies and design paradigms across six model families. Most families follow a consistent scaling approach, offering both 600M and 1.3B parameter versions with hidden dimensions doubling from 1024 to 2048, suggesting these represent standard benchmarks for architectural comparison. The models span three distinct paradigms: traditional Standard Transformers (Llama, LlamaByte) with 25 layers and 16 attention heads, Selective State Space Models (MambaByte) that eliminate attention mechanisms entirely while using 24 layers, and Hierarchical models (SpaceByte, AuNet, BFlowNet) featuring complex 2-level architectures with varying layer distributions and multi-level attention head configurations. 

## C.3 DETAILED ARCHITECTURE ANALYSIS

## C.3.1 BASELINE TRANSFORMERS

Our analysis begins with two baseline transformer architectures. The primary baseline is the canonical Llama model, which employs a traditional token-level attention mechanism with a standard vocabulary. Its design features Rotary Position Embeddings (RoPE) with θ = 10, 000, standard multi-head self-attention, and RMSNorm applied prior to both the attention and feed-forward network layers. The activation function used is SwiGLU. As a direct variant, we include the LlamaByte architecture. This model is architecturally identical to Llama but operates directly on UTF-8 byte sequences, utilizing a vocabulary of just 256 characters. This approach offers universal language support and eliminates out-of-vocabulary issues, though it comes with the challenge of processing significantly longer sequence lengths. 

## C.3.2 ADVANCED BYTE-LEVEL ARCHITECTURES

Moving beyond standard transformers, we explore architectures specifically optimized for byte-level processing. The MambaByte model leverages selective state-space models (SSMs), which confer a significant efficiency advantage with linear O(n) scaling complexity compared to the quadratic $O ( n ^ { 2 } )$ complexity of transformers. Its selection mechanism enables input-dependent state transitions, allowing it to effectively manage extended context windows of up to 4096 tokens with constant memory usage. In contrast, the SpaceByte architecture introduces an entropy-driven approach to byte-level processing. It uses an adaptive chunking strategy to segment sequences based on information boundaries, allowing for dynamic chunk sizes that adapt to content complexity. This intelligent boundary detection, combined with specialized attention patterns, enhances its overall performance and efficiency. 

## C.3.3 HIERARCHICAL CHUNKING ARCHITECTURES

We also evaluate two-level hierarchical models designed for sophisticated chunking. The AuNet architecture implements multi-resolution processing through dual-level attention with [512, 4096] sliding windows. It integrates a Canon layer with 4-token kernels to improve horizontal information flow and utilizes an extended RoPE with $\theta = 5 0 0 , 0 0 0$ to capture long-range dependencies. Its chunking strategy is guided by a regex rate-distortion optimization following a word1: 1@1 pattern. The BFlowNet model refines this hierarchical concept by focusing on optimized information flow. It employs specialized attention patterns for hierarchical propagation and an enhanced regex rate-distortion chunking method with adaptive boundaries. Designed for scalability, BFlowNet features optimized layer distributions for different model sizes and seamlessly integrates the Canon layer for local context enhancement. 

## C.4 TRAINING CONFIGURATION FRAMEWORK

## C.4.1 UNIFIED OPTIMIZATION PROTOCOL

To ensure a fair comparison, all models were trained under a standardized optimization protocol. We employed a learning rate of $4 \times 1 0 ^ { - 4 }$ with a cosine annealing schedule. Weight decay was set to either 0.033 or 0.1 depending on the model’s scale. Similarly, gradient clipping was configured to either 0.2 or 1.0 based on architectural requirements, and the number of warmup steps was set to 5000 or 10000 as appropriate for the model. 

## C.4.2 DATASET DISTRIBUTION STRATEGY

Our training data was carefully curated and distributed to align with the strengths of each architecture. Models specialized for programming languages were trained on the 10BT FineWeb Code dataset. For broad knowledge coverage, general-purpose models were trained on the FineWeb Education dataset, scaled from 10BT to 100BT tokens. To leverage their unique design, byte-level models were trained directly on raw byte sequences, thereby avoiding artifacts from sub-word tokenization. Finally, to properly evaluate their chunking capabilities, hierarchical models were trained on extended sequence lengths of 3200 tokens. 

## C.4.3 INFRASTRUCTURE AND IMPLEMENTATION

The entire training framework was built on a modern infrastructure stack. We utilized BF16 mixed precision across all architectures and employed Fully Sharded Data Parallel (FSDP) with modelspecific optimizations for efficient parallelization. Models were compiled with PyTorch 2.0, and selective activation checkpointing was used to manage memory consumption in larger models. For rigorous experimental control, all runs were comprehensively tracked and logged via WandB integration. 

## C.5 OTHER TRAINING DETAILS

Training Configuration. We train all models for up to 1.95M optimizer steps (ByteFlow Net) or 950K steps (baseline) using AdamW with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , weight decay 0.1, and cosine LR decay. The peak learning rate is $4 \times 1 0 ^ { - 4 }$ , with 10K warmup steps for ByteFlow Net and 5K for the baseline. Gradient clipping is set to 0.2 and 1.0, respectively. We use bf16 precision throughout, disable TF32 matmuls for reproducibility, and enable torch.compile to fuse kernels. 

Distributed Training. All models are trained on 8 NVIDIA A100 80GB GPUs, using PyTorch Fully Sharded Data Parallel (FSDP) in full shard mode. We keep activation checkpointing disabled unless otherwise stated and set tp size=1 (pure data parallelism). We cache compiled graphs to reduce startup overhead and cap compilation cache size to 16 GB. 

Regularization and Stability. All transformer feed-forward blocks use a multiple $\_ { \mathrm { O f } } = 2 5 6$ dimension rounding; rotary position embeddings (RoPE) are applied with $\theta = 5 \times 1 0 ^ { 5 }$ for ByteFlow Net and $\theta = 1 0 ^ { 4 }$ for the baseline. We schedule λ in the rate–distortion objective to target a desired compression ratio. Both models apply dropout implicitly via residual scaling and optimizer noise. 

## D ABLATION STUDIES

Understanding the individual contributions of ByteFlow Net’s architectural components is crucial for validating our design choices and identifying the sources of performance gains. We conduct compre-


Table 6: Zero-shot performance comparison with ablation studies. Evaluation results on six downstream tasks at both 0.6B (50B tokens) and 1.3B (500B tokens) scales, including ablation studies for Canon layer and compression ratios. We report average scores over three separate runs to ensure fair comparison.


<table><tr><td rowspan="2">Model</td><td rowspan="2">Tokenizer</td><td colspan="7">Accuracy (↑)</td></tr><tr><td>HellaSwag</td><td>WinoGrande</td><td>BoolQ</td><td>PIQA</td><td>ARC-e</td><td>ARC-c</td><td>Average</td></tr><tr><td colspan="9">600M Models Trained on 50B Tokens (1x Chincilla Ratio (Hoffmann et al., 2022))</td></tr><tr><td>LLaMA (Dubey et al., 2024)</td><td>BPE</td><td><eq>43.12_{±0.87}</eq></td><td><eq>42.74_{±1.92}</eq></td><td><eq>62.26_{±0.64}</eq></td><td><eq>59.43_{±1.25}</eq></td><td><eq>61.38_{±0.98}</eq></td><td><eq>25.95_{±1.76}</eq></td><td><eq>49.15_{±0.73}</eq></td></tr><tr><td>LlamaByte (Dubey et al., 2024)</td><td rowspan="5">Byte</td><td><eq>37.93_{±1.83}</eq></td><td><eq>41.84_{±0.59}</eq></td><td><eq>61.15_{±1.47}</eq></td><td><eq>58.31_{±0.91}</eq></td><td><eq>60.24_{±1.68}</eq></td><td><eq>25.18_{±0.52}</eq></td><td><eq>47.44_{±1.29}</eq></td></tr><tr><td>MambaByte (Wang et al., 2024)</td><td><eq>38.21_{±0.76}</eq></td><td><eq>41.97_{±1.95}</eq></td><td><eq>61.48_{±1.14}</eq></td><td><eq>58.67_{±0.68}</eq></td><td><eq>60.53_{±1.87}</eq></td><td><eq>25.42_{±1.03}</eq></td><td><eq>47.71_{±0.85}</eq></td></tr><tr><td>SpaceByte (Slagle, 2024b)</td><td><eq>37.76_{±1.56}</eq></td><td><eq>42.15_{±0.82}</eq></td><td><eq>61.04_{±1.39}</eq></td><td><eq>58.18_{±1.71}</eq></td><td><eq>60.12_{±0.55}</eq></td><td><eq>25.05_{±1.98}</eq></td><td><eq>47.38_{±1.22}</eq></td></tr><tr><td>AU-Net (Videau et al., 2025)</td><td><eq>40.34_{±0.93}</eq></td><td><eq>44.12_{±1.44}</eq></td><td><eq>63.85_{±0.71}</eq></td><td><eq>64.87_{±1.16}</eq></td><td><eq>62.91_{±1.89}</eq></td><td><eq>27.43_{±0.65}</eq></td><td><eq>49.38_{±1.22}</eq></td></tr><tr><td>ByteFlow Net (Ours)</td><td><eq>41.42_{±1.35}</eq></td><td><eq>44.93_{±0.78}</eq></td><td><eq>64.48_{±1.62}</eq></td><td><eq>62.25_{±0.94}</eq></td><td><eq>63.87_{±1.17}</eq></td><td><eq>28.36_{±1.81}</eq></td><td><eq>50.89_{±0.89}</eq></td></tr><tr><td colspan="9">Ablation Studies - Canon Layer (600M, 50B tokens)</td></tr><tr><td>ByteFlow Net w/o Canon</td><td>Byte</td><td><eq>39.78_{±1.52}</eq></td><td><eq>43.21_{±1.15}</eq></td><td><eq>62.15_{±1.84}</eq></td><td><eq>60.43_{±1.23}</eq></td><td><eq>61.92_{±1.41}</eq></td><td><eq>26.73_{±1.95}</eq></td><td><eq>49.04_{±1.35}</eq></td></tr><tr><td colspan="9">Ablation Studies - Compression Ratio (600M, 50B tokens)</td></tr><tr><td>ByteFlow Net (Seq=4096)</td><td>Byte</td><td><eq>42.15_{±1.28}</eq></td><td><eq>45.67_{±0.92}</eq></td><td><eq>65.32_{±1.45}</eq></td><td><eq>63.18_{±1.06}</eq></td><td><eq>64.73_{±1.23}</eq></td><td><eq>29.42_{±1.67}</eq></td><td><eq>51.74_{±1.02}</eq></td></tr><tr><td>ByteFlow Net (Seq=2400)</td><td>Byte</td><td><eq>40.87_{±1.61}</eq></td><td><eq>44.12_{±1.34}</eq></td><td><eq>63.75_{±1.79}</eq></td><td><eq>61.53_{±1.27}</eq></td><td><eq>62.94_{±1.52}</eq></td><td><eq>27.58_{±2.04}</eq></td><td><eq>50.13_{±1.26}</eq></td></tr><tr><td>ByteFlow Net (Seq=1600)</td><td>Byte</td><td><eq>39.23_{±1.84}</eq></td><td><eq>42.78_{±1.56}</eq></td><td><eq>61.92_{±2.03}</eq></td><td><eq>59.87_{±1.65}</eq></td><td><eq>61.15_{±1.89}</eq></td><td><eq>25.94_{±2.25}</eq></td><td><eq>48.48_{±1.67}</eq></td></tr><tr><td colspan="9">1.3B Models Trained on 500B Tokens (4x Chincilla Ratio (Hoffmann et al., 2022))</td></tr><tr><td>LLaMA (Dubey et al., 2024)</td><td>BPE</td><td><eq>54.12_{±1.58}</eq></td><td><eq>53.74_{±1.36}</eq></td><td><eq>73.26_{±1.62}</eq></td><td><eq>70.43_{±1.47}</eq></td><td><eq>72.38_{±1.54}</eq></td><td><eq>36.95_{±1.81}</eq></td><td><eq>60.15_{±1.59}</eq></td></tr><tr><td>LlamaByte (Dubey et al., 2024)</td><td rowspan="5">Byte</td><td><eq>48.93_{±1.46}</eq></td><td><eq>52.84_{±1.68}</eq></td><td><eq>72.15_{±1.39}</eq></td><td><eq>69.31_{±1.52}</eq></td><td><eq>71.24_{±1.43}</eq></td><td><eq>36.18_{±1.67}</eq></td><td><eq>58.44_{±1.55}</eq></td></tr><tr><td>MambaByte (Wang et al., 2024)</td><td><eq>49.21_{±1.35}</eq></td><td><eq>52.97_{±1.57}</eq></td><td><eq>72.48_{±1.48}</eq></td><td><eq>69.67_{±1.71}</eq></td><td><eq>71.53_{±1.76}</eq></td><td><eq>36.42_{±1.34}</eq></td><td><eq>58.71_{±1.53}</eq></td></tr><tr><td>SpaceByte (Slagle, 2024b)</td><td><eq>48.76_{±1.64}</eq></td><td><eq>53.15_{±1.42}</eq></td><td><eq>72.04_{±1.56}</eq></td><td><eq>69.18_{±1.38}</eq></td><td><eq>71.12_{±1.69}</eq></td><td><eq>36.05_{±1.41}</eq></td><td><eq>58.38_{±1.54}</eq></td></tr><tr><td>AU-Net (Videau et al., 2025)</td><td><eq>50.34_{±1.51}</eq></td><td><eq>54.12_{±1.45}</eq></td><td><eq>73.85_{±1.63}</eq></td><td><eq>74.87_{±1.37}</eq></td><td><eq>72.91_{±1.59}</eq></td><td><eq>37.43_{±1.82}</eq></td><td><eq>60.59_{±1.56}</eq></td></tr><tr><td>ByteFlow Net (Ours)</td><td><eq>55.42_{±1.44}</eq></td><td><eq>56.93_{±1.69}</eq></td><td><eq>76.48_{±1.38}</eq></td><td><eq>74.25_{±1.61}</eq></td><td><eq>75.87_{±1.46}</eq></td><td><eq>40.36_{±1.74}</eq></td><td><eq>63.19_{±1.57}</eq></td></tr><tr><td colspan="9">Ablation Studies - Canon Layer (1.3B, 500B tokens)</td></tr><tr><td>ByteFlow Net w/o Canon</td><td>Byte</td><td><eq>53.18_{±1.67}</eq></td><td><eq>54.85_{±1.82}</eq></td><td><eq>74.23_{±1.55}</eq></td><td><eq>72.41_{±1.84}</eq></td><td><eq>73.52_{±1.73}</eq></td><td><eq>38.19_{±2.03}</eq></td><td><eq>61.06_{±1.78}</eq></td></tr><tr><td colspan="9">Ablation Studies - Compression Ratio (1.3B, 500B tokens)</td></tr><tr><td>ByteFlow Net (Seq=4096)</td><td>Byte</td><td><eq>56.27_{±1.32}</eq></td><td><eq>58.14_{±1.48}</eq></td><td><eq>77.89_{±1.25}</eq></td><td><eq>75.68_{±1.43}</eq></td><td><eq>76.94_{±1.35}</eq></td><td><eq>41.73_{±1.61}</eq></td><td><eq>64.44_{±1.41}</eq></td></tr><tr><td>ByteFlow Net (Seq=2400)</td><td>Byte</td><td><eq>54.76_{±1.58}</eq></td><td><eq>56.42_{±1.73}</eq></td><td><eq>75.83_{±1.49}</eq></td><td><eq>73.91_{±1.67}</eq></td><td><eq>75.12_{±1.52}</eq></td><td><eq>39.68_{±1.89}</eq></td><td><eq>62.62_{±1.64}</eq></td></tr><tr><td>ByteFlow Net (Seq=1600)</td><td>Byte</td><td><eq>52.89_{±1.85}</eq></td><td><eq>54.67_{±1.96}</eq></td><td><eq>74.15_{±1.71}</eq></td><td><eq>72.34_{±1.89}</eq></td><td><eq>73.48_{±1.78}</eq></td><td><eq>37.92_{±2.14}</eq></td><td><eq>60.91_{±1.89}</eq></td></tr></table>

## E ABLATION STUDIES


hensive ablation studies to isolate the impact of key design decisions: the Canon layer integration for efficient token mixing, and the compression ratio controlled by global sequence length. These studies provide insights into the trade-offs between computational efficiency and model performance, while demonstrating the robustness of our approach across different architectural configurations. 


As shown in Table 6, Understanding the individual contributions of ByteFlow Net’s architectural components is crucial for validating our design choices and identifying the sources of performance gains. We conduct comprehensive ablation studies to isolate the impact of key design decisions: the Canon layer integration for efficient token mixing, and the compression ratio controlled by global sequence length. These studies provide insights into the trade-offs between computational efficiency and model performance, while demonstrating the robustness of our approach across different architectural configurations. 

The Canon layer represents a critical innovation in ByteFlow Net’s local processing pipeline, enabling efficient token mixing through causal convolution operations with minimal computational overhead. Unlike traditional attention mechanisms that scale quadratically, Canon layers provide linear-time token mixing by leveraging optimized CUDA kernels for causal convolution with a 4- token kernel size. 

The ablation results demonstrate the significant impact of Canon layers across both model scales. At the 600M parameter scale, removing Canon layers results in a 1.85-point drop in average accuracy (50.89% → 49.04%), with particularly notable degradation in reasoning-intensive tasks like ARCc (28.36% → 26.73%). The performance gap becomes even more pronounced at the 1.3B scale, where the absence of Canon layers leads to a 2.13-point decrease in average accuracy (63.19% → 61.06%). 

## E.1 CANON LAYER INTEGRATION ANALYSIS

This scaling-dependent performance degradation reveals an important architectural insight: as models grow larger and process longer sequences, the Canon layer’s role in facilitating information flow becomes increasingly critical. The layer’s ability to efficiently propagate information across positions through its causal convolution mechanism appears to be particularly valuable for maintaining coherent representations in the hierarchical architecture. 

## E.2 COMPRESSION RATIO ANALYSIS

The compression ratio in ByteFlow Net’s hierarchical architecture directly determines the trade-off between computational efficiency and information preservation. We systematically evaluate different compression settings by varying the global sequence length from 4096 (2.0× compression) to 1600 (5.12× compression), while maintaining the local sequence length at 8192 bytes. 

The results reveal an interesting trade-off between computational efficiency and model performance. The lowest compression setting (global seq len = 4096) achieves the best performance with 51.74% average accuracy, representing a 0.85-point improvement over the default setting (3200). However, this comes at the cost of increased computational overhead due to the larger global transformer operations. The highest compression setting (1600) shows graceful degradation with 48.48% average accuracy, only a 2.41-point drop from the default. 

The relatively modest performance degradation even at high compression ratios (5.12x) demonstrates the effectiveness of the information-theoretic chunking strategy in preserving the most critical semantic boundaries. Moving from 4096 to 1600 global sequence length reduces the quadratic attention operations in the global transformer by a factor of $( 4 0 9 6 / 1 6 0 \bar { 0 } ) ^ { 2 } \ : = \ : 6 . 6 $ , representing substantial computational savings with manageable performance trade-offs. 

## E.3 DESIGN IMPLICATIONS

The ablation studies collectively validate ByteFlow Net’s core design philosophy. The Canon layer analysis demonstrates that efficient local token mixing is crucial for maintaining information flow in compressed representations, while the compression ratio analysis reveals that information-theoretic chunking criteria can maintain model performance across a wide range of compression settings. The consistent improvements from Canon layers and the robust performance across compression ratios demonstrate that principled architectural design can effectively navigate the fundamental trade-offs in tokenizer-free modeling. 

## F LLM USAGE DISCLOSURE.

In preparing this manuscript, we used large language models solely for polishing the writing (e.g., grammar, readability, and style improvements). No ideas, experiments, analyses, or research contributions were generated by LLMs; all conceptual and technical content originated entirely from the authors. 