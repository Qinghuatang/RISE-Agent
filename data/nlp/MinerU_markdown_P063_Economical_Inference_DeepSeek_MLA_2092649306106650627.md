# Towards Economical Inference: Enabling DeepSeek’s Multi-Head Latent Attention in Any Transformer-based LLMs

Tao Ji<sup>1,6</sup>, Bin Guo<sup>2</sup>, Yuanbin Wu<sup>2</sup>, 

Qipeng Guo<sup>8</sup>, Lixing Shen<sup>7</sup>, Zhan Chen<sup>7</sup>, Xipeng Qiu<sup>1</sup>, Qi Zhang<sup>1,4,5</sup>, Tao Gui<sup>3,4,5</sup> <sup>1</sup>School of Computer Science, Fudan University 

<sup>2</sup>School of Computer Science and Technology, East China Normal University 

<sup>3</sup>Institute of Modern Languages and Linguistics, Fudan University 

<sup>4</sup>Institute of Trustworthy Embodied Artificial Intelligence, Fudan University 

<sup>5</sup>Shanghai Collaborative Innovation Center of Intelligent Visual Computing 

<sup>6</sup>Pengcheng Laboratory <sup>7</sup>Hikvision Inc <sup>8</sup>Shanghai Al Lab 

{taoji, tgui}@fudan.edu.cn 

{binguo@stu, ybwu@cs}.ecnu.edu.cn 

## Abstract

Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for lowrank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.6% to 1%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 1% drop in LongBench performance.<sup>1</sup> 

## 1 Introduction

The rapid advancement of large language models (LLMs) has significantly accelerated progress toward artificial general intelligence (AGI), with model capabilities scaling predictably with parameter counts (Kaplan et al., 2020). However, these gains come at a steep cost: escalating computational demands for training and degraded inference throughput, resulting in substantial energy consumption and carbon emissions (Strubell et al., 2019). 

As downstream tasks grow increasingly complex, long-context processing and computationally intensive inference have become central to LLM applications (An et al., 2024). A key bottleneck lies in the memory footprint of the Key-Value (KV) cache inherent to the Multi-Head Attention (MHA, 2017) mechanism, which scales linearly with sequence length and model size. To mitigate this, variants like Grouped-Query Attention (GQA, 2023) and Multi-Query Attention (MQA, 2019) have been explored. However, these methods reduce not only the KV cache size but also the number of parameters in the attention, leading to performance degradation. The DeepSeek introduces Multi-Head Latent Attention (MLA, 2024), an attention mechanism equipped with low-rank key-value joint compression. Empirically, MLA achieves superior performance compared with MHA, and meanwhile significantly reduces the KV cache during inference, thus boosting the inference efficiency. 

A critical yet unexplored question arises: Can LLMs originally well-trained for MHA be adapted to enabling MLA for inference? The inherent architectural disparities between MHA and MLA render zero-shot transfer impractical, while the prohibitive cost of pretraining from scratch makes this transition both technically challenging and underexplored in existing research. To address this gap, we propose the first carefully designed MHA2MLA framework that maximizes parameter reuse from pre-trained MHA networks while aligning the KV cache storage and inference process with MLA’s paradigm (Figure 1). Our framework features two pivotal technical innovations: partial rotary position embedding (partial RoPE) and low-rank approximation. The primary objective of MHA2MLA is to achieve data-efficient performance recovery - restoring architecture-induced capability degradation using minimal fine-tuning data. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/e45af8cffbbdd1a93e9a1e0eb35e48bab19b746baba5f70801e3396e844d2707.jpg)



Figure 1: The diagram illustrates the MHA, MLA, and our MHA2MLA. It can be seen that the “cached” part is fully aligned with MLA after MHA2MLA. The input to the attention module is also completely aligned with MLA (the aligned region below ). Meanwhile, the parameters in MHA2MLA maximize the use of pre-trained parameters from MHA (the aligned region above ).


The inherent incompatibility between MLA’s inference acceleration mechanism and RoPE necessitates architectural compromises. DeepSeek’s solution preserves PEs in limited dimensions while compressing others, requiring strategic removal of RoPE dimensions (converting them to NoPE) in MHA to achieve MLA alignment. While higher removal ratios enhance compression efficiency, they exacerbate performance degradation, creating an efficiency-capability trade-off. Through systematically exploring RoPE removal strategies, we identify that contribution-aware dimension selection (retaining top-k dimensions ranked by attention score impact) optimally balances these competing objectives. Although previous studies have investigated training partial-RoPE LLMs from scratch (Black et al., 2021; Barbero et al., 2024), our work pioneers data-efficient fine-tuning for fullto-partial RoPE conversion in LLMs. 

MLA reduces memory footprint by projecting keys and values into a low-rank latent representation space (stored in the KV cache). MHA2MLA can also apply low-rank approximation to the values and keys stripped of RoPE (NoPE dimensions). By performing Singular Value Decomposition (SVD) on the pre-trained parameter matrices $W _ { v }$ and $W _ { k }$ corresponding to the NoPE subspaces, we compress these components into a latent space while maximizing the retention of knowledge learned by the original model. 

Our main contributions are: 

• we introduce MHA2MLA, the first parameterefficient fine-tuning framework that adapts pretrained MHA-based LLMs to the MLA architecture using only 0.6% to 1% of training data without training from scratch. 

• we demonstrate that the MHA2MLA architecture can be integrated with KV-cache quantization to achieve more economical inference (up to a 96.87% reduction). 

• we conduct experiments across five model sizes (from 135M to 13B, covering both MHA and GQA), and detailed ablation studies to provide guidance and insights for MHA2MLA. 

## 2 Preliminary

## 2.1 Multi-Head Attention (MHA)

Given an input sequence $\{ \pmb { x } _ { 1 } , \dots , \pmb { x } _ { l } \} \ \in \ \mathbb { R } ^ { l \times d } ;$ standard MHA (Vaswani et al., 2017) projects each token $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { i } }$ into queries $\pmb q _ { i } ^ { ( h ) } = \pmb { x } _ { i } \pmb { W } _ { q } ^ { ( \bar { h } ) }$ , keys $\mathbf { \Delta } \mathbf { k } _ { i } ^ { ( h ) } \ = \ \mathbf { \Delta } \mathbf { x } _ { i } \mathbf { W } _ { k } ^ { ( h ) }$ , and values $\pmb { v } _ { i } ^ { ( h ) } \ = \ \pmb { x } _ { i } \pmb { W } _ { v } ^ { ( h ) }$ where ${ \pmb W } _ { q } ^ { ( h ) } , \tilde { { \pmb W } } _ { k } ^ { ( h ) } , { \pmb W } _ { v } ^ { ( h ) } \in \mathbb { R } ^ { d \times } \tilde { { \pmb d } } _ { h }$ for each head $h \in \{ 1 , \ldots , n _ { h } \}$ . The Rotary positional encoding (RoPE, 2024) is applied to queries and keys $( \mathrm { e . g . } , k _ { i , \mathrm { r o p e } } ^ { ( h ) } = \mathrm { R o P E } ( \bar { \pmb { k } _ { i } ^ { ( h ) } } ) )$ , followed by scaled 

dot-product attention<sup>2</sup>: 

$$
\begin{array}{l} \boldsymbol {o} _ {i} ^ {(h)} = \text { Softmax } \left(\boldsymbol {q} _ {i, \text { rope }} ^ {(h)} \boldsymbol {k} _ {\leq i, \text { rope }} ^ {(h) \top}\right) \boldsymbol {v} _ {\leq i} ^ {(h)}, \\ \text { MHA } (\boldsymbol {x} _ {i}) = \left[ \boldsymbol {o} _ {i} ^ {(1)}, \ldots , \boldsymbol {o} _ {i} ^ {(n _ {h})} \right] \boldsymbol {W} _ {o}, \end{array}\tag{1}
$$

where $W _ { o } ~ \in ~ \mathbb { R } ^ { ( n _ { h } d _ { h } ) \times d }$ and $[ \cdot , \cdot ]$ means vector concatenate. During autoregressive inference, MHA stores the KV cache $\{ k _ { \mathrm { r o p e } } ^ { ( h ) } , v ^ { ( h ) } \} _ { h = 1 } ^ { n _ { h } }$ of size $O ( 2 l n _ { h } d _ { h } )$ , growing linearly with sequence length l, posing memory bottlenecks. 

Variants: Grouped-Query Attention (GQA, 2023) shares keys/values across $n _ { g }$ groups $( n _ { g } \ll n _ { h } )$ to reduce the KV cache. For each head h, it maps to group $\begin{array} { r } { g = \lfloor \frac { h \times n _ { g } } { n _ { h } } \rfloor } \end{array}$ 

$$
\begin{array}{l} \boldsymbol {o} _ {i} ^ {(h)} = \text { Softmax } \left(\boldsymbol {q} _ {i, \text { rope }} ^ {(h)} \boldsymbol {k} _ {\leq i, \text { rope }} ^ {(g) \top}\right) \boldsymbol {v} _ {\leq i} ^ {(g)}, \\ \text { GQA } (\boldsymbol {x} _ {i}) = \left[ \boldsymbol {o} _ {i} ^ {(1)}, \ldots , \boldsymbol {o} _ {i} ^ {(n _ {h})} \right] \boldsymbol {W} _ {o}. \end{array}\tag{2}
$$

Multi-Query Attention (MQA, 2019) is a special case of GQA with $n _ { g } = 1 , \mathrm { i . e . }$ ., all heads share a single global key/value. While reducing the KV cache to $O ( 2 l n _ { g } d _ { h } )$ , these methods degrade performance due to parameter pruning. 

## 2.2 Multi-Head Latent Attention (MLA)

MLA (DeepSeek-AI et al., 2024) introduces a hybrid architecture that decouples PE from latent KV compression. For each head h, the input $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ is projected into two complementary components: 

Position-Aware Component A subset of dimensions retains PE to preserve positional sensitivity: 

$$
\boldsymbol {q} _ {i, \text { rope }} ^ {(h)}, \boldsymbol {k} _ {i, \text { rope }} = \operatorname{RoPE} \left(\boldsymbol {x} _ {i} \boldsymbol {W} _ {d q} \boldsymbol {W} _ {q r} ^ {(h)}, \boldsymbol {x} _ {i} \boldsymbol {W} _ {k r}\right),
$$

where $W _ { d q } \in \mathbb { R } ^ { d \times d _ { q } } , W _ { q r } ^ { ( h ) } \in \mathbb { R } ^ { d _ { q } \times d _ { r } } , W _ { k r } \in$ $\mathbb { R } ^ { d \times d _ { r } }$ project queries/keys into a RoPE-preserved component of dimension $d _ { r }$ 

Position-Agnostic Component The remaining dimensions $d _ { c }$ are stripped of PE (i.e., NoPE), $k _ { i , \mathrm { n o p e } } ^ { ( h ) }$ and $\pmb { v } _ { i } ^ { ( h ) }$ and compressed into a shared latent vector $\pmb { c } _ { i , k v } ^ { ( h ) }$ : 

$$
\begin{array}{r} \pmb {q} _ {i, \mathrm{nope}} ^ {(h)} = \pmb {x} _ {i} \pmb {W} _ {d q} \pmb {W} _ {q c} ^ {(h)}, \\ \pmb {c} _ {i, k v} = \pmb {x} _ {i} \pmb {W} _ {d k v}, \\ \pmb {k} _ {i, \mathrm{nope}} ^ {(h)}, \pmb {v} _ {i} ^ {(h)} = \pmb {c} _ {i, k v} \pmb {W} _ {u k} ^ {(h)}, \pmb {c} _ {i, k v} \pmb {W} _ {u v} ^ {(h)}, \end{array}
$$

where $\pmb { W } _ { q c } ^ { ( h ) } \in \mathbb { R } ^ { d _ { q } \times d _ { c } } , \pmb { W } _ { d k v } \in \mathbb { R } ^ { d \times d _ { k v } } , \pmb { W } _ { u k } ^ { ( h ) } \in$ $\mathbb { R } ^ { d _ { k v } \times d _ { c } } , \pmb { W } _ { u v } ^ { ( h ) } \in \mathbb { R } ^ { d _ { k v } \times d _ { h } }$ . Note that $d _ { r } + d _ { c } =$ $d _ { h }$ . The attention output of MLA combines both components: 

$$
\begin{array}{r l} \boldsymbol {o} _ {i} ^ {(h)} & = \text { Softmax } \left(\boldsymbol {q} _ {i, \text { rope }} ^ {(h)} \boldsymbol {k} _ {\leq i, \text { rope }} ^ {(h) \top} + \boldsymbol {q} _ {i, \text { nope }} \boldsymbol {k} _ {\leq i, \text { nope }} ^ {(h) \top}\right) \\ & \cdot \boldsymbol {v} _ {\leq i} ^ {(h)} \\ \mathrm{MLA} (\boldsymbol {x} _ {i}) & = \left[ \boldsymbol {o} _ {i} ^ {(1)}, \dots , \boldsymbol {o} _ {i} ^ {(n _ {h})} \right] \cdot \boldsymbol {W} _ {o}. \end{array} \tag {3}
$$

Unlike MHA and its variants, MLA stores the latent vector $c _ { k v }$ and $\pmb { k } _ { i , \mathrm { r o p e } } ^ { ( h ) } \left( \mathcal { O } \left( l d _ { r } + l d _ { k v } \right) \right) )$ instead of full-rank $k _ { i } , v _ { i } ( \mathcal { O } ( \dot { 2 } l n _ { h } d _ { h } ) )$ , where $( d _ { r } + d _ { k v } ) \ll$ $2 n _ { h } d _ { h }$ 

Why does MLA need to separate RoPE and NoPE? MLA introduces matrix merging techniques for the NoPE portion during inference, effectively reducing memory usage. For the dot product operation $\mathbf { \widetilde { \mathbf { q } } } _ { i , \mathrm { n o p e } } ^ { ( \breve { h } ) } k _ { j , \mathrm { n o p e } } ^ { ( h ) \ \dagger }$ , the following identity transformation can be applied $^ 3$ : 

$$
\begin{array}{c} \boldsymbol {q} _ {i, \text {nope}} \boldsymbol {k} _ {j, \text {nope}} ^ {\top} = (\boldsymbol {x} _ {i} \boldsymbol {W} _ {d q} \boldsymbol {W} _ {q c}) (\boldsymbol {c} _ {j, k v} \boldsymbol {W} _ {u k}) ^ {\top} \\ = \boldsymbol {x} _ {i} (\boldsymbol {W} _ {d q} \boldsymbol {W} _ {q c} \boldsymbol {W} _ {u k} ^ {\top}) \boldsymbol {c} _ {j, k v} ^ {\top} \end{array}
$$

where $\left( W _ { d q } W _ { q c } W _ { u k } ^ { \top } \right)$ can be pre-merged into a single matrix, and $\mathbf { } c _ { j , k v }$ is already stored in the KV cache. As for the RoPE portion, the RoPE( ) function multiplies the input vector by the rotation matrix $( { \mathrm { e . g . , R o P E } } ( q _ { i } ) = q _ { i } R _ { i } , R _ { i } { \mathrm { ' s } }$ specific form will be introduced in Section 3.1). Therefore, the identity transformation becomes as follows: 

$$
\begin{array}{c} \boldsymbol {q} _ {i, \text {rope}} \boldsymbol {k} _ {j, \text {rope}} ^ {\top} = (\boldsymbol {x} _ {i} \boldsymbol {W} _ {d q} \boldsymbol {W} _ {q r} \boldsymbol {R} _ {i}) (\boldsymbol {x} _ {j} \boldsymbol {W} _ {k r} \boldsymbol {R} _ {j}) ^ {\top} \\ = \boldsymbol {x} _ {i} (\boldsymbol {W} _ {d q} \boldsymbol {W} _ {q c} \boldsymbol {R} _ {j - i} \boldsymbol {W} _ {k r} ^ {\top}) \boldsymbol {x} _ {j} ^ {\top} \end{array}
$$

Since $\left( W _ { d q } W _ { q c } { \pmb R } _ { j - i } W _ { k r } ^ { \top } \right)$ is related to the relative position $j - i ,$ , it cannot be merged into a fixed matrix. Considering that the relative distances in LLMs can be very long, such as 128K, the RoPE portion is better suited to be computed using the original form. 

## 3 MHA2MLA

## 3.1 Partial-RoPE

To enable migration from standard MHA to MLA, we propose partial-RoPE finetuning, a strategy that removes RoPE from a targeted proportion of dimensions and converts them into NoPE. Critically, while prior work has explored training LLMs with partial-RoPE from scratch (achieving marginally better perplexity than full-RoPE (Black et al., 2021; Barbero et al., 2024)), no existing method addresses how to efficiently adapt pre-trained full-RoPE models (e.g., Llama) to partial-RoPE without costly retraining. Our work bridges this gap by systematically evaluating partial-RoPE variants to identify the most data-efficient fine-tuning protocol for recovering model performance post-adaptation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/43bd496e3ac704922e115c58de1aadd1ae0aff5781363d6af5d439c44ec8c72c.jpg)



Figure 2: Illustration of $S _ { \mathrm { h i g h } } .$ , $\boldsymbol { S _ { \mathrm { l o w } } }$ , <sub>uniform</sub>, $S _ { 2 }$ <sub>-norm</sub>. Where $d _ { h } = 8$ and $r = 2$


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/140e4022b2f0eca559d3ddd43eb0e617cebc7b0711bb3feb70b917ca2a48b277.jpg)



Figure 3: Visualization of Head-wise 2-norm Contribution for Llama2-7B. We randomly selected 4 heads, and the red dashed box highlights the top-4 frequency subspaces chosen when $r = 4$ . It can be seen that differ ent heads tend to focus on different frequency subspaces, which validates the rationality of our $S _ { 2 \mathrm { - n o r m } }$ method.


MHA’s Full-RoPE encodes positional information into queries and keys through frequencyspecific rotations. Formally, given a query vector $q _ { i } \in \mathbb { R } ^ { d _ { h } }$ and key vector $\pmb { k } _ { i } \in \mathbb { R } ^ { d _ { h } }$ , we partition them into 2D chunks: 

$$
\boldsymbol {q} _ {i}, \boldsymbol {k} _ {i} = \left[ \boldsymbol {q} _ {i} ^ {[ 2 k, 2 k + 1 ]} \right] _ {0 \leq k <   \frac {d _ {h}}{2}}, \left[ \boldsymbol {k} _ {i} ^ {[ 2 k, 2 k + 1 ]} \right] _ {0 \leq k <   \frac {d _ {h}}{2}},
$$

where $\pmb q _ { i } ^ { [ 2 k , 2 k + 1 ] } \in \mathbb { R } ^ { 2 }$ denotes the k-th 2D subspace. Each chunk undergoes a rotation by positiondependent angles $\theta _ { k } = \beta ^ { - 2 k / d _ { h } }$ , forming a spectrum of wavelengths. High-frequency components, e.g., $k = 0$ , rotate rapidly at 1 radian per token. Low-frequency components, e.g., $\begin{array} { r } { k = \frac { d _ { h } } { 2 } - 1 } \end{array}$ , rotate slowly at $\sim \beta ^ { 1 / d _ { h } }$ radians per token. The base wavelength $\beta ,$ , typically set to $1 0 ^ { 4 }$ (Su et al., 2024) or $\mathrm { 5 \times 1 0 ^ { 5 } }$ 

Formally, for each 2D chunk $\mathbf { \Delta } q _ { i } ^ { [ 2 k , 2 k + 1 ] }$ and $k _ { i } ^ { [ 2 k , 2 k + 1 ] }$ , the rotation matrix at position i is defined as: 

$$
\boldsymbol {R} _ {i} ^ {[ 2 k, 2 k + 1 ]} (\theta_ {k}) = \left[ \begin{array}{c c} \cos (i \theta_ {k}) & - \sin (i \theta_ {k}) \\ \sin (i \theta_ {k}) & \cos (i \theta_ {k}) \end{array} \right].
$$

Thus, applying RoPE to queries and keys becomes: 

$$
\begin{array}{l} \boldsymbol {q} _ {i, r o p e} = \left[ \boldsymbol {R} _ {i} ^ {[ 2 k, 2 k + 1 ]} (\theta_ {k}) \boldsymbol {q} _ {i} ^ {[ 2 k, 2 k + 1 ]} \right] _ {0 \leq k <   \frac {d _ {h}}{2}}, \\ \boldsymbol {k} _ {i, r o p e} = \left[ \boldsymbol {R} _ {i} ^ {[ 2 k, 2 k + 1 ]} (\theta_ {k}) \boldsymbol {k} _ {i} ^ {[ 2 k, 2 k + 1 ]} \right] _ {0 \leq k <   \frac {d _ {h}}{2}}. \end{array}
$$

Full-RoPE to Partial-RoPE Strategies Given r retained rotational subspaces $\begin{array} { r } { ( r = \frac { d _ { r } } { 2 } \ll } \end{array}$ total subspaces $\frac { d _ { h } } { 2 }$ , we propose four strategies (illustrated in Figure 2) to select which r subspaces preserve RoPE encoding: 

High-Frequency Preservation retain the r fastest-rotating (high-frequency) subspaces: 

$$
\mathcal {S} _ {\text { high }} = \left\{k \mid 0 \leq k <   r \right\}.
$$

It is consistent with the p-RoPE method proposed in Barbero et al. (2024), where they explored settings in which r constituted 25%, 50%, and 75% of the total subspaces, and observed a slight advantage over full-RoPE in LLMs trained from scratch. 

Low-Frequency Preservation retain the r slowest-rotating (low-frequency) subspaces: 

$$
\mathcal {S} _ {\text { low }} = \left\{k \mid \frac {d _ {h}}{2} - r \leq k <   \frac {d _ {h}}{2} \right\}.
$$

It was chosen as a controlled experiment for the high-frequency strategy. 

Uniform Sampling select r subspaces with equidistant intervals: 

$$
\mathcal {S} _ {\text { uniform }} = \left\{\left\lfloor k \frac {d _ {h}}{2 r} \right\rfloor \mid 0 \leq k <   r \right\}
$$

This balances high- and low-frequency components through geometric spacing. In practice, 2r typically divides $d _ { h }$ . It is similar to the partial RoPE used in GPT-Neo (Black et al., 2021). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/0a12a76f4ae80c44bdc77377c62b2103fe72ffae36d6f99b218c7693a3de6893.jpg)



Figure 4: Illustration of $\mathbf { S V D _ { s p l i t } }$ and $\mathbf { S V D _ { j o i n t } }$ . In the multi-head setting, we adhere to the standard MLA approach, performing SVD on the merged multi-heads rather than on each head individually (e.g., $U _ { k v } \in$ $\mathbb { R } ^ { n _ { h } d _ { h } \times n _ { h } d _ { k v } }$


Head-wise 2-norm Contribution Barbero et al. (2024) were the first to propose the 2-norm contribution to investigate whether these frequencies are utilized and how they are helpful. This approach is based on the observation that, according to the Cauchy-Schwarz inequality, the influence of the k-th frequency subspace on the attention logits is upper-bounded by the 2-norm of the corresponding query and key components, i.e., $\left| { \left. { \bf q } _ { i } ^ { \left[ 2 k , 2 \bar { k } + 1 \right] } , { \bf k } _ { j } ^ { \left[ 2 k , \bar { 2 } k + 1 \right] } \right. } \right| \leqslant$ $\left\| \mathbf { q } _ { i } ^ { \left[ 2 k , 2 k + 1 \right] } \right\| \left\| \mathbf { k } _ { j } ^ { \left[ 2 k , 2 k + 1 \right] } \right\|$ . For each head h, we compute the mean 2-norm score for each subspace in an LLM over long sequences <sup>4</sup>. Then, we propose to rank all subspaces by their 2-norm score and select the top-r: 

$$
\mathcal {S} _ {2 \text {-norm}} = \underset {0 \leq k <   \frac {d _ {h}}{2}} {\text {top - r}} \left(\left\| \mathbf {q} _ {*} ^ {[ 2 k, 2 k + 1 ]} \right\| \left\| \mathbf {k} _ {*} ^ {[ 2 k, 2 k + 1 ]} \right\|\right).
$$

This head-specific selection adaptively preserves rotation-critical subspaces. Figure 3 visualizes the 2-norm of Llama2-7B’s four heads. 

We will analyze the effectiveness of the four strategies in Section 4.3 and conduct an ablation study on the essential hyperparameter r in $\mathsf { A p - }$ pendix D. For all strategies, non-selected subspaces $( k \notin S )$ become NoPE dimensions, enabling seamless integration with MLA’s latent compression. 

## 3.2 Low-rank Approximation

After transitioning from full RoPE to partial RoPE, we obtain the first component of the 

KV cache in MLA, represented as: $k _ { i , r o p e } ~ =$ $\left[ R _ { i } ^ { [ 2 k , 2 k + 1 ] } ( \theta _ { k } ) { k } _ { i } ^ { [ 2 k , 2 k + \bar { 1 } ] } \right] _ { k \in \cal S } .$ . Our next goal is to derive the second component, $\boldsymbol { c } _ { i , k v } \in \mathbb { R } ^ { d _ { k v } }$ , which serves as a low-rank representation of $k _ { i , \mathrm { n o p e } }$ and ${ \mathbf { } } v _ { i }$ 

Given the keys $k _ { i } = \pmb { x } _ { i } \pmb { W } _ { k }$ and values ${ v } _ { i } =$ ${ \bf { { x } } } _ { i } { \bf { { W } } } _ { v }$ in MHA, we first extract the subspace of $W _ { k }$ corresponding to $k _ { i , \mathrm { n o p e } } , \mathrm { i . e . }$ ., the dimensions not included in $s ,$ yielding: $k _ { i , \mathrm { n o p e } } = \pmb { x } _ { i } \pmb { W } _ { k , \mathrm { n o p e } } .$ We propose two Singular Value Decomposition (SVD)-based strategies (Illustrated in Figure 4) to preserve pre-trained knowledge while achieving rank reduction: 

Decoupled SVD $\bf ( S V D _ { s p l i t } )$ Separately decompose $W _ { k , \mathrm { n o p e } }$ and $W _ { v }$ into truncated SVDs, allocating $d _ { k v } / 2$ dimensions to each: 

$$
\boldsymbol {W} _ {k, \text { nope }} = \boldsymbol {U} _ {k} \boldsymbol {\Sigma} _ {k} \boldsymbol {V} _ {k} ^ {\top}, \quad \boldsymbol {W} _ {v} = \boldsymbol {U} _ {v} \boldsymbol {\Sigma} _ {v} \boldsymbol {V} _ {v} ^ {\top},
$$

where $U _ { k } , U _ { v } , V _ { k } , V _ { v } \ \in \ \mathbb { R } ^ { d _ { h } \times \frac { d _ { k v } } { 2 } } , \ \Sigma _ { k } , \Sigma _ { v } \ \in$ $\mathbb { R } ^ { \frac { d _ { k v } } { 2 } \times \frac { d _ { k v } } { 2 } }$ . The down-projection matrices $W _ { d * }$ 米and up-projection matrices $W _ { u * }$ become:

$$
\begin{array}{r} \pmb {W} _ {d k} = \pmb {U} _ {k} \pmb {\Sigma} _ {k} ^ {1 / 2}, \quad \pmb {W} _ {u k} = \pmb {\Sigma} _ {k} ^ {1 / 2} \pmb {V} _ {k} ^ {\top}, \\ \pmb {W} _ {d v} = \pmb {U} _ {v} \pmb {\Sigma} _ {v} ^ {1 / 2}, \quad \pmb {W} _ {u v} = \pmb {\Sigma} _ {v} ^ {1 / 2} \pmb {V} _ {v} ^ {\top}. \end{array}
$$

The low-rank representation $_ { c _ { i , k v } }$ can be constructed using $\pmb { c } _ { i , k v } = [ \pmb { x } _ { i } \pmb { W } _ { d k } , \pmb { x } _ { i } \pmb { W } _ { d v } ]$ 

Joint SVD $\bf ( S V D _ { j o i n t } )$ To preserve interactions between $K _ { \mathrm { n o p e } }$ and $V$ , we jointly factorize the concatenated matrix: 

$$
[ \pmb {W} _ {k, \mathrm{nope}}, \pmb {W} _ {v} ] = \pmb {U} _ {k v} \pmb {\Sigma} _ {k v} \pmb {V} _ {k v} ^ {\top},
$$

where $U _ { k v } , V _ { k v } \in \mathbb R ^ { d _ { h } \times d _ { k v } } , \Sigma _ { k v } \in \mathbb R ^ { d _ { k v } \times d _ { k v } }$ . The latent projection is then: 

$$
\begin{array}{c} \boldsymbol {W} _ {d k v} = \boldsymbol {U} _ {k v} \boldsymbol {\Sigma} _ {k v} ^ {1 / 2}, \\ \boldsymbol {W} _ {u k} = \boldsymbol {\Sigma} _ {k v} ^ {1 / 2} \boldsymbol {V} _ {k v} [:,: - d _ {v} ], \boldsymbol {W} _ {u v} = \boldsymbol {\Sigma} _ {k v} ^ {1 / 2} \boldsymbol {V} _ {k v} [:, d _ {v}: ]. \end{array}
$$

This jointly optimizes the latent space for both keys and values, i.e., $\pmb { c } _ { i , k v } = \pmb { x } _ { i } \pmb { W } _ { d k v }$ , retaining crossparameter dependencies critical for autoregressive generation <sup>5</sup>. Section 4.3 shows $\mathbf { S V D _ { j o i n t } }$ outperforming $\mathbf { S V D _ { s p l i t } }$ , validating that joint factorization better preserves pre-trained knowledge. 

## 4 Experiment

We evaluate our method on LLMs of varying scales (SmolLM-135M/360M/1B7, Llama2-7B/13B) pretrained with MHA or GQA. We chose the SmolLMseries<sup>6</sup> because its pretraining data and framework are both open-source, which can minimize the gap in fine-tuning data and processes. We chose Llama2- $\mathbf { \cdot } 7 \mathbf { B } ^ { 7 }$ because it is one of the widely used open-source LLMs (but its pretraining data is not open-source, there is a potential gap in fine-tuning data). 

<table><tr><td colspan="2">Model</td><td>Tokens</td><td>KV Mem.</td><td>Avg.</td><td>MMLU</td><td>ARC</td><td>PIQA</td><td>HS</td><td>OBQA</td><td>WG</td></tr><tr><td><eq>135\text{M}_{\text{SmolLM}}</eq></td><td></td><td>600B</td><td></td><td>44.50</td><td>29.80</td><td>42.43</td><td>68.06</td><td>41.09</td><td>33.60</td><td>52.01</td></tr><tr><td rowspan="2">-GQA</td><td><eq>d_{kv}=128</eq></td><td></td><td></td><td>44.42</td><td>29.91</td><td>41.71</td><td>68.28</td><td>41.33</td><td>33.80</td><td>51.46</td></tr><tr><td><eq>d_{kv}=32</eq></td><td>6B</td><td>-68.75%</td><td>43.21 -1.21</td><td>29.50</td><td>40.84</td><td>67.08</td><td>38.34</td><td>33.20</td><td>50.28</td></tr><tr><td rowspan="2">-GQA2MLA</td><td><eq>d_{kv}=16</eq></td><td>(1%)</td><td>-81.25%</td><td>42.18 -2.24</td><td>28.79</td><td>40.11</td><td>65.94</td><td>36.68</td><td>31.20</td><td>50.36</td></tr><tr><td><eq>d_{kv}=8</eq></td><td></td><td>-87.50%</td><td>41.04 -3.38</td><td>28.49</td><td>38.17</td><td>64.36</td><td>33.93</td><td>31.60</td><td>49.72</td></tr><tr><td><eq>360\text{M}_{\text{SmolLM}}</eq></td><td></td><td>600B</td><td></td><td>49.60</td><td>33.70</td><td>49.82</td><td>71.87</td><td>51.65</td><td>37.60</td><td>52.96</td></tr><tr><td rowspan="2">-GQA</td><td><eq>d_{kv}=128</eq></td><td></td><td></td><td>49.51</td><td>34.08</td><td>49.89</td><td>71.60</td><td>51.67</td><td>37.20</td><td>52.64</td></tr><tr><td><eq>d_{kv}=32</eq></td><td>6B</td><td>-68.75%</td><td>48.14 -1.37</td><td>32.91</td><td>48.34</td><td>70.51</td><td>48.56</td><td>36.80</td><td>51.70</td></tr><tr><td rowspan="2">-GQA2MLA</td><td><eq>d_{kv}=16</eq></td><td>(1%)</td><td>-81.25%</td><td>46.88 -2.63</td><td>31.85</td><td>46.07</td><td>70.62</td><td>46.48</td><td>35.80</td><td>50.43</td></tr><tr><td><eq>d_{kv}=8</eq></td><td></td><td>-87.50%</td><td>45.84 -3.67</td><td>30.93</td><td>44.41</td><td>69.48</td><td>43.49</td><td>36.00</td><td>50.75</td></tr><tr><td><eq>1\text{B7}_{\text{SmolLM}}</eq></td><td></td><td>1T</td><td></td><td>55.90</td><td>39.27</td><td>59.87</td><td>75.73</td><td>62.93</td><td>42.80</td><td>54.85</td></tr><tr><td rowspan="2">-MHA</td><td><eq>d_{kv}=128</eq></td><td></td><td></td><td>55.71</td><td>38.66</td><td>59.02</td><td>75.79</td><td>62.60</td><td>43.20</td><td>55.01</td></tr><tr><td><eq>d_{kv}=32</eq></td><td>6B</td><td>-68.75%</td><td>54.66 -1.05</td><td>37.79</td><td>56.54</td><td>75.19</td><td>61.30</td><td>41.40</td><td>55.72</td></tr><tr><td rowspan="2">-MHA2MLA</td><td><eq>d_{kv}=16</eq></td><td>(0.6%)</td><td>-81.25%</td><td>54.28 -1.43</td><td>37.79</td><td>56.33</td><td>75.68</td><td>60.59</td><td>41.00</td><td>54.30</td></tr><tr><td><eq>d_{kv}=8</eq></td><td></td><td>-87.50%</td><td>52.79 -2.92</td><td>36.69</td><td>54.29</td><td>74.54</td><td>58.49</td><td>39.20</td><td>53.51</td></tr><tr><td><eq>7\text{B}_{\text{Llama2}}</eq></td><td></td><td>2T</td><td></td><td>59.85</td><td>41.43</td><td>59.24</td><td>78.40</td><td>73.29</td><td>41.80</td><td>64.96</td></tr><tr><td rowspan="2">-MHA</td><td><eq>d_{kv}=256</eq></td><td></td><td></td><td>59.50</td><td>40.30</td><td>60.05</td><td>77.91</td><td>70.20</td><td>45.00</td><td>63.54</td></tr><tr><td><eq>d_{kv}=64</eq></td><td>12B</td><td>-68.75%</td><td>59.21 -0.29</td><td>40.90</td><td>59.53</td><td>77.26</td><td>70.05</td><td>45.20</td><td>62.51</td></tr><tr><td rowspan="2">-MHA2MLA</td><td><eq>d_{kv}=32</eq></td><td>(0.6%)</td><td>-81.25%</td><td>59.20 -0.30</td><td>40.96</td><td>59.74</td><td>77.26</td><td>69.81</td><td>44.00</td><td>63.46</td></tr><tr><td><eq>d_{kv}=16</eq></td><td></td><td>-87.50%</td><td>58.48 -1.02</td><td>40.15</td><td>58.53</td><td>77.20</td><td>68.88</td><td>45.00</td><td>61.09</td></tr><tr><td><eq>13\text{B}_{\text{Llama2}}</eq></td><td></td><td>2T</td><td></td><td>62.65</td><td>43.56</td><td>62.68</td><td>80.41</td><td>76.99</td><td>43.80</td><td>68.43</td></tr><tr><td rowspan="2">-MHA</td><td><eq>d_{kv}=256</eq></td><td></td><td></td><td>60.72</td><td>42.89</td><td>61.75</td><td>77.69</td><td>72.22</td><td>44.00</td><td>65.75</td></tr><tr><td><eq>d_{kv}=64</eq></td><td>12B</td><td>-68.75%</td><td>60.45 -0.27</td><td>42.49</td><td>62.31</td><td>78.78</td><td>71.59</td><td>43.20</td><td>64.33</td></tr><tr><td rowspan="2">-MHA2MLA</td><td><eq>d_{kv}=32</eq></td><td>(0.6%)</td><td>-81.25%</td><td>60.49 -0.23</td><td>42.00</td><td>61.64</td><td>78.73</td><td>72.71</td><td>42.80</td><td>65.04</td></tr><tr><td><eq>d_{kv}=16</eq></td><td></td><td>-87.50%</td><td>59.68 -1.04</td><td>41.31</td><td>60.39</td><td>77.69</td><td>71.54</td><td>43.60</td><td>63.54</td></tr></table>


Table 1: Commonsense reasoning ability of four LLMs with MHA2MLA or GQA2MLA. The six benchmarks include MMLU (2021), ARC easy and challenge (ARC, 2018), PIQA (2020), HellaSwag (HS, 2019), OpenBookQA (OBQA, 2018), Winogrande (WG, 2021).


We denote the architectural migration using MHA2MLA and GQA2MLA, respectively.<sup>8</sup> Both adopt data-efficient full-parameter fine-tuning, with the head-wise 2-norm selection $( S _ { 2 - \mathrm { n o r m } } , r =$ $\frac { d _ { h } } { 1 6 } )$ for Partial-RoPE and joint SVD factorization $\mathbf { ( S V D _ { j o i n t } ) }$ for low-rank approximation as default configurations. Our experiments address three critical questions: 

1. How does MHA2MLA minimize accuracy degradation induced by architectural shifts? 

2. What does MHA2MLA achieve in the KV cache reduction ratio? 

3. Can MHA2MLA integrate with KV cache quan-

tization for compound gains? 

## 4.1 Commonsense Reasoning Tasks

Main Results As shown in Table 1, our method achieves efficient architectural migration across five model scales (135M to 13B) under varying KV cache compression ratios (via latent dimension $d _ { k v } )$ . First, when comparing the performance of our fine-tuning approach with the original LLM, we observe only minor changes in performance across the five base models: a -0.08% decrease on the 135M, -0.09% on the 360M, -0.19% on the 1B7, -0.35% on the 7B and -1.93% on the 13B. This suggests that the fine-tuning data does not significantly degrade or improve the performance of the original model, providing an appropriate experimental setting for the MHA2MLA framework. 

Next, as $d _ { k v }$ decreases (e.g., from 32 to 16 to 8), the KV cache reduction increases (i.e., from -68.75% to -81.25% to -87.5%), but the performance loss becomes more challenging to recover through fine-tuning. Figure 5 shows the fine-tuning loss curves of 135M (representing GQA) and 7B (representing MHA) under different compression ratios. As the compression ratio increases, the loss difference from the baseline becomes larger. Additionally, we observe that the fluctuation trends of the loss curves are almost identical, indicating that our architecture migration does not significantly harm the model’s internal knowledge. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/df99bc578e11bcc59f70e99f4031dd9241cea9d8da9ef42e62c9e40a0f9c9829.jpg)



Figure 5: The fine-tuning loss curves under different KV cache storage ratios (with colors ranging from light to dark representing 12.5%, 18.75%, 31.25%, and 100%).


We also find that larger models experience less performance degradation when transitioning to the MLA architecture. For example, with compression down to 18.75%, the performance drops by 2.24% for 135M, 2.63% for 360M, 1.43% for 1B7, 0.30% for 7B and 0.23% for 13B, revealing the potential scaling law of MHA2MLA. Finally, from the 135M model to the 13B model, the number of tokens required for fine-tuning is only about 0.6% to 1% of the pretraining tokens, demonstrating the data efficiency of our method. 

Overall, whether using GQA2MLA or MHA2MLA, the architecture transition is achieved with minimal cost, resulting in efficient and economical inference. 

## 4.2 Long Context Tasks

Settings To evaluate the generative capabilities of the model, we adopt LongBench (Bai et al., 2024) as the benchmark for generation performance. All models are tested using a greedy decoding strategy. The context window size is determined based on the sequence length used during model fine-tuning. We use HQQ (Badri and Shaji, 2023) and Quanto<sup>9</sup> to set caches with different levels of precision to evaluate the performance of the original model as the baseline. Since our method is compatible with KV cache quantization, we also conduct additional experiments to assess the combined effect of both approaches. 

<table><tr><td>Model</td><td>Precision</td><td>KV Mem.</td><td>Avg@LB</td></tr><tr><td rowspan="5"><eq>7B_{Llama2}</eq></td><td>BF16</td><td>100.0%</td><td>27.4</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-75.00%</td><td>27.5</td></tr><tr><td>Int4Quanto</td><td>27.3</td></tr><tr><td>Int2HQQ</td><td rowspan="2">-87.50%</td><td>21.2</td></tr><tr><td>Int2Quanto</td><td>18.5</td></tr><tr><td></td><td>BF16</td><td>-68.75%</td><td>26.7</td></tr><tr><td rowspan="2"><eq>d_{kv}=64</eq></td><td>Int4HQQ</td><td rowspan="2">-92.19%</td><td>26.4</td></tr><tr><td>Int4Quanto</td><td>26.3</td></tr><tr><td></td><td>BF16</td><td>-81.25%</td><td>26.0</td></tr><tr><td rowspan="2"><eq>d_{kv}=32</eq></td><td>Int4HQQ</td><td rowspan="2">-95.31%</td><td>25.8</td></tr><tr><td>Int4Quanto</td><td>25.5</td></tr><tr><td></td><td>BF16</td><td>-87.50%</td><td>25.1</td></tr><tr><td rowspan="2"><eq>d_{kv}=16</eq></td><td>Int4HQQ</td><td rowspan="2">-96.87%</td><td>25.0</td></tr><tr><td>Int4Quanto</td><td>24.6</td></tr></table>


Table 2: Evaluation results of Llama2-7B and MHA2MLA on LongBench. Bold indicates compression ratios greater than or equal to Int2 quantization while also achieving performance higher than Int2.


Main Results As evidenced in Table 2, MHA2MLA achieves competitive or superior efficiency-accuracy profiles compared to posttraining quantization methods on LongBench. While 4-bit quantization incurs modest degradation (-0.2% to -0.4%) at comparable compression ratios, aggressive 2-bit quantization suffers severe performance collapse (-6.2% to -9%) despite 87.5% KV cache reduction. In contrast, MHA2MLA alone attains 87.5% compression (at $d _ { k v } = 1 6 )$ with only 2.3% accuracy loss, and further synergizes with 4-bit quantization to reach 92.19%/96.87% compression $( d _ { k v } = 6 4 / 1 6 \small { + } \mathrm { I n t } 4 _ { \mathrm { H Q Q } } )$ while limiting degradation to -1.0%/-2.4%, outperforming all 2-bit baselines. This highlights that MHA2MLA’s latent space design remains orthogonal to numerical precision reduction, enabling compound efficiency gains without destructive interference. 

## 4.3 Ablation Study

Four Partial-RoPE strategies: $S _ { \mathbf { h i g h } } ,$ $\scriptstyle { S _ { \mathrm { l o w } } } ,$ <sub>uniform</sub>, <sub>2-norm</sub> Table 3 presents the results of four strategies for converting full-RoPE to partial-RoPE. First, when comparing the four strategies with full-RoPE, we observed that the low-frequency retention strategy, $\boldsymbol { S _ { \mathrm { l o w } } }$ , incurred the greatest performance loss (a reduction of - 5.25%@135M and -1.87%@1B7), whereas the high-frequency retention strategy, $S _ { \mathrm { h i g h } }$ , experienced significantly less degradation (a reduction of -0.82%@135M and -0.91%@1B7), underscoring the importance of high-frequency subspaces. Both $ { S _ { \mathrm { u n i f o r m } } }$ and $S _ { \mathrm { 2 - n o r m } }$ yielded better performance, the $ { S _ { \mathrm { u n i f o r m } } }$ preserves subspaces across the frequency spectrum, while the $S _ { \mathrm { 2 - n o r m } }$ retains subspaces based on their contribution to the attention scores. We choose $S _ { \mathrm { 2 - n o r m } }$ as the default configuration because the removed subspaces (i.e., NoPE) are more suitable for the (SVD-based) low-rank approximation. 

<table><tr><td>Model</td><td>Tokens</td><td>Avg@CS</td></tr><tr><td><eq>135M_{SmolLM}</eq></td><td>600B</td><td>44.50</td></tr><tr><td>- full-rope</td><td></td><td>44.42</td></tr><tr><td>- <eq>S_{high}</eq></td><td></td><td>43.60 -0.82</td></tr><tr><td>- <eq>S_{low}</eq></td><td>6B</td><td>39.17 -5.25</td></tr><tr><td>- <eq>S_{uniform}</eq></td><td></td><td>44.01 -0.41</td></tr><tr><td>- <eq>S_{2-norm}</eq></td><td></td><td>43.73 -0.69</td></tr><tr><td>- <eq>S_{high} + SVD_{joint}</eq></td><td></td><td>40.85 -3.57</td></tr><tr><td>- <eq>S_{uniform} + SVD_{joint}</eq></td><td>6B</td><td>41.79 -2.63</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{joint}</eq></td><td></td><td>42.18 -2.24</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{split}</eq></td><td></td><td>41.27 -3.15</td></tr><tr><td><eq>1B7_{SmolLM}</eq></td><td>1T</td><td>55.90</td></tr><tr><td>- full-rope</td><td></td><td>55.71</td></tr><tr><td>- <eq>S_{high}</eq></td><td></td><td>54.80 -0.91</td></tr><tr><td>- <eq>S_{low}</eq></td><td>6B</td><td>53.84 -1.87</td></tr><tr><td>- <eq>S_{uniform}</eq></td><td></td><td>55.30 -0.41</td></tr><tr><td>- <eq>S_{2-norm}</eq></td><td></td><td>54.98 -0.73</td></tr><tr><td>- <eq>S_{high} + SVD_{joint}</eq></td><td></td><td>54.17 -1.54</td></tr><tr><td>- <eq>S_{uniform} + SVD_{joint}</eq></td><td>6B</td><td>54.27 -1.44</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{joint}</eq></td><td></td><td>54.28 -1.43</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{split}</eq></td><td></td><td>52.90 -2.81</td></tr></table>


Table 3: Reasoning ability of ablation studies. The results of other models are provided in Appendix E.


Two SVD-based low-rank approximations: $\mathbf { S V D _ { s p l i t } } , \mathbf { S V D _ { j o i n t } }$ The last two rows of each group in Table 3 compare the effects of the two SVD methods. We observe that, on both LLMs, the $\mathrm { S V D _ { j o i n t } }$ method consistently outperforms $\mathrm { S V D _ { s p l i t } }$ yielding an average performance improvement of 0.91% on the 135M model and 1.38% on the 1B7 model. It indicates that $\mathrm { S V D _ { j o i n t } }$ emerges as the clear default choice. 

## 5 Related Work

Efficient Attention Architectures The standard Multi-Head Attention (MHA, 2017) mechanism’s quadratic complexity in context length has spurred numerous efficiency innovations. While MHA remains foundational, variants like Multi-Query Attention (MQA) and Grouped-Query Attention (GQA, 2023) reduce memory overhead by sharing keys/values across heads—albeit at the cost of parameter pruning and performance degradation. 

Parallel efforts, such as Linear Transformers (Guo et al., 2019; Katharopoulos et al., 2020; Choromanski et al., 2021), RWKV (Peng et al., 2023), and Mamba (Gu and Dao, 2023), replace softmax attention with linear recurrences or state-space models, but struggle to match the expressiveness of standard attention in autoregressive generation. 

Multi-Head Latent Attention (MLA, 2024) distinguishes itself by compressing KV caches into low-rank latent vectors without pruning attention parameters. Our work bridges MLA with mainstream architectures (MHA/GQA), enabling seamless migration via data-efficient fine-tuning. Notably, while many linear attention variants abandon softmax query-key interactions (e.g., through kernel approximations), architectures preserving a query-key dot product structure—even in factorized forms—remain compatible with our MHA2MLA framework. 

Economical Key-Value Cache The memory footprint of KV caches has become a critical bottleneck for long-context inference. Recent advances fall into three categories: 

Innovative Architecture methods like MLA (DeepSeek-AI et al., 2024), MiniCache (Liu et al., 2024a), and MLKV (Zuhri et al., 2024) share or compress KV representations across layers or heads. While effective, cross-layer sharing risks conflating distinct attention patterns, potentially harming task-specific performance. Only MLA has been successfully validated in Deepseek’s LLMs. 

Quantization techniques such as GPTQ (Frantar et al., 2022), FlexGen (Sheng et al., 2023), and KIVI (Liu et al., 2024b) store KV caches in lowbit formats (e.g., 2-bit), achieving memory savings with precision loss. 

Dynamic Pruning approaches like A2SF (Jo and Shin, 2024) and SnapKV (Li et al., 2024) prune “less important” tokens from the KV cache. However, token pruning risks discarding critical long-range dependencies, while head pruning (e.g., SliceGPT (Ashkboos et al., 2024), Sheared (Xia et al., 2024), and Simple Pruning (Sun et al., 2024)) irreversibly reduces model capacity. 

Our MHA2MLA method achieves the migration of standard Transformer-based LLMs to the more economical MLA architecture and has demonstrated its ability to integrate with KV quantization techniques to realize a ~97% cache saving. It is also theoretically compatible with other methods like pruning. 

## 6 Conclusion

This work addresses the critical challenge of adapting pre-trained MHA-based LLMs (or variants) to the KV-cache-efficient MLA architecture. By introducing MHA2MLA with contribution-aware partial-RoPE removal and SVD-driven low-rank projection, we achieve near-lossless compression of KV cache (up to 96.87% size reduction for Llama2-7B) while requiring only 0.6% to 1% of training data. The framework demonstrates strong compatibility with existing compression techniques and maintains commonsense reasoning and longcontext processing capabilities, offering a practical pathway for deploying resource-efficient LLMs without sacrificing performance. Our results underscore the feasibility of architectural migration for LLMs through targeted parameter reuse and data-efficient fine-tuning. 

## Limitations

Verification on More LLMs Considering that MHA2MLA can significantly reduce inference costs, it is worthwhile to validate it on larger and more diverse open-source LLMs. However, constrained by our computation resources, models like Llama3 require fine-tuning on a 128K context length to mitigate performance degradation from continued training, so we did not perform such experiments. Furthermore, since Deepseek has not yet open-sourced the tensor-parallel inference framework for MLA, it is currently challenging to explore models larger than 7B. This will be addressed in our future work. 

Parameter-Efficient MHA2MLA Fine-tuning This paper primarily focuses on the data efficiency of MHA2MLA. Since the architectural transformation does not involve the Feed-Forward (FFN) module, future work could explore parameter-efficient MHA2MLA fine-tuning, for example by freezing the FFN module and/or freezing the parameters in the queries and keys that correspond to the retained RoPE. This could further reduce the cost of the MHA2MLA transition. 

## Acknowledgments

The authors wish to thank all reviewers for their helpful comments and suggestions. The corresponding authors are Yuanbin Wu, Xipeng Qiu, Qi Zhang, Tao Gui. This work was partially funded by Guangdong S&T Program 2024B0101050003, 

National Natural Science Foundation of China (No.62076069,62206057,61976056), Shanghai Rising-Star Program (23QA1400200), and Natural Science Foundation of Shanghai (23ZR1403500), Program of Shanghai Academic Research Leader under grant 22XD1401100. The computations in this research were performed using the CFFF platform of Fudan University. 

## References



Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. GQA: training generalized multi-query transformer models from multi-head checkpoints. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 4895–4901. Association for Computational Linguistics. 





Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024. L-eval: Instituting standardized evaluation for long context language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11- 16, 2024, pages 14388–14411. Association for Computational Linguistics. 





Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari Do Nascimento, Torsten Hoefler, and James Hensman. 2024. Slicegpt: Compress large language models by deleting rows and columns. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. 





Hicham Badri and Appu Shaji. 2023. Half-quadratic quantization of large machine learning models. 





Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2024. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, Au gust 11-16, 2024, pages 3119–3137. Association for Computational Linguistics. 





Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Velickovic. 2024. Round and round we go! what makes rotary positional encodings useful? CoRR, abs/2410.06205. 





Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: reasoning about physical commonsense in natural language. In The 





Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications ofArtificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pages 7432– 7439. AAAI Press. 





Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021. GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow. If you use this software, please cite it using these metadata. 





Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. 2021. Rethinking attention with performers. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the AI2 reasoning challenge. CoRR, abs/1803.05457. 





DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, Hao Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuip ing Yu, Shunfeng Zhou, Size Zheng, Tao Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, and Xiaowen Sun. 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. CoRR, abs/2405.04434. 





Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2022. GPTQ: accurate post-training quantization for generative pre-trained transformers. CoRR, abs/2210.17323. 





Albert Gu and Tri Dao. 2023. Mamba: Linear-time sequence modeling with selective state spaces. CoRR, abs/2312.00752. 





Qipeng Guo, Xipeng Qiu, Pengfei Liu, Yunfan Shao, Xiangyang Xue, and Zheng Zhang. 2019. Startransformer. In Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 1315–1325. Association for Computational Linguistics. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. 





Hyun-rae Jo and Dongkun Shin. 2024. A2SF: accumulative attention scoring with forgetting factor for token pruning in transformer decoder. CoRR, abs/2407.20485. 





Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. CoRR, abs/2001.08361. 





Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR. 





Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv: LLM knows what you are looking for before generation. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024. 





Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Reza Haffari, and Bohan Zhuang. 2024a. Minicache: KV cache compression in depth dimension for large language models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024. 





Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024b. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. 





Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? A new dataset for open book question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 2381–2391. Association for Computational Linguistics. 





Bo Peng, Eric Alcaide, Quentin Anthony, Alon Al balak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Leon Derczynski, Xingjian Du, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Jiaju Lin, Krishna Sri Ipsit Mantri, Ferdi nand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Johan S. Wind, Stanislaw Wozniak, Zhenyuan Zhang, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. 2023. RWKV: reinventing rnns for the transformer era. In Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, pages 14048–14077. Association for Computational Linguistics. 





Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavat ula, and Yejin Choi. 2021. Winogrande: an adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106. 





Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need. CoRR, abs/1911.02150. 





Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, and Ce Zhang. 2023. Flexgen: High-throughput generative inference of large language models with a single GPU. In International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings ofMachine Learning Research, pages 31094–31116. PMLR. 





Emma Strubell, Ananya Ganesh, and Andrew McCallum. 2019. Energy and policy considerations for deep learning in NLP. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3645–3650, Florence, Italy. Association for Computational Linguistics. 





Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: En hanced transformer with rotary position embedding. Neurocomputing, 568:127063. 





Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. 2024. A simple and effective pruning approach for large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural 





Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 5998–6008. 





Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. 2024. Sheared llama: Accelerating language model pre-training via structured pruning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. 





Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings ofthe 57th Conference ofthe Associationfor Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 4791–4800. Association for Computational Linguistics. 





Zayd Muhammad Kawakibi Zuhri, Muhammad Farid Adilazuarda, Ayu Purwarianti, and Alham Fikri Aji. 2024. MLKV: multi-layer key-value heads for memory efficient transformer decoding. CoRR, abs/2406.09297. 



## A The Calculation of 2-norm Score

To compute the 2-norm scores for each attention head, we selected 1,024 samples from the training dataset. The proportions of the subsets and sequence length used during the 2-norm computation are consistent with those used during finetuning. First, we calculate the query vectors and key vectors for each head. Then, for each rotational subspace of the vectors, we compute the 2-norm scores. Finally, the 2-norm scores of the query and key vectors are aggregated within each subspace. If the model employs Grouped-Query Attention (GQA), the 2-norm scores are averaged within each GQA group, and the scores are shared between the groups. 

## B Inference Process of MHA2MLA

During inference in the MHA2MLA model, our input includes the hidden representation $x _ { i }$ of the i-th token, as well as the previously stored $\pmb { k } _ { < i , \mathrm { r o p e } } ^ { ( h ) }$ and $\mathbf { c } _ { < i , \mathrm { k v } }$ in the KV cache for the first $i - 1$ tokens. 

During the inference, our goal is to compute the h-th head’s dot product of these two parts $\mathbf { \Delta } q _ { i , \mathrm { r o p e } } ^ { ( h ) } k _ { \leq i , \mathrm { r o p e } } ^ { ( h ) \top }$ and $\mathbf { \Delta } q _ { i , \mathrm { n o p e } } ^ { ( \hat { h } ) } k _ { \leq i , \mathrm { n o p e } } ^ { ( h ) \top }$ . For the RoPE part, we can easily extract ${ \cal W } _ { q , \mathrm { r o p e } } ^ { ( h ) }$ and $W _ { k , \mathrm { r o p t } } ^ { ( h ) }$ e from the pre-trained parameter matrices ${ \cal W } _ { q } ^ { ( h ) }$ and $\boldsymbol { W } _ { k } ^ { ( h ) }$ (i.e., the rows corresponding to the subspace that retains RoPE) and then obtain the result through a linear transformation: 

$$
\begin{array}{c} \boldsymbol {q} _ {i, \text { rope}} ^ {(h)} = \boldsymbol {x} _ {i} \boldsymbol {W} _ {q, \text { rope }} ^ {(h)} \\ \boldsymbol {k} _ {i, \text { rope }} ^ {(h)} = \boldsymbol {x} _ {i} \boldsymbol {W} _ {k, \text { rope }} ^ {(h)} \\ \boldsymbol {k} _ {\leq i, \text { rope }} ^ {(h)} = [ \boldsymbol {k} _ {<   i, \text { rope }} ^ {(h)}, \boldsymbol {k} _ {i, \text { rope }} ^ {(h)} ] \\ \to \boldsymbol {q} _ {i, \text { rope }} ^ {(h)} \boldsymbol {k} _ {\leq i, \text { rope }} ^ {(h) \top}. \end{array}
$$

Note that $\boldsymbol { k } _ { < i , \mathrm { r o p e } } ^ { ( h ) }$ is already stored in the KV cache and can be directly retrieved. 

For the NoPE part, $\mathbf { \Delta } q _ { i , \mathrm { n o p e } } ^ { ( h ) }$ can still be easily obtained through a linear transformation ${ \cal W } _ { q , \mathrm { n o p e } } ^ { ( h ) }$ which extracted from the pre-trained parameter matrix ${ \cal W } _ { q } ^ { ( h ) }$ by separating the rows corresponding to the subspace with RoPE removed. However, $k _ { i , \mathrm { n o p e } } ^ { ( h ) }$ requires two linear transformations: a dimensionality reduction transformation using $W _ { d k v }$ , and a dimensionality expansion transformation using $\pmb { W } _ { u k } ^ { ( h ) }$ . Note that $W _ { d k v }$ is shared across all heads in the current layer, and both $W _ { d k v }$ and $W _ { u k } ^ { ( h ) }$ are constrained by the SVD decomposition of the pretrained parameter matrices ${ \cal W } _ { k , \mathrm { n o p e } } ^ { ( h ) }$ and $\boldsymbol { W } _ { v } ^ { ( h ) }$ , preserving most of the pre-trained knowledge: 

$$
\begin{array}{r} \pmb {q} _ {i, \mathrm{nope}} ^ {(h)} = \pmb {x} _ {i} \pmb {W} _ {q, \mathrm{nope}} ^ {(h)} \\ \pmb {c} _ {i, k v} = \pmb {x} _ {i} \pmb {W} _ {d k v}, \\ \pmb {k} _ {i, \mathrm{nope}} ^ {(h)} = \pmb {c} _ {i, k v} \pmb {W} _ {u k} ^ {(h)} \\ \pmb {k} _ {<   i, \mathrm{nope}} ^ {(h)} = \pmb {c} _ {<   i, k v} \pmb {W} _ {u k} ^ {(h)}. \end{array}
$$

During inference, the NoPE part can also leverage the standard MLA matrix merging algorithm to reduce memory consumption: 

$$
\begin{array}{c} \boldsymbol {k} _ {\leq i, \text {nope}} ^ {(h)} = [ \boldsymbol {c} _ {<   i, k v}, \boldsymbol {c} _ {i, k v} ] \boldsymbol {W} _ {u k} ^ {(h)} \\ \boldsymbol {q} _ {i, \text {nope}} ^ {(h)} \boldsymbol {k} _ {\leq i, \text {nope}} ^ {(h) \top} = (\boldsymbol {x} _ {i} \boldsymbol {W} _ {q, \text {nope}} ^ {(h)}) (\boldsymbol {c} _ {\leq i, k v} \boldsymbol {W} _ {u k} ^ {(h)}) ^ {\top} \\ = \boldsymbol {x} _ {i} (\boldsymbol {W} _ {q, \text {nope}} ^ {(h)} \boldsymbol {W} _ {u k} ^ {(h) \top}) \boldsymbol {c} _ {\leq i, k v} ^ {\top}. \end{array}
$$

We can pre-multiply the parameter matrices $( \hat { W } _ { q , \mathrm { n o p e } } ^ { ( h ) } W _ { u k } ^ { ( h ) \top } )$ , and let $\begin{array} { r l } { c _ { i , q } ^ { ( h ) } } & { { } = } \end{array}$ $\pmb { x } _ { i } ( \pmb { W } _ { q , \mathrm { n o p e } } ^ { ( h ) } \pmb { W } _ { u k } ^ { ( h ) \top } )$ In the end, the output of MHA2MLA is as follows: 

$$
\begin{array}{l} \boldsymbol {v} _ {i} ^ {(h)} = \boldsymbol {c} _ {i, k v} \boldsymbol {W} _ {u v} ^ {(h)} \\ \boldsymbol {o} _ {i} ^ {(h)} = \text {Softmax} \Big (\boldsymbol {q} _ {i, \text {rope}} ^ {(h)} \boldsymbol {k} _ {\leq i, \text {rope}} ^ {(h) \top} + \boldsymbol {c} _ {i, q} ^ {(h)} \boldsymbol {c} _ {\leq i, k v} ^ {\top} \Big)   \boldsymbol {c} _ {\leq i, k v} \\ \text {MHA2MLA} (\boldsymbol {x} _ {i}) = \Big [ \ldots , \boldsymbol {o} _ {i} ^ {(h)} \boldsymbol {W} _ {u v} ^ {(h)}, \ldots \Big ]   \boldsymbol {W} _ {o}. \end{array}
$$

Where $\boldsymbol { W } _ { u v } ^ { ( h ) }$ and $W _ { o }$ can also perform matrix merging to make inference more economical. 

Why doesn’t MHA2MLA perform low-rank representation on the query as DeepSeek does? Firstly, we found that the economical inference of MLA is not affected even if ${ \cal W } _ { q , \mathrm { n o p e } } ^ { ( h ) }$ is not decomposed into a dimension-reducing matrix $( \mathbf { e . g . } , W _ { d q } )$ and a dimension-increasing matrix $( \mathrm { e . g . , } W _ { u q } ^ { ( h ) } )$ Secondly, decomposing ${ \cal W } _ { q , \mathrm { n o p e } } ^ { ( h ) }$ introduces additional architectural migration loss (approximation loss) and further reduces the number of LLM parameters. Therefore, we believe there is no need to decompose ${ \cal W } _ { q , \mathrm { n o p e } } ^ { ( h ) }$ within the MHA2MLA framework. 

## C The Details of Fine-tuning

Data We fine-tune our model using the pretraining corpus from Smol $\mathbf { \delta M } ^ { 1 0 }$ . The dataset consists of fineweb-edu-dedup, cosmopedia-v2, python-edu, open-web-math, and StackOverflow. The first three datasets are part of the smollm-corpus<sup>11</sup> curated by HuggingFaceTB. Fineweb-edu-dedup is a high-quality dataset filtered by HuggingFaceTB from education-related webpages. Similarly, HuggingFaceTB filtered Python code snippets from The Stack to construct the python-edu dataset. Cosmopedia-v2 is a high-quality dataset generated by a model based on 34,000 topics defined by BISAC book classifications. Additionally, openweb-math<sup>12</sup> and StackOverflow<sup>13</sup> are sourced from high-quality mathematical texts available online and posts from StackOverflow, respectively. 

Hyperparameters The fine-tuning hyperparameters for models of all sizes are listed in Table 4. The training process employs a warmup phase followed by a decay strategy. A 1-sqrt decay strategy is applied to ensure a smooth and gradual reduction. 

## D Ablation Study on Partial-RoPE Dimensions

To better determine the strategy and dimensionality for partial-RoPE, we conducted an ablation study on the number of RoPE dimensions using the $1 3 5 \mathrm { M } _ { \mathrm { S m o l L M } }$ model. The experimental results are presented in Table 5. By comparing the performance of four different strategies in varying dimensionalities, we observed that the low-frequency strategy, $ { s _ { \mathrm { l o w } } }$ , suffered significant performance degradation (-11.8%) when the dimensionality was relatively low $( \leq 4 )$ . In contrast, both $ { S _ { \mathrm { u n i f o r m } } }$ and $S _ { 2 \mathrm { - n o r m } }$ consistently demonstrated superior performance regardless of dimensionality. Furthermore, increasing the dimensionality from 4 to 8 provided negligible performance gains. Based on these findings, we selected a dimensionality of 4 for partial-RoPE. 

## E Detailed Results

In this section, we present the detailed results. 

Detailed LongBench evaluation is reported in Table 6. 

Detailed ablation experiment is reported in Table 7. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/83f626fad81bbbc1a9e0d3c755aae5441bf71af94b1ea7298b1bc6568fca8b6d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/ef4c40dece588cb44f6bdc8a5899ff9aa18452cc003519e22b39110ca6859e1f.jpg)



Figure 6: The fine-tuning loss curves under different KV cache storage ratios (with colors ranging from light to dark representing 12.5%, 18.75%, 31.25%, and 100%).


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/d3d3ecfcda5ea40536c2c7c102873c0943b25a3a0af4650f07f59ddce1761b12.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/586cba84d5052159c29a13dcb2019f866046f278fcfbbaa9513e63b3adc2758b.jpg)



Figure 7: The fine-tuning loss curves under different partial-RoPE strategy.


Additional visualizations of fine-tuning loss We present the loss of the other two models finetuned, excluding the ones mentioned in the main text, in Figure 6. We observe that as fine-tuning progresses, the gap in loss between our approach and the baseline gradually decreases, and both exhibit similar fluctuations, demonstrating the effectiveness of our approach. In Figure 7, we show the loss under different partial-RoPE strategies. Except for $\boldsymbol { S _ { \mathrm { l o w } } }$ , the other three partial-RoPE schemes show little difference from the baseline. Additionally, $\boldsymbol { S _ { \mathrm { l o w } } }$ has a higher probability of convergence failure. In Figure 8, we show the loss under different SVD strategies. The loss curves on both $1 { \mathrm { B } } 7 _ { \mathrm { S m o l L M } }$ and $1 3 5 \mathrm { M } _ { \mathrm { S m o l L M } }$ reveal that $\mathrm { S V D _ { j o i n t } }$ outperforms $\mathrm { \Delta S V D _ { s p l i t } }$ 

<table><tr><td>Metrics</td><td>135MSmolLM</td><td>360MSmolLM</td><td>1B7SmolLM</td><td>7BLlama2</td><td>13BLlama2</td></tr><tr><td>n_batch × n_gpu</td><td>64×4</td><td>64×4</td><td>32×8</td><td>16×16</td><td>8×32</td></tr><tr><td>Learning Rate</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>Hardware</td><td>NVIDIA L20Y</td><td>NVIDIA L20Y</td><td>NVIDIA L20Y</td><td>NVIDIA L20Y</td><td>NVIDIA L20Y</td></tr><tr><td>Steps</td><td>12000</td><td>12000</td><td>12000</td><td>12000</td><td>12000</td></tr><tr><td>Warmup ratio</td><td>10.0%</td><td>10.0%</td><td>10.0%</td><td>10.0%</td><td>10.0%</td></tr><tr><td>Decay</td><td>16.7%</td><td>16.7%</td><td>16.7%</td><td>16.7%</td><td>16.7%</td></tr><tr><td>Time</td><td>4h</td><td>8h</td><td>16h</td><td>28h</td><td>36h</td></tr><tr><td>Seqlen</td><td>2048</td><td>2048</td><td>2048</td><td>4096</td><td>4096</td></tr><tr><td><eq>d_{kv}=128/256^{\dagger}</eq></td><td>134.52M</td><td>361.82M</td><td>1.71B</td><td>6.61B†</td><td>13.02B†</td></tr><tr><td rowspan="3">#Param.<eq>d_{kv}=32/64^{\dagger}</eq></td><td>130.99M</td><td>351.38M</td><td>1.67B</td><td>6.37B†</td><td>12.56B†</td></tr><tr><td><eq>d_{kv}=16/32^{\dagger}</eq></td><td>129.64M</td><td>1.59B</td><td>5.99B†</td><td>11.80B†</td></tr><tr><td><eq>d_{kv}=8/16^{\dagger}</eq></td><td>128.97M</td><td>1.56B</td><td>5.79B†</td><td>11.43B†</td></tr></table>


Table 4: Training detail information across different models.


<table><tr><td colspan="2">Model</td><td>Avg.</td><td>MMLU</td><td>ARC</td><td>PIQA</td><td>HS</td><td>OBQA</td><td>WG</td></tr><tr><td>135M</td><td>r=32</td><td>44.42</td><td>29.91</td><td>41.71</td><td>68.28</td><td>41.33</td><td>33.80</td><td>51.46</td></tr><tr><td rowspan="4"><eq>-\mathcal{S}_{high}</eq></td><td>r=1</td><td>42.88-1.54</td><td>29.24</td><td>40.15</td><td>66.81</td><td>37.90</td><td>33.40</td><td>49.80</td></tr><tr><td>r=2</td><td>43.07-1.35</td><td>29.73</td><td>40.60</td><td>67.25</td><td>38.82</td><td>32.40</td><td>49.64</td></tr><tr><td>r=4</td><td>43.60-0.82</td><td>29.87</td><td>41.29</td><td>67.08</td><td>39.58</td><td>32.80</td><td>50.99</td></tr><tr><td>r=8</td><td>43.90-0.52</td><td>29.79</td><td>40.89</td><td>68.01</td><td>40.71</td><td>33.40</td><td>50.59</td></tr><tr><td rowspan="4"><eq>-\mathcal{S}_{low}</eq></td><td>r=1</td><td>39.85-4.57</td><td>27.72</td><td>36.56</td><td>62.95</td><td>33.88</td><td>28.20</td><td>49.80</td></tr><tr><td>r=2</td><td>39.72-4.70</td><td>27.36</td><td>36.86</td><td>63.76</td><td>33.85</td><td>27.80</td><td>48.70</td></tr><tr><td>r=4</td><td>39.17-5.25</td><td>27.67</td><td>35.33</td><td>62.30</td><td>33.32</td><td>27.60</td><td>48.78</td></tr><tr><td>r=8</td><td>42.36-2.06</td><td>29.33</td><td>39.37</td><td>66.70</td><td>38.13</td><td>31.00</td><td>49.64</td></tr><tr><td rowspan="4"><eq>-\mathcal{S}_{uniform}</eq></td><td>r=1</td><td>42.72-1.70</td><td>29.34</td><td>40.20</td><td>66.76</td><td>37.60</td><td>32.60</td><td>49.80</td></tr><tr><td>r=2</td><td>43.50-0.92</td><td>29.41</td><td>41.30</td><td>67.63</td><td>39.31</td><td>33.40</td><td>49.96</td></tr><tr><td>r=4</td><td>44.01-0.41</td><td>29.79</td><td>41.09</td><td>67.95</td><td>40.54</td><td>34.20</td><td>50.51</td></tr><tr><td>r=8</td><td>43.79-0.66</td><td>29.85</td><td>40.72</td><td>67.57</td><td>40.84</td><td>32.80</td><td>50.99</td></tr><tr><td rowspan="4"><eq>-\mathcal{S}_{2-norm}</eq></td><td>r=1</td><td>43.27-1.15</td><td>29.58</td><td>40.83</td><td>67.25</td><td>39.14</td><td>33.00</td><td>49.80</td></tr><tr><td>r=2</td><td>43.77-0.65</td><td>29.82</td><td>40.76</td><td>68.28</td><td>39.32</td><td>34.40</td><td>50.04</td></tr><tr><td>r=4</td><td>43.73-0.69</td><td>30.00</td><td>41.29</td><td>68.17</td><td>39.83</td><td>33.20</td><td>49.88</td></tr><tr><td>r=8</td><td>44.18-0.24</td><td>30.01</td><td>41.52</td><td>68.12</td><td>40.70</td><td>34.00</td><td>50.75</td></tr></table>


Table 5: The impact of positional encoding dimensionality on model performance.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/8b53f724-f7f3-4fee-b200-ebf410b934a9/9950f65ab24b4ea7b5d7241469dcc985ea8c169fada65de73d7e30e76de46699.jpg)



Figure 8: The fine-tuning loss curves under the combi nation of $S _ { 2 \mathrm { - n o r m } }$ and different SVD strategies.


<table><tr><td rowspan="2"><eq>d_{kv}</eq></td><td rowspan="2">Precision</td><td rowspan="2">KV</td><td rowspan="2">Avg.</td><td colspan="3">S-Doc QA</td><td colspan="3">M-Doc QA</td><td colspan="3">Summ.</td><td colspan="3">Few-shot</td><td colspan="2">Synth.</td><td colspan="2">Code</td></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td><td>G</td><td>H</td><td>I</td><td>J</td><td>K</td><td>L</td><td>M</td><td>N</td><td>O</td><td>P</td></tr><tr><td colspan="20">7BLlama2 (Length=4K)</td></tr><tr><td rowspan="3"></td><td>BF16</td><td>100.0%</td><td>27.4</td><td>15.1</td><td>9.6</td><td>21.1</td><td>7.5</td><td>9.7</td><td>3.7</td><td>26.7</td><td>20.5</td><td>3.2</td><td>65.5</td><td>87.5</td><td>34.1</td><td>1.9</td><td>6.6</td><td>66.5</td><td>59.4</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-75.00%</td><td>27.5</td><td>16.1</td><td>9.1</td><td>22.0</td><td>7.3</td><td>9.9</td><td>3.6</td><td>26.5</td><td>21.1</td><td>3.4</td><td>65.5</td><td>87.2</td><td>34.3</td><td>1.5</td><td>6.7</td><td>66.0</td><td>59.9</td></tr><tr><td>Int4Quanto</td><td>27.3</td><td>14.4</td><td>9.5</td><td>20.5</td><td>7.5</td><td>9.7</td><td>3.5</td><td>25.8</td><td>20.7</td><td>3.1</td><td>65.5</td><td>87.7</td><td>34.3</td><td>1.4</td><td>7.3</td><td>66.8</td><td>59.3</td></tr><tr><td rowspan="2"></td><td>Int2HQQ</td><td rowspan="2">-87.50%</td><td>21.2</td><td>18.0</td><td>5.5</td><td>12.6</td><td>7.5</td><td>8.4</td><td>3.2</td><td>12.6</td><td>18.6</td><td>0.9</td><td>56.5</td><td>73.3</td><td>27.0</td><td>1.8</td><td>6.1</td><td>34.5</td><td>52.9</td></tr><tr><td>Int2Quanto</td><td>18.5</td><td>9.4</td><td>6.2</td><td>12.7</td><td>6.8</td><td>6.7</td><td>3.3</td><td>5.9</td><td>17.2</td><td>0.4</td><td>61.0</td><td>63.9</td><td>26.0</td><td>1.4</td><td>2.7</td><td>42.4</td><td>30.5</td></tr><tr><td rowspan="3">64</td><td>BF16</td><td>-68.75%</td><td>26.7</td><td>12.2</td><td>9.4</td><td>22.5</td><td>7.5</td><td>11.7</td><td>4.2</td><td>26.5</td><td>18.9</td><td>20.2</td><td>58.0</td><td>83.6</td><td>35.0</td><td>1.7</td><td>5.5</td><td>57.1</td><td>52.8</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-92.19%</td><td>26.4</td><td>12.9</td><td>9.3</td><td>22.7</td><td>7.8</td><td>12.1</td><td>3.3</td><td>26.5</td><td>18.7</td><td>18.6</td><td>58.0</td><td>82.4</td><td>35.5</td><td>1.4</td><td>4.9</td><td>56.2</td><td>51.8</td></tr><tr><td>Int4Quanto</td><td>26.3</td><td>9.5</td><td>8.7</td><td>22.7</td><td>7.6</td><td>11.0</td><td>4.0</td><td>26.0</td><td>18.3</td><td>19.8</td><td>58.5</td><td>84.4</td><td>35.3</td><td>1.3</td><td>5.1</td><td>56.7</td><td>51.4</td></tr><tr><td rowspan="3">32</td><td>BF16</td><td>-81.25%</td><td>26.0</td><td>13.4</td><td>8.7</td><td>21.2</td><td>5.9</td><td>9.9</td><td>2.5</td><td>25.3</td><td>19.2</td><td>17.6</td><td>65.5</td><td>85.5</td><td>25.5</td><td>3.0</td><td>7.0</td><td>54.0</td><td>51.4</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-95.31%</td><td>25.8</td><td>13.6</td><td>9.1</td><td>20.6</td><td>6.0</td><td>10.2</td><td>2.5</td><td>25.0</td><td>18.4</td><td>16.4</td><td>65.5</td><td>85.1</td><td>25.4</td><td>3.1</td><td>6.9</td><td>53.4</td><td>51.0</td></tr><tr><td>Int4Quanto</td><td>25.5</td><td>13.4</td><td>8.0</td><td>21.2</td><td>6.4</td><td>10.1</td><td>3.0</td><td>24.3</td><td>17.1</td><td>17.1</td><td>65.0</td><td>85.1</td><td>26.1</td><td>3.6</td><td>6.2</td><td>52.7</td><td>49.0</td></tr><tr><td rowspan="3">16</td><td>BF16</td><td>-87.50%</td><td>25.1</td><td>13.2</td><td>8.7</td><td>21.6</td><td>7.1</td><td>9.1</td><td>3.7</td><td>24.1</td><td>18.4</td><td>20.3</td><td>57.5</td><td>86.0</td><td>33.3</td><td>0.1</td><td>9.0</td><td>43.9</td><td>44.9</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-96.87%</td><td>25.0</td><td>13.7</td><td>8.8</td><td>23.7</td><td>7.1</td><td>9.2</td><td>4.5</td><td>22.8</td><td>18.7</td><td>18.4</td><td>57.5</td><td>86.6</td><td>32.1</td><td>0.1</td><td>8.8</td><td>43.5</td><td>44.3</td></tr><tr><td>Int4Quanto</td><td>24.6</td><td>9.9</td><td>8.4</td><td>22.3</td><td>7.2</td><td>9.0</td><td>4.2</td><td>22.6</td><td>18.4</td><td>18.6</td><td>57.0</td><td>85.4</td><td>33.6</td><td>0.4</td><td>8.8</td><td>43.5</td><td>45.0</td></tr><tr><td colspan="20">1B7SmolLM (Length=2K)</td></tr><tr><td rowspan="5"></td><td>BF16</td><td>100.0%</td><td>18.7</td><td>2.6</td><td>6.3</td><td>19.9</td><td>5.4</td><td>8.6</td><td>2.7</td><td>23.5</td><td>18.4</td><td>20.2</td><td>46.5</td><td>70.2</td><td>32.4</td><td>2.2</td><td>3.2</td><td>21.3</td><td>16.5</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-75.00%</td><td>18.6</td><td>2.5</td><td>6.2</td><td>19.1</td><td>5.5</td><td>8.2</td><td>2.7</td><td>23.4</td><td>18.3</td><td>20.0</td><td>46.5</td><td>69.4</td><td>32.1</td><td>2.7</td><td>3.2</td><td>21.5</td><td>16.0</td></tr><tr><td>Int4Quanto</td><td>18.6</td><td>2.6</td><td>6.2</td><td>17.4</td><td>5.1</td><td>8.6</td><td>2.6</td><td>23.0</td><td>18.1</td><td>20.1</td><td>46.0</td><td>70.2</td><td>31.9</td><td>2.9</td><td>3.6</td><td>21.9</td><td>16.7</td></tr><tr><td>Int2HQQ</td><td rowspan="2">-87.50%</td><td>16.3</td><td>2.5</td><td>5.6</td><td>13.0</td><td>4.8</td><td>7.5</td><td>2.7</td><td>14.8</td><td>16.3</td><td>9.3</td><td>46.0</td><td>70.4</td><td>26.9</td><td>2.6</td><td>3.4</td><td>18.3</td><td>16.8</td></tr><tr><td>Int2Quanto</td><td>13.3</td><td>1.6</td><td>3.8</td><td>10.3</td><td>3.9</td><td>7.3</td><td>1.4</td><td>5.9</td><td>13.4</td><td>6.3</td><td>40.0</td><td>64.3</td><td>14.6</td><td>3.1</td><td>3.5</td><td>15.6</td><td>17.5</td></tr><tr><td rowspan="3">32</td><td>BF16</td><td>-68.75%</td><td>16.3</td><td>2.6</td><td>6.1</td><td>17.8</td><td>5.3</td><td>9.2</td><td>2.4</td><td>21.8</td><td>14.7</td><td>18.8</td><td>52.5</td><td>55.6</td><td>28.8</td><td>1.7</td><td>4.7</td><td>12.5</td><td>5.7</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-92.19%</td><td>15.7</td><td>2.3</td><td>6.4</td><td>16.4</td><td>5.2</td><td>8.8</td><td>2.3</td><td>20.5</td><td>14.2</td><td>16.6</td><td>52.5</td><td>56.7</td><td>27.8</td><td>1.5</td><td>4.2</td><td>11.9</td><td>4.4</td></tr><tr><td>Int4Quanto</td><td>15.7</td><td>2.3</td><td>6.0</td><td>16.5</td><td>5.3</td><td>8.8</td><td>2.1</td><td>22.1</td><td>14.5</td><td>17.5</td><td>50.5</td><td>55.0</td><td>27.6</td><td>1.8</td><td>3.2</td><td>13.3</td><td>5.5</td></tr><tr><td rowspan="3">16</td><td>BF16</td><td>-81.25%</td><td>15.5</td><td>2.4</td><td>6.2</td><td>17.1</td><td>5.5</td><td>9.2</td><td>2.5</td><td>21.0</td><td>15.2</td><td>16.5</td><td>47.5</td><td>53.9</td><td>31.4</td><td>1.3</td><td>3.3</td><td>9.5</td><td>5.3</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-95.31%</td><td>15.3</td><td>2.4</td><td>5.7</td><td>17.0</td><td>4.8</td><td>9.0</td><td>2.1</td><td>20.0</td><td>15.5</td><td>16.8</td><td>47.5</td><td>53.1</td><td>30.1</td><td>2.0</td><td>3.4</td><td>10.6</td><td>5.3</td></tr><tr><td>Int4Quanto</td><td>15.1</td><td>2.3</td><td>5.9</td><td>16.0</td><td>6.0</td><td>9.4</td><td>2.5</td><td>19.1</td><td>14.4</td><td>15.5</td><td>47.5</td><td>52.5</td><td>28.4</td><td>2.0</td><td>3.2</td><td>11.2</td><td>5.3</td></tr><tr><td rowspan="3">8</td><td>BF16</td><td>-87.50%</td><td>14.0</td><td>2.6</td><td>5.6</td><td>16.5</td><td>5.1</td><td>8.9</td><td>2.1</td><td>19.8</td><td>15.7</td><td>14.2</td><td>40.0</td><td>51.0</td><td>28.0</td><td>2.1</td><td>3.3</td><td>7.1</td><td>2.6</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-96.87%</td><td>13.8</td><td>2.6</td><td>5.0</td><td>15.4</td><td>4.5</td><td>9.5</td><td>2.5</td><td>20.5</td><td>14.8</td><td>14.0</td><td>40.0</td><td>48.2</td><td>27.1</td><td>1.8</td><td>4.2</td><td>7.6</td><td>3.1</td></tr><tr><td>Int4Quanto</td><td>13.9</td><td>2.5</td><td>5.4</td><td>16.6</td><td>4.8</td><td>8.9</td><td>2.3</td><td>19.3</td><td>14.6</td><td>15.8</td><td>40.0</td><td>50.2</td><td>26.4</td><td>1.1</td><td>3.4</td><td>8.3</td><td>3.5</td></tr><tr><td colspan="20">360MSmolLM (Length=2K)</td></tr><tr><td rowspan="5"></td><td>BF16</td><td>100.0%</td><td>13.5</td><td>2.4</td><td>6.4</td><td>14.3</td><td>5.0</td><td>8.8</td><td>2.5</td><td>18.0</td><td>17.5</td><td>7.1</td><td>47.5</td><td>37.5</td><td>24.9</td><td>1.5</td><td>3.4</td><td>8.1</td><td>10.4</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-75.00%</td><td>13.4</td><td>2.7</td><td>6.1</td><td>14.1</td><td>5.5</td><td>8.4</td><td>3.0</td><td>16.2</td><td>15.4</td><td>11.2</td><td>47.5</td><td>37.5</td><td>23.4</td><td>1.3</td><td>3.7</td><td>9.0</td><td>10.1</td></tr><tr><td>Int4Quanto</td><td>13.3</td><td>2.4</td><td>6.2</td><td>13.7</td><td>5.4</td><td>8.7</td><td>2.6</td><td>15.4</td><td>17.4</td><td>7.3</td><td>47.5</td><td>37.3</td><td>24.4</td><td>1.0</td><td>3.7</td><td>8.4</td><td>11.0</td></tr><tr><td>Int2HQQ</td><td rowspan="2">-87.50%</td><td>10.8</td><td>2.7</td><td>4.7</td><td>8.3</td><td>5.4</td><td>5.9</td><td>1.9</td><td>9.9</td><td>10.0</td><td>8.4</td><td>45.2</td><td>27.5</td><td>14.2</td><td>2.1</td><td>4.2</td><td>10.0</td><td>11.9</td></tr><tr><td>Int2Quanto</td><td>8.6</td><td>2.6</td><td>2.2</td><td>4.4</td><td>3.9</td><td>4.8</td><td>1.4</td><td>5.6</td><td>8.9</td><td>2.9</td><td>44.0</td><td>26.8</td><td>9.6</td><td>1.0</td><td>1.9</td><td>7.2</td><td>9.7</td></tr><tr><td rowspan="3">32</td><td>BF16</td><td>-68.75%</td><td>13.3</td><td>2.5</td><td>6.0</td><td>13.6</td><td>5.0</td><td>8.4</td><td>2.8</td><td>19.2</td><td>15.4</td><td>10.4</td><td>43.5</td><td>35.0</td><td>29.8</td><td>1.0</td><td>3.0</td><td>10.5</td><td>6.5</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-92.19%</td><td>12.8</td><td>2.2</td><td>5.6</td><td>14.2</td><td>4.7</td><td>8.7</td><td>2.6</td><td>14.3</td><td>14.6</td><td>8.0</td><td>43.5</td><td>34.1</td><td>29.7</td><td>1.2</td><td>3.1</td><td>11.9</td><td>6.8</td></tr><tr><td>Int4Quanto</td><td>12.9</td><td>2.1</td><td>5.2</td><td>11.9</td><td>5.0</td><td>8.9</td><td>2.7</td><td>15.9</td><td>15.1</td><td>10.6</td><td>43.5</td><td>31.8</td><td>27.0</td><td>0.7</td><td>3.0</td><td>16.6</td><td>7.1</td></tr><tr><td rowspan="3">16</td><td>BF16</td><td>-81.25%</td><td>10.9</td><td>2.0</td><td>5.5</td><td>13.5</td><td>4.9</td><td>9.9</td><td>3.1</td><td>13.1</td><td>13.8</td><td>10.7</td><td>26.5</td><td>27.0</td><td>19.7</td><td>0.8</td><td>4.0</td><td>13.0</td><td>6.3</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-95.31%</td><td>10.4</td><td>2.1</td><td>4.7</td><td>13.1</td><td>4.9</td><td>9.3</td><td>2.8</td><td>11.5</td><td>12.8</td><td>8.0</td><td>26.5</td><td>25.5</td><td>20.5</td><td>0.7</td><td>4.1</td><td>14.1</td><td>6.5</td></tr><tr><td>Int4Quanto</td><td>10.2</td><td>2.0</td><td>5.0</td><td>13.2</td><td>4.4</td><td>9.0</td><td>2.5</td><td>12.0</td><td>12.5</td><td>9.7</td><td>27.5</td><td>24.0</td><td>18.9</td><td>0.6</td><td>3.2</td><td>11.3</td><td>7.6</td></tr><tr><td rowspan="3">8</td><td>BF16</td><td>-87.50%</td><td>9.4</td><td>1.9</td><td>4.5</td><td>11.7</td><td>4.3</td><td>8.5</td><td>2.9</td><td>12.5</td><td>12.5</td><td>9.5</td><td>24.0</td><td>20.3</td><td>14.4</td><td>0.9</td><td>3.7</td><td>10.9</td><td>8.3</td></tr><tr><td>Int4HQQ</td><td rowspan="2">-96.87%</td><td>9.0</td><td>1.8</td><td>4.4</td><td>11.2</td><td>4.3</td><td>8.0</td><td>2.4</td><td>10.5</td><td>11.4</td><td>7.2</td><td>23.5</td><td>20.8</td><td>12.5</td><td>0.9</td><td>4.2</td><td>11.5</td><td>8.7</td></tr><tr><td>Int4Quanto</td><td>8.8</td><td>2.2</td><td>3.8</td><td>10.7</td><td>3.8</td><td>7.3</td><td>2.8</td><td>11.2</td><td>11.1</td><td>7.5</td><td>22.5</td><td>21.0</td><td>12.0</td><td>0.7</td><td>4.6</td><td>10.5</td><td>8.7</td></tr></table>


Table 6: Evaluation results of all models on LongBench, including Task A: narrativeqa, B: qasper, C: multifieldqa_en, D: hotpotqa, E: 2wikimqa, F: musique, G: gov_report, H: qmsum, I: multi_news, J: trec, K: triviaqa, L: samsum, M: passage_count, N: passage_retrieval_en, O: lcc, P: repobench-p. Bold indicates compression ratios greater than or equal to Int2 quantization while also achieving performance higher than Int2.


<table><tr><td>Model</td><td>Tokens</td><td>Avg@CS</td><td>MMLU</td><td>ARC</td><td>PIQA</td><td>HS</td><td>OBQA</td><td>WG</td></tr><tr><td><eq>135M_{SmolLM}</eq></td><td>600B</td><td>44.50</td><td>29.80</td><td>42.43</td><td>68.06</td><td>41.09</td><td>33.60</td><td>52.01</td></tr><tr><td>- full-rope</td><td></td><td>44.42</td><td>29.91</td><td>41.71</td><td>68.28</td><td>41.33</td><td>33.80</td><td>51.46</td></tr><tr><td>- <eq>S_{high}</eq></td><td></td><td>43.60 -0.82</td><td>29.87</td><td>41.29</td><td>67.08</td><td>39.58</td><td>32.80</td><td>50.99</td></tr><tr><td>- <eq>S_{low}</eq></td><td>6B</td><td>39.17 -5.25</td><td>27.67</td><td>35.33</td><td>62.30</td><td>33.32</td><td>27.60</td><td>48.78</td></tr><tr><td>- <eq>S_{uniform}</eq></td><td></td><td>44.01 -0.41</td><td>29.79</td><td>41.09</td><td>67.95</td><td>40.54</td><td>34.20</td><td>50.51</td></tr><tr><td>- <eq>S_{2-norm}</eq></td><td></td><td>43.73 -0.69</td><td>30.00</td><td>41.29</td><td>68.17</td><td>39.83</td><td>33.20</td><td>49.88</td></tr><tr><td>- <eq>S_{high} + SVD_{joint}</eq></td><td></td><td>40.85 -3.57</td><td>28.46</td><td>37.25</td><td>64.85</td><td>35.31</td><td>30.20</td><td>49.01</td></tr><tr><td>- <eq>S_{uniform} + SVD_{joint}</eq></td><td>6B</td><td>41.79 -2.63</td><td>28.74</td><td>39.30</td><td>65.83</td><td>36.37</td><td>31.20</td><td>49.33</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{joint}</eq></td><td></td><td>42.18 -2.24</td><td>28.79</td><td>40.11</td><td>65.94</td><td>36.68</td><td>31.20</td><td>50.36</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{split}</eq></td><td></td><td>41.27 -3.15</td><td>28.05</td><td>38.65</td><td>65.51</td><td>34.04</td><td>31.20</td><td>49.17</td></tr><tr><td><eq>1B7_{SmolLM}</eq></td><td>1T</td><td>55.90</td><td>39.27</td><td>59.87</td><td>75.73</td><td>62.93</td><td>42.80</td><td>54.85</td></tr><tr><td>- full-rope</td><td></td><td>55.71</td><td>38.66</td><td>59.02</td><td>75.79</td><td>62.60</td><td>43.20</td><td>55.01</td></tr><tr><td>- <eq>S_{high}</eq></td><td></td><td>54.80 -0.91</td><td>38.18</td><td>57.57</td><td>75.08</td><td>60.66</td><td>42.40</td><td>54.93</td></tr><tr><td>- <eq>S_{low}</eq></td><td>6B</td><td>53.84 -1.87</td><td>37.49</td><td>55.24</td><td>74.16</td><td>59.22</td><td>42.60</td><td>54.30</td></tr><tr><td>- <eq>S_{uniform}</eq></td><td></td><td>55.30 -0.41</td><td>38.52</td><td>57.89</td><td>75.68</td><td>61.85</td><td>42.60</td><td>55.25</td></tr><tr><td>- <eq>S_{2-norm}</eq></td><td></td><td>54.98 -0.73</td><td>38.33</td><td>57.47</td><td>76.06</td><td>61.77</td><td>41.40</td><td>54.85</td></tr><tr><td>- <eq>S_{high} + SVD_{joint}</eq></td><td></td><td>54.17 -1.54</td><td>37.35</td><td>55.99</td><td>74.59</td><td>59.18</td><td>42.20</td><td>55.72</td></tr><tr><td>- <eq>S_{uniform} + SVD_{joint}</eq></td><td>6B</td><td>54.27 -1.44</td><td>37.95</td><td>56.78</td><td>74.86</td><td>60.23</td><td>41.00</td><td>54.78</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{joint}</eq></td><td></td><td>54.28 -1.43</td><td>37.79</td><td>56.33</td><td>75.68</td><td>60.59</td><td>41.00</td><td>54.30</td></tr><tr><td>- <eq>S_{2-norm} + SVD_{split}</eq></td><td></td><td>52.90 -2.81</td><td>36.99</td><td>53.80</td><td>73.39</td><td>58.55</td><td>41.60</td><td>53.04</td></tr></table>


Table 7: The complete results of the ablation experiment.
