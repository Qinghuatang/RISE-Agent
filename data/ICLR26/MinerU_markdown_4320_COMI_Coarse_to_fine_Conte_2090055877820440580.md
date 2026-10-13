# COMI: COARSE-TO-FINE CONTEXT COMPRESSION VIA MARGINAL INFORMATION GAIN

Jiwei Tang<sup>1,2</sup> Shilei Liu<sup>2</sup> Zhicheng Zhang<sup>1</sup> Yujin Yuan<sup>2</sup> Libin Zheng<sup>3∗</sup> Wenbo Su<sup>2</sup> Bo Zheng<sup>2∗</sup> 

<sup>1</sup>Tsinghua University <sup>2</sup>Future Living Lab of Alibaba <sup>3</sup>Sun Yat-sen University tangjw24@mails.tsinghua.edu.cn zhenglb6@mail.sysu.edu.cn bozheng@alibaba-inc.com 

## ABSTRACT

Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks. However, their deployment in long context scenarios remains hindered by computational inefficiency and information redundancy. Context compression methods address these challenges by significantly reducing input length and eliminating redundancy. We propose COMI, a coarse-to-fine adaptive context compression framework that jointly optimizes for semantic relevance and diversity under high compression rates. We introduce Marginal Information Gain (MIG), a metric defined as the relevance of a unit to the input query minus its semantic redundancy with other units, guiding the compression process to prioritize information that is both relevant and low redundant. The framework operates in two stages: (1) Coarse-Grained Group Reallocation, where the context is partitioned into groups and dynamically assigned compression rates based on inter-group MIG, ensuring compression budgets align with information value distribution; and (2) Fine-Grained Token Merging, where tokens within each group are fused via an intra-group MIG-based weighting mechanism, thereby preserving key semantics while avoiding the accumulation of redundancy. Extensive experiments across question-answering (e.g., NaturalQuestions, 2WikiMQA, HotpotQA and NarrativeQA), summarization (e.g., MultiNews) with various backbones (e.g., LLaMA-2-7B, Qwen2-7B) show that COMI outperforms existing baselines by a large margin, e.g., approximately 25-point Exact Match (EM) improvement under 32x compression constraint with Qwen2-7B on NaturalQuestions<sup>1</sup>. 

## 1 INTRODUCTION

Large Language Models (LLMs) achieve exceptional performance across a wide range of Natural Language Processing (NLP) tasks (Yang et al., 2024; DeepSeek-AI et al., 2025; Team et al., 2025; Lv et al., 2025; Zhao et al., 2025b; Liu et al., 2025a). However, recent advances in prompting techniques, such as Retrieval-Augmented Generation (RAG) (Lewis et al., 2020), inevitably increase input length, introducing two key challenges when deploying LLMs in long context scenarios: (1) computational cost, as the quadratic complexity of the attention mechanism in Transformer (Vaswani et al., 2017) leads to inefficiency with long sequences; and (2) information redundancy, where the presence of redundant content degrades model performance (Jiang et al., 2024; Ge et al., 2024; Liu et al., 2024b; Tang et al., 2025a;b). 

Context compression emerges as a promising solution in the NLP community to address these challenges by significantly reducing input length and eliminating redundancy. Existing prompt compression methods mainly fall into two categories: task-agnostic context compression methods (Xu et al., 2024; Pan et al., 2024; Ge et al., 2024; Tan et al., 2024; Cheng et al., 2024; Zhang et al., 2025; Ye et al., 2025; Li et al., 2025; Liao et al., 2025; Tang et al., 2025b) and task-aware context compression methods (Cao et al., 2024; Jiang et al., 2024; Zhao et al., 2025c; Tang et al., 2025a; Fang et al., 2025; Liskavets et al., 2025). LLMs typically allocate attention to only a small subset of query-relevant tokens (see Figure 1a). Consequently, task-agnostic methods that compress context without considering the input query inevitably lose or dilute relevant information, especially at high compression rates. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/49ac91fb-df74-46f9-b9b1-89a317c55626/1f18721cba55f9cbba832caa1ef23e0f54deb2642e578fcb0e6887989a854bc5.jpg)



(a) Attention Weight Contribution.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/49ac91fb-df74-46f9-b9b1-89a317c55626/61de5f1d8ab406c205db3bc590379de801610601447a899276e1df5c3d3002a4.jpg)



Token Index



(b) Top Query-Related Tokens Similarity.


Figure 1: Analysis of Attention Distribution and Similarity of Top Query-Related Tokens. (a) Only a small number of tokens related to the query occupy a large proportion of the attention weight allocation; for example, the 0.75% most relevant tokens occupy 99% of the attention weights. (b) These query-related tokens are highly similar to each other, with the lowest similarity exceeding 0.6. In contrast, task-aware methods generally include query in the compression process, preserving information according to relevance via query-guided strategies such as merging, deletion or summarization. However, existing task-aware methods rely solely on relevance as a criterion for compression, ignoring the inherent redundancy in natural language (Shannon, 1948). This leads to the retention of highly similar relevant content, with high redundancy (see Figure 1b). Such oversimilarity can mislead the LLM into producing erroneous outputs (relevance does not guarantee correctness) (Yang et al., 2025; Wang & Sun, 2025). This issue is especially amplified in long-context scenarios, where different segments carry varying information value and should be compressed with different compression rates (Jiang et al., 2023; 2024; Tang et al., 2025a). Existing dynamic compression rate allocation mechanisms are limited in several ways: they either follow predefined linear rules lacking adaptability (Jiang et al., 2023; 2024; Tang et al., 2025a), rely on model-internal understanding (Chen et al., 2025) (i.e., task-agnostic) that may ignore query-relevant content, or determine compression rates based solely on relevance (Cao et al., 2024) (i.e., assigning lower compression to high relevance content). None of these methods account for semantic redundancy among compression units, leading to repeated retention of similar information and compromising both compression effectiveness and information diversity. This naturally raises a research question: How can we retain query-relevant information while identifying and eliminating semantic redundancy among compressed representations, especially under high compression rates, to jointly optimize relevance and diversity? 

To this end, we propose COMI (COarse-to-fine Context Compression via Marginal Information Gain), a coarse-to-fine context compression framework that adaptively balances relevance information preservation and redundancy elimination. We introduce Marginal Information Gain (MIG), which is defined as the relevance of a unit to the query minus its semantic redundancy with other units. This metric jointly captures relevance and semantic uniqueness, guiding the compression process to prioritize information that is both relevant and low redundant. COMI employs a coarseto-fine compression strategy. In the first stage, Coarse-grained Group Reallocation, the input context is divided into segments of equal length, each treated as a compression group. MIG is computed for inter-group, and the compression rate for each group is dynamically adjusted accordingly: groups with high MIG (i.e., high relevance and low redundancy) are assigned lower compression rates. This enables the compression budget to be adaptively reallocated based on the distribution of information value in the context. In the second stage, Fine-Grained Token Merging, tokens are weighted by their intra-group MIG, and fine-grained semantic fusion is performed accordingly. Tokens with high MIG contribute more to the merged representation, ensuring that key semantic units are preserved while avoiding the accumulation of “relevant but redundant” content. Through this hierarchical compression mechanism, COMI effectively retains high-relevance, low-redundancy information even under high compression rates, ensuring that the final compressed representation is semantically complementary rather than redundancy, thereby increasing information diversity. 

Our contributions are three-fold: (1) We introduce Marginal Information Gain (MIG) as a metric for context compression, jointly modeling task relevance and semantic redundancy. This overcomes the limitations of existing relevance-only methods and provides a more discriminative framework for evaluating information value in long context compression. (2) We propose COMI, which employs a coarse-to-fine adaptive compression strategy. At the coarse level, reallocation based on inter-group MIG dynamically adjusts compression rates across different regions. At the fine level, intra-group MIG-guided weighted fusion eliminates group redundancy, preventing the accumulation of similar content. (3) We conduct comprehensive experiments on long-context tasks including Question-Answering (QA), summarization. Experimental results demonstrate that COMI outperforms existing methods by a large margin under high compression rates, e.g., with a 32x compression constraint and using Qwen2-7B as the backbone, COMI improves the Exact Match (EM) score by approximately 25-point over suboptimal baseline on NaturalQuestions. 

## 2 PRELIMINARY: GMSA

GMSA (Tang et al., 2025b) introduces an issue of cross-layer semantic misalignment in the encoderdecoder based framework and addresses it via Layer Semantic Alignment (LSA), which aligns highlevel summary vectors with low-level original input semantics, thereby bridging the semantic gap across different layers. Since COMI also relies on an encoder-decoder framework, we similarly employ LSA to achieve cross-layer semantic alignment (see Figure 2). Following Tang et al. (2025b), we set LSA to a single layer. 

## 3 RELATED WORK

Task-Agnostic Context Compression Methods. Task-agnostic context compression methods typically do not incorporate the input query and aim to preserve overall semantic information for broad downstream applicability. Main approaches include: (1) Encoder-decoder methods (Ge et al., 2024; Cheng et al., 2024; Tan et al., 2024; Liao et al., 2025; Li et al., 2025; Rau et al., 2024; Dai et al., 2025; Choi et al., 2025; Tang et al., 2025b; Zhao et al., 2025a; Liu et al., 2025b), which compress input via an encoder and decode answers from the compact representation; (2) Attention mask modification (Mu et al., 2023; Petrov et al., 2025; Ye et al., 2025), learning compact soft tokens guided by task losses; (3) Autoregressive modeling (Chevalier et al., 2023; Zhang et al., 2025), treating compression as a sequential generation process conditioned on prior compressed segments; (4) Metric-driven methods using entropy (Li et al., 2023; Jiang et al., 2023) or bidirectional semantics (Pan et al., 2024) to remove less important content; and (5) Summarization-based methods (Xu et al., 2024), training a task-agnostic summarizer for compression. Despite preserving general semantics, these methods lack awareness of query-related tokens, inevitably discarding useful information and degrading performance, especially under high compression constraints. 

Task-Aware Context Compression Methods. Task-aware context compression methods do not pursue the completeness of semantic over the compressed representation, but focus on retaining information relevant to downstream task. During the compression process, these methods simultaneously receive the original context and a query to merge relevant content (Cao et al., 2024), summarize (Yoon et al., 2024; Hwang et al., 2025) or delete irrelevant content (Jiang et al., 2024; Tang et al., 2025a; Liskavets et al., 2025; Fang et al., 2025; Zhao et al., 2025c). Although these methods can filter task-relevant content, they implicitly assume conditional independence among the retained tokens, leading to significant redundancy within the preserved tokens and making it easy to mislead LLM into generating erroneous outputs. 

KV-cache Compression. KV-cache Compression methods compresses KV-caches layer-wise, exploring strategies like inter-layer sharing (Brandon et al., 2024), reducing heads (Shazeer, 2019; Ainslie et al., 2023), or discarding less important KVs (Xiao et al., 2024; Li et al., 2024; Zhang et al., 2025)). The Key limitations on these methods include requiring identical compression and response models, and necessitating model-specific modificationsfor different KV-cache structures. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/49ac91fb-df74-46f9-b9b1-89a317c55626/44792062025f9af17acf8332f358c705755f19db765d667109812aab153a005d.jpg)



Figure 2: The Training Paradigm of COMI. COMI is based on an encoder-decoder architecture. The original context X and query Q are first encoded into hidden states, which are then compressed via a compression process (see Figure 3). The compressed representation is decoded and trained using cross-entropy loss. During training, the encoder and LSA are fully fine-tuned, while the decoder is partially fine-tuned, updating only the $W _ { Q } , W _ { K } , W _ { V }$ , and $W _ { O }$ matrices in each layer.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/49ac91fb-df74-46f9-b9b1-89a317c55626/60b7ccdcde886c35e8959a084598a3010561584836bd5ff46f00640a335cdf23.jpg)



Figure 3: The Compression Process of COMI. Specifically, it sequentially performs three steps: I. Average Pooling of Query Tokens. Obtain a single query vector via average pooling; II. Coarse-Grained Group Reallocation. Reallocate the sizes of compression groups based on inter-group Marginal Information Gain (MIG) $( i . e . ,$ , groups with higher MIG are assigned lower compression rates); III. Fine-Grained Token Merging. Compute the intra-group MIG for each token and merge all tokens within a group into a single compressed token according to weights $w _ { 1 } , . . . , w _ { L _ { i } - 1 }$


## 4 COMI

In this section, we elaborate on COMI, a coarse-to-fine context compression framework based on an encoder-decoder architecture. COMI simultaneously models the relevance of each compression unit to the input question and the redundancy among units via the Marginal Information Gain (MIG). It then performs Coarse-grained Group Reallocation followed by Fine-grained Group Token Merging to retain information that is both relevant and low redundant. 

## 4.1 MARGINAL INFORMATION GAIN

For a given token $x _ { i } ,$ query vector q, and the context X to which $x _ { i }$ belongs, we compute its Marginal Information Gain (MIG) $\dot { G } ( x _ { i } , q , \bar { X } )$ as follows: 

$$
G (x _ {i}, q, X) = \frac {x _ {i} ^ {\top} \cdot q}{\| x _ {i} \| \| q \|} - \max _ {x _ {j} \in X, j \neq i} \left(\frac {x _ {i} ^ {\top} \cdot x _ {j}}{\| x _ {i} \| \| x _ {j} \|}\right)\tag{1}
$$

Here, the first term measures the cosine similarity between $x _ { i }$ and the query q, representing its relevance; the second term captures the maximum cosine similarity between $x _ { i }$ and any other token in X, reflecting its redundancy. We demonstrate that using MIG yields superior expected performance compared to relying solely on relevance. (See Appendix $\mathbf { A } )$ 

## 4.2 COARSE-GRAINED GROUP REALLOCATION

For a given original context $X ,$ , we first encode it via a language model $( i . e . ,$ , encoder) to extract semantic information. 

$$
H = \operatorname{Encoder} (X),\tag{2}
$$

where H is the last hidden state. 

Then, we divide H into m equal-length, non-overlapping segments $H = \{ S _ { 1 } , S _ { 2 } , \dots , S _ { m } \}$ . Given an input query $Q = \{ q 1 , q 2 , . . . , q _ { n } \}$ , different segments exhibit varying degrees of relevance to $Q$ as well as differing levels of internal redundancy. Therefore, a fixed compression rate across all segments is suboptimal. Instead, we propose a dynamic compression rate reallocation strategy based on marginal information gain. 

First, we average pooling the query sequence $Q$ into a single query vector $\overline { { q } } { : }$ 

$$
\overline {{q}} = \frac {1}{| Q |} \sum_ {q _ {k} \in Q} q _ {k}.\tag{3}
$$

Then we select the representative vector $\hat { h _ { i } }$ of $S _ { i }$ that exhibits the highest relevance to the query q via: 

$$
\hat {h _ {i}} = \underset {h _ {j} \in S _ {i}} {\operatorname{argmax}} \left(\frac {h _ {j} ^ {\top} \cdot \overline {{q}}}{\| h _ {j} \| \| \overline {{q}} \|}\right).\tag{4}
$$

Then, we can compute the marginal information gain $G ( \hat { h _ { i } } , \overline { { q } } , H )$ for each segment $S _ { i }$ as defined in Equation (1), which now evaluates the trade-off between the segment’s relevance to the query and its redundancy with neighboring segments. 

Since segments with higher marginal gain are more informative and less redundant, they should be preserved more faithfully, i.e., assigned a smaller compression rate. To achieve this, we apply an inverse transformation to reverse the MIG ranking. We cant get the allocation weights via: 

$$
P _ {i} = \frac {e ^ {- G (\hat {h _ {i}} , \overline {{q}} , H)}}{\sum_ {j = 1} ^ {n} e ^ {- G (\hat {h _ {j}} , \overline {{q}} , H)}}, i = 1, 2, \ldots , n,\tag{5}
$$

where $\textstyle \sum _ { i = 1 } ^ { n } P _ { i } = 1$ . These weights $P _ { i }$ determine the proportion of the total allowed output length allocated to each segment. 

Finally, the target length $( i . e .$ ., number of tokens) for the compressed representation of segment $S _ { i }$ , denoted $L _ { i } ,$ , is computed as: 

$$
L _ {i} = L _ {\mathrm{org}} \cdot P _ {i},\tag{6}
$$

where $L _ { \mathrm { o r g } }$ is the length of the original input sequence. 

This dynamic reallocation mechanism enables COMI to efficiently allocate compression resources based on semantic importance and redundancy, preserving segments with higher marginal gain adaptively and improving downstream task performance under limited context length. 

## 4.3 FINE-GRAINED TOKEN MERGING

After dynamically reallocating the target lengths $L _ { i }$ for each compression segment $S _ { i }$ , we proceed to perform token merging within each segment to achieve compression. The goal is to preserve maximal informative content with respect to the query while minimizing redundancy among tokens. 

For each segment $S _ { i } = \left\{ h _ { 1 } , h _ { 2 } , \ldots , h _ { | S _ { i } | } \right\}$ , we compute the marginal information gain $G ( h _ { k } , S _ { i } )$ for every token $x _ { k } \in S _ { i }$ using the same formulation as in Equation (1). This score reflects a trade-off between the token’s relevance to the query (via cosine similarity with q) and its redundancy with the most similar token within the same segment. 

To merge all tokens in $S _ { i }$ into a single output token $\tilde { h } _ { i }$ that preserves the most informative content, we perform MIG-weighted merging via: 

$$
\tilde {h} _ {i} = \sum_ {h _ {k} \in S _ {i}} \frac {e ^ {G (h _ {k} , \overline {{q}} , S _ {i})} \cdot h _ {k}}{\sum_ {h _ {k} \in S _ {i}} e ^ {G (h _ {k} , \overline {{q}} , S _ {i})}}.\tag{7}
$$

That is, each token in the corresponding group is weighted by the softmax of its MIG, ensuring that tokens with higher information gain contribute more to the merged compressed representation $\tilde { h } _ { i }$ 

We can get the total compressed representation $\tilde { X }$ by: 

$$
\tilde {X} = \{\tilde {h} _ {1} \odot \tilde {h} _ {2} \odot \dots \odot \tilde {h} _ {m} \},\tag{8}
$$

where ⊙ denotes concatenation. 

## 4.4 TRAINING OBJECTIVE

We fine-tune COMI following the joint instruction-tuning approach (Lin et al., 2024), enabling it to generate correct outputs given a query and the original context. The difference lies in that we perform fine-tuning based on compressed representations, simultaneously training the encoder, LSA (to ensure the effectiveness of the compressed representations), and the decoder’s $W _ { Q } , W _ { K } , W _ { V }$ and $W _ { O }$ (to ensure knowledge extraction capability of the Decoder) (see Figure 2). 

$$
\mathcal {L} _ {n l l} = - \sum_ {i = 1} ^ {L _ {a}} \log p _ {\phi} \left(a _ {i} \mid \mathtt {L S A} (\widetilde {X}), q _ {1}, q _ {2},..., q _ {n}, a _ {<   i}\right),\tag{9}
$$

where $L _ { a }$ refers to the length of ground truth; LSA(·) denotes the LSA module; $p _ { \phi } ( \cdot )$ is the Decoder probability distribution obtained after the softmax function, and $a _ { i }$ denotes the i-th token in the predicted answer. 

## 5 EXPERIMENTS

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/49ac91fb-df74-46f9-b9b1-89a317c55626/7a6be82c3b05055c1d863ee2334a79abcaa0cf6cade130c2621a3edfc17a9551.jpg)



(a) Pressure test on NaturalQuestions.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/49ac91fb-df74-46f9-b9b1-89a317c55626/66ce40173cee68774b621779e3392d07ee470b87d8423ba44e369990fe29c763.jpg)



(b) Pressure test on 2WikiMQA.



Figure 4: Compression Pressure Test on NaturalQuestions and 2WikiMQA. As the compression constraint increases, although both COMI and Activation Beacon generally show a downward trend in EM and F1, COMI consistently remains higher than Activation Beacon.


In this section, we seek to answer the following four research questions: (1) How does COMI perform on various tasks? (RQ1) (2) What is the impact of compression constraint on COMI’s performance? (RQ2) (3) How effective is each individual component within COMI? (RQ3) (4) How does COMI impact native long context LLMs? 


Table 1: Experimental results on four QA benchmarks and the MultiNews summarization dataset. We bold the optimal and underline the suboptimal of baselines. EM refers to Exact Match and F1 refers to the F1 score. Closed-book indicates using only the input question as the input, while Original Prompt indicates using all retrieved documents as the input.


<table><tr><td rowspan="2">Methods</td><td colspan="2">NaturalQuestions</td><td colspan="2">2WikiMQA</td><td colspan="2">HotpotQA</td><td colspan="2">NarrativeQA</td><td>MultiNews</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>F1</td></tr><tr><td colspan="10">LLaMA-2-7B-Chat</td></tr><tr><td>Closed-book</td><td>20.21</td><td>21.89</td><td>24.92</td><td>27.81</td><td>17.22</td><td>24.03</td><td>1.2</td><td>10.78</td><td>-</td></tr><tr><td>Original Prompt</td><td>15.04</td><td>26.74</td><td>30.82</td><td>37.80</td><td>34.34</td><td>47.50</td><td>2.16</td><td>13.56</td><td>33.2</td></tr><tr><td colspan="10">Qwen2-7B-Instruct</td></tr><tr><td>Closed-book</td><td>22.64</td><td>20.15</td><td>30.07</td><td>20.97</td><td>2.82</td><td>10.18</td><td>2.82</td><td>10.18</td><td>-</td></tr><tr><td>Original Prompt</td><td>72.35</td><td>38.01</td><td>59.78</td><td>47.52</td><td>64.17</td><td>53.68</td><td>24.53</td><td>9.74</td><td>31.42</td></tr><tr><td colspan="10">16x compression constraint</td></tr><tr><td colspan="10">LLaMA-2-7B-Chat</td></tr><tr><td>StreamLLM</td><td>8.06</td><td>18.72</td><td>13.42</td><td>15.92</td><td>14.45</td><td>21.65</td><td>0.00</td><td>0.00</td><td>24.66</td></tr><tr><td>SnapKV</td><td>11.90</td><td>26.49</td><td>25.82</td><td>29.55</td><td>26.09</td><td>37.87</td><td>0.00</td><td>0.00</td><td>19.79</td></tr><tr><td>Activation Beacon</td><td>4.56</td><td>16.58</td><td>2.20</td><td>13.84</td><td>11.41</td><td>24.68</td><td>6.39</td><td>21.16</td><td>26.36</td></tr><tr><td>ICAE</td><td>1.50</td><td>5.93</td><td>12.92</td><td>21.52</td><td>8.24</td><td>16.90</td><td>0.00</td><td>3.86</td><td>20.85</td></tr><tr><td>LongLLMLingua</td><td>11.64</td><td>23.12</td><td>28.42</td><td>32.05</td><td>24.75</td><td>35.22</td><td>2.44</td><td>15.91</td><td>20.66</td></tr><tr><td>LLMLingua-2-large</td><td>17.74</td><td>27.91</td><td>24.26</td><td>26.60</td><td>15.71</td><td>25.95</td><td>4.51</td><td>17.38</td><td>20.83</td></tr><tr><td>GMSA</td><td>10.36</td><td>15.63</td><td>22.68</td><td>28.46</td><td>16.09</td><td>25.43</td><td>1.13</td><td>9.60</td><td>21.10</td></tr><tr><td>COMI</td><td>22.75</td><td>27.32</td><td>41.43</td><td>48.74</td><td>36.60</td><td>49.50</td><td>8.31</td><td>20.32</td><td>27.30</td></tr><tr><td colspan="10">Qwen2-7B-Instruct</td></tr><tr><td>SnapKV</td><td>6.06</td><td>19.64</td><td>8.09</td><td>25.52</td><td>13.61</td><td>30.84</td><td>0.00</td><td>11.12</td><td>25.56</td></tr><tr><td>Activation Beacon</td><td>11.67</td><td>27.53</td><td>36.53</td><td>49.11</td><td>39.73</td><td>55.25</td><td>6.63</td><td>24.07</td><td>33.60</td></tr><tr><td>LongLLMLingua</td><td>15.07</td><td>29.64</td><td>21.80</td><td>29.63</td><td>21.58</td><td>35.36</td><td>1.03</td><td>11.99</td><td>23.16</td></tr><tr><td>LLMLingua-2-large</td><td>10.36</td><td>20.31</td><td>18.26</td><td>25.35</td><td>2.44</td><td>11.28</td><td>8.27</td><td>11.28</td><td>21.39</td></tr><tr><td>GMSA</td><td>24.41</td><td>28.27</td><td>29.49</td><td>34.23</td><td>21.54</td><td>30.74</td><td>1.88</td><td>10.94</td><td>26.55</td></tr><tr><td>COMI</td><td>56.31</td><td>55.52</td><td>52.13</td><td>60.43</td><td>45.11</td><td>60.52</td><td>13.25</td><td>24.69</td><td>35.85</td></tr><tr><td colspan="10">32x compression constraint</td></tr><tr><td colspan="10">LLaMA-2-7B-Chat</td></tr><tr><td>StreamLLM</td><td>4.10</td><td>11.86</td><td>9.76</td><td>12.07</td><td>8.11</td><td>12.97</td><td>0.00</td><td>0.00</td><td>19.28</td></tr><tr><td>SnapKV</td><td>9.91</td><td>24.76</td><td>1.53</td><td>2.98</td><td>8.25</td><td>13.30</td><td>0.00</td><td>0.00</td><td>14.44</td></tr><tr><td>Activation Beacon</td><td>1.47</td><td>11.76</td><td>1.71</td><td>13.38</td><td>13.94</td><td>26.75</td><td>4.79</td><td>17.02</td><td>24.35</td></tr><tr><td>ICAE</td><td>1.31</td><td>10.82</td><td>12.91</td><td>21.68</td><td>8.20</td><td>16.95</td><td>0.00</td><td>3.16</td><td>19.00</td></tr><tr><td>LongLLMLingua</td><td>7.50</td><td>18.86</td><td>28.32</td><td>31.76</td><td>25.12</td><td>34.78</td><td>2.44</td><td>15.10</td><td>18.56</td></tr><tr><td>LLMLingua-2-large</td><td>14.09</td><td>22.76</td><td>21.98</td><td>23.89</td><td>14.02</td><td>23.33</td><td>3.57</td><td>15.50</td><td>16.46</td></tr><tr><td>GMSA</td><td>9.53</td><td>13.94</td><td>24.83</td><td>28.62</td><td>17.93</td><td>27.15</td><td>1.13</td><td>8.56</td><td>19.60</td></tr><tr><td>COMI</td><td>16.99</td><td>23.94</td><td>37.34</td><td>43.85</td><td>31.46</td><td>43.32</td><td>6.83</td><td>15.90</td><td>24.71</td></tr><tr><td colspan="10">Qwen2-7B-Instruct</td></tr><tr><td>SnapKV</td><td>5.76</td><td>17.93</td><td>7.16</td><td>21.36</td><td>10.57</td><td>25.70</td><td>0.00</td><td>10.98</td><td>21.35</td></tr><tr><td>Activation Beacon</td><td>8.52</td><td>22.26</td><td>38.79</td><td>49.91</td><td>37.77</td><td>52.76</td><td>5.18</td><td>21.29</td><td>32.38</td></tr><tr><td>LongLLMLingua</td><td>13.37</td><td>27.39</td><td>21.19</td><td>28.91</td><td>22.59</td><td>35.45</td><td>1.03</td><td>11.91</td><td>21.60</td></tr><tr><td>LLMLingua-2-large</td><td>9.60</td><td>18.16</td><td>18.81</td><td>24.90</td><td>2.26</td><td>10.27</td><td>2.26</td><td>10.27</td><td>17.70</td></tr><tr><td>GMSA</td><td>24.63</td><td>27.81</td><td>28.59</td><td>32.81</td><td>21.28</td><td>30.08</td><td>2.73</td><td>11.04</td><td>25.49</td></tr><tr><td>COMI</td><td>49.15</td><td>49.59</td><td>48.89</td><td>57.07</td><td>40.46</td><td>55.24</td><td>11.18</td><td>22.30</td><td>33.60</td></tr></table>

## 5.1 SETTINGS

Training. COMI requires only a single training run to be effectively applied to multiple downstream tasks (i.e., QA, summarization). We sampled 20,000 examples each from NaturalQuestions (Liu et al., 2024a), HotpotQA (Yang et al., 2018), 2WikiMQA (Ho et al., 2020), NarrativeQA (Kocisky´ et al., 2018), and MultiNews (Fabbri et al., 2019) to form the final training set. All training and testing samples have token lengths no longer than 32K. During training, the batch size is set to 64, the learning rate is set to 1e-5, and a linear decay schedule is employed. The training paradigm is illustrated in Figure 2. During training, we randomly sample a compression rate (e.g., 16x or 

32x) for each training sample. To ensure fair comparison, we train GMSA in the same datasets and settings. 

Implementation. Our implementation is based on LLaMA-2-7B (Chat) and Qwen2-7B (Instruct). To ensure a fair comparison, all baseline results are re-implemented using official open-source code. All experiments are conducted using the Hugging Face framework on 8 NVIDIA H20 (94GB) GPUs. 

Evaluation Metrics. For question answering (i.e., NaturalQuestions, HotpotQA, 2WikiMQA, and NarrativeQA), we report both Exact Match (EM) (Lewis et al., 2020) and F1 score (Yang et al., 2018). Summarization performance on MultiNews is measured by F1 score. 

Baselines. We conduct comprehensive comparisons with various methods in both context compression and KV-cache compression, including hard prompt compression methods (i.e., LongLLMLingua (Jiang et al., 2024), LLMLingua-2-large (Pan et al., 2024)), soft prompt compression methods (i.e., ICAE (Ge et al., 2024), GMSA (Tang et al., 2025b)), and KV-cache compression methods (i.e., StreamLLM (Xiao et al., 2024), SnapKV (Li et al., 2024), Activation Beacon (Zhang et al., 2025)). 

## 5.2 MAIN RESULT

For RQ1, COMI demonstrates superior performance on both question answering (QA) and summarization tasks (see Table 1). On QA tasks, where input questions are typically specific and relate to a particular piece of information within a long context, COMI nearly achieves state-of-the-art results across all compression constraints and evaluation metrics. This holds true for single-hop questions (i.e., NaturalQuestions), multi-hop questions (i.e., HotpotQA, 2WikiMQA), and QA on extremely long texts (i.e., NarrativeQA, which we test on samples with a maximum length of 32K). For instance, with a compression rate of 32x and using Qwen2-7B-Instruct as the backbone, COMI improves the Exact Match (EM) score by approximately 25 over suboptimal baseline. This highlights COMI’s exceptional performance under high compression rates. The EM metric reflects the upper bound of a model’s answer, while the F1 score additionally requires the model’s output length to be as close to the ground truth as possible. The strong performance on both metrics indicates that COMI not only answers questions more accurately but also produces outputs that are consistent in length with the reference answers. On summarization task, the input query is often a global request (e.g., “Please summarize the preceding text”). This setting requires the model to preserve and understand global information. COMI’s leading performance on these tasks demonstrates its strong comprehension capabilities even without explicit relevant information, confirming its high robustness. 

For RQ2, as shown in Figure 4, with the compression ratio gradually increasing (from 2x to 32x), while both COMI and Activation Beacon generally show a downward trend, COMI significantly outperforms Activation Beacon (especially in the EM metric, e.g., nearly 40 point higher on the NaturalQuestions dataset under the 32x compression constraint). EM requires the generated answer to perfectly match the ground truth, indicating that compared to Activation Beacon, COMI effectively retains key information (directly related to the ground truth) across various compression rates. 

## 5.3 ABLATION STUDY

To address RQ3, we conducted four ablation studies to examine how each component in COMI contributes to its performance (see Table 2): (1) Ours w/o Coarse-Grained Group Reallocation replaces dynamic grouping with a uniform partition, so every token group 


Table 2: Ablation study analysis on NaturalQuestions, 2WikiMQA under 32x compression constraint using Qwen2-7B.


<table><tr><td rowspan="2">Methods</td><td colspan="2">NaturalQuestions</td><td colspan="2">2WikiMQA</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>Default</td><td>56.31</td><td>55.52</td><td>52.13</td><td>60.43</td></tr><tr><td>w/o Coarse-Grained Group Reallocation</td><td>54.54</td><td>54.26</td><td>49.82</td><td>58.06</td></tr><tr><td>w/o Fine-Grained Tokens Merging</td><td>50.81</td><td>50.82</td><td>47.88</td><td>56.45</td></tr><tr><td>w/o Coarse-Grained level Redundancy</td><td>54.59</td><td>52.81</td><td>50.14</td><td>58.64</td></tr><tr><td>w/o Fine-Grained level Redundancy</td><td>51.53</td><td>52.89</td><td>49.50</td><td>57.87</td></tr></table>

## is compressed at the

same rate. (2) Ours w/o Fine-Grained Tokens Merging substitutes our information-gain-based weighting with plain average pooling. (3) Ours w/o Coarse-Grained level Redundancy reallocates groups based solely on relevance, ignoring inter-group redundancy. (4) Ours w/o Fine-Grained level Redundancy performs token merging without considering redundancy among tokens within the same group. 

Removing any component causes a clear drop in all metrics, demostrating the necessity and effectiveness of each one. Eliminating Coarse-Grained Group Reallocation misallocates the compression budget and risks losing key information, whereas dropping Fine-Grained Token Merging deprives COMI of its fine-grained intra-group sensitivity and dilutes key details (both harming overall results). Disregarding redundancy at either the coarse-grained or fine-grained stage retains redundant information, weakening the compressed representation, making it harder for the model to learn, and ultimately degrading performance. 

## 5.4 EFFICIENCY ANALYSIS

In this section, we analyze the computational efficiency of the COMI. By compressing the original context through coarse-grained group reallocation and fine-grained token merging, COMI significantly reduces the sequence length processed during generation, thereby lowering inference cost. The overall process consists of two main stages: compression and generation, whose floating-point operations (FLOPs) are modeled separately. 


Table 3: Latency Evaluation (seconds) under 32× compression constraint.


<table><tr><td>Method</td><td>Compression</td><td>Generation</td><td>End-to-End</td></tr><tr><td colspan="4">NarrativeQA</td></tr><tr><td>Original Prompt</td><td>-</td><td>-</td><td>7.04</td></tr><tr><td>SnapKV</td><td>-</td><td>-</td><td>2.10</td></tr><tr><td>Activation Beacon</td><td>-</td><td>-</td><td>3.93</td></tr><tr><td>LongLLMLingua</td><td>68.84</td><td>2.12</td><td>70.95</td></tr><tr><td>GMSA</td><td>1.57</td><td>0.53</td><td>2.10</td></tr><tr><td>COMI</td><td>2.76</td><td>0.50</td><td>3.27</td></tr><tr><td colspan="4">MultiNews</td></tr><tr><td>Original Prompt</td><td>-</td><td>-</td><td>8.58</td></tr><tr><td>SnapKV</td><td>-</td><td>-</td><td>5.03</td></tr><tr><td>Activation Beacon</td><td>-</td><td>-</td><td>7.94</td></tr><tr><td>LongLLMLingua</td><td>5.44</td><td>2.81</td><td>8.25</td></tr><tr><td>GMSA</td><td>0.52</td><td>3.34</td><td>3.86</td></tr><tr><td>COMI</td><td>0.66</td><td>3.43</td><td>4.09</td></tr></table>

The compression stage consists of two parts: first, the original context is partitioned into groups, and each group’s size is dynamically adjusted based on query relevance and inter-group redundancy; second, tokens within each group are pooled via weighted to generate compressed representations. 

Let $L _ { q }$ the query length, and $L _ { g } = \lceil L _ { o r g } / r \rceil$ the compressed length, where r is the compression rate. The FLOPs for the compression stage can be expressed as: 

$$
\mathrm{FLOPs} ^ {c o m p} = F ^ {\mathrm{GroupRealloc}} (L _ {o r g}) + F ^ {\mathrm{Pooling}} (L _ {o r g}, L _ {c}) + F ^ {\mathrm{LSA}} (L _ {c}),
$$

where $L _ { c }$ refers to compressed context length; $F ^ { \mathrm { G r o u p R e a l l o c } } ( L )$ denotes the group reallocation cost (complexity $O ( N _ { g } \log N _ { g } + N _ { g } ^ { 2 } )$ , with $L _ { g } \ll L _ { o r g } { ^ 2 } ) ; F ^ { \mathrm { m e r g i n g } } ( r , r )$ denotes the merging cost (complexity $O ( r ^ { 2 } )$ ), with $r \ll L _ { o r g } { } ^ { 2 } )$ . Due to their small input scales, both operations incur only lightweight overhead. 

For the generation stage, assuming the answer length is $L _ { a } , L _ { a }$ forward passes are required. The FLOPs for the i-th forward pass depend on the input sequence length, $i . e . , L _ { c }$ and query length $L _ { q } .$ 

Thus, the FLOPs for each forward pass are given by: 

$$
\mathrm{FLOPs} _ {i} ^ {\text {forward}} = F ^ {\text {Decoder}} (L _ {c}, L _ {q}, i),
$$

Combining both compression and generation stages, the total FLOPs is: 

$$
\mathrm{FLOPs} = \sum_ {i = 1} ^ {L _ {a}} \mathrm{FLOPs} _ {i} ^ {\text { forward }} + \mathrm{FLOPs} ^ {\text { comp }}.
$$

Whether in question-answering (QA) tasks or generation-centric summarization task, COMI achieves more than a 2× end-to-end speedup over the Original Prompt at a 32× compression constraint. For all compression methods, the end-to-end latency can be divided into compression and generation phases (see Table 3).<sup>2</sup> In contrast, the Original Prompt incurs only generation latency. 5.5 THE IMPACT OF COMI ON NATIVE LONG-CONTEXT LLMS 

For RQ4, to evaluate the impact of COMI on models with native long-context capabilities, we train COMI using Qwen3-4B-Instruct (which natively supports a 256K context length) and use F1 as an evaluation metric. As shown in Table 4, even compared to the strong baseline of feeding the full original prompt, COMI achieves superior performance across all datasets at both 16× and 32× compression constraint. For instance, on the NaturalQuestions dataset at 16× compression, COMI attains an F1 score of 34.79, compared to only 16.90 for the original prompt, demonstrating that COMI can enhance performance even for models with strong native long-context capabilities. 


Table 4: The performance of COMI on different QA datasets (Qwen3-4B-Instruct as backbone).


<table><tr><td>Method</td><td>NaturalQuestions</td><td>2WikiMQA</td><td>HotpotQA</td><td>NarrativeQA</td></tr><tr><td colspan="5">16× compression constraint</td></tr><tr><td>Original Prompt</td><td>16.90</td><td>22.51</td><td>34.16</td><td>11.35</td></tr><tr><td>COMI</td><td>34.79</td><td>45.01</td><td>44.44</td><td>15.71</td></tr><tr><td colspan="5">32× compression constraint</td></tr><tr><td>Original Prompt</td><td>16.90</td><td>22.51</td><td>34.16</td><td>11.35</td></tr><tr><td>COMI</td><td>28.89</td><td>41.19</td><td>39.71</td><td>13.25</td></tr></table>

## 6 CONCLUSION

We propose COMI, a coarse-to-fine context compression framework that dynamically optimizes for both task relevance and semantic diversity under high compression rates. By introducing Marginal Information Gain (MIG) (i.e., a metric that penalizes redundancy while rewarding query relevance) COMI adaptively reallocates compression budgets between segments and performs token merging within groups. Extensive experiments demonstrate that COMI significantly outperforms existing methods, e.g., achieving up to a 25-point EM gain under 32x compression. This work establishes MIG as a critical criterion for efficient and effective long-context modeling in LLMs. 

## ETHICS STATEMENT

This work introduces COMI, an encoder-decoder based framework designed to achieve adaptive coarse-to-fine context compression through marginal information gain. The data and models used in our research are released under open-source licenses and sourced from open platforms. Although our work may have various societal impacts, it does not introduce any additional ethical concerns compared to existing text compression methods. Therefore, we believe it is unnecessary to specifically highlight any particular ethical issues here. 

## ACKNOWLEDGEMENT

This work was supported by Alibaba Group through Alibaba Research Intern Program. Libin Zheng is supported by National Natural Science Foundation of China (No. 62472455, U22B2060). 

## REFERENCES



Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit´ Sanghai. GQA: training generalized multi-query transformer models from multi-head checkpoints. In EMNLP, pp. 4895–4901. Association for Computational Linguistics, 2023. 





Jeff A. Bilmes. Submodularity in machine learning and artificial intelligence. CoRR, abs/2202.00132, 2022. 





William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, and Jonathan Ragan-Kelley. Reducing transformer key-value cache size with cross-layer attention. In NeurIPS, 2024. 





Zhiwei Cao, Qian Cao, Yu Lu, Ningxin Peng, Luyang Huang, Shanbo Cheng, and Jinsong Su. Retaining key information under high compression ratios: Query-guided compressor for llms. In ACL (1), pp. 12685–12695. Association for Computational Linguistics, 2024. 





Shaoshen Chen, Yangning Li, Zishan Xu, Yongqin Zeng, Shunlong Wu, Xinshuo Hu, Zifei Shan, Xin Su, Jiwei Tang, Yinghui Li, and Hai-Tao Zheng. DAST: context-aware compression in llms via dynamic allocation of soft tokens. In ACL (Findings), pp. 20544–20552. Association for Computational Linguistics, 2025. 





Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one token. In NeurIPS, 2024. 





Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. In EMNLP, pp. 3829–3846. Association for Computational Linguistics, 2023. 





Eunseong Choi, June Park, Hyeri Lee, and Jongwuk Lee. Conflict-aware soft prompting for retrieval-augmented generation. CoRR, abs/2508.15253, 2025. 





Thomas M. Cover and Joy A. Thomas. Elements ofinformation theory (2. ed.). Wiley, 2006. 





Yuhong Dai, Jianxun Lian, Yitian Huang, Wei Zhang, Mingyang Zhou, Mingqi Wu, Xing Xie, and Hao Liao. Pretraining context compressor for large language models with embedding-based memory. In ACL (1), pp. 28715–28732. Association for Computational Linguistics, 2025. 





Maulik A. Dave. Review of ”information theory, inference, and learning algorithms by david j. c. mackay”, cambridge university press, 2003. SIGACT News, 37(4):34–36, 2006. 





DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025. 





Alexander R. Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir R. Radev. Multi-news: A largescale multi-document summarization dataset and abstractive hierarchical model. In ACL (1), pp. 1074–1084. Association for Computational Linguistics, 2019. 





Yixiong Fang, Tianran Sun, Yuling Shi, and Xiaodong Gu. Attentionrag: Attention-guided context pruning in retrieval-augmented generation. CoRR, abs/2503.10720, 2025. 





Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. In ICLR. OpenReview.net, 2024. 





James A Hanley and Barbara J McNeil. The meaning and use of the area under a receiver operating characteristic (roc) curve. Radiology, 143(1):29–36, 1982. 





Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing A multihop QA dataset for comprehensive evaluation of reasoning steps. In COLING, pp. 6609–6625. International Committee on Computational Linguistics, 2020. 





Taeho Hwang, Sukmin Cho, Soyeong Jeong, Hoyun Song, SeungYoon Han, and Jong C. Park. EXIT: context-aware extractive compression for enhancing retrieval-augmented generation. In ACL (Findings), pp. 4895–4924. Association for Computational Linguistics, 2025. 





Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. In EMNLP, pp. 13358–13376. Association for Computational Linguistics, 2023. 





Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. In ACL (1), pp. 1658–1677. Association for Computational Linguistics, 2024. 





Tomas Kocisk´ y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, G´ abor Melis,´ and Edward Grefenstette. The narrativeqa reading comprehension challenge. Trans. Assoc. Comput. Linguistics, 6:317–328, 2018. 





Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe¨ Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In NeurIPS, 2020. 





Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. Compressing context to enhance inference efficiency of large language models. In EMNLP, pp. 6342–6353. Association for Computational Linguistics, 2023. 





Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: LLM knows what you are looking for before generation. In NeurIPS, 2024. 





Zongqian Li, Yixuan Su, and Nigel Collier. 500xcompressor: Generalized prompt compression for large language models. In ACL (1), pp. 25081–25091. Association for Computational Linguistics, 2025. 





Huanxuan Liao, Wen Hu, Yao Xu, Shizhu He, Jun Zhao, and Kang Liu. Beyond hard and soft: Hybrid context compression for balancing local and global information retention. CoRR, abs/2505.15774, 2025. 





Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. RA-DIT: retrieval-augmented dual instruction tuning. In ICLR. OpenReview.net, 2024. 





Barys Liskavets, Maxim Ushakov, Shuvendu Roy, Mark Klibanov, Ali Etemad, and Shane K. Luke. Prompt compression with context-aware sentence encoding for fast and improved LLM inference. In AAAI, pp. 24595–24604. AAAI Press, 2025. 





Langming Liu, Shilei Liu, Yujin Yuan, Yizhen Zhang, Bencheng Yan, Zhiyuan Zeng, Zihao Wang, Jiaqi Liu, Di Wang, Wenbo Su, Pengjie Wang, Jian Xu, and Bo Zheng. Uqabench: Evaluating user embedding for prompting llms in personalized question answering. In KDD (2), pp. 5652–5661. ACM, 2025a. 





Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Trans. Assoc. Comput. Linguistics, 12:157–173, 2024a. 





Xin Liu, Runsong Zhao, Pengcheng Huang, Xinyu Liu, Junyi Xiao, Chunyang Xiao, Tong Xiao, Shengxiang Gao, Zhengtao Yu, and Jingbo Zhu. Autoencoding-free context compression for llms via contextual semantic anchors. CoRR, abs/2510.08907, 2025b. 





Xinyu Liu, Runsong Zhao, Pengcheng Huang, Chunyang Xiao, Bei Li, Jingang Wang, Tong Xiao, and JingBo Zhu. Forgetting curve: A reliable method for evaluating memorization capability for long-context models. In EMNLP, pp. 4667–4682. Association for Computational Linguistics, 2024b. 





Qingsong Lv, Yangning Li, Zihua Lan, Zishan Xu, Jiwei Tang, Yinghui Li, Wenhao Jiang, Hai-Tao Zheng, and Philip S. Yu. RAISE: reinforenced adaptive instruction selection for large language models. CoRR, abs/2504.07282, 2025. 





Jesse Mu, Xiang Li, and Noah D. Goodman. Learning to compress prompts with gist tokens. In NeurIPS, 2023. 





George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for maximizing submodular set functions - I. Math. Program., 14(1):265–294, 1978. 





Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Ruhle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang. Llmlingua-¨ 2: Data distillation for efficient and faithful task-agnostic prompt compression. In ACL (Findings), pp. 963–981. Association for Computational Linguistics, 2024. 





Hanchuan Peng, Fuhui Long, and Chris H. Q. Ding. Feature selection based on mutual information: Criteria of max-dependency, max-relevance, and min-redundancy. IEEE Trans. Pattern Anal. Mach. Intell., 27(8):1226–1238, 2005. 





Aleksandar Petrov, Mark Sandler, Andrey Zhmoginov, Nolan Miller, and Max Vladymyrov. Long context in-context compression by getting to the gist of gisting. CoRR, abs/2504.08934, 2025. 





David Rau, Shuai Wang, Herve D´ ejean, and St´ ephane Clinchant. Context embeddings for efficient´ answer generation in RAG. CoRR, abs/2407.09252, 2024. 





Claude E. Shannon. A mathematical theory of communication. Bell Syst. Tech. J., 27(3):379–423, 1948. 





Noam Shazeer. Fast transformer decoding: One write-head is all you need. CoRR, abs/1911.02150, 2019. 





Sijun Tan, Xiuyu Li, Shishir G. Patil, Ziyang Wu, Tianjun Zhang, Kurt Keutzer, Joseph Gonzalez, and Raluca A. Popa. Lloco: Learning long contexts offline. In EMNLP, pp. 17605–17621. Association for Computational Linguistics, 2024. 





Jiwei Tang, Jin Xu, Tingwei Lu, Zhicheng Zhang, Yiming Zhao, Lin Hai, and Hai-Tao Zheng. Perception compressor: A training-free prompt compression framework in long context scenarios. In NAACL (Findings), pp. 4093–4108. Association for Computational Linguistics, 2025a. 





Jiwei Tang, Zhicheng Zhang, Shunlong Wu, Jingheng Ye, Lichen Bai, Zitai Wang, Tingwei Lu, Jiaqi Chen, Lin Hai, Hai-Tao Zheng, and Hong-Gee Kim. GMSA: enhancing context compression via group merging and layer semantic alignment. CoRR, abs/2505.12215, 2025b. 





Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence, 2025. URL https://arxiv.org/abs/2507.20534. 





Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In NIPS, pp. 5998–6008, 2017. 





Chupei Wang and Jiaqiu Vince Sun. Unable to forget: Proactive lnterference reveals working memory limits in LLMs beyond context length. In ICML 2025 Workshop on Long-Context Foundation Models, 2025. URL https://openreview.net/forum?id=YUHksmL8aw. 





Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In ICLR. OpenReview.net, 2024. 





Fangyuan Xu, Weijia Shi, and Eunsol Choi. RECOMP: improving retrieval-augmented lms with context compression and selective augmentation. In ICLR. OpenReview.net, 2024. 





An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, et al. Qwen2 technical report. CoRR, abs/2407.10671, 2024. 





Diji Yang, Linda Zeng, Jinmeng Rao, and Yi Zhang. Knowing you don’t know: Learning when to continue search in multi-round RAG through self-practicing. In SIGIR, pp. 1305–1315. ACM, 2025. 





Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In EMNLP, pp. 2369–2380. Association for Computational Linguistics, 2018. 





Xubing Ye, Yukang Gan, Xiaoke Huang, Yixiao Ge, and Yansong Tang. Voco-llama: Towards vision compression with large language models. In CVPR, pp. 29836–29846. Computer Vision Foundation / IEEE, 2025. 





Chanwoong Yoon, Taewhoo Lee, Hyeon Hwang, Minbyul Jeong, and Jaewoo Kang. Compact: Compressing retrieved documents actively for question answering. In EMNLP, pp. 21424–21439. Association for Computational Linguistics, 2024. 





Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. Long context compression with activation beacon. In ICLR. OpenReview.net, 2025. 





Runsong Zhao, Xin Liu, Xinyu Liu, Pengcheng Huang, Chunyang Xiao, Tong Xiao, and Jingbo Zhu. Position ids matter: An enhanced position layout for efficient context compression in large language models, 2025a. URL https://arxiv.org/abs/2409.14364. 





Yiming Zhao, Jiwei Tang, Shimin Di, Libin Zheng, Jianxing Yu, and Jian Yin. Cos: Towards optimal event scheduling via chain-of-scheduling. CoRR, abs/2511.12913, 2025b. 





Yunlong Zhao, Haoran Wu, and Bo Xu. Leveraging attention to effectively compress prompts for long-context llms. In AAAI, pp. 26048–26056. AAAI Press, 2025c. 



## A THEORETICAL ANALYSIS: MIG VS. PURE RELEVANCE

## A.1 PRELIMINARIES

Let $X = \{ x _ { 1 } , \ldots , x _ { n } \} \subset \mathbb { R } ^ { d }$ be a set of token embeddings, assumed to be zero-mean and unit-norm. Let $q \in \mathbb { R } ^ { \dot { d } }$ be a query vector, also zero-mean and unit-norm, representing the target information or label. We use the cosine similarity to measure the relevance between embeddings. Specifically, for any two vectors $u , v \in \mathbb { R } ^ { d } , \cos ( u , v ) = u ^ { \top } v$ 

We define the “relevance” of a token $x _ { i }$ to the query $q$ as: 

$$
R (i) = \cos (x _ {i}, q) = \frac {x _ {i} ^ {\top} q}{\| x _ {i} \| \| q \|}.\tag{10}
$$

Since $x _ { i }$ and $q$ are unit-norm, $R ( i ) = x _ { i } ^ { \top } q .$ . This term captures the linear correlation between the token embedding and the query direction. 

We also define the “redundancy” of a token $x _ { i }$ with respect to a set of already selected tokens $S$ as: 

$$
\text { Redundancy } (i, S) = \max _ {x _ {j} \in S} \cos (x _ {i}, x _ {j}) = \max _ {x _ {j} \in S} \frac {x _ {i} ^ {\top} x _ {j}}{\| x _ {i} \| \| x _ {j} \|}.\tag{11}
$$

Again, assuming unit norm, Redundancy $( i , S ) = \operatorname* { m a x } _ { x _ { j } \in S } x _ { i } ^ { \top } x _ { j }$ . This measures how similar $x _ { i }$ is to the most similar token already in the set S. 

## A.2 MARGINAL INFORMATION GAIN (MIG)

Inspired by the max-relevance min-redundancy principle (Peng et al., 2005) and practical considerations in token compression, we define the Marginal Information Gain (MIG) of a token $x _ { i }$ with respect to a set S as: 

$$
G (i \mid S) = R (i) - \text { Redundancy } (i, S) = \cos (x _ {i}, q) - \max _ {x _ {j} \in S} \cos (x _ {i}, x _ {j}).\tag{12}
$$

When considering a single token $x _ { i }$ in isolation $( \mathrm { i . e . , } S = \emptyset )$ , the redundancy term is conventionally set to 0, simplifying MIG to: 

$$
G (i) = \cos (x _ {i}, q).\tag{13}
$$

However, the true power of MIG comes when selecting tokens sequentially or when comparing them against a context. For selection purposes, we are interested in $G ( i \mid \bar { S } \cup \{ x _ { i } \} )$ , which is $G ( i | S ) = \cos ( x _ { i } , q ) - \operatorname* { m a x } _ { x _ { j } \in S } \cos ( x _ { i } , x _ { j } )$ 

## A.3 COMPARISON: PURE RELEVANCE VS. MIG

Let $X _ { \mathrm { t o p - k } }$ be the set of k tokens with the highest relevance $R ( i )$ . Let $S _ { \mathrm { R E I } }$ be a set of tokens selected greedily using only relevance $R ( i )$ . Let S<sub>MIG</sub> be a set of tokens selected greedily using MIG $G ( i \mid S )$ 

Lemma A.1 (Information Preserved by Pure Relevance vs. MIG). Assume we are selecting a set of K tokens. 

1. A strategy based on pure relevance $R ( i )$ will select tokens that are highly correlated with the query q. 

2. A strategy based on MIG $G ( i \mid S )$ will select tokens that are highly correlated with q but are dissimilar to tokens already selected. 

Proof 1. Part 1. is direct: selecting based on $R ( i )$ means prioritizing tokens with the highest cos $( x _ { i } , q )$ 

Part 2. follows from the definition of $G ( i \mid S )$ . By subtracting ma $\mathfrak { c } _ { x _ { j } \in S } \cos ( x _ { i } , x _ { j } )$ , the score $G ( i \mid$ S) is naturally penalized when $x _ { i }$ is highly correlated with existing tokens in S. This encourages the selection of tokens that provide “new” information relative to what has already been gathered, as opposed to reinforcing existing information. 

Theorem 1 (Superiority of MIG under Redundancy). Let $f ( S ) = I ( S ; y )$ be the mutual information (Dave, 2006) between the selected set of tokens S and the target label y. Under the assumption that the token embeddings and the query are sampledfrom a distribution where high relevance often co-occurs with high redundancy among top-relevant tokens (a common scenario in natural language processing), and $i f f ( S )$ is approximately submodular (Bilmes, 2022), then a greedy selection strategy (Nemhauser et al., 1978) using MIG $( S _ { M I G } )$ is expected to yield a higher mutual information with the target y compared to a greedy strategy using only relevance $( S _ { R E L } )$ 

Proof 2. Let’s consider the scenario where there is significant redundancy among the most relevant tokens. Suppose we select K tokens. 

Pure Relevance Strategy $\mathrm { \Gamma } ( S _ { R E L } ) .$ : This strategy would greedily select tokens based on $R ( i ) =$ cos $( x _ { i } , q )$ . If several tokens $x _ { i _ { 1 } } , x _ { i _ { 2 } } , \ldots , x _ { i _ { p } }$ are all highly relevant to $\textit { l } ( i . e . , R ( i _ { m } )$ is large $f o r$ $m = 1 , \ldots , p ) ,$ , but also highly correlated with each other $( i . e . ,$ , cos $( x _ { i _ { m } } , x _ { i _ { l } } )$ is large for m $\neq l )$ the pure relevance strategy might select many of these redundant tokens. 

Under the approximation that mutual information is proportional to the square of the correlation coefficient, $\hat { I } ( x _ { i } ; y ) \propto R ( i ) ^ { 2 }$ . When multiple tokens are highly correlated with each other, the additional information they contribute about y diminishes. $H x _ { i }$ and $x _ { j }$ are selected and cos $( x _ { i } , x _ { j } ) = \tau$ the information from $x _ { j }$ that is $\ " \boldsymbol { n } e w \prime \boldsymbol { \cdot }$ with respect to $x _ { i }$ is reduced. In a simplified Gaussian setting, this redundancy can lead to an information overlap approximately proportional to $\tau ^ { 2 }$ (Cover $\&$ Thomas, 2006). The pure relevance strategy does not explicitly account for this overlap, potentially leading to $a \ ^ { \cdots } l o s s ^ { \prime \prime }$ in effective information captured compared to a strategy that penalizes redundancy. 

MIG Strategy (S<sub>MIG</sub>): This strategy selects tokens based on $G ( i \ \mathrm { ~  ~ { ~ \cal ~ S ~ } ~ } ) = \cos ( x _ { i } , q ) -$ $\mathrm { m a x } _ { x _ { j } \in S } \cos ( x _ { i } , x _ { j } )$ When considering a token $x _ { i }$ that is highly relevant to $\textit { q } ( l a r g e \cos ( x _ { i } , q ) )$ but also highly redundant with an already selected token $x _ { j }$ (large cos $\scriptstyle \left( x _ { i } , x _ { j } \right)$ for some $x _ { j } \in S )$ its MIG score $G ( i \mid S )$ will be significantly reduced. This actively discourages the selection of such redundant tokens. 

Therefore, when redundancy is present among top-relevant tokens, the MIG strategy is more likely to select a set of tokens that are both relevant to the query and collectively provide diverse, low redundant information. This leads to a better approximation ofthe true mutual information $I ( S ; y )$ $H f ( S ) = I ( S ; y )$ exhibits submodularity (a common assumption for information-theoretic objectives in feature selection), then a greedy approach based on marginal gain (MIG) is guaranteed to provide a $( 1 - 1 / e )$ approximation to the optimal set. The MIG’s explicit penalization of redundancy directly addresses the information loss incurred by redundantfeatures, thus yielding a higher expected mutual information. 

Conclusion: MIG provides a more robust criterion for token selection and compression compared to merely considering relevance. By explicitly penalizing redundancy with already selected tokens, MIG aims to capture a more diverse set of informative features, leading to a higher preservation of mutual information with the target query, especially in scenarios characterized by high token redundancy. 

## B EXPERIMENTAL EVIDENCE: MIG VS. PURE RELEVANCE

To directly validate that Marginal Information Gain (MIG) better captures token importance than pure relevance, we conduct a diagnostic study on NaturalQuestions, decoupled from the full COMI pipeline. We treat each context segment’s representative token and evaluate whether its score (MIG vs. relevance) predicts if the segment contains the ground-truth. We use AUC Score (Hanley & McNeil, 1982) as evaluation metric. 

As shown in Table 5, MIG achieves a higher AUC (0.5809) than relevance (0.5423), demonstrating its superior discriminative capability for identifying answer-critical segments. 

Furthermore, when retaining top segments by each metric, MIG yields significantly lower redundancy. We quantify redundancy as the average pairwise cosine similarity (excluding self-similarity) among the K compressed embeddings $\{ \mathbf { e } _ { 1 } , \hdots , \mathbf { e } _ { K } \}$ 

$$
\text{Redundancy Score} = \left\{ \begin{array}{ll}0 & \text{if $K\leq 1$},\\ \frac{1}{K(K - 1)}\sum_{i = 1}^{K}\sum_{\substack{j = 1\\ j\neq i}}^{K}\frac{\mathbf{e}_{i}^{\top}\mathbf{e}_{j}}{\|\mathbf{e}_{i}\| \| \mathbf{e}_{j}\|} & \text{if $K > 1$.} \end{array} \right.
$$


Table 6 shows MIG consistently reduces redundancy across different retention ratios.



Table 5: AUC Score for predicting answercontaining segments on NaturalQuestions.



Table 6: Redundancy of different metrics at different retention ratios.


<table><tr><td>Metric</td><td>AUC Score</td></tr><tr><td>Relevance</td><td>0.5423</td></tr><tr><td>MIG</td><td>0.5809</td></tr></table>

<table><tr><td rowspan="2">Metric</td><td colspan="2">Retention Ratio</td></tr><tr><td>25%</td><td>50%</td></tr><tr><td>Relevance</td><td>0.6839</td><td>0.6664</td></tr><tr><td>MIG</td><td>0.5673</td><td>0.5956</td></tr></table>

## C COMPREHENSIVE COMPARISON UNDER LOW COMPRESSION RATES

To comprehensively study the performance of different baselines under low compression rates, we evaluate COMI against additional baselines (including SnapKV, LongLLMLingua, and LLMLingua-2) on NaturalQuestions and 2WikiMQA using Qwen2-7B-Instruct as the backbone. As shown in Table 7, COMI consistently achieves the highest Exact Match (EM) scores across all compression rates (2× to 32×), demonstrating robust of COMI. 


Table 7: EM scores under compression rates (2×–32×) on NaturalQuestions and 2WikiMQA.


<table><tr><td rowspan="2">Method</td><td colspan="5">Compression Rate</td></tr><tr><td>2×</td><td>4×</td><td>8×</td><td>16×</td><td>32×</td></tr><tr><td colspan="6">NaturalQuestions</td></tr><tr><td>SnapKV</td><td>6.01</td><td>5.99</td><td>6.25</td><td>6.06</td><td>5.76</td></tr><tr><td>LongLLMLingua</td><td>19.81</td><td>22.49</td><td>19.44</td><td>15.07</td><td>13.37</td></tr><tr><td>LLMLingua-2-large</td><td>15.52</td><td>14.39</td><td>12.77</td><td>10.36</td><td>9.60</td></tr><tr><td>Activation Beacon</td><td>51.83</td><td>48.47</td><td>41.43</td><td>11.68</td><td>7.95</td></tr><tr><td>COMI</td><td>70.58</td><td>69.38</td><td>64.22</td><td>57.48</td><td>47.53</td></tr><tr><td colspan="6">2WikiMQA</td></tr><tr><td>SnapKV</td><td>8.63</td><td>8.78</td><td>8.29</td><td>8.09</td><td>1.53</td></tr><tr><td>LongLLMLingua</td><td>29.27</td><td>25.61</td><td>22.42</td><td>21.80</td><td>21.19</td></tr><tr><td>LLMLingua-2-large</td><td>27.72</td><td>20.87</td><td>18.20</td><td>18.26</td><td>18.81</td></tr><tr><td>Activation Beacon</td><td>45.75</td><td>45.29</td><td>45.12</td><td>36.59</td><td>39.00</td></tr><tr><td>COMI</td><td>61.80</td><td>60.73</td><td>57.71</td><td>52.85</td><td>47.56</td></tr></table>

## D CASE STUDY OF COARSE-GRAINED GROUP REALLOCATION

COMI dynamically reallocates compression budgets based on inter-group MIG. For example, for a context of 256 tokens (A truncated sample from 2WikiMQA) with a target 32× compression constraint, the initial group size is 32. MIG-guided reallocation adjusts group sizes to prioritize informative regions. As illustrated in Table 8, Segment 1 (containing key information relevant to query) receives a lower compression rate (final size = 18 vs. initial size = 32), while less informative segments expand. This adaptive behavior ensures information-dense regions are preserved with higher fidelity. 


Table 8: Example of MIG-guided group reallocation (context length = 256, target 32× compression).


<table><tr><td>Segment Index</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>MIG</td><td>0.2227</td><td>0.0078</td><td>-0.1719</td><td>-0.4546</td><td>-0.5583</td><td>-0.3682</td><td>-0.3203</td><td>-0.2832</td></tr><tr><td>Initial size</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>Reallo. size</td><td>18</td><td>22</td><td>26</td><td>35</td><td>39</td><td>32</td><td>31</td><td>30</td></tr></table>

## E SCALABILITY TO ULTRA-LONG CONTEXTS

To evaluate COMI’s scalability beyond the 32K context length used in main experiments, we train and test on NarrativeQA with contexts up to 64K tokens using Qwen3-4B-Instruct as the backbone. As shown in Table 9, COMI maintains strong performance under extreme lengths: at 16× compression, it achieves 22.79, more than double the original prompt (10.69). This demonstrates COMI’s scalability to ultra-long input scenarios. 

## F DATASETS

NaturalQuestions. NaturalQuestions (Liu et al., 2024a) is a large-scale dataset designed to evaluate open-domain question answering systems. It is based on real-world Google search queries and uses Wikipedia articles as its knowledge source. Unlike many other datasets, both the questions and answers in NQ are derived from authentic user behavior rather than being manually authored. Each question is paired with a complete answer, which can be either a short text span from a Wikipedia page (the “short answer”) or a longer text passage (the “long answer”), enabling the simultaneous assessment of a model’s precise extraction ability and its 


Table 9: F1 scores on NarrativeQA with 64K max length using Qwen3-4B-Instruct as backbone.


<table><tr><td rowspan="2">Method</td><td colspan="2">Compression Rate</td></tr><tr><td>16×</td><td>32×</td></tr><tr><td>Original Prompt</td><td>10.69</td><td>10.69</td></tr><tr><td>COMI</td><td>22.79</td><td>20.80</td></tr></table>

comprehension of long documents. The specific version we utilize contains 20 documents in total, with only one being the ground-truth document and the others serving as distractors. 

HotpotQA. HotpotQA (Yang et al., 2018) is a dataset for multi-hop question answering. Its unique characteristic is that each question necessitates finding and synthesizing information from multiple Wikipedia articles. The questions often involve several entities and facts, requiring models to perform cross-document reasoning and linking to construct a coherent and complete answer, rather than simply extracting a single piece of information from one document. 

2WikiMQA. 2WikiMQA (Ho et al., 2020) is another dataset specifically engineered for complex, multi-hop question answering. Similar to HotpotQA, it requires models to reason and integrate information from multiple Wikipedia documents. However, its questions typically involve more complex logic and inference chains, challenging models to not only identify relevant facts but also to understand their relationships (e.g., such as comparing, contrasting, or inferring causality) to generate an accurate response. 

MultiNews. MultiNews (Fabbri et al., 2019) is a dataset for multi-document summarization. It contains a large collection of news clusters, with each cluster composed of multiple articles reporting on the same event. The objective is to teach models how to extract key information from these disparate reports and synthesize it into a concise, coherent, and low redundant summary. This task requires models to identify redundant information, integrate knowledge from multiple sources, and generate fluent text. 

NarrativeQA. NarrativeQA (Kocisky et al., 2018) is a dataset designed to evaluate machine com-´ prehension and summarization capabilities. It includes full-length novels and movie scripts from Project Gutenberg, paired with naturally-occurring, human-generated questions and answers. The queries often require a deep understanding of plot development, character relationships, and event sequences, compelling models to perform sophisticated inference over long-form narratives to produce accurate responses. 

## G LANGUAGE MODEL USAGE STATEMENT

During the preparation of this manuscript, we utilize a large language model as a writing assistant. Its primary role is to refine and polish our paper, including the descriptions of our methodology and the presentation of mathematical derivations. This is done to improve the overall clarity, precision, and readability of the paper. All core ideas, experimental designs, and results are original work of the authors. 