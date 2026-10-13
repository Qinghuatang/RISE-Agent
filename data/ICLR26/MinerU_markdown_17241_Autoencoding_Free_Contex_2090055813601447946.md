# AUTOENCODING-FREE CONTEXT COMPRESSION FOR LLMS VIA CONTEXTUAL SEMANTIC ANCHORS

Xin Liu<sup>1,∗</sup> Runsong Zhao<sup>1,∗</sup> Pengcheng Huang<sup>1</sup> Xinyu Liu<sup>1</sup> Junyi Xiao<sup>1</sup> Chunyang Xiao Tong Xiao<sup>1,2,†</sup> Shengxiang Gao<sup>3</sup> Zhengtao Yu<sup>3</sup> Jingbo Zhu<sup>1,2</sup> <sup>1</sup> School of Computer Science and Engineering, Northeastern University, Shenyang, China 

<sup>2</sup> NiuTrans Research, Shenyang, China 

<sup>3</sup> Kunming University of Science and Technology, Kunming, China 

## ABSTRACT

Context compression is an advanced technique that accelerates large language model (LLM) inference by converting long inputs into compact representations. Existing methods primarily rely on autoencoding tasks to train special compression tokens to represent contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, we remark that such capabilities potentially conflict with actual downstream task requirements, prevent the models from learning the features more beneficial for real-world usage. Based on this observation, we propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability a priori. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. To ensure that anchors can effectively collect information, SAC introduces two key designs: (1) anchor embedding, a learnable embedding vector attached to the selected anchor tokens to mark compression carriers and (2) bidirectional attention modification, which enables anchor tokens to integrate information from the entire context. Experimental results show that SAC consistently outperforms existing context compression methods across different compression ratios and model sizes on question-answering and long-context summarization tasks. Our data, model and code have been released at https://github.com/lx-Meteors/SAC. 

## 1 INTRODUCTION

The expanding scope of large language models (LLMs) to tasks such as processing long documents (Liu et al., 2024b; Li et al., 2024; Duan et al., 2025; Zhao et al., 2026), maintaining multi-turn dialogue coherence (Zhang et al., 2025; Yi et al., 2025; Guan et al., 2025), and generating responses grounded in extensive external knowledge (Lewis et al., 2020; Karpukhin et al., 2020; Huang et al., 2025) necessitates the incorporation of vast contexts into the model input. However, directly processing such extremely long contexts is fraught with challenges, including prohibitive computational costs, significant inference latency, and performance degradation, largely caused by the “lost-in-the-middle” phenomenon (Liu et al., 2024a). 

To address these challenges, recent studies have proposed context compression (Chang et al., 2024; Li et al., 2025a; Lv et al., 2026; Tang et al., 2026a;b), a technique that typically appends special tokens (i.e. compression tokens) to the end of the context and leverages the LLM’s causal attention mechanism to compress contextual information into a compact representation within these tokens. Once this compact representation is obtained, the LLM can generate responses conditioned on it, rather than being conditioned on the entire original context. The reduction in context length leads to substantial decreases in both inference time and GPU memory consumption. While effective, these approaches face a key limitation: the compression tokens are randomly initialized and lack inherent semantic information. To address this limitation, prior works (Ge et al., 2024; Wang et al., 2024; Li et al., 2025b; Zhao et al., 2025; Tang et al., 2025b) typically rely on extensive pretraining on autoencoding (AE) and language modeling (LM) tasks (as shown in Fig. 1) to endow compression tokens with the ability to carry contextual information. However, while the model learns to reconstruct contexts from compression tokens through the AE task, such reconstruction objective can misalign with actual objectives (e.g. QA from compressd tokens) that downstream requires (we analyse such misalignment in subsection 5.1). Thus, this reliance on a suboptimal and costly pretraining stage raises a critical research question: is it possible to design a compression architecture that inherently understands context without a demanding AE phase? 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/5edb4a74df5b136096279258e48e4f97db9cfe72f772bc4caf27e1921fffeb4c.jpg)



Figure 1: Three tasks for training the context compressor introduced by ICAE and followed by numerous works. The training uses (a) Autoencoding task and (b) Language modeling task to pretrain the encoder, then finetunes on (c) Question answering task.


To answer this question, our work introduces Semantic-Anchor Compression (SAC) (Figure 2), a novel architecture for the context compression task. Instead of appending new special tokens and requiring extensive autoencoding pretraining to learn their representations, SAC directly selects representative tokens from the original context to act as “anchor tokens” for compression. By leveraging these semantically meaningful anchors from the input itself, SAC incorporates natural semantic priors that obviate the need for autoencoding pretraining. To mark their special role, these selected tokens are augmented with dedicated “anchor embeddings”, enabling the LLM to distinguish them from regular tokens. Furthermore, to enhance their compression capabilities, we modify the standard causal attention to a bidirectional attention mechanism. This allows anchor tokens to access information from the entire context, rather than being restricted to only preceding tokens. These modifications collectively foster a more effective context compression by providing anchor tokens with both distinct representations and comprehensive contextual awareness. To evaluate the effectiveness of SAC, we conduct experiments on question answering tasks and long-context summarization tasks, where it consistently outperforms strong baselines. Results also show that 1) our proposed method consistently improves over strong baselines across different compress ratios, 2) our proposed architecture achieves its best performance in a simpler training setting without autoencoding training, arguably because the anchor tokens already contain enough information about the original context. Our analysis reveals that SAC’s compressed representations more closely resemble original context token KVs in feature space, which arguably enables LLMs to better understand them during inference. 

## 2 RELATED WORKS

## 2.1 COMPRESSION METHOD

Many methods focus on reducing prompt lengths to improve LLM inference speed. CC (Wingate et al., 2022) utilizes contrastive learning to compress specific natural language prompts into shorter and unique soft prompt tokens. However, it cannot generalize to unseen prompts and requires retraining for new prompts. GIST (Mu et al., 2023) compresses original prompts into KV values through finetuning and can handle arbitrary unseen contexts. AutoCompressor (Chevalier et al., 2023) recursively combines compressed vectors with sub-prompts and aggregates all compressed vectors to construct the final representation, enabling compression of longer contexts. However, both GIST and AutoCompressor require finetuning the LLMs (referred to later as target LLMs) to perform inference, which may affect LLMs’ original capabilities. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/371347cce3971716d762f8d0babee9176b643045538b3acf498931fbb84c93bd.jpg)



Figure 2: The difference between SAC and previous methods. While previous methods (a) compress contextual information into dedicated special tokens (referred to as compression tokens), SAC (b) compresses the context directly into the original contextual tokens themselves. Here, M<sup>˜</sup> can represent either the output from the final layer of the LLM or the key-value pairs across all transformer layers, which are later used as compressed representations for LLM inference.


ICAE (Ge et al., 2024) formulates context compression as training a general encoder that compresses contexts into compact representations understandable by target LLMs without finetuning target LLMs. To train the encoder, ICAE introduces autoencoding tasks and performs joint pretraining with language modeling tasks, followed by finetuning on downstream tasks. 500xCompressor (Li et al., 2025b) improves upon ICAE by replacing the compression carrier from the last layer output of compression tokens with KV values at each layer, achieving higher compression ratios. EPL (Zhao et al., 2025) identifies that ICAE and 500xCompressor neglect the impact of positional encoding and proposes distributing compression token position IDs uniformly across the entire context rather than placing them at the end. However, these methods still rely on autoencoding tasks to endow the compression tokens with the ability to carry contextual information. 

Another category of prompt compression methods is based on token selection, which selects representative tokens from contexts based on token importance. SelectiveContext (Li et al., 2023), LLMLingua (Jiang et al., 2023), and LongLLMLingua (Jiang et al., 2024) employ causal small language models to evaluate token importance based on information entropy. LLMLingua-2 (Pan et al., 2024a) distills a token classifier to compute the probability of each token to be preserved. PerceptionCompressor (Tang et al., 2025a) preserves critical information at the token level while removing tokens that distract the LLM’s attention. These works demonstrate that LLMs can understand original contexts using a small number of representative tokens. However, they do not perform compressed tokens training which limits the usability of the selected tokens by target LLMs. Our proposed SAC can be seen as a combination of token selection methods and compressed token training methods: it derives and train compressed representations that are based on tokens selected directly from the context and indeed is compatible with the token selection methods above. 

## 2.2 BIDIRECTIONAL ATTENTION

Recent studies have shown that, removing the decoder’s unidirectional causal constraint and introducing bidirectional attention can effectively enhance the model’s representational capacity (Wang et al., 2020). For instance, NV-Embed (Lee et al., 2025) replaces causal attention with bidirectional attention during contrastive training, achieving strong performance on general text embedding and dense vector retrieval tasks. LLM2Vec (BehnamGhader et al., 2024), by enabling bidirectional attention alongside masked next-token prediction, significantly improves the model’s ability to capture global semantics in text embedding tasks. These works indicate that bidirectional attention is advantageous for acquiring global semantic information and robust contextual representations. However, its effectiveness in context compression tasks remains underexplored. Motivated by these findings, we incorporate bidirectional attention into the compressor to enhance contextual modeling during the compression phase. 

## 3 METHOD

## 3.1 TASK FORMULATION

Context compression is formally defined as follows: an encoder $\mathcal { E }$ compresses a context $C =$ $( c _ { 1 } , c _ { 2 } , \ldots , c _ { | C | } )$ into a compact representation $\tilde { M }$ with $\tilde { M } = \mathcal { E } ( C )$ . Subsequently, a target LLM leverages the compressed representation $\tilde { M }$ in place of the original context C to perform various tasks, such as question answering. 

To train the encoder E to effectively extract contextual information, ICAE introduces three objective functions. The autoencoding loss $\mathcal { L } _ { \mathbf { A E } }$ ensures that the compressed representation $\tilde { M }$ generated by $\mathcal { E }$ preserves all tokens in the context by learning to reconstruct the entire context C regardless of the token relative importance, as shown in Figure 1a; mathematically, $\mathcal { L } _ { \bf A E } = - \log P ( C | \tilde { M } )$ . The language modeling loss ${ \mathcal { L } } _ { \mathbf { L M } }$ encourages $\tilde { M }$ to maintain predictive capability for future context $C ^ { \prime } = ( \bar { c } _ { | C | + 1 } , c _ { | C | + 2 } , \bar { . } . . , c _ { | C | + | C ^ { \prime } | } )$ , enabling proactive information planning, as shown in Figure 1b; mathematically, $\mathcal { L } _ { \bf L M } = - \log P ( C ^ { \prime } | \tilde { M } )$ . During pretraining, $\mathcal { L } _ { \mathbf { A E } }$ and $\mathcal { L } _ { \bf L M }$ are jointly optimized to obtain an initially effective encoder $\mathcal { E } .$ 

Additionally, during finetuning, the downstream task loss $\mathcal { L } _ { \mathbf { Q A } }$ (taking QA as an exemplar downstream task) enhances the ability of $\tilde { M }$ to extract information that is potentially relevant for downstream tasks. For example, for question answering (QA) tasks, the encoder learns through $\mathcal { L } _ { \mathbf { Q A } }$ to identify and preserve information in the context that is likely to be queried, enabling accurate answer generation $A = \left( a _ { 1 } , a _ { 2 } , \dotsc , a _ { | A | } \right)$ when presented with subsequent questions $Q = ( q _ { 1 } , q _ { 2 } , \dotsc , q _ { | Q | } )$ as shown in Figure 1c; mathematically, $\mathcal { L } _ { \mathbf { Q A } } = - \log P ( A | \tilde { M } , Q )$ 

## 3.2 SEMANTIC-ANCHOR COMPRESSOR

A key distinction between our approach Semantic-Anchor Compression (SAC) and previous methods is that we derive compressed representations directly from selected context tokens, as shown in Figure 2. This involves selecting a subset of tokens S from context $C$ as anchor tokens $S \subseteq C$ . We believe that a good selection strategy benefits SAC. Following EPL, our default strategy divides the entire context $\bar { \boldsymbol { C } }$ into $| S |$ chunks and selects the middle token from each chunk. This setting helps maximize coverage of context C. As illustrated in Figure $^ { 3 \mathrm { a } , }$ each selected token $c _ { i } \in S$ is enhanced with the anchor embedding $e _ { A } ,$ , yielding an embedding sequence $E = ( e _ { 1 } , e _ { 2 } , \dots , e _ { | C | } )$ 

$$
e _ {i} = \mathbf {E m b} (c _ {i}) + \mathbf {1} _ {c _ {i} \in S} \cdot e _ {A}\tag{1}
$$

where $\mathbf { 1 } _ { c _ { i } \in S }$ is an indicator function that equals 1 when $c _ { i } ~ \in ~ S$ and 0 otherwise. Following previous works, we employ a LLM with LoRA parameters $\theta _ { L o R A }$ as the compressor: $\tilde { M } = \mathcal { E } ( C ) =$ $\mathbf { \bar { L } L M } ( E | \theta _ { L o R A } )$ . Overall, using original tokens from the context avoids learning compression tokens from scratch and we empirically validate that such design improves learning efficiency. 

We notice that because the encoder uses causal attention, the anchor tokens S do not have visibility to the full sentence, limiting their representation power. Hence in SAC, we modify the LLM to replace causal attention with bidirectional attention (see Figure 3b), to enhance the LLM’s encoding capability. Notably, this bidirectional attention operates across all tokens, rather than being limited to the anchor tokens, allowing the model to capture richer contextual dependencies. 

M<sup>˜</sup> can be either the output of anchor tokens from the LLM’s final layer or the key-value pairs from each layer. Following 500xCompressor, we use key-value pairs as the compressed representation $\tilde { M }$ For the encoder, we adopt the same language model as the decoder and leverage the original context’s KV cache, enabling the decoder to comprehend the compressed representation more readily without requiring additional semantic alignment.<sup>1</sup> During pretraining, we only use $\mathcal { L } _ { \bf L M }$ and do not use $\mathcal { L } _ { \mathbf { A E } }$ to train the compressor. Following previous work, we use $\mathcal { L } _ { \mathbf { Q A } }$ for finetuning. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/21be1ec20e03bcca9856f437b02d8e79d7664eed69eb6a9d51aa7c8aa6e6a2a6.jpg)



Figure 3: Key differentiators within SAC model architecture. (a) Representative tokens are transformed into anchor tokens through anchor embeddings. (b) The encoder in SAC adopts bidirectional attention, while the decoder operates with causal attention.


## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTING

Dataset. For continued pretraining, we use the corpus SlimPajama-6B (Soboleva et al., 2023). During finetuning and evaluation, we employ the standard MRQA (Fisch et al., 2019) questionanswering dataset, which consolidates multiple QA tasks and standardizes them into a unified format. We evaluate SAC on both test sets, namely in-domain (ID) and out-of-domain (OOD), to comprehensively assess its in-distribution fitting ability and cross-domain generalization performance. 

Implementation Details. SAC utilizes Llama-3.2-1B (Grattafiori & Dubey, 2024) as both the encoder and target LLM. We equip the encoder with trainable LoRA (Hu et al., 2022) adapters (rank = 128, $\alpha = 2 5 6 )$ , while the target LLM parameters remain frozen. For each context, we partition it into sub-contexts of 510 tokens each. The compressor compresses each sub-context into a sub-compressed representation, and subsequently concatenates these sub-compressed representations to form the complete compressed representation. The number of anchor tokens $| S | = \lfloor ^ { \cdot } L / r \rfloor$ is determined by the compression ratio r and the length of the sub-context L. We train all models in two stages: pretraining for 20,000 optimization steps followed by finetuning for an additional 20,000 steps, both conducted with a batch size of 16. Complete hyperparameter configurations are provided in Appendix A. 

Baselines. We use the Llama-3.2-1B model trained on the MRQA (Fisch et al., 2019) dataset as an uncompressed baseline (denoted as “Full-FT"). We compare our method against several context compression techniques: for hard compression, we choose LLMLingua-2 (Pan et al., 2024b) and evaluate its performance on the Full-FT model; for soft compression, we select ICAE (Ge et al., 2024), 500xCompressor (Li et al., 2025b), Activation Beacon (Zhang et al., 2024), DAST (Chen et al., 2025), and EPL (Zhao et al., 2025). To ensure a fair comparison, all these soft compression baselines are trained using the same experimental settings and data as our SAC method. 

## 4.2 FINE-TUNING RESULTS

Tables 1 and 2 report the evaluation results of SAC on in-domain and out-of-domain MRQA datasets. 

Overall Performance. For in domain results, SAC consistently outperforms all baseline models as shown in Table 1. At a 15× compression ratio, compared with the weaker ICAE and the stronger EPL, SAC shows a maximum improvement of 23.5% F1 / 26.8% EM and a minimum improvement of 6.7% F1 / 8.2% EM for in-domain evaluations. A similar trend can be observed for out-of-domain evaluations in Table 2 where the maximum improvement is 27.4% F1 / 28.9% EM, with a minimum improvement of 6.9% F1 / 9.2% EM compared to ICAE and EPL. 


Table 1: For the fine-tuning results, we report in-domain performance on the MRQA datasets using ROUGE-1 F1 (Lin, 2004) and exact match (EM) (Maalouly, 2022).The compression ratio of Lingua 2 (Pan et al., 2024b) is set to 5×.


<table><tr><td rowspan="2">Methods</td><td colspan="2">SQuAD</td><td colspan="2">NewsQA</td><td colspan="2">TriviaQA</td><td colspan="2">SearchQA</td><td colspan="2">HotpotQA</td><td colspan="2">NQ</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>Full-FT</td><td>77.69</td><td>59.71</td><td>63.50</td><td>46.04</td><td>68.80</td><td>60.54</td><td>73.25</td><td>62.07</td><td>74.78</td><td>59.26</td><td>71.01</td><td>53.47</td><td>71.51</td><td>56.85</td></tr><tr><td>Lingua-2</td><td>32.93</td><td>19.57</td><td>26.78</td><td>13.20</td><td>9.67</td><td>8.12</td><td>45.40</td><td>31.80</td><td>36.10</td><td>22.05</td><td>40.08</td><td>22.01</td><td>31.83</td><td>19.46</td></tr><tr><td colspan="15">15x compression constraint</td></tr><tr><td>ICAE</td><td>31.90</td><td>18.91</td><td>25.25</td><td>11.97</td><td>51.78</td><td>42.94</td><td>64.81</td><td>52.89</td><td>45.22</td><td>30.32</td><td>48.01</td><td>30.67</td><td>44.50</td><td>31.28</td></tr><tr><td>500x</td><td>40.68</td><td>24.97</td><td>32.01</td><td>16.76</td><td>53.84</td><td>44.86</td><td>65.65</td><td>53.70</td><td>53.01</td><td>36.30</td><td>50.93</td><td>33.26</td><td>49.35</td><td>34.98</td></tr><tr><td>Beacon</td><td>37.53</td><td>14.31</td><td>31.10</td><td>7.72</td><td>48.85</td><td>4.47</td><td>39.06</td><td>22.67</td><td>45.01</td><td>27.35</td><td>40.29</td><td>15.82</td><td>40.31</td><td>15.39</td></tr><tr><td>DAST</td><td>36.33</td><td>22.65</td><td>31.55</td><td>16.29</td><td>56.92</td><td>48.14</td><td>68.07</td><td>56.51</td><td>54.02</td><td>37.93</td><td>52.10</td><td>35.31</td><td>49.83</td><td>36.14</td></tr><tr><td>EPL</td><td>44.58</td><td>27.91</td><td>33.34</td><td>16.69</td><td>56.16</td><td>47.09</td><td>66.36</td><td>54.13</td><td>54.88</td><td>38.38</td><td>53.80</td><td>35.71</td><td>51.52</td><td>36.65</td></tr><tr><td>SAC</td><td>47.43</td><td>30.25</td><td>36.55</td><td>18.07</td><td>61.13</td><td>52.19</td><td>68.97</td><td>56.76</td><td>58.83</td><td>41.86</td><td>56.79</td><td>38.88</td><td>54.95</td><td>39.67</td></tr></table>


Table 2: For the fine-tuning results, we report out-of-domain performance on the MRQA datasets using ROUGE-1 F1 (Lin, 2004) and exact match (EM) (Maalouly, 2022).The compression ratio of Lingua-2 (Pan et al., 2024b) is set to 5×.


<table><tr><td rowspan="2">Methods</td><td colspan="2">BioASQ</td><td colspan="2">DROP</td><td colspan="2">DouRC</td><td colspan="2">RACE</td><td colspan="2">RE</td><td colspan="2">TQA</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>Full-FT</td><td>49.37</td><td>36.77</td><td>44.67</td><td>34.46</td><td>48.82</td><td>35.51</td><td>35.57</td><td>9.64</td><td>83.34</td><td>72.46</td><td>53.32</td><td>32.40</td><td>52.51</td><td>36.87</td></tr><tr><td>Lingua-2</td><td>27.76</td><td>19.48</td><td>27.28</td><td>18.83</td><td>27.07</td><td>18.32</td><td>17.54</td><td>4.15</td><td>39.30</td><td>20.59</td><td>28.42</td><td>15.83</td><td>27.90</td><td>16.20</td></tr><tr><td colspan="15">15x compression constraint</td></tr><tr><td>ICAE</td><td>35.51</td><td>24.47</td><td>30.39</td><td>21.96</td><td>13.78</td><td>9.06</td><td>15.21</td><td>3.71</td><td>55.24</td><td>40.33</td><td>34.75</td><td>21.56</td><td>30.81</td><td>20.18</td></tr><tr><td>500x</td><td>36.30</td><td>25.93</td><td>33.46</td><td>23.55</td><td>20.53</td><td>12.72</td><td>18.49</td><td>3.41</td><td>54.37</td><td>41.11</td><td>41.09</td><td>25.82</td><td>34.04</td><td>22.09</td></tr><tr><td>Beacon</td><td>36.07</td><td>7.65</td><td>34.39</td><td>14.90</td><td>33.78</td><td>14.99</td><td>26.68</td><td>3.71</td><td>53.28</td><td>23.91</td><td>37.27</td><td>12.97</td><td>36.91</td><td>13.02</td></tr><tr><td>DAST</td><td>36.57</td><td>27.06</td><td>31.90</td><td>22.42</td><td>21.56</td><td>14.26</td><td>16.31</td><td>3.26</td><td>48.54</td><td>35.92</td><td>36.69</td><td>22.75</td><td>31.93</td><td>20.95</td></tr><tr><td>EPL</td><td>40.52</td><td>28.52</td><td>32.16</td><td>22.29</td><td>25.70</td><td>16.39</td><td>20.97</td><td>4.01</td><td>59.75</td><td>46.34</td><td>41.31</td><td>25.42</td><td>36.74</td><td>23.83</td></tr><tr><td>SAC</td><td>41.31</td><td>28.66</td><td>36.72</td><td>27.48</td><td>28.94</td><td>18.99</td><td>23.35</td><td>4.90</td><td>61.04</td><td>47.90</td><td>44.21</td><td>28.21</td><td>39.26</td><td>26.02</td></tr></table>

Comparison with EPL. As shown in Tables 1 and 2, EPL performs strongly overall thanks to its position layout designed for soft compression. Interestingly, SAC can be seen as building upon EPL <sup>2</sup> with enhanced compressed token representations introduced in Section 3. The overall results show that the token representation enhancement is effective as SAC outperforms EPL not only overall but also for each individual datasets in MRQA. 

## 4.3 ABLATION STUDIES

In this subsection, we first show ablation studies on two SAC key components, namely anchor embeddings and bidirectional attentions. Then we study the effect of using different strategies to choose anchor tokens. Finally, we show empirically that our carefully designed SAC performs better without autoencoding training, a property that has motivated the SAC design. 


Table 3: Component ablation results. We report the average F1/EM performance of the model on in-domain (ID) and out-of-domain (OOD) tasks after removing the bidirectional attention (w/o mask) and the anchor embedding (w/o anchor). Full results on all tasks are provided in the Appendix B.3, including Table 16 and Table 17.


<table><tr><td rowspan="3">Methods</td><td colspan="6">In-domain</td><td colspan="6">Out-of-domain</td></tr><tr><td colspan="2">TriviaQA</td><td colspan="2">HotpotQA</td><td colspan="2">Average</td><td colspan="2">BioASQ</td><td colspan="2">TextbookQA</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>SAC</td><td>65.06</td><td>55.93</td><td>67.41</td><td>50.28</td><td>66.24</td><td>53.11</td><td>44.66</td><td>31.45</td><td>52.24</td><td>32.93</td><td>48.45</td><td>32.19</td></tr><tr><td>SAC(w/o mask)</td><td>62.60</td><td>53.27</td><td>64.63</td><td>47.43</td><td>63.62</td><td>50.35</td><td>41.93</td><td>30.65</td><td>48.29</td><td>29.67</td><td>45.11</td><td>30.16</td></tr><tr><td>SAC(w/o anchor)</td><td>63.90</td><td>54.81</td><td>65.25</td><td>48.31</td><td>64.58</td><td>51.56</td><td>43.70</td><td>31.78</td><td>51.59</td><td>32.20</td><td>47.65</td><td>31.99</td></tr></table>

Component Ablation. As shown in Table 3, our ablation study demonstrates the critical roles of the bidirectional attention and anchor embedding. Removing either component results in significant performance degradation in both in-domain (ID) and out-of-domain (OOD) settings. The bidirectional attention mechanism enables anchor tokens to more effectively integrate information from the entire context, producing compressed representations that are more beneficial for downstream tasks. Meanwhile, the anchor embedding provides explicit structural signals that guide the model to accurately identify and process these anchor tokens, thereby ensuring the effectiveness of information compression. 

Token Selection. Table 4 shows the results using different token selection strategies for SAC. The results indicate that random selection (Random) significantly degrades performance arguably for two reasons: the selected tokens lack semantic importance, and their random positions result in poor global coverage of the context, which together hinder effective representation. In contrast, information-based selection (Lingua-2) and our de-


Table 4: Token selection results. Different token selection strategies are compared, including Random selection, Lingua-2-based selection (Pan et al., 2024b), and our uniform selection (Zhao et al., 2025). Average F1/EM scores are reported across in-domain (ID) and out-of-domain (OOD) tasks.


<table><tr><td rowspan="2">Methods</td><td colspan="2">ID-Average</td><td colspan="2">OOD-Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>SAC</td><td>63.63</td><td>46.95</td><td>47.72</td><td>32.30</td></tr><tr><td>SAC(Random)</td><td>59.54</td><td>43.58</td><td>45.35</td><td>30.44</td></tr><tr><td>SAC(Lingua-2)</td><td>63.26</td><td>46.54</td><td>47.43</td><td>32.13</td></tr></table>

fault strategy achieve on par results, and both substantially outperform existing baselines in Tables 1 and 2. This demonstrates that the SAC architecture can effectively leverage and enhance any highquality token selection strategy, rather than relying on a specific choice, highlighting the generality and robustness of the SAC framework. 


Table 5: Ablation study on the effects of autoencoding (AE) and language modeling (LM) objectives.


<table><tr><td rowspan="3">Methods</td><td colspan="6">In-domain Average</td><td colspan="6">Out-of-domain Average</td></tr><tr><td colspan="2">5x</td><td colspan="2">15x</td><td colspan="2">51x</td><td colspan="2">5x</td><td colspan="2">15x</td><td colspan="2">51x</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>SAC</td><td>63.63</td><td>46.95</td><td>54.95</td><td>39.67</td><td>46.37</td><td>33.08</td><td>47.72</td><td>32.30</td><td>39.26</td><td>26.02</td><td>32.24</td><td>21.44</td></tr><tr><td>500x(w/ LM only)</td><td>53.23</td><td>38.70</td><td>49.76</td><td>35.71</td><td>44.46</td><td>31.41</td><td>38.22</td><td>25.73</td><td>33.99</td><td>22.05</td><td>30.09</td><td>18.99</td></tr><tr><td>500x(w/ AE+LM)</td><td>55.26</td><td>40.14</td><td>49.35</td><td>34.98</td><td>43.19</td><td>30.26</td><td>38.46</td><td>25.40</td><td>34.04</td><td>22.09</td><td>30.43</td><td>20.09</td></tr><tr><td>SAC(w/ AE only)</td><td>56.55</td><td>40.34</td><td>49.93</td><td>35.33</td><td>43.95</td><td>30.64</td><td>42.08</td><td>27.98</td><td>35.50</td><td>23.29</td><td>28.77</td><td>18.32</td></tr><tr><td>SAC(w/ AE+LM)</td><td>62.04</td><td>45.80</td><td>51.73</td><td>36.67</td><td>44.69</td><td>31.37</td><td>47.26</td><td>32.25</td><td>37.23</td><td>23.96</td><td>31.01</td><td>19.90</td></tr></table>

AE Effect. Soft compression methods often use AE so that the compressed representations learn to reconstruct the original context. However, AE objective itself can be a limiting factor as the reconstruction target can potentially be misaligned with downstream tasks. SAC is designed with compressed representations directly derived from the contexts to eliminate the need for AE and we show our empirical investigations in Table 5. The experimental results show that training with only the AE objective leads to a substantial performance drop, and even when combined with the LM objective, the performance remains inferior to that of the full SAC model. It is worth noting that while ICAE (Ge et al., 2024) empirically shows that combining AE with LM tasks yields better results, the results do not generalize to other architectures such as 500xCompressor where Table 5 shows a mixed picture for performance. Overall, the results in Table 5 raise questions regarding the necessity of autoencoding tasks and suggest that autoencoding may not be entirely essential for context compression methods. 

## 4.4 SCALABILITY RESULTS

To verify the generalizability of SAC across model scales, we conduct additional experiments on the larger Llama-3.2-3B and Llama-3.1-8B models. We show EPL as our baseline which performs the second best in main experiments shown in Table 1 and 2. As shown in Table 6, SAC outperforms EPL across all model sizes and evaluation settings, consistently achieving higher F1 and EM scores. On the 3B model, SAC achieves an average F1 score of 50.48 and an EM score of 34.73, outperforming the EPL baseline (47.46 / 31.82). On the 8B model, SAC scores 52.31 (F1) and 35.93 (EM), again surpassing the EPL baseline (50.82 / 34.42). We remark that as the model goes larger, the gain does not seem to diminish. These consistent gains demonstrate that SAC effectively generalizes to larger model sizes and maintains its performance advantage. More detailed results on MRQA can be found in Tables 18 and 19. 


Table 6: We compare the performance of SAC and EPL across different model sizes ( Llama-3.2-3B and Llama-3.1-8B ) under a fixed 15× compression ratio. Evaluations are conducted in both in-domain (ID) and out-of-domain (OOD) settings, and we report ROUGE-1 F1 and Exact Match (EM) scores.


<table><tr><td rowspan="3">Methods</td><td colspan="6">In-domain</td><td colspan="6">Out-of-domain</td></tr><tr><td colspan="2">SearchQA</td><td colspan="2">NQ</td><td colspan="2">Average</td><td colspan="2">DROP</td><td colspan="2">RE</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>EPL(3B)</td><td>73.61</td><td>61.56</td><td>63.59</td><td>44.43</td><td>68.60</td><td>53.00</td><td>46.20</td><td>36.46</td><td>65.73</td><td>53.70</td><td>55.97</td><td>45.08</td></tr><tr><td>SAC(3B)</td><td>73.88</td><td>62.29</td><td>65.98</td><td>47.18</td><td>69.93</td><td>54.74</td><td>48.46</td><td>38.39</td><td>75.11</td><td>62.89</td><td>61.76</td><td>50.64</td></tr><tr><td>EPL(8B)</td><td>74.86</td><td>63.23</td><td>67.24</td><td>48.71</td><td>71.05</td><td>55.97</td><td>49.98</td><td>39.92</td><td>69.77</td><td>57.84</td><td>59.88</td><td>48.88</td></tr><tr><td>SAC(8B)</td><td>76.92</td><td>65.29</td><td>67.77</td><td>49.21</td><td>72.35</td><td>57.25</td><td>51.55</td><td>40.65</td><td>78.91</td><td>67.71</td><td>65.23</td><td>54.18</td></tr></table>

## 4.5 SENSITIVITY ANALYSIS FOR COMPRESSION RATIOS

We conduct a detailed evaluation of model performance under different compression ratios (5×, 15×, and 51×), using the same experimental setup as in Table 1. The corresponding results are presented in Tables 7. 

As expected, the F1 and EM scores of all methods decrease as the compression ratio increases from 5× to 51×, since higher compression ratios result in more information being discarded. At the highest compression ratio of 51×, the performance of different compression methods varies across datasets (see more detailed results in Appendix B.2, Tables 11 and 15): some methods perform well on certain datasets but underperform on others. Nonetheless, SAC consis-


Table 7: In-domain and out-of-domain performance under different compression ratios.


<table><tr><td rowspan="2">Methods</td><td colspan="2">In-domain Average</td><td colspan="2">Out-of-domain Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td colspan="5">5x compression</td></tr><tr><td>ICAE</td><td>47.53</td><td>33.82</td><td>31.22</td><td>20.51</td></tr><tr><td>500x</td><td>55.26</td><td>40.14</td><td>38.46</td><td>25.40</td></tr><tr><td>EPL</td><td>62.90</td><td>46.33</td><td>46.95</td><td>31.30</td></tr><tr><td>SAC</td><td>63.63</td><td>46.95</td><td>47.72</td><td>32.30</td></tr><tr><td colspan="5">15x compression</td></tr><tr><td>ICAE</td><td>44.50</td><td>31.28</td><td>30.81</td><td>20.18</td></tr><tr><td>500x</td><td>49.35</td><td>34.98</td><td>34.04</td><td>22.09</td></tr><tr><td>EPL</td><td>51.52</td><td>36.65</td><td>36.74</td><td>23.83</td></tr><tr><td>SAC</td><td>54.95</td><td>39.67</td><td>39.26</td><td>26.02</td></tr><tr><td colspan="5">51x compression</td></tr><tr><td>ICAE</td><td>40.39</td><td>27.99</td><td>27.98</td><td>17.85</td></tr><tr><td>500x</td><td>43.19</td><td>30.26</td><td>30.43</td><td>20.09</td></tr><tr><td>EPL</td><td>43.22</td><td>30.26</td><td>30.22</td><td>19.48</td></tr><tr><td>SAC</td><td>46.37</td><td>33.08</td><td>32.24</td><td>21.44</td></tr></table>


tently achieves the best average performance across all compression ratios, demonstrating strong robustness and stability.


## 4.6 LONG-CONTEXT SUMMARIZATION RESULTS

To further validate the robustness and usefulness of SAC, we conduct additional experiments on two long-context summarization tasks using the QM-Sum (Zhong et al., 2021) and GovReport (Huang et al., 2021) datasets. In these experiments, the models are configured with a maximum input length of 32K tokens and are trained using 


Table 8: We compare the performance of SAC and EPL on longcontext summarization tasks, including QMSum (Zhong et al., 2021) and GovReport (Huang et al., 2021), reporting ROUGE-1 F1 scores.


<table><tr><td>Methods</td><td>QMSum</td><td>GovReport</td><td>Average</td></tr><tr><td>EPL</td><td>14.82</td><td>20.40</td><td>17.61</td></tr><tr><td>SAC</td><td>14.95</td><td>22.03</td><td>18.49</td></tr></table>

datasets with a 15× compression ratio. Both SAC and EPL models are trained and evaluated on the corresponding datasets. 

The results in Table 8 show that SAC maintains strong performance on long-context summarization tasks, surpassing EPL across all datasets. Because EPL is trained with AE objective, the results suggest that removing the AE objective for SAC does not impair the model’s capability to model long contexts. Moreover, SAC is able to capture sufficient global information to support tasks that require comprehensive understanding of the entire context, highlighting its adaptability and generalization performance in long-context generation scenarios. More details are provided in Appendix B.5. 

## 5 ANALYSIS

## 5.1 AE ANALYSIS

We attribute the negative impact of the AE task on SAC primarily to the following two related factors. 

Intuitive Perspective. The AE objective requires the compressed representations to reconstruct the complete context, forcing the model to encode a large number of tokens within its limited capacity; some tokens might require significant model memories while not useful for downstream tasks, thereby weakening model’s ability to retain task-critical information. 

AE-LM Gradient Perspective. In Figure 9, we visualize the gradient cosine similarity between the autoencoding and the language modeling losses during training. The figure shows that while the two gradients largely align at the very beginning, the cosine similarity quickly approaches zero, suggesting that the two tasks are largely orthogonal in parameter space. This orthogonality causes the optimization of autoencoding to interfere with language modeling updates, thereby hindering downstream task performance. 3 

## 5.2 ATTENTION VISUALIZATION

To understand the unique behavior of compressed models, we analyze the attention patterns of the final layer at a 5× compression rate. 

At a lower 5× compression rate, as shown in Figure 6, SAC attention map presents a clear positive diagonal, indicating that its compressed tokens primarily attend to local tokens, adhering to the attention prior similar to EPL. In contrast, the attention map of 500xCompressor appears more diffused which arguably complicates learning and impairs its performance. As the compression rate increases (as shown in Figure 7 and 8), EPL attention pattern starts also to diffuse while SAC exhibits a focused attention pattern, with its anchor tokens attending to only a few key original context tokens. Hypothetically, the focused attention pattern retains important token information to handle downstream tasks and explains the superior performance we empirically observe. 

## 5.3 REPRESENTATION ANALYSIS

Key Representation Analysis. In the key representation space (see Figure 4 up), the compression tokens (orange) of SAC and EPL are distributed relatively close to the context tokens (blue), while the compression tokens of 500xCompressor are clearly separated from the context tokens. This discrepancy arises from the architectural design of each method. EPL modifies the positional IDs of its additional compression tokens to share the same rotational angle (RoPE) as the original context tokens, thereby reducing the distance between them; without such modifications, 500xCompressor compression tokens do not align with context token key representations. In SAC, the anchor tokens are directly embedded within the original context, in consequence their representations maintain close semantic ties with the context tokens. 

Value Representation Analysis. In the value representation space (see Figure 4 bottom), the anchor tokens of SAC are uniformly distributed across all regions with the value representations of the context tokens, without forming independent sub-clusters. This suggests that SAC’s anchor embedding strategy allows for compressed value representations that more closely match the distribution of the original value space. In contrast, although EPL’s compression tokens also overlap with the context tokens, they overlap less than SAC’s: the compression tokens appear relatively sparse in the core regions and show a slight clustering tendency at the boundaries. This indicates that EPL’s value representations still exhibit a degree of semantic shift relative to the original value space, which is even more pronounced in the 500xCompressor. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/a89e7a26f088afc76ed968c89db18ec3d2b793871cea0a83200b64358b184a8e.jpg)



Figure 4: The t-SNE visualization shows the key representations of the final layer KV values for SAC, 500xCompressor (Li et al., 2025b), and EPL (Zhao et al., 2025), respectively.


## 6 CONCLUSION

In this paper, we propose a novel, autoencoding-free context compression method, Semantic-Anchor Compression (SAC). Unlike traditional context compression approaches, SAC does not rely on training compression tokens to reconstruct the original input. Instead, it directly selects representative anchor tokens from the context and aggregates contextual information into their key-value (KV) representations via a bidirectional attention mechanism. This approach effectively compresses lengthy contexts while avoiding any impairment to the language model’s original language modeling capabilities, while traditional context compression can hinder performance by introducing additional compression tokens and the autoencoding tasks, as our analysis shows. Experimental results show that SAC achieves high compression rates while significantly outperforming existing methods across multiple question answering and long-context summarization tasks, demonstrating strong cross-task generalization and an effective balance between compression efficiency and model performance. 

## 7 REPRODUCIBILITY STATEMENT

We declare that the work presented in this paper is reproducible. Our data, model, and code have been released at https://github.com/lx-Meteors/SAC. This code can be used to reproduce the experimental results. The repository includes detailed instructions for environment setup, running experiments, data processing, and result evaluation. 

## 8 ACKNOWLEDGEMENTS

This work was supported in part by the National Science Foundation of China (Nos. 62276056 and U24A20334), the Yunnan Fundamental Research Projects (No.202401BC070021), the Yunnan Science and Technology Major Project (No. 202502AD080014), the Fundamental Research Funds for the Central Universities (Nos. N25BSS054 and N25BSS094), and the Program of Introducing Talents of Discipline to Universities, Plan 111 (No.B16009). 

## REFERENCES



Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. Llm2vec: Large language models are secretly powerful text encoders, 2024. URL https://arxiv.org/abs/2404.05961. 





Kaiyan Chang, Songcheng Xu, Chenglong Wang, Yingfeng Luo, Xiaoqian Liu, Tong Xiao, and Jingbo Zhu. Efficient prompting methods for large language models: A survey, 2024. URL https://arxiv.org/abs/2404.01077. 





Shaoshen Chen, Yangning Li, Zishan Xu, Yinghui Li, Xin Su, Zifei Shan, and Hai tao Zheng. Dast: Context-aware compression in llms via dynamic allocation of soft tokens, 2025. URL https://arxiv.org/abs/2502.11493. 





Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3829–3846, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 232. URL https://aclanthology.org/2023.emnlp-main.232/. 





Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1246. URL https: //aclanthology.org/N19-1246/. 





Shaohua Duan, Xinze Li, Zhenghao Liu, Xiaoyuan Yi, Yukun Yan, Shuo Wang, Yu Gu, Ge Yu, and Maosong Sun. Chunks as arms: Multi-armed bandit-guided sampling for long-context llm preference optimization. arXiv preprint arXiv:2508.13993, 2025. 





Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Güney, Volkan Cirik, and Kyunghyun Cho. Searchqa: A new q&a dataset augmented with context from a search engine. CoRR, abs/1704.05179, 2017. URL http://arxiv.org/abs/1704.05179. 





Adam Fisch, Alon Talmor, Robin Jia, Minjoon Seo, Eunsol Choi, and Danqi Chen. MRQA 2019 shared task: Evaluating generalization in reading comprehension. In Proceedings of 2nd Machine Readingfor Reading Comprehension (MRQA) Workshop at EMNLP, 2019. 





Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model, 2024. URL https://arxiv.org/abs/ 2307.06945. 





Aaron Grattafiori and Abhimanyu Dubey. The llama 3 herd of models, 2024. URL https: //arxiv.org/abs/2407.21783. 





Shengyue Guan, Haoyi Xiong, Jindong Wang, Jiang Bian, Bin Zhu, and Jian guang Lou. Evaluating llm-based agents for multi-turn conversations: A survey, 2025. URL https://arxiv.org/ abs/2503.22458. 





Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9. 





Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization, 2021. URL https://arxiv.org/abs/2104.02112. 





Pengcheng Huang, Zhenghao Liu, Yukun Yan, Xiaoyuan Yi, Hao Chen, Zhiyuan Liu, Maosong Sun, Tong Xiao, Ge Yu, and Chenyan Xiong. Pip-kag: Mitigating knowledge conflicts in knowledgeaugmented generation via parametric pruning. arXiv preprint arXiv:2502.15543, 2025. 





Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. LLMLingua: Compressing prompts for accelerated inference of large language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 13358–13376, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.825. URL https: //aclanthology.org/2023.emnlp-main.825/. 





Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. LongLLMLingua: Accelerating and enhancing LLMs in long context scenarios via prompt compression. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1658–1677, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10. 18653/v1/2024.acl-long.91. URL https://aclanthology.org/2024.acl-long.91/. 





Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and Min-Yen Kan (eds.), Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1601–1611, Vancouver, Canada, July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1147. URL https: //aclanthology.org/P17-1147/. 





Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550/. 





Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017. 





Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Transactions ofthe Association of Computational Linguistics, 2019. 





Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel (eds.), Proceedings ofthe 2017 Conference on Empirical Methods in Natural Language Processing, pp. 785–794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URL https://aclanthology.org/D17-1082/. 





Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Nv-embed: Improved techniques for training llms as generalist embedding models, 2025. URL https://arxiv.org/abs/2405.17428. 





Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. In Roger Levy and Lucia Specia (eds.), Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), pp. 333–342, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/K17-1034. URL https://aclanthology.org/K17-1034/. 





Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546. 





Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. LooGLE: Can long-context language models understand long contexts? In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16304–16333, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.859. URL https://aclanthology.org/2024.acl-long.859/. 





Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. Compressing context to enhance inference efficiency of large language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 6342– 6353, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.emnlp-main.391. URL https://aclanthology.org/2023.emnlp-main.391/. 





Zongqian Li, Yinhong Liu, Yixuan Su, and Nigel Collier. Prompt compression for large language models: A survey. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguis tics: Human Language Technologies (Volume 1: Long Papers), pp. 7182–7195, Albuquerque, New Mexico, April 2025a. Association for Computational Linguistics. ISBN 979-8-89176- 189-6. doi: 10.18653/v1/2025.naacl-long.368. URL https://aclanthology.org/2025. naacl-long.368/. 





Zongqian Li, Yixuan Su, and Nigel Collier. 500xCompressor: Generalized prompt compression for large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 25081–25091, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/ 2025.acl-long.1219. URL https://aclanthology.org/2025.acl-long.1219/. 





Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pp. 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013/. 





Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Associationfor Computational Linguistics, 12:157–173, 2024a. doi: 10.1162/tacl_a_00638. URL https://aclanthology.org/2024.tacl-1.9/. 





Xinyu Liu, Runsong Zhao, Pengcheng Huang, Chunyang Xiao, Bei Li, Jingang Wang, Tong Xiao, and Jingbo Zhu. Forgetting curve: A reliable method for evaluating memorization capability for long-context models. arXiv preprint arXiv:2410.04727, 2024b. 





Kangtao Lv, Jiwei Tang, Langming Liu, Haibin Chen, Weidong Zhang, Shilei Liu, Yongwei Wang, Yujin Yuan, Wenbo Su, and Bo Zheng. Data distribution matters: A data-centric perspective on context compression for large language model, 2026. URL https://arxiv.org/abs/ 2602.01778. 





Nicolas El Maalouly. Exact matching: Algorithms and related problems, 2022. URL https: //arxiv.org/abs/2203.13899. 





Jesse Mu, Xiang Lisa Li, and Noah Goodman. Learning to compress prompts with gist tokens. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=2DtxPCL3T5. 





Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang. LLMLingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association for Computational Linguistics: ACL 2024, pp. 963–981, Bangkok, Thailand, August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.57. URL https://aclanthology.org/ 2024.findings-acl.57/. 





Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang. Llmlingua-2: Data distillation for efficient and faithful task-agnostic prompt compression, 2024b. URL https://arxiv.org/abs/2403.12968. 





Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Jian Su, Kevin Duh, and Xavier Carreras (eds.), Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1 D16-1264. URL https://aclanthology.org/D16-1264/. 





Amrita Saha, Rahul Aralikatte, Mitesh M. Khapra, and Karthik Sankaranarayanan. DuoRC: Towards complex language understanding with paraphrased reading comprehension. In Iryna Gurevych and Yusuke Miyao (eds.), Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1683–1693, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1156. URL https://aclanthology. org/P18-1156/. 





Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/blog/slimpajama, 2023. URL https://huggingface.co/ datasets/cerebras/SlimPajama-627B. 





Jiwei Tang, Jin Xu, Tingwei Lu, Zhicheng Zhang, Yiming Zhao, Lin Hai, and Hai-Tao Zheng. Perception compressor: A training-free prompt compression framework in long context scenarios. In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Findings of the Association for Computational Linguistics: NAACL 2025, pp. 4093–4108, Albuquerque, New Mexico, April 2025a. Association for Computational Linguistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/2025.findings-naacl. 229. URL https://aclanthology.org/2025.findings-naacl.229/. 





Jiwei Tang, Zhicheng Zhang, Shunlong Wu, Jingheng Ye, Lichen Bai, Zitai Wang, Tingwei Lu, Jiaqi Chen, Lin Hai, Hai-Tao Zheng, and Hong-Gee Kim. Gmsa: Enhancing context compression via group merging and layer semantic alignment, 2025b. URL https://arxiv.org/abs/ 2505.12215. 





Jiwei Tang, Shilei Liu, Zhicheng Zhang, Qingsong Lv, Runsong Zhao, Tingwei Lu, Langming Liu, Haibin Chen, Yujin Yuan, Hai-Tao Zheng, Wenbo Su, and Bo Zheng. Read as human: Compressing context via parallelizable close reading and skimming, 2026a. URL https://arxiv.org/ abs/2602.01840. 





Jiwei Tang, Shilei Liu, Zhicheng Zhang, Yujin Yuan, Libin Zheng, Wenbo Su, and Bo Zheng. Comi: Coarse-to-fine context compression via marginal information gain, 2026b. URL https: //arxiv.org/abs/2602.01719. 





Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni, Philip Bachman, and Kaheer Suleman. NewsQA: A machine comprehension dataset. In Phil Blunsom, Antoine Bordes, Kyunghyun Cho, Shay Cohen, Chris Dyer, Edward Grefenstette, Karl Moritz Hermann, Laura Rimell, Jason Weston, and Scott Yih (eds.), Proceedings of the 2nd Workshop on Representation Learning for NLP, pp. 191–200, Vancouver, Canada, August 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-2623. URL https: //aclanthology.org/W17-2623/. 





George Tsatsaronis, Georgios Balikas, Prodromos Malakasiotis, Ioannis Partalas, Matthias Zschunke, Michael R. Alvers, Dirk Weissenborn, Anastasia Krithara, Sergios Petridis, Dimitris Polychronopoulos, Yannis Almirantis, John Pavlopoulos, Nicolas Baskiotis, Patrick Gallinari, Thierry Artières, Axel-Cyrille Ngonga Ngomo, Norman Heino, Eric Gaussier, Liliana Barrio-Alvers, Michael Schroeder, Ion Androutsopoulos, and Georgios Paliouras. An overview of the BIOASQ large-scale biomedical semantic indexing and question answering competition. BMC Bioinformatics, 16(1):138, 2015. ISSN 1471-2105. doi: 10.1186/s12859-015-0564-6. URL https://doi.org/10.1186/s12859-015-0564-6. 





Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems, 2020. URL https://arxiv.org/abs/1905.00537. 





Xiangfeng Wang, Zaiyi Chen, Zheyong Xie, Tong Xu, Yongyi He, and Enhong Chen. In-context former: Lightning-fast compressing context for large language model, 2024. URL https: //arxiv.org/abs/2406.13618. 





David Wingate, Mohammad Shoeybi, and Taylor Sorensen. Prompt compression and contrastive conditioning for controllability and toxicity reduction in language models. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 5621–5634, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp.412. URL https://aclanthology.org/2022.findings-emnlp.412/. 





Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2369–2380, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https://aclanthology.org/D18-1259/. 





Zihao Yi, Jiarui Ouyang, Zhe Xu, Yuwen Liu, Tianhao Liao, Haohao Luo, and Ying Shen. A survey on recent advances in llm-based multi-turn dialogue systems, 2025. URL https://arxiv. org/abs/2402.18013. 





Chen Zhang, Xinyi Dai, Yaxiong Wu, Qu Yang, Yasheng Wang, Ruiming Tang, and Yong Liu. A survey on multi-turn interaction capabilities of large language models, 2025. URL https: //arxiv.org/abs/2501.09959. 





Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. Long context compression with activation beacon, 2024. URL https://arxiv.org/abs/2401.03462. 





Runsong Zhao, Xin Liu, Xinyu Liu, Pengcheng Huang, Chunyang Xiao, Tong Xiao, and Jingbo Zhu. Position ids matter: An enhanced position layout for efficient context compression in large language models, 2025. URL https://arxiv.org/abs/2409.14364. 





Runsong Zhao, Shilei Liu, Jiwei Tang, Langming Liu, Haibin Chen, Weidong Zhang, Yujin Yuan, Tong Xiao, Jingbo Zhu, Wenbo Su, and Bo Zheng. Comet: Collaborative memory transformer for efficient long context modeling, 2026. URL https://arxiv.org/abs/2602.01766. 





Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, and Dragomir Radev. Qmsum: A new benchmark for query-based multi-domain meeting summarization, 2021. URL https://arxiv.org/abs/ 2104.05938. 



## A EXPERIMENT DETAILS

We perform pretraining and fine-tuning using bf16 precision on 8 NVIDIA RTX 3090 GPUs (24GB). For pretraining, we randomly sample data from the SlimPajama-6B dataset with a token length ranging from 510 to 2040. This data is then split into two halves: one for the auto-encoding (AE) task and the other for the language modeling (LM) task (the AE half is discarded for models without the AE objective). For downstream tasks, we process the MRQA dataset into a (Context, Question, Answer) format for finetuning. Detailed hyperparameters can be found in Table 9. 


Table 9: Hyperparameters for training


<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Betas</td><td>(0.9, 0.95)</td></tr><tr><td>Weight decay</td><td>0.1</td></tr><tr><td rowspan="2">Learning rate</td><td>1e-4 (pretrain)</td></tr><tr><td>5e-5 (finetuning)</td></tr><tr><td>Scheduler</td><td>Constant</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Warmup</td><td>300</td></tr><tr><td rowspan="2">Training steps</td><td>20k (pretrain)</td></tr><tr><td>20k (finetuning)</td></tr><tr><td>Clip norm</td><td>2.0</td></tr></table>

## B DETAILED RESULTS

## B.1 PRETRAINING RESULTS

As shown in Table 10, our method, SAC, achieves the lowest perplexity (10.79) among all baseline models. This suggests that removing the autoencoding (AE) objective in SAC allows the model to better focus on the language modeling task, thereby improving its predictive capability. Furthermore, since SAC avoids the additional computational overhead from independent compression tokens and the AE task, its training is approximately 31% faster than ICAE and 26% faster than 500xCompressor and EPL. 


Table 10: Pretraining comparison of SAC and existing context compression methods, results on LM perplexity and training time.


<table><tr><td>Methods</td><td>LM-PPL</td><td>Training Time(hours)</td></tr><tr><td>ICAE</td><td>12.35</td><td>3.85</td></tr><tr><td>500x</td><td>11.83</td><td>3.60</td></tr><tr><td>EPL</td><td>10.88</td><td>3.60</td></tr><tr><td>SAC</td><td>10.79</td><td>2.66</td></tr></table>

## B.2 FINE-TUNING RESULTS

Tables 11 and 15 report the evaluation results of SAC on in-domain and out-of-domain MRQA datasets, which we analyze from three perspectives: overall performance, effect of compression ratio, and domain generalization. 

Overall Performance. SAC consistently outperforms all baselines across a variety of conditions, including compression ratios, and both in-domain and out-of-domain tests, as shown in Tables 11 and 15. Averaging the results of the context compression methods across different compression ratios, SAC shows a maximum improvement of 24.6% F1 / 28.6% EM and a minimum improvement of 4.6% F1 / 5.7% EM in in-domain evaluations. For out-of-domain tests, the maximum improvement is 32.5% F1 / 36.2% EM, with a minimum improvement of 4.6% F1 / 6.9% EM. 

Impact of Compression Ratio. We conduct a detailed evaluation of model performance under different compression ratios (5×, 15×, and 51×), as shown in Tables 11 and 15. As expected, F1 and EM scores of all methods decrease with increasing compression ratio, from 5× to 51×, since higher compression ratios result in more information being discarded. At the highest compression rate of 51×, the performance of different compression methods is not consistent. While one method may perform well on certain datasets, it may underperform on others. Nonetheless, SAC consistently achieves the best average performance. 

Cross-Domain Generalization. We evaluate the generalization capability of SAC on out-of-domain datasets, as shown in Table 15. Under all compression ratio constraints, SAC consistently achieves the highest average F1/EM scores among all methods. Specifically, at a 5× compression ratio, SAC attains average F1 and EM scores of 47.72 and 32.30, outperforming the second-best EPL method by 0.77 and 1.0 points, respectively. At a more challenging 15× compression ratio, SAC achieves average F1 and EM scores of 39.26 and 26.02, surpassing EPL by 2.52 and 2.19 points, with an EM improvement approaching 10%. Even at an extreme 51× compression ratio, SAC maintains average F1 and EM scores of 32.24 and 21.44, still leading EPL by 2.02 and 1.96 points, respectively. These results indicate that the compressed representations learned by SAC exhibit strong cross-domain robustness. 


Table 11: For the finetuning results, we report in-domain performance using ROUGE-1 F1 (Lin, 2004) and exact match (EM) (Maalouly, 2022) scores on the following datasets: SQuAD (Rajpurkar et al., 2016), NewsQA (Trischler et al., 2017), TriviaQA (Joshi et al., 2017), SearchQA (Dunn et al., 2017), HotpotQA (Yang et al., 2018), and NaturalQuestions (NQ) (Kwiatkowski et al., 2019).


<table><tr><td rowspan="2">Methods</td><td colspan="2">SQuAD</td><td colspan="2">NewsQA</td><td colspan="2">TriviaQA</td><td colspan="2">SearchQA</td><td colspan="2">HotpotQA</td><td colspan="2">NQ</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>Full-FT</td><td>77.69</td><td>59.71</td><td>63.50</td><td>46.04</td><td>68.80</td><td>60.54</td><td>73.25</td><td>62.07</td><td>74.78</td><td>59.26</td><td>71.01</td><td>53.47</td><td>71.51</td><td>56.85</td></tr><tr><td>Lingua-2</td><td>32.93</td><td>19.57</td><td>26.78</td><td>13.20</td><td>9.67</td><td>8.12</td><td>45.40</td><td>31.80</td><td>36.10</td><td>22.05</td><td>40.08</td><td>22.01</td><td>31.83</td><td>19.46</td></tr><tr><td colspan="15">5x compression constraint</td></tr><tr><td>ICAE</td><td>36.20</td><td>22.12</td><td>28.06</td><td>13.77</td><td>54.63</td><td>45.59</td><td>65.12</td><td>53.06</td><td>48.79</td><td>33.40</td><td>52.36</td><td>34.99</td><td>47.53</td><td>33.82</td></tr><tr><td>500x</td><td>51.62</td><td>33.63</td><td>39.70</td><td>22.63</td><td>57.62</td><td>48.76</td><td>66.43</td><td>54.38</td><td>59.10</td><td>42.20</td><td>57.11</td><td>39.26</td><td>55.26</td><td>40.14</td></tr><tr><td>EPL</td><td>64.72</td><td>44.28</td><td>48.74</td><td>27.45</td><td>63.75</td><td>54.54</td><td>69.69</td><td>57.73</td><td>67.16</td><td>49.79</td><td>63.32</td><td>44.16</td><td>62.90</td><td>46.33</td></tr><tr><td>SAC</td><td>65.37</td><td>44.83</td><td>49.39</td><td>27.14</td><td>65.06</td><td>55.93</td><td>69.99</td><td>58.06</td><td>67.41</td><td>50.28</td><td>64.56</td><td>45.44</td><td>63.63</td><td>46.95</td></tr><tr><td colspan="15">15x compression constraint</td></tr><tr><td>ICAE</td><td>31.90</td><td>18.91</td><td>25.25</td><td>11.97</td><td>51.78</td><td>42.94</td><td>64.81</td><td>52.89</td><td>45.22</td><td>30.32</td><td>48.01</td><td>30.67</td><td>44.50</td><td>31.28</td></tr><tr><td>500x</td><td>40.68</td><td>24.97</td><td>32.01</td><td>16.76</td><td>53.84</td><td>44.86</td><td>65.65</td><td>53.70</td><td>53.01</td><td>36.30</td><td>50.93</td><td>33.26</td><td>49.35</td><td>34.98</td></tr><tr><td>EPL</td><td>44.58</td><td>27.91</td><td>33.34</td><td>16.69</td><td>56.16</td><td>47.09</td><td>66.36</td><td>54.13</td><td>54.88</td><td>38.38</td><td>53.80</td><td>35.71</td><td>51.52</td><td>36.65</td></tr><tr><td>SAC</td><td>47.43</td><td>30.25</td><td>36.55</td><td>18.07</td><td>61.13</td><td>52.19</td><td>68.97</td><td>56.76</td><td>58.83</td><td>41.86</td><td>56.79</td><td>38.88</td><td>54.95</td><td>39.67</td></tr><tr><td colspan="15">51x compression constraint</td></tr><tr><td>ICAE</td><td>26.17</td><td>14.58</td><td>22.48</td><td>9.69</td><td>47.62</td><td>39.23</td><td>64.31</td><td>52.80</td><td>38.91</td><td>24.78</td><td>42.87</td><td>26.86</td><td>40.39</td><td>27.99</td></tr><tr><td>500x</td><td>30.09</td><td>17.11</td><td>25.06</td><td>12.20</td><td>50.84</td><td>42.13</td><td>64.92</td><td>53.29</td><td>42.15</td><td>27.32</td><td>46.07</td><td>29.53</td><td>43.19</td><td>30.26</td></tr><tr><td>EPL</td><td>30.09</td><td>17.49</td><td>24.49</td><td>11.54</td><td>51.15</td><td>42.38</td><td>65.12</td><td>53.16</td><td>42.19</td><td>27.23</td><td>46.29</td><td>29.77</td><td>43.22</td><td>30.26</td></tr><tr><td>SAC</td><td>31.81</td><td>18.78</td><td>27.36</td><td>13.56</td><td>56.73</td><td>47.85</td><td>65.82</td><td>53.76</td><td>48.28</td><td>32.84</td><td>48.22</td><td>31.70</td><td>46.37</td><td>33.08</td></tr></table>

## B.3 ABLATION RESULTS

In the main text, we discuss the significant performance gains of SAC over all baseline methods. To provide more detailed evidence, we present the full ablation study results here. As shown in Table 16 and Table 17, our conclusion holds not only in terms of average performance but is also consistently validated on each individual dataset. 

## B.4 SCALABILITY RESULTS

We use the same data and training settings as described in the paper, with a compression ratio of 15. The training is conducted on 8 A100 GPUs for Llama-3.2-3B and Llama-3.1-8B. 

The results are shown in Tables 18 and 19. Compared with the stronger baseline EPL, SAC achieved the best average performance in all cases, demonstrating its ability to effectively scale up to larger model sizes while maintaining its performance advantage. 

## B.5 LONG-CONTEXT SUMMARIZATION RESULTS

To further investigate whether this degradation phenomenon is specific to QA or reflects a broader conflict between AE and downstream tasks, we additionally evaluate SAC without AE and SAC with AE on other long-context tasks. The evaluation metric used is ROUGE-F1. 

For the summarization task, we train SAC and EPL on QMSum (Zhong et al., 2021) and GovReport (Huang et al., 2021) with a maximum input length of 32K tokens. The test results are shown in Table 20. The results indicate that SAC achieves the highest average performance on long-document summarization tasks. Notably, the training data size of QMSum (1.26k) is much smaller than that of GovReport (17.5k), which may explain why SAC(ae+lm) performs slightly better on QMSum, while SAC performs better on the larger GovReport dataset. We hypothesize that as redundant information increases in long contexts and the information capacity of compressed tokens is limited, the AE objective of reconstructing all information may impose additional burden on the model, thereby negatively affecting performance. 

We further evaluate on a long-context question answering benchmark with a context length of 24K tokens, using the model checkpoints from Table 1, which are trained with a 15× compression ratio on 2K-token contexts. The results are shown in Table 21. On both single-document and multi-document question answering tasks, SAC consistently outperforms the best baseline compression methods under a 15× compression ratio. These results indicate that our method can maintain its performance advantages even when applied to longer contexts. 

## B.6 FLOPS RESULTS

We want to clarify a critical advantage of SAC: while SAC does enable bidirectional attention for anchor tokens, it does not require appending additional k special tokens to the sequence like 500xCompressor. This means SAC operates on shorter sequences during inference, directly reducing computational overhead. 

Let’s quantify this advantage. Given a context with shape $[ b , s , h ]$ where b is batch size, s is sequence length, h is hidden size, and I is the FFN intermediate size, we compare the theoretical FLOPs: 


Table 12: Comparison of FLOPs between SAC and 500xCompressor for different modules.


<table><tr><td>Modules</td><td>SAC-FLOPs</td><td>500x-FLOPs</td></tr><tr><td><eq>\mathbf{x}(\mathbf{W}_{\mathbf{Q}}/\mathbf{W}_{\mathbf{K}}/\mathbf{W}_{\mathbf{V}})</eq></td><td><eq>3 \cdot 2bsh^{2}</eq></td><td><eq>3 \cdot 2b(s+k)h^{2}</eq></td></tr><tr><td><eq>\mathbf{QK}^{\mathrm{T}}</eq></td><td><eq>2bs^{2}h</eq></td><td><eq>b(s+k)^{2}h</eq></td></tr><tr><td>AV</td><td><eq>2bs^{2}h</eq></td><td><eq>b(s+k)^{2}h</eq></td></tr><tr><td><eq>\mathbf{xW}_{\mathbf{O}}</eq></td><td><eq>2bsh^{2}</eq></td><td><eq>2b(s+k)h^{2}</eq></td></tr><tr><td><eq>\mathbf{X}_{\text{out}}\mathbf{W}_{\text{up}}</eq></td><td><eq>2bshI</eq></td><td><eq>2b(s+k)hI</eq></td></tr><tr><td><eq>\mathbf{X}_{\text{out}}\mathbf{W}_{\text{gate}}</eq></td><td><eq>2bshI</eq></td><td><eq>2b(s+k)hI</eq></td></tr><tr><td><eq>\mathbf{X}_{\text{out}}\mathbf{W}_{\text{down}}</eq></td><td><eq>2bshI</eq></td><td><eq>2b(s+k)hI</eq></td></tr><tr><td>sum</td><td><eq>bhs(8h+4s+6I)</eq></td><td><eq>bh(s+k)[8h+2(s+k)+6I]</eq></td></tr></table>


Concrete comparison: With typical settings (b=1, s=510, h=2048, I=8192): 



• At 5× compression(k=102): 500xCompressor requires 1.19× more FLOPs than SAC 



• At 10× compression(k=51): 500xCompressor requires 1.08× more FLOPs than SAC 


To validate the theoretical analysis, we additionally measured empirical inference latency. Using a single GPU with batch size 10 over 1000 sequences, we recorded the compression time for both methods. The results are reported in Table 13. 


Table 13: Empirical compression time comparison (ms per batch) between SAC and 500xCompressor.


<table><tr><td>Methods</td><td>Compression Time</td></tr><tr><td>500x</td><td>257.43</td></tr><tr><td>SAC</td><td>243.87</td></tr></table>

These results show that SAC not only reduces theoretical FLOPs but also achieves lower empirical compression latency. Despite the use of bidirectional attention, SAC is more efficient than 500xCom pressor, as the computational cost of bidirectional attention is offset by operating on the original, shorter sequence. 

## C VISUALIZATION ANALYSIS

## C.1 TRAINING CURVES ANALYSIS

Figure 5 shows the training loss curves at different compression ratios on the MRQA dataset. The training loss of our SAC model consistently converges better than other baseline methods across all compression ratios, which demonstrates that the compressed representations obtained from the SAC architecture are more beneficial for language modeling tasks. Notably, as the compression ratio increases appropriately, the difference in convergence between SAC and the other baselines becomes more significant. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/6c1af9f94ca023ad8766ae61100a19493418d746fce6b57e8f6c777db471c70b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/822122b8259b8c39591a2f20476a2d90d0ea3330e04d5fb831a808bb6b52bdb6.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/8e43ef94d8bf8910c927ef38c0319b8a534c7b370dfe07ac35b0bdd37413a738.jpg)



Figure 5: Supervised finetuning loss curves. The figure illustrates the training loss trajectories of different models under three compression ratios: 5×, 15×, and 51×.


## D THE USE OF LARGE LANGUAGE MODELS

We used a large language model (LLM) as a general-purpose assist tool. The LLM’s primary role was in assisting with writing and text editing, such as refining prose and correcting grammar and spelling to ensure the paper’s professionalism and fluency. We explicitly state that the LLM was not involved in the core ideation or methodological design of this research. All core contributions of the paper, including the proposal of the methodology, the construction and execution of experiments, and the analysis of results, were performed independently by the authors. 


Table 14: Analysis of autoencoding (AE) reconstruction and language modeling performance under different training objectives.


<table><tr><td>Methods</td><td>AE-PPL</td><td>LM-PPL</td><td>AE-BLEU4</td></tr><tr><td>ICAE</td><td>4.08</td><td>12.35</td><td>12.02</td></tr><tr><td>500xCompress</td><td>1.09</td><td>11.83</td><td>87.21</td></tr><tr><td>EPL</td><td>1.03</td><td>10.88</td><td>97.50</td></tr><tr><td>SAC(ae+lm)</td><td>1.03</td><td>11.01</td><td>98.19</td></tr><tr><td>SAC(only ae)</td><td>1.00</td><td>-</td><td>99.94</td></tr><tr><td>SAC</td><td>-</td><td>10.79</td><td>-</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/15a62c29c687ee7ee61f333a620337efb27c1c65ed581403ed94a4a7fe2b5c88.jpg)



Figure 6: Attention maps of different models finetuned under a 5× compression rate. From top to bottom, the figure displays the final layer attention maps for the 500xCompressor, EPL, and SAC models, respectively. The x-axis represents the original context tokens, and the y-axis represents the compression tokens.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/5e1e7f39d56a73d3450b487b90c98f335e2ecb5a800ca545045b1736dfc33f5d.jpg)



Figure 7: Attention maps of different models finetuned under a 15× compression rate. From top to bottom, the figure displays the final layer attention maps for the 500xCompressor, EPL, and SAC models, respectively. The x-axis represents the original context tokens, and the y-axis represents the compression/anchor tokens.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/ddbb5fe3d7e35ea72171b39ff49a8107f9c0fc7997a326fc0e21cf5d1b07055a.jpg)



Figure 8: Attention maps of different models finetuned under a 51× compression rate. From top to bottom, the figure displays the final layer attention maps for the 500xCompressor, EPL, and SAC models, respectively. The x-axis represents the original context tokens, and the y-axis represents the compression tokens.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/f090cfde34b2bad638b588791e7ed9ea184752f591c1f0fd920a244bf37ebb83.jpg)



Figure 9: Gradient Cosine Similarity between AutoEncoding (AE) Loss and Language Modeling (LM) Loss.



Table 15: For the finetuning results, we report out-of-domain performance using ROUGE-1 F1 and exact match (EM) scores on the following datasets: BioASQ (Tsatsaronis et al., 2015), DROP (Dua et al., 2019), DuoRC (Saha et al., 2018), RACE (Lai et al., 2017), Relation Extraction (RE) (Levy et al., 2017), and TextbookQA (TQA) (Kembhavi et al., 2017).


<table><tr><td rowspan="2">Methods</td><td colspan="2">BioASQ</td><td colspan="2">DROP</td><td colspan="2">DouRC</td><td colspan="2">RACE</td><td colspan="2">RE</td><td colspan="2">TQA</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>Full-FT</td><td>49.37</td><td>36.77</td><td>44.67</td><td>34.46</td><td>48.82</td><td>35.51</td><td>35.57</td><td>9.64</td><td>83.34</td><td>72.46</td><td>53.32</td><td>32.4</td><td>52.51</td><td>36.87</td></tr><tr><td>Lingua-2</td><td>27.76</td><td>19.48</td><td>27.28</td><td>18.83</td><td>27.07</td><td>18.32</td><td>17.54</td><td>4.15</td><td>39.30</td><td>20.59</td><td>28.42</td><td>15.83</td><td>27.90</td><td>16.20</td></tr><tr><td colspan="15">5x compression constraint</td></tr><tr><td>ICAE</td><td>36.08</td><td>26.06</td><td>28.95</td><td>21.09</td><td>16.67</td><td>10.79</td><td>15.65</td><td>3.12</td><td>54.73</td><td>41.01</td><td>35.24</td><td>20.96</td><td>31.22</td><td>20.51</td></tr><tr><td>500x</td><td>40.30</td><td>28.99</td><td>35.40</td><td>25.55</td><td>29.43</td><td>19.32</td><td>21.57</td><td>4.90</td><td>65.43</td><td>50.88</td><td>38.62</td><td>22.75</td><td>38.46</td><td>25.40</td></tr><tr><td>EPL</td><td>46.05</td><td>32.58</td><td>39.94</td><td>28.94</td><td>39.10</td><td>27.12</td><td>30.99</td><td>6.08</td><td>76.07</td><td>62.31</td><td>49.54</td><td>30.74</td><td>46.95</td><td>31.30</td></tr><tr><td>SAC</td><td>44.66</td><td>31.45</td><td>41.55</td><td>30.87</td><td>39.48</td><td>26.92</td><td>30.53</td><td>6.23</td><td>77.87</td><td>65.40</td><td>52.24</td><td>32.93</td><td>47.72</td><td>32.30</td></tr><tr><td colspan="15">15x compression constraint</td></tr><tr><td>ICAE</td><td>35.51</td><td>24.47</td><td>30.39</td><td>21.96</td><td>13.78</td><td>9.06</td><td>15.21</td><td>3.71</td><td>55.24</td><td>40.33</td><td>34.75</td><td>21.56</td><td>30.81</td><td>20.18</td></tr><tr><td>500x</td><td>36.30</td><td>25.93</td><td>33.46</td><td>23.55</td><td>20.53</td><td>12.72</td><td>18.49</td><td>3.41</td><td>54.37</td><td>41.11</td><td>41.09</td><td>25.82</td><td>34.04</td><td>22.09</td></tr><tr><td>EPL</td><td>40.52</td><td>28.52</td><td>32.16</td><td>22.29</td><td>25.70</td><td>16.39</td><td>20.97</td><td>4.01</td><td>59.75</td><td>46.34</td><td>41.31</td><td>25.42</td><td>36.74</td><td>23.83</td></tr><tr><td>SAC</td><td>41.31</td><td>28.66</td><td>36.72</td><td>27.48</td><td>28.94</td><td>18.99</td><td>23.35</td><td>4.90</td><td>61.04</td><td>47.90</td><td>44.21</td><td>28.21</td><td>39.26</td><td>26.02</td></tr><tr><td colspan="15">51x compression constraint</td></tr><tr><td>ICAE</td><td>33.82</td><td>23.67</td><td>27.94</td><td>19.29</td><td>11.14</td><td>6.86</td><td>14.89</td><td>3.41</td><td>47.02</td><td>34.02</td><td>33.08</td><td>19.83</td><td>27.98</td><td>17.85</td></tr><tr><td>500x</td><td>32.17</td><td>23.07</td><td>30.11</td><td>21.76</td><td>13.42</td><td>8.53</td><td>15.18</td><td>2.67</td><td>54.62</td><td>41.86</td><td>37.10</td><td>22.62</td><td>30.43</td><td>20.09</td></tr><tr><td>EPL</td><td>32.52</td><td>22.21</td><td>29.64</td><td>20.89</td><td>13.16</td><td>8.13</td><td>17.15</td><td>3.12</td><td>53.72</td><td>40.37</td><td>35.15</td><td>22.16</td><td>30.22</td><td>19.48</td></tr><tr><td>SAC</td><td>36.95</td><td>26.86</td><td>29.52</td><td>20.89</td><td>21.85</td><td>14.26</td><td>15.87</td><td>4.00</td><td>48.19</td><td>36.43</td><td>41.05</td><td>26.21</td><td>32.24</td><td>21.44</td></tr></table>


Table 16: We conduct ablation studies for SAC under a 5× compression rate on the in-domain dataset in three sets: component ablation, token selection, and the influence of the autoencoding (AE) task.


<table><tr><td rowspan="2">Methods</td><td colspan="2">SQuAD</td><td colspan="2">NewsQA</td><td colspan="2">TriviaQA</td><td colspan="2">SearchQA</td><td colspan="2">HotpotQA</td><td colspan="2">NQ</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td colspan="15">Component Ablation</td></tr><tr><td>SAC</td><td>65.37</td><td>44.83</td><td>49.39</td><td>27.14</td><td>65.06</td><td>55.93</td><td>69.99</td><td>58.06</td><td>67.41</td><td>50.28</td><td>64.56</td><td>45.44</td><td>63.63</td><td>46.95</td></tr><tr><td>SAC(w/o mask)</td><td>60.21</td><td>39.93</td><td>45.93</td><td>25.74</td><td>62.60</td><td>53.27</td><td>66.66</td><td>54.82</td><td>64.63</td><td>47.43</td><td>61.53</td><td>42.55</td><td>60.26</td><td>43.96</td></tr><tr><td>SAC(w/o anchor)</td><td>61.69</td><td>41.72</td><td>46.52</td><td>25.45</td><td>63.90</td><td>54.81</td><td>68.03</td><td>56.17</td><td>65.25</td><td>48.31</td><td>62.21</td><td>43.88</td><td>61.27</td><td>45.06</td></tr><tr><td colspan="15">Token Selection</td></tr><tr><td>SAC(Random)</td><td>58.16</td><td>39.00</td><td>42.94</td><td>22.98</td><td>62.37</td><td>53.17</td><td>69.13</td><td>57.36</td><td>63.86</td><td>46.70</td><td>60.76</td><td>42.26</td><td>59.54</td><td>43.58</td></tr><tr><td>SAC(Lingua-2)</td><td>64.89</td><td>44.28</td><td>48.92</td><td>27.11</td><td>64.55</td><td>55.13</td><td>69.89</td><td>58.04</td><td>67.05</td><td>49.74</td><td>64.23</td><td>44.93</td><td>63.26</td><td>46.54</td></tr><tr><td colspan="15">AE Effect</td></tr><tr><td>500x(w/ LM only)</td><td>44.71</td><td>28.89</td><td>37.24</td><td>20.39</td><td>58.97</td><td>50.19</td><td>65.67</td><td>53.74</td><td>56.74</td><td>40.52</td><td>56.07</td><td>38.45</td><td>53.23</td><td>38.70</td></tr><tr><td>500x(w/ AE+LM)</td><td>51.62</td><td>33.63</td><td>39.70</td><td>22.63</td><td>57.62</td><td>48.76</td><td>66.43</td><td>54.38</td><td>59.10</td><td>42.20</td><td>57.11</td><td>39.26</td><td>55.26</td><td>40.14</td></tr><tr><td>SAC(w/ AE only)</td><td>56.98</td><td>37.60</td><td>41.09</td><td>20.61</td><td>58.19</td><td>49.08</td><td>64.02</td><td>51.65</td><td>61.58</td><td>44.13</td><td>57.23</td><td>38.98</td><td>56.55</td><td>40.34</td></tr><tr><td>SAC(w/ AE+LM)</td><td>64.68</td><td>44.62</td><td>46.64</td><td>25.62</td><td>63.34</td><td>54.27</td><td>68.40</td><td>56.48</td><td>66.61</td><td>49.72</td><td>62.56</td><td>44.06</td><td>62.04</td><td>45.80</td></tr></table>


Table 17: We conduct ablation studies for SAC under a 5× compression rate on the out-of-domain dataset in three sets: component ablation, token selection, and the influence of the autoencoding (AE) task.


<table><tr><td rowspan="2">Methods</td><td colspan="2">BioASQ</td><td colspan="2">DROP</td><td colspan="2">DouRC</td><td colspan="2">RACE</td><td colspan="2">RE</td><td colspan="2">TQA</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td colspan="15">Component Ablation</td></tr><tr><td>SAC</td><td>44.66</td><td>31.45</td><td>41.55</td><td>30.87</td><td>39.48</td><td>26.92</td><td>30.53</td><td>6.23</td><td>77.87</td><td>65.40</td><td>52.24</td><td>32.93</td><td>47.72</td><td>32.30</td></tr><tr><td>SAC(w/o mask)</td><td>41.93</td><td>30.65</td><td>40.24</td><td>28.48</td><td>36.48</td><td>23.58</td><td>28.21</td><td>5.49</td><td>69.09</td><td>55.63</td><td>48.29</td><td>29.67</td><td>44.04</td><td>28.92</td></tr><tr><td>SAC(w/o anchor)</td><td>43.70</td><td>31.78</td><td>40.55</td><td>30.34</td><td>36.97</td><td>25.58</td><td>30.05</td><td>6.82</td><td>75.88</td><td>62.35</td><td>51.59</td><td>32.20</td><td>46.46</td><td>31.51</td></tr><tr><td colspan="15">Token Selection</td></tr><tr><td>SAC(Random)</td><td>45.94</td><td>34.31</td><td>40.87</td><td>30.21</td><td>35.35</td><td>24.12</td><td>29.28</td><td>5.64</td><td>70.11</td><td>56.34</td><td>50.53</td><td>32.00</td><td>45.35</td><td>30.44</td></tr><tr><td>SAC(Lingua-2)</td><td>44.49</td><td>31.91</td><td>41.50</td><td>29.61</td><td>39.47</td><td>26.58</td><td>29.96</td><td>7.12</td><td>77.67</td><td>65.47</td><td>51.46</td><td>32.07</td><td>47.43</td><td>32.13</td></tr><tr><td colspan="15">AE Effect</td></tr><tr><td>500x(w/ LM only)</td><td>43.54</td><td>33.11</td><td>35.40</td><td>25.82</td><td>27.71</td><td>17.59</td><td>19.73</td><td>3.86</td><td>62.31</td><td>48.27</td><td>40.60</td><td>25.75</td><td>38.22</td><td>25.73</td></tr><tr><td>500x(w/ AE+LM)</td><td>40.30</td><td>28.99</td><td>35.40</td><td>25.55</td><td>29.43</td><td>19.32</td><td>21.57</td><td>4.90</td><td>65.43</td><td>50.88</td><td>38.62</td><td>22.75</td><td>38.46</td><td>25.40</td></tr><tr><td>SAC(w/ AE only)</td><td>40.85</td><td>29.39</td><td>35.32</td><td>25.28</td><td>31.55</td><td>21.32</td><td>25.86</td><td>4.90</td><td>72.29</td><td>57.90</td><td>46.61</td><td>29.08</td><td>42.08</td><td>27.98</td></tr><tr><td>SAC(w/ AE+LM)</td><td>44.84</td><td>32.31</td><td>41.47</td><td>31.14</td><td>39.29</td><td>27.58</td><td>30.11</td><td>6.23</td><td>77.12</td><td>64.42</td><td>50.74</td><td>31.87</td><td>47.26</td><td>32.26</td></tr></table>


Table 18: Experimental Results of the 1B/3B/8B Models on the In-Domain QA Tasks.


<table><tr><td rowspan="2">Methods</td><td colspan="2">SQuAD</td><td colspan="2">NewsQA</td><td colspan="2">TriviaQA</td><td colspan="2">SearchQA</td><td colspan="2">HotpotQA</td><td colspan="2">NQ</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>EPL(1B)</td><td>44.58</td><td>27.91</td><td>33.34</td><td>16.69</td><td>56.16</td><td>47.09</td><td>66.36</td><td>54.13</td><td>54.88</td><td>38.38</td><td>53.80</td><td>35.71</td><td>51.52</td><td>36.65</td></tr><tr><td>SAC(1B)</td><td>47.43</td><td>30.25</td><td>36.55</td><td>18.07</td><td>61.13</td><td>52.19</td><td>68.97</td><td>56.76</td><td>58.83</td><td>41.86</td><td>56.79</td><td>38.88</td><td>54.95</td><td>39.67</td></tr><tr><td>EPL(3B)</td><td>62.43</td><td>42.65</td><td>46.72</td><td>26.14</td><td>68.83</td><td>59.92</td><td>73.61</td><td>61.56</td><td>68.50</td><td>51.03</td><td>63.59</td><td>44.43</td><td>63.95</td><td>47.62</td></tr><tr><td>SAC(3B)</td><td>63.11</td><td>43.51</td><td>49.23</td><td>28.80</td><td>69.65</td><td>60.85</td><td>73.88</td><td>62.29</td><td>68.87</td><td>51.40</td><td>65.98</td><td>47.18</td><td>65.12</td><td>49.01</td></tr><tr><td>EPL(8B)</td><td>65.79</td><td>45.55</td><td>51.18</td><td>28.96</td><td>72.40</td><td>63.49</td><td>74.86</td><td>63.23</td><td>70.98</td><td>53.65</td><td>67.24</td><td>48.71</td><td>67.08</td><td>50.60</td></tr><tr><td>SAC(8B)</td><td>64.92</td><td>44.80</td><td>49.77</td><td>29.20</td><td>72.37</td><td>63.70</td><td>76.92</td><td>65.29</td><td>70.80</td><td>54.25</td><td>67.77</td><td>49.21</td><td>67.09</td><td>51.08</td></tr></table>


Table 19: Experimental Results of the 1B/3B/8B Models on the Out-of-Domain QA Tasks.


<table><tr><td rowspan="2">Methods</td><td colspan="2">BioASQ</td><td colspan="2">DROP</td><td colspan="2">DouRC</td><td colspan="2">RACE</td><td colspan="2">RE</td><td colspan="2">TQA</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td></tr><tr><td>EPL(1B)</td><td>40.52</td><td>28.52</td><td>32.16</td><td>22.29</td><td>25.70</td><td>16.39</td><td>20.97</td><td>4.01</td><td>59.75</td><td>46.34</td><td>41.31</td><td>25.42</td><td>36.74</td><td>23.83</td></tr><tr><td>SAC(1B)</td><td>41.31</td><td>28.66</td><td>36.72</td><td>27.48</td><td>28.94</td><td>18.99</td><td>23.35</td><td>4.90</td><td>61.04</td><td>47.90</td><td>44.21</td><td>28.21</td><td>39.26</td><td>26.02</td></tr><tr><td>EPL(3B)</td><td>46.76</td><td>32.98</td><td>46.20</td><td>36.46</td><td>36.81</td><td>25.32</td><td>33.80</td><td>7.57</td><td>65.73</td><td>53.70</td><td>55.46</td><td>34.86</td><td>47.46</td><td>31.82</td></tr><tr><td>SAC(3B)</td><td>47.96</td><td>34.51</td><td>48.46</td><td>38.39</td><td>42.21</td><td>29.85</td><td>33.61</td><td>7.72</td><td>75.11</td><td>62.89</td><td>55.53</td><td>35.00</td><td>50.48</td><td>34.73</td></tr><tr><td>EPL(8B)</td><td>52.52</td><td>37.57</td><td>49.98</td><td>39.92</td><td>40.09</td><td>27.24</td><td>35.22</td><td>7.27</td><td>69.77</td><td>57.84</td><td>57.36</td><td>36.66</td><td>50.82</td><td>34.42</td></tr><tr><td>SAC(8B)</td><td>48.54</td><td>34.04</td><td>51.55</td><td>40.65</td><td>41.44</td><td>29.18</td><td>35.52</td><td>7.72</td><td>78.91</td><td>67.71</td><td>57.87</td><td>36.26</td><td>52.31</td><td>35.93</td></tr></table>


Table 20: Model Performance on 32K Long-Context Summarization Tasks.


<table><tr><td>Methods</td><td>QMSum</td><td>GovReport</td><td>Average</td></tr><tr><td>SAC</td><td>14.95</td><td>22.03</td><td>18.49</td></tr><tr><td>SAC(ae+lm)</td><td>15.32</td><td>20.15</td><td>17.74</td></tr><tr><td>EPL</td><td>14.82</td><td>20.40</td><td>17.61</td></tr></table>


Table 21: Model Performance on 24K Long-Context QA Tasks.


<table><tr><td>Methods</td><td>MultiFieldQA</td><td>NarrativeQA</td><td>Qasper</td><td>Sigle-Doc-Avg.</td><td>2WikiMQA</td><td>MuSiQue</td><td>HotpotQA</td><td>Multi-Doc-Avg.</td></tr><tr><td>SAC</td><td>17.75</td><td>11.33</td><td>11.67</td><td>13.58</td><td>24.78</td><td>12.36</td><td>26.03</td><td>21.06</td></tr><tr><td>SAC(ae+lm)</td><td>17.38</td><td>10.49</td><td>9.05</td><td>12.31</td><td>22.47</td><td>7.76</td><td>23.88</td><td>18.04</td></tr><tr><td>EPL</td><td>14.34</td><td>10.50</td><td>7.58</td><td>10.81</td><td>21.99</td><td>8.43</td><td>23.72</td><td>18.05</td></tr></table>


ID F1 Performance across Models and Compression Ratios


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-19/90086f57-8f2f-46c4-bee7-c1bab09f413c/dde21ca1f5544f4ebbb88dc742aee3b6dd00bb91464f45c91a973b2aaaf1b210.jpg)



Figure 10: Efficiency and Performance Trade-off Curves on In-Domain (ID) Tasks.
