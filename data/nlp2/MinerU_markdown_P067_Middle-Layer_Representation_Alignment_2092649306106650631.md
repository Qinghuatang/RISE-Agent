# Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs

Danni Liu Jan Niehues Karlsruhe Institute of Technology, Germany {danni.liu, jan.niehues}@kit.edu 

## Abstract

While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective crosslingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available<sup>1</sup>. 

## 1 Introduction

Decoder-only large language models (LLMs) have emerged as the dominant paradigm in NLP. While these models exhibit promising zero-shot capabilities (Wei et al., 2022; Chowdhery et al., 2023), further task-specific fine-tuning remains crucial for optimal performance in many applications (Shen et al., 2024; Xu et al., 2024; Alves et al., 2024). During fine-tuning, a practical challenge is that the available training data rarely covers all languages supported by LLMs. This highlights the importance of cross-lingual transfer to extend taskspecific performance gains across languages. 

While cross-lingual transfer has been extensively studied (Wang and Zheng, 2015; Ruder et al., 2019; 

Artetxe and Schwenk, 2019b; Pfeiffer et al., 2020), achieving it on generative tasks with variablelength outputs remains challenging (Vu et al., 2022; Li and Murray, 2023) compared to classification tasks. This challenge is especially relevant for LLMs, which formulate all tasks as next-token prediction problems. 

The theoretical foundation of cross-lingual transfer lies in the analogous relationships between concepts across languages. This intuition was first demonstrated in cross-lingual word embeddings (Mikolov et al., 2013; Lample et al., 2018; Xu and Koehn, 2021), where these vector representations exhibit isometric relationships, i.e., the geometric structure of semantically equivalent items is preserved across different languages. This isometry property has proven crucial for transferring learned models across languages (Schuster et al., 2019; Wang et al., 2024b). Subsequent encoder-decoder models (Ha et al., 2016) and decoder-only models (Wu et al., 2024a) also exhibit similar properties in their internal representations. 

While pretrained multilingual models naturally develop some degree of unified multilingual representations (Pires et al., 2019; Conneau et al., 2020; Muller et al., 2021), explicitly strengthening the relationships between semantically equivalent content has shown benefits in various downstream tasks: cross-lingual retrieval (Yu et al., 2018), parallel text mining (Schwenk et al., 2021), zero-shot classification (Hu et al., 2021; Gritta and Iacobacci, 2021) and translation (Arivazhagan et al., 2019; Pham et al., 2019; Duquenne et al., 2022). Despite different approaches, these works share a common objective: aligning representations of semantically equivalent content across languages while preserving overall expressiveness. 

Cross-lingual alignment approaches have been successfully applied to models preceding LLMs. For encoder-only models, outputs can be aligned by e.g., minimizing distances between parallel sentence representations (Feng et al., 2022) or cross-lingual masked language modeling objectives (Conneau and Lample, 2019). These techniques are largely applicable to encoder-decoder models, where alignment is typically enforced to the encoder outputs (Duquenne et al., 2023). In contrast, decoder-only models lack such clear separation between input processing and output generation. This makes it less obvious where and how to optimize for cross-lingual alignment, as also highlighted in the survey by Hämmerl et al. (2024). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96d4abd9-69bd-4d63-b8df-36c77b8d83e4/c78073c9f4e655cd8303638d42bf25285f150edc8130c128ac2db20c81811af4.jpg)



(a) Cross-lingual semantic alignment (measured by average retrieval accuracy over 35 languages and 1190 language directions) varies by layer, with the middle layer showing the highest score. Lower-resource languages are poorly aligned.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96d4abd9-69bd-4d63-b8df-36c77b8d83e4/6f3c51f9e432342e1bcb6738ea7e94842c9de780885d544f3d8d4d87d634d89a.jpg)



(b) Positive correlation between base model cross-lingual semantic alignment and downstream transfer performance.



Figure 1: Two observations (§2) motivating our approach of aligning multilingual representations (§3).


In this work, we start by quantifying the degree of cross-lingual alignment present in two prominent LLMs, Llama 3 (AI @ Meta et al., 2024) and Qwen 2.5 (Qwen Team et al., 2025). We then apply these insights to improve cross-lingual transfer in task-specific fine-tuning. By alternatively training on alignment and task-specific data, we aim to improve the cross-lingual generalization to languages without fine-tuning data. We demonstrate transfer improvements across diverse tasks: slot filling, machine translation, and structured text generation. Our main findings include: 

• Applying alignment objectives to middle layers during LLM task-specific fine-tuning improves cross-lingual transfer (§5.1) and enhances alignment across all network depths (§5.2). 

• The transfer improvements extend beyond those languages seen in alignment (§5.1). 

• Our approach is robust to the choice of languages used for alignment training (§6.2, 6.3). 

• Task-specific and alignment modules trained separately can be combined post-hoc to improve transfer performance (§6.4). 

## 2 Analyzing Cross-Lingual Alignment

To understand how well LLM representations capture semantic equivalence across languages, we use translation retrieval as a diagnostic task. We choose this retrieval task over other metrics like cosine similarity or SVCCA score (Raghu et al., 2017) because it better captures relative semantic relationships. That is, if a model’s representations enable us to identify a sentence’s translation from a set of candidates, the exact numerical distance between the query and the retrieved translation is less important than the ability to rank translations as the most semantically similar. 

Specifically, we first extract model activations at each network layer for all language variants of the input text. To handle variable-length sequences, we create fixed-size sentence embeddings by meanpooling the activations over the sequence length dimension. For translation retrieval, given a query sentence in one language, we compare its embedding to the embeddings of candidate sentences in the target language using ratio-based margin similarity (Artetxe and Schwenk, 2019a)<sup>2</sup>. For N languages, we evaluate retrieval accuracy across all N(N 1) possible language pairs. We use the FLORES-200 dataset (NLLB Team, 2024), which provides high-quality multiway parallel texts across diverse languages (detailed setup in §4.2). 

Our investigation of LLama 3 and and Qwen 2.5 models<sup>3</sup> reveals three key findings: 

Overall weak semantic alignment, with peak in middle layers: As shown in Figure 1a, the average translation retrieval accuracy across 1,190 language pairs remains below 50%, with Llama 3 outperforming Qwen 2.5. Low-resource languages show especially weak alignment, achieving less than half of the overall average accuracy. In particular, the middle layers of both models demonstrate the strongest retrieval performance. This suggests stronger potential for cross-lingual transfer at these intermediate representations. 

Strong correlation between base LLM semantic alignment and downstream task transfer: To what extent can the semantic alignment present in the base LLM predict cross-lingual transfer performance after supervised fine-tuning? Using multilingual slot filling as a case study, we train models on 5 high-resource languages jointly and evaluate transfer performance on 25 additional languages (detailed setup in §4.1). As shown in Figure 1b, for both Llama 3 and Qwen 2.5, we observe strong positive correlations $( p < 0 . 0 1 )$ between middlelayer retrieval accuracy and downstream task performance. This correlation suggests that increasing cross-lingual alignment in LLM intermediate representations may improve cross-lingual transfer. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96d4abd9-69bd-4d63-b8df-36c77b8d83e4/d55c69604eaa5aeafe50cf81162f6897a219b9f13e0183de0e4805c5e80ae57a.jpg)



Figure 2: Illustration of our approach, alternating training between task-specific (left) and alignment (right) objectives. The alignment objective operates on middle-layer representations


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96d4abd9-69bd-4d63-b8df-36c77b8d83e4/b589745a33bf8ce372f8a703b413976135a445278c3845931a01e8d8b6810f15.jpg)



Figure 3: Task-specific fine-tuning shows minimal impact on semantic alignment.


Task-specific fine-tuning preserves but does not enhance semantic alignment: After analyzing the base LLMs, we examine how supervised finetuning affects the models’ internal semantic alignment. Using the same multilingual slot filling task as before, we study both English-only and multilingual fine-tuning. Despite multilingual fine-tuning being an established method for improving crosslingual transfer (Li and Murray, 2023; Chirkova and Nikoulina, 2024), we observe that neither training configuration alters the models’ cross-lingual semantic alignment (Figure 3). This preservation of baseline alignment patterns, even under multilingual training, indicates that pure fine-tuning does not sufficiently strengthen cross-lingual alignment. This further motivates us towards explicit cross-lingual alignment during fine-tuning. 

## 3 Explicit Alignment in fine-tuning

Alternate Training As shown in Figure 2, we optimize either the task-specific objective or the alignment objective in each training step. Compared to joint optimization that computes a combined loss for both objectives and performs a single backward pass, this approach does not involve manually balancing objective weights and mitigates potential gradient conflicts between objectives. It also showed stronger task performance empirically. 

Task Objective We follow standard causal language modeling, using a cross-entropy loss over the predicted text conditioned on the input prefix. 

Alignment Objective We use a contrastive loss motivated by its successful applications in sentence embedding (Feng et al., 2022), dense retrieval (Karpukhin et al., 2020) and modality alignment (Ye et al., 2022; Girdhar et al., 2023). The loss maximizes the similarity between translations while minimizing similarity between non-translations. Given a batch of n pairs of parallel sentences, the alignment loss for a sentence pair (s, t) is: 

$$
\mathcal {L} _ {\mathrm{align}} = - \log \frac {\exp (\mathrm{sim} (\mathbf {h} _ {s} ^ {i} , \mathbf {h} _ {t} ^ {i}))}{\sum_ {v \in \mathcal {B}} \exp (\mathrm{sim} (\mathbf {h} _ {s} ^ {i} , \mathbf {h} _ {v} ^ {i}))}\tag{1}
$$

where $\mathbf { h } _ { s } ^ { i }$ is the mean-pooled<sup>5</sup> hidden states at the $i ^ { t h }$ LLM layer for input s and sim( , ) is a similarity function. Motivated our finding that middle layers have the strongest cross-lingual alignment potential, we select i as the middle layer and compare its performance to other layer positions. We use cosine similarity following prior works (Gao et al., 2021; Ye et al., 2022). The similarity score is optionally scaled by a temperature parameter τ, which controls the peakiness of the softmax distribution and in turn determines the relative importance of non-translation pairs. This temperature parameter is tuned on the development sets. 

Activating Individual Objectives Note that the task and alignment losses can be activated separately. Deactivating the alignment loss degenerates to standard task-only training. Conversely, deactivating the task loss trains the model only for alignment. The modularity allows combining separately-trained task and alignment models. 

Data Requirement Our approach requires minimal parallel data. Later experiments show that for lower-resource languages, a few hundreds of sentences of parallel data is sufficient to improve transfer. Our approach also offers a practical advantage over alternatives that require monolingual language modeling training for each transfer target language (Ansell et al., 2022; Vu et al., 2022; Chronopoulou et al., 2024). 

<table><tr><td></td><td>Dataset</td><td>Languages</td></tr><tr><td colspan="3">Slot Filling</td></tr><tr><td>Task - train</td><td>MASSIVE</td><td>{ar, en, es, ru, zh}</td></tr><tr><td>Task - test</td><td>MASSIVE</td><td>supervised + {af, az, cy, de, el, fr, hi, is, ja, jv, sw, th, tl, tr, ur}</td></tr><tr><td>Alignment</td><td>Tatoeba</td><td>low-res.: {cy, jv, jp, sw, tl}-enmid-res.: {el, hi, th, tr}-enhigh-res.: {ar, es, ru, zh}-en</td></tr><tr><td colspan="3">Machine Translation</td></tr><tr><td>Task - train</td><td>ALMA</td><td>{cs, de, is, ru, zh}↔ en</td></tr><tr><td>Task - test</td><td>WMT 23</td><td rowspan="2">supervised + {he, ja, uk} ↔ en(same as “Task - train”)</td></tr><tr><td>Alignment</td><td></td></tr><tr><td colspan="3">JSON Generation (challenge task)</td></tr><tr><td>Task - train</td><td>UNER</td><td>{en, pt, zh}</td></tr><tr><td>Task - test</td><td>UNER</td><td>supervised + {da, hr, sk, sr, sv}</td></tr><tr><td>Alignment</td><td>Tatoeba</td><td>{da, sv}-en</td></tr><tr><td colspan="3">Semantic Alignment Evaluation (diagnostic task)</td></tr><tr><td>Alignment</td><td>FLoRes-200</td><td>N(N-1) pairs for N lang.</td></tr></table>


Table 1: Dataset statistics for three downstream tasks and one diagnostic task. “Train” refers to languages involved in SFT, and "test" includes SFT languages and additional transfer languages unseen during training. See Appendix B for more details.


## 4 Experimental Setup

## 4.1 Data

In general, we fine-tune on several high-resource languages and then evaluate transfer performance on additional languages. We do not focus on English-only fine-tuning, since our initial experiments demonstrated that multilingual finetuning substantially outperforms English-only finetuning<sup>6</sup>, thus establishing it as a stronger baseline. Table 1 presents a dataset overview. Descriptions of the language codes are in Appendix C. 

Main Task Data: We evaluate our approach on slot filling and machine translation, both modeled as generative tasks with templates shown in Appendix D.2. For slot filling, we use the MASSIVE dataset (FitzGerald et al., 2023). We train on 5 high-resource languages, and evaluate transfer performance on 15 additional diverse languages, 5 of which have non-Latin writing systems. This task presents a challenge due to the 60 possible slots, requiring strictly following the output format for correct parsing. For machine translation, we use ALMA (Xu et al., 2024)’s training and test data, and additionally test on 6 zero-shot directions from WMT 23 (Kocmi et al., 2023). 

Challenge Task Data: To assess performance on long-sequence processing and structured text generation, we include JSON generation as a challenge task. We use the UNER dataset (Mayhew et al., 2024) from the Aya collection (Singh et al., 2024), which requires following example instructions and extracting named entities into JSON format. A challenge not present in the previous tasks is the longer inputs, with an average input length exceeding 150 tokens in English. For this task, we train on 3 high-resource languages (en, pt, zh) and transfer to the 5 remaining languages. 

Alignment Data: For alignment, we mainly use parallel data to English from Tatoeba (Tiedemann, 2020), except for machine translation, where the training sentences are inherently parallel. For slot filling, our main experiments align the five languages with the weakest baseline<sup>7</sup> transfer performance (cy, jv, jp, sw, tl) reported by the dataset creators (FitzGerald et al., 2023). We choose them because their weak baseline performance suggests a lack of effective transfer, providing a strong testbed for evaluating the potential benefits of our alignment approach. For ablation, we alter the following factors of the alignment data: 

• Resource level (low, medium, high-resource) 

• Language coverage 

• Domain (oracle data, different, very distant) For machine translation, given the inherent semantic equivalence of translation pairs, we directly leverage the translation data for alignment. For JSON generation, we align the two lowestresourced in UNER (da and sv)<sup>8</sup> to English. For lower-resource languages, the alignment data are a few hundreds as detailed in Appendix B. 

## 4.2 Evaluation

Semantic Alignment Evaluation: As described in §2, we evaluate cross-lingual semantic alignment by retrieval accuracy. Given N languages, we perform many-to-many retrieval and average the accuracy over the N(N  1) language pairs. For the initial analyses (§2), the 35 languages are listed in Appendix C. We use the FLoRes-200 (NLLB Team, 2024) development set with 997 parallel sentences. While FLoRes partially overlaps with ALMA’s training data, it remains the only reliable massively multilingual multiway corpus to the best of our knowledge. Alternative such as Tatoeba have been advised against due to data imbalance and noise (Heffernan et al., 2022; Janeiro et al., 2024). We also demonstrate that this overlap does not result in memorization effects (§6.3). When reporting an aggregated retrieval accuracy for a model, we average over all language pairs at evennumbered layers’ retrieval accuracy, excluding the input embedding layer. 

Task Performance Evaluation: For slot filling, we report $\mathrm { F _ { 1 } }$ scores using the original evaluation script by FitzGerald et al. (2023). For machine translation, we report BLEU<sup>9</sup> (Papineni et al., 2002) and COMET-22 (Rei et al., 2022) scores. For JSON generation, we parse the generated outputs back to named entity tuples and then evaluate $\mathrm { F _ { 1 } }$ scores. 

## 4.3 Model, Training, and Inference

We build upon Llama (AI @ Meta et al., 2024) and Qwen (Qwen Team et al., 2025), specifically Meta-Llama-3-8B-Instruct<sup>10</sup> and Qwen2.5-7B-Instruct. We use LoRA (Hu et al., 2022) adapters with a rank of 8 for all attention components and linear projections. The effective batch size is 128 for both objectives, with minibatches of 32 examples considered for the contrastive objective<sup>11</sup>. Alignment data from different languages are re-sampled to an approximately uniform distribution. More details are in Appendix D. 

## 5 Main Results

The main results are summarized in Table 2. Before assessing our proposed approach, we first establish the necessity of supervised FT by comparing it with zero-shot usage of the LLMs (rows (2, 5) vs. (1, 4)). On slot filling, the zero-shot performance of Llama 3 is very poor, achieving only 

6.6% $\mathrm { F _ { 1 } }$ on English due to difficulties in adhering to task-specific formats. We therefore do not evaluate its zero-shot performance on all languages. In machine translation, supervised fine-tuning shows substantial gains of 4-6 COMET over zero-shot. 

## 5.1 Overall Performance Comparison

Gains in cross-lingual transfer with supervised performance preserved: Our approach improves cross-lingual transfer across different tasks and models. For slot filling, we observe gains in both supervised and transfer $( \mathrm { F _ { 1 } } ~ + 0 . 4 ~ \mathrm { a n d } ~ + 1 . 5 $ respectively) settings on Llama fine-tuning, with similar improvements on Qwen $( \mathrm { F } _ { 1 } \ + 0 . 7$ supervised, +1.8 transfer). In machine translation with Llama in row (3), our approach brings substantial gains when transferring to out-of-English directions (+1.5 BLEU, +1.1 COMET).<sup>12</sup> For into-English directions, there is a modest improvement in +0.5 BLEU and +0.2 COMET. The larger gains on out-of-English directions suggest the approach is more beneficial for non-English generation in this case. For Qwen in row (6), our approach shows minor gains in into-English translation (+1.1 BLEU but no change in COMET), and does not influence out-of-English scores. It also leads to a degradation ( 0.8 COMET) on supervised directions. This is potentially due to Qwen’s non-English-centric pretraining combined with our English-centric alignment data. With this exception, our approach maintains or improves supervised performance while enhancing transfer. 

Aligned languages improve the most, but gains extend to other languages: The diverse language coverage in the slot filling dataset allows us to com pare how the alignment objective benefits transfer to both aligned and non-aligned languages. While aligned languages show the strongest improvements $( \mathrm { F } _ { 1 } \ + 4 . 2$ and +4.9 for Llama and Qwen respectively), the benefits extend to other languages. Over the remaining 10 non-aligned languages, there is an average $\mathrm { F _ { 1 } }$ improvement of 0.4 (per-language results in Appendix E). This suggest that the alignment step enhances the model’s general cross-lingual transfer capabilities rather than optimizing for specific language pairs. 

Smaller gains on non-Latin script languages: Beyond overall performance improvements, we observe smaller gains on languages with diverse writing systems. Specifically, for the non-Latin script transfer languages in the slot filling task (Greek, Hindi, Japanese, Thai, Urdu), the average improvement is only $0 . 5 \mathrm { F _ { 1 } }$ in contrast to the overall average gain of 1.5. This reduced gain is likely related to suboptimal tokenization for these languages in multilingual models (Rust et al., 2021; Petrov et al., 2023; Hong et al., 2024). When tokens poorly align with linguistic units, the mean-pooled sentence representations may poorly capture semantics, thereby impacting our alignment objective. 

<table><tr><td rowspan="3">ID Model</td><td colspan="4">Slot Filling (MASSIVE)</td><td colspan="7">Machine Translation (WMT23)</td></tr><tr><td>Supervised(5 lang.)</td><td>Transfer(15 lang.)</td><td>Transfer(aligned)</td><td>Retrieval(all 20 lang.)</td><td colspan="2">Supervised(5 lang.↔En)</td><td colspan="2">Transfer(3 lang.→En)</td><td colspan="2">(En→3 lang.)</td><td>Retrieval(all 9 lang.)</td></tr><tr><td><eq>F_1</eq></td><td><eq>F_1</eq></td><td><eq>F_1</eq></td><td>Acc.</td><td>BLEU</td><td>COMET</td><td>BLEU</td><td>COMET</td><td>BLEU</td><td>COMET</td><td>Acc.</td></tr><tr><td>(1) LLAMA 3</td><td>-</td><td>-</td><td>-</td><td>39.1</td><td>25.8</td><td>75.5</td><td>27.8</td><td>75.8</td><td>14.8</td><td>71.3</td><td>51.5</td></tr><tr><td>(2) + SFT</td><td>76.6</td><td>60.2</td><td>51.7</td><td>39.4</td><td>30.0</td><td>81.5</td><td>31.8</td><td>82.8</td><td>15.5</td><td>79.6</td><td>(55.3)</td></tr><tr><td>(3) + alignment</td><td>77.0</td><td>61.7</td><td>55.5</td><td>73.2</td><td>29.9</td><td>81.5</td><td>32.3</td><td>83.0</td><td>17.0</td><td>80.7</td><td>(84.5)</td></tr><tr><td>(4) QWEN 2.5</td><td>-</td><td>-</td><td>-</td><td>21.4</td><td>23.0</td><td>74.5</td><td>28.5</td><td>81.3</td><td>12.6</td><td>71.2</td><td>36.5</td></tr><tr><td>(5) + SFT</td><td>76.3</td><td>53.5</td><td>41.6</td><td>20.9</td><td>27.4</td><td>78.4</td><td>29.7</td><td>82.7</td><td>14.6</td><td>76.9</td><td>(38.8)</td></tr><tr><td>(6) + alignment</td><td>77.0</td><td>55.3</td><td>46.5</td><td>20.5</td><td>27.2</td><td>77.6</td><td>30.8</td><td>82.7</td><td>14.7</td><td>76.9</td><td>(75.6)</td></tr></table>


Table 2: Overall supervised and transfer results. Retrieval accuracy are averaged over all language pairs and layers. Bold: highest task scores which outperforms the other setups. (Results in brackets): potentially inflated scores due to partial overlap between retrieval and translation data. Language-specific results are in Appendix E.


## 5.2 Alignment Loss Placement

To validate our choice of middle-layer alignment motivated by the analysis in §2, we compare performance when applying the alignment loss at different network depths: bottom $( 8 ^ { \mathrm { t h } } )$ , middle $( 1 6 ^ { \mathrm { { t h } } } )$ and top $( 3 2 ^ { \mathrm { n d } } )$ layers of Llama. 

Middle layer placement achieves more balanced improvements in transfer languages: As shown in Table 3, compared to the "middle" configuration, the "bottom" configuration clearly leads to poor overall performance in both supervised and transfer settings, with a particularly strong degradation on the slot filling task. While top-layer alignment maintains overall strong performance, it shows more unbalanced gains across transfer languages, as evidenced by the higher standard deviation of performance changes on transfer languages. 

Middle layer placement achieves better alignment across network depths: To better understand the effects of different loss placements, we run the translation retrieval task over model activations at from different intermediate layers. As shown in Figure 4, When the alignment loss is applied at the middle $( 1 6 ^ { \mathrm { { t h } } } )$ layer, semantic alignment is enhanced not only at that layer but also in multiple preceding layers. In contrast, top-layer alignment primarily affects only the final layer, and bottom-layer alignment shows limited improvement in alignment quality across all layers. This is likely because the lower layers are occupied with processing more fundamental text features (Belinkov et al., 2017; Peters et al., 2018) rather than abstract semantic meanings. 

<table><tr><td></td><td>Supervised↑</td><td>Transfer↑</td><td>Transfer SD↓</td></tr><tr><td colspan="4">Slot filling (MASSIVE): <eq>F_1</eq></td></tr><tr><td>SFT baseline</td><td>76.6</td><td>60.2</td><td>-</td></tr><tr><td>Middle (layer 16)</td><td>77.0</td><td>61.7</td><td>2.6</td></tr><tr><td>Top (layer 32)</td><td>76.6</td><td>62.0</td><td>3.3</td></tr><tr><td>Bottom (layer 8)</td><td>76.8</td><td>58.0</td><td>2.9</td></tr><tr><td colspan="4">Machine translation (WMT23): COMET</td></tr><tr><td>SFT baseline</td><td>81.5</td><td>79.6</td><td>-</td></tr><tr><td>Middle (layer 16)</td><td>81.5</td><td>80.7</td><td>3.7</td></tr><tr><td>Top (layer 32)</td><td>82.0</td><td>80.2</td><td>4.2</td></tr><tr><td>Bottom (layer 8)</td><td>81.2</td><td>80.1</td><td>5.6</td></tr></table>


Table 3: Impact of alignment loss placement on Llama 3. Last column: standard deviation of gains on transfer languages compared to baseline. “Top” leads to more uneven gains across languages, while “bottom” degrades both supervised and transfer performance.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-08-27/96d4abd9-69bd-4d63-b8df-36c77b8d83e4/5b2642a66b63ec132b1e189eaa86f3a712bb73b474af091aa2126ec9eae2a962.jpg)



Figure 4: Retrieval accuracy over model depths when aligning different layers of Llama 3. Middle layer placement leads to overall better alignment.


Aligning several layers does not show consistent gains: Results in Table 3 suggest that aligning at multiple layers may be complementary. In Appendix F, we show that adding alignment losses at both middle and top layers brings further improvements on slot filling, but does not on machine translation. This task-dependent behavior indicates that how to best align multiple layers still requires further investigation. 

<table><tr><td>Alignment Setup</td><td>Avg supervised F1↑ (5 lang.)</td><td>Avg transfer F1↑ (15 lang.)</td></tr><tr><td>5 lang↔en (Table 2 row 3)</td><td>77.0</td><td>61.7</td></tr><tr><td>All↔English (38 pairs)</td><td>77.5</td><td>63.6 (+1.9)</td></tr><tr><td>All↔all (238 pairs)</td><td>77.7</td><td>63.6 (+1.9)</td></tr></table>


Table 4: Effects of larger-scale alignment configurations on slot filling. Aligning all non-English languages to English (+1.9 F1) outperforms our base configuration, while fully connecting all 20 languages offers no additional gain beyond English-only alignment.


## 5.3 Impact on Representation Retrieval

To assess the impact of the alignment loss on the learned model representations, we also report the retrieval accuracy for all languages involved in each task (20 for slot filling and 9 for machine translation) after fine-tuning in Table 2. For Llama on the slot filling task, the alignment loss substantially improves retrieval accuracy from 39.4% to 73.2%. For Qwen, the alignment loss does not improve retrieval among the 20 slot filling languages, possibly due to the lower accuracy of the base model with many low-resource languages with 0% accuracy, making improvement more challenging. For machine translation, as noted earlier §4.2, the retrieval test data overlaps with part of the task training data, potentially inflating accuracy (marked in brackets in Table 2). However, we verify that this overlap does not lead to perfect retrieval accuracy: Specif ically, at the $1 6 ^ { \mathrm { { t h } } }$ layer where the alignment loss is applied, English-source retrieval accuracies for supervised languages show varying accuracy: cs (98.1%), de (96.5%), is (66.9%), ru (90.6%), and zh (94.8%). This suggests that the overlap does not make the retrieval diagnostic task trivial. 

## 6 Analyses

## 6.1 Larger-Scale Alignment

While our main configuration for slot filling (Table 1) allows studying performance on languages not involved in alignment, we also explore largerscale alignment scenarios as oracle setups where all languages have parallel data. We conduct additional experiments on slot filling with two expanded configurations: 

• All 19 non-English languages aligned to English (38 directional pairs) 

<table><tr><td>Alignment Language Resource Level</td><td>Super. (5 lang.)</td><td>Transfer (15 lang.)</td><td>Gain on Aligned (4/5 lang.)</td></tr><tr><td>SFT (row (2) Table 2)</td><td>76.6</td><td>60.2</td><td>-</td></tr><tr><td>Low (row (3) Table 2)</td><td>77.0</td><td>61.7</td><td>+3.8</td></tr><tr><td>Medium</td><td>77.8</td><td>61.4</td><td>+1.1</td></tr><tr><td>High</td><td>77.6</td><td>60.4</td><td>+0.7</td></tr></table>


Table 5: Effect of alignment language resource levels on slot filling F1 . In three groups of alignment languages: low ({cy, jv, jp, sw, tl}), medium ({el, hi, th, tr}), and high-resource ({ar, es, ru, zh}), languages involved in alignment consistently show improvements, with the strongest gains (+3.8 F1) in the low-resource group.


• All 20 languages aligned to each other (238 pairs with alignment data from all 380 possible pairs). The results in Table 4 show that expanding alignment to all languages further improves transfer performance (F1 +1.9) in an oracle setup where every transfer language has alignment data. However, multiway alignment data does not further improve transfer, suggesting that aligning to English implicitly creates multiway alignment effects. 

## 6.2 Resource Level of Alignment Languages

In our main experiments, we selected the 5 languages with the weakest performance from the MASSIVE baseline (FitzGerald et al., 2023) for alignment. We now vary the resource level of the alignment languages using a medium-resource group with {el, hi, th, tr} en and a high-resource group with {ar, es, ru, zh} en, which also have supervised task training data. As shown in Table 5, all three configurations improve $\mathrm { F _ { 1 } }$ scores for the languages involved in alignment. However, the lowresource group exhibit the largest gains (+3.8 F ), indicating that our approach is most beneficial to languages with weaker initial performance. Moreover, overall transfer gains relative to the SFT baseline diminish when using high-resource languages for alignment, likely because these languages already have well-aligned representations and aligning them provides little benefit to lower-resource languages in the transfer set. Overall, the results show that our approach is robust to the choice of alignment languages, but selecting initially poorly aligned languages could provide broader benefits across different languages. 

## 6.3 Generalization of Learned Alignment

Table 6 examines the language and domain generalization of our alignment component. To isolate the effects of task-specific joint training, we train the models using only the alignment loss, following the same setup as our previous experiments but without optimizing on task-specific data. We then evaluate retrieval accuracy as described in §4.2. 

<table><tr><td>Alignment Data</td><td>Overall (20 lang.)</td></tr><tr><td>Multi {ar,es,ru,zh,sw}-en</td><td>80.2</td></tr><tr><td>Only de-en</td><td>71.9</td></tr><tr><td>Only es-en</td><td>72.9</td></tr><tr><td>Only zh-en</td><td>72.7</td></tr><tr><td>de-en FLoRes (oracle)</td><td>77.7</td></tr><tr><td>Tatoeba (different)</td><td>71.9</td></tr><tr><td>IWSLT (very distant)</td><td>68.5</td></tr></table>


Table 6: Alignment generalization across languages and domains. Upper: Multilingual training improves overall alignment. Lower: Impacts of alignment transfer reasonably across domains, with performance drops when training data differs from test domain.


Language Generalization: While our main experiments align multiple language pairs, we now use single languages for alignment. As shown in Table 6 (upper portion), that single-language alignment training leads to diminished performance compared to multilingual training. Interestingly, we see comparable accuracy drops regardless of which individual language is used for alignment, suggesting that the gains of multilingual alignment come from the diversity of the training data rather than characteristics of individual languages. 

Domain Generalization: To isolate the effects of multilinguality, we focus on alignment between a single language pair (English-German). In Table 6 (lower portion), we first establish an oracle setup using models trained on FLoRes data (Wikipedia domain, overlapping with retrieval data). We then compare to two setups where the alignment data come from other domains: Tatoeba (short sentences for language learning; different) and IWSLT 2017 (public speaking transcriptions; very distant). While we observe a decrease in retrieval accuracy compared to the oracle setup, the results suggest that, to enforce alignment into the model, it is not strictly necessary to source alignment data from the same domain as the task-specific data. 

## 6.4 Merging Alignment and Task Modules

Our previous experiments focused on models jointly trained on both task and alignment objectives. However, in practice, it may be necessary to enhance existing task-specific models with crosslingual capabilities, where joint re-training is infeasible due to computational constraints or unavailability of the original task training data. Inspired by recent advances in model merging (Matena and Raffel, 2022; Ilharco et al., 2023), we explore the feasibility of combining separately-trained task and alignment modules. We merge two sets of trained LoRA adapters by averaging their weights<sup>13</sup>: the alignment module trained in isolation (§6.3), and task-specific modules (rows (2) and (5) in Table 2). 

<table><tr><td>Setup</td><td>Supervised</td><td>Transfer</td></tr><tr><td colspan="3">Slot filling (MASSIVE): <eq>F_{1\uparrow}</eq></td></tr><tr><td>SFT (row (2) Table 2)</td><td>76.6</td><td>60.2</td></tr><tr><td>Joint (row (3) Table 2)</td><td>77.0 (+0.4)</td><td>61.7 (+1.5)</td></tr><tr><td>Merge</td><td>76.9 (+0.3)</td><td>61.3 (+1.1)</td></tr><tr><td colspan="3">Machine translation (WMT23): <eq>COMET_{\uparrow}</eq></td></tr><tr><td>SFT (row (4) Table 2)</td><td>81.5</td><td>79.6</td></tr><tr><td>Joint (row (5) Table 2)</td><td>81.5 (+0.0)</td><td>80.7 (+1.1)</td></tr><tr><td>Merge</td><td>82.0 (+0.5)</td><td>80.2 (+0.6)</td></tr></table>


Table 7: Performance comparison of merged alignment and task modules versus joint training. Post-hoc merging of separately-trained LoRA adapters achieves comparable improvements to joint training.


<table><tr><td></td><td>Supervised (en, pt, zh)</td><td>Transfer (da, sv)</td><td>Transfer (5 lang.)</td></tr><tr><td>Llama SFT</td><td>83.4</td><td>82.1</td><td>79.3</td></tr><tr><td>+ alignment</td><td>82.4</td><td>83.1</td><td>79.8</td></tr></table>


Table 8: Results on JSON generation evaluated with $\mathrm { F _ { 1 } }$ , showing modest gains for aligned languages but decreased performance for supervised languages.


Table 7 shows that this post-hoc merging brings comparable improvements comparable to joint training. Moreover, the improvements are more evenly distributed across languages compared to the larger gains observed on languages used directly in alignment. These results demonstrate that our alignment approach is modular and can be combined with existing task-specific models. 

## 6.5 Long Sequence Processing

We investigate a more challenge task requiring longer input and output generation using UNER (§4.1). As shown in Table 8, while aligned languages still show improvements, the gains are more modest compared to previous experiments, with an $\mathrm { F _ { 1 } }$ increase of 1.0 on aligned languages and 0.5 across all transfer languages. Moreover, there is an average degradation of $1 . 0 \mathrm { F _ { 1 } }$ on supervised languages, mainly due to the decline in Chinese ( 2.2 $\mathrm { F } _ { 1 } )$ . We suspect that this is due to our sentencelevel alignment objective operates on fixed-length representations, which creates conflicts with processing longer sequences. As Chinese is the only character-based language in the JSON generation dataset, which has roughly twice the number of tokens compared to English of equivalent content, the conflict could be more influential for Chinese. 

## 7 Related Works

Multilingual Capabilities of LLMs LLM performance varies across languages due to imbalanced pre-training data volume. However, even predominantly English-centric models (Touvron et al., 2023) exhibit some degree of multilingual capability (Aycock and Bawden, 2024; Yuan et al., 2024), potentially due to the unintentional ingestion of multilingual data during pretraining (Briakou et al., 2023). Meanwhile, many recent LLMs have expanded their language coverage (AI @ Meta et al., 2024; Qwen Team et al., 2025). Despite these inherent multilingual capabilities, extending them to downstream tasks in low-resource settings (Adelani et al., 2024; Iyer et al., 2024) remains challenging. 

Multilingual Representation Alignment Enhancing meaningful cross-lingual relationships between model representations has been a wellstudied area in the context of many tasks, including intermediate tasks such as bilingual lexicon induction (Zhang et al., 2017) and sentence embeddings (Feng et al., 2022; Li et al., 2023), as well as more direct applications like information retrieval (Izacard et al., 2022) and translation (Pham et al., 2019). 

Multilingual representation alignment can be achieved by various mechanisms, such as similarity losses that push translations toward each other (Pham et al., 2019), contrastive losses (Hadsell et al., 2006) that additionally incorporate nontranslation pairs, and adversarial losses (Ganin and Lempitsky, 2015) that remove language-specific signals. The cross-lingual transfer capabilities of these approaches is extensively documented in the literature. In particular, contrastive learning methods have shown promising results (Pan et al., 2021; Chi et al., 2021; Qi et al., 2022). Our contribution is not applying contrastive learning itself, but rather investigating how to effectively align multilingual spaces specifically in decoder-only models. 

In the context of LLMs, Wang et al. (2024b) use linear projections learned offline to align non-English representations with English ones during decoding. Our work differs in that our alignment objective is parameterized by the same weights as task-specific fine-tuning, and is directly applicable to multilingual fine-tuning. Wu et al. (2024a) align LLM top-layer representations specifically for the task of semantic textual similarity (STS). Different from this work, they do not consider cross-lingual transfer in downstream tasks or explore intermediate LLM layers for alignment. 

LLM Representation Analysis Several recent works have analyzed LLM internal representations with geometric analysis of representation spaces (Razzhigaev et al., 2024; Lee et al., 2024), probing classifiers (Wang et al., 2024a; Li et al., 2025), or logit lens analysis (Wu et al., 2024b). Multiple studies (Wu et al., 2024b; Mao and Yu, 2024; Zhong et al., 2024) reported higher representational similarity in middle layers in various evaluation settings, complementing our findings. Wu et al. (2024b) identify “semantic hubs” in LLM middle layers that integrate information from various data types, while we focus specifically on cross-lingual representations rather than multi-modality. Mao and Yu (2024) show that SFT on machine translation increases similarity between parallel sentences from the same MT corpus, while we show that SFT on a non-translation task does not increase representation similarity, thereby motivating explicit alignment during SFT. Zhong et al. (2024) measure pairwise similarity to English representations ("latent language") on high resource languages, while we focus on pairwise similarity across a more diverse set of language pairs. Kargaran et al. (2024) use similarity between parallel sentences to estimate cross-lingual transfer capabilities. Our analysis in §2 shares the same motivation, and we additionally show that actively enforcing alignment can improve transfer performance. 

## 8 Conclusion

We presented a simple yet effective approach for enhancing cross-lingual transfer in LLMs through middle-layer representation alignment during finetuning. Our experimental results lead to several practical recommendations: 1) Aligning a few weakly-performing languages yields broad transfer benefits. A few hundreds of parallel sentences as alignment data are sufficient. 2) Alignment data can be sourced from different domains as the task. 3) Existing task-specific models can be enhanced with our approach via parameter merging without the need of full re-training. 

## Limitations

Performance on languages with diverse scripts: As discussed in §5.1, our approach shows smaller gains on languages non-Latin scripts. This limitation is likely related to fundamental tokenization challenges, where suboptimal token segmentation negatively impacts the quality of mean-pooled representations. While our initial experiments on attention pooling did not lead to improvements, exploring more sophisticated pooling mechanisms could potentially address this challenge in future work. 

Computational overhead during training: The alternating optimization between task and alignment objectives doubles the computational cost during training compared to standard fine-tuning. In computationally constrained settings, our merging approach (§6.4), which separates task-specific and alignment training, should be prioritized. Given that alignment can be effectively performed using only a small number of parallel sentences (a few hundred per language), this modular approach can significantly reduce the overall computational cost. 

Trade-offs between supervised and transfer performance in challenging scenarios: While our approach generally maintains or improves supervised task performance while improving transfer, we observe degradation in supervised performance in two specific scenarios. First, in structured text generation (§6.5), the method shows reduced effectiveness and can impair supervised performance ( 1.0 F ), suggesting that our sentence-level align ment may interfere with the processing of longer, structured sequences. Second, when applying the method to models with weak initial cross-lingual alignment (§5.1), there could be a trade-off between improved transfer and supervised performance. 

## Acknowledgments

We thank the reviewers for their feedback, as well as Felix Stahlberg and Google Research. Part of this work was funded by the KiKIT (The Pilot Program for Core-Informatics at the KIT) of the Helmholtz Association. The authors gratefully acknowledge the computing time provided on the high-performance computer HoreKa by the National High-Performance Computing Center at KIT (NHR@KIT). This center is jointly supported by the Federal Ministry of Education and Research and the Ministry of Science, Research and the Arts of Baden-Württemberg, as part of the National High-Performance Computing (NHR) joint funding program. HoreKa is partly funded by the German Research Foundation (DFG). 

## References



David Ifeoluwa Adelani, A. Seza Dogruöz, André˘ Coneglian, and Atul Kr. Ojha. 2024. Comparing LLM prompting with cross-lingual transfer performance on indigenous and low-resource Brazilian languages. In Proceedings ofthe 4th Workshop on Natural Language Processingfor Indigenous Languages ofthe Americas (AmericasNLP 2024), pages 34–41, Mexico City, Mexico. Association for Computational Linguistics. 





AI @ Meta, Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, and 543 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783. 





Duarte M. Alves, José Pombal, Nuno Miguel Guerreiro, Pedro Henrique Martins, João Alves, M. Amin Farajian, Ben Peters, Ricardo Rei, Patrick Fernandes, Sweta Agrawal, Pierre Colombo, José G. C. de Souza, and André F. T. Martins. 2024. Tower: An open multilingual large language model for translation-related tasks. CoRR, abs/2402.17733. 





Alan Ansell, Edoardo Ponti, Anna Korhonen, and Ivan Vulic. 2022.´ Composable sparse fine-tuning for crosslingual transfer. In Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1778–1796, Dublin, Ireland. Association for Computational Linguistics. 





Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Roee Aharoni, Melvin Johnson, and Wolfgang Macherey. 2019. The missing ingredient in zero-shot neural machine translation. Preprint, arXiv:1903.07091. 





Mikel Artetxe and Holger Schwenk. 2019a. Marginbased parallel corpus mining with multilingual sentence embeddings. In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 3197–3203, Florence, Italy. Association for Computational Linguistics. 





Mikel Artetxe and Holger Schwenk. 2019b. Massively multilingual sentence embeddings for zeroshot cross-lingual transfer and beyond. Transactions of the Association for Computational Linguistics, 7:597–610. 





Seth Aycock and Rachel Bawden. 2024. Topic-guided example selection for domain adaptation in LLM based machine translation. In Proceedings of the 





18th Conference of the European Chapter of the Associationfor Computational Linguistics: Student Re search Workshop, pages 175–195, St. Julian’s, Malta. Association for Computational Linguistics. 





Yonatan Belinkov, Nadir Durrani, Fahim Dalvi, Hassan Sajjad, and James Glass. 2017. What do neural machine translation models learn about morphology? In Proceedings of the 55th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 861–872, Vancouver, Canada. Association for Computational Linguistics. 





Eleftheria Briakou, Colin Cherry, and George Foster. 2023. Searching for needles in a haystack: On the role of incidental bilingualism in PaLM‘s translation capability. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9432–9452, Toronto, Canada. Association for Computational Linguistics. 





Mauro Cettolo, Marcello Federico, Luisa Bentivogli, Jan Niehues, Sebastian Stüker, Katsuhito Sudoh, Koichiro Yoshino, and Christian Federmann. 2017. Overview of the IWSLT 2017 evaluation campaign. In Proceedings ofthe 14th International Conference on Spoken Language Translation, pages 2–14, Tokyo, Japan. International Workshop on Spoken Language Translation. 





Changyou Chen, Jianyi Zhang, Yi Xu, Liqun Chen, Jiali Duan, Yiran Chen, Son Tran, Belinda Zeng, and Trishul Chilimbi. 2022. Why do we need large batchsizes in contrastive learning? A gradient-bias perspective. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022. 





Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, and Ming Zhou. 2021. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In Proceedings ofthe 2021 Conference ofthe North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 3576–3588, Online. Association for Computational Linguistics. 





Nadezhda Chirkova and Vassilina Nikoulina. 2024. Key ingredients for effective zero-shot cross-lingual knowledge transfer in generative tasks. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7222–7238, Mexico City, Mexico. Association for Computational Linguistics. 





Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, and 48 others. 2023. Palm: Scaling 





language modeling with pathways. J. Mach. Learn. Res., 24:240:1–240:113. 





Alexandra Chronopoulou, Jonas Pfeiffer, Joshua Maynez, Xinyi Wang, Sebastian Ruder, and Priyanka Agrawal. 2024. Language and task arithmetic with parameter-efficient layers for zero-shot summarization. In Proceedings of the Fourth Workshop on Multilingual Representation Learning (MRL 2024), pages 114–126, Miami, Florida, USA. Association for Computational Linguistics. 





Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 8440– 8451, Online. Association for Computational Linguistics. 





Alexis Conneau and Guillaume Lample. 2019. Crosslingual language model pretraining. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 7057–7067. 





Paul-Ambroise Duquenne, Hongyu Gong, Benoît Sagot, and Holger Schwenk. 2022. T-modules: Translation modules for zero-shot cross-modal machine translation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5794–5806, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Paul-Ambroise Duquenne, Holger Schwenk, and Benoît Sagot. 2023. SONAR: sentence-level multimodal and language-agnostic representations. CoRR, abs/2308.11466. 





Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. 2022. Language-agnostic BERT sentence embedding. In Proceedings of the 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 878–891, Dublin, Ireland. Association for Computational Linguistics. 





Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. 2023. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologically-diverse languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4277–4302, Toronto, Canada. Association for Computational Linguistics. 





Yaroslav Ganin and Victor S. Lempitsky. 2015. Unsupervised domain adaptation by backpropagation. In 





Proceedings ofthe 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6-11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 1180–1189. JMLR.org. 





Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics. 





Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. 2023. Imagebind one embedding space to bind them all. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 15180–15190. IEEE. 





Milan Gritta and Ignacio Iacobacci. 2021. XeroAlign: Zero-shot cross-lingual transformer alignment. In Findings ofthe Associationfor Computational Linguistics: ACL-IJCNLP 2021, pages 371–381, Online. Association for Computational Linguistics. 





Thanh-Le Ha, Jan Niehues, and Alexander Waibel. 2016. Toward multilingual neural machine translation with universal encoder and decoder. Preprint, arXiv:1611.04798. 





Raia Hadsell, Sumit Chopra, and Yann LeCun. 2006. Dimensionality reduction by learning an invariant mapping. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2006), 17-22 June 2006, New York, NY, USA, pages 1735–1742. IEEE Computer Society. 





Katharina Hämmerl, Jindˇrich Libovický, and Alexander Fraser. 2024. Understanding cross-lingual Alignment—A survey. In Findings of the Associa tionfor Computational Linguistics: ACL 2024, pages 10922–10943, Bangkok, Thailand. Association for Computational Linguistics. 





Mutian He and Philip N. Garner. 2023. Can chatgpt detect intent? evaluating large language models for spoken language understanding. In 24th Annual Conference ofthe International Speech Communication Association, Interspeech 2023, Dublin, Ireland, Au gust 20-24, 2023, pages 1109–1113. ISCA. 





Kevin Heffernan, Onur Çelebi, and Holger Schwenk. 2022. Bitext mining using distilled sentence representations for low-resource languages. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 2101–2112, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Jimin Hong, Gibbeum Lee, and Jaewoong Cho. 2024. Accelerating multilingual language model for excessively tokenized languages. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 11095–11111, Bangkok, Thailand. Association for Computational Linguistics. 





Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. 





Junjie Hu, Melvin Johnson, Orhan Firat, Aditya Siddhant, and Graham Neubig. 2021. Explicit alignment objectives for multilingual bidirectional encoders. In Proceedings of the 2021 Conference of the North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, pages 3633–3643, Online. Association for Computational Linguistics. 





Gabriel Ilharco, Marco Túlio Ribeiro, Mitchell Wortsman, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. 2023. Editing models with task arithmetic. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. 





Vivek Iyer, Bhavitvya Malik, Wenhao Zhu, Pavel Stepachev, Pinzhen Chen, Barry Haddow, and Alexandra Birch. 2024. Exploring very low-resource translation with LLMs: The University of Edinburgh‘s submission to AmericasNLP 2024 translation task. In Proceedings ofthe 4th Workshop on Natural Language Processingfor Indigenous Languages of the Americas (AmericasNLP 2024), pages 209–220, Mexico City, Mexico. Association for Computational Linguistics. 





Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Trans. Mach. Learn. Res., 2022. 





João Maria Janeiro, Benjamin Piwowarski, Patrick Gallinari, and Loïc Barrault. 2024. Mexma: Tokenlevel objectives improve sentence representations. Preprint, arXiv:2409.12737. 





Amir Hossein Kargaran, Ali Modarressi, Nafiseh Nikeghbal, Jana Diesner, François Yvon, and Hinrich Schütze. 2024. Mexa: Multilingual evaluation of english-centric llms via cross-lingual alignment. Preprint, arXiv:2410.05873. 





Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics. 





Tom Kocmi, Eleftherios Avramidis, Rachel Bawden, Ondˇrej Bojar, Anton Dvorkovich, Christian Federmann, Mark Fishel, Markus Freitag, Thamme Gowda, Roman Grundkiewicz, Barry Haddow, Philipp Koehn, Benjamin Marie, Christof Monz, Makoto Morishita, Kenton Murray, Makoto Nagata, 





Toshiaki Nakazawa, Martin Popel, and 2 others. 2023. Findings of the 2023 conference on machine translation (WMT23): LLMs are here but not quite there yet. In Proceedings ofthe Eighth Conference on Machine Translation, pages 1–42, Singapore. Association for Computational Linguistics. 





Guillaume Lample, Alexis Conneau, Marc’Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. 2018. Word translation without parallel data. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net. 





Hyunji Lee, Danni Liu, Supriti Sinhamahapatra, and Jan Niehues. 2024. How do multimodal foundation models encode text and speech? an analysis of crosslingual and cross-modal representations. Preprint, arXiv:2411.17666. 





Daoyang Li, Haiyan Zhao, Qingcheng Zeng, and Mengnan Du. 2025. Exploring multilingual probing in large language models: A cross-language analysis. Preprint, arXiv:2409.14459. 





Tianjian Li and Kenton Murray. 2023. Why does zeroshot cross-lingual generation fail? an explanation and a solution. In Findings of the Association for Computational Linguistics: ACL 2023, pages 12461–12476, Toronto, Canada. Association for Computational Linguistics. 





Ziheng Li, Shaohan Huang, Zihan Zhang, Zhi-Hong Deng, Qiang Lou, Haizhen Huang, Jian Jiao, Furu Wei, Weiwei Deng, and Qi Zhang. 2023. Dualalignment pre-training for cross-lingual sentence embedding. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3466–3478, Toronto, Canada. Association for Computational Linguistics. 





Zhuoyuan Mao and Yen Yu. 2024. Tuning LLMs with contrastive alignment instructions for machine translation in unseen, low-resource languages. In Proceedings of the Seventh Workshop on Technologies for Machine Translation ofLow-Resource Languages (LoResMT 2024), pages 1–25, Bangkok, Thailand. Association for Computational Linguistics. 





Michael Matena and Colin Raffel. 2022. Merging models with fisher-weighted averaging. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022. 





Stephen Mayhew, Terra Blevins, Shuheng Liu, Marek Suppa, Hila Gonen, Joseph Marvin Imperial, Börje Karlsson, Peiqin Lin, Nikola Ljubešic, Lester James´ Miranda, Barbara Plank, Arij Riabi, and Yuval Pinter. 2024. Universal NER: A gold-standard multilingual named entity recognition benchmark. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Lin guistics: Human Language Technologies (Volume 





1: Long Papers), pages 4322–4337, Mexico City, Mexico. Association for Computational Linguistics. 





Tomás Mikolov, Quoc V. Le, and Ilya Sutskever. 2013. Exploiting similarities among languages for machine translation. CoRR, abs/1309.4168. 





Benjamin Muller, Yanai Elazar, Benoît Sagot, and Djamé Seddah. 2021. First align, then predict: Understanding the cross-lingual ability of multilingual BERT. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 2214–2231, Online. Association for Computational Linguistics. 





NLLB Team. 2024. Scaling neural machine translation to 200 languages. Nat., 630(8018):841–846. 





Xiao Pan, Mingxuan Wang, Liwei Wu, and Lei Li. 2021. Contrastive learning for many-to-many multilingual neural machine translation. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 244–258, Online. Association for Computational Linguistics. 





Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting ofthe Associationfor Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics. 





Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In Proceedings of the 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 2227–2237, New Orleans, Louisiana. Association for Computational Linguistics. 





Aleksandar Petrov, Emanuele La Malfa, Philip H. S. Torr, and Adel Bibi. 2023. Language model tokenizers introduce unfairness between languages. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023. 





Jonas Pfeiffer, Ivan Vulic, Iryna Gurevych, and Se-´ bastian Ruder. 2020. MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7654–7673, Online. Association for Computational Linguistics. 





Ngoc-Quan Pham, Jan Niehues, Thanh-Le Ha, and Alexander Waibel. 2019. Improving zero-shot translation with language-independent constraints. In Proceedings ofthe Fourth Conference on Machine Translation (Volume 1: Research Papers), pages 13–23, 





Florence, Italy. Association for Computational Linguistics. 





Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996–5001, Florence, Italy. Association for Computational Linguistics. 





Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pages 186– 191, Brussels, Belgium. Association for Computational Linguistics. 





Kunxun Qi, Hai Wan, Jianfeng Du, and Haolan Chen. 2022. Enhancing cross-lingual natural language inference by prompt-learning from cross-lingual templates. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1910–1923, Dublin, Ireland. Association for Computational Linguistics. 





Qwen Team, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, and 24 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115. 





Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. 2017. SVCCA: singular vector canonical correlation analysis for deep learning dynamics and interpretability. In Advances in Neural Information Processing Systems 30: Annual Confer ence on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pages 6076–6085. 





Anton Razzhigaev, Matvey Mikhalchuk, Elizaveta Goncharova, Ivan Oseledets, Denis Dimitrov, and Andrey Kuznetsov. 2024. The shape of learning: Anisotropy and intrinsic dimensions in transformer-based models. In Findings ofthe Associationfor Computational Linguistics: EACL 2024, pages 868–874, St. Julian’s, Malta. Association for Computational Linguistics. 





Ricardo Rei, José G. C. de Souza, Duarte Alves, Chrysoula Zerva, Ana C Farinha, Taisiya Glushkova, Alon Lavie, Luisa Coheur, and André F. T. Martins. 2022. COMET-22: Unbabel-IST 2022 submission for the metrics shared task. In Proceedings of the Seventh Conference on Machine Translation (WMT), pages 578–585, Abu Dhabi, United Arab Emirates (Hybrid). Association for Computational Linguistics. 





Sebastian Ruder, Matthew E. Peters, Swabha Swayamdipta, and Thomas Wolf. 2019. Transfer learning in natural language processing. In Proceedings ofthe 2019 Conference ofthe North American Chapter of the Association for Computational Linguistics: Tutorials, pages 15–18, Minneapolis, Minnesota. Association for Computational Linguistics. 





Phillip Rust, Jonas Pfeiffer, Ivan Vulic, Sebastian Ruder,´ and Iryna Gurevych. 2021. How good is your tokenizer? on the monolingual performance of multilingual language models. In Proceedings of the 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3118–3135, Online. Association for Computational Linguistics. 





Tal Schuster, Ori Ram, Regina Barzilay, and Amir Globerson. 2019. Cross-lingual alignment of contextual word embeddings, with applications to zeroshot dependency parsing. In Proceedings ofthe 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 1599–1613, Minneapolis, Minnesota. Association for Computational Linguistics. 





Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong, and Francisco Guzmán. 2021. Wiki-Matrix: Mining 135M parallel sentences in 1620 language pairs from Wikipedia. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics: Main Volume, pages 1351–1361, Online. Association for Computational Linguistics. 





Junhong Shen, Neil A. Tenenholtz, James Brian Hall, David Alvarez-Melis, and Nicolò Fusi. 2024. Tagllm: Repurposing general-purpose llms for specialized domains. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net. 





Shivalika Singh, Freddie Vargus, Daniel D’souza, Börje Karlsson, Abinaya Mahendiran, Wei-Yin Ko, Herumb Shandilya, Jay Patel, Deividas Mataciunas, Laura O’Mahony, Mike Zhang, Ramith Hettiarachchi, Joseph Wilson, Marina Machado, Luisa Moura, Dominik Krzeminski, Hakimeh Fadaei, Irem´ Ergun, Ifeoma Okoh, and 14 others. 2024. Aya dataset: An open-access collection for multilingual instruction tuning. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11521– 11567, Bangkok, Thailand. Association for Computational Linguistics. 





Jörg Tiedemann. 2020. The tatoeba translation challenge – realistic data sets for low resource and multilingual MT. In Proceedings ofthe Fifth Conference on Machine Translation, pages 1174–1182, Online. Association for Computational Linguistics. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, and 49 others. 2023. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288. 





Tu Vu, Aditya Barua, Brian Lester, Daniel Cer, Mohit Iyyer, and Noah Constant. 2022. Overcoming catastrophic forgetting in zero-shot cross-lingual gen eration. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 9279–9300, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics. 





Dong Wang and Thomas Fang Zheng. 2015. Transfer learning for speech and language processing. In Asia-Pacific Signal and Information Processing Association Annual Summit and Conference, APSIPA 2015, Hong Kong, December 16-19, 2015, pages 1225–1237. IEEE. 





Hetong Wang, Pasquale Minervini, and Edoardo Ponti. 2024a. Probing the emergence of cross-lingual alignment during LLM training. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 12159–12173, Bangkok, Thailand. Association for Computational Linguistics. 





Weixuan Wang, Minghao Wu, Barry Haddow, and Alexandra Birch. 2024b. Bridging the language gaps in large language models with inference-time crosslingual intervention. Preprint, arXiv:2410.12462. 





Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, An drew M. Dai, and Quoc V. Le. 2022. Finetuned language models are zero-shot learners. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net. 





Di Wu, Yibin Lei, Andrew Yates, and Christof Monz. 2024a. Representational isomorphism and alignment of multilingual large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14074–14085, Miami, Florida, USA. Association for Computational Linguistics. 





Zhaofeng Wu, Xinyan Velocity Yu, Dani Yogatama, Jiasen Lu, and Yoon Kim. 2024b. The semantic hub hypothesis: Language models share semantic representations across languages and modalities. Preprint, arXiv:2411.04986. 





Haoran Xu, Young Jin Kim, Amr Sharaf, and Hany Hassan Awadalla. 2024. A paradigm shift in machine translation: Boosting translation performance of large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net. 





Haoran Xu and Philipp Koehn. 2021. Cross-lingual bert contextual embedding space mapping with isotropic and isometric conditions. Preprint, arXiv:2107.09186. 





Rong Ye, Mingxuan Wang, and Lei Li. 2022. Crossmodal contrastive learning for speech translation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5099–5113, Seattle, United States. Association for Computational Linguistics. 





Katherine Yu, Haoran Li, and Barlas Oguz. 2018. Multilingual seq2seq training with similarity loss for crosslingual document classification. In Proceedings of the Third Workshop on Representation Learningfor NLP, pages 175–179, Melbourne, Australia. Association for Computational Linguistics. 





Fei Yuan, Shuai Yuan, Zhiyong Wu, and Lei Li. 2024. How vocabulary sharing facilitates multilingualism in LLaMA? In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 12111–12130, Bangkok, Thailand. Association for Computational Linguistics. 





Meng Zhang, Yang Liu, Huanbo Luan, and Maosong Sun. 2017. Adversarial training for unsupervised bilingual lexicon induction. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1959–1970, Vancouver, Canada. Association for Computational Linguistics. 





Chengzhi Zhong, Fei Cheng, Qianying Liu, Junfeng Jiang, Zhen Wan, Chenhui Chu, Yugo Murawaki, and Sadao Kurohashi. 2024. Beyond english-centric llms: What language do multilingual language models think in? Preprint, arXiv:2408.10811. 



## A English-Only Fine-Tuning Results

Table 9 compares English-only and multilingual fine-tuning on MASSIVE. Multilingual finetuning substantially outperforms English-only in cross-lingual transfer performance. 

## B Dataset Details

All our task training data are retrieved from HuggingFace<sup>14</sup>. The translation test sets are hosted by WMT<sup>15</sup>. The alignment data are sourced from Tatoeba<sup>16</sup> with its default version of v2021-07-22 at the time of writing. We filter out translations that are empty or include multiple sentences. The lowest-resource alignment languages have a few hundred parallel sentences: Javanese (264), Swahili (371), Welsh (823). The ablation de-en alignment data is from IWSLT 2017<sup>17</sup> (Cettolo et al., 2017). 



<sup>14</sup>MASSIVE: https://huggingface.co/datasets/ AmazonScience/massive ALMA: https://huggingface.co/datasets/haoranxu/ ALMA-Human-Parallel UNER: https://huggingface.co/datasets/ CohereForAI/aya_collection/viewer/templated_ uner_llm 



<table><tr><td></td><td>ar</td><td>en</td><td>es</td><td>ru</td><td>zh</td><td>cy</td><td>ja</td><td>jv</td><td>sw</td><td>tl</td><td>af</td><td>az</td><td>de</td><td>el</td><td>fr</td><td>hi</td><td>is</td><td>th</td><td>tr</td><td>ur</td></tr><tr><td>English-only</td><td>59.8</td><td>82.5</td><td>82.4</td><td>65.8</td><td>61.6</td><td>60.3</td><td>39.7</td><td>37.8</td><td>39.8</td><td>57.5</td><td>60.3</td><td>39.6</td><td>71.1</td><td>64.8</td><td>68.2</td><td>62.1</td><td>39.2</td><td>75.3</td><td>52.9</td><td>49.9</td></tr><tr><td>Multilingual</td><td>75.5</td><td>81.7</td><td>74.5</td><td>77.6</td><td>73.8</td><td>44.0</td><td>65.8</td><td>41.0</td><td>42.8</td><td>65.0</td><td>66.0</td><td>49.0</td><td>75.0</td><td>69.4</td><td>71.9</td><td>70.0</td><td>45.0</td><td>79.9</td><td>60.4</td><td>57.1</td></tr></table>


Table 9: Per-languages $\mathrm { F _ { 1 } }$ results on slot filling of English-only finetuning compared to multilingual fine-tuning on {ar, en, es, ru, zh}. Multilingual fine-tuning shows stronger transfer performance.


<table><tr><td>Code</td><td>FLoRes Code</td><td>Full Name</td><td>Slot Filling</td><td>Machine Translation</td><td>JSON Generation</td></tr><tr><td>af</td><td>afr_Latn</td><td>Afrikaans</td><td>√</td><td></td><td></td></tr><tr><td>az</td><td>azj_Latn</td><td>North Azerbaijani</td><td>√</td><td></td><td></td></tr><tr><td>ar</td><td>arb_Arab</td><td>Modern Standard Arabic</td><td>√</td><td></td><td></td></tr><tr><td>cs</td><td>ces_Latn</td><td>Czech</td><td></td><td>√</td><td></td></tr><tr><td>cy</td><td>cym_Latn</td><td>Welsh</td><td>√</td><td></td><td></td></tr><tr><td>da</td><td>dan_Latn</td><td>Danish</td><td></td><td></td><td>√</td></tr><tr><td>de</td><td>deu_Latn</td><td>German</td><td>√</td><td>√</td><td></td></tr><tr><td>el</td><td>ell_Grek</td><td>Greek</td><td>√</td><td></td><td></td></tr><tr><td>en</td><td>eng_Latn</td><td>English</td><td>√</td><td>√</td><td>√</td></tr><tr><td>es</td><td>spa_Latn</td><td>Spanish</td><td>√</td><td></td><td></td></tr><tr><td>fr</td><td>fra_Latn</td><td>French</td><td>√</td><td></td><td></td></tr><tr><td>he</td><td>heb_Hebr</td><td>Hebrew</td><td></td><td>√</td><td></td></tr><tr><td>hi</td><td>hin_Deva</td><td>Hindi</td><td>√</td><td></td><td></td></tr><tr><td>hr</td><td>hrv_Latn</td><td>Croatian</td><td></td><td></td><td>√</td></tr><tr><td>is</td><td>isl_Latn</td><td>Icelandic</td><td>√</td><td>√</td><td></td></tr><tr><td>ja</td><td>jpn_Jpan</td><td>Japanese</td><td>√</td><td>√</td><td></td></tr><tr><td>jv</td><td>jav_Latn</td><td>Javanese</td><td>√</td><td></td><td></td></tr><tr><td>pt</td><td>por_Latn</td><td>Portuguese</td><td></td><td></td><td>√</td></tr><tr><td>ru</td><td>rus_Cyrl</td><td>Russian</td><td>√</td><td>√</td><td></td></tr><tr><td>sk</td><td>slk_Latn</td><td>Slovak</td><td></td><td></td><td>√</td></tr><tr><td>sr</td><td>srp_Cyrl</td><td>Serbian</td><td></td><td></td><td>√</td></tr><tr><td>sv</td><td>swe_Latn</td><td>Swedish</td><td></td><td></td><td>√</td></tr><tr><td>sw</td><td>swh_Latn</td><td>Swahili</td><td>√</td><td></td><td></td></tr><tr><td>th</td><td>tha_Thai</td><td>Thai</td><td>√</td><td></td><td></td></tr><tr><td>tl</td><td>tgl_Latn</td><td>Tagalog</td><td>√</td><td></td><td></td></tr><tr><td>tr</td><td>tur_Latn</td><td>Turkish</td><td>√</td><td></td><td></td></tr><tr><td>uk</td><td>ukr_Cyrl</td><td>Ukrainian</td><td></td><td>√</td><td></td></tr><tr><td>ur</td><td>urd_Arab</td><td>Urdu</td><td>√</td><td></td><td></td></tr><tr><td>zh</td><td>zho_Hans</td><td>Chinese (Simplified)</td><td>√</td><td>√</td><td>√</td></tr></table>


Table 10: List of languages evaluated on different downstream tasks.


## C List of Languages

The languages involved in our downstream tasks are listed in Table 10. The 35 languages in the initial analyses in §2 include all languages in slot fill and machine translation. They additionally include the following languages: am (Amharic), bn (Bengali), it (Italian), hu (Hungrian), hy (Armenian), id (Indonesian), kn (Kannada), ka (Georgian ), mn (Mongolian), km (Khmer), ko (Korean), and lv (Latvian). 

## D Training and Inference Details

## D.1 Training Hyperparameters

Fine-tuning is performed using LoRA (Hu et al., 2022) adapters with a rank of 8 for all attention components and linear projections (query, key, value, output, gate, up, down). We set LoRA’s α parameter to 16 and dropout to 0.1. The number of trainable parameter is 20,971,520 on Llama 3, and 20,185,088 on Qwen 2.5. We train at most 5 epochs on the task data. Training on all our tasks converged before reaching the max number of epochs. The learning rate is set to 5e-4 with inverse square root schedule and warmup up ratio 0.03. We save checkpoints and evaluate every 200 optimization steps, and early stop if the development loss does not improve for 5 consecutive evaluations. For the temperature parameter τ in the contrastive loss, we searched among {0.1, 1.0, 1.5, 2.0} based on development loss on machine translation. For Llama we 0.1, for Qwen we use 1.5. 

## D.2 Prompt Format

Slot Filling The system prompt is shortened from He and Garner (2023).<sup>18</sup> 

• System: Given a command from the user, a voice assistant will extract entities essential for carry out the command. Your task is to extract the entities as words from the command if they fall under a predefined list of entity types. 

• User: wake me up at five am this week 

• Assistant: time: five am; date: this week 

• User (de): wecke mich in dieser woche um fünf uhr auf 

• Assistant (de): date: dieser woche; time: fünf uhr 

For zero-shot slot filling experiments, we need to specify more requirements in the system prompt with the template also following He and Garner (2023): 

Given a command from the user, a voice assistant like Siri or Olly will extract entities from the command that are essential for carry out the the command. For example, for a command about playing a specific song, the name of the song mentioned by the user would be an entity, falling under the type of “song name”. 

Your task is to extract the entities as words from the command if they fall under any of the types given below according to the following description: 

transport_descriptor house_place music_album sport_type playlist_name movie_name song_name place_name radio_name cooking_type weather_descriptor person email_folder business_type audiobook_author transport_type general_frequency meal_type game_name device_type transport_name time_zone joke_type drink_type email_address food_type date relation currency_name ingredient player_setting movie_type definition_word game_type list_name artist_name personal_info audiobook_name timeofday transport_agency media_type podcast_name coffee_type business_name news_topic app_name podcast_descriptor color_type music_genre event_name time change_amount alarm_type order_type music_descriptor 

Please give answers like: 

1. person: john; contact_field: phone number 

2. transport_app: uber; time_of_day: tonight; time: ten pm 

3. None 

## 4. music_genre: jazz

etc., each taking a single line. The entity type must be one of the types given above, and the entity must be copied verbatim from the command. There could be zero, one, or multiple entities in a command. 

## Machine Translation

• System: Translate the following sentences from English to German. 

• User: Police arrest 15 after violent protest outside UK refugee hotel. 

• Assistant: Polizei verhaftet 15 Menschen nach gewalttätigen Protesten vor einer Flüchtlingsunterkunft in Großbritannien 

## JSON Generation

• User: Please identify all the named entities mentioned in the input sentence provided below. Use only the categories: PER - person, ORG - organization, and LOC - location. Remember, nationalities are neither locations nor organizations, and organizations can represent other groups of people. Pay attention to the provided example. You should only output the results in JSON format, following a similar structure to the example result provided. Example sentence and results: Where in the world is Iguazu? "Results": [ "TypeName": "LOC", "Text": "Iguazu", "Start": 22, "End": 28 ] Considering the input sentence below, what is the output result? Widely considered to be one of the most spectacular waterfalls in the world, the Iguazu Falls on the border of Argentina and Brazil, are a certainly must see attraction in the area. 

• Assistant: "Results": [ "TypeName": "LOC", "Text": "Iguazu Falls", "Start": 81, "End": 93 , "TypeName": "LOC", "Text": "Argentina", "Start": 111, "End": 120 , "TypeName": "LOC", "Text": "Brazil", "Start": 125, "End": 131 ] 

## D.3 Inference Details

We use greedy decoding in all experiments for easily reproducible results. For the model merging experiments, we searched among weights {0.5, 0.7, 0.9} for the task-specific LoRA modules on the MASSIVE development set and chose 0.9 for our experiments. 

<table><tr><td rowspan="2"></td><td colspan="5">Supervised</td><td colspan="6">Transfer (aligned)</td><td colspan="9">Transfer (other)</td></tr><tr><td>ar</td><td>en</td><td>es</td><td>ru</td><td>zh</td><td>cy</td><td>ja</td><td>jv</td><td>sw</td><td>tl</td><td>af</td><td>az</td><td>de</td><td>el</td><td>fr</td><td>hi</td><td>is</td><td>th</td><td>tr</td><td>ur</td></tr><tr><td>Llama 3 SFT</td><td>75.5</td><td>81.7</td><td>74.5</td><td>77.6</td><td>73.8</td><td>44.0</td><td>65.8</td><td>41.0</td><td>42.8</td><td>65.0</td><td>66.0</td><td>49.0</td><td>75.0</td><td>69.4</td><td>71.9</td><td>70.0</td><td>45.0</td><td>79.9</td><td>60.4</td><td>57.1</td></tr><tr><td>+ align (middle)</td><td>75.1</td><td>82.0</td><td>74.9</td><td>78.0</td><td>74.9</td><td>49.4</td><td>66.5</td><td>48.2</td><td>47.7</td><td>65.5</td><td>66.2</td><td>47.9</td><td>74.7</td><td>72.4</td><td>72.1</td><td>69.6</td><td>48.0</td><td>79.1</td><td>62.2</td><td>56.1</td></tr><tr><td>Qwen 2.5 SFT</td><td>74.7</td><td>81.1</td><td>74.0</td><td>77.5</td><td>74.1</td><td>27.0</td><td>67.3</td><td>32.9</td><td>23.5</td><td>57.4</td><td>58.9</td><td>45.9</td><td>74.6</td><td>63.3</td><td>70.8</td><td>60.0</td><td>34.4</td><td>79.9</td><td>59.9</td><td>46.5</td></tr><tr><td>+ align (middle)</td><td>74.9</td><td>82.5</td><td>74.8</td><td>78.0</td><td>75.1</td><td>36.5</td><td>68.3</td><td>39.6</td><td>30.4</td><td>57.8</td><td>63.1</td><td>42.5</td><td>74.6</td><td>63.3</td><td>70.9</td><td>61.3</td><td>35.8</td><td>80.2</td><td>58.1</td><td>47.2</td></tr></table>


Table 11: Per-languages $\mathrm { F _ { 1 } }$ results on slot filling.


<table><tr><td rowspan="2"></td><td colspan="5">Supervised X→En</td><td colspan="5">Supervised En→X</td><td colspan="3">Transfer X→En</td><td colspan="3">Transfer En→X</td></tr><tr><td>cs</td><td>de</td><td>is</td><td>ru</td><td>zh</td><td>cs</td><td>de</td><td>is</td><td>ru</td><td>zh</td><td>he</td><td>ja</td><td>uk</td><td>he</td><td>ja</td><td>uk</td></tr><tr><td colspan="17">BLEU</td></tr><tr><td>Llama 3 SFT</td><td>37.8</td><td>43.0</td><td>28.3</td><td>32.0</td><td>22.5</td><td>25.9</td><td>35.5</td><td>10.6</td><td>25.2</td><td>38.9</td><td>39.3</td><td>17.5</td><td>38.7</td><td>14.5</td><td>14.2</td><td>17.7</td></tr><tr><td>+ align (middle)</td><td>38.4</td><td>43.1</td><td>29.1</td><td>32.4</td><td>23.0</td><td>24.7</td><td>34.7</td><td>10.9</td><td>24.4</td><td>38.1</td><td>39.8</td><td>18.8</td><td>38.4</td><td>16.0</td><td>15.6</td><td>19.5</td></tr><tr><td>Qwen 2.5 SFT</td><td>36.1</td><td>40.8</td><td>20.5</td><td>30.6</td><td>23.2</td><td>21.5</td><td>33.7</td><td>6.8</td><td>25.3</td><td>45.3</td><td>34.6</td><td>18.9</td><td>35.6</td><td>13.3</td><td>17.6</td><td>13.0</td></tr><tr><td>+ align (middle)</td><td>36.6</td><td>41.4</td><td>21.2</td><td>30.9</td><td>24.0</td><td>20.5</td><td>32.7</td><td>4.8</td><td>25.0</td><td>45.3</td><td>36.3</td><td>19.4</td><td>36.8</td><td>12.7</td><td>17.8</td><td>13.5</td></tr><tr><td colspan="17">COMET</td></tr><tr><td>Llama 3 SFT</td><td>85.2</td><td>84.9</td><td>81.0</td><td>82.4</td><td>79.7</td><td>84.3</td><td>81.8</td><td>68.7</td><td>83.3</td><td>84.2</td><td>83.6</td><td>79.8</td><td>85.1</td><td>75.7</td><td>83.5</td><td>79.7</td></tr><tr><td>+ align (middle)</td><td>85.5</td><td>84.9</td><td>81.1</td><td>82.4</td><td>79.8</td><td>83.8</td><td>81.6</td><td>69.0</td><td>83.3</td><td>84.0</td><td>83.6</td><td>80.1</td><td>85.2</td><td>77.1</td><td>84.2</td><td>80.8</td></tr><tr><td>Qwen 2.5 SFT</td><td>84.8</td><td>84.7</td><td>74.1</td><td>82.6</td><td>80.2</td><td>80.8</td><td>80.6</td><td>52.0</td><td>83.3</td><td>86.1</td><td>82.3</td><td>81.3</td><td>84.5</td><td>70.7</td><td>85.5</td><td>74.6</td></tr><tr><td>+ align (middle)</td><td>85.1</td><td>84.7</td><td>74.4</td><td>82.6</td><td>80.4</td><td>79.5</td><td>80.1</td><td>46.5</td><td>83.1</td><td>85.8</td><td>82.2</td><td>81.4</td><td>84.6</td><td>70.7</td><td>85.7</td><td>74.4</td></tr></table>


Table 12: Per-languages BLEU and COMET results on machine translation.


<table><tr><td></td><td>Supervised↑</td><td>Transfer↑</td></tr><tr><td colspan="3">Slot filling (MASSIVE): <eq>F_1</eq></td></tr><tr><td>SFT baseline</td><td>76.6</td><td>60.2</td></tr><tr><td>Middle (layer 16)</td><td>77.0</td><td>61.7</td></tr><tr><td>Top (layer 32)</td><td>76.6</td><td>62.0</td></tr><tr><td>Bottom (layer 8)</td><td>76.8</td><td>58.0</td></tr><tr><td>Middle + Bottom</td><td>77.6</td><td>62.5</td></tr><tr><td colspan="3">Machine translation (WMT23): COMET</td></tr><tr><td>SFT baseline</td><td>81.5</td><td>79.6</td></tr><tr><td>Middle (layer 16)</td><td>81.5</td><td>80.7</td></tr><tr><td>Top (layer 32)</td><td>82.0</td><td>80.2</td></tr><tr><td>Bottom (layer 8)</td><td>81.2</td><td>80.1</td></tr><tr><td>Middle + Bottom</td><td>81.5</td><td>80.6</td></tr></table>


provements on slot filling, but does not on machine translation. This task-dependent behavior indicates that how to best align multiple layers still requires further investigation. 



Table 13: Impact of alignment loss placement on supervised and transfer performance on Llama 3.


## D.4 Details for Retrieval

To evaluate cross-lingual retrieval performance, we adapt the implementation from LASER<sup>19</sup> (Schwenk et al., 2021) to process representations extracted offline. 

## E Results for Individual Languages

The detailed results for Table 2 are in Table 11 (slot filling) and Table 12 (machine translation). 

## F Results of Aligning at Several Layers

In Table 13, we show that adding alignment losses at both middle and top layers brings further im-