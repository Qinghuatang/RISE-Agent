# Pre-Training Curriculum for Multi-Token Prediction in Language Models

Ansar Aynetdinov Humboldt-Universität zu Berlin aynetdia@hu-berlin.de 

Alan Akbik Humboldt-Universität zu Berlin alan.akbik@hu-berlin.de 

## Abstract

Multi-token prediction (MTP) is a recently proposed pre-training objective for language models. Rather than predicting only the next token (NTP), MTP predicts the next k tokens at each prediction step, using multiple prediction heads. MTP has shown promise in improving downstream performance, inference speed, and training efficiency, particularly for large models. However, prior work has shown that smaller language models (SLMs) struggle with the MTP objective. To address this, we propose a curriculum learning strategy for MTP training, exploring two variants: a forward curriculum, which gradually increases the complexity of the pre-training objective from NTP to MTP, and a reverse curriculum, which does the op posite. Our experiments show that the forward curriculum enables SLMs to better leverage the MTP objective during pre-training, improving downstream NTP performance and generative output quality, while retaining the benefits of self-speculative decoding. The reverse curriculum achieves stronger NTP performance and output quality, but fails to provide any selfspeculative decoding benefits. 

## 1 Introduction

In recent years, large language models (LLMs) have demonstrated remarkable capabilities in understanding and generating complex text, code, and other modalities. These advances have been driven primarily by improvements in model architectures, training data quality and scale, and optimization strategies, and yet the underlying objective for most widely-adopted LLMs remains the next-token prediction loss. Models such as GPT (Brown et al., 2020; OpenAI et al., 2024), LLaMA (Touvron et al., 2023a,b; Grattafiori et al., 2024), and Phi (Li et al., 2023; Abdin et al., 2024) are conventionally trained to predict a single token at each generation step, implicitly treating language modeling as a sequence of one-step-ahead predictions. While this formulation is simple and effective, it may not fully leverage the predictive capabilities of LLMs or reflect the underlying structure of natural language. 

Gloeckle et al. (2024) investigated the potential of a multi-token prediction (MTP) objective for LLMs. Instead of predicting just the next token (NTP), their approach aims to produce multiple subsequent tokens at each prediction step, with multiple output heads that are independent from each other but share a common model backbone. They found this approach to improve model’s downstream performance, inference speed, and training sample efficiency without significantly increasing training time. Recently, DeepSeek-AI adopted the MTP training objective for their V3 model (Liu et al., 2024a) that serves as a base model for the reasoning R1 model (Guo et al., 2025). 

So far, the previous work prioritized mid- and large-sized models with at least 7B parameters, since Gloeckle et al. (2024) observed that MTP leads to more performance gains as the model size increases. We hypothesize that smaller language models (SLMs) enjoy less benefits from the MTP objective during pre-training due to the fact that they struggle to deal with morphological and semantic dependencies between multiple tokens at once from the get-go. 

A curriculum-based approach to MTP. Inspired by the work of Bengio et al. (2009), who were the first to propose curriculum learning strategies in the context of machine learning, we explore using a pre-training curriculum to enable SLMs to better leverage the benefits of the MTP objective. Since the complexity of the MTP objective increases with the amount of tokens considered, it is easy to construct a curriculum by gradually changing the number of tokens considered in the objective. 

Figure 1 illustrates our proposed curriculum variants. Under a forward curriculum, models are given a simpler NTP task in the beginning of pre-training, and as the training progresses, they are guided towards a more complex MTP task. The task is expanded by incrementally including one additional token at uniform intervals throughout pre-training. The reverse curriculum, on the other hand, starts with a full MTP task, and incrementally removes one token from the task also at uniform intervals. The two opposing curricula are designed to clearly determine whether MTP SLMs perform better with an easy-to-hard or a hard-to-easy learning strategy. Contributions. This work makes the following contributions: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/2f973294-625b-432b-b54d-f8254456fceb/48ccd0a56b580ded92eaa2874fbc065d40351271199a742fc8574b827a813783.jpg)



Figure 1: Visualization of the forward and reverse MTP curricula. When training a language model on a 3-tokenprediction task for n steps, the forward curriculum starts with a vanilla NTP task, adding an additional token to the task every $\frac { n } { 3 }$ steps. The reverse curriculum does the opposite, starting with a full 3-token-prediction task, and dropping a token from the task every $\frac { n } { 3 }$ steps.


• We provide a comprehensive exploration of the MTP objective’s potential when used to train SLMs predominantly on natural language. 

• We propose and evaluate the validity of novel curriculum-based MTP training strategies. 

• We showcase that curriculum-based (dynamic) MTP pre-training strategies allow SLMs to learn the MTP task better than static MTP baselines. 

## 2 Preliminaries

In this section, we formalize the multi-token prediction objective and describe the design of the proposed curricula. 

## 2.1 Multi-Token Prediction Objective

Traditional large language models are trained using the next-token prediction (NTP) objective, where the model, given a context sequence x = $( x _ { 1 } , x _ { 2 } , \ldots , x _ { t } )$ , is tasked with predicting the next token $x _ { t + 1 }$ . This is accomplished by maximizing the log-likelihood of the target token: 

$$
\mathcal {L} _ {\mathrm{NTP}} = - \sum_ {t = 1} ^ {T} \log P (x _ {t + 1} | x _ {1}, \ldots , x _ {t}; \theta),
$$

where $\theta$ represents the model parameters. 

In contrast, the multi-token prediction (MTP) objective extends this task by requiring the model to predict a sequence of k tokens ${ \textbf { y } } =$ $( x _ { t + 1 } , x _ { t + 2 } , \ldots , x _ { t + k } )$ simultaneously. The MTP loss is defined as: 

$$
\mathcal {L} _ {\mathrm{MTP}} = - \sum_ {t = 1} ^ {T} \sum_ {i = 1} ^ {k} \log P (x _ {t + i} | x _ {1}, \dots , x _ {t}; \theta),
$$

where $P ( x _ { t + i } | x _ { 1 } , \dots , x _ { t } ; \theta )$ is computed using multiple output, or language modeling, heads conditioned on the shared model backbone. 

Predicting multiple tokens at each prediction step allows for self-speculative decoding (Stern et al., 2018) during inference without the need for an auxiliary model. This can speed up the inference by up to k times by reducing the number of forward passes needed to generate a sequence. In addition, the MTP objective promotes the learning of richer contextual dependencies, as the model must optimize for multiple interrelated predictions at each step. 

## 2.2 Forward and Reverse Multi-Token Curricula

The design of the curricula dynamically adjusts the number of output, or language modeling, heads $k ~ \in ~ \{ 1 , 2 , \dots , k _ { \operatorname* { m a x } } \}$ $k \in \mathbb { Z }$ , which corresponds to the number of tokens predicted by the model at each prediction step. These adjustments are based on the training epoch $e ,$ ensuring systematic changes every $E / k _ { \mathrm { m a x } }$ epochs, where $E$ is the total number of training epochs. While it is certainly possible to come up with a more adaptive curriculum that relies on the training dynamics, we aim to isolate and demonstrate the core benefit of using a curriculum-based MTP objective over a static one. Therefore in this paper we decide to focus on a simple pre-defined curriculum design. 

## 2.2.1 Forward Multi-Token Curriculum

The forward multi-token curriculum starts with the simplest prediction task $( k = 1 )$ and progressively increases the number of prediction heads. Formally, the number of active output heads at training epoch e can be defined as: 

$$
k _ {\mathrm{current}} (e) = \min \left(k _ {\max}, \left\lfloor \frac {e}{E / k _ {\max}} \right\rfloor + 1\right).
$$

This gradual increase in complexity ensures that the model first learns fundamental token-by-token predictions before transitioning to more complex multi-token prediction tasks. 

## 2.2.2 Reverse Multi-Token Curriculum

In contrast, the reverse multi-token curriculum starts with the maximum number of prediction heads $( k = k _ { \operatorname* { m a x } } )$ and gradually reduces the number of active heads over time. The number of active prediction heads at training epoch e is defined as: 

$$
k _ {\mathrm{current}} (e) = \max \left(1, k _ {\mathrm{max}} - \left\lfloor \frac {e}{E / k _ {\mathrm{max}}} \right\rfloor\right).
$$

This strategy relies on the insight reported by Gloeckle et al. (2024), that pre-training with a multi-token objective improves performance on single-token downstream tasks. We hypothesize that by gradually nudging the model towards NTP during the pre-training stage, we will be able to obtain a better main language modeling head. 

## 3 Experiments

In this section, we aim to address the following 3 aspects of our proposed curricula: 

• Main LM head performance. Since in both the forward and reverse curricula the MTP objective aligns with the NTP objective at some point during pre-training, we examine how the resulting models perform on standard NTP benchmarks using a single (main) language modeling (LM) head. 

• Inference Speed. One of the main advantages of the ability to output multiple tokens in one prediction step is the fact that it can be used to significantly speed-up the inference by performing self-speculative decoding. Therefore we measure whether the proposed curricula can offer additional speed-ups during inference. 

• Output Quality. Lastly, we evaluate generative capabilities of the models trained under our proposed curricula, i.e., the quality of the sequences that they generate. In the end, this is the most important aspect of language models, and therefore should be the main indicator of how useful the proposed curricula are in the context of multitoken prediction. 

## 3.1 Experimental Setup

## 3.1.1 Models

We conduct our experiments using transformerbased decoder-only LLMs, specifically the configuration used in the Llama model family (Touvron et al., 2023b). Regarding the model size, token types, and language modeling heads enabling the MTP, we follow the insights provided by Gloeckle et al. (2024) in order to achieve best possible performance with minimal resource requirements. 

We consider models of two sizes: 1.3B and 3B. We train our models on both subword and byte tokens. Byte-level models operate directly on raw bytes (e.g., UTF-8 encoded text), enabling them to handle diverse character sets and misspellings more robustly, while also reducing vocabulary size and memory usage (Xue et al., 2022). The resulting vocabulary sizes, or model output dimensionality, for each type of models are 32,000 and 320 respectively. In order to maintain relative similarity in the amount of training steps that models undergo during the pre-training stage, we limit the context window size of subword-level models to 1024 and that of byte-level models to 4096. 

## 3.1.2 Prediction Heads

As for the definition of additional language modeling heads, we consider the following two setups in our experiments: 

• Linear Layers (LL): We add additional $k -$ 1 output linear layers with $h i d d e n \_ s i z e \ x$ vocab_size dimensions to a regular NTP architecture in case of a k-token prediction model. This essentially translates to linear probing of the model’s hidden states. 

While this setup allows for predictions that can be obtained in parallel, and thus not leading to any computational overheads with regards to output latency, it introduces an additional memory overhead. For subword-level models, this leads to additional 65M parameters per head in case of a 1.3B model, and additional 98M parameters per head in case of a 3B model. For byte-level models, this setup translates to only 0.6M and 1M additional trainable parameters per head respectively. 

• Transformer Layers (TL): We use l k transformer layers as the model backbone and dedicate k out of l layers to handling token prediction in the latent space before passing them through the shared output linear layer. 

This setup not only does not introduce any additional trainable parameters, but also reduces the parameter count of the shared backbone by 51M (1.3B model) or 113M (3B model). 

## 3.1.3 Dataset

We conduct our experiments on the MiniPile dataset (Kaddour, 2023). It is a subset of The Pile corpus (Gao et al., 2020)<sup>1</sup> that consists of 1M documents in the training split and 10K documents in the test split. 

We train a dedicated BPE subword tokenizer (Sennrich et al., 2016) with a vocabulary size of 32K on this dataset, which yields about 1.7B subword tokens for training. The corresponding training byte count is around 5.9B. 

An important note regarding the discussion of our results is the fact that our models cannot be directly compared to other publicly released LLMs, given the size of the dataset. Our goal was not to train competitive LLMs, but rather to evaluate the viability of our proposed pre-training curricula. 

In order to confirm that our results hold true as the scale increases, we trained additional 1.3B MTP models with 4 TL heads together with a baseline 1.3B NTP model on 10B tokens of FineWeb-Edu (Lozhkov et al., 2024). The performance of these models is shown in Table 5 of the Appendix B. 

## 3.1.4 Training setup & Hyperparameters

We apply Best Fit Bin Packing (Albers et al., 2021) to the train split of the MiniPile in order to prepare model inputs for the pre-training stage. We then train all of our models for 1 epoch. The training batch size is 1024. This setup allows us to iterate quickly over various model configurations. 

In all experiments we schedule the learning rates with a linear warmup lasting 10% of total training steps and cosine decay (Loshchilov and Hutter, 2017) to 10% of the peak learning rate, which is 2e 4 for all models. All experiments use the AdamW optimizer (Loshchilov and Hutter, 2019) with $\beta _ { 1 } ~ = ~ 0 . 9 , ~ \beta _ { 2 } ~ = ~ 0 . 9 5$ and decoupled $L _ { 2 }$ weight decay coefficient of 0.1. We also clip gradients to a maximal Euclidean norm of 1.0 in all experiments. 

We plan to release the code used to train the models under our proposed curricula in our Github repository<sup>2</sup>. 

## 3.2 Experiment 1: Main LM Head Performance

## 3.2.1 Setup

In this experiment we evaluate the model’s NTP performance on the test set of the MiniPile dataset and on the OpenAI’s version of the LAMBADA benchmark (Paperno et al., 2016). We report the bits-per-byte metric instead of token perplexities, to allow for a direct comparison between subwordand byte-level models in our experiments. We use EleutherAI’s evaluation harness (v0.4.3) (Gao et al., 2024) to run our models on the benchmarks. 

Since we are training our models on a significantly limited amount of data compared to what LLMs are usually trained on, we believe that widely used knowledge-based tasks like e.g. MMLU (Hendrycks et al., 2021) would not be particularly meaningful to analyze their performance, as they training dataset does not cover an extended amount of factual knowledge. Thus we opt to focus on the BLiMP benchmark (Warstadt et al., 2020), since it allows for a fairer evaluation of the linguistic capabilities of our models. 

Nevertheless, we also include the evaluation on the easy set of the ARC challenge (ARC-E) (Clark et al., 2018) and on the OpenBookQA (OBQA) benchmark (Mihaylov et al., 2018) for the sake of a more comprehensive evaluation, as they focus on commonsense knowledge and reasoning, rather than area-specific knowledge. However we would like to note that these benchmarks do not provide an assessment of the underlying predictive capabilities associated with predicting multiple next tokens. 

<table><tr><td rowspan="2">Tokens</td><td rowspan="2">Heads</td><td rowspan="2">Curriculum</td><td colspan="5">1.3B</td><td colspan="5">3B</td></tr><tr><td>MiniPile BPB ↓</td><td>LAMBADA BPB ↓</td><td>BLiMP Acc. ↑</td><td>ARC-E Acc. ↑</td><td>OBQA Acc. ↑</td><td>MiniPile BPB ↓</td><td>LAMBADA BPB ↓</td><td>BLiMP Acc. ↑</td><td>ARC-E Acc. ↑</td><td>OBQA Acc. ↑</td></tr><tr><td rowspan="13">Subword</td><td>1 LL</td><td>-</td><td>1.08</td><td>1.34</td><td>71.80</td><td>34.60</td><td>25.80</td><td>1.03</td><td>1.17</td><td>74.93</td><td>34.60</td><td>24.80</td></tr><tr><td rowspan="3">2 LL</td><td>-</td><td>1.12</td><td>1.44</td><td>70.27</td><td>31.99</td><td>23.80</td><td>1.06</td><td>1.27</td><td>72.09</td><td>36.36</td><td>24.60</td></tr><tr><td>Forward</td><td>+0.15%</td><td>+0.37%</td><td>+0.75%</td><td>+8.29%</td><td>+7.56%</td><td>+0.46%</td><td>+0.01%</td><td>+2.16%</td><td>-2.66%</td><td>+3.25%</td></tr><tr><td>Reverse</td><td>-2.19%</td><td>-5.79%</td><td>+1.69%</td><td>+1.84%</td><td>+5.04%</td><td>-1.78%</td><td>-4.91%</td><td>+3.60%</td><td>-0.23%</td><td>-1.62%</td></tr><tr><td rowspan="3">2 TL</td><td>-</td><td>1.11</td><td>1.41</td><td>71.04</td><td>34.09</td><td>26.20</td><td>1.05</td><td>1.24</td><td>73.32</td><td>36.15</td><td>25.00</td></tr><tr><td>Forward</td><td>+0.49%</td><td>+1.82%</td><td>-0.45%</td><td>+3.09%</td><td>-4.58%</td><td>+0.88%</td><td>+0.26%</td><td>-0.08%</td><td>-1.98%</td><td>-1.60%</td></tr><tr><td>Reverse</td><td>-1.98%</td><td>-4.19%</td><td>+0.88%</td><td>+2.10%</td><td>0.00%</td><td>-1.75%</td><td>-3.75%</td><td>+0.83%</td><td>-3.03%</td><td>+0.80%</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>1.19</td><td>1.61</td><td>67.48</td><td>34.22</td><td>24.40</td><td>1.12</td><td>1.44</td><td>72.20</td><td>36.45</td><td>25.00</td></tr><tr><td>Forward</td><td>-2.46%</td><td>-3.54%</td><td>+3.07%</td><td>+3.81%</td><td>+4.92%</td><td>-1.76%</td><td>-4.85%</td><td>+0.22%</td><td>-2.89%</td><td>+1.60%</td></tr><tr><td>Reverse</td><td>-5.99%</td><td>-13.32%</td><td>+3.35%</td><td>+1.97%</td><td>0.00%</td><td>-5.19%</td><td>-12.88%</td><td>+2.08%</td><td>-3.00%</td><td>-4.00%</td></tr><tr><td rowspan="3">4 TL</td><td>-</td><td>1.18</td><td>1.57</td><td>67.07</td><td>33.96</td><td>23.40</td><td>1.11</td><td>1.38</td><td>71.34</td><td>36.41</td><td>23.20</td></tr><tr><td>Forward</td><td>-2.39%</td><td>-4.37%</td><td>+2.16%</td><td>+0.87%</td><td>0.00%</td><td>-1.42%</td><td>-4.46%</td><td>+0.77%</td><td>-3.01%</td><td>+9.48%</td></tr><tr><td>Reverse</td><td>-5.52%</td><td>-9.85%</td><td>+4.58%</td><td>+0.74%</td><td>+1.71%</td><td>-4.84%</td><td>-9.61%</td><td>+2.63%</td><td>+1.62%</td><td>-4.31%</td></tr><tr><td rowspan="7">Byte</td><td>1 LL</td><td>-</td><td>1.14</td><td>1.06</td><td>66.97</td><td>29.97</td><td>24.00</td><td>1.07</td><td>1.00</td><td>71.20</td><td>29.92</td><td>26.40</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>1.16</td><td>1.09</td><td>69.13</td><td>29.42</td><td>30.40</td><td>1.08</td><td>1.00</td><td>71.42</td><td>29.97</td><td>30.60</td></tr><tr><td>Forward</td><td>+3.16%</td><td>+2.82%</td><td>-2.76%</td><td>-1.43%</td><td>-6.58%</td><td>+2.62%</td><td>+3.12%</td><td>-1.40%</td><td>0.00%</td><td>+1.96%</td></tr><tr><td>Reverse</td><td>-3.44%</td><td>-3.95%</td><td>+1.77%</td><td>-0.72%</td><td>+6.58%</td><td>-2.43%</td><td>-4.45%</td><td>+1.46%</td><td>+7.30%</td><td>+1.31%</td></tr><tr><td rowspan="3">8 LL</td><td>-</td><td>1.23</td><td>1.19</td><td>68.16</td><td>28.79</td><td>34.40</td><td>1.13</td><td>1.07</td><td>69.60</td><td>29.71</td><td>30.40</td></tr><tr><td>Forward</td><td>+0.81%</td><td>0.00%</td><td>-4.41%</td><td>-1.17%</td><td>-19.77%</td><td>+0.79%</td><td>+1.28%</td><td>+0.83%</td><td>+2.27%</td><td>-7.24%</td></tr><tr><td>Reverse</td><td>-6.98%</td><td>-9.26%</td><td>+1.75%</td><td>-1.02%</td><td>-11.63%</td><td>-5.30%</td><td>-7.35%</td><td>+3.46%</td><td>+3.26%</td><td>-5.92%</td></tr></table>


Table 1: Performance of MTP models on NTP tasks using only their main LM heads. Percentage values indicate relative improvements or degradations in benchmark scores compared to the respective static MTP baselines trained without a curriculum. 1 LL head models refer to standard NTP models. The same table with absolute scores is provided in Appendix B.


## 3.2.2 Results

The performances<sup>3</sup> of all models on the NTP tasks using only the main language modeling heads are listed in Table 1. We make the following observations: 

Byte-level dynamic MTP models outperform subword-level ones. Despite starting from or converging to the NTP task during the pre-training, our proposed MTP curricula do not lead to performance improvements over the NTP baseline when applied to subword-level models of both sizes. However they lead to a noticeable improvement over the models trained with a static MTP objective, e.g. when they have 4 output heads, no matter if they are in the form of linear layers or transformer layers. Byte-level Reverse Curriculum MTP models, on the other hand, are able to score on par with or even outperform both static MTP and NTP models in non-knowledge-based benchmarks. 

This difference between subword- and byte-level models can be explained by the fact that smaller LLMs are constrained by how much future token information they can retain and leverage at each prediction step due to their parameter count. And given that subword tokens carry more semantic and morphological information than a single byte, modeling multiple bytes at once is much easier than multiple subword tokens. We argue that the Reverse Curriculum MTP objective leads to richer hidden state representations by making the model predict multiple future bytes at once. This, in turn, improves performance on the NTP task, especially compared to subword-level SLMs, when the model is trained to look ahead up to 4 future tokens during pre-training. 

Performance is similar for LL and TL heads. As for the performance difference between MTP models with multiple linear layers and transformer layers as language modeling heads, it is rather small, or sometimes even negligible, across both model sizes and both dynamic and static MTP objectives. And that is despite the fact that performing NTP with only the first LM heads results in Transformer Layer models "losing" parameters during inference. The effective parameter delta between a Transformer Layer MTP model with only the first LM head engaged and the corresponding Linear Layer model is around 210M per head in case of 3B model. 

Inference Speed-ups Across LM Head Types and Curricula 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/2f973294-625b-432b-b54d-f8254456fceb/124d21a7d4e2bc8ca0885e9252c1ae7a33013a6a5ac591ffd30892924832be48.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-17/2f973294-625b-432b-b54d-f8254456fceb/d41d9959f8af83c05eb101966be36de15f34d4f51b23045ee15b99ab21d75107.jpg)



Figure 2: Inference speed-ups achieved by MTP models in relation to the NTP baseline. We measure the speed-ups in terms of the amount of forward passes required to generate a sequence using self-speculative decoding formulation proposed by Stern et al. (2018)


## 3.3 Experiment 2: Inference speed

## 3.3.1 Setup

We implement greedy blockwise speculative decoding (Stern et al., 2018) and measure decoding speeds on completing inputs from a the MiniPile test split. We sample 512 documents, take the first 256 subword or 512 byte tokens as inputs and generate completions of the same length. All completions are generated with batch size 8. We measure the speedups in terms of the amount of forward passes relative to the NTP baseline. 

## 3.3.2 Results

Figure 2 showcases the maximum inference speedups that can be achieved by our MTP models, i.e. speed-ups when using all of the trained LM heads. Since self-speculative decoding requires at least two forward passes to predict and verify k > 1 tokens, 2-token subword models were excluded from this figure, as they do not allow for any inference speed-ups over a regular NTP setup. 

While the static MTP objective results in the largest inference speed-ups across both model types, sizes, and LM head types, the Forward Curriculum comes as a very close second. Reverse Curriculum, on the other hand, basically converges towards NTP inference speeds regardless of the LM head type used with subword-level models, and offers only modest speed-ups in case of byte-level models. 

Figure 3 provides a per head breakdown of the 

Figure 3: Breakdown of the per-head acceptance rate of tokens drafted via self-speculative decoding for 3B subword models. The y-axis indicates a head’s share of tokens accepted by the model in total. 

self-speculative performance for every curriculum based on 3B models with 4 LM heads. Once again, there is a small difference in the acceptance rate of the tokens drafted by the additional heads between models trained with No and Forward curricula regardless of the head type, while almost no tokens drafted by additional heads trained under the Reverse Curriculum were accepted. 

<table><tr><td rowspan="2">Tokens</td><td rowspan="2">Heads</td><td rowspan="2">Curriculum</td><td colspan="5">1.3B</td><td colspan="5">3B</td></tr><tr><td>BLEU ↑</td><td>ROUGE-L ↑</td><td>SemScore ↑</td><td>TTR ↑</td><td>G-Eval ↑</td><td>BLEU ↑</td><td>ROUGE-L ↑</td><td>SemScore ↑</td><td>TTR ↑</td><td>G-Eval ↑</td></tr><tr><td rowspan="13">Subword</td><td>1 LL</td><td>-</td><td>4.66</td><td>14.84</td><td>44.48</td><td>14.94</td><td>1.72</td><td>5.15</td><td>15.76</td><td>45.88</td><td>15.75</td><td>1.79</td></tr><tr><td rowspan="3">2 LL</td><td>-</td><td>4.31</td><td>14.83</td><td>42.05</td><td>13.78</td><td>1.72</td><td>4.82</td><td>15.50</td><td>45.15</td><td>14.92</td><td>1.78</td></tr><tr><td>Forward</td><td>-2.70%</td><td>-0.06%</td><td>+0.13%</td><td>-0.34%</td><td>+1.17%</td><td>+4.10%</td><td>+2.01%</td><td>+0.47%</td><td>+3.65%</td><td>-0.72%</td></tr><tr><td>Reverse</td><td>+7.21%</td><td>+0.46%</td><td>+3.01%</td><td>+7.46%</td><td>+1.42%</td><td>+1.25%</td><td>+1.35%</td><td>+1.24%</td><td>+8.78%</td><td>+0.58%</td></tr><tr><td rowspan="3">2 TL</td><td>-</td><td>4.31</td><td>14.64</td><td>42.01</td><td>13.70</td><td>1.71</td><td>5.08</td><td>15.65</td><td>45.35</td><td>15.60</td><td>1.79</td></tr><tr><td>Forward</td><td>+0.94%</td><td>+1.29%</td><td>-0.99%</td><td>+3.18%</td><td>+0.50%</td><td>-7.44%</td><td>-3.97%</td><td>-2.00%</td><td>-10.34%</td><td>-3.41%</td></tr><tr><td>Reverse</td><td>+4.72%</td><td>+2.81%</td><td>+1.76%</td><td>+1.68%</td><td>-2.44%</td><td>+3.88%</td><td>+1.29%</td><td>-0.74%</td><td>-3.06%</td><td>+1.40%</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>3.96</td><td>13.85</td><td>39.01</td><td>14.58</td><td>1.68</td><td>4.42</td><td>14.54</td><td>42.21</td><td>15.64</td><td>1.76</td></tr><tr><td>Forward</td><td>+4.05%</td><td>+3.63%</td><td>+2.23%</td><td>-1.22%</td><td>+0.25%</td><td>+4.45%</td><td>+2.95%</td><td>+2.62%</td><td>+1.19%</td><td>+0.63%</td></tr><tr><td>Reverse</td><td>+15.80%</td><td>+7.50%</td><td>+8.77%</td><td>+9.26%</td><td>+2.00%</td><td>+13.04%</td><td>+6.46%</td><td>+5.34%</td><td>+0.03%</td><td>+1.98%</td></tr><tr><td rowspan="3">4 TL</td><td>-</td><td>4.24</td><td>14.11</td><td>39.02</td><td>14.39</td><td>1.69</td><td>4.78</td><td>14.87</td><td>43.17</td><td>15.78</td><td>1.78</td></tr><tr><td>Forward</td><td>-8.87%</td><td>-0.14%</td><td>+3.07%</td><td>+1.19%</td><td>-0.21%</td><td>-3.83%</td><td>+1.16%</td><td>+1.44%</td><td>+2.91%</td><td>+0.14%</td></tr><tr><td>Reverse</td><td>+6.34%</td><td>+5.50%</td><td>+8.51%</td><td>+4.11%</td><td>+1.37%</td><td>+1.11%</td><td>+7.57%</td><td>+5.32%</td><td>-0.79%</td><td>+1.74%</td></tr><tr><td rowspan="7">Byte</td><td>1 LL</td><td>-</td><td>4.98</td><td>15.08</td><td>43.34</td><td>21.50</td><td>1.76</td><td>5.89</td><td>16.57</td><td>44.64</td><td>23.43</td><td>1.90</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>4.89</td><td>14.77</td><td>41.11</td><td>29.02</td><td>1.83</td><td>6.29</td><td>16.70</td><td>44.73</td><td>30.60</td><td>1.97</td></tr><tr><td>Forward</td><td>+3.79%</td><td>+0.81%</td><td>+0.53%</td><td>-1.99%</td><td>+1.63%</td><td>-6.98%</td><td>-4.82%</td><td>-5.88%</td><td>-4.87%</td><td>-2.37%</td></tr><tr><td>Reverse</td><td>+0.03%</td><td>+5.88%</td><td>+0.92%</td><td>+9.45%</td><td>+1.68%</td><td>-5.51%</td><td>-3.21%</td><td>-0.12%</td><td>+4.51%</td><td>-0.41%</td></tr><tr><td rowspan="3">8 LL</td><td>-</td><td>4.77</td><td>14.94</td><td>39.91</td><td>35.49</td><td>1.86</td><td>5.41</td><td>15.78</td><td>41.79</td><td>41.70</td><td>1.93</td></tr><tr><td>Forward</td><td>+2.80%</td><td>-0.48%</td><td>-5.20%</td><td>-0.51%</td><td>-2.91%</td><td>+4.65%</td><td>+1.21%</td><td>+0.59%</td><td>-12.72%</td><td>+0.14%</td></tr><tr><td>Reverse</td><td>+1.38%</td><td>+1.88%</td><td>+2.37%</td><td>-8.44%</td><td>+1.03%</td><td>+11.97%</td><td>+2.96%</td><td>+4.20%</td><td>-18.31%</td><td>+2.53%</td></tr></table>


Table 2: Evaluation of model outputs on the MiniPile test set. Percentage values indicate relative improvements or degradations in evaluation metrics compared to the respective static MTP baselines trained without a curriculum. 1 LL models correspond to regular NTP models and their scores provided for reference. The same table with absolute scores is provided in Appendix B.


## 3.4 Experiment 3: Output Quality

## 3.4.1 Setup

We take the outputs generated by the models when performing Experiment 2 and compare them with "gold" completions of the same length from the MiniPile test set. We rely on BLEU (Papineni et al., 2002), ROUGE-L (Lin, 2004) and SemScore (Aynetdinov and Akbik, 2024) metrics to determine how closely the model completions match the ground truth from the MiniPile test set. We also calculate the Type-Token Ratio (TTR) metric, which is a measure of lexical diversity. It is defined as the ratio of unique words to the total number of words in a text. 

We use the NLTK’s (v.3.8.2) (Bird and Loper, 2004) implementation of the BLEU score, as well as the RegExp tokenizer to tokenize the generated and gold sequences for BLEU, ROUGE-L and TTR. We provide the regular expression used to tokenize the text sequences in Appendix A. For SemScore we use the all-mpnet-base-v2 model (Reimers and Gurevych, 2019). As for the ROUGE-L, we use Google’s implementation of it. 

In addition to the aforementioned "traditional" metrics, we also perform an LLM-based (LLM as-a-judge) evaluation of generated outputs. We will refer to the resulting metric as G-Eval, as we rely on Liu et al. (2023) for the prompt design and probability-weighted final score calculation. Our evaluation prompt can be found in Appendix A. We define a 5-point rating scale with 1 being the worst score, and 5 being the best loosely based on Wang et al. (2023). The model of our choice for the role of an evaluator is OpenAI’s gpt-3.5-turbo-0125 due to its cost efficiency. 

## 3.4.2 Results

The results of evaluation are listed in Table 2. We also include the results for NTP models for reference. We make the following observations: 

Proposed curricula outperform static MTP baselines. In almost all of the model configurations both of our proposed curricula lead to better results than the static MTP approach on at least 2 metrics. The only model configurations in which No Curriculum models achieve better results are a byte-level 3B model with 4 linear layer heads and a subword-level 3B model with 2 transformer layer heads. 

Even though the Reverse Curriculum often showcases greater quality of improvements than Forward Curriculum over No curriculum, we have already shown previously that the Reverse Curriculum yields models that are closer in nature to NTP models, both in terms of the performance of the main LM head, as well as the lack of self-speculative decoding speed-ups. Therefore it would be more appropriate to compare Reverse Curriculum MTP models with NTP models, rather than No Curriculum MTP models. In this regard, the Reverse Curriculum models are not able to generate outputs that are qualitatively better than NTP models. 

When comparing No Curriculum models only against the Forward Curriculum MTP models, the latter still come out on top on at least two metrics, but with two additional exceptions: byte-level model with 1.3B parameters and 8 Linear Layer heads, as well as a subword-level model with 1.3B parameters and 2 Linear Layer heads. Nevertheless, in 8 out of 12 different configurations in total Forward Curriculum MTP generate qualitatively better sequences with almost the same speed-ups via self-speculative decoding as MTP models with No Curriculum. 

TL heads outperform LL. Additional model parameters that are introduced by using auxiliary linear layers as LM heads do not reliably translate to better generated sequences of better quality. Even the subword-level 3B models with 4 LM heads showcase better output quality with Transformer Layer LM Heads, rather than Linear Layers. 

This further reinforces the idea that MTP is not simply a matter of the parameter count. Linear Layers on their own might be too simplistic as LM heads to enable better MTP. 

Byte-level MTP models outperform subword models across all metrics. In Experiment 1 we already argued that by modeling multiple bytes at once, byte-level MTP models arrive at hidden representations that can handle more complex patterns, compared to their NTP counterparts. The output quality evaluation of 3B byte-level models further reinforces this point, and interestingly enough the MTP models generate significantly more lexically diverse outputs, as indicated by the TTR metric. 

In addition, byte-level models beat subwordlevel models across all metrics and model configurations. This also holds true not only when comparing byte-level MTP models against subword-level MTP counterparts, but also when comparing both static and dynamic MTP models against the subword NTP baseline. 

## 4 Related Work

Curriculum learning. After Bengio et al. (2009) first proposed to apply a curriculum learning strategy in the context of machine learning, it has been successfully applied on a number of tasks in various machine learning domains including natural language processing, computer vision and speech recognition (Soviany et al., 2022). In the context of language modeling, they have been shown to provide benefits both when pre-training encoder-only models (Xu et al., 2020; Nagatsuka et al., 2021; Ranaldi et al., 2023), as well as instruction-tuning large decoder-only models (Mukherjee et al., 2023; Lee et al., 2024). 

The use of curriculum learning approaches was not reported in pre-training any publicly available decoder-only foundation models trained on vast amounts of text data, although recently Feng et al. (2024) showed that using a two-stage curriculum based on text quality can lead better training outcomes. Meanwhile curriculum learning approaches have been very popular in data-constrained pretraining setups (Warstadt et al., 2023; Hu et al., 2024). While the curricula that focus on ordering the data based on various difficulty metrics were not found to be consistently better than noncurriculum baselines, an approach by Salhan et al. (2024) that involves a curriculum for pre-training objectives was able to reliably outperform noncurriculum baselines in a data-constrained setup. 

Multi-token prediction. ProphetNet (Qi et al., 2020) was the first large-scale transformer-based language model that was able to predict multiple ngrams in one prediction step. However, their model relies on n-stream self-attention mechanism that involves more computational overhead compared to regular transformers. 

Pal et al. (2023) showed that hidden states of next-token prediction models are able to encode more than a single token ahead by probing pretrained transformers, and that it’s possible to predict those to a certain extent. 

Gloeckle et al. (2024) improved upon the previous work by proposing slight architectural tweaks, such as using full transformer layers as language modeling heads, to account for the multi-token prediction task that resulted in a more computationally efficient, compute-matched with NTP models, and effective method for multi-token prediction. 

Self-speculative decoding. Stern et al. (2018) were the first to suggest a speculative decoding scheme for faster inference. Since then, a number self-speculative decoding methods were introduced. Some of these methods rely on the early-exit mechanism (Elhoushi et al., 2024; Liu et al., 2024b), others on skipping intermediate layers (Zhang et al., 2024a; Xia et al., 2024), and some on architectural transformations (Zhang et al., 2024b). Medusa (Cai et al., 2024) has gained the most prominence due to its simplicity and ability to relatively easily and cost-efficiently enable self-speculative decoding for LLMs that were pre-trained using the regular NTP objective. 

## 5 Conclusion

This paper introduces a novel curriculum-based training strategy to the multi-token prediction task. We conducted extensive experiments to determine their validity against both the regular NTP, as well as the static MTP training objectives. Our experiments demonstrated that the Forward MTP Curriculum offers the best trade-offs between the main LM head performance, inference speed, as well as the final output quality, when compared against both the static MTP and Reverse MTP Curriculum approaches. As a result, it allows SLMs to better leverage the MTP objective during pre-training. 

We have also shown that Reverse MTP Curriculum models outperform static MTP models on NTP benchmarks. In addition, Reverse MTP Curriculum models generate sequences of better quality than NTP models in case of both subword- and byte-level models. Unfortunately, they are not able to offer any meaningful inference speed-ups via self-speculative decoding, while coming short on NTP tasks in comparison to models trained using solely the NTP objective. 

In the future we are interested in investigating the potential benefits of the MTP objective for nontransformer architectures. 

## Limitations

One limitation of our proposed curricula is the fact that they are pre-defined in advance. The decision to progressively add or remove a token to or from an m-token task every <sup>m</sup> steps when training for n steps is somewhat arbitrary, since it does not rely on any metrics about the the models themselves or the training loss. This may lead to a situation that perhaps some LM heads require more steps to be fully "saturated", while others require less, and not accounting for this results in under- or overfitted LM heads. Since the goal of this paper was to establish that curriculum-based MTP training fundamentally benefits small language models over a static MTP objective in principle, we leave exploring the potentially very large space of possible schedules for future work. 

In addition, we acknowledge the fact that the language models considered in our experiments were trained on a dataset that is significantly smaller in size than datasets on which other contemporary LLMs are trained. We do not exclude the possibility that, when trained for a prolonged time and on significantly more data, the evaluation results of our proposed curricula may differ from the results reported in our paper in one way or another. However given the consistency in the results across various model configurations in all of our experiments, we argue that the training dataset size used to determine the validity of our proposed curricula does not undermine the reliability of our conclusions. 

## Acknowledgments

We thank all reviewers for their valuable feedback. The authors are supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Emmy Noether grant “Eidetic Representations of Natural Language” (project number 448414230). Further, Alan Akbik is supported by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy "Science of Intelligence" (EXC 2002/1, project number 390523135). 

The authors gratefully acknowledge the scientific support and HPC resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) under the NHR project c106fa. NHR funding is provided by federal and Bavarian state authorities. NHR@FAU hardware is partially funded by the German Research Foundation (DFG) – 440719683. 

## References



Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai, Matthew Dixon, Ronen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg, Allie Del Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, and Russell J. Hewett et al. 2024. Phi-3 





technical report: A highly capable language model locally on your phone. Preprint, arXiv:2404.14219. 





Susanne Albers, Arindam Khan, and Leon Ladewig. 2021. Best fit bin packing with random order revisited. Algorithmica, 83(9):2833–2858. 





Ansar Aynetdinov and Alan Akbik. 2024. Semscore: Automated evaluation of instruction-tuned llms based on semantic textual similarity. Preprint, arXiv:2401.17072. 





Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings ofthe 26th Annual International Conference on Machine Learning, ICML ’09, page 41–48, New York, NY, USA. Association for Computing Machinery. 





Steven Bird and Edward Loper. 2004. NLTK: The natural language toolkit. In Proceedings of the ACL In teractive Poster and Demonstration Sessions, pages 214–217, Barcelona, Spain. Association for Computational Linguistics. 





Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc. 





Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. Medusa: Simple llm inference acceleration framework with multiple decoding heads. Preprint, arXiv:2401.10774. 





Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1. 





Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed Aly, Beidi Chen, and Carole-Jean Wu. 2024. LayerSkip: Enabling early exit inference and self-speculative decoding. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12622–12642, Bangkok, Thailand. Association for Computational Linguistics. 





Steven Feng, Shrimai Prabhumoye, Kezhi Kong, Dan Su, Mostofa Patwary, Mohammad Shoeybi, and 





Bryan Catanzaro. 2024. Maximize your data’s potential: Enhancing llm accuracy with two-phase pretraining. Preprint, arXiv:2412.15285. 





Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. 2020. The pile: An 800gb dataset of diverse text for language modeling. Preprint, arXiv:2101.00027. 





Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024. A framework for few-shot language model evaluation. 





Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. 2024. Better & faster large language models via multi-token prediction. Preprint, arXiv:2404.19737. 





Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, and Damien Allonsius et al. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783. 





Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, and Jin Chen et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948. 





Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations. 





Michael Y. Hu, Aaron Mueller, Candace Ross, Adina Williams, Tal Linzen, Chengxu Zhuang, Ryan Cotterell, Leshem Choshen, Alex Warstadt, and Ethan Gotlieb Wilcox. 2024. Findings of the second babylm challenge: Sample-efficient pretraining on developmentally plausible corpora. Preprint, arXiv:2412.05149. 





Jean Kaddour. 2023. The minipile challenge for data-efficient language models. Preprint, arXiv:2304.08442. 





Bruce W Lee, Hyunsoo Cho, and Kang Min Yoo. 2024. Instruction tuning with human curriculum. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 1281–1309, Mexico City, Mexico. Association for Computational Linguistics. 





Yuanzhi Li, Sébastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. 2023. Textbooks are all you need ii: phi-1.5 technical report. Preprint, arXiv:2309.05463. 





Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics. 





Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, and Kai Hu et al. 2024a. Deepseek-v3 technical report. Preprint, arXiv:2412.19437. 





Fangcheng Liu, Yehui Tang, Zhenhua Liu, Yunsheng Ni, Kai Han, and Yunhe Wang. 2024b. Kangaroo: Lossless self-speculative decoding via double early exiting. Preprint, arXiv:2404.18911. 





Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics. 





Ilya Loshchilov and Frank Hutter. 2017. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations. 





Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Confer ence on Learning Representations. 





Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. 2024. Fineweb-edu: the finest collection of educational content. 





Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics. 





Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. 2023. Orca: Progressive learning from complex explanation traces of gpt-4. Preprint, arXiv:2306.02707. 





Koichi Nagatsuka, Clifford Broni-Bediako, and Masayasu Atsumi. 2021. Pre-training a BERT with curriculum learning by increasing block-size of input text. In Proceedings ofthe International Conference on Recent Advances in Natural Language Processing (RANLP 2021), pages 989–996, Held Online. INCOMA Ltd. 





OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, and Chester Cho et al. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774. 





Koyena Pal, Jiuding Sun, Andrew Yuan, Byron Wallace, and David Bau. 2023. Future lens: Anticipating subsequent tokens from a single hidden state. In Proceedings of the 27th Conference on Computa tional Natural Language Learning (CoNLL), pages 548–560, Singapore. Association for Computational Linguistics. 





Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The lambada dataset. 





Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th Annual Meeting on Association for Computational Linguistics, ACL ’02, page 311–318, USA. Association for Computational Linguistics. 





Weizhen Qi, Yu Yan, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, and Ming Zhou. 2020. ProphetNet: Predicting future n-gram for sequence-to-SequencePre-training. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2401–2410, Online. Association for Computational Linguistics. 





Leonardo Ranaldi, Giulia Pucci, and Fabio Massimo Zanzotto. 2023. Modeling easiness for training transformers with curriculum learning. In Proceedings ofthe 14th International Conference on Recent Advances in Natural Language Processing, pages 937– 948, Varna, Bulgaria. INCOMA Ltd., Shoumen, Bulgaria. 





Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics. 





Suchir Salhan, Richard Diehl Martinez, Zébulon Goriely, and Paula Buttery. 2024. Less is more: Pretraining cross-lingual small-scale language models with cognitively-plausible curriculum learning strategies. Preprint, arXiv:2410.22886. 





Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1715–1725, Berlin, Germany. Association for Computational Linguistics. 





Petru Soviany, Radu Tudor Ionescu, Paolo Rota, and Nicu Sebe. 2022. Curriculum learning: A survey. Preprint, arXiv:2101.10382. 





Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. Blockwise parallel decoding for deep autoregressive models. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc. 





Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. Preprint, arXiv:2302.13971. 





Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, and Pushkar Mishra et al. 2023b. Llama 2: Open foundation and fine-tuned chat models. Preprint, arXiv:2307.09288. 





Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. Self-instruct: Aligning language 





models with self-generated instructions. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, Toronto, Canada. Association for Computational Linguistics. 





Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjabe, Adina Williams, Tal Linzen, and Ryan Cotterell. 2023. Findings of the BabyLM challenge: Sample-efficient pretraining on developmentally plausible corpora. In Proceedings ofthe BabyLM Challenge at the 27th Conference on Computational Natural Language Learning, pages 1–34, Singapore. Association for Computational Linguistics. 





Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R. Bowman. 2020. Blimp: The benchmark of linguistic minimal pairs for english. Transactions ofthe Associationfor Computational Linguistics, 8:377–392. 





Heming Xia, Yongqi Li, Jun Zhang, Cunxiao Du, and Wenjie Li. 2024. Swift: On-the-fly self-speculative decoding for llm inference acceleration. Preprint, arXiv:2410.06916. 





Benfeng Xu, Licheng Zhang, Zhendong Mao, Quan Wang, Hongtao Xie, and Yongdong Zhang. 2020. Curriculum learning for natural language understanding. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 6095–6104, Online. Association for Computational Linguistics. 





Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, and Colin Raffel. 2022. ByT5: Towards a token-free future with pre-trained byte-to-byte models. Transactions ofthe Associationfor Computational Linguistics, 10:291–306. 





Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2024a. Draft& verify: Lossless large language model acceleration via self-speculative decoding. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11263–11282, Bangkok, Thailand. Association for Computational Linguistics. 





Kaiqi Zhang, Jing Zhao, and Rui Chen. 2024b. Koala: Enhancing speculative decoding for llm via multilayer draft heads with adversarial learning. Preprint, arXiv:2408.08146. 



## A Additional Evaluation Details

## A.1 NLTK RegExp Tokenizer Rule

Below we provide the regular expression rule assigned to a Python variable that we used for the NLTK’s RegExp tokenizer. This tokenizer was used to tokenize the sequences generated by the models we have trained in order to calculate the output quality metrics in Section 3.4, namely BLEU, ROUGE-L and TTR. 

pattern = r'''
# strings
"(?:[^\"\\]|\\.)*"
'(?:[^‘\\]|\\.)*'
# numeric literals
\d+(?:\..\d+)?
# words/unicode identifiers
[\w_]+
# punctuation, brackets, etc.
[!@#$%^&\*\(\)\-=+\\\|[\]\{\}; '",.<>/?~]+ 

## A.2 G-Eval Prompt

You will be given a sequence triplet consisting of: 

1. An input sequence (text or code) that serves as the starting point. 

2. An output sequence written as a continuation of the input. 

3. A target sequence that represents the expected continuation ofthe input sequence. 

Your task is to rate the written output sequence on one metric. 

Please make sure you read and understand these instructions carefully. Keep this document open while reviewing, and refer to it as needed. 

## Evaluation Criteria:

Overall Quality (1-5) - how well does the output sequence continue the input sequence and align with the target sequence? 

- A score of5 means that the output sequence is excellent. It provides a seamless continuation ofthe input sequence, closely aligns with the target sequence, and avoids any repetitions, irrelevant passages, or major errors. 

- A score of4 means that the output sequence is good. It continues the input sequence well and mostly aligns with the target sequence, but may include minor errors or imperfections, such as slight incoherence or small structural issues. 

- A score of 3 means that the output sequence is acceptable. It maintains some relevance to the input sequence and partial alignment with the target sequence, but contains noticeableflaws, such as incoherence, repetitions, or deviations that reduce its quality. 

- A score of2 means that the output sequence is poor. It struggles to continue the input sequence coherently or deviates significantly from the target sequence, with major errors, irrelevant sections, or repeated patterns. 

- A score of1 means that the output sequence is invalid. Itfails to continue the input sequence meaningfully, shows no alignment with the target sequence, or is completely incoherent. 

## Evaluation Steps:

1. Carefully read the input, output, and target sequences. 

2. Compare the output sequence to both the input sequence (continuity) and the target sequence (alignment). 

3. Rate the output on a scale of1-5for Quality, according to the criteria above. 

### Input Sequence: 

{Input Sequence} 

### Output Sequence: 

{Output Sequence} 

### Target Sequence: 

{Target Sequence} 

Evaluation Form (scores ONLY): 

- Quality: 

## B Additional Results

## B.1 Result tables with absolute values

Tables 3 and 4 report the absolute scores that correspond to those reported in Tables 1 and 2. 

## B.2 MTP models trained on 10B tokens

Table 5 lists the evaluation results of 1.3B models trained on 10B subword tokens of FineWeb-Edu. The MTP models with 4 TL layers showcase a performance that is consistent with 4 TL MTP models trained on MiniPile. As the scale increases, the dynamic MTP objective continues to result in performance improvements over the static one. 

Granted, the NTP objective still leads to a best performing model, however we believe that as the model size increases, the static MTP objective should start outperforming the NTP objective, as shown by Gloeckle et al. (2024). And given our observations, the dynamic MTP objective should result in even better performing scaled models. 

<table><tr><td rowspan="2">Tokens</td><td rowspan="2">Heads</td><td rowspan="2">Curriculum</td><td colspan="5">1.3B</td><td colspan="5">3B</td></tr><tr><td>MiniPile BPB ↓</td><td>LAMBADA BPB ↓</td><td>BLiMP Acc. ↑</td><td>ARC-E Acc. ↑</td><td>OBQA Acc. ↑</td><td>MiniPile BPB ↓</td><td>LAMBADA BPB ↓</td><td>BLiMP Acc. ↑</td><td>ARC-E Acc. ↑</td><td>OBQA Acc. ↑</td></tr><tr><td rowspan="13">Subword</td><td>1 LL</td><td>-</td><td>1.08</td><td>1.34</td><td>71.80</td><td>34.60</td><td>25.80</td><td>1.03</td><td>1.17</td><td>74.93</td><td>34.60</td><td>24.80</td></tr><tr><td rowspan="3">2 LL</td><td>-</td><td>1.12</td><td>1.44</td><td>70.27</td><td>31.99</td><td>23.80</td><td>1.06</td><td>1.27</td><td>72.09</td><td>36.36</td><td>24.60</td></tr><tr><td>Forward</td><td>1.12</td><td>1.45</td><td>70.79</td><td>34.64</td><td>25.60</td><td>1.06</td><td>1.27</td><td>73.65</td><td>35.40</td><td>25.40</td></tr><tr><td>Reverse</td><td>1.09</td><td>1.36</td><td>71.45</td><td>32.58</td><td>25.00</td><td>1.04</td><td>1.20</td><td>74.69</td><td>36.28</td><td>24.20</td></tr><tr><td rowspan="3">2 TL</td><td>-</td><td>1.11</td><td>1.41</td><td>71.04</td><td>34.09</td><td>26.20</td><td>1.05</td><td>1.24</td><td>73.32</td><td>36.15</td><td>25.00</td></tr><tr><td>Forward</td><td>1.12</td><td>1.43</td><td>70.72</td><td>35.14</td><td>25.00</td><td>1.06</td><td>1.24</td><td>73.26</td><td>35.44</td><td>24.60</td></tr><tr><td>Reverse</td><td>1.09</td><td>1.35</td><td>71.66</td><td>34.81</td><td>26.20</td><td>1.03</td><td>1.19</td><td>73.93</td><td>35.06</td><td>25.20</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>1.19</td><td>1.61</td><td>67.48</td><td>34.22</td><td>24.40</td><td>1.12</td><td>1.44</td><td>72.20</td><td>36.45</td><td>25.00</td></tr><tr><td>Forward</td><td>1.16</td><td>1.55</td><td>69.55</td><td>35.52</td><td>25.60</td><td>1.10</td><td>1.37</td><td>72.35</td><td>35.40</td><td>25.40</td></tr><tr><td>Reverse</td><td>1.12</td><td>1.43</td><td>69.74</td><td>34.89</td><td>24.40</td><td>1.06</td><td>1.26</td><td>73.70</td><td>35.35</td><td>24.00</td></tr><tr><td rowspan="3">4 TL</td><td>-</td><td>1.18</td><td>1.57</td><td>67.07</td><td>33.96</td><td>23.40</td><td>1.11</td><td>1.38</td><td>71.34</td><td>36.41</td><td>23.20</td></tr><tr><td>Forward</td><td>1.16</td><td>1.50</td><td>68.52</td><td>34.26</td><td>23.40</td><td>1.09</td><td>1.32</td><td>71.89</td><td>35.31</td><td>25.40</td></tr><tr><td>Reverse</td><td>1.12</td><td>1.42</td><td>70.14</td><td>34.22</td><td>23.80</td><td>1.06</td><td>1.25</td><td>73.21</td><td>36.99</td><td>22.20</td></tr><tr><td rowspan="7">Byte</td><td>1 LL</td><td>-</td><td>1.14</td><td>1.06</td><td>66.97</td><td>29.97</td><td>24.00</td><td>1.07</td><td>1.00</td><td>71.20</td><td>29.92</td><td>26.40</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>1.16</td><td>1.09</td><td>69.13</td><td>29.42</td><td>30.40</td><td>1.08</td><td>1.00</td><td>71.42</td><td>29.97</td><td>30.60</td></tr><tr><td>Forward</td><td>1.20</td><td>1.13</td><td>67.22</td><td>29.00</td><td>28.40</td><td>1.11</td><td>1.03</td><td>70.43</td><td>29.97</td><td>31.20</td></tr><tr><td>Reverse</td><td>1.12</td><td>1.05</td><td>70.36</td><td>29.21</td><td>32.20</td><td>1.06</td><td>0.96</td><td>72.47</td><td>32.15</td><td>31.00</td></tr><tr><td rowspan="3">8 LL</td><td>-</td><td>1.23</td><td>1.19</td><td>68.16</td><td>28.79</td><td>34.40</td><td>1.13</td><td>1.07</td><td>69.60</td><td>29.71</td><td>30.40</td></tr><tr><td>Forward</td><td>1.24</td><td>1.19</td><td>65.15</td><td>28.45</td><td>27.60</td><td>1.14</td><td>1.08</td><td>70.18</td><td>30.39</td><td>28.20</td></tr><tr><td>Reverse</td><td>1.15</td><td>1.08</td><td>69.35</td><td>28.49</td><td>30.40</td><td>1.07</td><td>0.99</td><td>72.01</td><td>30.68</td><td>28.60</td></tr></table>


Table 3: Performance of MTP models using only their main LM heads on NTP tasks. Best NTP performance across all model configurations are highlighted.


<table><tr><td rowspan="2">Tokens</td><td rowspan="2">Heads</td><td rowspan="2">Curriculum</td><td colspan="5">1.3B</td><td colspan="5">3B</td></tr><tr><td>BLEU↑</td><td>ROUGE-L↑</td><td>SemScore↑</td><td>TTR↑</td><td>G-Eval↑</td><td>BLEU↑</td><td>ROUGE-L↑</td><td>SemScore↑</td><td>TTR↑</td><td>G-Eval↑</td></tr><tr><td rowspan="13">Subword</td><td>1 LL</td><td>-</td><td>4.66</td><td>14.84</td><td>44.48</td><td>14.94</td><td>1.71</td><td>5.15</td><td>15.76</td><td>45.88</td><td>15.75</td><td>1.79</td></tr><tr><td rowspan="3">2 LL</td><td>-</td><td>4.31</td><td>14.83</td><td>42.05</td><td>13.78</td><td>1.72</td><td>4.82</td><td>15.50</td><td>45.15</td><td>14.92</td><td>1.78</td></tr><tr><td>Forward</td><td>4.20</td><td>14.82</td><td>42.11</td><td>13.74</td><td>1.74</td><td>5.02</td><td>15.81</td><td>45.36</td><td>15.47</td><td>1.77</td></tr><tr><td>Reverse</td><td>4.62</td><td>14.90</td><td>43.32</td><td>14.81</td><td>1.75</td><td>4.88</td><td>15.71</td><td>45.71</td><td>16.23</td><td>1.79</td></tr><tr><td rowspan="3">2 TL</td><td>-</td><td>4.31</td><td>14.64</td><td>42.01</td><td>13.70</td><td>1.71</td><td>5.08</td><td>15.65</td><td>45.35</td><td>15.60</td><td>1.79</td></tr><tr><td>Forward</td><td>4.35</td><td>14.83</td><td>41.59</td><td>14.14</td><td>1.72</td><td>4.70</td><td>15.03</td><td>44.44</td><td>13.98</td><td>1.72</td></tr><tr><td>Reverse</td><td>4.52</td><td>15.05</td><td>42.75</td><td>13.93</td><td>1.67</td><td>5.28</td><td>15.85</td><td>45.01</td><td>15.12</td><td>1.81</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>3.96</td><td>13.85</td><td>39.01</td><td>14.58</td><td>1.68</td><td>4.42</td><td>14.54</td><td>42.21</td><td>15.64</td><td>1.76</td></tr><tr><td>Forward</td><td>4.12</td><td>14.36</td><td>39.88</td><td>14.40</td><td>1.69</td><td>4.61</td><td>14.97</td><td>43.31</td><td>15.83</td><td>1.77</td></tr><tr><td>Reverse</td><td>4.59</td><td>14.89</td><td>42.42</td><td>15.92</td><td>1.72</td><td>4.99</td><td>15.48</td><td>44.46</td><td>15.65</td><td>1.80</td></tr><tr><td rowspan="3">4 TL</td><td>-</td><td>4.24</td><td>14.11</td><td>39.02</td><td>14.39</td><td>1.69</td><td>4.78</td><td>14.87</td><td>43.17</td><td>15.78</td><td>1.78</td></tr><tr><td>Forward</td><td>3.86</td><td>14.09</td><td>40.21</td><td>14.56</td><td>1.68</td><td>4.59</td><td>15.04</td><td>43.79</td><td>16.23</td><td>1.78</td></tr><tr><td>Reverse</td><td>4.51</td><td>14.88</td><td>42.34</td><td>14.98</td><td>1.71</td><td>4.83</td><td>15.99</td><td>45.47</td><td>15.65</td><td>1.81</td></tr><tr><td rowspan="7">Byte</td><td>1 LL</td><td>-</td><td>4.98</td><td>15.08</td><td>43.34</td><td>21.50</td><td>1.76</td><td>5.89</td><td>16.57</td><td>44.64</td><td>23.43</td><td>1.90</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>4.89</td><td>14.77</td><td>41.11</td><td>29.02</td><td>1.83</td><td>6.29</td><td>16.70</td><td>44.73</td><td>30.60</td><td>1.97</td></tr><tr><td>Forward</td><td>5.08</td><td>14.89</td><td>41.32</td><td>28.44</td><td>1.86</td><td>5.85</td><td>15.90</td><td>42.10</td><td>29.11</td><td>1.92</td></tr><tr><td>Reverse</td><td>4.89</td><td>15.64</td><td>41.48</td><td>31.76</td><td>1.86</td><td>5.94</td><td>16.17</td><td>44.68</td><td>31.98</td><td>1.96</td></tr><tr><td rowspan="3">8 LL</td><td>-</td><td>4.77</td><td>14.94</td><td>39.91</td><td>35.49</td><td>1.86</td><td>5.41</td><td>15.78</td><td>41.79</td><td>41.70</td><td>1.93</td></tr><tr><td>Forward</td><td>4.90</td><td>14.87</td><td>37.83</td><td>35.31</td><td>1.81</td><td>5.66</td><td>15.98</td><td>42.04</td><td>36.40</td><td>1.94</td></tr><tr><td>Reverse</td><td>4.83</td><td>15.22</td><td>40.85</td><td>32.50</td><td>1.88</td><td>6.06</td><td>16.25</td><td>43.54</td><td>34.07</td><td>1.98</td></tr><tr><td rowspan="2">Tokens</td><td rowspan="2">Heads</td><td rowspan="2">Curriculum</td><td colspan="5">Main LM Head</td><td colspan="5">Output Quality</td></tr><tr><td>MiniPile PPL ↓</td><td>LAMBADA PPL ↓</td><td>BLiMP Acc. ↑</td><td>ARC-E Acc. ↑</td><td>OBQA Acc. ↑</td><td>BLEU ↑</td><td>ROUGE-L ↑</td><td>SemScore ↑</td><td>TTR ↑</td><td>G-Eval ↑</td></tr><tr><td rowspan="4">Subword</td><td>1 LL</td><td>No</td><td>47</td><td>74</td><td>78.74</td><td>54.63</td><td>30.40</td><td>2.02</td><td>15.39</td><td>52.40</td><td>13.18</td><td>1.52</td></tr><tr><td rowspan="3">4 TL</td><td>No</td><td>57</td><td>100</td><td>75.96</td><td>55.22</td><td>27.40</td><td>1.97</td><td>14.80</td><td>51.69</td><td>16.11</td><td>1.54</td></tr><tr><td>Forward</td><td>57</td><td>101</td><td>76.56</td><td>54.30</td><td>28.40</td><td>2.00</td><td>15.08</td><td>52.71</td><td>15.48</td><td>1.56</td></tr><tr><td>Reverse</td><td>50</td><td>75</td><td>77.55</td><td>55.60</td><td>30.40</td><td>2.19</td><td>14.91</td><td>52.77</td><td>16.25</td><td>1.56</td></tr></table>


Table 4: Evaluation of the outputs generated by the models based on the MiniPile test set inputs. Highlighted are scores that are improvements over the corresponding static MTP baselines trained without a curriculum for each respective model configuration.


Table 5: Evaluation results of 1.3B models trained on 10B tokens of FineWeb-Edu. 

We recognize that 10B tokens is still a very modest token budget for moder LLM pre-training. Nevertheless, we hope that our results will serve as a useful starting point for future work in a scaled setup. 

## B.3 BLiMP breakdown

Tables 6 and 7 break down the performance of the models on the BLiMP benchmark from Table 1. We show the performance on each linguistic phenomenon considered in the benchmark. Please refer to Warstadt et al. (2020) for the precise definition of each phenomenon and additional details on how to interpret them. Below we discuss the results of one of the particularly interesting and illustrative English phenomenon pair. 

The category of tasks in BLiMP that fall under the Binding category deals with structural relationships between a pronoun and its antecedent. Often they are further apart from each other within a sentence, making them a good match for MTP models. Meanwhile, the Subject-Verb Agreement (SVA) phenomenon can be characterized as the agreement between subjects and verbs in number. Usually they are close to each other in a sentence, i.e. SVA often deals with relationships between adjacent tokens, which NTP models should excel at. 

The results show that even though the MTP models should be better suited to deal with the Binding phenomenon, the complexity of the MTP objective hinders the performance of 1.3B models with respect to this phenomenon. Only 3B MTP models are able to outperform their NTP counterparts, if trained with a dynamic MTP objective. Given the performance gains observed by 3B models with 4 LL or 2 TL heads, the performance degradation of 1.3B models in the Binding category can perhaps be explained by the limited capacity of the model due to its parameter count. 

The performance gap on the SVA phenomenon, on the other hand, is more expected, as NTP models do not have to account for multiple future tokens at once in its internal representations. Notably, the performance drop is especially noticeable in case of 4-token prediction models. Notably, the dynamic MTP objective significantly alleviates the performance drop in the SVA category, especially when it comes to 3B models. 

To conclude, these results make it evident that the proposed dynamic MTP objective offers some noticeable performance improvements in some categories of the English grammatical phenomena, despite the fact that MTP models on average perform worse than NTP models on the BLiMP benchmark, as shown in Tables 1 or 3 of the paper. 

<table><tr><td>Tokens</td><td>Heads</td><td>Curriculum</td><td>Anaphor Agreement</td><td>Argument Structure</td><td>Binding</td><td>Control/Raising</td><td>Ellipsis</td><td>Filler Gap</td><td>Irregular Forms</td><td>Island Effects</td><td>NPI Licensing</td><td>Quantifiers</td><td>Subject-Verb Agreement</td></tr><tr><td rowspan="13">Subword</td><td>1 LL</td><td>-</td><td>90.8</td><td>73.44</td><td>78.37</td><td>70.94</td><td>73.7</td><td>71.33</td><td>92.3</td><td>50.28</td><td>60.87</td><td>67.35</td><td>78.13</td></tr><tr><td rowspan="3">2 LL</td><td>-</td><td>88.2</td><td>73.14</td><td>76.57</td><td>70.42</td><td>75.45</td><td>69.73</td><td>86.9</td><td>48.72</td><td>61.11</td><td>62.35</td><td>76.98</td></tr><tr><td>Forward</td><td>88.35</td><td>72.91</td><td>76.87</td><td>69.54</td><td>74.8</td><td>69.27</td><td>93.15</td><td>53.59</td><td>61.21</td><td>54.52</td><td>78.37</td></tr><tr><td>Reverse</td><td>87.8</td><td>74.51</td><td>77.35</td><td>71.06</td><td>75.6</td><td>70.06</td><td>87.15</td><td>49.57</td><td>63.21</td><td>64.65</td><td>78.6</td></tr><tr><td rowspan="3">2 TL</td><td>-</td><td>89.75</td><td>72.59</td><td>76.95</td><td>68.34</td><td>71.95</td><td>69.94</td><td>91.65</td><td>50.55</td><td>62.69</td><td>67.35</td><td>77.42</td></tr><tr><td>Forward</td><td>90.95</td><td>72.82</td><td>77.67</td><td>68.78</td><td>72.8</td><td>68.83</td><td>90.9</td><td>49.84</td><td>59.66</td><td>63.95</td><td>77.7</td></tr><tr><td>Reverse</td><td>86.0</td><td>73.48</td><td>78.32</td><td>69.36</td><td>72.45</td><td>70.07</td><td>92.25</td><td>50.4</td><td>61.01</td><td>73.05</td><td>78.12</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>84.15</td><td>70.22</td><td>76.2</td><td>66.96</td><td>72.95</td><td>68.03</td><td>88.1</td><td>53.01</td><td>48.43</td><td>56.88</td><td>71.18</td></tr><tr><td>Forward</td><td>88.1</td><td>71.43</td><td>77.37</td><td>68.26</td><td>70.65</td><td>69.49</td><td>91.45</td><td>51.64</td><td>57.06</td><td>56.12</td><td>76.82</td></tr><tr><td>Reverse</td><td>85.95</td><td>73.19</td><td>77.78</td><td>68.38</td><td>73.7</td><td>68.4</td><td>89.25</td><td>52.21</td><td>55.17</td><td>62.05</td><td>74.15</td></tr><tr><td rowspan="3">4 TL</td><td>-</td><td>90.5</td><td>69.19</td><td>75.82</td><td>67.02</td><td>71.4</td><td>68.67</td><td>83.4</td><td>48.59</td><td>52.73</td><td>57.73</td><td>70.33</td></tr><tr><td>Forward</td><td>86.7</td><td>71.39</td><td>77.7</td><td>68.66</td><td>69.3</td><td>70.19</td><td>86.45</td><td>46.4</td><td>56.81</td><td>52.62</td><td>76.1</td></tr><tr><td>Reverse</td><td>86.8</td><td>72.47</td><td>76.68</td><td>70.4</td><td>73.0</td><td>69.19</td><td>86.75</td><td>50.61</td><td>60.29</td><td>64.53</td><td>75.67</td></tr><tr><td rowspan="7">Byte</td><td>1 LL</td><td>-</td><td>77.05</td><td>71.8</td><td>75.27</td><td>67.64</td><td>72.9</td><td>67.81</td><td>87.7</td><td>45.53</td><td>40.89</td><td>64.2</td><td>77.92</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>72.5</td><td>73.78</td><td>76.77</td><td>67.82</td><td>75.7</td><td>67.0</td><td>85.45</td><td>47.09</td><td>50.26</td><td>66.8</td><td>82.45</td></tr><tr><td>Forward</td><td>71.65</td><td>71.12</td><td>75.0</td><td>64.54</td><td>71.5</td><td>67.0</td><td>82.35</td><td>44.61</td><td>57.39</td><td>57.38</td><td>75.12</td></tr><tr><td>Reverse</td><td>72.75</td><td>73.78</td><td>75.03</td><td>67.98</td><td>74.55</td><td>69.94</td><td>87.3</td><td>47.7</td><td>57.03</td><td>65.83</td><td>84.18</td></tr><tr><td rowspan="3">8 LL</td><td>-</td><td>69.15</td><td>72.04</td><td>75.7</td><td>65.78</td><td>74.9</td><td>67.9</td><td>88.25</td><td>46.32</td><td>54.96</td><td>68.2</td><td>76.4</td></tr><tr><td>Forward</td><td>64.75</td><td>68.79</td><td>76.53</td><td>65.72</td><td>72.7</td><td>66.01</td><td>82.55</td><td>42.61</td><td>45.39</td><td>67.1</td><td>71.25</td></tr><tr><td>Reverse</td><td>75.9</td><td>74.6</td><td>76.2</td><td>66.86</td><td>71.8</td><td>69.9</td><td>88.4</td><td>48.24</td><td>49.06</td><td>69.8</td><td>77.88</td></tr></table>


Table 6: Detailed performance of 1.3B models on the BLiMP benchmark. These results correspond to the aggregated results reported in Table 1


<table><tr><td>Tokens</td><td>Heads</td><td>Curriculum</td><td>Anaphor Agreement</td><td>Argument Structure</td><td>Binding</td><td>Control/Raising</td><td>Ellipsis</td><td>Filler Gap</td><td>Irregular Forms</td><td>Island Effects</td><td>NPI Licensing</td><td>Quantifiers</td><td>Subject-Verb Agreement</td></tr><tr><td rowspan="13">Subword</td><td>1 LL</td><td>-</td><td>92.8</td><td>76.88</td><td>78.82</td><td>70.62</td><td>78.45</td><td>72.59</td><td>90.65</td><td>55.24</td><td>64.96</td><td>74.03</td><td>83.6</td></tr><tr><td rowspan="3">2 LL</td><td>-</td><td>91.4</td><td>74.43</td><td>78.63</td><td>71.84</td><td>75.0</td><td>71.47</td><td>92.0</td><td>54.4</td><td>48.01</td><td>72.88</td><td>81.4</td></tr><tr><td>Forward</td><td>90.65</td><td>75.57</td><td>78.62</td><td>71.4</td><td>77.15</td><td>72.4</td><td>90.5</td><td>54.3</td><td>64.56</td><td>64.05</td><td>82.2</td></tr><tr><td>Reverse</td><td>92.55</td><td>75.58</td><td>79.48</td><td>71.78</td><td>77.15</td><td>72.21</td><td>91.5</td><td>55.97</td><td>63.79</td><td>75.72</td><td>82.92</td></tr><tr><td rowspan="3">2 TL</td><td>-</td><td>89.7</td><td>75.56</td><td>78.85</td><td>70.06</td><td>76.3</td><td>72.04</td><td>91.6</td><td>57.3</td><td>62.37</td><td>69.1</td><td>77.73</td></tr><tr><td>Forward</td><td>89.3</td><td>74.9</td><td>78.47</td><td>71.14</td><td>75.35</td><td>71.07</td><td>85.1</td><td>50.8</td><td>65.49</td><td>73.6</td><td>80.2</td></tr><tr><td>Reverse</td><td>89.8</td><td>76.08</td><td>80.02</td><td>72.22</td><td>76.7</td><td>72.2</td><td>88.3</td><td>57.25</td><td>65.13</td><td>66.92</td><td>80.53</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>92.1</td><td>72.17</td><td>78.0</td><td>69.0</td><td>73.45</td><td>70.19</td><td>90.15</td><td>52.35</td><td>70.87</td><td>68.08</td><td>73.18</td></tr><tr><td>Forward</td><td>89.65</td><td>73.96</td><td>78.63</td><td>70.32</td><td>72.75</td><td>71.26</td><td>89.15</td><td>54.11</td><td>61.94</td><td>63.6</td><td>78.78</td></tr><tr><td>Reverse</td><td>91.7</td><td>75.66</td><td>79.4</td><td>70.6</td><td>76.2</td><td>71.61</td><td>91.0</td><td>54.05</td><td>69.7</td><td>70.03</td><td>76.7</td></tr><tr><td rowspan="3">4 TL</td><td>-</td><td>89.9</td><td>72.99</td><td>78.37</td><td>70.34</td><td>72.05</td><td>70.27</td><td>91.55</td><td>52.21</td><td>59.81</td><td>72.28</td><td>75.55</td></tr><tr><td>Forward</td><td>90.75</td><td>73.42</td><td>77.95</td><td>70.12</td><td>75.2</td><td>70.73</td><td>87.15</td><td>49.51</td><td>64.71</td><td>67.8</td><td>78.45</td></tr><tr><td>Reverse</td><td>91.65</td><td>75.0</td><td>78.28</td><td>71.24</td><td>77.2</td><td>70.36</td><td>90.65</td><td>54.76</td><td>64.81</td><td>70.78</td><td>76.88</td></tr><tr><td rowspan="7">Byte</td><td>1 LL</td><td>-</td><td>76.75</td><td>74.49</td><td>76.88</td><td>67.14</td><td>74.55</td><td>70.43</td><td>91.6</td><td>55.34</td><td>50.36</td><td>65.17</td><td>84.33</td></tr><tr><td rowspan="3">4 LL</td><td>-</td><td>77.5</td><td>74.83</td><td>77.0</td><td>69.66</td><td>76.55</td><td>68.76</td><td>82.1</td><td>51.98</td><td>59.54</td><td>68.08</td><td>82.2</td></tr><tr><td>Forward</td><td>78.8</td><td>74.84</td><td>75.98</td><td>67.98</td><td>77.25</td><td>69.26</td><td>91.05</td><td>50.78</td><td>54.03</td><td>63.05</td><td>82.7</td></tr><tr><td>Reverse</td><td>87.7</td><td>76.14</td><td>77.47</td><td>70.44</td><td>74.7</td><td>71.01</td><td>90.25</td><td>50.94</td><td>58.07</td><td>67.92</td><td>84.5</td></tr><tr><td rowspan="3">8 LL</td><td>-</td><td>81.6</td><td>74.7</td><td>74.08</td><td>67.98</td><td>75.2</td><td>68.27</td><td>84.25</td><td>48.49</td><td>52.16</td><td>69.85</td><td>80.83</td></tr><tr><td>Forward</td><td>78.9</td><td>74.02</td><td>75.72</td><td>67.96</td><td>76.45</td><td>68.07</td><td>87.35</td><td>52.41</td><td>56.76</td><td>62.45</td><td>79.97</td></tr><tr><td>Reverse</td><td>79.2</td><td>75.19</td><td>75.87</td><td>70.28</td><td>77.2</td><td>71.26</td><td>85.8</td><td>51.81</td><td>60.41</td><td>71.8</td><td>81.82</td></tr></table>


Table 7: Detailed performance of 3B models on the BLiMP benchmark. These results correspond to the aggregated results reported in Table 1
